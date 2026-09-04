"""
    FastPIDCCUDAExt

Package extension providing a CUDA-accelerated implementation of
[`FastPIDC.compute_puc_full_cuda`](@ref), loaded automatically once
`using CUDA` makes the `CUDA` package available alongside `FastPIDC`.
Selected by passing `config.backend = :cuda` (the default) to
[`FastPIDC.compute_puc_full`](@ref).

Rather than maintaining a second, CUDA.jl-native copy of the PUC kernels,
this extension compiles and drives the single canonical kernel source
shared with the Python package, `python/src/fastpidc/kernels/pidc_kernels.cu`
(see that file's header comment for the algorithm). It is plain CUDA C, so
it is compiled here with `nvcc` (required on `PATH` in addition to a
functional GPU) into PTX for the active device's compute capability, then
loaded with `CUDA.CuModule` and driven with `CUDA.cudacall` - the same
kernels Python's `cuda` backend loads via `cupy.RawModule`.
"""
module FastPIDCCUDAExt

using FastPIDC
using CUDA

# --- Shared kernel source: compile once per (session, GPU architecture) ---

const _MODULE_CACHE = Dict{String,CuModule}()

"""
    _kernel_source_path() -> String

Path to the canonical CUDA kernel source, shared with the Python package.
Resolved relative to this Julia package's root, since a git-based install
(`Pkg.add(url = ...)`) clones the whole repository, `python/` included.
"""
function _kernel_source_path()
    path = joinpath(pkgdir(FastPIDC), "python", "src", "fastpidc", "kernels", "pidc_kernels.cu")
    isfile(path) || error(
        "Shared CUDA kernel source not found at $path. FastPIDC.jl's CUDA " *
        "extension expects to find it relative to the package root (see " *
        "the FastPIDCCUDAExt module docstring); if FastPIDC.jl was " *
        "installed without the `python/` subdirectory, this backend is " *
        "unavailable.",
    )
    return path
end

"""
    _compile_ptx(arch::String) -> String

Compile the shared kernel source to PTX text targeting virtual architecture
`arch` (e.g. `"compute_89"`), using `nvcc`. Requires a CUDA toolkit
installation with `nvcc` on `PATH`.
"""
function _compile_ptx(arch::String)
    nvcc = Sys.which("nvcc")
    nvcc === nothing && error(
        "The CUDA backend needs `nvcc` (from a CUDA toolkit installation) " *
        "on PATH to compile the shared kernel source " *
        "(python/src/fastpidc/kernels/pidc_kernels.cu); none was found. " *
        "Install the CUDA toolkit, or use `config.backend = :cpu`.",
    )

    src = _kernel_source_path()
    mktempdir() do dir
        ptx_path = joinpath(dir, "pidc_kernels.ptx")
        cmd = `$nvcc --ptx -arch=$arch $src -o $ptx_path`
        out = IOBuffer()
        try
            run(pipeline(cmd; stdout=out, stderr=out))
        catch e
            error("nvcc failed to compile $src:\n$(String(take!(out)))")
        end
        return read(ptx_path, String)
    end
end

"""
    _get_module() -> CuModule

The compiled kernel module for the current device's compute capability,
compiling (and caching, per architecture, for the life of the Julia
session) on first use.
"""
function _get_module()
    cap = CUDA.capability(CUDA.device())
    arch = "compute_$(cap.major)$(cap.minor)"
    get!(_MODULE_CACHE, arch) do
        CuModule(_compile_ptx(arch))
    end
end

# --- Host implementation ---

"""
    FastPIDC.compute_puc_full_cuda(nodes, config, base) -> (mi_scores, puc_scores)

GPU implementation of [`FastPIDC.compute_puc_full`](@ref): computes the full
pairwise MI matrix and pre-context PUC matrix for `nodes` on the GPU,
processing genes along the target (`z`) axis in chunks of 256 to bound
device memory use. Moves discretized data and marginal probabilities to
the GPU once, then for each chunk launches the shared
`joint_counts_kernel`, `mi_si_kernel` and `puc_accumulation_kernel` (see
the `FastPIDCCUDAExt` module docstring) in sequence, symmetrizing the
resulting PUC matrix before returning both matrices to the CPU.
`config.verbose` enables progress printouts; `base` is currently unused
(mutual information is always computed in base 2 on the GPU, matching the
kernel source).

Device buffers use Julia's column-major layout with dimensions reversed
relative to the kernel source's documented (row-major) shapes - e.g. a
kernel-documented `(k_bins, n)` array is allocated here with Julia size
`(n, k_bins)` - so that the flat in-memory layout the kernels index into
with manual pointer arithmetic is identical in both languages, with no
transposition needed at the call boundary.
"""
function FastPIDC.compute_puc_full_cuda(nodes, config, base)
    md = _get_module()
    joint_counts_kernel = CuFunction(md, "joint_counts_kernel")
    mi_si_kernel = CuFunction(md, "mi_si_kernel")
    puc_accumulation_kernel = CuFunction(md, "puc_accumulation_kernel")

    num_nodes = length(nodes)
    num_samples = length(nodes[1].binned_values)
    k_bins = maximum(n -> n.number_of_bins, nodes)

    # Chunk configuration (256 target genes at a time)
    chunk_size = 256

    # Prepare static data on CPU and move to GPU. The kernels use 0-indexed
    # bin ids; FastPIDC.jl's bin ids are 1-indexed, so shift them down.
    data_cpu = zeros(Int32, num_nodes, num_samples)          # kernel shape (m, n), reversed
    marginals_cpu = zeros(Float64, num_nodes, k_bins)         # kernel shape (k_bins, n), reversed
    for i = 1:num_nodes
        data_cpu[i, :] .= Int32.(nodes[i].binned_values) .- Int32(1)
        p = nodes[i].probabilities
        marginals_cpu[i, 1:length(p)] .= Float64.(p)
    end

    data_gpu = CuArray(data_cpu)
    marginals_gpu = CuArray(marginals_cpu)

    # Global output matrices (kernel shape (n, n); square, so no reversal needed)
    puc_scores_gpu = CUDA.zeros(Float64, num_nodes, num_nodes)
    mi_matrix_gpu = CUDA.zeros(Float64, num_nodes, num_nodes)

    # Chunked intermediate buffers (pre-allocated once), reversed dims
    counts_chunk_gpu = CUDA.zeros(Int32, chunk_size, num_nodes, k_bins, k_bins)
    si_chunk_gpu = CUDA.zeros(Float64, chunk_size, num_nodes, k_bins)

    if config.verbose
        println("[FastPIDC] GPU Chunked PUC: Processing $num_nodes x $num_nodes pairs...")
        println("[FastPIDC] Using chunk size of $chunk_size (approx. $(ceil(Int, num_nodes/chunk_size)) iterations)")
    end

    threads = (16, 16)

    # Iterate over the Z-axis in chunks
    for z_start_1 in 1:chunk_size:num_nodes
        z_start = z_start_1 - 1  # 0-indexed, matching the kernel's convention
        z_end = min(z_start_1 + chunk_size - 1, num_nodes)
        z_curr_chunk_size = z_end - z_start_1 + 1

        CUDA.fill!(counts_chunk_gpu, Int32(0))
        CUDA.fill!(si_chunk_gpu, Float64(0))

        blocks = (cld(num_nodes, 16), cld(z_curr_chunk_size, 16))

        cudacall(
            joint_counts_kernel,
            (CuPtr{Cint}, CuPtr{Cint}, Cint, Cint, Cint, Cint, Cint),
            data_gpu, counts_chunk_gpu,
            Cint(num_nodes), Cint(num_samples), Cint(k_bins),
            Cint(z_start), Cint(z_curr_chunk_size);
            blocks=blocks, threads=threads,
        )

        cudacall(
            mi_si_kernel,
            (CuPtr{Cint}, CuPtr{Cdouble}, CuPtr{Cdouble}, CuPtr{Cdouble}, Cint, Cint, Cint, Cint, Cint),
            counts_chunk_gpu, marginals_gpu, mi_matrix_gpu, si_chunk_gpu,
            Cint(num_nodes), Cint(num_samples), Cint(k_bins),
            Cint(z_start), Cint(z_curr_chunk_size);
            blocks=blocks, threads=threads,
        )

        cudacall(
            puc_accumulation_kernel,
            (CuPtr{Cdouble}, CuPtr{Cdouble}, CuPtr{Cdouble}, CuPtr{Cdouble}, Cint, Cint, Cint, Cint),
            si_chunk_gpu, mi_matrix_gpu, puc_scores_gpu, marginals_gpu,
            Cint(num_nodes), Cint(k_bins),
            Cint(z_start), Cint(z_curr_chunk_size);
            blocks=blocks, threads=threads,
        )
    end

    # Copy results back. The kernels write mi_matrix[x*n+z] / puc_scores[x*n+z]
    # using their row-major (x, z) convention; with our reversed-dims layout
    # that is Julia index [z+1, x+1], i.e. the arrays come back transposed
    # relative to the (x, z) naming.
    mi_matrix_cpu = permutedims(Array(mi_matrix_gpu))
    puc_scores_cpu = permutedims(Array(puc_scores_gpu))

    # Symmetrize PUC scores: each ordered pair (x, z) only holds one of the
    # two directional contributions (see the kernel source's header comment).
    for i = 1:num_nodes
        for j = (i+1):num_nodes
            val = puc_scores_cpu[i, j] + puc_scores_cpu[j, i]
            puc_scores_cpu[i, j] = val
            puc_scores_cpu[j, i] = val
        end
    end

    return mi_matrix_cpu, puc_scores_cpu
end

end # module
