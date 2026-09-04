"""
    FastPIDCCUDAExt

Package extension providing a CUDA-accelerated implementation of
[`FastPIDC.compute_puc_full_cuda`](@ref), loaded automatically once
`using CUDA` makes the `CUDA` package available alongside `FastPIDC`.
Selected by passing `config.backend = :cuda` (the default) to
[`FastPIDC.compute_puc_full`](@ref).
"""
module FastPIDCCUDAExt

using FastPIDC
using CUDA

# --- Kernels ---

"""
    joint_counts_kernel_chunked!(data, counts, n, m, k_bins, z_start, z_chunk_size)

CUDA kernel: for a chunk of `z_chunk_size` target genes starting at
`z_start`, accumulate the joint bin-count histogram `counts[u, v, x,
z_local]` (co-occurrences of bin `u` for gene `x` and bin `v` for gene
`z_global = z_start + z_local - 1`) across all `m` samples in `data`. One
GPU thread handles one `(x, z_local)` pair; `n` is the number of genes and
`k_bins` the number of discretization bins.
"""
function joint_counts_kernel_chunked!(data, counts, n, m, k_bins, z_start, z_chunk_size)
    x = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    z_local = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    
    if x > n || z_local > z_chunk_size; return nothing; end
    
    z_global = z_start + z_local - 1
    if z_global > n || x == z_global; return nothing; end
    
    # Each thread computes joint counts for pair (x, z_global). Derive the
    # increment from the count buffer so UInt16/UInt32 storage remains generic.
    count_increment = one(eltype(counts))
    for s in 1:m
        # Bin IDs may use a compact unsigned storage type on the GPU; widen
        # them for bounds checks and native N-D indexing.
        u = Int32(data[s, x])
        v = Int32(data[s, z_global])
        if u >= 1 && u <= k_bins && v >= 1 && v <= k_bins
            counts[u, v, x, z_local] += count_increment
        end
    end

    return nothing
end

"""
    mi_si_kernel_chunked!(counts, marginals, mi_matrix, si_matrix, n, m, k_bins, z_start, z_chunk_size)

CUDA kernel: from the joint bin counts `counts` (as produced by
[`joint_counts_kernel_chunked!`](@ref)) and per-gene marginal bin
probabilities `marginals`, compute the mutual information
`mi_matrix[x, z_global]` and the specific information
`si_matrix[:, x, z_local]` of gene `x` with respect to target gene
`z_global = z_start + z_local - 1`, for the chunk of `z_chunk_size` targets
starting at `z_start`. One GPU thread handles one `(x, z_local)` pair; `n`
is the number of genes, `m` the number of samples, and `k_bins` the number
of discretization bins.
"""
function mi_si_kernel_chunked!(counts, marginals, mi_matrix, si_matrix, n, m, k_bins, z_start, z_chunk_size)
    x = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    z_local = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    
    if x > n || z_local > z_chunk_size; return nothing; end
    
    z_global = z_start + z_local - 1
    if z_global > n || x == z_global; return nothing; end
    
    inv_m = 1.0 / Float64(m)
    mi_val = 0.0
    
    for v in 1:k_bins
        p_z_v = marginals[v, z_global]
        if p_z_v <= 0.0; continue; end
        
        si_v = 0.0f0 # Use Float32 for Specific Information buffer
        for u in 1:k_bins
            p_x_u = marginals[u, x]
            if p_x_u <= 0.0; continue; end
            
            c_uv = counts[u, v, x, z_local]
            p_uv = Float64(c_uv) * inv_m

            if p_uv > 0.0
                mi_val += p_uv * log2(p_uv / (p_x_u * p_z_v))
                p_u_cond_v = p_uv / p_z_v
                si_v += Float64(p_u_cond_v * log2(p_u_cond_v / p_x_u))
            end
        end
        si_matrix[v, x, z_local] = si_v
    end

    mi_matrix[x, z_global] = mi_val
    return nothing
end

"""
    puc_accumulation_kernel_chunked!(si_matrix, mi_matrix, puc_scores, marginals, n, k_bins, z_start, z_chunk_size)

CUDA kernel: for each target gene `z_global` in the current chunk and each
source gene `x`, accumulate the PUC contribution
`puc_scores[x, z_global] += (MI(x, z_global) - redundancy(x, y, z_global)) /
MI(x, z_global)` (clamped to be non-negative) summed over all other genes
`y`, using specific information values from `si_matrix` and marginal
probabilities from `marginals`. One GPU thread handles one `(x, z_local)`
pair, looping internally over `y`; `n` is the number of genes and `k_bins`
the number of discretization bins. Contributions still need to be
symmetrized (`puc_scores[i,j] + puc_scores[j,i]`) by the caller, since each
thread only writes `puc_scores[x, z_global]`.
"""
function puc_accumulation_kernel_chunked!(si_matrix, mi_matrix, puc_scores, marginals, n, k_bins, z_start, z_chunk_size)
    x = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    z_local = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    
    if x > n || z_local > z_chunk_size; return nothing; end
    
    z_global = z_start + z_local - 1
    if z_global > n || x == z_global; return nothing; end
    
    mi_xz = mi_matrix[x, z_global]
    if mi_xz <= 1e-12; return nothing; end
    
    local_puc = 0.0
    # Y loop: to compute redundancy, we need SI for every other node Y with Z.
    for y in 1:n
        if y == x || y == z_global; continue; end
        
        redundancy = 0.0
        for k in 1:k_bins
            p_z_k = marginals[k, z_global]
            if p_z_k <= 0.0; continue; end
            
            # Read from the Float64 SI matrix, convert back to Float64 for math
            si_x = Float64(si_matrix[k, x, z_local])
            si_y = Float64(si_matrix[k, y, z_local])
            
            redundancy += p_z_k * min(si_x, si_y)
        end

        score = (mi_xz - redundancy) / mi_xz
        if isfinite(score) && score > 0.0
            local_puc += score
        end
    end
    
    puc_scores[x, z_global] = local_puc
    return nothing
end

# --- Host Implementation ---

function _smallest_unsigned_type(max_value::Integer)
    max_value >= 0 || throw(ArgumentError("max_value must be nonnegative"))

    if max_value <= 255
        return UInt8
    elseif max_value <= 65_535
        return UInt16
    elseif max_value <= 4_294_967_295
        return UInt32
    else
        return UInt64
    end
end

function _compute_puc_full_cuda_typed(
    nodes,
    config,
    base,
    ::Type{BinT},
    ::Type{CountT},
) where {BinT<:Integer,CountT<:Integer}
    num_nodes = length(nodes)
    num_samples = length(nodes[1].binned_values)
    k_bins = maximum(n -> n.number_of_bins, nodes)

    # Prepare static data on CPU and move to GPU.
    data_cpu = Matrix{BinT}(undef, num_samples, num_nodes)
    marginals_cpu = zeros(Float64, k_bins, num_nodes)
    for i = 1:num_nodes
        data_cpu[:, i] .= nodes[i].binned_values
        p = nodes[i].probabilities
        marginals_cpu[1:length(p), i] .= Float64.(p)
    end

    data_gpu = CuArray(data_cpu)
    marginals_gpu = CuArray(marginals_cpu)

    # Global output matrices.
    puc_scores_gpu = CUDA.zeros(Float64, num_nodes, num_nodes)
    mi_matrix_gpu = CUDA.zeros(Float64, num_nodes, num_nodes)

    # Chunk configuration: `counts_chunk_gpu`/`si_chunk_gpu` scale with
    # k_bins^2 * num_nodes * chunk_size and k_bins * num_nodes * chunk_size,
    # respectively. Size the chunk to fit the GPU memory currently free rather
    # than always requesting a fixed chunk size of 256. Because BB_prefix uses
    # the smallest exact joint-count type, account for CountT rather than
    # assuming Int32 storage.
    bytes_per_chunk_col =
        k_bins^2 * num_nodes * sizeof(CountT) +  # counts_chunk_gpu
        k_bins * num_nodes * sizeof(Float64)     # si_chunk_gpu
    free_bytes, _ = CUDA.memory_info()
    safety_factor = 0.8  # headroom for fixed buffers + allocator overhead/fragmentation
    max_chunk_size = floor(Int, free_bytes * safety_factor / bytes_per_chunk_col)
    chunk_size = clamp(max_chunk_size, 1, min(256, num_nodes))

    if max_chunk_size < 1
        error(
            "compute_puc_full_cuda: even a single-gene chunk would require " *
            "$(round(bytes_per_chunk_col / 2^30, digits = 2)) GiB of GPU memory " *
            "(only $(round(free_bytes * safety_factor / 2^30, digits = 2)) GiB " *
            "usable), because the discretizer selected k_bins=$k_bins bins per " *
            "gene. This is usually caused by an adaptive discretizer (e.g. " *
            "\"bayesian_blocks\", the default) picking an unbounded number of " *
            "bins on a dataset with many samples. Try discretizer=\"uniform_width\" " *
            "with a fixed, small number_of_bins (e.g. 10-20), or config.backend = :cpu.",
        )
    end

    # Chunked intermediate buffers, pre-allocated once.
    counts_chunk_gpu = CUDA.zeros(CountT, k_bins, k_bins, num_nodes, chunk_size)
    si_chunk_gpu = CUDA.zeros(Float64, k_bins, num_nodes, chunk_size)

    if config.verbose
        println(
            "[FastPIDC] GPU Chunked PUC: Processing $num_nodes x $num_nodes pairs " *
            "(k_bins=$k_bins)...",
        )
        println(
            "[FastPIDC] Using chunk size of $chunk_size " *
            "(approx. $(ceil(Int, num_nodes / chunk_size)) iterations), " *
            "sized to fit $(round(free_bytes / 2^30, digits = 2)) GiB free GPU memory",
        )
        println(
            "[FastPIDC] GPU storage types: bin IDs=$(BinT) (max bins=$k_bins), " *
            "joint counts=$(CountT) (cells=$num_samples)",
        )
    end

    # Iterate over the Z-axis in chunks
    for z_start in 1:chunk_size:num_nodes
        z_end = min(z_start + chunk_size - 1, num_nodes)
        z_curr_chunk_size = z_end - z_start + 1

        # Wipe the intermediate buffers clean before the next chunk.
        CUDA.fill!(counts_chunk_gpu, zero(CountT))
        CUDA.fill!(si_chunk_gpu, 0.0)

        threads = (16, 16)
        blocks = (cld(num_nodes, 16), cld(z_curr_chunk_size, 16))

        @cuda threads=threads blocks=blocks joint_counts_kernel_chunked!(
            data_gpu,
            counts_chunk_gpu,
            Int32(num_nodes),
            Int32(num_samples),
            Int32(k_bins),
            Int32(z_start),
            Int32(z_curr_chunk_size),
        )

        @cuda threads=threads blocks=blocks mi_si_kernel_chunked!(
            counts_chunk_gpu,
            marginals_gpu,
            mi_matrix_gpu,
            si_chunk_gpu,
            Int32(num_nodes),
            Int32(num_samples),
            Int32(k_bins),
            Int32(z_start),
            Int32(z_curr_chunk_size),
        )

        @cuda threads=threads blocks=blocks puc_accumulation_kernel_chunked!(
            si_chunk_gpu,
            mi_matrix_gpu,
            puc_scores_gpu,
            marginals_gpu,
            Int32(num_nodes),
            Int32(k_bins),
            Int32(z_start),
            Int32(z_curr_chunk_size),
        )
    end

    # Copy results back
    puc_scores_cpu = Array(puc_scores_gpu)
    mi_matrix_cpu = Array(mi_matrix_gpu)

    # Symmetrize PUC scores
    for i = 1:num_nodes
        for j = (i+1):num_nodes
            val = puc_scores_cpu[i, j] + puc_scores_cpu[j, i]
            puc_scores_cpu[i, j] = val
            puc_scores_cpu[j, i] = val
        end
    end

    return mi_matrix_cpu, puc_scores_cpu
end

"""
    FastPIDC.compute_puc_full_cuda(nodes, config, base) -> (mi_scores, puc_scores)

GPU implementation of [`FastPIDC.compute_puc_full`](@ref): computes the full
pairwise MI matrix and pre-context PUC matrix for `nodes` on the GPU,
processing genes along the target (`z`) axis in chunks (up to 256 genes at
a time) sized to fit the GPU memory currently free, to bound device memory
use even when the discretizer has picked a large number of bins (see the
chunk-sizing comment in the implementation). Moves discretized data and
marginal probabilities to the GPU once, then for each chunk launches
[`joint_counts_kernel_chunked!`](@ref), [`mi_si_kernel_chunked!`](@ref) and
[`puc_accumulation_kernel_chunked!`](@ref) in sequence, symmetrizing the
resulting PUC matrix before returning both matrices to the CPU.
`config.verbose` enables progress printouts; `base` is currently unused
(mutual information is always computed in base 2 on the GPU path). Raises
an `ErrorException` with a suggested remedy if even a single-gene chunk
would not fit in the currently-free GPU memory.
"""
function FastPIDC.compute_puc_full_cuda(nodes, config, base)
    num_samples = length(nodes[1].binned_values)
    k_bins = maximum(n -> n.number_of_bins, nodes)

    # Bin IDs only need to represent the largest per-gene bin index.
    BinT = _smallest_unsigned_type(k_bins)

    # A joint count can reach the total number of cells. Choose the smallest
    # exact unsigned type that can hold num_samples, guarding against overflow
    # as datasets with larger cell counts are processed.
    CountT = _smallest_unsigned_type(num_samples)

    return _compute_puc_full_cuda_typed(nodes, config, base, BinT, CountT)
end


# --- Bayesian-block CUDA backend -------------------------------------------

FastPIDC.bayesian_blocks_cuda_available() = CUDA.functional()

# Deterministic comparison used by the Bayesian Blocks reduction: higher
# score wins, with exact ties resolved in favor of the smaller candidate index.
@inline function _bb_take_other(
    other_score::Float64,
    other_i::Int32,
    current_score::Float64,
    current_i::Int32,
)::Bool
    return other_score > current_score ||
           (other_score == current_score && other_i < current_i)
end

"""
    bayesian_blocks_dp_kernel!(...)

Assign one CUDA block to each gene. Endpoints `K` remain sequential because
`best[K]` depends on earlier endpoints, while threads within the block evaluate
candidate starts `i <= K` in parallel. The reduction uses a deterministic
first-maximum rule: higher score wins, and exact ties choose the smaller `i`.
"""
function bayesian_blocks_dp_kernel!(
    prefix_counts::CuDeviceArray{CountT,1},
    block_lengths::CuDeviceArray{Float64,1},
    state_offsets::CuDeviceArray{Int64,1},
    block_offsets::CuDeviceArray{Int64,1},
    unique_counts::CuDeviceArray{Int32,1},
    best::CuDeviceArray{Float64,1},
    last::CuDeviceArray{IndexT,1},
    final_scores::CuDeviceArray{Float64,1},
    priors::CuDeviceArray{Float64,1},
) where {CountT<:Unsigned,IndexT<:Unsigned}
    gene = Int32(blockIdx().x)
    tid = Int32(threadIdx().x)
    nthreads = Int32(blockDim().x)
    lane = ((tid - 1) & Int32(31)) + 1
    warp = ((tid - 1) >>> 5) + 1
    nwarps = nthreads >>> 5

    # Keep one candidate per thread and one reduced candidate per warp. The
    # reduction uses shared memory rather than generic shuffle helpers, keeping
    # device dispatch fully concrete while requiring only two block barriers.
    thread_scores = CuStaticSharedArray(Float64, 256)
    thread_indices = CuStaticSharedArray(Int32, 256)
    warp_scores = CuStaticSharedArray(Float64, 8)
    warp_indices = CuStaticSharedArray(Int32, 8)

    @inbounds begin
        state_start = state_offsets[gene]
        block_start = block_offsets[gene]
        n_unique = unique_counts[gene]

        # Match the CPU reference's singleton behavior explicitly. The generic
        # event-fitness expression has zero block width when U_g == 1, whereas a
        # constant gene should deterministically return its two outer edges and
        # an objective score of zero.
        if n_unique == 1
            if tid == 1
                best[state_start] = 0.0
                last[state_start] = one(IndexT)
                final_scores[gene] = 0.0
            end
            return nothing
        end

        K = Int32(1)
        while K <= n_unique
            block_length_K1 = block_lengths[block_start + Int64(K)]
            prefix_K = Float64(prefix_counts[state_start + Int64(K) - 1])
            prior = priors[K]

            local_best = -Inf
            local_i = typemax(Int32)
            i = tid
            while i <= K
                prefix_before = i == 1 ? 0.0 :
                                Float64(prefix_counts[state_start + Int64(i) - 2])
                count = prefix_K - prefix_before
                width =
                    block_lengths[block_start + Int64(i) - 1] - block_length_K1

                fit = count * log(count / width) - prior
                if i > 1
                    fit += best[state_start + Int64(i) - 2]
                end

                if _bb_take_other(fit, i, local_best, local_i)
                    local_best = fit
                    local_i = i
                end
                i += nthreads
            end

            thread_scores[tid] = local_best
            thread_indices[tid] = local_i
            sync_threads()

            # Each warp leader scans its 32 thread-local candidates in a fixed
            # order. Exact ties still choose the smaller i, matching the CPU
            # strict-`>` scan independently of launch size.
            if lane == 1
                warp_start = (warp - 1) * Int32(32) + 1
                warp_end = warp_start + Int32(31)
                warp_best = thread_scores[warp_start]
                warp_i = thread_indices[warp_start]
                slot = warp_start + 1
                while slot <= warp_end
                    other_score = thread_scores[slot]
                    other_i = thread_indices[slot]
                    if _bb_take_other(other_score, other_i, warp_best, warp_i)
                        warp_best = other_score
                        warp_i = other_i
                    end
                    slot += 1
                end
                warp_scores[warp] = warp_best
                warp_indices[warp] = warp_i
            end
            sync_threads()

            if tid == 1
                block_best = warp_scores[1]
                block_i = warp_indices[1]
                warp_slot = Int32(2)
                while warp_slot <= nwarps
                    other_score = warp_scores[warp_slot]
                    other_i = warp_indices[warp_slot]
                    if _bb_take_other(other_score, other_i, block_best, block_i)
                        block_best = other_score
                        block_i = other_i
                    end
                    warp_slot += 1
                end

                state_index = state_start + Int64(K) - 1
                best[state_index] = block_best
                # IndexT was selected from max(U_g), so this modular conversion
                # is exact and avoids a checked integer constructor in device code.
                last[state_index] = block_i % IndexT
                if K == n_unique
                    final_scores[gene] = block_best
                end
            end

            # `best[K]` is stored in global memory and is required by every
            # thread during the next endpoint. Block synchronization makes that
            # write visible before advancing K.
            sync_threads()
            K += 1
        end
    end

    return nothing
end

function _bb_threads_for_max_u(max_u::Integer)
    if max_u <= 32
        return 32
    elseif max_u <= 512
        return 64
    elseif max_u <= 4_096
        return 128
    else
        return 256
    end
end

function _bb_quantile_buckets(problems::Vector{FastPIDC.BayesianBlocksProblem})
    n = length(problems)
    n == 0 && return Vector{Vector{Int}}()

    # U_g is already available from required preprocessing. Sorting only these
    # gene indices is a lightweight O(G log G) operation and avoids a second
    # scan of the expression matrix merely to choose GPU workload buckets.
    order = sortperm(eachindex(problems); by = i -> length(problems[i].prefix_counts))
    n_buckets = min(4, n)
    buckets = Vector{Vector{Int}}()
    for bucket = 1:n_buckets
        lo = fld((bucket - 1) * n, n_buckets) + 1
        hi = fld(bucket * n, n_buckets)
        lo <= hi && push!(buckets, collect(order[lo:hi]))
    end
    return buckets
end

function _bb_prior_values(max_u::Integer)
    # The prior depends only on endpoint K, not on the gene. Compute it once on
    # the CPU with the reference expression, avoiding one pow/log pair per gene
    # per endpoint and eliminating that source of CPU/CUDA numeric variation.
    return [4 - log(73.53 * 0.05 * ((K)^-0.478)) for K = 1:max_u]
end

function _bb_problem_bytes(
    problem::FastPIDC.BayesianBlocksProblem,
    ::Type{CountT},
    ::Type{IndexT},
) where {CountT<:Integer,IndexT<:Integer}
    u = length(problem.prefix_counts)
    return (
        sizeof(Float64) * (u + 1) + # block lengths
        sizeof(CountT) * u +        # prefix counts
        sizeof(Float64) * u +       # best scores
        sizeof(IndexT) * u          # back-pointers
    )
end

function _bb_memory_batches(
    bucket::Vector{Int},
    problems::Vector{FastPIDC.BayesianBlocksProblem},
    budget_bytes::Integer,
    ::Type{CountT},
    ::Type{IndexT},
) where {CountT<:Integer,IndexT<:Integer}
    batches = Vector{Vector{Int}}()
    current = Int[]
    current_bytes = 0

    for problem_index in bucket
        problem_bytes = _bb_problem_bytes(problems[problem_index], CountT, IndexT)
        problem_bytes <= budget_bytes || throw(
            ArgumentError(
                "One Bayesian-block problem requires $(problem_bytes) bytes, " *
                "which exceeds the CUDA batch budget of $(budget_bytes) bytes. " *
                "Reduce the number of unique input values for that gene.",
            ),
        )

        if !isempty(current) && current_bytes + problem_bytes > budget_bytes
            push!(batches, current)
            current = Int[]
            current_bytes = 0
        end
        push!(current, problem_index)
        current_bytes += problem_bytes
    end

    !isempty(current) && push!(batches, current)
    return batches
end

function _flatten_bb_batch(
    problems::Vector{FastPIDC.BayesianBlocksProblem},
    problem_indices::Vector{Int},
    ::Type{CountT},
) where {CountT<:Integer}
    n_genes = length(problem_indices)
    total_states = sum(i -> length(problems[i].prefix_counts), problem_indices)
    total_blocks = total_states + n_genes

    prefix_counts = Vector{CountT}(undef, total_states)
    block_lengths = Vector{Float64}(undef, total_blocks)
    state_offsets = Vector{Int64}(undef, n_genes)
    block_offsets = Vector{Int64}(undef, n_genes)
    unique_counts = Vector{Int32}(undef, n_genes)

    state_cursor = 1
    block_cursor = 1
    for (local_gene, problem_index) in enumerate(problem_indices)
        problem = problems[problem_index]
        u = length(problem.prefix_counts)
        u <= typemax(Int32) || throw(
            ArgumentError("Bayesian blocks CUDA backend supports at most $(typemax(Int32)) unique values per gene"),
        )

        state_offsets[local_gene] = state_cursor
        block_offsets[local_gene] = block_cursor
        unique_counts[local_gene] = Int32(u)

        @inbounds for j = 1:u
            prefix_counts[state_cursor+j-1] = CountT(problem.prefix_counts[j])
        end
        edge_end = problem.edges[end]
        @inbounds for j = 1:(u+1)
            block_lengths[block_cursor+j-1] = edge_end - problem.edges[j]
        end

        state_cursor += u
        block_cursor += u + 1
    end

    return prefix_counts, block_lengths, state_offsets, block_offsets, unique_counts
end

function _change_points_from_last(last_values, offset::Int, n::Int)
    n >= 1 || throw(ArgumentError("Bayesian-block backtracking requires n >= 1"))

    # A valid partition may place every unique value in its own block. In that
    # case the returned edge-index path contains U_g + 1 entries, so allocating
    # only U_g slots can underflow to Julia index zero during backtracking.
    change_points = Vector{Int64}(undef, n + 1)
    i_cp = n + 2
    ind = n + 1
    while true
        i_cp -= 1
        change_points[i_cp] = ind
        ind == 1 && break

        state = ind - 1
        1 <= state <= n || throw(
            ArgumentError("invalid Bayesian-block state $state while backtracking"),
        )
        next_ind = Int(last_values[offset + state - 1])
        1 <= next_ind <= state || throw(
            ArgumentError(
                "invalid Bayesian-block back-pointer $next_ind for state $state",
            ),
        )
        ind = next_ind
    end
    return change_points[i_cp:end]
end

function _solve_bb_cuda_batch_with_priors(
    problems::Vector{FastPIDC.BayesianBlocksProblem},
    problem_indices::Vector{Int},
    threads::Int,
    ::Type{CountT},
    ::Type{IndexT},
    priors_gpu,
) where {CountT<:Integer,IndexT<:Integer}
    threads in (32, 64, 128, 256) || throw(
        ArgumentError(
            "CUDA Bayesian blocks requires a power-of-two thread count " *
            "from 32, 64, 128, or 256; got $threads",
        ),
    )

    prefix_counts, block_lengths, state_offsets, block_offsets, unique_counts =
        _flatten_bb_batch(problems, problem_indices, CountT)

    prefix_gpu = CuArray(prefix_counts)
    block_gpu = CuArray(block_lengths)
    state_offsets_gpu = CuArray(state_offsets)
    block_offsets_gpu = CuArray(block_offsets)
    unique_counts_gpu = CuArray(unique_counts)
    best_gpu = CUDA.zeros(Float64, length(prefix_counts))
    last_gpu = CUDA.zeros(IndexT, length(prefix_counts))
    final_scores_gpu = CUDA.zeros(Float64, length(problem_indices))

    try
        @cuda threads=threads blocks=length(problem_indices) bayesian_blocks_dp_kernel!(
            prefix_gpu,
            block_gpu,
            state_offsets_gpu,
            block_offsets_gpu,
            unique_counts_gpu,
            best_gpu,
            last_gpu,
            final_scores_gpu,
            priors_gpu,
        )

        last_values = Array(last_gpu)
        final_scores = Array(final_scores_gpu)

        solutions =
            Vector{FastPIDC.BayesianBlocksSolution}(undef, length(problem_indices))
        for local_gene = eachindex(problem_indices)
            offset = state_offsets[local_gene]
            n = Int(unique_counts[local_gene])
            change_points = _change_points_from_last(last_values, offset, n)
            solutions[local_gene] = FastPIDC.BayesianBlocksSolution(
                change_points,
                final_scores[local_gene],
            )
        end
        return solutions
    finally
        # Explicitly return batch allocations to CUDA's pool. The CUDA backend
        # may process many U_g buckets, so relying on a later GC cycle can retain
        # unnecessary pressure between batches or after an exception.
        for array in (
            prefix_gpu,
            block_gpu,
            state_offsets_gpu,
            block_offsets_gpu,
            unique_counts_gpu,
            best_gpu,
            last_gpu,
            final_scores_gpu,
        )
            CUDA.unsafe_free!(array)
        end
    end
end

function _solve_bb_cuda_batch(
    problems::Vector{FastPIDC.BayesianBlocksProblem},
    problem_indices::Vector{Int},
    threads::Int,
    ::Type{CountT},
    ::Type{IndexT},
) where {CountT<:Integer,IndexT<:Integer}
    max_u = maximum(i -> length(problems[i].prefix_counts), problem_indices)
    priors_gpu = CuArray(_bb_prior_values(max_u))
    try
        return _solve_bb_cuda_batch_with_priors(
            problems,
            problem_indices,
            threads,
            CountT,
            IndexT,
            priors_gpu,
        )
    finally
        CUDA.unsafe_free!(priors_gpu)
    end
end

function FastPIDC.solve_bayesian_blocks_cuda(
    problems::Vector{FastPIDC.BayesianBlocksProblem},
    verbose::Bool,
)
    CUDA.functional() || return nothing
    isempty(problems) && return FastPIDC.BayesianBlocksSolution[]

    sample_count = maximum(p -> Int(round(p.prefix_counts[end])), problems)
    max_u = maximum(p -> length(p.prefix_counts), problems)

    # A cumulative prefix count can reach the number of cells, so select the
    # smallest exact unsigned type that guards against overflow for this input.
    CountT = _smallest_unsigned_type(sample_count)
    # Back-pointers only need to represent candidate indices up to U_g.
    IndexT = _smallest_unsigned_type(max_u)

    free_bytes = Int(CUDA.free_memory())
    # Keep headroom for the CUDA context, allocator bookkeeping, and other
    # active package allocations while still using most of the currently free
    # device memory.
    budget_bytes = max(1, floor(Int, 0.65 * Float64(free_bytes)))

    buckets = _bb_quantile_buckets(problems)
    solutions = Vector{FastPIDC.BayesianBlocksSolution}(undef, length(problems))
    priors_gpu = CuArray(_bb_prior_values(max_u))

    if verbose
        unique_counts = sort!(collect(length(p.prefix_counts) for p in problems))
        median_u = unique_counts[cld(length(unique_counts), 2)]
        println(
            "[FastPIDC] CUDA Bayesian blocks: $(length(problems)) genes, " *
            "U_g median=$median_u, max=$(unique_counts[end]), " *
            "prefix counts=$(CountT), back-pointers=$(IndexT)",
        )
        println(
            "[FastPIDC] CUDA Bayesian blocks memory budget: " *
            "$(round(budget_bytes / 2.0^30; digits = 2)) GiB",
        )
    end

    try
        for (bucket_number, bucket) in enumerate(buckets)
            bucket_max_u = maximum(i -> length(problems[i].prefix_counts), bucket)
            threads = _bb_threads_for_max_u(bucket_max_u)
            batches = _bb_memory_batches(
                bucket,
                problems,
                budget_bytes,
                CountT,
                IndexT,
            )

            if verbose
                bucket_min_u = minimum(i -> length(problems[i].prefix_counts), bucket)
                println(
                    "[FastPIDC] CUDA BB bucket $bucket_number/$(length(buckets)): " *
                    "$(length(bucket)) genes, U_g=$bucket_min_u:$bucket_max_u, " *
                    "threads=$threads, batches=$(length(batches))",
                )
            end

            for batch in batches
                batch_solutions = _solve_bb_cuda_batch_with_priors(
                    problems,
                    batch,
                    threads,
                    CountT,
                    IndexT,
                    priors_gpu,
                )
                for (problem_index, solution) in zip(batch, batch_solutions)
                    solutions[problem_index] = solution
                end
            end
        end
        return solutions
    finally
        CUDA.unsafe_free!(priors_gpu)
    end
end

end # module
