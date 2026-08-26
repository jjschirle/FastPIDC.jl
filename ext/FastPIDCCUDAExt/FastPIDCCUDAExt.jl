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
    
    # Each thread computes joint counts for pair (x, z_global)
    for s in 1:m
        u = data[s, x]
        v = data[s, z_global]
        if u >= 1 && u <= k_bins && v >= 1 && v <= k_bins
            # Using native N-D indexing
            counts[u, v, x, z_local] += Int32(1)
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

"""
    FastPIDC.compute_puc_full_cuda(nodes, config, base) -> (mi_scores, puc_scores)

GPU implementation of [`FastPIDC.compute_puc_full`](@ref): computes the full
pairwise MI matrix and pre-context PUC matrix for `nodes` on the GPU,
processing genes along the target (`z`) axis in chunks of 256 to bound
device memory use. Moves discretized data and marginal probabilities to
the GPU once, then for each chunk launches
[`joint_counts_kernel_chunked!`](@ref), [`mi_si_kernel_chunked!`](@ref) and
[`puc_accumulation_kernel_chunked!`](@ref) in sequence, symmetrizing the
resulting PUC matrix before returning both matrices to the CPU.
`config.verbose` enables progress printouts; `base` is currently unused
(mutual information is always computed in base 2 on the GPU path).
"""
function FastPIDC.compute_puc_full_cuda(nodes, config, base)
    num_nodes = length(nodes)
    num_samples = length(nodes[1].binned_values)
    k_bins = maximum(n -> n.number_of_bins, nodes)
    
    # Chunk configuration (512 genes at a time)
    chunk_size = 256 
    
    # Prepare static data on CPU and move to GPU
    data_cpu = zeros(Int32, num_samples, num_nodes)
    marginals_cpu = zeros(Float64, k_bins, num_nodes)
    for i = 1:num_nodes
        data_cpu[:, i] .= Int32.(nodes[i].binned_values)
        p = nodes[i].probabilities
        marginals_cpu[1:length(p), i] .= Float64.(p)
    end

    data_gpu = CuArray(data_cpu)
    marginals_gpu = CuArray(marginals_cpu)
    
    # Global output matrices
    puc_scores_gpu = CUDA.zeros(Float64, num_nodes, num_nodes)
    mi_matrix_gpu = CUDA.zeros(Float64, num_nodes, num_nodes)
    
    # Chunked intermediate buffers (Pre-allocated once!)
    counts_chunk_gpu = CUDA.zeros(Int32, k_bins, k_bins, num_nodes, chunk_size)
    si_chunk_gpu = CUDA.zeros(Float64, k_bins, num_nodes, chunk_size)
    
    if config.verbose
        println("[FastPIDC] GPU Chunked PUC: Processing $num_nodes x $num_nodes pairs...")
        println("[FastPIDC] Using chunk size of $chunk_size (approx. $(ceil(Int, num_nodes/chunk_size)) iterations)")
    end
    
    # Iterate over the Z-axis in chunks
    for z_start in 1:chunk_size:num_nodes
        z_end = min(z_start + chunk_size - 1, num_nodes)
        z_curr_chunk_size = z_end - z_start + 1
        
        # Wipe the intermediate buffers clean before the next chunk!
        CUDA.fill!(counts_chunk_gpu, Int32(0))
        CUDA.fill!(si_chunk_gpu, Float64(0))
        
        threads = (16, 16)
        blocks = (cld(num_nodes, 16), cld(z_curr_chunk_size, 16))
        
        @cuda threads=threads blocks=blocks joint_counts_kernel_chunked!(
            data_gpu, counts_chunk_gpu,
            Int32(num_nodes), Int32(num_samples), Int32(k_bins), 
            Int32(z_start), Int32(z_curr_chunk_size)
        )
        
        @cuda threads=threads blocks=blocks mi_si_kernel_chunked!(
            counts_chunk_gpu, marginals_gpu, mi_matrix_gpu, si_chunk_gpu,
            Int32(num_nodes), Int32(num_samples), Int32(k_bins), 
            Int32(z_start), Int32(z_curr_chunk_size)
        )
        
        @cuda threads=threads blocks=blocks puc_accumulation_kernel_chunked!(
            si_chunk_gpu, mi_matrix_gpu, puc_scores_gpu, marginals_gpu,
            Int32(num_nodes), Int32(k_bins), 
            Int32(z_start), Int32(z_curr_chunk_size)
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

end # module
