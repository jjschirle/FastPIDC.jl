// Shared CUDA kernels for PIDC/PUC network inference.
//
// This is the canonical GPU implementation of the chunked PUC algorithm used
// by FastPIDC: for each chunk of "target" genes z, it (1) accumulates joint
// bin-count histograms between every source gene x and each target z in the
// chunk, (2) turns those histograms into pairwise mutual information and
// specific-information values, and (3) accumulates the resulting
// Proportional Unique Contribution (PUC) scores.
//
// It is written in plain CUDA C (no Julia- or Python-specific glue) so it
// is compiled and launched from either language's GPU bindings, both at
// runtime, from this one file:
//   * Python: loaded via `cupy.RawModule` (see `fastpidc.cuda`), which
//     invokes nvrtc to compile this source.
//   * Julia: the FastPIDC.jl CUDA extension (`ext/FastPIDCCUDAExt`) shells
//     out to `nvcc --ptx` to compile this source for the active device's
//     compute capability, then loads the result with `CUDA.CuModule` and
//     drives it with `CUDA.cudacall`.
//
// Both hosts pass 0-indexed bin ids and marginals in the shapes documented
// per kernel below; FastPIDC.jl's bin ids are 1-indexed internally, so its
// host code shifts them down by one before upload (and un-shifts nothing
// on the way back out, since only float scores cross that boundary).
//
// All arrays are indexed 0-based, row-major (C order), and passed as flat
// buffers with the shapes documented per kernel.

extern "C" {

// data:        (m, n) int32         -- data[s * n + x] = bin id of gene x, sample s
// counts:      (k_bins, k_bins, n, chunk) int32, zero-initialized by the caller
//              counts[((u * k_bins + v) * n + x) * chunk + z_local]
// One thread handles one (x, z_local) pair, looping over all m samples.
__global__ void joint_counts_kernel(
    const int* __restrict__ data,
    int* counts,
    int n, int m, int k_bins,
    int z_start, int z_chunk_size)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int z_local = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= n || z_local >= z_chunk_size) return;

    int z_global = z_start + z_local;
    if (z_global >= n || x == z_global) return;

    for (int s = 0; s < m; ++s) {
        int u = data[s * n + x];
        int v = data[s * n + z_global];
        if (u >= 0 && u < k_bins && v >= 0 && v < k_bins) {
            int idx = ((u * k_bins + v) * n + x) * z_chunk_size + z_local;
            atomicAdd(&counts[idx], 1);
        }
    }
}

// counts:      (k_bins, k_bins, n, chunk) int32, as produced above
// marginals:   (k_bins, n) float64        -- marginals[v * n + x] = P(gene x == bin v)
// mi_matrix:   (n, n) float64             -- mi_matrix[x * n + z]
// si_matrix:   (k_bins, n, chunk) float64 -- si_matrix[(v * n + x) * chunk + z_local]
//              specific information of source x with respect to target z_global,
//              at target bin v.
// One thread handles one (x, z_local) pair.
__global__ void mi_si_kernel(
    const int* __restrict__ counts,
    const double* __restrict__ marginals,
    double* mi_matrix,
    double* si_matrix,
    int n, int m, int k_bins,
    int z_start, int z_chunk_size)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int z_local = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= n || z_local >= z_chunk_size) return;

    int z_global = z_start + z_local;
    if (z_global >= n || x == z_global) return;

    double inv_m = 1.0 / (double)m;
    double mi_val = 0.0;

    for (int v = 0; v < k_bins; ++v) {
        double p_z_v = marginals[v * n + z_global];
        if (p_z_v <= 0.0) continue;

        double si_v = 0.0;
        for (int u = 0; u < k_bins; ++u) {
            double p_x_u = marginals[u * n + x];
            if (p_x_u <= 0.0) continue;

            int c_uv = counts[((u * k_bins + v) * n + x) * z_chunk_size + z_local];
            double p_uv = (double)c_uv * inv_m;

            if (p_uv > 0.0) {
                mi_val += p_uv * log2(p_uv / (p_x_u * p_z_v));
                double p_u_cond_v = p_uv / p_z_v;
                si_v += p_u_cond_v * log2(p_u_cond_v / p_x_u);
            }
        }
        si_matrix[(v * n + x) * z_chunk_size + z_local] = si_v;
    }

    mi_matrix[x * n + z_global] = mi_val;
}

// si_matrix:   (k_bins, n, chunk) float64, as produced above
// mi_matrix:   (n, n) float64
// puc_scores:  (n, n) float64 -- puc_scores[x * n + z_global] (one direction only;
//              the caller must symmetrize puc_scores[i,j] + puc_scores[j,i])
// marginals:   (k_bins, n) float64
// One thread handles one (x, z_local) pair, looping internally over source y.
__global__ void puc_accumulation_kernel(
    const double* __restrict__ si_matrix,
    const double* __restrict__ mi_matrix,
    double* puc_scores,
    const double* __restrict__ marginals,
    int n, int k_bins,
    int z_start, int z_chunk_size)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int z_local = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= n || z_local >= z_chunk_size) return;

    int z_global = z_start + z_local;
    if (z_global >= n || x == z_global) return;

    double mi_xz = mi_matrix[x * n + z_global];
    if (mi_xz <= 1e-12) return;

    double local_puc = 0.0;
    for (int y = 0; y < n; ++y) {
        if (y == x || y == z_global) continue;

        double redundancy = 0.0;
        for (int k = 0; k < k_bins; ++k) {
            double p_z_k = marginals[k * n + z_global];
            if (p_z_k <= 0.0) continue;

            double si_x = si_matrix[(k * n + x) * z_chunk_size + z_local];
            double si_y = si_matrix[(k * n + y) * z_chunk_size + z_local];
            redundancy += p_z_k * fmin(si_x, si_y);
        }

        double score = (mi_xz - redundancy) / mi_xz;
        if (isfinite(score) && score > 0.0) {
            local_puc += score;
        }
    }

    puc_scores[x * n + z_global] = local_puc;
}

} // extern "C"
