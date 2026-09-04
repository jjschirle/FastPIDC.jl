"""Step 3 of STATE.md's "immediate next actions": install/validate the cupy
CUDA backend before attempting a full-scale PUC run (interventional plan
Stage 3, an O(N^3) computation -- strictly worse than Stage 4's O(N log N)
energy distance, which already needed real memory/layout care at N ~ 12,000
genes, see LOG.md).

Two checks:
1. Correctness: CPU (`compute_puc_full`) vs CUDA (`compute_puc_full_cuda`)
   backends must agree on a small synthetic case.
2. Timing/memory at representative scale: one chunk-sized slice of node
   counts close to what a real gene-count run would look like, to size a
   full run before committing to it (same discipline as Stage 4's memory
   check -- see LOG.md's "check peak memory on one representative unit of
   work before fanning out, not after").
"""

from __future__ import annotations

import time

import numpy as np

from fastpidc.puc import compute_puc_full
from fastpidc.cuda import compute_puc_full_cuda, cuda_available
from fastpidc.types import Node, PIDCConfig


def make_synthetic_nodes(n_nodes: int, n_cells: int, n_bins: int, seed: int = 0) -> list[Node]:
    rng = np.random.default_rng(seed)
    # Correlated-ish synthetic data: each node is a noisy function of a shared
    # latent plus its own noise, so MI/PUC aren't all trivially zero.
    latent = rng.normal(size=n_cells)
    nodes = []
    for i in range(n_nodes):
        raw = 0.4 * latent + rng.normal(size=n_cells)
        binned = np.clip((np.argsort(np.argsort(raw)) * n_bins) // n_cells, 0, n_bins - 1).astype(np.int64)
        counts = np.bincount(binned, minlength=n_bins).astype(np.float64)
        probs = counts / counts.sum()
        nodes.append(Node(label=str(i), binned_values=binned, number_of_bins=n_bins, probabilities=probs))
    return nodes


def correctness_check():
    print("=== Correctness: CPU vs CUDA on a small synthetic case ===")
    nodes = make_synthetic_nodes(n_nodes=40, n_cells=2000, n_bins=6)
    mi_cpu, puc_cpu = compute_puc_full(nodes, estimator="maximum_likelihood", base=2)
    mi_gpu, puc_gpu = compute_puc_full_cuda(nodes, base=2, verbose=False)

    mi_diff = np.abs(mi_cpu - mi_gpu).max()
    puc_diff = np.abs(puc_cpu - puc_gpu).max()
    print(f"max|MI_cpu - MI_gpu|   = {mi_diff:.3e}")
    print(f"max|PUC_cpu - PUC_gpu| = {puc_diff:.3e}")
    print(f"MI allclose (atol=1e-6): {np.allclose(mi_cpu, mi_gpu, atol=1e-6)}")
    print(f"PUC allclose (atol=1e-6): {np.allclose(puc_cpu, puc_gpu, atol=1e-6)}")
    return mi_diff, puc_diff


def timing_memory_profile(cases: list[tuple[int, int]], n_bins: int, chunk_size: int = 256):
    print()
    print("=== Timing / memory profile (CUDA backend) ===")
    import cupy as cp

    mempool = cp.get_default_memory_pool()
    for n_nodes, n_cells in cases:
        nodes = make_synthetic_nodes(n_nodes=n_nodes, n_cells=n_cells, n_bins=n_bins)
        mempool.free_all_blocks()
        free0, total0 = cp.cuda.runtime.memGetInfo()

        t0 = time.time()
        mi, puc = compute_puc_full_cuda(nodes, base=2, verbose=False, chunk_size=chunk_size)
        cp.cuda.Stream.null.synchronize()
        elapsed = time.time() - t0

        # total_bytes() = pool's reserved size right after the call, before
        # the next free_all_blocks() -- approximates this call's peak device
        # allocation (cupy's pool doesn't shrink until freed).
        peak_reserved = mempool.total_bytes()
        free1, _ = cp.cuda.runtime.memGetInfo()
        print(
            f"n_nodes={n_nodes:6d} n_cells={n_cells:7d} n_bins={n_bins} chunk={chunk_size}: "
            f"{elapsed:.3f}s, pool reserved={peak_reserved / 1e9:.2f} GB, "
            f"device free before/after={free0/1e9:.2f}/{free1/1e9:.2f} GB"
        )
        del nodes, mi, puc


if __name__ == "__main__":
    if not cuda_available():
        raise SystemExit("cupy / GPU not available")
    correctness_check()
    # Representative sizes: the real Stage-3 run is ~11,942 genes x ~221,273
    # cells. Probe increasing N at real (or near-real) cell count -- cell
    # count dominates device memory via the (m, n) binned-data array and
    # dominates compute via the joint_counts_kernel's O(n^2 * m / chunk)
    # inner loop -- rather than a small n_cells stand-in, to get a timing/
    # memory extrapolation that's actually trustworthy at Stage-3 scale.
    timing_memory_profile(
        cases=[(512, 221_273), (1024, 221_273), (2048, 221_273)],
        n_bins=10,
        chunk_size=256,
    )
