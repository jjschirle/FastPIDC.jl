"""Stage 4 of the interventional plan: the directed effect matrix E[g, j] --
the distributional shift (energy distance) of every gene j under CRISPRi
perturbation of target g, vs. non-targeting (NTC) control cells.

Energy distance (Szekely & Rizzo), not a mean-shift test, because Perturb-seq
responses are frequently variance changes or bimodality rather than pure mean
shifts (plan's own framing). Implemented with an O((m+n) log(m+n))-per-gene
sort trick (see `energy_distance_batch`), vectorized across genes and
parallelized across targets with multiprocessing -- a naive O(m*n) pairwise
implementation, or a Python-level loop calling scipy.stats.energy_distance
once per (target, gene) pair (150 x ~12,000 = ~1.8M calls), is not tractable
at this scale in an interactive session.

Significance: the plan specifies calibrating against non-targeting-vs-
non-targeting splits rather than a hardcoded parametric null, to absorb
batch/depth structure. A literal permutation p-value with per-target,
per-gene resolution would need thousands of same-size NTC/NTC splits per
target (each costing about as much as one real target's computation) --
not tractable today. Approximation actually used, and why it's still in the
spirit of "empirical, not assumed" calibration: draw NTC/NTC splits at a
handful of representative sample sizes spanning the observed target-group-size
range, several replicates each, to get an empirically *estimated* (not
assumed-functional-form) null mean/SD of energy distance as a function of
group size, then z-score each target's observed per-gene energy distance
against the null level interpolated for that target's actual group size.
This is documented explicitly in LOG.md as an approximation to revisit with
more compute if literal permutation p-values are needed for a specific
downstream claim (e.g. a small shortlist of hits worth a dedicated
permutation test).
"""

from __future__ import annotations

import pickle
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

SCRATCH = Path("/tmp/claude-1000/-home-asm-FastPIDC-jl/564c6d04-ee7a-44e7-b8c1-702a37701a63/scratchpad")
X_PATH = SCRATCH / "X_norm_log.npy"  # (n_cells, n_genes) float32, size-factor normalized + log1p
# (n_genes_filtered, n_cells) float32, contiguous, rows in the same order as the
# gene-filter mask's True entries -- see genemajor_rebuild step in LOG.md. Built
# once from X_PATH so that a contiguous gene-chunk read (this file's natural
# row-major layout) doesn't have to gather ~1,500 scattered columns out of an
# 18,080-wide row per cell, which is what made the first two attempts at this
# script I/O-bound (workers pegged "R" in `ps` but barely accumulating CPU
# time -- symptomatic of mmap page-fault-bound scattered reads, not compute).
X_GENEMAJOR_PATH = SCRATCH / "X_genemajor_filtered.npy"


def energy_distance_batch(
    a: np.ndarray, b: np.ndarray, b_within_mean: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Energy-distance^2 between group `a` (m, G) and group `b` (n, G),
    one value per gene (column), via the merge-rank trick (verified against
    `scipy.stats.energy_distance` on random data: `sqrt(e2) == scipy result`).

    Pass `b_within_mean` (from a prior call) to skip recomputing b's
    within-group term when `b` (e.g. the full NTC pool) is reused across
    many calls.

    Memory note: every array below is shaped ((m+n), G) -- this is the thing
    that blew up a first attempt at parallelizing over 150 targets at full
    gene width (G ~ 12,000): each worker's peak was ~15-20 GB (dominated by
    the fixed n=38,176-cell NTC block, not by the much smaller target group),
    and 46 concurrent workers pushed the machine into swap thrashing and an
    OOM kill. `compute_effect_matrix` now chunks G explicitly so a worker's
    peak is bounded by `chunk_size`, independent of `max_workers`.
    """
    m, G = a.shape
    n = b.shape[0]
    combined = np.concatenate([a, b], axis=0)
    order = np.argsort(combined, axis=0).astype(np.int32)  # (m+n) fits int32; halves this array's footprint vs default int64
    sorted_vals = np.take_along_axis(combined, order, axis=0)
    label = np.concatenate([np.ones(m, dtype=np.int8), np.full(n, -1, dtype=np.int8)])
    sorted_label = label[order]
    is_a = sorted_label == 1
    is_b = ~is_a
    cumB = np.cumsum(is_b, axis=0, dtype=np.int32)
    cumBval = np.cumsum(np.where(is_b, sorted_vals, 0), axis=0)
    totalBval = cumBval[-1]
    contrib = np.where(is_a, sorted_vals * (2 * cumB - n) - 2 * cumBval + totalBval, 0.0)
    mean_cross = contrib.sum(axis=0) / (m * n)

    a_sorted = np.sort(a, axis=0)
    ranks = np.arange(1, m + 1)[:, None]
    mean_within_a = 2 * np.sum((2 * ranks - m - 1) * a_sorted, axis=0) / (m * m)

    if b_within_mean is None:
        b_sorted = np.sort(b, axis=0)
        ranksb = np.arange(1, n + 1)[:, None]
        b_within_mean = 2 * np.sum((2 * ranksb - n - 1) * b_sorted, axis=0) / (n * n)

    e2 = 2 * mean_cross - mean_within_a - b_within_mean
    return e2, b_within_mean


def _worker(args):
    target, cell_indices, ntc_indices, chunk_start, chunk_end, chunk_id = args
    # Genes-major layout: a contiguous row-block read (fast, sequential), then
    # cell selection is fancy-indexing on an already-in-RAM small block instead
    # of scattered reads against the full 16 GB mmap.
    Xg = np.load(X_GENEMAJOR_PATH, mmap_mode="r")
    block = np.asarray(Xg[chunk_start:chunk_end, :])  # (chunk_genes, n_cells), contiguous read
    a = block[:, cell_indices].T  # (m, chunk_genes)
    b = block[:, ntc_indices].T  # (n, chunk_genes)
    e2, _ = energy_distance_batch(a, b)
    return target, chunk_id, e2


def compute_effect_matrix(max_workers: int = 16, chunk_size: int = 1500) -> None:
    targets_obs = pickle.load(open(SCRATCH / "obs_target_gene.pkl", "rb"))
    var_names = pickle.load(open(SCRATCH / "var_names.pkl", "rb"))
    gene_mask = np.load(SCRATCH / "gene_filter_mask.npy")
    gene_cols = np.where(gene_mask)[0]
    filtered_genes = [var_names[i] for i in gene_cols]
    n_genes = len(filtered_genes)
    chunk_bounds = [(i, min(i + chunk_size, n_genes)) for i in range(0, n_genes, chunk_size)]

    ntc_indices = np.where(targets_obs == "non-targeting")[0]
    target_list = sorted(set(targets_obs) - {"non-targeting"})

    jobs = []
    for g in target_list:
        cell_indices = np.where(targets_obs == g)[0]
        for ci, (start, end) in enumerate(chunk_bounds):
            jobs.append((g, cell_indices, ntc_indices, start, end, ci))

    E = np.zeros((len(target_list), len(filtered_genes)), dtype=np.float32)
    target_to_row = {g: i for i, g in enumerate(target_list)}

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(_worker, job) for job in jobs]
        n_done = 0
        for fut in as_completed(futs):
            g, chunk_id, e2 = fut.result()
            start, _ = chunk_bounds[chunk_id]
            E[target_to_row[g], start : start + len(e2)] = e2
            n_done += 1
            if n_done % 50 == 0:
                print(f"[{n_done}/{len(jobs)}] elapsed {time.time() - t0:.1f}s", flush=True)
    print(f"done in {time.time() - t0:.1f}s, {len(jobs)} jobs total")

    np.save(SCRATCH / "E_matrix.npy", E)
    with open(SCRATCH / "E_matrix_meta.pkl", "wb") as f:
        pickle.dump({"target_list": target_list, "filtered_genes": filtered_genes}, f)


if __name__ == "__main__":
    compute_effect_matrix()
