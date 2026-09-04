"""Significance calibration for the Stage-4 effect matrix (`effect_matrix.py`),
per the interventional plan: "Calibrate significance against non-targeting-
vs-non-targeting splits, not a parametric null -- this absorbs batch and
depth structure."

A literal permutation p-value with per-target, per-gene resolution would need
many same-size NTC/NTC splits *for every one of the 150 targets' exact group
sizes* -- each split costs about as much as one real target's computation
(dominated by the fixed ~38k-cell NTC pool, not the smaller draw), so that's
not tractable in an interactive session (it would roughly multiply Stage 4's
~1000s runtime by however many splits-per-target resolution requires, e.g.
20-50x). What's implemented instead: NTC/NTC splits at a handful of
*representative* sample sizes spanning the actually-observed target-group-size
range (33 to 4760 cells), several replicates each, giving an *empirically
estimated* (not assumed-functional-form) null mean/SD of energy distance as a
function of group size. Each real target is then scored against the null
level for its nearest representative size. This is an approximation to the
plan's literal instruction, made explicit here and in LOG.md -- revisit with
per-target-exact-size permutation if a specific downstream claim needs it
(e.g. a short list of top hits worth a dedicated test).
"""

from __future__ import annotations

import pickle
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from effect_matrix import SCRATCH, X_GENEMAJOR_PATH, energy_distance_batch

REPRESENTATIVE_SIZES = [50, 200, 500, 1000, 2000, 3000, 4500]
N_REPLICATES = 8
CHUNK_SIZE = 1500


def _null_worker(args):
    size, rep_seed, chunk_start, chunk_end, chunk_id = args
    targets_obs = pickle.load(open(SCRATCH / "obs_target_gene.pkl", "rb"))
    ntc_indices = np.where(targets_obs == "non-targeting")[0]

    rng = np.random.default_rng(rep_seed)
    perm = rng.permutation(ntc_indices)
    pseudo_target = perm[:size]
    pseudo_control = perm[size:]  # rest of the NTC pool

    Xg = np.load(X_GENEMAJOR_PATH, mmap_mode="r")
    block = np.asarray(Xg[chunk_start:chunk_end, :])
    a = block[:, pseudo_target].T
    b = block[:, pseudo_control].T
    e2, _ = energy_distance_batch(a, b)
    return size, rep_seed, chunk_id, e2


def compute_null(max_workers: int = 16) -> None:
    gene_mask = np.load(SCRATCH / "gene_filter_mask.npy")
    n_genes = int(gene_mask.sum())
    chunk_bounds = [(i, min(i + CHUNK_SIZE, n_genes)) for i in range(0, n_genes, CHUNK_SIZE)]

    jobs = []
    for size in REPRESENTATIVE_SIZES:
        for rep in range(N_REPLICATES):
            seed = hash((size, rep)) % (2**31)
            for ci, (start, end) in enumerate(chunk_bounds):
                jobs.append((size, seed, start, end, ci))

    # e2_by_size[size] -> (N_REPLICATES, n_genes) array
    e2_by_size = {s: np.zeros((N_REPLICATES, n_genes), dtype=np.float32) for s in REPRESENTATIVE_SIZES}
    rep_index = {s: {} for s in REPRESENTATIVE_SIZES}  # seed -> rep row index, assigned on first sight

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(_null_worker, job) for job in jobs]
        n_done = 0
        for fut in as_completed(futs):
            size, seed, chunk_id, e2 = fut.result()
            if seed not in rep_index[size]:
                rep_index[size][seed] = len(rep_index[size])
            row = rep_index[size][seed]
            start, _ = chunk_bounds[chunk_id]
            e2_by_size[size][row, start : start + len(e2)] = e2
            n_done += 1
            if n_done % 50 == 0:
                print(f"[{n_done}/{len(jobs)}] elapsed {time.time() - t0:.1f}s", flush=True)
    print(f"null calibration done in {time.time() - t0:.1f}s")

    null_mean = np.stack([e2_by_size[s].mean(axis=0) for s in REPRESENTATIVE_SIZES])  # (n_sizes, n_genes)
    null_sd = np.stack([e2_by_size[s].std(axis=0, ddof=1) for s in REPRESENTATIVE_SIZES])

    np.save(SCRATCH / "null_mean_by_size.npy", null_mean)
    np.save(SCRATCH / "null_sd_by_size.npy", null_sd)
    with open(SCRATCH / "null_meta.pkl", "wb") as f:
        pickle.dump({"sizes": REPRESENTATIVE_SIZES, "n_replicates": N_REPLICATES}, f)


if __name__ == "__main__":
    compute_null()
