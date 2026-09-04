"""Apply the representative-size NTC/NTC null (`null_calibration.py`) to the
real Stage-4 effect matrix (`effect_matrix.py`): z-score each target's
per-gene energy distance against the null level for its own group size
(log-linear interpolation between the two bracketing representative sizes),
one-sided normal-approximation p-value (energy distance is only ever
*elevated* by a real effect, never suppressed), BH-FDR across all
(target, gene) pairs -- per the interventional plan's Stage 4 instruction
("Benjamini-Hochberg across all (g, j)").

Feeds:
  - companion plan Checkpoint D0: non-degenerate spread in measured
    effective out-degree k^out_g = #{j : E[g,j] significant}.
  - interventional plan Checkpoint 3 (second half): known pluripotency
    edges / hub detectability, now with an actual (approximately)
    calibrated effect matrix rather than the Checkpoint-0 mean-shift proxy.
"""

from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from effect_matrix import SCRATCH


def load_all(
    e_matrix_name: str = "E_matrix.npy",
    e_meta_name: str = "E_matrix_meta.pkl",
    null_mean_name: str = "null_mean_by_size.npy",
    null_sd_name: str = "null_sd_by_size.npy",
    null_meta_name: str = "null_meta.pkl",
):
    E = np.load(SCRATCH / e_matrix_name)
    np.clip(E, 0, None, out=E)  # tiny negative floating-point noise near zero
    meta = pickle.load(open(SCRATCH / e_meta_name, "rb"))
    target_list = meta["target_list"]
    genes = meta["filtered_genes"]

    null_mean = np.load(SCRATCH / null_mean_name)
    null_sd = np.load(SCRATCH / null_sd_name)
    null_meta = pickle.load(open(SCRATCH / null_meta_name, "rb"))
    sizes = np.array(null_meta["sizes"], dtype=np.float64)

    targets_obs = pickle.load(open(SCRATCH / "obs_target_gene.pkl", "rb"))
    n_pert = {g: int((targets_obs == g).sum()) for g in target_list}
    return E, target_list, genes, null_mean, null_sd, sizes, n_pert


def interpolate_null(n: int, sizes: np.ndarray, null_mean: np.ndarray, null_sd: np.ndarray):
    """Log-log interpolate the null mean/SD (per gene) to sample size n.
    Energy distance's null scales roughly like 1/n (verified empirically:
    mean null e2 drops ~1/size across the tested range) so interpolating
    log(null) vs log(size) is closer to linear than a raw linear interp."""
    n = np.clip(n, sizes.min(), sizes.max())
    log_sizes = np.log(sizes)
    log_n = np.log(n)
    i = np.searchsorted(log_sizes, log_n, side="right")
    i = np.clip(i, 1, len(sizes) - 1)
    lo, hi = i - 1, i
    t = (log_n - log_sizes[lo]) / (log_sizes[hi] - log_sizes[lo])

    def interp(arr):
        log_arr = np.log(np.clip(arr, 1e-12, None))
        return np.exp(log_arr[lo] * (1 - t) + log_arr[hi] * t)

    return interp(null_mean), interp(null_sd)


def calibrate(
    load_kwargs: dict | None = None,
    q_name: str = "E_qvalues.npy",
    z_name: str = "E_zscores.npy",
    kout_name: str = "k_out.csv",
):
    E, target_list, genes, null_mean, null_sd, sizes, n_pert = load_all(**(load_kwargs or {}))
    gi = {g: i for i, g in enumerate(genes)}

    Z = np.zeros_like(E)
    for ti, g in enumerate(target_list):
        mu, sd = interpolate_null(n_pert[g], sizes, null_mean, null_sd)
        Z[ti] = (E[ti] - mu) / sd

    p = stats.norm.sf(Z)  # one-sided: only elevated energy distance is "significant"
    q_flat = multipletests(p.ravel(), method="fdr_bh")[1]
    Q = q_flat.reshape(p.shape)

    np.save(SCRATCH / q_name, Q)
    np.save(SCRATCH / z_name, Z)

    # measured effective out-degree per target, excluding the target's own gene
    k_out = {}
    for ti, g in enumerate(target_list):
        sig = Q[ti] < 0.05
        if g in gi:
            sig = sig.copy()
            sig[gi[g]] = False
        k_out[g] = int(sig.sum())

    df = pd.DataFrame({"target": target_list, "n_pert": [n_pert[g] for g in target_list], "k_out": [k_out[g] for g in target_list]})
    df.to_csv(SCRATCH / kout_name, index=False)

    print("=== Checkpoint D0 (companion plan): measured k^out_g spread ===")
    print(df["k_out"].describe())
    print("n targets with k_out == 0:", (df["k_out"] == 0).sum())
    print("n targets with k_out >= 10:", (df["k_out"] >= 10).sum())
    print()
    print("Correlation of k_out with n_pert (power confound check):", df["k_out"].corr(df["n_pert"]))
    print()
    print("Top 10 by k_out:")
    print(df.sort_values("k_out", ascending=False).head(10))
    print()
    print("Bottom 10 by k_out:")
    print(df.sort_values("k_out").head(10))

    # pluripotency / hub check (interventional Checkpoint 3, second half)
    print()
    print("=== Pluripotency / hub check ===")
    for g in ["POU5F1", "NANOG", "SOX2"]:
        row = df[df["target"] == g]
        if len(row):
            print(f"{g} (perturbation target): k_out = {row['k_out'].values[0]}, n_pert = {row['n_pert'].values[0]}")
        else:
            print(f"{g}: not a perturbation target (measured only)")
    if "SOX2" in target_list:
        ti = target_list.index("SOX2")
        for other in ["POU5F1", "NANOG"]:
            if other in gi:
                q = Q[ti, gi[other]]
                e = E[ti, gi[other]]
                print(f"SOX2 -> {other}: q = {q:.3g}, energy_d^2 = {e:.4g}")


if __name__ == "__main__":
    calibrate()
