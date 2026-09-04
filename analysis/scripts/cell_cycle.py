"""Step 1 of STATE.md's "immediate next actions": condition Stage 4's effect
matrix on cell-cycle/cell-state before trusting the companion plan's k^out_g
degree prior (Checkpoint D0). LOG.md's own analysis of the Stage-4 run found
some targets "significant" against ~97% of the filtered genome -- almost
certainly hESC cell-cycle/state composition shifting under perturbation, not
150 independent direct regulatory programs, exactly the risk the interventional
plan's Checkpoint 0 "Secondary concern" flagged in advance.

Two-part fix, both implemented here:

1. Score every cell for S-phase and G2M-phase activity using the standard
   Tirosh et al. / Seurat cc.genes.updated.2019 marker lists, via the same
   algorithm as scanpy.tl.score_genes / Seurat AddModuleScore: each cell's
   score for a gene set is (mean expression of the set) minus (mean expression
   of a size-matched control gene set drawn from the same expression-level
   bins), computed directly here rather than pulling in scanpy as a dependency
   for one function.
2. Regress S_score and G2M_score out of the gene-major filtered expression
   matrix (linear regression per gene, vectorized across all genes at once via
   the normal equations -- NOT a per-gene Python loop, which would be ~12,000
   separate lstsq calls) and write the residualized matrix. Stage 4's
   `effect_matrix.py` / `null_calibration.py` / `calibrate_effects.py` are then
   re-run against the residualized array (see `rerun_stage4_residualized.py`)
   to get a cell-state-conditioned k^out_g.

Vectorized regression trick (avoids transposing the 11,942 x 221,273 gene-major
array, which would cost ~11 GB and a slow transpose): for design matrix D
(n_cells x 3: [intercept, S_score, G2M_score]) and gene-major data M
(n_genes x n_cells), the per-gene OLS coefficients are
    B = (D^T D)^-1 (M D)^T
because D^T M^T = (M D)^T -- so the only large matmul needed is M @ D
(n_genes x 3), not M^T (n_cells x n_genes). Fitted values are then B^T @ D^T.
"""

from __future__ import annotations

import pickle

import numpy as np

from effect_matrix import SCRATCH, X_GENEMAJOR_PATH

X_NORM_LOG_PATH = SCRATCH / "X_norm_log.npy"  # (n_cells, n_genes) full 18,080-gene set

# Seurat cc.genes.updated.2019 (Tirosh et al. 2016 cell-cycle marker lists),
# public/standard gene symbol lists shipped with Seurat -- not project-specific.
S_GENES = [
    "MCM5", "PCNA", "TYMS", "FEN1", "MCM2", "MCM4", "RRM1", "UNG", "GINS2",
    "MCM6", "CDCA7", "DTL", "PRIM1", "UHRF1", "MLF1IP", "HELLS", "RFC2",
    "RPA2", "NASP", "RAD51AP1", "GMNN", "WDR76", "SLBP", "CCNE2", "UBR7",
    "POLD3", "MSH2", "ATAD2", "RAD51", "RRM2", "CDC45", "CDC6", "EXO1",
    "TIPIN", "DSCC1", "BLM", "CASP8AP2", "USP1", "CLSPN", "POLA1", "CHAF1B",
    "BRIP1", "E2F8",
]
G2M_GENES = [
    "HMGB2", "CDK1", "NUSAP1", "UBE2C", "BIRC5", "TPX2", "TOP2A", "NDC80",
    "CKS2", "NUF2", "CKS1B", "MKI67", "TMPO", "CENPF", "TACC3", "PIMREG",
    "SMC4", "CCNB2", "CKAP2L", "CKAP2", "AURKB", "BUB1", "KIF11", "ANP32E",
    "TUBB4B", "GTSE1", "KIF20B", "HJURP", "CDCA3", "JPT1", "CDC20", "TTK",
    "CDC25C", "KIF2C", "RANGAP1", "NCAPD2", "DLGAP5", "CDCA2", "CDCA8",
    "ECT2", "KIF23", "HMMR", "AURKA", "PSRC1", "ANLN", "LBR", "CKAP5",
    "CENPE", "CTCF", "NEK2", "G2E3", "GAS2L3", "CBX5", "CENPA",
]

N_BINS = 25
CTRL_SIZE = 50


def _score_gene_set(X: np.ndarray, gene_idx: np.ndarray, bin_of_gene: np.ndarray,
                     genes_by_bin: dict[int, np.ndarray], rng: np.random.Generator) -> np.ndarray:
    """scanpy/Seurat-style module score: mean(set) - mean(size-matched control)."""
    ctrl_idx = []
    for g in gene_idx:
        b = bin_of_gene[g]
        pool = genes_by_bin[b]
        take = min(CTRL_SIZE, len(pool))
        ctrl_idx.append(rng.choice(pool, size=take, replace=False))
    ctrl_idx = np.unique(np.concatenate(ctrl_idx))
    return X[:, gene_idx].mean(axis=1) - X[:, ctrl_idx].mean(axis=1)


def compute_cc_scores(seed: int = 0) -> None:
    var_names = pickle.load(open(SCRATCH / "var_names.pkl", "rb"))
    name_to_idx = {n: i for i, n in enumerate(var_names)}

    s_idx = np.array([name_to_idx[g] for g in S_GENES if g in name_to_idx])
    g2m_idx = np.array([name_to_idx[g] for g in G2M_GENES if g in name_to_idx])
    print(f"S genes found: {len(s_idx)}/{len(S_GENES)}; G2M genes found: {len(g2m_idx)}/{len(G2M_GENES)}")

    X = np.load(X_NORM_LOG_PATH)  # load fully into RAM (16 GB) -- avoid scattered mmap reads (LOG.md lesson)

    avg_expr = X.mean(axis=0)  # (n_genes,)
    bin_of_gene = np.clip((np.argsort(np.argsort(avg_expr)) * N_BINS) // len(avg_expr), 0, N_BINS - 1)
    genes_by_bin = {b: np.where(bin_of_gene == b)[0] for b in range(N_BINS)}

    rng = np.random.default_rng(seed)
    s_score = _score_gene_set(X, s_idx, bin_of_gene, genes_by_bin, rng)
    g2m_score = _score_gene_set(X, g2m_idx, bin_of_gene, genes_by_bin, rng)
    del X

    phase = np.full(len(s_score), "G1", dtype=object)
    is_cycling = (s_score > 0) | (g2m_score > 0)
    phase[is_cycling & (s_score >= g2m_score)] = "S"
    phase[is_cycling & (g2m_score > s_score)] = "G2M"

    np.save(SCRATCH / "cc_s_score.npy", s_score.astype(np.float32))
    np.save(SCRATCH / "cc_g2m_score.npy", g2m_score.astype(np.float32))
    with open(SCRATCH / "cc_phase.pkl", "wb") as f:
        pickle.dump(phase, f)

    print("Phase counts:", {p: int((phase == p).sum()) for p in ["G1", "S", "G2M"]})
    print("S_score range:", s_score.min(), s_score.max(), "mean", s_score.mean())
    print("G2M_score range:", g2m_score.min(), g2m_score.max(), "mean", g2m_score.mean())


def regress_out_cc() -> None:
    s_score = np.load(SCRATCH / "cc_s_score.npy")
    g2m_score = np.load(SCRATCH / "cc_g2m_score.npy")
    n_cells = len(s_score)

    D = np.column_stack([np.ones(n_cells), s_score, g2m_score]).astype(np.float64)  # (N, 3)
    DtD = D.T @ D
    DtD_inv = np.linalg.inv(DtD)

    M = np.load(X_GENEMAJOR_PATH)  # (G, N) float32, loaded fully into RAM (~11 GB)
    MD = M.astype(np.float64) @ D  # (G, 3) -- cheap; avoids transposing M
    B = DtD_inv @ MD.T  # (3, G)
    fitted = (B.T @ D.T).astype(np.float32)  # (G, N)
    resid = M - fitted
    del M, fitted

    out_path = SCRATCH / "X_resid_genemajor.npy"
    np.save(out_path, resid)
    print(f"Residualized matrix written to {out_path}, shape {resid.shape}")
    print("Residual variance retained (mean per-gene R^2 removed):",
          float(1 - resid.var(axis=1).mean() / np.load(X_GENEMAJOR_PATH, mmap_mode="r").var(axis=1).mean()))


if __name__ == "__main__":
    compute_cc_scores()
    regress_out_cc()
