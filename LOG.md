# Analysis log

Working log for evaluating `Interventional PID for GRN inference—testing the gap with FastPIDC.jl on Arc H1 hESC Perturb-seq.md`
(**interventional plan**) and `Degree-aware recalibration of PIDC—a companion plan.md` (**companion plan**) against real data.
Branch: `perturbation-analysis`. See `STATE.md` for the checklist view of plan items.

---

## 2026-09-04 — Environment setup

- Created branch `perturbation-analysis` off `analysis`.
- `python/` (the `fastpidc` package) is left untouched for now; its `requires-python = ">=3.10"`
  conflicts with `vsparse==0.2.0`'s `>=3.12` floor, so analysis dependencies are **not** added
  there. Instead: `analysis/pyproject.toml`, a `package = false` uv project (Python ≥3.12) that
  pulls in `fastpidc` (editable, `../python`) and `vsparse` (editable, `/home/asm/vsparse`) as
  path sources, plus `anndata`, `pandas`, `statsmodels`, `matplotlib`. Run everything from
  `analysis/` with `uv run python ...`.
  - **Caveat:** the `vsparse` path source is an absolute local path (`/home/asm/vsparse`), not a
    pinned release — fine for exploratory work on this machine, but not reproducible elsewhere.
    Revisit once/if `vsparse` cuts a real release.
- Loading `hESC.h5ad` (and presumably `K562-genome-wide.h5ad`) requires, in this order:
  `import hdf5plugin` (registers the HDF5 filter plugin directory — without it, reads fail with
  `OSError: can't open directory (/usr/local/lib/plugin)`), then `import vsparse` (registers the
  `ivcsr`/`ivcsc` AnnData element readers), then load with
  **`vsparse.VCSCAnnData.read_h5ad(path)`** — plain `anndata.read_h5ad` fails: current `anndata`
  rejects a bare `VCSRArray`/`VCSCArray` as a valid `X` type at the top-level constructor, even
  though vsparse's element reader for `/X` works fine. `VCSCAnnData` wraps that correctly.
- Machine: 188 GiB RAM, RTX 4090 present but **`cupy` not installed** anywhere on the system —
  `fastpidc`'s CUDA backend is unusable until that's added. `PIDCConfig.backend` defaults to
  `"cuda"`, so every call in this analysis needs `backend="cpu"` explicit until cupy is set up.

## 2026-09-04 — FastPIDC.jl Python API: resolving the plans' open questions

Both plans flagged their assumed API as unverified (interventional plan §2, companion plan
item 10.4). Read `python/src/fastpidc/{api,network,puc,information,types}.py` directly. Findings:

1. **No dense/sparse/edge-list ambiguity** — `infer_network_from_nodes` returns an
   `InferredNetwork(nodes, edges)`; `edges` is a flat sorted list of `Edge(nodes, weight)`, not a
   matrix. Full PUC/MI as matrices *are* available separately (see #3).
2. **Discretization** happens inside `Node.from_raw_values` / `get_nodes`, not accepted as
   pre-binned integers through the main entry point — but `Node` itself is a public dataclass
   with `binned_values` already binned, so nodes can be constructed directly, bypassing the
   provided discretizers if we bin ourselves (relevant for Stage 2's custom
   zero-as-own-bin discretizer, which doesn't match any of `discretizers.py`'s built-ins and
   will need to be added or built externally and wrapped into `Node` objects by hand).
3. **Per-gene PUC vectors and disabling calibration — both already supported, no patch needed.**
   `PUCNetworkInference` (`apply_context=False, get_puc=True`) returns *raw, uncalibrated* PUC as
   edge weights — this **is** the `calibrate=false` companion plan item 10.4 asked to confirm.
   The full dense PUC matrix is also directly accessible via `compute_puc_full(nodes, ...)` →
   `(mi_scores, puc_scores)`, or dumped to disk with `PIDCConfig(dump_puc_path=...)`. A gene's PUC
   vector is just a row of that matrix. This unblocks companion-plan V2/V1/V3/V4 immediately —
   §2's per-gene gamma work operates on `puc_scores` rows before `get_weights()`'s gamma-fit step
   ever runs.
4. **Redundancy measure**: only `I_min` (`apply_redundancy_formula` = `E[min(SI_1, SI_2)]`),
   matching interventional plan's own flagged weakness ("`I_min` is the weakest link", Checkpoint
   4). No BROJA or `I_ccs` implementation exists in this package — Checkpoint 4's
   redundancy-measure robustness check will need an external PID library (e.g. `dit`) or a new
   implementation; out of scope to add speculatively.
5. **`pid_triple`-equivalent (interventional plan's load-bearing primitive, §2 item 3): does
   NOT exist.** There is no function that, given an arbitrary triple `(source1, source2, target)`,
   returns `(redundancy, unique1, unique2, synergy)`. What exists are the pieces it would be built
   from: `network.get_mi_and_si(node1, node2, estimator, base)` (pairwise MI + specific
   information) and `information.apply_redundancy_formula(p_z, si1, si2, base)`. From these,
   for a triple `(Ig, X, Y)`: `redundancy = apply_redundancy_formula(p(Y), SI(Ig;Y), SI(X;Y))`,
   `unique(Ig→Y|X) = MI(Ig,Y) - redundancy`, `unique(X→Y|Ig) = MI(X,Y) - redundancy`. Synergy
   needs the joint MI of `(Ig,X)` on `Y`, which requires a joint-binned "combined" node (bin id =
   `Ig * n_bins_X + X`) fed through the same `get_mi_and_si` machinery — buildable, not present.
   **Action taken:** none yet on the package itself; per the interventional plan's own instruction
   ("if `pid_triple` or equivalent does not exist, stop and add it — it is the core of the
   contribution"), this is the first code change needed before Stage 6 (H2) can run, and will be
   added as a new `fastpidc.pid` module on this branch before that stage starts. Not needed for
   Checkpoint 0 or the companion plan's V2, so it did not block today's work.
6. **Threading/CUDA-loop safety**: not evaluated yet (no cupy installed to test against); the CPU
   path is plain NumPy with Python-level loops over node pairs, no threading primitives visible,
   so nothing to trip over there. Revisit when the CUDA extra is actually installed.

## 2026-09-04 — Checkpoint 0 (interventional plan): reciprocal-pair count — RUN FIRST

This is the plan's own designated first action, before any hypothesis test proper. Ran directly
against `/opt/IVCSC/hESC.h5ad` (chosen over `K562-genome-wide.h5ad` per instruction: hESC has
much better read depth despite narrower breadth, and is the plan's primary dataset).

**Data reality vs. the plan's assumptions — first discrepancy found:**
- Plan text says "~300 CRISPRi perturbations." Actual file has **150** distinct `target_gene`
  values (excluding `non-targeting`), not ~300. `obs` has `target_gene` (categorical, 151
  categories incl. NTC), `guide_id`, `batch`. 221,273 cells × 18,080 genes; 38,176 NTC
  (non-targeting) cells. This roughly halves every pair-count estimate in both plans and should be
  corrected wherever "300 targets" appears in method text later (companion plan §2's `C(300,2) =
  44,850` becomes `C(150,2) = 11,175`; the "~9,800 targets" Replogle comparison in interventional
  plan §1 is unaffected, that's a different dataset).

**Checks run** (script logic below; not yet committed as a package script — see `STATE.md`):
1. Cells per target: **min 33, median 1045, max 4760** — all 150 targets clear the plan's own
   power floor (`n_pert >= 30`, interventional plan Stage 4) with room to spare; none dropped.
2. All 150 target-gene symbols are present in `var_names` (i.e., measurable as response genes,
   not just guide labels) — no targets lost to a target/measured-gene mismatch.
3. Self-knockdown effect (Checkpoint 3's positive control, run early since it's nearly free once
   the above is done): size-factor-normalized + log1p expression of each target gene, perturbed
   vs. NTC cells, one-sided Mann–Whitney (knockdown ⇒ down), BH-FDR across the 150 tests.
   **All 150/150 significant at q<0.05 and log2FC<0.** Effect sizes vary a lot though — log2FC
   ranges from −4.98 (TMSB4X) to −0.16 (KAT2A); the weak end is "significant because n is huge,"
   not "large knockdown," and should not be over-read as uniformly strong CRISPRi efficiency.
   Full table saved to the session scratchpad as `self_effect_screen.csv` (not committed — see
   "Scratch outputs" below).
4. Downstream phenotype detectability (proxy for the plan's energy-distance screen — see caveat
   below): per target, Welch mean-shift z-test of every one of the 18,080 genes vs. NTC,
   BH-FDR<0.05, excluding the target's own gene. **All 150/150 targets clear even a generous bar**
   (min significant-gene count across targets = 153, vs. a nominal ≥5-gene bar) — no targets look
   like "no detectable phenotype" cases the way a meaningful fraction of Replogle targets reportedly
   were.
   - **Caveat, stated plainly:** this is a linear mean-shift proxy, not the energy distance the
     interventional plan specifies for the real Stage-4 effect matrix (Perturb-seq responses are
     "frequently variance changes or bimodality," which mean-shift can miss). It was chosen only
     because it's cheap enough to run genome-wide today; it is **not** a substitute for the real
     Stage 4 computation, which should use `scipy`/`dcor` energy distance as specified. At >50k
     UMI/cell and hundreds-to-thousands of cells per group, this test has enormous power — "q<0.05"
     on its own is a very low bar here and should not be read as "biologically meaningful effect."
5. Known pluripotency genes (Stage 9 / companion-plan D3 relevance): `POU5F1` and `NANOG` are
   measured but **not** perturbation targets (can only ever be response genes / hub-recovery
   targets, never PID sources). `SOX2` is both measured and a perturbation target.

**Gate result: PASS, comfortably.** Reciprocal-eligible unordered pairs (both genes perturbed,
both pass the cell-count and phenotype screens) = `C(150,2) = 11,175`, vs. the plan's stated gate
of "≥~2,000 pairs." Even a much stricter phenotype filter than the one applied here would have to
throw out the large majority of targets to endanger the gate. **No fallback to Replogle needed for
feasibility; hESC alone supports H1/H2 pair-count requirements.** (Replogle may still be worth
using for Stage 9 context-transfer, per the plan — that's a separate question from this gate.)

Companion plan's own Checkpoint D0 (out-degree prior fitting) is a *different*, stricter
requirement — non-degenerate spread in *measured effective out-degree* per target, not just "a
detectable phenotype exists." That needs the real Stage 4 effect matrix (not the mean-shift proxy
above) and hasn't been run yet; see `STATE.md`.

---

## 2026-09-04 — Stage 4: real effect matrix (energy distance)

Implemented the actual Stage-4 deliverable this time (not the mean-shift proxy from Checkpoint 0):
per-gene **energy distance** (Székely & Rizzo) between each of the 150 perturbation targets and
the NTC pool, vectorized across genes via an O((m+n)·log(m+n))-per-gene merge-rank identity
(`analysis/scripts/effect_matrix.py::energy_distance_batch`) rather than the naive O(m·n) pairwise
form or 1.8M individual `scipy.stats.energy_distance` calls (150 targets × 11,942
gene-filtered genes) — both were checked and are not tractable interactively at this scale.
Correctness verified against `scipy.stats.energy_distance` on random synthetic data (`sqrt(e2)`
matches to 1e-6) before running on real data.

Gene set for this and downstream work: Stage 2's own filter (detected in ≥10% of cells, plus
union of perturbation targets) — 18,080 → **11,942 genes**. All 150 targets already satisfied the
union clause (they're all in `var_names`), so the filter is really just the detection threshold.

**Near-miss: OOM from over-parallelizing.** First attempt parallelized across all 150 targets at
full gene width (11,942 genes/target) with `ProcessPoolExecutor(max_workers=46)` on this
48-core machine. Killed the box: `free -h` showed 187/188 GiB used and 92 GiB of swap in use a
few minutes in, and the job died (exit 1, empty stderr — almost certainly an OOM kill of one or
more workers, silently swallowed by `ProcessPoolExecutor`). Root cause: every worker's arrays are
shaped `(m+n, G)` where `n` is the *fixed* 38,176-cell NTC pool — so per-worker peak memory
(~15–20 GB at G=11,942, dominated by the `argsort` output and its int64 default dtype) barely
depends on which target it's processing, and 46 workers all inflate to that peak roughly
simultaneously. Fixed by (1) chunking genes explicitly (1,500/chunk, so peak scales with chunk
size, not total gene count) as the actual unit of parallel work — 150 targets × 8 chunks = 1,200
jobs, and (2) downcasting the sort-order and cumulative-count arrays from `argsort`'s default
int64 to int32 (safe: array length ≤ ~43,000, far under int32 range), and (3) dropping
`max_workers` to 16. Measured single-job peak at chunk_size=1500 (worst-case target): **5.7 GB
RSS** — 16 concurrent ⇒ ~91 GB peak, comfortably under 188 GB even before accounting for jobs not
all hitting worst-case simultaneously. Verified stable (`free -h` steady in the 45–57 GiB used
range, swap flat) before leaving it to run unattended. **Lesson for later stages**: this
same pitfall applies directly to Stage 3's observational-skeleton PUC computation and to any
GPU-backed run — check peak memory on one representative unit of work before fanning out, not
after.

**Two more infrastructure snags, both fixed, both worth remembering for Stage 3:**

- **OOM near-miss.** First parallelization attempt used `ProcessPoolExecutor(max_workers=46)`
  over all 150 targets at full gene width (11,942 genes/worker). `free -h` showed 187/188 GiB
  used and 92 GiB of swap within minutes; the job died silently (exit 1, empty stderr — an OOM
  kill of one or more workers, swallowed by the executor). Root cause: every worker's arrays are
  shaped `(m+n, G)` where `n` is the *fixed* 38,176-cell NTC pool, so per-worker peak memory
  (~15–20 GB, dominated by `argsort`'s default int64 output) barely depends on the target being
  processed — 46 workers all inflate to that peak roughly simultaneously. Fixed by chunking genes
  explicitly (1,500/chunk, so peak scales with chunk size, not total gene count — 150 targets × 8
  chunks = 1,200 jobs), downcasting `argsort`/cumulative-count arrays to int32, and dropping to 16
  workers. Measured single-job peak after the fix: 5.7 GB RSS.
- **I/O bottleneck (looked like a hang, wasn't).** Even after the memory fix, the run appeared to
  stall for many minutes with workers stuck at "R" state and CPU time barely accumulating —
  because each worker's gene-chunk was a *fancy-indexed column gather* out of a cell-major
  `(cells, genes)` array (`X[np.ix_(cell_rows, gene_cols)]`): for every selected row, ~1,500
  scattered non-contiguous reads out of an 18,080-wide row, times up to ~43,000 rows, times 16
  concurrent workers hitting the same 16 GB mmap. Fixed by precomputing a **gene-major, gene-filtered
  contiguous array** once (`(11,942 genes, 221,273 cells)`, `analysis/scratch: X_genemajor_filtered.npy`)
  so a gene-chunk read is a fast contiguous row-block; cell selection then happens as fancy-indexing
  on an already-in-RAM small block, not scattered disk reads. That one-time rebuild itself had the
  same pitfall the first time (built via `X[:, gene_mask]` fancy-indexing on the mmap'd cell-major
  file — also slow) and was fixed by loading the source array fully into RAM first (sequential read,
  fast) before slicing/transposing in memory.
- **Final, healthy run**: 1,200 jobs, 16 workers, chunk_size=1,500 — **995.1s (~16.6 min)**, memory
  stable at 45–70 GiB throughout, workers genuinely CPU-bound (verified via `top`: 1:1 CPU-time-to-
  wall-time ratio). `E_matrix.npy` (150 × 11,942, float32) saved.
- **Correctness sanity check**: `E_matrix.npy` has a handful of tiny negative values (min
  ≈ −1.4e-5) — expected floating-point noise near zero for this identity, not a bug; clipped to 0
  downstream. Self-effect signal is enormous and correctly recovered: for each target's own gene,
  median energy-distance² is ≈2.35 vs. a median of ≈0.0008 across random (target, gene) pairs —
  roughly 3000×, as expected given how strong the CRISPRi self-knockdown signal already was in
  Checkpoint 0.

**Null calibration** (`analysis/scripts/null_calibration.py`): NTC/NTC splits at 7 representative
sizes (50, 200, 500, 1000, 2000, 3000, 4500 — spanning the observed 33–4760 target-size range), 8
replicates each, same chunked/parallel machinery — 448 jobs, 442.8s, memory-safe throughout (no
repeat of the earlier issues, since the fixes above applied directly). Null mean and SD both
scale smoothly and roughly as 1/size across the tested range (mean null e² drops from 0.0096 at
size 50 to 0.00012 at size 4500 — about 77× over a 90× size range, consistent with what's expected
for this kind of two-sample statistic) — a useful confirmation that the representative-size
approach is behaving sensibly, not an artifact.

**Calibration applied** (`analysis/scripts/calibrate_effects.py`): each target's per-gene energy
distance is z-scored against its own group size's null (log-log interpolated between the two
bracketing representative sizes), one-sided normal-approximation p-value (energy distance is only
ever elevated by a real effect), BH-FDR across all 150 × 11,942 = 1,791,300 (target, gene) pairs,
per the plan's own instruction to correct "across all (g, j)."

**Checkpoint D0 (companion plan) result: technically PASSES, but with an important caveat.**
Measured effective out-degree $\hat k^{\mathrm{out}}_g$ (genes at q<0.05, excluding the target's own
gene) ranges from **1,003 to 11,559** out of 11,942 genes across the 150 targets (mean 4,481,
median 4,133) — no target near zero, real dynamic range (>11× spread), so the letter of the D0 gate
("non-degenerate $\hat k^{\mathrm{out}}$ spread... not near-zero for most targets") is satisfied.
Correlation between $\hat k^{\mathrm{out}}_g$ and the target's own cell count is weak and slightly
*negative* (r = −0.12), so this isn't simply a power/n artifact in the crude sense.
**But**: the top of that range means some targets have "significant" shifts in **97% of the
filtered genome** — a number that is not a credible functional out-degree for a single gene's
regulatory targets by any biological standard, and is exactly the failure mode the interventional
plan's own Checkpoint 0 "Secondary concern" flagged in advance: *"hESC are a self-renewing
pluripotent population with strong cell-cycle structure and differentiation-propensity
heterogeneity. That variance will dominate MI. Cell-cycle regression or explicit conditioning is
not optional here."* At this UMI depth and these cell counts, any perturbation that nudges cell
state/cycle composition will register as "significant" against thousands of genes through that
shared axis, not through 150 independent direct regulatory programs. **Conclusion: D0 passes as
literally specified, but $\hat k^{\mathrm{out}}_g$ as currently computed is not yet trustworthy as
an out-degree *prior* for the companion plan's V3/V4 — cell-cycle/state conditioning (regression or
explicit stratification) needs to happen before this number means what the companion plan wants it
to mean.** This is the next concrete blocker to close, not a stopping point — flagged here rather
than glossed over.

**Interventional Checkpoint 3, second half (known pluripotency edges): recovered, cleanly.**
`SOX2` (the one pluripotency gene that's both measured and a perturbation target) shows extremely
significant shifts in both `POU5F1` (q≈0, energy-distance²=0.071) and `NANOG` (q≈0,
energy-distance²=0.135) — the canonical hESC pluripotency circuit is recovered as a strong positive
control. Caveat: `SOX2` itself has one of the largest $\hat k^{\mathrm{out}}$ values (8,941/11,942,
~75%), so this specific pair recovering "significant" carries the same cell-state-confound caveat
as D0 above — it's necessary-but-not-sufficient evidence, not proof the calibration is cleanly
isolating direct regulatory edges yet.

Full per-target table: `k_out.csv` in the session scratchpad (not committed).

---

## 2026-09-04 — Zero-as-own-bin discretizer

Implemented `fastpidc.discretizers.get_bin_ids_zero_as_own_bin` (Stage 2's discretizer, per
STATE.md item 2): exact zeros get bin 0, nonzero values are equal-frequency ("uniform_count")
binned into the remaining `number_of_bins - 1` bins. Wired into the existing `get_bin_ids`
dispatcher as `mode="zero_as_own_bin"`, so it's usable directly through the public
`Node.from_raw_values(..., discretizer="zero_as_own_bin", ...)` API — turned out **not** to need
the "construct `Node` by hand, bypassing the provided discretizers" workaround flagged in the
2026-09-04 API note #2; adding one function and one dispatch branch was enough. Falls back to a
plain zero/nonzero binarization if there isn't enough nonzero data to also equal-frequency-split it
(fewer nonzero values than requested bins minus one, or no nonzero values at all). Unit tests added
to `python/tests/test_discretizers.py` (zero isolation, equal-frequency counts on the nonzero
remainder, all-zero collapse, small-data fallback, dispatch-vs-direct-call equivalence) — 17/17
pass in that file, 78 passed / 1 skipped across the full `python/` suite. Not yet run against real
data; that's interventional Checkpoint 1 (MI invariance under equal-frequency binning), still open.

## 2026-09-04 — Add `pid_triple`

Added `fastpidc.pid` (new module): `pid_triple(source1, source2, target, estimator=, base=)` →
`PIDTriple(redundancy, unique1, unique2, synergy, mi1, mi2, mi_joint)`, closing the gap flagged in
the 2026-09-04 API note #5 and required by the interventional plan's own instruction to add this
before Stage 6 (H2). Built from the existing pieces, per that note's sketch:
`redundancy = apply_redundancy_formula(target.probabilities, si1, si2, base)` where `si1`/`si2` come
from `get_mi_and_si(source_i, target, ...)` (using `target.probabilities` directly rather than
recomputing the marginal, since `Node.from_raw_values` already computed it with a matching
estimator); `unique_i = mi_i - redundancy`; `synergy = mi_joint - redundancy - unique1 - unique2`,
where `mi_joint` is the MI of a **joint-binned "combined" node** (new helper `combined_node`: bin id
= `source1_bin * source2.number_of_bins + source2_bin`) against `target`. Still only supports
`I_min` as the redundancy measure — no new formula introduced, so Checkpoint 4's
redundancy-measure-robustness caveat (no BROJA/I_ccs in this package) is unchanged.

Verified against known-analytic PID cases in `python/tests/test_pid.py` rather than just
structural checks: XOR gate (X, Y independent fair bits, Z = X⊕Y) recovers ≈0 redundancy/uniques
and ≈1 bit of pure synergy; identical sources (Y := X, Z := X) recover ≈0 unique/synergy and
redundancy ≈ MI(X,Z) ≈ 1 bit; fully independent (X, Y, Z all independent) recovers ≈0 everywhere;
a general random-XOR-target case checks the decomposition identity
`mi_joint == redundancy + unique1 + unique2 + synergy` holds to floating-point precision. All 5
pass. Exposed at the package top level (`fastpidc.pid_triple`, `fastpidc.combined_node`,
`fastpidc.PIDTriple`). Not yet exercised on real intervention-indicator/gene/gene triples from the
Arc data — that's Stage 6 itself, still not started (needs Stage 3's observational skeleton and a
binned intervention-indicator node first).

## 2026-09-04 — Cell-cycle / cell-state conditioning (companion plan D0 blocker)

Implemented `analysis/scripts/cell_cycle.py` to address the D0 caveat: score every cell for S-phase
and G2M-phase activity using the standard Tirosh et al. / Seurat `cc.genes.updated.2019` marker
lists (42/43 S genes and 54/54 G2M genes found in `var_names`), via the same algorithm as
`scanpy.tl.score_genes` / Seurat `AddModuleScore` (mean expression of the gene set minus mean
expression of a size-matched, expression-bin-matched control set) — implemented directly rather
than adding `scanpy` as a dependency for one function. Then regressed both scores out of the
gene-major filtered expression matrix (11,942 × 221,273) via **vectorized OLS across all genes at
once**, not a per-gene Python loop: for design matrix `D` (n_cells × 3: intercept, S_score,
G2M_score) and gene-major data `M` (n_genes × n_cells), used the identity
`B = (D^T D)^-1 (M D)^T` to get per-gene coefficients without ever transposing the 11 GB `M` array
(`D^T M^T = (M D)^T`, and `M @ D` is a cheap n_genes×3 matmul) — the same "watch memory layout"
lesson from Stage 4's I/O bottleneck, applied preemptively this time instead of discovered the hard
way. Output: `X_resid_genemajor.npy` (same shape as the input).

**Only ~1.5% of per-gene variance was explained by the two cell-cycle scores.** Smaller than hoped
given how dramatic the D0 out-degree inflation looked (up to 97% of the genome "significant" for
one target) — suggests cell-cycle proper is only part of the confound, and the plan's own
"differentiation-propensity heterogeneity" axis (a separate, likely larger source of shared
variance in a self-renewing-but-heterogeneous hESC population) is not captured by cell-cycle
scoring alone. Re-ran Stage 4 end-to-end (effect matrix → representative-size null calibration →
BH-FDR) against the residualized array via `rerun_stage4_residualized.py` (parallel output files,
suffixed `_resid`, rather than overwriting the originals — `E_matrix_resid.npy`,
`k_out_resid.csv`, etc.) to see whether even that modest variance removal meaningfully shrinks the
implausible tail of $\hat k^{\mathrm{out}}_g$, given how much statistical power this dataset has
(a small mean shift can still clear q<0.05 at these sample sizes). **Result: negative. Cell-cycle regression alone does not fix the D0 confound — if anything, the
residualized numbers are slightly worse.**

| | original (`k_out.csv`) | cell-cycle-residualized (`k_out_resid.csv`) |
|---|---|---|
| range | 1,003 – 11,559 | 1,036 – 11,680 |
| mean | 4,481 | 4,656 |
| median | 4,133 | 4,196.5 |
| targets with $\hat k^{\mathrm{out}} \geq 10$ | 150/150 | 150/150 |

Per-target comparison: median change **+72.5**, mean change **+175.8** ($\hat k^{\mathrm{out}}$
went *up* after residualization for 135/150 targets, down for only 15/150). `PRDM14` is now the
single largest offender at 11,680/11,942 (97.8% of the filtered genome), essentially unchanged from
the pre-residualization tail. Correlation with `n_pert` is still weakly negative (r = −0.117,
matching the original −0.12) — still not simply a power artifact in the crude sense, but that was
never in question.

**Interventional Checkpoint 3 (SOX2 → POU5F1/NANOG) also got no better, and SOX2's own confound
caveat got slightly worse**: both edges remain q≈0 after residualization (POU5F1 energy-distance²
0.071 → 0.053, NANOG 0.135 → 0.128 — both still highly significant, magnitudes shrank slightly as
expected since *some* shared variance was removed), so the positive-control recovery itself is
robust to this conditioning. But SOX2's own $\hat k^{\mathrm{out}}$ **increased** from 8,941/11,942
(~75%) to 9,536/11,942 (~80%) — the exact opposite of what conditioning on the confound should do
if cell-cycle were the dominant driver.

Sanity checks before trusting this null result: (1) the residualized null's mean/SD-vs-size scaling
matches the original run's within ~0.5% at every representative size (e.g. mean null e² at size 50:
0.009565 residualized vs. 0.009570 original) — the null-calibration machinery itself did not break
on the residualized input, so the flat/negative result isn't a null-calibration artifact; (2) only
~1.5% of per-gene variance was removed by the two cell-cycle scores in the first place (see the
entry above) — in retrospect this result is exactly what that small an adjustment predicts, given
how much statistical power this dataset has (thousands of cells per group, >50k UMI/cell): a 1.5%
variance reduction is nowhere near enough to move q-values computed at that power.

**Conclusion, stated plainly: cell-cycle scoring/regression, on its own, is not the fix for the D0
out-degree inflation.** The interventional plan's own "Secondary concern" language named both "cell
cycle" *and* "differentiation-propensity heterogeneity" as risks — this result is consistent with
the latter (or some other shared axis not captured by the Tirosh/Seurat S/G2M marker genes) being
the actual dominant confound, not cell-cycle phase per se. **This is now a re-opened, harder
problem, not a closed one**: before the companion plan's V3/V4 can trust $\hat k^{\mathrm{out}}_g$
as a real degree prior, a different or additional conditioning strategy is needed — candidates worth
trying next: (a) a data-driven state axis (e.g. top PCs of the NTC-only cells, or a diffusion
pseudotime/differentiation score) rather than a curated marker-gene score, regressed out the same
way; (b) stratifying the NTC-null calibration by state bin instead of (or in addition to)
regression, so the null itself absorbs the heterogeneity rather than assuming a linear correction
removes it; (c) checking whether the top-$\hat k^{\mathrm{out}}$ targets (`PRDM14`, `METTL14`,
`METTL3`, `KDM1A`, `SMARCA4`, ...) share a annotatable biological theme (several of these are
chromatin/epigenetic regulators, which is *itself* a plausible reason for broad transcriptional
disruption rather than proof of confounding — worth a literature gut-check before assuming this is
purely technical). Not resolved in this session; flagged as the concrete next blocker in `STATE.md`.

## 2026-09-04 — CUDA backend validation (Prerequisites item 3)

Installed `cupy-cuda12x==14.2.0` into `analysis/` via `uv add "cupy-cuda12x>=12.0"` (matches
`python/pyproject.toml`'s `cuda` extra pin) — resolved cleanly against the existing Python ≥3.12
project in 279ms, no dependency conflicts, only pulled in `cuda-pathfinder` as a transitive
dependency. Machine has CUDA 12.4 toolkit/headers (`nvcc --version`) under driver 595.84 (CUDA
13.2, backward-compatible) and an idle RTX 4090 (24 GB, 0% util, no other processes holding it) —
confirmed via `nvidia-smi` before starting, to make sure this wouldn't collide with the CPU-only
cell-cycle-conditioning rerun happening in parallel this session. `fastpidc.cuda.cuda_available()`
returns `True` and `cp.cuda.runtime.getDeviceCount()` reports the GPU correctly; no `CUDA_PATH`
workaround needed (the headers are found under `/usr` already, one of `_FALLBACK_CUDA_HEADER_DIRS`
in `fastpidc/cuda.py`).

**Correctness**: `analysis/scripts/cuda_backend_check.py` builds a 40-node synthetic case (latent
factor + noise, 2,000 cells, 6 bins) and compares `fastpidc.puc.compute_puc_full` (CPU) against
`fastpidc.cuda.compute_puc_full_cuda` (GPU). Exact bitwise match: `max|MI_cpu - MI_gpu| = 0.0`,
`max|PUC_cpu - PUC_gpu| = 0.0`. Both backends implement the same closed-form arithmetic on the same
binned integer data, so exact agreement (not just "close") is the right bar here and it's met.

**Timing/memory profile**, done at the *real* cell count (221,273, not a small stand-in) since both
compute cost (joint-counts kernel is `O(n^2 * m)`-ish) and device memory (the `(m, n)` int32 binned-
data array) depend on cell count, not just gene count — using a small `n_cells` would have given a
memory estimate that's wrong in exactly the way that matters for sizing the real run. Default
`chunk_size=256`, `n_bins=10` (matches `PIDCConfig`'s actual default number of bins):

| N (genes) | time | GPU pool reserved | device free after |
|---|---|---|---|
| 512  | 4.8s   | 0.52 GB | 24.31 / 24.84 GB |
| 1024 | 31.4s  | 1.05 GB | 23.78 / 24.84 GB |
| 2048 | 127.5s | 2.13 GB | 22.70 / 24.84 GB |

Time scales quadratically in N as expected (1024→2048 is a 4.06× runtime increase, consistent with
`O(N^2)` pair-work at fixed cell count); memory scales linearly in N (doubling N roughly doubles
reserved memory), consistent with the dominant term being the `(cells, genes)` binned-data array
(`221,273 * N * 4 bytes` for the int32 array alone — e.g. at N=2048 that's already 1.81 GB of the
2.13 GB measured, the rest split between the kernel's per-chunk counts buffer and the N×N MI/PUC
output matrices).

**Extrapolation to the real Stage-3 scale** (N=11,942 filtered genes, same 221,273 cells): memory
≈ 221,273 × 11,942 × 4 bytes (binned data) + 10² × 11,942 × 256 × 4 bytes (counts chunk, chunk_size
default 256) + 2 × 11,942² × 8 bytes (MI + PUC output matrices) ≈ 10.6 + 1.2 + 2.3 ≈ **~14 GB**,
comfortably inside the RTX 4090's 24 GB. Time ≈ 127.5s × (11,942 / 2,048)² ≈ **~72 minutes** for one
full dense PUC pass over the entire filtered gene set. Both numbers are extrapolations from N≤2048
measurements, not a direct measurement at N=11,942 — a spot-check at an intermediate size (e.g.
N=4096) before committing a ~70-minute run would be cheap insurance, but nothing in the scaling
behavior so far suggests a surprise at full width.

**Recommendation, superseding STATE.md's earlier assumption that Stage 3 would need the same
gene-chunked multiprocessing treatment Stage 4 needed on CPU**: run the full 11,942-gene dense PUC
in **one GPU call** (`compute_puc_full_cuda`, letting its own internal chunking over the z-axis at
`chunk_size=256` bound device memory) rather than pre-splitting into a CPU-style job queue — there's
only one GPU to serialize onto anyway, and 14 GB of 24 GB leaves enough headroom that chunk-size
tuning for memory is not expected to be necessary. This closes STATE.md Prerequisites item 3; Stage
3 itself (running this against the real, zero-as-own-bin-discretized, filtered gene set) is still
not started — it also needs interventional Checkpoint 1 (binning-invariance check) run first, per
the plan's own ordering.

## Scratch outputs (not committed)

Intermediate arrays/CSVs from today's session live in the session scratchpad
(`/tmp/claude-1000/.../scratchpad/`), not in the repo — regenerate from the h5ad rather than
relying on these persisting: `X_dense.npy` / `X_norm_log.npy` (16 GB, cell-major raw / normalized),
`X_genemajor_filtered.npy` (10.6 GB, gene-major + gene-filtered, the layout `effect_matrix.py` and
`null_calibration.py` actually read), `gene_filter_mask.npy`, `self_effect_screen.csv`,
`phenotype_screen.csv`, `E_matrix.npy` + `E_matrix_meta.pkl` (Stage-4 effect matrix),
`null_mean_by_size.npy` / `null_sd_by_size.npy` + `null_meta.pkl` (representative-size null),
`E_zscores.npy` / `E_qvalues.npy` (calibrated), `k_out.csv` (measured effective out-degree per
target). The three scripts (`effect_matrix.py`, `null_calibration.py`, `calibrate_effects.py`, all
under `analysis/scripts/`) are committed and reproduce all of this from the h5ad + cached
normalization step; only the huge intermediate arrays themselves aren't kept.
