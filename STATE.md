# Plan status

Tracks completion of items in `Interventional PID for GRN inference—testing the gap with
FastPIDC.jl on Arc H1 hESC Perturb-seq.md` (**interventional plan**) and `Degree-aware
recalibration of PIDC—a companion plan.md` (**companion plan**). Detail and rationale in
`LOG.md`. Branch: `perturbation-analysis`.

Legend: ✅ done · 🟡 in progress / partial · ⬜ not started · 🚫 blocked

---

## Prerequisites (both plans' §"immediate next actions")

| Item | Status | Note |
|---|---|---|
| Locate & verify real FastPIDC.jl Python API | ✅ | `LOG.md` 2026-09-04. Package lives at `python/`. |
| Confirm per-gene PUC access + calibration can be disabled | ✅ | `PUCNetworkInference` / `compute_puc_full` — already exposed, no patch needed. |
| Confirm `pid_triple`-equivalent (triple PID: redundancy/unique/synergy) exists | ✅ | Added `fastpidc.pid.pid_triple` (+ `combined_node`, `PIDTriple`) — see `LOG.md` 2026-09-04 "Add `pid_triple`". |
| Set up environment to load Arc hESC/K562 h5ad | ✅ | `analysis/` uv project; `vsparse` 0.2.0 + `hdf5plugin`, load via `vsparse.VCSCAnnData.read_h5ad`. |
| Literature re-check (degree-corrected graphical models; recent PID/interventional-GRN work) | ⬜ | Not started — needs web search, out of scope for a data-only session so far. |
| Read ADAPRE, D-SPIN properly | ⬜ | Not started. |

## Interventional plan — checkpoints (§5 summary table)

| # | Gate | Status | Result |
|---|---|---|---|
| **0** | ≥~2,000 reciprocal testable pairs | ✅ **PASS** | 150 targets (not the assumed ~300 — see `LOG.md`), all clear cell-count + phenotype screens. `C(150,2) = 11,175` pairs. No Replogle fallback needed for feasibility. |
| 1 | MI invariant under equal-frequency binning | ⬜ | Not started (discretizer now ready — see Stage 2 below — but the actual invariance run hasn't happened yet). |
| 2 | Bootstrap edge recovery >50% at top-k | ⬜ | Not started. Needs Stage 3 (observational skeleton on NTC cells) first, which needs a working discretization pipeline. CUDA backend now installed and validated (`cupy-cuda12x`, see Prerequisites item 3) so the compute path is no longer the blocker — Stage 3 itself just hasn't been run yet. |
| 3 | Self-effects present; known pluripotency edges recovered | ✅ | Self-effect: **150/150 pass** (q<0.05, down; Checkpoint-0 byproduct). Pluripotency edges: **recovered, and robust to cell-cycle residualization** — `SOX2` (measured + targeted) shows q≈0 shifts in both `POU5F1` (energy-distance² 0.071→0.053) and `NANOG` (0.135→0.128) before and after cell-cycle regression. Caveat unchanged/worsened: `SOX2`'s own $\hat k^{\mathrm{out}}$ is ~75–80% of the filtered genome (went *up* slightly after cell-cycle residualization, 8,941→9,536), so this is still necessary-but-not-sufficient — see D0 below, now a re-opened problem, not resolved by the cell-cycle fix that was tried. |
| 4 | Sibling calls stable across redundancy measures (I_min vs BROJA/I_ccs) | ⬜ | Not started; blocked on Stage 6 existing at all, and on sourcing a non-I_min PID implementation (`fastpidc` only has I_min — see `LOG.md`). |
| 5 | PID beats mediation regression (H3, the kill criterion) | ⬜ | Not started; last in the pipeline by design. |

### Stages (§4), beyond what the checkpoints above already cover

| Stage | Status | Note |
|---|---|---|
| 1 — Load & QC | 🟡 | Load path solved. Per-cell filters (UMI/gene-count/mito) and mixscape-style per-cell knockdown-efficiency estimate: not started. |
| 2 — Normalization & discretization | 🟡 | Size-factor + log1p normalization exercised (as a means to Checkpoint 0, not yet as a committed pipeline step). Zero-as-own-bin discretizer now implemented as `fastpidc.discretizers.get_bin_ids_zero_as_own_bin` / `get_bin_ids(..., mode="zero_as_own_bin")` — not yet run against real data (that's Checkpoint 1). |
| 3 — Observational skeleton | ⬜ | Not started, but unblocked: CUDA backend validated (see Prerequisites item 3 / `LOG.md` 2026-09-04) — correctness matches CPU exactly, and a full 11,942-gene dense PUC run is projected to fit in ~14 GB / ~70 min on the RTX 4090. Still needs the zero-as-own-bin discretizer actually run against real data (Checkpoint 1) before Stage 3 itself starts. |
| 4 — Directed effect matrix (energy distance) | ✅ | Done: `analysis/scripts/effect_matrix.py`. Real energy distance (not the Checkpoint-0 mean-shift proxy), 150 targets × 11,942 genes, NTC-split significance calibration via `null_calibration.py` + `calibrate_effects.py` (BH-FDR across all pairs, per the plan). Hit and fixed an OOM and an I/O-layout bottleneck along the way — see `LOG.md`. **Open issue carried forward**: measured effective out-degree is inflated by hESC cell-state/cycle confounding, exactly as the plan's own Checkpoint-0 "Secondary concern" anticipated — see D0 below. |
| 5 — Orientation (H1) | ⬜ | Not started; depends on Stage 4. |
| 6 — Intervention-indicator PID (H2) | ⬜ | `pid_triple` primitive now exists (see Prerequisites) — Stage 6 itself (running it on real intervention-indicator/gene/gene triples) not started. |
| 7 — Baselines (H3) | ⬜ | Not started. |
| 8 — Invariance filtering (ICP) | ⬜ | Not started. |
| 9 — External validation (ATAC, Replogle, literature) | ⬜ | Not started. |

## Companion plan (degree-aware recalibration) — checkpoints (§8 summary table)

| # | Gate | Status | Result |
|---|---|---|---|
| **D0** | Enough targets with non-degenerate measured $\hat k^{\mathrm{out}}$ | 🚫 **PASS on the letter of the gate, but the confound is NOT resolved — cell-cycle regression tried and failed** | $\hat k^{\mathrm{out}}_g$ ranges 1,003–11,559 / 11,942 genes originally (>11× spread, none near zero) — technically clears the gate. Tried the obvious fix (regress S/G2M cell-cycle scores out of expression, re-run Stage 4 on residuals — `LOG.md` 2026-09-04 "Cell-cycle / cell-state conditioning"): **it didn't work.** Residualized $\hat k^{\mathrm{out}}_g$ range 1,036–11,680, mean 4,656 (was 4,481), median 4,196.5 (was 4,133) — *higher* for 135/150 targets (median +72.5). SOX2's own out-degree got worse (8,941→9,536/11,942, ~75%→~80%). Only ~1.5% of per-gene variance was explained by cell-cycle scores in the first place, which in hindsight predicts exactly this null result given the dataset's statistical power. **Conclusion: the dominant confound is not cell-cycle phase** (or not fully — the plan's own "differentiation-propensity heterogeneity" language covers exactly this possibility). **This is a re-opened blocker, not a closed one** — see next actions below for candidate follow-ups (data-driven state axis instead of curated marker genes; stratified null instead of linear regression; check whether top-$\hat k^{\mathrm{out}}$ targets like `PRDM14`/`METTL14`/`METTL3`/`KDM1A`/`SMARCA4` share a chromatin/epigenetic-regulator theme that might be a real broad effect, not pure confound). Tables: `k_out.csv` / `k_out_resid.csv` (scratchpad, not committed). |
| D1 | V2 (hierarchical gamma) does not reduce bootstrap stability | ⬜ | Not started; depends on interventional-plan Stage 3 existing. |
| D2 | V1 (direction-aware calibration) beats V2 on held-out effect recovery | ⬜ | Not started. |
| D3 | Known hubs (POU5F1, NANOG, SOX2) survive increasing λ in V3 | ⬜ | Not started; note all three are measured, `SOX2` is also itself a perturbation target. |
| D4 | Gains don't appear equally for a correlation skeleton | ⬜ | Not started. |

Implementation order per the plan (V2 → V1 → V4 → V3) not started; V2 is unblocked from the API
side (raw PUC access confirmed) whenever Stage 3's skeleton exists to calibrate.

---

## Immediate next actions (carried over from both plans, reordered by what's actually next)

1. **Cell-cycle / cell-state conditioning before trusting $\hat k^{\mathrm{out}}_g$.** 🚫 **Tried,
   failed, re-opened.** `analysis/scripts/cell_cycle.py` scores every cell for S/G2M activity
   (Tirosh/Seurat marker lists, scanpy-style module scoring) and regresses both scores out of the
   gene-major filtered expression matrix; `rerun_stage4_residualized.py` reran Stage 4 end-to-end
   against the residuals. **Result: no improvement** — residualized $\hat k^{\mathrm{out}}_g$ went
   *up* for 135/150 targets (median +72.5), and SOX2's own out-degree got worse (~75%→~80%). Only
   ~1.5% of per-gene variance was explained by the two cell-cycle scores, which predicts this null
   result in hindsight. See `LOG.md` 2026-09-04 "Cell-cycle / cell-state conditioning" for full
   numbers and sanity checks (null-scaling verified unaffected, so this isn't a calibration bug).
   **Next candidate fixes, not yet tried:** (a) a data-driven state axis (top PCs of NTC cells, or
   a pseudotime/differentiation score) instead of curated cell-cycle marker genes; (b) stratifying
   the null calibration by state bin instead of linear regression; (c) a literature/annotation
   check on whether the current top-$\hat k^{\mathrm{out}}$ targets (`PRDM14`, `METTL14`,
   `METTL3`, `KDM1A`, `SMARCA4`, ...) are chromatin/epigenetic regulators whose broad effect might
   be partly real, not purely a confound artifact.
2. ✅ Implemented the zero-as-own-bin discretizer (`fastpidc.discretizers.get_bin_ids_zero_as_own_bin`,
   wired into `get_bin_ids(..., mode="zero_as_own_bin")`, so it's usable directly via
   `Node.from_raw_values(..., discretizer="zero_as_own_bin", ...)` — no bypass of the public API
   needed after all). Not yet run against real data — running interventional Checkpoint 1
   (binning-invariance sanity check) is still open.
3. ✅ Installed `cupy-cuda12x==14.2.0` in `analysis/` (`uv add`, clean resolve, no conflicts with
   the existing Python ≥3.12 project) and validated the CUDA backend end-to-end in
   `analysis/scripts/cuda_backend_check.py` — see `LOG.md` 2026-09-04 "CUDA backend validation"
   for full numbers. Correctness: CPU (`compute_puc_full`) vs CUDA (`compute_puc_full_cuda`) match
   exactly (`max|diff| = 0.0`) on a 40-node synthetic case. Memory/timing at real cell count
   (221,273 cells, 10 bins, chunk_size=256): time scales ~quadratically in gene count as expected
   (N=1024→2048 is a 4.06× runtime increase), device memory scales ~linearly in gene count
   (0.52/1.05/2.13 GB at N=512/1024/2048) and is dominated by the (cells × genes) int32 binned-data
   array. **Extrapolated full-genome estimate (N=11,942, real cell count): ~14 GB device memory
   (fits comfortably in the RTX 4090's 24 GB) and ~70 minutes wall-clock** — both numbers are
   extrapolations from the measured N≤2048 points, not a direct measurement at full N; worth a
   spot-check at, say, N=4096 before committing to the full run, but nothing here suggests Stage 3
   is infeasible on this GPU. **Recommendation: proceed with Stage 3 at the full 11,942-gene
   filtered set in one GPU pass (no gene-chunking needed across the outer call, only the kernel's
   own internal z-chunking at chunk_size=256), rather than pre-splitting into a job queue like
   Stage 4's CPU multiprocessing did** — the memory headroom and single-GPU serialization make
   that unnecessary here.
4. ✅ Added `fastpidc.pid` (`pid_triple`, `combined_node`, `PIDTriple`) implementing the triple PID
   decomposition (`MI_joint = redundancy + unique1 + unique2 + synergy`) via the existing
   `get_mi_and_si` / `apply_redundancy_formula` building blocks, plus a joint-binned "combined
   node" for the synergy term. Unit-tested against known-analytic cases (XOR → pure synergy,
   identical sources → pure redundancy, independent → all-zero) in `python/tests/test_pid.py`; 78
   passed / 1 skipped across the full `python/` suite after adding it. Still uses `I_min` only
   (Checkpoint 4's redundancy-measure-robustness check is unaffected — still needs an external PID
   library). Not yet exercised on real intervention-indicator/gene/gene triples (that's Stage 6
   itself, still not started).
5. Literature re-check and ADAPRE/D-SPIN reading — delegated to a background agent (see
   `LITERATURE_REVIEW.md` once it lands); verdict not yet folded into this file.
