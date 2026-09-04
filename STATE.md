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
| Confirm `pid_triple`-equivalent (triple PID: redundancy/unique/synergy) exists | 🚫 | **Does not exist.** Building blocks (`get_mi_and_si`, `apply_redundancy_formula`) do. Must be added as a new module before Stage 6 — see below. |
| Set up environment to load Arc hESC/K562 h5ad | ✅ | `analysis/` uv project; `vsparse` 0.2.0 + `hdf5plugin`, load via `vsparse.VCSCAnnData.read_h5ad`. |
| Literature re-check (degree-corrected graphical models; recent PID/interventional-GRN work) | ⬜ | Not started — needs web search, out of scope for a data-only session so far. |
| Read ADAPRE, D-SPIN properly | ⬜ | Not started. |

## Interventional plan — checkpoints (§5 summary table)

| # | Gate | Status | Result |
|---|---|---|---|
| **0** | ≥~2,000 reciprocal testable pairs | ✅ **PASS** | 150 targets (not the assumed ~300 — see `LOG.md`), all clear cell-count + phenotype screens. `C(150,2) = 11,175` pairs. No Replogle fallback needed for feasibility. |
| 1 | MI invariant under equal-frequency binning | ⬜ | Not started. Needs Stage 2 discretizer implementation first (zero-as-own-bin isn't one of `fastpidc`'s built-ins — must be hand-rolled and fed in via `Node` construction, see `LOG.md` API note #2). |
| 2 | Bootstrap edge recovery >50% at top-k | ⬜ | Not started. Needs Stage 3 (observational skeleton on NTC cells) first, which needs a working discretization pipeline and — at 18,080 genes — realistically needs the CUDA backend (`cupy` not yet installed, RTX 4090 available). |
| 3 | Self-effects present; known pluripotency edges recovered | ✅ | Self-effect: **150/150 pass** (q<0.05, down; Checkpoint-0 byproduct). Pluripotency edges: **recovered** — `SOX2` (measured + targeted) shows q≈0 shifts in both `POU5F1` and `NANOG`, using the real Stage-4 effect matrix (not the Checkpoint-0 proxy). Caveat in `LOG.md`: `SOX2`'s own $\hat k^{\mathrm{out}}$ is ~75% of the filtered genome, so this is necessary-but-not-sufficient given the cell-state-confound issue flagged under D0 below. |
| 4 | Sibling calls stable across redundancy measures (I_min vs BROJA/I_ccs) | ⬜ | Not started; blocked on Stage 6 existing at all, and on sourcing a non-I_min PID implementation (`fastpidc` only has I_min — see `LOG.md`). |
| 5 | PID beats mediation regression (H3, the kill criterion) | ⬜ | Not started; last in the pipeline by design. |

### Stages (§4), beyond what the checkpoints above already cover

| Stage | Status | Note |
|---|---|---|
| 1 — Load & QC | 🟡 | Load path solved. Per-cell filters (UMI/gene-count/mito) and mixscape-style per-cell knockdown-efficiency estimate: not started. |
| 2 — Normalization & discretization | 🟡 | Size-factor + log1p normalization exercised (as a means to Checkpoint 0, not yet as a committed pipeline step). The zero-as-own-bin equal-frequency discretizer is not implemented against `fastpidc.discretizers` yet. |
| 3 — Observational skeleton | ⬜ | Not started. |
| 4 — Directed effect matrix (energy distance) | ✅ | Done: `analysis/scripts/effect_matrix.py`. Real energy distance (not the Checkpoint-0 mean-shift proxy), 150 targets × 11,942 genes, NTC-split significance calibration via `null_calibration.py` + `calibrate_effects.py` (BH-FDR across all pairs, per the plan). Hit and fixed an OOM and an I/O-layout bottleneck along the way — see `LOG.md`. **Open issue carried forward**: measured effective out-degree is inflated by hESC cell-state/cycle confounding, exactly as the plan's own Checkpoint-0 "Secondary concern" anticipated — see D0 below. |
| 5 — Orientation (H1) | ⬜ | Not started; depends on Stage 4. |
| 6 — Intervention-indicator PID (H2) | 🚫 | Blocked on adding the `pid_triple`-equivalent primitive to `fastpidc` (see Prerequisites). |
| 7 — Baselines (H3) | ⬜ | Not started. |
| 8 — Invariance filtering (ICP) | ⬜ | Not started. |
| 9 — External validation (ATAC, Replogle, literature) | ⬜ | Not started. |

## Companion plan (degree-aware recalibration) — checkpoints (§8 summary table)

| # | Gate | Status | Result |
|---|---|---|---|
| **D0** | Enough targets with non-degenerate measured $\hat k^{\mathrm{out}}$ | 🟡 **PASS, with a caveat that blocks trusting the number yet** | $\hat k^{\mathrm{out}}_g$ ranges 1,003–11,559 / 11,942 genes (>11× spread, none near zero) — technically clears the gate. But the top of that range (~97% of the genome "significant" for one target) is not a credible functional out-degree; almost certainly reflects hESC cell-state/cycle composition shifting under perturbation rather than 150 independent direct regulatory programs, exactly the risk the plan's own Checkpoint 0 "Secondary concern" flagged. **Needs cell-cycle/state regression or stratification before $\hat k^{\mathrm{out}}_g$ is fit as a real degree prior** — that's the concrete next blocker, not a re-run of what's already done. Full table: `k_out.csv` (scratchpad, not committed). |
| D1 | V2 (hierarchical gamma) does not reduce bootstrap stability | ⬜ | Not started; depends on interventional-plan Stage 3 existing. |
| D2 | V1 (direction-aware calibration) beats V2 on held-out effect recovery | ⬜ | Not started. |
| D3 | Known hubs (POU5F1, NANOG, SOX2) survive increasing λ in V3 | ⬜ | Not started; note all three are measured, `SOX2` is also itself a perturbation target. |
| D4 | Gains don't appear equally for a correlation skeleton | ⬜ | Not started. |

Implementation order per the plan (V2 → V1 → V4 → V3) not started; V2 is unblocked from the API
side (raw PUC access confirmed) whenever Stage 3's skeleton exists to calibrate.

---

## Immediate next actions (carried over from both plans, reordered by what's actually next)

1. **Cell-cycle / cell-state conditioning before trusting $\hat k^{\mathrm{out}}_g$.** D0 passes
   literally but the measured out-degree numbers are almost certainly inflated by shared
   cell-state/cycle shifts, not 150 independent regulatory programs (see D0 note above). Regress
   out or stratify by a cell-cycle score before the companion plan's V3/V4 fit a degree prior on
   this. This is the concrete blocker uncovered by today's work — resolve before building on
   `k_out.csv` further.
2. Implement the zero-as-own-bin discretizer and wire it into `fastpidc.Node` construction
   (Stage 2), then run interventional Checkpoint 1 (binning-invariance sanity check).
3. Install `cupy` and exercise the CUDA backend before attempting a full 18,080-gene PUC run —
   the CPU backend's Python-loop-over-pairs cost is very unlikely to be workable at that gene
   count (companion + interventional plans both note PUC is inherently O(N³)). Note from today:
   even the much cheaper O((m+n)log(m+n)) energy-distance computation needed real attention to
   memory (chunk sizing) and data layout (gene-major contiguity) to run efficiently in parallel —
   the same two issues will apply, likely more sharply, to Stage 3's O(N³) PUC computation; check
   peak memory on one representative unit of work before fanning out, not after.
4. Add the `pid_triple`-equivalent module to `fastpidc` (new branch work, per the interventional
   plan's own instruction) — needed before Stage 6 / H2, not before.
5. Literature re-check and ADAPRE/D-SPIN reading — still not started, no dependency on the above.
