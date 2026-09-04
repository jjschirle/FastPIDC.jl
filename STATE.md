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
| 3 | Self-effects present; known pluripotency edges recovered | 🟡 | Self-effect half done today as a Checkpoint-0 byproduct: **150/150 pass** (q<0.05, down). Pluripotency-edge recovery half not started (needs the actual network, i.e. Stage 3). `POU5F1`/`NANOG` measured-not-targeted, `SOX2` measured-and-targeted — noted for later. |
| 4 | Sibling calls stable across redundancy measures (I_min vs BROJA/I_ccs) | ⬜ | Not started; blocked on Stage 6 existing at all, and on sourcing a non-I_min PID implementation (`fastpidc` only has I_min — see `LOG.md`). |
| 5 | PID beats mediation regression (H3, the kill criterion) | ⬜ | Not started; last in the pipeline by design. |

### Stages (§4), beyond what the checkpoints above already cover

| Stage | Status | Note |
|---|---|---|
| 1 — Load & QC | 🟡 | Load path solved. Per-cell filters (UMI/gene-count/mito) and mixscape-style per-cell knockdown-efficiency estimate: not started. |
| 2 — Normalization & discretization | 🟡 | Size-factor + log1p normalization exercised (as a means to Checkpoint 0, not yet as a committed pipeline step). The zero-as-own-bin equal-frequency discretizer is not implemented against `fastpidc.discretizers` yet. |
| 3 — Observational skeleton | ⬜ | Not started. |
| 4 — Directed effect matrix (energy distance) | ⬜ | Not started — **note**: today's phenotype screen used a mean-shift proxy, explicitly *not* this. Real Stage 4 must use energy distance per the plan. |
| 5 — Orientation (H1) | ⬜ | Not started; depends on Stage 4. |
| 6 — Intervention-indicator PID (H2) | 🚫 | Blocked on adding the `pid_triple`-equivalent primitive to `fastpidc` (see Prerequisites). |
| 7 — Baselines (H3) | ⬜ | Not started. |
| 8 — Invariance filtering (ICP) | ⬜ | Not started. |
| 9 — External validation (ATAC, Replogle, literature) | ⬜ | Not started. |

## Companion plan (degree-aware recalibration) — checkpoints (§8 summary table)

| # | Gate | Status | Result |
|---|---|---|---|
| **D0** | Enough targets with non-degenerate measured $\hat k^{\mathrm{out}}$ | ⬜ | Not started. Needs the *real* Stage-4 effect matrix (energy distance, BH-significant vs NTC-split null), not today's mean-shift proxy — measuring $\hat k^{\mathrm{out}}_g$ per target requires that matrix. |
| D1 | V2 (hierarchical gamma) does not reduce bootstrap stability | ⬜ | Not started; depends on interventional-plan Stage 3 existing. |
| D2 | V1 (direction-aware calibration) beats V2 on held-out effect recovery | ⬜ | Not started. |
| D3 | Known hubs (POU5F1, NANOG, SOX2) survive increasing λ in V3 | ⬜ | Not started; note all three are measured, `SOX2` is also itself a perturbation target. |
| D4 | Gains don't appear equally for a correlation skeleton | ⬜ | Not started. |

Implementation order per the plan (V2 → V1 → V4 → V3) not started; V2 is unblocked from the API
side (raw PUC access confirmed) whenever Stage 3's skeleton exists to calibrate.

---

## Immediate next actions (carried over from both plans, reordered by what's actually next)

1. Implement Stage 4's real effect matrix (energy distance, NTC-split null, BH-FDR) — feeds
   both interventional Checkpoint 3's second half and companion Checkpoint D0. Highest-value next
   step; nothing else downstream can be evaluated honestly without it.
2. Implement the zero-as-own-bin discretizer and wire it into `fastpidc.Node` construction
   (Stage 2), then run interventional Checkpoint 1 (binning-invariance sanity check).
3. Install `cupy` and exercise the CUDA backend before attempting a full 18,080-gene PUC run —
   the CPU backend's Python-loop-over-pairs cost is very unlikely to be workable at that gene
   count (companion + interventional plans both note PUC is inherently O(N³)).
4. Add the `pid_triple`-equivalent module to `fastpidc` (new branch work, per the interventional
   plan's own instruction) — needed before Stage 6 / H2, not before.
5. Literature re-check and ADAPRE/D-SPIN reading — still not started, no dependency on the above.
