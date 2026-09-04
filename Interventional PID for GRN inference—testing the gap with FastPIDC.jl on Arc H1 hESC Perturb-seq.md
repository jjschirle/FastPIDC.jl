**Status:** draft plan. Sections marked ⚠️ contain assumptions that must be corrected before implementation.

---

## 0. The claim being tested

Partial information decomposition has been applied to observational scRNA-seq (PIDC; Chan, Stumpf & Babtie 2017) and interventional data has been used to _evaluate_ PIDC (SCING; Littman et al. 2023), but no published method uses **CRISPR intervention indicators as PID sources**. That is the gap.

Three sub-hypotheses, in increasing ambition and decreasing confidence:

| # | Hypothesis | Prior |
|---|---|---|
| **H1** | Reciprocal perturbation asymmetry orients a useful fraction of PIDC edges | High — near-direct measurement |
| **H2** | `unique(X→Y \| I_g) ≈ 0` identifies co-downstream sibling pairs that observational PIDC calls edges | Moderate — the novel contribution |
| **H3** | The PID mediation decomposition beats a plain mediation regression at the same task | **Low** — this is the honest kill criterion |

H3 is the one that decides whether the PID framing is doing work or is decoration. If a linear mediation test or a conditional-independence test matches the decomposition, publish H2 as a mediation result and drop the information theory. Design the study so that comparison is unavoidable rather than optional.

### Prior art to position against

- **PIDC** — Chan, Stumpf & Babtie, _Cell Systems_ 2017. Observational, undirected, `InformationMeasures.jl`.
- **SCING** — used Perturb-seq to benchmark PIDC and concluded PID approaches are "more accurately described as measures of coexpression, rather than gene regulation." This is the criticism the project must answer.
- **D-SPIN** — Jiang et al. Closest in spirit: perturbations as external fields in a maximum-entropy model, explicitly motivated by the case where two genes have "weak negative correlation and low mutual information" but a real relationship. Not PID. **Primary conceptual competitor.**
- **ADAPRE** (bioRxiv, Feb 2026) — CRISPRi guides as instrumental variables in a Poisson–lognormal model; handles knockdown-efficiency bias and cyclic structure. **Primary methodological competitor**, and better than this plan on soft interventions and feedback. Read before starting.
- **CausalBench** — found interventional methods do _not_ beat observational ones on real data, contrary to synthetic benchmarks. The single most important tempering result.

---

## 1. Data

**Arc Virtual Cell Atlas, Virtual Cell Challenge H1 hESC dataset.** ~300k cells, 300 CRISPRi perturbations, 10x Flex, >50k UMI/cell. CC0.

Two properties matter and pull in opposite directions:

- **Depth is excellent.** >50k UMI/cell is far above typical droplet data. Contingency tables will be better populated than in any dataset PIDC was originally tested on. This is the main reason the project is feasible at all.
- **Perturbation breadth is narrow.** 300 targets, not genome-wide.

### ⚠️ Checkpoint 0 — the reciprocal-pair count (RUN THIS FIRST, BEFORE ANY CODE)

H1 and H2 both require pairs where **both** genes were perturbed. Upper bound is `C(300,2) = 44,850` ordered-pair-eligible pairs, but the real number is smaller after:

- restricting to targets with a detectable transcriptomic phenotype (in Replogle, a large share of perturbations had none),
- restricting to targets that are themselves well-detected as _response_ genes,
- requiring adequate cells per condition.

**Gate:** if fewer than ~2,000 pairs survive with both directions testable, H1/H2 are underpowered on this dataset alone. Do not proceed to implementation. Options at that point: add Replogle K562/RPE1 (genome-wide, ~9,800 targets) as the primary discovery set and use hESC as the context-transfer test, or restrict the entire study to the 300 targets and drop genome-scale ambitions.

This checkpoint is cheap — one pass over the obs metadata plus a per-target energy-distance screen — and it determines the shape of everything downstream. It is also where the project is most likely to die, so it goes first.

**Secondary concern:** hESC are a self-renewing pluripotent population with strong cell-cycle structure and differentiation-propensity heterogeneity. That variance will dominate MI. Cell-cycle regression or explicit conditioning is not optional here.

---

## 2. ⚠️ FastPIDC.jl — assumed API

**I could not locate FastPIDC.jl.** Nothing in the Julia General registry, GitHub search, JOSS, or bioRxiv. Everything below is an assumed interface based on `InformationMeasures.jl` conventions and the original PIDC implementation. **Replace this section with the real API before writing any code.**

Specifically, I need to know:

1. Signature and return type of the main network call. Dense matrix, sparse, or edge list?
2. Is discretization done internally or does it accept pre-binned integer data? **Critical** — the whole normalization argument depends on binning scheme.
3. Is the low-level PID accessible, i.e. can I get `redundancy` / `unique` / `synergy` for an arbitrary triple `(source1, source2, target)`? **This is the load-bearing requirement.** If only the aggregated PUC score is exposed, H2 and H3 cannot be implemented without patching the package.
4. Which redundancy measure(s)? Is anything beyond `I_min` available?
5. How is the per-gene gamma calibration exposed, and can it be disabled?
6. Threading/GPU model, and whether it is safe to call from inside a `Threads.@threads` loop.

Assumed for the purposes of drafting:

```julia
# ⚠️ ALL ASSUMED
using FastPIDC

net = pidc_network(X;                      # X :: Matrix (cells × genes) or CSC
                   discretizer = :equal_frequency,
                   estimator   = :maximum_likelihood,
                   calibrate   = true)      # per-gene gamma

# The critical primitive:
pid = pid_triple(s1, s2, target;            # AbstractVector, discretized
                 measure = :imin)
# pid.redundancy, pid.unique1, pid.unique2, pid.synergy
```

If `pid_triple` or equivalent does not exist, **stop and add it** before anything else. It is the core of the contribution.

---

## 3. Environment

```julia
# Project.toml
[deps]
FastPIDC          = "..."   # ⚠️ path/UUID unknown
InformationMeasures = "..."
Muon             = "..."    # AnnData .h5ad reader
HDF5             = "..."
SparseArrays     = "..."
Statistics       = "..."
StatsBase        = "..."
Distributions    = "..."
Random           = "..."
Distances        = "..."
GLM              = "..."    # baselines for H3
MultipleTesting  = "..."
Arrow            = "..."    # intermediate storage
DataFrames       = "..."
ProgressMeter    = "..."
```

Pin versions and commit `Manifest.toml`. Set `JULIA_NUM_THREADS` explicitly. Seed every RNG and record seeds in output metadata — resampling stability is a headline result, so irreproducibility there is fatal.

---

## 4. Pipeline

### Stage 1 — Load and QC

```julia
using Muon, SparseArrays

ad = readh5ad("data/vcc_h1_hesc.h5ad")

# ⚠️ verify actual column names in ad.obs
target_col = "target_gene"
control_label = "non-targeting"

counts = ad.X                     # cells × genes, sparse counts
targets = ad.obs[!, target_col]
```

QC: standard per-cell filters (UMI, gene count, mitochondrial fraction). Per-guide knockdown efficiency, mixscape-style, retaining a per-cell continuous perturbation strength rather than binary assignment — CRISPRi is a soft intervention and ADAPRE showed that efficacy variation inflates out-degree of strongly knocked-down genes. Keep the continuous value even if the first pass uses binary.

### Stage 2 — Normalization and discretization

Per earlier reasoning: **size-factor normalize, log1p, equal-frequency binning, zeros as their own category.** Deviance residuals are explicitly rejected — for zero counts the residual is a deterministic decreasing function of library size, which smears depth into the zero block and manufactures MI between sparse gene pairs.

```julia
function discretize_gene(v::AbstractVector{<:Real}; nbins::Int=4)
    # zeros get bin 1; nonzeros get equal-frequency bins 2..nbins
    out = ones(Int, length(v))
    nz = findall(!iszero, v)
    isempty(nz) && return out
    vals = v[nz]
    qs = quantile(vals, range(0, 1, length=nbins))
    @inbounds for (k, i) in enumerate(nz)
        out[i] = 1 + clamp(searchsortedlast(qs, vals[k]), 1, nbins - 1)
    end
    return out
end
```

**Checkpoint 1 — invariance sanity check.** Under equal-frequency binning, MI must be numerically identical for raw, log1p, and rank-transformed input. If it isn't, the binning is not rank-based and the transformation-invariance argument in the paper is wrong. Then deliberately repeat with uniform-width binning and confirm it _does_ change — that contrast is a figure.

Gene filter: detected in ≥10% of cells, plus the union of perturbation targets regardless of detection rate. Record the gene set in output metadata, because PUC depends on which genes are in the matrix.

### Stage 3 — Observational skeleton

Run FastPIDC on **control (non-targeting) cells only**. This is the candidate skeleton whose job is to cut the O(p²) space to something affordable for causal testing.

```julia
ctrl = findall(==(control_label), targets)
Xd_ctrl = Xd[ctrl, :]
skeleton = pidc_network(Xd_ctrl)   # ⚠️ assumed API
```

Retain top-k edges at several densities (k = 10p, 50p, 100p). Report everything at matched density — gamma-calibrated scores have no absolute meaning and are not comparable across runs.

**Checkpoint 2 — scaling and stability.** Time and memory on 500 / 2,000 / 5,000 genes; confirm the claimed scaling before committing to the full run. Then bootstrap cells (20 replicates) and compute edge-recovery frequency. **Gate: if top-k edges don't recover at >50% frequency across bootstraps, no downstream causal result is interpretable.** Fix estimation before proceeding.

### Stage 4 — Directed effect matrix

Distributional shift, not mean shift — Perturb-seq responses are frequently variance changes or bimodality.

```julia
using Distances

# E[g, j] = distributional shift of gene j under perturbation of g
function effect_matrix(Xn, targets, control_label, target_list)
    E = zeros(length(target_list), size(Xn, 2))
    ctrl = findall(==(control_label), targets)
    for (gi, g) in enumerate(target_list)
        pert = findall(==(g), targets)
        length(pert) < 30 && continue          # power floor
        for j in axes(Xn, 2)
            E[gi, j] = energy_distance(view(Xn, pert, j), view(Xn, ctrl, j))
        end
    end
    return E
end
```

Calibrate significance against **non-targeting-vs-non-targeting splits**, not a parametric null — this absorbs batch and depth structure. Benjamini–Hochberg across all (g, j).

**Checkpoint 3 — positive control.** Every perturbation must show a strong self-effect (`E[g,g]` large, direction = down for CRISPRi). Targets failing this had ineffective knockdown; exclude them from orientation but keep them as response genes. Also confirm known hESC pluripotency relationships (POU5F1, NANOG, SOX2 and their documented targets) appear. If they don't, something is wrong upstream and no amount of downstream sophistication will rescue it.

### Stage 5 — Orientation (H1)

For each skeleton edge with both genes perturbed, compare `E[X,Y]` vs `E[Y,X]`:

- asymmetric → orient
- both large → feedback / cycle, flag, do not force a direction
- neither → likely confounding or artifact, flag for removal

Report the fraction of skeleton edges orientable, and the fraction flagged cyclic. The cyclic fraction is itself a result worth reporting, since acyclicity is an assumption most competing methods make and ADAPRE specifically relaxes.

### Stage 6 — Intervention-indicator PID (H2, the novel part)

For candidate pair (X, Y) and perturbation g upstream of both:

```julia
# ⚠️ depends on pid_triple existing
function mediation_pid(Xd, targets, g, x_idx, y_idx, control_label)
    cells = findall(t -> t == g || t == control_label, targets)
    Ig = Int.(targets[cells] .== g) .+ 1      # binary source, 1-indexed bins
    xs = Xd[cells, x_idx]
    ys = Xd[cells, y_idx]
    return pid_triple(Ig, xs, ys; measure = :imin)
end
```

Interpretation:

| Pattern | Reading |
|---|---|
| `unique(Ig→Y \| X) ≈ 0` | complete mediation: `g → X → Y` |
| `unique(X→Y \| Ig) ≈ 0` | **siblings, not connected — the false-positive filter** |
| high redundancy, both uniques low | confounding by g |
| synergy | interaction / gating |

Estimation is easier than the gene–gene case because `Ig` is binary. Aggregate across all upstream g per pair.

**Checkpoint 4 — redundancy-measure robustness.** `I_min` systematically overstates redundancy (Bertschinger et al.). Recompute the sibling calls under BROJA and Ince's `I_ccs` on a subsample. **Gate: if sibling classification flips substantially between measures, the result is an artifact of `I_min` and must be reported as such, not papered over.**

### Stage 7 — Baselines (H3, the kill criterion)

Run on the identical candidate set:

1. Linear mediation regression: `Y ~ X + Ig`, test whether the X coefficient survives conditioning on Ig.
2. Conditional independence test `X ⊥ Y | Ig` (partial correlation, and a nonparametric alternative).
3. Correlation-only skeleton with the same orientation and pruning layers — isolates whether PID contributes anything over correlation.
4. GRNBOOST2 / ppcor comparators, per SCING's framing.

**This is the decisive comparison.** If (1) or (2) matches the PID sibling classification at equal cost, say so plainly in the paper. A clean negative on H3 with a solid H1/H2 is still a publishable, useful result — and given CausalBench's finding that interventional methods fail to beat observational baselines on real data, a well-executed negative is arguably the more valuable contribution.

### Stage 8 — Invariance filtering

For surviving oriented edges, test stability of `P(Y | pa(Y))` across perturbation regimes that did not target Y (Invariant Causal Prediction, Peters/Bühlmann/Meinshausen). Catches parent sets that are merely predictive.

### Stage 9 — External validation

- **H1 ATAC.** ENCODE H1 accessibility (e.g. GSE267154 / ChromBPNet-associated data) plus motif scanning — is the oriented regulator's motif in an accessible region near the target? Independent of expression.
- **Replogle K562/RPE1.** Which edges replicate across cell type? Directly tests the context-transfer question. Expect core machinery to replicate and lineage-specific regulation not to.
- **Pluripotency literature.** Curated hESC network as a partial gold standard, with the caveat that it's incomplete and biased toward well-studied genes.

---

## 5. Checkpoint summary

| # | Gate | Failure action |
|---|---|---|
| **0** | ≥~2,000 reciprocal testable pairs | Add Replogle as discovery set, or restrict scope |
| **1** | MI invariant under equal-frequency binning | Fix discretization; the invariance claim is wrong |
| **2** | Bootstrap edge recovery >50% at top-k | Fix estimation before any causal claim |
| **3** | Self-effects present; known pluripotency edges recovered | Debug upstream; do not proceed |
| **4** | Sibling calls stable across redundancy measures | Report as `I_min` artifact |
| **5** | PID beats mediation regression | Publish as negative result on H3 |

Checkpoints 0, 2 and 5 are the ones most likely to fail. Reaching each of them early, rather than after building the full pipeline, is the main design goal of this ordering.

---

## 6. Known limitations to state explicitly in any write-up

- **Timescale.** CRISPRi Perturb-seq reads steady state days post-knockdown. "Direct" here means _not mediated by other measured genes at equilibrium_, not direct transcriptional binding. This silently redefines what an edge means and should be stated in the abstract, not buried.
- **Absence of effect ≠ absence of edge.** Paralog redundancy, buffering, partial knockdown.
- **Soft interventions.** CRISPRi attenuates rather than severs incoming edges; the do-calculus argument for edge deletion is approximate.
- **Cycles.** Reciprocal asymmetry reads genuine feedback as ambiguity.
- **PUC is gene-set dependent.** Edge scores change with the gene filter. Report the filter.
- **Single cell line.** H1 hESC only; context transfer is a hypothesis tested in Stage 9, not an assumption.
- **`I_min` is the weakest link** in the PID literature, and it is what PIDC uses.

---

## 7. Immediate next actions

1. **Send me the real FastPIDC.jl API**, especially whether triple-level PID components are accessible. Section 2 is unusable until this is resolved.
2. **Run Checkpoint 0.** One pass over `ad.obs` plus per-target energy distance. Decides project shape.
3. **Read ADAPRE and D-SPIN properly** before writing method text — they define the positioning.
4. **Literature re-check.** My search was one round and ADAPRE is from February 2026; search bioRxiv and Scholar for PID/information-theoretic interventional GRN work from the last few months before assuming the gap is open.
