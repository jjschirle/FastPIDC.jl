**Status:** draft. Companion to `interventional_pidc_plan.md`. Shares Stages 1–5 of that pipeline; this document covers only the calibration layer. Sections marked ⚠️ contain assumptions to correct before implementation.

---

## 0. What is being changed, and why it is legitimate

PIDC's final step fits, for each gene $i$, a gamma distribution to the empirical set $\{\mathrm{PUC}(i,j)\}_{j\neq i}$ and converts each edge score into a tail probability under that gene's own fit. The step is usually described as removing gene-specific nuisance variation, and it does. But it also carries an unstated topological assumption:

> every gene's score distribution has the same _shape_, therefore at a fixed confidence threshold every gene has approximately the same expected degree.

That is a homogeneity prior on degree. It is not neutral and it is almost certainly wrong for regulatory networks, where out-degree is heavy-tailed (a TF can act on thousands of loci) while in-degree is sharply constrained (finite promoter real estate). So the question is not whether to impose a degree assumption — PIDC already does — but whether a better one improves recovery.

**This is a re-calibration project, not a new score.** PUC is untouched. Only the mapping from PUC to edge confidence changes. That keeps the contribution narrow, cheap to implement, and directly comparable against stock PIDC as a baseline.

### Four variants, in increasing ambition and decreasing confidence

| # | Variant | Assumption added | Prior |
|---|---|---|---|
| **V2** | Hierarchical/empirical-Bayes gamma fits | none beyond stock PIDC | **High** — pure variance reduction |
| **V1** | Direction-aware calibration (post-orientation) | in/out asymmetry | High — mechanistically grounded |
| **V4** | Degree-corrected null (configuration-model style) | expected-degree structure | Moderate |
| **V3** | Global degree-regularized thresholding | explicit $P_{\mathrm{deg}}$ | Moderate, with a known hub-suppression failure mode |

Implement in the order **V2 → V1 → V4 → V3**. V2 is a strict improvement with no new assumptions and plausibly fixes real instability; V3 is the most likely to actively harm results.

---

## 1. Prior art and the honest novelty position

- Degree-corrected and degree-penalized estimation is established in the sparse graphical model literature (degree-corrected graphical lasso and relatives), and degree priors appear in Bayesian network structure learning.
- Configuration-model nulls are standard in community detection.
- **I am not aware of published work modifying PIDC's gamma calibration specifically.** Same epistemic status as the mediation idea in the companion plan: a construction that follows from available pieces, not something citable.

**Action before writing method text:** search the degree-corrected graphical model literature properly. The statistical problem there is close enough that someone may have solved a borrowable form, and if so this becomes an application rather than a method paper. That is a fine outcome but it changes the framing.

---

## 2. The circularity trap — read before designing evaluation

If $P_{\mathrm{deg}}$ is fit to a curated network (TRRUST, DoRothEA, ChIP-derived) and then used to regularize inference, **you cannot claim the inferred network recovers realistic topology.** You assumed it. Any evaluation scoring topological realism becomes vacuous, and reviewers will say so.

Three compounding problems with importing a prior from curated sources:

1. **Ascertainment bias.** Khanin & Wit (2006): observed scale-freeness in biological networks partly reflects that well-studied genes accumulate more recorded edges.
2. **Scale-freeness is contested.** Broido & Clauset (_Nat Commun_ 2019) tested ~1,000 real networks and found strict power laws rare, with log-normal fitting at least as well. Do not hard-code $P(k)\sim k^{-\gamma}$.
3. **Binding degree ≠ transcriptomic degree.** This is the most serious for this application. A TF with 3,000 binding sites may move 40 genes at steady state. Curated degree distributions are binding-derived; PIDC edges are steady-state covariation. The relevant distribution is that of the _effective functional_ network, and it is not the one in the literature.

### The way out: fit the prior on held-out perturbation targets

The Arc hESC dataset provides ~300 CRISPRi targets and a directed effect matrix $E$ (companion plan, Stage 4). Effective out-degree can therefore be **measured** in the correct cell context, functionally defined, with no curation bias:

$$\hat{k}^{\mathrm{out}}_g = \#\{\,j : E[g,j] \text{ significant vs. NTC-split null}\,\}$$

Fit $P_{\mathrm{deg}}$ on a subset of targets, evaluate calibration on held-out targets. 5-fold splits over 300 targets. This is the experiment that makes the whole line defensible, and it is only possible because interventional data is available.

**⚠️ Checkpoint D0 (gate, run first):** confirm enough targets have well-estimated effective out-degree — sufficient cells per condition, self-effect present (companion Checkpoint 3), non-degenerate $\hat k^{\mathrm{out}}$ spread. If $\hat k^{\mathrm{out}}$ is near-zero for most targets or has no dynamic range, there is no prior to fit and the project reduces to V2 alone. Cheap to check; decides scope.

Note the asymmetry this leaves: **in-degree cannot be measured this way.** Estimating $\hat k^{\mathrm{in}}_j$ requires knowing how many _regulators_ of $j$ were perturbed, and with 300 targets out of ~20,000 genes the sampling fraction is far too low. So in-degree constraints must come from either a literature prior (with all the caveats above, stated) or from ATAC/motif counts in accessible regions near the promoter — which is at least context-matched, if still binding-derived rather than functional. Prefer the latter and be explicit that it is a weaker leg than the out-degree side.

---

## 3. V2 — Hierarchical gamma calibration

**Do this first.** Stock PIDC fits each gene's gamma independently from $p-1$ PUC values. For genes with few informative partners — low-expression, low-variance, heavily filtered — that fit is noisy, and the noise propagates directly into edge confidence. This is a plausible contributor to the bootstrap instability flagged in companion Checkpoint 2.

Fix: put a prior on the gamma parameters across genes and shrink each gene's fit toward the global distribution, with shrinkage inversely proportional to that gene's effective information content.

Parameterize gamma by shape $\alpha_i$, rate $\beta_i$; work on the log scale:

$$\log\alpha_i \sim \mathcal{N}(\mu_\alpha,\sigma_\alpha^2), \qquad \log\beta_i \sim \mathcal{N}(\mu_\beta,\sigma_\beta^2)$$

Method-of-moments per gene, then James–Stein-style shrinkage toward the pooled mean is adequate and avoids MCMC over ~20,000 genes:

```julia
# ⚠️ assumes access to per-gene PUC vectors; see companion plan §2
using Statistics, Distributions

"""
Shrink per-gene gamma fits toward a global fit. `puc` is genes × genes
(or a vector of per-gene PUC vectors). Returns shrunk (α, β) per gene.
"""
function hierarchical_gamma_fits(puc_by_gene::Vector{<:AbstractVector})
    fits = [fit_mle_gamma_mom(v) for v in puc_by_gene]
    logα = [log(f[1]) for f in fits]
    logβ = [log(f[2]) for f in fits]
    μα, μβ = mean(logα), mean(logβ)
    τα, τβ = var(logα), var(logβ)

    out = similar(fits)
    for i in eachindex(fits)
        n  = length(puc_by_gene[i])
        # per-gene sampling variance of logα ~ O(1/n); crude but serviceable
        vα = 1 / max(n - 1, 1)
        vβ = 1 / max(n - 1, 1)
        wα = τα / (τα + vα)          # shrinkage weight toward own estimate
        wβ = τβ / (τβ + vβ)
        out[i] = (exp(wα * logα[i] + (1 - wα) * μα),
                  exp(wβ * logβ[i] + (1 - wβ) * μβ))
    end
    return out
end

function fit_mle_gamma_mom(v)
    m, s2 = mean(v), var(v)
    s2 <= 0 && return (1.0, 1.0)
    β = m / s2; α = m * β
    return (α, β)
end
```

Refinements worth trying: use _effective_ sample size rather than raw $n$ (genes with many tied/zero-bin cells have less information than $n$ suggests); consider shrinking toward a covariate-dependent mean, e.g. regressing $\log\alpha$ on detection rate and mean expression, so shrinkage respects known nuisance structure rather than pulling everything to one point.

**Checkpoint D1 (gate):** bootstrap edge-recovery frequency at matched density, V2 vs. stock PIDC, on control cells. V2 must **not** reduce stability. Expected result is an increase concentrated in low-information genes — stratify the comparison by detection rate to see it. If stability is unchanged, V2 is a null result: report it and move on, since it costs little.

---

## 4. V1 — Direction-aware calibration

PUC is symmetric, so a single per-gene gamma cannot express in/out asymmetry. After companion Stage 5 supplies orientations from reciprocal effect asymmetry, calibrate the two roles separately:

- gene acting as **regulator** (out-edges): permissive, heavy-tailed null — do not penalize high out-degree
- gene acting as **target** (in-edges): strict, thin-tailed null — penalize in-degree beyond $k^{\mathrm{in}}_{\max}$

Simplest defensible implementation is a soft in-degree penalty rather than a hard cap:

```julia
"""
Re-rank oriented edges with a soft in-degree penalty.
`edges` :: Vector{(src, dst, score)} sorted descending by score.
Greedy accept with penalty growing in the target's current in-degree.
"""
function indegree_penalized_select(edges, n_genes; kmax = 8, λ = 1.0, budget = Inf)
    indeg = zeros(Int, n_genes)
    kept = similar(edges, 0)
    for (s, d, sc) in edges
        penalty = λ * max(0, indeg[d] - kmax)
        sc - penalty <= 0 && continue
        push!(kept, (s, d, sc))
        indeg[d] += 1
        length(kept) >= budget && break
    end
    return kept
end
```

Sweep $k^{\mathrm{in}}_{\max}$ and $\lambda$; select on **held-out perturbation targets**, never on the fitting set.

**Checkpoint D2 (gate):** V1 must improve recovery of held-out $E$-derived edges over V2 at matched density. If it does not, the in-degree constraint is either wrong for the effective-functional network or the orientations are too noisy to support it — diagnose which by re-running with orientations restricted to the most confidently asymmetric pairs only.

---

## 5. V4 — Degree-corrected null

Rather than changing the prior, change the null. Score an edge by excess over a null that preserves expected degrees — the configuration-model logic from community detection, applied to a weighted score matrix.

For gene strengths $s_i=\sum_j \mathrm{PUC}(i,j)$ and total $S=\sum_i s_i$, the expected score under a degree-preserving null is $\mathbb{E}[\mathrm{PUC}(i,j)] \approx s_i s_j / S$, giving

$$\tilde s_{ij} = \mathrm{PUC}(i,j) - \gamma \frac{s_i s_j}{S}$$

with resolution parameter $\gamma$. This asks the question PIDC's gamma step was groping toward — _is this edge surprising given how connected these two genes generally are_ — but does so jointly rather than per-gene. It is O(p) to compute from row sums, so it is essentially free.

Calibrate significance by permutation within degree strata rather than parametrically. Note the tension with V2: both address gene-level nuisance, so run them separately before combining, and check they are not double-correcting.

---

## 6. V3 — Global degree-regularized thresholding

The most ambitious and the most dangerous. Choose the edge set jointly:

$$\hat E = \arg\max_E \ \sum_{(i,j)\in E} s_{ij} \ + \ \lambda \sum_i \log P_{\mathrm{deg}}(k_i)$$

Combinatorial; a greedy add/swap or Lagrangian relaxation is adequate. Practical effect: redistribute edges away from genes that accumulated many marginal ones toward genes with a few strong ones.

**Known failure mode, stated up front:** if $P_{\mathrm{deg}}$ has too thin a tail, V3 actively suppresses genuine hubs — and hubs are usually the thing most worth finding. Mitigations: fit $P_{\mathrm{deg}}$ from measured $\hat k^{\mathrm{out}}$ (§2) rather than a curated network; use a flexible heavy-tailed family (log-normal, Weibull, discrete power-law-with-cutoff) and select by AIC rather than assuming a form; report results across $\lambda$ including $\lambda=0$ so hub suppression is visible.

**Checkpoint D3 (gate):** track recovery of _known_ hESC hubs (POU5F1, NANOG, SOX2) as a function of $\lambda$. If hub out-degree collapses before overall recovery improves, V3 fails on this data — report the negative and stop. This is a likely outcome and worth saying so in advance.

---

## 7. Evaluation, shared across variants

Everything at **matched network density**; gamma-calibrated scores have no absolute meaning across runs.

Primary metrics:

1. Held-out perturbation-target edge recovery (AUPR against significant $E$ entries for targets excluded from prior fitting). **Primary.**
2. Bootstrap edge stability at matched density.
3. Hub identification: do measured high-$\hat k^{\mathrm{out}}$ genes rank as high-degree?
4. In-degree distribution realism, evaluated _only_ on the leg not used for fitting.
5. Replication in Replogle K562/RPE1 — separates cell-type-invariant structure from hESC-specific.

Baselines that must appear in every comparison: stock PIDC; PIDC with a plain global (non-hierarchical) gamma; correlation skeleton with identical calibration applied. The third isolates whether any of this interacts specifically with PID or would help any score matrix equally — the same kill-criterion logic as H3 in the companion plan.

---

## 8. Checkpoint summary

| # | Gate | Failure action |
|---|---|---|
| **D0** | Enough targets with non-degenerate $\hat k^{\mathrm{out}}$ | Restrict to V2; no fitted prior possible |
| **D1** | V2 does not reduce bootstrap stability | Report as null result, retain stock calibration |
| **D2** | V1 beats V2 on held-out $E$ recovery | In-degree prior wrong, or orientations too noisy |
| **D3** | Known hubs survive increasing $\lambda$ | V3 fails on this data; report negative |
| **D4** | Gains do not appear equally for a correlation skeleton | Reframe as generic recalibration, not PIDC-specific |

D0 and D3 are the likely failure points. D4 is the framing risk.

---

## 9. Limitations to state explicitly

- **Effective functional degree ≠ binding degree.** The prior is fit to steady-state transcriptomic response, days post-CRISPRi. That is the right target for calibrating PIDC, but it is not the degree distribution reported in the ChIP literature and should not be compared to it.
- **In-degree leg is weak.** Not measurable from 300 targets; sourced from ATAC/motif or literature, both binding-derived.
- **Soft interventions.** CRISPRi attenuates rather than severs; $\hat k^{\mathrm{out}}$ is a knockdown-efficiency-weighted quantity. Use per-cell efficiency estimates (mixscape-style) per companion Stage 1.
- **Single context.** Degree structure is itself context-dependent; hESC-fit priors need not transfer, which is what metric 5 tests.
- **Circularity remains partial.** Held-out splits address it for out-degree; the in-degree leg cannot be fully cleared and any topological-realism claim must be scoped to out-degree only.

---

## 10. Immediate next actions

1. **Run Checkpoint D0** — requires only the effect matrix from companion Stage 4.
2. **Implement V2** and run D1. Smallest change, best expected value, independent of everything else here.
3. **Literature search on degree-corrected graphical models** before writing any method text (§1).
4. **⚠️ Confirm FastPIDC.jl exposes per-gene PUC vectors and allows calibration to be disabled** (`calibrate = false` or equivalent). Every variant here requires raw uncalibrated PUC. If the gamma step is baked in and not bypassable, all four variants are blocked and the package needs patching first.
