# Literature review — ADAPRE, D-SPIN, degree-corrected models, recent PID/interventional-GRN work

Follow-up to both plan documents' §"immediate next actions" (interventional plan item 3,
companion plan item 3). Read against the current state of the branch: `STATE.md`,
`LOG.md` (Stage 4 / Checkpoint D0, cell-cycle-conditioning negative result, commit `11a7fcb`).

---

## (a) ADAPRE — summary + relevance

**Causal gene regulatory network inference from Perturb-seq via adaptive instrumental variable
modeling**, bioRxiv, Feb 2026 ([link](https://www.biorxiv.org/content/10.64898/2026.02.18.706642v1)).

Treats each CRISPRi guide's intervention indicator as an instrumental variable and models UMI
counts with a **Poisson–lognormal observation layer**, separating measurement noise from true
expression. Its central methodological move, and the one most relevant here: it applies
**gene-specific adaptive penalties to correct strength-dependent degree bias** — the explicit
finding is that genes with stronger/more efficient knockdowns get estimated with spuriously
higher out-degree and are disproportionately inferred as network hubs, distorting topology under
heterogeneous CRISPRi efficiency. It also relaxes the acyclicity assumption most competing methods
(NOTEARS-style) impose, recovering potentially cyclic structure. Evaluated on genome-wide K562
Perturb-seq; networks enriched for known biological interactions, with coherent leukemia-associated
subnetworks recovered.

**Relevance:** this is a close structural cousin of this project's reopened Checkpoint D0 problem
(`LOG.md`, commit `11a7fcb`) — both are "some perturbation targets look like they hit everything"
inflation problems in $\hat k^{\mathrm{out}}$. ADAPRE's diagnosis is **knockdown-efficiency
heterogeneity**, not cell-state/cell-cycle — a different specific covariate than the one already
tried and rejected here, but the same general shape of fix: **regress/penalize by a per-target
covariate that predicts spurious degree, not a global correction.** Concretely worth checking as
a next step before inventing a new cell-state axis: **is measured $\hat k^{\mathrm{out}}_g$
correlated with per-target knockdown efficiency** (already computed in Checkpoint 0's self-effect
screen, `self_effect_screen.csv` — log2FC of the target's own gene)? If the worst offenders in
`k_out.csv` (or `k_out_resid.csv`) are also the strongest knockdowns, this is a cheap, already-half-computed
check and a plausible mechanism distinct from the cell-cycle hypothesis that just failed. This
should be checked before the three follow-ups already listed in `STATE.md`'s reopened D0 row (PCA/
pseudotime state axis, stratified null, chromatin-regulator plausibility check) — it's cheaper than
all three and directly informed by a paper doing almost exactly this correction on closely related
data (also K562/CRISPRi Perturb-seq).

Also directly relevant to the interventional plan's own item 3 ("ADAPRE... better than this plan on
soft interventions and feedback... Primary methodological competitor"): confirmed as still the
right framing — ADAPRE remains the closest published competitor and this project's positioning
against it (PID-specific triple decomposition for sibling/mediation calls, not just network
recovery) is unchanged by anything found here.

## (b) D-SPIN — summary + relevance

**D-SPIN constructs regulatory network models from scRNA-seq that reveal organizing principles of
perturbation response**, *Cell*, 2026 ([PubMed](https://pubmed.ncbi.nlm.nih.gov/42127893/),
[GitHub](https://github.com/JialongJiang/DSPIN)).

Models the joint distribution of transcriptional states as a **spin-glass / Markov random field**
(maximum-entropy), with pairwise regulatory weights $J$ and condition-specific external fields $h$
representing perturbations (genetic, chemical, or physiological). A single unified network is fit
jointly across all conditions — perturbations shift $h$, not $J$ — which lets weakly-correlated gene
pairs still register a real edge if a perturbation moves them together. The key claimed advantage
over correlation/MI-based approaches, stated explicitly in the paper: genes under **persistent
multi-input inhibition** in the unperturbed state can have near-zero correlation and near-zero
mutual information with their true regulators (because their expression is pinned near floor by
redundant suppression) — a perturbation that lifts one input reveals the hidden coupling by
changing the *pattern* of correlation, not by increasing correlation with any single measured gene.
D-SPIN is explicitly **not** an information-theoretic/PID method — it's parametric (fits a
maximum-entropy graphical model), scales to thousands of genes/conditions/millions of cells via a
different inference machinery entirely (pseudolikelihood / contrastive-divergence-style fitting,
not binning + information estimation).

**Relevance:** unchanged from the interventional plan's own framing ("Not PID. Primary conceptual
competitor.") — nothing found here softens or sharpens that positioning. One point worth adding to
the eventual write-up: D-SPIN's "hidden regulatory interaction via persistent suppression" failure
mode is a **different** blind spot than PIDC's degree-homogeneity problem (this repo's companion
plan) — a gene pinned at floor by redundant inhibition would show *low* $\hat k^{\mathrm{out}}$ *and
in* under both approaches for reasons neither addresses, so D-SPIN's contribution is complementary
to, not overlapping with, either of this repo's two plans. Not a reason to change scope, just a
citation-accuracy note for the eventual methods section.

## (c) Degree-corrected graphical models — what's new

No hits specifically on "degree-corrected graphical lasso" as a named method beyond what the
companion plan's §1 already cites (degree-weighted Lasso, Khanin & Wit ascertainment-bias critique,
Broido & Clauset's scale-freeness skepticism). Adjacent 2025–2026 work found but not overlapping in
approach:

- **tvsfglasso** (time-varying scale-free graphical lasso, *PLOS Comp Bio* 2025) — extends
  graphical lasso with a scale-free-network penalty for time-series data. Different problem
  (temporal, not perturbation-based degree correction) but confirms scale-free-penalized graphical
  estimation is still an active line; worth a citation as "concurrent, different setting" rather
  than a competitor.
- **GRNFormer** (graph transformer for GRN inference, *Bioinformatics* 2026) — deep-learning
  approach, no explicit degree-correction step comparable to this project's V1–V4; not a direct
  competitor to the companion plan's calibration-layer framing.

**Verdict:** the companion plan's claim in §1 ("I am not aware of published work modifying PIDC's
gamma calibration specifically... same epistemic status as the mediation idea") still holds. No
new work closes this gap. The novelty framing is unchanged.

## (d) Recent PID/interventional-GRN work — what's new

- **ADAPRE** (above) — already the plan's primary methodological competitor; nothing new to add
  beyond the degree-bias-correction mechanism noted in (a).
- **CausalBench follow-up**: the original CausalBench finding cited in the interventional plan
  ("interventional methods do not beat observational ones on real data") has a 2024–2025 sequel —
  the **CausalBench challenge** (arXiv 2308.15395) reports that methods built specifically for the
  challenge do meaningfully better than the priors CausalBench originally benchmarked, "constitut[ing]
  a major step towards alleviating the limitations identified with CausalBench... utilization of the
  interventional information." **This partially tempers the plan's own citation** — the H3 kill
  criterion (does PID beat plain mediation regression?) is still the right experiment to run, but
  the interventional plan's framing that interventional methods broadly "fail" against observational
  ones on real data is now dated by ~2 years and should be stated with that caveat, not as settled.
- **"When Does GRN Inference Break?"** (arXiv 2605.04930, 2026) — a controlled diagnostic study of
  causal/correlational GRN methods under injected pathologies (dropout, latent confounders,
  cell-type mixing, network density, feedback). Two findings bear directly on open items in this
  project: (1) **latent confounders degrade all methods equally** ("the great equalizer") —
  without interventional data, no observational method can separate confounding from causation,
  which is exactly the shape of this project's own reopened D0 problem, just framed from the
  benchmark side rather than the applied side; (2) **mutual-information/discretization-based
  methods are the most fragile to dropout specifically** (ΔAUPRC ≈ −0.7 vs. Pearson's −0.28 at high
  dropout), because "MI relies on equal-frequency discretization into 6 bins, and at high dropout
  most observed entries are forced to zero" — this is a direct, if indirect, endorsement of Stage
  2's zero-as-own-bin discretizer design already implemented on this branch (commit `b139fc9`):
  giving zeros their own bin rather than letting dropout collapse into the discretization is exactly
  the kind of fix this diagnostic paper's failure mode calls for. Worth citing as external support
  for that design choice.
- **PSGRN** (*Science Advances*, self-training with synthetic gold standards) and a Dec-2025 bioRxiv
  **"Comparison of Interventional Causal Structure Learning Algorithms for GRN Inference"** — both
  benchmark-type papers, neither uses PID/information-theoretic decomposition specifically. No PID
  triple-decomposition-with-intervention-indicators approach was found anywhere in the 2025–2026
  literature searched. **The gap the interventional plan claims in §0 ("no published method uses
  CRISPR intervention indicators as PID sources") still appears open.**

## (e) Relevance to the reopened D0 cell-state-confound problem

Two concrete, actionable leads, both cheaper than the three already-listed follow-ups in
`STATE.md`'s D0 row:

1. **Knockdown-efficiency confound (from ADAPRE, (a) above).** Check whether $\hat k^{\mathrm{out}}_g$
   (either version — original or cell-cycle-residualized) correlates with each target's own
   knockdown strength (log2FC from `self_effect_screen.csv`, already computed in Checkpoint 0).
   This is a five-minute correlation check against data that already exists on disk, and it's the
   mechanism a directly comparable published method (ADAPRE, also CRISPRi/K562 Perturb-seq)
   identified and corrected for. If it correlates, the fix pattern is the same shape already
   attempted (residualize/penalize by a covariate) but with a different, better-targeted covariate.
2. **"Latent confounder = great equalizer" framing (from (d) above).** The diagnostic-study
   finding that latent confounders degrade *all* methods equally, observationally, is a useful
   framing point for the eventual write-up regardless of which correction (if any) works: it
   predicts, independent of this project's own results, that no purely observational fix inside
   the energy-distance/PUC framework can fully separate a shared-cell-state confound from real
   direct effects — only genuinely interventional structure (e.g. comparing across perturbations
   with matched cell-state shift but different targets, or explicit ICP-style invariance testing,
   already Stage 8 in the interventional plan) can. This argues for **not over-investing** in
   finding the exact right regression covariate for D0 and instead treating a residual confound as
   expected, to be filtered later by Stage 8 (Invariance filtering) rather than fully resolved at
   Stage 4.

Neither of these invalidates the cell-cycle regression attempt already run (commit `11a7fcb`) —
it was a reasonable first thing to try and its negative result is itself informative (rules out
cell-cycle specifically, ~1.5% variance explained is a real, checked number). But (1) above is a
strictly higher-priority next check than the three items STATE.md currently lists, given it reuses
existing data and is motivated by a directly comparable published result.

## (f) Verdict — does this change STATE.md's checklist or next steps?

**Yes, one concrete addition; everything else confirmed as-is.**

- **Add to STATE.md's reopened D0 row / "Immediate next actions"**: check $\hat k^{\mathrm{out}}_g$
  vs. per-target knockdown efficiency (from `self_effect_screen.csv`) before trying the
  PCA/pseudotime state axis or stratified-null follow-ups — cheaper, reuses existing data, and
  motivated by ADAPRE's directly comparable, published fix for the same failure shape.
- The interventional plan's CausalBench citation should be stated with a "this is now ~2 years old
  and partially superseded by the CausalBench challenge results" caveat in any eventual write-up,
  not changed in approach.
- The zero-as-own-bin discretizer decision (already implemented, commit `b139fc9`) now has
  independent literature support from the 2026 diagnostic-study finding that MI/discretization
  methods are uniquely fragile to dropout-driven zero-inflation — no code change needed, just a
  citation to add later.
- No new competing PID-with-intervention-indicators method was found; the core novelty claim in the
  interventional plan's §0 stands unchallenged.
- No published degree-corrected-PIDC-calibration work was found; the companion plan's novelty claim
  in §1 stands unchallenged.
