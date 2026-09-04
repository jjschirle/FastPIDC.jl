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

## Scratch outputs (not committed)

Intermediate arrays/CSVs from today's session live in the session scratchpad
(`/tmp/claude-1000/.../scratchpad/`), not in the repo: `X_dense.npy` (16 GB densified matrix — 
regenerate, don't commit), `self_effect_screen.csv`, `phenotype_screen.csv`. Regenerate from the
h5ad rather than relying on these persisting.
