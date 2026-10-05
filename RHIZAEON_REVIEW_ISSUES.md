# RhizAeon Code Review — Proposed GitHub Issues

Statistical and scientific issues in `feat/rhizaeon`, ordered by severity.
Scope: code-level issues only (benchmark/baseline-comparison items excluded).
Each entry is formatted as a ready-to-file issue title + body.

---

## Issue 1 — Critical

**Title:** `Post-selection inference: Fisher gate p-values are computed on argmax-selected triplets and positions`

**Body:**

In `rhizaeon/fda.py::extract_fda_breakpoints`, the pipeline selects:

1. Peak position via `argmax` over M cutpoints (`find_peaks` on `max_z`)
2. Recombinant taxon via `argmax` over N taxa
3. Parental pair via `argmax` over N² triplets (`evaluate_triplets_for_taxon`)

…then evaluates `verify_crossover_support`'s Fisher exact test on the selected
triplet at the selected position as if it had been specified a priori.

The only multiplicity correction is `p_crit = min(0.005, 0.15 / N)` (N taxa).
It does not account for the N² parental-pair search or the M-position scan.
The reported p-values are therefore anti-conservative by an unquantified factor.

**Suggested fix:** calibrate the gate empirically — permutation of flank labels
or site order under the clonal null (the prefix engine makes null replicates
cheap), or relabel the Fisher output as a heuristic score rather than a p-value.

---

## Issue 2 — Critical

**Title:** `Fisher crossover gate assumes independent informative sites; flanking sites are maximally linked`

**Body:**

`rhizaeon/fda.py::verify_crossover_support` builds a 2×2 table of informative
sites (parental allele matches) in left vs. right flanks and applies
`fisher_exact`. Fisher's exact test assumes independent draws under the null.

Informative sites within a flank share a single local genealogy — they are in
complete linkage, not independent Bernoulli draws. Under the clonal null this
inflates the effective sample size and produces anti-conservative p-values.
This is the same problem GARD handles via phylogenetic likelihood and PHI via
permutation.

**Suggested fix:** site-permutation or flank-label-permutation null
distribution for the 2×2 statistic, or an explicit effective-sample-size
correction for linked sites.

---

## Issue 3 — High

**Title:** `Streaming medoid recruitment voids the conformal coverage guarantee it advertises`

**Body:**

`rhizaeon/conformal.py::ConformalMetricCalibrator.compute_p_value` implements
the correct finite-sample conformal p-value `(1 + #{s_calib ≥ s_Q})/(n+1)`,
valid under exchangeability.

But `MultiSegmentConformalEngine.stream_queries_dynamic` mutates the medoid
set mid-stream (dynamic recruitment). Post-recruitment p-values are computed
against a calibration set that depends on prior queries — exchangeability is
broken and the "finite-sample coverage guarantee" no longer holds.

Additionally, calibration genomes drawn from the same outbreak/clade are
phylogenetically correlated, so effective n < n_calib even in the static case.

**Suggested fix:** either (a) document recruitment mode as a heuristic without
coverage guarantees, or (b) hold out a fixed calibration set and treat
recruited medoids as a separate, uncalibrated track.

---

## Issue 4 — High

**Title:** `Kinetic Z > 3 described as "statistically significant" but is an uncalibrated heuristic`

**Body:**

`rhizaeon/fda.py::compute_fda_trajectories` computes robust Z-scores
(median/MAD) of kinetic energy, and the docs describe `Z > 3.0` as
"statistically significant topological dislocation."

Problems:

- Kinetic energy is a squared-norm (χ²-like, right-skewed) quantity;
  MAD-normalized scores do not have Gaussian tails.
- Overlapping sliding windows induce strong autocorrelation, so the effective
  number of independent tests is far below M.
- Per-taxon median normalization masks taxa that are recombinant across most
  of their genome (elevated baseline → missed events).

**Suggested fix:** calibrate the threshold against a clonal-null distribution
(permutation or simulation), or present Z as a ranking score and let the
conformal layer carry the statistical interpretation.

---

## Issue 5 — High

**Title:** `No multiple-testing control on the final breakpoint list`

**Body:**

RP-FDA recursion + `find_peaks` evaluates ~M candidate cutpoints genome-wide,
but the only correction lives inside the Fisher gate (`0.15/N` cap). There is
no family-wise or FDR statement for the emitted breakpoint set.

`conformal.py` already implements Simes/Bonferroni/BH aggregation — but it is
wired to the surveillance/segment-typing path, not to breakpoint calls.

**Suggested fix:** apply the existing Simes/BH machinery (or a permutation
null) to the candidate-breakpoint p-values before emitting the final list.

---

## Issue 6 — Medium

**Title:** `Tier-2 trigger constants are hardcoded from single-dataset anecdotes`

**Body:**

`rhizaeon/adaptive.py` / `embed.py` encode thresholds drawn from individual
examples:

- `τ_root = 0.35` (root-token attention, ghost-parent detection)
- `r_ghost ≥ 1.50` (orthogonal residual surge; "84× surge" is one XBB.1
  observation)
- `ρ_retic ≥ 0.70` / `< 0.25` (reticulation concordance classification)
- "0.00% empirical error" claim for Trigger 5 rests on n≈1 datasets

These are plausible but unvalidated as general constants.

**Suggested fix:** evaluate each trigger across the existing benchmark matrix
and report operating curves; until then, label them as illustrative defaults
in docs and CLI help.

---

## Issue 7 — Medium

**Title:** `Inconsistent min_pir defaults across entry points (0.25 / 0.10 / 0.08)`

**Body:**

The same L-PIR statistic has three different significance thresholds:

- `segmentation.py::RhizAeonDetector` — `pir_threshold = 0.25`
- `fda.py::extract_fda_breakpoints` — `min_pir = 0.10`
- `cli.py` `rp-fda` — `--min-pir` default `0.08`

Same statistic, three defaults → irreproducible results across subcommands.

**Suggested fix:** define one canonical default in a shared constants module;
let CLI flags override explicitly.

---

## Issue 8 — Medium

**Title:** `ci_left / ci_right fields mislabeled — plateau bounds are an identifiability interval, not a confidence interval`

**Body:**

`rhizaeon/polisher.py::PolishedBreakpoint` exposes `ci_left` / `ci_right`.
These are the argmax level-set bounds of the profile log-likelihood — the
uninformative plateau where likelihood is flat. That is an identifiability
(support) interval, not a frequentist CI with coverage properties.

The docs mostly call it a "plateau"; the field names invite misreading.

**Suggested fix:** rename to `plateau_left` / `plateau_right` (or
`support_left` / `support_right`) throughout code, JSON output, and docs.

---

## Issue 9 — Medium

**Title:** `ML polisher likelihood model fixes ε = 0.01 with no sensitivity analysis`

**Body:**

`polish_breakpoint_ml` uses `γ = log(3(1−ε)/ε)` — a uniform 3-way error model
with `error_rate = 0.01` hardcoded as default. The cumulative profile is a
random walk whose plateau geometry depends on ε; misspecification sensitivity
is unexamined. Sites are also treated as independent (linkage ignored, same
caveat as Issue 2).

**Suggested fix:** document the error model explicitly; expose sensitivity to
ε in tests; consider estimating ε from non-informative-site mismatch rates.

---

## Issue 10 — Medium

**Title:** `Divergence-weighted parent selection biases attribution toward distant lineages`

**Body:**

`pir.py::evaluate_triplets_for_taxon` selects parents via
`score = PIR × (D1 + D2)`. This systematically prefers maximally divergent
parental pairs — intended to prevent clonal siblings shadowing real parents,
but it biases attribution toward deep lineages when multiple valid pairs exist.

**Suggested fix:** document the bias; consider reporting top-k candidate pairs
(already supported) in output rather than only the argmax.

---

## Issue 11 — Low

**Title:** `W-projection attenuates breakpoints orthogonal to top-k global eigendirections`

**Body:**

`compute_fda_trajectories` projects local Gram matrices through the global
`W = H V_k Λ^{-1/2}`. A local topology change whose signal lives in global
eigendirections > k is attenuated or invisible. `F_frame` partially diagnoses
this, but the attenuation is not quantified per-window.

**Suggested fix:** report per-window residual energy `‖B(s) − B(s)WW⁺‖` as a
diagnostic channel alongside kinetic Z.

---

## Issue 12 — Low

**Title:** `Conformal calibration includes medoid self-distances (score = 0)`

**Body:**

`ConformalMetricCalibrator.fit` computes min-distance-to-medoid over the
calibration set that contains the medoids themselves → each medoid contributes
score 0, inflating the left tail of the calibration distribution (conservative
direction, but distorting).

**Suggested fix:** exclude self-distances (leave-one-out) when computing
calibration scores for medoid members.

---

## Issue 13 — Low

**Title:** `hyphaeon recombine delegation: undeclared dependency + PATH-order bug`

**Body:**

`hyphaeon/cli.py` adds a `recombine` subcommand that subprocess-delegates to
`rhizaeon`, but `rhizaeon` is not in `hyphaeon`'s dependencies:

- `shutil.which("rhizaeon")` prefers any PATH binary (possibly a different
  env/version) over the current interpreter; the same-env fallback
  (`sys.executable -m rhizaeon.cli`) only runs when PATH finds nothing —
  priority order is backwards.
- If rhizaeon is absent, users get a raw `ModuleNotFoundError`, not an
  actionable message.
- It re-exports rhizaeon's entire evolving CLI surface with no version pin.

**Suggested fix:** remove the delegation (each wheel keeps its own CLI), or
move cross-package dispatch to `aeon-core` as a plugin-style umbrella entry
point. If kept, prefer `sys.executable -m rhizaeon.cli` first and emit a clear
"pip install rhizaeon" message on ImportError.

---

## Issue 14 — Low

**Title:** `rhizaeon scan --codon defaults to True; nucleotide alignments silently truncated to L//3`

**Body:**

`cli.py` `scan` sets `--codon` `default=True`; a raw nucleotide alignment run
without `--nt` is silently treated as codons (units = L//3, remainder dropped).

**Suggested fix:** default to nucleotide mode, or validate `L % 3 == 0` and
warn when `--codon` is in effect.

---

## Issue 15 — Low

**Title:** `rhizaeon/pyproject.toml depends on aeon-core>=0.1.0; dist renamed to hyphaeon-core on main`

**Body:**

Branch predates the `aeon-core` → `hyphaeon-core` PyPI rename (main commit
`1586e21`). After rebase, `pip install rhizaeon` will fail to resolve.
`rhizaeon/` also lacks a `LICENSE` file (main commit `d11eada` ships one per
package).

**Suggested fix:** rebase onto main; update dependency to
`hyphaeon-core>=0.1.0`; add `rhizaeon/LICENSE`.

---

## Issue 16 — Low

**Title:** `Duplicate typo'd docs: ALIGMENT.md and ALIGNMENT.md both committed`

**Body:**

`benchmarks/bacterial_recombination_meta_analysis/` contains both
`ALIGMENT.md` and `ALIGNMENT.md` (root and `docs/`).

**Suggested fix:** delete the misspelled copies.
