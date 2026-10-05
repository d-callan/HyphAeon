# SymbAeon `screen` — design plan

Goal: given a **collection** of host-gene MSAs and pathogen-gene MSAs that all
share the same individual axis (the same paired samples), rank every
host_gene × pathogen_gene pair by how worth-while a full `coselect` deep-dive
is. The screen must be **cheap enough to run on hundreds/thousands of pairs**,
so it deliberately avoids the counterfactual driver matrix (`delta_lrt` /
`delta_lrt_aa`) — that stays in `coselect` for the shortlist.

## 1. The cost structure (why the screen is possible)

`coselect` spends its time in two places:

| step | cost | needed for screen? |
|---|---|---|
| `_side_base` per gene: dedup → `load_alignment_and_tree` → `precompute_tree_cache` → `compute_transformer_attributions` (attrs + LRTs) | **expensive** (one transformer pass) | yes — but **per gene, not per pair** |
| `_driver_matrix` counterfactual (per-site reversion) | **very expensive** (many forward passes/site) | **no** — this is what we skip |
| `cross_cosine` / `_side_burden` / correlations | cheap linear algebra | yes |

The win: `_side_base` is **per-gene**. For H host genes × P pathogen genes we
pay `H + P` transformer passes once, cache each gene's `SideData`, then every
one of the `H×P` pairs is just cheap numpy on cached attributions. The
counterfactual — the part that doesn't scale — is deferred to `coselect` on
the shortlisted pairs.

## 2. Signals

### 2a. Per-gene (computed once, cached)

From `_side_base` we already get `attrs_uniq`, `lrts`, `uniq_seqs`, `ind2uniq`,
`d_mat`, `z`. Cheap derived stats per gene:

- **`frac_sel`** — fraction of sites with `lrt >= min_lrt` (fraction, not
  count, so gene length doesn't dominate). A gene with ~0 selected sites
  can't drive concentrated coselection.
- **`lrt_mass`** — total clipped LRT over selected sites (`sum clip(lrt,0)`).
  Captures "how much selection activity" better than a count — one strong
  site vs many weak ones.
- **`burden_var`** — variance of the per-individual burden. If ~0, there's no
  inter-individual variation to correlate → diffuse coselection impossible.

**"Activity" not "rate".** What we actually want is *genes with a lot of
selection activity* (more likely coevolving than a housekeeping gene), and
`frac_sel`/`lrt_mass`/`burden_var` measure exactly that. A raw substitution-
rate proxy (mean pairwise TN93, root-to-tip divergence) would be **misleading
here**: it conflates neutral drift/relaxed constraint with selection — a gene
can evolve fast without any coevolution-relevant selection. So no rate axis in
the score. (If a collection *is* dated, a `chronaeon` clock rate —
`optimize_latent_convex_hull_root` → `mu_ols` — is a nice bonus annotation
column, but it's not essential and not part of the score.)

**Not used:** `hyphaeon busted` — dev-only, not yet validated; excluded.

### 2b. Per-pair (cheap, on cached SideData)

- **`burden_r_resid` / `p_resid`** — the ancestry-corrected diffuse readout
  (`_side_burden` + Pearson + a small permutation null). **This is the primary
  screening signal** — it needs only base attributions, no counterfactual.
  The correction is a **relatedness kernel**, not a regression on embedding
  coordinates: each individual's burden is predicted from its `resid_knn`
  nearest neighbours in neutral-site TN93 distance space (leave-one-out) and
  the residual keeps the non-ancestry elevation. An earlier MDS+OLS version
  silently erased scattered (non-ancestry) signal — a linear fit on MDS coords
  absorbs *any* group that is linearly separable in embedding space, related
  or not. The kernel only removes elevation shared with actual relatives, so
  a clade member's resid vanishes (its neighbours are its high-burden
  clade-mates) while a scattered individual's survives. `resid_knn` must stay
  below the smallest clade we want to remove (default 5).
- **`burden_r` / `p_raw`** — the raw burden correlation, reported for
  transparency but **not** used in the score (see caveat below).
- **`edge_density`** — `cross_cosine` on the **base** `attrs_uniq` (the `attn`
  method, no counterfactual) → fraction of the L_h×L_p site-pair space with
  `|cos| >= min_sim`. A cheap proxy for concentrated co-structure; the real
  `delta_lrt_aa` edges are computed later by `coselect` on the shortlist.
  Density (not raw `n_attn_edges`) is scored because the raw count scales with
  gene length — a long housekeeping gene accrues edges by size alone.

**Caveat — the raw burden correlation is misleading as a screen.** Two genes
can show a high raw `burden_r` purely because both burdens track the same
shared-ancestry / relatedness structure across individuals, with no real
coevolution (the positive-control clade case: raw r=0.89 but resid r=−0.11).
So the screen ranks on **`burden_r_resid`** (non-ancestry coupling) and treats
the raw value as context. The cost is real: clade-driven coselection *is*
ancestry, so a resid-only screen will under-rank genuinely clade-driven pairs
— acceptable for a triage that prioritizes non-trivial signal, but the raw
column is kept so a reviewer can spot a strong raw+weak-resid pattern
(clade-driven candidate) vs weak-both (no signal).

### 2c. Composite score

Rank pairs by a transparent, documented score — e.g.

```
score = z(burden_r_resid)              # non-ancestry individual coupling
      + z(edge_density)                # concentrated co-structure proxy (scale-free)
      + z(min(frac_sel_h, frac_sel_p)) # BOTH partners under selection
      + z(min(lrt_mass_h, lrt_mass_p)) # BOTH partners have selection activity
```

`min()` on the per-gene terms so a pair only scores high when *both* genes are
active — a hyperactive gene paired with a housekeeping gene shouldn't rank.
Weights configurable; keep it simple and report every component so the ranking
is auditable (per the project rule: document arbitrary thresholds/weights).

## 3. CLI sketch

```
symbaeon screen \
    --host-dir host_msas/  --par-dir par_msas/ \
    --pairs manifest.tsv  [--samples keep.txt] \
    --min-lrt 1.0 --min-sim 0.5 --permute 200 \
    --top 20 --out screen_ranked.tsv
```

- `--host-dir` / `--par-dir`: glob of per-gene codon MSAs. All must share the
  individual axis (same sample_ids) — enforced via the pairing layer.
- `--top`: how many shortlisted pairs to echo as ready-to-run `coselect`
  commands.
- Output TSV: one row per pair with every component stat + the composite score.
- (Optional later) `--rate clock` to annotate a `chronaeon` subs/site/yr column
  when the collection is dated — annotation only, never part of the score.

## 4. Data requirements / open questions

- **Shared individual axis is the hard requirement.** Every host and pathogen
  MSA must be keyed by the same sample_ids. For HIV Gag×Protease this holds
  (173 patients). For a real screen we need a collection where many genes are
  aligned over the *same* individuals — e.g. all HIV genes over the same
  patient cohort, or a host GWAS panel × pathogen genes.
- **Test data: simulated first.** Build a panel of H host × P pathogen genes
  over the same individuals with a few planted coevolving pairs (extend the
  existing `sim_mg94.py` clade/scatter simulators to emit a *collection* of
  genes — some pairs sharing drivers, most not). Confirm the planted pairs
  rank top and housekeeping-like genes rank bottom. Then expand to real data:
  more HIV genes (pol/vif/vpr/env…) over the 173-patient cohort, or a malaria
  panel (CR1 × several Pf genes), or a new collection if needed.
- **Scoring weights** are arbitrary — expose them, and validate the ranking
  against pairs we've already characterized (Gag×Protease should rank high;
  CR1×PfRH4 low).

## 5. Implementation plan

1. `symbaeon/screen.py`:
   - `load_collection(dir, role, manifest)` → list of `PairedSide`.
   - `gene_stats(side, model, device, ...)` → cached `SideData` + per-gene
     stats (reuses `_side_base`, `_side_burden`, `_pearson` from `model.py`).
   - `pair_stats(Sh, Sp, ...)` → burden corr + attn-edge count.
   - `run_screen(...)` → ranked `DataFrame`/TSV.
2. `cli.py`: add `screen` subcommand.
3. Reuse, don't duplicate: `_side_base`, `_side_burden`, `cross_cosine`,
   `edges_from`, `_pearson` are already in `model.py` — import them.
4. Rate: TN93 proxy from `d_mat` first; wire `chronaeon` clock behind
   `--rate clock` once we confirm dated collections.
5. Test: build a small panel (HIV genes over the 173-patient cohort, or a
   simulated scattered/clade panel), confirm Gag×Protease ranks top and the
   ranking is stable to `--permute`.

## 6. What the screen is *not*

- It does **not** replace `coselect` — it nominates pairs for it. The
  counterfactual driver matrix (the actual concentrated-coselection evidence)
  only runs on the shortlist.
- It does **not** detect allele-correlation — that's `correlate`, a different
  question the screen doesn't try to approximate.

## 7. Real-panel result (HSPF malaria, 600-individual subset)

Screened 5 host (CR1, DARC, GYPA, GYPB, GYPE) × 8 parasite (EBA140/175/181,
PfRH1/2a/2b/4/5) genes over 600 shared individuals
(`examples/hpi/results/screen_panel/`, `ranked.tsv`). Honest findings:

- **`edge_density` is dead on real data.** Max `|cos|` of base `attrs_uniq`
  across all pairs is ~0.17 (p99.9 = 0) — far under `min_sim=0.5`. In the sim,
  coupled genes had shared drivers strong enough to show in base attributions;
  real selection is too diffuse/noisy. The concentrated proxy genuinely needs
  the counterfactual `delta_lrt_aa` matrix — i.e. it is not cheap. On real
  data the score effectively runs on 3 of 4 terms.
- **The score reduces to an activity filter.** With `e_dens`≈0 and resid weak,
  ranking is driven by `min(frac_sel)`/`min(lrt_mass)` — CR1 (lrt_mass 160)
  tops the list paired with whatever par gene is active. That's "both genes
  under selection," not "coupled."
- **`coselect` top-vs-bottom contrast: no detectable coselection anywhere.**
  0 counterfactual edges on all tested pairs; resid max |r|=0.13, all
  non-significant. The screen's resid ordering is *faithful* to coselect's
  (top +0.051 → bottom −0.131) — it's the same computation — but the signal is
  too weak to be significant. Monomorphic genes (DARC, GYPE, PfRH2b) correctly
  sink to score −0.95.
- **No assumed positives.** Textbook interactions (EBA175×GYPA, PfRH4×CR1) are
  not ground truth here — a binding interaction need not leave a population
  coselection signature, and GYPA has only 4 unique haplotypes so it can't show
  signal regardless. Validation should be a ranking contrast, not "did we
  recover a known pair."

Caveats on the negative result: n=600 (from 3346) limits power; host genes are
diploid but `_representative_seqs` uses only the first haplotype (h1), dropping
heterozygosity; and the sim's "scattered-individual" coupling model may not
match real coselection, which may be lineage-driven (correctly removed by
resid). Open question: whether `edge_density` should be dropped, replaced by a
cheap counterfactual approximation, or accepted as sim-only — and whether
activity should gate rather than score.
