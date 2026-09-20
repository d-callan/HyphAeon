# Host–Pathogen Interaction Analysis in the *aeon Stack

Design plan for adding host–pathogen coevolution to the HyphAeon monorepo.
Working name for the new package: **`symbaeon`** (symbiosis) — sits next to
`hyphaeon` and `chronaeon`, depends on `aeon-core`. Name is a placeholder;
alternatives: `coaeon`, `hpiaeon`.

---

## 1. What HyphAeon consumes today

`aeon_core.dataset.load_alignment_and_tree(fa_path, nwk_path)` returns:

```
(c, a, d, z, inv, taxa, L, tree_cache)
  c    codon tokens      [L, N, 1]   0-63 sense codons, 64 = stop/ambig
  a    amino-acid tokens [L, N, 1]   0-19 AA, 20 = gap/other
  d    patristic dist    [N, N]      from tree, or TN93 if tree-free
  z    MDS coords        [N, 4]      4-D classical MDS embedding of the tree
  inv  invariable mask   [L]
  taxa taxon names       [N]
  L    codon length
```

`PhyloAxialTransformer.forward_cached(msa_codons, msa_aas, tree_cache)` runs
2-D axial attention over the **(site × taxa)** grid with patristic
distance-decay attention and returns per-site LRTs plus root→leaf attention.

`hyphaeon.epistasis.compute_transformer_attributions` produces
`leaf_attributions [L, N] = root_attn ⊙ delta` (delta = non-consensus
indicator), `lrts [L]`, `pvals`, `consensus_aas`.

**Existing coselection** (`phenotype.py` / `epistasis.py`) is
*within-alignment*: it computes cosine similarity between the attribution
vectors of two sites **across the same N taxa / shared tree**, then
`CESI = sim · sqrt(lrt_u · lrt_v)` → t-test → BH-FDR → TSE sector mining.

## 2. The fundamental gap

Host and parasite genes live in **different alignments with different taxon
axes**. There is no shared tree. The pairing is on the **individual axis**:

```
host MSA:    L_h sites  ×  N_h host haplotypes   (h1/h2 per diploid host)
parasite MSA: L_p sites  ×  N_p parasite clones   (c1/c2 per mixed infection)
pairing:     sample_id  →  {host haplotypes} × {parasite clones}
```

So cross-alignment coselection cannot reuse the shared-taxon-axis cosine
directly. It needs a **pairing layer** that aligns the two MSAs on a common
individual axis, then a **cross-alignment association statistic** computed
across individuals rather than across tree leaves.

Two regimes, same machinery:

- **Population-level** (Band et al., our data): N individuals, shallow
  population trees (TN93 or none). Pairing = exact `sample_id` match.
  Coselection = allele co-occurrence across individuals — a cross-species
  epistasis / GWAS-like scan.
- **Deep-time** (species-level, future): host and parasite phylogenies paired
  by cophylogeny mapping. Same pairing-layer abstraction, different pairing
  source (host↔parasite association table instead of sample_id).

## 3. Proposed architecture

```
aeon-core   (unchanged)  model, tokenization, tree/MDS embedding, io
hyphaeon    (unchanged)  per-alignment LRTs, attributions, epistasis, dms
chronaeon   (unchanged)  dating, phylodynamics
symbaeon    (NEW)        pairing layer + cross-alignment coselection + DMS landscape
```

Dependency decision (2026-09-20): `symbaeon` imports **all of `hyphaeon`**
(which itself pulls `aeon-core`). It needs `meme`, `epistasis`, and `dms`
anyway, so a package-on-package-on-package stack is accepted for now — a
three-layer monorepo stack. Refactor shared scoring primitives down into
`aeon-core` later only if the dependency proves heavy in practice.

`symbaeon` adds only what is genuinely new:

- **`symbaeon.pairing`** — load two MSAs + a pairing manifest, build the
  joint individual axis, handle host diploidy (h1/h2) and parasite mixed
  infection (c1/c2) by expanding each individual to its allele set.
- **`symbaeon.coselect`** — cross-alignment site-site association across
  paired individuals (see §5).
- **`symbaeon.screen`** — en-masse driver: many host genes × many parasite
  genes, emit ranked candidate pairs.
- **`symbaeon.landscape`** — targeted follow-up: within-host, within-parasite,
  and cross epistasis + DMS-informed fitness landscape for a flagged pair.

### Input contract for symbaeon

```
--host-aln   host CDS MSA (FASTA)      headers: {sample}|{gene}|{h1|h2}
--par-aln    parasite CDS MSA (FASTA)  headers: {sample}|{gene}|{c1|c2}
--pairs      pairing manifest (TSV)    seq_id, sample_id, species, gene, tag
--host-tree  optional (else TN93)
--par-tree   optional (else TN93)
--host-cov   optional host covariates (e.g. glycophorin SV genotypes)
```

This is exactly what `reconstruct_paired_seqs.py` already emits.

## 4. Handling ploidy & mixed infection on the pairing axis

Each individual contributes a *set* of sequences per side:

- host: {h1, h2} (diploid pseudo-haplotypes, unphased)
- parasite: {c1} or {c1, c2} (pseudo-clones from het calls)

For a host site u and parasite site v, the per-individual genotype is a
*set* of (host allele, parasite allele) combinations. Reductions (pick one,
document why):

- **Allele-presence encoding** (default): for individual i, host allele at u
  = the set of bases in {h1,h2}; parasite allele at v = set in {c1,c2}.
  Association = do the allele *sets* co-vary across individuals. Robust to
  phasing ambiguity; loses dosage.
- **Dosage encoding**: alt-allele count (0/1/2 host, 0/1/2 parasite).
  Preserves dosage; assumes additive effect.
- **Best-pair encoding**: for each individual pick the single (h,c) pair
  maximizing a linkage score — fragile, only if needed.

Default: **allele-presence**, with dosage as a flag. Both are simple
per-individual feature vectors → standard association statistics apply.

## 5. Stage A — en-masse screen for interacting pairs

Goal: given H host-gene MSAs and P parasite-gene MSAs (all sharing the same
individual axis via the manifest), rank candidate interacting pairs.

Per (host_gene, parasite_gene) pair:

1. Score each MSA separately with `hyphaeon` → per-site LRTs + attributions
   (reuses `compute_transformer_attributions` unchanged; the taxon axis is
   now individuals, tree is TN93/shallow).
2. Build the joint individual axis from the manifest (intersect sample_ids).
3. For each variable host site u × variable parasite site v:
   - encode per-individual allele sets (§4)
   - association statistic across individuals:
     - categorical: chi-square / Fisher on the allele-set contingency table
     - continuous: correlation of attribution-weighted allele indicators
   - cross-CESI analogue: `sim_ind · sqrt(lrt_u · lrt_v)` where `sim_ind` is
     the across-individual association, `lrt_*` the per-site selection LRTs
4. BH-FDR across all (u, v) pairs; rank gene pairs by top site-pair signal
   and by a gene-level aggregate (e.g. mean top-k CESI, or a network
   density score over significant edges).

Output: ranked `candidate_pairs.tsv` (host_gene, parasite_gene, n_sig_sites,
top_cesi, gene_score) + per-pair `site_pairs.tsv`.

Screen scale: for the pilot H≈{GYPA+SVs}, P≈{EBA175, PfRH5}; en-masse mode
targets all expressed blood-group / invasion-ligand genes.

## 6. Stage B — targeted analysis of flagged pairs

For each flagged (host_gene, parasite_gene):

- **Within-host selection**: `hyphaeon meme`/`busted` on the host MSA.
- **Within-parasite selection**: same on the parasite MSA.
- **Cross epistasis**: the Stage-A site-pair map, refined — which specific
  host residues co-vary with which parasite residues; map onto protein
  structure / interface (EBA-175–GYPA, PfRH5–BSG have solved structures).
- **DMS landscape**: `hyphaeon dms` on the parasite ligand to get the
  in-silico mutational landscape; overlay the cross-epistasis sites to ask
  whether host-associated parasite variants sit in high-fitness or
  compensatory regions. Optionally `dms` on the host receptor too, giving a
  two-sided landscape — the coevolutionary "interface" is where both DMS
  maps and the epistasis map coincide.
- **SV integration**: host SV genotypes (Dantu etc.) enter as categorical
  host covariates — test parasite-site association against SV state the same
  way as against a host SNP (the pairing layer treats an SV as a host
  "site" with allele set {I,D}).

## 7. What is reused vs new

| Component | Source | Change |
|---|---|---|
| Codon/AA tokenization, model, tree/MDS embed | `aeon-core` | none |
| Per-site LRTs, attributions | `hyphaeon.epistasis` | none — taxon axis = individuals |
| Within-alignment epistasis, DMS | `hyphaeon` | none — run per MSA |
| Pairing manifest → joint individual axis | `symbaeon.pairing` | **new** |
| Allele-set / dosage encoding | `symbaeon.pairing` | **new** |
| Cross-alignment association + cross-CESI | `symbaeon.coselect` | **new** |
| En-masse pair screening driver | `symbaeon.screen` | **new** |
| Two-sided DMS + epistasis overlay | `symbaeon.landscape` | **new** (wraps `hyphaeon dms`) |

## 8. Open design questions

- **Tree for population data**: TN93 pairwise distances are probably right
  for shallow population MSAs; confirm the model behaves with near-star
  trees. Deep-time mode needs real trees + cophylogeny pairing.
- **Multiple testing**: en-masse mode is (H genes × P genes × sites²) —
  need a gene-level pre-filter (only variable sites, only sites with LRT
  signal) to keep the FDR tractable.
- **Mixed-infection weighting**: should c2 clones from high-MOI samples
  count equally? Option to weight by within-sample allele balance (AD) or
  restrict to single-clone infections for a cleaner signal.
- **SV as sequence vs covariate**: SVs are not codon sites; encoding them as
  categorical host covariates (not in the MSA) is cleaner than fake bases.
- **Package boundary**: if deep-time cophylogeny becomes a goal, the pairing
  layer should accept a generic "association table" so the same code serves
  both regimes.

## 9. Phased implementation

- **Phase 0 (done)**: `reconstruct_paired_seqs.py` → paired FASTAs +
  manifest + SV table from Band et al.
- **Phase 1**: `symbaeon.pairing` + `symbaeon.coselect` on the pilot pair
  (GYPA+SVs × EBA-175; PfRH5 as second parasite gene). Validate that known
  biology (Dantu–EBA-175) surfaces.
- **Phase 2**: `symbaeon.screen` — extend host side to all blood-group genes
  in the bgen, parasite side to all invasion ligands in the VCF.
- **Phase 3**: `symbaeon.landscape` — DMS + epistasis overlay on flagged
  pairs; structure mapping.
- **Phase 4**: managed-access (EGA) dense human genotypes → full BSG +
  genome-wide host screen; deep-time cophylogeny mode.
