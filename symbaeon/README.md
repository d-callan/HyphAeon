# SymbAeon

Host–pathogen coevolution analysis for the *aeon stack.

SymbAeon pairs host and pathogen gene alignments on a shared **individual
axis** (rather than the shared phylogenetic-taxon axis used by HyphAeon's
within-alignment coselection) and screens for cross-species coevolution.

See `examples/hpi/hpi_integration_plan.md` for the design plan.

## Paradigms

SymbAeon detects three distinct kinds of cross-species coupling. They are
complementary — a real interaction can show one, several, or none.

| Paradigm | Question | Command | Engine |
| :--- | :--- | :--- | :--- |
| **Allele-correlation** | Do the same *individuals* carry non-reference alleles at host site X and parasite site Y? | `correlate` | encoded carrier vectors → Fisher/CMH/Spearman + FDR |
| **Selection-correlation (concentrated)** | Do the same *lineages/alleles* drive the selection signal at host site X and parasite site Y? | `coselect --mode edges` | transformer counterfactual driver profiles → cross-cosine edges |
| **Selection-correlation (diffuse)** | Do the same *individuals* carry an elevated selected-divergence burden in both genes? | `coselect --mode burden` | transformer LRT-weighted burden → ancestry-corrected correlation |

The model-based `coselect` runs the PhyloAxialTransformer separately on each
MSA — each side gets its own TN93 distance / positional encoding, so neither
side's substitution rate swamps the other. The concentrated and diffuse
readouts share that single per-side model pass, so `--mode both` (the default)
reports edges *and* the burden correlation for the price of one inference;
`--mode burden` skips the counterfactual driver matrix entirely and is the
cheap option. `correlate` remains for the distinct allele-correlation question.

## Commands

| Command | Description |
| :--- | :--- |
| `symbaeon coselect` | Model-based selection-correlation. `--mode {edges,burden,both}` picks the readout(s); `--method {delta_lrt,delta_lrt_aa,attn}` picks the edges driver granularity (`delta_lrt` reverts one haplotype at a time; `delta_lrt_aa` reverts all carriers of each derived amino acid jointly — more power). `--permute` gives a leaf-shuffle null. |
| `symbaeon correlate` | Allele-correlation site-site association across paired individuals (presence/dosage encoding → Fisher/CMH/Spearman, optional relatedness-PC and strata control, BH-FDR). |

## Install

```bash
pip install -e ./symbaeon   # pulls aeon-core + hyphaeon
```
