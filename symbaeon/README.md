# SymbAeon

Host–pathogen coevolution analysis for the *aeon stack.

SymbAeon pairs host and pathogen gene alignments on a shared **individual
axis** (rather than the shared phylogenetic-taxon axis used by HyphAeon's
within-alignment coselection) and screens for cross-species coevolution.

See `examples/hpi/hpi_integration_plan.md` for the design plan.

## Commands

| Command | Description |
| :--- | :--- |
| `symbaeon coselect` | Cross-alignment site-site association between a host MSA and a parasite MSA across paired individuals |

## Install

```bash
pip install -e ./symbaeon   # pulls aeon-core + hyphaeon
```
