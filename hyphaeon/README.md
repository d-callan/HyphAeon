# HyphAeon

**The Depth of Evolution** — Neural selection inference using a phylogenetic foundation model.

HyphAeon operates along the **SITE** axis of the foundation model, providing episodic positive selection detection, 3D macromolecular epistasis, in silico DMS sweeps, and phenotype attribution.

## Target Audience

Comparative evolutionary biologists, structural virologists, protein biochemists.

## Installation

HyphAeon requires Python ≥ 3.8 and PyTorch ≥ 2.0. At runtime it auto-selects
the best available device (CUDA → Apple MPS → CPU), so no manual configuration
is needed regardless of which install path you choose.

```bash
pip install hyphaeon
```

| Method | Command | Torch | GPU? |
| :--- | :--- | :--- | :--- |
| **pip** (default) | `pip install hyphaeon` | CUDA-bundled wheel (~550 MB) | NVIDIA GPU if driver matches; else CPU |
| **pip** (CPU-only) | `pip install torch --index-url https://download.pytorch.org/whl/cpu` then `pip install hyphaeon` | CPU-only wheel (~200 MB) | CPU |
| **Bioconda** | `conda install -c bioconda hyphaeon` | CPU-only `pytorch` from conda-forge | CPU only |
| **NVIDIA Jetson** | See [issue #31](https://github.com/veg/HyphAeon/issues/31) | JetPack-native wheel (cp38 only) | Jetson GPU |

You can always install a specific PyTorch build before installing HyphAeon if
none of the above defaults suit your system (e.g. a particular CUDA version,
a custom wheel, or a CPU-only build on a server without GPU).

> [!NOTE]
> **Model weights** are downloaded automatically from [Hugging Face](https://huggingface.co/datamonkey/hyphaeon)
> on first use (cached in `~/.cache/hyphaeon/`). No authentication or token is
> required. Use `--model-variant viral` to select the viral-tuned variant, or
> `--weights /path/to/checkpoint` to use a local file.

## CLI Subcommands

| Command | Aliases | Description |
| :--- | :--- | :--- |
| `hyphaeon meme` | `predict`, `site-selection` | Episodic positive selection (MEME/BUSTED) |
| `hyphaeon epistasis` | `coselection`, `sector`, `network` | 3D macromolecular epistasis |
| `hyphaeon dms` | `essm`, `digital-dms` | In silico DMS sweeps |
| `hyphaeon phenotype` | `phylowas`, `trait` | PhyloWAS phenotype-genotype association |
| `hyphaeon disease` | `pathogenicity`, `variant`, `clinvar` | Disease pathogenicity prediction |
| `hyphaeon filter` | `mask`, `qc`, `clean` | Alignment QC / masking |
| `hyphaeon splits` | `split`, `clades`, `bisection` | Spectral bisection |
| `hyphaeon temporal` | `surveillance`, `longitudinal` | Temporal selection dynamics |
| `hyphaeon list-models` | | List available model variants |

## Key Capabilities

- **Selection Inference** — MEME/BUSTED-style episodic positive selection via neural LRTs
- **Epistasis** — 3D macromolecular epistatic sector detection (CESI)
- **DMS** — In silico deep mutational scanning sweeps (ESSM)
- **Phenotype Attribution** — PhyloWAS/PARS phenotype-genotype associations
- **Spectral Splits** — Phylogenetic spectral bisection for clade analysis

## The Radar & Microscope Flywheel

HyphAeon is the **Microscope**: takes clades flagged by ChronAeon (the **Radar**) and dissects *why* they emerged — identifying positive selection bursts, functional epistatic rewiring, or host-jump phenotypic signatures. Together they form a collaborative flywheel for genomic surveillance and deep evolutionary analysis.

## Reproducible Benchmark Examples

### Example 1: Inter-Site Epistasis & Branch Co-Selection in HIV-1 Reverse Transcriptase

```bash
# Run branch co-selection, sector mining, and export co-selection network with Monte Carlo permutation testing
hyphaeon epistasis \
  -a examples/HIV1_RT.fasta \
  -t examples/HIV1_RT.nwk \
  --n-permutations 10000 \
  --max-perm-p 0.05 \
  -o examples/HIV1_RT_epistasis.json \
  -c examples/HIV1_RT_edges.csv \
  --graphml examples/HIV1_RT_coselection.graphml
```

#### Key Biological Discoveries:
1. **Unsupervised Discovery of Multi-Drug Catalytic Complexes (Q151M MDR Complex)**:
   * HyphAeon places the co-evolution of residue 116 with residue 151 at **#1 overall** across all candidate pairs:
     > **F116 ⟷ Q151** (Co-Sel = 0.8660, p<sub>hyper</sub> = 7.02 × 10⁻⁹, FDR q = 1.17 × 10⁻⁷)
2. **Autonomous Dissection of Mutually Exclusive Pathways (TAM-1 vs. TAM-2)**:
   * HyphAeon's branch co-selection metric autonomously isolates the **TAM-1 triad** (`M41L + L210W + T215Y`, q < 10⁻⁷) from the mutually antagonistic **TAM-2 cluster** (`D67N + K70R + K219Q`, q < 10⁻³).

#### Monte Carlo Permutation Testing for Epistatic Sectors:
To distinguish authentic structural/functional sectors from stochastic subsets of variable sites, HyphAeon tests the spectral coherence of candidate sectors against an empirical null distribution:
* **Vectorized Permutation Engine (`--n-permutations <int>`, default: `10000`)**: For a discovered sector S of size K, samples B random K-site subgraphs uniformly without replacement from active candidate sites. Coherence is computed across null batches via tensor contraction and Hermitian eigenvalue decomposition:
  ```text
  C(S) = λ₁(A[S, :] A[S, :]ᵀ) / Tr(A[S, :] A[S, :]ᵀ)
  ```
* **Output Metrics**: Each sector reports empirical one-sided permutation p-value:
  ```text
  p_perm = (1/B) Σ I(C(S^(b)) ≥ C(S))
  ```
  along with null mean E[C<sub>null</sub>], standard deviation, 95th percentile cutoff C<sub>95</sub>, and theoretical isotropic baseline 1/K. Set `--n-permutations 0` to disable permutation testing.
* **Empirical Filtering (`--max-perm-p <float>`, default: `None`)**: Retains only sectors whose spectral coherence satisfies `p_perm ≤ threshold` (e.g., `--max-perm-p 0.05`).

---

### Example 2: In Silico Selection Deep Mutational Scanning (Digital DMS / ESSM)

```bash
# Run digital DMS sweep on HIV-1 RT
hyphaeon dms -a examples/HIV1_RT.fasta -t examples/HIV1_RT.nwk -o examples/HIV1_RT_dms.json -c examples/HIV1_RT_dms.csv
```

---

### Example 3: Convergent Sensory Adaptation & Spectral Tuning in Rhodopsin

```bash
# Run PhyloWAS with trait sector permutation testing and gene-level phylogenetic permulations
hyphaeon phenotype \
  -a examples/RHO.fasta \
  -fg "turTru,balMus,balPhys,orcOrc,delDelp,phyCat,phoVit,halGryp,mirLeo,zalCali,odoRos" \
  --n-permutations 10000 \
  --max-perm-p 0.05 \
  --permulations 1000 \
  -o examples/RHO_marine_phenotype.json \
  -c examples/RHO_marine_sites.csv
```

#### Multi-Scale Permutation & Null Testing in PhyloWAS:
HyphAeon implements two complementary null testing layers addressing distinct evolutionary hypotheses:
1. **Macromolecular Trait Sector Permutations (`--n-permutations <int>`, default: `10000`; `--max-perm-p <float>`, default: `None`)**:
   * Following single-site phenotype association (FDR q ≤ α), HyphAeon extracts coherent epistatic sectors among trait-associated residues.
   * Tests whether trait sector coherence C(S) significantly exceeds random K-site subgraphs sampled across the alignment (p<sub>perm</sub> ≤ max_perm_p), confirming that convergent phenotype adaptation drives coordinated macromolecular re-organization rather than unlinked mutations.
2. **Gene-Level Brownian Motion Liability Permulations (`--permulations <int>`, default: `0` / parametric)**:
   * Simulates neutral continuous phenotype evolution along the phylogenetic tree using Brownian motion (Saputra et al. 2021 / RERconverge null model).
   * Computes empirical gene-level p-values (p<sub>gene</sub>) testing whether the length-normalized spectral energy (Ψ̄) or maximum site association (ρ<sub>max</sub>) exceeds neutral phylogenetic drift.

---

### Example 4: Ultra-Fast Episodic Positive Selection, Feature Attribution & Alignment Error Filtering

```bash
# Standard per-codon episodic selection inference
hyphaeon meme -a examples/Smc6.fasta -t examples/Smc6.nwk -o examples/Smc6_results.json -c examples/Smc6_results.csv

# Enable mechanistic feature attribution (identifies driving species & evolutionary timing)
hyphaeon meme -a examples/Smc6.fasta -t examples/Smc6.nwk --attribute --attribution-min-lrt 3.84 -o examples/Smc6_attributed.json

# Run inference with automated dual-stage alignment error filtering & export cleaned alignment
hyphaeon meme -a examples/Smc6.fasta -t examples/Smc6.nwk --filter --filter-out-aln examples/Smc6_cleaned.fasta -c examples/Smc6_clean.csv
```

#### 1. Mechanistic Feature Attribution (`--attribute`):
* **Single-Taxon Counterfactual Perturbation (ΔLRT)**: In silico mutates each non-consensus species back to ancestral state to rank driving taxa by marginal selection evidence explained (% Signal Explained).
* **Evolutionary Epoch Decomposition**: Classifies selection timing by weighted root patristic depth into **Recent Terminal / Tip Sweep** (≥ 0.60), **Intermediate Subclade Burst** (0.35–0.60), and **Deep Ancestral / Basal Divergence** (< 0.35), separating **Recurrent Multi-Lineage Adaptation** from single-lineage sweeps.

#### 2. Automated Alignment Error Screening (`--filter`):
* **Dual-Stage Algorithm**: Detects 1D selective clusters via exact upper-tail hypergeometric scan (p<sub>local</sub> ≤ 0.01), then evaluates the Outlier Contamination Index (OCI ≥ 0.25) to flag private frameshifts (≥ 3 contiguous radical mutations in an isolated leaf against conserved species).
* **Surgical In-Place Masking**: Automatically masks only the guilty taxon's anomalous span with `NNN` and re-evaluates the cleaned alignment in milliseconds, eliminating false positives while preserving legitimate multi-species selection.

---

### Example 5: Spectral Graph Bisection & Tree-Free Phylogenetic Splits

```bash
# Basic Tree-Free Macro-Split Discovery (Outputs Newick Tree & Clade CSV)
hyphaeon splits \
  -a examples/bat_oas1.fasta \
  --no-tree \
  -o examples/bat_oas1_spectral_tree.nwk \
  -c examples/bat_oas1_clades.csv \
  --cpu
```

#### Spectral Bisection Architecture:
* **Multi-Modal Affinity Fusion**: Combines cross-taxa attention matrices ($\bar{\mathbf{A}}$) from the axial transformer, continuous 4D metric space from Multidimensional Scaling (MDS) on pairwise distances, and sequence-level latent representations into a fused affinity matrix $\mathbf{A}_{\text{fused}} = \mathbf{S}_{\text{attn}} \odot \mathbf{K}_{\text{MDS}} \odot \mathbf{K}_{\text{emb}}$.
* **Normalized Graph Laplacian & Fiedler Vector**: Partitions taxa along the Fiedler vector $\mathbf{v}_2$ of $\mathbf{L}_{\text{sym}} = \mathbf{I} - \mathbf{D}^{-1/2} \mathbf{A}_{\text{fused}} \mathbf{D}^{-1/2}$, quantifying macro-clade split stability via the spectral eigengap $\Delta\lambda = \lambda_3 - \lambda_2$.
* **Comprehensive Benchmarks**: See [`SPECTRAL_SPLITS_BENCHMARK.md`](SPECTRAL_SPLITS_BENCHMARK.md) for full benchmarks against IQ-TREE 2, RAxML-NG, FastTree, and Neighbor-Joining across empirical datasets.

---

## Retraining & Fine-Tuning

### 1. Build per-gene training tensors

Prepare one alignment and one official HyPhy MEME JSON result per gene. Trees may be supplied as matching Newick files or embedded in the alignments:

```bash
python training/build_training_npz.py \
  --alignment_dir /path/to/training_alignments/ \
  --tree_dir /path/to/trees/ \
  --meme_dir /path/to/meme_results/ \
  --output_dir /path/to/training_npz/
```

### 2. Fine-tune the foundation model or reproduce full CORAL foundation training

To fine-tune from the released foundation checkpoint on per-gene `.npz` tensors using the 16-threshold CORAL ordinal cross-entropy objective:

```bash
python training/train.py \
  --data_dir /path/to/training_npz/ \
  --init_checkpoint weights/axomeme_v1.pt \
  --loss coral \
  --num_thresholds 16 \
  --stratified \
  --epochs 20 \
  --batch_size 128 \
  --accum_steps 16 \
  --lr 3e-4 \
  --embed_dim 384 \
  --layers 6 \
  --heads 12 \
  --bf16 \
  --output_dir /path/to/run_weights/
```

The standalone Google Colab / TPU / CUDA foundation training script that produced the released checkpoint (`weights/axomeme_v1.pt` / `model.safetensors`, Epoch 16, `loss_type='coral'`, `num_thresholds=16`, `val_spearman=0.6134`, `val_mse=3.6411`) directly from the SQLite/MSA corpus is archived at `training/train_transformer_selection.py`.

---

## Documentation

- [Architecture](ARCHITECTURE.md)
- [Migration Guide](MIGRATION_GUIDE.md)
- [Spectral Splits Benchmark](SPECTRAL_SPLITS_BENCHMARK.md)
- [Temporal Analysis Guide](TEMPORAL_ANALYSIS_GUIDE.md)
