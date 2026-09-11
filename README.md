<div align="center">

<img src="assets/hyphaeon_logo.png" alt="HyphAeon Logo" width="280"/>

# HyphAeon
### Attention on Evolution Across Deep Time Transforms Comparative Genomics

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![bioRxiv](https://img.shields.io/badge/bioRxiv-2026.09.06.749597-b31b1b.svg)](https://www.biorxiv.org/content/10.64898/2026.09.06.749597v1)
[![GitHub Pages](https://img.shields.io/badge/GitHub%20Pages-Interactive%20Presentation-00e5ff.svg)](https://veg.github.io/HyphAeon/)

</div>

---

**HyphAeon** is a deep-time phylogenetic foundation model designed to bridge computational phylogenetics, structural biology, and foundation AI. Built upon a 2D axial transformer backbone (**`PhyloAxialTransformer`**) with patristic distance-decay attention and classical multidimensional scaling (MDS) tree embeddings, HyphAeon ingests multi-species codon alignments and explicit evolutionary trees spanning 200 million years of deep time.

---

> [!TIP]
> **Migrating from HyPhy?** See our comprehensive [**HyPhy to HyphAeon Migration Guide**](MIGRATION_GUIDE.md) for direct method-by-method translations (`hyphy meme` → `hyphaeon meme`, `contrast-fel` → `hyphaeon phenotype`, `prime` → `hyphaeon dms`) and biological recipes categorized by empirical data regime.

## 🚀 Key Capabilities & Unified Commands

HyphAeon integrates six complementary phylogenetic deep learning and geometric
projection engines:

1. **`hyphaeon meme` (Site-Level Diversifying Selection)**:
   Neural episodic positive selection inference (100×–1,100× faster than standard numerical MLE and codon-MCMC models like HyPhy MEME/FEL; see ARCHITECTURE.md for detailed benchmarks) using Tree-RoPE 4D geometric branch embeddings and axial tree attention.
2. **`hyphaeon epistasis` (3D Co-Evolution & Epistatic Sectors)**:
   Multi-scale epistatic sector mining implementing phylogenetic branch attribution, exact tree hypergeometric tests, Jaccard overlap suppression, contact map recovery (C<sub>β</sub>–C<sub>β</sub> < 8 Å), and vectorized Monte Carlo permutation significance testing (`--n-permutations`, `--max-perm-p`).
3. **`hyphaeon dms` (Digital Deep Mutational Scanning & CPDs)**:
   In silico Selection Deep Mutational Scanning. Performs high-throughput sweeps of all 19 alternative amino acids across every codon position in seconds, calculating the **Epistatic Selection Sensitivity Matrix (ESSM)**, Intrinsic Mutational Plasticity (E<sub>i,i</sub>), and de novo predicting compensatory partners (s<sub>comp</sub>) that rescue human disease mutations (Compensated Pathogenic Deviations).
4. **`hyphaeon phenotype` (PhyloWAS)**:
   Directional phenotype-genotype association mapping on the unit hypersphere S<sup>M-1</sup>. Computes spectral trait energies (Ψ<sub>Spectral</sub>), exact sequenced-taxa null scaling p-values, Benjamini-Hochberg FDR q-values, **Phenotype-Associated Residue Signatures (PARS)**, macromolecular trait sector permutation testing (`--n-permutations`, `--max-perm-p`), and gene-level Brownian motion liability permulations (`--permulations`).
5. **`hyphaeon temporal` (Continuous Surveillance Dynamics & Sweep Velocity)**:
   Time-resolved episodic selection tracking using continuous logistic trajectory regression, positive sweep velocity v<sub>s</sub>(t) = max(0, d/dt â<sub>s</sub>(t)), Dynamic Time Warping (DTW) wave decomposition, and temporal SVD factor loadings. See the [**Temporal Analysis Operational Guide**](TEMPORAL_ANALYSIS_GUIDE.md).
6. **`hyphaeon splits` (Spectral Graph Bisection & Tree-Free Clade Discovery)**:
   Recovers well-supported phylogenetic macro-clades and deep hierarchical bipartitions by fusing pairwise continuous 4D MDS geometry with discrete cross-taxa attention maps. Delivers up to 28× speedups over traditional ML tree search without requiring pre-computed phylogenies. See the [**Spectral Splits & Benchmarking Report**](SPECTRAL_SPLITS_BENCHMARK.md).

---

## 📦 Installation

HyphAeon requires Python ≥ 3.8 and PyTorch ≥ 2.0. At runtime it auto-selects
the best available device (CUDA → Apple MPS → CPU), so no manual configuration
is needed regardless of which install path you choose.

| Method | Command | Torch | GPU? |
| :--- | :--- | :--- | :--- |
| **pip** (default) | `pip install hyphaeon` | CUDA-bundled wheel (~550 MB) | NVIDIA GPU if driver matches; else CPU |
| **pip** (CPU-only) | `pip install torch --index-url https://download.pytorch.org/whl/cpu` then `pip install hyphaeon` | CPU-only wheel (~200 MB) | CPU |
| **Bioconda** | `conda install -c bioconda hyphaeon` | CPU-only `pytorch` from conda-forge | CPU by default; swap in `pytorch-gpu` for GPU |
| **NVIDIA Jetson** | See [issue #31](https://github.com/veg/HyphAeon/issues/31) | JetPack-native wheel (cp38 only) | Jetson GPU |

To use a GPU with Bioconda, install conda-forge's GPU PyTorch variant first:

```bash
conda create -n hyphaeon-gpu -c conda-forge pytorch-gpu
conda activate hyphaeon-gpu
conda install -c bioconda hyphaeon
```

You can always install a specific PyTorch build before installing HyphAeon if
none of the above defaults suit your system (e.g. a particular CUDA version,
a custom wheel, or a CPU-only build on a server without GPU).

> [!NOTE]
> **Model weights** are downloaded automatically from [Hugging Face](https://huggingface.co/datamonkey/hyphaeon)
> on first use (cached in `~/.cache/hyphaeon/`). No authentication or token is
> required. Use `--model-variant viral` to select the viral-tuned variant, or
> `--weights /path/to/checkpoint` to use a local file.

---

## 📂 Included Benchmark Datasets

All example alignments and phylogenetic trees required to reproduce these analyses are bundled directly in the `examples/` directory:

| Dataset | Alignment File | Tree File | Taxa (N) | Codons (L) | Description & Biological Domain |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **HIV-1 RT** | [`examples/HIV1_RT.fasta`](examples/HIV1_RT.fasta) | [`examples/HIV1_RT.nwk`](examples/HIV1_RT.nwk) | 476 | 335 | Retroviral Reverse Transcriptase polymerase domain (drug resistance & epistasis). |
| **Rhodopsin** | [`examples/RHO.fasta`](examples/RHO.fasta) | Auto (TN93) | 710 | 349 | Mammalian Rhodopsin visual pigments (deep-sea diving sensory adaptation). No tree file provided; uses TN93 distance estimation. |
| **Smc6** | [`examples/Smc6.fasta`](examples/Smc6.fasta) | [`examples/Smc6.nwk`](examples/Smc6.nwk) | 20 | 1,097 | Primate Smc6 structural maintenance of chromosomes (antiviral host restriction). |
| **Bat OAS1** | [`examples/bat_oas1.fasta`](examples/bat_oas1.fasta) | [`examples/bat_oas1.nwk`](examples/bat_oas1.nwk) | 18 | 351 | Chiropteran OAS1 2'-5'-oligoadenylate synthetase (innate immunity escape). |
| **Camelid VHH** | [`examples/camelid.fasta`](examples/camelid.fasta) | [`examples/camelid.nwk`](examples/camelid.nwk) | 212 | 96 | Camelid single-domain antibody heavy-chain variable domain (antigenic diversity). Used for integration testing; no dedicated example section. |
| **Randall FP** | [`examples/randall_asr/fp_leaves_codon.fasta`](examples/randall_asr/fp_leaves_codon.fasta) | [`examples/randall_asr/RandallBenchmarkTree.newick`](examples/randall_asr/RandallBenchmarkTree.newick) | 19 | 226 | Experimental phylogeny with known true ancestors (Randall et al. 2016). ASR benchmark; see [validation notes](examples/randall_asr/VALIDATION_NOTES.md). |
| **AMA1** | [`examples/ama1/ama1_codon_aln.fasta`](examples/ama1/ama1_codon_aln.fasta) | — | 7 | 470 | Apicomplexan AMA1 adhesin domain (malarial invasion). ASR example dataset. |

---

## 🔬 Reproducible Benchmark Examples

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

### Example 4: Ultra-Fast Episodic Positive Selection (`predict`), Feature Attribution (`--attribute`), & Alignment Error Filtering (`--filter`)

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

### Example 5: Ancestral Sequence Candidate Scoring (`hyphaeon ancestral`)

```bash
# Centroid mode: single ancestor, no tree (TN93 distances), outgroup-rooted
hyphaeon ancestral -a examples/randall_asr/fp_leaves_codon.fasta --no-tree --outgroup 01 --lambda-dist 1.0 -k 5 --fasta examples/randall_asr/ancestors.fasta -o examples/randall_asr/results.json

# Per-node mode: one ancestor per internal tree node (requires tree)
hyphaeon ancestral -a examples/randall_asr/fp_leaves_codon.fasta -t examples/randall_asr/RandallBenchmarkTree.newick --per-node --fasta examples/randall_asr/per_node.fasta -o examples/randall_asr/per_node.json
```

**EXPERIMENTAL / PROTOTYPE** — HyphAeon-informed ancestral sequence candidate scoring. This is *not* classical tree-based ASR (PAML, FastML, IQ-TREE). HyphAeon is topology-blind (star-tree invariant per `model_eval/invariance/`) and exposes no character-state decoder head, so it cannot emit per-internal-node states the way classical marginal reconstruction can.

Instead, this reframes the problem to match what the model actually is:

- The natural object is a **single "centroid ancestor"** sitting at the origin of the model's 4D MDS distance embedding — exactly where the model places its internal `[ROOT]` token.
- The base reconstruction signal comes from observed tips, weighted by patristic proximity to the centroid. This needs no model at all.
- HyphAeon enters **only as re-scorers / constraints** on top of that base:
  - **Selection LRT** → per-site reconstruction confidence
  - **Digital DMS** → per-state selection-signal sensitivity (a regulariser)
  - **Co-selection** → joint / epistatic compatibility across site pairs

The key contribution over standard ASR is the **epistatic term**: classical marginal ASR reconstructs each site independently, whereas the co-selection network penalises ancestral candidates that combine individually-plausible states that are jointly never observed.

**Key parameters:**
- `--outgroup`: Root the ancestor at an outgroup taxon's MDS position (fuzzy match).
- `--lambda-dist`: Steepness of exponential distance weighting (lower = flatter, outgroup contributes more).
- `--lambda-plast`: Weight of DMS selection-sensitivity regulariser.
- `--lambda-epi`: Weight of epistatic co-occurrence term.
- `--per-node`: Reconstruct one ancestor per internal tree node (requires `-t/--tree`).
- `--node-tip-context-weight`: Soft descendant mask (0.0 = hard mask, 1.0 = no mask).
- `--joint-pass`: Joint consistency pass strength (mixes child logprobs with parent's reconstructed sequence).
- `--weak-node-boost`: Scale up DMS/epistatic lambdas for nodes with few descendants.

**Benchmark:** Validated against the Randall et al. (2016) experimental FP phylogeny with 18 known true ancestors. See [`examples/randall_asr/VALIDATION_NOTES.md`](examples/randall_asr/VALIDATION_NOTES.md) for full results, and [`model_eval/concordance/test_ancestral_concordance.py`](model_eval/concordance/test_ancestral_concordance.py) for automated concordance tests. Parameter sweeps are available via `python scripts/sweep_ancestral_params.py`.

---

### Example 6: Spectral Graph Bisection & Tree-Free Phylogenetic Splits (`hyphaeon splits`)

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

## 🛠️ Retraining & Fine-Tuning HyphAeon

### 1. Build per-gene training tensors

Prepare one alignment and one official HyPhy MEME JSON result per gene. Trees may be supplied as matching Newick files or embedded in the alignments:

```bash
python training/build_training_npz.py \
  --alignment_dir /path/to/training_alignments/ \
  --tree_dir /path/to/trees/ \
  --meme_dir /path/to/meme_results/ \
  --output_dir /path/to/training_npz/
```

### 2. Fine-tune the foundation model

```bash
python training/train.py \
  --data_dir /path/to/training_npz/ \
  --epochs 30 \
  --batch_size 1 \
  --lr 3e-4 \
  --embed_dim 384 \
  --layers 6 \
  --heads 12 \
  --fp16 \
  --output_dir /path/to/run_weights/
```

---

## ⚡ CLI Reference Summary

| Command | Action | Description |
| :--- | :--- | :--- |
| `hyphaeon meme` | Site-Level Selection | Fast per-codon LRT & selection rate prediction (100×–1,100× faster than MLE). |
| `hyphaeon epistasis` | 3D Epistatic Sectors | Co-selection networks, hypergeometric tree overlaps, and Monte Carlo sector permutations. |
| `hyphaeon dms` | Digital DMS | 19-AA in silico perturbation sweeps and Compensated Pathogenic Deviation mapping. |
| `hyphaeon phenotype`| Directional PhyloWAS | Directional trait mapping on the unit hypersphere, trait sector permutations, and liability permulations. |
| `hyphaeon temporal` | Dynamic Surveillance | Continuous logistic trajectory regression, sweep velocity, DTW waves, and temporal SVD. |
| `hyphaeon splits` | Spectral Bisection | Tree-free phylogenetic macro-splits via cross-taxa attention and MDS graph Laplacian. |
| `hyphaeon disease` | Pathogenicity Prediction | Predict disease variant effects and pathogenicity using HyphAeon attention attributions. |
| `hyphaeon filter` | Alignment QC | Automated alignment error detection and surgical masking of anomalous regions. |
| `hyphaeon ancestral` | Ancestral Scoring | **(EXPERIMENTAL)** HyphAeon-informed ancestral sequence candidate scoring with epistatic constraints. |

### Key Permutation Testing Arguments:

#### `hyphaeon epistasis`
| Flag | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `--n-permutations` | `int` | `10000` | Number of random K-site subset Monte Carlo permutations for sector significance testing (set `0` to disable). |
| `--max-perm-p` | `float` | `None` | Maximum empirical permutation p-value threshold to retain sectors (default retains all C(S) ≥ min_coherence). |
| `--min-coherence` | `float` | `0.50` | Minimum spectral coherence ratio C(S) = λ₁ / Tr for candidate sectors. |
| `--min-clique-size` | `int` | `3` | Minimum clique seed size for epistatic sectors. |
| `--max-overlap` | `float` | `0.50` | Maximum Jaccard overlap allowed between discovered sectors. |
| `--no-tree` / `--use-tn93` | `flag` | `False` | Estimate pairwise evolutionary distances directly from alignment via TN93 (skips tree). Requires the optional `tn93` package (`pip install hyphaeon[tn93]`) or the `tn93` binary on PATH. |

#### `hyphaeon phenotype`
| Flag | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `--n-permutations` | `int` | `10000` | Number of random K-site subset Monte Carlo permutations for trait sector significance testing (set `0` to disable). |
| `--max-perm-p` | `float` | `None` | Maximum permutation p-value threshold to retain trait sectors (default retains all C(S) ≥ 0.45). |
| `--permulations` | `int` | `0` | Number of Brownian motion phylogenetic permulations for gene-level empirical p-values (RERconverge null model; default `0` / parametric). |
| `--alpha` | `float` | `0.05` | Benjamini-Hochberg FDR significance threshold for trait-associated sites. |
| `--continuous` | `flag` | `False` | Treat trait values as continuous phylogenetic contrasts rather than discrete foreground/background. |
| `--min-taxa` | `int` | `4` | Minimum sequenced taxa required per site. |

#### `hyphaeon splits`
| Flag | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `-a` / `--alignment` | `path` | Required | Path to in-frame codon FASTA or NEXUS alignment. |
| `-t` / `--tree` | `path` | `None` | Optional Newick/NEXUS phylogenetic tree (optional if embedded, or if `--no-tree`/`--use-tn93` is set). |
| `--no-tree` / `--use-tn93` | `flag` | `False` | Skip phylogenetic tree and estimate pairwise evolutionary distances directly from alignment via TN93. Requires `tn93` (`pip install hyphaeon[tn93]`) or the `tn93` binary on PATH. |
| `--min-clade-size` | `int` | `2` | Minimum clade size floor to terminate recursive bisection. |
| `--max-depth` | `int` | `10` | Maximum tree hierarchy recursion depth. |
| `-o` / `--output` | `path` | `None` | Optional path to export derived hierarchical Newick tree (`.nwk`). |
| `-c` / `--csv` | `path` | `None` | Optional path to export split clade membership assignments (`.csv`). |
| `-w` / `--weights` | `path` | `None` | Path to local model weights file (overrides HF download). |
| `--cpu` | `flag` | `False` | Force CPU execution. |

---

## 📜 Citation

If you use **HyphAeon** in your research, please cite:

> Sergei L. Kosakovsky Pond, Steven Weaver, Danielle Callan, Jordan D. Zehr, Alexander G. Lucaci, Hannah Verdonk, Avery Selberg, Gallean Brown, Maria Chikina, Nathan L. Clark, Kateryna D. Makova, Darren P. Martin, and Anton Nekrutenko.  
> **HyphAeon: Attention on Evolution Across Deep Time Transforms Comparative Genomics**.  
> *bioRxiv* 2026.09.06.749597; doi: [https://doi.org/10.64898/2026.09.06.749597](https://www.biorxiv.org/content/10.64898/2026.09.06.749597v1)

```bibtex
@article{kosakovskypond2026hyphaeon,
  title={HyphAeon: Attention on Evolution Across Deep Time Transforms Comparative Genomics},
  author={Kosakovsky Pond, Sergei L. and Weaver, Steven and Callan, Danielle and Zehr, Jordan D. and Lucaci, Alexander G. and Verdonk, Hannah and Selberg, Avery and Brown, Gallean and Chikina, Maria and Clark, Nathan L. and Makova, Kateryna D. and Martin, Darren P. and Nekrutenko, Anton},
  journal={bioRxiv},
  pages={2026.09.06.749597},
  year={2026},
  doi={10.64898/2026.09.06.749597},
  url={https://www.biorxiv.org/content/10.64898/2026.09.06.749597v1}
}
```
