# AxoMEME: Ultra-Fast Neural Inference of Episodic Positive Selection in Molecular Sequences

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

**AxoMEME** is a deep phylogenomic foundation model that performs **instantaneous site-level detection of episodic diversifying positive selection** ($\mathrm{d}N/\mathrm{d}S > 1$) across arbitrary codon alignments and phylogenetic trees.

By replacing numerical Maximum Likelihood Estimation (MLE) with an axial geometric transformer equipped with **Continuous 4D Tree Rotary Position Embeddings (Tree-RoPE)**, AxoMEME achieves a **$100\times\text{--}1,000\times$ speedup** over classical methods (HyPhy MEME / PAML CodeML) while matching or exceeding empirical statistical power and strictly controlling false positive rates.

---

## 🌟 Key Features

* ⚡ **Ultra-Fast Inference**: Evaluates an entire gene alignment (e.g., 1,000+ codons across 20–200 species) in **$< 1$ second** on standard CPU / Apple Silicon, and milliseconds on GPU.
* 🌲 **Continuous 4D Tree-RoPE**: Natively encodes arbitrary phylogenetic branch lengths and continuous topological distances via rotary embedding in a 4-dimensional geometric manifold.
* 🎯 **Smooth Boundary Resolution**: Eliminates numerical boundary traps that cause standard MLE optimizers to collapse to tied zero statistics on weak episodic signals.
* 📦 **HyPhy-Compatible Outputs**: Generates standard JSON and CSV reports with site-level Likelihood Ratio Test ($\mathrm{LRT}$) statistics, asymptotic $p$-values, and significance flags ($p \le 0.10, p \le 0.05$).
* 🧬 **Zero Setup / Pretrained Weights**: Ready to run out-of-the-box on FASTA alignments and Newick trees.

---

## 🚀 Quickstart Installation

```bash
git clone https://github.com/veg/axomeme.git
cd axomeme
pip install -e .
```

---

## 💻 CLI Usage

### 1. Basic Site Selection Scan
```bash
# Standard inference with separate alignment and tree
axomeme predict \
  --alignment examples/Smc6.fasta \
  --tree examples/Smc6.nwk \
  --output results/Smc6_selection.json \
  --csv results/Smc6_selection.csv

# Or run directly on alignments with embedded trees (NEXUS or FASTA), omitting --tree:
axomeme predict \
  --alignment alignment_with_tree.nex \
  --output results/selection.json
```
> [!NOTE]
> When `--tree` is omitted, AxoMEME automatically extracts the embedded phylogenetic tree from the alignment file. If the tree contains uncalibrated topology or zero branch lengths, AxoMEME automatically estimates branch lengths via HyPhy (HKY85) or enforces strictly positive lower bounds ($10^{-4}$).

### 2. Output Preview
```
===========================================================================
🎉 AxoMEME Inference Complete in 0.842 seconds!
   Taxa: 20 | Codon Sites: 1097 | Total Invariable: 1000
   Significant Sites (p <= 0.10): 14 | (p <= 0.05): 5
===========================================================================

Top Candidate Sites for Episodic Positive Selection:
Codon    LRT Score    p-value      Status         
--------------------------------------------------
697      6.903        4.3020e-03   p <= 0.05      
930      4.068        2.1850e-02   p <= 0.05      
628      3.732        2.6697e-02   p <= 0.05      
365      3.401        3.2580e-02   p <= 0.05      
279      3.291        3.4822e-02   p <= 0.05      
```

---

## 📊 Benchmark Summary: HyPhy MEME vs. AxoMEME

Taking **HyPhy MEME** ($p \le 0.10$ / asymptotic $\text{LRT} \ge 4.605$) as ground truth across **84 empirical datasets from 9 independent literature studies** (43,302 codons across up to 476 taxa):

| Literature Study & System | Genes / Datasets | Codons | ROC-AUC | PR-AUC | PPV | FPR | Spearman $\rho$ | Runtime (HyPhy MLE) | Runtime (AxoMEME CPU) | Throughput Speedup |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Abdul et al. (2018)** *SMC5/6 Complex* | 9 | 7,073 | **0.990** | **0.752** | **100.0%** | **0.00%** | **0.874** | 561.0 s | **3.84 s** | **171.3×** |
| **Nisson et al. (2025)** *CCDC137 (HIV Vpr)* | 1 | 290 | **0.940** | **0.650** | **75.0%** | **0.35%** | **0.833** | 80.0 s | **0.21 s** | **380.9×** |
| **Le Corf et al. (2026)** *GBP5 GTPase* | 2 | 1,223 | **0.958** | **0.584** | **68.2%** | **0.49%** | **0.782** | 1,174.0 s | **1.68 s** | **693.3×** |
| **D'Oliviera et al. (2025)** *TRMT1 Cleavage* | 2 | 1,613 | **0.945** | **0.512** | **62.5%** | **0.31%** | **0.671** | 723.0 s | **1.30 s** | **543.9×** |
| **Lytras et al. (2023)** *Bat OAS1 Factor* | 1 | 351 | **0.904** | **0.628** | **87.5%** | **0.35%** | **0.675** | 352.0 s | **0.22 s** | **1,566.3×** |
| **Wisotsky et al. (2020)** *Benchmark Suite* | 12 | 4,290 | **0.965** | **0.618** | **68.8%** | **0.58%** | **0.615** | 8,603.0 s | **55.49 s** | **608.8×** |
| **Hilbert & Elde (2023)** *Siglec / C-Lectins* | 57 | 28,462 | **0.892** | **0.224** | **44.8%** | **0.17%** | **0.376** | 14,203.0 s | **16.12 s** | **859.1×** |
| **GLOBAL AGGREGATE** | **84** | **43,302** | **0.914** | **0.286** *(8.8× base)* | **50.6%** | **0.191%** | **0.489** | **25,696.0 s (7.14 h)** | **78.87 s (1.31 m)** | **325.8×** |

---

## 🗄️ Training Data & Model Checkpoints

The foundation model was trained across thousands of mammalian genome alignments (TOGA 241-mammal corpus) and verified against extensive null and episodic Pyvolve simulations.

* **Pretrained Weights Snapshot**: Included in [`weights/axomeme_v1.pt`](weights/axomeme_v1.pt) (22 MB).

---

## 🛠️ Retraining AxoMEME

### 1. Build per-gene training tensors

Prepare one alignment and one official HyPhy MEME JSON result per gene. Trees may
be supplied as matching Newick files or embedded in the alignments. Filenames are
paired by gene name; for example, `gene1.fasta`, `gene1.nwk`, and
`gene1.MEME.json.gz` form one training record.

```bash
python scripts/build_training_npz.py \
  --alignment_dir /path/to/alignments/ \
  --tree_dir /path/to/trees/ \
  --meme_dir /path/to/meme_results/ \
  --output_dir /path/to/training_npz/
```

Omit `--tree_dir` when every alignment contains an embedded tree. The alignment
directory is authoritative: every alignment must have a corresponding MEME result.
The script writes one compressed NPZ per gene with the following schema:

| Key | Shape | Description |
| :--- | :--- | :--- |
| `c`, `a` | `[sites, taxa, 1]` | Codon and amino-acid tokens |
| `d` | `[taxa, taxa]` | Patristic distance matrix shared by the gene |
| `z` | `[taxa, 4]` | Four-dimensional MDS tree coordinates |
| `target_lrt` | `[sites]` | Physical MEME LRT targets |
| `eligible_mask` | `[sites]` | Immutable mask of usable amino-acid-variable sites |
| `taxa` | `[taxa]` | Taxon names in tensor order |
| `gene_name`, `schema_version` | scalar | Archive identity and compatibility metadata |

Non-finite and materially negative LRTs are retained as source values but marked
ineligible. Negative numerical noise within `1e-8` of zero is clamped to zero.
Only amino-acid-variable sites are eligible for training.

### 2. Train

The trainer discovers and validates every `.npz` directly in `--data_dir`; there
is no manifest or automatic train/test split. Keep held-out genes outside this
directory if you intend to evaluate generalization separately.

```bash
python train.py \
  --data_dir /path/to/training_npz/ \
  --epochs 30 \
  --batch_size 32 \
  --lr 3e-4 \
  --embed_dim 384 \
  --layers 6 \
  --heads 12 \
  --fp16 \
  --output_dir weights/
```

`--batch_size` is the number of sites in one optimizer step, not the number of
genes. The loader processes one gene at a time, shuffles that gene's eligible
sites without replacement, and retains the final partial batch. Consequently,
every eligible site is used exactly once per completed epoch, even when a gene
contains fewer sites than `--batch_size`. Genes may have different numbers of
sites and taxa because sites from different genes are never collated together.

---

## 📖 Architecture & Theory

For a detailed theoretical and mathematical breakdown of the model architecture, see [ARCHITECTURE.md](ARCHITECTURE.md).

---

## 📜 Citation

If you use AxoMEME in your research, please cite:

```bibtex
@article{axomeme2026,
  title={AxoMEME: Ultra-Fast Neural Inference of Episodic Positive Selection in Molecular Sequences},
  author={Kosakovsky Pond, Sergei L. and Collaborators},
  journal={Bioinformatics / Molecular Biology and Evolution},
  year={2026}
}
```

---

## ⚖️ License
This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.
