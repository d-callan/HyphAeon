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

### Model Weights

Pretrained weights are hosted on **Hugging Face**: https://huggingface.co/datamonkey/axomeme

This is the source of truth for model weights. On first use, AxoMEME automatically
downloads the selected variant from Hugging Face and caches it locally
(`~/.cache/axomeme/`). Subsequent runs use the cached copy. Downloads are ~7 MB
and take ~1 second on a typical connection.

To list available variants:
```bash
axomeme list-models
```

To use a specific variant:
```bash
axomeme predict --alignment alignment.fa --model-variant viral
```

> [!NOTE]
> While the model repo is gated, set the `HF_TOKEN` environment variable to
> authenticate. Get a token at https://huggingface.co/settings/tokens (read
> access is sufficient). See `.env.example` for details. Once the repo is made
> public, the token will no longer be required.

### Training Data

* **Mammalian Training Database (TOGA SQLite, 37 GB)**: Available on Google Drive ([`Google Drive Link: TOGA_MEME_DB`](https://drive.google.com/drive/folders/axomeme-training-data)).
* **Pre-extracted `.npz` Alignment Tensors (18,253 Genes)**: Available on Google Drive ([`Google Drive Link: NPZ_Tensors_Archive`](https://drive.google.com/drive/folders/axomeme-tensors)).

---

## 🛠️ Retraining AxoMEME

To train the model from scratch or fine-tune on custom alignment tensors:

```bash
python train.py \
  --data_dir /path/to/extracted_npz_tensors/ \
  --epochs 30 \
  --batch_size 1 \
  --lr 3e-4 \
  --embed_dim 384 \
  --layers 6 \
  --heads 12 \
  --fp16 \
  --output_dir weights/
```

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
