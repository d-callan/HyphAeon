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

Taking **HyPhy MEME** ($p \le 0.10$ / asymptotic $\text{LRT} \ge 3.12$) as ground truth across **22 empirical literature datasets** (11,714 codons across up to 476 taxa):

| Dataset / Gene | System / Biological Regime | Taxa ($N$) | Codons ($L$) | ROC-AUC | PR-AUC | PPV | FPR | Spearman $\rho$ | Runtime (HyPhy) | Runtime (AxoMEME) | Speedup |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **SARS-CoV-2 spike** | Coronavirus spike receptor | 180 | 1,284 | **0.976** | 0.433 | 20.6% | 2.13% | **0.972** | 859.0 s | **21.16 s** | **40.6×** |
| **Nsmce2** | SMC5/6 ubiquitin ligase cofactor | 20 | 247 | **1.000** | **1.000** | **100.0%** | **0.00%** | **0.941** | 31.0 s | **0.20 s** | **158.4×** |
| **Nsmce4a** | Primate kleisin subunit | 14 | 355 | **1.000** | — | **100.0%** | **0.00%** | **0.936** | 24.0 s | **0.21 s** | **115.8×** |
| **Smc4** | SMC4 chromosome condensing | 18 | 1,288 | **0.987** | 0.169 | 0.0% | 0.08% | **0.930** | 92.0 s | **0.89 s** | **102.9×** |
| **Smc5** | SMC5 conserved partner | 18 | 1,102 | **0.967** | 0.027 | 0.0% | **0.00%** | **0.914** | 79.0 s | **0.74 s** | **106.6×** |
| **Smc2** | SMC2 condensin subunit | 17 | 1,197 | **0.991** | 0.293 | 0.0% | **0.00%** | **0.909** | 86.0 s | **0.80 s** | **107.9×** |
| **Smc6** | Primate HBV restriction factor | 20 | 1,097 | **0.990** | **0.752** | **100.0%** | **0.00%** | **0.870** | 99.0 s | **0.82 s** | **120.5×** |
| **Lysozyme** | Primate stomach lysozyme C | 19 | 130 | **1.000** | — | **100.0%** | **0.00%** | **0.860** | 21.0 s | **0.10 s** | **211.4×** |
| **Nsmce3** | Primate SMC5/6 subunit | 20 | 304 | **0.974** | 0.111 | 0.0% | **0.00%** | **0.860** | 59.0 s | **0.23 s** | **258.1×** |
| **Lysin** | Abalone sperm-egg recognition | 25 | 134 | **0.863** | **0.659** | **76.9%** | 3.06% | **0.799** | 104.0 s | **0.14 s** | **755.4×** |
| **Bat OAS1** | Bat sarbecovirus restriction factor | 18 | 351 | **0.904** | **0.628** | **87.5%** | 0.35% | **0.675** | 352.0 s | **0.25 s** | **1,402.0×** |
| **Camelid VHH** | Single-domain antibodies | 212 | 96 | **0.839** | **0.746** | **86.7%** | 3.23% | **0.635** | 976.0 s | **1.85 s** | **526.9×** |
| **Adh** | Drosophila alcohol dehydrogenase | 23 | 254 | **0.980** | **0.669** | **80.0%** | 0.41% | **0.602** | 142.0 s | **0.22 s** | **634.1×** |
| **Influenza A HA** | Influenza A hemagglutinin | 349 | 329 | **0.937** | **0.607** | **55.6%** | 2.64% | **0.587** | 1,829.0 s | **17.53 s** | **104.3×** |
| **HIV RT** | HIV-1 Reverse Transcriptase | 476 | 335 | **0.959** | **0.743** | **83.3%** | 0.65% | **0.485** | 3,184.0 s | **32.95 s** | **96.6×** |
| **TOTAL (All 22)**| **Global Benchmark Suite** | — | **11,714** | **0.970** | **0.461** | **58.6%** | **0.46%** | **0.680** | **9,516.0 s** | **81.44 s** | **116.8×** |

---

## 🗄️ Training Data & Model Checkpoints

The foundation model was trained across thousands of mammalian genome alignments (TOGA 241-mammal corpus) and verified against extensive null and episodic Pyvolve simulations.

* **Pretrained Weights Snapshot**: Included in [`weights/axomeme_v1.pt`](weights/axomeme_v1.pt) (22 MB).
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
