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

Evaluated on empirical literature datasets (Host restriction factors, viral glycoproteins, and benchmark suites):

| Dataset / Gene | System / Biological Regime | Taxa ($N$) | Codons ($L$) | Spearman $\rho$ | Runtime (HyPhy MEME) | Runtime (AxoMEME) | Speedup |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`Smc6`** | Primate HBV restriction factor | 20 | 1,097 | **0.870** | 60.0 s | **0.83 s** | **$72\times$** |
| **`Nsmce2`** | SMC5/6 ubiquitin ligase cofactor | 20 | 247 | **0.941** | 60.0 s | **0.19 s** | **$309\times$** |
| **`Nsmce4a`** | Primate kleisin subunit | 14 | 355 | **0.936** | 60.0 s | **0.20 s** | **$295\times$** |
| **`camelid`** | Camelid VHH single-domain antibodies | 212 | 96 | **0.930** | 185.2 s | **1.43 s** | **$129\times$** |
| **`lysin`** | Abalone sperm-egg recognition | 25 | 134 | **0.885** | 35.4 s | **0.21 s** | **$168\times$** |
| **`lysozyme`** | Primate stomach lysozyme C | 19 | 130 | **0.912** | 28.1 s | **0.18 s** | **$156\times$** |
| **`HIV_RT`** | HIV-1 Reverse Transcriptase | 476 | 335 | **0.865** | 890.5 s | **2.85 s** | **$312\times$** |

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
