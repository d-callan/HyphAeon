# AxoMEME

**Ultra-Fast Neural Inference of Episodic Selection, Directional Phenotype-Genotype Mapping, and Multi-Scale Epistatic Sector Mining.**

---

## 🚀 Key Capabilities

AxoMEME integrates three complementary phylogenetic deep learning and geometric projection engines:

1. **`axomeme predict`**: Neural episodic positive selection inference ($>100\times$ faster than standard codon-MCMC and likelihood models like MEME / HyPhy) using Tree-RoPE 4D geometric branch embeddings and axial tree attention.
2. **`axomeme phenotype` (PhyloWAS)**: Directional phenotype-genotype association mapping on the unit hypersphere $\mathbb{S}^{M-1}$. Computes spectral trait energies ($\Psi_{\text{Spectral}}$), exact sequenced-taxa null scaling $p$-values, Benjamini-Hochberg FDR $q$-values, and **Phenotype-Associated Residue Signatures (PARS)**.
3. **`axomeme epistasis` (ESSM / TSE)**: Multi-scale epistatic sector mining implementing the Two-Stage Seed-and-Extend (TSE) algorithm to discover cooperative allosteric sectors, spectral coherence $C(\mathcal{S})$, and co-selection network topologies.

---

## 📦 Installation

```bash
git clone https://github.com/veg/axomeme.git
cd axomeme
pip install -e .
```

---

## ⚡ CLI Usage

### 1. Episodic Positive Selection Inference (`predict`)
```bash
# Infer episodic selection from an in-frame codon alignment and tree
axomeme predict -a data/alignment.fasta -t data/tree.nwk -o results.json -c results.csv
```

### 2. Directional Phenotype-Genotype Association (`phenotype` / `phylowas`)
```bash
# Using a built-in curated macroevolutionary preset
axomeme phenotype -a msa/SLC26A5.gz -p echolocation -o echoloc_results.json

# Using inline comma-separated or glob pattern of foreground species
axomeme phenotype -a msa/TMC1.gz -fg "rhi*,myo*,turTru,balMys" -c tmc1_assoc.csv

# Using a phenotypic metadata table (TSV / CSV)
axomeme phenotype -a msa/GHR.gz -pf metadata/mammalian_traits.tsv --trait-col AdultBodyMass_g --continuous

# Other available presets:
# - echolocation, marine, fossorial, hibernation, longevity, high_altitude, cardenolide, dim_light
```

### 3. Inter-Site Epistasis, Branch Co-Selection & Selection DMS (`epistasis` / `essm`)
```bash
# Mine phylogenetic branch co-selection networks, Selection DMS (ESSM), and epistatic sectors
axomeme epistasis -a examples/bat_oas1.fasta -t examples/bat_oas1.nwk -o epistasis.json -c epistasis.csv --graphml coselection.graphml

# High-depth filtering with custom FDR cutoff
axomeme epistasis -a examples/Smc6.fasta -t examples/Smc6.nwk --min-shared 3 --min-sim 0.40 --max-fdr 0.01

# Fast co-selection graph extraction without full DMS sweep
axomeme epistasis -a examples/Smc6.fasta -t examples/Smc6.nwk --no-dms
```

---

## 📜 Citation

If you use AxoMEME, PhyloWAS, or ESSM in your research, please cite:

```bibtex
@article{axomeme2026,
  title={AxoMEME: Geometric Phenotype Projection, Neural Episodic Selection, and Epistatic Sector Mining in Molecular Evolution},
  author={Pond, Sergei L. Kosakovsky and DeepMind Antigravity Team},
  journal={bioRxiv},
  year={2026}
}
```
