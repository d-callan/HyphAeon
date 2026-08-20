# AxoMEME

**Ultra-Fast Neural Inference of Episodic Selection, Directional Phenotype-Genotype Mapping, In Silico Selection Deep Mutational Scanning (Digital DMS), and Multi-Scale Epistatic Sector Mining.**

---

## 🚀 Key Capabilities

AxoMEME integrates four complementary phylogenetic deep learning and geometric projection engines:

1. **`axomeme predict`**: Neural episodic positive selection inference ($>100\times$ faster than standard numerical MLE and codon-MCMC models like HyPhy MEME) using Tree-RoPE 4D geometric branch embeddings and axial tree attention.
2. **`axomeme phenotype` (PhyloWAS)**: Directional phenotype-genotype association mapping on the unit hypersphere $\mathbb{S}^{M-1}$. Computes spectral trait energies ($\Psi_{\text{Spectral}}$), exact sequenced-taxa null scaling $p$-values, Benjamini-Hochberg FDR $q$-values, and **Phenotype-Associated Residue Signatures (PARS)**.
3. **`axomeme dms` (Digital DMS / ESSM)**: In silico Selection Deep Mutational Scanning. Performs high-throughput sweeps of all 19 alternative amino acids across every codon position in seconds, calculating the **Epistatic Selection Sensitivity Matrix (ESSM)**, Intrinsic Mutational Plasticity ($\mathbf{E}_{i,i}$), and allosteric selection shifts ($\Delta \text{LRT}$).
4. **`axomeme epistasis` (Branch Co-Selection & Sectors)**: Multi-scale epistatic sector mining implementing phylogenetic branch attribution, exact tree hypergeometric tests, Jaccard overlap suppression, and the Two-Stage Seed-and-Extend (TSE) sector discovery algorithm.

---

## 📦 Installation

```bash
git clone https://github.com/veg/axomeme.git
cd axomeme
pip install -e .
```

---

## 🧠 Model Weights

Weights are hosted on **Hugging Face**: https://huggingface.co/datamonkey/axomeme

On first use, weights are downloaded automatically (~7 MB, ~1 second) and cached
locally. Subsequent runs use the cached copy.

### Choosing a Model Variant

```bash
# List available variants
axomeme list-models

# General model (default — trained on diverse alignments)
axomeme predict --alignment alignment.fa --tree tree.nwk

# Viral fine-tuned variant
axomeme predict --alignment alignment.fa --tree tree.nwk --model-variant viral
```

### Using Local Weights

Bypass HF download with `--weights` or `AXOMEME_WEIGHTS` (supports both `.pt`
and `.safetensors`):

```bash
axomeme predict --alignment alignment.fa --tree tree.nwk --weights /path/to/model.pt
axomeme predict --alignment alignment.fa --tree tree.nwk --weights /path/to/model.safetensors
```

### Environment Variables

| Variable | Description | Default |
|---|---|---|
| `HF_TOKEN` | Hugging Face token (required while repo is gated) | — |
| `AXOMEME_VARIANT` | Model variant to download | `general` |
| `AXOMEME_WEIGHTS` | Path to local weights file (overrides HF download) | — |
| `AXOMEME_CACHE` | Cache directory for downloaded weights | `~/.cache/axomeme` |

See [`.env.example`](.env.example) for details.

> [!NOTE]
> While the model repo is gated, set `HF_TOKEN` to authenticate. Get a token at
> https://huggingface.co/settings/tokens (read access is sufficient). Once the
> repo is made public, the token will no longer be required.

---

## 📂 Included Benchmark Datasets

All example alignments and phylogenetic trees required to reproduce these analyses are bundled directly in the `examples/` directory:

| Dataset | Alignment File | Tree File | Taxa ($N$) | Codons ($L$) | Description & Biological Domain |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **HIV-1 RT** | [`examples/HIV1_RT.fasta`](examples/HIV1_RT.fasta) | [`examples/HIV1_RT.nwk`](examples/HIV1_RT.nwk) | 476 | 335 | Retroviral Reverse Transcriptase polymerase domain (drug resistance & epistasis). |
| **Rhodopsin** | [`examples/RHO.fasta`](examples/RHO.fasta) | Embedded / Auto | 710 | 349 | Mammalian Rhodopsin visual pigments (deep-sea diving sensory adaptation). |
| **Smc6** | [`examples/Smc6.fasta`](examples/Smc6.fasta) | [`examples/Smc6.nwk`](examples/Smc6.nwk) | 20 | 1,097 | Primate Smc6 structural maintenance of chromosomes (antiviral host restriction). |
| **Bat OAS1** | [`examples/bat_oas1.fasta`](examples/bat_oas1.fasta) | [`examples/bat_oas1.nwk`](examples/bat_oas1.nwk) | 18 | 351 | Chiropteran OAS1 2'-5'-oligoadenylate synthetase (innate immunity escape). |
| **Camelid VHH** | [`examples/camelid.fasta`](examples/camelid.fasta) | [`examples/camelid.nwk`](examples/camelid.nwk) | 212 | 96 | Camelid single-domain antibody heavy-chain variable domain (antigenic diversity). |

---

## 🔬 Reproducible Benchmark Examples

### Example 1: Inter-Site Epistasis & Branch Co-Selection in HIV-1 Reverse Transcriptase

Evaluate phylogenetic branch attribution, pairwise co-selection networks, and non-redundant epistatic sectors across all codons of the HIV-1 RT polymerase domain ($N = 476$ taxa, $L = 335$ codons):

```bash
# Run branch co-selection, sector mining, and export co-selection network
axomeme epistasis -a examples/HIV1_RT.fasta -t examples/HIV1_RT.nwk -o examples/HIV1_RT_epistasis.json -c examples/HIV1_RT_edges.csv --graphml examples/HIV1_RT_coselection.graphml
```

#### Key Biological Discoveries:
1. **Unsupervised Discovery of Multi-Drug Catalytic Complexes (Q151M MDR Complex)**:
   * AxoMEME places the co-evolution of residue 116 with residue 151 at **#1 overall** across all candidate pairs:
     $$\text{F116} \longleftrightarrow \text{Q151} \quad (\text{Co-Sel} = 0.8660, \; p_{\text{hyper}} = 7.02 \times 10^{-9}, \; \text{FDR } q = 1.17 \times 10^{-7})$$
   * In clinical antiretroviral genetics, the Q151M mutation coordinates directly with F116Y in the catalytic dNTP-binding cleft to confer broad cross-resistance across the entire NRTI class (AZT, ddI, ddC, d4T, ABC).
2. **Autonomous Dissection of Mutually Exclusive Pathways (TAM-1 vs. TAM-2)**:
   * AxoMEME's branch co-selection metric autonomously isolates the **TAM-1 triad** (`M41L + L210W + T215Y`, $\text{Sim} = 0.53\text{--}0.61, q < 10^{-7}$) from the mutually antagonistic **TAM-2 cluster** (`D67N + K70R + K219Q`, $q < 10^{-3}$), mirroring clinical fitness landscapes without any pre-existing pharmacological knowledge.

---

### Example 2: In Silico Selection Deep Mutational Scanning (Digital DMS / ESSM)

Perform an exhaustive 19-amino-acid in silico mutational sweep across every site to calculate the Epistatic Selection Sensitivity Matrix (ESSM) and measure Intrinsic Mutational Plasticity ($\mathbf{E}_{i,i}$):

```bash
# Run digital DMS sweep on HIV-1 RT
axomeme dms -a examples/HIV1_RT.fasta -t examples/HIV1_RT.nwk -o examples/HIV1_RT_dms.json -c examples/HIV1_RT_dms.csv
```

#### Key Biological Discoveries:
* **Strict Catalytic Invariance**: The active triad ($\text{Asp110}, \text{Asp185}, \text{Asp186}$) exhibits low baseline selection ($\widehat{\text{LRT}} \le 1.08, p \ge 0.14$) but triggers massive selection shocks under in silico perturbation ($\Delta\text{LRT} \approx 2.65\text{--}3.13$), confirming rigid purifying constraint.
* **Permissive Drug Escape**: Known clinical resistance positions ($\text{Lys103}, \text{Tyr181}, \text{Thr215}$) exhibit low perturbation shifts ($\Delta\text{LRT} \approx 0.15\text{--}0.47$), reflecting high intrinsic mutational tolerance.

---

### Example 3: Convergent Sensory Adaptation & Spectral Tuning in Rhodopsin

PhyloWAS maps directional selection shifts across vertebrate visual pigments (RHO / RH1, $N = 710$ mammalian taxa, $L = 349$ codons) to identify convergent spectral tuning substitutions in deep-sea diving marine mammals (cetaceans, pinnipeds, sirenians):

```bash
# Run PhyloWAS for marine diving mammal visual adaptation
axomeme phenotype -a examples/RHO.fasta -fg "turTru,balMus,balPhys,orcOrc,delDelp,phyCat,phoVit,halGryp,mirLeo,zalCali,odoRos" -o examples/RHO_marine_phenotype.json -c examples/RHO_marine_sites.csv
```

#### Key Biological Discoveries:
* **Site 292 (A292S Spectral Blue-Shift)**:
  $$\text{Ala292Ser} \quad (\rho = 0.2948, \; p = 2.11 \times 10^{-5}, \; \text{FDR } q = 3.06 \times 10^{-3})$$
  * Reaches **58.3% frequency** in marine diving lineages vs. only **5.8%** across terrestrial background mammals. Site 292 is the primary molecular mechanism responsible for the $-10\text{ nm}$ blue-shift tuning required for vision in deep-sea blue-green oceanic water.
* **Site 83 (D83N Spectral Shift)**:
  $$\text{Asp83Asn} \quad (\rho = 0.2324, \; p = 3.74 \times 10^{-4}, \; \text{FDR } q = 0.019)$$
  * Reaches **75.0% frequency** in marine foreground taxa vs. 16.8% in background.
* **Site 101 (G101A)**:
  $$\text{Gly101Ala} \quad (\rho = 0.2357, \; p = 3.89 \times 10^{-4}, \; \text{FDR } q = 0.019)$$

---

### Example 4: Ultra-Fast Episodic Positive Selection (`predict`)

```bash
# Infer episodic selection on primate Smc6 antiviral restriction factor
axomeme predict -a examples/Smc6.fasta -t examples/Smc6.nwk -o examples/Smc6_results.json -c examples/Smc6_results.csv

# Infer episodic selection on bat OAS1 interferon-stimulated antiviral factor
axomeme predict -a examples/bat_oas1.fasta -t examples/bat_oas1.nwk -c examples/bat_oas1_results.csv

# Infer episodic selection on camelid antibody repertoire (auto branch-length estimation)
axomeme predict -a examples/camelid.fasta -t examples/camelid.nwk -c examples/camelid_results.csv
```

* **Throughput**: Processes $>1,000$ codon positions across hundreds of species in **$<0.15\text{ seconds}$** ($397\times$ speedup over numerical MLE).
* **Automatic Tree Inference**: Automatically estimates HKY85 maximum-likelihood branch lengths via HyPhy when omitted.
* **Haplotype Compression**: Automatically detects and prunes 100% identical sequence duplicates by default to enforce strict mathematical invariance.

---

## ⚡ CLI Reference Summary

### 1. `axomeme predict`
```bash
axomeme predict -a <alignment> [-t <tree>] [-w <weights>] [-s <max_species>] [--cpu] [-o <out.json>] [-c <out.csv>]
```

### 2. `axomeme phenotype` (PhyloWAS)
```bash
# Using curated presets (echolocation, marine, fossorial, hibernation, longevity, high_altitude, cardenolide, dim_light)
axomeme phenotype -a <alignment> -p echolocation

# Using inline regex or species lists
axomeme phenotype -a <alignment> -fg "turTru,balMus,orcOrc"

# Using quantitative continuous trait tables (e.g. body mass, longevity quotient)
axomeme phenotype -a <alignment> -pf traits.tsv --trait-col BodyMass --continuous
```

### 3. `axomeme dms` (Digital DMS / ESSM)
```bash
# Exhaustive 19-amino-acid in silico Selection DMS sweep
axomeme dms -a <alignment> -t <tree> [-o <out.json>] [-c <out.csv>]
```

### 4. `axomeme epistasis` (Branch Co-Selection & Sectors)
```bash
# Co-Selection network, Jaccard sector suppression, and GraphML export
axomeme epistasis -a <alignment> -t <tree> --min-sim 0.30 --min-shared 2 --max-overlap 0.50 --max-fdr 0.05 --graphml network.graphml

# Fast co-selection graph without 19-amino-acid DMS sweep
axomeme epistasis -a <alignment> -t <tree> --no-dms
```

---

## 📜 License

MIT License. Copyright (c) 2026 Sergei L. Kosakovsky Pond.
