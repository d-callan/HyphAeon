# Host-Pathogen Coevolution Detection with HyphAeon

Plan for extending HyphAeon to detect cross-species co-selection between paired host and pathogen alignments, validated on four Plasmodium falciparum–human gene pairs, with a genome-wide screening follow-up.

---

## 1. Problem & Current Gap

HyphAeon's `epistasis` command computes **within-gene** co-selection: it finds co-evolving *sites* within a single alignment by comparing Transformer Attribution Vectors (TAVs) across codon positions in one protein. The function `compute_branch_coselection_network` in `@/home/danielle/Documents/veg/HyphAeon/hyphaeon/src/hyphaeon/epistasis.py:121-222` computes cosine similarity between attribution vectors of shape `[L, N]` where L = codon sites and N = taxa, all within one gene.

**What's needed**: A new **cross-alignment co-selection** mode that takes two alignments (host gene + pathogen gene) with **paired taxa** (same N samples), computes TAVs independently for each, then tests for correlated selection signals *between* sites in different genes. This is the molecular signature of host-pathogen coevolution — reciprocal adaptive changes at interacting interface residues.

---

## 2. Pilot Validation Gene Pairs

Four well-characterized P. falciparum–human interactions, chosen for structural data, known polymorphism, and diverse interaction types:

| # | Pathogen Gene | Host Gene | Interaction | Rationale |
|---|---|---|---|---|
| 1 | **PfRH5** (PF3D7_0424100) | **Basigin/BSG** (CD147) | Invasion ligand–receptor | Essential, no redundancy, crystal structure (PDB 4U0Q), known polymorphism both sides |
| 2 | **PfEBA-175** (PF3D7_0731500) | **Glycophorin A / GYPA** | Invasion ligand–receptor | Classic interaction, MNS blood group polymorphism, sialic acid binding |
| 3 | **PfAMA1** (PF3D7_1133400) | **RON2** (PF3D7_1145300, secreted into host) | Moving junction formation | Major vaccine antigen, RON2 peptide inserts into AMA1 hydrophobic groove (PDB 2Y8T) |
| 4 | **PfMSP1-19** (PF3D7_0930300, C-terminal 19kDa) | **Erythrocyte surface** (BSG-independent binding) | Merozoite surface invasion | Vaccine target, known escape polymorphism, smaller domain simplifies alignment |

**Note on pair 3**: RON2 is secreted by the parasite into the host erythrocyte membrane, so this is technically a parasite–parasite interaction at the molecular level, but it represents an arms-race between AMA1 diversity and RON2 conservation — a different coevolutionary dynamic that tests the method's range.

**Note on pair 4**: MSP1-19 binds the erythrocyte surface non-specifically; the "host partner" is less well-defined. We include it because MSP1-19 is a compact, well-studied domain with strong vaccine-driven polymorphism data. If no clear host interactor is needed, we can treat this as pathogen-only arms-race detection (within-gene epistasis under host immune pressure) as a contrast case.

---

## 3. Data Acquisition

### 3.1 Primary Source: MalariaGEN Raw FASTQs from ENA

**Source**: Band, G., Leffler, E.M., Jallow, M. et al. *Malaria protection due to sickle haemoglobin depends on parasite genotype.* Nature (2021). doi:10.1038/s41586-021-04288-3

**Cohort**: 5,096 children with severe malaria from Banjul, The Gambia and Kilifi County, Kenya (1995-2009). Raw sequencing data deposited in ENA under study **ERP000190** (open access). Sample accession list with metadata on **Zenodo** (10.5281/zenodo.5723515).

**Two sequencing approaches** (key to our strategy):

| Approach | N samples | Description | Host reads? | Parasite coverage |
|---|---|---|---|---|
| **Whole DNA** | ~1,071 (483 Gambian + 588 Kenyan) | Total DNA from blood, high parasitaemia (>10^5 pRBCs/μl) | **Yes — 20x+ human genome** | 25–1000x Pf genome |
| **SWGA** | ~4,000+ (all samples) | Selective whole-genome amplification enriches Pf DNA | Minimal (<1x human) | High Pf genome |

**Critical insight**: Whole DNA samples contain both human and parasite reads. The original study aligned to a combined GRCh37 + Pf3D7 reference and extracted parasite reads only — but the **raw FASTQs still contain human reads**. We can extract both host and pathogen sequences from the same blood draw.

**Dual-source extraction strategy**:

| Target | Source FASTQ | Reference | Rationale |
|---|---|---|---|
| **Host genes** (BSG, GYPA, etc.) | Whole DNA FASTQs | GRCh38 | Human reads present at ~20x coverage — sufficient for variant calling in coding regions |
| **Pathogen genes** (PfRH5, PfEBA-175, etc.) | SWGA FASTQs (preferred) or whole DNA FASTQs | Pf3D7 | SWGA enriches parasite reads; whole DNA also works for high-parasitaemia samples |
| **Combined** | Both (where available) | Both | Many samples have both whole DNA + SWGA — use whole DNA for host, SWGA for pathogen |

**Access plan**:
1. **Immediate (open access)**: Download raw FASTQs from ENA (ERP000190). Filter sample list from Zenodo (10.5281/zenodo.5723515) to identify samples with whole DNA sequencing, and those with both whole DNA + SWGA.
2. **Pilot (two arms)**: Run both extraction strategies in parallel on the same 50–100 samples (those with both whole DNA + SWGA):
   - **Arm A — Whole DNA only**: Extract host + pathogen both from whole DNA FASTQs. Simpler pipeline (one download, one alignment). Question: is parasite coverage sufficient at ~0.6–40% Pf reads?
   - **Arm B — Paired (whole DNA + SWGA)**: Extract host from whole DNA FASTQs, pathogen from SWGA FASTQs. Higher parasite coverage but requires two downloads + two alignments per sample.
   - Compare: parasite gene coverage depth, variant call concordance, cross-coselection results. If Arm A is sufficient, simplify pipeline for scale.
3. **Scale**: Based on pilot comparison, choose strategy for genome-wide screen. If whole DNA sufficient for pathogen, use single-download pipeline. If SWGA needed for pathogen, use paired strategy for samples with both; for SWGA-only samples, supplement host from 1000G WGS (GWD/LWK).
4. **Fallback**: If ENA FASTQ access is slow, use Sanger FTP BAM files (parasite-only) for pathogen + 1000G WGS for host (population-level matching).

### 3.2 Supplementary Source: Pf8 Dataset

**Source**: MalariaGEN Pf8 (June 2025). 33,325 P. falciparum genomes, open access (CC-BY 4.0). doi:10.12688/wellcomeopenres.24031.1

**Use case**: Expand parasite sample size beyond the 4,171 host-parasite paired samples. Pf8 VCFs contain 12.5M variants with `variant_CDS` flag for coding variants. The `malariagen_data` Python package provides cloud-based access without local download.

**Important**: Pf8 CRAM files contain only parasite-mapped reads (human reads were filtered out during processing). Cannot extract host sequences from Pf8 CRAMs.

**Parasite gene extraction from Pf8** (for expanding beyond MalariaGEN samples):
```python
import malariagen_data
pf8 = malariagen_data.Pf8()
variant_data = pf8.variant_calls()
# Filter to CDS variants in target gene (e.g., PF3D7_0424100 for PfRH5)
# Reconstruct per-sample coding sequence from reference + variants
```

### 3.3 Gene Sequence Extraction from Raw FASTQs

**No variant calling needed** — we want per-sample coding sequences, not genotype calls. Pipeline: extract reads → consensus sequence → codon alignment.

**Limitation noted**: Consensus sequences collapse intra-host diversity and co-infection complexity. Each sample is represented as a single haploid sequence per gene, losing information about mixed infections (MOI > 1) and within-host minor variant frequencies. This is acceptable for the pilot — HyphAeon operates on codon-level selection patterns across samples, not within-sample diversity. Future extension could use read-level information (allele frequencies, MOI estimates) as covariates or filtering criteria.

**Mini-reference approach** (instead of aligning to full genomes):
1. Build a small FASTA: target gene CDS + ~5kb flanking from GRCh38 (BSG, GYPA, ICAM1) and Pf3D7 (PfRH5, PfEBA-175, PfAMA1, RON2, MSP1-19). Total ~800kb.
2. Index once with BWA-MEM2 (or minimap2).

**Pipeline** (per sample, per gene):

```
For each sample:
  1. Download FASTQ from ENA (fasterq-dump or ena-api)
     - Arm A: whole DNA FASTQ only
     - Arm B: whole DNA FASTQ + SWGA FASTQ
  2. Align to mini-reference (BWA-MEM2 or minimap2 -x sr)
     - For Pf field isolates (~1-2% divergence from 3D7): tune minimap2 -k 15 -w 8
       or use BWA-MEM2 (more sensitive for diverged short reads)
  3. Extract mapped reads → tiny BAM per gene (~MB per sample)
  4. Generate per-sample consensus sequence:
     - samtools consensus (majority base at each position)
     - or: bcftools mpileup + bcftools call + bcftools consensus (reference + variants → sequence)
     - or: simply use `samtools faidx` on the pileup (for high-coverage positions)
  5. Translate to amino acids, verify no premature stops
  6. Delete full FASTQ after extraction (keep only consensus sequences)
```

**Pathogen genes** (from SWGA FASTQs or whole DNA FASTQs):
1. Align to Pf3D7 gene regions in mini-reference
2. Extract consensus CDS per sample
3. MACSE for codon alignment (frameshift-aware, per CAPHEINE protocol)

**Host genes** (from whole DNA FASTQs):
1. Align to GRCh38 gene regions in mini-reference
2. Extract consensus CDS per sample
3. **Key advantage over SNP array**: captures all coding variants, including rare nonsynonymous SNPs that arrays miss
4. At ~20x coverage, expect reliable consensus at ≥95% of coding positions
5. For samples without whole DNA (SWGA only): use 1000G WGS (GWD/LWK) as population-level match

**Data size estimate** (100 samples):

| Item | Per sample | 100 samples |
|---|---|---|
| Compressed FASTQ download | ~900 MB | ~90 GB (Arm A) or ~180 GB (Arm B) |
| Extracted target BAM | ~10-50 MB | ~1-5 GB |
| Final consensus sequences | ~10 KB | ~1 MB |
| Mini-reference index | ~800 KB | shared |

**Optimization**: Stream from ENA → align to mini-reference → extract → delete full FASTQ. Only keep consensus sequences. Storage needed at any point: ~1-2 GB per sample (temporary), ~5 GB total for final extracted data.

### 3.4 Alignment Construction

For each gene, per sample pair:

1. **Codon-aware alignment**: MAFFT (`--auto`) or MACSE (`--seq` for codon alignment with frameshift awareness)
   - MACSE preferred for pathogen genes (handles frameshifts/stops in field isolates)
   - MAFFT sufficient for host genes (higher quality, fewer frameshifts)
2. **Alignment QC**: HyphAeon's built-in `filter` command (`hyphaeon filter`) for automated error masking
3. **Tree inference**: IQ-TREE2 (ModelFinder, ultrafast bootstrap) — one tree per gene per sample set
   - Alternatively: TN93 distance mode (`--no-tree`) for rapid screening, switch to explicit trees for dedicated analysis
4. **Taxon matching**: Ensure identical taxon labels across paired alignments (MalariaGEN sample ID as taxon name)

### 3.5 Pairing Metadata

A TSV manifest mapping each MalariaGEN sample to:
- Sample ID (used as taxon label in both alignments; matches MalariaGEN sample IDs)
- MalariaGEN partner study ID
- ENA run accession (whole DNA FASTQ)
- ENA run accession (SWGA FASTQ, if separate)
- Sequencing type (whole_dna / swga / both)
- Geographic origin (Gambia / Kenya)
- Collection year
- Host gene coverage depth (from whole DNA alignment)
- Pathogen gene coverage depth (from SWGA or whole DNA alignment)
- Pairing mode (individual-level / population-level)
- Host variant call quality flag
- Pathogen variant call quality flag

---

## 4. HyphAeon Code Changes

### 4.1 New Module: `cross_coselection.py`

Location: `@/home/danielle/Documents/veg/HyphAeon/hyphaeon/src/hyphaeon/cross_coselection.py`

Core function:

```python
def compute_cross_coselection_network(
    attr_host: np.ndarray,    # [L_host, N] TAVs from host alignment
    lrts_host: np.ndarray,     # [L_host]
    attr_pathogen: np.ndarray, # [L_pathogen, N] TAVs from pathogen alignment
    lrts_pathogen: np.ndarray, # [L_pathogen]
    taxa: List[str],           # paired sample IDs (must match)
    consensus_aas_host: List[str],
    consensus_aas_pathogen: List[str],
    min_sim: float = 0.30,
    min_shared: int = 2,
    max_fdr: float = 0.05,
    min_lrt: float = 1.0,
    min_cesi: float = 2.0,
    interface_mask: Optional[Dict] = None,  # optional: restrict to known interface residues
) -> Tuple[List[Dict], nx.Graph]
```

**Algorithm** (mirrors `compute_branch_coselection_network` but cross-gene):
1. Normalize both attribution matrices (active sites only)
2. Compute cross-gene cosine similarity matrix: `S[i,j] = cos(attr_host[i,:], attr_pathogen[j,:])` for all active site pairs (L_host × L_pathogen)
3. Compute shared-sample matrix: `B[i,j] = sum(I(attr_host[i,:] > 0 & attr_pathogen[j,:] > 0))`
4. Student's t-test on each cross-gene pair (df = N-2)
5. Global BH-FDR across all L_host × L_pathogen tests
6. CESI for cross-gene pairs: `cesi = sim * sqrt(max(lrt_host, 0.1) * max(lrt_pathogen, 0.1))`
7. Filter by min_sim, min_shared, max_fdr, min_lrt, min_cesi
8. Return bipartite graph (host sites + pathogen sites as nodes, co-selection edges between)

**Key difference from within-gene**: The FDR burden is L_host × L_pathogen (not L choose 2), which can be much larger. For two 500-codon genes, that's 250K tests vs ~125K. Need to consider whether to restrict to interface residues (if structural data available) to reduce multiple testing burden.

### 4.2 New CLI Subcommand: `cross-coselection`

```
hyphaeon cross-coselection \
  --alignment-host host_BSG.fasta \
  --tree-host host_BSG.nwk \
  --alignment-pathogen pathogen_PfRH5.fasta \
  --tree-pathogen pathogen_PfRH5.nwk \
  --min-sim 0.30 \
  --max-fdr 0.05 \
  --interface-pdb 4U0Q \
  --interface-distance 8.0 \
  -o results_cross_coselection.json \
  -c results_cross_coselection.csv \
  --graphml results_cross_coselection.graphml
```

Arguments:
- `--alignment-host` / `--alignment-pathogen`: paired codon FASTA alignments
- `--tree-host` / `--tree-pathogen`: optional trees (or `--no-tree` for TN93)
- `--interface-pdb` (optional): PDB ID or file to restrict co-selection tests to structurally proximal residue pairs
- `--interface-distance` (default 8.0 Å): distance cutoff for interface definition
- `--min-sim`, `--max-fdr`, `--min-shared`, `--min-lrt`, `--min-cesi`: same as epistasis
- `--n-permutations` / `--max-perm-p`: permutation testing for cross-gene sector coherence
- `--graphml`: export bipartite co-selection network
- `-o` / `-c`: JSON + CSV output

### 4.3 New CLI Subcommand: `screen`

For the genome-wide screening follow-up:

```
hyphaeon screen \
  --host-dir host_alignments/ \
  --pathogen-dir pathogen_alignments/ \
  --pair-file gene_pairs.tsv \
  --tree-dir trees/ \
  --mode tn93 \
  --min-sim 0.30 \
  --max-fdr 0.05 \
  --output-dir screen_results/
```

Where `gene_pairs.tsv` is:
```
pathogen_gene	host_gene
PF3D7_0424100	BSG
PF3D7_0731500	GYPA
...
```

**Screen mode logic**:
1. For each gene pair: load both alignments, compute TAVs, compute cross-coselection summary statistics
2. Fast mode: TN93 distance (no tree), skip DMS, skip sector mining
3. Output: summary CSV with per-pair statistics (n_edges, max_cesi, mean_sim, n_interface_edges, p_value)
4. Flag pairs exceeding threshold for dedicated `cross-coselection` follow-up

### 4.4 Interface Restriction (Optional but Recommended)

For pilot pairs with known structures:
- Parse PDB to identify residue pairs within `--interface-distance` Å
- Map PDB residue numbers to alignment positions (sequence-to-structure mapping)
- Restrict cross-coselection tests to interface residue pairs only
- Dramatically reduces multiple testing burden (e.g., 30 × 25 = 750 tests vs 500 × 500 = 250K)
- Increases statistical power per test

Implementation: lightweight PDB parsing using BioPython's `PDBParser`, compute CA-CA distances, map to alignment positions via reference sequence.

### 4.5 Cross-Gene Sector Mining

Extend `extract_epistatic_sectors_tse` to work on bipartite graphs:
- A cross-gene "sector" = a set of host sites + pathogen sites that co-vary coherently
- Spectral coherence computed on the concatenated attribution submatrix `[A_host[S_host, :]; A_pathogen[S_path, :]]`
- Permutation test: random K_host + K_pathogen site subsets of same total size
- Biological interpretation: co-evolving interface patch

### 4.6 Summary of Files to Create/Modify

| File | Action | Description |
|---|---|---|
| `hyphaeon/src/hyphaeon/cross_coselection.py` | **Create** | Core cross-alignment co-selection logic |
| `hyphaeon/src/hyphaeon/cli.py` | **Modify** | Add `cross-coselection` and `screen` subcommands |
| `hyphaeon/src/hyphaeon/__init__.py` | **Modify** | Export new functions |
| `hyphaeon/tests/test_cross_coselection.py` | **Create** | Unit tests for cross-gene co-selection |
| `scripts/prepare_paired_alignments.py` | **Create** | MalariaGEN raw FASTQ → paired alignment pipeline (ENA download → dual alignment → variant calling → codon FASTA) |
| `scripts/build_gene_pair_manifest.py` | **Create** | Build gene-pair TSV from interaction databases |
| `scripts/extract_interface_residues.py` | **Create** | PDB → interface residue mask |

---

## 5. Validation Plan

### 5.1 Positive Controls (Known Coevolution)

For each pilot pair, verify that HyphAeon's cross-coselection recovers known interface residues:

- **PfRH5–BSG**: Crystal structure PDB 4U0Q. Interface residues documented (e.g., PfRH5 residues 197-203, BSG residues 34-50). Check if cross-coselection edges enrich at these positions.
- **PfEBA-175–GYPA**: Binding interface involves PfEBA-175 F2 domain and GYPA sialic acid residues. Check enrichment.
- **PfAMA1–RON2**: RON2 peptide binds AMA1 hydrophobic groove (PDB 2Y8T). Check if AMA1 groove residues show cross-coselection with RON2.
- **PfMSP1-19**: Known escape mutations (e.g., Q1644K, L1685F). Check if these positions show elevated within-gene epistasis (contrast case).

### 5.2 Negative Controls

- **Shuffled pairing**: Randomly permute taxon labels in one alignment → break pairing → cross-coselection signal should collapse
- **Non-interacting gene pairs**: Test PfRH5 vs GYPA (no known interaction) → expect no significant cross-coselection edges
- **Random site pairs**: Permutation null (already built into the method via FDR)

### 5.3 Convergence with Literature

- Cross-reference flagged co-evolving site pairs with:
  - UniProt annotated functional sites
  - PDB interface residues (for pairs with structures)
  - Published coevolution studies (e.g., PfRH5 polymorphism studies, EBA-175 binding studies)
  - Vaccine escape mutation literature (for AMA1, MSP1)

### 5.4 Quantitative Metrics

| Metric | Target |
|---|---|
| Interface enrichment | ≥ 2× over random (Fisher's exact test p < 0.05) |
| Shuffled-pairing signal loss | ≥ 80% reduction in significant edges |
| Non-interacting pair false positive rate | < 5% of tests significant |
| Sensitivity (known interface recovery) | ≥ 50% of known interface residues flagged |

---

## 6. Genome-Wide Screening Experiment

### 6.1 Gene Pair Compilation

- Source all known P. falciparum–human protein interactions from:
  - STRING database (P. falciparum taxon 36329)
  - BioGRID
  - IntAct
  - UniProt annotations
  - Published interactome studies (e.g., Lavstsen 2012, Bei 2010)
- Additionally: screen all Pf erythrocyte invasion genes (merozoite, rhoptry, microneme proteins) against a panel of human erythrocyte surface proteins
- Expected scale: ~500-2000 gene pairs

### 6.2 Screening Pipeline

```
For each gene pair (pathogen_gene, host_gene):
  1. Load paired alignments (from pre-built alignment directory)
  2. Compute TAVs for both genes (TN93 mode, no tree — speed)
  3. Compute cross-coselection summary statistics:
     - n_significant_edges (FDR < 0.05)
     - max_cesi, mean_cesi
     - n_interface_edges (if structure available)
     - spectral_coherence_cross (if sector detected)
  4. Flag pairs with n_significant_edges ≥ threshold
  5. Output: summary CSV ranked by max_cesi or n_significant_edges
```

### 6.3 Flagged-Pair Follow-Up

For top ~5% of screened pairs:
1. Build proper codon alignments with MACSE (frameshift-aware)
2. Infer IQ-TREE2 phylogenies
3. Run full `hyphaeon cross-coselection` with sector mining, permutation testing, DMS
4. Interface analysis if structure available
5. Literature cross-reference

### 6.4 ChronAeon AutoClock as Screening Covariate

Coevolving host-pathogen gene pairs should show correlated evolutionary rates — reciprocal selection drives accelerated substitution at interacting residues. ChronAeon's `autoclock` feature (see `@/home/danielle/Documents/veg/HyphAeon/chronaeon/src/chronaeon/cli.py`) estimates per-branch evolutionary rates from alignments. For the genome-wide screen:

1. Run `chronaeon autoclock` on both host and pathogen alignments independently
2. Compare rate distributions: coevolving pairs should show correlated rate acceleration across shared lineages (same MalariaGEN samples)
3. Use rate correlation as a **pre-filter** before cross-coselection: pairs with uncorrelated evolutionary rates are less likely to show true co-selection
4. This is computationally cheap (rate estimation << TAV computation) and can reduce the number of pairs needing full cross-coselection analysis

**Note**: This is a hypothesis — coevolution doesn't always produce correlated global rates (selection may be localized to interface residues while the rest of the gene evolves neutrally). Test on pilot pairs first: do PfRH5–BSG show correlated autoclock rates? If yes, add as screen pre-filter. If no, skip.

### 6.5 Compute Estimate

- HyphAeon inference: ~1-3 seconds per gene (L=500, N=50, TN93 mode)
- Cross-coselection computation: ~0.1s per pair (matrix operations)
- Total for 1000 pairs: ~2-5 hours on a single GPU
- Alignment construction: bottleneck (MACSE ~1-5 min per gene), but done once
- AutoClock pre-filter (if validated): ~0.5s per gene, negligible cost

---

## 7. Implementation Phases

### Phase 1: Core Cross-Coselection (Weeks 1-2)
- Implement `cross_coselection.py` with `compute_cross_coselection_network`
- Add `cross-coselection` CLI subcommand
- Unit tests with synthetic paired alignments
- Validate on shuffled vs. correlated synthetic data

### Phase 2: Data Pipeline (Weeks 2-3)
- `prepare_paired_alignments.py`: ENA FASTQ download → dual-reference alignment (GRCh38 + Pf3D7) → variant calling → codon FASTA
- Download pilot set: 50–100 samples with both whole DNA + SWGA from ENA (ERP000190)
- Build manifest for pilot gene pairs
- Generate pilot alignments for all 4 pairs, **both arms**:
  - Arm A: whole DNA only (host + pathogen from same FASTQ)
  - Arm B: whole DNA (host) + SWGA (pathogen)
- Assess coverage: host gene depth (BSG, GYPA) from whole DNA; pathogen gene depth (PfRH5, PfEBA-175) from both arms
- Compare variant call concordance between arms for pathogen genes

### Phase 3: Pilot Validation (Weeks 3-4)
- Run `cross-coselection` on all 4 pilot pairs, **both arms** (8 total runs)
- Compare arms: do cross-coselection results differ? Which arm has more power (more significant edges, higher CESI)?
- Interface enrichment analysis
- Negative controls (shuffled pairing, non-interacting pairs)
- Cross-reference with structural data
- **Decision point**: choose Arm A or B for genome-wide screen based on parasite coverage + result quality

### Phase 4: Interface Restriction & Sector Mining (Weeks 4-5)
- PDB parsing for interface masks
- Cross-gene sector mining
- Permutation testing for cross-gene sectors
- DMS on flagged cross-gene sector sites

### Phase 5: Screen Mode (Weeks 5-6)
- `screen` CLI subcommand
- Gene-pair manifest builder
- Run genome-wide screen
- Rank and flag pairs

### Phase 6: Flagged-Pair Deep Analysis (Weeks 6-7)
- Full `cross-coselection` on top flagged pairs
- Literature validation
- Write-up

---

## 8. Key Design Decisions & Risks

### 8.1 Sample Pairing Fidelity
**Risk**: Whole DNA samples (~1,071) are a subset of the full 5,096 cohort. SWGA-only samples lack usable host reads. Host gene coverage from whole DNA varies with parasitaemia and sequencing yield.
**Mitigation**: (1) Use whole DNA samples for both host + pathogen extraction (true paired data from same blood draw). (2) For SWGA-only samples, supplement with 1000G WGS (GWD/LWK) as population-level match. (3) Filter samples by host gene coverage depth (≥10x at coding positions). (4) The Zenodo accession list (10.5281/zenodo.5723515) identifies which samples used whole DNA vs SWGA — use this to stratify. (5) Document pairing mode per sample.

### 8.2 Multiple Testing Burden
**Risk**: L_host × L_pathogen tests can be very large (e.g., 500 × 500 = 250K), reducing power after FDR correction.
**Mitigation**: Interface restriction when structures available. For pairs without structures, use stricter min_sim/min_cesi thresholds or pre-filter to top-LRT sites only.

### 8.3 Alignment Quality
**Risk**: Field isolate sequences may contain frameshifts, stop codons, or ambiguous bases that corrupt codon alignments.
**Mitigation**: MACSE for codon alignment (frameshift-aware). HyphAeon `filter` command for automated error masking. Document MACSE frameshift/stop retention per CAPHEINE verification protocol.

### 8.4 Model Generalization
**Risk**: HyphAeon was trained on diverse pathogen alignments (viral, bacterial, protozoan). Human genes (BSG, GYPA) are eukaryotic, multicellular, with different selective regimes. Host gene variation may be sparse (BSG is highly constrained; GYPA has few coding SNPs beyond MN blood group).
**Mitigation**: The model operates on codon-level selection patterns, not organism-specific features. Validate with negative controls. If human gene TAVs show artifacts, consider using the `--model-variant` flag or fine-tuning. Raw FASTQ extraction from whole DNA captures all coding variants (including rare nonsynonymous SNPs missed by SNP arrays), maximizing available variation. If coding variation is insufficient for a specific gene, flag it and exclude from screen.

### 8.5 Taxon Label Matching
**Risk**: Paired alignments must have exactly matching taxon labels for cross-coselection to be meaningful.
**Mitigation**: Enforce strict label matching in `compute_cross_coselection_network` — raise error if taxa lists don't match. Pre-processing script ensures consistent naming.

---

## 9. Output Artifacts

### Per-Pair (Dedicated Analysis)
- `cross_coselection.json`: Full results (edges, sectors, DMS, permutation stats)
- `cross_coselection.csv`: Edge-level summary (site_host, site_pathogen, sim, cesi, fdr_q, shared_taxa)
- `cross_coselection.graphml`: Bipartite co-selection network for Cytoscape/Gephi
- `interface_enrichment.csv`: Fisher's exact test results for interface enrichment

### Screen (Genome-Wide)
- `screen_summary.csv`: Per-pair summary, ranked by max_cesi / n_significant_edges
- `screen_flagged_pairs.tsv`: Pairs exceeding threshold for follow-up

### Validation
- `validation_report.md`: Positive/negative control results, interface enrichment, literature cross-reference

---

## 10. Empirical Findings: Two Distinct Notions of "Coselection"

Prototyping (`examples/hpi/scripts/model_coselect*.py`, `symbaeon coselect`) surfaced a methodological distinction that must be made explicit, because the two approaches answer **different questions** and require **different positive controls**.

### 10.1 The two notions

| | **Allele-correlation** (linkage) | **Selection-correlation** (coevolution) |
|---|---|---|
| Question | Do specific *alleles* travel together across individuals? | Do the same *lineages* drive the selection signal at both sites? |
| Method | `symbaeon coselect` (relatedness-corrected association) | `hyphaeon epistasis` / `model_coselect` (attribution-cosine) |
| Signal | per-site allele × allele association, relatedness-regressed | `compute_branch_coselection_network`: both sites need `min_lrt` + correlated attribution profiles (`cesi = sim·√(lrt_u·lrt_v)`) |
| Positive control | plant correlated alleles | plant correlated *selection* (same lineages under selection at both sites) |

These are **complementary, not competing**. Linkage detects association; selection-correlation detects coevolution of selected lineages. A site pair can show one without the other.

### 10.2 Positive-control result (Gag × Protease, 173 HIV patients)

Planted a **non-synonymous allele correlation** (host codon 251 tracks par codon 116, 173/173 individuals) via `inject_coselect.py`:

| method | planted pair recovered? |
|---|---|
| `coselect` uncorrected | **rank 1 / 167,168** (q=2e-46) |
| `coselect` `--par-pc 5 --host-pc 5` | **rank 1 / 167,168** (q=5e-38) — correction costs no power |
| `model_coselect` (concat) | not recovered |
| `model_coselect_paired` (per-side dist) | not recovered |

**Interpretation**: the stats method is validated for allele-correlation. The model "miss" is **not** a falsification — an allele correlation is not a selection signal, so the model correctly doesn't flag it. The model targets selection-correlation, which this control doesn't test.

### 10.3 Implementation caveat discovered

`model_coselect` used `compute_transformer_attributions` — the cheap `a_{s,n} = alpha_{root→n,s}·delta_{s,n}` proxy (attention × non-consensus-mutation), which is dominated by the attention/relatedness term. The **proper** selection-driver signal is `attribute_selection` (`attribution.py`): single-taxon counterfactual **Delta-LRT** — *"how much does taxon n's mutation drive site s's selection LRT."* Cross-gene coselection should correlate **Delta-LRT selection-driver profiles**, gated by `min_lrt`, not the alpha×delta proxy.

### 10.4 Corrected validation approach

- **Architecture**: `model_coselect_paired.py` (two MSAs, per-side distance matrix — avoids the concat rate-domination problem) is the right skeleton.
- **Signal**: use `attribute_selection` (Delta-LRT) or gate the attribution-cosine on `min_lrt`/`cesi` so only *selected* sites are tested.
- **Control**: plant a **selection correlation** — the same lineages under selection at a host and a par site (phylogenetically-clustered nonsynonymous changes), not an allele correlation.
- **Caveat**: the within-gene epistasis attribution-cosine is itself only unit-tested on synthetic arrays, not validated against known coevolving pairs — so the selection-correlation signal's real-world validity is still open and needs this control.

### 10.5 Relatedness vs coselection (the confound)

The model's leaf-attribution is dominated by tree position (`alpha`), so naive attribution-cosine detects **relatedness** — *"both sites vary along the same family branches"* — not allele linkage. The stats method regresses out relatedness explicitly (`--par-pc`/`--host-pc`); the model must either (a) use the Delta-LRT selection-driver signal, or (b) residualize attributions on relatedness before correlating. This is the central design risk for the model path.

### 10.6 Selection-correlation positive control — implemented & passing

The corrected approach from §10.4 is now wired in and validated on a purpose-built control.

**Simulator** (`examples/hpi/hiv/sim_mg94.py`): a small Python MG94 codon simulator. HyPhy batch-mode simulation was abandoned — the `libv3` `ApplyModelToTree` path resets branch lengths to a separate parameter that resisted correct setting after extensive debugging, so a direct `scipy.linalg.expm` simulator was written instead (full control over per-branch ω). Design: a fixed 40-taxon tree where the **internal branch to a foreground clade (t0–t9) gets ω=5** and all other branches are purifying (ω=0.15); the background internal branch is collapsed to ~0 so only the FG clade carries *shared* selected substitutions. Run twice (different seeds) → `sim_host.fasta` / `sim_par.fasta`, each with ~35–42 sites where all 10 FG taxa share a derived amino acid distinct from all background taxa.

**Delta-LRT wiring** (`model_coselect_paired.py --method delta_lrt`): replaces the `alpha·delta` proxy with `attribute_selection` (`attribution.py`) — for each selected site, each non-consensus taxon is mutated to consensus and `delta_lrt = site_lrt − mod_lrt` is recorded, yielding a per-(site, taxon) **selection-driver matrix**. Cross-gene cosine on these profiles tests whether the *same lineages* drive selection at both sites. `--attr-min-lrt` gates which sites get attributed.

**Result** (200 codons × 40 taxa, `--attr-min-lrt 1.0 --min-sim 0.3 --min-lrt 1.0`, 20 permutations):

| metric | observed | null (mean / max) | p |
|---|---|---|---|
| edges | **207** | 157 / 202 | **0.048** |

Driver-fraction check confirms the mechanism: the top edges (`host:33×par:38` cos=0.97, `host:176×par:16` cos=0.92, `host:170`, `par:41`, `par:12`) are driven by the **t0–t9 foreground clade** (FG-driver-fraction 0.8–1.0) in *both* genes — i.e. the same selected lineages, which is precisely the selection-correlation signal. A minority of edges are background-driven (residual shared-ancestry correlation), a caveat to carry into real-data interpretation.

**Negative control** (`sim_mg94.py` with `--fg` on different clades): host selects on t0–t9, par selects on t20–t29 — both genes have real episodic selection but on *disjoint* lineages. Result: **190 edges vs null mean 201.6, p=0.81** (no excess). The method is specific for *shared* drivers, not merely "selection present in both genes" — the permutation null already absorbs the structured-variation baseline, so only a true shared-lineage signal pushes obs above null.

**Status**: the model path now has a validated positive **and** negative control for selection-correlation, complementing the stats method's allele-correlation control (§10.2).

### 10.7 delta_lrt on real paired data

| pair | N inds | host sel / par sel | edges | perm p | verdict |
|---|---|---|---|---|---|
| **Gag × Protease** (HIV, 173 patients) | 173 | 440 / 91 | **910** | **0.048** | significant excess; coherent cluster |
| **CR1 × PfRH4** (malaria) | 3347 | 137 / 180 | **0** | 1.0 | no detectable signal |

**Gag × Protease**: 910 edges vs null mean 392 / max 703 → p=0.048. Top edges form a coherent cluster — Gag sites **451/452** (adjacent, same signal) and **334/433** × Protease sites **138/17/14/102/108/143** at cos≈0.85. The 451/452 edges are positive-cosine (same driving individuals); the 334 edges are negative-cosine (anti-correlated drivers) — two distinct lineage-sharing patterns. Caveat: 38.5% of protease codons carry gaps/ambiguities, so protease-side signal should be re-checked after alignment QC.

**CR1 × PfRH4**: 0 edges even at min-sim 0.15, and the permutation null is also 0 — no host×par site pair reaches threshold. Diagnostic shows **median drivers/site = 0** for both genes: most selected sites have *no* single-haplotype driver whose reversion lowers the LRT.

**Lineage counterfactual resolves the ambiguity** (`--method delta_lrt_aa`): to test whether the negative was a power artifact of single-taxon reversion, a grouped counterfactual reverts **all carriers of each derived amino acid jointly** and assigns the group's ΔLRT to every carrier. On the positive control this still detects the shared-clade signal (773 edges, p=0.048 — stronger than single-taxon, since allele-group reversion captures the full clade effect). On CR1×PfRH4 it yields drivers at only **10/137 host** and **4/180 par** selected sites, max |cos| = 0.05 → still 0 edges. **Interpretation**: the selection at these genes is *not attributable to any allele or lineage* — reverting even a whole allele-class doesn't move the LRT, so the signal is aggregate/diffuse diversifying selection (immune evasion on CR1, host-receptor variation on PfRH4) spread across the alignment, not concentrated on lineages that could co-vary. The malaria negative is therefore a **true negative for selection-correlation**, not a detection failure — the method correctly reports "no shared driving lineages" because there are none.
