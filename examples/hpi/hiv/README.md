# HIV HLA × Gag — a positive-control dataset for symbaeon

A fully public, paired host-genotype × pathogen-genotype dataset used to
validate `symbaeon coselect` on real biology with known ground truth:
**HLA class-I alleles in the host drive specific escape mutations in HIV Gag.**

## Source

- **LANL HIV Sequence Database** — search interface with the
  *"Only sequences with HLA information"* filter + Gag region. The patient's
  HLA type is a curated field on each record. Bulk sequence download mail-backs
  on large result sets, so we pull the **background table** (all accessions +
  HLA + subtype + patient code) and fetch the sequences from **NCBI by
  accession** instead.
- `lanl_gag_hla_table.tsv` — the LANL "Save Background Info" download
  (15,846 Gag-region records, all with HLA type).

## Cohort

- **836 patients** with both an HLA type and a Gag sequence: **393 subtype B +
  443 subtype C**. One representative sequence per patient (longest).
- **69 HLA alleles** (≥3 carriers), normalized to locus + 2-digit group
  (`A*31:01`/`A3101` → `A31`).
- **764 patients** yield a Gag protein (annotated CDS translation); the rest are
  unannotated/unverified genomes (recoverable via nt alignment if needed).
- Parasite MSA = Gag amino acids aligned with `mafft`, projected onto **HXB2
  coordinates** (500 columns = Gag residues 1–500).

## Pipeline

| script | role |
|---|---|
| `build_selection.py` | parse LANL table → per-patient accession, HLA carrier matrix, subtype strata |
| `fetch_gag.py` | NCBI efetch GenBank → extract `gag` CDS `/translation` |
| `prep_symbaeon.py` | patient-keyed parasite MSA + invariant host dummy + `host_cov.tsv` |

Host variation is HLA allele carrier status → passed via `--host-cov` (a dummy
invariant host MSA contributes 0 host sites). Run:

```
symbaeon coselect --host-aln host_dummy.fasta --par-aln gag_patients.fasta \
  --host-cov host_cov.tsv --strata strata.tsv --strata-id-col sample_id \
  --strata-col subtype --encoding presence --min-carriers 5 --out out.tsv
```

## Results

**Subtype B alone (343 patients)** — top genome-wide hit is the canonical escape:

| host × parasite | q | note |
|---|---|---|
| **B57 × Gag-242 (T242N)** | 3.6e-10 | the most famous HIV escape (TW10 epitope) |

**Subtype B+C, stratified by subtype (764 patients)** — 94 hits; stratification
removes the population-structure confound:

| host × parasite | q | note |
|---|---|---|
| **B57 × Gag-242 (T242N)** | 4.5e-24 | canonical, stronger pooled |
| **B81 × Gag-186 / Gag-177** | 7e-10 / 3e-7 | B*81:01 → TL9-epitope escape, a subtype-C signal |
| B57 × Gag-163, B27 × Gag-273, B18/B14/B37/B67 | <1e-4 | coherent |

**Confounding control:** unstratified B+C gives 1,516 "hits" of which 99/100 top
are at subtype-defining sites (HLA frequencies differ between the B and C
cohorts). With `--strata subtype` only 4/94 hits sit at subtype-defining sites.

## Caveats

- `stat=inf` rows are zero-cell Fisher artifacts at small carrier counts.
- Moderate escapes (B27×Gag-264, B57×Gag-147) have strong raw p but are diluted
  by the ~11k-test FDR burden — expected conservative behaviour.
- HLA is a free-text field; normalization to 2-digit groups trades resolution
  for carrier counts.

## Files

- Inputs: `gag_patients.fasta`, `host_dummy.fasta`, `host_cov.tsv`, `strata.tsv`
- Metadata: `patient_map.tsv`, `hla_covariates.tsv`, `accessions.txt`
- Results: `coselect_aa.tsv` (B only), `coselect_bc_strat.tsv` (B+C stratified),
  `coselect_bc_unstrat.tsv` (B+C unstratified, confounded contrast)
- Reference: `hxb2_gag.fasta`
