#!/usr/bin/env python3
"""Prepare symbaeon coselect inputs for the HIV HLA x Gag dataset.

Outputs:
  gag_patients.fasta   parasite MSA, headers {patient}|gag (HXB2 -> |ref)
  host_dummy.fasta     invariant host MSA, headers {patient}|hla (0 var sites)
  host_cov.tsv         sample_id + one binary col per HLA allele
"""
import csv
from Bio import SeqIO

# patient -> accession
pmap = {}
for r in csv.DictReader(open("patient_map.tsv"), delimiter="\t"):
    pmap[r["accession"].strip()] = r["patient"].strip()
acc2pat = pmap

# 1) parasite MSA keyed by patient
with open("gag_patients.fasta", "w") as out:
    for rec in SeqIO.parse("gag_msa_hxb2.fasta", "fasta"):
        if rec.id == "HXB2":
            out.write(">HXB2|gag|ref\n" + str(rec.seq) + "\n")
        elif rec.id in acc2pat:
            out.write(f">{acc2pat[rec.id]}|gag\n{rec.seq}\n")

# patients that actually have a parasite sequence
pat_with_seq = {acc2pat[r.id] for r in SeqIO.parse("gag_msa_hxb2.fasta", "fasta")
                if r.id in acc2pat}

# 2) invariant host dummy MSA (one seq per patient) -> 0 variable host sites
with open("host_dummy.fasta", "w") as out:
    for p in sorted(pat_with_seq):
        out.write(f">{p}|hla\n" + "A" * 12 + "\n")

# 3) covariate file with sample_id header, restricted to patients with seqs
rows = list(csv.reader(open("hla_covariates.tsv"), delimiter="\t"))
hdr = rows[0]
hdr[0] = "sample_id"
with open("host_cov.tsv", "w") as out:
    w = csv.writer(out, delimiter="\t")
    w.writerow(hdr)
    for r in rows[1:]:
        if r[0] in pat_with_seq:
            w.writerow(r)

print("patients with gag seq:", len(pat_with_seq))
print("wrote gag_patients.fasta, host_dummy.fasta, host_cov.tsv")
