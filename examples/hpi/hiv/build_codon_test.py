#!/usr/bin/env python3
"""Build a codon-aligned test set for model_coselect.

Fetches the Gag *nucleotide* CDS for a subset of patients and threads each
onto the existing protein MSA (gag_msa_hxb2.fasta) -> a codon MSA in HXB2
coordinates. Then splits it into two halves (geneA/geneB) so the
cross-block machinery can be exercised on real sequences.
"""
import time, urllib.request, urllib.parse
from Bio import SeqIO
from io import StringIO

N = 60
accs = [l.strip() for l in open("accessions.txt") if l.strip()][:N]
EMAIL = "brc@example.org"

def efetch(ids):
    q = urllib.parse.urlencode({"db": "nuccore", "id": ",".join(ids),
        "rettype": "gb", "retmode": "text", "email": EMAIL, "tool": "symbaeon"})
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?" + q
    with urllib.request.urlopen(url, timeout=120) as r:
        return r.read().decode("utf-8", "replace")

def gag_cds_nt(rec):
    """Return the nt subsequence of the gag CDS (by feature location)."""
    best = None
    for f in rec.features:
        if f.type != "CDS":
            continue
        gene = "".join(f.qualifiers.get("gene", [])).lower()
        prod = "".join(f.qualifiers.get("product", [])).lower()
        if "gag" in gene or ("gag" in prod and "pol" not in prod):
            nt = str(f.extract(rec.seq))
            if best is None or len(nt) > len(best):
                best = nt
    return best

# fetch nt CDS
nt = {}
for i in range(0, len(accs), 60):
    for rec in SeqIO.parse(StringIO(efetch(accs[i:i+60])), "genbank"):
        acc = rec.annotations.get("accessions", [rec.id])[0]
        s = gag_cds_nt(rec)
        if s:
            nt[acc] = s
    time.sleep(0.4)
print("fetched nt CDS:", len(nt))

# protein MSA keyed by accession
prot = {r.id: str(r.seq) for r in SeqIO.parse("gag_msa_hxb2.fasta", "fasta")}

# thread nt onto protein MSA -> codon MSA
codon = {}
for acc, pseq in prot.items():
    if acc == "HXB2" or acc not in nt:
        continue
    s = nt[acc]
    pos = 0
    out = []
    ok = True
    for aa in pseq:
        if aa == "-":
            out.append("---")
        else:
            if pos + 3 > len(s):
                ok = False
                break
            out.append(s[pos:pos+3]); pos += 3
    if ok:
        codon[acc] = "".join(out)
print("codon-aligned:", len(codon), "width", len(next(iter(codon.values()))))

# split into two halves -> geneA / geneB fastas keyed {patient}|{gene}
# map accession -> patient
import csv
acc2pat = {r["accession"]: r["patient"] for r in csv.DictReader(open("patient_map.tsv"), delimiter="\t")}
half = len(next(iter(codon.values()))) // 2
half -= half % 3
with open("codonA.fasta", "w") as fa, open("codonB.fasta", "w") as fb:
    for acc, s in codon.items():
        p = acc2pat.get(acc, acc)
        fa.write(f">{p}|geneA\n{s[:half]}\n")
        fb.write(f">{p}|geneB\n{s[half:]}\n")
print("wrote codonA.fasta / codonB.fasta (", len(codon), "inds )")
