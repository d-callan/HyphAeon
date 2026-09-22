#!/usr/bin/env python3
"""Build paired Gag x Protease codon + protein MSAs for the head-to-head.

Fetches GenBank records for a subset of patients, extracts:
  - Gag CDS   : nt subsequence + translation (protein)
  - Protease  : first 99 codons of the pol CDS (nt + protein)
Threads each nt onto its protein MSA -> codon MSAs in alignment coords.
Writes paired fastas keyed {patient}|{gene} for both stats and model runs.
"""
import csv, time, urllib.request, urllib.parse
from io import StringIO
from Bio import SeqIO
from Bio.Seq import Seq

N = 200
accs = [l.strip() for l in open("accessions.txt") if l.strip()][:N]
EMAIL = "brc@example.org"
acc2pat = {r["accession"]: r["patient"] for r in csv.DictReader(open("patient_map.tsv"), delimiter="\t")}

def efetch(ids):
    q = urllib.parse.urlencode({"db": "nuccore", "id": ",".join(ids),
        "rettype": "gb", "retmode": "text", "email": EMAIL, "tool": "symbaeon"})
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?" + q
    for _ in range(3):
        try:
            return urllib.request.urlopen(url, timeout=120).read().decode("utf-8", "replace")
        except Exception:
            time.sleep(2)
    return ""

def cds_by_gene(rec, want):
    """Return (nt, protein) of the CDS whose gene matches `want`."""
    for f in rec.features:
        if f.type != "CDS":
            continue
        gene = "".join(f.qualifiers.get("gene", [])).lower()
        if gene == want:
            nt = str(f.extract(rec.seq))
            prot = f.qualifiers.get("translation", [""])[0]
            if not prot:
                prot = str(Seq(nt).translate(to_stop=True))
            return nt, prot
    return None, None

gag_nt, gag_pr, prot_nt, prot_pr = {}, {}, {}, {}
for i in range(0, len(accs), 60):
    for rec in SeqIO.parse(StringIO(efetch(accs[i:i+60])), "genbank"):
        acc = rec.annotations.get("accessions", [rec.id])[0]
        g_nt, g_pr = cds_by_gene(rec, "gag")
        p_nt, p_pr = cds_by_gene(rec, "pol")
        if g_nt and g_pr:
            gag_nt[acc], gag_pr[acc] = g_nt, g_pr
        if p_nt and p_pr:
            prot_nt[acc], prot_pr[acc] = p_nt[:297], p_pr[:99]   # protease = pol[0:99]
    time.sleep(0.4)
print("gag:", len(gag_nt), "protease:", len(prot_nt))

# --- protein MSAs ---
# Gag: reuse existing HXB2-projected protein MSA (keyed by accession)
gag_msa = {r.id: str(r.seq) for r in SeqIO.parse("gag_msa_hxb2.fasta", "fasta")}
# Protease: ~99 aa, near-constant length -> align with mafft for safety
with open("_prot_raw.fasta", "w") as f:
    for acc, p in prot_pr.items():
        f.write(f">{acc}\n{p}\n")
import subprocess, os
subprocess.run(["mafft", "--auto", "--quiet", "_prot_raw.fasta"],
               stdout=open("_prot_aln.fasta", "w"), check=False)
prot_msa = {r.id: str(r.seq) for r in SeqIO.parse("_prot_aln.fasta", "fasta")}

def thread(nt, aln_prot):
    """Thread ungapped nt onto an aligned protein -> codon alignment."""
    out, pos = [], 0
    for aa in aln_prot:
        if aa == "-":
            out.append("---")
        else:
            if pos + 3 > len(nt):
                return None
            out.append(nt[pos:pos+3]); pos += 3
    return "".join(out)

# --- write paired fastas: protein (stats) + codon (model), keyed {patient}|{gene} ---
def dump(path, msa):
    with open(path, "w") as f:
        for acc, s in msa.items():
            p = acc2pat.get(acc)
            if p:
                f.write(f">{p}|{path.split('_')[0]}\n{s}\n")

# protein MSAs (for stats) — restrict to patients present in BOTH genes
gag_prot_msa = {a: s for a, s in gag_msa.items() if a in prot_msa}
prot_prot_msa = {a: s for a, s in prot_msa.items() if a in gag_msa}
shared = sorted(set(gag_prot_msa) & set(prot_prot_msa))
print("shared patients:", len(shared))

with open("gag_protein.fasta", "w") as f:
    for a in shared:
        f.write(f">{acc2pat[a]}|gag\n{gag_prot_msa[a]}\n")
with open("prot_protein.fasta", "w") as f:
    for a in shared:
        f.write(f">{acc2pat[a]}|prot\n{prot_prot_msa[a]}\n")

# codon MSAs (for model)
ng = np = 0
with open("gag_codon.fasta", "w") as fg, open("prot_codon.fasta", "w") as fp:
    for a in shared:
        cg = thread(gag_nt[a], gag_prot_msa[a])
        cp = thread(prot_nt[a], prot_prot_msa[a])
        if cg and cp:
            fg.write(f">{acc2pat[a]}|gag\n{cg}\n"); ng += 1
            fp.write(f">{acc2pat[a]}|prot\n{cp}\n"); np += 1
print(f"codon MSAs: gag={ng} prot={np}")
