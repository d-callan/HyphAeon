#!/usr/bin/env python3
"""Fetch GenBank records for selected accessions and extract Gag protein.

Writes gag_proteins.fasta keyed by accession. Also fetches HXB2 (K03455) Gag
as the coordinate reference. Uses NCBI efetch in batches; polite delay between.
"""
import time, sys, urllib.request, urllib.parse
from Bio import SeqIO
from io import StringIO

ACCS = [l.strip() for l in open("accessions.txt") if l.strip()]
# include HXB2 reference genome
ALL = ACCS + ["K03455"]
OUT = "gag_proteins.fasta"
BATCH = 120
EMAIL = "brc@example.org"

def efetch_gb(ids):
    q = urllib.parse.urlencode({
        "db": "nuccore", "id": ",".join(ids),
        "rettype": "gb", "retmode": "text", "email": EMAIL, "tool": "symbaeon",
    })
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?" + q
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                return r.read().decode("utf-8", "replace")
        except Exception as e:
            sys.stderr.write(f"efetch retry {attempt}: {e}\n")
            time.sleep(3 * (attempt + 1))
    return ""

def gag_translation(rec):
    """Return the gag CDS translation (longest CDS whose gene/product is gag)."""
    best = None
    for f in rec.features:
        if f.type != "CDS":
            continue
        gene = "".join(f.qualifiers.get("gene", [])).lower()
        prod = "".join(f.qualifiers.get("product", [])).lower()
        if "gag" in gene or ("gag" in prod and "pol" not in prod):
            tr = f.qualifiers.get("translation", [None])[0]
            if tr and (best is None or len(tr) > len(best)):
                best = tr
    return best

found, missing = {}, []
for i in range(0, len(ALL), BATCH):
    chunk = ALL[i:i + BATCH]
    txt = efetch_gb(chunk)
    for rec in SeqIO.parse(StringIO(txt), "genbank"):
        acc = rec.annotations.get("accessions", [rec.id])[0]
        tr = gag_translation(rec)
        if tr:
            found[acc] = tr
    time.sleep(0.4)

for a in ALL:
    if a not in found:
        missing.append(a)

with open(OUT, "w") as f:
    for a in ALL:
        if a in found:
            f.write(f">{a}\n{found[a]}\n")

print(f"accessions requested={len(ALL)}  with gag translation={len(found)}  missing={len(missing)}")
if missing:
    print("missing:", missing[:20])
