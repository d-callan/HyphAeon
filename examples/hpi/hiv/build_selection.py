#!/usr/bin/env python3
"""Build a paired host-HLA x HIV-Gag dataset from the LANL background table.

Input: lanl_gag_hla_table.tsv  (LANL "Save Background Info" download, tab-delimited)
Outputs:
  accessions.txt        one representative GenBank accession per patient
  hla_covariates.tsv    patient x HLA-allele binary carrier matrix (for symbaeon host_cov)
  patient_map.tsv       patient -> chosen accession + subtype + country + raw HLA string

We keep subtype B only (best-characterized HLA-escape ground truth) and pick a
single representative sequence per patient (longest = most complete Gag coverage).
"""
import csv, re, collections, sys

TABLE = "lanl_gag_hla_table.tsv"
SUBTYPES = {"B", "C"}   # the two well-characterized subtypes for HLA escape

def norm_hla(tok):
    """Normalize an HLA allele token to a canonical 'LOCUS*GROUP' form.

    Handles 'A*31:01', 'A3101', 'A*0201', 'Cw*0302', 'C12:02', 'B48', etc.
    We collapse to locus + 2-digit allele group (protein-level escape is usually
    ascribed at the allele-group / 2-digit level, and 2-digit maximizes carrier
    counts for power). Returns e.g. 'A31', 'B57', 'C07'.
    """
    t = tok.strip().upper()
    if not t or t in ("-", "NULL", "NONE", "?"):
        return None
    t = t.replace("HLA-", "").replace("CW", "C").replace("*", "").replace(":", "")
    m = re.match(r"^([ABC])(\d{2})", t)   # locus letter + first 2 digits = group
    if not m:
        return None
    return m.group(1) + m.group(2)

def parse_hla(s):
    out = set()
    for tok in re.split(r"[,\s;/]+", s):
        a = norm_hla(tok)
        if a:
            out.add(a)
    return out

def main():
    rows = list(csv.reader(open(TABLE), delimiter="\t"))
    hi = [i for i, r in enumerate(rows) if r and r[0].strip() == "#"][0]
    hdr = [c.strip() for c in rows[hi]]
    ix = {c: i for i, c in enumerate(hdr)}
    data = [r for r in rows[hi + 1:] if len(r) >= len(hdr)]

    sub = [r for r in data if r[ix["Subtype"]].strip() in SUBTYPES]

    # group sequences by patient; keep longest as representative
    bypat = collections.defaultdict(list)
    for r in sub:
        bypat[r[ix["Patient Code"]].strip()].append(r)

    chosen = {}   # patient -> row
    for pat, rs in bypat.items():
        def seqlen(r):
            try:
                return int(r[ix["Sequence Length"]].strip())
            except ValueError:
                return 0
        chosen[pat] = max(rs, key=seqlen)

    # build HLA carrier matrix
    pat_hla = {p: parse_hla(r[ix["HLA type"]]) for p, r in chosen.items()}
    alleles = sorted({a for s in pat_hla.values() for a in s})
    # keep alleles seen in >=3 patients (power) — drop singletons/rare
    freq = collections.Counter(a for s in pat_hla.values() for a in s)
    alleles = [a for a in alleles if freq[a] >= 3]

    pats = sorted(chosen)
    with open("accessions.txt", "w") as f:
        for p in pats:
            f.write(chosen[p][ix["Accession"]].strip() + "\n")

    with open("patient_map.tsv", "w") as f:
        f.write("patient\taccession\tsubtype\tcountry\thla_raw\n")
        for p in pats:
            r = chosen[p]
            f.write("\t".join([p, r[ix["Accession"]].strip(),
                               r[ix["Subtype"]].strip(),
                               r[ix["Country"]].strip(),
                               r[ix["HLA type"]].strip()]) + "\n")

    with open("hla_covariates.tsv", "w") as f:
        f.write("sample\t" + "\t".join(alleles) + "\n")
        for p in pats:
            f.write(p + "\t" + "\t".join("1" if a in pat_hla[p] else "0"
                                         for a in alleles) + "\n")

    # strata file: sample_id -> subtype (for symbaeon --strata)
    with open("strata.tsv", "w") as f:
        f.write("sample_id\tsubtype\n")
        for p in pats:
            f.write(p + "\t" + chosen[p][ix["Subtype"]].strip() + "\n")

    print(f"patients={len(pats)}  alleles(>=3 carriers)={len(alleles)}")
    print("top alleles:", freq.most_common(15))
    print("wrote accessions.txt, patient_map.tsv, hla_covariates.tsv")

if __name__ == "__main__":
    main()
