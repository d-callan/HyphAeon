#!/usr/bin/env python
"""
fetch_cds_exons.py
------------------
Fetch canonical-transcript CDS exon coordinates for a gene and emit the
`GENES` config entry + reference CDS FASTA used by reconstruct_paired_seqs.py.

Human genes  -> Ensembl GRCh37 REST (grch37.rest.ensembl.org)
Parasite     -> Ensembl (rest.ensembl.org), species plasmodium_falciparum_3d7,
                looked up by Pf3D7 gene id (e.g. PF3D7_0424200)

Output (per gene):
  - a GENES dict entry printed to stdout (chrom, strand, cds_exons, ref_fasta)
  - a reference CDS FASTA written to --ref-dir/{GENE}_cds.fa

cds_exons are GENOMIC 1-based inclusive coords in TRANSCRIPTION order
(descending for minus-strand genes), clipped to the CDS (translation) bounds.
"""

import argparse
import json
import sys
import urllib.request

HUMAN = "https://grch37.rest.ensembl.org"
PARA = "https://rest.ensembl.org"


def _get(url):
    req = urllib.request.Request(url, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode())


def fetch_human(symbol):
    """Return (chrom, strand, cds_exons, cds_seq) for a human gene symbol."""
    data = _get(f"{HUMAN}/lookup/symbol/homo_sapiens/{symbol}?expand=1")
    chrom = data["seq_region_name"]
    strand = data["strand"]
    canon_id = data.get("canonical_transcript", "").split(".")[0]
    txs = data["Transcript"]
    # prefer canonical if it has a Translation, else first protein_coding with one
    tx = next((t for t in txs if t["id"] == canon_id and "Translation" in t), None)
    if tx is None:
        tx = next((t for t in txs if t.get("biotype") == "protein_coding"
                   and "Translation" in t), None)
    if tx is None:
        raise ValueError(f"{symbol}: no protein-coding transcript with a Translation")
    tr = tx["Translation"]
    cds_start, cds_end = tr["start"], tr["end"]  # genomic CDS bounds
    # clip each exon to CDS bounds, keep coding portion
    cds_exons = []
    for ex in tx["Exon"]:
        s, e = ex["start"], ex["end"]
        lo, hi = max(s, cds_start), min(e, cds_end)
        if lo <= hi:
            cds_exons.append((lo, hi))
    # transcription order: ascending for +, descending for -
    cds_exons.sort(key=lambda x: x[0], reverse=(strand == -1))
    return chrom, strand, cds_exons


def fetch_parasite(gene_id):
    """Return (chrom, strand, cds_exons) for a Pf3D7 gene id via Ensembl."""
    data = _get(f"{PARA}/lookup/id/{gene_id}?expand=1")
    chrom = data["seq_region_name"]
    strand = data["strand"]
    # pick the protein_coding transcript (usually only one)
    txs = [t for t in data.get("Transcript", []) if t.get("biotype") == "protein_coding"]
    tx = txs[0] if txs else data["Transcript"][0]
    tr = tx.get("Translation")
    if tr:
        cds_start, cds_end = tr["start"], tr["end"]
        cds_exons = []
        for ex in tx["Exon"]:
            s, e = ex["start"], ex["end"]
            lo, hi = max(s, cds_start), min(e, cds_end)
            if lo <= hi:
                cds_exons.append((lo, hi))
    else:
        cds_exons = [(ex["start"], ex["end"]) for ex in tx["Exon"]]
    cds_exons.sort(key=lambda x: x[0], reverse=(strand == -1))
    return chrom, strand, cds_exons


_COMP = str.maketrans("ACGTacgt", "TGCAtgca")


def _load_genome(path):
    """Load a genome FASTA -> {contig: seq}. Maps Pf3D7_{NN}_v3 <-> {NN}."""
    from Bio import SeqIO
    genome = {}
    for rec in SeqIO.parse(path, "fasta"):
        genome[rec.id] = str(rec.seq).upper()
    return genome


def _genome_contig(genome, chrom):
    """Resolve a VCF-style contig (Pf3D7_04_v3) to a genome FASTA id (4)."""
    if chrom in genome:
        return chrom
    if chrom.startswith("Pf3D7_"):
        num = chrom.split("_")[1].lstrip("0")
        if num in genome:
            return num
    return None


def fetch_cds_seq(chrom, cds_exons, strand, species, genome=None):
    """Return the spliced CDS sequence.

    If `genome` (dict contig->seq) is given, extract locally; else fetch via
    the Ensembl sequence endpoint (human only).
    """
    parts = []
    if genome is not None:
        contig = _genome_contig(genome, chrom)
        if contig is None:
            raise ValueError(f"contig {chrom} not in genome FASTA")
        gseq = genome[contig]
        for s, e in cds_exons:  # 1-based inclusive
            parts.append(gseq[s - 1:e])
    else:
        base = HUMAN if species == "human" else PARA
        sp = "homo_sapiens" if species == "human" else "plasmodium_falciparum_3d7"
        for s, e in cds_exons:
            d = _get(f"{base}/sequence/region/{sp}/{chrom}:{s}..{e}:1")
            parts.append(d["seq"])
    # exons are in transcription order; for - strand reverse-complement each
    if strand == -1:
        parts = [p.translate(_COMP)[::-1] for p in parts]
    return "".join(parts).upper()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("genes", nargs="+", help="Gene symbols (human) or Pf3D7 ids (parasite)")
    ap.add_argument("--species", choices=["human", "parasite"], required=True)
    ap.add_argument("--ref-dir", default=".", help="Where to write {GENE}_cds.fa")
    ap.add_argument("--genome", default=None,
                    help="Local genome FASTA for CDS extraction (parasite); avoids REST")
    ap.add_argument("--no-seq", action="store_true", help="Skip fetching CDS sequence")
    args = ap.parse_args()

    genome = _load_genome(args.genome) if args.genome else None

    for gene in args.genes:
        try:
            if args.species == "human":
                chrom, strand, exons = fetch_human(gene)
            else:
                chrom, strand, exons = fetch_parasite(gene)
        except Exception as e:
            print(f"# ERROR {gene}: {e}", file=sys.stderr)
            continue

        cds_len = sum(e - s + 1 for s, e in exons)
        if not args.no_seq:
            try:
                seq = fetch_cds_seq(chrom, exons, strand, args.species, genome)
                with open(f"{args.ref_dir}/{gene}_cds.fa", "w") as fh:
                    fh.write(f">{gene}_ref\n")
                    for i in range(0, len(seq), 60):
                        fh.write(seq[i:i + 60] + "\n")
                seq_note = f"  # wrote {args.ref_dir}/{gene}_cds.fa ({len(seq)} bp)"
            except Exception as e:
                seq_note = f"  # SEQ FETCH FAILED: {e}"
        else:
            seq_note = ""

        exon_str = ",\n            ".join(f"({s}, {e})" for s, e in exons)
        print(f'    "{gene}": {{')
        print(f'        "species": "{args.species}",')
        print(f'        "chrom": "{chrom}",')
        print(f'        "strand": {strand},')
        print(f'        "cds_exons": [\n            {exon_str}\n        ],')
        print(f'        "ref_fasta": "{gene}_cds.fa",')
        print(f'    }},  # cds_len={cds_len}{seq_note}')


if __name__ == "__main__":
    main()
