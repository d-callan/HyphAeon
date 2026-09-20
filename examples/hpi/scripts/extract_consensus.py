#!/usr/bin/env python3
"""Extract per-sample consensus coding sequences from ENA FASTQ files.

Pipeline: download FASTQ → align to genomic mini-reference → call consensus → extract CDS → write FASTA.
No variant calling — just majority-base consensus at each position.

Usage:
    python extract_consensus.py --accessions accessions.txt --mini-ref mini_reference_genomic.fasta --cds-map cds_map.json --output-dir results/

Accessions file format (one per line):
    ERR10153096
    ERR10153110
    ...
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pysam
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord


def get_ena_fastq_urls(run_accession: str) -> list[str]:
    """Query ENA filereport API for FASTQ download URLs for a run accession."""
    import requests

    url = (
        f"https://www.ebi.ac.uk/ena/portal/api/filereport?"
        f"accession={run_accession}&result=read_run"
        f"&fields=run_accession,fastq_ftp,fastq_bytes&format=tsv"
    )
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    lines = resp.text.strip().split("\n")
    if len(lines) < 2:
        return []
    fields = lines[1].split("\t")
    ftp_paths = fields[1].split(";")
    urls = []
    for p in ftp_paths:
        p = p.strip()
        if not p:
            continue
        if p.startswith("ftp://"):
            p = "http://" + p[6:]
        elif not p.startswith("http"):
            p = "http://" + p
        urls.append(p)
    return urls


def download_fastq(urls: list[str], dest_dir: str) -> list[str]:
    """Download FASTQ files to dest_dir using curl. Returns list of local paths."""
    local_paths = []
    for url in urls:
        filename = url.split("/")[-1]
        dest = os.path.join(dest_dir, filename)
        print(f"  Downloading {filename}...", flush=True)
        subprocess.run(
            ["curl", "-L", "-s", "--retry", "3", "-o", dest, url],
            check=True,
        )
        size_mb = os.path.getsize(dest) / 1e6
        print(f"    Done: {size_mb:.1f} MB", flush=True)
        local_paths.append(dest)
    return local_paths


def align_to_reference(fastq_paths: list[str], ref_index: str, output_bam: str, threads: int = 4):
    """Align FASTQs to reference using bwa mem, output sorted BAM."""
    sam_file = output_bam.replace(".bam", ".sam")

    cmd = ["bwa", "mem", "-t", str(threads), ref_index] + fastq_paths
    with open(sam_file, "w") as sam_out:
        subprocess.run(cmd, stdout=sam_out, stderr=subprocess.PIPE, check=True)

    subprocess.run(["samtools", "sort", "-o", output_bam, "-@", str(threads), sam_file], check=True)
    subprocess.run(["samtools", "index", output_bam], check=True)
    os.remove(sam_file)


def call_consensus(bam_path: str, ref_fasta: str, min_depth: int = 5, min_baseq: int = 20) -> dict:
    """Call majority-base consensus for each reference sequence in the BAM.

    Returns dict of {ref_name: consensus_sequence}.
    """
    samfile = pysam.AlignmentFile(bam_path, "rb")
    ref_names = samfile.references

    consensus_seqs = {}
    for ref_name in ref_names:
        ref_len = samfile.get_reference_length(ref_name)
        consensus = list("N" * ref_len)

        for col in samfile.pileup(ref_name, min_base_quality=min_baseq):
            pos = col.reference_pos
            if pos < 0 or pos >= ref_len:
                continue
            if col.n < min_depth:
                continue
            bases = {}
            for read in col.pileups:
                if read.is_del or read.is_refskip:
                    continue
                base = read.query_base
                if base is None:
                    continue
                base = base.upper()
                bases[base] = bases.get(base, 0) + 1
            if not bases:
                continue
            best_base = max(bases, key=bases.get)
            consensus[pos] = best_base

        consensus_seqs[ref_name] = "".join(consensus)

    samfile.close()
    return consensus_seqs


def extract_cds(consensus_seq: str, cds_exons: list[dict]) -> str:
    """Extract CDS from genomic consensus using exon coordinates."""
    cds_parts = []
    for exon in cds_exons:
        cds_parts.append(consensus_seq[exon["start"]:exon["end"]])
    return "".join(cds_parts)


def translate_check(seq: str) -> tuple[str, bool]:
    """Translate nucleotide sequence, check for premature stops.

    Returns (amino_acid_seq, has_premature_stop).
    """
    coding = Seq(seq)
    aa = str(coding.translate())
    if aa.endswith("*"):
        aa = aa[:-1]
    has_premature = "*" in aa
    return aa, has_premature


def write_consensus_fasta(consensus_seqs: dict, cds_map: dict, sample_id: str, output_dir: str):
    """Write CDS sequences extracted from genomic consensus to FASTA files."""
    for gene_name, genomic_seq in consensus_seqs.items():
        if gene_name not in cds_map:
            print(f"  WARNING: {gene_name} not in CDS map, skipping")
            continue

        gene_info = cds_map[gene_name]
        cds_seq = extract_cds(genomic_seq, gene_info["cds_exons"])

        # Stats
        genomic_coverage = sum(1 for c in genomic_seq if c != "N") / len(genomic_seq) if len(genomic_seq) > 0 else 0
        cds_coverage = sum(1 for c in cds_seq if c != "N") / len(cds_seq) if len(cds_seq) > 0 else 0
        aa, has_stop = translate_check(cds_seq)

        outfile = os.path.join(output_dir, f"{gene_name}_{sample_id}.fasta")
        record = SeqRecord(
            Seq(cds_seq),
            id=f"{sample_id}|{gene_name}",
            description=f"cds_len={len(cds_seq)} genomic_cov={genomic_coverage:.1%} cds_cov={cds_coverage:.1%} premature_stop={has_stop}"
        )
        with open(outfile, "w") as out:
            SeqIO.write(record, out, "fasta")

        print(f"  {gene_name}: CDS={len(cds_seq)}bp, genomic_cov={genomic_coverage:.1%}, cds_cov={cds_coverage:.1%}, premature_stop={has_stop}", flush=True)


def process_sample(run_accession: str, ref_fasta: str, ref_index: str, cds_map: dict, output_dir: str, threads: int = 4):
    """Process a single ENA run accession: download → align → consensus → extract CDS → write."""
    print(f"\n=== Processing {run_accession} ===", flush=True)

    with tempfile.TemporaryDirectory(prefix=f"ena_{run_accession}_") as tmp_dir:
        # Download FASTQs
        urls = get_ena_fastq_urls(run_accession)
        if not urls:
            print(f"  ERROR: No FASTQ URLs found for {run_accession}")
            return None

        fastq_paths = download_fastq(urls, tmp_dir)
        print(f"  Downloaded {len(fastq_paths)} files", flush=True)

        # Align to genomic mini-reference
        bam_path = os.path.join(output_dir, f"{run_accession}.bam")
        align_to_reference(fastq_paths, ref_index, bam_path, threads)

        # Alignment stats
        flagstat = subprocess.run(
            ["samtools", "flagstat", bam_path], capture_output=True, text=True, check=True
        )
        for line in flagstat.stdout.split("\n"):
            if "mapped" in line.lower() and "%" not in line:
                print(f"  {line.strip()}", flush=True)

        # Call consensus on genomic reference
        consensus_seqs = call_consensus(bam_path, ref_fasta)

        # Extract CDS and write
        write_consensus_fasta(consensus_seqs, cds_map, run_accession, output_dir)

        # Clean up BAM
        os.remove(bam_path)
        bam_bai = bam_path + ".bai"
        if os.path.exists(bam_bai):
            os.remove(bam_bai)

    return run_accession


def main():
    parser = argparse.ArgumentParser(description="Extract consensus CDS from ENA FASTQs")
    parser.add_argument("--accessions", required=True, help="File with one ENA run accession per line")
    parser.add_argument("--mini-ref", required=True, help="Genomic mini-reference FASTA")
    parser.add_argument("--cds-map", required=True, help="JSON file mapping gene names to CDS exon coordinates")
    parser.add_argument("--output-dir", required=True, help="Output directory for consensus sequences")
    parser.add_argument("--threads", type=int, default=4, help="Threads for alignment")
    parser.add_argument("--min-depth", type=int, default=5, help="Min depth for consensus calling")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load CDS map
    with open(args.cds_map) as f:
        cds_map = json.load(f)
    print(f"Loaded CDS map: {list(cds_map.keys())}", flush=True)

    # Index the reference
    print(f"Indexing mini-reference: {args.mini_ref}", flush=True)
    subprocess.run(["bwa", "index", args.mini_ref], check=True)
    subprocess.run(["samtools", "faidx", args.mini_ref], check=True)

    # Read accessions
    with open(args.accessions) as f:
        accessions = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    print(f"Processing {len(accessions)} accessions...", flush=True)

    results = []
    for acc in accessions:
        try:
            result = process_sample(acc, args.mini_ref, args.mini_ref, cds_map, args.output_dir, args.threads)
            if result:
                results.append(result)
        except Exception as e:
            print(f"  ERROR for {acc}: {e}", flush=True)
            import traceback
            traceback.print_exc()
            continue

    print(f"\nDone. {len(results)}/{len(accessions)} samples processed successfully.")
    print(f"Consensus sequences in: {args.output_dir}/")

    # Summary
    fasta_files = list(Path(args.output_dir).glob("*.fasta"))
    print(f"Total FASTA files: {len(fasta_files)}")
    for gene in cds_map.keys():
        gene_files = [f for f in fasta_files if f.name.startswith(gene + "_")]
        print(f"  {gene}: {len(gene_files)} samples")


if __name__ == "__main__":
    main()
