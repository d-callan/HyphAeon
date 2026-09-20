"""
symbaeon/pairing.py
-------------------
Pairing layer: load two MSAs plus a pairing manifest and align them on a
common *individual* axis.

Unlike HyphAeon's within-alignment coselection (which shares a taxon axis and
tree), host and pathogen genes live in different alignments. Pairing maps
each sequence to its source individual (sample_id) via the manifest, then
expands each individual into its allele set:

  host:      {h1, h2}   diploid pseudo-haplotypes (unphased)
  parasite:  {c1, c2}   pseudo-clones from mixed infections (haploid if single)

Sequence headers / manifest seq_id format:  {sample_id}|{gene}|{tag}
"""

import csv
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from Bio import SeqIO


@dataclass
class PairedSide:
    """One side (host or parasite) of a paired alignment.

    Attributes
    ----------
    name : str
        Label, e.g. 'host' or 'parasite'.
    seqs : dict
        seq_id -> sequence string (uppercase).
    seq_to_sample : dict
        seq_id -> sample_id (from manifest, or parsed from header).
    sample_to_seqs : dict
        sample_id -> [seq_id, ...] preserving FASTA order.
    length : int
        Alignment length (sites). All sequences must share it.
    ref_id : str or None
        seq_id of the reference sequence (tag == 'ref'), if present.
    """
    name: str
    seqs: Dict[str, str] = field(default_factory=dict)
    seq_to_sample: Dict[str, str] = field(default_factory=dict)
    sample_to_seqs: Dict[str, List[str]] = field(default_factory=dict)
    length: int = 0
    ref_id: Optional[str] = None

    def samples(self) -> List[str]:
        return list(self.sample_to_seqs.keys())


def load_manifest(path: str) -> Dict[str, str]:
    """Load pairing manifest TSV -> {seq_id: sample_id}.

    Expects columns including 'seq_id' and 'sample_id' (as emitted by
    reconstruct_paired_seqs.py). Extra columns are ignored.
    """
    mapping = {}
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        if "seq_id" not in reader.fieldnames or "sample_id" not in reader.fieldnames:
            raise ValueError(
                f"Manifest {path} must contain 'seq_id' and 'sample_id' columns; "
                f"found {reader.fieldnames}"
            )
        for row in reader:
            mapping[row["seq_id"]] = row["sample_id"]
    return mapping


def _sample_from_header(seq_id: str) -> str:
    """Fallback: sample_id is the first |-delimited field of the header."""
    return seq_id.split("|")[0]


def load_paired_alignment(
    fasta_path: str,
    name: str,
    manifest: Optional[Dict[str, str]] = None,
) -> PairedSide:
    """Load a FASTA MSA into a PairedSide.

    seq_id -> sample_id comes from the manifest when provided, else parsed
    from the header (first |-field). The sequence tagged 'ref' (header third
    field) is recorded as ref_id and excluded from the individual axis.
    """
    side = PairedSide(name=name)
    for rec in SeqIO.parse(fasta_path, "fasta"):
        seq_id = rec.id
        seq = str(rec.seq).upper()
        side.seqs[seq_id] = seq
        if side.length == 0:
            side.length = len(seq)
        elif len(seq) != side.length:
            raise ValueError(
                f"{name} alignment {fasta_path}: sequence {seq_id} has length "
                f"{len(seq)} != {side.length}; MSAs must be aligned."
            )

        tag = seq_id.split("|")[-1] if "|" in seq_id else ""
        if tag == "ref":
            side.ref_id = seq_id
            continue

        sample_id = manifest.get(seq_id) if manifest else None
        if sample_id is None:
            sample_id = _sample_from_header(seq_id)
        side.seq_to_sample[seq_id] = sample_id
        side.sample_to_seqs.setdefault(sample_id, []).append(seq_id)
    return side


def build_individual_axis(host: PairedSide, parasite: PairedSide) -> List[str]:
    """Return the sorted intersection of sample_ids present on both sides."""
    shared = sorted(set(host.sample_to_seqs) & set(parasite.sample_to_seqs))
    return shared


def site_allele_sets(
    side: PairedSide, site: int, individuals: List[str]
) -> List[set]:
    """For a 0-based site, return the set of observed alleles per individual.

    Alleles are single characters from the aligned sequences. 'N', '-', and
    other non-ACGT characters are treated as missing and excluded from the
    set; an individual whose set is empty is fully missing at this site.
    """
    out = []
    for ind in individuals:
        alleles = set()
        for seq_id in side.sample_to_seqs.get(ind, []):
            ch = side.seqs[seq_id][site]
            if ch in "ACGT":
                alleles.add(ch)
        out.append(alleles)
    return out
