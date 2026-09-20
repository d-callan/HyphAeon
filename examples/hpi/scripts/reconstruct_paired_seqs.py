#!/usr/bin/env python3
"""
reconstruct_paired_seqs.py

Rebuild per-individual CDS haplotype sequences for host and parasite genes
from the Band et al. (2022) joint genotype dataset (MalariaGEN hspf release,
Zenodo doi:10.5281/zenodo.4973477).

Data model
----------
Host (human, bgen, diploid):
    Emit two pseudo-haplotypes h1/h2 per sample. Genotypes are unphased, so
    allele1 is assigned to h1 and allele2 to h2 at every heterozygous site.
    This preserves both alleles present in the individual but does NOT recover
    true phase across sites. Homozygous sites give identical h1/h2 bases.

Parasite (P. falciparum, VCF, called "as if diploid"):
    Homozygous calls -> single sequence c1.
    Heterozygous calls -> emit c1 (allele1) and c2 (allele2) as pseudo-clones.
    Real multiplicity of infection can exceed 2 and phase across sites is
    unknown, so c1/c2 preserve allele content, not true clone sequences.
    n_het_sites in the manifest flags likely mixed infections.

Shared rules:
    Missing genotypes -> 'N' at that CDS position (both haplotypes).
    Indels and multiallelic sites are skipped and logged (they would break
    the CDS frame); see skipped_variants.tsv.
    Only variants falling inside CDS exons are applied; intron/UTR variants
    in the bgen (it covers gene +/- 2kb) are ignored for CDS output.

Outputs
-------
<outdir>/<gene>.fasta          headers: >{sample_id}|{gene}|{tag}
                               plus a >REFERENCE|{gene}|ref record first
<outdir>/pairing_manifest.tsv  seq_id, sample_id, species, gene, tag,
                               n_alt, n_missing, n_het_sites
<outdir>/skipped_variants.tsv  gene, chrom, pos, reason
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import pysam
from bgen_reader import open_bgen
from Bio import SeqIO
from Bio.Seq import Seq

# ---------------------------------------------------------------------------
# Gene configuration.
#
# cds_exons are GENOMIC coordinates (1-based, inclusive) listed in
# TRANSCRIPTION order (i.e. for minus-strand genes, descending genomic
# coordinate). Verified against:
#   GYPA   : Ensembl GRCh37 ENST00000360771 / ENSP00000354003 (150 aa + stop)
#   EBA175 : NCBI NC_004328 CDS join, XM_001349171.2 (4509 bp)
#   PfRH5  : cds_map.json built earlier (1581 bp), minus strand on Pf3D7_04_v3
# ---------------------------------------------------------------------------

GENES = {
    "GYPA": {
        "species": "human",
        "chrom": "04",
        "strand": -1,
        "cds_exons": [
            (145061752, 145061788),
            (145041643, 145041741),
            (145040839, 145040934),
            (145039867, 145039905),
            (145038007, 145038092),
            (145035842, 145035920),
            (145032547, 145032563),
        ],
        "ref_fasta": "GYPA_cds_grch37.fa",
    },
    "EBA175": {
        "species": "parasite",
        "chrom": "Pf3D7_07_v3",
        "strand": 1,
        "cds_exons": [
            (1358055, 1362318),
            (1362444, 1362522),
            (1362624, 1362706),
            (1362847, 1362929),
        ],
        "ref_fasta": "EBA175_cds.fa",
    },
    "PfRH5": {
        "species": "parasite",
        "chrom": "Pf3D7_04_v3",
        "strand": -1,
        "cds_exons": [
            (1084087, 1084150),
            (1082363, 1083879),
        ],
        "ref_fasta": "PfRH5_cds.fa",
    },
    # ------------------------------------------------------------------
    # Expanded panel (coords via scripts/fetch_cds_exons.py, Ensembl GRCh37
    # canonical transcript for human; Ensembl/PlasmoDB for Pf3D7).
    # Human chrom names are zero-padded to match the bgen ("04","01",...).
    # ------------------------------------------------------------------
    "GYPB": {
        "species": "human", "chrom": "04", "strand": -1,
        "cds_exons": [
            (144940404, 144940440), (144922338, 144922436),
            (144920564, 144920602), (144918693, 144918787),
            (144917452, 144917457),
        ],
        "ref_fasta": "GYPB_cds.fa",
    },
    "GYPE": {
        "species": "human", "chrom": "04", "strand": -1,
        "cds_exons": [
            (144826624, 144826660), (144801564, 144801662),
            (144797908, 144798008),
        ],
        "ref_fasta": "GYPE_cds.fa",
    },
    "DARC": {  # ACKR1, Duffy antigen (P. vivax receptor)
        "species": "human", "chrom": "01", "strand": 1,
        "cds_exons": [(159175224, 159176240)],
        "ref_fasta": "DARC_cds.fa",
    },
    "CR1": {  # complement receptor 1, PfRH4 ligand (canonical ENST00000367049)
        "species": "human", "chrom": "01", "strand": 1,
        "cds_exons": [
            (207669613, 207669733), (207679249, 207679428),
            (207680059, 207680158), (207684914, 207684999),
            (207696956, 207697354), (207700098, 207700274),
            (207700366, 207700468), (207702061, 207702146),
            (207705525, 207705743), (207706754, 207706933),
            (207707542, 207707641), (207713326, 207713411),
            (207715527, 207715925), (207718653, 207718829),
            (207718921, 207719023), (207720616, 207720701),
            (207724080, 207724298), (207725309, 207725488),
            (207726097, 207726196), (207731883, 207731968),
            (207734084, 207734482), (207737209, 207737385),
            (207737477, 207737579), (207739183, 207739268),
            (207741169, 207741387), (207742404, 207742583),
            (207743192, 207743291), (207748940, 207749025),
            (207751150, 207751548), (207753585, 207753761),
            (207753856, 207753958), (207755263, 207755348),
            (207757994, 207758221), (207760731, 207760910),
            (207762020, 207762119), (207780596, 207780681),
            (207782635, 207783033), (207785022, 207785198),
            (207785284, 207785386), (207787749, 207787834),
            (207789920, 207790147), (207791416, 207791592),
            (207793225, 207793410), (207795318, 207795341),
            (207796338, 207796413), (207803862, 207803966),
            (207812742, 207812754),
        ],
        "ref_fasta": "CR1_cds.fa",
    },
    # Parasite invasion ligands (Pf3D7 ids; ref fasta named by gene id)
    "EBA140": {  # BAEBL, binds GYPC
        "species": "parasite", "chrom": "Pf3D7_13_v3", "strand": -1,
        "cds_exons": [(79441, 79470), (79167, 79328), (78614, 79165)],
        "ref_fasta": "PF3D7_1301300_cds.fa",
    },
    "EBA181": {  # EBL1
        "species": "parasite", "chrom": "Pf3D7_04_v3", "strand": 1,
        "cds_exons": [(155877, 155948), (156067, 156984)],
        "ref_fasta": "PF3D7_0402500_cds.fa",
    },
    "PfRH1": {
        "species": "parasite", "chrom": "Pf3D7_04_v3", "strand": 1,
        "cds_exons": [(137640, 137694), (137793, 146653)],
        "ref_fasta": "PF3D7_0402300_cds.fa",
    },
    "PfRH2a": {
        "species": "parasite", "chrom": "Pf3D7_13_v3", "strand": 1,
        "cds_exons": [(1440779, 1440836), (1441051, 1450385)],
        "ref_fasta": "PF3D7_1335400_cds.fa",
    },
    "PfRH2b": {
        "species": "parasite", "chrom": "Pf3D7_13_v3", "strand": -1,
        "cds_exons": [(1438795, 1438852), (1428874, 1438580)],
        "ref_fasta": "PF3D7_1335300_cds.fa",
    },
    "PfRH4": {  # binds CR1
        "species": "parasite", "chrom": "Pf3D7_04_v3", "strand": -1,
        "cds_exons": [(1091223, 1091277), (1085979, 1091074)],
        "ref_fasta": "PF3D7_0424200_cds.fa",
    },
}

GT_CALL_THRESHOLD = 0.9  # min genotype probability to call from bgen probs


def build_pos_map(gene):
    """Return dict mapping genomic position (1-based) -> CDS position (0-based)."""
    pos_map = {}
    cds_offset = 0
    for start, end in gene["cds_exons"]:
        for g in range(start, end + 1):
            if gene["strand"] == 1:
                pos_map[g] = cds_offset + (g - start)
            else:
                pos_map[g] = cds_offset + (end - g)
        cds_offset += end - start + 1
    return pos_map


def maybe_complement(allele, gene):
    """Complement an allele if the gene is on the minus strand."""
    if gene["strand"] == -1:
        return str(Seq(allele).complement())
    return allele


def load_ref_cds(path):
    rec = SeqIO.read(str(path), "fasta")
    return str(rec.seq).upper()


# ---------------------------------------------------------------------------
# Human side (bgen)
# ---------------------------------------------------------------------------

def reconstruct_human(gene, bgen_path, ref_cds, samples_wanted):
    """
    Returns dict sample_id -> (h1_list, h2_list, stats) for one gene.
    Also returns list of skipped-variant dicts.
    """
    pos_map = build_pos_map(gene)
    cds_len = len(ref_cds)
    bgen = open_bgen(str(bgen_path), verbose=False)
    bgen_samples = [str(s) for s in bgen.samples]

    # index of wanted samples within the bgen sample array
    if samples_wanted is None:
        wanted_idx = list(range(len(bgen_samples)))
    else:
        wanted_idx = [i for i, s in enumerate(bgen_samples) if s in samples_wanted]
    wanted_ids = [bgen_samples[i] for i in wanted_idx]

    # initialise both haplotypes to reference
    haps = {
        sid: {"h1": list(ref_cds), "h2": list(ref_cds),
              "n_alt": 0, "n_missing": 0, "n_het": 0}
        for sid in wanted_ids
    }
    skipped = []

    chroms = bgen.chromosomes
    positions = bgen.positions
    alleles = bgen.allele_ids  # array of "a1,a2" per variant
    n_var = len(bgen.ids)

    for vidx in range(n_var):
        if str(chroms[vidx]) != gene["chrom"]:
            continue
        gpos = int(positions[vidx])
        cds_pos = pos_map.get(gpos)
        if cds_pos is None:
            continue  # outside CDS exons (intron/UTR/flank)

        a1, a2 = alleles[vidx].split(",")
        if len(a1) != 1 or len(a2) != 1:
            skipped.append({"gene": "", "chrom": gene["chrom"], "pos": gpos,
                            "reason": f"indel/multibp ({a1}/{a2})"})
            continue

        # determine which allele matches the reference base at this CDS pos
        ref_base = ref_cds[cds_pos]
        c1 = maybe_complement(a1, gene)
        c2 = maybe_complement(a2, gene)
        if c1 == ref_base:
            alt_base = c2
        elif c2 == ref_base:
            alt_base = c1
        else:
            skipped.append({"gene": "", "chrom": gene["chrom"], "pos": gpos,
                            "reason": f"no allele matches ref {ref_base} ({a1}/{a2})"})
            continue

        gt = bgen.read(vidx)[:, 0, :]  # (n_samples, 3): p(a1a1),p(a1a2),p(a2a2)
        for i, sid in zip(wanted_idx, wanted_ids):
            p = gt[i]
            best = int(p.argmax())
            if p[best] < GT_CALL_THRESHOLD:
                haps[sid]["h1"][cds_pos] = "N"
                haps[sid]["h2"][cds_pos] = "N"
                haps[sid]["n_missing"] += 1
                continue
            # genotype classes: 0 = a1a1, 1 = a1a2, 2 = a2a2
            if best == 0:
                b1 = b2 = c1
            elif best == 2:
                b1 = b2 = c2
            else:
                b1, b2 = c1, c2
                haps[sid]["n_het"] += 1
            haps[sid]["h1"][cds_pos] = b1
            haps[sid]["h2"][cds_pos] = b2
            if b1 != ref_base:
                haps[sid]["n_alt"] += 1
            if b2 != ref_base:
                haps[sid]["n_alt"] += 1

    return haps, skipped


# ---------------------------------------------------------------------------
# Parasite side (VCF)
# ---------------------------------------------------------------------------

def reconstruct_parasite(gene, vcf_path, ref_cds, samples_wanted):
    """
    Returns dict sample_id -> {"c1": list, "c2": list|None, stats}.
    c2 is only created if the sample has >=1 het call in this gene.
    """
    pos_map = build_pos_map(gene)
    vcf = pysam.VariantFile(str(vcf_path))
    if samples_wanted is None:
        vcf_samples = list(vcf.header.samples)
    else:
        vcf_samples = [s for s in vcf.header.samples if s in samples_wanted]

    haps = {
        sid: {"c1": list(ref_cds), "c2": None,
              "n_alt": 0, "n_missing": 0, "n_het": 0}
        for sid in vcf_samples
    }
    skipped = []

    seen = set()
    for start, end in gene["cds_exons"]:
        for rec in vcf.fetch(gene["chrom"], start - 1, end):
            if rec.pos in seen:
                continue
            seen.add(rec.pos)
            cds_pos = pos_map.get(rec.pos)
            if cds_pos is None:
                continue
            if len(rec.alts) != 1 or len(rec.ref) != 1 or len(rec.alts[0]) != 1:
                skipped.append({"gene": "", "chrom": gene["chrom"], "pos": rec.pos,
                                "reason": f"indel/multiallelic ({rec.ref}/{','.join(rec.alts)})"})
                continue

            ref_base = ref_cds[cds_pos]
            ref_allele = maybe_complement(rec.ref, gene)
            alt_allele = maybe_complement(rec.alts[0], gene)
            if ref_allele != ref_base:
                skipped.append({"gene": "", "chrom": gene["chrom"], "pos": rec.pos,
                                "reason": f"VCF ref {ref_allele} != CDS ref {ref_base}"})
                continue

            for sid in vcf_samples:
                gt = rec.samples[sid]["GT"]
                if gt is None or None in gt:
                    haps[sid]["c1"][cds_pos] = "N"
                    if haps[sid]["c2"] is not None:
                        haps[sid]["c2"][cds_pos] = "N"
                    haps[sid]["n_missing"] += 1
                    continue
                a, b = gt[0], gt[1]
                if a == b:
                    base = ref_base if a == 0 else alt_allele
                    haps[sid]["c1"][cds_pos] = base
                    if haps[sid]["c2"] is not None:
                        haps[sid]["c2"][cds_pos] = base
                    if a != 0:
                        haps[sid]["n_alt"] += 1
                else:
                    # het -> pseudo-clones: c1 gets allele a, c2 gets allele b
                    if haps[sid]["c2"] is None:
                        haps[sid]["c2"] = list(haps[sid]["c1"])
                    ba = ref_base if a == 0 else alt_allele
                    bb = ref_base if b == 0 else alt_allele
                    haps[sid]["c1"][cds_pos] = ba
                    haps[sid]["c2"][cds_pos] = bb
                    haps[sid]["n_het"] += 1
                    haps[sid]["n_alt"] += (ba != ref_base) + (bb != ref_base)

    return haps, skipped


# ---------------------------------------------------------------------------
# Host structural-variant genotypes (glycophorin DEL1/DUP1/DUP4)
#
# These cannot be applied to the CDS as base substitutions, so they are
# emitted as a per-sample diploid genotype table that joins to the pairing
# manifest on sample_id. Per the release README, the "-mid" imputed calls
# are the recommended versions; we emit all three breakpoints anyway.
# ---------------------------------------------------------------------------

SV_IDS = ("DEL1-start", "DEL1-mid", "DEL1-end",
          "DUP1-start", "DUP1-mid", "DUP1-end",
          "DUP4-start", "DUP4-mid")


def extract_host_svs(bgen_path, samples_wanted, out_path):
    """Write per-sample diploid genotypes for the glycophorin SVs."""
    bgen = open_bgen(str(bgen_path), verbose=False)
    bgen_samples = [str(s) for s in bgen.samples]
    if samples_wanted is None:
        wanted = list(range(len(bgen_samples)))
    else:
        wanted = [i for i, s in enumerate(bgen_samples) if s in samples_wanted]

    rsids = [str(r) for r in bgen.rsids]  # SVs are named in rsid, not id
    sv_idx = {sv: rsids.index(sv) for sv in SV_IDS if sv in rsids}
    missing = [sv for sv in SV_IDS if sv not in sv_idx]
    if missing:
        print(f"  warning: SV ids not found in bgen: {missing}", flush=True)

    with open(out_path, "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["sample_id"] + list(sv_idx))
        probs = {sv: bgen.read(i)[:, 0, :] for sv, i in sv_idx.items()}
        alleles = {sv: bgen.allele_ids[i].split(",") for sv, i in sv_idx.items()}
        for i in wanted:
            row = [bgen_samples[i]]
            for sv in sv_idx:
                p = probs[sv][i]
                a1, a2 = alleles[sv]
                if p.max() < GT_CALL_THRESHOLD:
                    row.append("NA")
                    continue
                row.append({0: f"{a1}{a1}", 1: f"{a1}{a2}", 2: f"{a2}{a2}"}[int(p.argmax())])
            w.writerow(row)
    print(f"SV genotypes: {out_path} ({len(wanted)} samples, {len(sv_idx)} SVs)",
          flush=True)


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--release-dir", required=True,
                    help="Path to MalariaGEN_hspf_r1_release/")
    ap.add_argument("--ref-dir", required=True,
                    help="Directory containing reference CDS FASTAs")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--genes", nargs="+", default=list(GENES),
                    choices=list(GENES))
    ap.add_argument("--samples", default=None,
                    help="Optional file of sample IDs to restrict to")
    args = ap.parse_args()

    release = Path(args.release_dir)
    refdir = Path(args.ref_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    bgen_path = release / "hs" / "MalariaGEN_hspf_r1_discovery_hs_genotypes.bgen"
    vcf_path = release / "pf" / "MalariaGEN_hspf_r1_discovery_pf_genotypes.vcf.gz"

    samples_wanted = None
    if args.samples:
        samples_wanted = set(Path(args.samples).read_text().split())

    manifest_rows = []
    all_skipped = []

    for gname in args.genes:
        gene = GENES[gname]
        ref_cds = load_ref_cds(refdir / gene["ref_fasta"])
        expected = sum(e - s + 1 for s, e in gene["cds_exons"])
        assert len(ref_cds) == expected, \
            f"{gname}: ref CDS {len(ref_cds)}bp != exon sum {expected}bp"

        if gene["species"] == "human":
            haps, skipped = reconstruct_human(gene, bgen_path, ref_cds, samples_wanted)
            tags = ("h1", "h2")
        else:
            haps, skipped = reconstruct_parasite(gene, vcf_path, ref_cds, samples_wanted)
            tags = ("c1", "c2")

        for s in skipped:
            s["gene"] = gname
        all_skipped.extend(skipped)

        fa = outdir / f"{gname}.fasta"
        with open(fa, "w") as fh:
            fh.write(f">REFERENCE|{gname}|ref\n")
            for i in range(0, len(ref_cds), 60):
                fh.write(ref_cds[i:i + 60] + "\n")
            for sid, d in sorted(haps.items()):
                for tag in tags:
                    seq = d.get(tag)
                    if seq is None:
                        continue
                    seq_id = f"{sid}|{gname}|{tag}"
                    s = "".join(seq)
                    fh.write(f">{seq_id}\n")
                    for i in range(0, len(s), 60):
                        fh.write(s[i:i + 60] + "\n")
                    manifest_rows.append({
                        "seq_id": seq_id, "sample_id": sid,
                        "species": gene["species"], "gene": gname, "tag": tag,
                        "n_alt": d["n_alt"], "n_missing": d["n_missing"],
                        "n_het_sites": d["n_het"],
                    })
        n_het_samples = sum(1 for d in haps.values() if d["n_het"] > 0)
        print(f"{gname}: {len(haps)} samples -> {fa} "
              f"({n_het_samples} samples with het/mixed calls)", flush=True)

    with open(outdir / "pairing_manifest.tsv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(manifest_rows[0]), delimiter="\t")
        w.writeheader()
        w.writerows(manifest_rows)

    with open(outdir / "skipped_variants.tsv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["gene", "chrom", "pos", "reason"],
                           delimiter="\t")
        w.writeheader()
        w.writerows(all_skipped)

    if any(GENES[g]["species"] == "human" for g in args.genes):
        extract_host_svs(bgen_path, samples_wanted,
                         outdir / "host_sv_genotypes.tsv")

    print(f"Manifest: {outdir/'pairing_manifest.tsv'} "
          f"({len(manifest_rows)} sequences)", flush=True)
    print(f"Skipped variants: {len(all_skipped)}", flush=True)


if __name__ == "__main__":
    main()
