import os
import pandas as pd
from Bio import SeqIO
from collections import defaultdict

# Setup directories
OUT_DIR = "/home/danielle/Documents/veg/HyphAeon/examples/hpi/klebsiella"
os.makedirs(f"{OUT_DIR}/host", exist_ok=True)
os.makedirs(f"{OUT_DIR}/par", exist_ok=True)

# 1. Map PC to proteinID/prophageID
print("[*] Loading PCs...")
pcs = pd.read_csv("PROPHAGES_METADATA/pcs2proteins.tsv", sep="\t", header=None, names=["prophageID", "PC", "proteinID"])
# PCs of interest
TARGET_PCS = ["PC_00366", "PC_00371", "PC_00276"]
pc_map = pcs[pcs["PC"].isin(TARGET_PCS)]

# 2. Get KL2 genomes
print("[*] Loading KL2 genomes...")
bac_meta = pd.read_csv("BACTERIA_METADATA/bacteria_metadata.tsv", sep="\t")
kl2_genomes = set(bac_meta[bac_meta["K_locus"] == "KL2"]["genomeID"])

# 3. Identify pairs
print("[*] Identifying genome-phage pairs...")
# genomeID -> {PC -> proteinID}
pairs = defaultdict(dict)
for _, row in pc_map.iterrows():
    pid = row["prophageID"]
    # Find matching genomeID
    for g in kl2_genomes:
        if pid.startswith(g):
            pairs[g][row["PC"]] = (pid, row["proteinID"])
            break

# Filter to genomes that have at least one PC and are in KL2
final_genomes = sorted(list(pairs.keys()))
print(f"[*] Found {len(final_genomes)} genomes with KL2 and target PCs")

# 4. Extract host genes
HOST_GENES = ["wza", "wzb", "wzc", "wzx", "wzy"]
host_seqs = defaultdict(list) # gene -> [records]

print("[*] Extracting host genes...")
for g in final_genomes:
    fpath = f"BACTERIA_FASTA/CDS/{g}.fasta"
    if not os.path.exists(fpath): continue
    for rec in SeqIO.parse(fpath, "fasta"):
        for hg in HOST_GENES:
            if f"[gene={hg}]" in rec.description:
                # Rename record to just genomeID for pairing
                rec.id = g
                rec.description = ""
                host_seqs[hg].append(rec)

# 5. Extract par genes
par_seqs = defaultdict(list) # PC -> [records]

print("[*] Extracting par genes...")
for g, pc_data in pairs.items():
    for pc, (prophageID, proteinID) in pc_data.items():
        fpath = f"PROPHAGES_FASTA/CDS/{prophageID}.fasta"
        if not os.path.exists(fpath): continue
        for rec in SeqIO.parse(fpath, "fasta"):
            if proteinID in rec.id:
                # Rename record to just genomeID for pairing
                rec.id = g
                rec.description = ""
                par_seqs[pc].append(rec)

# 6. Save FASTA files (unaligned)
for hg, recs in host_seqs.items():
    if len(recs) > 10:
        SeqIO.write(recs, f"{OUT_DIR}/host/{hg}.fasta", "fasta")

for pc, recs in par_seqs.items():
    if len(recs) > 10:
        SeqIO.write(recs, f"{OUT_DIR}/par/{pc}.fasta", "fasta")

print("[*] Done. Files written to /home/danielle/Documents/veg/HyphAeon/examples/hpi/klebsiella/")
