import os
import json
import requests
import subprocess
import gzip
import shutil
from Bio import SeqIO
from io import BytesIO

# Configuration
OUT_DIR = "/home/danielle/Documents/veg/HyphAeon/examples/hpi/vibrio"
os.makedirs(f"{OUT_DIR}/host", exist_ok=True)
os.makedirs(f"{OUT_DIR}/par", exist_ok=True)

# 1. Unambiguous pairs (from clinical surveillance data)
PAIRS = [
    ("BFS1003", "PFS375"), ("BFS1017", "PFS381"), ("BFS1021", "PFS386"),
    ("BFS1022", "PFS383"), ("BFS1026", "PFS385"), ("BFS1150", "PFS433"),
    ("BFS1155", "PFS437"), ("BFS1159", "PFS441"), ("BFS1160", "PFS442"),
    ("BFS1211", "PFS464"), ("BFS1213", "PFS466"), ("BFS1214", "PFS467"),
    ("BFS1215", "PFS468"), ("BFS1216", "PFS469"), ("BFS1217", "PFS470"),
    ("BFS1218", "PFS471"), ("BFS1221", "PFS474"), ("BFS1320", "PFS525"),
    ("BFS1335", "PFS543"), ("BFS1403", "PFS590"), ("BFS783", "PFS285"),
    ("BFS814", "PFS301"), ("BFS845", "PFS322"), ("BFS853", "PFS352")
]

# 2. Load GCA mapping
with open("assembly_mapping.json") as f:
    MAPPING = json.load(f)

def get_ftp_url(acc):
    url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=assembly&id={acc}&retmode=json"
    res = requests.get(url).json()
    uid = list(res['result']['uids'])[0]
    ftp_path = res['result'][uid]['ftppath']
    name = res['result'][uid]['assemblyname']
    acc = res['result'][uid]['assemblyaccession']
    return f"{ftp_path}/{acc}_{name}_genomic.fna.gz"

def extract_gene(asm_path, ref_path, out_fasta, tag):
    # Use minimap2 to find the gene
    cmd = ["minimap2", "-x", "asm5", asm_path, ref_path]
    res = subprocess.run(cmd, capture_output=True, text=True)
    # Parse PAF to get coords
    lines = res.stdout.strip().split('\n')
    if not lines or not lines[0]: return False
    cols = lines[0].split('\t')
    target_name = cols[5]
    start = int(cols[7])
    end = int(cols[8])
    strand = cols[4]
    
    # Extract from FASTA
    for rec in SeqIO.parse(asm_path, "fasta"):
        if rec.id == target_name:
            sub = rec.seq[start:end]
            if strand == '-':
                sub = sub.reverse_complement()
            new_rec = rec[start:end]
            new_rec.seq = sub
            new_rec.id = tag
            new_rec.description = f"extracted from {target_name}:{start}-{end} ({strand})"
            with open(out_fasta, "a") as f:
                SeqIO.write(new_rec, f, "fasta")
            return True
    return False

# 3. Process pairs
for bfs, pfs in PAIRS:
    print(f"[*] Processing {bfs} <-> {pfs}")
    try:
        if bfs not in MAPPING or pfs not in MAPPING:
            print(f"  [!] Missing mapping for {bfs} or {pfs}")
            continue
            
        host_acc = MAPPING[bfs]
        phage_acc = MAPPING[pfs]
        
        host_url = get_ftp_url(host_acc)
        phage_url = get_ftp_url(phage_acc)
        
        # Host side: rta
        print(f"  Downloading host assembly {host_acc}...")
        r = requests.get(host_url)
        with open("tmp_host.fna.gz", "wb") as f: f.write(r.content)
        with gzip.open("tmp_host.fna.gz", "rb") as f_in:
            with open("tmp_host.fna", "wb") as f_out: shutil.copyfileobj(f_in, f_out)
        if not extract_gene("tmp_host.fna", "rta_ref.fasta", f"{OUT_DIR}/host/rta.fasta", bfs):
            print(f"  [!] Failed to extract rta for {bfs}")
        
        # Phage side: tmp
        print(f"  Downloading phage assembly {phage_acc}...")
        r = requests.get(phage_url)
        with open("tmp_par.fna.gz", "wb") as f: f.write(r.content)
        with gzip.open("tmp_par.fna.gz", "rb") as f_in:
            with open("tmp_par.fna", "wb") as f_out: shutil.copyfileobj(f_in, f_out)
        if not extract_gene("tmp_par.fna", "tmp_ref.fasta", f"{OUT_DIR}/par/tmp.fasta", pfs):
            print(f"  [!] Failed to extract tmp for {pfs}")
            
        os.remove("tmp_host.fna.gz"); os.remove("tmp_host.fna")
        os.remove("tmp_par.fna.gz"); os.remove("tmp_par.fna")
            
    except Exception as e:
        print(f"  [!] Error: {e}")

print("[*] Done.")
