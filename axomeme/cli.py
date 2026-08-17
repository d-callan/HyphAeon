"""
axomeme/cli.py
--------------
Command-line interface for AxoMEME inference and evaluation.
"""

import os
import sys
import time
import json
import argparse
import numpy as np
import pandas as pd
import scipy.stats as stats
import torch

from .model import PhyloAxialTransformer
from .dataset import load_alignment_and_tree

_REPO_WEIGHTS = os.path.join(os.path.dirname(os.path.dirname(__file__)), "weights", "axomeme_v1.pt")
DEFAULT_WEIGHTS = os.environ.get("AXOMEME_WEIGHTS", _REPO_WEIGHTS)

def ensure_parent_directory(path):
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)

def predict_single(args):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    
    if not os.path.exists(args.weights):
        print(f"[!] Error: Model weights not found at '{args.weights}'.")
        print("    Please download pretrained weights or specify --weights /path/to/axomeme_v1.pt")
        sys.exit(1)
        
    print(f"[*] Loading AxoMEME model from: {args.weights}")
    ckpt = torch.load(args.weights, map_location=device, weights_only=False)
    ckpt_args = ckpt.get('args', {}) if isinstance(ckpt, dict) else {}
    model = PhyloAxialTransformer(
        embed_dim=ckpt_args.get('embed_dim', 384),
        num_layers=ckpt_args.get('layers', 6),
        num_heads=ckpt_args.get('heads', 12),
        window_size=ckpt_args.get('window_size', 1),
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt)
    model.eval()
    
    print(f"[*] Parsing Alignment: {args.alignment}")
    if args.tree:
        print(f"[*] Parsing Tree:      {args.tree}")
    else:
        print(f"[*] Tree argument not provided; extracting tree from alignment...")
    
    t0 = time.time()
    try:
        c, a, d, z, inv, taxa, L = load_alignment_and_tree(args.alignment, args.tree)
    except Exception as e:
        print(f"\n[!] Error loading alignment and tree: {e}")
        sys.exit(1)

    c, a, d, z = c.to(device), a.to(device), d.to(device), z.to(device)
    
    with torch.no_grad():
        y_lrt_soft, _ = model(c, a, d, z)
        lrts = torch.clamp(y_lrt_soft.squeeze(-1), min=0.0).cpu().numpy().flatten()
        
    lrts[inv] = 0.0
    elapsed = time.time() - t0
    
    # Asymptotic p-values based on 0.5 delta(0) + 0.5 chi^2(1)
    # p-value = 0.5 * (1 - chi2.cdf(LRT, df=1)) for LRT > 0
    pvals = np.ones(L, dtype=np.float32)
    pos_mask = lrts > 0.0
    pvals[pos_mask] = 0.5 * (1.0 - stats.chi2.cdf(lrts[pos_mask], df=1))
    
    sig_10 = (pvals <= 0.10).sum()
    sig_05 = (pvals <= 0.05).sum()
    
    print("\n" + "=" * 75)
    print(f"🎉 AxoMEME Inference Complete in {elapsed:.3f} seconds!")
    print(f"   Taxa: {len(taxa)} | Codon Sites: {L} | Total Invariable: {inv.sum()}")
    print(f"   Significant Sites (p <= 0.10): {sig_10} | (p <= 0.05): {sig_05}")
    print("=" * 75)
    
    # Print top candidate sites
    top_indices = np.argsort(lrts)[::-1][:10]
    print("\nTop Candidate Sites for Episodic Positive Selection:")
    print(f"{'Codon':<8} {'LRT Score':<12} {'p-value':<12} {'Status':<15}")
    print("-" * 50)
    for idx in top_indices:
        l = lrts[idx]
        p = pvals[idx]
        status = "p <= 0.05" if p <= 0.05 else ("p <= 0.10" if p <= 0.10 else "not significant")
        print(f"{idx+1:<8} {l:<12.3f} {p:<12.4e} {status:<15}")
        
    # Export outputs
    results_list = [
        {
            "site": i + 1,
            "axomeme_lrt": float(lrts[i]),
            "p_value": float(pvals[i]),
            "is_invariable": bool(inv[i])
        }
        for i in range(L)
    ]
    
    tree_meta = args.tree if args.tree else "embedded_in_alignment"
    
    if args.output:
        ensure_parent_directory(args.output)
        with open(args.output, "w") as f:
            json.dump({
                "alignment": args.alignment,
                "tree": tree_meta,
                "taxa_count": len(taxa),
                "codon_count": L,
                "runtime_sec": elapsed,
                "sites": results_list
            }, f, indent=2)
        print(f"\n[✓] JSON results written to: {args.output}")
        
    if args.csv:
        ensure_parent_directory(args.csv)
        df = pd.DataFrame(results_list)
        df.to_csv(args.csv, index=False)
        print(f"[✓] CSV results written to: {args.csv}")

def main():
    parser = argparse.ArgumentParser(
        description="AxoMEME: Ultra-Fast Neural Inference of Episodic Positive Selection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")
    
    # Predict parser
    pred_parser = subparsers.add_parser("predict", help="Run selection inference on a codon alignment and tree")
    pred_parser.add_argument("-a", "--alignment", required=True, help="Path to in-frame codon FASTA or NEXUS alignment")
    pred_parser.add_argument("-t", "--tree", required=False, default=None, help="Path to Newick/NEXUS phylogenetic tree (optional if tree is embedded in alignment)")
    pred_parser.add_argument("-w", "--weights", default=DEFAULT_WEIGHTS, help="Path to pretrained model checkpoint")
    pred_parser.add_argument("-o", "--output", help="Optional path to output JSON results")
    pred_parser.add_argument("-c", "--csv", help="Optional path to output CSV results")
    pred_parser.add_argument("--cpu", action="store_true", help="Force CPU inference")
    
    args = parser.parse_args()
    if args.command == "predict":
        predict_single(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
