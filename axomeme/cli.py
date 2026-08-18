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
from .weights import (
    resolve_weights_path,
    load_model_config,
    load_weights,
    list_available_variants,
    DEFAULT_VARIANT,
    HF_REPO_ID,
)

DEFAULT_VARIANT_ENV = os.environ.get("AXOMEME_VARIANT", DEFAULT_VARIANT)
# If set, AXOMEME_WEIGHTS points to a local weights file and bypasses HF download.
DEFAULT_WEIGHTS_ENV = os.environ.get("AXOMEME_WEIGHTS")

def ensure_parent_directory(path):
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)

def determine_adaptive_batch_size(num_species: int, total_sites: int, device: torch.device, user_batch_size: int = None) -> int:
    """
    Dynamically computes the optimal, safe site batch size based on available
    hardware memory (VRAM / RAM) and the quadratic complexity O(N^2) of tree self-attention.
    """
    if user_batch_size is not None and user_batch_size > 0:
        return min(user_batch_size, total_sites)
        
    n = num_species + 1  # including root token
    
    # 1. Determine available memory in bytes
    available_bytes = 4 * (1024 ** 3)  # default conservative 4 GB budget
    if device.type == 'cuda' and torch.cuda.is_available():
        try:
            free_mem, _ = torch.cuda.mem_get_info(device)
            available_bytes = free_mem
        except Exception:
            available_bytes = 8 * (1024 ** 3)
    else:
        try:
            import psutil
            vm = psutil.virtual_memory()
            available_bytes = vm.available
        except Exception:
            available_bytes = 8 * (1024 ** 3)
            
    # Apply safety headroom (use at most 40% of available memory for batch tensor allocations)
    target_budget_bytes = max(int(available_bytes * 0.40), 256 * (1024 ** 2))
    
    # 2. Estimate dynamic activation memory per site in float32 bytes:
    # - Pairwise genetic code masks: 4 * n^2 * 4 bytes = 16 * n^2
    # - Embeddings and intermediate activations ~ 9216 * n
    # - Attention weights across 6 layers x 12 heads = 288 * n^2
    # Total dynamic activation memory per site ~ 320 * n^2 + 10000 * n bytes
    bytes_per_site = 320 * (n ** 2) + 10000 * n + 4096
    
    calculated_batch = max(1, target_budget_bytes // bytes_per_site)
    batch_size = min(total_sites, int(calculated_batch))
    
    return batch_size

def predict_single(args):
    if torch.cuda.is_available() and not args.cpu:
        device = torch.device('cuda')
    elif torch.backends.mps.is_available() and not args.cpu:
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"[*] Hardware device selected: {device.type.upper()}")

    # Resolve weights: explicit --weights path > --model-variant (download from HF) > default variant
    try:
        weights_path = resolve_weights_path(
            weights=args.weights,
            variant=args.model_variant,
        )
    except RuntimeError as e:
        print(f"[!] {e}")
        sys.exit(1)
    print(f"[*] Loading AxoMEME model from: {weights_path}")

    # Load architecture config.
    # If using an explicit local --weights path, try to read config from the checkpoint
    # (legacy .pt files store args inline). If using HF variant download, fetch config.json from HF.
    config = {}
    if args.weights and os.path.exists(args.weights) and args.weights.endswith(".pt"):
        ckpt = torch.load(args.weights, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict) and "args" in ckpt:
            config = ckpt["args"]
        elif isinstance(ckpt, dict):
            # Legacy .pt without 'args' dict — extract known keys
            config = {k: ckpt[k] for k in ("embed_dim", "num_layers", "num_heads", "window_size") if k in ckpt}
    else:
        try:
            config = load_model_config(variant=args.model_variant)
        except Exception:
            pass  # Fall back to defaults

    model = PhyloAxialTransformer(
        embed_dim=config.get('embed_dim', 384),
        num_layers=config.get('num_layers', config.get('layers', 6)),
        num_heads=config.get('num_heads', config.get('heads', 12)),
        window_size=config.get('window_size', 1),
    ).to(device)

    state_dict = load_weights(weights=weights_path, variant=args.model_variant, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    
    print(f"[*] Parsing Alignment: {args.alignment}")
    if args.tree:
        print(f"[*] Parsing Tree:      {args.tree}")
    else:
        print(f"[*] Tree argument not provided; extracting tree from alignment...")
    
    t0 = time.time()
    try:
        c, a, d, z, inv, taxa, L = load_alignment_and_tree(args.alignment, args.tree, max_species=args.max_species)
    except Exception as e:
        print(f"\n[!] Error loading alignment and tree: {e}")
        sys.exit(1)

    # 1. Fast Invariable Site Filter (Invariable codons mathematically have LRT=0, p=1.0)
    variable_indices = np.where(~inv)[0]
    num_variable = len(variable_indices)

    # Determine adaptive batch size to prevent OOM
    batch_size = determine_adaptive_batch_size(len(taxa), max(1, num_variable), device, args.batch_size)
    num_chunks = (num_variable + batch_size - 1) // batch_size if num_variable > 0 else 0
    mode_desc = f"manual override" if args.batch_size else "hardware adaptive"
    print(f"[*] Site Batch Sizing ({mode_desc}): {batch_size} sites/chunk ({num_chunks} chunk{'s' if num_chunks != 1 else ''} for {num_variable}/{L} variable codons)")
    
    # 2. Pre-compute static tree kernel cache (Markov log-bias + Tree-RoPE cos/sin)
    d_dev = d.to(device)  # [1, N, N]
    z_dev = z.to(device)  # [1, N, 4]
    tree_cache = model.precompute_tree_cache(d_dev, z_dev)
    
    lrts = np.zeros(L, dtype=np.float32)
    if num_variable > 0:
        with torch.no_grad():
            for start_idx in range(0, num_variable, batch_size):
                end_idx = min(start_idx + batch_size, num_variable)
                batch_site_idx = variable_indices[start_idx:end_idx]
                
                c_chunk = c[batch_site_idx].to(device)  # [cur_bs, N, 1]
                a_chunk = a[batch_site_idx].to(device)  # [cur_bs, N, 1]
                
                y_soft, _ = model.forward_cached(c_chunk, a_chunk, tree_cache)
                chunk_lrts = torch.clamp(y_soft.squeeze(-1), min=0.0).cpu().numpy().flatten()
                lrts[batch_site_idx] = chunk_lrts
                
    if device.type == 'mps':
        torch.mps.synchronize()
    elif device.type == 'cuda':
        torch.cuda.synchronize()

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

def list_models():
    """List available model variants from Hugging Face."""
    try:
        variants = list_available_variants()
    except Exception as e:
        print(f"[!] Could not fetch model list from Hugging Face: {e}")
        if "401" in str(e) or "Unauthorized" in str(e):
            print("    The model repo may be gated. Set HF_TOKEN env var to authenticate.")
            print("    Get a token at: https://huggingface.co/settings/tokens")
        return

    if not variants:
        print("No model variants found on Hugging Face.")
        return

    print(f"Available AxoMEME model variants ({HF_REPO_ID}):")
    print()
    for v in variants:
        default = " (default)" if v["variant"] == DEFAULT_VARIANT else ""
        print(f"  {v['variant']:15s}  {v['description']}{default}")
    print()
    print("Use with:  axomeme predict -a alignment.fa --model-variant <variant>")
    print(f"Default variant: {DEFAULT_VARIANT}")


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
    pred_parser.add_argument("-w", "--weights", default=DEFAULT_WEIGHTS_ENV, help="Path to local model weights file (overrides HF download). Can also be set via AXOMEME_WEIGHTS env var.")
    pred_parser.add_argument("--model-variant", default=DEFAULT_VARIANT_ENV, help=f"Model variant to download from Hugging Face (default: {DEFAULT_VARIANT})")
    pred_parser.add_argument("-b", "--batch-size", type=int, default=None, help="Site batch size (default: auto-selected dynamically based on available VRAM/RAM and taxa count)")
    pred_parser.add_argument("-s", "--max-species", type=int, default=None, help="Maximum number of species/taxa to include (applies greedy Faith's PD downsampling)")
    pred_parser.add_argument("-o", "--output", help="Optional path to output JSON results")
    pred_parser.add_argument("-c", "--csv", help="Optional path to output CSV results")
    pred_parser.add_argument("--cpu", action="store_true", help="Force CPU inference")

    # List-models parser
    list_parser = subparsers.add_parser("list-models", help="List available model variants from Hugging Face")

    args = parser.parse_args()
    if args.command == "predict":
        predict_single(args)
    elif args.command == "list-models":
        list_models()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
