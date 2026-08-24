"""
axomeme/cli.py
--------------
Command-line interface for AxoMEME:
1. 'predict': Ultra-Fast Neural Inference of Episodic Positive Selection (AxoMEME Transformer)
2. 'phenotype' (phylowas): Directional Phenotype-Genotype Association & PARS Signature Extraction
3. 'epistasis' (essm): Multi-Scale Epistatic Sector Mining (Two-Stage Seed-and-Extend TSE)
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
from .phenotype import run_phenotype_association, PRESETS
from .epistasis import run_epistatic_sector_mining

_REPO_WEIGHTS = os.path.join(os.path.dirname(os.path.dirname(__file__)), "weights", "axomeme_v1.pt")
DEFAULT_WEIGHTS = os.environ.get("AXOMEME_WEIGHTS", _REPO_WEIGHTS)

def ensure_parent_directory(path):
    if path:
        parent = os.path.dirname(os.path.abspath(path))
        os.makedirs(parent, exist_ok=True)

def determine_adaptive_batch_size(num_species: int, total_sites: int, device: torch.device, user_batch_size: int = None) -> int:
    if user_batch_size is not None and user_batch_size > 0:
        return min(user_batch_size, total_sites)
        
    n = num_species + 1
    available_bytes = 4 * (1024 ** 3)
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
            
    target_budget_bytes = max(int(available_bytes * 0.40), 256 * (1024 ** 2))
    bytes_per_site = 320 * (n ** 2) + 10000 * n + 4096
    calculated_batch = max(1, target_budget_bytes // bytes_per_site)
    return min(total_sites, int(calculated_batch))

def cmd_predict(args):
    if torch.cuda.is_available() and not args.cpu:
        device = torch.device('cuda')
    elif torch.backends.mps.is_available() and not args.cpu:
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"[*] Hardware device selected: {device.type.upper()}")
    
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
        prune_dups = not getattr(args, "no_prune_duplicates", False)
        c, a, d, z, inv, taxa, L = load_alignment_and_tree(
            args.alignment, args.tree, max_species=args.max_species, prune_duplicates=prune_dups
        )
    except Exception as e:
        print(f"\n[!] Error loading alignment and tree: {e}")
        sys.exit(1)

    variable_indices = np.where(~inv)[0]
    num_variable = len(variable_indices)

    batch_size = determine_adaptive_batch_size(len(taxa), max(1, num_variable), device, args.batch_size)
    num_chunks = (num_variable + batch_size - 1) // batch_size if num_variable > 0 else 0
    mode_desc = "manual override" if args.batch_size else "hardware adaptive"
    print(f"[*] Site Batch Sizing ({mode_desc}): {batch_size} sites/chunk ({num_chunks} chunk{'s' if num_chunks != 1 else ''} for {num_variable}/{L} variable codons)")
    
    d_dev = d.to(device)
    z_dev = z.to(device)
    tree_cache = model.precompute_tree_cache(d_dev, z_dev)
    
    lrts = np.zeros(L, dtype=np.float32)
    if num_variable > 0:
        with torch.no_grad():
            for start_idx in range(0, num_variable, batch_size):
                end_idx = min(start_idx + batch_size, num_variable)
                batch_site_idx = variable_indices[start_idx:end_idx]
                
                c_chunk = c[batch_site_idx].to(device)
                a_chunk = a[batch_site_idx].to(device)
                
                y_soft, _ = model.forward_cached(c_chunk, a_chunk, tree_cache)
                chunk_lrts = torch.clamp(y_soft.squeeze(-1), min=0.0).cpu().numpy().flatten()
                lrts[batch_site_idx] = chunk_lrts
                
    if device.type == 'mps':
        torch.mps.synchronize()
    elif device.type == 'cuda':
        torch.cuda.synchronize()

    elapsed = time.time() - t0
    
    # 0.5 * chi2.sf(LRT, df=1) under Self & Liang (1987) mixture null: 0.5 * delta(0) + 0.5 * chi2(1)
    pvals = np.ones(L, dtype=np.float32)
    pos_mask = lrts > 0.0
    pvals[pos_mask] = 0.5 * stats.chi2.sf(lrts[pos_mask], df=1)
    
    # Compute Benjamini-Hochberg False Discovery Rate (FDR) q-values
    order = np.argsort(pvals)
    ranks = np.empty(L, dtype=int)
    ranks[order] = np.arange(1, L + 1)
    raw_q = pvals * (L / ranks)
    sorted_q = raw_q[order]
    for i in range(L - 2, -1, -1):
        sorted_q[i] = min(sorted_q[i], sorted_q[i + 1])
    raw_q[order] = sorted_q
    qvals = np.clip(raw_q, 0.0, 1.0).astype(np.float32)
    
    sig_10 = (pvals <= 0.10).sum()
    sig_05 = (pvals <= 0.05).sum()
    fdr_05 = (qvals <= 0.05).sum()
    fdr_10 = (qvals <= 0.10).sum()
    
    print("\n" + "=" * 78)
    print(f"🎉 AxoMEME Selection Inference Complete in {elapsed:.3f} seconds!")
    print(f"   Taxa: {len(taxa)} | Codon Sites: {L} | Total Invariable: {inv.sum()}")
    print(f"   Nominal Significance: (p <= 0.05): {sig_05} | (p <= 0.10): {sig_10}")
    print(f"   FDR Significance:     (q <= 0.05): {fdr_05} | (q <= 0.10): {fdr_10}")
    print("=" * 78)
    
    top_indices = np.argsort(lrts)[::-1][:10]
    print("\nTop Candidate Sites for Episodic Positive Selection:")
    print(f"{'Codon':<8} {'LRT Score':<12} {'p-value':<12} {'FDR q-val':<12} {'Status':<15}")
    print("-" * 62)
    for idx in top_indices:
        l = lrts[idx]
        p = pvals[idx]
        q = qvals[idx]
        status = "p <= 0.05" if p <= 0.05 else ("p <= 0.10" if p <= 0.10 else "not significant")
        print(f"{idx+1:<8} {l:<12.3f} {p:<12.4e} {q:<12.4e} {status:<15}")
        
    results_list = [
        {
            "site": i + 1,
            "axomeme_lrt": float(lrts[i]),
            "p_value": float(pvals[i]),
            "q_value": float(qvals[i]),
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

def cmd_phenotype(args):
    print(f"[*] Executing Directional Phenotype-Genotype Mapping (PhyloWAS)...")
    print(f"[*] Alignment: {args.alignment}")
    
    t0 = time.time()
    try:
        res = run_phenotype_association(
            alignment_path=args.alignment,
            preset=args.preset,
            foreground=args.foreground,
            background=args.background,
            phenotype_file=args.phenotype_file,
            trait_col=args.trait_col,
            species_col=args.species_col,
            continuous=args.continuous,
            min_taxa_per_site=args.min_taxa,
            alpha=args.alpha
        )
    except Exception as e:
        print(f"\n[!] Phenotype Association Error: {e}")
        sys.exit(1)
        
    elapsed = time.time() - t0
    meta = res["phenotype_meta"]
    
    print("\n" + "=" * 80)
    print(f"🎉 PhyloWAS Phenotype Association Complete in {elapsed:.3f} seconds!")
    print(f"   Target Trait: {meta['description']}")
    print(f"   Taxa: {res['taxa_count']} (Foreground: {meta.get('foreground_count', 'N/A')}) | Codon Sites: {res['codon_count']}")
    print(f"   Spectral Energy (Psi): {res['spectral_energy']:.3f} | Normalized Ratio: {res['norm_spectral_ratio']:.4f}")
    print(f"   Significant Sites (FDR q <= {args.alpha}): {res['significant_sites_count']}")
    print(f"   Compact PARS Signature: {res['compact_pars_signature']}")
    print("=" * 80)
    
    sites = res["sites"]
    top_sites = sites[:15]
    if top_sites:
        print("\nTop Trait-Associated Codon Sites:")
        print(f"{'Site':<6} {'Ref':<5} {'Derived':<9} {'Mut':<5} {'Shared':<8} {'Assoc (rho)':<12} {'p-value':<12} {'q-value (FDR)':<14} {'Fg %':<7} {'Bg %':<7}")
        print("-" * 90)
        for s in top_sites:
            q_str = f"{s.get('q_value', 1.0):.3e}" if s.get('q_value', 1.0) < 0.01 else f"{s.get('q_value', 1.0):.3f}"
            print(f"{s['site']:<6d} {s['ref_aa']:<5s} {s['derived_aa']:<9s} {s['total_mutations']:<5d} {s['shared_foreground_mutations']:<8d} {s['association_rho']:<12.4f} {s['p_value']:<12.3e} {q_str:<14s} {s['foreground_freq_pct']:<7.1f} {s['background_freq_pct']:<7.1f}")

    if args.output:
        ensure_parent_directory(args.output)
        with open(args.output, "w") as f:
            json.dump(res, f, indent=2)
        print(f"\n[✓] JSON results written to: {args.output}")

    if args.csv:
        ensure_parent_directory(args.csv)
        df = pd.DataFrame(sites)
        df.to_csv(args.csv, index=False)
        print(f"[✓] CSV results written to: {args.csv}")

def cmd_epistasis(args):
    print(f"[*] Executing Phylogenetic Branch Attribution, Co-Selection Networks & Selection DMS (ESSM)...")
    print(f"[*] Alignment: {args.alignment}")
    if args.tree:
        print(f"[*] Tree:      {args.tree}")
    else:
        print(f"[*] Tree:      (extracting from alignment)")
    
    t0 = time.time()
    try:
        from .epistasis import run_epistasis_analysis
        res = run_epistasis_analysis(
            alignment_path=args.alignment,
            tree_path=args.tree,
            weights_path=getattr(args, "weights", DEFAULT_WEIGHTS),
            focal_taxon=getattr(args, "focal_taxon", None),
            min_sim=getattr(args, "min_sim", 0.30),
            min_shared=getattr(args, "min_shared", 2),
            max_fdr=getattr(args, "max_fdr", 0.05),
            min_lrt=getattr(args, "min_lrt", 1.0),
            min_clique_size=getattr(args, "min_clique_size", 3),
            max_overlap=getattr(args, "max_overlap", 0.50),
            run_dms=not getattr(args, "no_dms", False),
            cpu=getattr(args, "cpu", False)
        )
    except Exception as e:
        print(f"\n[!] Epistasis / ESSM Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    elapsed = time.time() - t0
    edges = res["edges"]
    sectors = res["sectors"]
    plasticity = res["plasticity"]
    
    print("\n" + "=" * 82)
    print(f"🎉 Epistatic Analysis Complete in {elapsed:.3f} seconds!")
    print(f"   Taxa: {res['taxa_count']} | Codons: {res['codon_count']} | Evaluated Branches: {res['branch_count']}")
    max_fdr = getattr(args, "max_fdr", 0.05)
    print(f"   Co-Selection Edges (FDR q <= {max_fdr}): {res['coevolution_edges_count']} | Discovered Sectors: {res['sectors_discovered']}")
    print("=" * 82)
    
    # 1. Top Co-Selection Pairs
    if edges:
        print("\nTop Phylogenetic Branch Co-Selection Pairs (Ranked by CESI):")
        print(f"{'Rank':<5} {'Residue Pair':<16} {'LRT 1/2':<12} {'Co-Sel':<9} {'Shared':<8} {'CESI':<10} {'FDR q-val':<12}")
        print("-" * 76)
        for idx, e in enumerate(edges[:12]):
            pair_str = f"{e['ref_u']}{e['site_u']} <-> {e['ref_v']}{e['site_v']}"
            lrt_str = f"{e['lrt_u']:.1f} / {e['lrt_v']:.1f}"
            q_str = f"{e['fdr_q']:.2e}" if e['fdr_q'] < 0.01 else f"{e['fdr_q']:.3f}"
            print(f"#{idx+1:<4d} {pair_str:<16s} {lrt_str:<12s} {e['similarity']:<9.4f} {e['shared_branches']:<8d} {e['cesi']:<10.3f} {q_str:<12s}")

    # 2. Epistatic Sectors
    if sectors:
        print("\nDiscovered Epistatic Sectors (Two-Stage Seed-and-Extend):")
        for sec in sectors[:8]:
            print(f"\nSector #{sec['sector_id']} (Size K = {sec['size']} residues): Sites {sec['sites']}")
            print(f"  • Spectral Coherence C(S): {sec['spectral_coherence']:.4f} | Mean LRT: {sec['mean_lrt']:.2f} | Shared Branches: {sec['shared_branches']}")
            print(f"  • Signature: {sec['pars_signature']}")

    # 3. Selection DMS Mutational Plasticity
    if plasticity:
        df_plas = pd.DataFrame(plasticity)
        top_plastic = df_plas.sort_values(by="intrinsic_plasticity", ascending=False).head(5)
        top_rigid = df_plas.sort_values(by="intrinsic_plasticity", ascending=True).head(5)
        
        print("\nSelection Deep Mutational Scanning (ESSM Intrinsic Plasticity):")
        print("  Top Permissive / Evolvable Sites (High Plasticity):")
        for _, r in top_plastic.iterrows():
            print(f"    • Site {int(r['site']):<4d} ({r['wt_aa']}): Plasticity = {r['intrinsic_plasticity']:.3f} | Baseline LRT = {r['baseline_lrt']:.2f} (p = {r['p_value']:.3e})")
            
        print("  Top Rigid / Catalytic Backbone Sites (Low Plasticity):")
        for _, r in top_rigid.iterrows():
            print(f"    • Site {int(r['site']):<4d} ({r['wt_aa']}): Plasticity = {r['intrinsic_plasticity']:.3f} | Baseline LRT = {r['baseline_lrt']:.2f} (p = {r['p_value']:.3e})")

    if getattr(args, "output", None):
        ensure_parent_directory(args.output)
        with open(args.output, "w") as f:
            json.dump(res, f, indent=2)
        print(f"\n[✓] JSON results written to: {args.output}")

    if getattr(args, "csv", None):
        ensure_parent_directory(args.csv)
        if getattr(args, "command", "") in ["dms", "essm", "digital-dms"] and plasticity:
            df_out = pd.DataFrame(plasticity)
        elif edges:
            df_out = pd.DataFrame(edges)
        elif plasticity:
            df_out = pd.DataFrame(plasticity)
        else:
            df_out = pd.DataFrame(sectors)
        df_out.to_csv(args.csv, index=False)
        print(f"[✓] CSV results written to: {args.csv}")

    if getattr(args, "graphml", None):
        ensure_parent_directory(args.graphml)
        G = nx.Graph()
        for e in edges:
            G.add_edge(
                str(e["site_u"]),
                str(e["site_v"]),
                weight=float(e["similarity"]),
                cesi=float(e["cesi"]),
                shared=int(e["shared_branches"]),
                fdr_q=float(e["fdr_q"])
            )
        nx.write_graphml(G, args.graphml)
        print(f"[✓] Co-selection network GraphML written to: {args.graphml}")

def main():
    parser = argparse.ArgumentParser(
        prog="axomeme",
        description="AxoMEME: Ultra-Fast Neural Selection Inference, Phenotype-Genotype Mapping, and Epistatic Sector Mining",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")
    
    # 1. Predict Subcommand
    pred_parser = subparsers.add_parser("predict", help="Run episodic positive selection inference (AxoMEME Transformer)")
    pred_parser.add_argument("-a", "--alignment", required=True, help="Path to in-frame codon FASTA or NEXUS alignment")
    pred_parser.add_argument("-t", "--tree", required=False, default=None, help="Path to Newick/NEXUS phylogenetic tree (optional if embedded)")
    pred_parser.add_argument("-w", "--weights", default=DEFAULT_WEIGHTS, help="Path to pretrained model checkpoint")
    pred_parser.add_argument("-b", "--batch-size", type=int, default=None, help="Site batch size (default: auto-selected)")
    pred_parser.add_argument("-s", "--max-species", type=int, default=None, help="Maximum number of species to include (PD downsampling)")
    pred_parser.add_argument("--no-prune-duplicates", action="store_true", help="Disable automatic collapsing of 100% identical sequence duplicates")
    pred_parser.add_argument("-o", "--output", help="Optional path to output JSON results")
    pred_parser.add_argument("-c", "--csv", help="Optional path to output CSV results")
    pred_parser.add_argument("--cpu", action="store_true", help="Force CPU inference")
    
    # 2. Phenotype Subcommand (PhyloWAS)
    pheno_parser = subparsers.add_parser("phenotype", aliases=["phylowas", "trait"], help="Run directional phenotype-genotype association & PARS signature extraction")
    pheno_parser.add_argument("-a", "--alignment", required=True, help="Path to in-frame codon FASTA or NEXUS alignment")
    pheno_parser.add_argument("-p", "--preset", choices=list(PRESETS.keys()), help=f"Curated phenotype preset: {', '.join(PRESETS.keys())}")
    pheno_parser.add_argument("-fg", "--foreground", help="Inline comma-separated list or regex pattern of foreground species")
    pheno_parser.add_argument("-bg", "--background", help="Optional explicit list of background control species")
    pheno_parser.add_argument("-pf", "--phenotype-file", help="Path to CSV/TSV metadata file mapping taxa to trait values")
    pheno_parser.add_argument("-tc", "--trait-col", help="Name of the trait column in phenotype file")
    pheno_parser.add_argument("-sc", "--species-col", help="Name of the species/taxa column in phenotype file")
    pheno_parser.add_argument("--continuous", action="store_true", help="Treat trait values as continuous phylogenetic contrasts")
    pheno_parser.add_argument("--min-taxa", type=int, default=10, help="Minimum sequenced taxa required per site")
    pheno_parser.add_argument("--alpha", type=float, default=0.05, help="FDR significance threshold")
    pheno_parser.add_argument("-o", "--output", help="Optional path to output JSON results")
    pheno_parser.add_argument("-c", "--csv", help="Optional path to output CSV results")

    # 3. Epistasis Subcommand (Branch Co-Selection & Sectors)
    epi_parser = subparsers.add_parser("epistasis", aliases=["coselection", "sector", "network"], help="Run phylogenetic branch co-selection and epistatic sector mining")
    epi_parser.add_argument("-a", "--alignment", required=True, help="Path to in-frame codon FASTA or NEXUS alignment")
    epi_parser.add_argument("-t", "--tree", default=None, help="Optional Newick/NEXUS phylogenetic tree (optional if embedded)")
    epi_parser.add_argument("-w", "--weights", default=DEFAULT_WEIGHTS, help="Path to pretrained model checkpoint")
    epi_parser.add_argument("--focal-taxon", help="Focal taxon for in silico Selection DMS sweep (default: auto/consensus)")
    epi_parser.add_argument("--min-sim", type=float, default=0.30, help="Pairwise cosine similarity threshold for co-selection edges")
    epi_parser.add_argument("--min-shared", type=int, default=2, help="Minimum shared mutated phylogenetic branches")
    epi_parser.add_argument("--max-fdr", type=float, default=0.05, help="Benjamini-Hochberg FDR q-value threshold")
    epi_parser.add_argument("--min-lrt", type=float, default=1.0, help="Minimum site selection drive (LRT threshold)")
    epi_parser.add_argument("--min-clique-size", type=int, default=3, help="Minimum clique seed size for epistatic sectors")
    epi_parser.add_argument("--max-overlap", type=float, default=0.50, help="Maximum Jaccard overlap allowed between discovered sectors")
    epi_parser.add_argument("--no-dms", action="store_true", help="Skip 19-amino-acid in silico Selection DMS sweep")
    epi_parser.add_argument("--cpu", action="store_true", help="Force CPU execution")
    epi_parser.add_argument("-o", "--output", help="Optional path to output JSON results")
    epi_parser.add_argument("-c", "--csv", help="Optional path to output CSV results")
    epi_parser.add_argument("--graphml", help="Export co-selection network to GraphML for Cytoscape/Gephi")

    # 4. Digital DMS / ESSM Subcommand
    dms_parser = subparsers.add_parser("dms", aliases=["essm", "digital-dms"], help="Run in silico Selection Deep Mutational Scanning (Digital DMS / ESSM)")
    dms_parser.add_argument("-a", "--alignment", required=True, help="Path to in-frame codon FASTA or NEXUS alignment")
    dms_parser.add_argument("-t", "--tree", default=None, help="Optional Newick/NEXUS phylogenetic tree (optional if embedded)")
    dms_parser.add_argument("-w", "--weights", default=DEFAULT_WEIGHTS, help="Path to pretrained model checkpoint")
    dms_parser.add_argument("--focal-taxon", help="Focal taxon for in silico Selection DMS sweep (default: auto/consensus)")
    dms_parser.add_argument("--cpu", action="store_true", help="Force CPU execution")
    dms_parser.add_argument("-o", "--output", help="Optional path to output JSON results")
    dms_parser.add_argument("-c", "--csv", help="Optional path to output CSV results")

    args = parser.parse_args()
    if args.command == "predict":
        cmd_predict(args)
    elif args.command in ["phenotype", "phylowas", "trait"]:
        cmd_phenotype(args)
    elif args.command in ["epistasis", "coselection", "sector", "network"]:
        cmd_epistasis(args)
    elif args.command in ["dms", "essm", "digital-dms"]:
        cmd_epistasis(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
