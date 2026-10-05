"""
symbaeon/cli.py
---------------
Command-line interface for SymbAeon host-pathogen coevolution analysis.
"""

import argparse
import sys

from . import __version__
from .coselect import run_correlate
from .model import run_coselect
from .screen import run_screen


def cmd_correlate(args):
    res = run_correlate(
        host_fasta=args.host_aln,
        par_fasta=args.par_aln,
        manifest_path=args.pairs,
        host_cov_path=args.host_cov,
        strata_path=args.strata,
        strata_id_col=args.strata_id_col,
        strata_col=args.strata_col,
        samples_path=args.samples,
        encoding=args.encoding,
        hom_only=args.hom_only,
        min_carriers=args.min_carriers,
        par_pc=args.par_pc,
        host_pc=args.host_pc,
        alpha=args.alpha,
        out_path=args.out,
    )
    sig = [p for p in res.pairs if p["significant"]]
    print(f"\n=== SymbAeon correlate (allele-correlation) ===")
    print(f"Individuals: {res.n_individuals} | host sites: {res.n_host_sites} "
          f"(+{res.n_covariates} cov) | parasite sites: {res.n_par_sites}")
    print(f"Significant pairs (q<={args.alpha}): {len(sig)}")
    for p in sig[:20]:
        print(f"  {p['host_feature']:>10} x site{p['par_site']:<5} "
              f"stat={p['stat']:.3g}  q={p['q_value']:.3g}")


def cmd_coselect(args):
    res = run_coselect(
        host_fasta=args.host_aln,
        par_fasta=args.par_aln,
        manifest_path=args.pairs,
        samples_path=args.samples,
        mode=args.mode,
        method=args.method,
        attr_min_lrt=args.attr_min_lrt,
        min_sim=args.min_sim,
        min_lrt=args.min_lrt,
        min_cesi=args.min_cesi,
        permute=args.permute,
        seed=args.seed,
        resid_knn=args.resid_knn,
        weights=args.weights,
        variant=args.variant,
        edges_out=args.edges_out,
        burden_out=args.burden_out,
    )
    print(f"\n=== SymbAeon coselect (mode={res.mode}"
          + (f", {res.method}" if res.mode != "burden" else "") + ") ===")
    print(f"Individuals: {res.n_individuals} | selected sites: "
          f"host={res.n_host_sel} par={res.n_par_sel}")

    if res.mode in ("edges", "both"):
        print(f"\n-- concentrated (site x site edges) --")
        print(f"Edges (|cos|>={args.min_sim}, min_lrt={args.min_lrt}, "
              f"min_cesi={args.min_cesi}): {len(res.edges)}")
        if res.perm_p_edges is not None:
            print(f"Permutation: edges p={res.perm_p_edges:.3g} "
                  f"(null mean {res.null_edge_mean:.1f}) | hub p={res.perm_p_hub:.3g}")
        for h, p, s, ce in res.edges[:25]:
            print(f"  host:{h:<5} x par:{p:<5} cos={s:.3f} cesi={ce:.2f}")

    if res.mode in ("burden", "both"):
        print(f"\n-- diffuse (per-individual burden) --")
        print(f"RAW burden corr        : r={res.r_raw:+.4f} "
              f"null mean={res.null_raw_mean:+.4f}  p={res.p_raw:.4f}")
        print(f"ANCESTRY-CORRECTED corr: r={res.r_resid:+.4f} "
              f"null mean={res.null_resid_mean:+.4f}  p={res.p_resid:.4f}")
        print("[i] resid = burden after regressing out neutral-site ancestry; "
              "a significant resid corr = same individuals carry excess "
              "selected divergence in both genes beyond shared descent.")


def cmd_screen(args):
    res = run_screen(
        host_dir=args.host_dir,
        par_dir=args.par_dir,
        manifest_path=args.pairs,
        samples_path=args.samples,
        host_pattern=args.host_pattern,
        par_pattern=args.par_pattern,
        min_lrt=args.min_lrt,
        min_sim=args.min_sim,
        min_cesi=args.min_cesi,
        permute=args.permute,
        seed=args.seed,
        resid_knn=args.resid_knn,
        weights=args.weights,
        variant=args.variant,
        top=args.top,
        out_path=args.out,
    )
    print(f"\n=== SymbAeon screen ===")
    print(f"{res.n_host_genes} host x {res.n_par_genes} pathogen genes | "
          f"{res.n_individuals} shared individuals | {len(res.rows)} pairs")
    print(f"\nTop {min(args.top, len(res.rows))} pairs for `coselect`:")
    print(f"  {'host':<14}{'par':<14}{'score':>7}{'r_resid':>8}{'p_res':>7}"
          f"{'r_raw':>7}{'e_dens':>7}")
    for r in res.rows[:args.top]:
        print(f"  {r['host_gene']:<14}{r['par_gene']:<14}{r['score']:>7.2f}"
              f"{r['burden_r_resid']:>8.3f}{r['p_resid']:>7.3g}"
              f"{r['burden_r']:>7.3f}{r['edge_density']:>7.3f}")
    print("\n[i] score = z(burden_r_resid) + z(edge_density) + "
          "z(min frac_sel) + z(min lrt_mass); raw burden_r shown for context "
          "(high raw + low resid = clade-driven, not non-ancestry coupling).")
    print("[i] run `symbaeon coselect --host-aln <h> --par-aln <p>` on a "
          "shortlisted pair for the counterfactual edges + burden.")


def main():
    parser = argparse.ArgumentParser(
        prog="symbaeon",
        description="SymbAeon: host-pathogen coevolution analysis",
    )
    parser.add_argument("--version", action="version", version=f"symbaeon {__version__}")
    sub = parser.add_subparsers(dest="command")

    # --- allele-correlation (encoded carrier vectors -> Fisher/CMH/Spearman) ---
    cs = sub.add_parser("correlate", aliases=["cor"],
                        help="Allele-correlation: do the same individuals carry "
                             "non-reference alleles at host site X and parasite site Y?")
    cs.add_argument("--host-aln", required=True, help="Host CDS MSA (FASTA)")
    cs.add_argument("--par-aln", required=True, help="Parasite CDS MSA (FASTA)")
    cs.add_argument("--pairs", default=None, help="Pairing manifest TSV (seq_id -> sample_id)")
    cs.add_argument("--host-cov", default=None,
                    help="Optional host covariate TSV (sample_id + one col per covariate, e.g. SV genotypes)")
    cs.add_argument("--strata", default=None,
                    help="Optional TSV mapping sample_id -> stratum (e.g. country) for structure control")
    cs.add_argument("--strata-id-col", default=None, help="Sample-id column in strata file (default: auto)")
    cs.add_argument("--strata-col", default=None, help="Stratum column in strata file (default: auto)")
    cs.add_argument("--samples", default=None,
                    help="Optional file of sample_ids to restrict analysis (TSV with sample_id/ID col or plain list)")
    cs.add_argument("--encoding", choices=["presence", "dosage"], default="presence",
                    help="Allele encoding per individual (default: presence)")
    cs.add_argument("--hom-only", action="store_true",
                    help="Presence encoding: count a parasite sample as an alt carrier "
                         "only when alt is its sole allele (excludes mixed-infection "
                         "minority clones). Host diploid encoding unaffected.")
    cs.add_argument("--min-carriers", type=int, default=10,
                    help="Min individuals carrying alt allele to test a site (default: 10)")
    cs.add_argument("--par-pc", type=int, default=0,
                    help="Parasite relatedness correction: # of MSA-distance PCs to "
                         "regress out of parasite sites (controls viral phylogeny). 0=off")
    cs.add_argument("--host-pc", type=int, default=0,
                    help="Host relatedness correction: # of MSA-distance PCs to "
                         "regress out of host sites/covariates (controls kinship). 0=off")
    cs.add_argument("--alpha", type=float, default=0.05, help="FDR threshold (default: 0.05)")
    cs.add_argument("--out", default=None, help="Output TSV of all tested pairs")
    cs.set_defaults(func=cmd_correlate)

    # --- model-based selection-correlation (transformer; edges + burden) ---
    mc = sub.add_parser("coselect", aliases=["co"],
                        help="Model-based selection-correlation: one shared "
                             "transformer pass feeds the concentrated (edges) "
                             "and diffuse (burden) readouts")
    mc.add_argument("--host-aln", required=True, help="Host codon MSA (FASTA)")
    mc.add_argument("--par-aln", required=True, help="Parasite codon MSA (FASTA)")
    mc.add_argument("--pairs", default=None, help="Pairing manifest TSV (seq_id -> sample_id)")
    mc.add_argument("--samples", default=None,
                    help="Optional file of sample_ids to restrict the individual axis")
    mc.add_argument("--mode", choices=["edges", "burden", "both"], default="both",
                    help="edges=concentrated site-site drivers | "
                         "burden=diffuse per-individual burden | "
                         "both=run both on one shared model pass (default)")
    mc.add_argument("--method", choices=["attn", "delta_lrt", "delta_lrt_aa"],
                    default="delta_lrt_aa",
                    help="Edges driver profile: attn=alpha*delta | "
                         "delta_lrt=single-taxon counterfactual | "
                         "delta_lrt_aa=allele-group counterfactual (default, most power)")
    mc.add_argument("--attr-min-lrt", type=float, default=3.84,
                    help="Delta-LRT: min site LRT to attribute (focal-site gate)")
    mc.add_argument("--min-sim", type=float, default=0.5,
                    help="Min |cosine| between host/par driver profiles for an edge")
    mc.add_argument("--min-lrt", type=float, default=1.0,
                    help="Min selection LRT — gates edges and the burden site set")
    mc.add_argument("--min-cesi", type=float, default=0.0,
                    help="Min CESI = |cos|*sqrt(lrt_u*lrt_v)")
    mc.add_argument("--permute", type=int, default=0,
                    help="Permutation null: shuffle parasite individual axis P times")
    mc.add_argument("--resid-knn", type=int, default=5,
                    help="Neighbours for the relatedness-kernel ancestry "
                         "correction (burden resid). Must stay below the "
                         "smallest clade to remove; larger k smooths across "
                         "clade boundaries and leaks clade signal back in.")
    mc.add_argument("--seed", type=int, default=0)
    mc.add_argument("--weights", default=None, help="Model weights path (default: bundled)")
    mc.add_argument("--variant", default=None, help="Model variant (default: bundled)")
    mc.add_argument("--edges-out", default=None, help="Output TSV of edges")
    mc.add_argument("--burden-out", default=None,
                    help="Output TSV of per-individual burden/residuals")
    mc.set_defaults(func=cmd_coselect)

    # --- collection screen: rank host_gene x pathogen_gene pairs for coselect ---
    sc = sub.add_parser("screen",
                        help="Screen a collection of host x pathogen gene MSAs "
                             "over a shared individual axis; rank pairs by "
                             "ancestry-corrected burden coupling + attn-edge "
                             "proxy + per-gene selection activity")
    sc.add_argument("--host-dir", required=True,
                    help="Directory of host-gene codon MSAs (one per gene)")
    sc.add_argument("--par-dir", required=True,
                    help="Directory of pathogen-gene codon MSAs (one per gene)")
    sc.add_argument("--host-pattern", default="*.fasta",
                    help="Glob for host MSAs in --host-dir (default: *.fasta)")
    sc.add_argument("--par-pattern", default="*.fasta",
                    help="Glob for pathogen MSAs in --par-dir (default: *.fasta)")
    sc.add_argument("--pairs", default=None, help="Pairing manifest TSV (seq_id -> sample_id)")
    sc.add_argument("--samples", default=None,
                    help="Optional file of sample_ids to restrict the individual axis")
    sc.add_argument("--min-lrt", type=float, default=1.0,
                    help="Min selection LRT — gates selected sites and the burden site set")
    sc.add_argument("--min-sim", type=float, default=0.5,
                    help="Min |cosine| between base attribution profiles for an attn-edge")
    sc.add_argument("--min-cesi", type=float, default=0.0, help="Min CESI for attn-edges")
    sc.add_argument("--permute", type=int, default=200,
                    help="Permutation null for the burden correlation (default: 200)")
    sc.add_argument("--resid-knn", type=int, default=5,
                    help="Neighbours for the relatedness-kernel ancestry "
                         "correction (burden resid). Must stay below the "
                         "smallest clade to remove; larger k smooths across "
                         "clade boundaries and leaks clade signal back in.")
    sc.add_argument("--seed", type=int, default=0)
    sc.add_argument("--weights", default=None, help="Model weights path (default: bundled)")
    sc.add_argument("--variant", default=None, help="Model variant (default: bundled)")
    sc.add_argument("--top", type=int, default=20,
                    help="How many top-ranked pairs to print (default: 20)")
    sc.add_argument("--out", default=None, help="Output TSV of all ranked pairs")
    sc.set_defaults(func=cmd_screen)

    args = parser.parse_args()
    if not getattr(args, "func", None):
        parser.print_help()
        sys.exit(0)
    args.func(args)


if __name__ == "__main__":
    main()
