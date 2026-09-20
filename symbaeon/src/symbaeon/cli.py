"""
symbaeon/cli.py
---------------
Command-line interface for SymbAeon host-pathogen coevolution analysis.
"""

import argparse
import sys

from . import __version__
from .coselect import run_coselect


def cmd_coselect(args):
    res = run_coselect(
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
        alpha=args.alpha,
        out_path=args.out,
    )
    sig = [p for p in res.pairs if p["significant"]]
    print(f"\n=== SymbAeon coselect ===")
    print(f"Individuals: {res.n_individuals} | host sites: {res.n_host_sites} "
          f"(+{res.n_covariates} cov) | parasite sites: {res.n_par_sites}")
    print(f"Significant pairs (q<={args.alpha}): {len(sig)}")
    for p in sig[:20]:
        print(f"  {p['host_feature']:>10} x site{p['par_site']:<5} "
              f"stat={p['stat']:.3g}  q={p['q_value']:.3g}")


def main():
    parser = argparse.ArgumentParser(
        prog="symbaeon",
        description="SymbAeon: host-pathogen coevolution analysis",
    )
    parser.add_argument("--version", action="version", version=f"symbaeon {__version__}")
    sub = parser.add_subparsers(dest="command")

    cs = sub.add_parser("coselect", aliases=["co"],
                        help="Cross-alignment site-site association across paired individuals")
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
    cs.add_argument("--alpha", type=float, default=0.05, help="FDR threshold (default: 0.05)")
    cs.add_argument("--out", default=None, help="Output TSV of all tested pairs")
    cs.set_defaults(func=cmd_coselect)

    args = parser.parse_args()
    if not getattr(args, "func", None):
        parser.print_help()
        sys.exit(0)
    args.func(args)


if __name__ == "__main__":
    main()
