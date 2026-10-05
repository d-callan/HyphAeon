"""Diagnostic: is the screen's attn-edge proxy (edge_density) dead on real
data because min_sim is miscalibrated, or because base attributions genuinely
don't correlate?

On the sim panel coupled pairs had edge_density ~0.36 at min_sim=0.5. On the
real malaria panel EVERY pair scored 0.0000 — not a single |cos|>=0.5. This
computes the full |cos| distribution for a few pairs so we can see where the
mass sits and pick a realistic threshold (or conclude the proxy is dead).
"""
import os
import sys
import glob
import tempfile

import numpy as np

from symbaeon.pairing import load_paired_alignment
from symbaeon.model import (_load_model, _representative_seqs, _side_base,
                            cross_cosine)

PANEL = "/home/danielle/Documents/veg/HyphAeon/examples/hpi/results/screen_panel"
SAMPLES = os.path.join(PANEL, "samples600.txt")
# a few genes spanning high/low diversity
GENES = ["CR1", "GYPB", "PfRH4", "EBA175", "PfRH1", "EBA181"]
PAIRS = [("CR1", "PfRH4"), ("CR1", "EBA175"), ("CR1", "PfRH1"),
         ("GYPB", "EBA175"), ("GYPB", "PfRH4")]


def main():
    keep = set(open(SAMPLES).read().split())
    sides = {}
    for g in GENES:
        role = "host" if os.path.exists(f"{PANEL}/host/{g}.fasta") else "par"
        sides[g] = load_paired_alignment(f"{PANEL}/{role}/{g}.fasta", role, None)
    inds = sorted(set.intersection(*[set(s.sample_to_seqs)
                                     for s in sides.values()]) & keep)
    print(f"[*] {len(inds)} shared individuals")
    model, device = _load_model(None, None)
    sd = {}
    with tempfile.TemporaryDirectory() as td:
        for g in GENES:
            sd[g] = _side_base(_representative_seqs(sides[g], inds), inds,
                               model, device, td, g)
            print(f"[*] {g}: L={sd[g].L} uniq={len(sd[g].uniq_seqs)} "
                  f"n_sel(lrt>=1)={int((sd[g].lrts>=1).sum())}")
        print("\n|cos| distribution of base-attribution cross_cosine:")
        for h, p in PAIRS:
            Ah = sd[h].attrs_uniq[:, sd[h].ind2uniq]
            Ap = sd[p].attrs_uniq[:, sd[p].ind2uniq]
            C = np.abs(cross_cosine(Ah, Ap))
            qs = np.percentile(C, [50, 90, 99, 99.9, 100])
            cnt = {t: int((C >= t).sum()) for t in (0.3, 0.5, 0.7)}
            print(f"  {h:5s}x{p:7s} max={qs[4]:.3f} p99.9={qs[3]:.3f} "
                  f"p99={qs[2]:.3f} p90={qs[1]:.3f} | "
                  f">=0.3:{cnt[0.3]} >=0.5:{cnt[0.5]} >=0.7:{cnt[0.7]}")


if __name__ == "__main__":
    main()
