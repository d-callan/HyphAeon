#!/usr/bin/env python3
"""Generate the minimum-overlap toy alignment and its tn93-binary oracle.

Writes, next to this script:
  overlap_toy.fasta        toy alignment (pairs straddling tn93's -l 100 rule)
  overlap_toy_RMYSKB.csv   `tn93 -t 1.0 -a RMYSKB` on it (the oracle; default -l 100)

Shared-site counts (both characters non-gap; N/? count, as in tn93's totalNonGap):
  ref~o100 100   ref~o101 101   o100~o101 100   ref~even101 101
  ref~o100N 101  o100~o100N 100 o101~o100N 101  ref~tail100 100
tn93 keeps a pair only when shared sites > 100 (tn93_shared.cc: `if (totalNonGap
<= min_overlap) return -1.;`), so exactly the 101-site pairs should appear.

Run: python3 gen_overlap_toy.py   (needs `tn93` on PATH; used v1.1.0)
"""
import os
import random
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
W = 300
rng = random.Random(20261012)
ref = [rng.choice("ACGT") for _ in range(W)]

TS = {"A": "G", "G": "A", "C": "T", "T": "C"}


def with_ts(seq, positions):
    s = list(seq)
    for p in positions:
        s[p] = TS[s[p]]
    return s


def window(seq, keep):
    return [c if i in keep else "-" for i, c in enumerate(seq)]


seqs = {
    "ref": ref,
    "o100": window(with_ts(ref, [10, 50]), set(range(100))),
    "o101": window(with_ts(ref, [10, 50]), set(range(101))),
    "even101": window(with_ts(ref, [20]), set(range(0, 201, 2))),
    "o100N": window(with_ts(ref, [30]), set(range(101))),
    "tail100": window(with_ts(ref, [250]), set(range(200, 300))),
}
seqs["o100N"][100] = "N"

fasta = os.path.join(HERE, "overlap_toy.fasta")
with open(fasta, "w") as fh:
    for name, s in seqs.items():
        fh.write(f">{name}\n{''.join(s)}\n")

out = os.path.join(HERE, "overlap_toy_RMYSKB.csv")
# tn93 prints its run summary JSON on stdout; the pair list goes to -o
subprocess.run(["tn93", "-q", "-t", "1.0", "-a", "RMYSKB", "-o", out, fasta],
               check=True, stdout=subprocess.DEVNULL)
print(open(out).read())
