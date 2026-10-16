#!/usr/bin/env python3
"""Builds static_only.fasta: an alignment whose static TN93 network has edges
(identical sequence pairs, d = 0) but whose DANNO network has none, because
each identical pair is sampled 15 years apart (beyond max_delta_t = 2 y) and
the remaining pairs are ~15% divergent.

With --control, builds control.fasta instead: the same sequences, with each
identical pair sampled half a year apart (2000-06-15 / 2000-12-15), so DANNO
does support edges and N_act is estimable.

Headers are DC-style (B.US.YYYY.ID_0YYMMDD_PR_RT.ACC) so both the Rust
ingest and validation/run_danno.py parse the same collection dates.
Expected values for the test are taken from running tn93 + run_danno.py on
this file (see oracle.sh), not typed in.
"""
import random
import sys

control = '--control' in sys.argv
args = [a for a in sys.argv[1:] if a != '--control']
rng = random.Random(20261012)
L = 300
anc = [rng.choice('ACGT') for _ in range(L)]
out = []
for k in range(3):
    seq = list(anc)
    for i in range(L):
        if rng.random() < 0.08:
            seq[i] = rng.choice([b for b in 'ACGT' if b != seq[i]])
    s = ''.join(seq)
    out.append((f'B.US.2000.Z{k}a_0000615_PR_RT.ACC{k}A', s))
    if control:
        out.append((f'B.US.2000.Z{k}b_0001215_PR_RT.ACC{k}B', s))
    else:
        out.append((f'B.US.2015.Z{k}b_0150615_PR_RT.ACC{k}B', s))
default = 'control.fasta' if control else 'static_only.fasta'
with open(args[0] if args else default, 'w') as fh:
    for h, s in out:
        fh.write(f'>{h}\n{s}\n')
