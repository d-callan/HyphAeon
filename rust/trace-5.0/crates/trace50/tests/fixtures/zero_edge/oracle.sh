#!/bin/sh
# Regenerates the oracle for static_only.fasta (run from this directory).
# tn93 v1.1.0 (~/.local/bin/tn93); run_danno_edges.py = EV/ingest copy of
# validation/run_danno.py that also writes <out>.edges.csv.
set -e
EV=/home/sweaver/projects/programming/trace5/validation/results/Perez-Losada-2017-28961263/parity_2026-10-12
python3 make_fixture.py static_only.fasta
tn93 -t 1.0 -a RMYSKB -o static_only.tn93.csv static_only.fasta
python3 $EV/ingest/run_danno_edges.py --msa static_only.fasta --tn93 static_only.tn93.csv --out static_only.py_danno.json
python3 make_fixture.py --control control.fasta
tn93 -t 1.0 -a RMYSKB -o control.tn93.csv control.fasta
python3 $EV/ingest/run_danno_edges.py --msa control.fasta --tn93 control.tn93.csv --out control.py_danno.json
