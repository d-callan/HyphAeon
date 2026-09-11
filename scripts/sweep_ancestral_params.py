#!/usr/bin/env python3
"""
Sweep per-node ASR parameters against the Randall et al. (2016) benchmark.

Sweeps three parameters independently against a baseline (hard mask, no
joint pass, no boost), comparing per-node reconstructions to known true
ancestors:

1. --node-tip-context-weight (soft descendant mask)
2. --joint-pass (joint consistency pass strength)
3. --weak-node-boost (DMS/epi lambda scaling for weak-signal nodes)

Usage:
    python scripts/sweep_ancestral_params.py
    python scripts/sweep_ancestral_params.py --cpu
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from hyphaeon.ancestral import run_ancestral_reconstruction

RANDALL_DIR = REPO_ROOT / "examples" / "randall_asr"
ALN = RANDALL_DIR / "fp_leaves_codon.fasta"
TREE = RANDALL_DIR / "RandallBenchmarkTree.newick"
TRUE_ANC = RANDALL_DIR / "fp_true_ancestors.fasta"
MREF = RANDALL_DIR / "fp_mrfp1_reference.fasta"
OUT_DIR = REPO_ROOT / "scripts" / "sweep_ancestral_output"
OUT_DIR.mkdir(exist_ok=True)


def parse_fasta(path):
    seqs = {}
    header = None
    seq = []
    for line in Path(path).read_text().strip().split("\n"):
        if line.startswith(">"):
            if header is not None:
                seqs[header] = "".join(seq)
            header = line[1:].split()[0]
            seq = []
        else:
            seq.append(line.strip())
    if header is not None:
        seqs[header] = "".join(seq)
    return seqs


def identity(s1, s2):
    if len(s1) != len(s2):
        min_len = min(len(s1), len(s2))
        s1, s2 = s1[:min_len], s2[:min_len]
    matches = sum(a == b for a, b in zip(s1, s2))
    return 100.0 * matches / len(s1) if s1 else 0.0


def evaluate(res, true_ancs, mref_seq):
    """Compare predicted per-node ancestors to true ancestors."""
    preds = {}
    for node_info in res["nodes"]:
        label = node_info["node_label"]
        preds[label] = node_info["candidates"][0]["sequence"]

    results = {}
    for label, pred_seq in preds.items():
        true_label = None
        if label.startswith("A"):
            try:
                true_label = str(int(label[1:]))
            except ValueError:
                pass
        if true_label and true_label in true_ancs:
            results[label] = identity(pred_seq, true_ancs[true_label])
        elif label == "node_01-02-03":
            results[label] = identity(pred_seq, mref_seq)
    avg = np.mean(list(results.values())) if results else 0.0
    return results, avg


def run_config(label, extra_kwargs, true_ancs, mref_seq, cpu=False):
    """Run ancestral reconstruction with given kwargs, return (runtime, results)."""
    print(f"\n{'='*60}")
    print(f"  Config: {label}")
    print(f"  Extra: {extra_kwargs}")
    print(f"{'='*60}")
    t0 = time.time()
    res = run_ancestral_reconstruction(
        alignment_path=str(ALN),
        tree_path=str(TREE),
        outgroup="01",
        lambda_dist=4.0,
        top_k=2,
        per_node=True,
        cpu=cpu,
        progress=False,
        **extra_kwargs,
    )
    elapsed = time.time() - t0
    per_node, avg = evaluate(res, true_ancs, mref_seq)
    print(f"  Done in {elapsed:.1f}s — avg identity: {avg:.1f}%")
    for node, pid in sorted(per_node.items()):
        print(f"    {node}: {pid:.1f}%")
    return elapsed, per_node, avg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu", action="store_true", help="Force CPU execution")
    args = parser.parse_args()

    true_ancs = parse_fasta(TRUE_ANC)
    mref_seq = list(parse_fasta(MREF).values())[0]
    true_ancs["20"] = mref_seq

    all_results = []

    sweeps = [
        ("node_tip_context_weight", "node_tip_context_weight", [0.0, 0.05, 0.1, 0.2, 0.5]),
        ("joint_pass", "joint_pass", [0.0, 0.1, 0.3, 0.5]),
        ("weak_node_boost", "weak_node_boost", [0.0, 1.0, 2.0, 5.0]),
    ]

    for sweep_name, kwarg_name, values in sweeps:
        print(f"\n{'#'*60}")
        print(f"# SWEEP: {sweep_name}")
        print(f"{'#'*60}")
        for val in values:
            label = f"{sweep_name}_{val}"
            elapsed, per_node, avg = run_config(
                label, {kwarg_name: val}, true_ancs, mref_seq, cpu=args.cpu
            )
            all_results.append({
                "sweep": sweep_name,
                "value": val,
                "runtime_s": elapsed,
                "avg_identity": avg,
                "per_node": per_node,
            })

    print("\n\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for sweep_name, _, _ in sweeps:
        print(f"\n--- {sweep_name} ---")
        sweep_results = [r for r in all_results if r["sweep"] == sweep_name]
        print(f"  {'Value':>8}  {'Avg ID':>8}  {'Runtime':>8}")
        for r in sweep_results:
            print(f"  {r['value']:>8.2f}  {r['avg_identity']:>7.1f}%  {r['runtime_s']:>7.1f}s")

    out_json = OUT_DIR / "sweep_results.json"
    with open(out_json, "w") as fh:
        json.dump(all_results, fh, indent=2)
    print(f"\nFull results saved to {out_json}")


if __name__ == "__main__":
    main()
