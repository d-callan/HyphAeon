#!/usr/bin/env bash
# Reproduce CLI outputs for rhizaeon + trace50 and diff against committed
# normalized baselines under rust/tests/baselines/.
#
# Usage:
#   ./scripts/compare_cli_baseline.sh            # compare (exit 1 on diff)
#   ./scripts/compare_cli_baseline.sh --regen    # regenerate baselines
#
# Timing fields (run_time_ms, execution_time_seconds, prose durations, and
# absolute out-dir paths) are normalized away before diffing.
set -euo pipefail

RUST_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASELINES="$RUST_ROOT/tests/baselines"
FIXTURES="$RUST_ROOT/tests/fixtures"
NORMALIZE="$RUST_ROOT/scripts/normalize_baseline.py"
RHIZ_ALN="$RUST_ROOT/rhizaeon/tests/empirical/data/hiv1_kal153_crf02_ag.fasta"
TRACE_ALN="$FIXTURES/pair_test.fasta"

REGEN=0
[ "${1:-}" = "--regen" ] && REGEN=1

RUN_DIR="$(mktemp -d)"
trap 'rm -rf "$RUN_DIR"' EXIT

# CLIs ship via the rhizaeon/trace50 Python packages (maturin); run them
# through `python3 -m` so the comparison needs no PATH assumptions.
RHIZ_BIN=(python3 -m rhizaeon)
TRACE_BIN=(python3 -m trace50)
if ! python3 -c "import rhizaeon._native" 2>/dev/null || \
   ! python3 -c "import trace50._native" 2>/dev/null; then
    echo "[!] rhizaeon/trace50 Python packages not installed." >&2
    echo "    Run: cd rhizaeon && maturin develop   (same for trace50/)" >&2
    exit 1
fi

mkdir -p "$RUN_DIR/rhizaeon" "$RUN_DIR/trace50"

echo "[*] rhizaeon -> $RUN_DIR/rhizaeon"
"${RHIZ_BIN[@]}" \
    --input "$RHIZ_ALN" \
    --output "$RUN_DIR/rhizaeon/rhiz.json" \
    --viz "$RUN_DIR/rhizaeon/rhiz_viz.json" \
    --html "$RUN_DIR/rhizaeon/rhiz.html" \
    > "$RUN_DIR/rhizaeon/rhiz_stdout.txt" 2>&1

echo "[*] trace50 -> $RUN_DIR/trace50"
"${TRACE_BIN[@]}" \
    --fasta "$TRACE_ALN" \
    --out-dir "$RUN_DIR/trace50" \
    > "$RUN_DIR/trace50/trace_stdout.txt" 2>&1

echo "[*] Normalizing timing fields / paths"
python3 "$NORMALIZE" "--out-dir=$RUN_DIR" \
    "$RUN_DIR/rhizaeon/rhiz.json" \
    "$RUN_DIR/rhizaeon/rhiz_viz.json" \
    "$RUN_DIR/rhizaeon/rhiz.html" \
    "$RUN_DIR/rhizaeon/rhiz_stdout.txt" \
    "$RUN_DIR/trace50/trace_stdout.txt" \
    "$RUN_DIR/trace50/trace50_visualization_dossier.json" \
    "$RUN_DIR/trace50/acute_transmission_dyads.csv" \
    "$RUN_DIR/trace50/autoclock_communities.csv" \
    "$RUN_DIR/trace50/patient_surveillance_assignments.csv"

if [ "$REGEN" -eq 1 ]; then
    cp "$RUN_DIR/rhizaeon/rhiz.json"            "$BASELINES/rhizaeon/"
    cp "$RUN_DIR/rhizaeon/rhiz_viz.json"        "$BASELINES/rhizaeon/"
    cp "$RUN_DIR/rhizaeon/rhiz.html"            "$BASELINES/rhizaeon/"
    cp "$RUN_DIR/rhizaeon/rhiz_stdout.txt"      "$BASELINES/rhizaeon/"
    cp "$RUN_DIR/trace50/trace_stdout.txt"      "$BASELINES/trace50/"
    cp "$RUN_DIR/trace50/trace50_visualization_dossier.json" "$BASELINES/trace50/"
    cp "$RUN_DIR/trace50/acute_transmission_dyads.csv"       "$BASELINES/trace50/"
    cp "$RUN_DIR/trace50/autoclock_communities.csv"          "$BASELINES/trace50/"
    cp "$RUN_DIR/trace50/patient_surveillance_assignments.csv" "$BASELINES/trace50/"
    echo "[+] Baselines regenerated in $BASELINES"
    exit 0
fi

echo "[*] Diffing against $BASELINES"
FAILED=0
for f in rhiz.json rhiz_viz.json rhiz.html rhiz_stdout.txt; do
    if ! diff -q "$BASELINES/rhizaeon/$f" "$RUN_DIR/rhizaeon/$f" > /dev/null; then
        echo "  DIFF: rhizaeon/$f"
        diff "$BASELINES/rhizaeon/$f" "$RUN_DIR/rhizaeon/$f" | head -20
        FAILED=1
    fi
done
for f in trace_stdout.txt trace50_visualization_dossier.json \
         acute_transmission_dyads.csv autoclock_communities.csv \
         patient_surveillance_assignments.csv; do
    if ! diff -q "$BASELINES/trace50/$f" "$RUN_DIR/trace50/$f" > /dev/null; then
        echo "  DIFF: trace50/$f"
        diff "$BASELINES/trace50/$f" "$RUN_DIR/trace50/$f" | head -20
        FAILED=1
    fi
done

if [ "$FAILED" -eq 0 ]; then
    echo "[+] All CLI outputs byte-identical to baselines (timing normalized)"
else
    echo "[!] Baseline diffs found" >&2
    exit 1
fi
