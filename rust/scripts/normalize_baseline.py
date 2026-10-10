#!/usr/bin/env python3
"""Normalize nondeterministic fields out of CLI outputs for baseline diffing.

Handles:
- JSON files: strips keys whose names match TIMING_KEYS (recursively)
- Text files (stdout logs): regex-replaces duration patterns like "1.19ms",
  "67.00ms", and absolute paths to the run's output directory
- HTML files: contains an embedded JSON blob; timing keys appear inline, so a
  regex zeroes them without parsing the whole document

Usage: normalize_baseline.py <file> [<out_dir_placeholder_base> ...]
Normalizes each file in place.
"""
import json
import re
import sys
from pathlib import Path

TIMING_KEYS = re.compile(r"^(run_time_ms|execution_time_seconds|duration_ms|elapsed_ms)$")
# Durations in prose output: "1.19ms", "67.00 ms", "0.000038595"
DURATION_RE = re.compile(r"\b\d+\.\d+\s*(?:ns|µs|ms|s)\b")
# Embedded timing keys in HTML blobs: "run_time_ms": 66.99
EMBEDDED_TIMING_RE = re.compile(
    r'("(?:run_time_ms|execution_time_seconds|duration_ms|elapsed_ms)"\s*:\s*)[\d.eE+-]+'
)


def strip_timing_keys(obj):
    if isinstance(obj, dict):
        return {
            k: strip_timing_keys(v)
            for k, v in obj.items()
            if not TIMING_KEYS.match(k)
        }
    if isinstance(obj, list):
        return [strip_timing_keys(v) for v in obj]
    return obj


def normalize_file(path: Path, out_dirs: list[Path]) -> None:
    text = path.read_text()
    suffix = path.suffix.lower()

    if suffix == ".json":
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            data = None
        if data is not None:
            data = strip_timing_keys(data)
            path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
            return

    # Text / HTML path: zero embedded timing keys, durations, and out-dir paths
    text = EMBEDDED_TIMING_RE.sub(r"\g<1>0", text)
    text = DURATION_RE.sub("X.XXms", text)
    for out_dir in out_dirs:
        text = text.replace(str(out_dir), "<OUT_DIR>")
    path.write_text(text)


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__, file=sys.stderr)
        return 2
    files = [Path(a) for a in sys.argv[1:] if not a.startswith("--out-dir=")]
    out_dirs = [
        Path(a.split("=", 1)[1]) for a in sys.argv[1:] if a.startswith("--out-dir=")
    ]
    for f in files:
        normalize_file(f, out_dirs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
