# Phase 2 Baseline — Test & Reference State

Recorded 2026-10-10, conda env `axomeme`, Rust toolchain in axomeme.
This is the regression gate for `feat/phase2-dedup` work items: any refactor
must reproduce these numbers exactly.

## Rust workspace

`cargo test --workspace` — **all green** (see per-crate counts below).

| Crate | Unit tests | Notes |
|---|---|---|
| aeon-manifold | 10 | ln_gamma, gammaincc, poisson_cdf, Jacobi 2x2 + P3-Laplacian Fiedler, plus **new scipy golden tests** for norm_ppf (9 quantiles @1e-8, tails @1e-4), student_t_ppf (C-F expansion, df-scaled tolerances), linregress (slope/intercept/r/std_err vs scipy; p_value asserted small only — impl uses normal approx, not t-dist) |
| rhizaeon-core | 25 | **new**: `fasta::tests` (encode_char table incl. ambiguity→0 collapse, parse_fasta strict-length + first-token header + blank-line skip, extract_informative_snp_alignment map), `screener::tests` (median, normal_ppf goldens + boundary clamp contract: never ±inf) |
| trace50-core | 27 | **new**: `tn93` (IUPAC mask table, nuc_int, base-prob sums, `from_fasta_str` ragged→pad-to-max with `-` + NaN-date contract + dual-encoding bit↔int consistency, `parse_date_from_header` decimal/ISO/out-of-range/NAN), `chin` (percentile endpoints, numpy-linear interpolation on rank=pct/100·(n−1), degenerate inputs) |
| trace50-core (test_pipeline.rs) | 2 | end-to-end run_trace50 smoke |
| rhizaeon-core (other suites) | 4 + 2 | empirical benchmarks + integration |

### CLI byte-exact baselines

`rust/scripts/compare_cli_baseline.sh` regenerates CLI outputs and diffs
against committed goldens in `rust/tests/baselines/{rhizaeon,trace50}/`.
Normalization (via `rust/scripts/normalize_baseline.py`): strips
`run_time_ms`/`execution_time_seconds`/`duration_ms`/`elapsed_ms` JSON keys,
prose durations (`ns|µs|ms|s`), and absolute out-dir paths.

- rhizaeon: `rhizaeon --input rhizaeon/tests/empirical/data/hiv1_kal153_crf02_ag.fasta --output … --viz … --html …` → rhiz.json, rhiz_viz.json, rhiz.html, rhiz_stdout.txt
- trace50: `trace50 --fasta tests/fixtures/pair_test.fasta --out-dir …` → trace50_visualization_dossier.json + 3 CSVs + trace_stdout.txt (dates parsed from headers; fixture copied from upstream `trace-5.0/data/`)

Regen goldens deliberately: `./scripts/compare_cli_baseline.sh --regen`.

## Python packages (pytest, env `axomeme`)

| Package | Result |
|---|---|
| aeon-core | **158 passed, 2 skipped** (0.52s) |
| chronaeon | **189 passed**, 6 warnings (31.1s) |
| hyphaeon | **161 passed, 5 skipped** (48.6s) |

Known non-fatal warning: `chronaeon/autoclock.py:254` `np.power(..., where=deg>0)`
without `out=` — uninitialized-memory warning, pre-existing, leave for W11.

## Cross-language conformance (`rust/tests/conformance/`)

Driver: `cargo build --release --bin conformance <fasta> <out.json>` dumps
headers, parsed dates, flat TN93 matrix, and autoclock community tree.
Pytest: `python -m pytest rust/tests/conformance/ -q` — **6 passed**.

Documented contract divergences (pinned as goldens, NOT bugs to fix silently):

- **TN93 saturation**: Python `tn93` package caps at 1.0; Rust closed-form
  returns finite estimates (pair_test MASE pair: py=1.0, rust=0.7535).
- **TN93 solver gap**: divergent pairs differ ≤1.01e-4 rel (outbreak) /
  3.56e-4 rel (hcv); close pairs (d≤2%) agree within float32 quantization.
- **Date regex coverage**: Python `parse_header_timestamp` requires
  `\d{4}\.\d{2,4}` for decimal years → NaN on all `|YYYY.D` outbreak
  headers; Rust's float-token scan accepts them.
- **First-token-wins**: Rust takes the first in-range token
  (`HCV_REPEAT_2015|2015-01-10` → 2015.0); Python prefers the ISO match.
- **Fractional-year convention**: Rust `(m-1)/12 + (d-1)/365.25` vs Python
  day-of-year — same date, ~4e-3 apart on ISO headers.
- **Autoclock partitions differ** on identical (dates, D) inputs — Rust
  (modularity Cheeger + 0.65 conductance) vs Python split criteria are
  different algorithms; each side's partition is pinned per-engine.

## Dependency snapshots

- `rust/snapshots/cargo-tree.txt` — `cargo tree --workspace` (106 lines)
- `rust/snapshots/pip-list-axomeme.txt` — `pip list --format=freeze` sorted (124 pkgs)

Refresh after any dependency change; diff before/after each work item.
