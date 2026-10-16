# Rust-vs-Python parity golden files

Used by `tests/test_parity_golden.rs`. These files pin the Rust engine to the
Python reference on a fixed 200-sequence subset of the Washington, DC cohort
(Perez-Losada et al. 2017). The tests need neither Python nor tn93 at run time.

Every golden value was produced by running an external tool. None was typed in.
Regenerate them with

```bash
python3 scripts/regen_parity_golden.py            # goldens for the committed subset
python3 scripts/regen_parity_golden.py --select   # also re-derive subset.fasta and meta.csv
```

The second form needs the full DC alignment
(`validation/results/Perez-Losada-2017-28961263/msa.fasta`, sha256 `0f88f1e2…2aee`)
and the Python edge list `parity_2026-10-12/ingest/py_L1023_edges.csv`.
Run twice, the script gives the same `MANIFEST.sha256` both times.

## Cohort selection rule

1. Take the connected components of the 151 Python-supported DC edges at
   L = 1023 (`py_L1023_edges.csv`). Every member of every component with at
   least 3 members is kept: 22 clusters, 85 sequences.
2. Order the remaining DC sequences by `sha256(header)` and keep the first 115
   as background.
3. Write the result in DC file order, with rows unmodified. The alignment width
   stays at 2841, while the median non-gap length is 1023.

All 200 sequences are dated from their headers (199 to the day, 1 to the year
only, placed at year + 0.5). `meta.csv` holds `run_danno.parse_date` for each
header, at full precision.

## Files

| file | produced by |
|---|---|
| `subset.fasta` | the selection rule above |
| `meta.csv` | `run_danno.parse_date` (id, decimal year) |
| `tn93_RMYSKB.csv.gz` | `tn93 -q -t 1.0 -a RMYSKB -o tn93_RMYSKB.csv subset.fasta` (default `-l 100`). Data rows are sorted because tn93's row order varies between runs. Gzipped with mtime 0. |
| `golden_distances.csv` | the same 16,300 rows as `i,j,distance`, where `i < j` are row indices in `subset.fasta`. The distance text is tn93's, unchanged. |
| `run_danno.json`, `run_danno.log` | `validation/run_danno.py --msa subset.fasta --tn93 tn93_RMYSKB.csv`, unmodified (Python path (b): L = median non-gap length) |
| `golden.json` | see below |
| `MANIFEST.sha256` | sha256 of every file above |

`golden.json` contains the following.

- **Supported edges.** These are the dyads that `DannoEstimator.book_em_danno` returns with BF > 0, each with its BF, PEP and q. run_danno.py runs in-process so that they can be captured, with the script itself unmodified. A plain subprocess run writes a byte-identical `run_danno.json`. The capture also agrees exactly with the `run_danno_edges.py` variant.
- **Calibrated background.** `p_bg`, `r_bg` and `t_span`.
- **DANNO cluster partition.**
- **Static partition at d ≤ 0.015.** run_danno's static arm produces it. `hivnetworkcsv -f plain -t 0.015 -c` must give the identical cluster list, and the script exits if it does not.
- **CHIN values.** `trace50.chin.ChinEstimator(1.5)` supplies N_act, its CI and the Borel statistics (cap 12).

Tool versions: tn93 v1.1.0, hivclustering 1.9.2, Python 3.9.21, numpy 1.26.1,
scipy 1.11.3. The same versions are recorded in `golden.json`.

Golden values: L = 1023, width = 2841, 16,300 tn93 pairs, 73 supported edges
in 23 clusters (max 7), 19 static clusters, p_bg = 0.788907, r_bg = 15.1847,
N_act = 410.96.

## What the tests assert

The tests in `test_parity_golden.rs` check these properties.

- **TN93 pair set.** The internal TN93 pair set (`Alignment::compute_distance_matrix`, RMYSKB) equals tn93's.
- **TN93 distances.** Each internal distance is within 1e-6 of tn93's 6-significant-digit text. The measured maximum is 5.0e-7.
- **Default L and dates.** The default L is the golden L (1023, not the width). The header dates equal `meta.csv`.
- **Background calibration.** `calibrate_from_data` reproduces `p_bg`, `r_bg` and `t_span`. On golden distances the tolerance is 1e-9 relative (measured 4e-15). On internal distances it is 1e-4 (measured 6e-8).
- **Pipeline runs.** The checks below hold for `run_trace50`, which uses the internal TN93, and for `run_trace50_with_matrix`, which uses the golden distances:
  - The supported-edge set equals the golden set. Edges are compared as unordered ID pairs.
  - BF, PEP and q are within 1e-4 relative on the internal path (measured 3.6e-5) and 1e-9 on the matrix path (measured 4.6e-14).
  - The DANNO and static partitions equal the golden partitions, compared as sets of sets of IDs.
  - L is the golden L.
  - N_act, the CI and the Borel slope, R², p and cluster counts are within 1e-9.

## The harness catches known regressions

Each regression below was reintroduced in a scratch copy of the crate, outside
this repository, and `test_parity_golden` was run against it. The logs and
diffs are in
`validation/results/Perez-Losada-2017-28961263/parity_2026-10-12/phase2/parity-harness/`.

| perturbation | result |
|---|---|
| L reverted to the alignment width (`perturb_L_width`) | 4 of 5 tests fail. L is 2841 instead of 1023, and p_bg is 0.924 instead of 0.789. |
| unscorable pairs get d = 0 again (pre-C1; `perturb_overlap_zero`) | 3 of 5 fail. There are 19,900 pairs instead of 16,300, and the internal pipeline finds **0** supported edges instead of 73. |
| default ambiguity mode `resolve` instead of RMYSKB (`perturb_ambig_resolve`) | 3 of 5 fail. Distances are off by up to 3.8e-3, and the internal pipeline finds 78 edges instead of 73. |
