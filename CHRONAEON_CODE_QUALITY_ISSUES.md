# ChronAeon Code Review — Proposed GitHub Issues (Factoring & Quality)

Code-factoring and code-quality issues in `chronaeon/` (reviewed on
`feat/rhizaeon`), ordered by severity. Same format as
`RHIZAEON_CODE_QUALITY_ISSUES.md`.

---

## Issue 1 — Critical

**Title:** `Shell injection + implicit PATH probing in geo.py FastTree invocation`

**Body:**

`chronaeon/geo.py` (~line 208) builds a shell command by f-string
interpolation and runs it with `shell=True`:

```python
cmd = f"{fasttree_bin} -nt -gtr {alignment_path} > {tmp_tree}"
subprocess.run(cmd, shell=True, check=True, ...)
```

- `alignment_path` is user-controlled → shell injection via crafted filename.
- The binary is discovered by probing hardcoded paths
  (`/usr/local/bin/FastTree`, `/usr/bin/FastTree`, `FastTree`, `fasttree`)
  with a `-help` subprocess each — slow and order-dependent.
- The output tree is written next to the input alignment
  (`with_suffix(".fasttree.nwk")`) then deleted — pollutes the user's
  directory and races on concurrent runs.

**Suggested fix:** `shutil.which("FastTree")` for discovery; argv-list
`subprocess.run` (no `shell=True`); write the tree to `tempfile`.

---

## Issue 2 — High

**Title:** `autoclock.py is a 2,600-line monolith with two near-duplicate engine classes`

**Body:**

`autoclock.py` (~120 KB) contains `AutoClockDeconvolution` and
`HierarchicalAutoClock`, which independently re-implement the same lifecycle:
`_log`, `load_and_validate`, `calibrate_*`, `plot_diagnostics`, `run`, plus
duplicated spectral-splitting and manifold-fallback logic. Even the `_log`
methods differ only by `flush=True`.

**Suggested fix:** extract a shared `ClockEngineBase` (logging, validation,
plot scaffolding); split the file into `autoclock/` package modules
(spectral, deconvolution, hierarchical, plotting).

---

## Issue 3 — High

**Title:** `cli.py is a ~1,000-line monolith mixing parsing, orchestration, and report formatting`

**Body:**

`cli.py` (~63 KB) holds all `cmd_*` handlers with inline business logic —
alignment loading, date matching, model dispatch, and formatted report
printing all in one function per subcommand. Untestable without subprocess
invocation.

**Suggested fix:** move each `cmd_*` body into a callable in the relevant
module returning a result dict; keep `cli.py` to argparse + printing.

---

## Issue 4 — High

**Title:** `Broad except Exception fallbacks silently degrade model fitting`

**Body:**

Multiple sites swallow all exceptions and continue with a degraded result:

- `dating_pipeline.py` (~671, 796, 891, 904): spline/power-law/latent fits
  fall back to linear on any error — a bug in the spline fit is reported as
  "fell back to linear," masking real failures.
- `autoclock.py` (~719): header sniff `except Exception: pass`.
- `dating_plots.py` (~245, 254, 323): plot failures silently skipped.
- `geo.py` (~89, 202, 792): transformer extraction failure → silent distance
  fallback.
- `triage.py` (~91): tn93 subprocess failure → silent NumPy fallback.

**Suggested fix:** catch specific exceptions (`LinAlgError`, `ValueError`,
`FileNotFoundError`); log at warning level with the original exception
chained; add a `--strict` mode that re-raises.

---

## Issue 5 — High

**Title:** `-o/--output` has inconsistent semantics across subcommands: output prefix for date/phylogeo, verbatim path for triage/sketch/align`

**Body:**

The same flag means two different things depending on the subcommand:

- `date` and `phylogeo` pass `args.output` as `output_prefix`
  (`cli.py:69`, `cli.py:297`), and the pipelines emit *multiple* files
  derived from it: `PREFIX.json`, `PREFIX.csv`, `{prefix}_diagnostic.pdf`
  (`dating_pipeline.py:3152-3195`), `PREFIX.json` + `PREFIX.pdf`
  (`geo.py:898-918`). `-o out` silently produces `out.json`, `out.csv`,
  `out_diagnostic.pdf`.
- `triage` treats `-o` as a verbatim CSV path (`cli.py:465`), `sketch`
  as verbatim JSON (`cli.py:614`), `align` as verbatim FASTA
  (`cli.py:647`), `autoclock` as verbatim JSON but only when the suffix
  is `.json` (`cli.py:576-585`), and `r0` as verbatim JSON while also
  deriving the alluvial path via `os.path.splitext` (`cli.py:225`).

Consequences: users can't predict what `-o out` produces; scripts and
Galaxy wrappers must know per-subcommand semantics — the chronaeon Galaxy
wrappers currently need `from_work_dir` on JSON outputs because the tool
writes a prefix-derived filename rather than the path Galaxy provides.

**Suggested fix:** standardize `-o/--output PATH` to mean the verbatim
JSON output file across all subcommands (matching sketch/triage/align),
and move the multi-file prefix behavior to a separate `--output-prefix
PREFIX` flag emitting `PREFIX.json` / `PREFIX.csv` / `PREFIX.pdf`.
Backward-compatible for the common `-o out.json` case, and lets Galaxy
pass `-o '$json_output'` directly, dropping `from_work_dir`.

---

## Issue 6 — Medium

**Title:** `Weights/variant CLI surface inconsistent with hyphaeon: autoclock lacks --model-variant; flag spelling differs`

**Body:**

Model-selection flags don't match the hyphaeon CLI conventions:

- `date` and `phylogeo` expose `-w/--weights` + `--model-variant`
  (`cli.py:714-715`, `741-742`), but `autoclock` accepts `--weights`
  with **no** `--model-variant` (`cli.py:805`) — users can point at a
  local checkpoint but can't select an HF variant.
- Flag spelling differs from hyphaeon: chronaeon uses `--model-variant`
  (dest=`variant`) where hyphaeon uses `--variant`.
- `triage`, `sketch`, `align`, `r0`, and `dynamics` expose neither flag.
- Env vars exist (`CHRONAEON_WEIGHTS`/`CHRONAEON_VARIANT` with
  `HYPHAEON_*` fallbacks, `cli.py:20-23`) — good, but only honored by
  the subcommands that define the flags.

**Suggested fix:** expose `--weights` + `--variant` (hyphaeon spelling)
on every subcommand that consumes the transformer, keep
`--model-variant` as an alias for backward compatibility, and route all
of them through the same `resolve_weights_path(weights, variant)` call.

---

## Issue 7 — Medium

**Title:** `np.random.seed(seed) global seeding in geo.py permutation test`

**Body:**

`geo.py` (~line 359) calls `np.random.seed(seed)` inside the transmission-
route permutation test — mutating global RNG state for the whole process and
making results non-reproducible when called alongside other seeded code.

**Suggested fix:** `rng = np.random.default_rng(seed)` threaded through the
permutation loop.

---

## Issue 8 — Medium

**Title:** `dating.py facade re-exports private _-prefixed functions as public API`

**Body:**

`dating.py` re-exports `_spectral_eigh`, `_gls_fit`, `_generalized_r2`,
`_compute_precision_weighted_ensemble`, `_select_clock_model`,
`_compute_taxon_predictions`, `_export_dating_results` — private helpers
become de-facto public API, so internal refactors break downstream users.

**Suggested fix:** drop private names from the facade (and from
`__init__.__all__`); expose only the `run_*`/`compute_*` public surface.

---

## Issue 9 — Medium

**Title:** `Silent alignment mutation: auto-trim to divisible-by-3 in dating_io.py`

**Body:**

`verify_coding_alignment` (~line 62) silently drops trailing nucleotides when
`L % 3 != 0`. It prints a notice, but the returned matrix no longer matches
the input file — downstream coordinates (site numbers in reports) shift
relative to the user's alignment.

**Suggested fix:** require explicit opt-in (`--trim-to-codon`) or raise;
at minimum report the trimmed length in the output metadata.

---

## Issue 10 — Medium

**Title:** `Per-query subprocess + temp-file overhead in triage.py TN93 path`

**Body:**

`triage.py` (~lines 70–95) writes two temp FASTAs and shells out to `tn93`
per query sequence, then parses CSV with pandas — despite the module's own
comment claiming "Zero disk I/O, zero subprocess overhead" for the fallback.
On failure it silently falls back (`except Exception: pass`).

**Suggested fix:** batch all queries into one tn93 call; use `tempfile`
contexts; log the fallback transition.

---

## Issue 11 — Low

**Title:** `Duplicated date-parsing logic between dating_io.py and aeon_core.temporal`

**Body:**

`dating_io.py` implements `parse_header_timestamp`, `_parse_timestamp_flexible`,
`parse_sample_dates` while also importing `parse_date_to_decimal`,
`parse_dates_from_auspice_json`, `extract_date_from_string` from
`aeon_core.temporal` — two partially overlapping date-parsing stacks.

**Suggested fix:** consolidate on `aeon_core.temporal`; keep only
chronaeon-specific header-regex logic locally.

---

## Issue 12 — Low

**Title:** `print()-based progress/reporting throughout; no logging or verbosity control`

**Body:**

All modules report via `print` with `[*]`/`[✓]`/`[!]` prefixes; `quiet` flags
exist only on some classes (`autoclock`) and are inconsistent (`flush=True`
in one `_log`, not the other). Library callers can't silence or redirect
output.

**Suggested fix:** standardize on `logging` with a `chronaeon` logger; keep
`print` only for the final CLI report block.

---

## Issue 13 — Low

**Title:** `PATHOGEN_PRESETS embedded as a large inline dict in r0.py`

**Body:**

`r0.py` (~line 38) embeds pathogen generation-time presets as a hardcoded
dict in source. Adding a pathogen requires a code change and release.

**Suggested fix:** move presets to a packaged data file (JSON/YAML) loaded via
`importlib.resources`; allow user-supplied preset files.
