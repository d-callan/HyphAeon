# HyphAeon Code Review — Proposed GitHub Issues (Factoring & Quality)

Code-factoring and code-quality issues in `hyphaeon/` (reviewed on
`feat/rhizaeon`), ordered by severity. Same format as
`RHIZAEON_CODE_QUALITY_ISSUES.md`. The `recombine` delegation issue is
already filed in the RhizAeon document and not repeated here.

---

## Issue 1 — High

**Title:** `cli.py is a ~2,000-line monolith; every cmd_* mixes parsing, model loading, inference, and report formatting`

**Body:**

`cli.py` (~80 KB) implements all subcommand handlers inline — `cmd_meme`,
`cmd_busted`, `cmd_phenotype`, etc. each do device selection, weight
resolution, alignment prep, inference, and formatted printing in one body.
None of it is unit-testable without subprocess invocation, and the same
setup boilerplate is copied between handlers.

**Suggested fix:** extract per-command `run_*` functions into their domain
modules returning result dicts; keep `cli.py` to argparse + output
formatting.

---

## Issue 2 — High

**Title:** `Model-loading + alignment-prep boilerplate duplicated verbatim across 6+ call sites`

**Body:**

The sequence `get_device` → `resolve_weights_path`/`load_model` →
`load_alignment_and_tree`/`prepare_alignment` → `precompute_tree_cache`
is copy-pasted in `epistasis.py` (twice, ~lines 661 and 743), `disease.py`
(~430), `filter.py` (~125), and `cli.py` (3+ sites). Divergent copies have
already appeared: `cmd_busted` bypasses `load_model` entirely and
hand-builds `PhyloAxialTransformer` + `load_state_dict(strict=False)` +
prefix-stripping for the BUSTED head — `strict=False` silently ignores
mismatched checkpoint keys.

**Suggested fix:** one `load_model_and_alignment(args)` helper in
`aeon_core.inference` (or a hyphaeon `_common.py`); make `cmd_busted` use
`load_model` and add a `strict=True` head-loading path with explicit key
validation.

---

## Issue 3 — High

**Title:** `np.random.seed(seed) global seeding in phenotype.py permutation test`

**Body:**

`phenotype.py` (~line 346) calls `np.random.seed(seed)` inside the
phylogenetic permutation routine — mutating process-global RNG state, so
results depend on call order and contaminate any other seeded code in the
same process.

**Suggested fix:** `rng = np.random.default_rng(seed)` threaded through the
permutation loop.

---

## Issue 4 — Medium

**Title:** `Permutation p-value silently set to None on any exception`

**Body:**

`phenotype.py` (~line 470) wraps the gene-level permutation null in
`except Exception` → `null_rhos = None; gene_p_perm = None`. A bug in the
null computation produces a result object with a missing p-value and no
warning — indistinguishable from "test not run."

**Suggested fix:** catch specific exceptions; emit a warning with the
chained exception; mark the field explicitly as failed vs. skipped.

---

## Issue 5 — Medium

**Title:** `161 KB of committed HyPhy result-cache JSONs inside the package source tree`

**Body:**

`hyphaeon/src/hyphaeon/model_eval/_cache/` contains 12 committed
`meme_*_hyphy2.5.101(MP).json` files (~161 KB total) — test fixtures living
inside `src/`. They bloat the repo, risk being packaged into the wheel
depending on `package_data` config, and put eval data under the import
path.

**Suggested fix:** move `_cache/` to `tests/fixtures/` or
`model_eval/fixtures/` outside `src/`; add a `.gitignore` for regenerated
caches.

---

## Issue 6 — Medium

**Title:** `splits.py weight resolution falls back to a literal "model.safetensors" path on any error`

**Body:**

`splits.py` (~line 169): after probing candidate paths, a broad
`except Exception` sets `weights_path = "model.safetensors"` — a relative
path that almost certainly doesn't exist, deferring a confusing
`FileNotFoundError` to `load_model` far from the real cause.

**Suggested fix:** let `resolve_weights_path` raise its own informative
error; remove the string fallback.

---

## Issue 7 — Medium

**Title:** `temporal.py swallows CSV-parse failure then retries with different defaults`

**Body:**

`temporal.py` (~line 79): `except Exception` around `pd.read_csv(...,
engine='python')` retries with `pd.read_csv(src_str)` — if the file is
genuinely malformed, the second parse error surfaces without the context
of the first; if the first failed for a non-format reason (permissions,
OOM), the retry wastes it.

**Suggested fix:** sniff the delimiter once (`csv.Sniffer` or extension +
fallback on `ParserError` only).

---

## Issue 8 — Low

**Title:** `except Exception: pass` on GPU cache cleanup in filter.py`

**Body:**

`filter.py` (~lines 44, 49) silently ignores failures of
`torch.mps.empty_cache()`/`torch.cuda.empty_cache()`. Benign, but a bare
`pass` also hides real device errors during debugging.

**Suggested fix:** `except RuntimeError` with a debug-level log.

---

## Issue 9 — Low

**Title:** `matplotlib imported optionally at module level in temporal.py`

**Body:**

`temporal.py` (~line 43) does `matplotlib.use('Agg')` + imports under
`try/except ImportError` setting `HAS_MATPLOTLIB` — importing the package
pulls matplotlib even for non-plotting callers, and backend selection at
import time can conflict with embedding applications.

**Suggested fix:** defer matplotlib import into the plotting functions;
set the backend inside the call, not at module import.

---

## Issue 10 — Low

**Title:** `pyproject.toml still depends on aeon-core>=0.1.0 (pre-rename dist name)`

**Body:**

On this branch `hyphaeon/pyproject.toml` lists `aeon-core>=0.1.0`; main
renamed the distribution to `hyphaeon-core` (import stays `aeon_core`).
Rebase will need the dependency updated or the wheel won't resolve.

**Suggested fix:** update to `hyphaeon-core>=0.1.0` on rebase.
