# RhizAeon Code Review — Proposed GitHub Issues (Factoring & Quality)

Code-factoring and code-quality issues in `feat/rhizaeon`, ordered by severity.
Scope: implementation quality only — statistical/methodology issues are tracked
separately in `RHIZAEON_REVIEW_ISSUES.md` (issues there are not repeated here).
Each entry is formatted as a ready-to-file issue title + body.

---

## Issue 1 — Critical

**Title:** `Hardcoded developer-machine checkpoint paths committed in adaptive.py`

**Body:**

`rhizaeon/adaptive.py` (~line 410) contains literal paths from a developer's
machine as fallback weight locations:

```python
default_ckpts = [
    str(Path.home() / ".cache" / "hyphaeon" / "axomeme_5_dim384_nonull.pt"),
    str(Path.home() / ".cache" / "hyphaeon" / "model.safetensors"),
    "/Users/sergei/Projects/TOGA_MEME/axomeme_5_dim384_nonull.pt",
    "/Users/sergei/Projects/TOGA_MEME/phylomlm_dim384.pt",
]
```

These are dead paths on every other machine, leak internal directory
structure, and silently change behavior depending on whose machine runs the
code.

**Suggested fix:** remove the `/Users/sergei/...` entries; resolve weights
exclusively through `aeon_core.weights.resolve_weights_path` or an explicit
`--weights` flag / env var.

---

## Issue 2 — Critical

**Title:** `Tier-2 neural hand-off silently disables itself on ImportError`

**Body:**

`adaptive.py` wraps the `aeon_core`/`torch` import in nested
`try/except ImportError` blocks that fall back to a `sys.path` hack and then
`return None`. If rhizaeon is installed without the monorepo checkout (the
normal PyPI case), Tier-2 analysis silently never runs — the user gets Tier-1
results with no indication the neural path was skipped.

**Suggested fix:** declare the dependency properly (or gate it behind an
extras_require like `rhizaeon[neural]`), and emit an explicit warning/log line
when Tier-2 is requested but unavailable. Never `return None` silently.

---

## Issue 3 — High

**Title:** `sys.path manipulation to locate aeon_core instead of a declared dependency`

**Body:**

`adaptive.py` (~line 388) does:

```python
repo_root = Path(__file__).resolve().parent.parent.parent.parent
aeon_core_path = repo_root / "aeon-core" / "src"
if aeon_core_path.exists() and str(aeon_core_path) not in sys.path:
    sys.path.insert(0, str(aeon_core_path))
```

This assumes a specific monorepo checkout layout, breaks under any installed
wheel, and mutates global interpreter state at call time. `embed.py` has a
similar pattern.

**Suggested fix:** depend on `hyphaeon-core` normally (it's on PyPI); delete
the path-insertion fallback entirely.

---

## Issue 4 — High

**Title:** `Breakpoint schema drift: 3+ incompatible representations normalized by duck-typing with fabricated defaults`

**Body:**

Breakpoints exist as (a) `FDABreakpoint` dataclass, (b) `PolishedBreakpoint`
dataclass, (c) plain dicts from `RhizAeonDetector`, and (d) JSON round-trips —
with divergent key names for the same fields:

| Concept | Spellings seen |
|---|---|
| recombinant | `recombinant`, `recombinant_taxon` |
| parents | `parent_1`/`parent_2`, `parent_left`/`parent_right` |
| position | `breakpoint`, `breakpoint_nt`, `polished_bp` |
| PIR | `l_pir`, `refined_pir` |
| Z-score | `ghost_z`, `kinetic_z` |
| plateau | `ci_left`/`ci_right`, `plateau_interval` |
| LL gain | `log_likelihood_gain`, `support_metrics.delta_ln_l` |

`visualizer.py` (~lines 215–260) contains a ~50-line normalization block that
duck-types both dicts and dataclasses — and when a field is missing it
*fabricates* values: `recombinant` defaults to `taxa[0]`,
`log_likelihood_gain` defaults to `15.0`, `kinetic_z` to `3.0`, `pir` to
`0.15`. A malformed record silently produces plausible-looking output.

**Suggested fix:** define one canonical `Breakpoint` dataclass (or TypedDict)
in a shared module; emit it from all detectors; make the visualizer accept
only that type and raise on missing fields.

---

## Issue 5 — High

**Title:** `Divergent engine attribute names force getattr-chains at every call site`

**Body:**

`PrefixDistanceEngine` exposes `seq_matrix`/`L`; `SNPCompressedPrefixEngine`
exposes `full_seq_matrix`/`L_nt`/`num_units`. Every consumer compensates with
repeated duck-typing — the pattern

```python
seq_mat = getattr(engine, "full_seq_matrix", getattr(engine, "seq_matrix", None))
```

appears 5+ times in `fda.py` alone, plus `cli.py`, `adaptive.py`, and
`visualizer.py`. `scale_coord = 3 if getattr(engine, "codon_aligned", False)
else 1` is likewise inlined at 4+ sites.

**Suggested fix:** give both engines a common protocol (properties
`seq_matrix`, `length_nt`, `coord_scale`); delete the getattr chains.

---

## Issue 6 — High

**Title:** `visualizer.py is a 2,443-line monolith with ~1,900 lines of embedded HTML/CSS/JS in an f-string`

**Body:**

`build_html_template` returns a single f-string spanning lines ~507–2443
containing the entire frontend (CSS, D3/vanilla JS, JSON injection). This is
uneditable without Python syntax risk, defeats frontend tooling (linting,
minification, source maps), and makes the f-string interpolation surface a
constant escaping hazard (`{`/`}` in JS must be doubled).

**Suggested fix:** move the template to a static asset file (Jinja2 or plain
`str.replace`/`json.dumps` injection of the data payload); keep
`visualizer.py` to data-bundle construction only.

---

## Issue 7 — Medium

**Title:** `Three parallel detection pipelines with duplicated orchestration and divergent defaults`

**Body:**

Recombination detection is orchestrated independently in:

- `segmentation.py::RhizAeonDetector` (Tier-1 recursive segmentation)
- `fda.py::run_recursive_partition_fda_screen` (RP-FDA)
- `adaptive.py` dual-architecture runner
- `visualizer.py` auto-detect path with its own hardcoded
  `min_len=40, max_depth=5, min_z=1.8, min_pir=0.08`

Each re-implements peak-finding → triplet evaluation → polishing with
different default constants, so results differ depending on which entry point
the user happened to call.

**Suggested fix:** extract one `detect_breakpoints(engine, config)` core;
make CLI, visualizer, and adaptive runner thin wrappers over a single
`DetectionConfig` dataclass.

---

## Issue 8 — Medium

**Title:** `torch.load(weights_only=False) on user-supplied checkpoint paths`

**Body:**

`embed.py::load_embedding_matrices` calls
`torch.load(path, map_location="cpu", weights_only=False)` on paths that can
come from CLI arguments. `weights_only=False` permits arbitrary pickle
execution — a malicious `.pt` file is a code-execution vector.

**Suggested fix:** prefer safetensors; if `.pt` must be supported, use
`weights_only=True` where possible and document the trust requirement.

---

## Issue 9 — Medium

**Title:** `sys.modules monkeypatching for NumPy 1.x/2.x unpickling bridge`

**Body:**

`embed.py` (~line 100) mutates `sys.modules` to alias `numpy._core` →
`numpy.core` before `torch.load`. This is global interpreter state mutation
inside a library call; it persists for the whole process and can mask real
import problems elsewhere.

**Suggested fix:** scope the shim (e.g. a context manager that restores
`sys.modules`), or migrate checkpoints to safetensors so the bridge is
unnecessary.

---

## Issue 10 — Medium

**Title:** `except Exception: pass` swallows weight-resolution failures`

**Body:**

`embed.py::resolve_checkpoint_path` (~line 76) and `adaptive.py` (~line 407)
catch bare `Exception` and continue silently. A broken `aeon_core.weights`
install produces a confusing downstream `FileNotFoundError` instead of the
real error.

**Suggested fix:** catch specific exceptions, or log-and-reraise with context.

---

## Issue 11 — Medium

**Title:** `No model-variant selection: --weights only, no --variant, no list-models, no env vars`

**Body:**

`cli.py` (~line 70) exposes `--weights` ("Custom path to HyphAeon
transformer weights (.pt)") but nothing else: no `--variant`/
`--model-variant`, no `list-models` subcommand, and no
`RHIZAEON_WEIGHTS`/`HYPHAEON_WEIGHTS` env-var support. `adaptive.py`
(~line 404) calls `resolve_weights_path(weights=cfg.weights_path)`
without a variant parameter, so users can only ever get the default
HF variant unless they hand-manage checkpoint files.

Both hyphaeon (`--weights` + `--variant`) and chronaeon (`-w/--weights`
+ `--model-variant`, `list-models`, `CHRONAEON_*`/`HYPHAEON_*` env vars)
expose fuller weight-management surfaces — the three CLIs should share
one convention.

**Suggested fix:** add `--variant` (matching hyphaeon spelling) and a
`list-models` subcommand; honor `RHIZAEON_WEIGHTS`/`RHIZAEON_VARIANT`
with `HYPHAEON_*` fallbacks (same pattern as chronaeon `cli.py:20-23`);
pass `variant` through to `resolve_weights_path` in `adaptive.py`.

---

## Issue 12 — Low

**Title:** `Repeated deferred imports of parse_fasta inside export.py function bodies`

**Body:**

`export.py` re-imports `from rhizaeon.tensor import parse_fasta` inside three
separate functions (lines ~94, 153, 189). If this was to avoid a circular
import, the cycle should be broken structurally; otherwise hoist to module
level.

**Suggested fix:** single top-level import; if circularity exists, move the
shared parser to a leaf module.

---

## Issue 13 — Low

**Title:** `Magic-number detection defaults duplicated between CLI and visualizer auto-detect`

**Body:**

`visualizer.py` hardcodes `min_len=40, max_depth=5, min_z=1.8, min_pir=0.08`;
`cli.py` exposes overlapping flags with different defaults. The two paths can
emit different breakpoint sets for the same alignment.

**Suggested fix:** one `DetectionConfig` with canonical defaults (see Issue 7);
both entry points consume it.

---

## Issue 14 — Low

**Title:** `No defined public API surface (__init__.py star-imports everything)`

**Body:**

`rhizaeon/__init__.py` re-exports symbols from all submodules with no
`__all__`, so the public API is implicit and any internal rename is a silent
breaking change for downstream users.

**Suggested fix:** define `__all__` explicitly; keep internal helpers
underscore-prefixed and unexported.

---

## Issue 15 — Low

**Title:** `print()-based reporting in cli.py mixes presentation with logic`

**Body:**

`cli.py` builds the inference report via inline `print` blocks with `[*]`
prefixes and banner rules. Acceptable for a CLI, but the report text is
untestable and unlocalizable, and there's no `--quiet`/verbosity control.

**Suggested fix:** route through `logging` or a small `report()` function
returning a string; add `-q/--quiet`.

---

## Issue 16 — Low

**Title:** `Global np.random.seed() in benchmark generator affects caller state`

**Body:**

`benchmarks/.../generator_envelope.py::VectorizedMarkovEvolver.__init__` calls
`np.random.seed(seed)` — mutating global RNG state for the whole process.

**Suggested fix:** use a `np.random.default_rng(seed)` instance passed through
the evolver.
