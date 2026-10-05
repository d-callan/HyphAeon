# Aeon-Core Code Review — Proposed GitHub Issues (Factoring & Quality)

Code-factoring and code-quality issues in `aeon-core/` (reviewed on
`feat/rhizaeon`), ordered by severity. Same format as
`RHIZAEON_CODE_QUALITY_ISSUES.md`.

---

## Issue 1 — High

**Title:** `sys.modules monkeypatching at import time in weights.py (numpy._core bridge)`

**Body:**

`weights.py` (~lines 24–29) mutates `sys.modules` at module import:

```python
try:
    import numpy as np
    if not hasattr(np, '_core') and hasattr(np, 'core'):
        sys.modules['numpy._core'] = np.core
        sys.modules['numpy._core.multiarray'] = np.core.multiarray
except Exception:
    pass
```

Because this runs on `import aeon_core`, every downstream process inherits
the aliased modules — it can mask genuine import errors and affects
unrelated unpickling in the same interpreter. The same pattern is
copy-pasted into `rhizaeon/embed.py` (function scope there, module scope
here).

**Suggested fix:** scope the shim to the `torch.load` call site (context
manager that restores `sys.modules`), or migrate checkpoints to
safetensors so the bridge is unnecessary.

---

## Issue 2 — High

**Title:** `Newick extraction in dataset.py is regex/line-scraping with 4+ silent except-pass fallback stages`

**Body:**

`extract_tree_from_string_or_file` (~lines 385–430) tries, in order: BEAST
XML parse → regex `tree\s+[^=]+=\s*\(...` → per-line `startswith('(')`
heuristic → whole-content parse — each stage wrapped in
`except Exception: pass`. A malformed tree silently falls through all
stages and returns `None`, surfacing far downstream as a missing-tree
error with no indication of what was tried.

**Suggested fix:** collect per-stage errors and report them when all stages
fail; tighten the regexes to actual Nexus/Newick grammar or use a real
parser for each candidate format.

---

## Issue 3 — Medium

**Title:** `estimate_tree_branch_lengths_hyphy shells out to HyPhy via a generated .bf batch file and stdout scraping`

**Body:**

`dataset.py` (~line 500) writes a HyPhy batch file to disk, runs
`hyphy temp_bf`, then extracts the tree by `stdout.find('(')` /
`rfind(')')`. HyPhy is an undeclared external binary dependency, the
output contract is stdout text scraping, and failure prints a warning and
returns `None` — callers can't distinguish "HyPhy missing" from "HyPhy
crashed."

**Suggested fix:** detect HyPhy via `shutil.which` up front and raise a
clear error when required; parse stdout with a real Newick parse attempt;
use `tempfile` for the .bf file.

---

## Issue 4 — Medium

**Title:** `get_live_available_memory returns hardcoded 2/3/4 GB guesses on any probing failure`

**Body:**

`inference.py` (~lines 31–56) nests `except Exception` fallbacks: CUDA
probe failure → 3 GB; MPS failure → 4 GB; psutil missing → `sysconf` →
2 GB. These silently wrong budgets feed `compute_adaptive_safe_batch_size`,
so a probing failure can over- or under-allocate with no signal.

**Suggested fix:** return `Optional[int]` and let callers decide; log which
fallback fired; make the constants named module-level config.

---

## Issue 5 — Medium

**Title:** `dataset.py is a 49 KB god-module mixing FASTA, BEAST XML, Nexus, Newick, TN93, HyPhy, MDS, and downsampling`

**Body:**

One module owns: alignment parsing, genetic-code tables, BEAST XML,
tree extraction heuristics, tn93 subprocess wrapper, HyPhy batch-file
wrapper, MDS coordinates, Faith-PD downsampling, and duplicate pruning.
Any change risks touching unrelated paths; the file is the single biggest
merge-conflict surface in the repo.

**Suggested fix:** split into `dataset/` subpackage: `alignment.py`,
`tree_io.py`, `distances.py` (tn93 + fast dist), `external.py` (subprocess
wrappers), `sampling.py` (prune/downsample).

---

## Issue 6 — Medium

**Title:** `tn93 distance computation writes temp FASTA + CSV per call`

**Body:**

`compute_tn93_distance_matrix` (~line 734) writes a sanitized temp FASTA,
runs the `tn93` binary, parses the CSV output — per invocation. The
threshold>1.0 retry for stock tn93 ≤1.0.15 is handled, but the temp-file
pattern repeats the same overhead triage.py already has.

**Suggested fix:** keep the binary path but batch inputs; ensure temp files
are under `tempfile.TemporaryDirectory`; document the Python fallback's
performance envelope.

---

## Issue 7 — Low

**Title:** `print()-based progress in weights.py library code`

**Body:**

`resolve_weights_path` prints download progress (`[*] Downloading...`,
`[✓] Weights cached at:`) from library code — no way for embedding
applications to silence or redirect it.

**Suggested fix:** use `logging.getLogger("aeon_core.weights")`.

---

## Issue 8 — Low

**Title:** `except Exception: pass around HF cache lookup hides cache-layer errors`

**Body:**

`weights.py` (~line 146) wraps `try_to_load_from_cache` in bare
`except Exception: pass` — a corrupted cache dir is indistinguishable from
a cache miss and triggers a full re-download.

**Suggested fix:** catch `Exception` but log at debug level.

---

## Issue 9 — Low

**Title:** `__init__.py re-exports ~40 symbols with no __all__`

**Body:**

`aeon_core/__init__.py` star-exports the entire public surface without
`__all__`; private helpers imported into submodule namespaces can leak.

**Suggested fix:** add explicit `__all__`.

---

## Issue 10 — Low

**Title:** `pyproject.toml name is aeon-core (pre-rename); rebase to hyphaeon-core needed`

**Body:**

On this branch the dist name is still `aeon-core`; main renamed it to
`hyphaeon-core` (import stays `aeon_core`). Rebase must update
`[project] name` and all downstream `dependencies` entries.

**Suggested fix:** handled at rebase; verify `pip install hyphaeon-core`
resolves `aeon_core` imports.
