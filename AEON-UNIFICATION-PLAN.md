# Aeon Unification Plan: Shared AutoClock, Crate Collapse, Maturin aeon-core

**Repo:** github.com/veg/HyphAeon (monorepo)
**Audience:** implementing agent with no prior context on this codebase
**Status:** plan, unimplemented

## Goals

1. **One autoclock implementation in Rust**, shared by `trace50` (Rust) and `chronaeon` (Python via PyO3 bindings).
2. **Collapse each Rust sister's `-core`/`-cli` pair into a single crate** (`rhizaeon`, `trace50`) with the binary feature-gated so `*-wasm` consumers do not pull `clap`/`rayon`.
3. **Fold `aeon-py` (PyO3 crate) into `aeon-core` (Python package)** as a maturin-built extension inside the same wheel — one PyPI artifact instead of two.
4. Extend publishing workflows (PyPI wheel matrix, crates.io chain) and update all install docs.

## Current state (verified)

- Rust workspace root: `rust/Cargo.toml`, members: `libs/aeon-manifold`, `libs/aeon-py`, `rhizaeon/crates/rhizaeon-{core,cli,wasm}`, `trace-5.0/crates/trace50-{core,cli,wasm}`. Workspace deps declared with `path = ..., version = ...` (crates.io-ready).
- `trace50-core/src/autoclock.rs` (~545 LoC) exports `SteveCommunity`, `compute_fieller_mrca`, `compute_bartlett_neff`, `fit_clock_community`, `optimize_fiedler_modularity`, `recursive_autoclock_deconvolution`. Used by `pipeline.rs:347` (the STEVE stage). Rust impl = `git log`-verified descendant of the standalone trace-5.0 repo.
- `chronaeon/src/chronaeon/autoclock.py` (~2600 LoC): kernel functions `fit_clock`, `recursive_spectral_autoclock`, `classify_community`, `classify_leaf_community`, `detect_contemporaneous_dyads`, `select_adaptive_n_landmarks`, plus orchestration classes (`AutoClockDeconvolution`, `HierarchicalAutoClock`) doing CSV/JSON/figure output. Exported via `chronaeon/__init__.py`.
- `aeon-py` (`rust/libs/aeon-py`, `crate-type = ["cdylib"]`, module name `aeon_py`) exports: `percentile`, `median`, `parse_date_from_header`, `char_to_iupac_mask`, `iupac_base_probabilities`, `fasta_tokenize`, `fasta_normalize_lengths`, `tn93_distance_matrix`, `tn93_cross_distance_matrix`, `pvals_from_lrt_meme`, `pvals_from_lrt_self_liang`, `benjamini_hochberg`, `cauchy_combination_p`. Python consumers use `import aeon_py` guarded by try/except (see `aeon-core/src/aeon_core/dataset.py`, test probes `_HAS_TN93` in `aeon-core/tests/test_load_alignment.py`, `hyphaeon/tests/test_{integration,splits}.py`).
- `aeon-core` = pure-Python setuptools package, `pyproject.toml` at `aeon-core/pyproject.toml`, PyPI name **`hyphaeon-core`**, src-layout `aeon-core/src/aeon_core/`.
- `rhizaeon-cli` = 2 bins (`rhizaeon` from `main.rs`, `export_figure1_data` from `bin/`); `trace50-cli` = 1 bin (`trace50`). Both wasm crates depend only on their `-core` and wasm-bindgen stack.
- `trace50-core` already has `parallel = ["dep:rayon"]` feature. `rhizaeon-core` uses rayon indirectly via `rhizaeon-cli`.
- `.github/workflows/publish.yml`: tag glob `hyphaeon-core-v*`/`hyphaeon-v*`/`chronaeon-v*` → package dir → `python -m build` + `twine check` + OIDC trusted-publisher upload (no token secret).
- `.github/workflows/rust.yml`: fmt + `cargo test --workspace` + `cargo build --release -p rhizaeon-cli -p trace50-cli` + wasm-pack job.
- **No `license` field in any `rust/**/Cargo.toml`** — required by crates.io. `publish = false` set nowhere.
- **crates.io availability verified (2026-10-12, crates.io API)**: `aeon-manifold`, `aeon-autoclock`, `aeon-py`, `rhizaeon`, `trace50`, `rhizaeon-cli`, `trace50-cli` all return 404 — every candidate name is free.
- Conformance suite `rust/tests/conformance/test_rust_python_parity.py` documents intentional divergences between the two autoclock impls: **Bartlett n_eff penalty, tier taxonomy, n<3 fields, cohesive stopping criterion** — merging eliminates these; canonical semantics must be chosen (see Decisions).

---

## Phase A — Shared autoclock crate `aeon-autoclock`

Autoclock is an algorithm, not a math kernel — do **not** put it inside `aeon-manifold`. New lib crate mirrors the existing `libs/` pattern.

### A1. Extract to `rust/libs/aeon-autoclock/`

- Move `rust/trace-5.0/crates/trace50-core/src/autoclock.rs` → `rust/libs/aeon-autoclock/src/lib.rs` (rename file to lib.rs or keep `mod autoclock` inside lib.rs — either fine).
- `Cargo.toml`: name `aeon-autoclock`, version `0.1.0`, `license` field (match repo LICENSE), deps: `aeon-manifold.workspace = true`, `serde`/`serde_json` if `SteveCommunity` derives serde (check current derives — `pipeline.rs` serializes community results, so keep whatever derives exist).
- Add to `rust/Cargo.toml` members + `[workspace.dependencies]` as `aeon-autoclock = { path = "libs/aeon-autoclock", version = "0.1.0" }`.
- Fix internal `use crate::...` references → `use aeon_manifold::...` (the file uses `linregress`, `symmetric_jacobi_eigh`, `percentile`, `norm_ppf` etc. from manifold).

### A2. Extend the API to cover chronaeon's needs

Python's `recursive_spectral_autoclock(sub_idx, dates, D, min_size, max_depth, depth, path, is_ref_array)` returns `List[(sub_idx, fit_dict, stop_reason, path)]`. The Rust signature `recursive_autoclock_deconvolution(&indices, &dates, &d_matrix, n, min_size, max_depth, depth, path, 0.65)` lacks:

- **`is_ref_array: Option<&[bool]>`** — reference-panel anchor support used by chronaeon's triage.
- **Structured fit results** — Python `fit_clock` returns a dict of fields (`mu`, `d0`, `se_mu`, `t_mrca`, `t_mrca_ci`, `r_squared`, `tier`, `stop_reason`, ...). Add a `pub struct ClockFit` (serde-serializable) collecting exactly the fields the Python dict emits today; document field-by-field mapping in the struct docstring.
- **`SteveCommunity` extension** — carry `stop_reason: String` and `path: String` (Python's tuple elements 2,3) plus the `ClockFit`.

**Canonical semantics decision (RESOLVED 2026-10-13 — Sergei: Rust is canonical):** the trace50 STEVE engine is the reference implementation. The four conformance-documented divergences are resolved by making the Python kernels delegates to `aeon_core._native` bindings of `aeon-autoclock`; the pure-Python bodies remain only as the no-extension fallback (legacy semantics, pinned separately in the conformance suite).
- Bartlett n_eff: Rust's `df_eff = n_eff - 2` penalty is canonical; Python's `n - 2` is fallback-only.
- Tier taxonomy + n<3 fields: Rust's 4-tier relative-horizon taxonomy and real `mean_pairwise_dist`/`time_span_years` on dyads are canonical; the delegated fit dict additionally exposes `n_eff`, `ci_mu`, `is_active_outbreak`, `operational_tier` for Python consumers.
- Cohesive stopping: Rust's modularity/conductance gate is canonical; Python's Welch-z child-rate stop is fallback-only.
- `is_ref_array`: dead plumbing in Python (threaded, never read) — the Rust binding accepts and reserves it but does not consume it.
- `SteveCommunity` extended with `rss`, `tot_var`, `aicc` (Hurvich-Tsai, Bartlett-penalized sigma^2), `max_pairwise_dist`, `d0` — additive serde fields covering the full Python fit-dict contract.

### A3. Rewire trace50

- `trace50-core`: replace `autoclock.rs` content with a re-export shim (same pattern as existing `tn93.rs`/`math.rs` shims): `pub use aeon_autoclock::*;` — keeps `crate::autoclock::...` call sites (`pipeline.rs:15,347`) untouched.
- Add `aeon-autoclock.workspace = true` to trace50-core deps.

### A4. Bind for Python via `aeon-py`

In `rust/libs/aeon-py/src/lib.rs` add (after the crate is moved in Phase C, or before — order flexible):

```rust
#[pyfunction]
fn fit_clock_community(sub_idx: Vec<usize>, dates: Vec<f64>, d_matrix_flat: Vec<f64>, n: usize) -> PyResult<PyObject>  // serialize ClockFit -> PyDict via pythonize or serde_json
#[pyfunction]
fn recursive_spectral_autoclock(sub_idx: Vec<usize>, dates: Vec<f64>, d_matrix_flat: Vec<f64>, n: usize, min_size: usize, max_depth: usize, is_ref: Option<Vec<bool>>) -> PyResult<Vec<PyObject>>
```

Return types: `ClockFit`/`SteveCommunity` → Python `dict` — use `pythonize` crate or serialize to JSON string and `json.loads` in Python (JSON is simpler; pick `pythonize` if already a dep pattern elsewhere — it is not, so JSON is the zero-new-dep option).

### A5. Rewire chronaeon

- `chronaeon/src/chronaeon/autoclock.py`: `fit_clock` and `recursive_spectral_autoclock` become thin delegates — call `aeon_py.*` when importable, **keep the pure-Python bodies as fallback** (pure-Python installs from sdist without a Rust toolchain remain possible).
- Orchestration classes (`AutoClockDeconvolution`, `HierarchicalAutoClock`, `detect_contemporaneous_dyads`) stay Python — they do I/O/figures, not kernels.

### A6. Tests

- Port autoclock unit tests from trace50's autoclock.rs `#[cfg(test)]` into the new crate.
- `rust/tests/conformance/test_rust_python_parity.py`: remove the intentional-divergence header block for the four now-unified behaviors; add a test that Rust-bound `recursive_spectral_autoclock` output == Python fallback output on a fixture alignment (field equality within float tolerance).
- `cargo test --workspace`, `pytest chronaeon/tests`, conformance suite — all green.

---

## Phase B — Collapse `-core`/`-cli` into one crate per product

> **SUPERSEDED (Python-fronted CLI pivot, executed):** the `-core`/`-cli`
> collapse itself happened (single `rhizaeon` and `trace50` crates), but the
> user-facing binaries were then REMOVED from the lib crates entirely.
> Instead of cargo binaries, each product got a thin Python sister package
> mirroring `aeon-core`: `rhizaeon/` (native/ PyO3 crate `rhizaeon-native`
> exposing `rhizaeon_cli` + `export_figure1_data`, console scripts
> `rhizaeon`/`export_figure1_data`, PyPI name `rhizaeon`) and `trace50/`
> (crate `trace50-native` exposing `trace50_cli`, console script `trace50`,
> PyPI name `trace50`). The Python wrappers forward `sys.argv` verbatim to
> the same clap parsers — CLI surface identical to the old binaries.
> `conformance` stays as trace50's only Rust bin (dev/test driver).
> Rationale: better UX, single PyPI publishing path (publish.yml handles
> `rhizaeon-v*`/`trace50-v*` tags via the same maturin wheel matrix;
> publish-crates.yml now covers only the shared lib crates).
>
> Remaining sections below describe the crate collapse mechanics that were
> kept; the "ship as cargo bins" framing is superseded.

Target: workspace members shrink from 8 → 7: `aeon-autoclock`, `aeon-manifold`, `aeon-py` (moved to `aeon-core/native` in Phase C), `rhizaeon`, `rhizaeon-wasm`, `trace50`, `trace50-wasm` (wasm crates stay separate).

### B1. `rhizaeon` crate

- `rust/rhizaeon/crates/rhizaeon-core/` becomes the crate dir; rename package to `rhizaeon`.
- Move `rhizaeon-cli/src/main.rs` → `rhizaeon-core/src/bin/rhizaeon.rs`; move `rhizaeon-cli/src/bin/export_figure1_data.rs` alongside.
- `Cargo.toml`:

```toml
[package]
name = "rhizaeon"
version = "0.1.0"
edition = "2021"
license = "MIT"          # verify against repo LICENSE file
description = "RhizAeon PA-FF: engine + CLI"

[dependencies]
aeon-manifold.workspace = true
serde.workspace = true
serde_json.workspace = true
clap = { workspace = true, optional = true }
rayon = { workspace = true, optional = true }

[features]
default = ["cli"]
cli = ["dep:clap", "dep:rayon"]

[[bin]]
name = "rhizaeon"
path = "src/bin/rhizaeon.rs"
required-features = ["cli"]

[[bin]]
name = "export_figure1_data"
path = "src/bin/export_figure1_data.rs"
required-features = ["cli"]
```

- Fix `use rhizaeon_core::` → `use rhizaeon::` inside the bins (`cargo fix` handles most).
- `main.rs` currently 183 LoC, `export_figure1_data.rs` 624 LoC — pure moves, no logic edits expected.

### B2. `trace50` crate — same pattern

- Package rename `trace50-core` → `trace50`, version `5.0.0`.
- Move `trace50-cli/src/main.rs` (498 LoC) → `src/bin/trace50.rs`, `required-features = ["cli"]`.
- Keep existing `parallel = ["dep:rayon"]` feature; add `cli = ["dep:clap"]` (trace50-cli does not use rayon — check `main.rs` imports; if it does, add rayon to `cli`).
- `default = ["cli", "parallel"]` so `cargo build`/`cargo install` get the full binary; wasm uses `default-features = false` (+`features=["parallel"]` if it wants rayon — wasm target: rayon does not compile to wasm32 without atomics; **verify trace50-wasm currently builds with `parallel` off — it does, since `parallel` is default-off today; keep it off for wasm**).

### B3. Update wasm + workspace

- `rhizaeon-wasm` dep → `rhizaeon = { workspace = true, default-features = false }`; `trace50-wasm` likewise.
- `rust/Cargo.toml`: members list drops the `-cli` entries; `[workspace.dependencies]` renames `rhizaeon-core`→`rhizaeon`, `trace50-core`→`trace50` (path stays `rhizaeon/crates/rhizaeon-core` — optionally `git mv` dirs to drop `-core` suffix; renaming dirs is cosmetic but breaks muscle memory less if done now: `rhizaeon/crates/rhizaeon`, `trace-5.0/crates/trace50`; update all paths).
- Delete empty `rhizaeon-cli`, `trace50-cli` dirs.
- `use rhizaeon_core::`/`use trace50_core::` across wasm crates → new names.
- Regenerate `rust/Cargo.lock` (`cargo check --workspace`).
- `rust.yml` CI: `cargo build --release -p rhizaeon-cli -p trace50-cli` → `-p rhizaeon -p trace50 --all-features` (or default features — `cli` is default).

**Note on crates.io naming:** `rhizaeon`/`trace50` are the lib+bin crates — all names verified free (see Current state).

---

## Phase C — Fold `aeon-py` into `aeon-core` (maturin)

### C1. Relocate the Rust crate

- `git mv rust/libs/aeon-py aeon-core/native` — maturin requires the Rust source inside the Python package dir for sdist completeness.
- `rust/Cargo.toml` members: `libs/aeon-py` → `../aeon-core/native` — workspace member paths can escape the rust/ dir; alternatively keep a nested `aeon-core/native` member entry. Both work; pick one and make `cargo check --workspace` green.
- Keep `[lib] crate-type = ["cdylib"]`, but rename module output per C2.

### C2. Module name — DECIDED

Ship the extension as `aeon_core._native`. **Decision (2026-10-12): do not keep `aeon_py` as a top-level module** — the compiled extension inside the `hyphaeon-core` wheel should live under the `aeon_core` namespace, not squat a top-level name.

- `native/Cargo.toml`: `[lib] name = "aeon_core_native"`.
- `aeon-core/pyproject.toml`:

```toml
[build-system]
requires = ["maturin>=1.5,<2.0"]
build-backend = "maturin"

[tool.maturin]
python-source = "src"
manifest-path = "native/Cargo.toml"
module-name = "aeon_core._native"
bindings = "pyo3"
```

- Keep all existing `[project]` metadata/deps unchanged.

### C3. Python-side import updates

- In `native/src/lib.rs` the `#[pymodule]` fn name must match module name tail: rename `fn aeon_py(m: ...)` → `fn _native(m: ...)`.
- `aeon-core/src/aeon_core/dataset.py`, `stats.py` (and any `import aeon_py` site — `grep -rn 'aeon_py' aeon-core/ chronaeon/ hyphaeon/`): change to `from aeon_core import _native as aeon_py` inside the existing try/except so call sites (`aeon_py.tn93_distance_matrix`) are untouched.
- Test probes `_HAS_TN93` in the three test files: update to probe `from aeon_core import _native` (binary fallback unchanged).

### C4. Wheel story

`hyphaeon-core` becomes a **platform-specific wheel**. CI matrix (in publish workflow):

- linux x86_64 + aarch64 (maturin-action, manylinux)
- macOS x86_64 + arm64
- Windows: **DECIDED (2026-10-12): include `windows-latest` in the matrix.** Cost is one extra CI leg in `PyO3/maturin-action` (no special config — maturin handles MSVC + pyo3 abi3 the same as other targets). Rationale: today `hyphaeon-core` is a single OS-independent pure-Python wheel, so Windows users can `pip install` it now; shipping no Windows wheel would silently remove that. A pure `py3-none-any` dual-wheel alternative was considered and rejected as unusual/risky. Caveat to watch: verify the compiled extension doesn't hit MSVC-specific issues (unlikely — the crate is pure Rust, no C deps, pyo3 is MSVC-supported).
- Python: build abi3 wheels (`bindings = "pyo3"` + `abi3-py38` feature on pyo3 matching `requires-python = ">=3.8"`) → one wheel per platform covers all py≥3.8.
- sdist still published; installing from sdist requires a Rust toolchain (document in README).

---

## Phase D — Packaging metadata, CI, docs

### D1. Crate metadata (crates.io blockers)

Add to every publishable `Cargo.toml` (`aeon-manifold`, `aeon-autoclock`, `rhizaeon`, `trace50`):

```toml
license = "MIT"    # confirm against repo LICENSE text
repository = "https://github.com/veg/HyphAeon"
homepage = "https://github.com/veg/HyphAeon"
```

Add `publish = false` to `rhizaeon-wasm`, `trace50-wasm`, `aeon-py/native` crates.

Verify names free: `cargo search aeon-manifold` etc. (or crates.io API).

### D2. Extend `publish.yml` — wheels

For `hyphaeon-core-v*` tags replace the `python -m build` step with a wheel matrix:

- New job `build_wheels` matrix over `[ubuntu-latest, ubuntu-24.04-arm, macos-13, macos-latest, windows-latest]` using `PyO3/maturin-action@v1` (`maturin generate-ci github` scaffolds this) producing abi3 wheels + sdist → upload artifacts. Windows leg included per C4 decision.
- `publish` job `needs: build_wheels`, downloads artifacts to `dist/`, `twine check`, OIDC upload (existing trusted publisher).
- `hyphaeon`/`chronaeon` unchanged (pure wheels).

### D3. New `publish-crates.yml`

Trigger on tags `aeon-manifold-v*`, `aeon-autoclock-v*`, `rhizaeon-v*`, `trace50-v*`:

```yaml
- run: cargo publish -p <crate> --token ${{ secrets.CARGO_REGISTRY_TOKEN }} --allow-dirty=false
```

**Ordering constraint:** `rhizaeon`/`trace50` depend on `aeon-manifold` + `aeon-autoclock` being published first — release order: tag/push shared crates → wait for registry propagation (~5–10 min; add `sleep 300` + `cargo search` poll step) → tag/push products. Optionally single workflow with `needs:`-chained publish jobs so one tag push orchestrates all.
- Secret needed: `CARGO_REGISTRY_TOKEN` (crates.io has OIDC trusted publishing as an alternative — supported since 2025; evaluate, else token).

### D4. Release binaries (optional, recommended)

New `release-binaries.yml` on `rhizaeon-v*`/`trace50-v*` tags: `cargo build --release -p rhizaeon -p trace50` matrix (linux gnu, macos intel+arm, windows) → attach to GitHub Release. Gives non-cargo users a `pip install`-equivalent.

### D5. Bioconda path (note, not in-repo)

Post-publish, bioconda recipes (in user's `bioconda-recipes` fork) can source the crates.io tarball — recipe is `cargo install --locked --root {{ PREFIX }} --path .` — self-contained tarball, deps resolve from registry. Flag as follow-up, not part of this plan's in-repo changes.

### D6. README/doc updates

- **Root `README.md`**: install table — `pip install hyphaeon` / `pip install chronaeon` (pull `hyphaeon-core` w/ native ext on supported platforms), `cargo install rhizaeon` / `cargo install trace50` or GH-release binaries, wasm bundles for browser. Dev-install section: `cd aeon-core && maturin develop` (requires rustup toolchain); `cd rust && cargo build --workspace`.
- **`rust/trace-5.0/README.md` + `rust/rhizaeon/README.md`**: replace "build the cli crate" instructions with single-crate equivalents; note `--no-default-features` for lib-only/wasm consumers; add `cargo install` once published.
- **`aeon-core/README.md`**: document maturin build, sdist→cargo requirement, platform wheel coverage, `aeon_core._native` availability probe.
- **`rust/tests/conformance` header**: remove resolved-divergence notes (Phase A6).
- Any docs mentioning `aeon_py` top-level import or `rhizaeon-cli`/`trace50-cli`/`rhizaeon-core`/`trace50-core` crate names — `grep -rn` and update.

---

## Execution order

1. **B (crate collapse)** — mechanical, no API change; do first to stabilize naming before CI/docs.
2. **A (autoclock)** — largest semantic work; **gate on the canonical-semantics decision (A2)**.
3. **C (maturin fold)** — after A so `native/` already contains autoclock bindings.
4. **D** — metadata early (needed to test `cargo publish --dry-run`), workflows last.

Independent: B ∥ A (different crates), but A's bindings land in the crate C relocates — sequence A→C avoids a double move.

## Acceptance criteria

- `cargo test --workspace` green; `cargo build --release -p rhizaeon -p trace50` produces `rhizaeon`/`trace50` bins; `cargo build -p rhizaeon --no-default-features` succeeds without clap.
- `wasm-pack build` (existing rust.yml job) green for both wasm crates — proves clap/rayon stayed out.
- `cd aeon-core && maturin build` produces a wheel containing `aeon_core/_native.*.so`; `pip install` of that wheel makes `from aeon_core import _native` work and `aeon_core.dataset` uses it.
- Conformance suite green, including new autoclock-parity test; four documented divergences removed from header.
- `cargo publish --dry-run` passes for `aeon-manifold`, `aeon-autoclock`, `rhizaeon`, `trace50` (license/metadata complete).

## Risks / decisions needing sign-off

- **Canonical autoclock semantics (A2)** — with Sergei; upstream-fix-then-pull vs. we implement. Recommended direction if it's ours: Python semantics win.
- ~~Windows wheel (C4)~~ — **decided: `windows-latest` in the matrix**.
- ~~`aeon_py` top-level~~ — **decided: `aeon_core._native`**.
- **Python fallback retention (A5)** — recommended (sdist/no-cargo installs); alternative is hard-fail making chronaeon unusable without native ext.
- ~~crates.io name availability~~ — **verified: all free**.
