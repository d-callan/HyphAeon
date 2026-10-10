# Phase 2: Codebase-Wide Redundancy Reduction

Status: draft, not yet started.
Prerequisite: phase-1 merge (`feat/rust-workspace-import`) — `aeon-manifold` hosts the
shared Rust math/stats kernel; CLI outputs verified byte-identical modulo timing fields.
Scope: the **entire HyphAeon codebase** — `aeon-core`, `chronaeon`, `hyphaeon`, and
the `rust/` workspace (rhizaeon + trace-5.0). `symbaeon` is excluded (branch WIP).

## Governing Principle: No Refactoring Before Regression Detection

Phase-1 verification leaned on ad-hoc `/tmp` baselines and a 28-test Rust suite that
leaves several refactor targets untested. Every work item below is gated on a
committed, reproducible baseline that would flag a regression.

## Phase 0 — Test & Baseline Infrastructure (MUST COMPLETE FIRST)

### P0.1 Commit the Rust CLI baseline harness
- `rust/scripts/compare_cli_baseline.sh`: rebuilds release binaries, runs the two
  fixtures (`hiv1_kal153_crf02_ag.fasta`, `pair_test.fasta`), JSON-normalizes
  `run_time_ms`/`execution_time_seconds`, diffs byte-exact.
- Commit normalized golden outputs under `rust/tests/baselines/`.
- Wire into `cargo test` as an integration test (`#[ignore]` default; `--ignored`
  in CI release job) or a dedicated CI step.

### P0.2 Backfill Rust unit tests for phase-2 refactor targets
Current coverage gaps (evidence: `grep #[test]` audit):

- `rhizaeon-core/src/fasta.rs::parse_fasta` — **no direct unit test**; only
  exercised end-to-end. Needs: strict-length rejection, header tokenization,
  multiline sequence accumulation, ambiguity encoding round-trip.
- `rhizaeon-core/src/fasta.rs::encode_char` — untested. Needs: all IUPAC codes
  (ambiguous → 0), case handling.
- `rhizaeon-core/src/screener.rs::median` — private, untested.
- `rhizaeon-core/src/screener.rs::normal_ppf` — delegate, untested at edges
  (clamp behavior at p <= 0, p >= 1).
- `trace50-core/src/tn93.rs::Alignment::from_fasta_str` — only via integration.
  Needs: ragged-pad behavior, date extraction, dual-encoding consistency.
- `trace50-core/src/tn93.rs::char_to_nuc_int` — untested standalone.
- `trace50-core/src/tn93.rs::parse_date_from_header` — **untested**. Needs: decimal
  passthrough, ISO YYYY-MM-DD, YYYY-MM, delimited tokens, NaN fallback.
- `trace50-core/src/chin.rs::percentile` — tested only indirectly.
- `aeon-manifold` — has 5 tests; add norm_ppf/student_t_ppf/linregress against
  reference values (scipy-generated goldens committed as a CSV or inline consts).

### P0.3 Verify Python test baseline is green
- ~512 test functions exist (aeon-core 160, chronaeon 187, hyphaeon 137,
  model_eval 28) — but no recorded all-green baseline in this environment.
- Run `pytest` per package in `axomeme`; document pass/fail/skip counts in
  `PHASE2_BASELINE.md`. Any pre-existing reds are triaged (fix or `xfail` with
  issue ref) — refactoring on a red suite hides regressions.

### P0.4 Cross-language conformance tests (the critical gap)
Python and Rust implement the same algorithms independently with **zero**
equivalence checking. Before any dedup, add golden-vector tests:

- **TN93**: `aeon_core.dataset.compute_tn93_distance_matrix` vs
  `trace50-core::compute_pairwise_tn93` — same input alignment -> same matrix
  (tolerance 1e-12). Include ambiguity codes, Ns, ragged/short sequences.
- **Date parsing**: `aeon_core.temporal.parse_date_to_decimal` /
  `extract_date_from_string` vs `trace50::parse_date_from_header` — agreed
  header corpus -> agreed decimal dates (or documented intentional divergence).
- **Autoclock/STEVE**: `chronaeon.autoclock` (~2600 LoC, full feature set) vs
  `trace50-core::autoclock` (~500 LoC, spectral cut core) — establish which
  sub-behaviors are *supposed* to match on `pair_test.fasta` scale fixtures
  (community count, Fiedler sign partition) before touching either side.
- Harness: Python pytest that shells to the Rust CLI (or `wasm-pack test`) and
  compares against scipy/numpy-computed references. Lives in
  `rust/tests/conformance/` + a pytest marker.

### P0.5 Dependency-drift guard
- `cargo tree` + `pip list` snapshot committed under `tests/` so dedup PRs can't
  silently bump versions.

## Duplication Inventory — Whole Codebase

### Rust <-> Rust (carried from phase-1 audit)

- **`Alignment` x2**: rhizaeon flat `Vec<u8>` matrix w/ 0–4 encoding
  (`types.rs:5`) vs trace50 headers+dates+triple-encoding row-vecs (`tn93.rs:201`).
- **FASTA parsers**: same scan loop; rhizaeon strict-errors on ragged
  (`fasta.rs:17`), trace50 pads + extracts dates (`tn93.rs:212`).
- **Encoding tables x3**: `encode_char` (0–4), `char_to_iupac_mask` (4-bit),
  `char_to_nuc_int` (i8) — projections of one IUPAC alphabet.
- **Stats helpers**: `median` (screener.rs:412, private), `percentile`
  (chin.rs:353) -> `aeon-manifold`.
- **Deps**: serde/clap/wasm-bindgen across 7 manifests -> `[workspace.dependencies]`.

### Python <-> Python

- **Date parsing x3**: `aeon_core.temporal.parse_date_to_decimal` /
  `extract_date_from_string` (canonical), `chronaeon.r0.parse_calendar_date`
  (`r0.py:101`, local reimpl), `chronaeon.dating_io._parse_timestamp_flexible`
  (third variant). Consolidate on aeon_core — verify format coverage first:
  r0's version handles `DD/MM/YYYY` which aeon_core may not.
- **aeon_core is already the kernel** — chronaeon/hyphaeon import heavily from it
  (dataset, temporal, stats, inference, io, cli, _progress). Residual work is
  hunting local reimplementations like r0's, not building new infra.
- **Stats**: `pvals_from_lrt_*`, `benjamini_hochberg`, `cauchy_combination_p`
  already centralized in `aeon_core.stats` — verify no bypassed copies in
  model_eval/`_harness.py` (it aliases imports; check bodies).

### Python -> Rust (canonical direction: Rust, Python impls get deleted)

DECISION: where the same algorithm exists in both languages, **Rust is canonical**.
Path per component: parity audit (P0.4 conformance) -> port missing features
into the Rust impl -> expose to Python via PyO3/maturin bindings ->
Python call sites switch to the binding -> delete the Python implementation.

The binding crate lives at `rust/libs/aeon-py` (PyO3, built with maturin into
the axomeme env). Python keeps only non-algorithmic glue (torch model inference,
plotting, CLI plumbing, dataframe I/O); all math/algorithmic cores move to Rust.

- **TN93 distance** — `aeon_core.dataset.compute_tn93_distance_matrix` (numpy)
  vs `trace50 tn93.rs`. Rust is nearly at parity already; needs the
  cross-distance-matrix variant and a numpy-friendly binding signature.
- **Autoclock/STEVE** — `chronaeon.autoclock` vs `trace50-core autoclock.rs`.
  Largest gap. Python has model selection, manifold representation, community
  classification, admissible-dyad detection; Rust has spectral cut + Fieller CI.
  Parity audit must enumerate every Python stage and decide: port to Rust or
  keep as Python glue (e.g., anything consuming transformer embeddings stays
  Python-side, calling into Rust for the spectral/statistical core).
- **Date parsing** — see Python<->Python section; Rust's `parse_date_from_header`
  plus W7 consolidation should converge to one shared format matrix.
- **IUPAC encoding** — `aeon_core.dataset` encoding tables vs Rust tables.
  W3 makes Rust canonical; Python table becomes a re-export or is generated
  from the Rust definition.
- **Stats kernels** — `aeon_core.stats` (LRT p-values, BH, Cauchy) vs
  `aeon-manifold`. Longer-term: port the Python stats functions that have no
  Rust equivalent (chi2 sf, LRT mixtures) into aeon-manifold, bind via aeon-py.

## Work Items (each gated on its P0 tests)

| # | Item | Lang | Gate |
|---|------|------|------|
| W1 | `[workspace.dependencies]` consolidation | Rust | cargo tree unchanged |
| W2 | `median`/`percentile` -> aeon-manifold | Rust | P0.2 tests + baseline |
| W3 | Encoding module -> aeon-manifold | Rust | P0.2 encode/iupac tests |
| W4 | Shared FASTA tokenizer (ParseMode strict/pad) | Rust | P0.2 parser tests + CLI baseline |
| W5 | `parse_date_from_header` -> aeon-manifold | Rust | P0.2 + P0.4 date goldens |
| W6 | Alignment type strategy (trait vs unified) | Rust | design decision + bench |
| W7 | Python date consolidation (r0/dating_io -> aeon_core.temporal) | Py | P0.3 + format-coverage audit |
| W8 | Local-parser sweep (r0.py etc. -> aeon_core.dataset) | Py | P0.3 green suite |
| W9 | TN93 -> Rust canonical: parity audit, port missing variants, aeon-py binding, delete Python impl | Py+Rs | P0.4 goldens green |
| W10 | aeon-py PyO3/maturin crate skeleton + packaging into axomeme | Py+Rs | binding imports + smoke test |
| W11 | Autoclock parity audit (enumerate Python stages -> port/keep-glue) then port gaps to trace50-core | Py+Rs | P0.4 + parity spec |
| W12 | Python stats kernels (chi2 sf, LRT mixtures) -> aeon-manifold + aeon-py bindings | Py+Rs | aeon_core.stats test suite passes via bindings |

## Traps (updated)

- **Ambiguity collapse**: `encode_char` maps ambiguous->0; IUPAC masks preserve.
  Port projections, never unify inputs.
- **Ragged contract**: trace50 pads silently; rhizaeon rejects; Python's
  `parse_alignment_sequences` may do a third thing — audit all three before
  defining shared semantics.
- **Date-format coverage**: `r0.parse_calendar_date` accepts `DD/MM/YYYY` and
  int/float passthrough; consolidating blindly loses formats. Write the format
  matrix first.
- **Interpolation semantics**: `median` vs `percentile` may interpolate
  differently (linear vs midpoint). Verify before subsumption.
- **Parity before deletion**: Python autoclock is ~5x the Rust version. The
  Rust impl must provably cover every Python stage's *used* behavior (not just
  overlap) before the Python path is removed. The parity spec is the contract.
- **Layout regression**: trait-vs-unified Alignment (W6) needs a benchmark before
  flattening rhizaeon's matrix into row-vecs.

## Sequencing

P0.1 -> P0.2 -> P0.3 -> P0.4 -> P0.5 (Phase 0 gate) -> W1/W2/W3 -> W4/W5 ->
W7/W8 -> W10 (aeon-py skeleton, needed early as the binding substrate) ->
W9 (TN93, smallest migration) -> W11 (autoclock, largest) -> W12 (stats) ->
W6 (Alignment design, deferred since W4/W5 reshape the input layer anyway).
All on `feat/phase2-dedup`, one commit per work item, baseline harness re-run
green before each commit. A Python implementation is only deleted after its
binding replacement is in-tree, tested, and the caller sweep is complete.
