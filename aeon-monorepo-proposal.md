# Proposal: Consolidate the Aeon family into the HyphAeon monorepo on a shared Rust core

> **Status: rough draft for discussion.** This is a direction-setting proposal, not an implementation plan. Crate names, boundaries, sequencing, and even whether some of this happens are all open — expect the details to change substantially if we decide to move forward and start doing it. The intent is to agree on *whether* the direction is right, and roughly *why*, before sorting *how*.

## Why move in this direction

The value of consolidating `rhizaeon-code` and `trace-5.0` into the `HyphAeon` monorepo, with a shared Rust core that everything incrementally ports onto:

- **Kill real duplication, not theoretical duplication.** The same metric-manifold math is currently implemented independently at least four times: rhizaeon's Rust kernel, chronaeon's `autoclock.py` (TN93 → MDS → eigengap), trace-5's AutoClock (a reimplementation of chronaeon's logic, not a dependency), and even rhizaeon internally duplicates its own distance/MDS code across its engine and visualization layers. One shared Rust implementation ends all of it.
- **WASM "for free" for chronaeon and hyphaeon.** rhizaeon already ships browser-native WASM. Once the shared primitives (FASTA IO, distance models, embedding, attention kernels) and the deterministic engines are in Rust behind `wasm-bindgen`, chronaeon's triage and autoclock functionality — and eventually hyphaeon *inference* — get browser/WASM builds essentially as a side effect of the port. This matters because we're already manually porting this code anyway; doing it in Rust buys the wasm target at marginal extra cost.
- **Single source of truth kills drift.** trace-5's dual Python+Rust implementations have already diverged (the Rust path lacks the Python `--reference-panel` feature). Maintaining parallel implementations in two languages guarantees skew; a single Rust core consumed by both Python (PyO3) and JS (wasm-bindgen) can't drift.
- **Performance ceiling goes up everywhere.** Rust cores replace numpy/sklearn pipelines and subprocess-to-`tn93` hacks with zero-copy, allocation-controlled implementations — rhizaeon's O(N·K·L) profile is the template for what the rest of the family could run like.
- **Each step is independently shippable.** Nothing below requires a big-bang migration: crates move, then get refactored, then get consumed — Python APIs and CLIs stay stable throughout, so downstream users never see a breaking change.

## Current state (one-liners)

- **HyphAeon monorepo**: Python/PyTorch — `aeon-core` (trained transformer, dataset/weights plumbing), `chronaeon` (autoclock, triage, dating), `hyphaeon`. No Rust.
- **rhizaeon-code**: pure Rust, 3-crate workspace (`rhizaeon-core`, `rhizaeon-cli`, `rhizaeon-wasm`), deterministic recombination engine, already wasm-ready.
- **trace-5.0**: HIV-TRACE successor — application layer with its own engines (DANNO/STEVE/CHIN), dual Python+Rust implementations, AutoClock as a reimplemented (not linked) chronaeon concept, HIV-specific assets bundled on the Python side.

## Sketch of the target shape

Everything below is illustrative — real boundaries get decided during implementation.

```
HyphAeon/
├── rust/                     # new Cargo workspace
│   ├── aeon-manifold/        # shared kernel: FASTA, TN93/Hamming/JC distances,
│   │                         #   double-centering MDS, eigendecomposition, softmax kernels
│   ├── rhizaeon-core/        # moved as-is, then refactored onto aeon-manifold
│   ├── rhizaeon-cli/         # moved as-is
│   ├── rhizaeon-wasm/        # moved as-is
│   ├── autoclock-core/       # Rust port of chronaeon's autoclock/triage math
│   ├── autoclock-py/         # PyO3/maturin bindings — chronaeon calls this
│   ├── trace5-core/          # transmission-network engine(s) from trace-5.0
│   ├── aeon-infer/           # hyphaeon inference-only forward pass (candle/burn, safetensors)
│   ├── aeon-infer-py/        # PyO3 wrapper — aeon-core delegates inference here
│   └── aeon-wasm/            # wasm-bindgen surface over the engines
├── aeon-core/                # unchanged Python package (training, weights, datasets)
├── chronaeon/                # unchanged Python API; internals progressively delegate to Rust
├── hyphaeon/                 # unchanged Python package; inference path progressively delegates
└── ...                       # docs, examples, benchmarks as today
```

Principles behind the sketch:

- **One shared kernel crate** (`aeon-manifold`) owns the primitives every engine needs. This is the highest-value piece and the least controversial.
- **Bindings live in separate crates** (`*-py`, `*-wasm`) so PyO3 and wasm-bindgen feature-gate cleanly — wasm targets never see Python deps.
- **Inference-only scope for the learned model.** Porting hyphaeon's forward pass (a custom but standard-primitive transformer) to Rust on candle/burn is tractable; porting training is not in scope. Weights stay in safetensors/HF, loaded by the Rust crate.
- **Python packages keep their public APIs.** chronaeon/hyphaeon become progressively thinner wrappers — existing scripts, tests, and CLI surfaces are the regression contract.

## Rough sequencing

Each phase stands alone; we can stop after any of them and still be better off.

1. **Extract `aeon-manifold` inside rhizaeon** — rhizaeon already contains ~3 copies of its own distance/embedding math; consolidate those first. Pure refactor, no API change.
2. **Move the rhizaeon workspace into HyphAeon** (`rust/` dir), refactor onto `aeon-manifold`. Bring tests; preserve git history (subtree merge or `git filter-repo`).
3. **Port autoclock to `autoclock-core` + PyO3 wrapper**, keeping chronaeon's Python API and test suite as the contract. scipy/sklearn pieces (spectral clustering, stats) need Rust equivalents (linfa or hand-rolled) — the first real porting effort.
4. **Fold trace-5's engines in as `trace5-core`**, collapsing its dual implementations onto the shared cores; HIV-specific/reference-panel assets stay as external data (like hyphaeon's HF weights), not vendored code.
5. **`aeon-infer`**: candle/burn forward-pass port, safetensors weights, golden-output parity tests against PyTorch as the acceptance gate.
6. **`aeon-wasm`** once the cores stabilize: browser builds of each engine, shared where sensible.

## Open questions / things likely to change

- **Monorepo vs. separate repos sharing a published crate.** Alternative: keep repos separate and publish `aeon-manifold` + engine crates to crates.io that each repo deps on. Leaning monorepo for a specific reason: this ecosystem is still actively evolving, and where the boundaries between engines sit will likely keep shifting as it develops — a monorepo makes cross-boundary refactors cheap exactly when the architecture is least settled, while one CI and coordinated versioning keep the moving pieces compatible. Separate repos become more attractive once boundaries stabilize and independent release cadence matters more than refactor velocity.
- **scipy/sklearn coverage** — spectral clustering, KDE/stats helpers need Rust equivalents; some may justify keeping a Python fallback.
- **Naming and branding** — crate names here are placeholders; whether trace-5 becomes an "Aeon" member or stays the flagship application is a community decision.
- **Release engineering** — maturin/PyO3 wheels per platform, wasm-pack publishing, conda recipe updates, versioning coordination across the workspace.
- **Crate boundaries** — e.g. whether `aeon-wasm` is one bundle or per-engine packages; whether `trace5-core` splits per engine.
