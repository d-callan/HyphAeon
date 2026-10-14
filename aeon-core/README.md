# aeon-core

Shared infrastructure for the HyphAeon family of packages (HyphAeon, ChronAeon).

Contains the `PhyloAxialTransformer` foundation model, pretrained weight resolution (HuggingFace Hub), and core utilities for alignment processing, inference, spectral splits, temporal parsing, I/O, and statistics.

## Installation

```bash
pip install hyphaeon-core
```

Prebuilt wheels (abi3, one per platform covering all Python >= 3.8) ship a
compiled Rust extension at `aeon_core._native` providing the canonical TN93
distance and statistical kernels. Installing from the sdist instead requires a
Rust toolchain (`rustup` recommended) — maturin compiles `native/` at install
time. The extension is required for TN93-backed functions — a missing
extension means a broken install and raises `ImportError` at module load.

## Components

- **`model`** — `PhyloAxialTransformer`, `BustedMultiTaskHead`, `decode_soft_ordinal_lrt`
- **`weights`** — Weight resolution from HuggingFace Hub (`datamonkey/hyphaeon`), safetensors loading
- **`inference`** — Device management, adaptive batch sizing, model loading, alignment preparation
- **`dataset`** — Alignment parsing, TN93 distance matrices, tree handling, MDS coordinates
- **`splits`** — Cross-taxa attention extraction, fused affinity matrices
- **`temporal`** — Date parsing utilities (`parse_date_to_decimal`, `extract_date_from_string`)
- **`io`** — JSON/CSV writers, directory helpers
- **`stats`** — LRT p-value computation, Benjamini-Hochberg, Cauchy combination
- **`_progress`** — `ChunkProgress` progress bar utility
