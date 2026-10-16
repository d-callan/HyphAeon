# RhizAeon — PA-FF v2.1 Recombination Scanner (Python package)

Python-fronted distribution of the RhizAeon PA-FF engine. The compiled
extension (`rhizaeon._native`, built by maturin from `native/`) carries the
clap parser that used to live in the standalone `rhizaeon` cargo bin, so the
command-line surface is unchanged:

```console
$ pip install rhizaeon
$ rhizaeon --input alignment.fasta --output scan.json --viz viz.json --html report.html
```

`python -m rhizaeon` is equivalent to `rhizaeon`. Manuscript figure/benchmark
helpers (`export_figure1_data`, `export_figure2_data`, `grand_sweep`) live in
the unpublished `rust/rhizaeon/crates/rhizaeon-tools` crate — run them via
`cargo run -p rhizaeon-tools --release --bin <name>` from a source checkout.

## Development install

```console
$ cd rhizaeon && maturin develop
```

requires a Rust toolchain (`rustup`). The engine crate itself lives at
`rust/rhizaeon/crates/rhizaeon` and stays a pure library (no clap, no bins)
so `rhizaeon-wasm` builds clean.
