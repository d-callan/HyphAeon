# TRACE-5.0 — Molecular Transmission Surveillance (Python package)

Python-fronted distribution of the TRACE-5.0 engine. The compiled extension
(`trace50._native`, built by maturin from `native/`) carries the clap parser
that used to live in the standalone `trace50` cargo bin, so the command-line
surface is unchanged:

```console
$ pip install trace50
$ trace50 --fasta alignment.fasta --meta dates.csv --out-dir trace50_results
```

`python -m trace50` is equivalent to `trace50`.

## Development install

```console
$ cd trace50 && maturin develop
```

requires a Rust toolchain (`rustup`). The engine crate itself lives at
`rust/trace-5.0/crates/trace50` and stays a pure library (no clap; the only
bin is the `conformance` dev driver) so `trace50-wasm` builds clean.
