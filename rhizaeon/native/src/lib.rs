//! PyO3 bindings powering the `rhizaeon` Python package.
//!
//! The clap argument parsers live here — not in the `rhizaeon` lib crate —
//! so the lib and, transitively, `rhizaeon-wasm` stay clap-free. The Python
//! sister package forwards `sys.argv` verbatim; help text, usage errors,
//! and output are identical to the former standalone binaries.

use pyo3::prelude::*;

mod cli;

/// `rhizaeon` console-script entry point. `argv` includes the program name
/// at index 0; returns a process exit code.
#[pyfunction]
fn rhizaeon_cli(argv: Vec<String>) -> i32 {
    cli::cli_main(argv)
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(rhizaeon_cli, m)?)?;
    Ok(())
}
