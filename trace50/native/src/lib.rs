//! PyO3 bindings powering the `trace50` Python package.
//!
//! The clap argument parser lives here — not in the `trace50` lib crate —
//! so the lib and, transitively, `trace50-wasm` stay clap-free. The Python
//! sister package forwards `sys.argv` verbatim; help text, usage errors,
//! and output are identical to the former standalone binary.

use pyo3::prelude::*;

mod cli;

// Re-exported so integration tests can drive the CLI in-process.
pub use cli::cli_main;

/// `trace50` console-script entry point. `argv` includes the program name
/// at index 0; returns a process exit code.
#[pyfunction]
fn trace50_cli(argv: Vec<String>) -> i32 {
    cli::cli_main(argv)
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(trace50_cli, m)?)?;
    Ok(())
}
