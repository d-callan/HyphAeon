//! Shared console-script boilerplate for the Aeon native crates
//! (`rhizaeon-native`, `trace50-native`). Both previously carried
//! byte-identical copies of `cli_main` and `pep440` — this crate is the
//! single home so the versions cannot drift.
//!
//! Contract (pinned end-to-end by `rhizaeon/tests/test_rhizaeon_cli.py`
//! and `trace50/tests/test_trace50_cli.py`):
//!
//! - `--version` / `-V` anywhere after argv[0] prints `<name> <PEP 440>`
//!   and returns 0, ahead of clap parsing.
//! - `--help` and usage errors come from `try_parse_from`: clap writes
//!   the message itself and `cli_main` forwards its exit code
//!   (0 for help, 2 for usage errors), so a programmatic caller's
//!   process is never exited underneath it.
//! - `run` returning `Err` prints `Error: <e>` to stderr and returns 1.

use clap::Parser;
use std::error::Error;

/// Render a cargo semver string in PEP 440 form (the version `pip show`
/// reports): `0.1.0-beta.1` -> `0.1.0b1`, `0.1.0-alpha.2` -> `0.1.0a2`,
/// `0.1.0-rc.1` -> `0.1.0rc1`, `0.1.0-dev.3` -> `0.1.0.dev3`. Stable
/// releases and unrecognized prerelease tags pass through unchanged.
pub fn pep440(cargo_version: &str) -> String {
    let (version, suffix) = match cargo_version.split_once('-') {
        Some(pair) => pair,
        None => return cargo_version.to_string(),
    };
    let normalized = match suffix.split_once('.') {
        Some(("alpha", n)) => format!("a{}", n),
        Some(("beta", n)) => format!("b{}", n),
        Some(("rc", n)) => format!("rc{}", n),
        Some(("dev", n)) => format!(".dev{}", n),
        _ => format!("-{}", suffix),
    };
    format!("{}{}", version, normalized)
}

/// Console-script entry point. `argv` includes the program name at index 0
/// (clap `parse_from` semantics). `bin_name` and `pkg_version` must come
/// from the calling crate (`env!("CARGO_PKG_VERSION")` resolves to the
/// crate where the macro is expanded, not to this one). Returns a process
/// exit code: 0 success/help/version, 2 clap usage error, 1 `run` error.
pub fn cli_main<A, F>(bin_name: &str, pkg_version: &str, argv: Vec<String>, run: F) -> i32
where
    A: Parser,
    F: FnOnce(A) -> Result<(), Box<dyn Error>>,
{
    // Intercept --version/-V ahead of clap so the printed version is the
    // PEP 440 form `pip show` reports (Cargo semver writes
    // `0.1.0-beta.1`; PyPI normalizes it to `0.1.0b1`). Same
    // short-circuit semantics as clap's built-in version flag.
    if argv.iter().skip(1).any(|a| a == "--version" || a == "-V") {
        println!("{} {}", bin_name, pep440(pkg_version));
        return 0;
    }
    let args = match A::try_parse_from(&argv) {
        Ok(a) => a,
        Err(e) => {
            let _ = e.print();
            return e.exit_code();
        }
    };
    match run(args) {
        Ok(()) => 0,
        Err(e) => {
            eprintln!("Error: {}", e);
            1
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `pep440` had no unit tests while it lived as duplicated private
    /// helpers; consolidate both coverage and code here.
    #[test]
    fn pep440_normalizes_prerelease_tags() {
        assert_eq!(pep440("0.1.0-beta.1"), "0.1.0b1");
        assert_eq!(pep440("0.1.0-alpha.2"), "0.1.0a2");
        assert_eq!(pep440("0.1.0-rc.1"), "0.1.0rc1");
        assert_eq!(pep440("0.1.0-dev.3"), "0.1.0.dev3");
        assert_eq!(pep440("5.0.0-beta.1"), "5.0.0b1");
    }

    #[test]
    fn pep440_passes_stable_and_unknown_tags_through() {
        assert_eq!(pep440("1.2.3"), "1.2.3");
        assert_eq!(pep440("0.1.0-nightly"), "0.1.0-nightly");
        assert_eq!(pep440("0.1.0-beta"), "0.1.0-beta");
    }

    #[derive(Parser)]
    struct TestArgs {
        input: Option<String>,
    }

    fn ok(_args: TestArgs) -> Result<(), Box<dyn Error>> {
        Ok(())
    }

    #[test]
    fn cli_main_exit_codes() {
        let argv = |extra: &[&str]| {
            std::iter::once("prog".to_string())
                .chain(extra.iter().map(|s| s.to_string()))
                .collect::<Vec<String>>()
        };
        assert_eq!(cli_main::<TestArgs, _>("prog", "1.0.0", argv(&[]), ok), 0);
        // --version short-circuits ahead of clap.
        assert_eq!(
            cli_main::<TestArgs, _>("prog", "1.0.0", argv(&["--version"]), ok),
            0
        );
        assert_eq!(
            cli_main::<TestArgs, _>("prog", "1.0.0", argv(&["-V"]), ok),
            0
        );
        // clap usage error forwards clap's exit code (2).
        assert_eq!(
            cli_main::<TestArgs, _>("prog", "1.0.0", argv(&["--no-such-flag"]), ok),
            2
        );
        // run() error maps to 1.
        assert_eq!(
            cli_main::<TestArgs, _>("prog", "1.0.0", argv(&[]), |_| { Err("boom".into()) }),
            1
        );
    }
}
