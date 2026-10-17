"""Python-level contract tests for the ``rhizaeon`` console script.

The clap parser lives inside the compiled ``rhizaeon._native`` extension
and writes to process file descriptors directly, so output capture must
use ``capfd`` — Python-level ``capsys`` cannot see Rust's ``println!``.
"""

import re
import sys

import pytest

import rhizaeon
import rhizaeon.cli
from rhizaeon._native import rhizaeon_cli

# PEP 440 release segment plus optional prerelease (a/b/rc/.devN).
PEP440_VERSION = re.compile(r"^rhizaeon \d+\.\d+\.\d+(?:(?:a|b|rc)\d+|\.dev\d+)?$")


def run(*args: str) -> int:
    return rhizaeon_cli(["rhizaeon", *args])


@pytest.mark.parametrize("flag", ["--version", "-V"])
def test_version_flag_pep440(capfd, flag):
    assert run(flag) == 0
    out = capfd.readouterr().out.strip()
    assert PEP440_VERSION.match(out), out
    # The native engine and the Python package must not drift apart.
    assert out == f"rhizaeon {rhizaeon.__version__}"


def test_help_exits_zero_and_names_tool(capfd):
    assert run("--help") == 0
    out = capfd.readouterr().out
    assert "rhizaeon" in out
    assert "__main__" not in out


def test_missing_input_is_clap_error(capfd):
    assert run() == 2
    err = capfd.readouterr().err
    assert "required" in err
    assert "--input" in err


@pytest.mark.parametrize("flag", ["-f", "--fasta", "-i", "--input"])
def test_all_input_aliases_parse(capfd, flag):
    # A nonexistent path parses fine and fails later at runtime (rc 1),
    # whereas a parse error would return clap's exit code 2.
    assert run(flag, "/nonexistent_xyz.fa") == 1
    err = capfd.readouterr().err
    assert "unexpected argument" not in err


def test_viz_is_long_only(capfd):
    # `-v` was reserved for --version style muscle memory; the dossier
    # output flag is long-only so the short flag stays free.
    assert run("-v", "out.png") == 2
    err = capfd.readouterr().err
    assert "unexpected argument '-v'" in err

    # `--viz` parses; the failure must be the missing required input,
    # not an unrecognized flag.
    assert run("--viz", "out.png") == 2
    err = capfd.readouterr().err
    assert "required" in err
    assert "unexpected argument" not in err


def test_runtime_error_uses_error_prefix(capfd):
    assert run("--fasta", "/nonexistent_xyz.fa") == 1
    err = capfd.readouterr().err
    assert "Error:" in err


def test_shim_normalizes_argv0(capfd, monkeypatch):
    # Whatever sys.argv[0] happens to be (console script path,
    # `python -m`, pytest runner), clap must see the tool name so help
    # and error text read `rhizaeon ...`.
    monkeypatch.setattr(sys, "argv", ["/weird/path/__main__.py", "--version"])
    with pytest.raises(SystemExit) as exc:
        rhizaeon.cli.main()
    assert exc.value.code == 0
    out = capfd.readouterr().out.strip()
    assert out.startswith("rhizaeon ")


def test_main_module_delegates_to_cli_main():
    import rhizaeon.__main__

    assert rhizaeon.__main__.main is rhizaeon.cli.main
