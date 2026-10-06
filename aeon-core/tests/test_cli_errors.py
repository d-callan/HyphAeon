"""Tests for aeon_core.cli shared CLI error handling."""

import argparse
from types import SimpleNamespace

import pytest

from aeon_core.cli import handle_cli_errors, add_weights_args, weights_kwargs


class TestHandleCliErrors:
    def test_file_not_found_clean_exit(self, capsys):
        @handle_cli_errors
        def main():
            raise FileNotFoundError("bad weights path")

        with pytest.raises(SystemExit) as ei:
            main()
        assert ei.value.code == 1
        assert "bad weights path" in capsys.readouterr().err

    def test_runtime_error_clean_exit(self, capsys):
        @handle_cli_errors
        def main():
            raise RuntimeError("incompatible weights file")

        with pytest.raises(SystemExit) as ei:
            main()
        assert ei.value.code == 1
        assert "incompatible weights file" in capsys.readouterr().err

    def test_other_errors_keep_traceback(self):
        @handle_cli_errors
        def main():
            raise ValueError("a code bug")

        # Bugs outside CLI_ERROR_TYPES must NOT be swallowed.
        with pytest.raises(ValueError):
            main()


class TestAddWeightsArgs:
    """The shared -w/--weights + --model-variant declaration."""

    def test_attaches_both_flags(self):
        p = argparse.ArgumentParser()
        add_weights_args(p)
        args = p.parse_args([])
        assert hasattr(args, "weights")
        assert args.variant == "general"

    def test_env_var_supplies_weights_default(self, monkeypatch, tmp_path):
        w = tmp_path / "w.pt"
        w.write_text("x")
        monkeypatch.setenv("TESTPKG_WEIGHTS", str(w))
        p = argparse.ArgumentParser()
        add_weights_args(p, weights_envs=("TESTPKG_WEIGHTS",))
        assert p.parse_args([]).weights == str(w)

    def test_env_var_supplies_variant_default(self, monkeypatch):
        monkeypatch.setenv("TESTPKG_VARIANT", "viral")
        p = argparse.ArgumentParser()
        add_weights_args(p, variant_envs=("TESTPKG_VARIANT",))
        assert p.parse_args([]).variant == "viral"

    def test_env_precedence_order(self, monkeypatch):
        """First env var wins; fallbacks only consulted when primary unset."""
        monkeypatch.setenv("PKG_VARIANT", "primary")
        monkeypatch.setenv("OLD_VARIANT", "fallback")
        p = argparse.ArgumentParser()
        add_weights_args(p, variant_envs=("PKG_VARIANT", "OLD_VARIANT"))
        assert p.parse_args([]).variant == "primary"
        monkeypatch.delenv("PKG_VARIANT")
        p2 = argparse.ArgumentParser()
        add_weights_args(p2, variant_envs=("PKG_VARIANT", "OLD_VARIANT"))
        assert p2.parse_args([]).variant == "fallback"

    def test_cli_flag_overrides_env_default(self, monkeypatch):
        monkeypatch.setenv("PKG_VARIANT", "envvar")
        p = argparse.ArgumentParser()
        add_weights_args(p, variant_envs=("PKG_VARIANT",))
        assert p.parse_args(["--model-variant", "flagged"]).variant == "flagged"


class TestWeightsKwargs:
    """The shared forwarding contract: {weights, variant} from parsed args."""

    def test_returns_standard_keys(self):
        args = SimpleNamespace(weights="/x.pt", variant="viral")
        assert weights_kwargs(args) == {"weights": "/x.pt", "variant": "viral"}

    def test_missing_attrs_fail_loudly(self):
        """A parser that skipped add_weights_args must AttributeError at the
        call site — never silently drop the flags."""
        with pytest.raises(AttributeError):
            weights_kwargs(SimpleNamespace())

    def test_round_trip_with_add_weights_args(self):
        p = argparse.ArgumentParser()
        add_weights_args(p)
        args = p.parse_args(["-w", "/m.pt", "--model-variant", "viral"])
        assert weights_kwargs(args) == {"weights": "/m.pt", "variant": "viral"}
