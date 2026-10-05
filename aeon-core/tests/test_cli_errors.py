"""Tests for aeon_core.cli shared CLI error handling."""

import pytest

from aeon_core.cli import handle_cli_errors


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
