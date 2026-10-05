"""
Tests for hyphaeon/weights.py — HF weights discovery, download, and loading.

These tests mock the huggingface_hub API calls so they don't require network
access or a valid HF token. Tests that do require HF access are skipped if
HF_TOKEN is not set or the repo is unreachable.
"""

import os
import json
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from aeon_core.weights import (
    HF_REPO_ID,
    DEFAULT_VARIANT,
    ARCH_METADATA_KEY,
    get_variant_filename,
    resolve_weights_path,
    load_model_config,
    load_arch_config,
    load_weights,
    load_checkpoint,
    _torch_load,
    save_safetensors,
    list_available_variants,
)


@pytest.fixture(autouse=True)
def _clear_config_cache():
    """load_model_config is lru_cached; clear it so per-test mocks apply."""
    load_model_config.cache_clear()
    yield
    load_model_config.cache_clear()


class TestGetVariantFilename:
    def test_general_variant(self):
        assert get_variant_filename("general") == "model.safetensors"

    def test_viral_variant(self):
        assert get_variant_filename("viral") == "model.viral.safetensors"

    def test_arbitrary_variant(self):
        assert get_variant_filename("custom") == "model.custom.safetensors"


class TestResolveWeightsPath:
    def test_explicit_path_takes_precedence(self, tmp_path):
        """If --weights points to an existing file, use it directly."""
        weights_file = tmp_path / "model.pt"
        weights_file.write_text("dummy")
        result = resolve_weights_path(weights=str(weights_file), variant="general")
        assert result == str(weights_file)

    def test_nonexistent_explicit_path_raises(self, tmp_path):
        """If --weights points to a nonexistent file, raise instead of silently downloading."""
        nonexistent = str(tmp_path / "does_not_exist.pt")
        with pytest.raises(FileNotFoundError):
            resolve_weights_path(weights=nonexistent, variant="general")

    def test_nonexistent_explicit_path_does_not_download(self, tmp_path):
        """A bad --weights path must not silently fall through to HF download."""
        nonexistent = str(tmp_path / "does_not_exist.pt")
        with patch("aeon_core.weights.hf_hub_download") as mock_dl, \
             pytest.raises(FileNotFoundError):
            resolve_weights_path(weights=nonexistent, variant="general")
        mock_dl.assert_not_called()

    def test_cached_variant_skips_download(self, tmp_path):
        """If the variant is already in the HF cache, don't download."""
        cache_file = str(tmp_path / "model.safetensors")
        Path(cache_file).write_text("cached")
        with patch("huggingface_hub.try_to_load_from_cache", return_value=cache_file), \
             patch("aeon_core.weights.hf_hub_download") as mock_dl:
            result = resolve_weights_path(weights=None, variant="general")
            assert result == cache_file
            mock_dl.assert_not_called()

    def test_download_failure_raises_error(self, tmp_path):
        """If HF download fails and no local weights exist, raise RuntimeError."""
        with patch("huggingface_hub.try_to_load_from_cache", return_value=None), \
             patch("aeon_core.weights.hf_hub_download", side_effect=Exception("401 Unauthorized")):
            with pytest.raises(RuntimeError, match="Could not download weights"):
                resolve_weights_path(weights=None, variant="general")


class TestListAvailableVariants:
    def test_parses_safetensors_files(self):
        """Should extract variant names from model.*.safetensors filenames."""
        mock_files = [
            ".gitattributes",
            "README.md",
            "config.json",
            "model.safetensors",
            "model.viral.safetensors",
            "model.viral.pt",
            "model.viral.onnx",
        ]
        with patch("aeon_core.weights.list_repo_files", return_value=mock_files):
            variants = list_available_variants()
        names = [v["variant"] for v in variants]
        assert "general" in names
        assert "viral" in names
        assert len(variants) == 2  # only .safetensors files

    def test_general_listed_first(self):
        """General variant should be listed first."""
        mock_files = ["model.viral.safetensors", "model.safetensors", "config.json"]
        with patch("aeon_core.weights.list_repo_files", return_value=mock_files):
            variants = list_available_variants()
        assert variants[0]["variant"] == "general"

    def test_no_safetensors_returns_empty(self):
        mock_files = ["config.json", "README.md"]
        with patch("aeon_core.weights.list_repo_files", return_value=mock_files):
            variants = list_available_variants()
        assert len(variants) == 0


class TestLoadModelConfig:
    def test_loads_shared_config(self, tmp_path):
        """Should load config.json from HF when no variant-specific config exists."""
        config = {"embed_dim": 384, "num_layers": 6, "num_heads": 12}
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(config))

        mock_files = ["config.json", "model.safetensors"]
        with patch("aeon_core.weights.list_repo_files", return_value=mock_files), \
             patch("aeon_core.weights.hf_hub_download", return_value=str(config_path)):
            result = load_model_config(variant="general")
        assert result["embed_dim"] == 384
        assert result["num_layers"] == 6

    def test_variant_specific_config_preferred(self, tmp_path):
        """If model.{variant}.config.json exists, use it over the shared config.json."""
        general_config = {"embed_dim": 384, "num_layers": 6}
        viral_config = {"embed_dim": 512, "num_layers": 8}

        general_path = tmp_path / "config.json"
        general_path.write_text(json.dumps(general_config))
        viral_path = tmp_path / "model.viral.config.json"
        viral_path.write_text(json.dumps(viral_config))

        mock_files = ["config.json", "model.safetensors", "model.viral.safetensors", "model.viral.config.json"]

        def fake_download(repo_id, filename, **kwargs):
            return str(tmp_path / filename)

        with patch("aeon_core.weights.list_repo_files", return_value=mock_files), \
             patch("aeon_core.weights.hf_hub_download", side_effect=fake_download):
            result = load_model_config(variant="viral")
        assert result["embed_dim"] == 512
        assert result["num_layers"] == 8

    def test_no_config_returns_empty(self):
        """If no config files exist on HF, return empty dict (CLI uses defaults)."""
        mock_files = ["model.safetensors"]
        with patch("aeon_core.weights.list_repo_files", return_value=mock_files):
            result = load_model_config(variant="general")
        assert result == {}


class TestLoadWeights:
    def test_load_safetensors(self, tmp_path):
        """Should load a .safetensors file directly as a state_dict."""
        pytest.importorskip("safetensors")
        from safetensors.torch import save_file

        state_dict = {"weight": torch.zeros(3, 3)}
        st_path = tmp_path / "model.safetensors"
        save_file(state_dict, str(st_path))

        result = load_weights(weights=str(st_path), variant="general")
        assert "weight" in result
        assert result["weight"].shape == (3, 3)

    def test_load_pt_with_model_state_dict(self, tmp_path):
        """Should extract model_state_dict from a .pt checkpoint."""
        state_dict = {"weight": torch.ones(2, 2)}
        ckpt = {"epoch": 10, "model_state_dict": state_dict, "optimizer_state_dict": {}}
        pt_path = tmp_path / "model.pt"
        torch.save(ckpt, str(pt_path))

        result = load_weights(weights=str(pt_path), variant="general")
        assert "weight" in result
        assert result["weight"].shape == (2, 2)

    def test_load_pt_with_state_dict_key(self, tmp_path):
        """Should extract state_dict from a .pt checkpoint with 'state_dict' key."""
        state_dict = {"weight": torch.ones(2, 2)}
        ckpt = {"state_dict": state_dict}
        pt_path = tmp_path / "model.pt"
        torch.save(ckpt, str(pt_path))

        result = load_weights(weights=str(pt_path), variant="general")
        assert "weight" in result

    def test_load_pt_unified_suite_remaps_keys(self, tmp_path):
        """Unified-suite .pt (backbone./head_meme. prefixes) remaps like .safetensors."""
        sd = {"backbone.layer.weight": torch.ones(2, 2),
              "head_meme.fc.weight": torch.zeros(3),
              "other.weight": torch.ones(1)}
        pt_path = tmp_path / "model.pt"
        torch.save({"model_state_dict": sd}, str(pt_path))

        result = load_weights(weights=str(pt_path))
        assert "layer.weight" in result
        assert "lrt_ordinal_head.fc.weight" in result
        assert "other.weight" in result
        assert not any(k.startswith("backbone.") for k in result)

    def test_load_checkpoint_single_open(self, tmp_path):
        """load_checkpoint returns (path, arch, state_dict) from one file open."""
        ckpt = {"model_state_dict": {"w": torch.ones(2, 2)},
                "args": {"embed_dim": 256, "num_layers": 4, "num_heads": 8,
                          "window_size": 2}}
        pt_path = tmp_path / "model.pt"
        torch.save(ckpt, str(pt_path))

        path, arch, sd = load_checkpoint(weights=str(pt_path))
        assert path == str(pt_path)
        assert arch["embed_dim"] == 256 and arch["window_size"] == 2
        assert torch.equal(sd["w"], torch.ones(2, 2))

    def test_load_pt_raw_state_dict(self, tmp_path):
        """Should handle a .pt file that is just a raw state_dict (no wrapper)."""
        state_dict = {"weight": torch.ones(2, 2)}
        pt_path = tmp_path / "model.pt"
        torch.save(state_dict, str(pt_path))

        result = load_weights(weights=str(pt_path), variant="general")
        assert "weight" in result


class TestLoadArchConfig:
    def test_load_from_pt_checkpoint(self, tmp_path):
        """Should extract architecture config from a .pt checkpoint's args dict."""
        state_dict = {"weight": torch.ones(2, 2)}
        ckpt = {"args": {"embed_dim": 512, "layers": 8, "heads": 16, "window_size": 2},
                "model_state_dict": state_dict}
        pt_path = tmp_path / "model.pt"
        torch.save(ckpt, str(pt_path))

        config = load_arch_config(weights=str(pt_path))
        assert config["embed_dim"] == 512
        assert config["num_layers"] == 8
        assert config["num_heads"] == 16
        assert config["window_size"] == 2

    def test_pt_with_num_layers_key(self, tmp_path):
        """Should handle checkpoints that use 'num_layers' instead of 'layers'."""
        state_dict = {"weight": torch.ones(2, 2)}
        ckpt = {"args": {"embed_dim": 256, "num_layers": 4, "num_heads": 8, "window_size": 1},
                "model_state_dict": state_dict}
        pt_path = tmp_path / "model.pt"
        torch.save(ckpt, str(pt_path))

        config = load_arch_config(weights=str(pt_path))
        assert config["num_layers"] == 4
        assert config["num_heads"] == 8

    def test_pt_bare_state_dict_warns_and_uses_defaults(self, tmp_path, capsys):
        """A bare .pt state_dict has no arch info — must warn, not silently default."""
        state_dict = {"weight": torch.ones(2, 2)}
        pt_path = tmp_path / "model.pt"
        torch.save(state_dict, str(pt_path))

        config = load_arch_config(weights=str(pt_path))
        assert config["embed_dim"] == 384
        assert config["num_layers"] == 6
        assert "assuming default arch params" in capsys.readouterr().out

    def test_pt_args_missing_fields_warns(self, tmp_path, capsys):
        """A partial args dict warns about fields that fall back to defaults."""
        ckpt = {"model_state_dict": {"w": torch.zeros(1)},
                "args": {"embed_dim": 512}}
        pt_path = tmp_path / "model.pt"
        torch.save(ckpt, str(pt_path))

        config = load_arch_config(weights=str(pt_path))
        assert config["embed_dim"] == 512
        out = capsys.readouterr().out
        assert "window_size" in out and "assuming defaults" in out


class _UnpicklableByRestriction:
    """Module-level custom class: picklable, but rejected by the restricted
    (weights_only=True) unpickler — exercises the hard-rejection path."""


class TestTorchLoadSecurity:
    def test_unsafe_objects_rejected(self, tmp_path, capsys):
        """weights_only failure raises an actionable error — no unsafe fallback."""
        p = tmp_path / "legacy.pt"
        torch.save({"args": _UnpicklableByRestriction()}, str(p))

        with pytest.raises(RuntimeError, match="Cannot safely load"):
            _torch_load(str(p))
        out = capsys.readouterr().out
        assert out == ""  # no fallback warning: the load is refused outright

    def test_safe_load_no_warning(self, tmp_path, capsys):
        """Restriction-compatible checkpoints load with weights_only silently."""
        p = tmp_path / "ok.pt"
        torch.save({"model_state_dict": {"w": torch.zeros(1)}}, str(p))

        ckpt = _torch_load(str(p))
        assert "w" in ckpt["model_state_dict"]
        assert capsys.readouterr().out == ""


class TestLoadModelKeyMismatch:
    def _arch(self):
        return {"embed_dim": 16, "num_layers": 1, "num_heads": 2, "window_size": 1}

    def test_mismatched_keys_warn(self, tmp_path, capsys):
        """Unexpected keys / missing backbone keys must warn, not stay silent."""
        pytest.importorskip("safetensors")
        from aeon_core.inference import load_model
        st = tmp_path / "mismatch.safetensors"
        save_safetensors({"bogus.weight": torch.zeros(2)}, str(st), arch=self._arch())

        load_model(weights=str(st), device="cpu")
        out = capsys.readouterr().out
        assert "Weight key mismatch" in out
        assert "bogus.weight" in out

    def test_matching_keys_silent(self, tmp_path, capsys):
        """A fully-matching checkpoint produces no mismatch warning."""
        pytest.importorskip("safetensors")
        from aeon_core.inference import load_model
        from aeon_core.model import PhyloAxialTransformer
        model = PhyloAxialTransformer(**self._arch())
        st = tmp_path / "match.safetensors"
        save_safetensors(model.state_dict(), str(st), arch=self._arch())

        load_model(weights=str(st), device="cpu")
        assert "Weight key mismatch" not in capsys.readouterr().out

    def test_extra_head_keys_silent(self, tmp_path, capsys):
        """Unified checkpoints carry heads the backbone ignores (head_busted.,
        etc.) — unexpected keys under head_* must not warn."""
        pytest.importorskip("safetensors")
        from aeon_core.inference import load_model
        from aeon_core.model import PhyloAxialTransformer
        model = PhyloAxialTransformer(**self._arch())
        sd = model.state_dict()
        sd["head_busted.fc.weight"] = torch.zeros(4, 4)
        st = tmp_path / "unified.safetensors"
        save_safetensors(sd, str(st), arch=self._arch())

        load_model(weights=str(st), device="cpu")
        assert "Weight key mismatch" not in capsys.readouterr().out


class TestSaveSafetensors:
    def test_roundtrip_with_arch_metadata(self, tmp_path):
        """save_safetensors embeds arch params readable via load_arch_config."""
        pytest.importorskip("safetensors")
        st_path = tmp_path / "model.safetensors"
        save_safetensors(
            {"weight": torch.ones(2, 2)},
            str(st_path),
            arch={"embed_dim": 512, "num_layers": 8, "num_heads": 16, "window_size": 3},
        )
        config = load_arch_config(weights=str(st_path))
        assert config["embed_dim"] == 512
        assert config["num_layers"] == 8
        assert config["num_heads"] == 16
        assert config["window_size"] == 3
        # state_dict still loads back identically
        sd = load_weights(weights=str(st_path))
        assert torch.equal(sd["weight"], torch.ones(2, 2))

    def test_noncontiguous_tensors_handled(self, tmp_path):
        """save_safetensors normalizes non-contiguous tensors internally."""
        pytest.importorskip("safetensors")
        st_path = tmp_path / "model.safetensors"
        save_safetensors({"w": torch.ones(2, 3).t()},  # non-contiguous
                         str(st_path), arch={"embed_dim": 64})
        sd = load_weights(weights=str(st_path))
        assert sd["w"].shape == (3, 2)

    def test_writes_metadata_header(self, tmp_path):
        """The __metadata__ block carries 'arch' JSON and format=pt."""
        pytest.importorskip("safetensors")
        from safetensors import safe_open
        st_path = tmp_path / "model.safetensors"
        save_safetensors({"w": torch.zeros(1)}, str(st_path),
                         arch={"embed_dim": 128}, metadata={"variant": "test"})
        with safe_open(str(st_path), framework="pt") as f:
            md = f.metadata()
        assert md["format"] == "pt"
        assert md["variant"] == "test"
        assert json.loads(md[ARCH_METADATA_KEY])["embed_dim"] == 128


class TestSafetensorsArchResolution:
    def test_sibling_stem_config_json(self, tmp_path):
        """model.safetensors reads model.config.json beside it when no metadata."""
        pytest.importorskip("safetensors")
        from safetensors.torch import save_file
        st_path = tmp_path / "model.safetensors"
        save_file({"w": torch.zeros(1)}, str(st_path))
        (tmp_path / "model.config.json").write_text(json.dumps({"embed_dim": 256}))
        config = load_arch_config(weights=str(st_path))
        assert config["embed_dim"] == 256

    def test_sibling_plain_config_json(self, tmp_path):
        """Falls back to plain config.json in the same directory."""
        pytest.importorskip("safetensors")
        from safetensors.torch import save_file
        st_path = tmp_path / "model.safetensors"
        save_file({"w": torch.zeros(1)}, str(st_path))
        (tmp_path / "config.json").write_text(json.dumps({"embed_dim": 192}))
        config = load_arch_config(weights=str(st_path))
        assert config["embed_dim"] == 192

    def test_metadata_beats_sibling(self, tmp_path):
        """Embedded __metadata__ takes precedence over sibling config files."""
        pytest.importorskip("safetensors")
        st_path = tmp_path / "model.safetensors"
        save_safetensors({"w": torch.zeros(1)}, str(st_path), arch={"embed_dim": 999})
        (tmp_path / "config.json").write_text(json.dumps({"embed_dim": 256}))
        config = load_arch_config(weights=str(st_path))
        assert config["embed_dim"] == 999

    def test_no_info_warns_and_uses_defaults(self, tmp_path, capsys):
        """Local safetensors with no arch info warns and returns defaults — no HF call."""
        pytest.importorskip("safetensors")
        from safetensors.torch import save_file
        st_path = tmp_path / "model.safetensors"
        save_file({"w": torch.zeros(1)}, str(st_path))
        with patch("aeon_core.weights.load_model_config") as mock_cfg:
            config = load_arch_config(weights=str(st_path))
        mock_cfg.assert_not_called()
        assert config["embed_dim"] == 384
        assert "assuming default arch params" in capsys.readouterr().out

    def test_hf_cache_path_keeps_hf_config_fallback(self, tmp_path):
        """A resolved path under the HF hub cache is NOT 'explicit local' —
        callers that pre-resolve via resolve_weights_path must still reach
        load_model_config for HF files without embedded metadata."""
        pytest.importorskip("safetensors")
        from safetensors.torch import save_file
        # File lives under the (patched) HF cache root, like a real
        # hf_hub_download result handed back in as `weights`.
        hf_cache = tmp_path / "huggingface"
        st_path = hf_cache / "hub" / "models--x--y" / "snapshots" / "abc" / "model.safetensors"
        st_path.parent.mkdir(parents=True)
        save_file({"w": torch.zeros(1)}, str(st_path))

        with patch("aeon_core.weights.HF_HUB_CACHE", str(hf_cache)), \
             patch("aeon_core.weights.load_model_config",
                   return_value={"embed_dim": 512}) as mock_cfg:
            config = load_arch_config(weights=str(st_path), variant="general")
        mock_cfg.assert_called_once()
        assert config["embed_dim"] == 512


@pytest.mark.skipif(not os.environ.get("HF_TOKEN"), reason="HF_TOKEN not set")
class TestHFAccess:
    """Integration tests that actually hit Hugging Face. Skipped without HF_TOKEN."""

    def test_list_variants_real(self):
        variants = list_available_variants()
        assert len(variants) >= 1
        assert any(v["variant"] == "general" for v in variants)

    def test_load_config_real(self):
        config = load_model_config(variant="general")
        assert "embed_dim" in config
        assert config["embed_dim"] == 384
