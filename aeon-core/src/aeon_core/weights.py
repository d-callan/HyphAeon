"""
aeon_core/weights.py
--------------------
Handles discovery, download, and caching of Aeon model weights from Hugging Face.

Weights are hosted at https://huggingface.co/datamonkey/hyphaeon and that repo is
the source of truth. On first use, weights are downloaded and cached locally
in the Hugging Face cache directory (~/.cache/huggingface/ by default).
Subsequent runs use the cached copy.

The repo may contain multiple model variants (e.g. general, viral). Available
variants are enumerated dynamically from the HF Hub API so new variants appear
without a package update.
"""

import os
import sys
import json
import functools
from pathlib import Path
from typing import Optional, Dict, List, Tuple

# NumPy 1.x / 2.x unpickling compatibility bridge
try:
    import numpy as np
    if not hasattr(np, '_core') and hasattr(np, 'core'):
        sys.modules['numpy._core'] = np.core
        sys.modules['numpy._core.multiarray'] = np.core.multiarray
except Exception:
    pass

from huggingface_hub import list_repo_files, hf_hub_download
from huggingface_hub.constants import HF_HUB_CACHE

HF_REPO_ID = "datamonkey/hyphaeon"
DEFAULT_VARIANT = "general"
DEFAULT_CONFIG_FILENAME = "config.json"

# __metadata__ key under which self-describing .safetensors files carry their
# architecture config (JSON string). Written by save_safetensors(), read by
# load_arch_config().
ARCH_METADATA_KEY = "arch"

# Downloads go to the standard HF hub cache (HF_HUB_CACHE, env-controlled
# via HF_HUB_CACHE / HF_HOME) so we interoperate with other HF tooling
# instead of maintaining a private parallel layout.


def default_weights(env_var: str, fallback_env_vars: Tuple[str, ...] = ()) -> Optional[str]:
    """Default weights spec for a package CLI.

    Checks `env_var`, then each fallback (e.g. pre-refactor HYPHAEON_WEIGHTS),
    then a monorepo-checkout `model.safetensors` at the repo root. Returns the
    first set env var, or the repo-root file as a string if it exists, else None.
    """
    for var in (env_var,) + tuple(fallback_env_vars):
        value = os.environ.get(var)
        if value:
            return value
    repo_root_weights = Path(__file__).resolve().parents[3] / "model.safetensors"
    return str(repo_root_weights) if repo_root_weights.exists() else None


def default_variant(env_var: str, fallback_env_vars: Tuple[str, ...] = ()) -> str:
    """Default model variant for a package CLI (env var -> fallbacks -> default)."""
    for var in (env_var,) + tuple(fallback_env_vars):
        value = os.environ.get(var)
        if value:
            return value
    return DEFAULT_VARIANT


def list_available_variants() -> List[Dict[str, str]]:
    """
    Enumerate available model variants from the HF repo by listing .safetensors files.

    Returns a list of dicts with keys: variant, filename, description.
    The 'general' variant (model.safetensors) is listed first.
    """
    files = list_repo_files(HF_REPO_ID)
    variants = []

    for f in files:
        if f == "model.safetensors":
            variants.append({
                "variant": "general",
                "filename": f,
                "description": "General model (trained on diverse alignments)",
            })
        elif f.startswith("model.") and f.endswith(".safetensors"):
            # e.g. model.viral.safetensors -> variant "viral"
            variant_name = f[len("model."):-len(".safetensors")]
            variants.append({
                "variant": variant_name,
                "filename": f,
                "description": f"{variant_name} variant",
            })

    # General first, then alphabetical
    variants.sort(key=lambda v: (v["variant"] != "general", v["variant"]))
    return variants


def print_available_variants(cli_name: str = "aeon"):
    """
    Fetch and print available model variants from Hugging Face.

    Shared by all Aeon-family CLI tools (hyphaeon, chronaeon).

    Args:
        cli_name: Name of the calling CLI (e.g. 'hyphaeon', 'chronaeon')
                  used in the usage example line.
    """
    try:
        variants = list_available_variants()
    except Exception as e:
        print(f"[!] Could not fetch model list from Hugging Face: {e}")
        if "401" in str(e) or "Unauthorized" in str(e):
            print("    Could not authenticate with Hugging Face. If accessing a private repo, set HF_TOKEN env var.")
        return

    if not variants:
        print("No model variants found on Hugging Face.")
        return

    print(f"Available model variants ({HF_REPO_ID}):")
    print()
    for v in variants:
        default = " (default)" if v["variant"] == DEFAULT_VARIANT else ""
        print(f"  {v['variant']:15s}  {v['description']}{default}")
    print()
    print(f"Use with:  {cli_name} <subcommand> --model-variant <variant>")
    print(f"Default variant: {DEFAULT_VARIANT}")


def get_variant_filename(variant: str) -> str:
    """Map a variant name to its HF filename."""
    if variant == "general":
        return "model.safetensors"
    return f"model.{variant}.safetensors"


def resolve_weights_path(
    weights: Optional[str] = None,
    variant: Optional[str] = None,
) -> str:
    """
    Resolve the path to a weights file, downloading from HF if needed.

    Resolution order:
    1. If `weights` is an explicit path to an existing file, use it directly.
    2. If `variant` is specified (or default), check the HF cache.
    3. If not cached, download from HF and cache locally.

    Returns the local path to the weights file.
    """
    # 1. Explicit path takes precedence
    if weights:
        weights = os.fspath(weights)
        if os.path.exists(weights):
            return weights
        pkg_root_weights = Path(__file__).resolve().parent.parent.parent.parent / weights
        if pkg_root_weights.exists():
            return str(pkg_root_weights)
        raise FileNotFoundError(
            f"Explicit weights path does not exist: {weights}. "
            f"Omit --weights to use a Hugging Face variant instead."
        )

    # 2. Determine which variant to use
    v = variant or DEFAULT_VARIANT
    filename = get_variant_filename(v)

    # 3. Check HF cache (from prior hf_hub_download calls)
    try:
        from huggingface_hub import try_to_load_from_cache
        hf_cache_path = try_to_load_from_cache(
            repo_id=HF_REPO_ID, filename=filename,
        )
        if hf_cache_path is not None and os.path.exists(hf_cache_path):
            return hf_cache_path
    except Exception:
        pass

    # 4. Download from HF
    print(f"[*] Downloading HyphAeon weights ({v} variant) from Hugging Face...")
    try:
        downloaded = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=filename,
        )
        print(f"[✓] Weights cached at: {downloaded}")
        return downloaded
    except Exception as e:
        raise RuntimeError(
            f"Could not download weights from Hugging Face ({e}). "
            f"Specify --weights /path/to/checkpoint or set HYPHAEON_WEIGHTS env var."
        ) from e


@functools.lru_cache(maxsize=None)
def load_model_config(variant: Optional[str] = None) -> Dict:
    """
    Load model architecture config from the HF repo's config.json.

    All variants share the same architecture (config.json at repo root).
    If a variant-specific config exists (model.{variant}.config.json), it
    will be used instead.
    """
    v = variant or DEFAULT_VARIANT

    # Check for variant-specific config first
    variant_config = f"model.{v}.config.json"
    files = list_repo_files(HF_REPO_ID)
    if variant_config in files:
        path = hf_hub_download(repo_id=HF_REPO_ID, filename=variant_config)
        with open(path) as f:
            return json.load(f)

    # Fall back to shared config.json
    if DEFAULT_CONFIG_FILENAME in files:
        path = hf_hub_download(repo_id=HF_REPO_ID, filename=DEFAULT_CONFIG_FILENAME)
        with open(path) as f:
            return json.load(f)

    # No config available — return empty dict (CLI will use defaults)
    return {}


def save_safetensors(
    state_dict,
    path,
    arch: Optional[Dict] = None,
    metadata: Optional[Dict[str, str]] = None,
) -> str:
    """
    Save a state_dict as .safetensors with self-describing metadata.

    The safetensors header carries an arbitrary string->string __metadata__
    block; we write the architecture config there (JSON under 'arch') plus
    'format=pt' following HF convention, so the file can describe its own
    arch params without a companion config.json.

    Args:
        state_dict: Model state dict. Tensors are detached, moved to CPU, and
              made contiguous here, so callers can pass a live state_dict.
        path: Destination .safetensors path.
        arch: Architecture dict (embed_dim, num_layers, num_heads, window_size)
              embedded as JSON metadata. Strongly recommended.
        metadata: Additional string key/value metadata.

    Returns the written path.

    Note: safetensors rejects tensors sharing storage (tied/aliased weights);
    detach such tensors before calling.
    """
    from safetensors.torch import save_file

    md = {"format": "pt"}
    if metadata:
        md.update({k: str(v) for k, v in metadata.items()})
    if arch:
        md[ARCH_METADATA_KEY] = json.dumps(arch)

    tensors = {k: v.detach().cpu().contiguous() for k, v in state_dict.items()}
    path = os.fspath(path)
    save_file(tensors, path, metadata=md)
    return path


def _torch_load(path: str, map_location="cpu"):
    """torch.load a .pt checkpoint with the restricted (weights_only=True) unpickler.

    SECURITY: .pt files are pickles; the restricted unpickler refuses to
    execute arbitrary code. Checkpoints that fail under it (e.g. pickled
    custom objects or other non-tensor globals) are rejected outright —
    we deliberately do NOT fall back to weights_only=False, since that
    would let a crafted checkpoint execute arbitrary code. The standard
    .safetensors path (save_safetensors) carries no code and is always safe.
    """
    import torch
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except Exception as e:
        raise RuntimeError(
            f"Cannot safely load {path}: {e}. "
            f"This .pt checkpoint contains objects the restricted "
            f"(weights_only) loader refuses to deserialize — either a "
            f"legacy checkpoint or a crafted file. Unrestricted pickle "
            f"loading is not supported for security. To convert a trusted "
            f"legacy checkpoint, re-save it from its original environment "
            f"as self-describing weights: "
            f"aeon_core.weights.save_safetensors(state_dict, 'model.safetensors', arch=...)"
        ) from e


def _extract_state_dict(ckpt):
    """Pull a state_dict out of a .pt checkpoint dict (model_state_dict > state_dict > ckpt itself)."""
    if isinstance(ckpt, dict):
        return ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt
    return ckpt


def _remap_unified_keys(state_dict: Dict) -> Dict:
    """Map unified-suite prefixes onto standalone model names, if present.

    Checkpoints saved as a unified suite prefix backbone params with
    'backbone.' and the MEME head with 'head_meme.'; a standalone
    PhyloAxialTransformer expects unprefixed keys and 'lrt_ordinal_head.'.
    Applied for both .safetensors and .pt so prefixed heads round-trip
    regardless of container format. Only triggered when a 'backbone.' key
    exists, preserving verbatim passthrough otherwise.
    """
    if not any(k.startswith("backbone.") for k in state_dict.keys()):
        return state_dict
    mapped = {}
    for k, v in state_dict.items():
        if k.startswith("backbone."):
            mapped[k[len("backbone."):]] = v
        elif k.startswith("head_meme."):
            mapped["lrt_ordinal_head." + k[len("head_meme."):]] = v
        else:
            mapped[k] = v
    return mapped


def _load_from_resolved_path(path: str, map_location="cpu") -> Tuple:
    """Open a resolved weights file once -> (raw_checkpoint_or_None, state_dict).

    For .safetensors, raw is None (there is no checkpoint wrapper). For .pt,
    raw is the loaded checkpoint dict, which also carries arch params.
    """
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file
        return None, _remap_unified_keys(load_file(path, device=str(map_location)))
    ckpt = _torch_load(path, map_location)
    return ckpt, _remap_unified_keys(_extract_state_dict(ckpt))


def load_weights(
    weights: Optional[str] = None,
    variant: Optional[str] = None,
    map_location="cpu",
):
    """
    Load model weights as a state_dict.

    Supports both .pt and .safetensors formats. For .pt files, extracts
    model_state_dict/state_dict. Unified-suite prefixed keys are remapped
    for both formats.
    """
    path = os.fspath(resolve_weights_path(weights=weights, variant=variant))
    return _load_from_resolved_path(path, map_location)[1]


# Default architecture parameters, used when no config is available.
_DEFAULT_ARCH = {"embed_dim": 384, "num_layers": 6, "num_heads": 12, "window_size": 1}


# Canonical arch keys, plus the legacy spellings seen in train.py args dicts
# (canonical -> alias).
_ARCH_KEYS = ("embed_dim", "num_layers", "num_heads", "window_size")
_ARCH_ALIASES = {"num_layers": "layers", "num_heads": "heads"}


def _normalize_arch(a: Dict, source: Optional[str] = None) -> dict:
    """Map various arch-param key spellings onto the canonical names.

    Missing fields fall back to _DEFAULT_ARCH. When `source` is given, a
    warning lists which fields were defaulted, so a partial config can't
    silently masquerade as complete.
    """
    arch = {
        "embed_dim": a.get("embed_dim", _DEFAULT_ARCH["embed_dim"]),
        "num_layers": a.get("num_layers", a.get("layers", _DEFAULT_ARCH["num_layers"])),
        "num_heads": a.get("num_heads", a.get("heads", _DEFAULT_ARCH["num_heads"])),
        "window_size": a.get("window_size", _DEFAULT_ARCH["window_size"]),
    }
    missing = [k for k in _ARCH_KEYS
               if k not in a and _ARCH_ALIASES.get(k) not in a]
    if missing and source:
        defaulted = {k: arch[k] for k in missing}
        print(f"[!] {source}: arch param(s) {missing} not found; assuming defaults {defaulted}.")
    return arch


def _arch_from_safetensors_metadata(path: str) -> Optional[Dict]:
    """Read embedded arch config from a .safetensors __metadata__ block, if present."""
    try:
        from safetensors import safe_open
        with safe_open(path, framework="pt") as f:
            md = f.metadata() or {}
        raw = md.get(ARCH_METADATA_KEY)
        if raw:
            return json.loads(raw)
    except Exception:
        pass
    return None


def _arch_from_sibling_config(path: str) -> Tuple[Optional[Path], Optional[Dict]]:
    """Read arch config from <stem>.config.json or config.json beside the weights file.

    Returns (config_path, config_dict) or (None, None).
    """
    p = Path(path)
    candidates = [p.with_suffix(".config.json"), p.parent / DEFAULT_CONFIG_FILENAME]
    for c in candidates:
        if c.exists():
            try:
                with open(c) as f:
                    return c, json.load(f)
            except Exception:
                pass
    return None, None


def _arch_from_pt(ckpt, path: str) -> dict:
    """Arch config from a loaded .pt checkpoint: 'args' dict, else top-level arch keys.

    A bare state_dict (tensor-valued keys, no arch keys) is NOT mistaken for
    an args dict — it yields defaults *with* a warning.
    """
    a = ckpt.get("args") if isinstance(ckpt, dict) else None
    if not a and isinstance(ckpt, dict):
        a = {k: ckpt[k] for k in _ARCH_KEYS + tuple(_ARCH_ALIASES.values())
             if k in ckpt}
    if a:
        return _normalize_arch(a, f"{path} (checkpoint args)")
    print(f"[!] {path}: no arch params in checkpoint; "
          f"assuming default arch params {_DEFAULT_ARCH}.")
    return dict(_DEFAULT_ARCH)


def _is_explicit_local(weights: Optional[str]) -> bool:
    """Whether a `weights` spec is an explicit local path.

    A path under the HF hub cache (HF_HUB_CACHE) is a variant-resolved HF
    artifact, not an explicit local file — this lets callers that pre-resolve
    via resolve_weights_path keep full arch-resolution semantics (the HF
    config.json fallback stays reachable for HF files lacking embedded
    metadata or a sibling config).
    """
    # HF_HUB_CACHE is a str; parents are Path objects — normalize before
    # membership test or the comparison is silently always-True.
    return bool(weights) and Path(HF_HUB_CACHE).resolve() not in \
        Path(weights).resolve().parents


def _resolve_arch(path: str, variant: Optional[str], is_explicit_local: bool) -> dict:
    """Arch config for non-.pt weights: embedded metadata -> sibling config ->
    HF config (variant flows only) -> defaults with a warning."""
    meta = _arch_from_safetensors_metadata(path)
    if meta:
        return _normalize_arch(meta, f"{path} (embedded metadata)")

    sibling_path, sibling = _arch_from_sibling_config(path)
    if sibling:
        return _normalize_arch(sibling, str(sibling_path))

    # HF config only makes sense when the file came from the HF variant flow;
    # skip the network call for an explicit local file that isn't self-describing.
    if not is_explicit_local:
        try:
            cfg = load_model_config(variant=variant)
            if cfg:
                return _normalize_arch(cfg, f"{HF_REPO_ID} config.json")
        except Exception:
            pass

    sources = ("embedded metadata or sibling config" if is_explicit_local
               else "embedded metadata, sibling config, or HF config")
    print(f"[!] Could not determine architecture for {path} "
          f"(no {sources}); "
          f"assuming default arch params {_DEFAULT_ARCH}. "
          f"To embed arch params in the file, re-save via aeon_core.weights.save_safetensors().")
    return dict(_DEFAULT_ARCH)


def load_arch_config(
    weights: Optional[str] = None,
    variant: Optional[str] = None,
) -> dict:
    """
    Determine architecture hyperparameters (embed_dim, num_layers, num_heads, window_size).

    Resolution order:
      .pt            : 'args' dict embedded in the checkpoint, else top-level
                       arch keys. Bare state_dicts warn and use defaults.
      .safetensors   : __metadata__['arch'] (written by save_safetensors)
                       -> sibling <stem>.config.json / config.json
                       -> HF config.json (only if resolving by variant)
      fallback       : _DEFAULT_ARCH (warns — arch params are assumed).

    Prefer load_checkpoint() when you also need the state_dict, to avoid
    opening the file twice.
    """
    path = os.fspath(resolve_weights_path(weights=weights, variant=variant))
    if path.endswith(".pt"):
        return _arch_from_pt(_torch_load(path), path)
    return _resolve_arch(path, variant, _is_explicit_local(weights))


def load_checkpoint(
    weights: Optional[str] = None,
    variant: Optional[str] = None,
    map_location="cpu",
) -> Tuple[str, dict, Dict]:
    """Resolve a weights source and open it once.

    Returns (resolved_path, arch_config, state_dict). Use this instead of
    calling resolve_weights_path/load_arch_config/load_weights separately —
    for .pt files those each deserialize the whole checkpoint.
    """
    path = os.fspath(resolve_weights_path(weights=weights, variant=variant))
    raw, state_dict = _load_from_resolved_path(path, map_location)
    arch = (_arch_from_pt(raw, path) if path.endswith(".pt")
            else _resolve_arch(path, variant, _is_explicit_local(weights)))
    return path, arch, state_dict
