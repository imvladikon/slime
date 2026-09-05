"""Version-independent GLM-5.3-Flash configuration loading and validation."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any


def checkpoint_tensor_names(path: str | Path) -> list[str]:
    """Read checkpoint tensor names from the index or safetensors headers only."""
    checkpoint = Path(path)
    index_path = checkpoint / "model.safetensors.index.json"
    if index_path.is_file():
        with index_path.open(encoding="utf-8") as stream:
            return sorted(json.load(stream)["weight_map"])

    from safetensors import safe_open

    names: list[str] = []
    for shard in sorted(checkpoint.glob("*.safetensors")):
        with safe_open(shard, framework="pt", device="cpu") as tensors:
            names.extend(tensors.keys())
    if not names:
        raise FileNotFoundError(f"No safetensors checkpoint found in {checkpoint}")
    return sorted(names)


def _namespace(value: Any) -> Any:
    if isinstance(value, dict):
        return SimpleNamespace(**{key: _namespace(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_namespace(item) for item in value]
    return value


def get_text_config(config):
    """Return the nested text config used by the released VLM checkpoint."""
    return getattr(config, "text_config", None) or config


def load_glm5_next_config(path: str | Path):
    """Load a local ``glm5_next`` config without relying on Transformers registration."""
    config_path = Path(path) / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"GLM-5.3 config must be a local file: {config_path}")
    with config_path.open(encoding="utf-8") as stream:
        config = _namespace(json.load(stream))
    validate_glm5_next_config(config)
    return config


def attention_schedules(text_config) -> tuple[list[int], list[int]]:
    """Return zero-based KDA and DSA layer indices from either supported schema."""
    linear = getattr(text_config, "linear_attn_config", None)
    layer_types = getattr(text_config, "layer_types", None)
    if isinstance(linear, SimpleNamespace):
        kda = getattr(linear, "kda_layers", None)
        dsa = getattr(linear, "full_attn_layers", None)
        if kda is not None and dsa is not None:
            explicit = sorted(int(layer) for layer in kda), sorted(int(layer) for layer in dsa)
            if layer_types is not None:
                derived = (
                    [index for index, kind in enumerate(layer_types) if kind == "linear_attention"],
                    [index for index, kind in enumerate(layer_types) if kind != "linear_attention"],
                )
                if explicit != derived:
                    raise ValueError(
                        "GLM-5.3 explicit attention schedules disagree with layer_types: "
                        f"explicit={explicit}, derived={derived}"
                    )
            return explicit

    if layer_types is None:
        raise ValueError("GLM-5.3 config must define layer_types or linear_attn_config schedules")
    kda = [index for index, kind in enumerate(layer_types) if kind == "linear_attention"]
    dsa = [index for index, kind in enumerate(layer_types) if kind != "linear_attention"]
    return kda, dsa


def validate_glm5_next_config(config) -> None:
    """Fail early when a similarly named GLM config is not GLM-5.3-Flash."""
    if getattr(config, "model_type", None) != "glm5_next":
        raise ValueError(f"Expected model_type='glm5_next', got {getattr(config, 'model_type', None)!r}")
    text = get_text_config(config)
    required = (
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "q_lora_rank",
        "kv_lora_rank",
        "qk_nope_head_dim",
        "qk_rope_head_dim",
        "v_head_dim",
        "index_n_heads",
        "index_head_dim",
        "index_topk",
        "index_kpool",
        "hc_mult",
        "hc_eps",
        "hc_sinkhorn_iters",
    )
    missing = [name for name in required if not hasattr(text, name)]
    if missing:
        raise ValueError(f"Incomplete GLM-5.3 text config; missing {missing}")
    if int(text.qk_rope_head_dim) != 0:
        raise ValueError("GLM-5.3-Flash DSA is NoPE and requires qk_rope_head_dim=0")
    expected_qk_head_dim = int(text.qk_nope_head_dim) + int(text.qk_rope_head_dim)
    if int(getattr(text, "qk_head_dim", expected_qk_head_dim)) != expected_qk_head_dim:
        raise ValueError(
            "GLM-5.3-Flash qk_head_dim disagrees with its NoPE dimensions: "
            f"qk_head_dim={text.qk_head_dim}, expected={expected_qk_head_dim}"
        )
    if int(text.index_kpool) != 4:
        raise ValueError(f"Released GLM-5.3-Flash requires index_kpool=4, got {text.index_kpool}")
    if not getattr(text, "index_kpool_compress", False):
        raise ValueError("GLM-5.3-Flash requires pooled DSA index compression")
    if not getattr(text, "index_kpool_always_select_tail", False):
        raise ValueError("GLM-5.3-Flash requires the incomplete KPool tail to remain causal")
    if int(text.hc_mult) != 4 or float(text.hc_eps) != 1e-6:
        raise ValueError(f"Unsupported GLM-5.3 mHC contract: hc_mult={text.hc_mult}, hc_eps={text.hc_eps}")

    kda, dsa = attention_schedules(text)
    expected = list(range(int(text.num_hidden_layers)))
    if sorted(kda + dsa) != expected or set(kda) & set(dsa):
        raise ValueError(f"KDA/DSA schedules do not partition all decoder layers: kda={kda}, dsa={dsa}")

    mlp_types = getattr(text, "mlp_layer_types", None)
    if mlp_types is None or len(mlp_types) != int(text.num_hidden_layers):
        raise ValueError("GLM-5.3 config must define one mlp_layer_types entry per decoder layer")
    invalid_mlp = sorted(set(mlp_types) - {"dense", "sparse"})
    if invalid_mlp:
        raise ValueError(f"Unsupported GLM-5.3 MLP layer types: {invalid_mlp}")
    dense_prefix = int(getattr(text, "first_k_dense_replace", 0))
    expected_mlp = ["dense"] * dense_prefix + ["sparse"] * (int(text.num_hidden_layers) - dense_prefix)
    if list(mlp_types) != expected_mlp:
        raise ValueError(
            "GLM-5.3 mlp_layer_types disagrees with first_k_dense_replace: "
            f"declared={list(mlp_types)}, expected={expected_mlp}"
        )

def validate_glm5_next_checkpoint(path: str | Path, config=None) -> dict[str, int | bool]:
    """Validate the released nested state layout without materializing any tensor."""
    config = load_glm5_next_config(path) if config is None else config
    validate_glm5_next_config(config)
    text = get_text_config(config)
    names = checkpoint_tensor_names(path)
    prefix = "model.language_model.layers."
    layers = {
        int(parts[0])
        for name in names
        if name.startswith(prefix) and (parts := name[len(prefix) :].split(".", 1))[0].isdigit()
    }
    backbone = set(range(int(text.num_hidden_layers)))
    missing = sorted(backbone - layers)
    if missing:
        raise ValueError(f"GLM-5.3 checkpoint is missing backbone layers: {missing}")
    unexpected = sorted(layer for layer in layers if layer > int(text.num_hidden_layers))
    if unexpected:
        raise ValueError(f"GLM-5.3 checkpoint contains unexpected decoder layers: {unexpected}")

    required_globals = {
        "model.language_model.embed_tokens.weight",
        "model.language_model.norm.weight",
        "lm_head.weight",
    }
    missing_globals = sorted(required_globals - set(names))
    if missing_globals:
        raise ValueError(f"GLM-5.3 checkpoint is missing global tensors: {missing_globals}")

    mtp_layer = int(text.num_hidden_layers)
    has_mtp = mtp_layer in layers
    configured_mtp = int(getattr(text, "num_nextn_predict_layers", 0))
    if has_mtp and configured_mtp != 1:
        raise ValueError(f"Checkpoint contains MTP layer {mtp_layer} but config declares {configured_mtp}")
    if configured_mtp and not has_mtp:
        raise ValueError(
            f"GLM-5.3 config declares MTP={configured_mtp} but checkpoint has no layer {mtp_layer}"
        )

    vision_prefix = "model.visual.blocks."
    vision_blocks = {
        int(parts[0])
        for name in names
        if name.startswith(vision_prefix)
        and (parts := name[len(vision_prefix) :].split(".", 1))[0].isdigit()
    }
    expected_vision = set(range(int(config.vision_config.depth)))
    if vision_blocks != expected_vision:
        raise ValueError(
            "GLM-5.3 vision block layout mismatch: "
            f"found={sorted(vision_blocks)}, expected={sorted(expected_vision)}"
        )

    return {
        "tensor_count": len(names),
        "backbone_layers": len(backbone),
        "has_mtp": has_mtp,
        "vision_blocks": len(vision_blocks),
    }


__all__ = [
    "attention_schedules",
    "checkpoint_tensor_names",
    "get_text_config",
    "load_glm5_next_config",
    "validate_glm5_next_checkpoint",
    "validate_glm5_next_config",
]
