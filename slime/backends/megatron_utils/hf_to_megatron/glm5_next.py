"""Hugging Face -> Megatron tensor mapping for GLM-5.3-Flash."""

from __future__ import annotations

import re

import torch

from slime_plugins.models.glm5_next.config import get_text_config

from .common import SafetensorReader, merge_gate_up, strip_mcore_wrappers
from .deepseek import deepseek_hf_tensor

_TEXT_PREFIX = "model.language_model."

_KDA_MAPPING = {
    f"self_attention.kda.{name}": f"self_attn.{name}"
    for name in (
        "q_proj.weight",
        "k_proj.weight",
        "v_proj.weight",
        "b_proj.weight",
        "f_a_proj.weight",
        "f_b_proj.weight",
        "g_a_proj.weight",
        "g_b_proj.weight",
        "A_log",
        "dt_bias",
        "o_norm.weight",
        "o_proj.weight",
    )
}

_INDEXER_MAPPING = {
    "self_attention.wq_b.weight": "self_attn.indexer.wq_b.weight",
    "self_attention.wk.weight": "self_attn.indexer.wk.weight",
    "self_attention.weights_proj.weight": "self_attn.indexer.weights_proj.weight",
    "self_attention.k_norm.weight": "self_attn.indexer.k_norm.weight",
    "self_attention.k_norm.bias": "self_attn.indexer.k_norm.bias",
    "self_attention.index_kpool_compress_gate": "self_attn.indexer.index_kpool_compress_gate",
    "self_attention.index_kpool_compress_ape": "self_attn.indexer.index_kpool_compress_ape",
}

_DSA_LOCAL_MAPPING = {
    "self_attention.q_layernorm.weight": "self_attn.q_a_layernorm.weight",
    "self_attention.kv_layernorm.weight": "self_attn.kv_a_layernorm.weight",
}

_HC_MAPPING = {
    "self_attention_hyper_connection.mapping_proj.weight": "hc_attn_fn",
    "self_attention_hyper_connection.bias": "hc_attn_base",
    "mlp_hyper_connection.mapping_proj.weight": "hc_ffn_fn",
    "mlp_hyper_connection.bias": "hc_ffn_base",
}

_HC_ALPHA_INDEX = {"alpha_pre": 0, "alpha_post": 1, "alpha_res": 2}
_LAYER = re.compile(r"decoder\.layers\.(\d+)\.(.+)")
_HC_ALPHA = re.compile(r"(self_attention_hyper_connection|mlp_hyper_connection)\.(alpha_pre|alpha_post|alpha_res)")
_LOCAL_EXPERT = re.compile(r"mlp\.experts\.local_experts\.(\d+)\.linear_fc([12])\.weight")


class _NestedLanguageReader:
    """Present ``model.language_model`` tensors to the existing DeepSeek mapper."""

    def __init__(self, reader: SafetensorReader):
        self.reader = reader

    @staticmethod
    def _nested(name: str) -> str:
        return _TEXT_PREFIX + name.removeprefix("model.") if name.startswith("model.") else name

    def __contains__(self, name: str) -> bool:
        return self._nested(name) in self.reader

    def get_tensor(self, name: str) -> torch.Tensor:
        return self.reader.get_tensor(self._nested(name))


def glm5_next_hf_tensor(name: str, reader: SafetensorReader, config) -> torch.Tensor:
    """Return the exact released HF tensor for one local Megatron parameter."""
    normalized = strip_mcore_wrappers(name)
    match = _LAYER.fullmatch(normalized)
    if match:
        layer, rest = match.groups()
        prefix = f"{_TEXT_PREFIX}layers.{layer}."
        if rest in _KDA_MAPPING:
            return reader.get_tensor(prefix + _KDA_MAPPING[rest])
        if rest == "self_attention.kda.conv1d.weight":
            return torch.cat(
                [reader.get_tensor(prefix + f"self_attn.{kind}_conv1d.weight") for kind in ("q", "k", "v")],
                dim=0,
            ).contiguous()
        if rest in _INDEXER_MAPPING:
            # GLM-5.3's indexer tensor layout is already native.  In particular,
            # do not apply the GLM-5.2 RoPE half-swap even when the compatibility
            # config contains indexer_rope_interleave=true.
            return reader.get_tensor(prefix + _INDEXER_MAPPING[rest])
        if rest in _DSA_LOCAL_MAPPING:
            return reader.get_tensor(prefix + _DSA_LOCAL_MAPPING[rest])
        if rest in _HC_MAPPING:
            return reader.get_tensor(prefix + _HC_MAPPING[rest])
        alpha = _HC_ALPHA.fullmatch(rest)
        if alpha:
            site, alpha_name = alpha.groups()
            scale = "hc_attn_scale" if site == "self_attention_hyper_connection" else "hc_ffn_scale"
            return reader.get_tensor(prefix + scale).reshape(-1)[_HC_ALPHA_INDEX[alpha_name]].reshape(1).clone()
        local_expert = _LOCAL_EXPERT.fullmatch(rest)
        if local_expert:
            from megatron.core import mpu

            local_index, projection = (int(local_expert.group(1)), local_expert.group(2))
            num_experts = int(get_text_config(config).n_routed_experts)
            experts_per_rank = num_experts // mpu.get_expert_model_parallel_world_size()
            expert = local_index + mpu.get_expert_model_parallel_rank() * experts_per_rank
            expert_prefix = prefix + f"mlp.experts.{expert}."
            if projection == "1":
                return merge_gate_up(
                    reader.get_tensor(expert_prefix + "gate_proj.weight"),
                    reader.get_tensor(expert_prefix + "up_proj.weight"),
                )
            return reader.get_tensor(expert_prefix + "down_proj.weight")

    nested_reader = _NestedLanguageReader(reader)
    return deepseek_hf_tensor(name, nested_reader, config)


__all__ = ["glm5_next_hf_tensor"]
