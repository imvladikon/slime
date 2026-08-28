"""Megatron -> Hugging Face conversion for GLM-5.3-Flash."""

from __future__ import annotations

import re

import torch

from .deepseekv3 import convert_deepseekv3_to_hf

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

_HC_ORDER = ("alpha_pre", "alpha_post", "alpha_res")
_HC_SITE_SCALE = {
    "self_attention_hyper_connection": "hc_attn_scale",
    "mlp_hyper_connection": "hc_ffn_scale",
}
_LAYER = re.compile(r"(?:module\.)*decoder\.layers\.(\d+)\.(.+)")
_HC_ALPHA = re.compile(r"(self_attention_hyper_connection|mlp_hyper_connection)\.(alpha_pre|alpha_post|alpha_res)")
_LOCAL_EXPERT = re.compile(r"mlp\.experts\.local_experts\.(\d+)\.linear_fc([12])\.weight")
_HC_BUFFERS: dict[tuple[int, str, str], dict[str, torch.Tensor]] = {}


def _nested(name: str) -> str:
    return _TEXT_PREFIX + name.removeprefix("model.") if name.startswith("model.") else name


def _hc_scale(args, layer: str, site: str, alpha: str, tensor: torch.Tensor):
    key = (id(args), layer, site)
    values = _HC_BUFFERS.setdefault(key, {})
    if alpha in values:
        raise ValueError(f"Duplicate GLM-5.3 mHC scale component: layer={layer}, site={site}, alpha={alpha}")
    values[alpha] = tensor
    if len(values) != len(_HC_ORDER):
        return []
    _HC_BUFFERS.pop(key)
    scale = torch.cat([values[name].reshape(1) for name in _HC_ORDER]).contiguous()
    return [(f"{_TEXT_PREFIX}layers.{layer}.{_HC_SITE_SCALE[site]}", scale)]


def convert_glm5_next_to_hf(args, name, param):
    """Convert a single Megatron tensor without GLM-5.2 indexer reordering."""
    stripped = name
    while stripped.startswith("module."):
        stripped = stripped.removeprefix("module.")

    if stripped == "embedding.word_embeddings.weight":
        return [(f"{_TEXT_PREFIX}embed_tokens.weight", param)]
    if stripped == "decoder.final_layernorm.weight":
        return [(f"{_TEXT_PREFIX}norm.weight", param)]
    if stripped == "output_layer.weight":
        return [("lm_head.weight", param)]

    match = _LAYER.fullmatch(name)
    if match:
        layer, rest = match.groups()
        prefix = f"{_TEXT_PREFIX}layers.{layer}."
        suffix = (
            _KDA_MAPPING.get(rest)
            or _INDEXER_MAPPING.get(rest)
            or _DSA_LOCAL_MAPPING.get(rest)
            or _HC_MAPPING.get(rest)
        )
        if suffix is not None:
            # MCore deliberately trains the dynamic mHC projection in FP32,
            # while both the released checkpoint and SGLang store hc_*_fn in
            # the model parameter dtype.  Static bases/scales remain FP32.
            if rest.endswith("hyper_connection.mapping_proj.weight"):
                param = param.to(getattr(args, "params_dtype", torch.bfloat16))
            return [(prefix + suffix, param)]
        if rest == "self_attention.kda.conv1d.weight":
            q_conv, k_conv, v_conv = param.chunk(3, dim=0)
            return [
                (prefix + "self_attn.q_conv1d.weight", q_conv),
                (prefix + "self_attn.k_conv1d.weight", k_conv),
                (prefix + "self_attn.v_conv1d.weight", v_conv),
            ]
        alpha = _HC_ALPHA.fullmatch(rest)
        if alpha:
            site, alpha_name = alpha.groups()
            return _hc_scale(args, layer, site, alpha_name, param)
        local_expert = _LOCAL_EXPERT.fullmatch(rest)
        if local_expert:
            from megatron.core import mpu

            local_index, projection = (int(local_expert.group(1)), local_expert.group(2))
            experts_per_rank = int(args.num_experts) // mpu.get_expert_model_parallel_world_size()
            expert = local_index + mpu.get_expert_model_parallel_rank() * experts_per_rank
            expert_prefix = prefix + f"mlp.experts.{expert}."
            if projection == "1":
                gate, up = param.chunk(2, dim=0)
                return [
                    (expert_prefix + "gate_proj.weight", gate),
                    (expert_prefix + "up_proj.weight", up),
                ]
            return [(expert_prefix + "down_proj.weight", param)]

    return [(_nested(hf_name), tensor) for hf_name, tensor in convert_deepseekv3_to_hf(args, name, param)]


__all__ = ["convert_glm5_next_to_hf"]
