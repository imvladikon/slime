"""Megatron layer-spec integration for GLM-5.3-Flash."""

from __future__ import annotations

import copy

from slime_plugins.models.glm5_next.config import (
    attention_schedules,
    get_text_config,
    load_glm5_next_config,
    validate_glm5_next_checkpoint,
)

_MHC_EPS = 1e-6


def _make_te_norms_mhc_compatible(submodules, backend, identity_op) -> None:
    """Use TE tensor-only norms around mHC-owned residual connections.

    HyperConnectionTransformerLayer owns the n-stream residual captured before
    normalization. TE's ``has_residual=True`` norm returns a single-stream
    residual alongside the normalized tensor, which cannot replace that value.
    Keep TE norm kernels but request their tensor-only ABI. Dense TE MLPs fuse
    their norm into FC1 and therefore retain IdentityOp.
    """
    submodules.input_layernorm = backend.layer_norm()
    if submodules.pre_mlp_layernorm is not identity_op:
        submodules.pre_mlp_layernorm = backend.layer_norm()


def _apply_config(config, text_config) -> list[int]:
    if float(text_config.hc_eps) != _MHC_EPS:
        raise ValueError(f"Megatron mHC uses eps={_MHC_EPS}, checkpoint uses {text_config.hc_eps}")
    config.enable_mhc_connections = True
    config.mhc_num_residual_streams = int(text_config.hc_mult)
    config.mhc_sinkhorn_iterations = int(text_config.hc_sinkhorn_iters)
    config.mhc_rms_epsilon_inside_sqrt = True
    config.mhc_mapping_proj_fp32 = False
    config.use_fused_mhc = False
    if getattr(config, "recompute_granularity", None) == "full":
        raise ValueError("GLM-5.3 mHC requires selective recompute; full-layer recompute is unsupported")
    if getattr(config, "recompute_granularity", None) == "selective":
        modules = list(getattr(config, "recompute_modules", None) or [])
        if "mhc" not in modules:
            raise ValueError("GLM-5.3 selective recompute must include --recompute-modules mhc")
    if (getattr(config, "mtp_num_layers", None) or 0) != 0:
        raise ValueError("GLM-5.3 training intentionally disables the released inference-only MTP layer")
    config.index_num_attention_heads = int(text_config.index_n_heads)
    config.index_head_dim = int(text_config.index_head_dim)
    config.index_topk = int(text_config.index_topk)
    config.index_kpool = int(text_config.index_kpool)
    _kda, dsa = attention_schedules(text_config)
    config.glm5_next_full_attn_layers = dsa

    expected = {
        "num_layers": int(text_config.num_hidden_layers),
        "hidden_size": int(text_config.hidden_size),
        "num_attention_heads": int(text_config.num_attention_heads),
        "layernorm_epsilon": float(text_config.rms_norm_eps),
        "hidden_dropout": 0.0,
        "attention_dropout": float(getattr(text_config, "attention_dropout", 0.0)),
        "ffn_hidden_size": int(text_config.intermediate_size),
        "num_moe_experts": int(text_config.n_routed_experts),
        "moe_ffn_hidden_size": int(text_config.moe_intermediate_size),
        "moe_router_topk": int(text_config.num_experts_per_tok),
        "q_lora_rank": int(text_config.q_lora_rank),
        "kv_lora_rank": int(text_config.kv_lora_rank),
        "qk_head_dim": int(text_config.qk_nope_head_dim),
        "qk_pos_emb_head_dim": int(text_config.qk_rope_head_dim),
        "v_head_dim": int(text_config.v_head_dim),
        "moe_router_load_balancing_type": "none",
        "moe_router_pre_softmax": bool(text_config.norm_topk_prob),
        "moe_router_score_function": "sigmoid",
        "moe_router_dtype": "fp32",
        "moe_router_enable_expert_bias": True,
        "moe_router_bias_update_rate": 1e-3,
        "moe_router_topk_scaling_factor": float(text_config.routed_scaling_factor),
        "moe_router_num_groups": int(text_config.n_group),
        "moe_router_group_topk": int(text_config.topk_group),
        "moe_aux_loss_coeff": 0.0,
        "activation_func_clamp_value": float(text_config.swiglu_limit),
    }
    mismatches = {
        name: (getattr(config, name, None), value)
        for name, value in expected.items()
        if getattr(config, name, None) != value
    }
    expected_moe = [1 if kind == "sparse" else 0 for kind in text_config.mlp_layer_types]
    if list(getattr(config, "moe_layer_freq", [])) != expected_moe:
        mismatches["moe_layer_freq"] = (getattr(config, "moe_layer_freq", None), expected_moe)
    if mismatches:
        raise ValueError(f"Megatron arguments do not match GLM-5.3 checkpoint config: {mismatches}")
    return dsa


def get_glm5_next_spec(args, config, vp_stage=None):
    """Build the checkpoint-declared hybrid KDA/DSA decoder block."""
    from megatron.core.models.backends import LocalSpecProvider
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
    from megatron.core.post_training.modelopt.layers import Linear
    from megatron.core.transformer.enums import AttnMaskType
    from megatron.core.transformer.hyper_connection import HyperConnectionModule
    from megatron.core.transformer.identity_op import IdentityOp
    from megatron.core.transformer.spec_utils import ModuleSpec
    from megatron.core.transformer.transformer_block import get_num_layers_to_build
    from megatron.core.transformer.transformer_layer import (
        HyperConnectionTransformerLayer,
        get_transformer_layer_offset,
    )

    from slime_plugins.models.glm5.glm5 import DSASelfAttentionSubmodules
    from slime_plugins.models.glm5_next.dsa import Glm5NextDSAAttention
    from slime_plugins.models.glm5_next.kda import Glm5NextKDAAttention

    hf_config = load_glm5_next_config(args.hf_checkpoint)
    validate_glm5_next_checkpoint(args.hf_checkpoint, hf_config)
    text_config = get_text_config(hf_config)
    dsa_layers = set(_apply_config(config, text_config))

    if getattr(config, "pipeline_model_parallel_layout", None) is not None:
        raise NotImplementedError("GLM-5.3 currently uses the standard first/last-layer PP split")
    if int(getattr(config, "pipeline_model_parallel_size", 1)) != 1:
        raise NotImplementedError(
            "The pinned Megatron mHC implementation requires pipeline_model_parallel_size=1; "
            "scale GLM-5.3 with TP/EP/CP until native mHC pipeline buffers are supported"
        )
    use_transformer_engine = getattr(args, "transformer_impl", "transformer_engine") == "transformer_engine"
    block_kwargs = {"use_transformer_engine": use_transformer_engine}
    if vp_stage is not None:
        block_kwargs["vp_stage"] = vp_stage
    layer_spec = get_gpt_decoder_block_spec(config, **block_kwargs)
    if use_transformer_engine:
        from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider

        backend = TESpecProvider()
        duplicated_linear = backend.linear()
        q_up = backend.column_parallel_layer_norm_linear()
        kv_up = backend.column_parallel_layer_norm_linear()
        q_norm = IdentityOp
        kv_norm = IdentityOp
    else:
        backend = LocalSpecProvider()
        duplicated_linear = Linear
        q_up = backend.column_parallel_linear()
        kv_up = backend.column_parallel_linear()
        q_norm = backend.layer_norm(rms_norm=True, for_qk=True)
        kv_norm = backend.layer_norm(rms_norm=True, for_qk=True)
    dsa_spec = ModuleSpec(
        module=Glm5NextDSAAttention,
        params={"attn_mask_type": AttnMaskType.causal},
        submodules=DSASelfAttentionSubmodules(
            linear_q_down_proj=duplicated_linear,
            linear_q_up_proj=q_up,
            linear_kv_down_proj=duplicated_linear,
            linear_kv_up_proj=kv_up,
            core_attention=backend.core_attention(),
            linear_proj=backend.row_parallel_linear(),
            q_layernorm=q_norm,
            kv_layernorm=kv_norm,
            linear_v_up_proj=IdentityOp,
            wq_b=duplicated_linear,
            wk=duplicated_linear,
            k_norm=backend.layer_norm(),
            weights_proj=duplicated_linear,
        ),
    )
    count = get_num_layers_to_build(config, vp_stage=vp_stage)
    offset = get_transformer_layer_offset(config, vp_stage=vp_stage)
    for local_layer in range(count):
        current = copy.deepcopy(layer_spec.layer_specs[local_layer])
        current.module = HyperConnectionTransformerLayer
        current.submodules.self_attention_hyper_connection = HyperConnectionModule
        current.submodules.mlp_hyper_connection = HyperConnectionModule
        if use_transformer_engine:
            _make_te_norms_mhc_compatible(current.submodules, backend, IdentityOp)
        if local_layer + offset in dsa_layers:
            current.submodules.self_attention = dsa_spec
        else:
            current.submodules.self_attention = ModuleSpec(module=Glm5NextKDAAttention, params={"args": args})
        layer_spec.layer_specs[local_layer] = current
    return layer_spec


__all__ = ["get_glm5_next_spec"]
