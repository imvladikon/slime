import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

NUM_GPUS = 1


def _tiny_checkpoint() -> Path:
    path = os.environ.get("GLM53_TINY_CHECKPOINT")
    if not path:
        pytest.skip("Set GLM53_TINY_CHECKPOINT to run the real tiny-model training gate")
    checkpoint = Path(path)
    if not (checkpoint / "model.safetensors").is_file():
        pytest.skip(f"Tiny GLM-5.3 safetensors are unavailable at {checkpoint}")
    return checkpoint


def _tiny_config(MLATransformerConfig):
    return MLATransformerConfig(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        expert_model_parallel_size=1,
        num_layers=5,
        hidden_size=256,
        num_attention_heads=4,
        num_query_groups=4,
        ffn_hidden_size=256,
        kv_channels=64,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        layernorm_epsilon=1e-5,
        add_bias_linear=False,
        gated_linear_unit=True,
        activation_func=F.silu,
        activation_func_clamp_value=10.0,
        normalization="RMSNorm",
        bf16=True,
        params_dtype=torch.bfloat16,
        transformer_impl="local",
        num_moe_experts=8,
        moe_layer_freq=[0, 0, 0, 1, 1],
        moe_ffn_hidden_size=128,
        moe_shared_expert_intermediate_size=128,
        moe_router_topk=4,
        moe_router_pre_softmax=True,
        moe_router_score_function="sigmoid",
        moe_router_dtype="fp32",
        moe_router_enable_expert_bias=True,
        moe_router_bias_update_rate=0.0,
        moe_router_load_balancing_type="none",
        moe_router_num_groups=1,
        moe_router_group_topk=1,
        moe_router_topk_scaling_factor=2.5,
        moe_token_dispatcher_type="allgather",
        q_lora_rank=128,
        kv_lora_rank=64,
        qk_head_dim=64,
        qk_pos_emb_head_dim=0,
        v_head_dim=64,
        rope_type="rope",
        enable_mhc_connections=True,
        mhc_num_residual_streams=4,
        mhc_sinkhorn_iterations=20,
        mhc_rms_epsilon_inside_sqrt=True,
        use_fused_mhc=False,
        mtp_num_layers=0,
    )


@pytest.mark.integration
@pytest.mark.skipif(not torch.cuda.is_available(), reason="GLM-5.3 tiny-model training gate requires CUDA")
@pytest.mark.skipif(importlib.util.find_spec("fla") is None, reason="flash-linear-attention is not installed")
def test_real_tiny_mcore_model_loads_and_trains_one_step(tmp_path: Path):
    if importlib.util.find_spec("megatron") is None:
        pytest.skip("The pinned Megatron-LM fork is not importable")
    if dist.is_initialized():
        pytest.skip("This world-one integration gate owns its process group")

    from megatron.core import parallel_state
    from megatron.core.models.gpt import GPTModel
    from megatron.core.packed_seq_params import PackedSeqParams
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer.transformer_config import MLATransformerConfig

    from slime.backends.megatron_utils.hf_to_megatron import load_hf_weights
    from slime.backends.megatron_utils.model_provider import freeze_model_params
    from slime_plugins.models.glm5_next.glm5_next import get_glm5_next_spec

    checkpoint = _tiny_checkpoint()
    args = SimpleNamespace(
        hf_checkpoint=str(checkpoint),
        transformer_impl="local",
        sequence_parallel=False,
        num_experts=8,
        padded_vocab_size=154880,
        params_dtype=torch.bfloat16,
        hidden_size=256,
        num_attention_heads=4,
        num_query_groups=4,
        kv_channels=64,
        freeze_indexer=True,
        freeze_params_name_list=None,
        only_train_params_name_list=None,
    )
    init_file = tmp_path / "process-group"
    dist.init_process_group("nccl", init_method=f"file://{init_file}", rank=0, world_size=1)
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        expert_model_parallel_size=1,
    )
    try:
        model_parallel_cuda_manual_seed(123)
        config = _tiny_config(MLATransformerConfig)
        with torch.device("cuda"):
            model = GPTModel(
                config=config,
                transformer_layer_spec=get_glm5_next_spec(args, config),
                vocab_size=154880,
                max_sequence_length=1048576,
                pre_process=True,
                post_process=True,
                fp16_lm_cross_entropy=False,
                parallel_output=False,
                share_embeddings_and_output_weights=False,
                position_embedding_type="none",
                rotary_percent=1.0,
                rotary_base=10000,
            )
        assert sum(parameter.numel() for parameter in model.parameters()) == 83_486_558
        load_hf_weights(args, [model], checkpoint)
        freeze_model_params(model, args)
        assert len(model._slime_frozen_indexer_param_names) == 7

        input_ids = torch.arange(1, 9, device="cuda").unsqueeze(0)
        position_ids = torch.arange(8, device="cuda").unsqueeze(0)
        labels = torch.arange(2, 10, device="cuda").unsqueeze(0)
        cu_seqlens = torch.tensor([0, 8], device="cuda", dtype=torch.int32)
        packed = PackedSeqParams(
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            max_seqlen_q=8,
            max_seqlen_kv=8,
            qkv_format="thd",
        )
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=1e-5)
        optimizer.zero_grad(set_to_none=True)
        loss = model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=None,
            labels=labels,
            packed_seq_params=packed,
        ).float().mean()
        loss.backward()

        assert torch.isfinite(loss)
        assert all(parameter.grad is not None for parameter in trainable)
        assert all(torch.isfinite(parameter.grad).all() for parameter in trainable)
        optimizer.step()
    finally:
        parallel_state.destroy_model_parallel()
        dist.destroy_process_group()
