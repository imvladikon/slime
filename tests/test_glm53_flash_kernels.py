import importlib.util

import pytest
import torch
import torch.nn.functional as F

from slime_plugins.models.glm5_next.dsa import _reference_sparse_mla
from slime_plugins.models.glm5_next.ops.kpool_indexer import (
    build_pooled_keys,
    kpool_select_topk,
    pool_boundaries,
)

NUM_GPUS = 1


@pytest.mark.integration
@pytest.mark.skipif(not torch.cuda.is_available(), reason="GLM-5.3 kernel smoke requires CUDA")
@pytest.mark.skipif(importlib.util.find_spec("fla") is None, reason="flash-linear-attention is not installed")
def test_compact_kda_forward_backward_is_finite():
    import triton

    native_next_power_of_2 = triton.next_power_of_2
    from slime.backends.megatron_utils.megatron_to_hf.processors.quantizer_fp8 import (
        quantize_params_fp8,
    )
    from sglang.srt.utils.common import next_power_of_2 as sglang_next_power_of_2
    from slime_plugins.models.glm5_next.kda import Glm5NextKDA

    # Importing the rollout weight converter imports SGLang.  It must not
    # replace Triton's JIT intrinsic, because FLA captures this object while
    # compiling the KDA kernels.
    assert quantize_params_fp8 is not None
    assert triton.next_power_of_2 is native_next_power_of_2
    assert triton.next_power_of_2 is not sglang_next_power_of_2

    torch.manual_seed(17)
    module = Glm5NextKDA(
        hidden_size=256,
        num_heads=4,
        head_dim=64,
        conv_kernel_size=4,
        gate_lower_bound=-5.0,
        rms_norm_eps=1e-5,
        device=torch.device("cuda"),
        dtype=torch.bfloat16,
    )
    hidden = torch.randn(1, 16, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    cu_seqlens = torch.tensor([0, 16], device="cuda", dtype=torch.int32)

    output = module(hidden, cu_seqlens)
    loss = output.float().square().mean()
    loss.backward()

    assert output.shape == hidden.shape
    assert output.dtype == torch.bfloat16
    assert torch.isfinite(output).all()
    assert torch.isfinite(hidden.grad).all()
    assert module.A_log.dtype == torch.float32
    assert module.dt_bias.dtype == torch.float32
    assert module.q_proj.weight.dtype == torch.bfloat16
    assert module.q_proj.weight.grad is not None
    assert module.A_log.grad is not None


@pytest.mark.integration
@pytest.mark.skipif(not torch.cuda.is_available(), reason="GLM-5.3 kernel smoke requires CUDA")
def test_compact_kpool_and_sparse_attention_are_causal_and_differentiable():
    torch.manual_seed(23)
    tokens, head_dim, heads = 17, 64, 4
    cu_seqlens = torch.tensor([0, 7, 17], device="cuda", dtype=torch.int32)
    index_k = torch.randn(tokens, head_dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    gate = torch.randn_like(index_k)
    ape = torch.randn(4, head_dim, device="cuda", dtype=torch.bfloat16)
    index_q = torch.randn(
        tokens,
        heads,
        head_dim,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    head_weights = torch.randn(tokens, heads, device="cuda", dtype=torch.float32)

    pool_cu_seqlens = pool_boundaries(cu_seqlens, 4)
    pooled = build_pooled_keys(index_k, gate, ape, cu_seqlens, 4)
    routes = kpool_select_topk(
        index_q.detach(),
        pooled.detach(),
        head_weights,
        cu_seqlens,
        pool_cu_seqlens,
        64,
        4,
    )

    for token in range(tokens):
        sequence_start = 0 if token < 7 else 7
        valid = routes[token, 0] >= 0
        assert (routes[token, 0, valid] >= sequence_start).all()
        assert (routes[token, 0, valid] <= token).all()

    output = _reference_sparse_mla(
        F.pad(index_q, (0, 64)),
        F.pad(index_k, (0, 64)),
        routes,
        head_dim**-0.5,
    )
    loss = output.float().square().mean()
    loss.backward()

    assert output.shape == (tokens, heads, head_dim)
    assert torch.isfinite(output).all()
    assert torch.isfinite(index_q.grad).all()
    assert torch.isfinite(index_k.grad).all()
