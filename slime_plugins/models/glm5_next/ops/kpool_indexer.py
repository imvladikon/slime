"""Causal KPool compression and token expansion for GLM-5.3 DSA."""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from slime_plugins.models.glm5.ops.tilelang_indexer_fwd import indexer_fwd_interface

SPARSE_MLA_BLOCK = 64
_SELECT_BLOCK = 256
_TILELANG_SHARED_MEMORY_BYTES = 114688
_MAX_REFERENCE_LOGIT_ELEMENTS = 16 * 1024 * 1024


def pool_boundaries(cu_seqlens: torch.Tensor, kpool: int) -> torch.Tensor:
    sequence_lengths = cu_seqlens[1:] - cu_seqlens[:-1]
    pool_counts = torch.div(sequence_lengths, kpool, rounding_mode="floor")
    boundaries = torch.zeros_like(cu_seqlens)
    boundaries[1:] = torch.cumsum(pool_counts, dim=0)
    return boundaries


@triton.jit
def _pooled_keys_kernel(
    index_k_ptr,
    gate_ptr,
    ape_ptr,
    cu_seqlens_ptr,
    pool_cu_seqlens_ptr,
    num_sequences,
    output_ptr,
    D: tl.constexpr,
    KPOOL: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pool_id = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_D)
    dimension_mask = offsets < D
    output_offsets = pool_id.to(tl.int64) * D + offsets
    num_pools = tl.load(pool_cu_seqlens_ptr + num_sequences)
    if pool_id >= num_pools:
        tl.store(output_ptr + output_offsets, 0.0, mask=dimension_mask)
        return

    low = 0
    high = num_sequences
    while low < high:
        middle = (low + high + 1) // 2
        if tl.load(pool_cu_seqlens_ptr + middle) <= pool_id:
            low = middle
        else:
            high = middle - 1
    sequence_start = tl.load(cu_seqlens_ptr + low)
    pool_start = tl.load(pool_cu_seqlens_ptr + low)
    token_start = (sequence_start + (pool_id - pool_start) * KPOOL).to(tl.int64)

    peak = tl.full((BLOCK_D,), float("-inf"), dtype=tl.float32)
    for slot in tl.static_range(KPOOL):
        gate = tl.load(gate_ptr + (token_start + slot) * D + offsets, mask=dimension_mask, other=0.0)
        ape = tl.load(ape_ptr + slot * D + offsets, mask=dimension_mask, other=0.0)
        peak = tl.maximum(peak, gate.to(tl.float32) + ape)
    denominator = tl.zeros((BLOCK_D,), dtype=tl.float32)
    for slot in tl.static_range(KPOOL):
        gate = tl.load(gate_ptr + (token_start + slot) * D + offsets, mask=dimension_mask, other=0.0)
        ape = tl.load(ape_ptr + slot * D + offsets, mask=dimension_mask, other=0.0)
        denominator += libdevice.exp(gate.to(tl.float32) + ape - peak)
    pooled = tl.zeros((BLOCK_D,), dtype=tl.float32)
    for slot in tl.static_range(KPOOL):
        gate = tl.load(gate_ptr + (token_start + slot) * D + offsets, mask=dimension_mask, other=0.0)
        ape = tl.load(ape_ptr + slot * D + offsets, mask=dimension_mask, other=0.0)
        key = tl.load(index_k_ptr + (token_start + slot) * D + offsets, mask=dimension_mask, other=0.0)
        weight = libdevice.div_rn(libdevice.exp(gate.to(tl.float32) + ape - peak), denominator)
        pooled += libdevice.mul_rn(weight, key.to(tl.float32))
    tl.store(output_ptr + output_offsets, pooled.to(output_ptr.dtype.element_ty), mask=dimension_mask)


def build_pooled_keys(
    index_k: torch.Tensor,
    gate_score: torch.Tensor,
    ape: torch.Tensor,
    cu_seqlens: torch.Tensor,
    kpool: int,
) -> torch.Tensor:
    total_tokens, head_dim = index_k.shape
    max_pools = total_tokens // kpool
    if max_pools == 0:
        return index_k.new_zeros((0, head_dim))
    pool_cu_seqlens = pool_boundaries(cu_seqlens, kpool)
    pooled = torch.empty((max_pools, head_dim), dtype=index_k.dtype, device=index_k.device)
    _pooled_keys_kernel[(max_pools,)](
        index_k.contiguous(),
        gate_score.contiguous(),
        ape.float().contiguous(),
        cu_seqlens.contiguous(),
        pool_cu_seqlens.contiguous(),
        cu_seqlens.shape[0] - 1,
        pooled,
        D=head_dim,
        KPOOL=kpool,
        BLOCK_D=triton.next_power_of_2(head_dim),
    )
    return pooled


def _can_use_tilelang_indexer(device: torch.device) -> bool:
    if device.type != "cuda":
        return False
    properties = torch.cuda.get_device_properties(device)
    available = getattr(properties, "shared_memory_per_block_optin", properties.shared_memory_per_block)
    return available >= _TILELANG_SHARED_MEMORY_BYTES


def _reference_indexer_logits(
    index_q: torch.Tensor,
    pooled_k: torch.Tensor,
    head_weights: torch.Tensor,
    starts: torch.Tensor,
    ends: torch.Tensor,
) -> torch.Tensor:
    """Exact short-context fallback for GPUs below the TileLang shared-memory floor."""
    num_elements = index_q.shape[0] * pooled_k.shape[0] * index_q.shape[1]
    if num_elements > _MAX_REFERENCE_LOGIT_ELEMENTS:
        raise RuntimeError(
            "GLM-5.3 Torch indexer fallback would materialize too many logits; "
            "use a Hopper/Blackwell GPU with the TileLang production kernel"
        )
    scores = torch.einsum("thd,pd->tph", index_q.float(), pooled_k.float()).relu_()
    logits = torch.einsum("tph,th->tp", scores, head_weights.float())
    columns = torch.arange(pooled_k.shape[0], device=logits.device).unsqueeze(0)
    valid = (columns >= starts.unsqueeze(1)) & (columns < ends.unsqueeze(1))
    return logits.masked_fill(~valid, float("-inf"))


@triton.jit
def _expand_topk_kernel(
    pools_ptr,
    scores_ptr,
    sequence_base_ptr,
    local_position_ptr,
    pool_base_ptr,
    output_ptr,
    topk,
    group_topk,
    output_width,
    KPOOL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    first_column = tl.program_id(1) * BLOCK
    columns = first_column + tl.arange(0, BLOCK)
    mask = columns < output_width
    sequence_base = tl.load(sequence_base_ptr + token)
    local_position = tl.load(local_position_ptr + token)
    if (local_position + 1) <= topk:
        values = tl.where((columns < topk) & (columns <= local_position), sequence_base + columns, -1)
    else:
        pool_base = tl.load(pool_base_ptr + token)
        groups = first_column // KPOOL + tl.arange(0, BLOCK // KPOOL)
        group_mask = groups < group_topk
        pool = tl.load(pools_ptr + token * group_topk + groups, mask=group_mask, other=0).to(tl.int32)
        score = tl.load(scores_ptr + token * group_topk + groups, mask=group_mask, other=float("-inf"))
        finite = (score == score) & (score != float("inf")) & (score != float("-inf"))
        slots = tl.arange(0, KPOOL)
        candidates = (sequence_base + (pool - pool_base) * KPOOL)[:, None] + slots[None, :]
        values = tl.reshape(tl.where((group_mask & finite)[:, None], candidates, -1), (BLOCK,))
        tail_start = sequence_base + ((local_position + 1) // KPOOL) * KPOOL
        tail_slot = columns - topk
        in_tail = (tail_slot >= 0) & (tail_slot < (local_position + 1) % KPOOL)
        values = tl.where(in_tail, tail_start + tail_slot, values)
    tl.store(output_ptr + token * output_width + columns, values.to(tl.int32), mask=mask)


def kpool_select_topk(
    index_q: torch.Tensor,
    pooled_k: torch.Tensor,
    head_weights: torch.Tensor,
    cu_seqlens: torch.Tensor,
    pool_cu_seqlens: torch.Tensor,
    index_topk: int,
    kpool: int,
) -> torch.Tensor:
    num_tokens = index_q.shape[0]
    token_ids = torch.arange(num_tokens, device=index_q.device)
    sequence_indices = torch.searchsorted(cu_seqlens, token_ids, right=True) - 1
    sequence_base = cu_seqlens[sequence_indices].to(torch.int32)
    pool_base = pool_cu_seqlens[sequence_indices].to(torch.int32)
    local_positions = (token_ids - sequence_base).to(torch.int32)
    eligible_pools = torch.div(local_positions + 1, kpool, rounding_mode="floor")

    if pooled_k.shape[0]:
        with torch.no_grad():
            pool_ends = (pool_base + eligible_pools).to(torch.int32)
            if _can_use_tilelang_indexer(index_q.device):
                logits = indexer_fwd_interface(
                    index_q,
                    pooled_k,
                    head_weights,
                    pool_base,
                    pool_ends,
                    clean_logits=True,
                )
            else:
                logits = _reference_indexer_logits(
                    index_q,
                    pooled_k,
                    head_weights,
                    pool_base,
                    pool_ends,
                )
    else:
        logits = torch.full((num_tokens, 1), float("-inf"), dtype=torch.float32, device=index_q.device)

    group_topk = min(index_topk // kpool, logits.shape[1])
    if group_topk < 1:
        raise ValueError(f"KPool top-k has no eligible groups: topk={index_topk}, kpool={kpool}")
    scores, pools = torch.topk(logits.float(), group_topk, dim=-1)
    output_width = (index_topk + kpool - 1 + SPARSE_MLA_BLOCK - 1) // SPARSE_MLA_BLOCK * SPARSE_MLA_BLOCK
    output = torch.empty((num_tokens, output_width), dtype=torch.int32, device=index_q.device)
    grid = (num_tokens, triton.cdiv(output_width, _SELECT_BLOCK))
    _expand_topk_kernel[grid](
        pools,
        scores,
        sequence_base,
        local_positions,
        pool_base,
        output,
        index_topk,
        group_topk,
        output_width,
        KPOOL=kpool,
        BLOCK=_SELECT_BLOCK,
    )
    return output.unsqueeze(1)


__all__ = [
    "build_pooled_keys",
    "kpool_select_topk",
    "pool_boundaries",
]
