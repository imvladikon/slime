"""Pooled DeepSeek Sparse Attention used by GLM-5.3-Flash."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from megatron.core import parallel_state
from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.moe.moe_utils import RouterGatingLinearFunction

from slime_plugins.models.glm5.glm5 import DSAMLASelfAttention
from slime_plugins.models.glm5.ops.sparse_mla import SparseMLA
from slime_plugins.models.glm5_next.ops.kpool_indexer import build_pooled_keys, kpool_select_topk, pool_boundaries

_SPARSE_MLA_TAIL_DIM = 64
_PRODUCTION_SPARSE_MLA_QK_DIM = 576
_MAX_REFERENCE_ATTENTION_ELEMENTS = 64 * 1024 * 1024


def _reference_sparse_mla(
    query: torch.Tensor,
    key: torch.Tensor,
    topk_indices: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """Differentiable exact sparse attention for the public compact checkpoint."""
    routes = topk_indices.squeeze(1).to(torch.long)
    value_dim = key.shape[-1] - _SPARSE_MLA_TAIL_DIM
    num_elements = routes.numel() * query.shape[1] * key.shape[-1]
    if num_elements > _MAX_REFERENCE_ATTENTION_ELEMENTS:
        raise RuntimeError(
            "GLM-5.3 Torch sparse-attention fallback would exceed its safe allocation; "
            "the full checkpoint must use the production SparseMLA kernel"
        )
    valid = routes >= 0
    selected = key[routes.clamp_min(0)]
    scores = torch.einsum("thd,tkd->thk", query.float(), selected.float()) * softmax_scale
    scores = scores.masked_fill(~valid.unsqueeze(1), float("-inf"))
    probabilities = torch.softmax(scores, dim=-1).to(selected.dtype)
    probabilities = torch.where(valid.unsqueeze(1), probabilities, 0)
    return torch.einsum("thk,tkd->thd", probabilities, selected[..., :value_dim])


def _sparse_mla(
    query: torch.Tensor,
    key: torch.Tensor,
    topk_indices: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    if query.shape[-1] == _PRODUCTION_SPARSE_MLA_QK_DIM:
        output, _ = SparseMLA.apply(query, key, topk_indices, softmax_scale)
        return output
    return _reference_sparse_mla(query, key, topk_indices, softmax_scale)


class Glm5NextDSAAttention(DSAMLASelfAttention):
    """NoPE MLA plus causal KPool index selection."""

    def __init__(
        self,
        config,
        submodules,
        layer_number: int,
        attn_mask_type=AttnMaskType.padding,
        is_mtp_layer: bool = False,
        cp_comm_type: str | None = None,
        model_comm_pgs=None,
        pg_collection=None,
        name: str | None = None,
    ) -> None:
        if config.qk_pos_emb_head_dim != 0:
            raise ValueError("GLM-5.3 DSA is NoPE and requires qk_pos_emb_head_dim=0")
        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            is_mtp_layer=is_mtp_layer,
            cp_comm_type=cp_comm_type,
            model_comm_pgs=model_comm_pgs,
            pg_collection=pg_collection,
            name=name,
        )
        self.softmax_scale = self.q_head_dim**-0.5
        self.index_topk = int(config.index_topk)
        self.index_kpool = int(config.index_kpool)
        device = torch.device("cpu") if config.use_cpu_initialization else torch.device("cuda", torch.cuda.current_device())
        self.index_kpool_compress_gate = torch.nn.Parameter(
            torch.zeros(
                config.index_head_dim,
                config.hidden_size,
                device=device,
                dtype=config.params_dtype,
            )
        )
        self.index_kpool_compress_ape = torch.nn.Parameter(
            torch.zeros(
                self.index_kpool,
                config.index_head_dim,
                device=device,
                dtype=config.params_dtype,
            )
        )
        # The released checkpoint treats the complete DSA indexer as frozen.
        self.index_kpool_compress_gate.requires_grad_(False)
        self.index_kpool_compress_ape.requires_grad_(False)

    def get_absorb_query_key_value_tensors(self, hidden_states, packed_seq_params):
        if hidden_states.ndim != 3:
            raise ValueError(f"Expected [sequence, batch, hidden], got {tuple(hidden_states.shape)}")
        if packed_seq_params is None:
            raise ValueError("GLM-5.3 DSA requires packed sequence metadata")
        if parallel_state.get_context_parallel_world_size() > 1:
            raise NotImplementedError(
                "GLM-5.3 KPool DSA currently requires context_parallel_size=1; use TP/PP/EP for production scale"
            )

        q_compressed, _ = self.linear_q_down_proj(hidden_states)
        q_compressed = q_compressed.squeeze(1)
        q_index_input = self._get_indexer_q_input(q_compressed).detach()
        q_projection_input = (
            q_compressed
            if getattr(self.linear_q_up_proj, "layer_norm_weight", None) is not None
            else self.q_layernorm(q_compressed)
        )
        query_heads, _ = self.linear_q_up_proj(q_projection_input)
        query_heads = query_heads.view(
            *query_heads.shape[:-1], self.num_attention_heads_per_partition, self.q_head_dim
        )

        kv_compressed, _ = self.linear_kv_down_proj(hidden_states)
        if self.config.sequence_parallel:
            kv_compressed = gather_from_sequence_parallel_region(kv_compressed)
        fused_kv_norm_weight = getattr(self.linear_kv_up_proj, "layer_norm_weight", None)
        if fused_kv_norm_weight is None:
            kv_compressed = self.kv_layernorm(kv_compressed)
        else:
            kv_compressed = torch.nn.functional.rms_norm(
                kv_compressed.float(),
                normalized_shape=(kv_compressed.shape[-1],),
                weight=fused_kv_norm_weight.float(),
                eps=self.config.layernorm_epsilon,
            ).to(kv_compressed.dtype)

        key_weight, value_weight = self.linear_kv_up_proj.weight.unflatten(
            0,
            (-1, self.config.qk_head_dim + self.config.v_head_dim),
        ).split([self.config.qk_head_dim, self.config.v_head_dim], dim=1)
        query = torch.einsum("thd,hdm->thm", query_heads, key_weight).contiguous()
        key = kv_compressed.squeeze(1).contiguous()

        index_q, _ = self.wq_b(q_index_input)
        index_q = index_q.view(-1, self.config.index_num_attention_heads, self.config.index_head_dim)
        if self.config.sequence_parallel:
            index_q = gather_from_sequence_parallel_region(index_q)

        detached_hidden = hidden_states.detach()
        index_k, _ = self.wk(detached_hidden)
        index_k = self.k_norm(index_k.squeeze(1).float()).to(index_k.dtype)
        if self.config.sequence_parallel:
            index_k = gather_from_sequence_parallel_region(index_k)

        gate_score = F.linear(detached_hidden.squeeze(1), self.index_kpool_compress_gate)
        if self.config.sequence_parallel:
            gate_score = gather_from_sequence_parallel_region(gate_score)

        head_weights = RouterGatingLinearFunction.apply(
            detached_hidden, self.weights_proj.weight, None, torch.float32
        ).squeeze(1)
        head_weights = head_weights * (
            (self.config.index_num_attention_heads**-0.5) * (self.config.index_head_dim**-0.5)
        )
        if self.config.sequence_parallel:
            head_weights = gather_from_sequence_parallel_region(head_weights)
        return query, key, value_weight, index_q, index_k, head_weights, gate_score

    def _select(self, index_q, index_k, head_weights, gate_score, packed_seq_params):
        cu_seqlens = packed_seq_params.cu_seqlens_kv
        pool_cu_seqlens = pool_boundaries(cu_seqlens, self.index_kpool)
        pooled_k = build_pooled_keys(
            index_k,
            gate_score,
            self.index_kpool_compress_ape,
            cu_seqlens,
            self.index_kpool,
        )
        return kpool_select_topk(
            index_q,
            pooled_k,
            head_weights,
            cu_seqlens,
            pool_cu_seqlens,
            self.index_topk,
            self.index_kpool,
        )

    def forward(
        self,
        hidden_states,
        attention_mask,
        key_value_states=None,
        inference_context=None,
        rotary_pos_emb=None,
        rotary_pos_cos=None,
        rotary_pos_sin=None,
        rotary_pos_cos_sin=None,
        attention_bias=None,
        packed_seq_params=None,
        position_ids=None,
        sequence_len_offset=None,
        *,
        inference_params=None,
        router_token_masks=None,
        loss_mask=None,
    ):
        del (
            attention_mask,
            key_value_states,
            inference_context,
            rotary_pos_cos,
            rotary_pos_sin,
            rotary_pos_cos_sin,
            position_ids,
            sequence_len_offset,
            inference_params,
            router_token_masks,
            loss_mask,
        )
        if rotary_pos_emb is not None or attention_bias is not None:
            raise ValueError("GLM-5.3 DSA must not receive RoPE or attention bias")
        query, key, value_weight, index_q, index_k, head_weights, gate_score = (
            self.get_absorb_query_key_value_tensors(hidden_states, packed_seq_params)
        )
        topk_indices = self._select(index_q, index_k, head_weights, gate_score, packed_seq_params)
        query = F.pad(query, (0, _SPARSE_MLA_TAIL_DIM)).contiguous()
        key = F.pad(key, (0, _SPARSE_MLA_TAIL_DIM)).contiguous()
        output = _sparse_mla(query, key, topk_indices, self.softmax_scale)
        output = torch.einsum("thm,hdm->thd", output, value_weight).reshape(output.shape[0], 1, -1)
        return self.linear_proj(output)


__all__ = ["Glm5NextDSAAttention"]
