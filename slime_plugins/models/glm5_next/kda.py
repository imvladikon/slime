"""Kimi Delta Attention layers used by GLM-5.3-Flash."""

from __future__ import annotations

import copy

import torch
from torch import nn
from megatron.core import parallel_state, tensor_parallel
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer.module import mark_keep_in_fp32

try:
    from fla.modules import FusedRMSNormGated, ShortConvolution
    from fla.ops.kda import chunk_kda
    from fla.ops.kda.gate import fused_kda_gate
except ImportError:
    FusedRMSNormGated = ShortConvolution = chunk_kda = fused_kda_gate = None

from slime_plugins.models.glm5_next.config import get_text_config
from slime_plugins.models.hf_attention import HuggingfaceAttention


def _linear_attention_fields(text_config) -> dict[str, int | float]:
    linear = getattr(text_config, "linear_attn_config", None)

    def field(name: str, fallback: str, default=None):
        value = linear.get(name) if isinstance(linear, dict) else getattr(linear, name, None)
        return getattr(text_config, fallback, default) if value is None else value

    lower_bound = field("gate_lower_bound", "linear_lower_bound")
    if lower_bound is None:
        raise ValueError("GLM-5.3 KDA requires its safe gate_lower_bound")
    return {
        "num_heads": int(field("num_heads", "linear_num_heads", 64)),
        "head_dim": int(field("head_dim", "linear_head_dim", 128)),
        "conv_kernel_size": int(field("short_conv_kernel_size", "linear_conv_kernel_dim", 4)),
        "gate_lower_bound": float(lower_bound),
    }


class Glm5NextKDA(nn.Module):
    """Reference-faithful trainable KDA block backed by FLA kernels."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        conv_kernel_size: int,
        gate_lower_bound: float,
        rms_norm_eps: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        config=None,
    ) -> None:
        super().__init__()
        if ShortConvolution is None or chunk_kda is None or FusedRMSNormGated is None:
            raise ImportError("GLM-5.3 KDA requires flash-linear-attention >= 0.4.2")
        tp_size = parallel_state.get_tensor_model_parallel_world_size() if config is not None else 1
        if num_heads % tp_size:
            raise ValueError(f"GLM-5.3 KDA heads={num_heads} must be divisible by TP={tp_size}")
        self.tp_size = tp_size
        self.num_heads = num_heads // tp_size
        self.head_dim = head_dim
        self.projection_size = self.num_heads * head_dim
        self.gate_lower_bound = gate_lower_bound

        factory_kwargs = {"device": device, "dtype": dtype}
        if config is None:
            self.q_proj = nn.Linear(hidden_size, self.projection_size, bias=False, **factory_kwargs)
            self.k_proj = nn.Linear(hidden_size, self.projection_size, bias=False, **factory_kwargs)
            self.v_proj = nn.Linear(hidden_size, self.projection_size, bias=False, **factory_kwargs)
            self.b_proj = nn.Linear(hidden_size, self.num_heads, bias=False, **factory_kwargs)
            self.f_b_proj = nn.Linear(head_dim, self.projection_size, bias=False, **factory_kwargs)
            self.g_b_proj = nn.Linear(head_dim, self.projection_size, bias=False, **factory_kwargs)
            self.o_proj = nn.Linear(self.projection_size, hidden_size, bias=False, **factory_kwargs)
        else:
            # HuggingfaceAttention gathers sequence-parallel inputs before KDA,
            # so these head-parallel projections must not perform a second SP
            # gather/reduce-scatter internally.
            tp_config = copy.copy(config)
            tp_config.sequence_parallel = False

            def column(input_size: int, output_size: int):
                return ColumnParallelLinear(
                    input_size,
                    output_size,
                    config=tp_config,
                    init_method=config.init_method,
                    bias=False,
                    gather_output=False,
                )

            self.q_proj = column(hidden_size, num_heads * head_dim)
            self.k_proj = column(hidden_size, num_heads * head_dim)
            self.v_proj = column(hidden_size, num_heads * head_dim)
            self.b_proj = column(hidden_size, num_heads)
            self.f_b_proj = column(head_dim, num_heads * head_dim)
            self.g_b_proj = column(head_dim, num_heads * head_dim)
            self.o_proj = RowParallelLinear(
                num_heads * head_dim,
                hidden_size,
                config=tp_config,
                init_method=config.output_layer_init_method,
                bias=False,
                input_is_parallel=True,
                skip_bias_add=False,
            )
        self.conv1d = ShortConvolution(
            hidden_size=3 * self.projection_size,
            kernel_size=conv_kernel_size,
            bias=False,
            activation="silu",
            **factory_kwargs,
        )
        self.f_a_proj = nn.Linear(hidden_size, head_dim, bias=False, **factory_kwargs)
        self.g_a_proj = nn.Linear(hidden_size, head_dim, bias=False, **factory_kwargs)
        self.A_log = mark_keep_in_fp32(
            nn.Parameter(torch.zeros(self.num_heads, device=device, dtype=torch.float32))
        )
        self.dt_bias = mark_keep_in_fp32(
            nn.Parameter(torch.zeros(self.projection_size, device=device, dtype=torch.float32))
        )
        self.o_norm = FusedRMSNormGated(
            head_dim,
            eps=rms_norm_eps,
            activation="sigmoid",
            **factory_kwargs,
        )
        if config is not None and tp_size > 1:
            tensor_parallel.set_tensor_model_parallel_attributes(self.conv1d.weight, True, 0, 1)
            tensor_parallel.set_tensor_model_parallel_attributes(self.A_log, True, 0, 1)
            tensor_parallel.set_tensor_model_parallel_attributes(self.dt_bias, True, 0, 1)
            # This affine is shared by head partitions and its gradients must be
            # reduced across TP ranks by Megatron's replicated-parameter pass.
            self.o_norm.weight.sequence_parallel = True

    @staticmethod
    def _linear(module: nn.Module, value: torch.Tensor) -> torch.Tensor:
        output = module(value)
        return output[0] if isinstance(output, tuple) else output

    def forward(self, hidden_states: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
        mixed_qkv = torch.cat(
            tuple(self._linear(module, hidden_states) for module in (self.q_proj, self.k_proj, self.v_proj)),
            dim=-1,
        )
        mixed_qkv, _ = self.conv1d(x=mixed_qkv, cu_seqlens=cu_seqlens)
        query, key, value = torch.split(mixed_qkv, [self.projection_size] * 3, dim=-1)
        query = query.unflatten(-1, (self.num_heads, self.head_dim))
        key = key.unflatten(-1, (self.num_heads, self.head_dim))
        value = value.unflatten(-1, (self.num_heads, self.head_dim))

        beta = torch.sigmoid(self._linear(self.b_proj, hidden_states).float())
        forget_low_rank = self.f_a_proj(hidden_states)
        gate_low_rank = self.g_a_proj(hidden_states)
        if self.tp_size > 1:
            # Forward is an identity; backward sums the contributions from all
            # head partitions into the replicated low-rank projections.
            forget_low_rank = tensor_parallel.copy_to_tensor_model_parallel_region(forget_low_rank)
            gate_low_rank = tensor_parallel.copy_to_tensor_model_parallel_region(gate_low_rank)
        forget = self._linear(self.f_b_proj, forget_low_rank)
        decay = fused_kda_gate(
            forget.unflatten(-1, (self.num_heads, self.head_dim)),
            self.A_log,
            self.dt_bias,
            lower_bound=self.gate_lower_bound,
        )
        output, _ = chunk_kda(
            query,
            key,
            value,
            g=decay,
            beta=beta,
            initial_state=None,
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=cu_seqlens,
            safe_gate=True,
            transpose_state_layout=True,
        )
        norm_gate = self._linear(self.g_b_proj, gate_low_rank)
        output_shape = output.shape
        output = self.o_norm(output.reshape(-1, self.head_dim), norm_gate.reshape(-1, self.head_dim))
        output = output.reshape(output_shape[0], output_shape[1], -1)
        return self._linear(self.o_proj, output)


class Glm5NextKDAAttention(HuggingfaceAttention):
    """Megatron attention adapter for a KDA decoder layer."""

    def __init__(
        self,
        args,
        config,
        layer_number: int,
        cp_comm_type: str = "p2p",
        model_comm_pgs=None,
        pg_collection=None,
        name: str | None = None,
    ):
        # ``model_comm_pgs`` and ``name`` are part of the current MCore
        # attention-construction ABI.  This adapter does not consume them, but
        # accepting them keeps the custom KDA layer buildable by ModuleSpec.
        super().__init__(args, config, layer_number, cp_comm_type, pg_collection)
        text = get_text_config(self.hf_config)
        fields = _linear_attention_fields(text)
        device = torch.device("cpu") if config.use_cpu_initialization else torch.device("cuda", torch.cuda.current_device())
        self.kda = Glm5NextKDA(
            hidden_size=text.hidden_size,
            num_heads=fields["num_heads"],
            head_dim=fields["head_dim"],
            conv_kernel_size=fields["conv_kernel_size"],
            gate_lower_bound=fields["gate_lower_bound"],
            rms_norm_eps=text.rms_norm_eps,
            device=device,
            dtype=config.params_dtype,
            config=config,
        )

    def hf_forward(self, hidden_states, packed_seq_params):
        return self.kda(hidden_states, cu_seqlens=packed_seq_params.cu_seqlens_q)


__all__ = ["Glm5NextKDA", "Glm5NextKDAAttention"]
