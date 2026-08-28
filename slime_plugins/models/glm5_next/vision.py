"""Frozen GLM-5.3 visual tower and multimodal Megatron model provider."""

from __future__ import annotations

from contextlib import contextmanager

import torch
from torch import nn
from transformers.activations import ACT2FN
from transformers.models.glm_ocr.modeling_glm_ocr import (
    GlmOcrRMSNorm,
    GlmOcrVisionAttention,
    GlmOcrVisionPatchEmbed,
    GlmOcrVisionRotaryEmbedding,
)

try:
    # Transformers 5.16 moved these model-independent helpers out of the
    # individual vision model modules.
    from transformers.vision_utils import get_vision_cu_seqlens, get_vision_position_ids
except ImportError:  # pragma: no cover - exercised by the production 5.12 image
    from transformers.models.glm_ocr.modeling_glm_ocr import get_vision_cu_seqlens, get_vision_position_ids

from slime.backends.megatron_utils.hf_to_megatron.common import SafetensorReader
from slime_plugins.models.glm5_next.config import load_glm5_next_config

_VISUAL_PREFIX = "model.visual."


class Glm5NextVisionMLP(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.attention_bias)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.attention_bias)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.attention_bias)
        self.act_fn = ACT2FN[config.hidden_act]
        self.swiglu_limit = config.swiglu_limit

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(hidden_states).clamp(max=self.swiglu_limit)
        up = self.up_proj(hidden_states).clamp(min=-self.swiglu_limit, max=self.swiglu_limit)
        return self.down_proj(self.act_fn(gate) * up)


class Glm5NextVisionBlock(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.norm1 = GlmOcrRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.norm2 = GlmOcrRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn = GlmOcrVisionAttention(config)
        self.mlp = Glm5NextVisionMLP(config)

    def forward(self, hidden_states, *, cu_seqlens, position_embeddings):
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            cu_seqlens=cu_seqlens,
            position_embeddings=position_embeddings,
        )
        return hidden_states + self.mlp(self.norm2(hidden_states))


class Glm5NextVisionPatchMerger(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        dim = config.out_hidden_size
        context = config.projection_intermediate_size
        self.proj = nn.Linear(dim, dim, bias=False)
        self.post_projection_norm = nn.LayerNorm(dim)
        self.gate_proj = nn.Linear(dim, context, bias=False)
        self.up_proj = nn.Linear(dim, context, bias=False)
        self.down_proj = nn.Linear(context, dim, bias=False)
        self.swiglu_limit = config.swiglu_limit

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = torch.nn.functional.gelu(self.post_projection_norm(self.proj(hidden_states)))
        gate = self.gate_proj(hidden_states).clamp(max=self.swiglu_limit)
        up = self.up_proj(hidden_states).clamp(min=-self.swiglu_limit, max=self.swiglu_limit)
        return self.down_proj(torch.nn.functional.silu(gate) * up)


class Glm5NextVisionModel(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_embed = GlmOcrVisionPatchEmbed(config)
        head_dim = config.hidden_size // config.num_heads
        self.rotary_pos_emb = GlmOcrVisionRotaryEmbedding(head_dim // 2)
        self.blocks = nn.ModuleList(Glm5NextVisionBlock(config) for _ in range(config.depth))
        self.post_layernorm = GlmOcrRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.downsample = nn.Conv2d(
            config.hidden_size,
            config.out_hidden_size,
            kernel_size=config.spatial_merge_size,
            stride=config.spatial_merge_size,
        )
        self.merger = Glm5NextVisionPatchMerger(config)

    def forward(self, pixel_values: torch.Tensor, image_grid_thw: torch.Tensor) -> torch.Tensor:
        positions = get_vision_position_ids(image_grid_thw, self.spatial_merge_size)
        cu_seqlens = get_vision_cu_seqlens(image_grid_thw)
        hidden_states = self.patch_embed(pixel_values)
        rotary = self.rotary_pos_emb(positions)
        rotary = torch.cat((rotary, rotary), dim=-1)
        position_embeddings = (rotary.cos(), rotary.sin())
        for block in self.blocks:
            hidden_states = block(
                hidden_states,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
            )
        hidden_states = self.post_layernorm(hidden_states)
        merge = self.spatial_merge_size
        hidden_states = hidden_states.view(-1, merge, merge, hidden_states.shape[-1]).permute(0, 3, 1, 2)
        hidden_states = self.downsample(hidden_states).view(-1, self.config.out_hidden_size)
        return self.merger(hidden_states)


@contextmanager
def _default_dtype(dtype: torch.dtype):
    previous = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


def build_glm5_next_visual(checkpoint: str, *, device: torch.device, dtype: torch.dtype) -> Glm5NextVisionModel:
    config = load_glm5_next_config(checkpoint).vision_config
    config._attn_implementation = "sdpa"
    with torch.device(device), _default_dtype(dtype):
        visual = Glm5NextVisionModel(config)
    reader = SafetensorReader(checkpoint)
    state_dict = {
        name.removeprefix(_VISUAL_PREFIX): reader.get_tensor(name)
        for name in reader.weight_map
        if name.startswith(_VISUAL_PREFIX)
    }
    if not state_dict:
        raise RuntimeError(f"No {_VISUAL_PREFIX} tensors found in {checkpoint}")
    incompatible = visual.load_state_dict(state_dict, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"GLM-5.3 visual mismatch: missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    visual.requires_grad_(False)
    visual.eval()
    return visual


def _scatter_image_embeddings(model, decoder_input, positions, image_embeddings):
    from megatron.core import parallel_state

    positions = positions.to(decoder_input.device)
    image_embeddings = image_embeddings.to(device=decoder_input.device, dtype=decoder_input.dtype)
    if model.config.sequence_parallel and parallel_state.get_tensor_model_parallel_world_size() > 1:
        local_sequence = decoder_input.shape[0]
        rank = parallel_state.get_tensor_model_parallel_rank()
        local_positions = positions - rank * local_sequence
        selected = (local_positions >= 0) & (local_positions < local_sequence)
        local_positions = local_positions[selected]
        image_embeddings = image_embeddings[selected]
    else:
        local_positions = positions
    decoder_input = decoder_input.clone()
    if local_positions.numel():
        decoder_input[local_positions, 0] = image_embeddings
    return decoder_input


def wire_glm5_next_visual(model, checkpoint: str) -> None:
    """Replace image-token embeddings while keeping text parameter names stable."""
    original_forward = model.forward
    if not model.pre_process:
        def passthrough(*args, pixel_values=None, image_grid_thw=None, **kwargs):
            del pixel_values, image_grid_thw
            return original_forward(*args, **kwargs)

        model.forward = passthrough
        return

    device = torch.device("cuda", torch.cuda.current_device())
    config = load_glm5_next_config(checkpoint)
    visual = build_glm5_next_visual(checkpoint, device=device, dtype=model.config.params_dtype)
    # Deliberately keep the frozen tower out of Megatron DDP/optimizer/checkpoint
    # state. Raw HF save and rollout sync preserve it from the source checkpoint.
    model.__dict__["_glm5_next_visual"] = visual

    def multimodal_forward(*args, pixel_values=None, image_grid_thw=None, **kwargs):
        if pixel_values is None:
            return original_forward(*args, **kwargs)
        if image_grid_thw is None:
            raise ValueError("pixel_values requires image_grid_thw")
        input_ids = kwargs.get("input_ids", args[0] if args else None)
        if input_ids is None or input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("GLM-5.3 VLM training expects packed input_ids with shape [1, sequence]")
        decoder_input = model.embedding(input_ids=input_ids, position_ids=kwargs.get("position_ids"))
        with torch.no_grad():
            image_embeddings = model.__dict__["_glm5_next_visual"](
                pixel_values.to(device=device, dtype=model.config.params_dtype),
                image_grid_thw.to(device=device),
            )
        image_positions = torch.nonzero(input_ids.reshape(-1) == config.image_token_id, as_tuple=False).flatten()
        if image_positions.numel() != image_embeddings.shape[0]:
            raise ValueError(
                f"GLM-5.3 image token/feature mismatch: {image_positions.numel()} tokens, "
                f"{image_embeddings.shape[0]} features"
            )
        kwargs["decoder_input"] = _scatter_image_embeddings(
            model,
            decoder_input,
            image_positions,
            image_embeddings,
        )
        return original_forward(*args, **kwargs)

    model.forward = multimodal_forward


def glm5_next_vlm_model_provider(pre_process=True, post_process=True, vp_stage=None):
    from megatron.core.models.gpt import GPTModel
    from megatron.training import get_args
    from megatron.training.arguments import core_transformer_config_from_args

    from slime_plugins.models.glm5_next.glm5_next import get_glm5_next_spec

    args = get_args()
    config = core_transformer_config_from_args(args)
    kwargs = {}
    if vp_stage is not None:
        kwargs["vp_stage"] = vp_stage
    model = GPTModel(
        config=config,
        transformer_layer_spec=get_glm5_next_spec(args, config, vp_stage),
        vocab_size=args.padded_vocab_size,
        max_sequence_length=args.max_position_embeddings,
        pre_process=pre_process,
        post_process=post_process,
        fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
        parallel_output=True,
        share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
        position_embedding_type=args.position_embedding_type,
        rotary_percent=args.rotary_percent,
        rotary_base=args.rotary_base,
        rope_scaling=args.use_rope_scaling,
        **kwargs,
    )
    wire_glm5_next_visual(model, args.hf_checkpoint)
    return model


__all__ = ["Glm5NextVisionModel", "build_glm5_next_visual", "glm5_next_vlm_model_provider"]
