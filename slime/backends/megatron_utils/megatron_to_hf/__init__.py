from contextlib import contextmanager

from .deepseekv3 import convert_deepseekv3_to_hf
from .glm4 import convert_glm4_to_hf
from .glm4moe import convert_glm4moe_to_hf
from .glm5_next import convert_glm5_next_to_hf
from .llama import convert_llama_to_hf
from .mimo import convert_mimo_to_hf
from .minimax_m2 import convert_minimax_m2_to_hf
from .processors import quantize_params, remove_padding
from .qwen2 import convert_qwen2_to_hf
from .qwen3_5 import convert_qwen3_5_to_hf
from .qwen3_next import convert_qwen3_next_to_hf
from .qwen3_vl import convert_qwen3vl_to_hf
from .qwen3moe import convert_qwen3moe_to_hf


# TODO optimize code details
def convert_to_hf(args, model_name, name, param, quantization_config=None, transform_ue8m0=True):
    hf_name = name
    while hf_name.startswith("module."):
        hf_name = hf_name.removeprefix("module.")
    if hf_name.startswith("model.visual."):
        return [(hf_name, param)]

    param = remove_padding(name, param, args.vocab_size)
    converted_named_tensors = _convert_to_hf_core(args, model_name, name, param)

    return quantize_params(args, name, converted_named_tensors, quantization_config, transform_ue8m0)


# TODO optimize
_cached_tensors = {}
_conversion_scope_depth = 0


def _clear_conversion_caches() -> None:
    from .glm5_next import _HC_BUFFERS

    _cached_tensors.clear()
    _HC_BUFFERS.clear()


def _assert_conversion_caches_empty() -> None:
    from .glm5_next import _HC_BUFFERS

    if _cached_tensors or _HC_BUFFERS:
        raise RuntimeError(
            "Megatron-to-HF conversion ended with incomplete paired tensors: "
            f"projection_pairs={sorted(_cached_tensors)}, mhc_groups={sorted(_HC_BUFFERS)}"
        )


@contextmanager
def conversion_cache_scope():
    """Keep stateful tensor pairing local to one complete conversion iterator."""
    global _conversion_scope_depth
    outermost = _conversion_scope_depth == 0
    if outermost:
        _clear_conversion_caches()
    _conversion_scope_depth += 1
    completed = False
    try:
        yield
        completed = True
    finally:
        _conversion_scope_depth -= 1
        if outermost:
            try:
                if completed:
                    _assert_conversion_caches_empty()
            finally:
                _clear_conversion_caches()


# TODO optimize code details
def _convert_to_hf_core(args, model_name, name, param):
    model_name = model_name.lower().replace("_", "").replace("-", "")
    if "glm5next" in model_name:
        converted_named_tensors = convert_glm5_next_to_hf(args, name, param)
    elif "minimaxm2" in model_name:
        converted_named_tensors = convert_minimax_m2_to_hf(args, name, param)
    elif any(family in model_name for family in ("glm4moelite", "deepseekv3", "deepseekv32", "glmmoedsa", "kimi")):
        converted_named_tensors = convert_deepseekv3_to_hf(args, name, param)
    elif "glm4moe" in model_name:
        converted_named_tensors = convert_glm4moe_to_hf(args, name, param)
    elif "glm4" in model_name:
        converted_named_tensors = convert_glm4_to_hf(args, name, param)
    elif "qwen3next" in model_name:
        converted_named_tensors = convert_qwen3_next_to_hf(args, name, param)
    elif "qwen35" in model_name:
        converted_named_tensors = convert_qwen3_5_to_hf(args, name, param)
    elif "qwen3vl" in model_name:
        converted_named_tensors = convert_qwen3vl_to_hf(args, name, param)
    elif "qwen2moe" in model_name or "qwen3moe" in model_name:
        converted_named_tensors = convert_qwen3moe_to_hf(args, name, param)
    elif "qwen2" in model_name or "qwen3" in model_name:
        converted_named_tensors = convert_qwen2_to_hf(args, name, param)
    elif "llama" in model_name:
        converted_named_tensors = convert_llama_to_hf(args, name, param)
    elif "mimo" in model_name:
        converted_named_tensors = convert_mimo_to_hf(args, name, param)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    # to compatible with sglang implementation
    if args.q_lora_rank is not None:
        old_converted_named_tensors = converted_named_tensors
        converted_named_tensors = []
        for converted_name, converted_param in old_converted_named_tensors:
            if "q_a_proj" in converted_name:
                pair_name = converted_name.replace("q_a_proj", "kv_a_proj_with_mqa")
                if pair_name in _cached_tensors:
                    converted_named_tensors += [
                        (converted_name, converted_param),
                        (pair_name, _cached_tensors[pair_name]),
                    ]
                    del _cached_tensors[pair_name]
                else:
                    _cached_tensors[converted_name] = converted_param
            elif "kv_a_proj_with_mqa" in converted_name:
                pair_name = converted_name.replace("kv_a_proj_with_mqa", "q_a_proj")
                if pair_name in _cached_tensors:
                    converted_named_tensors += [
                        (converted_name, converted_param),
                        (pair_name, _cached_tensors[pair_name]),
                    ]
                    del _cached_tensors[pair_name]
                else:
                    _cached_tensors[converted_name] = converted_param
            else:
                converted_named_tensors.append((converted_name, converted_param))
    return converted_named_tensors
