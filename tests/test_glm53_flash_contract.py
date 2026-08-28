import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image
from safetensors.torch import save_file

from slime.backends.megatron_utils.hf_to_megatron.common import SafetensorReader
from slime.backends.megatron_utils.hf_to_megatron.glm5_next import glm5_next_hf_tensor
from slime.backends.megatron_utils.megatron_to_hf.glm5_next import convert_glm5_next_to_hf
from slime.backends.megatron_utils.megatron_to_hf.processors.quantizer_fp8 import quantize_params_fp8
from slime_plugins.models.glm5_next.config import (
    attention_schedules,
    get_text_config,
    load_glm5_next_config,
    validate_glm5_next_checkpoint,
)

NUM_GPUS = 0


def _tiny_checkpoint() -> Path:
    path = os.environ.get("GLM53_TINY_CHECKPOINT")
    if not path:
        pytest.skip("Set GLM53_TINY_CHECKPOINT to run the real tiny-checkpoint contract test")
    checkpoint = Path(path)
    if not (checkpoint / "model.safetensors").is_file():
        pytest.skip(f"Tiny GLM-5.3 safetensors are unavailable at {checkpoint}")
    return checkpoint


def test_real_tiny_checkpoint_matches_hybrid_contract():
    checkpoint = _tiny_checkpoint()

    report = validate_glm5_next_checkpoint(checkpoint)
    text = get_text_config(load_glm5_next_config(checkpoint))

    assert report == {"tensor_count": 223, "backbone_layers": 5, "has_mtp": False, "vision_blocks": 2}
    assert attention_schedules(text) == ([0, 1, 2, 4], [3])
    assert text.mlp_layer_types == ["dense", "dense", "dense", "sparse", "sparse"]
    assert (text.n_routed_experts, text.num_experts_per_tok) == (8, 4)
    assert (text.hc_mult, text.index_kpool) == (4, 4)


def test_real_tiny_checkpoint_round_trips_representative_state_families():
    checkpoint = _tiny_checkpoint()
    config = load_glm5_next_config(checkpoint)
    reader = SafetensorReader(checkpoint)
    args = SimpleNamespace(
        hidden_size=256,
        num_attention_heads=4,
        num_query_groups=4,
        kv_channels=64,
        q_lora_rank=128,
        params_dtype=torch.bfloat16,
    )

    direct = {
        "module.module.embedding.word_embeddings.weight": "model.language_model.embed_tokens.weight",
        "module.module.decoder.final_layernorm.weight": "model.language_model.norm.weight",
        "module.module.output_layer.weight": "lm_head.weight",
        "module.module.decoder.layers.0.input_layernorm.weight": (
            "model.language_model.layers.0.input_layernorm.weight"
        ),
        "module.module.decoder.layers.0.self_attention.kda.q_proj.weight": (
            "model.language_model.layers.0.self_attn.q_proj.weight"
        ),
        "module.module.decoder.layers.0.self_attention.kda.A_log": (
            "model.language_model.layers.0.self_attn.A_log"
        ),
        "module.module.decoder.layers.0.self_attention_hyper_connection.mapping_proj.weight": (
            "model.language_model.layers.0.hc_attn_fn"
        ),
        "module.module.decoder.layers.3.self_attention.linear_q_down_proj.weight": (
            "model.language_model.layers.3.self_attn.q_a_proj.weight"
        ),
        "module.module.decoder.layers.3.self_attention.linear_kv_up_proj.weight": (
            "model.language_model.layers.3.self_attn.kv_b_proj.weight"
        ),
        "module.module.decoder.layers.3.self_attention.wq_b.weight": (
            "model.language_model.layers.3.self_attn.indexer.wq_b.weight"
        ),
        "module.module.decoder.layers.3.self_attention.index_kpool_compress_ape": (
            "model.language_model.layers.3.self_attn.indexer.index_kpool_compress_ape"
        ),
        "module.module.decoder.layers.3.mlp.router.expert_bias": (
            "model.language_model.layers.3.mlp.gate.e_score_correction_bias"
        ),
        "module.module.decoder.layers.3.mlp.experts.linear_fc2.weight7": (
            "model.language_model.layers.3.mlp.experts.7.down_proj.weight"
        ),
        "module.module.decoder.layers.3.mlp.shared_experts.linear_fc2.weight": (
            "model.language_model.layers.3.mlp.shared_experts.down_proj.weight"
        ),
    }
    for megatron_name, hf_name in direct.items():
        loaded = glm5_next_hf_tensor(megatron_name, reader, config)
        converted = dict(convert_glm5_next_to_hf(args, megatron_name, loaded))
        assert set(converted) == {hf_name}
        assert torch.equal(converted[hf_name], reader.get_tensor(hf_name).to(converted[hf_name].dtype))

    conv_name = "module.module.decoder.layers.4.self_attention.kda.conv1d.weight"
    conv = glm5_next_hf_tensor(conv_name, reader, config)
    converted_conv = dict(convert_glm5_next_to_hf(args, conv_name, conv))
    for kind in ("q", "k", "v"):
        hf_name = f"model.language_model.layers.4.self_attn.{kind}_conv1d.weight"
        assert torch.equal(converted_conv[hf_name], reader.get_tensor(hf_name))

    fc1_name = "module.module.decoder.layers.3.mlp.experts.linear_fc1.weight6"
    fc1 = glm5_next_hf_tensor(fc1_name, reader, config)
    converted_fc1 = dict(convert_glm5_next_to_hf(args, fc1_name, fc1))
    for projection in ("gate_proj", "up_proj"):
        hf_name = f"model.language_model.layers.3.mlp.experts.6.{projection}.weight"
        assert torch.equal(converted_fc1[hf_name], reader.get_tensor(hf_name))

    scale_names = []
    for alpha in ("alpha_pre", "alpha_post", "alpha_res"):
        name = f"module.module.decoder.layers.3.mlp_hyper_connection.{alpha}"
        tensor = glm5_next_hf_tensor(name, reader, config)
        scale_names.extend(convert_glm5_next_to_hf(args, name, tensor))
    assert len(scale_names) == 1
    hf_scale_name, hf_scale = scale_names[0]
    assert hf_scale_name == "model.language_model.layers.3.hc_ffn_scale"
    assert torch.equal(hf_scale, reader.get_tensor(hf_scale_name))


def test_real_tiny_checkpoint_exhaustively_round_trips_all_text_tensors():
    import slime.backends.megatron_utils.megatron_to_hf as conversion
    from slime.backends.megatron_utils.megatron_to_hf.glm5_next import _HC_BUFFERS

    checkpoint = _tiny_checkpoint()
    config = load_glm5_next_config(checkpoint)
    text = get_text_config(config)
    reader = SafetensorReader(checkpoint)
    args = SimpleNamespace(
        vocab_size=text.vocab_size,
        hidden_size=text.hidden_size,
        num_attention_heads=text.num_attention_heads,
        num_query_groups=text.num_key_value_heads,
        kv_channels=64,
        q_lora_rank=text.q_lora_rank,
        num_experts=text.n_routed_experts,
        params_dtype=torch.bfloat16,
        force_fp8_ue8m0_scale=False,
    )
    names = [
        "module.module.embedding.word_embeddings.weight",
        "module.module.decoder.final_layernorm.weight",
        "module.module.output_layer.weight",
    ]
    common = ["input_layernorm.weight", "pre_mlp_layernorm.weight"]
    hc = [
        f"{site}.{name}"
        for site in ("self_attention_hyper_connection", "mlp_hyper_connection")
        for name in ("mapping_proj.weight", "bias", "alpha_pre", "alpha_post", "alpha_res")
    ]
    kda = [
        f"self_attention.kda.{name}"
        for name in (
            "q_proj.weight",
            "k_proj.weight",
            "v_proj.weight",
            "conv1d.weight",
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
    ]
    dsa = [
        f"self_attention.{name}"
        for name in (
            "linear_q_down_proj.weight",
            "q_layernorm.weight",
            "linear_q_up_proj.weight",
            "linear_kv_down_proj.weight",
            "kv_layernorm.weight",
            "linear_kv_up_proj.weight",
            "linear_proj.weight",
            "wq_b.weight",
            "wk.weight",
            "weights_proj.weight",
            "k_norm.weight",
            "k_norm.bias",
            "index_kpool_compress_gate",
            "index_kpool_compress_ape",
        )
    ]
    kda_layers, dsa_layers = map(set, attention_schedules(text))
    for layer in range(text.num_hidden_layers):
        layer_names = common + hc + (kda if layer in kda_layers else dsa)
        if text.mlp_layer_types[layer] == "dense":
            layer_names += ["mlp.linear_fc1.weight", "mlp.linear_fc2.weight"]
        else:
            layer_names += [
                "mlp.router.weight",
                "mlp.router.expert_bias",
                "mlp.shared_experts.linear_fc1.weight",
                "mlp.shared_experts.linear_fc2.weight",
            ]
            layer_names += [
                f"mlp.experts.linear_fc{projection}.weight{expert}"
                for expert in range(text.n_routed_experts)
                for projection in (1, 2)
            ]
        names += [f"module.module.decoder.layers.{layer}.{name}" for name in layer_names]

    conversion._cached_tensors.clear()
    _HC_BUFFERS.clear()
    exported = {}
    for name in names:
        parameter = glm5_next_hf_tensor(name, reader, config)
        for hf_name, tensor in conversion.convert_to_hf(args, "glm5_next", name, parameter):
            assert hf_name not in exported, f"Duplicate exported tensor {hf_name}"
            exported[hf_name] = tensor

    expected_names = {name for name in reader.weight_map if not name.startswith("model.visual.")}
    assert len(names) == 175
    assert len(expected_names) == len(exported) == 184
    assert set(exported) == expected_names
    for name in expected_names:
        expected = reader.get_tensor(name)
        actual = exported[name]
        assert actual.shape == expected.shape, name
        assert actual.dtype == expected.dtype, name
        assert torch.equal(actual, expected), name
    assert not conversion._cached_tensors
    assert not _HC_BUFFERS


def test_conversion_cache_scope_fails_closed_and_cleans_up():
    import slime.backends.megatron_utils.megatron_to_hf as conversion
    from slime.backends.megatron_utils.megatron_to_hf.glm5_next import _HC_BUFFERS

    with pytest.raises(RuntimeError, match="incomplete paired tensors"):
        with conversion.conversion_cache_scope():
            conversion._cached_tensors["unpaired"] = torch.ones(1)
            _HC_BUFFERS[(1, "3", "attn")] = {"alpha_pre": torch.ones(1)}
    assert not conversion._cached_tensors
    assert not _HC_BUFFERS

    with pytest.raises(ValueError, match="interrupted"):
        with conversion.conversion_cache_scope():
            conversion._cached_tensors["unpaired"] = torch.ones(1)
            raise ValueError("interrupted conversion")
    assert not conversion._cached_tensors
    assert not _HC_BUFFERS


@pytest.mark.parametrize(
    "hf_name",
    [
        "model.language_model.layers.3.self_attn.kv_b_proj.weight",
        "model.language_model.layers.3.self_attn.indexer.wq_b.weight",
        "model.language_model.layers.3.self_attn.indexer.wk.weight",
    ],
)
def test_glm53_fp8_export_preserves_bf16_only_dsa_tensors(hf_name):
    value = torch.randn(4, 4, dtype=torch.bfloat16)
    output = quantize_params_fp8(
        SimpleNamespace(force_fp8_ue8m0_scale=False),
        "module.module.decoder.layers.3.self_attention.linear_kv_up_proj.weight",
        [(hf_name, value)],
        {
            "quant_method": "fp8",
            "fmt": "e4m3",
            "activation_scheme": "dynamic",
            "weight_block_size": [128, 128],
        },
    )

    assert output[0][0] == hf_name
    assert output[0][1] is value


def test_config_validation_rejects_glm53_and_flash_schedule_mixup(tmp_path: Path):
    source = _tiny_checkpoint() / "config.json"
    config = json.loads(source.read_text(encoding="utf-8"))
    config["text_config"]["layer_types"] = ["deepseek_sparse_attention"] * 5
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="schedules disagree"):
        load_glm5_next_config(tmp_path)


def test_config_validation_rejects_stale_compact_qk_and_mtp_contracts(tmp_path: Path):
    checkpoint = _tiny_checkpoint()
    config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    config["text_config"]["qk_head_dim"] = 256
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="qk_head_dim disagrees"):
        load_glm5_next_config(tmp_path)

    config["text_config"]["qk_head_dim"] = 64
    config["text_config"]["num_nextn_predict_layers"] = 1
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (tmp_path / "model.safetensors").symlink_to(checkpoint / "model.safetensors")
    with pytest.raises(ValueError, match="declares MTP=1"):
        validate_glm5_next_checkpoint(tmp_path)


def test_fp8_reader_dequantizes_in_fp32_before_casting_to_bf16(tmp_path: Path):
    torch.manual_seed(31)
    source = (torch.randn(129, 130) * 3).to(torch.float8_e4m3fn)
    scale = torch.tensor([[0.12345, 1.71337], [2.517, 0.371]], dtype=torch.float32)
    save_file({"weight": source, "weight_scale_inv": scale}, tmp_path / "model.safetensors")

    actual = SafetensorReader(tmp_path).get_tensor("weight")
    expanded_scale = scale.repeat_interleave(128, 0).repeat_interleave(128, 1)[:129, :130]
    expected = (source.float() * expanded_scale).to(torch.bfloat16)
    premature_bf16 = source.to(torch.bfloat16).mul(expanded_scale.to(torch.bfloat16))

    assert torch.equal(actual, expected)
    assert not torch.equal(actual, premature_bf16)


def test_safetensor_reader_bounds_open_full_checkpoint_shards(tmp_path: Path):
    weight_map = {}
    for index, name in enumerate(("a", "b", "c"), start=1):
        filename = f"model-{index:05d}-of-00003.safetensors"
        save_file({name: torch.full((2,), index, dtype=torch.float32)}, tmp_path / filename)
        weight_map[name] = filename
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": weight_map}), encoding="utf-8"
    )

    with SafetensorReader(tmp_path, max_open_files=2) as reader:
        first = reader.get_tensor("a")
        reader.get_tensor("b")
        reader.get_tensor("c")
        assert len(reader._files) == 2
        assert list(reader._files) == [weight_map["b"], weight_map["c"]]
        torch.testing.assert_close(first, torch.ones(2))
    assert not reader._files


def test_real_tiny_image_sft_rollout_preserves_media_and_mask_alignment(monkeypatch):
    from slime.rollout import sft_rollout
    from slime.utils.processing_utils import build_processor_kwargs, load_processor, process_vision_info
    from slime.utils.types import Sample
    from slime_plugins.models.glm5_next.vision import build_glm5_next_visual

    checkpoint = _tiny_checkpoint()
    image = Image.new("RGB", (28, 28), "red")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Describe."},
            ],
        },
        {"role": "assistant", "content": "A red square."},
    ]
    processor = load_processor(str(checkpoint), trust_remote_code=True)
    raw_inputs = process_vision_info(messages, processor)
    sample = Sample(prompt=messages, multimodal_inputs=raw_inputs)

    class Buffer:
        @staticmethod
        def get_samples(count):
            assert count == 1
            return [[sample]]

    args = SimpleNamespace(
        rollout_global_dataset=True,
        hf_checkpoint=str(checkpoint),
        loss_mask_type="qwen",
        rollout_batch_size=1,
        apply_chat_template_kwargs={},
    )
    monkeypatch.setattr(sft_rollout, "TOKENIZER", None)
    monkeypatch.setattr(sft_rollout, "PROCESSOR", None)
    monkeypatch.setattr(sft_rollout, "MASK_GENERATOR", None)
    monkeypatch.setattr(sft_rollout, "SAMPLE_PRINTED", False)

    output = sft_rollout.generate_rollout(args, 0, Buffer())[0][0]
    full_mask = [0] * (len(output.tokens) - len(output.loss_mask)) + output.loss_mask

    assert output.multimodal_inputs is raw_inputs
    assert len(output.tokens) == 37
    assert output.tokens.count(processor.image_token_id) == 16
    assert output.response_length == sum(output.loss_mask) == 5
    assert set(output.multimodal_train_inputs) == {"pixel_values", "image_grid_thw"}
    assert output.multimodal_train_inputs["pixel_values"].shape == (64, 1176)
    assert output.multimodal_train_inputs["image_grid_thw"].tolist() == [[1, 8, 8]]
    assert sft_rollout.MASK_GENERATOR.get_text_from_loss_mask(output.tokens, full_mask) == [
        "</think>A red square."
    ]
    visual = build_glm5_next_visual(str(checkpoint), device=torch.device("cpu"), dtype=torch.bfloat16)
    with torch.no_grad():
        image_embeddings = visual(
            output.multimodal_train_inputs["pixel_values"].to(torch.bfloat16),
            output.multimodal_train_inputs["image_grid_thw"],
        )
    assert image_embeddings.shape == (16, 256)
    assert torch.isfinite(image_embeddings).all()
    assert not any(parameter.requires_grad for parameter in visual.parameters())

    blue = Image.new("RGB", (28, 28), "blue")
    interleaved = [
        {
            "role": "user",
            "content": [{"type": "image", "image": image}, {"type": "text", "text": "First?"}],
        },
        {"role": "assistant", "content": "Red.", "step_loss_mask": 0},
        {
            "role": "user",
            "content": [{"type": "text", "text": "Second?"}, {"type": "image", "image": blue}],
        },
        {"role": "assistant", "content": "Blue."},
    ]
    rendered = processor.apply_chat_template(interleaved, tokenize=False, add_generation_prompt=False)
    processed = processor(
        text=rendered,
        **build_processor_kwargs({"images": [image, blue]}),
    )
    interleaved_ids = list(processed["input_ids"][0])
    _, interleaved_mask = sft_rollout.MASK_GENERATOR.get_loss_mask_with_multimodal_alignment(
        interleaved,
        interleaved_ids,
    )
    media_ids = {
        processor.image_token_id,
        processor.tokenizer.convert_tokens_to_ids("<|begin_of_image|>"),
        processor.tokenizer.convert_tokens_to_ids("<|end_of_image|>"),
    }
    assert not any(
        mask for token, mask in zip(interleaved_ids, interleaved_mask, strict=True) if token in media_ids
    )
    assert sft_rollout.MASK_GENERATOR.get_text_from_loss_mask(interleaved_ids, interleaved_mask) == [
        "</think>Blue."
    ]
    with pytest.raises(ValueError, match="does not preserve"):
        sft_rollout.MASK_GENERATOR.get_loss_mask_with_multimodal_alignment(
            interleaved,
            interleaved_ids[:-1],
        )
