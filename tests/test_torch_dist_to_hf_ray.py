import importlib.util
import io
import sys
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "convert_torch_dist_to_hf_ray.py"
SPEC = importlib.util.spec_from_file_location("convert_torch_dist_to_hf_ray", MODULE_PATH)
converter = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = converter
SPEC.loader.exec_module(converter)

_find_megatron_args = converter._find_megatron_args
copy_frozen_source_tensors = converter.copy_frozen_source_tensors
hf_expert_weight_name = converter.hf_expert_weight_name
plan_whole_source_tasks = converter.plan_whole_source_tasks
validate_path_layout = converter.validate_path_layout


def test_glm53_stateful_conversion_groups_are_atomic():
    metadata = {
        "decoder.layers.3.self_attention.linear_q_down_proj.weight": (torch.Size([8, 16]), torch.bfloat16),
        "decoder.layers.3.self_attention.linear_kv_down_proj.weight": (torch.Size([4, 16]), torch.bfloat16),
        "decoder.layers.3.self_attention_hyper_connection.alpha_pre": (torch.Size([1]), torch.float32),
        "decoder.layers.3.self_attention_hyper_connection.alpha_post": (torch.Size([1]), torch.float32),
        "decoder.layers.3.self_attention_hyper_connection.alpha_res": (torch.Size([1]), torch.float32),
    }

    tasks = plan_whole_source_tasks(metadata, q_lora_rank=8, task_group_bytes=0)
    key_sets = {frozenset(task.keys) for task in tasks}

    assert frozenset(
        {
            "decoder.layers.3.self_attention.linear_q_down_proj.weight",
            "decoder.layers.3.self_attention.linear_kv_down_proj.weight",
        }
    ) in key_sets
    assert frozenset(
        {
            "decoder.layers.3.self_attention_hyper_connection.alpha_pre",
            "decoder.layers.3.self_attention_hyper_connection.alpha_post",
            "decoder.layers.3.self_attention_hyper_connection.alpha_res",
        }
    ) in key_sets


def test_glm53_prefixed_stateful_conversion_groups_are_atomic():
    prefix = "language_model.decoder.layers.3."
    metadata = {
        prefix + "self_attention.linear_q_down_proj.weight": (torch.Size([8, 16]), torch.bfloat16),
        prefix + "self_attention.linear_kv_down_proj.weight": (torch.Size([4, 16]), torch.bfloat16),
        prefix + "mlp_hyper_connection.alpha_pre": (torch.Size([1]), torch.float32),
        prefix + "mlp_hyper_connection.alpha_post": (torch.Size([1]), torch.float32),
        prefix + "mlp_hyper_connection.alpha_res": (torch.Size([1]), torch.float32),
    }

    tasks = plan_whole_source_tasks(metadata, q_lora_rank=8, task_group_bytes=0)
    key_sets = {frozenset(task.keys) for task in tasks}

    assert frozenset(
        {
            prefix + "self_attention.linear_q_down_proj.weight",
            prefix + "self_attention.linear_kv_down_proj.weight",
        }
    ) in key_sets
    assert frozenset(
        {
            prefix + "mlp_hyper_connection.alpha_pre",
            prefix + "mlp_hyper_connection.alpha_post",
            prefix + "mlp_hyper_connection.alpha_res",
        }
    ) in key_sets


def test_incomplete_glm53_stateful_group_fails_closed():
    metadata = {
        "decoder.layers.3.mlp_hyper_connection.alpha_pre": (torch.Size([1]), torch.float32),
        "decoder.layers.3.mlp_hyper_connection.alpha_post": (torch.Size([1]), torch.float32),
    }

    with pytest.raises(ValueError, match="Incomplete mHC scale group"):
        plan_whole_source_tasks(metadata, q_lora_rank=None, task_group_bytes=0)


def test_direct_expert_names_use_glm53_composite_prefix():
    assert (
        hf_expert_weight_name("", 7, 19, "gate_proj", "glm5next")
        == "model.language_model.layers.7.mlp.experts.19.gate_proj.weight"
    )


def test_frozen_visual_tensors_are_streamed_from_source(tmp_path):
    source = tmp_path / "source"
    staging = tmp_path / "staging"
    source.mkdir()
    staging.mkdir()
    save_file(
        {
            "model.language_model.norm.weight": torch.ones(2),
            "model.visual.patch_embed.weight": torch.arange(8, dtype=torch.float32).reshape(2, 4),
            "model.visual.norm.weight": torch.ones(2),
        },
        source / "model.safetensors",
    )

    result = copy_frozen_source_tensors(
        str(source),
        str(staging),
        task_id=9,
        max_file_bytes=32,
        prefix="model.visual.",
    )

    assert result.weights == 2
    assert len(result.shards) == 2
    exported = set()
    for shard in result.shards:
        with safe_open(staging / shard.temp_filename, framework="pt", device="cpu") as handle:
            exported.update(handle.keys())
    assert exported == {"model.visual.norm.weight", "model.visual.patch_embed.weight"}


def test_find_megatron_args_in_dcp_common_state():
    expected = Namespace(num_layers=45)

    assert _find_megatron_args([{"args": expected, "iteration": 2}]) is expected


@pytest.mark.parametrize("output", ("input", "input/export", "."))
def test_converter_rejects_output_overlapping_checkpoint(tmp_path, output):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    output_dir = tmp_path / output

    with pytest.raises(ValueError, match="output_dir must not overlap input_dir"):
        validate_path_layout(str(input_dir), str(output_dir), None)


def test_converter_rejects_output_overlapping_hf_source(tmp_path):
    input_dir = tmp_path / "input"
    origin = tmp_path / "origin"
    input_dir.mkdir()
    origin.mkdir()

    with pytest.raises(ValueError, match="output_dir must not overlap origin_hf_dir"):
        validate_path_layout(str(input_dir), str(origin / "export"), str(origin))


def test_load_megatron_args_from_current_dcp_common_state(tmp_path):
    expected = Namespace(num_layers=45, original_hf_model_name="glm5next")
    payload = io.BytesIO()
    torch.save([{"args": expected, "iteration": 2}], payload)
    checkpoint_file = tmp_path / "__0_0.distcp"
    checkpoint_file.write_bytes(payload.getvalue())
    index = converter.MetadataIndex(fqn="common_state/shard_0_1")
    metadata = SimpleNamespace(
        storage_data={
            index: SimpleNamespace(
                relative_path=checkpoint_file.name,
                offset=0,
                length=len(payload.getvalue()),
                transform_descriptors=(),
            )
        }
    )

    actual = converter._load_megatron_args_from_dcp(str(tmp_path), metadata)

    assert actual.num_layers == 45
    assert actual.original_hf_model_name == "glm5next"


def test_load_megatron_args_from_legacy_common_pt(tmp_path):
    expected = Namespace(num_layers=45, original_hf_model_name="glm5next")
    torch.save({"args": expected}, tmp_path / "common.pt")

    actual, model_name = converter.load_megatron_args(
        str(tmp_path),
        SimpleNamespace(storage_data=None),
        model_name_override=None,
        vocab_size=154880,
    )

    assert actual.num_layers == 45
    assert actual.vocab_size == 154880
    assert model_name == "glm5next"


def test_gpu_node_selection_excludes_cpu_only_nodes(monkeypatch):
    monkeypatch.setattr(
        converter.ray,
        "nodes",
        lambda: [
            {"NodeID": "cpu", "Alive": True, "Resources": {"CPU": 8}},
            {"NodeID": "gpu", "Alive": True, "Resources": {"CPU": 8, "GPU": 1}},
            {"NodeID": "dead", "Alive": False, "Resources": {"GPU": 8}},
        ],
    )

    assert converter.live_ray_node_ids(requires_gpu=True) == ["gpu"]
    assert converter.live_ray_node_ids() == ["cpu", "gpu"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA staging")
def test_quantized_writer_stages_cpu_groups_on_reserved_cuda_device(tmp_path, monkeypatch):
    observed = {}

    def fake_quantize(_args, _source_name, tensors, _config, transform_ue8m0):
        observed["devices"] = [tensor.device.type for _name, tensor in tensors]
        observed["transform_ue8m0"] = transform_ue8m0
        return tensors

    monkeypatch.setattr(converter.m2hf, "quantize_params", fake_quantize)
    group = converter.PreparedTensorGroup(
        source_name="module.module.decoder.layers.3.mlp.experts.linear_fc2.weight0",
        tensors=(("model.language_model.layers.3.mlp.experts.0.down_proj.weight", torch.ones(4, 4)),),
    )

    shards, _size = converter.write_prepared_tensor_groups(
        str(tmp_path),
        task_id=0,
        groups=(group,),
        megatron_args=Namespace(),
        quantization_config={"quant_method": "fp8"},
        max_file_bytes=1024,
        cuda_device_id=0,
    )

    assert observed == {"devices": ["cuda"], "transform_ue8m0": False}
    with safe_open(tmp_path / shards[0].temp_filename, framework="pt", device="cpu") as handle:
        assert handle.get_tensor(group.tensors[0][0]).device.type == "cpu"
