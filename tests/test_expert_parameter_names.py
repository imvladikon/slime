from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from slime.backends.megatron_utils.megatron_to_hf.glm5_next import convert_glm5_next_to_hf
from slime.backends.megatron_utils.update_weight.common import named_params_and_buffers
from slime.backends.megatron_utils.update_weight.hf_weight_iterator_direct import (
    _get_megatron_local_param_infos,
)
from slime.utils.types import ParamInfo

NUM_GPUS = 0


class _LocalExpertModel:
    config = SimpleNamespace()

    def named_parameters(self):
        for layer in (3, 4):
            for expert in range(4):
                for projection in (1, 2):
                    name = (
                        f"module.decoder.layers.{layer}.mlp.experts.local_experts.{expert}."
                        f"linear_fc{projection}.weight"
                    )
                    shape = (4, 2) if projection == 1 else (2, 2)
                    yield name, torch.nn.Parameter(torch.full(shape, layer + expert + projection, dtype=torch.float32))

    @staticmethod
    def named_buffers():
        return []


class _EP2LocalExpertModel(_LocalExpertModel):
    def __init__(self):
        from megatron.core.transformer.transformer_config import TransformerConfig

        self.config = TransformerConfig(num_layers=5, hidden_size=8, num_attention_heads=1)

    def named_parameters(self):
        yield "module.decoder.final_layernorm.weight", torch.nn.Parameter(torch.ones(8))
        yield (
            "module.decoder.layers.3.mlp.experts.local_experts.0.linear_fc1.weight",
            torch.nn.Parameter(torch.ones(4, 8)),
        )
        yield (
            "module.decoder.layers.3.mlp.experts.local_experts.0.linear_fc2.weight",
            torch.nn.Parameter(torch.ones(8, 2)),
        )


def _run_ep2_metadata_contract(rank, init_method):
    from megatron.core import parallel_state
    from slime.utils.distributed_utils import set_gloo_group

    dist.init_process_group("gloo", init_method=init_method, rank=rank, world_size=2)
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        expert_model_parallel_size=2,
    )
    set_gloo_group(dist.group.WORLD)
    original_all_gather_object = dist.all_gather_object
    ep_payloads = []

    def spy_all_gather_object(object_list, obj, group=None):
        if isinstance(obj, tuple) and len(obj) == 2 and isinstance(obj[1], dict):
            ep_payloads.append(obj[1])
        return original_all_gather_object(object_list, obj, group=group)

    dist.all_gather_object = spy_all_gather_object
    try:
        infos = _get_megatron_local_param_infos(
            SimpleNamespace(num_experts=2), [_EP2LocalExpertModel()]
        )
        by_name = {info.name: info for info in infos}
        dense_name = "module.module.decoder.final_layernorm.weight"
        fc1 = "module.module.decoder.layers.3.mlp.experts.linear_fc1.weight"
        fc2 = "module.module.decoder.layers.3.mlp.experts.linear_fc2.weight"
        assert set(by_name) == {dense_name, fc1 + "0", fc1 + "1", fc2 + "0", fc2 + "1"}
        assert by_name[dense_name].src_rank == rank
        assert by_name[fc1 + "0"].src_rank == by_name[fc2 + "0"].src_rank == 0
        assert by_name[fc1 + "1"].src_rank == by_name[fc2 + "1"].src_rank == 1
        assert len(ep_payloads) == 1
        assert ep_payloads[0]
        assert all(".experts." in name for name in ep_payloads[0])
    finally:
        dist.all_gather_object = original_all_gather_object
        set_gloo_group(None)
        parallel_state.destroy_model_parallel()
        dist.destroy_process_group()


def _canonical_names(monkeypatch, ep_rank):
    monkeypatch.setattr(
        "slime.backends.megatron_utils.update_weight.common.get_transformer_layer_offset",
        lambda _config: 0,
    )
    monkeypatch.setattr(
        "slime.backends.megatron_utils.update_weight.common.mpu.get_expert_model_parallel_world_size",
        lambda: 2,
    )
    monkeypatch.setattr(
        "slime.backends.megatron_utils.update_weight.common.mpu.get_expert_model_parallel_rank",
        lambda: ep_rank,
    )
    return list(named_params_and_buffers(SimpleNamespace(num_experts=8), [_LocalExpertModel()]))


def test_local_mlp_ep2_names_cover_every_global_expert_and_hf_tensor(monkeypatch):
    rank0 = _canonical_names(monkeypatch, 0)
    rank1 = _canonical_names(monkeypatch, 1)

    assert len(rank0) == len(rank1) == 16
    all_named = rank0 + rank1
    names = [name for name, _param in all_named]
    assert len(names) == len(set(names)) == 32
    assert not any("local_experts" in name for name in names)
    expected = {
        f"module.module.decoder.layers.{layer}.mlp.experts.linear_fc{projection}.weight{expert}"
        for layer in (3, 4)
        for expert in range(8)
        for projection in (1, 2)
    }
    assert set(names) == expected

    args = SimpleNamespace(
        num_experts=8,
        params_dtype=torch.bfloat16,
        hidden_size=2,
        num_attention_heads=1,
        num_query_groups=1,
        kv_channels=2,
        q_lora_rank=2,
    )
    exported = {}
    for name, tensor in all_named:
        for hf_name, value in convert_glm5_next_to_hf(args, name, tensor):
            assert hf_name not in exported
            exported[hf_name] = value
    assert len(exported) == 48
    assert {
        f"model.language_model.layers.{layer}.mlp.experts.{expert}.{projection}_proj.weight"
        for layer in (3, 4)
        for expert in range(8)
        for projection in ("gate", "up", "down")
    } == set(exported)


def test_ep_metadata_exchange_sends_only_routed_experts(monkeypatch):
    dense = torch.nn.Parameter(torch.ones(2))
    local_expert = torch.nn.Parameter(torch.ones(2, 2))
    local_name = "module.module.decoder.layers.3.mlp.experts.linear_fc1.weight0"
    remote_name = "module.module.decoder.layers.3.mlp.experts.linear_fc1.weight4"
    captured = []

    monkeypatch.setattr(
        "slime.backends.megatron_utils.update_weight.hf_weight_iterator_direct.named_params_and_buffers",
        lambda _args, _model: iter([("module.module.decoder.final_layernorm.weight", dense), (local_name, local_expert)]),
    )
    module = "slime.backends.megatron_utils.update_weight.hf_weight_iterator_direct"
    monkeypatch.setattr(f"{module}.mpu.get_pipeline_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(f"{module}.mpu.get_expert_model_parallel_world_size", lambda: 2)
    monkeypatch.setattr(f"{module}.mpu.get_expert_model_parallel_group", lambda: "ep")
    monkeypatch.setattr(f"{module}.dist.get_rank", lambda: 0)
    monkeypatch.setattr(f"{module}.dist.get_world_size", lambda: 2)
    monkeypatch.setattr(f"{module}.get_gloo_group", lambda: "gloo")

    remote = ParamInfo(remote_name, torch.float32, torch.Size([2, 2]), {}, 16, 1)

    def all_gather_object(obj, object_list, group):
        captured.append((obj, group))
        if group == "ep":
            rank, infos = obj
            object_list[:] = [(rank, infos), (1, {remote_name: remote})]
        else:
            object_list[:] = [obj, obj]

    monkeypatch.setattr(f"{module}.dist.all_gather_object", all_gather_object)
    infos = _get_megatron_local_param_infos(SimpleNamespace(), [object()])

    ep_payload, ep_group = captured[0]
    assert ep_group == "ep"
    assert set(ep_payload[1]) == {local_name}
    assert {info.name for info in infos} == {
        "module.module.decoder.final_layernorm.weight",
        local_name,
        remote_name,
    }
    assert next(info.src_rank for info in infos if info.name == remote_name) == 1


def test_ep_metadata_exchange_rejects_duplicate_expert_owners(monkeypatch):
    name = "module.module.decoder.layers.3.mlp.experts.linear_fc1.weight0"
    parameter = torch.nn.Parameter(torch.ones(2, 2))
    module = "slime.backends.megatron_utils.update_weight.hf_weight_iterator_direct"
    monkeypatch.setattr(f"{module}.named_params_and_buffers", lambda _args, _model: iter([(name, parameter)]))
    monkeypatch.setattr(f"{module}.mpu.get_pipeline_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(f"{module}.mpu.get_expert_model_parallel_world_size", lambda: 2)
    monkeypatch.setattr(f"{module}.mpu.get_expert_model_parallel_group", lambda: "ep")
    monkeypatch.setattr(f"{module}.dist.get_rank", lambda: 0)

    duplicate = ParamInfo(
        name,
        torch.float32,
        torch.Size([2, 2]),
        {
            "tensor_model_parallel": False,
            "partition_dim": -1,
            "partition_stride": 1,
            "parallel_mode": None,
        },
        16,
        1,
    )

    def all_gather_object(obj, object_list, group):
        object_list[:] = [obj, (1, {name: duplicate})]

    monkeypatch.setattr(f"{module}.dist.all_gather_object", all_gather_object)
    with pytest.raises(RuntimeError, match="multiple physical owners"):
        _get_megatron_local_param_infos(SimpleNamespace(), [object()])


def test_real_gloo_ep2_metadata_contract(tmp_path):
    init_method = f"file://{tmp_path / 'ep2-init'}"
    mp.spawn(_run_ep2_metadata_contract, args=(init_method,), nprocs=2, join=True)
