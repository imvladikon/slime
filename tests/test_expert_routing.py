from argparse import Namespace
from dataclasses import replace

import pytest
import torch

from slime.backends.megatron_utils.update_weight.expert_routing import (
    _build_expert_transfer_plan,
    _expert_transfer_size,
    _ExpertParam,
    _set_expert_source_ranks,
    configure_expert_routing,
)
from slime.backends.megatron_utils.update_weight import hf_weight_iterator_direct
from slime.utils.types import ParamInfo

NUM_GPUS = 0


def _param(*, expert: int, projection: int, source_rank: int, target_rank: int, size: int) -> _ExpertParam:
    info = ParamInfo(
        name=f"module.module.decoder.layers.3.mlp.experts.linear_fc{projection}.weight{expert}",
        dtype=torch.bfloat16,
        shape=torch.Size([size // 2]),
        attrs={},
        size=size,
        src_rank=source_rank,
    )
    return _ExpertParam(
        info=info,
        layer=3,
        expert=expert,
        target_ranks=(target_rank,),
    )


def test_transfer_plan_splits_same_rank_experts_at_expert_boundaries():
    params = []
    for expert, rank in ((0, 0), (1, 0), (2, 1), (3, 1)):
        params.extend(
            [
                _param(expert=expert, projection=1, source_rank=rank, target_rank=rank, size=60),
                _param(expert=expert, projection=2, source_rank=rank, target_rank=rank, size=60),
            ]
        )

    plan = _build_expert_transfer_plan(params, buffer_size=150)

    assert len(plan) == 1
    assert len(plan[0]) == 2
    transfers = [transfer for batch in plan[0] for transfer in batch]
    assert len(transfers) == 4
    assert all(_expert_transfer_size(transfer) == 120 for transfer in transfers)
    assert all(len({param.expert for param in transfer.params}) == 1 for transfer in transfers)
    assert {(param.expert, param.info.name) for transfer in transfers for param in transfer.params} == {
        (param.expert, param.info.name) for param in params
    }


def test_transfer_plan_rejects_one_expert_larger_than_buffer():
    first = _param(expert=0, projection=1, source_rank=0, target_rank=0, size=80)
    second = replace(
        first,
        info=replace(
            first.info,
            name="module.module.decoder.layers.3.mlp.experts.linear_fc2.weight0",
            size=80,
        ),
    )

    with pytest.raises(ValueError, match="exceeds update_weight_buffer_size"):
        _build_expert_transfer_plan([first, second], buffer_size=150)


def test_required_rank_local_expert_update_rejects_fallback():
    with pytest.raises(RuntimeError, match="parameter metadata is unavailable"):
        configure_expert_routing(
            args=Namespace(require_rank_local_expert_update=True),
            full_param_info_buckets=None,
            get_local_weight_names=lambda: (),
            engine_gpu_counts=(),
            engine_gpu_offsets=(),
            engine_parallel_configs=None,
            use_distribute=False,
        )


def test_optional_rank_local_expert_update_keeps_legacy_fallback():
    assert configure_expert_routing(
        args=Namespace(require_rank_local_expert_update=False),
        full_param_info_buckets=None,
        get_local_weight_names=lambda: (),
        engine_gpu_counts=(),
        engine_gpu_offsets=(),
        engine_parallel_configs=None,
        use_distribute=False,
    ) == (None, [])


def test_replicated_expert_sources_are_byte_balanced():
    infos = [
        _param(expert=expert, projection=1, source_rank=0, target_rank=0, size=64).info
        for expert in range(4)
    ]
    names = [info.name for info in infos]

    selected = _set_expert_source_ranks(infos, [names, names])

    assert [info.src_rank for info in selected] == [0, 1, 0, 1]


def test_live_weight_buckets_keep_glm53_qkv_pair_atomic(monkeypatch):
    monkeypatch.setattr(
        hf_weight_iterator_direct.mpu,
        "get_tensor_model_parallel_world_size",
        lambda: 1,
    )
    monkeypatch.setattr(
        hf_weight_iterator_direct.mpu,
        "get_expert_tensor_parallel_world_size",
        lambda: 1,
    )
    names = (
        "module.module.decoder.layers.3.self_attention.linear_q_down_proj.weight",
        "module.module.decoder.layers.3.input_layernorm.weight",
        "module.module.decoder.layers.3.self_attention.linear_kv_down_proj.weight",
    )
    infos = [
        ParamInfo(
            name=name,
            dtype=torch.bfloat16,
            shape=torch.Size([size // 2]),
            attrs={},
            size=size,
            src_rank=0,
        )
        for name, size in zip(names, (60, 20, 60), strict=True)
    ]

    buckets = hf_weight_iterator_direct.pack_param_info_buckets(infos, 100)

    assert [[info.name for info in bucket] for bucket in buckets] == [
        [names[0], names[2]],
        [names[1]],
    ]


def test_live_weight_buckets_keep_prefixed_glm53_qkv_pair_atomic(monkeypatch):
    monkeypatch.setattr(
        hf_weight_iterator_direct.mpu,
        "get_tensor_model_parallel_world_size",
        lambda: 1,
    )
    monkeypatch.setattr(
        hf_weight_iterator_direct.mpu,
        "get_expert_tensor_parallel_world_size",
        lambda: 1,
    )
    prefix = "module.module.language_model.decoder.layers.3.self_attention."
    names = (
        prefix + "linear_q_down_proj.weight",
        prefix + "linear_kv_down_proj.weight",
    )
    infos = [
        ParamInfo(
            name=name,
            dtype=torch.bfloat16,
            shape=torch.Size([30]),
            attrs={},
            size=60,
            src_rank=0,
        )
        for name in names
    ]

    assert hf_weight_iterator_direct.pack_param_info_buckets(infos, 100) == [infos]


def test_live_weight_buckets_keep_glm53_mhc_scale_atomic(monkeypatch):
    monkeypatch.setattr(
        hf_weight_iterator_direct.mpu,
        "get_tensor_model_parallel_world_size",
        lambda: 1,
    )
    monkeypatch.setattr(
        hf_weight_iterator_direct.mpu,
        "get_expert_tensor_parallel_world_size",
        lambda: 1,
    )
    names = tuple(
        f"module.module.decoder.layers.3.mlp_hyper_connection.alpha_{suffix}"
        for suffix in ("pre", "post", "res")
    )
    infos = [
        ParamInfo(
            name=name,
            dtype=torch.float32,
            shape=torch.Size([4]),
            attrs={},
            size=16,
            src_rank=0,
        )
        for name in names
    ]

    buckets = hf_weight_iterator_direct.pack_param_info_buckets(infos, 20)

    assert len(buckets) == 1
    assert [info.name for info in buckets[0]] == list(names)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
