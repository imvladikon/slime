import pytest
import torch

from slime.backends.megatron_utils.update_weight.hf_weight_iterator_direct import (
    _merge_param_info,
    _param_info_fingerprint,
    _validate_param_infos_consistent,
)
from slime.utils.types import ParamInfo

NUM_GPUS = 0


def _infos():
    return [
        ParamInfo("a", torch.bfloat16, torch.Size([2, 3]), {}, 12, 0),
        ParamInfo("b", torch.float32, torch.Size([5]), {}, 20, 1),
    ]


def test_param_info_fingerprint_covers_order_shape_and_dtype():
    infos = _infos()
    baseline = _param_info_fingerprint(infos)

    assert baseline[0] == 2
    assert len(baseline[1]) == 64
    assert _param_info_fingerprint(list(reversed(infos))) != baseline
    assert _param_info_fingerprint([infos[0], ParamInfo("b", torch.float32, torch.Size([6]), {}, 24, 1)]) != baseline
    assert _param_info_fingerprint([infos[0], ParamInfo("b", torch.bfloat16, torch.Size([5]), {}, 10, 1)]) != baseline
    assert _param_info_fingerprint([infos[0], ParamInfo("b", torch.float32, torch.Size([5]), {}, 24, 1)]) != baseline
    assert (
        _param_info_fingerprint(
            [infos[0], ParamInfo("b", torch.float32, torch.Size([5]), {"partition_dim": 0}, 20, 1)]
        )
        != baseline
    )
    assert _param_info_fingerprint([infos[0], ParamInfo("b", torch.float32, torch.Size([5]), {}, 20, 99)]) == baseline


def test_param_info_world_validation_gathers_only_constant_size_fingerprints(monkeypatch):
    captured = {}

    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 3)

    def all_gather_object(obj, object_list, group):
        captured["object"] = obj
        captured["group"] = group
        object_list[:] = [obj, obj, obj]

    monkeypatch.setattr(torch.distributed, "all_gather_object", all_gather_object)
    monkeypatch.setattr(
        "slime.backends.megatron_utils.update_weight.hf_weight_iterator_direct.get_gloo_group",
        lambda: "gloo",
    )

    _validate_param_infos_consistent(_infos())

    assert captured == {"object": _param_info_fingerprint(_infos()), "group": "gloo"}


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("tensor_model_parallel", True),
        ("partition_dim", 1),
        ("partition_stride", 2),
        ("parallel_mode", "duplicated"),
    ],
)
def test_param_info_fingerprint_covers_every_parallel_attribute(field, value):
    baseline = ParamInfo("weight", torch.bfloat16, torch.Size([2, 3]), {}, 12, 0)
    changed = ParamInfo("weight", torch.bfloat16, torch.Size([2, 3]), {field: value}, 12, 0)
    assert _param_info_fingerprint([changed]) != _param_info_fingerprint([baseline])


def test_param_info_merge_rejects_mismatch_and_uses_minimum_pp_owner():
    first = ParamInfo("weight", torch.bfloat16, torch.Size([2, 3]), {}, 12, 7)
    earlier = ParamInfo("weight", torch.bfloat16, torch.Size([2, 3]), {}, 12, 3)
    infos = {first.name: first}

    _merge_param_info(infos, earlier, reject_distinct_owner=False)
    assert infos["weight"].src_rank == 3

    mismatch = ParamInfo("weight", torch.bfloat16, torch.Size([3, 2]), {}, 12, 3)
    with pytest.raises(RuntimeError, match="metadata mismatch"):
        _merge_param_info(infos, mismatch, reject_distinct_owner=False)


def test_param_info_merge_rejects_duplicate_ep_owner():
    first = ParamInfo("expert", torch.bfloat16, torch.Size([2, 3]), {}, 12, 0)
    duplicate = ParamInfo("expert", torch.bfloat16, torch.Size([2, 3]), {}, 12, 1)
    with pytest.raises(RuntimeError, match="multiple physical owners"):
        _merge_param_info({first.name: first}, duplicate, reject_distinct_owner=True)
