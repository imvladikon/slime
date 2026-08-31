import struct

import pybase64

from slime.backends.sglang_utils.router_compat import (
    _merge_prefill_json,
    _merge_routed_experts,
    _generate_with_r3,
    patch_sglang_router_r3,
)


def _encode(values):
    return pybase64.b64encode(struct.pack(f"<{len(values)}i", *values)).decode()


def _decode(value):
    raw = pybase64.b64decode(value, validate=True)
    return struct.unpack(f"<{len(raw) // 4}i", raw)


def test_merge_prefill_json_preserves_complete_r3_and_logprob_metadata():
    prefill = {
        "meta_info": {
            "input_token_logprobs": [[-1.0, 1, None]],
            "routed_experts": _encode([1, 2]),
        },
        "sglext": {"routed_experts": _encode([5, 6])},
    }
    decode = {
        "meta_info": {
            "input_token_logprobs": [[-2.0, 2, None]],
            "routed_experts": _encode([9, 9, 3, 4]),
        },
        "sglext": {"routed_experts": _encode([9, 9, 7, 8])},
    }

    _merge_prefill_json(prefill, decode)

    assert decode["meta_info"]["input_token_logprobs"] == [
        [-1.0, 1, None],
        [-2.0, 2, None],
    ]
    assert _decode(decode["meta_info"]["routed_experts"]) == (1, 2, 3, 4)
    assert _decode(decode["sglext"]["routed_experts"]) == (5, 6, 7, 8)


def test_merge_routed_experts_requires_both_halves():
    decode = {"routed_experts": _encode([3, 4])}

    assert not _merge_routed_experts({}, decode)
    assert _decode(decode["routed_experts"]) == (3, 4)


def test_old_router_generate_is_patched_once(monkeypatch):
    from sglang_router import mini_lb

    monkeypatch.delattr(mini_lb, "_merge_prefill_json", raising=False)
    monkeypatch.setattr(mini_lb.MiniLoadBalancer, "generate", object())

    assert patch_sglang_router_r3()
    assert mini_lb.MiniLoadBalancer.generate is _generate_with_r3
