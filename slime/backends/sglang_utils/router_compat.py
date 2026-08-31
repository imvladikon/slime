"""Compatibility for routed-expert metadata in older SGLang router wheels."""

import asyncio

import aiohttp
import pybase64
from fastapi.responses import ORJSONResponse


def _merge_routed_experts(prefill: dict, decode: dict) -> bool:
    if "routed_experts" not in prefill or "routed_experts" not in decode:
        return False

    prefill_bytes = pybase64.b64decode(prefill["routed_experts"], validate=True)
    decode_bytes = pybase64.b64decode(decode["routed_experts"], validate=True)
    decode["routed_experts"] = pybase64.b64encode(prefill_bytes + decode_bytes[len(prefill_bytes) :]).decode()
    return True


def _merge_prefill_json(prefill_json: dict, decode_json: dict) -> None:
    if "meta_info" in prefill_json and "meta_info" in decode_json:
        prefill_meta = prefill_json["meta_info"]
        decode_meta = decode_json["meta_info"]
        if "input_token_logprobs" in prefill_meta and "input_token_logprobs" in decode_meta:
            decode_meta["input_token_logprobs"] = prefill_meta["input_token_logprobs"] + decode_meta["input_token_logprobs"]
        _merge_routed_experts(prefill_meta, decode_meta)

    if "sglext" in prefill_json and "sglext" in decode_json:
        _merge_routed_experts(prefill_json["sglext"], decode_json["sglext"])


async def _generate_with_r3(self, modified_request: dict, prefill_server: str, decode_server: str, endpoint: str) -> ORJSONResponse:
    assert not endpoint.startswith("/"), f"Endpoint should not start with '/': {endpoint}"

    expected_decode_dp_rank = None
    if self.test_external_dp_routing:
        await self._ensure_dp_sizes()
        prefill_req, decode_req, expected_decode_dp_rank = self._fork_dp_requests(modified_request)
    else:
        prefill_req = modified_request
        decode_req = modified_request

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
        prefill_response, decode_response = await asyncio.gather(
            session.post(f"{prefill_server}/{endpoint}", json=prefill_req),
            session.post(f"{decode_server}/{endpoint}", json=decode_req),
        )
        if "return_logprob" in modified_request or "return_routed_experts" in modified_request:
            prefill_json = await prefill_response.json()
            ret_json = await decode_response.json()
            _merge_prefill_json(prefill_json, ret_json)
        else:
            ret_json = await decode_response.json()

        if expected_decode_dp_rank is not None:
            actual = ret_json.get("meta_info", {}).get("dp_rank")
            if actual != expected_decode_dp_rank:
                return ORJSONResponse(
                    content={"error": (f"DP rank mismatch: expected {expected_decode_dp_rank}, got {actual}")},
                    status_code=500,
                )

        return ORJSONResponse(content=ret_json, status_code=decode_response.status)


def patch_sglang_router_r3() -> bool:
    """Patch only router wheels predating native PD routed-expert merging."""
    from sglang_router import mini_lb

    if hasattr(mini_lb, "_merge_prefill_json"):
        return False
    mini_lb.MiniLoadBalancer.generate = _generate_with_r3
    return True
