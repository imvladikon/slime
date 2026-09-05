#!/usr/bin/env python3
"""Metadata-only GLM-5.3-Flash production topology and memory preflight.

This tool never opens a safetensors payload.  It consumes the released config,
weight index, and an aggregate cache of safetensors JSON headers, then computes
per-rank persistent-state floors for the Megatron actor and SGLang rollout.
Runtime activation and kernel-workspace reserves remain explicit inputs because
they cannot be proven from checkpoint metadata.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import struct
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import requests

FULL_REPOSITORY = "zai-org/GLM-5.3-Flash"
FULL_REVISION = "04c4e9e95c5da8862dced7e5056455116f83a7e0"
EXPECTED_TENSORS = 76_108
EXPECTED_SHARDS = 62
EXPECTED_NON_SCALE_PARAMETERS = 321_323_031_390
EXPECTED_NATIVE_BYTES = 328_326_771_576
SOURCE_CONFIG_SHA256 = "bb8f01c42cb92a52ca72e65afb4d5bd8d11aef083cd210e8de25dfb904f23e9f"
SOURCE_INDEX_SHA256 = "3c3f40366a53c3fd7974b4eab7881a365a98c2a4329150befebab99fe7c18b05"
SOURCE_HEADERS_SHA256 = "b01c2e4b5ed1595b7ad2cbcc9b70800a1f0009651f8581fa342c36ed41cdd813"
MAX_SINGLE_HEADER_BYTES = 64 * 1024 * 1024
MAX_AGGREGATE_HEADER_BYTES = 256 * 1024 * 1024
EXPECTED_TRAINING = {
    "regular_tp": (8_625_539_200, 17_251_639_808),
    "trainable_replicated": (213_262_974, 426_530_808),
    "routed_experts": (304_405_807_104, 608_811_614_208),
    "frozen_replicated": (82_202_688, 164_429_568),
    "vision": (563_627_008, 1_127_254_016),
}

DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}
LAYER_PATTERN = re.compile(r"\.layers\.(\d+)\.")


@dataclass
class CategoryStats:
    tensors: int = 0
    parameter_numel: int = 0
    training_bytes: int = 0
    checkpoint_bytes: int = 0
    rollout_bytes: int = 0
    training_numel_by_dtype: dict[str, int] = field(
        default_factory=lambda: defaultdict(int)
    )

    def serializable(self) -> dict[str, Any]:
        result = asdict(self)
        result["training_numel_by_dtype"] = dict(
            sorted(self.training_numel_by_dtype.items())
        )
        result["training_gib"] = gib(self.training_bytes)
        result["checkpoint_gib"] = gib(self.checkpoint_bytes)
        result["rollout_gib"] = gib(self.rollout_bytes)
        return result


def gib(value: float) -> float:
    return value / 2**30


def numel(entry: dict[str, Any]) -> int:
    return math.prod(entry["shape"])


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_source_hashes(config: Path, index: Path, headers: Path) -> dict[str, str]:
    actual = {
        "config": sha256(config),
        "index": sha256(index),
        "headers": sha256(headers),
    }
    expected = {
        "config": SOURCE_CONFIG_SHA256,
        "index": SOURCE_INDEX_SHA256,
        "headers": SOURCE_HEADERS_SHA256,
    }
    if actual != expected:
        raise ValueError(f"official metadata hash mismatch: {actual}")
    return actual


def strict_range(url: str, start: int, end: int) -> bytes:
    response = requests.get(
        url,
        headers={"Accept-Encoding": "identity", "Range": f"bytes={start}-{end}"},
        allow_redirects=True,
        stream=True,
        timeout=120,
    )
    try:
        if response.status_code != 206:
            raise RuntimeError(
                f"refusing non-range response {response.status_code} for {url}"
            )
        expected_range = f"bytes {start}-{end}/"
        content_range = response.headers.get("Content-Range", "")
        if not content_range.startswith(expected_range):
            raise RuntimeError(f"unexpected Content-Range {content_range!r}")
        expected_size = end - start + 1
        payload = response.raw.read(expected_size + 1, decode_content=False)
        if len(payload) != expected_size:
            raise RuntimeError(
                f"range returned {len(payload)} bytes, expected {expected_size}"
            )
        return payload
    finally:
        response.close()


def fetch_remote_header(filename: str) -> dict[str, Any]:
    url = f"https://huggingface.co/{FULL_REPOSITORY}/resolve/{FULL_REVISION}/{filename}"
    header_size = struct.unpack("<Q", strict_range(url, 0, 7))[0]
    if header_size > MAX_SINGLE_HEADER_BYTES:
        raise RuntimeError(f"refusing oversized safetensors header in {filename}")
    return json.loads(strict_range(url, 8, 7 + header_size))


def cache_remote_headers(index: dict[str, Any], output: Path) -> dict[str, Any]:
    combined: dict[str, Any] = {}
    aggregate_size = 0
    for filename in sorted(set(index["weight_map"].values())):
        shard_header = fetch_remote_header(filename)
        shard_header.pop("__metadata__", None)
        overlap = set(combined).intersection(shard_header)
        if overlap:
            raise ValueError(f"duplicate tensors across shard headers: {sorted(overlap)[:3]}")
        combined.update(shard_header)
        aggregate_size += len(json.dumps(shard_header, separators=(",", ":")))
        if aggregate_size > MAX_AGGREGATE_HEADER_BYTES:
            raise RuntimeError("refusing oversized aggregate safetensors headers")
    if set(combined) != set(index["weight_map"]):
        raise ValueError("fetched safetensors headers do not match the weight index")
    document = {
        "repo": FULL_REPOSITORY,
        "revision": FULL_REVISION,
        "header": combined,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(document, handle, separators=(",", ":"), sort_keys=True)
    temporary.replace(output)
    return document


def reject_weight_payloads(paths: list[Path]) -> None:
    checked: set[Path] = set()
    payloads: list[Path] = []
    for path in paths:
        directory = path.resolve().parent
        if directory in checked:
            continue
        checked.add(directory)
        payloads.extend(sorted(directory.glob("*.safetensors")))
    if payloads:
        names = ", ".join(str(path) for path in payloads[:3])
        raise ValueError(f"refusing metadata directory containing weight payloads: {names}")


def base_parameter_name(key: str) -> str:
    return key.removesuffix("_scale_inv")


def classify_parameter(key: str) -> str:
    key = base_parameter_name(key)
    if ".mlp.experts." in key:
        return "routed_experts"
    if key.startswith("model.visual."):
        return "vision"
    if ".self_attn.indexer." in key or key.endswith(
        ".mlp.gate.e_score_correction_bias"
    ):
        return "frozen_replicated"

    replicated = (
        ".hc_" in key
        or ".input_layernorm." in key
        or ".post_attention_layernorm." in key
        or key == "model.language_model.norm.weight"
        or key.endswith(".mlp.gate.weight")
        or key.endswith(".mlp.shared_experts.gate.weight")
        or any(
            marker in key
            for marker in (
                ".self_attn.f_a_proj.",
                ".self_attn.g_a_proj.",
                ".self_attn.o_norm.",
                ".self_attn.q_a_proj.",
                ".self_attn.kv_a_proj_with_mqa.",
                ".self_attn.q_a_layernorm.",
                ".self_attn.kv_a_layernorm.",
            )
        )
    )
    return "trainable_replicated" if replicated else "regular_tp"


def training_dtype(key: str, checkpoint_dtype: str) -> str:
    if key.endswith(
        (".hc_attn_base", ".hc_attn_scale", ".hc_ffn_base", ".hc_ffn_scale")
    ):
        return "F32"
    if checkpoint_dtype in {"F8_E4M3", "F8_E5M2"}:
        return "BF16"
    return checkpoint_dtype


def rollout_dtype(key: str, checkpoint_dtype: str) -> str:
    """Return the dtype materialized by the pinned SGLang GLM runtime."""
    if key.endswith(
        (
            ".hc_attn_fn",
            ".hc_ffn_fn",
            ".q_conv1d.weight",
            ".k_conv1d.weight",
            ".v_conv1d.weight",
            ".self_attn.indexer.weights_proj.weight",
            ".self_attn.indexer.k_norm.weight",
            ".self_attn.indexer.k_norm.bias",
            ".self_attn.indexer.index_kpool_compress_ape",
        )
    ):
        return "F32"
    return checkpoint_dtype


def is_scale_metadata(key: str) -> bool:
    return key.endswith(".weight_scale_inv")


def layer_index(key: str) -> int | None:
    match = LAYER_PATTERN.search(key)
    return int(match.group(1)) if match else None


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    text = config["text_config"]
    layer_types = text["layer_types"]
    checks = {
        "model_type": config.get("model_type") == "glm5_next",
        "architecture": config.get("architectures")
        == ["Glm5NextForConditionalGeneration"],
        "text_model_type": text.get("model_type") == "glm5_next_text",
        "layers": text.get("num_hidden_layers") == 45
        and len(layer_types) == 45,
        "attention_schedule": layer_types
        == [
            "deepseek_sparse_attention" if index % 4 == 3 else "linear_attention"
            for index in range(45)
        ],
        "mlp_schedule": text.get("mlp_layer_types")
        == ["dense"] * 3 + ["sparse"] * 42,
        "dense_prefix": text.get("first_k_dense_replace") == 3,
        "experts": text.get("n_routed_experts") == 288
        and text.get("num_experts_per_tok") == 8,
        "mhc": text.get("mhc") is True
        and text.get("hc_mult") == 4
        and text.get("hc_eps") == 1e-6
        and text.get("hc_sinkhorn_iters") == 20,
        "normalization": text.get("rms_norm_eps") == 1e-5,
        "kpool": text.get("index_kpool") == 4
        and text.get("index_topk") == 2048
        and text.get("index_n_heads") == 32
        and text.get("index_head_dim") == 128
        and text.get("index_kpool_compress") is True
        and text.get("index_kpool_always_select_tail") is True,
        "router": text.get("norm_topk_prob") is True
        and text.get("routed_scaling_factor") == 2.5
        and text.get("n_group") == 1
        and text.get("topk_group") == 1
        and text.get("n_shared_experts") == 1,
        "mtp_release": text.get("num_nextn_predict_layers") == 1,
        "core_dimensions": all(
            (
                text.get("hidden_size") == 4096,
                text.get("num_attention_heads") == 64,
                text.get("q_lora_rank") == 1536,
                text.get("kv_lora_rank") == 512,
                text.get("qk_nope_head_dim") == 256,
                text.get("qk_head_dim") == 256,
                text.get("qk_rope_head_dim") == 0,
                text.get("v_head_dim") == 256,
                text.get("intermediate_size") == 12288,
                text.get("moe_intermediate_size") == 2048,
                text.get("vocab_size") == 154880,
            )
        ),
        "vision": config.get("vision_config", {}).get("depth") == 24,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"GLM-5.3-Flash config contract failed: {failed}")
    return checks


def build_inventory(
    config: dict[str, Any],
    index: dict[str, Any],
    header_document: dict[str, Any],
    *,
    strict_official: bool,
) -> tuple[dict[str, CategoryStats], dict[str, Any]]:
    if "header" in header_document:
        if strict_official and (
            header_document.get("repo") != FULL_REPOSITORY
            or header_document.get("revision") != FULL_REVISION
        ):
            raise ValueError("safetensors header cache repository/revision mismatch")
        header = header_document["header"]
    else:
        header = header_document

    weight_map = index["weight_map"]
    if set(weight_map) != set(header):
        missing = sorted(set(weight_map) - set(header))
        extra = sorted(set(header) - set(weight_map))
        raise ValueError(
            f"index/header mismatch: missing={missing[:3]} extra={extra[:3]}"
        )

    num_backbone_layers = config["text_config"]["num_hidden_layers"]
    categories: dict[str, CategoryStats] = defaultdict(CategoryStats)
    total_non_scale_numel = 0
    native_bytes = 0
    mtp_tensors = 0
    mtp_parameters = 0
    largest_training_tensor = (0, "")

    for key, entry in header.items():
        count = numel(entry)
        checkpoint_bytes = count * DTYPE_BYTES[entry["dtype"]]
        native_bytes += checkpoint_bytes
        scale = is_scale_metadata(key)
        if not scale:
            total_non_scale_numel += count

        current_layer = layer_index(key)
        if current_layer is not None and current_layer >= num_backbone_layers:
            mtp_tensors += 1
            if not scale:
                mtp_parameters += count
            continue

        category = classify_parameter(key)
        stats = categories[category]
        stats.tensors += 1
        stats.checkpoint_bytes += checkpoint_bytes
        rollout_type = rollout_dtype(key, entry["dtype"])
        stats.rollout_bytes += count * DTYPE_BYTES[rollout_type]
        if scale:
            continue

        dtype = training_dtype(key, entry["dtype"])
        if dtype not in {"BF16", "F32"}:
            raise ValueError(f"unsupported training dtype {dtype} for {key}")
        train_bytes = count * DTYPE_BYTES[dtype]
        stats.parameter_numel += count
        stats.training_bytes += train_bytes
        stats.training_numel_by_dtype[dtype] += count
        if category != "vision" and train_bytes > largest_training_tensor[0]:
            largest_training_tensor = (train_bytes, key)

    summary = {
        "repository": FULL_REPOSITORY,
        "revision": FULL_REVISION,
        "tensor_count": len(header),
        "shard_count": len(set(weight_map.values())),
        "non_scale_parameters": total_non_scale_numel,
        "native_checkpoint_bytes": native_bytes,
        "native_checkpoint_gib": gib(native_bytes),
        "mtp_tensors": mtp_tensors,
        "mtp_parameters": mtp_parameters,
        "largest_training_tensor": {
            "name": largest_training_tensor[1],
            "bytes": largest_training_tensor[0],
            "gib": gib(largest_training_tensor[0]),
        },
    }

    if strict_official:
        exact = {
            "tensor_count": EXPECTED_TENSORS,
            "shard_count": EXPECTED_SHARDS,
            "non_scale_parameters": EXPECTED_NON_SCALE_PARAMETERS,
            "native_checkpoint_bytes": EXPECTED_NATIVE_BYTES,
        }
        mismatches = {
            key: (summary[key], expected)
            for key, expected in exact.items()
            if summary[key] != expected
        }
        for category, (expected_numel, expected_bytes) in EXPECTED_TRAINING.items():
            actual = categories[category]
            if (
                actual.parameter_numel != expected_numel
                or actual.training_bytes != expected_bytes
            ):
                mismatches[category] = (
                    (actual.parameter_numel, actual.training_bytes),
                    (expected_numel, expected_bytes),
                )
        if mismatches:
            raise ValueError(f"official state contract mismatch: {mismatches}")

    return categories, summary


def optimizer_bytes(
    stats: CategoryStats, bf16_bytes: int, fp32_bytes: int
) -> int:
    return (
        stats.training_numel_by_dtype.get("BF16", 0) * bf16_bytes
        + stats.training_numel_by_dtype.get("F32", 0) * fp32_bytes
    )


def add_gate(gates: list[dict[str, str]], severity: str, message: str) -> None:
    gates.append({"severity": severity, "message": message})


def calculate(
    args: argparse.Namespace,
    config: dict[str, Any],
    categories: dict[str, CategoryStats],
    inventory: dict[str, Any],
) -> dict[str, Any]:
    gates: list[dict[str, str]] = []
    text = config["text_config"]
    world_size = args.tp * args.dp * args.pp * args.cp

    if args.pp != 1:
        add_gate(gates, "error", "current mHC layer ownership requires PP=1")
    if args.cp != 1:
        add_gate(gates, "error", "current KPool DSA implementation requires CP=1")
    if text["num_attention_heads"] % args.tp:
        add_gate(gates, "error", "attention heads must be divisible by actor TP")
    num_experts = text["n_routed_experts"]
    if num_experts % args.ep:
        add_gate(gates, "error", "288 routed experts must be divisible by actor EP")
    expert_grid = args.etp * args.ep * args.pp
    if world_size % expert_grid:
        add_gate(gates, "error", "actor world size must be divisible by ETP*EP*PP")
    expert_dp = world_size // expert_grid if world_size % expert_grid == 0 else 0
    if text["moe_intermediate_size"] % args.etp:
        add_gate(gates, "error", "MoE intermediate size must be divisible by actor ETP")

    if args.mode == "rl" and args.etp != 1:
        add_gate(
            gates,
            "error",
            "current rank-local Slime rollout synchronization requires actor ETP=1",
        )
    if args.mode == "rl" and args.colocate and args.offload_train:
        add_gate(
            gates,
            "error",
            "colocated CUDA IPC is not qualified with torch-memory-saver train offload",
        )
    if args.mode == "rl" and args.offload_rollout and not args.rollout_weights_cpu_backup:
        add_gate(
            gates,
            "error",
            "rollout offload without a main-weight CPU backup discards the frozen vision tower",
        )
    if (
        args.mode == "rl"
        and not args.colocate
        and not args.allow_unsharded_remote_sync
    ):
        add_gate(
            gates,
            "error",
            "non-colocated full-model sync broadcasts unsharded tensors to rollout ranks",
        )

    regular = categories["regular_tp"]
    replicated = categories["trainable_replicated"]
    routed = categories["routed_experts"]
    frozen = categories["frozen_replicated"]
    vision = categories["vision"]

    actor_language_parameter_bytes = (
        regular.training_bytes / args.tp
        + replicated.training_bytes
        + routed.training_bytes / (args.ep * args.etp)
        + frozen.training_bytes
    )
    actor_parameter_bytes = actor_language_parameter_bytes + vision.training_bytes
    actor_gradient_bytes = 4 * (
        regular.parameter_numel / args.tp
        + replicated.parameter_numel
        + routed.parameter_numel / (args.ep * args.etp)
    )

    dense_optimizer_shards = args.dp if args.distributed_optimizer else 1
    expert_optimizer_shards = expert_dp if args.distributed_optimizer else 1
    actor_optimizer_bytes = (
        optimizer_bytes(regular, args.bf16_optimizer_bytes, args.fp32_optimizer_bytes)
        / args.tp
        / dense_optimizer_shards
        + optimizer_bytes(
            replicated, args.bf16_optimizer_bytes, args.fp32_optimizer_bytes
        )
        / dense_optimizer_shards
        + optimizer_bytes(routed, args.bf16_optimizer_bytes, args.fp32_optimizer_bytes)
        / (args.ep * args.etp)
        / max(expert_optimizer_shards, 1)
    )
    optimizer_gpu_bytes = 0 if args.optimizer_offload else actor_optimizer_bytes
    optimizer_cpu_bytes = actor_optimizer_bytes if args.optimizer_offload else 0
    actor_persistent_bytes = (
        actor_parameter_bytes + actor_gradient_bytes + optimizer_gpu_bytes
    )

    actor = {
        "world_size": world_size,
        "dense_data_parallel_size": args.dp,
        "expert_data_parallel_size": expert_dp,
        "parameter_gib_per_rank": gib(actor_parameter_bytes),
        "language_parameter_gib_per_rank": gib(actor_language_parameter_bytes),
        "frozen_vision_gib_per_rank": gib(vision.training_bytes),
        "fp32_gradient_gib_per_rank": gib(actor_gradient_bytes),
        "optimizer_gpu_gib_per_rank": gib(optimizer_gpu_bytes),
        "optimizer_cpu_gib_per_rank": gib(optimizer_cpu_bytes),
        "persistent_gpu_floor_gib_per_rank": gib(actor_persistent_bytes),
    }

    rollout_model_bytes = 0.0
    rollout_static_bytes = 0.0
    rollout_replicas = 0
    rollout = None
    if args.mode == "rl":
        if args.sglang_moe_a2a_backend != "deepep":
            add_gate(gates, "error", "full-scale rollout requires the DeepEP MoE A2A backend")
        if args.sglang_deepep_mode not in {"auto", "low_latency"}:
            add_gate(gates, "error", "full-scale rollout requires DeepEP auto or low_latency mode")
        if not args.use_rollout_routing_replay:
            add_gate(
                gates,
                "error",
                "the all-to-all GLM-5.3-Flash actor requires a qualified R3 route-replay path",
            )
        if text["num_attention_heads"] % args.rollout_tp:
            add_gate(gates, "error", "attention heads must be divisible by rollout TP")
        if num_experts % args.rollout_ep:
            add_gate(gates, "error", "288 routed experts must be divisible by rollout EP")
        if args.rollout_gpus_per_engine % args.rollout_tp:
            add_gate(gates, "error", "rollout GPUs per engine must be divisible by rollout TP")
        if args.rollout_gpus_per_engine % args.rollout_ep:
            add_gate(gates, "error", "rollout GPUs per engine must be divisible by rollout EP")
        if args.rollout_tp != args.rollout_gpus_per_engine:
            add_gate(
                gates,
                "error",
                "Slime PP=1 derives rollout TP from rollout GPUs per engine",
            )
        if args.rollout_gpus_per_engine != args.rollout_ep * args.rollout_moe_dp:
            add_gate(
                gates,
                "error",
                "rollout GPUs per engine must equal SGLang EP times MoE-DP",
            )
        if args.rollout_ep <= 1:
            add_gate(gates, "error", "full-scale RL requires sharded SGLang expert routing (EP>1)")
        if args.sglang_eplb or args.sglang_redundant_experts or args.sglang_elastic_ep:
            add_gate(
                gates,
                "error",
                "the qualified expert-routing seam requires EPLB, redundant experts, and elastic EP disabled",
            )
        rollout_model_bytes = (
            regular.rollout_bytes / args.rollout_tp
            + replicated.rollout_bytes
            + frozen.rollout_bytes
            + routed.rollout_bytes / args.rollout_ep
            + vision.rollout_bytes
        )
        gpu_capacity_bytes = args.gpu_memory_gib * 2**30
        # SGLang applies mem_fraction_static to the memory visible before it
        # loads model weights.  In the colocated lane the resident actor has
        # already consumed its persistent parameter/gradient/optimizer floor.
        rollout_pre_model_available_bytes = gpu_capacity_bytes
        if args.colocate:
            rollout_pre_model_available_bytes -= actor_persistent_bytes
        if rollout_pre_model_available_bytes <= 0:
            add_gate(
                gates,
                "error",
                "the actor persistent floor leaves no memory for the colocated rollout",
            )
        rollout_static_bytes = max(rollout_pre_model_available_bytes, 0) * args.sglang_mem_fraction_static
        if rollout_static_bytes < rollout_model_bytes:
            add_gate(
                gates,
                "error",
                "SGLang static memory allocation is smaller than the rollout model floor",
            )
        rollout_static_headroom_bytes = rollout_static_bytes - rollout_model_bytes
        requested_cache_reserve_bytes = args.rollout_cache_reserve_gib * 2**30
        if rollout_static_headroom_bytes < requested_cache_reserve_bytes:
            add_gate(
                gates,
                "error",
                "SGLang static memory allocation leaves less KV/KDA cache headroom than requested",
            )
        rollout_runtime_slack_bytes = (
            rollout_pre_model_available_bytes - rollout_static_bytes
        )
        requested_rollout_runtime_reserve_bytes = (
            args.rollout_runtime_reserve_gib * 2**30
        )
        if rollout_runtime_slack_bytes < requested_rollout_runtime_reserve_bytes:
            add_gate(
                gates,
                "error",
                "SGLang mem_fraction_static leaves less activation/workspace slack than requested",
            )
        if args.rollout_num_gpus % args.rollout_gpus_per_engine:
            add_gate(
                gates,
                "error",
                "total rollout GPUs must be divisible by rollout GPUs per engine",
            )
        if args.colocate:
            if args.rollout_num_gpus != world_size:
                add_gate(
                    gates,
                    "error",
                    "colocated full-scale lane requires total rollout GPUs to equal actor world size",
                )
            rollout_replicas = args.rollout_num_gpus // args.rollout_gpus_per_engine
        else:
            rollout_replicas = args.rollout_replicas
        rollout = {
            "tp": args.rollout_tp,
            "ep": args.rollout_ep,
            "moe_dp": args.rollout_moe_dp,
            "total_gpus": args.rollout_num_gpus,
            "gpus_per_engine": args.rollout_gpus_per_engine,
            "replicas": rollout_replicas,
            "model_floor_gib_per_rank": gib(rollout_model_bytes),
            "pre_model_available_gib_per_rank": gib(rollout_pre_model_available_bytes),
            "mem_fraction_static": args.sglang_mem_fraction_static,
            "static_allocation_gib_per_rank": gib(rollout_static_bytes),
            "cache_headroom_gib_per_rank": gib(rollout_static_headroom_bytes),
            "cache_reserve_required_gib_per_rank": args.rollout_cache_reserve_gib,
            "runtime_slack_gib_per_rank": gib(rollout_runtime_slack_bytes),
            "runtime_reserve_required_gib_per_rank": args.rollout_runtime_reserve_gib,
            "moe_a2a_backend": args.sglang_moe_a2a_backend,
            "deepep_mode": args.sglang_deepep_mode,
            "r3": {
                "enabled": args.use_rollout_routing_replay,
                "bytes_per_generated_token": (
                    text["num_hidden_layers"] * text["num_experts_per_tok"] * 4
                    if args.use_rollout_routing_replay
                    else 0
                ),
                "raw_upper_gib_per_rollout": (
                    args.rollout_batch_size
                    * args.n_samples_per_prompt
                    * args.rollout_max_response_len
                    * text["num_hidden_layers"]
                    * text["num_experts_per_tok"]
                    * 4
                    / 2**30
                    if args.use_rollout_routing_replay
                    else 0
                ),
            },
            "mtp_loaded": False,
            "vision_assumption": "fully replicated upper bound",
        }

    actor_train_peak = actor_persistent_bytes + args.actor_runtime_reserve_gib * 2**30
    rollout_peak = rollout_static_bytes + args.rollout_runtime_reserve_gib * 2**30
    dense_sync_transient_bytes = 2 * inventory["largest_training_tensor"]["bytes"]
    expert_sync_transient_bytes = (
        2 * args.update_weight_buffer_bytes if args.mode == "rl" else 0
    )
    sync_transient_bytes = max(
        dense_sync_transient_bytes,
        expert_sync_transient_bytes,
    )
    sync_peak = actor_persistent_bytes + sync_transient_bytes
    if args.mode == "rl" and args.colocate:
        # SGLang reserves its static model/KV pool while the colocated actor is
        # resident, including during the actor's activation peak.
        actor_train_peak += rollout_static_bytes
        rollout_peak += actor_persistent_bytes
        sync_peak += rollout_static_bytes

    projected_peak = max(actor_train_peak, rollout_peak, sync_peak)
    usable_gpu_bytes = args.gpu_memory_gib * args.memory_safety_fraction * 2**30
    if projected_peak > usable_gpu_bytes:
        add_gate(
            gates,
            "error",
            f"projected phase peak {gib(projected_peak):.2f} GiB exceeds the "
            f"{gib(usable_gpu_bytes):.2f} GiB safety budget",
        )

    actor_backup_per_node = (
        actor_language_parameter_bytes * args.backup_tags + optimizer_cpu_bytes
    ) * args.gpus_per_node
    vision_loader_staging_per_node = vision.training_bytes * args.gpus_per_node
    rollout_backup_per_node = (
        rollout_model_bytes * args.gpus_per_node
        if args.mode == "rl" and args.rollout_weights_cpu_backup
        else 0
    )
    host_pinned_per_node = (
        actor_backup_per_node + vision_loader_staging_per_node + rollout_backup_per_node
    )
    if args.host_memory_gib <= 0:
        add_gate(
            gates,
            "error",
            "an explicit positive host-memory budget is required for production preflight",
        )
    elif host_pinned_per_node > args.host_memory_gib * args.memory_safety_fraction * 2**30:
        add_gate(
            gates,
            "error",
            "actor backup, optimizer, vision staging, and rollout backup exceed the host-memory safety budget",
        )

    add_gate(
        gates,
        "warning",
        "the runtime reserves must be replaced by measured activation and kernel peaks on target hardware",
    )
    add_gate(
        gates,
        "warning",
        "MTP is intentionally omitted; speculative rollout must remain disabled",
    )
    if args.mode == "rl" and args.colocate:
        add_gate(
            gates,
            "warning",
            "the colocated estimate conservatively keeps actor and rollout weights resident together",
        )
        add_gate(
            gates,
            "warning",
            "target runtime must verify balanced rank-local expert source bytes and per-node egress",
        )

    source_language_bytes = (
        regular.training_bytes
        + replicated.training_bytes
        + routed.training_bytes
        + frozen.training_bytes
    )
    target_language_bytes = (
        regular.rollout_bytes
        + replicated.rollout_bytes
        + routed.rollout_bytes
        + frozen.rollout_bytes
    )
    target_write_per_replica = 0
    if args.mode == "rl":
        target_write_per_replica = (
            regular.rollout_bytes
            + routed.rollout_bytes * args.rollout_moe_dp
            + (replicated.rollout_bytes + frozen.rollout_bytes)
            * args.rollout_gpus_per_engine
        )
    aggregate_target_bytes = target_write_per_replica * rollout_replicas
    expert_target_copies = (
        rollout_replicas * args.rollout_moe_dp if args.mode == "rl" else 0
    )
    num_moe_layers = sum(
        layer_type == "sparse" for layer_type in text["mlp_layer_types"]
    )
    expert_target_bytes_per_layer = (
        routed.training_bytes / num_moe_layers / args.rollout_ep
        if args.mode == "rl" and num_moe_layers
        else 0
    )
    minimum_expert_batches_per_layer = (
        math.ceil(expert_target_bytes_per_layer / args.update_weight_buffer_bytes)
        if expert_target_bytes_per_layer
        else 0
    )
    expert_bf16_p2p_upper_bytes = routed.training_bytes * expert_target_copies
    expert_bf16_p2p_one_local_bytes = routed.training_bytes * max(
        expert_target_copies - (1 if args.colocate else 0), 0
    )
    balanced_expert_source_bytes_per_rank = (
        routed.training_bytes / world_size if args.mode == "rl" else 0
    )
    sync = {
        "bf16_actor_language_payload_gib": gib(source_language_bytes),
        "rollout_runtime_language_payload_gib_per_replica": gib(target_language_bytes),
        "target_write_gib_per_replica": gib(target_write_per_replica),
        "largest_source_tensor": inventory["largest_training_tensor"],
        "dense_ipc_transient_gib_per_sending_rank": gib(dense_sync_transient_bytes),
        "expert_staging_and_ipc_upper_gib_per_rank": gib(expert_sync_transient_bytes),
        "maximum_sync_transient_gib_per_rank": gib(sync_transient_bytes),
        "expert_update_buffer_gib_per_rank": gib(args.update_weight_buffer_bytes),
        "expert_target_gib_per_layer_per_rank": gib(expert_target_bytes_per_layer),
        "minimum_expert_transfer_batches_per_layer": minimum_expert_batches_per_layer,
        "minimum_expert_transfer_batches_per_update": (
            minimum_expert_batches_per_layer * num_moe_layers
        ),
        "rollout_replicas": rollout_replicas,
        "aggregate_target_write_tib_per_update": aggregate_target_bytes / 2**40,
        "expert_bf16_p2p_upper_tib_per_update": expert_bf16_p2p_upper_bytes / 2**40,
        "expert_bf16_p2p_if_one_local_target_tib_per_update": (
            expert_bf16_p2p_one_local_bytes / 2**40
        ),
        "balanced_unique_expert_source_gib_per_rank": gib(
            balanced_expert_source_bytes_per_rank
        ),
        "balanced_expert_egress_upper_gib_per_rank": gib(
            balanced_expert_source_bytes_per_rank * expert_target_copies
        ),
        "balanced_expert_egress_upper_gib_per_node": gib(
            balanced_expert_source_bytes_per_rank
            * expert_target_copies
            * args.gpus_per_node
        ),
    }
    if args.aggregate_sync_bandwidth_gib_s > 0:
        sync["bandwidth_lower_bound_seconds"] = gib(aggregate_target_bytes) / args.aggregate_sync_bandwidth_gib_s

    memory = {
        "gpu_memory_gib": args.gpu_memory_gib,
        "safety_fraction": args.memory_safety_fraction,
        "usable_gpu_gib": gib(usable_gpu_bytes),
        "actor_train_phase_gib": gib(actor_train_peak),
        "rollout_phase_gib": gib(rollout_peak),
        "sync_phase_gib": gib(sync_peak),
        "projected_phase_peak_gib": gib(projected_peak),
        "safety_margin_gib": gib(usable_gpu_bytes - projected_peak),
        "host_pinned_and_optimizer_gib_per_node": gib(host_pinned_per_node),
        "actor_backup_gib_per_node": gib(actor_backup_per_node),
        "vision_loader_staging_gib_per_node": gib(vision_loader_staging_per_node),
        "rollout_weight_backup_gib_per_node": gib(rollout_backup_per_node),
    }
    export_output_bytes = sum(category.checkpoint_bytes for category in categories.values())
    # Training checkpoints contain the dequantized language state; the frozen
    # vision tower is copied from the pinned HF source by the offline exporter.
    model_only_dcp_bytes = source_language_bytes
    trainable_language_numel = (
        regular.parameter_numel + replicated.parameter_numel + routed.parameter_numel
    )
    full_adam_resume_estimate_bytes = (
        trainable_language_numel * 14 + frozen.training_bytes
    )
    official_checkpoint_bytes = inventory.get("native_checkpoint_bytes", export_output_bytes)
    export = {
        "output_checkpoint_gib": gib(export_output_bytes),
        "model_only_torch_dist_source_gib": gib(model_only_dcp_bytes),
        "source_plus_output_disk_gib": gib(model_only_dcp_bytes + export_output_bytes),
        "full_adam_resume_estimate_tib": full_adam_resume_estimate_bytes / 2**40,
        "minimum_gather_and_conversion_transient_gib_per_writer": gib(
            dense_sync_transient_bytes
        ),
        "full_rank_zero_weight_gather": False,
        "conversion_mode": "key_chunk_targeted_dcp",
        "nccl_collectives": False,
        "live_full_hf_export_supported": False,
        "default_task_group_gib": 2.0,
        "default_output_shard_gib": 5.0,
    }
    rollout_initial_load = {
        "prefetch_partition_required": args.mode == "rl",
        "ideal_physical_read_tib": (
            official_checkpoint_bytes * rollout_replicas / 2**40
            if args.mode == "rl"
            else 0
        ),
        "unpartitioned_per_rank_read_upper_tib": (
            official_checkpoint_bytes
            * rollout_replicas
            * args.rollout_gpus_per_engine
            / 2**40
            if args.mode == "rl"
            else 0
        ),
    }
    errors = [gate for gate in gates if gate["severity"] == "error"]
    return {
        "verdict": "FAIL" if errors else "PASS_WITH_RUNTIME_GATES",
        "contract": {
            "repository": FULL_REPOSITORY,
            "revision": FULL_REVISION,
            "training_backbone_layers": 45,
            "released_mtp_layers": 1,
            "training_mtp_layers": 0,
        },
        "topology": {
            "mode": args.mode,
            "actor": {
                "tp": args.tp,
                "pp": args.pp,
                "cp": args.cp,
                "ep": args.ep,
                "etp": args.etp,
                "dp": args.dp,
            },
            "rollout": rollout,
            "colocate": args.colocate,
        },
        "inventory": inventory,
        "categories": {
            name: stats.serializable() for name, stats in sorted(categories.items())
        },
        "actor": actor,
        "memory": memory,
        "sync": sync,
        "export": export,
        "rollout_initial_load": rollout_initial_load,
        "gates": gates,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--headers", type=Path, required=True)
    parser.add_argument("--fetch-missing-headers", action="store_true")
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--strict-official", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mode", choices=("sft", "rl"), default="rl")
    parser.add_argument("--tp", type=int, default=8)
    parser.add_argument("--pp", type=int, default=1)
    parser.add_argument("--cp", type=int, default=1)
    parser.add_argument("--ep", type=int, default=72)
    parser.add_argument("--etp", type=int, default=1)
    parser.add_argument("--dp", type=int, default=72)
    parser.add_argument("--gpu-memory-gib", type=float, default=141.0)
    parser.add_argument("--gpus-per-node", type=int, default=8)
    parser.add_argument("--host-memory-gib", type=float, default=0.0)
    parser.add_argument("--memory-safety-fraction", type=float, default=0.90)
    parser.add_argument("--actor-runtime-reserve-gib", type=float, default=24.0)
    parser.add_argument("--rollout-runtime-reserve-gib", type=float, default=16.0)
    parser.add_argument("--rollout-cache-reserve-gib", type=float, default=16.0)
    parser.add_argument("--sglang-mem-fraction-static", type=float, default=0.54)
    parser.add_argument("--backup-tags", type=int, default=1)
    parser.add_argument("--distributed-optimizer", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--optimizer-offload", action="store_true")
    parser.add_argument("--bf16-optimizer-bytes", type=int, default=12)
    parser.add_argument("--fp32-optimizer-bytes", type=int, default=8)
    parser.add_argument("--offload-train", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--offload-rollout", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--rollout-weights-cpu-backup",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--colocate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-unsharded-remote-sync", action="store_true")
    parser.add_argument("--rollout-tp", type=int, default=8)
    parser.add_argument("--rollout-ep", type=int, default=8)
    parser.add_argument("--rollout-moe-dp", type=int, default=1)
    parser.add_argument("--rollout-num-gpus", type=int, default=576)
    parser.add_argument("--rollout-gpus-per-engine", type=int, default=8)
    parser.add_argument("--rollout-replicas", type=int, default=1)
    parser.add_argument("--rollout-batch-size", type=int, default=8)
    parser.add_argument("--n-samples-per-prompt", type=int, default=8)
    parser.add_argument("--rollout-max-response-len", type=int, default=65536)
    parser.add_argument(
        "--use-rollout-routing-replay",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--sglang-moe-a2a-backend", choices=("deepep", "none"), default="deepep")
    parser.add_argument(
        "--sglang-deepep-mode",
        choices=("auto", "low_latency", "normal"),
        default="auto",
    )
    parser.add_argument("--aggregate-sync-bandwidth-gib-s", type=float, default=0.0)
    parser.add_argument("--update-weight-buffer-bytes", type=int, default=512 * 1024**2)
    parser.add_argument("--sglang-eplb", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--sglang-redundant-experts", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--sglang-elastic-ep", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args(argv)

    positive = (
        "tp",
        "pp",
        "cp",
        "ep",
        "etp",
        "dp",
        "gpu_memory_gib",
        "gpus_per_node",
        "memory_safety_fraction",
        "rollout_tp",
        "rollout_ep",
        "rollout_moe_dp",
        "rollout_num_gpus",
        "rollout_gpus_per_engine",
        "rollout_replicas",
        "rollout_batch_size",
        "n_samples_per_prompt",
        "rollout_max_response_len",
        "update_weight_buffer_bytes",
        "backup_tags",
        "bf16_optimizer_bytes",
        "fp32_optimizer_bytes",
    )
    invalid = [name for name in positive if getattr(args, name) <= 0]
    if invalid:
        parser.error(f"positive values required for: {invalid}")
    if not 0 < args.memory_safety_fraction <= 1:
        parser.error("--memory-safety-fraction must be in (0, 1]")
    if not 0 < args.sglang_mem_fraction_static <= 1:
        parser.error("--sglang-mem-fraction-static must be in (0, 1]")
    nonnegative = (
        "host_memory_gib",
        "actor_runtime_reserve_gib",
        "rollout_runtime_reserve_gib",
        "rollout_cache_reserve_gib",
        "aggregate_sync_bandwidth_gib_s",
    )
    invalid = [name for name in nonnegative if getattr(args, name) < 0]
    if invalid:
        parser.error(f"nonnegative values required for: {invalid}")
    finite = positive + nonnegative
    invalid = [
        name
        for name in finite
        if isinstance(getattr(args, name), float)
        and not math.isfinite(getattr(args, name))
    ]
    if invalid:
        parser.error(f"finite values required for: {invalid}")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    reject_weight_payloads([args.config, args.index, args.headers])
    config = read_json(args.config)
    index = read_json(args.index)
    if args.headers.exists():
        header_document = read_json(args.headers)
    elif args.fetch_missing_headers:
        header_document = cache_remote_headers(index, args.headers)
    else:
        raise ValueError(
            f"header cache is missing: {args.headers}; use --fetch-missing-headers"
        )
    if args.strict_official:
        validate_source_hashes(args.config, args.index, args.headers)
    validate_config(config)
    categories, inventory = build_inventory(
        config,
        index,
        header_document,
        strict_official=args.strict_official,
    )
    report = calculate(args, config, categories, inventory)

    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        with args.json_output.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
            handle.write("\n")

    print(f"verdict: {report['verdict']}")
    print(
        "actor: "
        f"world={report['actor']['world_size']} "
        f"params={report['actor']['parameter_gib_per_rank']:.2f} GiB/rank "
        f"grads={report['actor']['fp32_gradient_gib_per_rank']:.2f} GiB/rank "
        f"optimizer_gpu={report['actor']['optimizer_gpu_gib_per_rank']:.2f} GiB/rank"
    )
    print(
        "memory: "
        f"peak={report['memory']['projected_phase_peak_gib']:.2f} GiB "
        f"budget={report['memory']['usable_gpu_gib']:.2f} GiB "
        f"margin={report['memory']['safety_margin_gib']:.2f} GiB"
    )
    print(
        "sync: "
        f"source={report['sync']['bf16_actor_language_payload_gib']:.2f} GiB "
        f"target/replica={report['sync']['rollout_runtime_language_payload_gib_per_replica']:.2f} GiB "
        f"aggregate={report['sync']['aggregate_target_write_tib_per_update']:.2f} TiB/update"
    )
    for gate in report["gates"]:
        print(f"{gate['severity']}: {gate['message']}")
    return 1 if report["verdict"] == "FAIL" else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, RuntimeError, TypeError, ValueError) as error:
        print(f"preflight error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
