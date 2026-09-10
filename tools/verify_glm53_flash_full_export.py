#!/usr/bin/env python3
"""Verify a full GLM-5.3-Flash HF export without loading the 321B model.

The mandatory path compares only JSON and safetensors headers.  When the
released shards are already present locally, an optional gate streams frozen
vision tensors one at a time to prove value equality without materializing the
full checkpoint.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

FULL_REPOSITORY = "zai-org/GLM-5.3-Flash"
FULL_REVISION = "04c4e9e95c5da8862dced7e5056455116f83a7e0"
EXPECTED_SOURCE_TENSORS = 76_108
EXPECTED_EXPORT_TENSORS = 74_348
EXPECTED_EXPORT_SCALE_TENSORS = 36_467
EXPECTED_VISION_TENSORS = 347
EXPECTED_FROZEN_INDEXER_TENSORS = 77
SOURCE_CONFIG_SHA256 = "bb8f01c42cb92a52ca72e65afb4d5bd8d11aef083cd210e8de25dfb904f23e9f"
SOURCE_INDEX_SHA256 = "3c3f40366a53c3fd7974b4eab7881a365a98c2a4329150befebab99fe7c18b05"
SOURCE_HEADERS_SHA256 = "b01c2e4b5ed1595b7ad2cbcc9b70800a1f0009651f8581fa342c36ed41cdd813"
LAYER_PATTERN = re.compile(r"\.layers\.(\d+)\.")


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


def is_scale_metadata(key: str) -> bool:
    return key.endswith(".weight_scale_inv")


def is_mtp(key: str, num_backbone_layers: int = 45) -> bool:
    match = LAYER_PATTERN.search(key)
    return match is not None and int(match.group(1)) >= num_backbone_layers


def checkpoint_weight_map(root: Path) -> dict[str, str]:
    index = root / "model.safetensors.index.json"
    if index.is_file():
        mapping = read_json(index).get("weight_map")
        if not isinstance(mapping, dict) or not mapping:
            raise ValueError(f"invalid candidate weight map: {index}")
        return mapping
    single = root / "model.safetensors"
    if not single.is_file():
        raise ValueError(f"no safetensors checkpoint found in {root}")
    with safe_open(single, framework="pt", device="cpu") as handle:
        return {key: single.name for key in handle.keys()}


def checkpoint_headers(root: Path, mapping: dict[str, str]) -> dict[str, dict[str, Any]]:
    names_by_file: dict[str, list[str]] = defaultdict(list)
    for key, filename in mapping.items():
        names_by_file[filename].append(key)
    result: dict[str, dict[str, Any]] = {}
    for filename, expected_keys in sorted(names_by_file.items()):
        path = root / filename
        if not path.is_file():
            raise ValueError(f"candidate shard is missing: {path}")
        with safe_open(path, framework="pt", device="cpu") as handle:
            actual_keys = set(handle.keys())
            if actual_keys != set(expected_keys):
                raise ValueError(f"candidate shard/index key mismatch: {filename}")
            for key in expected_keys:
                tensor_slice = handle.get_slice(key)
                result[key] = {
                    "shape": list(tensor_slice.get_shape()),
                    "dtype": tensor_slice.get_dtype(),
                }
    return result


def expected_export_headers(
    source_index: dict[str, Any], source_header_document: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    if (
        source_header_document.get("repo") != FULL_REPOSITORY
        or source_header_document.get("revision") != FULL_REVISION
    ):
        raise ValueError("source header cache repository/revision mismatch")
    source = source_header_document.get("header", {})
    weight_map = source_index.get("weight_map", {})
    if set(source) != set(weight_map):
        raise ValueError("source index/header keys differ")
    if len(source) != EXPECTED_SOURCE_TENSORS:
        raise ValueError(f"unexpected source tensor count: {len(source)}")

    expected = {}
    for key, entry in source.items():
        if is_mtp(key):
            continue
        expected[key] = {
            "shape": entry["shape"],
            "dtype": entry["dtype"],
        }
    if len(expected) != EXPECTED_EXPORT_TENSORS:
        raise ValueError(f"unexpected FP8 export tensor count: {len(expected)}")
    scale_tensors = sum(is_scale_metadata(key) for key in expected)
    if scale_tensors != EXPECTED_EXPORT_SCALE_TENSORS:
        raise ValueError(f"unexpected FP8 export scale tensor count: {scale_tensors}")
    return expected


def validate_export_config(source: dict[str, Any], candidate: dict[str, Any]) -> None:
    expected = copy.deepcopy(source)
    text = expected["text_config"]
    text["num_nextn_predict_layers"] = 0
    text["index_share_for_mtp_iteration"] = False
    if candidate != expected:
        differing = sorted(
            key for key in set(candidate).union(expected) if candidate.get(key) != expected.get(key)
        )
        raise ValueError(f"candidate config differs outside the training contract: {differing}")


def _tensor(root: Path, mapping: dict[str, str], key: str) -> torch.Tensor:
    with safe_open(root / mapping[key], framework="pt", device="cpu") as handle:
        return handle.get_tensor(key)


def compare_frozen_values(
    source_root: Path,
    source_mapping: dict[str, str],
    candidate_root: Path,
    candidate_mapping: dict[str, str],
) -> int:
    keys = sorted(
        key
        for key in candidate_mapping
        if key.startswith("model.visual.") or ".self_attn.indexer." in key
    )
    vision_count = sum(key.startswith("model.visual.") for key in keys)
    indexer_count = sum(".self_attn.indexer." in key for key in keys)
    if vision_count != EXPECTED_VISION_TENSORS:
        raise ValueError(f"unexpected frozen vision tensor count: {vision_count}")
    if indexer_count != EXPECTED_FROZEN_INDEXER_TENSORS:
        raise ValueError(f"unexpected frozen indexer tensor count: {indexer_count}")
    if any(key not in source_mapping for key in keys):
        raise ValueError("source checkpoint is missing frozen vision keys")
    for key in keys:
        expected = _tensor(source_root, source_mapping, key)
        actual = _tensor(candidate_root, candidate_mapping, key)
        if not torch.equal(expected, actual):
            raise ValueError(f"frozen tensor changed: {key}")
        del expected, actual
    return len(keys)


def validate_candidate_values(root: Path, mapping: dict[str, str]) -> int:
    names_by_file: dict[str, list[str]] = defaultdict(list)
    for key, filename in mapping.items():
        names_by_file[filename].append(key)
    checked = 0
    for filename, keys in sorted(names_by_file.items()):
        with safe_open(root / filename, framework="pt", device="cpu") as handle:
            for key in sorted(keys):
                tensor = handle.get_tensor(key)
                if tensor.is_floating_point():
                    values = tensor.float()
                    if not torch.isfinite(values).all():
                        raise ValueError(f"non-finite candidate tensor: {key}")
                    if is_scale_metadata(key) and not torch.all(values > 0):
                        raise ValueError(f"nonpositive FP8 scale tensor: {key}")
                    del values
                del tensor
                checked += 1
    return checked


def verify(
    source_config_path: Path,
    source_index_path: Path,
    source_headers_path: Path,
    candidate: Path,
    *,
    source_weights: Path | None = None,
    reload_transformers_config: bool = True,
    validate_values: bool = True,
    strict_source_hashes: bool = True,
) -> dict[str, Any]:
    source_hashes = (
        validate_source_hashes(source_config_path, source_index_path, source_headers_path)
        if strict_source_hashes
        else None
    )
    source_config = read_json(source_config_path)
    source_index = read_json(source_index_path)
    source_headers = read_json(source_headers_path)
    expected = expected_export_headers(source_index, source_headers)

    candidate_config = read_json(candidate / "config.json")
    validate_export_config(source_config, candidate_config)
    candidate_mapping = checkpoint_weight_map(candidate)
    actual = checkpoint_headers(candidate, candidate_mapping)
    if set(actual) != set(expected):
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        raise ValueError(f"candidate keys differ: missing={missing[:3]} extra={extra[:3]}")
    mismatches = [key for key in expected if actual[key] != expected[key]]
    if mismatches:
        key = mismatches[0]
        raise ValueError(f"candidate metadata mismatch for {key}: {actual[key]} != {expected[key]}")

    checked_values: int | str = "NOT_RUN"
    if validate_values:
        checked_values = validate_candidate_values(candidate, candidate_mapping)

    config_class = None
    if reload_transformers_config:
        from transformers import AutoConfig

        config_class = type(
            AutoConfig.from_pretrained(candidate, local_files_only=True)
        ).__name__
        if config_class != "Glm5NextConfig":
            raise ValueError(f"unexpected Transformers config class: {config_class}")

    frozen_values = "NOT_RUN"
    if source_weights is not None:
        source_mapping = checkpoint_weight_map(source_weights)
        frozen_values = compare_frozen_values(
            source_weights, source_mapping, candidate, candidate_mapping
        )

    total_numel = sum(math.prod(entry["shape"]) for entry in actual.values())
    return {
        "source_repository": FULL_REPOSITORY,
        "source_revision": FULL_REVISION,
        "candidate": str(candidate),
        "header_only_contract": "PASS",
        "source_metadata_sha256": source_hashes,
        "export_tensors": len(actual),
        "export_numel": total_numel,
        "mtp_tensors": 0,
        "scale_metadata_tensors": sum(is_scale_metadata(key) for key in actual),
        "candidate_value_tensors_checked": checked_values,
        "transformers_config_class": config_class,
        "frozen_value_comparison": frozen_values,
        "full_model_reload": "NOT_RUN_BY_DESIGN",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--source-index", type=Path, required=True)
    parser.add_argument("--source-headers", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--source-weights", type=Path)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument(
        "--reload-transformers-config",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--validate-values",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--strict-source-hashes",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()
    report = verify(
        args.source_config,
        args.source_index,
        args.source_headers,
        args.candidate,
        source_weights=args.source_weights,
        reload_transformers_config=args.reload_transformers_config,
        validate_values=args.validate_values,
        strict_source_hashes=args.strict_source_hashes,
    )
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
