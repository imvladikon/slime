#!/usr/bin/env python3
"""Verify a tiny GLM-5.3-Flash HF export after an optimizer step."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open

MAX_CHECKPOINT_BYTES = 2 * 2**30
EXPECTED_MODEL_CLASS = "Glm5NextForConditionalGeneration"
EXPECTED_PARAMETERS = 84_361_950
EXPECTED_TENSORS = 223
EXPECTED_VISION_TENSORS = 39
EXPECTED_LANGUAGE_TENSORS = 184
EXPECTED_LAYER_TYPES = [
    "linear_attention",
    "linear_attention",
    "linear_attention",
    "deepseek_sparse_attention",
    "linear_attention",
]


def weight_map(root: Path) -> dict[str, str]:
    index = root / "model.safetensors.index.json"
    if index.is_file():
        with index.open(encoding="utf-8") as handle:
            return json.load(handle)["weight_map"]
    single = root / "model.safetensors"
    if not single.is_file():
        raise ValueError(f"no safetensors checkpoint found in {root}")
    with safe_open(single, framework="pt", device="cpu") as handle:
        return {key: single.name for key in handle.keys()}


def load_tensors(root: Path) -> dict[str, torch.Tensor]:
    mapping = weight_map(root)
    files = {root / filename for filename in mapping.values()}
    size = sum(path.stat().st_size for path in files)
    if size > MAX_CHECKPOINT_BYTES:
        raise ValueError(
            f"refusing {size / 2**30:.2f} GiB checkpoint; this verifier is tiny-only"
        )
    by_file: dict[str, list[str]] = defaultdict(list)
    for key, filename in mapping.items():
        by_file[filename].append(key)
    tensors = {}
    for filename, keys in by_file.items():
        with safe_open(root / filename, framework="pt", device="cpu") as handle:
            for key in keys:
                tensors[key] = handle.get_tensor(key)
    return tensors


def _validate_tiny_config(config: dict) -> None:
    text = config.get("text_config", {})
    checks = {
        "model_type": config.get("model_type") == "glm5_next",
        "architecture": config.get("architectures")
        == ["Glm5NextForConditionalGeneration"],
        "text_model_type": text.get("model_type") == "glm5_next_text",
        "layers": text.get("num_hidden_layers") == 5,
        "layer_schedule": text.get("layer_types") == EXPECTED_LAYER_TYPES,
        "mlp_schedule": text.get("mlp_layer_types")
        == ["dense", "dense", "dense", "sparse", "sparse"],
        "qk_width": text.get("qk_head_dim") == 64
        and text.get("qk_nope_head_dim") == 64
        and text.get("qk_rope_head_dim") == 0,
        "mtp": text.get("num_nextn_predict_layers") == 0,
        "mhc": text.get("mhc") is True and text.get("hc_mult") == 4,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"tiny config contract failed: {failed}")


def verify(
    source: Path,
    candidate: Path,
    reload_transformers: bool,
    strict_contract: bool = True,
) -> dict:
    source_state = load_tensors(source)
    candidate_state = load_tensors(candidate)
    if set(source_state) != set(candidate_state):
        missing = sorted(set(source_state) - set(candidate_state))
        extra = sorted(set(candidate_state) - set(source_state))
        raise ValueError(f"state keys differ: missing={missing[:3]} extra={extra[:3]}")

    vision_keys = [key for key in candidate_state if key.startswith("model.visual.")]
    language_keys = [key for key in candidate_state if not key.startswith("model.visual.")]
    if strict_contract and (
        len(candidate_state) != EXPECTED_TENSORS
        or len(vision_keys) != EXPECTED_VISION_TENSORS
        or len(language_keys) != EXPECTED_LANGUAGE_TENSORS
    ):
        raise ValueError(
            "tiny tensor partition mismatch: "
            f"total={len(candidate_state)} vision={len(vision_keys)} "
            f"language={len(language_keys)}"
        )

    changed_language = []
    changed_vision = []
    for key, expected in source_state.items():
        actual = candidate_state[key]
        if expected.shape != actual.shape or expected.dtype != actual.dtype:
            raise ValueError(
                f"metadata mismatch for {key}: "
                f"{tuple(expected.shape)}/{expected.dtype} != "
                f"{tuple(actual.shape)}/{actual.dtype}"
            )
        if strict_contract and actual.is_floating_point() and not torch.isfinite(actual).all():
            raise ValueError(f"non-finite candidate tensor: {key}")
        if not torch.equal(expected, actual):
            if key.startswith("model.visual."):
                changed_vision.append(key)
            else:
                changed_language.append(key)
    if changed_vision:
        raise ValueError(f"frozen vision tensors changed: {changed_vision[:3]}")
    if not changed_language:
        raise ValueError("no language tensor changed after the optimizer step")
    changed_indexer = [
        key for key in changed_language if ".self_attn.indexer." in key
    ]
    if strict_contract and changed_indexer:
        raise ValueError(f"frozen DSA indexer tensors changed: {changed_indexer[:3]}")

    with (candidate / "config.json").open(encoding="utf-8") as handle:
        config = json.load(handle)
    if strict_contract:
        _validate_tiny_config(config)
    else:
        if config.get("model_type") != "glm5_next":
            raise ValueError("candidate is not a glm5_next checkpoint")
        if config["text_config"].get("num_nextn_predict_layers", 0) != 0:
            raise ValueError("candidate export incorrectly enables the absent tiny MTP layer")

    model_class = None
    parameters = None
    if reload_transformers:
        from transformers import AutoModelForImageTextToText

        model = AutoModelForImageTextToText.from_pretrained(
            candidate, local_files_only=True, dtype=torch.float32
        )
        model_class = type(model).__name__
        parameters = sum(parameter.numel() for parameter in model.parameters())
        del model
        if model_class != EXPECTED_MODEL_CLASS:
            raise ValueError(
                f"unexpected Transformers class {model_class}; expected {EXPECTED_MODEL_CLASS}"
            )
        if parameters != EXPECTED_PARAMETERS:
            raise ValueError(
                f"unexpected parameter count {parameters}; expected {EXPECTED_PARAMETERS}"
            )

    return {
        "source": str(source),
        "candidate": str(candidate),
        "tensors": len(candidate_state),
        "changed_language_tensors": len(changed_language),
        "changed_vision_tensors": len(changed_vision),
        "frozen_indexer_tensors": sum(
            ".self_attn.indexer." in key for key in candidate_state
        ),
        "transformers_model_class": model_class,
        "transformers_parameters": parameters,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument(
        "--reload-transformers", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--strict-contract", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args()
    report = verify(
        args.source,
        args.candidate,
        args.reload_transformers,
        strict_contract=args.strict_contract,
    )
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        with args.json_output.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
            handle.write("\n")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
