#!/usr/bin/env python3
"""Normalize the public 84M GLM-5.3-Flash proxy for lifecycle qualification.

The public proxy predates the final production serialization contract. This
tool reads only that small proxy, never the full released model, and writes a
separate fixture with an exact provenance and content-hash contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

SOURCE_REPOSITORY = "inference-optimization/GLM-5.3-Flash-0.1B-A0.1B"
SOURCE_REVISION = "7c3a6d3dc51732dd8ab230888e06ba8c93a381ac"
SOURCE_MODEL_SHA256 = "0f0645f3da199f6c2f381c4f50f2729d6e455dbd3e87f91974202807ee1e5df8"
NORMALIZED_MODEL_SHA256 = "c8859ffe8b82f4e7346f49abaef98b52378f95a32df2c32e18b5156890857d84"
EXPECTED_TENSORS = 223
EXPECTED_DTYPE_CHANGES = 32
HC_FP32_SUFFIXES = (
    ".hc_attn_base",
    ".hc_attn_scale",
    ".hc_ffn_base",
    ".hc_ffn_scale",
)
KDA_BF16_SUFFIXES = (
    ".q_conv1d.weight",
    ".k_conv1d.weight",
    ".v_conv1d.weight",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def normalize(source: Path, output: Path) -> dict:
    source = source.resolve()
    output = output.resolve()
    source_checkpoint = source / "model.safetensors"
    if not source_checkpoint.is_file():
        raise FileNotFoundError(source_checkpoint)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    if output == source or source in output.parents:
        raise ValueError("output must not be the source or a child of the source")

    source_hash = sha256(source_checkpoint)
    if source_hash != SOURCE_MODEL_SHA256:
        raise ValueError(
            f"tiny source hash {source_hash} does not match {SOURCE_MODEL_SHA256}"
        )
    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    text = config["text_config"]
    layer_types = text["layer_types"]
    if config.get("model_type") != "glm5_next":
        raise ValueError("source is not a glm5_next checkpoint")
    if config.get("architectures") != ["Glm5NextForConditionalGeneration"]:
        raise ValueError("source architecture is not GLM-5.3-Flash")
    if text.get("num_hidden_layers") != 5 or layer_types != [
        "linear_attention",
        "linear_attention",
        "linear_attention",
        "deepseek_sparse_attention",
        "linear_attention",
    ]:
        raise ValueError("unexpected tiny KDA/DSA schedule")

    kda_layers = [i for i, kind in enumerate(layer_types) if kind == "linear_attention"]
    dsa_layers = [
        i for i, kind in enumerate(layer_types) if kind == "deepseek_sparse_attention"
    ]
    runtime_qk_head_dim = text["qk_nope_head_dim"] + text["qk_rope_head_dim"]
    if runtime_qk_head_dim != 64:
        raise ValueError(f"unexpected tiny runtime QK width {runtime_qk_head_dim}")
    text["qk_head_dim"] = runtime_qk_head_dim
    text["num_nextn_predict_layers"] = 0
    text["linear_attn_config"].update(
        {"kda_layers": kda_layers, "full_attn_layers": dsa_layers}
    )

    tensors = load_file(source_checkpoint, device="cpu")
    if len(tensors) != EXPECTED_TENSORS:
        raise ValueError(f"unexpected tiny tensor count {len(tensors)}")
    converted: dict[str, torch.Tensor] = {}
    dtype_changes: dict[str, dict[str, str]] = {}
    for key, value in tensors.items():
        target_dtype = value.dtype
        if key.endswith(HC_FP32_SUFFIXES):
            target_dtype = torch.float32
        elif key.endswith(KDA_BF16_SUFFIXES):
            target_dtype = torch.bfloat16
        converted[key] = value.to(target_dtype).contiguous()
        if target_dtype != value.dtype:
            dtype_changes[key] = {
                "from": str(value.dtype).removeprefix("torch."),
                "to": str(target_dtype).removeprefix("torch."),
            }
    if len(dtype_changes) != EXPECTED_DTYPE_CHANGES:
        raise ValueError(f"unexpected dtype change count {len(dtype_changes)}")

    with safe_open(source_checkpoint, framework="pt", device="cpu") as reader:
        metadata = reader.metadata()
    provenance = {
        "status": "contract-normalized tiny fixture; not an upstream release",
        "source_repository": SOURCE_REPOSITORY,
        "source_revision": SOURCE_REVISION,
        "source_model_sha256": source_hash,
        "normalized_model_sha256": NORMALIZED_MODEL_SHA256,
        "full_model_tensor_payloads_read": False,
        "config_changes": {
            "text_config.qk_head_dim": runtime_qk_head_dim,
            "text_config.num_nextn_predict_layers": 0,
            "text_config.linear_attn_config.kda_layers": kda_layers,
            "text_config.linear_attn_config.full_attn_layers": dsa_layers,
        },
        "dtype_changes": dtype_changes,
        "dtype_change_count": len(dtype_changes),
        "tensor_count": len(converted),
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        for item in source.iterdir():
            if item.is_file() and item.name != source_checkpoint.name:
                shutil.copy2(item, temporary / item.name)
        write_json(temporary / "config.json", config)
        save_file(converted, temporary / source_checkpoint.name, metadata=metadata)
        normalized_hash = sha256(temporary / source_checkpoint.name)
        if normalized_hash != NORMALIZED_MODEL_SHA256:
            raise ValueError(
                f"normalized hash {normalized_hash} does not match {NORMALIZED_MODEL_SHA256}"
            )
        write_json(temporary / "contract_normalization.json", provenance)
        temporary.replace(output)
    except BaseException:
        shutil.rmtree(temporary)
        raise
    return provenance


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(normalize(args.source, args.output), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
