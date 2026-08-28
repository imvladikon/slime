import importlib.util
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "tools"
    / "verify_glm53_flash_full_export.py"
)
SPEC = importlib.util.spec_from_file_location("verify_glm53_flash_full_export", MODULE_PATH)
verifier = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = verifier
SPEC.loader.exec_module(verifier)


def test_full_export_verifier_uses_headers_and_drops_mtp_scales(tmp_path, monkeypatch):
    monkeypatch.setattr(verifier, "EXPECTED_SOURCE_TENSORS", 3)
    monkeypatch.setattr(verifier, "EXPECTED_EXPORT_TENSORS", 1)

    source_config = {
        "model_type": "glm5_next",
        "text_config": {
            "num_nextn_predict_layers": 1,
            "index_share_for_mtp_iteration": True,
        },
    }
    candidate_config = {
        "model_type": "glm5_next",
        "text_config": {
            "num_nextn_predict_layers": 0,
            "index_share_for_mtp_iteration": False,
        },
    }
    source_headers = {
        "repo": verifier.FULL_REPOSITORY,
        "revision": verifier.FULL_REVISION,
        "header": {
            "model.language_model.layers.0.weight": {
                "shape": [2],
                "dtype": "F8_E4M3",
            },
            "model.language_model.layers.0.weight_scale_inv": {
                "shape": [1],
                "dtype": "F32",
            },
            "model.language_model.layers.45.weight": {
                "shape": [2],
                "dtype": "BF16",
            },
        },
    }
    source_index = {
        "weight_map": {key: "source.safetensors" for key in source_headers["header"]}
    }

    source_config_path = tmp_path / "source-config.json"
    source_index_path = tmp_path / "source-index.json"
    source_headers_path = tmp_path / "source-headers.json"
    source_config_path.write_text(json.dumps(source_config), encoding="utf-8")
    source_index_path.write_text(json.dumps(source_index), encoding="utf-8")
    source_headers_path.write_text(json.dumps(source_headers), encoding="utf-8")

    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "config.json").write_text(json.dumps(candidate_config), encoding="utf-8")
    save_file(
        {"model.language_model.layers.0.weight": torch.ones(2, dtype=torch.bfloat16)},
        candidate / "model.safetensors",
    )

    report = verifier.verify(
        source_config_path,
        source_index_path,
        source_headers_path,
        candidate,
        reload_transformers_config=False,
    )

    assert report["header_only_contract"] == "PASS"
    assert report["export_tensors"] == 1
    assert report["full_model_reload"] == "NOT_RUN_BY_DESIGN"
