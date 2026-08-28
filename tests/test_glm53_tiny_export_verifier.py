import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "verify_glm53_flash_tiny_export.py"
SPEC = importlib.util.spec_from_file_location("verify_glm53_flash_tiny_export", MODULE_PATH)
verifier = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = verifier
SPEC.loader.exec_module(verifier)


def _checkpoint(root: Path, language: torch.Tensor, vision: torch.Tensor):
    root.mkdir()
    save_file(
        {
            "model.language_model.layers.0.weight": language,
            "model.visual.blocks.0.weight": vision,
        },
        root / "model.safetensors",
    )
    (root / "config.json").write_text(
        json.dumps(
            {
                "model_type": "glm5_next",
                "text_config": {"num_nextn_predict_layers": 0},
            }
        ),
        encoding="utf-8",
    )


def test_tiny_export_verifier_requires_language_update_and_frozen_vision(tmp_path):
    source = tmp_path / "source"
    candidate = tmp_path / "candidate"
    vision = torch.ones(2, dtype=torch.bfloat16)
    _checkpoint(source, torch.zeros(2, dtype=torch.bfloat16), vision)
    _checkpoint(candidate, torch.ones(2, dtype=torch.bfloat16), vision)

    report = verifier.verify(
        source, candidate, reload_transformers=False, strict_contract=False
    )

    assert report["tensors"] == 2
    assert report["changed_language_tensors"] == 1
    assert report["changed_vision_tensors"] == 0


def test_tiny_export_verifier_rejects_vision_update(tmp_path):
    source = tmp_path / "source"
    candidate = tmp_path / "candidate"
    _checkpoint(
        source,
        torch.zeros(2, dtype=torch.bfloat16),
        torch.zeros(2, dtype=torch.bfloat16),
    )
    _checkpoint(
        candidate,
        torch.ones(2, dtype=torch.bfloat16),
        torch.ones(2, dtype=torch.bfloat16),
    )

    with pytest.raises(ValueError, match="vision tensors changed"):
        verifier.verify(
            source, candidate, reload_transformers=False, strict_contract=False
        )


def test_tiny_export_verifier_rejects_incomplete_contract_by_default(tmp_path):
    source = tmp_path / "source"
    candidate = tmp_path / "candidate"
    vision = torch.ones(2, dtype=torch.bfloat16)
    _checkpoint(source, torch.zeros(2, dtype=torch.bfloat16), vision)
    _checkpoint(candidate, torch.ones(2, dtype=torch.bfloat16), vision)

    with pytest.raises(ValueError, match="tensor partition"):
        verifier.verify(source, candidate, reload_transformers=False)
