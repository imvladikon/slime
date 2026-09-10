import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from slime.ray.train_actor import _remove_expandable_segments_from_allocator_env


def test_remove_expandable_segments_preserves_other_allocator_options(monkeypatch):
    monkeypatch.setenv(
        "PYTORCH_ALLOC_CONF",
        "max_split_size_mb:256,expandable_segments:True,garbage_collection_threshold:0.8",
    )
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:False")

    changed = _remove_expandable_segments_from_allocator_env()

    assert changed == ["PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF"]
    assert os.environ["PYTORCH_ALLOC_CONF"] == "max_split_size_mb:256,garbage_collection_threshold:0.8"
    assert "PYTORCH_CUDA_ALLOC_CONF" not in os.environ


def test_remove_expandable_segments_leaves_unrelated_values(monkeypatch):
    monkeypatch.setenv("PYTORCH_ALLOC_CONF", "backend:native,max_split_size_mb:128")
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)

    changed = _remove_expandable_segments_from_allocator_env()

    assert changed == []
    assert os.environ["PYTORCH_ALLOC_CONF"] == "backend:native,max_split_size_mb:128"
