import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from slime_plugins.models.glm5_next.smoke_reward import score


def test_glm53_smoke_reward_alternates_per_sample():
    rewards = [
        asyncio.run(score(None, SimpleNamespace(index=index)))
        for index in range(4)
    ]

    assert rewards == [0.0, 1.0, 0.0, 1.0]


def test_glm53_smoke_reward_requires_sample_index():
    with pytest.raises(ValueError, match="sample index"):
        asyncio.run(score(None, SimpleNamespace(index=None)))
