"""Deterministic non-constant reward used by the tiny GLM lifecycle gate."""

from __future__ import annotations


async def score(_args, sample, **_kwargs):
    """Return stable alternating rewards so the smoke run has a policy signal."""
    if sample.index is None:
        raise ValueError("The GLM smoke reward requires a rollout sample index")
    return float(sample.index % 2)


__all__ = ["score"]
