"""Deterministic non-constant reward used by the tiny GLM lifecycle gate."""

from __future__ import annotations


async def score(_args, samples, **_kwargs):
    """Return stable alternating rewards so the smoke run has a policy signal."""
    return [float(sample.index % 2) for sample in samples]


__all__ = ["score"]
