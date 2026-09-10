import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import train


class _RemoteMethod:
    def __init__(self, events):
        self.events = events

    def remote(self, *, action):
        self.events.append(("check", action))
        return action


class _RolloutManager:
    def __init__(self, events):
        self.check_weights = _RemoteMethod(events)


class _ActorModel:
    def __init__(self, events):
        self.events = events

    def update_weights(self):
        self.events.append(("actor", "update"))


def test_post_step_weight_check_resets_and_resynchronizes(monkeypatch):
    events = []
    monkeypatch.setattr(train.ray, "get", lambda value: events.append(("get", value)))

    train._update_rollout_weights(
        SimpleNamespace(check_weight_update_equal=True),
        _ActorModel(events),
        _RolloutManager(events),
        refresh_snapshot=True,
    )

    assert events == [
        ("actor", "update"),
        ("check", "snapshot"),
        ("get", "snapshot"),
        ("check", "reset_tensors"),
        ("get", "reset_tensors"),
        ("actor", "update"),
        ("check", "compare"),
        ("get", "compare"),
    ]


def test_regular_weight_update_is_not_duplicated(monkeypatch):
    events = []
    monkeypatch.setattr(train.ray, "get", lambda value: events.append(("get", value)))

    train._update_rollout_weights(
        SimpleNamespace(check_weight_update_equal=False),
        _ActorModel(events),
        _RolloutManager(events),
        refresh_snapshot=True,
    )

    assert events == [("actor", "update")]


def test_frozen_weight_fingerprint_selects_only_requested_prefixes():
    responses = [
        {
            "success": True,
            "ranks": [
                {
                    "checksums": {
                        "model.layers.0.weight": "language",
                        "visual.blocks.0.weight": "vision-0",
                        "visual.blocks.1.weight": "vision-1",
                    }
                }
            ],
        }
    ]

    assert train._frozen_weight_fingerprint(responses, ["visual."]) == (
        (
            ("visual.blocks.0.weight", "vision-0"),
            ("visual.blocks.1.weight", "vision-1"),
        ),
    )
