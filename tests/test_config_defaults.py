"""Built-in defaults must not drift from the tracked caps.json or point at a
machine-specific path."""

import json
from pathlib import Path

from claude_budget.config import DEFAULT_ALERT_PATH, DEFAULT_CAPS

REPO = Path(__file__).resolve().parent.parent


def test_default_caps_match_caps_json():
    tracked = json.loads((REPO / "caps.json").read_text(encoding="utf-8"))
    assert DEFAULT_CAPS["reset_day"] == tracked["reset_day"]
    assert float(DEFAULT_CAPS["global"]) == float(tracked["global"])
    assert {k: float(v) for k, v in DEFAULT_CAPS["consumers"].items()} == {
        k: float(v) for k, v in tracked["consumers"].items()
    }


def test_default_alert_path_is_under_home():
    assert DEFAULT_ALERT_PATH.is_absolute()
    assert Path.home() in DEFAULT_ALERT_PATH.parents
