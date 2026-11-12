"""80% global-cap alert file for the CyberDash portal.

The portal is not wired yet; this only writes a small JSON object. The file
is replaced atomically so a reader never sees a half-written document.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime
from pathlib import Path

from .config import ALERT_THRESHOLD, Cycle

log = logging.getLogger(__name__)


def build_alert(*, spent: float, cap: float, cycle: Cycle, now: datetime) -> dict[str, object]:
    pct = spent / cap if cap else 1.0
    return {
        "source": "claude-budget",
        "level": "critical" if pct >= 1.0 else "warning",
        "message": f"Claude API global spend ${spent:.2f} is {pct:.0%} of the ${cap:.2f} cap",
        "spent_usd": round(spent, 4),
        "cap_usd": cap,
        "pct": round(pct, 4),
        "threshold": ALERT_THRESHOLD,
        "cycle_start": cycle.start.date().isoformat(),
        "refill_date": cycle.end.date().isoformat(),
        "days_to_refill": cycle.days_to_refill(now),
        "generated_at": now.isoformat(timespec="seconds"),
    }


def maybe_write_alert(path: Path, *, spent: float, cap: float, cycle: Cycle, now: datetime) -> bool:
    """Write the alert when spend is at or above the threshold. Never raises."""
    if cap <= 0 or spent < ALERT_THRESHOLD * cap:
        return False
    payload = build_alert(spent=spent, cap=cap, cycle=cycle, now=now)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=".alert-", suffix=".json", dir=path.parent)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp, path)
        return True
    except OSError as exc:  # missing drive, permissions: never break a call over this
        log.warning("claude_budget: could not write alert %s: %s", path, exc)
        return False
