"""Paths, caps and billing-cycle math."""

from __future__ import annotations

import calendar
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .errors import UnknownConsumer

# Environment variables. ANTHROPIC_API_KEY is intentionally never read or set:
# a user-wide value would move Claude Code and `claude -p` off the Max plan.
ENV_API_KEY = "DEX_ANTHROPIC_API_KEY"
ENV_BASE_URL = "DEX_ANTHROPIC_BASE_URL"
ENV_DB = "CLAUDE_BUDGET_DB"
ENV_CAPS = "CLAUDE_BUDGET_CAPS"
ENV_ALERT = "CLAUDE_BUDGET_ALERT_PATH"

DEFAULT_BASE_URL = "https://api.anthropic.com"
DEFAULT_DB = Path(r"D:\DexProjects\claude-budget\ledger.db")
DEFAULT_ALERT_PATH = Path(r"G:\CyberDash\alerts-claude-budget.json")
ALERT_THRESHOLD = 0.80

DEFAULT_CAPS: dict[str, Any] = {
    "reset_day": 29,
    "global": 190.0,
    "consumers": {
        "trading": 45.0,
        "portal-ingest": 20.0,
        "portal-chat": 25.0,
        "house-advice": 5.0,
        "job-classify": 12.0,
        "cmdb": 15.0,
        "rc-nuggets": 10.0,
        "finance": 8.0,
        "hermes-fallback": 10.0,
        "reserve": 35.0,
    },
}


def db_path() -> Path:
    return Path(os.environ.get(ENV_DB) or DEFAULT_DB)


def alert_path() -> Path:
    return Path(os.environ.get(ENV_ALERT) or DEFAULT_ALERT_PATH)


def caps_path() -> Path | None:
    """Resolve the caps file: $CLAUDE_BUDGET_CAPS, else caps.local.json then
    caps.json next to the ledger. None means use DEFAULT_CAPS."""
    env = os.environ.get(ENV_CAPS)
    if env:
        return Path(env)
    folder = db_path().parent
    for name in ("caps.local.json", "caps.json"):
        candidate = folder / name
        if candidate.is_file():
            return candidate
    return None


@dataclass(frozen=True)
class Caps:
    global_cap: float
    consumers: dict[str, float]
    reset_day: int = 29
    source: str = "builtin"

    def cap_for(self, consumer: str) -> float:
        try:
            return self.consumers[consumer]
        except KeyError:
            known = ", ".join(sorted(self.consumers))
            raise UnknownConsumer(f"Consumer {consumer!r} has no cap (known: {known})") from None

    @classmethod
    def from_dict(cls, data: dict[str, Any], source: str = "dict") -> "Caps":
        reset_day = int(data.get("reset_day", 29))
        if not 1 <= reset_day <= 31:
            raise ValueError(f"reset_day must be 1-31, got {reset_day}")
        g = float(data["global"])
        consumers = {str(k): float(v) for k, v in dict(data.get("consumers", {})).items()}
        for name, v in [("global", g), *consumers.items()]:
            if v < 0:
                raise ValueError(f"cap for {name!r} is negative")
        return cls(global_cap=g, consumers=consumers, reset_day=reset_day, source=source)


def load_caps(path: Path | None = None) -> Caps:
    path = path if path is not None else caps_path()
    if path is None:
        return Caps.from_dict(DEFAULT_CAPS, source="builtin")
    with open(path, encoding="utf-8") as f:
        return Caps.from_dict(json.load(f), source=str(path))


# ---------------------------------------------------------------------------
# Billing cycle. Cycles run from reset_day 00:00 UTC to the next reset_day.
# In a month shorter than reset_day the boundary clamps to its last day
# (reset_day 29 in a non-leap February resets on Feb 28).
# ---------------------------------------------------------------------------


def _boundary(year: int, month: int, reset_day: int) -> datetime:
    day = min(reset_day, calendar.monthrange(year, month)[1])
    return datetime(year, month, day, tzinfo=timezone.utc)


def _shift_month(year: int, month: int, delta: int) -> tuple[int, int]:
    idx = year * 12 + (month - 1) + delta
    return idx // 12, idx % 12 + 1


@dataclass(frozen=True)
class Cycle:
    start: datetime
    end: datetime  # exclusive; the next refill moment

    def contains(self, ts: datetime) -> bool:
        return self.start <= ts < self.end

    def days_to_refill(self, now: datetime) -> int:
        return max((self.end.date() - now.astimezone(timezone.utc).date()).days, 0)

    @property
    def label(self) -> str:
        last = self.end - timedelta(days=1)
        return f"{self.start.date().isoformat()}..{last.date().isoformat()}"


def cycle_for(now: datetime, reset_day: int = 29) -> Cycle:
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    now = now.astimezone(timezone.utc)
    this_month = _boundary(now.year, now.month, reset_day)
    if now >= this_month:
        start = this_month
        ny, nm = _shift_month(now.year, now.month, 1)
    else:
        py, pm = _shift_month(now.year, now.month, -1)
        start = _boundary(py, pm, reset_day)
        ny, nm = now.year, now.month
    return Cycle(start=start, end=_boundary(ny, nm, reset_day))


def resolve_cycle(cycle: str | datetime | Cycle, now: datetime, reset_day: int) -> Cycle:
    """``cycle`` is 'current', 'previous', a Cycle, a datetime, or an ISO date
    inside the wanted cycle."""
    if isinstance(cycle, Cycle):
        return cycle
    if isinstance(cycle, datetime):
        return cycle_for(cycle, reset_day)
    if cycle == "current":
        return cycle_for(now, reset_day)
    if cycle == "previous":
        cur = cycle_for(now, reset_day)
        return cycle_for(cur.start - timedelta(seconds=1), reset_day)
    return cycle_for(datetime.fromisoformat(cycle), reset_day)


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class Settings:
    """Runtime settings; tests build one directly instead of using env vars."""

    db: Path = field(default_factory=db_path)
    caps_file: Path | None = field(default_factory=caps_path)
    alert_file: Path = field(default_factory=alert_path)
