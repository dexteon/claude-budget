"""Shared fixtures: a fake Anthropic client and an isolated ClaudeBudget.

No network and no API key are ever used.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import anthropic
import pytest

from claude_budget import ClaudeBudget, Settings
from claude_budget.config import Caps, DEFAULT_CAPS

NOW = datetime(2026, 10, 7, 12, 0, tzinfo=timezone.utc)


def make_usage(inp: int = 1000, out: int = 500, cache_read: int = 0, cache_write: int = 0,
               iterations: list[Any] | None = None) -> anthropic.types.Usage | SimpleNamespace:
    if iterations is not None:
        return SimpleNamespace(input_tokens=inp, output_tokens=out, cache_read_input_tokens=cache_read,
                               cache_creation_input_tokens=cache_write, cache_creation=None,
                               iterations=iterations)
    return anthropic.types.Usage(input_tokens=inp, output_tokens=out, cache_read_input_tokens=cache_read,
                                 cache_creation_input_tokens=cache_write)


def make_message(model: str = "claude-haiku-4-5", *, msg_id: str = "msg_test", **usage_kw: Any) -> SimpleNamespace:
    return SimpleNamespace(
        id=msg_id,
        type="message",
        model=model,
        content=[SimpleNamespace(type="text", text="ok")],
        stop_reason="end_turn",
        usage=make_usage(**usage_kw),
        _request_id="req_test",
        parsed_output=None,
    )


class FakeStream:
    def __init__(self, message: Any) -> None:
        self._message = message
        self.request_id = "req_stream"

    def __enter__(self) -> "FakeStream":
        return self

    def __exit__(self, *exc: Any) -> None:
        return None

    def get_final_message(self) -> Any:
        return self._message


class FakeMessages:
    def __init__(self, owner: "FakeClient", label: str) -> None:
        self.owner = owner
        self.label = label

    def _respond(self, method: str, kwargs: dict[str, Any]) -> Any:
        self.owner.calls.append((self.label, method, kwargs))
        if self.owner.raise_exc is not None:
            raise self.owner.raise_exc
        return self.owner.response or make_message(kwargs["model"])

    def create(self, **kwargs: Any) -> Any:
        return self._respond("create", kwargs)

    def parse(self, **kwargs: Any) -> Any:
        msg = self._respond("parse", kwargs)
        msg.parsed_output = "parsed"
        return msg

    def stream(self, **kwargs: Any) -> FakeStream:
        return FakeStream(self._respond("stream", kwargs))

    def count_tokens(self, **kwargs: Any) -> Any:
        self.owner.calls.append((self.label, "count_tokens", kwargs))
        return SimpleNamespace(input_tokens=self.owner.counted_tokens)


class FakeBatches:
    def __init__(self, owner: "FakeClient") -> None:
        self.owner = owner
        self.status = "in_progress"
        self.results_list: list[Any] = []
        self.submitted: list[Any] = []

    def create(self, *, requests: list[Any]) -> Any:
        self.submitted.append(requests)
        return SimpleNamespace(id="msgbatch_test", processing_status="in_progress")

    def retrieve(self, batch_id: str) -> Any:
        return SimpleNamespace(id=batch_id, processing_status=self.status)

    def results(self, batch_id: str) -> Any:
        return iter(self.results_list)


class FakeClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict[str, Any]]] = []
        self.response: Any = None
        self.raise_exc: BaseException | None = None
        self.counted_tokens = 100
        self.messages = FakeMessages(self, "messages")
        self.messages.batches = FakeBatches(self)
        self.beta = SimpleNamespace(messages=FakeMessages(self, "beta"))


class Clock:
    def __init__(self, now: datetime = NOW) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # Never touch a real key or the real ledger / alert paths.
    monkeypatch.delenv("DEX_ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("DEX_ANTHROPIC_GATEWAY_KEY", raising=False)
    monkeypatch.delenv("DEX_ANTHROPIC_BASE_URL", raising=False)
    monkeypatch.setenv("CLAUDE_BUDGET_DB", str(tmp_path / "env-ledger.db"))
    monkeypatch.setenv("CLAUDE_BUDGET_ALERT_PATH", str(tmp_path / "env-alert.json"))
    monkeypatch.delenv("CLAUDE_BUDGET_CAPS", raising=False)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(db=tmp_path / "ledger.db", caps_file=None, alert_file=tmp_path / "alert.json")


@pytest.fixture
def fake_client() -> FakeClient:
    return FakeClient()


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def make_budget(settings: Settings, fake_client: FakeClient, clock: Clock):
    def _make(caps: dict[str, Any] | None = None, **kw: Any) -> ClaudeBudget:
        kw.setdefault("count_tokens", False)
        return ClaudeBudget(
            settings=settings,
            client=fake_client,
            caps=Caps.from_dict(caps or DEFAULT_CAPS),
            clock=clock,
            **kw,
        )

    return _make
