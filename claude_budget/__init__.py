"""claude_budget: call the Anthropic API under a hard monthly budget.

Module-level functions use a lazily built default ``ClaudeBudget`` that
reads its settings from the environment (``DEX_ANTHROPIC_API_KEY``,
``CLAUDE_BUDGET_DB``, ``CLAUDE_BUDGET_CAPS``, ``CLAUDE_BUDGET_ALERT_PATH``).
"""

from __future__ import annotations

import threading
from typing import Any, Iterable, Mapping

from .budget import ClaudeBudget, apply_defaults
from .config import Caps, Cycle, Settings, cycle_for, load_caps
from .errors import (
    BatchNotReady,
    BudgetError,
    BudgetExceeded,
    BudgetUnavailable,
    ClaudeAPIError,
    CredentialError,
    NonRetryableAPIError,
    RetryableAPIError,
    UnknownConsumer,
    UnknownModel,
)
from .pricing import PRICING, PRICING_AS_OF, TokenUsage, cost_usd

__version__ = "0.1.0"

_default: ClaudeBudget | None = None
_lock = threading.Lock()


def default_budget() -> ClaudeBudget:
    global _default
    with _lock:
        if _default is None:
            _default = ClaudeBudget()
        return _default


def configure(budget: ClaudeBudget | None = None, **kwargs: Any) -> ClaudeBudget:
    """Replace the default instance (``configure(count_tokens=False)``, or
    pass a ready ``ClaudeBudget``)."""
    global _default
    with _lock:
        _default = budget or ClaudeBudget(**kwargs)
        return _default


def call(consumer: str, **kwargs: Any) -> Any:
    return default_budget().call(consumer, **kwargs)


def parse(consumer: str, output_format: Any, **kwargs: Any) -> Any:
    return default_budget().parse(consumer, output_format=output_format, **kwargs)


def batch_submit(consumer: str, requests: Iterable[Mapping[str, Any]]) -> str:
    return default_budget().batch_submit(consumer, requests)


def batch_collect(batch_id: str, **kwargs: Any) -> dict[str, Any]:
    return default_budget().batch_collect(batch_id, **kwargs)


def spend(consumer: str | None = None, cycle: str = "current") -> dict[str, Any]:
    return default_budget().spend(consumer=consumer, cycle=cycle)


__all__ = [
    "BatchNotReady", "BudgetError", "BudgetExceeded", "BudgetUnavailable", "Caps",
    "ClaudeAPIError", "ClaudeBudget", "CredentialError", "Cycle", "NonRetryableAPIError",
    "PRICING", "PRICING_AS_OF", "RetryableAPIError", "Settings", "TokenUsage",
    "UnknownConsumer", "UnknownModel", "apply_defaults", "batch_collect", "batch_submit",
    "call", "configure", "cost_usd", "cycle_for", "default_budget", "load_caps", "parse",
    "spend",
]
