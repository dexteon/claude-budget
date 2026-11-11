"""Price table and cost math.

Source: the ``claude-api`` skill (models table cached 2026-09-25, and
``shared/prompt-caching.md`` "Economics"):

* Opus 5.5 $4 / $20 per MTok, cache read $0.20 (0.05x input).
* Sonnet 5.5 $2 / $10 per MTok, cache read $0.20 (0.1x input).
* Haiku 4.5 $1 / $5 per MTok, cache read 0.1x input = $0.10.
* Cache writes: 1.25x input for the 5 minute TTL, 2x input for the 1 hour TTL.
* Batch API: 50% of every line item.

The older / fallback models are listed so that a server-side refusal
fallback (which can be served by e.g. ``claude-opus-4-8``) is still priced.
Their cache-read price uses the standard 0.1x multiplier.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .errors import UnknownModel

PRICING_AS_OF = "2026-09-25"
MTOK = 1_000_000

CACHE_WRITE_5M_MULTIPLIER = 1.25
CACHE_WRITE_1H_MULTIPLIER = 2.0
BATCH_DISCOUNT = 0.5


@dataclass(frozen=True)
class ModelPrice:
    input: float  # USD per MTok, uncached input
    output: float  # USD per MTok
    cache_read: float  # USD per MTok

    @property
    def cache_write_5m(self) -> float:
        return self.input * CACHE_WRITE_5M_MULTIPLIER

    @property
    def cache_write_1h(self) -> float:
        return self.input * CACHE_WRITE_1H_MULTIPLIER


PRICING: dict[str, ModelPrice] = {
    # Primary models used by the pipelines.
    "claude-opus-5-5": ModelPrice(input=4.00, output=20.00, cache_read=0.20),
    "claude-sonnet-5-5": ModelPrice(input=2.00, output=10.00, cache_read=0.20),
    "claude-haiku-4-5": ModelPrice(input=1.00, output=5.00, cache_read=0.10),
    # Possible refusal-fallback targets and other current models.
    "claude-fable-5-1": ModelPrice(input=10.00, output=50.00, cache_read=0.25),
    "claude-fable-5": ModelPrice(input=10.00, output=50.00, cache_read=1.00),
    "claude-opus-5": ModelPrice(input=5.00, output=25.00, cache_read=0.50),
    "claude-opus-4-8": ModelPrice(input=5.00, output=25.00, cache_read=0.50),
    "claude-opus-4-7": ModelPrice(input=5.00, output=25.00, cache_read=0.50),
    "claude-opus-4-6": ModelPrice(input=5.00, output=25.00, cache_read=0.50),
    "claude-sonnet-5": ModelPrice(input=2.00, output=10.00, cache_read=0.20),
    "claude-sonnet-4-6": ModelPrice(input=3.00, output=15.00, cache_read=0.30),
}

# Models that the server-side ``fallbacks: "default"`` mode may route to.
# Used only to bound the worst case of a call with fallbacks enabled.
FALLBACK_CANDIDATES: dict[str, tuple[str, ...]] = {
    "claude-opus-5-5": ("claude-opus-5", "claude-opus-4-8"),
    "claude-sonnet-5-5": ("claude-opus-5", "claude-opus-4-8"),
}

# The most expensive row, used to price an unexpected served model so that
# an unknown fallback never under-counts.
_MOST_EXPENSIVE = max(PRICING.values(), key=lambda p: p.output)


def price_for(model: str, *, strict: bool = True) -> ModelPrice:
    """Return the price row for ``model``.

    Accepts exact IDs and Bedrock-style ``anthropic.`` prefixes. With
    ``strict=False`` an unknown model is priced at the most expensive row.
    """
    key = model.removeprefix("anthropic.")
    if key in PRICING:
        return PRICING[key]
    if strict:
        raise UnknownModel(f"No price for model {model!r}; add it to claude_budget.pricing.PRICING")
    return _MOST_EXPENSIVE


@dataclass(frozen=True)
class TokenUsage:
    input_tokens: int = 0
    cache_creation_5m_tokens: int = 0
    cache_creation_1h_tokens: int = 0
    cache_read_tokens: int = 0
    output_tokens: int = 0

    @property
    def cache_creation_tokens(self) -> int:
        return self.cache_creation_5m_tokens + self.cache_creation_1h_tokens

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        return TokenUsage(
            self.input_tokens + other.input_tokens,
            self.cache_creation_5m_tokens + other.cache_creation_5m_tokens,
            self.cache_creation_1h_tokens + other.cache_creation_1h_tokens,
            self.cache_read_tokens + other.cache_read_tokens,
            self.output_tokens + other.output_tokens,
        )


def cost_usd(model: str, usage: TokenUsage, *, batch: bool = False, strict: bool = True) -> float:
    """Exact list-price cost of one attempt."""
    p = price_for(model, strict=strict)
    total = (
        usage.input_tokens * p.input
        + usage.cache_creation_5m_tokens * p.cache_write_5m
        + usage.cache_creation_1h_tokens * p.cache_write_1h
        + usage.cache_read_tokens * p.cache_read
        + usage.output_tokens * p.output
    ) / MTOK
    if batch:
        total *= BATCH_DISCOUNT
    return total


def _get(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def usage_from_sdk(u: Any) -> TokenUsage:
    """Build a TokenUsage from an SDK ``Usage`` / ``BetaUsage`` / iteration entry.

    ``cache_creation`` (when present) splits writes into the 5 minute and
    1 hour TTL buckets; otherwise every write is priced as 5 minute.
    """
    input_tokens = int(_get(u, "input_tokens", 0) or 0)
    output_tokens = int(_get(u, "output_tokens", 0) or 0)
    read = int(_get(u, "cache_read_input_tokens", 0) or 0)
    created = int(_get(u, "cache_creation_input_tokens", 0) or 0)
    breakdown = _get(u, "cache_creation")
    c5 = int(_get(breakdown, "ephemeral_5m_input_tokens", 0) or 0)
    c1 = int(_get(breakdown, "ephemeral_1h_input_tokens", 0) or 0)
    if breakdown is None or (c5 + c1) != created:
        # No breakdown, or it disagrees with the total: price the remainder at 5m.
        c5 = max(created - c1, 0)
    return TokenUsage(input_tokens, c5, c1, read, output_tokens)


@dataclass(frozen=True)
class AttemptCost:
    model: str
    usage: TokenUsage
    cost_usd: float


def message_cost(requested_model: str, message: Any, *, batch: bool = False) -> tuple[float, TokenUsage, list[AttemptCost]]:
    """Cost of a returned Message, including every refusal-fallback attempt.

    Per the skill docs, ``usage.iterations`` is the per-attempt source of
    truth when present (top-level usage then covers only the attempt that
    produced the message). Each iteration carries its own ``model``; an
    iteration with no model is billed at the requested model.
    """
    usage = _get(message, "usage")
    iterations = _get(usage, "iterations") or []
    served_model = _get(message, "model") or requested_model
    attempts: list[AttemptCost] = []
    if iterations:
        for it in iterations:
            m = _get(it, "model") or requested_model
            tu = usage_from_sdk(it)
            attempts.append(AttemptCost(m, tu, cost_usd(m, tu, batch=batch, strict=False)))
    else:
        tu = usage_from_sdk(usage)
        attempts.append(AttemptCost(served_model, tu, cost_usd(served_model, tu, batch=batch, strict=False)))
    total_usage = TokenUsage()
    for a in attempts:
        total_usage = total_usage + a.usage
    return sum(a.cost_usd for a in attempts), total_usage, attempts
