"""Worst-case cost estimation before a request is sent."""

from __future__ import annotations

import json
import math
from typing import Any, Iterable, Mapping

from .pricing import (
    BATCH_DISCOUNT,
    CACHE_WRITE_1H_MULTIPLIER,
    CACHE_WRITE_5M_MULTIPLIER,
    FALLBACK_CANDIDATES,
    MTOK,
    price_for,
)

CHARS_PER_TOKEN = 4  # heuristic from the spec; count_tokens is used when it matters
_MEDIA_TYPES = {"image", "document", "search_result", "container_upload"}
_INPUT_KEYS = ("system", "messages", "tools")


def _walk(obj: Any) -> Iterable[Any]:
    yield obj
    if isinstance(obj, Mapping):
        for v in obj.values():
            yield from _walk(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _walk(v)


def _plain(obj: Any) -> Any:
    """Turn SDK pydantic objects into plain data so they can be measured."""
    if hasattr(obj, "model_dump"):
        return obj.model_dump(exclude_none=True)
    if isinstance(obj, Mapping):
        return {k: _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    return obj


def has_media(params: Mapping[str, Any]) -> bool:
    for node in _walk(_plain({k: params.get(k) for k in _INPUT_KEYS})):
        if isinstance(node, Mapping) and node.get("type") in _MEDIA_TYPES:
            return True
    return False


def heuristic_input_tokens(params: Mapping[str, Any]) -> int:
    payload = _plain({k: params[k] for k in _INPUT_KEYS if k in params})
    text = json.dumps(payload, ensure_ascii=False, default=str)
    return math.ceil(len(text) / CHARS_PER_TOKEN)


def cache_write_multiplier(params: Mapping[str, Any]) -> float:
    """Worst case: every input token is a cache write at the longest TTL used."""
    mult = 1.0
    for node in _walk(_plain(dict(params))):
        if isinstance(node, Mapping) and "cache_control" in node and node["cache_control"]:
            cc = node["cache_control"]
            ttl = cc.get("ttl") if isinstance(cc, Mapping) else None
            mult = max(mult, CACHE_WRITE_1H_MULTIPLIER if ttl == "1h" else CACHE_WRITE_5M_MULTIPLIER)
    return mult


def worst_case_usd(
    model: str,
    input_tokens: int,
    max_tokens: int,
    *,
    cache_multiplier: float = 1.0,
    batch: bool = False,
    fallbacks: bool = False,
) -> float:
    """List-price upper bound: all input priced as (cache-write) input, plus
    the full ``max_tokens`` as output. With refusal fallbacks enabled the
    bound uses the most expensive model the request could be served by."""
    candidates = [model, *(FALLBACK_CANDIDATES.get(model, ()) if fallbacks else ())]
    worst = 0.0
    for m in candidates:
        p = price_for(m)
        worst = max(worst, (input_tokens * p.input * cache_multiplier + max_tokens * p.output) / MTOK)
    return worst * (BATCH_DISCOUNT if batch else 1.0)
