"""The budget-enforcing wrapper around the Anthropic client."""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Callable, Iterable, Mapping

import anthropic

from . import config
from .alerts import maybe_write_alert
from .config import Caps, Cycle, Settings, cycle_for, load_caps, resolve_cycle, utcnow
from .errors import BatchNotReady, BudgetUnavailable, wrap_sdk_error
from .estimate import cache_write_multiplier, has_media, heuristic_input_tokens, worst_case_usd
from .ledger import Ledger
from .pricing import TokenUsage, message_cost, price_for

log = logging.getLogger(__name__)

# Models that get adaptive thinking and refusal fallbacks by default.
ADAPTIVE_MODELS = frozenset({"claude-opus-5-5", "claude-sonnet-5-5"})
FALLBACK_MODELS = frozenset({"claude-opus-5-5", "claude-sonnet-5-5"})
# Haiku 4.5 rejects output_config.effort; every other current model takes it.
NO_EFFORT_MODELS = frozenset({"claude-haiku-4-5"})

FALLBACK_BETA = "server-side-fallback-2026-07-01"
DEFAULT_EFFORT = "medium"
_MISSING = object()


def apply_defaults(
    params: Mapping[str, Any],
    *,
    default_effort: str = DEFAULT_EFFORT,
    allow_fallbacks: bool = True,
) -> tuple[dict[str, Any], bool]:
    """Fill in thinking / effort / fallbacks the caller did not set.

    Returns ``(params, use_beta)``. ``use_beta`` is True when the request
    needs ``client.beta.messages`` (fallbacks or caller-supplied betas).
    Passing ``fallbacks=None`` opts a single call out of refusal fallbacks.
    """
    p = dict(params)
    model = p["model"]

    if model in ADAPTIVE_MODELS and "thinking" not in p:
        p["thinking"] = {"type": "adaptive"}

    if model not in NO_EFFORT_MODELS:
        oc = dict(p.get("output_config") or {})
        oc.setdefault("effort", default_effort)
        p["output_config"] = oc

    fb = p.pop("fallbacks", _MISSING)
    if fb is _MISSING:
        fb = "default" if (allow_fallbacks and model in FALLBACK_MODELS) else None
    if fb is not None:
        if not allow_fallbacks:
            raise ValueError("fallbacks are not accepted by the Batches API")
        p["fallbacks"] = fb
        betas = list(p.get("betas") or [])
        # The scalar "default" form needs the -07-01 header; the array form
        # needs -06-01 and is left to the caller to declare.
        if fb == "default" and FALLBACK_BETA not in betas:
            betas.append(FALLBACK_BETA)
        p["betas"] = betas

    use_beta = bool(p.get("betas")) or "fallbacks" in p
    return p, use_beta


class ClaudeBudget:
    """Budgeted access to the Claude API.

    Parameters are injectable for tests: ``client`` (any object shaped like
    ``anthropic.Anthropic``), ``caps`` (a fixed Caps), ``clock`` (returns an
    aware UTC datetime), ``settings`` (ledger / caps / alert paths).
    ``count_tokens`` is ``"auto"`` (use the endpoint for media requests or
    when the heuristic estimate is above half the remaining headroom),
    ``True`` (always) or ``False`` (never).
    """

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        client: Any = None,
        caps: Caps | None = None,
        clock: Callable[[], Any] = utcnow,
        count_tokens: bool | str = "auto",
        default_effort: str = DEFAULT_EFFORT,
    ) -> None:
        self.settings = settings or Settings()
        self.ledger = Ledger(self.settings.db)
        self._client = client
        self._caps = caps
        self._clock = clock
        self.count_tokens_mode = count_tokens
        self.default_effort = default_effort

    # -- plumbing ------------------------------------------------------------

    @property
    def caps(self) -> Caps:
        # Re-read every time so edits to caps.json apply without a restart.
        return self._caps or load_caps(self.settings.caps_file)

    def client(self) -> Any:
        if self._client is None:
            key = os.environ.get(config.ENV_API_KEY, "").strip()
            if not key:
                raise BudgetUnavailable(
                    f"{config.ENV_API_KEY} is not set; use the free chain"
                )
            # api_key is explicit, so the SDK skips ANTHROPIC_API_KEY /
            # ANTHROPIC_AUTH_TOKEN entirely; base_url is pinned so a
            # user-wide ANTHROPIC_BASE_URL (e.g. a LiteLLM proxy) is ignored.
            self._client = anthropic.Anthropic(
                api_key=key,
                base_url=os.environ.get(config.ENV_BASE_URL) or config.DEFAULT_BASE_URL,
            )
        return self._client

    def _cycle(self, caps: Caps) -> Cycle:
        return cycle_for(self._clock(), caps.reset_day)

    def _headroom(self, consumer: str, caps: Caps, cycle: Cycle) -> float:
        return min(
            caps.cap_for(consumer) - self.ledger.spent(cycle, consumer),
            caps.global_cap - self.ledger.spent(cycle),
        )

    def _estimate(
        self,
        client: Any,
        params: Mapping[str, Any],
        *,
        consumer: str,
        caps: Caps,
        cycle: Cycle,
        extra_input_tokens: int = 0,
    ) -> float:
        model = params["model"]
        max_tokens = int(params["max_tokens"])
        mult = cache_write_multiplier(params)
        fallbacks = "fallbacks" in params

        def bound(n: int) -> float:
            return worst_case_usd(model, n, max_tokens, cache_multiplier=mult, fallbacks=fallbacks)

        heuristic = heuristic_input_tokens(params) + extra_input_tokens
        estimate = bound(heuristic)

        use = self.count_tokens_mode
        if use == "auto":
            use = has_media(params) or estimate > 0.5 * self._headroom(consumer, caps, cycle)
        if use:
            count_kw = {k: params[k] for k in ("messages", "system", "tools") if k in params}
            try:
                counted = client.messages.count_tokens(model=model, **count_kw).input_tokens
                estimate = bound(int(counted) + extra_input_tokens)
            except anthropic.APIError as exc:
                log.info("claude_budget: count_tokens failed (%s); using char/4 heuristic", exc)
        return estimate

    def _after_spend(self, caps: Caps, cycle: Cycle) -> None:
        maybe_write_alert(
            self.settings.alert_file,
            spent=self.ledger.spent(cycle),
            cap=caps.global_cap,
            cycle=cycle,
            now=self._clock(),
        )

    # -- messages ------------------------------------------------------------

    def call(self, consumer: str, **kwargs: Any) -> Any:
        """Budgeted ``messages.create``. ``stream=True`` streams and returns
        the final message. Returns ``Message`` (or ``BetaMessage`` when
        fallbacks / betas are in play)."""
        return self._run(consumer, kwargs, output_format=None)

    def parse(self, consumer: str, output_format: Any, **kwargs: Any) -> Any:
        """Budgeted structured output: ``messages.parse(output_format=Model)``.
        The validated object is on ``.parsed_output``."""
        if output_format is None:
            raise TypeError("parse() needs an output_format (a Pydantic model class)")
        return self._run(consumer, kwargs, output_format=output_format)

    def _run(self, consumer: str, kwargs: dict[str, Any], *, output_format: Any) -> Any:
        if "model" not in kwargs or "max_tokens" not in kwargs:
            raise TypeError("model and max_tokens are required (max_tokens bounds the budget check)")
        price_for(kwargs["model"])  # UnknownModel before anything else
        stream = bool(kwargs.pop("stream", False))
        client = self.client()  # BudgetUnavailable before any ledger write
        caps = self.caps
        caps.cap_for(consumer)  # UnknownConsumer early
        cycle = self._cycle(caps)

        params, use_beta = apply_defaults(kwargs, default_effort=self.default_effort)
        extra = 0
        if output_format is not None and hasattr(output_format, "model_json_schema"):
            extra = len(str(output_format.model_json_schema())) // 4

        estimate = self._estimate(client, params, consumer=consumer, caps=caps, cycle=cycle,
                                  extra_input_tokens=extra)
        row = self.ledger.reserve(consumer=consumer, model=params["model"], estimate_usd=estimate,
                                  caps=caps, cycle=cycle, now=self._clock())

        api = client.beta.messages if use_beta else client.messages
        if output_format is not None:
            params["output_format"] = output_format
        request_id = None
        try:
            if stream:
                with api.stream(**params) as s:
                    message = s.get_final_message()
                    request_id = getattr(s, "request_id", None)
            elif output_format is not None:
                message = api.parse(**params)
            else:
                message = api.create(**params)
        except anthropic.APIStatusError as exc:
            # The API answered with an error status: nothing was generated.
            self.ledger.finalize(row, status="error", cost_usd=0.0,
                                 request_id=getattr(exc, "request_id", None), error=str(exc)[:500])
            raise wrap_sdk_error(exc) from exc
        except BaseException as exc:
            # Timeout, dropped connection or interrupted stream: tokens may have
            # been billed, so keep the worst case on the books.
            self.ledger.finalize(row, status="error", cost_usd=estimate, error=repr(exc)[:500])
            if isinstance(exc, anthropic.APIError):
                raise wrap_sdk_error(exc) from exc
            raise

        cost, usage, _ = message_cost(params["model"], message)
        request_id = request_id or getattr(message, "_request_id", None) or getattr(message, "id", None)
        self.ledger.finalize(row, status="ok", cost_usd=cost, usage=usage,
                             served_model=getattr(message, "model", None), request_id=request_id)
        self._after_spend(caps, cycle)
        return message

    # -- batches -------------------------------------------------------------

    def batch_submit(self, consumer: str, requests: Iterable[Mapping[str, Any]]) -> str:
        """Submit a Message Batch (50% price). Each request is
        ``{"custom_id": str, "params": {...messages.create kwargs...}}``.
        The whole batch's worst case is reserved against the caps until
        ``batch_collect`` records the actual per-result cost."""
        client = self.client()
        caps = self.caps
        caps.cap_for(consumer)
        cycle = self._cycle(caps)

        prepared: list[dict[str, Any]] = []
        estimate = 0.0
        for req in requests:
            params = dict(req["params"])
            if "betas" in params or params.get("stream"):
                raise ValueError("batch params cannot carry betas or stream")
            if "max_tokens" not in params:
                raise TypeError(f"{req['custom_id']}: max_tokens is required")
            price_for(params["model"])
            params, _ = apply_defaults(params, default_effort=self.default_effort, allow_fallbacks=False)
            estimate += worst_case_usd(
                params["model"], heuristic_input_tokens(params), int(params["max_tokens"]),
                cache_multiplier=cache_write_multiplier(params), batch=True,
            )
            prepared.append({"custom_id": req["custom_id"], "params": params})
        if not prepared:
            raise ValueError("batch_submit needs at least one request")

        models = sorted({r["params"]["model"] for r in prepared})
        row = self.ledger.reserve(consumer=consumer, model=",".join(models), estimate_usd=estimate,
                                  caps=caps, cycle=cycle, now=self._clock(), batch=True)
        try:
            batch = client.messages.batches.create(requests=prepared)
        except anthropic.APIError as exc:
            self.ledger.finalize(row, status="error", cost_usd=0.0, error=str(exc)[:500])
            raise wrap_sdk_error(exc) from exc
        self.ledger.attach_batch(row, batch.id)
        return batch.id

    def batch_collect(
        self,
        batch_id: str,
        *,
        consumer: str | None = None,
        wait: bool = False,
        poll_interval: float = 30.0,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """Fetch results of an ended batch, record each result's cost at the
        batch price, and release the reservation. Returns
        ``{custom_id: result}`` where ``result.type`` is succeeded / errored /
        canceled / expired. Safe to call twice; costs are recorded once."""
        client = self.client()
        deadline = None if timeout is None else time.monotonic() + timeout
        try:
            while True:
                batch = client.messages.batches.retrieve(batch_id)
                if batch.processing_status == "ended":
                    break
                if not wait or (deadline is not None and time.monotonic() >= deadline):
                    raise BatchNotReady(f"batch {batch_id} is {batch.processing_status}")
                time.sleep(poll_interval)
            raw = list(client.messages.batches.results(batch_id))
        except anthropic.APIError as exc:
            raise wrap_sdk_error(exc) from exc

        reservation = self.ledger.batch_reservation(batch_id)
        consumer = consumer or (reservation["consumer"] if reservation is not None else None)
        if consumer is None:
            raise ValueError(f"no reservation for batch {batch_id}; pass consumer=")
        default_model = reservation["model"] if reservation is not None else "unknown"

        out: dict[str, Any] = {}
        rows: list[dict[str, object]] = []
        for item in raw:
            res = item.result
            out[item.custom_id] = res
            if res.type == "succeeded":
                msg = res.message
                cost, usage, _ = message_cost(msg.model, msg, batch=True)
                rows.append({"consumer": consumer, "model": msg.model, "served_model": msg.model,
                             "usage": usage, "cost_usd": cost, "request_id": msg.id,
                             "custom_id": item.custom_id, "status": "ok"})
            else:
                err = getattr(res, "error", None)
                rows.append({"consumer": consumer, "model": default_model, "usage": TokenUsage(),
                             "cost_usd": 0.0, "custom_id": item.custom_id, "status": "error",
                             "error": f"{res.type}: {err}"[:500] if err else res.type})
        self.ledger.settle_batch(batch_id, rows, now=self._clock())
        caps = self.caps
        self._after_spend(caps, self._cycle(caps))
        return out

    # -- reporting -----------------------------------------------------------

    def spend(self, consumer: str | None = None, cycle: str | Any = "current") -> dict[str, Any]:
        caps = self.caps
        now = self._clock()
        cyc = resolve_cycle(cycle, now, caps.reset_day)
        per = self.ledger.by_consumer(cyc)
        cycle_info = {
            "start": cyc.start.date().isoformat(),
            "end": cyc.end.date().isoformat(),
            "label": cyc.label,
            "refill_date": cyc.end.date().isoformat(),
            "days_to_refill": cyc.days_to_refill(now),
        }

        def row(name: str) -> dict[str, Any]:
            s = per.get(name)
            cap = caps.consumers.get(name)
            spent = s.spent_usd if s else 0.0
            return {
                "consumer": name,
                "spent_usd": round(spent, 6),
                "cap_usd": cap,
                "remaining_usd": None if cap is None else round(cap - spent, 6),
                "pct": None if not cap else round(spent / cap, 4),
                "reserved_usd": round(s.reserved_usd, 6) if s else 0.0,
                "calls": s.calls if s else 0,
                "errors": s.errors if s else 0,
                "open_reservations": s.open_reservations if s else 0,
            }

        if consumer is not None:
            return {**row(consumer), "cycle": cycle_info}

        names = list(caps.consumers) + sorted(set(per) - set(caps.consumers))
        total = sum(s.spent_usd for s in per.values())
        return {
            "cycle": cycle_info,
            "global": {
                "spent_usd": round(total, 6),
                "cap_usd": caps.global_cap,
                "remaining_usd": round(caps.global_cap - total, 6),
                "pct": round(total / caps.global_cap, 4) if caps.global_cap else None,
                "alert": total >= config.ALERT_THRESHOLD * caps.global_cap,
            },
            "consumers": {n: row(n) for n in names},
            "caps_source": caps.source,
            "ledger": str(self.settings.db),
        }
