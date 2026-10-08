"""Exception hierarchy for claude_budget.

Callers that have a free-model fallback chain should catch
``(BudgetExceeded, BudgetUnavailable)``. That covers:

* the budget refusing the call (``BudgetExceeded``),
* no key configured (``BudgetUnavailable``),
* transient API failures after the SDK's own retries are exhausted
  (``RetryableAPIError``, which is also a ``BudgetUnavailable``),
* a bad or revoked key, billing problems (``CredentialError``, also a
  ``BudgetUnavailable``).

A malformed request (HTTP 400/404/413/422) raises ``NonRetryableAPIError``
which is deliberately NOT a ``BudgetUnavailable``: it is a bug in the caller
and should surface rather than silently fall back.
"""

from __future__ import annotations

from typing import Any


class BudgetError(Exception):
    """Base class for every error raised by claude_budget."""


class BudgetUnavailable(BudgetError):
    """Claude cannot be used right now; the caller should use its free chain."""


class BudgetExceeded(BudgetError):
    """The worst-case cost of the call would cross a consumer or global cap."""

    def __init__(
        self,
        message: str,
        *,
        consumer: str,
        scope: str,
        estimate_usd: float,
        spent_usd: float,
        cap_usd: float,
    ) -> None:
        super().__init__(message)
        self.consumer = consumer
        self.scope = scope  # "consumer" or "global"
        self.estimate_usd = estimate_usd
        self.spent_usd = spent_usd
        self.cap_usd = cap_usd


class UnknownConsumer(BudgetUnavailable):
    """The consumer name has no cap in caps.json."""


class UnknownModel(BudgetError, ValueError):
    """The model has no entry in the pricing table, so cost cannot be bounded."""


class BatchNotReady(BudgetError):
    """batch_collect was called before the batch reached ``ended``."""


class ClaudeAPIError(BudgetError):
    """Wraps an ``anthropic`` SDK exception. ``original`` holds the SDK error."""

    retryable: bool = False

    def __init__(self, message: str, *, original: BaseException, status_code: int | None = None,
                 request_id: str | None = None) -> None:
        super().__init__(message)
        self.original = original
        self.status_code = status_code
        self.request_id = request_id

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{type(self).__name__}(status={self.status_code}, request_id={self.request_id!r})"


class RetryableAPIError(ClaudeAPIError, BudgetUnavailable):
    """429, 5xx, 529 overloaded, timeouts, connection errors. Safe to retry later."""

    retryable = True


class NonRetryableAPIError(ClaudeAPIError):
    """4xx errors caused by the request itself. Retrying unchanged will fail again."""

    retryable = False


class CredentialError(NonRetryableAPIError, BudgetUnavailable):
    """401/402/403: bad key, billing problem, or no access. Fall back to the free chain."""


def wrap_sdk_error(exc: BaseException) -> ClaudeAPIError:
    """Translate an ``anthropic`` SDK exception into the claude_budget hierarchy."""
    import anthropic

    status: int | None = getattr(exc, "status_code", None)
    request_id: Any = getattr(exc, "request_id", None)
    msg = f"{type(exc).__name__}: {exc}"

    if isinstance(exc, (anthropic.APITimeoutError, anthropic.APIConnectionError)):
        return RetryableAPIError(msg, original=exc, status_code=status, request_id=request_id)
    if isinstance(exc, (anthropic.AuthenticationError, anthropic.PermissionDeniedError)):
        return CredentialError(msg, original=exc, status_code=status, request_id=request_id)
    if isinstance(exc, anthropic.RateLimitError):
        return RetryableAPIError(msg, original=exc, status_code=status, request_id=request_id)
    if isinstance(exc, anthropic.APIStatusError):
        if status == 402:
            return CredentialError(msg, original=exc, status_code=status, request_id=request_id)
        if status is not None and (status >= 500 or status in (408, 409, 429)):
            return RetryableAPIError(msg, original=exc, status_code=status, request_id=request_id)
        return NonRetryableAPIError(msg, original=exc, status_code=status, request_id=request_id)
    return NonRetryableAPIError(msg, original=exc, status_code=status, request_id=request_id)
