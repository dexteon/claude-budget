# claude-budget

A small shared library through which every pipeline calls the Anthropic Claude API, under a hard monthly budget. A global cap and per-consumer caps live in `caps.json` (the example caps $190 a month against a $200 API credit that resets on a set day), every call is checked against its worst-case cost before it is sent, and every call is logged to a SQLite ledger.

## Install into another venv

```powershell
D:\path\to\other\.venv\Scripts\python.exe -m pip install -e D:\DexProjects\claude-budget
```

Or from an activated venv: `pip install -e D:\DexProjects\claude-budget`. Requires Python 3.10+ and `anthropic` 1.x (pulled in automatically).

## The key rule

The key is read **only** from `DEX_ANTHROPIC_API_KEY` and passed explicitly to `anthropic.Anthropic(api_key=...)`.

**Never set `ANTHROPIC_API_KEY`** (user-wide or machine-wide). Claude Code and every `claude -p` job would silently switch from the Max subscription onto this credit. The library never reads that variable. Because the key is passed explicitly, the SDK also ignores `ANTHROPIC_AUTH_TOKEN`. The base URL is pinned to `https://api.anthropic.com`, so a user-wide `ANTHROPIC_BASE_URL` (a LiteLLM proxy, say) cannot redirect these calls. To override it on purpose, set `DEX_ANTHROPIC_BASE_URL`.

If `DEX_ANTHROPIC_API_KEY` is missing, every call raises `BudgetUnavailable` before anything else happens.

## Calling it, with a free-chain fallback

```python
from claude_budget import call, BudgetExceeded, BudgetUnavailable

def summarize(text: str) -> str:
    try:
        msg = call(
            "portal-chat",                      # consumer name from caps.json
            model="claude-sonnet-5-5",
            max_tokens=2000,                    # required: it bounds the budget check
            system=[{"type": "text", "text": STABLE_PROMPT, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": text}],
        )
    except (BudgetExceeded, BudgetUnavailable):
        return free_chain(text)                 # Gemini / local / whatever the pipeline had before
    if msg.stop_reason == "refusal":
        return free_chain(text)
    return next(b.text for b in msg.content if b.type == "text")
```

`(BudgetExceeded, BudgetUnavailable)` covers every case where the free chain is the right answer:

| Raised | When | Is a `BudgetUnavailable` |
|---|---|---|
| `BudgetExceeded` | the worst case would cross the consumer or global cap | (own class, catch it too) |
| `BudgetUnavailable` | no `DEX_ANTHROPIC_API_KEY` | yes |
| `UnknownConsumer` | consumer name not in caps.json | yes |
| `RetryableAPIError` | 429, 5xx, 529 overloaded, timeout, connection error (after the SDK's own 2 retries) | yes |
| `CredentialError` | 401 / 402 / 403 | yes |
| `NonRetryableAPIError` | 400 / 404 / 413 / 422: the request itself is wrong | **no**, it surfaces as a bug |
| `UnknownModel` | model not in the price table | no (also a `ValueError`) |

Every API error has `.retryable`, `.status_code`, `.request_id` and `.original` (the SDK exception).

### Streaming

```python
msg = call("trading", model="claude-sonnet-5-5", max_tokens=64000, stream=True, messages=[...])
```

`stream=True` uses the SDK streaming helper and returns the final message (`get_final_message()`). Use it for large `max_tokens`.

### Structured output

```python
from pydantic import BaseModel
from claude_budget import parse

class Verdict(BaseModel):
    clearance_required: bool
    remote: bool

msg = parse("job-classify", output_format=Verdict, model="claude-haiku-4-5", max_tokens=300,
            messages=[{"role": "user", "content": posting}])
verdict = msg.parsed_output
```

### Batch API (50% price)

```python
from claude_budget import batch_submit, batch_collect, BatchNotReady

batch_id = batch_submit("job-classify", [
    {"custom_id": f"post-{i}", "params": {"model": "claude-haiku-4-5", "max_tokens": 300,
                                         "messages": [{"role": "user", "content": p}]}}
    for i, p in enumerate(postings)
])
# later (results stay available for 29 days):
try:
    results = batch_collect(batch_id)          # or batch_collect(batch_id, wait=True, timeout=3600)
except BatchNotReady:
    ...
for custom_id, r in results.items():
    if r.type == "succeeded":
        handle(custom_id, r.message)
```

`batch_submit` reserves the whole batch's worst case against the caps. `batch_collect` records each result's actual cost at the 50% price and releases the reservation. A second collect records nothing.

### Spend

```python
from claude_budget import spend
spend()                       # cycle info, global total, every consumer
spend("trading")              # one consumer
spend(cycle="previous")       # or a date inside the wanted cycle: "2026-09-15"
```

```powershell
python -m claude_budget status            # table: spend vs cap per consumer, global, days to refill
python -m claude_budget status --json     # same data as JSON (for the portal tile)
python -m claude_budget status --cycle previous
```

## Defaults applied when the caller does not set them

| Model | thinking | `output_config.effort` | refusal fallbacks |
|---|---|---|---|
| `claude-opus-5-5` | `{"type": "adaptive"}` | `"medium"` | `fallbacks="default"` + beta `server-side-fallback-2026-07-01` via `client.beta.messages` |
| `claude-sonnet-5-5` | `{"type": "adaptive"}` | `"medium"` | same |
| `claude-haiku-4-5` | none (pass `{"type": "enabled", "budget_tokens": N}` yourself if wanted) | not set (Haiku 4.5 rejects `effort`) | none |
| other priced models | untouched | `"medium"` | none |

Caller values always win. `output_config` is merged, so passing `{"format": ...}` keeps the default effort. Pass `fallbacks=None` to turn fallbacks off for one call. Fallbacks also apply to `parse` and streaming, because `client.beta.messages.parse` / `.stream` accept them. They are **not** applied to batches: the Batches API rejects the parameter.

Always check `msg.stop_reason == "refusal"` before reading content: a refusal can still come back if the fallback model also declines.

## How the budget is enforced

1. **Before the call** the library computes a worst case: input tokens at list price (multiplied by 1.25, or by 2 for a 1-hour TTL, when the request carries `cache_control`), plus the full `max_tokens` as output. With fallbacks enabled it prices at the most expensive model the request could be served by (Opus 5 / Opus 4.8 at $5 / $25). Input tokens come from a chars/4 heuristic. `client.messages.count_tokens` is called instead when the request has images or documents, or when the heuristic worst case exceeds half of the remaining headroom. `ClaudeBudget(count_tokens=True|False)` forces either way.
2. **Atomic check and reserve.** Inside one SQLite `BEGIN IMMEDIATE` transaction, the library checks the consumer cap and the global cap, then inserts a `reserved` row holding the estimate. Two processes can never both spend the same remaining headroom.
3. **After the call** the row becomes `ok`, with the actual cost from `response.usage`. When a refusal fallback ran, `usage.iterations` is summed, each attempt priced at its own model.
4. **Errors.** An HTTP error status is recorded as `error` at $0. A timeout, dropped connection or interrupted stream keeps the worst case on the books, since tokens may have been billed. A reservation left open by a killed process also keeps counting at its estimate. The CLI's `open` column shows these.

### 80% alert

After each recorded spend, if global spend for the cycle is at least 80% of the global cap, the library writes `G:\CyberDash\alerts-claude-budget.json` (override with `CLAUDE_BUDGET_ALERT_PATH`). The file is replaced atomically and looks like this:

```json
{"source": "claude-budget", "level": "warning", "message": "...", "spent_usd": 152.1, "cap_usd": 190.0,
 "pct": 0.8005, "threshold": 0.8, "cycle_start": "2026-10-29", "refill_date": "2026-11-29",
 "days_to_refill": 12, "generated_at": "..."}
```

`level` becomes `critical` at 100%. Check `cycle_start` to ignore an alert left over from a previous cycle. A failure to write the alert is logged and never breaks a call.

## Configuration

| Env var | Default | Meaning |
|---|---|---|
| `DEX_ANTHROPIC_API_KEY` | (none) | the API key; missing means `BudgetUnavailable` |
| `DEX_ANTHROPIC_BASE_URL` | `https://api.anthropic.com` | API base URL |
| `CLAUDE_BUDGET_DB` | `D:\DexProjects\claude-budget\ledger.db` | ledger path |
| `CLAUDE_BUDGET_CAPS` | `caps.local.json`, then `caps.json` next to the ledger, then built-in | caps file |
| `CLAUDE_BUDGET_ALERT_PATH` | `G:\CyberDash\alerts-claude-budget.json` | alert file |

`caps.json` (tracked) holds the plan allocation. Put machine-specific changes in `caps.local.json` (git-ignored). Caps are re-read on every call, so an edit applies without a restart.

```json
{"reset_day": 29, "global": 190.0,
 "consumers": {"trading": 45, "portal-ingest": 20, "portal-chat": 25, "house-advice": 5, "job-classify": 12,
               "cmdb": 15, "rc-nuggets": 10, "finance": 8, "hermes-fallback": 10, "reserve": 35}}
```

**Cycle:** from `reset_day` 00:00 UTC to the next `reset_day`, so the 29th through the 28th. In a month shorter than `reset_day`, the boundary falls on that month's last day: in a non-leap February the cycle resets on Feb 28.

## Prices (as of 2026-09-25, from the `claude-api` skill)

| Model | Input | Output | Cache read | Cache write 5m / 1h |
|---|---|---|---|---|
| `claude-opus-5-5` | $4 | $20 | $0.20 | $5 / $8 |
| `claude-sonnet-5-5` | $2 | $10 | $0.20 | $2.50 / $4 |
| `claude-haiku-4-5` | $1 | $5 | $0.10 | $1.25 / $2 |

Per MTok. The Batch API is 50% of every line. Possible fallback targets (Opus 5, Opus 4.8, etc.) are in `claude_budget/pricing.py`, so a fallback attempt is priced correctly.

## Ledger schema

Table `ledger` in `ledger.db` (WAL mode), one row per call or per batch result: `ts` (UTC ISO), `consumer`, `model`, `served_model`, `input_tokens`, `cache_creation_tokens`, `cache_read_tokens`, `output_tokens`, `batch`, `estimate_usd`, `cost_usd`, `request_id`, `batch_id`, `custom_id`, `status` (`reserved` / `ok` / `error` / `released`), `error`.

## Tests

```powershell
D:\DexProjects\claude-budget\.venv\Scripts\python.exe -m pytest
```

The tests mock the Anthropic client: no network, no key. `tests/test_sdk_wire.py` drives the real SDK through an `httpx2.MockTransport` to check the exact request body and headers.

<!-- ca-tracked-link -->
[cyberautomations.com](https://cyberautomations.com/?utm_source=github&utm_medium=readme&utm_campaign=claude-budget)
