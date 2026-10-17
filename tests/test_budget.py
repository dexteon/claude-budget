import json
from types import SimpleNamespace

import anthropic
import httpx2
import pytest
from pydantic import BaseModel

import claude_budget
from claude_budget import (
    BatchNotReady,
    BudgetExceeded,
    BudgetUnavailable,
    ClaudeBudget,
    CredentialError,
    NonRetryableAPIError,
    RetryableAPIError,
    UnknownConsumer,
    UnknownModel,
)
from claude_budget.budget import FALLBACK_BETA
from conftest import make_message

MSG = [{"role": "user", "content": "hello"}]


def _sdk_error(cls, status):
    req = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    resp = httpx2.Response(status, request=req, headers={"request-id": "req_err"})
    return cls(f"status {status}", response=resp, body=None)


# -- key handling --------------------------------------------------------------


def test_missing_key_raises_budget_unavailable(settings, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-must-not-be-used")
    b = ClaudeBudget(settings=settings)
    with pytest.raises(BudgetUnavailable):
        b.call("finance", model="claude-haiku-4-5", max_tokens=10, messages=MSG)
    assert b.ledger.rows() == []


def test_module_level_call_without_key(monkeypatch, tmp_path):
    claude_budget.configure()  # fresh default reading env from the autouse fixture
    with pytest.raises(BudgetUnavailable):
        claude_budget.call("finance", model="claude-haiku-4-5", max_tokens=10, messages=MSG)


def test_client_uses_dex_key_and_pins_base_url(settings, monkeypatch):
    seen = {}

    def fake_ctor(**kw):
        seen.update(kw)
        return SimpleNamespace()

    monkeypatch.setenv("DEX_ANTHROPIC_API_KEY", "sk-dex-test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-wrong")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://litellm.local:4000")
    monkeypatch.setattr(claude_budget.budget.anthropic, "Anthropic", fake_ctor)
    ClaudeBudget(settings=settings).client()
    assert seen == {"api_key": "sk-dex-test", "base_url": "https://api.anthropic.com"}


def test_real_sdk_client_ignores_anthropic_api_key(settings, monkeypatch):
    monkeypatch.setenv("DEX_ANTHROPIC_API_KEY", "sk-dex-test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-wrong")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "tok-wrong")
    c = ClaudeBudget(settings=settings).client()
    assert c.api_key == "sk-dex-test"
    assert c.auth_token is None
    assert str(c.base_url).startswith("https://api.anthropic.com")


# -- caps ------------------------------------------------------------------------


def test_consumer_cap_trips_before_the_call(make_budget, fake_client):
    b = make_budget({"global": 190, "consumers": {"house-advice": 0.01}})
    # Sonnet worst case: 20k output tokens * $10/MTok = $0.20 > $0.01
    with pytest.raises(BudgetExceeded) as ei:
        b.call("house-advice", model="claude-sonnet-5-5", max_tokens=20_000, messages=MSG)
    assert ei.value.scope == "consumer"
    assert fake_client.calls == []
    assert b.ledger.rows() == []


def test_global_cap_trips_before_the_call(make_budget, fake_client):
    b = make_budget({"global": 1.0, "consumers": {"a": 50, "b": 50}})
    fake_client.response = make_message("claude-haiku-4-5", inp=0, out=190_000)  # $0.95
    b.call("a", model="claude-haiku-4-5", max_tokens=1000, messages=MSG)
    fake_client.calls.clear()
    with pytest.raises(BudgetExceeded) as ei:
        b.call("b", model="claude-haiku-4-5", max_tokens=20_000, messages=MSG)  # +$0.10 worst case
    assert ei.value.scope == "global"
    assert fake_client.calls == []


def test_cap_allows_until_limit_then_refuses(make_budget, fake_client):
    b = make_budget({"global": 190, "consumers": {"finance": 0.02}})
    fake_client.response = make_message("claude-haiku-4-5", inp=0, out=1000)  # $0.005 each
    for _ in range(3):
        b.call("finance", model="claude-haiku-4-5", max_tokens=1000, messages=MSG)
    with pytest.raises(BudgetExceeded):
        b.call("finance", model="claude-haiku-4-5", max_tokens=1000, messages=MSG)
    assert b.spend("finance")["spent_usd"] == pytest.approx(0.015)


def test_unknown_consumer_and_model(make_budget):
    b = make_budget()
    with pytest.raises(UnknownConsumer):
        b.call("nobody", model="claude-haiku-4-5", max_tokens=10, messages=MSG)
    assert issubclass(UnknownConsumer, BudgetUnavailable)
    with pytest.raises(UnknownModel):
        b.call("finance", model="claude-imaginary-9", max_tokens=10, messages=MSG)


def test_max_tokens_required(make_budget):
    with pytest.raises(TypeError):
        make_budget().call("finance", model="claude-haiku-4-5", messages=MSG)


# -- recording -----------------------------------------------------------------


def test_call_records_actual_cost(make_budget, fake_client):
    b = make_budget()
    fake_client.response = make_message("claude-haiku-4-5", inp=2000, out=300, cache_read=1000)
    msg = b.call("finance", model="claude-haiku-4-5", max_tokens=500, messages=MSG)
    assert msg.id == "msg_test"
    (row,) = b.ledger.rows()
    assert row["status"] == "ok"
    assert row["consumer"] == "finance" and row["model"] == "claude-haiku-4-5"
    assert (row["input_tokens"], row["cache_read_tokens"], row["output_tokens"]) == (2000, 1000, 300)
    assert row["cost_usd"] == pytest.approx((2000 * 1 + 1000 * 0.10 + 300 * 5) / 1e6)
    assert row["request_id"] == "req_test"
    assert row["batch"] == 0
    assert row["estimate_usd"] > 0


# -- defaults --------------------------------------------------------------------


def test_opus_defaults_adaptive_effort_and_fallbacks(make_budget, fake_client):
    make_budget().call("reserve", model="claude-opus-5-5", max_tokens=1000, messages=MSG)
    label, method, kw = fake_client.calls[-1]
    assert (label, method) == ("beta", "create")
    assert kw["thinking"] == {"type": "adaptive"}
    assert kw["output_config"] == {"effort": "medium"}
    assert kw["fallbacks"] == "default"
    assert kw["betas"] == [FALLBACK_BETA]


def test_sonnet_defaults_respect_caller_values(make_budget, fake_client):
    make_budget().call("portal-chat", model="claude-sonnet-5-5", max_tokens=1000, messages=MSG,
                       thinking={"type": "between_tools"},
                       output_config={"effort": "low", "format": {"type": "json_schema", "schema": {}}},
                       betas=["some-beta"])
    _, _, kw = fake_client.calls[-1]
    assert kw["thinking"] == {"type": "between_tools"}
    assert kw["output_config"]["effort"] == "low" and "format" in kw["output_config"]
    assert kw["betas"] == ["some-beta", FALLBACK_BETA]


def test_fallbacks_opt_out(make_budget, fake_client):
    make_budget().call("reserve", model="claude-opus-5-5", max_tokens=100, messages=MSG, fallbacks=None)
    label, _, kw = fake_client.calls[-1]
    assert label == "messages" and "fallbacks" not in kw and "betas" not in kw


def test_haiku_gets_no_thinking_no_effort_no_beta(make_budget, fake_client):
    make_budget().call("finance", model="claude-haiku-4-5", max_tokens=100, messages=MSG)
    label, _, kw = fake_client.calls[-1]
    assert label == "messages"
    assert "thinking" not in kw and "output_config" not in kw and "fallbacks" not in kw


def test_haiku_budget_thinking_passes_through(make_budget, fake_client):
    t = {"type": "enabled", "budget_tokens": 2048}
    make_budget().call("finance", model="claude-haiku-4-5", max_tokens=4096, messages=MSG, thinking=t)
    assert fake_client.calls[-1][2]["thinking"] == t


def test_stream_returns_final_message(make_budget, fake_client):
    b = make_budget()
    fake_client.response = make_message("claude-sonnet-5-5", inp=10, out=10)
    msg = b.call("portal-chat", model="claude-sonnet-5-5", max_tokens=64_000, messages=MSG, stream=True)
    label, method, kw = fake_client.calls[-1]
    assert (label, method) == ("beta", "stream") and "stream" not in kw
    assert msg.model == "claude-sonnet-5-5"
    assert b.ledger.rows()[0]["request_id"] == "req_stream"


class Verdict(BaseModel):
    remote: bool


def test_parse_uses_beta_parse_with_output_format(make_budget, fake_client):
    msg = make_budget().parse("job-classify", output_format=Verdict, model="claude-sonnet-5-5",
                              max_tokens=500, messages=MSG)
    label, method, kw = fake_client.calls[-1]
    assert (label, method) == ("beta", "parse")
    assert kw["output_format"] is Verdict and kw["fallbacks"] == "default"
    assert msg.parsed_output == "parsed"


def test_parse_on_haiku_uses_plain_messages(make_budget, fake_client):
    make_budget().parse("job-classify", output_format=Verdict, model="claude-haiku-4-5", max_tokens=50, messages=MSG)
    assert fake_client.calls[-1][:2] == ("messages", "parse")


# -- errors ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "cls, status, expected, fallback_ok",
    [
        (anthropic.RateLimitError, 429, RetryableAPIError, True),
        (anthropic.InternalServerError, 500, RetryableAPIError, True),
        (anthropic.APIStatusError, 529, RetryableAPIError, True),
        (anthropic.AuthenticationError, 401, CredentialError, True),
        (anthropic.BadRequestError, 400, NonRetryableAPIError, False),
        (anthropic.NotFoundError, 404, NonRetryableAPIError, False),
    ],
)
def test_sdk_errors_are_typed(make_budget, fake_client, cls, status, expected, fallback_ok):
    b = make_budget()
    fake_client.raise_exc = _sdk_error(cls, status)
    with pytest.raises(expected) as ei:
        b.call("finance", model="claude-haiku-4-5", max_tokens=100, messages=MSG)
    assert isinstance(ei.value, BudgetUnavailable) is fallback_ok
    assert ei.value.retryable is (expected is RetryableAPIError)
    assert ei.value.status_code == status
    assert isinstance(ei.value.original, cls)
    (row,) = b.ledger.rows()
    assert row["status"] == "error" and row["cost_usd"] == 0


def test_timeout_keeps_worst_case_on_the_books(make_budget, fake_client):
    b = make_budget()
    fake_client.raise_exc = anthropic.APITimeoutError(request=httpx2.Request("POST", "https://x"))
    with pytest.raises(RetryableAPIError):
        b.call("finance", model="claude-haiku-4-5", max_tokens=1000, messages=MSG)
    (row,) = b.ledger.rows()
    assert row["status"] == "error"
    assert row["cost_usd"] == pytest.approx(row["estimate_usd"]) and row["cost_usd"] > 0


# -- token counting -------------------------------------------------------------


def test_count_tokens_used_for_media_in_auto_mode(make_budget, fake_client):
    b = make_budget(count_tokens="auto")
    content = [{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "A" * 400_000}},
               {"type": "text", "text": "what is this"}]
    fake_client.counted_tokens = 1500
    b.call("portal-ingest", model="claude-sonnet-5-5", max_tokens=1000,
           messages=[{"role": "user", "content": content}])
    methods = [m for _, m, _ in fake_client.calls]
    assert methods[0] == "count_tokens"
    # estimate based on 1500 counted tokens at the fallback-bounded rate, not 100k chars/4
    row = b.ledger.rows()[0]
    assert row["estimate_usd"] == pytest.approx((1500 * 5 + 1000 * 25) / 1e6)


def test_count_tokens_skipped_when_cheap(make_budget, fake_client):
    make_budget(count_tokens="auto").call("finance", model="claude-haiku-4-5", max_tokens=100, messages=MSG)
    assert [m for _, m, _ in fake_client.calls] == ["create"]


# -- batches ---------------------------------------------------------------------


def _batch_requests(n=3):
    return [{"custom_id": f"job-{i}", "params": {"model": "claude-haiku-4-5", "max_tokens": 200,
                                                "messages": [{"role": "user", "content": f"post {i}"}]}}
            for i in range(n)]


def test_batch_submit_and_collect(make_budget, fake_client):
    b = make_budget()
    batch_id = b.batch_submit("job-classify", _batch_requests())
    assert batch_id == "msgbatch_test"
    (sent,) = fake_client.messages.batches.submitted
    assert all("fallbacks" not in r["params"] and "betas" not in r["params"] for r in sent)
    res = b.ledger.batch_reservation(batch_id)
    assert res["status"] == "reserved" and res["batch"] == 1 and res["estimate_usd"] > 0

    with pytest.raises(BatchNotReady):
        b.batch_collect(batch_id)

    batches = fake_client.messages.batches
    batches.status = "ended"
    batches.results_list = [
        SimpleNamespace(custom_id="job-0", result=SimpleNamespace(
            type="succeeded", message=make_message("claude-haiku-4-5", msg_id="m0", inp=1000, out=100))),
        SimpleNamespace(custom_id="job-1", result=SimpleNamespace(
            type="succeeded", message=make_message("claude-haiku-4-5", msg_id="m1", inp=2000, out=200))),
        SimpleNamespace(custom_id="job-2", result=SimpleNamespace(
            type="errored", error=SimpleNamespace(type="invalid_request"))),
    ]
    out = b.batch_collect(batch_id)
    assert set(out) == {"job-0", "job-1", "job-2"}
    rows = [r for r in b.ledger.rows(batch_id=batch_id) if r["custom_id"]]
    costs = {r["custom_id"]: r["cost_usd"] for r in rows}
    assert costs["job-0"] == pytest.approx((1000 * 1 + 100 * 5) / 1e6 / 2)
    assert costs["job-1"] == pytest.approx((2000 * 1 + 200 * 5) / 1e6 / 2)
    assert costs["job-2"] == 0
    assert all(r["batch"] == 1 for r in rows)
    assert b.ledger.batch_reservation(batch_id)["status"] == "released"
    spent = b.spend("job-classify")["spent_usd"]
    assert spent == pytest.approx(costs["job-0"] + costs["job-1"])

    b.batch_collect(batch_id)  # idempotent
    assert b.spend("job-classify")["spent_usd"] == pytest.approx(spent)


def test_batch_reservation_counts_against_cap(make_budget):
    b = make_budget({"global": 190, "consumers": {"job-classify": 0.0015}})
    b.batch_submit("job-classify", _batch_requests(2))  # 2 * 200 * $5/MTok / 2 ~ $0.001 + input
    with pytest.raises(BudgetExceeded):
        b.batch_submit("job-classify", _batch_requests(2))


def test_batch_rejects_betas(make_budget):
    reqs = _batch_requests(1)
    reqs[0]["params"]["betas"] = ["x"]
    with pytest.raises(ValueError):
        make_budget().batch_submit("job-classify", reqs)


# -- alert ----------------------------------------------------------------------


def test_alert_written_at_80_percent(make_budget, fake_client, settings):
    b = make_budget({"global": 1.0, "consumers": {"reserve": 5}})
    fake_client.response = make_message("claude-haiku-4-5", inp=0, out=150_000)  # $0.75
    b.call("reserve", model="claude-haiku-4-5", max_tokens=1000, messages=MSG)
    assert not settings.alert_file.exists()
    fake_client.response = make_message("claude-haiku-4-5", inp=0, out=10_000)  # +$0.05 -> 80%
    b.call("reserve", model="claude-haiku-4-5", max_tokens=1000, messages=MSG)
    alert = json.loads(settings.alert_file.read_text())
    assert alert["source"] == "claude-budget" and alert["level"] == "warning"
    assert alert["pct"] == pytest.approx(0.8)
    assert alert["refill_date"] == "2026-10-29" and alert["days_to_refill"] == 22


def test_unwritable_alert_path_never_breaks_a_call(make_budget, fake_client, settings, tmp_path):
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("x")
    settings.alert_file = blocker / "alert.json"  # parent is a file: mkdir fails
    b = make_budget({"global": 1.0, "consumers": {"reserve": 5}})
    fake_client.response = make_message("claude-haiku-4-5", inp=0, out=180_000)  # $0.90 -> 90%
    msg = b.call("reserve", model="claude-haiku-4-5", max_tokens=1000, messages=MSG)
    assert msg.id == "msg_test"
    assert b.ledger.rows()[0]["status"] == "ok"


# -- spend -------------------------------------------------------------------------


def test_spend_shape(make_budget, fake_client):
    b = make_budget()
    fake_client.response = make_message("claude-haiku-4-5", inp=0, out=200_000)  # $1
    b.call("finance", model="claude-haiku-4-5", max_tokens=1000, messages=MSG)
    s = b.spend()
    assert s["global"]["spent_usd"] == pytest.approx(1.0) and s["global"]["cap_usd"] == 190
    assert s["consumers"]["finance"]["remaining_usd"] == pytest.approx(7.0)
    assert s["consumers"]["trading"]["spent_usd"] == 0
    assert list(s["consumers"])[:2] == ["trading", "portal-ingest"]
    one = b.spend("finance")
    assert one["calls"] == 1 and one["cap_usd"] == 8 and one["cycle"]["days_to_refill"] == 22


# -- gateway mode (DEX_ANTHROPIC_BASE_URL points at the LiteLLM gateway) -------------

GW = "http://gateway-host:10100"


def test_gateway_mode_uses_gateway_key_only(settings, monkeypatch):
    seen = {}

    def fake_ctor(**kw):
        seen.update(kw)
        return SimpleNamespace()

    monkeypatch.setenv("DEX_ANTHROPIC_BASE_URL", GW)
    monkeypatch.setenv("DEX_ANTHROPIC_API_KEY", "sk-dex-direct")
    monkeypatch.setattr(claude_budget.budget.anthropic, "Anthropic", fake_ctor)
    with pytest.raises(BudgetUnavailable):
        ClaudeBudget(settings=settings).client()
    monkeypatch.setenv("DEX_ANTHROPIC_GATEWAY_KEY", "sk-gw-test")
    ClaudeBudget(settings=settings).client()
    assert seen == {"api_key": "sk-gw-test", "base_url": GW}


def test_gateway_mode_tags_consumer_and_skips_fallbacks(make_budget, fake_client, monkeypatch):
    monkeypatch.setenv("DEX_ANTHROPIC_BASE_URL", GW)
    b = make_budget(count_tokens=True)
    b.call("finance", model="claude-sonnet-5-5", max_tokens=10, messages=MSG)
    (_, m1, count_kw), (label, m2, kw) = fake_client.calls
    assert m1 == "count_tokens" and count_kw["extra_headers"] == {"x-litellm-tags": "consumer:finance"}
    assert label == "messages" and m2 == "create"
    assert "fallbacks" not in kw and FALLBACK_BETA not in (kw.get("betas") or [])
    assert kw["extra_headers"] == {"x-litellm-tags": "consumer:finance"}


def test_gateway_mode_refuses_batches(make_budget, monkeypatch):
    monkeypatch.setenv("DEX_ANTHROPIC_BASE_URL", GW)
    with pytest.raises(BudgetUnavailable):
        make_budget().batch_submit("finance", [{"custom_id": "a", "params": {
            "model": "claude-haiku-4-5", "max_tokens": 10, "messages": MSG}}])
