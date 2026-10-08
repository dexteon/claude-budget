"""Drive the real anthropic SDK through a mock HTTP transport (no network) to
confirm the wire request the budget layer produces, and that real response
types are costed correctly."""

import json

import anthropic
import httpx2
import pytest
from pydantic import BaseModel

from claude_budget import ClaudeBudget
from claude_budget.config import Caps, DEFAULT_CAPS

MSG = [{"role": "user", "content": "hello"}]


def _response_body(model, text="ok", iterations=None):
    usage = {"input_tokens": 1200, "output_tokens": 80, "cache_read_input_tokens": 3000,
             "cache_creation_input_tokens": 0}
    if iterations is not None:
        usage["iterations"] = iterations
    return {"id": "msg_wire", "type": "message", "role": "assistant", "model": model,
            "content": [{"type": "text", "text": text}], "stop_reason": "end_turn",
            "stop_sequence": None, "usage": usage}


@pytest.fixture
def wire(settings, clock):
    seen = []
    state = {"body": None}

    def handler(request):
        seen.append(request)
        return httpx2.Response(200, json=state["body"], headers={"request-id": "req_wire"})

    client = anthropic.Anthropic(
        api_key="sk-test", base_url="https://api.anthropic.com", max_retries=0,
        http_client=anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(handler)),
    )
    budget = ClaudeBudget(settings=settings, client=client, caps=Caps.from_dict(DEFAULT_CAPS),
                          clock=clock, count_tokens=False)
    return budget, seen, state


def test_opus_request_on_the_wire(wire):
    budget, seen, state = wire
    state["body"] = _response_body("claude-opus-5-5")
    msg = budget.call("reserve", model="claude-opus-5-5", max_tokens=2000, messages=MSG)
    req = seen[-1]
    body = json.loads(req.content)
    assert req.url.path == "/v1/messages"
    assert "server-side-fallback-2026-07-01" in req.headers["anthropic-beta"]
    assert req.headers["x-api-key"] == "sk-test"
    assert body["fallbacks"] == "default"
    assert body["thinking"] == {"type": "adaptive"}
    assert body["output_config"] == {"effort": "medium"}
    assert "betas" not in body
    row = budget.ledger.rows()[0]
    assert row["cost_usd"] == pytest.approx((1200 * 4 + 3000 * 0.20 + 80 * 20) / 1e6)
    assert row["request_id"] == "req_wire"
    assert msg.content[0].text == "ok"


def test_fallback_iterations_from_real_beta_types(wire):
    budget, seen, state = wire
    state["body"] = _response_body("claude-opus-4-8", iterations=[
        {"type": "message", "model": "claude-opus-5-5", "input_tokens": 1200, "output_tokens": 0,
         "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0},
        {"type": "fallback_message", "model": "claude-opus-4-8", "input_tokens": 1200, "output_tokens": 80,
         "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0},
    ])
    budget.call("reserve", model="claude-opus-5-5", max_tokens=2000, messages=MSG)
    row = budget.ledger.rows()[0]
    assert row["served_model"] == "claude-opus-4-8"
    assert row["cost_usd"] == pytest.approx((1200 * 4) / 1e6 + (1200 * 5 + 80 * 25) / 1e6)


def test_haiku_request_is_plain_messages(wire):
    budget, seen, state = wire
    state["body"] = _response_body("claude-haiku-4-5")
    budget.call("finance", model="claude-haiku-4-5", max_tokens=100, messages=MSG)
    req = seen[-1]
    body = json.loads(req.content)
    assert "anthropic-beta" not in req.headers
    assert "thinking" not in body and "output_config" not in body and "fallbacks" not in body


class Verdict(BaseModel):
    remote: bool


def test_parse_on_the_wire(wire):
    budget, seen, state = wire
    state["body"] = _response_body("claude-sonnet-5-5", text='{"remote": true}')
    msg = budget.parse("job-classify", output_format=Verdict, model="claude-sonnet-5-5",
                       max_tokens=300, messages=MSG)
    assert msg.parsed_output == Verdict(remote=True)
    body = json.loads(seen[-1].content)
    assert body["output_config"]["effort"] == "medium"
    assert body["output_config"]["format"]["type"] == "json_schema"
    assert body["fallbacks"] == "default"
