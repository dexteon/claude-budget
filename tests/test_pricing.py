from types import SimpleNamespace

import anthropic
import pytest

from claude_budget.errors import UnknownModel
from claude_budget.estimate import cache_write_multiplier, has_media, heuristic_input_tokens, worst_case_usd
from claude_budget.pricing import PRICING, TokenUsage, cost_usd, message_cost, price_for, usage_from_sdk

M = 1_000_000


@pytest.mark.parametrize(
    "model, inp, out",
    [("claude-opus-5-5", 4.0, 20.0), ("claude-sonnet-5-5", 2.0, 10.0), ("claude-haiku-4-5", 1.0, 5.0)],
)
def test_list_prices(model, inp, out):
    assert cost_usd(model, TokenUsage(input_tokens=M)) == pytest.approx(inp)
    assert cost_usd(model, TokenUsage(output_tokens=M)) == pytest.approx(out)
    assert cost_usd(model, TokenUsage(input_tokens=M, output_tokens=M)) == pytest.approx(inp + out)


@pytest.mark.parametrize(
    "model, read, write5, write1h",
    [
        ("claude-opus-5-5", 0.20, 5.0, 8.0),
        ("claude-sonnet-5-5", 0.20, 2.5, 4.0),
        ("claude-haiku-4-5", 0.10, 1.25, 2.0),
    ],
)
def test_cache_prices(model, read, write5, write1h):
    assert cost_usd(model, TokenUsage(cache_read_tokens=M)) == pytest.approx(read)
    assert cost_usd(model, TokenUsage(cache_creation_5m_tokens=M)) == pytest.approx(write5)
    assert cost_usd(model, TokenUsage(cache_creation_1h_tokens=M)) == pytest.approx(write1h)


@pytest.mark.parametrize("model", ["claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-4-5"])
def test_batch_is_half_of_everything(model):
    u = TokenUsage(input_tokens=12_345, cache_creation_5m_tokens=2_000, cache_creation_1h_tokens=1_000,
                   cache_read_tokens=50_000, output_tokens=7_777)
    assert cost_usd(model, u, batch=True) == pytest.approx(cost_usd(model, u) / 2)


def test_mixed_usage_opus_by_hand():
    # 10k input, 2k 5m write, 30k read, 1k output on Opus 5.5
    u = TokenUsage(input_tokens=10_000, cache_creation_5m_tokens=2_000, cache_read_tokens=30_000, output_tokens=1_000)
    expected = (10_000 * 4 + 2_000 * 5 + 30_000 * 0.20 + 1_000 * 20) / M
    assert cost_usd("claude-opus-5-5", u) == pytest.approx(expected)
    assert cost_usd("claude-opus-5-5", u, batch=True) == pytest.approx(expected / 2)


def test_usage_from_real_sdk_usage_with_ttl_breakdown():
    u = anthropic.types.Usage(
        input_tokens=100, output_tokens=40, cache_read_input_tokens=900, cache_creation_input_tokens=300,
        cache_creation=anthropic.types.CacheCreation(ephemeral_5m_input_tokens=100, ephemeral_1h_input_tokens=200),
    )
    tu = usage_from_sdk(u)
    assert tu == TokenUsage(100, 100, 200, 900, 40)
    assert tu.cache_creation_tokens == 300


def test_usage_without_breakdown_prices_writes_as_5m():
    tu = usage_from_sdk(anthropic.types.Usage(input_tokens=1, output_tokens=1, cache_creation_input_tokens=50))
    assert tu.cache_creation_5m_tokens == 50 and tu.cache_creation_1h_tokens == 0


def test_message_cost_sums_fallback_iterations_at_each_models_price():
    usage = SimpleNamespace(
        input_tokens=1000, output_tokens=200, cache_read_input_tokens=0, cache_creation_input_tokens=0,
        iterations=[
            {"type": "message", "model": "claude-opus-5-5", "input_tokens": 1000, "output_tokens": 50,
             "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0},
            {"type": "fallback_message", "model": "claude-opus-4-8", "input_tokens": 1000, "output_tokens": 200,
             "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0},
        ],
    )
    msg = SimpleNamespace(model="claude-opus-4-8", usage=usage)
    total, tu, attempts = message_cost("claude-opus-5-5", msg)
    expected = (1000 * 4 + 50 * 20) / M + (1000 * 5 + 200 * 25) / M
    assert total == pytest.approx(expected)
    assert [a.model for a in attempts] == ["claude-opus-5-5", "claude-opus-4-8"]
    assert tu.output_tokens == 250


def test_unknown_model_is_strict_by_default():
    with pytest.raises(UnknownModel):
        price_for("claude-imaginary-9")
    assert price_for("claude-imaginary-9", strict=False) == max(PRICING.values(), key=lambda p: p.output)
    assert price_for("anthropic.claude-haiku-4-5") == PRICING["claude-haiku-4-5"]


def test_worst_case_and_fallback_bound():
    base = worst_case_usd("claude-opus-5-5", 1000, 1000)
    assert base == pytest.approx((1000 * 4 + 1000 * 20) / M)
    with_fb = worst_case_usd("claude-opus-5-5", 1000, 1000, fallbacks=True)
    assert with_fb == pytest.approx((1000 * 5 + 1000 * 25) / M)  # Opus 4.8 / Opus 5 rates
    assert worst_case_usd("claude-haiku-4-5", 1000, 1000, batch=True) == pytest.approx(3000 / M)


def test_estimate_helpers():
    p = {"system": [{"type": "text", "text": "x" * 400, "cache_control": {"type": "ephemeral", "ttl": "1h"}}],
         "messages": [{"role": "user", "content": "hi"}]}
    assert cache_write_multiplier(p) == 2.0
    assert cache_write_multiplier({"messages": [], "cache_control": {"type": "ephemeral"}}) == 1.25
    assert cache_write_multiplier({"messages": []}) == 1.0
    assert heuristic_input_tokens(p) >= 100
    assert not has_media(p)
    img = {"messages": [{"role": "user", "content": [{"type": "image", "source": {"type": "url", "url": "u"}}]}]}
    assert has_media(img)
