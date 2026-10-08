from datetime import datetime, timezone

from claude_budget.config import cycle_for, resolve_cycle

UTC = timezone.utc


def d(*a):
    return datetime(*a, tzinfo=UTC)


def test_28th_is_last_day_of_cycle_and_29th_starts_new_one():
    last = cycle_for(d(2026, 10, 28, 23, 59, 59))
    assert (last.start, last.end) == (d(2026, 9, 29), d(2026, 10, 29))
    first = cycle_for(d(2026, 10, 29, 0, 0, 0))
    assert (first.start, first.end) == (d(2026, 10, 29), d(2026, 11, 29))
    assert last.label == "2026-09-29..2026-10-28"


def test_year_rollover():
    dec = cycle_for(d(2026, 12, 29, 1))
    assert (dec.start, dec.end) == (d(2026, 12, 29), d(2027, 1, 29))
    jan = cycle_for(d(2027, 1, 10))
    assert jan.start == d(2026, 12, 29)
    jan_end = cycle_for(d(2027, 1, 28, 23, 59))
    assert jan_end.start == d(2026, 12, 29) and jan_end.end == d(2027, 1, 29)
    assert cycle_for(d(2027, 1, 29)).start == d(2027, 1, 29)


def test_short_february_clamps_to_last_day():
    assert cycle_for(d(2027, 2, 27)).start == d(2027, 1, 29)
    assert cycle_for(d(2027, 2, 27)).end == d(2027, 2, 28)
    feb = cycle_for(d(2027, 2, 28, 5))
    assert (feb.start, feb.end) == (d(2027, 2, 28), d(2027, 3, 29))
    assert cycle_for(d(2028, 2, 29)).start == d(2028, 2, 29)  # leap year


def test_configurable_reset_day():
    c = cycle_for(d(2026, 10, 7), reset_day=1)
    assert (c.start, c.end) == (d(2026, 10, 1), d(2026, 11, 1))
    c = cycle_for(d(2026, 10, 14), reset_day=15)
    assert (c.start, c.end) == (d(2026, 9, 15), d(2026, 10, 15))


def test_days_to_refill_and_previous():
    now = d(2026, 10, 7, 12)
    c = cycle_for(now)
    assert c.days_to_refill(now) == 22
    prev = resolve_cycle("previous", now, 29)
    assert (prev.start, prev.end) == (d(2026, 8, 29), d(2026, 9, 29))
    assert resolve_cycle("2026-10-30", now, 29).start == d(2026, 10, 29)


def test_naive_datetime_treated_as_utc():
    assert cycle_for(datetime(2026, 10, 29)).start == d(2026, 10, 29)


def test_spend_resets_at_boundary(make_budget, fake_client, clock):
    b = make_budget()
    clock.now = d(2026, 10, 28, 23, 0)
    b.call("finance", model="claude-haiku-4-5", max_tokens=100, messages=[{"role": "user", "content": "a"}])
    assert b.spend("finance")["spent_usd"] > 0

    clock.now = d(2026, 10, 29, 0, 0, 1)
    assert b.spend("finance")["spent_usd"] == 0
    assert b.spend("finance", cycle="previous")["spent_usd"] > 0
    b.call("finance", model="claude-haiku-4-5", max_tokens=100, messages=[{"role": "user", "content": "a"}])
    assert b.spend("finance")["calls"] == 1


def test_year_rollover_spend_isolated(make_budget, clock):
    b = make_budget()
    clock.now = d(2026, 12, 28, 23, 59)
    b.call("finance", model="claude-haiku-4-5", max_tokens=100, messages=[{"role": "user", "content": "a"}])
    clock.now = d(2027, 1, 5)
    assert b.spend()["global"]["spent_usd"] == 0
    assert b.spend()["cycle"]["start"] == "2026-12-29"
    assert b.spend(cycle="2026-12-01")["global"]["spent_usd"] > 0
