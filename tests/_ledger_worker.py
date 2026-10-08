"""Child-process worker for the concurrency tests.

usage: python _ledger_worker.py DB CONSUMER N ESTIMATE CAP START_FILE
Waits for START_FILE to exist (so both workers start together), then tries
N reserve+finalize cycles and prints how many were accepted.
"""

import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from claude_budget.config import Caps, cycle_for
from claude_budget.errors import BudgetExceeded
from claude_budget.ledger import Ledger
from claude_budget.pricing import TokenUsage


def main() -> None:
    db, consumer, n, estimate, cap, start_file = sys.argv[1:7]
    n, estimate, cap = int(n), float(estimate), float(cap)
    now = datetime(2026, 10, 7, 12, tzinfo=timezone.utc)
    caps = Caps(global_cap=cap, consumers={"a": cap, "b": cap})
    cycle = cycle_for(now)
    ledger = Ledger(db)
    deadline = time.monotonic() + 30
    while not Path(start_file).exists():
        if time.monotonic() > deadline:
            raise SystemExit("start file never appeared")
        time.sleep(0.005)
    accepted = 0
    for i in range(n):
        try:
            row = ledger.reserve(consumer=consumer, model="claude-haiku-4-5", estimate_usd=estimate,
                                 caps=caps, cycle=cycle, now=now)
        except BudgetExceeded:
            continue
        ledger.finalize(row, status="ok", cost_usd=estimate, usage=TokenUsage(input_tokens=1, output_tokens=1),
                        request_id=f"{consumer}-{i}")
        accepted += 1
    print(accepted)


if __name__ == "__main__":
    main()
