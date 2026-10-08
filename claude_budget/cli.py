"""``python -m claude_budget status [--json] [--cycle previous|YYYY-MM-DD]``"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Sequence


def _money(v: float | None) -> str:
    return "-" if v is None else f"${v:,.2f}"


def _pct(v: float | None) -> str:
    return "-" if v is None else f"{v * 100:.0f}%"


def render_status(data: dict[str, Any]) -> str:
    cyc = data["cycle"]
    g = data["global"]
    lines = [
        f"Claude API budget, cycle {cyc['label']}  (refill {cyc['refill_date']}, "
        f"{cyc['days_to_refill']} days)",
        "",
        f"{'consumer':<18}{'spent':>10}{'cap':>10}{'used':>7}{'calls':>7}{'errors':>8}{'open':>6}",
    ]
    for c in data["consumers"].values():
        lines.append(
            f"{c['consumer']:<18}{_money(c['spent_usd']):>10}{_money(c['cap_usd']):>10}"
            f"{_pct(c['pct']):>7}{c['calls']:>7}{c['errors']:>8}{c['open_reservations']:>6}"
        )
    lines += [
        "",
        f"{'GLOBAL':<18}{_money(g['spent_usd']):>10}{_money(g['cap_usd']):>10}{_pct(g['pct']):>7}",
        f"days to refill: {cyc['days_to_refill']}",
    ]
    if g.get("alert"):
        lines.append("ALERT: global spend is at or above 80% of the cap")
    lines.append(f"caps: {data['caps_source']}   ledger: {data['ledger']}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None, budget: Any = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m claude_budget")
    sub = parser.add_subparsers(dest="cmd", required=True)
    st = sub.add_parser("status", help="month-to-date spend per consumer against its cap")
    st.add_argument("--json", action="store_true", help="print JSON instead of a table")
    st.add_argument("--cycle", default="current", help="current (default), previous, or a date inside the cycle")
    args = parser.parse_args(argv)

    if budget is None:
        from . import default_budget

        budget = default_budget()
    data = budget.spend(cycle=args.cycle)
    if args.json:
        print(json.dumps(data, indent=2))
    else:
        print(render_status(data))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
