import subprocess
import sys
from pathlib import Path

import pytest

from claude_budget.config import cycle_for
from claude_budget.ledger import Ledger
from conftest import NOW

WORKER = Path(__file__).with_name("_ledger_worker.py")


def _run_pair(db: Path, tmp_path: Path, n: int, estimate: float, cap: float) -> list[int]:
    Ledger(db)  # create schema + WAL before the race
    start = tmp_path / "go"
    procs = [
        subprocess.Popen(
            [sys.executable, str(WORKER), str(db), consumer, str(n), str(estimate), str(cap), str(start)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        for consumer in ("a", "b")
    ]
    start.write_text("go")
    out = []
    for p in procs:
        stdout, stderr = p.communicate(timeout=120)
        assert p.returncode == 0, stderr
        out.append(int(stdout.strip()))
    return out


def test_two_processes_write_without_losing_rows(tmp_path):
    db = tmp_path / "ledger.db"
    accepted = _run_pair(db, tmp_path, n=100, estimate=0.01, cap=1000.0)
    assert accepted == [100, 100]
    ledger = Ledger(db)
    rows = ledger.rows()
    assert len(rows) == 200
    assert all(r["status"] == "ok" for r in rows)
    assert len({r["request_id"] for r in rows}) == 200
    assert ledger.spent(cycle_for(NOW)) == pytest.approx(2.0)
    with ledger._connect() as con:
        assert con.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"


def test_cap_check_is_atomic_across_processes(tmp_path):
    db = tmp_path / "ledger.db"
    # Global cap $10 at $1 per reservation: exactly 10 may pass in total,
    # however the two processes interleave.
    accepted = _run_pair(db, tmp_path, n=10, estimate=1.0, cap=10.0)
    assert sum(accepted) == 10
    assert Ledger(db).spent(cycle_for(NOW)) == pytest.approx(10.0)
