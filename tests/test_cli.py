import json
import os
import subprocess
import sys

from claude_budget.cli import main
from conftest import make_message

MSG = [{"role": "user", "content": "hi"}]


def _seed(make_budget, fake_client):
    b = make_budget()
    fake_client.response = make_message("claude-haiku-4-5", inp=0, out=2_000_000)  # $10.00
    b.call("trading", model="claude-haiku-4-5", max_tokens=1000, messages=MSG)
    fake_client.response = make_message("claude-haiku-4-5", inp=0, out=400_000)  # $2.00
    b.call("finance", model="claude-haiku-4-5", max_tokens=1000, messages=MSG)
    return b


def test_status_table(make_budget, fake_client, capsys):
    b = _seed(make_budget, fake_client)
    assert main(["status"], budget=b) == 0
    out = capsys.readouterr().out
    assert "cycle 2026-09-29..2026-10-28" in out
    assert "refill 2026-10-29" in out
    lines = {ln.split()[0]: ln.split() for ln in out.splitlines() if ln.strip()}
    assert lines["trading"][1:4] == ["$10.00", "$45.00", "22%"]
    assert lines["finance"][1:4] == ["$2.00", "$8.00", "25%"]
    assert lines["GLOBAL"][1:4] == ["$12.00", "$190.00", "6%"]
    assert "days to refill: 22" in out
    assert "ALERT" not in out
    assert chr(0x2014) not in out and chr(0x2013) not in out


def test_status_json(make_budget, fake_client, capsys):
    b = _seed(make_budget, fake_client)
    main(["status", "--json"], budget=b)
    data = json.loads(capsys.readouterr().out)
    assert data["global"]["spent_usd"] == 12.0
    assert data["consumers"]["trading"]["cap_usd"] == 45.0
    assert data["cycle"]["days_to_refill"] == 22


def test_python_dash_m_entrypoint(tmp_path):
    env = dict(os.environ)
    env["CLAUDE_BUDGET_DB"] = str(tmp_path / "ledger.db")
    env.pop("CLAUDE_BUDGET_CAPS", None)
    env.pop("DEX_ANTHROPIC_API_KEY", None)
    r = subprocess.run([sys.executable, "-m", "claude_budget", "status", "--json"],
                       capture_output=True, text=True, env=env, timeout=60)
    assert r.returncode == 0, r.stderr
    data = json.loads(r.stdout)
    assert data["global"]["cap_usd"] == 190.0
    assert set(data["consumers"]) >= {"trading", "reserve", "hermes-fallback"}
    r2 = subprocess.run([sys.executable, "-m", "claude_budget", "status"],
                        capture_output=True, text=True, env=env, timeout=60)
    assert r2.returncode == 0 and "GLOBAL" in r2.stdout
