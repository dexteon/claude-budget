"""SQLite spend ledger.

WAL mode plus ``BEGIN IMMEDIATE`` around check-and-reserve makes the cap
check atomic across processes: two pipelines cannot both pass the check on
the same remaining headroom. A call first inserts a ``reserved`` row holding
its worst-case estimate; when the response arrives the row is finalised to
``ok`` / ``error`` with the actual cost. A reservation that is never
finalised (process killed mid-call) keeps counting at its estimate, which
errs on the side of spending less.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from .config import Caps, Cycle, utcnow
from .errors import BudgetExceeded
from .pricing import TokenUsage

SCHEMA = """
CREATE TABLE IF NOT EXISTS ledger (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    ts                    TEXT    NOT NULL,      -- ISO 8601 UTC
    consumer              TEXT    NOT NULL,
    model                 TEXT    NOT NULL,      -- requested model
    served_model          TEXT,                  -- model that produced the message
    input_tokens          INTEGER NOT NULL DEFAULT 0,
    cache_creation_tokens INTEGER NOT NULL DEFAULT 0,
    cache_read_tokens     INTEGER NOT NULL DEFAULT 0,
    output_tokens         INTEGER NOT NULL DEFAULT 0,
    batch                 INTEGER NOT NULL DEFAULT 0,
    estimate_usd          REAL    NOT NULL DEFAULT 0,
    cost_usd              REAL    NOT NULL DEFAULT 0,
    request_id            TEXT,
    batch_id              TEXT,
    custom_id             TEXT,
    status                TEXT    NOT NULL,      -- reserved | ok | error | released
    error                 TEXT
);
CREATE INDEX IF NOT EXISTS ix_ledger_ts ON ledger (ts);
CREATE INDEX IF NOT EXISTS ix_ledger_consumer_ts ON ledger (consumer, ts);
CREATE INDEX IF NOT EXISTS ix_ledger_batch ON ledger (batch_id);
"""

# Reserved rows count at their estimate, released rows count zero.
_SPEND_EXPR = (
    "COALESCE(SUM(CASE status WHEN 'reserved' THEN estimate_usd "
    "WHEN 'released' THEN 0 ELSE cost_usd END), 0)"
)


def _iso(ts: datetime) -> str:
    return ts.astimezone(timezone.utc).isoformat(timespec="microseconds")


@dataclass(frozen=True)
class ConsumerSpend:
    consumer: str
    spent_usd: float
    reserved_usd: float
    calls: int
    errors: int
    open_reservations: int


class Ledger:
    def __init__(self, path: Path | str, *, timeout: float = 30.0) -> None:
        self.path = Path(path)
        self.timeout = timeout
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as con:
            con.execute("PRAGMA journal_mode=WAL")
            con.executescript(SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        con = sqlite3.connect(self.path, timeout=self.timeout, isolation_level=None)
        try:
            con.execute(f"PRAGMA busy_timeout={int(self.timeout * 1000)}")
            con.execute("PRAGMA synchronous=NORMAL")
            yield con
        finally:
            con.close()

    @contextmanager
    def _immediate(self) -> Iterator[sqlite3.Connection]:
        with self._connect() as con:
            con.execute("BEGIN IMMEDIATE")
            try:
                yield con
            except BaseException:
                con.execute("ROLLBACK")
                raise
            else:
                con.execute("COMMIT")

    # -- reads ---------------------------------------------------------------

    @staticmethod
    def _spent(con: sqlite3.Connection, cycle: Cycle, consumer: str | None = None) -> float:
        sql = f"SELECT {_SPEND_EXPR} FROM ledger WHERE ts >= ? AND ts < ?"
        args: list[object] = [_iso(cycle.start), _iso(cycle.end)]
        if consumer is not None:
            sql += " AND consumer = ?"
            args.append(consumer)
        return float(con.execute(sql, args).fetchone()[0])

    def spent(self, cycle: Cycle, consumer: str | None = None) -> float:
        with self._connect() as con:
            return self._spent(con, cycle, consumer)

    def by_consumer(self, cycle: Cycle) -> dict[str, ConsumerSpend]:
        sql = f"""
            SELECT consumer,
                   {_SPEND_EXPR},
                   COALESCE(SUM(CASE status WHEN 'reserved' THEN estimate_usd ELSE 0 END), 0),
                   SUM(CASE WHEN status IN ('ok', 'error') THEN 1 ELSE 0 END),
                   SUM(CASE status WHEN 'error' THEN 1 ELSE 0 END),
                   SUM(CASE status WHEN 'reserved' THEN 1 ELSE 0 END)
            FROM ledger WHERE ts >= ? AND ts < ?
            GROUP BY consumer
        """
        with self._connect() as con:
            rows = con.execute(sql, (_iso(cycle.start), _iso(cycle.end))).fetchall()
        return {
            r[0]: ConsumerSpend(r[0], float(r[1]), float(r[2]), int(r[3] or 0), int(r[4] or 0), int(r[5] or 0))
            for r in rows
        }

    def rows(self, **where: object) -> list[sqlite3.Row]:
        sql = "SELECT * FROM ledger"
        if where:
            sql += " WHERE " + " AND ".join(f"{k} = ?" for k in where)
        sql += " ORDER BY id"
        with self._connect() as con:
            con.row_factory = sqlite3.Row
            return con.execute(sql, tuple(where.values())).fetchall()

    # -- writes --------------------------------------------------------------

    def reserve(
        self,
        *,
        consumer: str,
        model: str,
        estimate_usd: float,
        caps: Caps,
        cycle: Cycle,
        now: datetime | None = None,
        batch: bool = False,
        batch_id: str | None = None,
    ) -> int:
        """Atomically check both caps and insert a reservation row.

        Raises BudgetExceeded without writing anything when the estimate would
        cross the consumer cap or the global cap.
        """
        consumer_cap = caps.cap_for(consumer)
        now = now or utcnow()
        with self._immediate() as con:
            spent_c = self._spent(con, cycle, consumer)
            if spent_c + estimate_usd > consumer_cap:
                raise BudgetExceeded(
                    f"{consumer}: worst case ${estimate_usd:.4f} + spent ${spent_c:.4f} "
                    f"> cap ${consumer_cap:.2f}",
                    consumer=consumer, scope="consumer", estimate_usd=estimate_usd,
                    spent_usd=spent_c, cap_usd=consumer_cap,
                )
            spent_g = self._spent(con, cycle)
            if spent_g + estimate_usd > caps.global_cap:
                raise BudgetExceeded(
                    f"global: worst case ${estimate_usd:.4f} + spent ${spent_g:.4f} "
                    f"> cap ${caps.global_cap:.2f}",
                    consumer=consumer, scope="global", estimate_usd=estimate_usd,
                    spent_usd=spent_g, cap_usd=caps.global_cap,
                )
            cur = con.execute(
                "INSERT INTO ledger (ts, consumer, model, batch, estimate_usd, batch_id, status) "
                "VALUES (?, ?, ?, ?, ?, ?, 'reserved')",
                (_iso(now), consumer, model, int(batch), estimate_usd, batch_id),
            )
            return int(cur.lastrowid)

    def finalize(
        self,
        row_id: int,
        *,
        status: str,
        cost_usd: float = 0.0,
        usage: TokenUsage | None = None,
        served_model: str | None = None,
        request_id: str | None = None,
        batch_id: str | None = None,
        error: str | None = None,
    ) -> None:
        u = usage or TokenUsage()
        with self._immediate() as con:
            con.execute(
                """UPDATE ledger SET status = ?, cost_usd = ?, input_tokens = ?,
                       cache_creation_tokens = ?, cache_read_tokens = ?, output_tokens = ?,
                       served_model = COALESCE(?, served_model), request_id = COALESCE(?, request_id),
                       batch_id = COALESCE(?, batch_id), error = ?
                   WHERE id = ?""",
                (status, cost_usd, u.input_tokens, u.cache_creation_tokens, u.cache_read_tokens,
                 u.output_tokens, served_model, request_id, batch_id, error, row_id),
            )

    def attach_batch(self, row_id: int, batch_id: str) -> None:
        """Tag a still-open reservation with the batch id returned by the API."""
        with self._immediate() as con:
            con.execute("UPDATE ledger SET batch_id = ? WHERE id = ?", (batch_id, row_id))

    def settle_batch(
        self,
        batch_id: str,
        results: list[dict[str, object]],
        *,
        now: datetime | None = None,
    ) -> bool:
        """Record per-result rows for a batch and release its reservation.

        Idempotent: returns False (and writes nothing) if the batch was
        already settled.
        """
        now = now or utcnow()
        with self._immediate() as con:
            open_rows = con.execute(
                "SELECT id FROM ledger WHERE batch_id = ? AND status = 'reserved' AND custom_id IS NULL",
                (batch_id,),
            ).fetchall()
            settled = con.execute(
                "SELECT COUNT(*) FROM ledger WHERE batch_id = ? AND custom_id IS NOT NULL",
                (batch_id,),
            ).fetchone()[0]
            if settled:
                return False
            for r in results:
                u: TokenUsage = r.get("usage") or TokenUsage()  # type: ignore[assignment]
                con.execute(
                    """INSERT INTO ledger (ts, consumer, model, served_model, input_tokens,
                           cache_creation_tokens, cache_read_tokens, output_tokens, batch,
                           cost_usd, request_id, batch_id, custom_id, status, error)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?, ?, ?, ?)""",
                    (_iso(now), r["consumer"], r["model"], r.get("served_model"), u.input_tokens,
                     u.cache_creation_tokens, u.cache_read_tokens, u.output_tokens,
                     float(r.get("cost_usd", 0.0)), r.get("request_id"), batch_id, r["custom_id"],
                     r["status"], r.get("error")),
                )
            for (rid,) in open_rows:
                con.execute("UPDATE ledger SET status = 'released' WHERE id = ?", (rid,))
            return True

    def batch_reservation(self, batch_id: str) -> sqlite3.Row | None:
        rows = [r for r in self.rows(batch_id=batch_id) if r["custom_id"] is None]
        return rows[0] if rows else None
