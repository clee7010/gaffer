"""SQLite audit log: every run, tool call and guardrail verdict is persisted.

This is what makes the agent measurable: guardrail pass rates, retries per
run, tool latency and token cost all come straight out of these tables.
"""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at      TEXT NOT NULL,
    team_id         INTEGER,
    gameweek        INTEGER,
    question        TEXT,
    model           TEXT,
    status          TEXT,          -- running | accepted | rejected | error
    answer          TEXT,
    recommendation  TEXT,          -- JSON, only set once guardrails pass
    guardrail_attempts INTEGER DEFAULT 0,
    input_tokens    INTEGER DEFAULT 0,
    output_tokens   INTEGER DEFAULT 0,
    cache_read_tokens INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS tool_calls (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      INTEGER REFERENCES runs(id),
    name        TEXT,
    input       TEXT,
    is_error    INTEGER,
    latency_ms  REAL
);
CREATE TABLE IF NOT EXISTS guardrail_events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      INTEGER REFERENCES runs(id),
    attempt     INTEGER,
    passed      INTEGER,
    errors      TEXT,
    warnings    TEXT
);
"""


def default_db_path() -> Path:
    return Path(os.getenv("GAFFER_DB", ".cache/gaffer.db"))


class Store:
    def __init__(self, path: Path | str | None = None):
        self.path = Path(path) if path else default_db_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(SCHEMA)

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def start_run(self, team_id: int | None, gameweek: int, question: str, model: str) -> int:
        with self._conn() as c:
            cur = c.execute(
                "INSERT INTO runs (created_at, team_id, gameweek, question, model, status) VALUES (?,?,?,?,?, 'running')",
                (datetime.now(UTC).isoformat(), team_id, gameweek, question, model),
            )
            return int(cur.lastrowid)

    def log_tool(self, run_id: int, name: str, tool_input: dict, is_error: bool, latency_ms: float):
        with self._conn() as c:
            c.execute(
                "INSERT INTO tool_calls (run_id, name, input, is_error, latency_ms) VALUES (?,?,?,?,?)",
                (run_id, name, json.dumps(tool_input), int(is_error), latency_ms),
            )

    def log_guardrail(self, run_id: int, attempt: int, report: dict):
        with self._conn() as c:
            c.execute(
                "INSERT INTO guardrail_events (run_id, attempt, passed, errors, warnings) VALUES (?,?,?,?,?)",
                (run_id, attempt, int(report["passed"]), json.dumps(report["errors"]), json.dumps(report["warnings"])),
            )
            c.execute("UPDATE runs SET guardrail_attempts = ? WHERE id = ?", (attempt, run_id))

    def finish_run(self, run_id: int, status: str, answer: str, recommendation: dict | None, usage: dict):
        with self._conn() as c:
            c.execute(
                """UPDATE runs SET status=?, answer=?, recommendation=?, input_tokens=?, output_tokens=?,
                   cache_read_tokens=? WHERE id=?""",
                (
                    status,
                    answer,
                    json.dumps(recommendation) if recommendation else None,
                    usage.get("input_tokens", 0),
                    usage.get("output_tokens", 0),
                    usage.get("cache_read_input_tokens", 0),
                    run_id,
                ),
            )

    def get_run(self, run_id: int) -> dict | None:
        with self._conn() as c:
            row = c.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
            if not row:
                return None
            run = dict(row)
            run["recommendation"] = json.loads(run["recommendation"]) if run["recommendation"] else None
            run["guardrail_events"] = [
                {**dict(g), "errors": json.loads(g["errors"]), "warnings": json.loads(g["warnings"])}
                for g in c.execute("SELECT * FROM guardrail_events WHERE run_id = ? ORDER BY attempt", (run_id,))
            ]
            run["tool_calls"] = [
                dict(t) for t in c.execute("SELECT name, is_error, latency_ms FROM tool_calls WHERE run_id = ?", (run_id,))
            ]
            return run

    def stats(self, model: str | None = None) -> dict:
        """Aggregate guardrail metrics across finished runs, optionally for one model."""
        where, args = ("WHERE status != 'running'", ())
        if model:
            where, args = (where + " AND model = ?", (model,))
        with self._conn() as c:
            runs = [dict(r) for r in c.execute(f"SELECT id, status FROM runs {where}", args)]
            ids = [r["id"] for r in runs]
            if not ids:
                return {"runs": 0}
            marks = ",".join("?" * len(ids))
            events = [
                dict(e) for e in c.execute(f"SELECT * FROM guardrail_events WHERE run_id IN ({marks}) ORDER BY attempt", ids)
            ]
            tools = c.execute(f"SELECT COUNT(*), AVG(latency_ms) FROM tool_calls WHERE run_id IN ({marks})", ids).fetchone()

        by_run: dict[int, list[dict]] = {}
        for e in events:
            by_run.setdefault(e["run_id"], []).append(e)
        submitted = [rid for rid in ids if rid in by_run]
        first_pass = [rid for rid in submitted if by_run[rid][0]["passed"]]
        accepted = {r["id"] for r in runs if r["status"] == "accepted"}
        corrected = [rid for rid in submitted if not by_run[rid][0]["passed"] and rid in accepted]
        blocked = [rid for rid in submitted if rid not in accepted]
        codes: dict[str, int] = {}
        for e in events:
            for err in json.loads(e["errors"]):
                codes[err["code"]] = codes.get(err["code"], 0) + 1
        pct = lambda n, d: round(100 * n / d, 1) if d else None
        return {
            "model": model or "all",
            "runs": len(ids),
            "accepted": len(accepted),
            "runs_that_submitted": len(submitted),
            "submissions": len(events),
            "submissions_rejected": sum(1 for e in events if not e["passed"]),
            "pct_submissions_rejected": pct(sum(1 for e in events if not e["passed"]), len(events)),
            "pct_first_attempt_pass": pct(len(first_pass), len(submitted)),
            "self_corrected_after_rejection": len(corrected),
            "invalid_advice_blocked": len(blocked),
            "violations_caught": dict(sorted(codes.items(), key=lambda kv: -kv[1])),
            "tool_calls": tools[0],
            "avg_tool_calls_per_run": round(tools[0] / len(ids), 1),
            "avg_tool_latency_ms": round(tools[1] or 0, 1),
        }
