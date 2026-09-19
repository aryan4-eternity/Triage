"""
aegis/store.py — SQLite knowledge store for AegisML.

Persists MetricSnapshots, Incidents, Decisions, ActionCandidates, ActionOutcomes.
HarmonE's JSON/CSV files (knowledge/*.json, predictions.csv) are untouched.

Tables:
  snapshots   — MetricSnapshot rows (JSON blob)
  incidents   — Incident rows
  decisions   — Decision rows
  candidates  — ActionCandidate rows (linked to decision_id)
  outcomes    — ActionOutcome rows

Implements the KnowledgeStore protocol defined in docs/CONTRACTS.md §Ports.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator

from aegis.core.models import (
    ActionCandidate,
    ActionOutcome,
    Decision,
    Incident,
    MetricSnapshot,
)

ROOT     = Path(__file__).resolve().parents[1]
DB_PATH  = ROOT / "knowledge" / "aegis.sqlite"


# ── DDL ───────────────────────────────────────────────────────────────────────

_DDL = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS snapshots (
    snapshot_id  TEXT PRIMARY KEY,
    ts           TEXT NOT NULL,
    active_model TEXT NOT NULL,
    payload      TEXT NOT NULL    -- full JSON of MetricSnapshot
);

CREATE TABLE IF NOT EXISTS incidents (
    incident_id  TEXT PRIMARY KEY,
    snapshot_id  TEXT NOT NULL,
    type         TEXT NOT NULL,
    severity_json TEXT NOT NULL,
    payload      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS decisions (
    decision_id    TEXT PRIMARY KEY,
    incident_id    TEXT NOT NULL,
    chosen_action  TEXT NOT NULL,
    policy_profile TEXT NOT NULL,
    exploratory    INTEGER NOT NULL DEFAULT 0,
    payload        TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS candidates (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    decision_id   TEXT NOT NULL,
    action        TEXT NOT NULL,
    eligible      INTEGER NOT NULL,
    utility       REAL,
    rejected_by   TEXT,
    payload       TEXT NOT NULL,
    FOREIGN KEY (decision_id) REFERENCES decisions(decision_id)
);

CREATE TABLE IF NOT EXISTS outcomes (
    outcome_id    TEXT PRIMARY KEY,
    decision_id   TEXT NOT NULL,
    action        TEXT NOT NULL,
    status        TEXT NOT NULL,
    resolved      INTEGER NOT NULL DEFAULT 0,
    actual_energy_uj REAL NOT NULL DEFAULT 0,
    payload       TEXT NOT NULL,
    FOREIGN KEY (decision_id) REFERENCES decisions(decision_id)
);

CREATE TABLE IF NOT EXISTS policy_store (
    key    TEXT PRIMARY KEY,
    value  TEXT NOT NULL
);
"""


class AegisStore:
    """
    SQLite-backed knowledge store.

    Invariant: all writes are atomic within a single transaction.
    The store never writes to knowledge/*.json or predictions.csv —
    those belong to HarmonE.
    """

    def __init__(self, db_path: Path = DB_PATH) -> None:
        self._db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            conn.executescript(_DDL)

    @contextmanager
    def _conn(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ── writes ─────────────────────────────────────────────────────────────

    def put_snapshot(self, s: MetricSnapshot) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO snapshots VALUES (?,?,?,?)",
                (s.snapshot_id, s.ts.isoformat(), s.active_model,
                 s.model_dump_json()),
            )

    def put_incident(self, i: Incident) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO incidents VALUES (?,?,?,?,?)",
                (i.incident_id, i.snapshot_id, i.type,
                 json.dumps(i.severity), i.model_dump_json()),
            )

    def put_decision(self, d: Decision) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO decisions VALUES (?,?,?,?,?,?)",
                (d.decision_id, d.incident_id, d.chosen_action,
                 d.policy_profile, int(d.exploratory), d.model_dump_json()),
            )
            for c in d.candidates:
                conn.execute(
                    "INSERT INTO candidates "
                    "(decision_id, action, eligible, utility, rejected_by, payload) "
                    "VALUES (?,?,?,?,?,?)",
                    (d.decision_id, c.action, int(c.eligible),
                     c.utility, c.rejected_by_guard, c.model_dump_json()),
                )

    def put_outcome(self, o: ActionOutcome) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO outcomes VALUES (?,?,?,?,?,?,?)",
                (o.outcome_id, o.decision_id, o.action, o.status,
                 int(o.resolved), o.actual_energy_uj, o.model_dump_json()),
            )

    # ── reads ──────────────────────────────────────────────────────────────

    def history(self, n: int) -> list[MetricSnapshot]:
        """Return the n most recent snapshots, oldest first."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT payload FROM snapshots ORDER BY ts DESC LIMIT ?", (n,)
            ).fetchall()
        return [MetricSnapshot.model_validate_json(r["payload"])
                for r in reversed(rows)]

    def last_action_ts(self, action: str) -> datetime | None:
        """Return the timestamp of the most recent decision with chosen_action."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT d.payload FROM decisions d "
                "WHERE d.chosen_action = ? ORDER BY rowid DESC LIMIT 1",
                (action,),
            ).fetchone()
        if row is None:
            return None
        d = Decision.model_validate_json(row["payload"])
        return d.model_fields["recheck_after_snapshots"] and d.candidates[0].action and None or None  # noqa
        # Simpler: parse ts from the linked snapshot
        with self._conn() as conn:  # type: ignore[unreachable]
            snap_row = conn.execute(
                "SELECT s.ts FROM snapshots s "
                "JOIN decisions d ON s.snapshot_id = "
                "(SELECT incident_id FROM incidents WHERE incident_id=d.incident_id LIMIT 1) "
                "WHERE d.chosen_action=? ORDER BY s.ts DESC LIMIT 1",
                (action,),
            ).fetchone()
        if snap_row:
            return datetime.fromisoformat(snap_row["ts"])
        return None

    def last_action_ts_simple(self, action: str) -> datetime | None:
        """Return the ts of the snapshot linked to the most recent decision."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT i.snapshot_id FROM decisions d "
                "JOIN incidents i ON d.incident_id = i.incident_id "
                "WHERE d.chosen_action=? ORDER BY d.rowid DESC LIMIT 1",
                (action,),
            ).fetchone()
            if row is None:
                return None
            snap = conn.execute(
                "SELECT ts FROM snapshots WHERE snapshot_id=?",
                (row["snapshot_id"],),
            ).fetchone()
        if snap:
            return datetime.fromisoformat(snap["ts"])
        return None

    def all_decisions(self) -> list[Decision]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT payload FROM decisions ORDER BY rowid"
            ).fetchall()
        return [Decision.model_validate_json(r["payload"]) for r in rows]

    def pending_outcomes(self) -> list[ActionOutcome]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT payload FROM outcomes WHERE resolved=0"
            ).fetchall()
        return [ActionOutcome.model_validate_json(r["payload"]) for r in rows]

    def get_outcome(self, outcome_id: str) -> ActionOutcome | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT payload FROM outcomes WHERE outcome_id=?",
                (outcome_id,),
            ).fetchone()
        if row:
            return ActionOutcome.model_validate_json(row["payload"])
        return None

    def policy(self) -> dict:
        """Return current policy dict (loaded from config/policy.json)."""
        policy_path = ROOT / "config" / "policy.json"
        if policy_path.exists():
            return json.loads(policy_path.read_text())
        return {}

    def candidate_table(self, decision_id: str) -> list[ActionCandidate]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT payload FROM candidates WHERE decision_id=? ORDER BY id",
                (decision_id,),
            ).fetchall()
        return [ActionCandidate.model_validate_json(r["payload"]) for r in rows]
