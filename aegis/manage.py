"""
aegis/manage.py — AegisML MAPE-K loop ("aegis" approach).

Mirrors mape/manage.py's threading structure.
Controller overhead is logged to knowledge/mape_log.csv in HarmonE's format
so the two are directly comparable in the evaluation table.

Starts when approach.conf contains "aegis".
"""
from __future__ import annotations

import csv
import json
import os
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aegis.core.analyze  import analyze
from aegis.core.execute  import execute as aegis_execute
from aegis.core.learn    import resolve_outcomes
from aegis.core.monitor  import produce_snapshot, read_window
from aegis.core.plan     import plan
from aegis.energy        import get_meter
from aegis.store         import AegisStore

import numpy as np

PREDICTIONS_FILE = ROOT / "knowledge" / "predictions.csv"
MODEL_CSV        = ROOT / "knowledge" / "model.csv"
LOG_FILE         = ROOT / "knowledge" / "mape_log.csv"
CONFIG_FILE      = ROOT / "approach.conf"

_WINDOW_SIZE = 1200
_SLEEP_S     = 40.0         # same cadence as mape/manage.py run_execute_mape

_meter = get_meter()
_store = AegisStore()
_rng   = np.random.default_rng(42)

_violation_counts: dict[str, int] = {}
_pending_outcomes: list          = []
_snapshot_history: list          = []
_MAX_HISTORY = 25


def _log_energy(label: str, energy_uj: float) -> None:
    """Append to mape_log.csv in HarmonE's format (energy in Joules)."""
    energy_j = energy_uj / 1_000_000.0
    with open(LOG_FILE, mode="a", newline="") as fh:
        csv.writer(fh).writerow([label, round(energy_j, 8)])


def _ensure_log() -> None:
    if not LOG_FILE.exists():
        with open(LOG_FILE, mode="w", newline="") as fh:
            csv.writer(fh).writerow(["function", "energy_joules"])


def _active_model() -> str:
    try:
        return MODEL_CSV.read_text().strip()
    except FileNotFoundError:
        return "lstm"


def run_aegis_cycle() -> None:
    """One full MONITOR → ANALYZE → PLAN → EXECUTE → LEARN cycle."""
    global _pending_outcomes, _snapshot_history

    window_df, _ = read_window(PREDICTIONS_FILE, _WINDOW_SIZE)
    if window_df.empty:
        print("[aegis] waiting for enough predictions…")
        return

    active = _active_model()

    with _meter.measure("aegis_cycle") as r:
        # MONITOR
        try:
            snapshot = produce_snapshot(window_df, active)
        except Exception as exc:
            print(f"[aegis] MONITOR error: {exc}")
            return

        _store.put_snapshot(snapshot)

        # ANALYZE
        incident = analyze(snapshot, _snapshot_history, _violation_counts)
        _store.put_incident(incident)

        # Keep history bounded
        _snapshot_history.append(snapshot)
        if len(_snapshot_history) > _MAX_HISTORY:
            _snapshot_history.pop(0)

        # PLAN
        decision = plan(incident, snapshot, _store, _rng)
        _store.put_decision(decision)

        # EXECUTE
        if incident.type != "NONE" or decision.exploratory:
            outcome = aegis_execute(decision, snapshot)
            _pending_outcomes.append(outcome)
            _store.put_outcome(outcome)

        # LEARN — resolve pending
        _pending_outcomes = resolve_outcomes(
            _pending_outcomes, snapshot, _store
        )

        print(
            f"[aegis] snap={snapshot.snapshot_id}  "
            f"incident={incident.type}  "
            f"action={decision.chosen_action}  "
            f"energy_backend={snapshot.window.energy_backend}"
        )

    _log_energy("aegis_cycle", r.micro_joules)


def _loop() -> None:
    _ensure_log()
    print(f"[aegis] starting loop  energy_backend={_meter.backend}")
    while True:
        time.sleep(_SLEEP_S)
        try:
            run_aegis_cycle()
        except Exception as exc:
            print(f"[aegis] cycle error: {exc}")


if __name__ == "__main__":
    t = threading.Thread(target=_loop, daemon=True)
    t.start()
    print("[aegis] manager running. Ctrl-C to stop.")
    threading.Event().wait()
