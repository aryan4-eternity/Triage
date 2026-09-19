"""
aegis/actuators/model_switch.py — Writes knowledge/model.csv.

Writes exactly what HarmonE's execute.py writes, so the inference loop
picks up the new model within one iteration.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

ROOT       = Path(__file__).resolve().parents[2]
_MODEL_CSV = ROOT / "knowledge" / "model.csv"


def switch_model(model_name: str) -> dict:
    """
    Write model_name to knowledge/model.csv.

    Returns an outcome-compatible dict (actual_energy_uj from config).
    """
    _MODEL_CSV.write_text(model_name)
    print(f"[actuator] switch_model → {model_name}")
    return {
        "action": f"SWITCH_MODEL:{model_name}",
        "ts":     datetime.now(timezone.utc).isoformat(),
        "notes":  f"wrote {model_name} to {_MODEL_CSV}",
    }
