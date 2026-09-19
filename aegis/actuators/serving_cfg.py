"""
aegis/actuators/serving_cfg.py — Writes knowledge/serving.json.

inference.py re-reads this on every iteration (same cadence as model.csv).
Changing batch_size / seq_length / sampling_rate takes effect within one loop.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

ROOT          = Path(__file__).resolve().parents[2]
_SERVING_JSON = ROOT / "knowledge" / "serving.json"

_DEFAULTS = {"batch_size": 1, "seq_length": 5, "sampling_rate": 1.0}


def _load() -> dict:
    if _SERVING_JSON.exists():
        try:
            return json.loads(_SERVING_JSON.read_text())
        except Exception:
            pass
    return dict(_DEFAULTS)


def set_serving(
    batch_size: Optional[int]   = None,
    seq_length: Optional[int]   = None,
    sampling_rate: Optional[float] = None,
) -> dict:
    """
    Update serving.json with the provided fields.
    Unspecified fields keep their current values.
    """
    cfg = _load()
    if batch_size    is not None:
        cfg["batch_size"]    = int(batch_size)
    if seq_length    is not None:
        cfg["seq_length"]    = int(seq_length)
    if sampling_rate is not None:
        cfg["sampling_rate"] = float(sampling_rate)
    cfg["updated_at"] = datetime.now(timezone.utc).isoformat()

    _SERVING_JSON.write_text(json.dumps(cfg, indent=2))
    print(f"[actuator] set_serving → {cfg}")
    return {"action": "SET_SERVING", "ts": cfg["updated_at"], "cfg": cfg}


def batch_inference(batch_size: int = 8) -> dict:
    """Convenience: set batch_size to batch_size."""
    return set_serving(batch_size=batch_size)


def lower_sampling(rate: float = 0.5) -> dict:
    """Convenience: set sampling_rate."""
    return set_serving(sampling_rate=rate)


def reduce_window(seq_length: int = 3) -> dict:
    """Convenience: set seq_length."""
    return set_serving(seq_length=seq_length)
