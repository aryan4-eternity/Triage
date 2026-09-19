"""
aegis/actuators/retrain_act.py — Triggers model retraining via retrain.py.
"""
from __future__ import annotations

import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def retrain(model_name: str | None = None) -> dict:
    """
    Call retrain.py for the given model (or active model if None).

    Returns outcome-compatible dict.
    """
    cmd = [sys.executable, str(ROOT / "retrain.py")]
    if model_name:
        cmd += ["--model", model_name]

    print(f"[actuator] retrain {model_name or '(active)'} …")
    result = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    ok     = result.returncode == 0
    print(f"[actuator] retrain done  rc={result.returncode}")
    if not ok:
        print(f"[actuator] stderr: {result.stderr[:500]}")

    return {
        "action": f"RETRAIN_CURRENT:{model_name or 'active'}",
        "ts":     datetime.now(timezone.utc).isoformat(),
        "ok":     ok,
        "stdout": result.stdout[-500:] if result.stdout else "",
    }
