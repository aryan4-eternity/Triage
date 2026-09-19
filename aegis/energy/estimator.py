"""
aegis/energy/estimator.py — Time × TDP proxy energy meter.

Formula: µJ ≈ elapsed_seconds × TDP_watts × utilisation × 1e6

TDP_watts and utilisation come from config/hardware.json.
measured=False — always labelled as an estimate.

This backend is always available (no external deps beyond stdlib).
"""
from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

from aegis.energy.meter import Reading

_DEFAULT_TDP_WATTS = 65.0       # generic desktop/laptop default
_DEFAULT_UTILISATION = 0.35     # conservative fraction of TDP during inference

# Resolve hardware config relative to repo root regardless of CWD
_ROOT = Path(__file__).resolve().parents[2]
_HARDWARE_JSON = _ROOT / "config" / "hardware.json"


def _load_tdp() -> tuple[float, float]:
    """Return (tdp_watts, utilisation) from hardware.json if present."""
    if _HARDWARE_JSON.exists():
        try:
            with open(_HARDWARE_JSON) as fh:
                cfg = json.load(fh)
            tdp = float(cfg.get("tdp_watts", _DEFAULT_TDP_WATTS))
            util = float(cfg.get("utilisation", _DEFAULT_UTILISATION))
            return tdp, util
        except Exception:
            pass
    return _DEFAULT_TDP_WATTS, _DEFAULT_UTILISATION


class EstimatorMeter:
    """
    Energy meter that estimates µJ from elapsed time and hardware TDP.

    Always available; always measured=False.
    """

    backend: str = "estimator"
    measured: bool = False

    def __init__(self) -> None:
        self._tdp, self._util = _load_tdp()

    @contextmanager
    def measure(self, label: str) -> Generator[Reading, None, None]:  # noqa: ARG002
        reading = Reading(
            micro_joules=0.0,
            seconds=0.0,
            backend=self.backend,
            measured=self.measured,
        )
        t0 = time.perf_counter()
        try:
            yield reading
        finally:
            elapsed = time.perf_counter() - t0
            uj = elapsed * self._tdp * self._util * 1_000_000.0
            reading.micro_joules = uj
            reading.seconds = elapsed
