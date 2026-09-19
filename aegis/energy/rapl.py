"""
aegis/energy/rapl.py — RAPL backend wrapping pyRAPL.

Only usable on Linux + Intel + powercap read permissions.
Raises EnergyBackendUnavailable at construction on any other platform.
"""
from __future__ import annotations

import os
import time
from contextlib import contextmanager
from typing import Generator

from aegis.energy.meter import EnergyBackendUnavailable, Reading

_POWERCAP_PATH = "/sys/class/powercap/intel-rapl"


def _check_availability() -> None:
    """Raise EnergyBackendUnavailable if RAPL is not accessible."""
    if not os.path.exists(_POWERCAP_PATH):
        raise EnergyBackendUnavailable(
            f"RAPL powercap interface not found at {_POWERCAP_PATH}. "
            "Requires Linux + Intel CPU."
        )
    try:
        import pyRAPL  # noqa: F401
    except ImportError as exc:
        raise EnergyBackendUnavailable(
            "pyRAPL is not installed. Run: pip install pyrapl"
        ) from exc


class RaplMeter:
    """
    Energy meter backed by Intel RAPL via pyRAPL.

    measured=True — the only backend that reports physically measured energy.
    """

    backend: str = "rapl"
    measured: bool = True

    def __init__(self) -> None:
        _check_availability()
        import pyRAPL
        pyRAPL.setup()
        self._pyrapl = pyRAPL

    @contextmanager
    def measure(self, label: str) -> Generator[Reading, None, None]:
        meter = self._pyrapl.Measurement(label)
        meter.begin()
        t0 = time.perf_counter()
        reading = Reading(
            micro_joules=0.0,
            seconds=0.0,
            backend=self.backend,
            measured=self.measured,
        )
        try:
            yield reading
        finally:
            meter.end()
            elapsed = time.perf_counter() - t0
            pkg = meter.result.pkg
            uj = float(pkg[0]) if pkg else 0.0
            reading.micro_joules = uj
            reading.seconds = elapsed
