"""
aegis/energy/meter.py — EnergyMeter protocol and factory.

Selection via AEGIS_ENERGY=rapl|codecarbon|estimator|auto (default: auto).
Every Reading carries its backend so mixed-backend windows can be detected
and refused downstream (CONTRACTS.md §predictions.csv).
"""
from __future__ import annotations

import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Generator, Protocol, runtime_checkable


class EnergyBackendUnavailable(RuntimeError):
    """Raised when a requested energy backend cannot initialise."""


@dataclass
class Reading:
    micro_joules: float
    seconds: float
    backend: str   # "rapl" | "codecarbon" | "estimator"
    measured: bool  # True only for rapl


@runtime_checkable
class EnergyMeter(Protocol):
    backend: str
    measured: bool

    @contextmanager
    def measure(self, label: str) -> Generator[Reading, None, None]:
        ...


def get_meter() -> EnergyMeter:
    """
    Factory: return the best available EnergyMeter for this machine.

    Order when AEGIS_ENERGY=auto (default):
      1. rapl    — real Intel RAPL (Linux/Intel only)
      2. codecarbon — hardware-model estimate with RAPL where available
      3. estimator — time × TDP proxy, always available

    Raises EnergyBackendUnavailable only if a specific backend is requested
    and unavailable.  auto never raises.
    """
    requested = os.environ.get("AEGIS_ENERGY", "auto").lower()

    def _try_rapl():
        from aegis.energy.rapl import RaplMeter
        return RaplMeter()

    def _try_codecarbon():
        from aegis.energy.codecarbon import CodeCarbonMeter
        return CodeCarbonMeter()

    def _try_estimator():
        from aegis.energy.estimator import EstimatorMeter
        return EstimatorMeter()

    if requested == "rapl":
        try:
            m = _try_rapl()
            _log_backend(m)
            return m
        except Exception as exc:
            raise EnergyBackendUnavailable(
                "RAPL backend unavailable. "
                "Requires Linux + Intel CPU + powercap read permissions. "
                f"Original error: {exc}"
            ) from exc

    if requested == "codecarbon":
        try:
            m = _try_codecarbon()
            _log_backend(m)
            return m
        except Exception as exc:
            raise EnergyBackendUnavailable(
                f"CodeCarbon backend unavailable: {exc}"
            ) from exc

    if requested == "estimator":
        m = _try_estimator()
        _log_backend(m)
        return m

    # auto: try in order
    for factory in [_try_rapl, _try_codecarbon, _try_estimator]:
        try:
            m = factory()
            _log_backend(m)
            return m
        except Exception:
            continue

    # estimator never fails, so we should never reach here
    raise EnergyBackendUnavailable("No energy backend could be initialised.")


def _log_backend(meter: EnergyMeter) -> None:
    label = "MEASURED" if meter.measured else "ESTIMATED"
    print(f"[aegis.energy] backend={meter.backend} ({label})")
