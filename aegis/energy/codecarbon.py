"""
aegis/energy/codecarbon.py — CodeCarbon backend.

Uses RAPL where available; otherwise falls back to its own hardware-model
estimate.  measured=False in both cases (we can't distinguish RAPL-via-CC
from CC's estimator at runtime without introspecting CC internals, and the
conservative label is the honest one).
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Generator

from aegis.energy.meter import EnergyBackendUnavailable, Reading


class CodeCarbonMeter:
    """
    Energy meter backed by codecarbon's EmissionsTracker.

    measured is always False — codecarbon may use RAPL internally but
    we cannot confirm it at the Reading level, so we label conservatively.
    """

    backend: str = "codecarbon"
    measured: bool = False

    def __init__(self) -> None:
        try:
            from codecarbon import EmissionsTracker  # noqa: F401
        except ImportError as exc:
            raise EnergyBackendUnavailable(
                "codecarbon is not installed. Run: pip install codecarbon"
            ) from exc

    @contextmanager
    def measure(self, label: str) -> Generator[Reading, None, None]:
        from codecarbon import EmissionsTracker

        reading = Reading(
            micro_joules=0.0,
            seconds=0.0,
            backend=self.backend,
            measured=self.measured,
        )
        tracker = EmissionsTracker(
            project_name=label,
            log_level="error",
            save_to_file=False,
            save_to_api=False,
        )
        t0 = time.perf_counter()
        tracker.start()
        try:
            yield reading
        finally:
            emissions_kg = tracker.stop()  # kg CO2-eq
            elapsed = time.perf_counter() - t0
            # Convert kg CO2 → kWh → µJ
            # 1 kg CO2 ≈ 0.233 kWh (global average, labelled as estimate)
            # 1 kWh = 3.6e9 µJ
            kwh = (emissions_kg or 0.0) * 0.233
            uj = kwh * 3.6e9
            reading.micro_joules = uj
            reading.seconds = elapsed
