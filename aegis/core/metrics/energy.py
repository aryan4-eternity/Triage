"""
aegis/core/metrics/energy.py — Energy metric extractor.

Pure function. No torch, no pyRAPL, no mape imports.

Normalisation formula reused from mape/monitor.py (HarmonE):
    energy_normalized = (mean(energy) - E_m) / (E_M - E_m)
Cite: mape/monitor.py::monitor_mape()

AegisML adds:
  - Hard clamp to [0, 1] (BUG-5 fix)
  - Integral-controller threshold tracking (matches HarmonE's analyse.py)
  - Backend label forwarded to every Reading
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from aegis.core.models import EnergyMetrics, SCHEMA

_CLAMP_LOWER_FACTOR = 0.10   # clamp integral thr to [0.1·orig, 2·orig]
_CLAMP_UPPER_FACTOR = 2.0


def compute_energy(
    window: pd.DataFrame,
    e_m: float,
    e_M: float,
    current_threshold: float,
    original_threshold: float,
) -> tuple[EnergyMetrics, float]:
    """
    Compute normalised energy and update the integral-controller threshold.

    Invariant: returned energy.normalized is in [0, 1] (BUG-5 fix).
    Invariant: returned new_threshold is clamped to
               [0.1·original_threshold, 2·original_threshold].

    Integral controller (HarmonE formula, cite: mape/analyse.py::analyse_mape()):
        new_thr = current_thr + 0.95·(original_thr − used_energy_norm)

    Args:
        window:             DataFrame with energy_uj column.
        e_m:                Minimum energy baseline (µJ) from thresholds.json.
        e_M:                Maximum energy baseline (µJ) from thresholds.json.
        current_threshold:  Current dynamic threshold (normalised [0,1]).
        original_threshold: Original threshold from thresholds.json.

    Returns:
        (EnergyMetrics, new_threshold)
    """
    col = SCHEMA["energy_uj"]
    if col not in window.columns or window.empty:
        return (
            EnergyMetrics(
                uj_per_inference=0.0,
                normalized=0.0,
                current_threshold=current_threshold,
                measured=False,
            ),
            current_threshold,
        )

    mean_uj = float(window[col].mean())
    denom   = e_M - e_m
    if denom > 0:
        normalized = (mean_uj - e_m) / denom
    else:
        normalized = 0.0
    normalized = float(np.clip(normalized, 0.0, 1.0))  # BUG-5 fix

    # HarmonE integral controller (cite: mape/analyse.py)
    new_threshold = current_threshold + 0.95 * (original_threshold - normalized)
    # Clamp: prevent runaway negative threshold (POLICY_ENGINE.md §1)
    lo = _CLAMP_LOWER_FACTOR * original_threshold
    hi = _CLAMP_UPPER_FACTOR * original_threshold
    new_threshold = float(np.clip(new_threshold, lo, hi))

    # Determine measured flag from backend column
    backend_col = SCHEMA["energy_backend"]
    measured = False
    if backend_col in window.columns:
        backends = window[backend_col].dropna().unique()
        measured = len(backends) == 1 and backends[0] == "rapl"

    return (
        EnergyMetrics(
            uj_per_inference=mean_uj,
            normalized=normalized,
            current_threshold=new_threshold,
            measured=measured,
        ),
        new_threshold,
    )
