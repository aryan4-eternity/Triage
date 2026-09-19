"""
aegis/core/metrics/equity.py — Per-station equity metric extractor.

Pure function. No torch, no pyRAPL, no mape imports.

Equity here means performance disparity across sensor stations
(not demographic fairness — PEMS has no protected attributes).

Design choice (documented here per EXTENSION_PLAN.md §2.2):
  With 8 stations and a 1200-row window, each station contributes ~150 rows
  per window — thin for R². We use per-station MAE instead of R² as the
  primary within-window equity metric (more stable at small n), but also
  compute per-station R² when enough rows exist (≥30 per station).
  The worst_r2 field falls back to 1.0 - worst_mae_norm when n < 30.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

from aegis.core.models import EquityMetrics, SCHEMA

_MIN_ROWS_FOR_R2 = 30


def compute_equity(window: pd.DataFrame) -> EquityMetrics:
    """
    Compute per-station performance equity from a prediction window.

    Invariant: if station_id column is absent, all rows are treated as
    a single station "ST_00".

    Returns:
        EquityMetrics with worst_station, worst_r2, gap, per_station_r2.
    """
    true_col    = SCHEMA["true_value"]
    pred_col    = SCHEMA["predicted_value"]
    station_col = SCHEMA["station_id"]

    if window.empty or true_col not in window.columns:
        return EquityMetrics(
            worst_station="ST_00", worst_r2=1.0, gap=0.0
        )

    if station_col not in window.columns:
        window = window.copy()
        window[station_col] = "ST_00"

    per_station_r2: dict[str, float] = {}

    for station, grp in window.groupby(station_col):
        if len(grp) < 2:
            continue
        y_true = grp[true_col].values
        y_pred = grp[pred_col].values
        if len(grp) >= _MIN_ROWS_FOR_R2:
            r2 = float(r2_score(y_true, y_pred))
        else:
            # Fallback: normalised 1-MAE proxy
            mae  = mean_absolute_error(y_true, y_pred)
            span = float(np.ptp(y_true)) if np.ptp(y_true) > 0 else 1.0
            r2   = float(np.clip(1.0 - mae / span, -1.0, 1.0))
        per_station_r2[str(station)] = round(r2, 6)

    if not per_station_r2:
        return EquityMetrics(worst_station="ST_00", worst_r2=1.0, gap=0.0)

    worst_station = min(per_station_r2, key=per_station_r2.__getitem__)
    worst_r2      = per_station_r2[worst_station]
    best_r2       = max(per_station_r2.values())
    gap           = round(best_r2 - worst_r2, 6)

    return EquityMetrics(
        worst_station=worst_station,
        worst_r2=worst_r2,
        gap=gap,
        per_station_r2=per_station_r2,
    )
