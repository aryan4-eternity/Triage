"""
aegis/core/metrics/accuracy.py — Accuracy metric extractor.

Pure function over a DataFrame window. No torch, no pyRAPL, no mape imports.

EMA formula reused verbatim from mape/monitor.py (HarmonE):
    model_score = β·r2 + (1-β)·(1 - energy_norm)
    ema         = γ·model_score + (1-γ)·prev_ema
Cite: mape/monitor.py::monitor_mape()

NOTE: ema_score already blends energy into accuracy. AegisML's utility uses
raw r2 for accuracy severity to avoid double-counting energy. ema_score is
kept for comparability with the base arms only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

from aegis.core.models import AccuracyMetrics, SCHEMA


def compute_accuracy(
    window: pd.DataFrame,
    prev_ema: float,
    beta: float,
    gamma: float,
    energy_normalized: float,
) -> AccuracyMetrics:
    """
    Compute accuracy metrics for a prediction window.

    Invariant: window must have true_value and predicted_value columns.

    Args:
        window:            DataFrame with SCHEMA columns present.
        prev_ema:          Previous EMA score for the active model.
        beta:              HarmonE β weight (accuracy vs energy in EMA).
        gamma:             HarmonE γ EMA smoothing factor.
        energy_normalized: Pre-computed normalised energy [0, 1].

    Returns:
        AccuracyMetrics with r2, mae, ema_score.
    """
    true_col = SCHEMA["true_value"]
    pred_col = SCHEMA["predicted_value"]

    y_true = window[true_col].values
    y_pred = window[pred_col].values

    r2  = float(r2_score(y_true, y_pred))
    mae = float(mean_absolute_error(y_true, y_pred))

    # HarmonE EMA formula — cite: mape/monitor.py::monitor_mape()
    model_score = beta * r2 + (1.0 - beta) * (1.0 - energy_normalized)
    ema_score   = gamma * model_score + (1.0 - gamma) * prev_ema

    return AccuracyMetrics(r2=r2, mae=mae, ema_score=ema_score)
