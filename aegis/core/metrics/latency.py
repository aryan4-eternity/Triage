"""
aegis/core/metrics/latency.py — Latency metric extractor.

Pure function. No torch, no pyRAPL, no mape imports.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from aegis.core.models import LatencyMetrics, SCHEMA


def compute_latency(window: pd.DataFrame) -> LatencyMetrics:
    """
    Compute p50 and p95 inference latency from a prediction window.

    Invariant: window must contain the inference_time column (seconds).
    Returns latency in milliseconds.

    Args:
        window: DataFrame with SCHEMA columns present.

    Returns:
        LatencyMetrics with p50_ms and p95_ms.
    """
    col = SCHEMA["inference_time"]
    if col not in window.columns or window.empty:
        return LatencyMetrics(p50_ms=0.0, p95_ms=0.0)

    times_ms = window[col].values.astype(float) * 1000.0  # s → ms
    p50 = float(np.percentile(times_ms, 50))
    p95 = float(np.percentile(times_ms, 95))
    return LatencyMetrics(p50_ms=p50, p95_ms=p95)
