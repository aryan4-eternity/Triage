"""
aegis/core/monitor.py — AegisML MONITOR stage.

Reads predictions.csv and produces a MetricSnapshot.
Pure functions — no torch, no pyRAPL, no mape imports.

Invariants enforced here:
  1. Mixed-backend window raises MixedBackendError (CONTRACTS.md §predictions.csv).
  2. scenario_tag is written to the snapshot but NEVER read by any core logic.
     tests/test_invariants.py enforces this by grepping aegis/core/.
  3. Window must have at least 2 rows to produce a valid snapshot.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from aegis.core.metrics.accuracy import compute_accuracy
from aegis.core.metrics.drift    import compute_drift
from aegis.core.metrics.energy   import compute_energy
from aegis.core.metrics.equity   import compute_equity
from aegis.core.metrics.latency  import compute_latency
from aegis.core.models import (
    SCHEMA, CostMetrics, MetricSnapshot, ServingConfig, WindowInfo,
)

ROOT = Path(__file__).resolve().parents[2]


class MixedBackendError(ValueError):
    """Raised when a window contains more than one energy backend."""


class InsufficientDataError(ValueError):
    """Raised when the window has fewer than 2 rows."""


def _load_thresholds() -> dict:
    path = ROOT / "knowledge" / "thresholds.json"
    if path.exists():
        return json.loads(path.read_text())
    return {"E_m": 0.0, "E_M": 25000.0, "beta": 0.95, "gamma": 0.8,
            "max_energy": 1.0}


def _load_mape_info() -> dict:
    path = ROOT / "knowledge" / "mape_info.json"
    if path.exists():
        return json.loads(path.read_text())
    return {"ema_scores": {"lstm": 0.5, "svm": 0.5, "linear": 0.5},
            "current_energy_threshold": 1.0}


def _load_serving() -> ServingConfig:
    path = ROOT / "knowledge" / "serving.json"
    if path.exists():
        try:
            cfg = json.loads(path.read_text())
            return ServingConfig(**cfg)
        except Exception:
            pass
    return ServingConfig()


def produce_snapshot(
    window_df: pd.DataFrame,
    active_model: str,
    scenario_tag: Optional[str] = None,
) -> MetricSnapshot:
    """
    Produce a MetricSnapshot from a window of predictions.

    Args:
        window_df:    DataFrame slice of predictions.csv (SCHEMA columns).
        active_model: Name of the currently active model.
        scenario_tag: Simulator tag — written but NEVER read by core logic.

    Raises:
        InsufficientDataError: window has < 2 rows.
        MixedBackendError:     window contains multiple energy backends.

    Returns:
        Immutable MetricSnapshot.
    """
    if len(window_df) < 2:
        raise InsufficientDataError(
            f"Window has only {len(window_df)} rows; need ≥ 2."
        )

    # ── Backend consistency check ─────────────────────────────────────────
    backend_col = SCHEMA["energy_backend"]
    if backend_col in window_df.columns:
        backends = window_df[backend_col].dropna().unique()
        if len(backends) > 1:
            raise MixedBackendError(
                f"Window contains multiple energy backends: {sorted(backends)}. "
                "Cannot produce a valid snapshot — runs must use a single backend."
            )
        energy_backend = str(backends[0]) if len(backends) == 1 else "estimator"
    else:
        energy_backend = "estimator"

    # ── Load config ───────────────────────────────────────────────────────
    thresholds = _load_thresholds()
    mape_info  = _load_mape_info()

    e_m = float(thresholds.get("E_m", 0.0))
    e_M = float(thresholds.get("E_M", 25000.0))
    beta  = float(thresholds.get("beta",  0.95))
    gamma = float(thresholds.get("gamma", 0.8))
    current_thr  = float(mape_info.get("current_energy_threshold", 1.0))
    original_thr = float(thresholds.get("max_energy", 1.0))
    prev_ema     = float(mape_info.get("ema_scores", {}).get(active_model, 0.5))

    # ── Metrics ───────────────────────────────────────────────────────────
    energy_metrics, new_thr = compute_energy(
        window_df, e_m, e_M, current_thr, original_thr
    )
    accuracy_metrics = compute_accuracy(
        window_df, prev_ema, beta, gamma, energy_metrics.normalized
    )
    latency_metrics  = compute_latency(window_df)
    equity_metrics   = compute_equity(window_df)

    # Drift: need 2 × window for reference vs current
    drift_half = len(window_df) // 2
    if drift_half >= 2:
        ref  = window_df[SCHEMA["true_value"]].iloc[:drift_half]
        curr = window_df[SCHEMA["true_value"]].iloc[drift_half:]
        drift_metrics = compute_drift(ref, curr)
    else:
        from aegis.core.models import DriftMetrics
        drift_metrics = DriftMetrics(kl_div=0.0, psi=0.0)

    # Cost: simple wall-clock / energy proxy
    inf_col = SCHEMA["inference_time"]
    wall_s  = float(window_df[inf_col].sum()) if inf_col in window_df.columns else 0.0
    # Rough cost: ₹0.005 per 1k inferences (placeholder; calibrate in T2.9)
    inr_per_1k = round(wall_s * 0.005 / max(len(window_df), 1) * 1000.0, 6)
    cost_metrics = CostMetrics(inr_per_1k=inr_per_1k, wall_clock_s=wall_s)

    stations = int(window_df[SCHEMA["station_id"]].nunique()
                   if SCHEMA["station_id"] in window_df.columns else 1)

    snapshot = MetricSnapshot(
        snapshot_id  = f"snap_{uuid.uuid4().hex[:8]}",
        ts           = datetime.now(timezone.utc),
        active_model = active_model,
        window       = WindowInfo(
            rows=len(window_df),
            energy_backend=energy_backend,
            stations=stations,
        ),
        accuracy = accuracy_metrics,
        drift    = drift_metrics,
        latency  = latency_metrics,
        energy   = energy_metrics,
        cost     = cost_metrics,
        equity   = equity_metrics,
        serving  = _load_serving(),
        scenario_tag = scenario_tag,
    )
    return snapshot


def read_window(
    predictions_path: Path,
    window_size: int = 1200,
    last_line: int = 0,
) -> tuple[pd.DataFrame, int]:
    """
    Read the most recent `window_size` rows from predictions.csv.

    Returns (window_df, new_last_line).
    """
    try:
        df = pd.read_csv(predictions_path)
        df.columns = df.columns.str.strip()
    except FileNotFoundError:
        return pd.DataFrame(), last_line

    if len(df) < window_size:
        return pd.DataFrame(), last_line

    window = df.iloc[-window_size:].reset_index(drop=True)
    return window, len(df)
