"""
aegis/core/boundaries.py — Rolling boundary computation for AegisML ANALYZE.

Pure functions. No torch, no pyRAPL, no mape imports.
No float literals — all thresholds from config/boundaries.json.

Modes:
  "dynamic"  — rolling mean ± k·std with hard clamps (POLICY_ENGINE.md §1)
  "hard"     — hard bounds only (cold start, < window samples)
  "integral" — HarmonE's energy integral controller (cite: mape/analyse.py)

Cite: ARCHITECTURE.md §4, POLICY_ENGINE.md §1.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np

from aegis.core.models import BoundaryReport

ROOT           = Path(__file__).resolve().parents[2]
_BOUNDARIES_JSON = ROOT / "config" / "boundaries.json"

# Zero-variance widening: use max(sd, zero_var_fraction · |µ|)
_ZERO_VAR_FRACTION = 0.02


def _load_cfg() -> dict:
    if _BOUNDARIES_JSON.exists():
        return json.loads(_BOUNDARIES_JSON.read_text())
    return {"window": 20, "k": 2.0, "persistence": 2, "metrics": {}}


def compute_boundary(
    metric_key: str,
    history: Sequence[float],
    current_value: float,
    consecutive_violations: int,
) -> BoundaryReport:
    """
    Compute the boundary and violation status for a single metric.

    Invariant: severity ∈ [0, 1].
    Invariant: mode is "dynamic", "hard", or "integral".

    Args:
        metric_key:             e.g. "accuracy.r2" or "drift.kl_div"
        history:                Recent metric values (oldest first).
        current_value:          The value from the current snapshot.
        consecutive_violations: How many consecutive violations before this one.

    Returns:
        BoundaryReport
    """
    cfg      = _load_cfg()
    window   = int(cfg.get("window", 20))
    k        = float(cfg.get("k", 2.0))
    m_cfg    = cfg.get("metrics", {}).get(metric_key, {})

    direction   = m_cfg.get("direction", "upper")
    mode_hint   = m_cfg.get("mode", "dynamic")
    hard_floor  = m_cfg.get("hard_floor")
    hard_ceiling = m_cfg.get("hard_ceiling")

    hist = list(history)

    # ── Integral mode (HarmonE energy controller) ─────────────────────────
    if mode_hint == "integral":
        orig_thr  = float(m_cfg.get("original_threshold", 1.0))
        # current_value is energy.normalized (clamped [0,1]).
        # The dynamic threshold is in the most recent history snapshot.
        # Pull current_threshold from most recent history entry if available.
        if hist:
            # We need the snapshot's energy.current_threshold — but we only
            # have floats in hist (the extractor gives energy.normalized).
            # Use the integral formula directly: after N overspend cycles
            # the effective threshold decays. We approximate by using
            # current_value itself vs orig_thr with the integral formula.
            dyn_thr = orig_thr
            for h_val in hist[-10:]:
                dyn_thr = dyn_thr + 0.95 * (orig_thr - h_val)
        else:
            dyn_thr = orig_thr

        # Clamp threshold (POLICY_ENGINE.md §1)
        lo = 0.1 * orig_thr; hi = 2.0 * orig_thr
        dyn_thr = float(np.clip(dyn_thr, lo, hi))

        violated  = current_value > dyn_thr
        hard_b    = hard_ceiling or orig_thr * 2.0
        severity  = _severity(current_value, dyn_thr, hard_b, "upper") if violated else 0.0
        return BoundaryReport(
            metric=metric_key,
            value=current_value,
            lower=None,
            upper=dyn_thr,
            mode="integral",
            hard_ceiling=hard_ceiling,
            violated=violated,
            severity=severity,
            consecutive_violations=consecutive_violations,
        )

    # ── Cold start — not enough history ───────────────────────────────────
    if len(hist) < window:
        lower = float(hard_floor)  if hard_floor  is not None else None
        upper = float(hard_ceiling) if hard_ceiling is not None else None
        if direction == "lower":
            violated = lower is not None and current_value < lower
            bound    = lower
            hard_b   = lower
        else:
            violated = upper is not None and current_value > upper
            bound    = upper
            hard_b   = upper
        severity = _severity(current_value, bound, hard_b, direction)
        return BoundaryReport(
            metric=metric_key, value=current_value,
            lower=lower, upper=upper, mode="hard",
            hard_floor=hard_floor, hard_ceiling=hard_ceiling,
            violated=violated, severity=severity,
            consecutive_violations=consecutive_violations,
        )

    # ── Dynamic mode ──────────────────────────────────────────────────────
    arr = np.array(hist[-window:], dtype=float)
    mu  = float(arr.mean())
    sd  = float(arr.std())

    # Zero-variance widening
    if sd < 1e-6:
        sd = max(sd, _ZERO_VAR_FRACTION * abs(mu))

    lower_dyn  = None
    upper_dyn  = None
    bound_used = None
    hard_b     = None

    if direction == "lower":
        lower_dyn = mu - k * sd
        if hard_floor is not None:
            lower_dyn = max(lower_dyn, float(hard_floor))
        violated  = current_value < lower_dyn
        bound_used = lower_dyn
        hard_b    = hard_floor
    else:
        upper_dyn = mu + k * sd
        if hard_ceiling is not None:
            upper_dyn = min(upper_dyn, float(hard_ceiling))
        violated  = current_value > upper_dyn
        bound_used = upper_dyn
        hard_b    = hard_ceiling

    severity = _severity(current_value, bound_used, hard_b, direction)

    return BoundaryReport(
        metric=metric_key,
        value=current_value,
        lower=lower_dyn,
        upper=upper_dyn,
        mode="dynamic",
        baseline_mean=mu, baseline_std=sd, k=k,
        hard_floor=hard_floor, hard_ceiling=hard_ceiling,
        violated=violated, severity=severity,
        consecutive_violations=consecutive_violations,
    )


def _severity(
    value: float,
    bound: float | None,
    hard_bound: float | None,
    direction: str,
) -> float:
    """
    Severity = clamp(|value − bound| / |hard_bound − bound|, 0, 1).

    When bound ≈ hard_bound (or they are equal), the denominator collapses.
    In that case we use the ratio |value - bound| / max(|bound|, 1) clamped
    to [0, 1] — a saturating fallback that still produces a non-zero severity
    when the value massively overshoots the boundary.
    """
    if bound is None:
        return 0.0
    diff  = abs(value - bound)
    if hard_bound is None:
        # No hard limit: normalise against the bound magnitude itself
        denom = max(abs(bound), 1.0)
        return float(np.clip(diff / denom, 0.0, 1.0))
    denom = abs(hard_bound - bound)
    if denom < 1e-6:
        # hard_bound ≈ bound: use saturating fallback
        denom = max(abs(bound), 1.0)
    return float(np.clip(diff / denom, 0.0, 1.0))
