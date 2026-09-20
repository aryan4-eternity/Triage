"""
aegis/core/metrics/drift.py — Drift metric extractor (KL divergence + PSI).

Pure function. No torch, no pyRAPL, no mape imports.

KL formula reused from mape/monitor.py::monitor_drift() (HarmonE).
Cite: mape/monitor.py::monitor_drift()

PSI (Population Stability Index):
    PSI = Σ (actual% - expected%) · ln(actual% / expected%)
    Rule of thumb: PSI < 0.1 → stable, 0.1–0.25 → minor, > 0.25 → major shift.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import entropy

from aegis.core.models import DriftMetrics, SCHEMA

_BINS     = 50
_EPSILON  = 1e-10
_PSI_BINS = 10


def compute_drift(
    reference: pd.Series,
    current: pd.Series,
) -> DriftMetrics:
    """
    Compute KL divergence and PSI between reference and current windows.

    Invariant: both series must be numeric and non-empty.
    KL parity: uses identical histogram parameters to mape/monitor.py so
    drift detection is comparable across HarmonE and AegisML arms.
    Shared bin edges are derived from the union of both series to correctly
    detect distribution shifts between non-overlapping ranges.

    Args:
        reference: Earlier window of true_value (1200 rows).
        current:   More recent window of true_value (1200 rows).

    Returns:
        DriftMetrics with kl_div and psi.
    """
    ref  = reference.values.astype(float)
    curr = current.values.astype(float)

    # Compute KL divergence with SHARED bin edges (union range)
    # This ensures separated distributions produce high KL, not low KL
    combined_min = min(ref.min(), curr.min())
    combined_max = max(ref.max(), curr.max())
    shared_edges = np.linspace(combined_min, combined_max, _BINS + 1)

    h_ref,  _ = np.histogram(ref,  bins=shared_edges, density=True)
    h_curr, _ = np.histogram(curr, bins=shared_edges, density=True)
    kl_div = float(entropy(h_ref + _EPSILON, h_curr + _EPSILON))

    # PSI — use shared bin edges derived from reference
    bin_edges = np.percentile(ref, np.linspace(0, 100, _PSI_BINS + 1))
    bin_edges[0]  -= 1e-6
    bin_edges[-1] += 1e-6

    ref_counts  = np.histogram(ref,  bins=bin_edges)[0]
    curr_counts = np.histogram(curr, bins=bin_edges)[0]

    ref_pct  = ref_counts  / (ref_counts.sum()  + _EPSILON)
    curr_pct = curr_counts / (curr_counts.sum() + _EPSILON)

    ref_pct  = np.clip(ref_pct,  _EPSILON, None)
    curr_pct = np.clip(curr_pct, _EPSILON, None)

    psi = float(np.sum((curr_pct - ref_pct) * np.log(curr_pct / ref_pct)))

    return DriftMetrics(kl_div=kl_div, psi=psi)
