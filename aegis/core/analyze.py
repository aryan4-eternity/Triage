"""
aegis/core/analyze.py — AegisML ANALYZE stage.

Classifies metric violations into incidents using the 7-rule ordered
classification from ARCHITECTURE.md §4 and POLICY_ENGINE.md.

Pure functions. No torch, no pyRAPL, no mape imports. No float literals.

Classification rules (first match wins):
  1. equity violated                         → EQUITY_VIOLATION
  2. accuracy violated AND drift violated    → MODEL_DEGRADATION
  3. accuracy violated, drift fine           → ACCURACY_DROP
  4. drift violated, accuracy fine           → DRIFT_ONLY
  5. energy violated                         → ENERGY_PRESSURE
  6. latency violated                        → LATENCY_PRESSURE
  7. ≥ 3 families violated                   → COMPOSITE
  (none)                                     → NONE
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Sequence

from aegis.core.boundaries import compute_boundary
from aegis.core.models import (
    BoundaryReport, Incident, IncidentType, MetricSnapshot,
)

ROOT = Path(__file__).resolve().parents[2]


def _load_persistence() -> int:
    path = ROOT / "config" / "boundaries.json"
    if path.exists():
        cfg = json.loads(path.read_text())
        return int(cfg.get("persistence", 2))
    return 2


# Mapping from metric key to (snapshot_attribute_chain, history_key)
_METRIC_EXTRACTORS: dict[str, callable] = {
    "accuracy.r2":       lambda s: s.accuracy.r2,
    "drift.kl_div":      lambda s: s.drift.kl_div,
    "latency.p95_ms":    lambda s: s.latency.p95_ms,
    "energy.normalized": lambda s: s.energy.normalized,
    "equity.gap":        lambda s: s.equity.gap,
    "equity.worst_r2":   lambda s: s.equity.worst_r2,
}


def analyze(
    snapshot: MetricSnapshot,
    history: Sequence[MetricSnapshot],
    violation_counts: dict[str, int] | None = None,
) -> Incident:
    """
    Analyse a snapshot against historical boundaries and produce an Incident.

    Invariant: severity values in [0, 1].
    Invariant: does not read snapshot.scenario_tag.

    Args:
        snapshot:         Current MetricSnapshot.
        history:          Previous snapshots (oldest first).
        violation_counts: Per-metric consecutive violation counters
                          (mutated in-place on violation).

    Returns:
        Incident describing what (if anything) violated and why.
    """
    if violation_counts is None:
        violation_counts = {}

    persistence = _load_persistence()

    # Extract per-metric history arrays
    boundary_reports: list[BoundaryReport] = []
    raw_violated: dict[str, bool] = {}       # violated before persistence filter
    severity: dict[str, float] = {
        "accuracy": 0.0, "drift": 0.0, "latency": 0.0,
        "energy": 0.0, "cost": 0.0, "equity": 0.0,
    }

    for metric_key, extractor in _METRIC_EXTRACTORS.items():
        hist_vals = [extractor(s) for s in history]
        current   = extractor(snapshot)
        consec    = violation_counts.get(metric_key, 0)

        report = compute_boundary(metric_key, hist_vals, current, consec)

        # Update consecutive counter
        if report.violated:
            violation_counts[metric_key] = consec + 1
        else:
            violation_counts[metric_key] = 0

        # Apply persistence filter: incident only fires after N consecutive violations
        persistent = violation_counts[metric_key] >= persistence

        raw_violated[metric_key] = report.violated and persistent

        # Aggregate severity by family
        family = metric_key.split(".")[0]
        if family in severity and persistent:
            severity[family] = max(severity[family], report.severity)

        boundary_reports.append(report)

    violated_metrics = [k for k, v in raw_violated.items() if v]

    # ── 7-rule classification ─────────────────────────────────────────────
    incident_type, reason = _classify(violated_metrics, severity)

    return Incident(
        incident_id  = f"inc_{uuid.uuid4().hex[:8]}",
        snapshot_id  = snapshot.snapshot_id,
        type         = incident_type,
        violated     = violated_metrics,
        severity     = severity,
        boundaries   = boundary_reports,
        classification_reason = reason,
    )


def _classify(
    violated: list[str],
    severity: dict[str, float],
) -> tuple[IncidentType, str]:
    """Apply the 7 ordered classification rules. First match wins."""

    acc_v     = any("accuracy" in m for m in violated)
    drift_v   = any("drift" in m    for m in violated)
    energy_v  = any("energy" in m   for m in violated)
    latency_v = any("latency" in m  for m in violated)
    equity_v  = any("equity" in m   for m in violated)

    n_families = sum([acc_v, drift_v, energy_v, latency_v, equity_v])

    # Rule 1 — equity dominates (safety constraint)
    if equity_v:
        return ("EQUITY_VIOLATION",
                "equity boundary exceeded — safety constraint; "
                f"gap severity={severity['equity']:.2f}")

    # Rule 2 — accuracy + drift together
    if acc_v and drift_v:
        return ("MODEL_DEGRADATION",
                f"accuracy (sev={severity['accuracy']:.2f}) and drift "
                f"(sev={severity['drift']:.2f}) both violated — "
                "model no longer fits the current distribution")

    # Rule 3 — accuracy alone
    if acc_v:
        return ("ACCURACY_DROP",
                f"accuracy below boundary (sev={severity['accuracy']:.2f}), "
                "drift inside boundary — suspect pipeline or label issue, "
                "not a data distribution shift")

    # Rule 4 — drift alone
    if drift_v:
        return ("DRIFT_ONLY",
                f"drift boundary exceeded (sev={severity['drift']:.2f}), "
                "accuracy still inside boundary — "
                "retraining not yet justified by accuracy loss")

    # Rule 5 — energy
    if energy_v:
        return ("ENERGY_PRESSURE",
                f"energy normalised above threshold (sev={severity['energy']:.2f})")

    # Rule 6 — latency
    if latency_v:
        return ("LATENCY_PRESSURE",
                f"p95 latency above boundary (sev={severity['latency']:.2f})")

    # Rule 7 — composite
    if n_families >= 3:
        names = [k for k, v in severity.items() if v > 0.0]
        return ("COMPOSITE",
                f"≥3 families violated: {names}")

    # No violation
    return ("NONE", "all metrics inside boundaries")
