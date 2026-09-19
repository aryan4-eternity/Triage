"""
aegis/core/models.py — All data contracts for AegisML.

Rules (from CLAUDE.md):
  - No torch, no pyRAPL, no mape imports here.
  - No float literals — all thresholds come from config/*.json.
  - Every public function documents the invariant it preserves.

Pydantic v2 models.  Every JSON example in docs/CONTRACTS.md must round-trip.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator

# ── SCHEMA — single source of truth for predictions.csv column names ──────────
# Import this everywhere instead of writing column names as string literals.
SCHEMA: dict[str, str] = {
    "true_value":      "true_value",
    "predicted_value": "predicted_value",
    "model_used":      "model_used",
    "inference_time":  "inference_time",
    "energy_uj":       "energy_uj",          # BUG-2 fix: one name, used everywhere
    "station_id":      "station_id",
    "energy_backend":  "energy_backend",
}

# Allowed energy backend labels
EnergyBackend = Literal["rapl", "codecarbon", "estimator"]

# Allowed model names
ModelName = Literal["lstm", "svm", "linear"]

# Incident types (ordered by classification priority — see ARCHITECTURE.md §4)
IncidentType = Literal[
    "NONE",
    "EQUITY_VIOLATION",
    "MODEL_DEGRADATION",
    "ACCURACY_DROP",
    "DRIFT_ONLY",
    "ENERGY_PRESSURE",
    "LATENCY_PRESSURE",
    "COMPOSITE",
]

# Action names
ActionName = Literal[
    "OBSERVE",
    "SWITCH_MODEL",
    "REUSE_VERSION",
    "REDUCE_WINDOW",
    "BATCH_INFERENCE",
    "LOWER_SAMPLING",
    "RETRAIN_CURRENT",
    "RETRAIN_ALL",
    "EQUITY_REVIEW",
]

OutcomeStatus = Literal["SUCCESS", "FAILED", "PENDING_HUMAN", "TIMED_OUT"]
BoundaryMode  = Literal["dynamic", "hard", "integral"]


# ── Snapshot sub-models ───────────────────────────────────────────────────────

class AccuracyMetrics(BaseModel):
    r2:        float
    mae:       float
    ema_score: float


class DriftMetrics(BaseModel):
    kl_div: float
    psi:    float = Field(default=0.0)


class LatencyMetrics(BaseModel):
    p50_ms: float
    p95_ms: float


class EnergyMetrics(BaseModel):
    uj_per_inference:  float
    normalized:        float
    current_threshold: float
    measured:          bool


class CostMetrics(BaseModel):
    inr_per_1k:  float
    wall_clock_s: float


class EquityMetrics(BaseModel):
    worst_station:  str
    worst_r2:       float
    gap:            float
    per_station_r2: Dict[str, float] = Field(default_factory=dict)


class WindowInfo(BaseModel):
    rows:           int
    energy_backend: str
    stations:       int


class ServingConfig(BaseModel):
    batch_size:    int   = Field(default=1)
    seq_length:    int   = Field(default=5)
    sampling_rate: float = Field(default=1.0)


# ── MetricSnapshot ────────────────────────────────────────────────────────────

class MetricSnapshot(BaseModel):
    """
    Produced by MONITOR. Immutable once created.

    Invariant: window.energy_backend is a single value (MONITOR refuses
    to produce a snapshot from a mixed-backend window).
    """
    snapshot_id:  str
    ts:           datetime
    active_model: str
    window:       WindowInfo

    accuracy: AccuracyMetrics
    drift:    DriftMetrics
    latency:  LatencyMetrics
    energy:   EnergyMetrics
    cost:     CostMetrics
    equity:   EquityMetrics
    serving:  ServingConfig

    # Simulator metadata — the controller MUST NEVER read this.
    # tests/test_invariants.py enforces this by grepping aegis/core/.
    scenario_tag: Optional[str] = None


# ── BoundaryReport ────────────────────────────────────────────────────────────

class BoundaryReport(BaseModel):
    """
    Result of a single metric boundary check.

    Invariant: if mode == "integral", lower is None and upper tracks the
    integral-controller threshold (ARCHITECTURE.md §4).
    """
    metric:      str
    value:       float
    lower:       Optional[float]
    upper:       Optional[float]
    mode:        BoundaryMode
    baseline_mean: Optional[float] = None
    baseline_std:  Optional[float] = None
    k:             Optional[float] = None
    hard_ceiling:  Optional[float] = None
    hard_floor:    Optional[float] = None
    violated:      bool
    severity:      float   # clamp(|value - bound| / |hard_bound - bound|, 0, 1)
    consecutive_violations: int = Field(default=0)


# ── Incident ──────────────────────────────────────────────────────────────────

class Incident(BaseModel):
    """
    Classification output from ANALYZE.

    Invariant: severity values are in [0, 1].  The 'violated' list names
    metric keys that crossed their boundary.
    """
    incident_id:  str
    snapshot_id:  str
    type:         IncidentType
    violated:     List[str]
    severity:     Dict[str, float]   # {accuracy, drift, latency, energy, cost, equity}
    boundaries:   List[BoundaryReport]
    classification_reason: str

    @model_validator(mode="after")
    def _severity_range(self) -> "Incident":
        for k, v in self.severity.items():
            assert 0.0 <= v <= 1.0, f"severity[{k}]={v} out of [0,1]"
        return self


# ── ActionCandidate ───────────────────────────────────────────────────────────

class ActionCandidate(BaseModel):
    """
    One candidate action evaluated during PLAN.

    Both eligible and ineligible candidates are persisted — the rejected list
    is the demo figure and the paper differentiator.

    Invariant: if eligible is False, utility must be None and
    rejected_by_guard must be set.
    """
    action:           str
    params:           Dict[str, Any] = Field(default_factory=dict)
    eligible:         bool
    rejected_by_guard: Optional[str] = None
    terms:            Optional[Dict[str, float]] = None  # Q, E, L, C, F, R
    weights:          Optional[Dict[str, float]] = None
    utility:          Optional[float] = None
    projected_effect: Optional[Dict[str, float]] = None
    reason:           Optional[str]  = None

    @model_validator(mode="after")
    def _ineligible_has_no_utility(self) -> "ActionCandidate":
        if not self.eligible:
            assert self.utility is None, (
                "Ineligible candidate must have utility=None"
            )
            assert self.rejected_by_guard is not None, (
                "Ineligible candidate must have rejected_by_guard set"
            )
        return self


# ── Decision ──────────────────────────────────────────────────────────────────

class Decision(BaseModel):
    """
    Output of PLAN.

    Invariant: chosen_action must appear in candidates with eligible=True
    (unless exploratory, where it's still eligible — exploration is post-guard).
    """
    decision_id:    str
    incident_id:    str
    chosen_action:  str
    params:         Dict[str, Any] = Field(default_factory=dict)
    candidates:     List[ActionCandidate]
    policy_profile: str
    policy_version: str
    exploratory:    bool = False
    reason:         str
    recheck_after_snapshots: int = Field(default=3)


# ── ActionOutcome ─────────────────────────────────────────────────────────────

class ActionOutcome(BaseModel):
    """
    Resolved outcome written by LEARN.

    Invariant: actual_energy_uj is the MEASURED cost from mape_log.csv —
    never estimated here (that would defeat the purpose of the ledger).
    """
    outcome_id:    str
    decision_id:   str
    action:        str
    status:        OutcomeStatus
    pre:           Dict[str, float]
    post:          Dict[str, float]
    resolved:      bool
    snapshots_to_resolve: Optional[int] = None
    actual_energy_uj:     float = Field(default=0.0)
    notes:                str   = Field(default="")
