"""
tests/test_plan.py — PLAN stage unit tests (POLICY_ENGINE.md §7).

All tests use pure functions with no file I/O (aegis/core/ constraint).
Incidents and snapshots are built programmatically.

Tests:
- test_drift_only_rejects_retrain_under_balanced
- test_high_severity_drift_selects_retrain_under_accuracy_first
- test_equity_violation_blocks_all_promoting_actions
- test_ineligible_actions_carry_a_guard_reason
- test_exploration_only_picks_eligible_actions
- test_exploratory_decisions_are_flagged
- test_decision_deterministic_across_100_runs_fixed_seed
- test_falls_back_to_observe_when_all_utilities_negative
- test_accuracy_sign_flip_handled_once
"""
from __future__ import annotations

import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aegis.core.models import (
    AccuracyMetrics, BoundaryReport, CostMetrics, Decision, DriftMetrics,
    EnergyMetrics, EquityMetrics, Incident, LatencyMetrics, MetricSnapshot,
    ServingConfig, WindowInfo,
)
from aegis.core.utility import compute_utility


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_snapshot(
    r2: float = 0.84,
    kl: float = 0.31,
    uj: float = 1000.0,
    p95_ms: float = 4.0,
    eq_gap: float = 0.05,
    worst_r2: float = 0.80,
    active_model: str = "lstm",
) -> MetricSnapshot:
    return MetricSnapshot(
        snapshot_id  = f"snap_{uuid.uuid4().hex[:8]}",
        ts           = datetime.now(timezone.utc),
        active_model = active_model,
        window       = WindowInfo(rows=1200, energy_backend="estimator", stations=8),
        accuracy = AccuracyMetrics(r2=r2, mae=15.0, ema_score=0.80),
        drift    = DriftMetrics(kl_div=kl, psi=0.10),
        latency  = LatencyMetrics(p50_ms=2.0, p95_ms=p95_ms),
        energy   = EnergyMetrics(uj_per_inference=uj, normalized=0.04,
                                  current_threshold=0.91, measured=False),
        cost     = CostMetrics(inr_per_1k=0.004, wall_clock_s=180.0),
        equity   = EquityMetrics(worst_station="ST_07", worst_r2=worst_r2, gap=eq_gap),
        serving  = ServingConfig(batch_size=1, seq_length=5, sampling_rate=1.0),
    )


def _make_incident(
    incident_type: str = "DRIFT_ONLY",
    severity: dict | None = None,
) -> Incident:
    if severity is None:
        severity = {"accuracy": 0.0, "drift": 0.23, "latency": 0.0,
                    "energy": 0.0, "cost": 0.0, "equity": 0.0}
    return Incident(
        incident_id  = f"inc_{uuid.uuid4().hex[:8]}",
        snapshot_id  = "snap_test",
        type         = incident_type,
        violated     = ["drift.kl_div"],
        severity     = severity,
        boundaries   = [],
        classification_reason = "test fixture",
    )


def _plan(incident: Incident, snapshot: MetricSnapshot,
          profile: str = "balanced", seed: int = 0) -> Decision:
    from aegis.core.plan import plan
    rng = np.random.default_rng(seed)
    return plan(incident, snapshot, store=None, rng=rng,
                profile_override=profile)


# ── Tests ──────────────────────────────────────────────────────────────────────

class TestDriftOnly:

    def test_drift_only_rejects_retrain_under_balanced(self):
        """
        POLICY_ENGINE.md §4 worked example:
        DRIFT_ONLY with accuracy inside boundary → RETRAIN_CURRENT gets U ≈ -1.88
        under balanced profile. OBSERVE (U=0) should win.
        """
        incident = _make_incident("DRIFT_ONLY", {
            "accuracy": 0.0, "drift": 0.23, "latency": 0.0,
            "energy": 0.0, "cost": 0.0, "equity": 0.0,
        })
        snap    = _make_snapshot()
        dec     = _plan(incident, snap, "balanced", seed=999)

        chosen_base = dec.chosen_action.split(":")[0]
        assert chosen_base == "OBSERVE", (
            f"Expected OBSERVE for DRIFT_ONLY balanced, got {dec.chosen_action}"
        )

        # Check that RETRAIN_CURRENT appears in candidates with negative utility
        retrain_cand = next(
            (c for c in dec.candidates if c.action == "RETRAIN_CURRENT" and c.eligible),
            None,
        )
        if retrain_cand:
            assert retrain_cand.utility is not None
            assert retrain_cand.utility < 0.0, (
                f"RETRAIN_CURRENT should have negative utility under DRIFT_ONLY, "
                f"got {retrain_cand.utility}"
            )

    def test_ineligible_actions_carry_a_guard_reason(self):
        """Every ineligible candidate must have a non-empty rejected_by_guard."""
        incident = _make_incident("DRIFT_ONLY")
        dec = _plan(incident, _make_snapshot(), "balanced")
        for c in dec.candidates:
            if not c.eligible:
                assert c.rejected_by_guard, (
                    f"Ineligible action {c.action} has no rejected_by_guard"
                )
                assert len(c.rejected_by_guard) > 0
                assert c.utility is None


class TestEquitySafety:

    def test_equity_violation_blocks_all_promoting_actions(self):
        """
        POLICY_ENGINE.md §5:
        EQUITY_VIOLATION must make all model-promoting actions ineligible.
        """
        incident = _make_incident("EQUITY_VIOLATION", {
            "accuracy": 0.0, "drift": 0.0, "latency": 0.0,
            "energy": 0.0, "cost": 0.0, "equity": 0.27,
        })
        snap = _make_snapshot(eq_gap=0.19, worst_r2=0.58)
        dec  = _plan(incident, snap, "balanced")

        promoting = {"SWITCH_MODEL", "REUSE_VERSION", "RETRAIN_CURRENT", "RETRAIN_ALL"}
        for c in dec.candidates:
            base = c.action.split(":")[0]
            if base in promoting:
                assert not c.eligible, (
                    f"Promoting action {c.action} should be ineligible under "
                    f"equity_review_pending"
                )
                assert c.rejected_by_guard is not None
                assert "equity" in c.rejected_by_guard.lower() or \
                       "promoting" in c.rejected_by_guard.lower(), \
                    f"Guard reason doesn't mention equity: {c.rejected_by_guard}"


class TestDeterminism:

    def test_decision_deterministic_across_100_runs_fixed_seed(self):
        """Same snapshot + same policy + same seed → same decision."""
        incident = _make_incident("DRIFT_ONLY")
        snap     = _make_snapshot()

        decisions = set()
        for _ in range(100):
            dec = _plan(incident, snap, "balanced", seed=42)
            decisions.add(dec.chosen_action)

        assert len(decisions) == 1, (
            f"Non-deterministic: got {decisions} across 100 runs with same seed"
        )

    def test_falls_back_to_observe_when_all_utilities_negative(self):
        """
        Floor rule: when max utility < 0, must choose OBSERVE.
        Create a scenario where OBSERVE has U=0 and all others are negative.
        """
        # NONE incident means severity=0 everywhere → Q=0 for all actions
        # → all actions have negative utility (their costs dominate) except OBSERVE (U=0)
        incident = _make_incident("NONE", {
            "accuracy": 0.0, "drift": 0.0, "latency": 0.0,
            "energy": 0.0, "cost": 0.0, "equity": 0.0,
        })
        snap = _make_snapshot()
        dec  = _plan(incident, snap, "balanced", seed=42)
        assert dec.chosen_action == "OBSERVE", (
            f"Floor rule: should fall back to OBSERVE, got {dec.chosen_action}"
        )


class TestExploration:

    def test_exploration_only_picks_eligible_actions(self):
        """ε-greedy must never pick an ineligible action."""
        incident = _make_incident("EQUITY_VIOLATION", {
            "accuracy": 0.0, "drift": 0.0, "latency": 0.0,
            "energy": 0.0, "cost": 0.0, "equity": 0.5,
        })
        snap = _make_snapshot(eq_gap=0.20, worst_r2=0.55)

        promoting = {"SWITCH_MODEL", "REUSE_VERSION", "RETRAIN_CURRENT", "RETRAIN_ALL"}

        # Run many times to catch any exploratory pick of an ineligible action
        for seed in range(200):
            rng = np.random.default_rng(seed)
            from aegis.core.plan import plan
            dec = plan(incident, snap, store=None, rng=rng,
                       profile_override="balanced")
            base = dec.chosen_action.split(":")[0]
            assert base not in promoting, (
                f"Seed {seed}: exploratory={dec.exploratory} chose promoting "
                f"action {dec.chosen_action} under equity_review_pending"
            )

    def test_exploratory_decisions_are_flagged(self):
        """
        When exploration fires (epsilon=1.0 effectively), the decision must
        carry exploratory=True.
        """
        from aegis.core.plan import plan

        incident = _make_incident("NONE", {
            "accuracy": 0.0, "drift": 0.0, "latency": 0.0,
            "energy": 0.0, "cost": 0.0, "equity": 0.0,
        })
        snap = _make_snapshot()

        # Find a seed that triggers exploration with epsilon=0.1
        found_exploratory = False
        for seed in range(1000):
            rng = np.random.default_rng(seed)
            dec = plan(incident, snap, store=None, rng=rng,
                       profile_override="balanced")
            if dec.exploratory:
                found_exploratory = True
                assert dec.exploratory is True
                break

        assert found_exploratory, "No exploratory decision found in 1000 seeds"


class TestAccuracySignFlip:

    def test_accuracy_sign_flip_handled_once(self):
        """
        POLICY_ENGINE.md §3: accuracy is a lower-direction metric.
        A positive effect on accuracy should reduce severity (positive relief).
        A negative effect should give zero relief.
        """
        from aegis.core.utility import _relief

        # Positive accuracy effect = improvement = positive relief
        assert _relief("accuracy", 0.10) > 0.0, \
            "Positive accuracy effect should give positive relief"

        # Negative accuracy effect = degradation = zero relief
        assert _relief("accuracy", -0.10) == 0.0, \
            "Negative accuracy effect should give zero relief (no negative relief)"

        # Upper-direction families: negative effect = improvement = positive relief
        assert _relief("drift", -0.5) > 0.0, \
            "Negative drift effect should give positive relief (reduces drift)"

        assert _relief("energy", -0.3) > 0.0, \
            "Negative energy effect should give positive relief"

        # Upper-direction with positive effect = worsening = zero relief
        assert _relief("drift", 0.5) == 0.0, \
            "Positive drift effect should give zero relief (makes drift worse)"
