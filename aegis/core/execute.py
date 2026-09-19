"""
aegis/core/execute.py — AegisML EXECUTE stage.

Dispatches a Decision to the appropriate actuator and creates an
ActionOutcome (status=PENDING until LEARN resolves it).

Pure dispatch — no file I/O except through actuators.
No torch, no pyRAPL, no mape imports.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from aegis.core.models import ActionOutcome, Decision, MetricSnapshot


def execute(
    decision: Decision,
    snapshot: MetricSnapshot,
) -> ActionOutcome:
    """
    Execute a Decision and return a pending ActionOutcome.

    Invariant: only OBSERVE returns a resolved=True outcome immediately
    (nothing was changed, nothing to verify).

    Args:
        decision: The Decision from PLAN.
        snapshot: The current MetricSnapshot (provides pre-state).

    Returns:
        ActionOutcome with status=PENDING (except OBSERVE → SUCCESS).
    """
    action = decision.chosen_action
    params = decision.params

    pre = {
        "accuracy.r2":      snapshot.accuracy.r2,
        "drift.kl_div":     snapshot.drift.kl_div,
        "energy.normalized": snapshot.energy.normalized,
        "equity.gap":        snapshot.equity.gap,
        "latency.p95_ms":    snapshot.latency.p95_ms,
    }

    result = _dispatch(action, params)

    status   = "SUCCESS" if result.get("ok", True) else "FAILED"
    resolved = action == "OBSERVE"

    return ActionOutcome(
        outcome_id    = f"out_{uuid.uuid4().hex[:8]}",
        decision_id   = decision.decision_id,
        action        = action,
        status        = "SUCCESS" if resolved else "PENDING_HUMAN" if not resolved and action == "EQUITY_REVIEW" else status,
        pre           = pre,
        post          = {},   # filled by LEARN after N snapshots
        resolved      = resolved,
        snapshots_to_resolve = 0 if resolved else 3,
        actual_energy_uj = 0.0,   # filled from mape_log.csv by LEARN
        notes         = result.get("notes", ""),
    )


def _dispatch(action: str, params: dict) -> dict:
    """Route action name to actuator call."""
    base = action.split(":")[0]

    if base == "OBSERVE":
        return {"ok": True, "notes": "no intervention"}

    if base == "SWITCH_MODEL":
        from aegis.actuators.model_switch import switch_model
        model = params.get("model") or (action.split(":")[1] if ":" in action else "lstm")
        return switch_model(model)

    if base == "REUSE_VERSION":
        from aegis.actuators.version_reuse import reuse_version
        model   = params.get("model", "")
        version = params.get("version", "")
        return reuse_version(model, version)

    if base == "RETRAIN_CURRENT":
        from aegis.actuators.retrain_act import retrain
        return retrain(params.get("model"))

    if base == "BATCH_INFERENCE":
        from aegis.actuators.serving_cfg import batch_inference
        return batch_inference(int(params.get("batch_size", 8)))

    if base == "LOWER_SAMPLING":
        from aegis.actuators.serving_cfg import lower_sampling
        return lower_sampling(float(params.get("rate", 0.5)))

    if base == "REDUCE_WINDOW":
        from aegis.actuators.serving_cfg import reduce_window
        return reduce_window(int(params.get("seq_length", 3)))

    if base == "EQUITY_REVIEW":
        from aegis.actuators.review import open_equity_review
        return open_equity_review(
            incident_id   = params.get("incident_id", "unknown"),
            worst_station = params.get("worst_station", "unknown"),
            gap           = float(params.get("gap", 0.0)),
        )

    print(f"[execute] unknown action: {action}")
    return {"ok": False, "notes": f"unknown action {action}"}
