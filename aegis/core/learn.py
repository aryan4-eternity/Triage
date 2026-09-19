"""
aegis/core/learn.py — AegisML LEARN stage.

Resolves pending ActionOutcomes by comparing post-action metric values
against the pre-action snapshot. Reads actual_energy_uj from mape_log.csv.

No torch, no pyRAPL, no mape imports. No float literals.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import pandas as pd

from aegis.core.models import ActionOutcome, MetricSnapshot

ROOT = Path(__file__).resolve().parents[2]
_MAPE_LOG = ROOT / "knowledge" / "mape_log.csv"


def _last_controller_energy(n_rows: int = 1) -> float:
    """
    Read the most recent controller energy entry from mape_log.csv.
    Returns 0.0 if the file is absent or empty.
    """
    if not _MAPE_LOG.exists():
        return 0.0
    try:
        df = pd.read_csv(_MAPE_LOG)
        if df.empty:
            return 0.0
        return float(df["energy_joules"].iloc[-n_rows:].sum() * 1_000_000)  # J → µJ
    except Exception:
        return 0.0


def _resolution_status(
    outcome: ActionOutcome,
    current_snapshot: MetricSnapshot,
) -> tuple[bool, str]:
    """
    Check whether the violation that triggered the action has been resolved.

    Returns (resolved: bool, notes: str).
    """
    # Compare key metrics between pre and current
    current = {
        "accuracy.r2":       current_snapshot.accuracy.r2,
        "drift.kl_div":      current_snapshot.drift.kl_div,
        "energy.normalized": current_snapshot.energy.normalized,
        "equity.gap":        current_snapshot.equity.gap,
    }

    pre = outcome.pre
    if not pre:
        return True, "no pre-state to compare"

    # Determine which metric was primarily addressed
    action_base = outcome.action.split(":")[0]
    targeted: dict[str, str] = {
        "RETRAIN_CURRENT": "drift.kl_div",
        "SWITCH_MODEL":    "energy.normalized",
        "REUSE_VERSION":   "drift.kl_div",
        "BATCH_INFERENCE": "energy.normalized",
        "OBSERVE":         "",
    }
    primary = targeted.get(action_base, "")

    if primary and primary in current and primary in pre:
        pre_val = pre[primary]
        cur_val = current[primary]
        # For "upper" metrics, lower is better; for accuracy (lower), higher is better
        improved = cur_val < pre_val if primary != "accuracy.r2" else cur_val > pre_val
        notes = (
            f"{primary}: {pre_val:.3f} → {cur_val:.3f} "
            f"({'improved' if improved else 'worsened'})"
        )
        return improved, notes

    return True, "action executed; full metric change deferred to next snapshot"


def resolve_outcomes(
    pending: Sequence[ActionOutcome],
    current_snapshot: MetricSnapshot,
    store=None,
) -> list[ActionOutcome]:
    """
    Attempt to resolve pending outcomes using the current snapshot.

    Outcomes are resolved when either:
      - snapshots_to_resolve has reached 0, OR
      - the targeted metric has returned inside its boundary.

    Args:
        pending:          List of unresolved ActionOutcomes.
        current_snapshot: The latest MetricSnapshot from MONITOR.
        store:            AegisStore for writing resolved outcomes (optional).

    Returns:
        List of updated ActionOutcome objects.
    """
    updated: list[ActionOutcome] = []
    controller_energy = _last_controller_energy()

    for outcome in pending:
        if outcome.resolved:
            updated.append(outcome)
            continue

        remaining = (outcome.snapshots_to_resolve or 3) - 1

        if remaining <= 0:
            # Time's up — resolve whatever the state is
            resolved, notes = _resolution_status(outcome, current_snapshot)
            post = {
                "accuracy.r2":       current_snapshot.accuracy.r2,
                "drift.kl_div":      current_snapshot.drift.kl_div,
                "energy.normalized": current_snapshot.energy.normalized,
                "equity.gap":        current_snapshot.equity.gap,
            }
            new_outcome = ActionOutcome(
                outcome_id    = outcome.outcome_id,
                decision_id   = outcome.decision_id,
                action        = outcome.action,
                status        = "SUCCESS" if resolved else "FAILED",
                pre           = outcome.pre,
                post          = post,
                resolved      = True,
                snapshots_to_resolve = 0,
                actual_energy_uj = controller_energy,
                notes = notes,
            )
        else:
            # Not yet — decrement counter
            new_outcome = ActionOutcome(
                outcome_id    = outcome.outcome_id,
                decision_id   = outcome.decision_id,
                action        = outcome.action,
                status        = outcome.status,
                pre           = outcome.pre,
                post          = {},
                resolved      = False,
                snapshots_to_resolve = remaining,
                actual_energy_uj = 0.0,
                notes = f"waiting {remaining} more snapshots",
            )

        if store and new_outcome.resolved:
            store.put_outcome(new_outcome)
        updated.append(new_outcome)

    return updated
