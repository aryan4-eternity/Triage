"""
aegis/core/utility.py — Multi-objective utility scorer for PLAN.

Pure functions. No torch, no pyRAPL, no mape imports. No float literals.

Utility formula (POLICY_ENGINE.md §3, ARCHITECTURE.md §5):
    U(a) = w_acc·Q(a) − w_e·E(a) − w_l·L(a) − w_c·C(a) − w_eq·F(a) − w_r·R(a)

where:
    Q(a) = min(1, Σ_f severity[f] · max(0, relief(a, f)))  — projected relief
    E(a) = action_energy_uj / max_action_energy              — action energy cost
    L(a) = latency_factor from policy.json                   — time-to-effect
    C(a) = cost_factor from policy.json                      — operational cost
    F(a) = equity penalty if action promotes model & equity_sev > 0
    R(a) = declared risk from policy.json

Sign convention for effects (POLICY_ENGINE.md §3):
    For upper-direction metrics (drift, energy, latency, cost, equity):
        effect > 0 means severity INCREASES — bad.
        effect < 0 means severity DECREASES — good. relief = -effect.
    For lower-direction metrics (accuracy):
        effect > 0 means value INCREASES — good. relief = +effect.
        effect < 0 means value DECREASES — bad.

This sign flip is handled ONCE here, in _relief(), with a comment.
"""
from __future__ import annotations

import numpy as np

from aegis.core.models import ActionCandidate, Incident

_UPPER_DIRECTION_FAMILIES = frozenset({"drift", "energy", "latency", "cost", "equity"})
_LOWER_DIRECTION_FAMILIES = frozenset({"accuracy"})


def _relief(family: str, effect: float) -> float:
    """
    Convert a declared effect to a severity-relief value in [0, ∞).

    For upper-direction families: negative effect = improvement = positive relief.
    For lower-direction families: positive effect = improvement = positive relief.
    Returns 0 if the effect worsens the family.
    """
    if family in _UPPER_DIRECTION_FAMILIES:
        # effect < 0 means "reduces severity" → relief = -effect
        return max(0.0, -effect)
    else:
        # lower-direction (accuracy): effect > 0 means improvement
        return max(0.0, effect)


def compute_utility(
    action_name: str,
    action_cfg: dict,
    incident: Incident,
    weights: dict[str, float],
    ctx: dict,
    max_energy_uj: float,
) -> tuple[float, dict[str, float]]:
    """
    Compute the utility score for one candidate action.

    Args:
        action_name:   e.g. "RETRAIN_CURRENT"
        action_cfg:    The action's dict from policy.json["actions"].
        incident:      Current incident (provides severity per family).
        weights:       Weight profile e.g. {"w_acc":1.0,"w_e":0.8,...}
        ctx:           PlanContext dict (for params like model name).
        max_energy_uj: Maximum energy_uj across all actions (for normalisation).

    Returns:
        (utility: float, terms: dict[str,float]) where terms = {Q,E,L,C,F,R}
    """
    severity = incident.severity  # {accuracy, drift, latency, energy, cost, equity}

    # ── Effect lookup ─────────────────────────────────────────────────────
    # Parameterised actions (SWITCH_MODEL) have per-model effects
    effect_raw: dict[str, float] = {}
    if "effect" in action_cfg:
        eff = action_cfg["effect"]
        if action_name == "SWITCH_MODEL":
            model_param = ctx.get("params", {}).get("model", "")
            effect_raw  = eff.get(model_param, {}) if isinstance(eff, dict) else {}
        elif isinstance(eff, dict) and not any(
            isinstance(v, dict) for v in eff.values()
        ):
            effect_raw = eff

    # ── Q — projected relief weighted by severity ─────────────────────────
    q_raw = 0.0
    for family, sev in severity.items():
        if sev <= 0.0:
            continue
        eff_value = effect_raw.get(family, 0.0)
        rel       = _relief(family, eff_value)
        q_raw    += sev * rel
    Q = float(min(1.0, q_raw))

    # ── E — action's own energy cost (normalised) ─────────────────────────
    action_uj = float(action_cfg.get("energy_uj", 0))
    E = float(np.clip(action_uj / max(max_energy_uj, 1.0), 0.0, 1.0))

    # ── L — time-to-effect ────────────────────────────────────────────────
    L = float(action_cfg.get("latency_factor", 0.0))

    # ── C — operational cost ─────────────────────────────────────────────
    C = float(action_cfg.get("cost_factor", 0.0))

    # ── F — equity penalty ───────────────────────────────────────────────
    # Non-zero only for promoting actions when equity_severity > 0
    eq_sev = severity.get("equity", 0.0)
    from aegis.core.guards import _PROMOTING_ACTIONS
    F = float(eq_sev) if (action_name in _PROMOTING_ACTIONS and eq_sev > 0.0) else 0.0

    # ── R — declared operational risk ────────────────────────────────────
    R = float(action_cfg.get("risk", 0.0))

    # ── U — final utility ────────────────────────────────────────────────
    w_acc = weights.get("w_acc", 1.0)
    w_e   = weights.get("w_e",   0.8)
    w_l   = weights.get("w_l",   0.4)
    w_c   = weights.get("w_c",   0.3)
    w_eq  = weights.get("w_eq",  1.4)
    w_r   = weights.get("w_r",   0.4)

    U = w_acc * Q - w_e * E - w_l * L - w_c * C - w_eq * F - w_r * R

    terms = {"Q": round(Q,4), "E": round(E,4), "L": round(L,4),
             "C": round(C,4), "F": round(F,4), "R": round(R,4)}
    return round(float(U), 6), terms
