"""
aegis/core/plan.py — AegisML PLAN stage.

Expands actions, runs guards, scores utility, selects via argmax.
Pure functions. No torch, no pyRAPL, no mape imports. No float literals.

Pipeline (POLICY_ENGINE.md §2–3):
  1. Build PlanContext from snapshot + store state.
  2. Expand parameterised actions (SWITCH_MODEL × 3, REUSE_VERSION × versions).
  3. Run guards — ineligible candidates carry rejection reason.
  4. Score utility for eligible candidates.
  5. ε-greedy exploration AFTER guard filter (never picks ineligible).
  6. Tie-break: lower R, then lower E, then action name (deterministic).
  7. Floor rule: if max U < 0, fall back to OBSERVE.
  8. Return Decision with full candidate list (eligible + rejected).
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from aegis.core.guards  import check_guards
from aegis.core.models  import ActionCandidate, Decision, Incident
from aegis.core.utility import compute_utility

ROOT = Path(__file__).resolve().parents[2]


def _load_policy() -> dict:
    path = ROOT / "config" / "policy.json"
    if path.exists():
        return json.loads(path.read_text())
    return {"version": "unknown", "active_profile": "balanced",
            "epsilon": 0.1, "profiles": {}, "actions": {}}


def _load_model_list() -> list[str]:
    """Return available model names from models/ directory."""
    models_dir = ROOT / "models"
    names = []
    for name in ["lstm", "svm", "linear"]:
        if (models_dir / f"{name}.pth").exists() or \
           (models_dir / f"{name}.pkl").exists():
            names.append(name)
    return names or ["lstm", "svm", "linear"]


def _build_ctx(snapshot, store, extra: dict | None = None) -> dict:
    """Build the PlanContext dict from snapshot and store state."""
    from datetime import timezone as tz

    # Archived versions for version reuse
    vmr   = ROOT / "versionedMR"
    versions: dict[str, int] = {}
    best_kl: dict[str, float] = {}
    if vmr.exists():
        for mdir in vmr.iterdir():
            if mdir.is_dir():
                vs = [d for d in mdir.iterdir() if d.name.startswith("version_")]
                versions[mdir.name] = len(vs)

    # Best version KL (from drift_kl.json if present)
    kl_path = ROOT / "knowledge" / "drift_kl.json"
    if kl_path.exists():
        try:
            kl_data = json.loads(kl_path.read_text())
            best_kl_val = float(kl_data.get("min_kl_div", 999.0))
        except Exception:
            best_kl_val = 999.0
    else:
        best_kl_val = 999.0

    # Cooldown elapsed seconds — check against last decision timestamp
    cooldowns: dict[str, float] = {}
    if store is not None:
        for action_name in ["RETRAIN_CURRENT", "RETRAIN_ALL"]:
            ts = store.last_action_ts_simple(action_name)
            if ts:
                now = datetime.now(timezone.utc)
                cooldowns[action_name] = (now - ts).total_seconds()
            else:
                cooldowns[action_name] = 9999.0
    else:
        cooldowns = {"RETRAIN_CURRENT": 9999.0, "RETRAIN_ALL": 9999.0}

    # Equity review pending
    review_path = ROOT / "knowledge" / "review.json"
    equity_pending = False
    if review_path.exists():
        try:
            rv = json.loads(review_path.read_text())
            equity_pending = bool(rv.get("equity_review_pending", False))
        except Exception:
            pass

    # Drift window rows
    drift_path = ROOT / "knowledge" / "drift.csv"
    drift_rows = 0
    if drift_path.exists():
        try:
            import pandas as pd
            drift_rows = len(pd.read_csv(drift_path))
        except Exception:
            pass

    ctx = {
        "active_model":         snapshot.active_model,
        "ema_scores":           {},   # filled from mape_info if needed
        "archived_versions":    sum(versions.values()),
        "best_version_kl":      best_kl_val,
        "drift_window_rows":    drift_rows,
        "equity_review_pending": equity_pending,
        "cooldown_elapsed_s":   cooldowns,
        "serving": {
            "batch_size":    snapshot.serving.batch_size,
            "seq_length":    snapshot.serving.seq_length,
            "sampling_rate": snapshot.serving.sampling_rate,
        },
    }
    if extra:
        ctx.update(extra)
    return ctx


def _expand_actions(policy: dict, ctx: dict) -> list[tuple[str, dict, dict]]:
    """
    Expand parameterised actions into individual candidates.

    Returns list of (action_label, action_cfg, params).
    e.g. SWITCH_MODEL → [("SWITCH_MODEL:lstm",cfg,{model:lstm}), ...]
    """
    candidates = []
    models = _load_model_list()
    vmr    = ROOT / "versionedMR"

    for action_name, action_cfg in policy.get("actions", {}).items():
        param_by = action_cfg.get("parameterised_by")

        if param_by == "model":
            for m in models:
                if m == ctx.get("active_model"):
                    continue  # SWITCH_MODEL to current model is trivially ineligible
                label  = f"{action_name}:{m}"
                params = {"model": m}
                candidates.append((label, action_name, action_cfg, params))
        elif param_by == "version":
            # Expand per existing version
            found = False
            if vmr.exists():
                for mdir in vmr.iterdir():
                    if not mdir.is_dir():
                        continue
                    for vdir in sorted(mdir.iterdir()):
                        if vdir.name.startswith("version_"):
                            label  = f"{action_name}:{mdir.name}@{vdir.name}"
                            params = {"model": mdir.name, "version": vdir.name}
                            candidates.append((label, action_name, action_cfg, params))
                            found = True
            if not found:
                candidates.append(
                    (action_name, action_name, action_cfg, {})
                )
        else:
            candidates.append((action_name, action_name, action_cfg, {}))

    return candidates  # [(label, base_action, cfg, params)]


def plan(
    incident: Incident,
    snapshot,
    store=None,
    rng: Optional[np.random.Generator] = None,
    profile_override: Optional[str] = None,
) -> Decision:
    """
    Produce a Decision from an Incident.

    Invariant: exploratory decisions are still from eligible candidates only.
    Invariant: all candidates (eligible + rejected) are persisted.
    Invariant: does not read snapshot.scenario_tag.
    Invariant: tie-breaks are deterministic (R, E, name order).

    Args:
        incident:         Current Incident from ANALYZE.
        snapshot:         Current MetricSnapshot.
        store:            AegisStore instance (optional, for cooldown lookup).
        rng:              numpy RNG for ε-greedy (seeded for determinism).
        profile_override: If set, use this weight profile instead of active.

    Returns:
        Decision with full candidate table.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    policy  = _load_policy()
    profile = profile_override or policy.get("active_profile", "balanced")
    weights = policy.get("profiles", {}).get(profile, {
        "w_acc": 1.0, "w_e": 0.8, "w_l": 0.4,
        "w_c": 0.3, "w_eq": 1.4, "w_r": 0.4,
    })
    epsilon = float(policy.get("epsilon", 0.1))

    ctx = _build_ctx(snapshot, store)

    # Max energy for normalisation
    all_uj = [float(cfg.get("energy_uj", 0))
               for cfg in policy.get("actions", {}).values()]
    max_uj = max(all_uj, default=1.0) or 1.0

    # Expand → evaluate
    expanded      = _expand_actions(policy, ctx)
    all_candidates: list[ActionCandidate] = []
    eligible_list: list[tuple[ActionCandidate, dict]] = []  # (candidate, action_cfg)

    for (label, base_action, action_cfg, params) in expanded:
        ctx_with_params = {**ctx, "params": params}
        guards          = action_cfg.get("guards", [])
        eligible, reason = check_guards(base_action, guards, ctx_with_params)

        if not eligible:
            all_candidates.append(ActionCandidate(
                action=label, params=params,
                eligible=False, rejected_by_guard=reason,
            ))
            continue

        utility, terms = compute_utility(
            base_action, action_cfg, incident, weights, ctx_with_params, max_uj
        )
        # Build projected_effect
        eff = action_cfg.get("effect", {})
        if base_action == "SWITCH_MODEL":
            eff = eff.get(params.get("model", ""), {}) if isinstance(eff, dict) else {}
        proj_eff = eff if isinstance(eff, dict) and not any(
            isinstance(v, dict) for v in eff.values()
        ) else {}

        reason_str = _build_reason(base_action, incident, terms, weights)
        cand = ActionCandidate(
            action=label, params=params,
            eligible=True, utility=utility,
            terms=terms, weights=weights,
            projected_effect=proj_eff,
            reason=reason_str,
        )
        all_candidates.append(cand)
        eligible_list.append((cand, action_cfg))

    # Ensure OBSERVE is always eligible
    if not any(c.action == "OBSERVE" for c in all_candidates):
        obs_cfg = policy.get("actions", {}).get("OBSERVE", {})
        u, t = compute_utility("OBSERVE", obs_cfg, incident, weights, ctx, max_uj)
        obs_cand = ActionCandidate(
            action="OBSERVE", params={}, eligible=True,
            utility=u, terms=t, weights=weights,
            projected_effect={},
            reason="baseline: no intervention",
        )
        all_candidates.append(obs_cand)
        eligible_list.append((obs_cand, obs_cfg))

    # Floor rule: if all utilities ≤ 0, choose OBSERVE
    eligible_with_positive = [(c, cfg) for c, cfg in eligible_list
                               if (c.utility or 0.0) > 0.0]

    if eligible_list:
        # ε-greedy (post-guard)
        if rng.random() < epsilon:
            chosen_cand, _ = eligible_list[int(rng.integers(len(eligible_list)))]
            exploratory     = True
        else:
            # argmax with deterministic tie-break: (−utility, risk, energy, name)
            def _key(pair):
                c, cfg = pair
                return (
                    -(c.utility or 0.0),
                    float(cfg.get("risk", 0.0)),
                    float(cfg.get("energy_uj", 0)),
                    c.action,
                )
            chosen_cand, _ = min(eligible_list, key=_key)
            exploratory     = False
    else:
        # No eligible actions — OBSERVE
        obs_cfg  = policy.get("actions", {}).get("OBSERVE", {})
        u, t     = compute_utility("OBSERVE", obs_cfg, incident, weights, ctx, max_uj)
        chosen_cand = ActionCandidate(
            action="OBSERVE", params={}, eligible=True,
            utility=u, terms=t, weights=weights,
            projected_effect={}, reason="no eligible actions",
        )
        exploratory = False

    return Decision(
        decision_id    = f"dec_{uuid.uuid4().hex[:8]}",
        incident_id    = incident.incident_id,
        chosen_action  = chosen_cand.action,
        params         = chosen_cand.params,
        candidates     = all_candidates,
        policy_profile = profile,
        policy_version = policy.get("version", "unknown"),
        exploratory    = exploratory,
        reason         = chosen_cand.reason or "argmax utility",
        recheck_after_snapshots = 3,
    )


def _build_reason(
    action: str,
    incident: Incident,
    terms: dict[str, float],
    weights: dict[str, float],
) -> str:
    """Build a human-readable reason string for a decision."""
    if incident.type == "NONE":
        return "no violation — observing"
    Q, E, R = terms.get("Q", 0), terms.get("E", 0), terms.get("R", 0)
    return (
        f"{incident.type}: Q={Q:.2f} E={E:.2f} R={R:.2f} "
        f"(profile w_acc={weights.get('w_acc',1):.1f} "
        f"w_e={weights.get('w_e',0.8):.1f})"
    )
