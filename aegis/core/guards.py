"""
aegis/core/guards.py — Declarative eligibility guards for PLAN.

Pure functions. No torch, no pyRAPL, no mape imports. No float literals.

Guards are declarative conditions in config/policy.json. Each guard is:
    { "field": "<ctx_path>", "op": "<operator>", "value": <literal_or_$ref> }

Supported operators: eq, ne, gt, gte, lt, lte.

Context fields resolved from a PlanContext dict:
    active_model, archived_versions, best_version_kl, drift_window_rows,
    equity_review_pending, serving.batch_size, serving.seq_length,
    serving.sampling_rate, ema_scores.<model>, cooldown_elapsed_s.<ACTION>,
    params.model  (resolved at candidate expansion time)

Global guard (CONTRACTS.md §Invariant):
    When equity_review_pending is True, every model-promoting action is
    ineligible regardless of any other guard.

Promoting actions: SWITCH_MODEL, REUSE_VERSION, RETRAIN_CURRENT (any retrain).
"""
from __future__ import annotations

import operator
from typing import Any

_OPS: dict[str, callable] = {
    "eq":  operator.eq,
    "ne":  operator.ne,
    "gt":  operator.gt,
    "gte": operator.ge,
    "lt":  operator.lt,
    "lte": operator.le,
}

_PROMOTING_ACTIONS = frozenset({
    "SWITCH_MODEL", "REUSE_VERSION", "RETRAIN_CURRENT", "RETRAIN_ALL",
})


def _resolve(ctx: dict, field: str) -> Any:
    """Resolve a dot-path field from the context dict."""
    parts = field.split(".")
    val   = ctx
    for p in parts:
        if isinstance(val, dict):
            val = val.get(p)
        else:
            return None
    return val


def _resolve_value(ctx: dict, value: Any) -> Any:
    """Resolve $-prefixed references against the context."""
    if isinstance(value, str) and value.startswith("$"):
        return _resolve(ctx, value[1:])
    return value


def check_guards(
    action_name: str,
    guards: list[dict],
    ctx: dict,
) -> tuple[bool, str | None]:
    """
    Check all guards for an action.

    Invariant: when equity_review_pending is True and the action is
    model-promoting, returns (False, reason) regardless of other guards.
    This is a hard filter, not a utility penalty.

    Args:
        action_name: e.g. "RETRAIN_CURRENT"
        guards:      list of guard dicts from policy.json
        ctx:         PlanContext dict

    Returns:
        (eligible: bool, rejection_reason: str | None)
        rejection_reason is None when eligible is True.
    """
    # Global safety guard
    if ctx.get("equity_review_pending") and action_name in _PROMOTING_ACTIONS:
        return False, (
            "equity_review_pending=True — all model-promoting actions are "
            "ineligible regardless of utility (safety constraint)"
        )

    for g in guards:
        field    = g["field"]
        op_name  = g["op"]
        expected = _resolve_value(ctx, g["value"])
        actual   = _resolve(ctx, field)

        if op_name not in _OPS:
            return False, f"unknown guard operator '{op_name}'"

        try:
            if not _OPS[op_name](actual, expected):
                return False, (
                    f"{field} {actual!r} fails {op_name} {expected!r}"
                )
        except TypeError:
            return False, (
                f"guard type error: cannot compare {actual!r} {op_name} {expected!r}"
            )

    return True, None
