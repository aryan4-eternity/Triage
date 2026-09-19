"""
aegis/actuators/review.py — Opens an equity review (sets equity_review_pending).

Writes knowledge/review.json. While this flag is set, all model-promoting
actions are ineligible (guards.py checks this — safety invariant).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

ROOT         = Path(__file__).resolve().parents[2]
_REVIEW_JSON = ROOT / "knowledge" / "review.json"


def open_equity_review(
    incident_id: str,
    worst_station: str,
    gap: float,
    notes: str = "",
) -> dict:
    """
    Set equity_review_pending = True in review.json.

    This blocks all model-promoting actions until the review is resolved.
    """
    review = {
        "equity_review_pending": True,
        "opened_at":    datetime.now(timezone.utc).isoformat(),
        "incident_id":  incident_id,
        "worst_station": worst_station,
        "gap":          gap,
        "resolved_by":  None,
        "notes":        notes,
    }
    _REVIEW_JSON.write_text(json.dumps(review, indent=2))
    print(f"[actuator] equity_review OPENED  "
          f"station={worst_station}  gap={gap:.3f}")
    return {
        "action": "EQUITY_REVIEW",
        "ts":     review["opened_at"],
        "notes":  f"pending review for {worst_station} (gap={gap:.3f})",
    }


def resolve_equity_review(resolved_by: str) -> dict:
    """Clear equity_review_pending — call manually after human review."""
    if not _REVIEW_JSON.exists():
        return {"action": "RESOLVE_REVIEW", "error": "no review open"}

    review = json.loads(_REVIEW_JSON.read_text())
    review["equity_review_pending"] = False
    review["resolved_by"]           = resolved_by
    review["resolved_at"]           = datetime.now(timezone.utc).isoformat()
    _REVIEW_JSON.write_text(json.dumps(review, indent=2))
    print(f"[actuator] equity_review RESOLVED by {resolved_by}")
    return {"action": "RESOLVE_REVIEW", "resolved_by": resolved_by}


def is_review_pending() -> bool:
    if not _REVIEW_JSON.exists():
        return False
    try:
        return bool(json.loads(_REVIEW_JSON.read_text()).get(
            "equity_review_pending", False
        ))
    except Exception:
        return False
