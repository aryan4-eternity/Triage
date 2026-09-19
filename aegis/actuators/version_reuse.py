"""
aegis/actuators/version_reuse.py — Restores an archived model version.

Uses the BUG-1-fixed path logic: resolves artifact by name+extension from
the version DIRECTORY, not from data.csv.
"""
from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT       = Path(__file__).resolve().parents[2]
_MODELS    = ROOT / "models"
_VMR       = ROOT / "versionedMR"


def reuse_version(model_name: str, version: str) -> dict:
    """
    Copy the model artifact from versionedMR/<model>/<version>/ to models/.

    Args:
        model_name: "lstm", "svm", or "linear"
        version:    e.g. "version_2"

    Returns outcome-compatible dict.
    """
    ext     = ".pth" if model_name == "lstm" else ".pkl"
    src     = _VMR / model_name / version / f"{model_name}{ext}"
    dst     = _MODELS / f"{model_name}{ext}"

    if not src.exists():
        msg = f"artifact not found: {src}"
        print(f"[actuator] reuse_version ERROR: {msg}")
        return {"action": f"REUSE_VERSION:{model_name}@{version}",
                "error": msg}

    shutil.copy(src, dst)
    print(f"[actuator] reuse_version {model_name}@{version} → {dst}")
    return {
        "action": f"REUSE_VERSION:{model_name}@{version}",
        "ts":     datetime.now(timezone.utc).isoformat(),
        "notes":  f"copied {src.name} to {dst}",
    }
