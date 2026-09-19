"""
tests/test_version_reuse.py — T1.1 acceptance tests for BUG-1 fix.

Acceptance criterion:
  - get_best_version() returns a dict with the version DIRECTORY, not data.csv
  - execute_drift() copies the actual model artifact (.pth / .pkl)
  - models/data.pth is NEVER created
"""
from __future__ import annotations

import json
import os
import pickle
import shutil
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "mape"))


# ── fixture factory ───────────────────────────────────────────────────────────

def _make_fake_repo(tmp: Path, model_name: str = "linear") -> Path:
    """
    Minimal fake repo under tmp:
      knowledge/{drift.csv, model.csv, mape_info.json, thresholds.json}
      versionedMR/<model>/version_1/  data.csv (far from drift) + artifact
      versionedMR/<model>/version_2/  data.csv (close to drift) + artifact
      models/  (empty target)
    """
    k = tmp / "knowledge"
    k.mkdir()

    # drift window: tightly clustered around 200
    rng = np.random.default_rng(0)
    drift_vals = rng.normal(loc=200.0, scale=5.0, size=1200)
    pd.DataFrame({"true_value": drift_vals}).to_csv(k / "drift.csv", index=False)

    (k / "model.csv").write_text(model_name)
    (k / "mape_info.json").write_text(json.dumps({
        "last_line": 0, "current_energy_threshold": 1,
        "recovery_cycles": 0,
        "ema_scores": {"lstm": 0.82, "linear": 0.75, "svm": 0.79},
    }))
    (k / "thresholds.json").write_text(json.dumps({
        "min_score": 0.78, "max_energy": 1, "beta": 0.95,
        "gamma": 0.8, "alpha": 0.1, "E_m": 0, "E_M": 25000,
    }))

    ext = ".pth" if model_name == "lstm" else ".pkl"
    vmr = tmp / "versionedMR" / model_name

    # version_1 — training data far from drift (clustered around 800)
    v1 = vmr / "version_1"
    v1.mkdir(parents=True)
    v1_data = np.random.default_rng(1).normal(800.0, 5.0, 1200)
    pd.DataFrame({"train_data": v1_data}).to_csv(v1 / "data.csv", index=False)
    if model_name == "lstm":
        (v1 / f"{model_name}{ext}").write_bytes(b"fake_v1")
    else:
        with open(v1 / f"{model_name}{ext}", "wb") as fh:
            pickle.dump({"version": 1}, fh)

    # version_2 — training data close to drift (clustered around 200)
    v2 = vmr / "version_2"
    v2.mkdir(parents=True)
    v2_data = np.random.default_rng(2).normal(200.0, 5.0, 1200)
    pd.DataFrame({"train_data": v2_data}).to_csv(v2 / "data.csv", index=False)
    if model_name == "lstm":
        (v2 / f"{model_name}{ext}").write_bytes(b"fake_v2")
    else:
        with open(v2 / f"{model_name}{ext}", "wb") as fh:
            pickle.dump({"version": 2}, fh)

    (tmp / "models").mkdir()
    return tmp


def _get_analyse(repo: Path):
    """Import analyse with all paths redirected to repo."""
    # purge cached modules to force fresh import with new ROOT
    for key in [k for k in sys.modules if k.startswith("mape")]:
        del sys.modules[key]

    import mape.analyse as m
    m.ROOT              = repo
    m.thresholds_file   = repo / "knowledge" / "thresholds.json"
    m.mape_info_file    = repo / "knowledge" / "mape_info.json"
    m.base_version_dir  = repo / "versionedMR"
    m.current_model_file = repo / "knowledge" / "model.csv"
    m.drift_kl_file     = repo / "knowledge" / "drift_kl.json"
    m.drift_data_file   = repo / "knowledge" / "drift.csv"
    return m


def _get_execute(repo: Path, model_name: str, plan_result: dict):
    """Import execute with paths redirected and plan_drift stubbed."""
    for key in [k for k in sys.modules if k.startswith("mape")]:
        del sys.modules[key]

    import mape.execute as m
    m.ROOT       = repo
    m.model_file = repo / "knowledge" / "model.csv"
    m.models_dir = repo / "models"

    # stub plan_drift to return a canned replace decision
    v2_dir = repo / "versionedMR" / model_name / "version_2"
    m.plan_drift = lambda: {
        "action":  "replace",
        "version": {"dir": str(v2_dir), "kl": 0.05, "model": model_name},
    }

    # stub time.sleep so the 400s sleep is a no-op
    fake_time = types.ModuleType("time")
    fake_time.sleep = lambda *_: None
    m.time = fake_time

    return m


# ── get_best_version unit tests ───────────────────────────────────────────────

class TestGetBestVersion:

    def test_returns_dict_not_data_csv_path(self, tmp_path):
        repo   = _make_fake_repo(tmp_path, "linear")
        analyse = _get_analyse(repo)
        result  = analyse.get_best_version("linear")

        assert result is not None, "Expected a best version"
        assert isinstance(result, dict), "Must be a dict"
        assert "dir" in result and "kl" in result and "model" in result
        assert not result["dir"].endswith("data.csv"), (
            "BUG-1 regression: dir must be the version directory, not data.csv"
        )

    def test_returns_lower_kl_version(self, tmp_path):
        repo   = _make_fake_repo(tmp_path, "linear")
        analyse = _get_analyse(repo)
        result  = analyse.get_best_version("linear")

        assert result is not None
        assert "version_2" in result["dir"], (
            f"version_2 (close to drift) should win, got: {result['dir']}"
        )

    def test_returns_none_when_only_one_version(self, tmp_path):
        repo = _make_fake_repo(tmp_path, "linear")
        shutil.rmtree(repo / "versionedMR" / "linear" / "version_2")
        analyse = _get_analyse(repo)
        assert analyse.get_best_version("linear") is None


# ── execute_drift integration tests ──────────────────────────────────────────

class TestExecuteDrift:

    def test_model_artifact_is_copied_not_data_csv(self, tmp_path):
        """execute_drift must copy the .pkl, not data.csv."""
        repo    = _make_fake_repo(tmp_path, "linear")
        execute = _get_execute(repo, "linear", {})
        execute.execute_drift()

        copied = repo / "models" / "linear.pkl"
        assert copied.exists(), "linear.pkl must be present in models/"
        with open(copied, "rb") as fh:
            obj = pickle.load(fh)
        assert obj == {"version": 2}, "Should have copied version_2 artifact"

    def test_data_pth_never_created(self, tmp_path):
        """models/data.pth must never exist after execute_drift."""
        repo    = _make_fake_repo(tmp_path, "linear")
        execute = _get_execute(repo, "linear", {})
        execute.execute_drift()

        bad = repo / "models" / "data.pth"
        assert not bad.exists(), (
            "BUG-1 regression: models/data.pth was created"
        )

    def test_lstm_pth_extension_resolved(self, tmp_path):
        """LSTM model correctly gets .pth extension."""
        repo    = _make_fake_repo(tmp_path, "lstm")
        execute = _get_execute(repo, "lstm", {})
        execute.execute_drift()

        copied = repo / "models" / "lstm.pth"
        assert copied.exists(), "lstm.pth must be present in models/"
        assert copied.read_bytes() == b"fake_v2", "Should have copied version_2 bytes"
