"""
mape/analyse.py — Triage MAPE-K: Analyse stage.

BUG-1 FIX (T1.1): get_best_version() now returns a dict with the version
DIRECTORY path (not data.csv), the winning KL divergence, and the model name.
execute.py uses this to resolve the actual model artifact (.pth / .pkl).
"""
import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import entropy

# ── package-safe imports (works both as script and as package) ────────────────
if __package__:
    from mape.monitor import monitor_mape, monitor_drift
else:
    from monitor import monitor_mape, monitor_drift

ROOT = Path(__file__).resolve().parents[1]

thresholds_file  = ROOT / "knowledge" / "thresholds.json"
mape_info_file   = ROOT / "knowledge" / "mape_info.json"
base_version_dir = ROOT / "versionedMR"
current_model_file = ROOT / "knowledge" / "model.csv"
drift_kl_file    = ROOT / "knowledge" / "drift_kl.json"
drift_data_file  = ROOT / "knowledge" / "drift.csv"


def load_mape_info():
    with open(mape_info_file) as f:
        return json.load(f)


def save_mape_info(data):
    with open(mape_info_file, "w") as f:
        json.dump(data, f, indent=4)


def analyse_mape():
    """Analyse performance; decide if a model switch is needed."""
    mape_data = monitor_mape()
    if not mape_data:
        print("⚠️ No MAPE data available for analysis.")
        return None

    with open(thresholds_file) as f:
        thresholds = json.load(f)

    min_score              = thresholds["min_score"]
    original_energy_threshold = thresholds["max_energy"]

    mape_info = load_mape_info()
    current_energy_threshold = mape_info.get("current_energy_threshold",
                                              original_energy_threshold)
    recovery_cycles = mape_info["recovery_cycles"]

    used_energy = mape_data["normalized_energy"]
    new_energy_threshold = (current_energy_threshold
                            + 0.95 * (original_energy_threshold - used_energy))
    mape_info["current_energy_threshold"] = new_energy_threshold

    switch_needed     = False
    threshold_violated = None

    if recovery_cycles > 0:
        recovery_cycles -= 1
        print(f"⏳ Recovery mode active: {recovery_cycles} cycles remaining.")
    else:
        if mape_data["score"] < min_score:
            print("⚠️ Model score too low! Switch required.")
            switch_needed     = True
            threshold_violated = "score"

        if used_energy > current_energy_threshold:
            print(f"⚠️ Energy exceeded! Used: {used_energy:.4f}, "
                  f"Limit: {current_energy_threshold:.4f}")
            switch_needed     = True
            threshold_violated = "energy"
            recovery_cycles   = 3

    mape_info["recovery_cycles"] = recovery_cycles
    save_mape_info(mape_info)
    print(f"📊 Updated Energy Threshold: {new_energy_threshold:.4f}")

    return {
        "switch_needed":    switch_needed,
        "score":            mape_data["score"],
        "threshold_violated": threshold_violated,
    }


def get_model_versions(model_name: str) -> list[str]:
    """Return sorted list of version directory names for a given model."""
    model_dir = base_version_dir / model_name
    if not model_dir.exists():
        return []
    return sorted(
        [d for d in os.listdir(model_dir) if d.startswith("version_")],
        key=lambda x: int(x.split("_")[-1]),
    )


def get_best_version(model_name: str) -> dict | None:
    """
    BUG-1 FIX: Return the version DIRECTORY with the lowest KL divergence,
    along with the KL value and model name — NOT the path to data.csv.

    Returns:
        {"dir": str, "kl": float, "model": str}  if a suitable version exists
        None                                       otherwise
    """
    versions = get_model_versions(model_name)
    if len(versions) <= 1:
        return None

    if not drift_data_file.exists():
        print("⚠️ No drift.csv found. Cannot compare versions.")
        return None

    try:
        drift_data = pd.read_csv(drift_data_file)["true_value"].values
        drift_hist, _ = np.histogram(drift_data, bins=50, density=True)
        drift_hist   += 1e-10
    except Exception as exc:
        print(f"❌ Error reading drift.csv: {exc}")
        return None

    min_kl_div   = float("inf")
    best_dir     = None

    for version in versions:
        version_dir      = base_version_dir / model_name / version
        version_data_path = version_dir / "data.csv"
        if not version_data_path.exists():
            continue

        try:
            version_data = pd.read_csv(version_data_path)["train_data"].values
            version_hist, _ = np.histogram(version_data, bins=50, density=True)
            version_hist   += 1e-10
        except Exception as exc:
            print(f"❌ Error reading {version_data_path}: {exc}")
            continue

        kl_div = float(np.clip(entropy(drift_hist, version_hist), 0, 10))
        print(f"🔎 KL divergence for {version}: {kl_div:.4f}")

        if kl_div < min_kl_div:
            min_kl_div = kl_div
            best_dir   = str(version_dir)  # ← directory, not data.csv

    with open(drift_kl_file, "w") as f:
        json.dump({"best_version_dir": best_dir, "min_kl_div": min_kl_div}, f,
                  indent=4)

    if min_kl_div < 0.75 and best_dir is not None:
        return {"dir": best_dir, "kl": min_kl_div, "model": model_name}
    return None


def analyse_drift():
    """Analyse drift; decide between version reuse and retrain."""
    drift_data = monitor_drift()
    if not drift_data:
        return None

    kl_div         = drift_data["kl_div"]
    drift_detected = kl_div > 0.75

    if drift_detected:
        print(f"🚨 Drift detected! KL divergence = {kl_div:.4f}")
        try:
            df = pd.read_csv(ROOT / "knowledge" / "predictions.csv")
            df.columns = df.columns.str.strip()
            df.tail(1200).to_csv(drift_data_file, index=False)
        except FileNotFoundError:
            print("No predictions file found to store drift data.")

        with open(current_model_file) as f:
            current_model = f.read().strip()

        best = get_best_version(current_model)

        if best:
            print(f"✔ Best version: dir={best['dir']}  kl={best['kl']:.4f}")
            return {"drift_detected": True, "best_version": best}

        return {"drift_detected": True, "best_version": None}

    return {"drift_detected": False}
