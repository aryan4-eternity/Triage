"""
mape/monitor.py — HarmonE MAPE-K: Monitor stage.

T1.4 change: reads energy column via SCHEMA["energy_uj"] (BUG-2 fix).
Paths resolved from ROOT so the module works from any CWD (SMELL-2 fix).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import entropy
from sklearn.metrics import r2_score

ROOT = Path(__file__).resolve().parents[1]

mape_info_file  = ROOT / "knowledge" / "mape_info.json"
thresholds_file = ROOT / "knowledge" / "thresholds.json"
model_file      = ROOT / "knowledge" / "model.csv"
predictions_file = ROOT / "knowledge" / "predictions.csv"

# Column name via SCHEMA (BUG-2 fix)
try:
    from aegis.core.models import SCHEMA as _SCHEMA
    _ENERGY_COL = _SCHEMA["energy_uj"]
except ImportError:
    _ENERGY_COL = "energy_uj"


def load_mape_info() -> dict:
    with open(mape_info_file) as f:
        return json.load(f)


def save_mape_info(data: dict) -> None:
    with open(mape_info_file, "w") as f:
        json.dump(data, f, indent=4)


def get_current_model() -> str | None:
    try:
        return model_file.read_text().strip()
    except FileNotFoundError:
        return None


def monitor_mape() -> dict | None:
    """Monitor R² score and normalised energy; update EMA scores."""
    info = load_mape_info()
    last_line     = info["last_line"]
    current_model = get_current_model()

    if current_model is None:
        print("⚠️ No model currently in use.")
        return None

    try:
        df = pd.read_csv(predictions_file, skiprows=range(1, last_line + 1))
        df.columns = df.columns.str.strip()
        if df.empty:
            print("📉 No new data in predictions.csv")
            return None
    except FileNotFoundError:
        print("⚠️ No predictions.csv found.")
        return None

    print(f"🆕 Processing {len(df)} new rows for {current_model.upper()}")

    r2 = r2_score(df["true_value"], df["predicted_value"])

    with open(thresholds_file) as f:
        thresholds = json.load(f)

    energy_min = thresholds["E_m"]
    energy_max = thresholds["E_M"]

    # Use SCHEMA column name; fall back to legacy "energy" if column absent
    if _ENERGY_COL in df.columns:
        energy_mean = df[_ENERGY_COL].mean()
    elif "energy" in df.columns:
        energy_mean = df["energy"].mean()
    else:
        print(f"⚠️ Energy column '{_ENERGY_COL}' not found — defaulting to 0.")
        energy_mean = 0.0

    denom = energy_max - energy_min
    energy_normalized = (
        (energy_mean - energy_min) / denom if denom > 0 else 0.0
    )
    energy_normalized = float(np.clip(energy_normalized, 0.0, 1.0))

    beta        = thresholds.get("beta", 0.95)
    model_score = beta * r2 + (1 - beta) * (1 - energy_normalized)

    gamma      = thresholds.get("gamma", 0.8)
    prev_score = info["ema_scores"].get(current_model, 0.5)
    final_score = gamma * model_score + (1 - gamma) * prev_score

    info["ema_scores"][current_model] = final_score
    info["last_line"] += len(df)
    save_mape_info(info)

    print(f"🔹 R²={r2:.4f}  energy_norm={energy_normalized:.4f}  "
          f"EMA({current_model})={final_score:.4f}")

    return {
        "r2_score":         r2,
        "normalized_energy": energy_normalized,
        "score":            final_score,
    }


def monitor_drift() -> dict | None:
    """Monitor KL divergence between two consecutive 1200-row windows."""
    try:
        df = pd.read_csv(predictions_file)
        df.columns = df.columns.str.strip()
        if df.empty:
            print("Drift Monitor: No predictions yet.")
            return None
    except FileNotFoundError:
        print("Drift Monitor: predictions.csv not found.")
        return None

    window_size = 1200
    if len(df) < window_size * 2:
        print(f"Not enough data for drift detection "
              f"({len(df)} rows, need {window_size * 2})")
        return None

    reference = df["true_value"].iloc[-2 * window_size: -window_size]
    current   = df["true_value"].iloc[-window_size:]

    kl_div = float(entropy(
        np.histogram(reference, bins=50, density=True)[0] + 1e-10,
        np.histogram(current,   bins=50, density=True)[0] + 1e-10,
    ))
    print(f"🌊 Drift: KL={kl_div:.4f}")
    return {"kl_div": kl_div}
