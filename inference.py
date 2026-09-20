"""
inference.py — Triage managed inference loop.

BUG-3 FIX (T1.2): model is now cached as (name, mtime, obj).
  torch.load / pickle.load called once per model switch, not once per inference.
  Invalidated when model.csv mtime or contents change.

BUG-2 / T1.4: column renamed to energy_uj (one SCHEMA constant, shared).
BUG-4 / T1.4: scaler loaded from artifacts/scaler.pkl (fit once on train data).

New (T0.1): energy measured via EnergyMeter abstraction (estimator on Windows,
  rapl on Linux/Intel). Backend written to every predictions.csv row.

New (T2.6): serving.json read each loop — supports batch_size, seq_length,
  sampling_rate set by AegisML actuators.

station_id column: written from data if present, else "ST_00".
"""
from __future__ import annotations

import os
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent

# ── schema constant (shared with mape/monitor.py, aegis/core/) ───────────────
# Import lazily to avoid circular dependency at module level
def _get_schema():
    try:
        from aegis.core.models import SCHEMA
        return SCHEMA
    except ImportError:
        # Fallback before T1.4/T2.1 are wired in
        return {
            "true_value": "true_value",
            "predicted_value": "predicted_value",
            "model_used": "model_used",
            "inference_time": "inference_time",
            "energy_uj": "energy_uj",
            "station_id": "station_id",
            "energy_backend": "energy_backend",
        }

# ── energy meter ──────────────────────────────────────────────────────────────
from aegis.energy import get_meter
_meter = get_meter()

# ── paths ─────────────────────────────────────────────────────────────────────
DATA_DIR         = ROOT / "data" / "pems"
PREDICTIONS_FILE = ROOT / "knowledge" / "predictions.csv"
MODEL_CSV        = ROOT / "knowledge" / "model.csv"
SERVING_JSON     = ROOT / "knowledge" / "serving.json"
SCALER_PKL       = ROOT / "artifacts" / "scaler.pkl"

os.makedirs(ROOT / "knowledge", exist_ok=True)
os.makedirs(ROOT / "models", exist_ok=True)

# ── LSTM definition (matches train_models.py) ─────────────────────────────────
class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=50, batch_first=True)
        self.fc   = nn.Linear(50, 1)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1])


# ── model cache (BUG-3 fix) ───────────────────────────────────────────────────
_model_cache: dict = {"name": None, "mtime": None, "obj": None}

def _load_model(name: str):
    """Load model from disk; cache the object; invalidate on mtime change."""
    model_path = ROOT / "models" / (
        f"{name}.pth" if name == "lstm" else f"{name}.pkl"
    )
    try:
        mtime = model_path.stat().st_mtime
    except FileNotFoundError:
        return None

    cached = _model_cache
    if cached["name"] == name and cached["mtime"] == mtime:
        return cached["obj"]

    # Cache miss — load from disk
    if name == "lstm":
        m = LSTMModel()
        m.load_state_dict(torch.load(str(model_path), weights_only=False))
        m.eval()
    else:
        with open(model_path, "rb") as fh:
            m = pickle.load(fh)

    cached["name"]  = name
    cached["mtime"] = mtime
    cached["obj"]   = m
    return m


def _read_model_name() -> str:
    """Read active model from knowledge/model.csv."""
    try:
        return MODEL_CSV.read_text().strip().lower()
    except FileNotFoundError:
        return "lstm"


def _read_serving() -> dict:
    """Read serving.json; return defaults if absent."""
    defaults = {"batch_size": 1, "seq_length": 5, "sampling_rate": 1.0}
    try:
        import json
        with open(SERVING_JSON) as fh:
            cfg = json.load(fh)
        defaults.update(cfg)
    except (FileNotFoundError, Exception):
        pass
    return defaults


def _load_scaler():
    """Load shared scaler; fit a fresh one if not present (fallback only)."""
    if SCALER_PKL.exists():
        with open(SCALER_PKL, "rb") as fh:
            return pickle.load(fh)
    # Fallback: fit on train data (BUG-4 partial fix — full fix in T1.4)
    from sklearn.preprocessing import MinMaxScaler
    train_path = DATA_DIR / "flow_data_train.csv"
    if train_path.exists():
        data = pd.read_csv(train_path)["flow"].values
        sc = MinMaxScaler()
        sc.fit(data.reshape(-1, 1))
        return sc
    return None


def _predict(model, name: str, x_input: np.ndarray) -> float:
    if name == "lstm":
        t = torch.tensor(x_input, dtype=torch.float32).unsqueeze(-1)
        with torch.no_grad():
            return float(model(t).numpy().flatten()[0])
    return float(model.predict(x_input)[0])


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    SCHEMA = _get_schema()
    scaler = _load_scaler()

    print("Loading data stream…")
    test_path = DATA_DIR / "flow_data_test.csv"
    df_full   = pd.read_csv(test_path)
    has_station = "station_id" in df_full.columns
    data        = df_full["flow"].values
    stations    = df_full["station_id"].values if has_station else ["ST_00"] * len(data)

    if scaler is None:
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        scaler.fit(data.reshape(-1, 1))

    data_scaled = scaler.transform(data.reshape(-1, 1)).flatten()

    serving    = _read_serving()
    seq_length = int(serving.get("seq_length", 5))

    def _make_sequences(d, sl):
        X, y = [], []
        for i in range(len(d) - sl):
            X.append(d[i:i + sl])
            y.append(d[i + sl])
        return np.array(X), np.array(y)

    X_stream, y_stream = _make_sequences(data_scaled, seq_length)
    station_stream     = stations[seq_length:]

    print(f"Stream ready: {len(X_stream):,} steps  seq_length={seq_length}  "
          f"energy_backend={_meter.backend}")

    # Initialise predictions CSV
    cols = [SCHEMA["true_value"], SCHEMA["predicted_value"], SCHEMA["model_used"],
            SCHEMA["inference_time"], SCHEMA["energy_uj"],
            SCHEMA["station_id"], SCHEMA["energy_backend"]]
    if not PREDICTIONS_FILE.exists():
        pd.DataFrame(columns=cols).to_csv(PREDICTIONS_FILE, index=False)

    sample_rate = float(serving.get("sampling_rate", 1.0))
    _sample_counter = 0

    for i in range(len(X_stream)):
        # Reload serving config each iteration (actuators may update it)
        if i % 50 == 0:
            serving     = _read_serving()
            seq_length_new = int(serving.get("seq_length", 5))
            sample_rate = float(serving.get("sampling_rate", 1.0))
            if seq_length_new != seq_length:
                # seq_length changed — rebuild sequences
                seq_length = seq_length_new
                X_stream, y_stream = _make_sequences(data_scaled, seq_length)
                station_stream     = stations[seq_length:]
                if i >= len(X_stream):
                    break

        # Sampling rate — skip rows probabilistically
        _sample_counter += sample_rate
        if _sample_counter < 1.0:
            continue
        _sample_counter -= 1.0

        chosen_model = _read_model_name()
        model        = _load_model(chosen_model)
        if model is None:
            print(f"Model '{chosen_model}' not found, skipping.")
            continue

        x_input  = X_stream[i].reshape(1, -1)
        station  = station_stream[i] if i < len(station_stream) else "ST_00"

        with _meter.measure("inference") as reading:
            t0         = time.perf_counter()
            prediction = _predict(model, chosen_model, x_input)
            inf_time   = time.perf_counter() - t0

        true_val  = scaler.inverse_transform([[y_stream[i]]])[0, 0]
        pred_val  = scaler.inverse_transform([[prediction]])[0, 0]

        row = pd.DataFrame([[
            true_val, pred_val, chosen_model,
            inf_time, reading.micro_joules,
            station, reading.backend,
        ]], columns=cols)
        row.to_csv(PREDICTIONS_FILE, mode="a", header=False, index=False)

        print(f"[{i+1}/{len(X_stream)}] "
              f"true={true_val:.1f}  pred={pred_val:.1f}  "
              f"model={chosen_model}  "
              f"energy={reading.micro_joules:.0f}µJ({reading.backend})  "
              f"time={inf_time*1000:.2f}ms  station={station}")

        time.sleep(0.15)

    print("\nInference complete. Predictions saved to", PREDICTIONS_FILE)


if __name__ == "__main__":
    main()
