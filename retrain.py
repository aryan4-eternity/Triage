"""
retrain.py — Triage model retraining on drift data.

T1.4 changes:
  - BUG-4 fix: loads shared scaler from artifacts/scaler.pkl (never refits on
    test/drift data unless no scaler exists).
  - --model flag: retrain a specific model (default: active model from model.csv).
  - Uses Path-based constants so it works from any CWD.
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

BASE_DIR   = ROOT / "versionedMR"
MODEL_DIR  = ROOT / "models"
DRIFT_FILE = ROOT / "knowledge" / "drift.csv"
MODEL_FILE = ROOT / "knowledge" / "model.csv"
SCALER_PKL = ROOT / "artifacts" / "scaler.pkl"

BASE_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)


def get_next_version(model_name: str) -> int:
    md = BASE_DIR / model_name
    md.mkdir(exist_ok=True)
    existing = [int(d.split("_")[-1])
                for d in os.listdir(md) if d.startswith("version_")]
    return max(existing, default=0) + 1


def save_model_and_data(model, model_name: str, train_data: pd.DataFrame) -> None:
    version      = get_next_version(model_name)
    version_path = BASE_DIR / model_name / f"version_{version}"
    version_path.mkdir(parents=True, exist_ok=True)

    if model_name == "lstm":
        pth = MODEL_DIR / "lstm.pth"
        torch.save(model.state_dict(), str(pth))
        torch.save(model.state_dict(), str(version_path / "lstm.pth"))
    else:
        pkl = MODEL_DIR / f"{model_name}.pkl"
        with open(pkl, "wb") as fh:
            pickle.dump(model, fh)
        with open(version_path / f"{model_name}.pkl", "wb") as fh:
            pickle.dump(model, fh)

    train_data.to_csv(version_path / "data.csv", index=False)
    print(f"✔ {model_name} saved at {version_path} and {MODEL_DIR / model_name}")


def create_sequences(data: np.ndarray, seq_length: int = 5):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i: i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)


class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=50, batch_first=True)
        self.fc   = nn.Linear(50, 1)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1])


def train_lstm(X_train: np.ndarray, y_train: np.ndarray) -> LSTMModel:
    X_t = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)
    y_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)
    model     = LSTMModel()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loader    = DataLoader(TensorDataset(X_t, y_t), batch_size=16, shuffle=True)
    for _ in range(50):
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
    return model


def _load_scaler():
    """Load shared scaler; warn if falling back to a new fit."""
    if SCALER_PKL.exists():
        with open(SCALER_PKL, "rb") as fh:
            return pickle.load(fh), False
    print("⚠️  artifacts/scaler.pkl not found — fitting fresh scaler on drift data. "
          "Run tools/train_models.py first to persist the canonical scaler.")
    return None, True   # caller must fit


def retrain(model_name: str | None = None) -> None:
    if not DRIFT_FILE.exists() or not MODEL_FILE.exists():
        print("❌ Missing drift.csv or model.csv.")
        return

    try:
        drift_data = pd.read_csv(DRIFT_FILE)["true_value"].values
        if model_name is None:
            model_name = MODEL_FILE.read_text().strip()
    except Exception as exc:
        print(f"❌ Error loading files: {exc}")
        return

    print(f"🚀 Retraining {model_name} using drift data…")

    scaler, needs_fit = _load_scaler()
    if needs_fit:
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        scaler.fit(drift_data.reshape(-1, 1))

    data_scaled = scaler.transform(drift_data.reshape(-1, 1)).flatten()
    X_train, y_train = create_sequences(data_scaled)

    if model_name == "linear":
        model = Ridge(alpha=200)
        model.fit(X_train, y_train)
    elif model_name == "svm":
        model = SVR(kernel="linear", C=0.05, tol=0.16)
        model.fit(X_train, y_train)
    elif model_name == "lstm":
        model = train_lstm(X_train, y_train)
    else:
        print(f"❌ Unknown model: {model_name}")
        return

    # Inverse-transform to save original-scale training data for KL comparison
    train_data_orig = scaler.inverse_transform(data_scaled.reshape(-1, 1)).flatten()
    save_model_and_data(model, model_name,
                        pd.DataFrame({"train_data": train_data_orig}))
    print(f"✔ {model_name} retraining complete.")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Retrain a Triage model.")
    parser.add_argument("--model", type=str, default=None,
                        help="Model to retrain (lstm|svm|linear). "
                             "Default: active model from knowledge/model.csv")
    args = parser.parse_args(argv)
    retrain(args.model)


if __name__ == "__main__":
    main()
