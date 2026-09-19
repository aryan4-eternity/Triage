"""
tools/profile_models.py — warm-cache energy and latency profiler.

Run AFTER BUG-3 fix (model cache) so torch.load is not in the hot path.
Writes results to config/hardware.json and updates knowledge/thresholds.json
(E_m, E_M from measured data).

Usage:
    python3 tools/profile_models.py
    python3 tools/profile_models.py --n 500 --repeats 2
    python3 tools/profile_models.py --retrain       # measure retrain cost too
    AEGIS_ENERGY=estimator python3 tools/profile_models.py
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import platform
import subprocess
import sys
import time
from pathlib import Path
from statistics import median

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aegis.energy import get_meter

MODELS_DIR     = ROOT / "models"
HARDWARE_JSON  = ROOT / "config" / "hardware.json"
THRESHOLDS_JSON = ROOT / "knowledge" / "thresholds.json"
DATA_DIR       = ROOT / "data" / "pems"


class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=50, batch_first=True)
        self.fc   = nn.Linear(50, 1)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1])


def _load_model(name: str):
    if name == "lstm":
        m = LSTMModel()
        m.load_state_dict(torch.load(
            str(MODELS_DIR / "lstm.pth"), weights_only=False))
        m.eval()
    else:
        ext = ".pkl"
        with open(MODELS_DIR / f"{name}{ext}", "rb") as fh:
            m = pickle.load(fh)
    return m


def _make_inputs(n: int, seq_length: int = 5) -> np.ndarray:
    """Generate random normalised input windows."""
    return np.random.default_rng(42).uniform(0.0, 1.0, (n, seq_length))


def _predict_one(model, name: str, x: np.ndarray):
    if name == "lstm":
        t = torch.tensor(x.reshape(1, seq_length := x.shape[0]),
                         dtype=torch.float32).unsqueeze(-1)
        with torch.no_grad():
            model(t)
    else:
        model.predict(x.reshape(1, -1))


def _warmup(model, name: str, inputs: np.ndarray, n: int = 20):
    for i in range(min(n, len(inputs))):
        _predict_one(model, name, inputs[i])


def profile_model(
    name: str,
    meter,
    inputs: np.ndarray,
    batch_sizes: list[int],
    n_inferences: int,
    repeats: int,
) -> dict:
    """
    Profile one model at multiple batch sizes.
    Returns nested dict: batch_size -> {uj_median, ms_median}.
    """
    model = _load_model(name)
    _warmup(model, name, inputs)

    results: dict[int, dict] = {}

    for bs in batch_sizes:
        uj_readings: list[float] = []
        ms_readings: list[float] = []

        for _ in range(repeats):
            for i in range(0, n_inferences, bs):
                batch = inputs[i: i + bs]
                if len(batch) == 0:
                    break

                with meter.measure(f"{name}_bs{bs}") as r:
                    t0 = time.perf_counter()
                    for row in batch:
                        _predict_one(model, name, row)
                    elapsed = time.perf_counter() - t0

                per_inf_uj = r.micro_joules / len(batch)
                per_inf_ms = elapsed * 1000.0 / len(batch)
                uj_readings.append(per_inf_uj)
                ms_readings.append(per_inf_ms)

        results[bs] = {
            "uj_per_inference_median": round(median(uj_readings), 2),
            "ms_per_inference_median": round(median(ms_readings), 4),
        }
        print(f"  {name} bs={bs:2d}: "
              f"{results[bs]['uj_per_inference_median']:.1f} µJ/inf  "
              f"{results[bs]['ms_per_inference_median']:.3f} ms/inf")

    return results


def measure_idle(meter, duration_s: float = 10.0) -> float:
    """Measure idle power in µW (µJ/s) over duration_s seconds."""
    print(f"Measuring idle power ({duration_s:.0f}s)…", flush=True)
    with meter.measure("idle") as r:
        time.sleep(duration_s)
    uw = r.micro_joules / r.seconds if r.seconds > 0 else 0.0
    print(f"  Idle: {uw/1e6:.3f} W  ({uw:.0f} µW)")
    return uw


def measure_retrain(model_name: str, meter) -> float:
    """Measure energy for one full retrain cycle (µJ)."""
    print(f"Measuring retrain cost for {model_name}…", flush=True)
    retrain_script = ROOT / "retrain.py"
    with meter.measure(f"retrain_{model_name}") as r:
        subprocess.run(
            [sys.executable, str(retrain_script)],
            cwd=str(ROOT), check=False,
        )
    print(f"  {model_name} retrain: {r.micro_joules:.0f} µJ")
    return r.micro_joules


def main(argv=None):
    parser = argparse.ArgumentParser(description="Profile model energy/latency.")
    parser.add_argument("--n",        type=int,   default=1000,
                        help="Inferences per model per batch size (default 1000)")
    parser.add_argument("--repeats",  type=int,   default=3,
                        help="Repeats per measurement (default 3)")
    parser.add_argument("--retrain",  action="store_true",
                        help="Also measure retrain cost per model")
    parser.add_argument("--idle-s",   type=float, default=10.0,
                        help="Idle measurement duration seconds (default 10)")
    args = parser.parse_args(argv)

    meter = get_meter()
    print(f"Energy backend: {meter.backend}  measured={meter.measured}")

    batch_sizes = [1, 2, 4, 8]
    inputs      = _make_inputs(args.n + 8, seq_length=5)

    model_names = ["lstm", "svm", "linear"]
    profiles: dict = {}

    for name in model_names:
        model_path = MODELS_DIR / (f"{name}.pth" if name == "lstm" else f"{name}.pkl")
        if not model_path.exists():
            print(f"  Skipping {name} — model file not found at {model_path}")
            continue
        print(f"\nProfiling {name}…")
        profiles[name] = profile_model(
            name, meter, inputs, batch_sizes, args.n, args.repeats
        )

    # Idle power
    idle_uw = measure_idle(meter, args.idle_s)

    # Retrain costs
    retrain_costs: dict = {}
    if args.retrain:
        for name in model_names:
            retrain_costs[name] = measure_retrain(name, meter)
    else:
        # Load existing values if present (preserve calibrated values)
        if HARDWARE_JSON.exists():
            existing = json.loads(HARDWARE_JSON.read_text())
            retrain_costs = existing.get("retrain_cost_uj", {})

    # ── Write config/hardware.json ────────────────────────────────────────────
    hw: dict = {}
    if HARDWARE_JSON.exists():
        hw = json.loads(HARDWARE_JSON.read_text())

    hw["cpu_model"]   = platform.processor() or "unknown"
    hw["os"]          = platform.system()
    hw["energy_backend"] = meter.backend
    hw["measured"]    = meter.measured
    hw["_date"]       = time.strftime("%Y-%m-%d")
    hw["idle_power_uw"] = round(idle_uw, 1)

    model_profiles = hw.get("model_profiles", {})
    for name, bs_data in profiles.items():
        mp = model_profiles.get(name, {})
        for bs, vals in bs_data.items():
            mp[f"uj_per_inference_batch{bs}"] = vals["uj_per_inference_median"]
            mp[f"ms_per_inference_batch{bs}"] = vals["ms_per_inference_median"]
        mp.pop("_note", None)
        model_profiles[name] = mp
    hw["model_profiles"] = model_profiles

    rc = hw.get("retrain_cost_uj", {})
    rc.update({k: round(v, 1) for k, v in retrain_costs.items()})
    hw["retrain_cost_uj"] = rc

    HARDWARE_JSON.write_text(json.dumps(hw, indent=2))
    print(f"\nWritten → {HARDWARE_JSON}")

    # ── Update knowledge/thresholds.json (E_m, E_M) ──────────────────────────
    if profiles:
        all_uj = [
            v["uj_per_inference_median"]
            for bs_data in profiles.values()
            for v in bs_data.values()
        ]
        e_m = max(0.0, min(all_uj) * 0.8)
        e_M = max(all_uj) * 1.5

        if THRESHOLDS_JSON.exists():
            thr = json.loads(THRESHOLDS_JSON.read_text())
        else:
            thr = {"min_score": 0.78, "max_energy": 1, "beta": 0.95,
                   "gamma": 0.8, "alpha": 0.1}

        thr["E_m"] = round(e_m, 2)
        thr["E_M"] = round(e_M, 2)
        thr["_provenance"] = (
            f"calibrated from profile_models.py  "
            f"backend={meter.backend}  date={time.strftime('%Y-%m-%d')}"
        )
        THRESHOLDS_JSON.write_text(json.dumps(thr, indent=4))
        print(f"Updated thresholds.json: E_m={e_m:.1f}  E_M={e_M:.1f}")
        print(f"Written → {THRESHOLDS_JSON}")

    print("\nProfile complete.")


if __name__ == "__main__":
    main()
