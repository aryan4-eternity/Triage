"""
mape/manage.py — Triage MAPE-K manager.

T0.1 change: pyRAPL replaced with EnergyMeter abstraction so the process
  starts on any platform (Windows, macOS, AMD), not just Linux/Intel.
T2.8 change: "aegis" approach launches aegis/manage.py instead.
"""
import csv
import os
import sys
import threading
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "mape"))

from execute import execute_mape, execute_drift
from aegis.energy import get_meter as _get_meter

_meter = _get_meter()

log_file         = str(ROOT / "knowledge" / "mape_log.csv")
predictions_file = str(ROOT / "knowledge" / "predictions.csv")
drift_file       = str(ROOT / "knowledge" / "drift.csv")
config_file      = str(ROOT / "approach.conf")

# Ensure log file exists with header
if not os.path.exists(log_file):
    with open(log_file, mode="w", newline="") as file:
        csv.writer(file).writerow(["function", "energy_joules"])


def log_energy(function_name: str, energy_uj: float) -> None:
    with open(log_file, mode="a", newline="") as file:
        csv.writer(file).writerow([function_name, round(energy_uj / 1_000_000, 8)])


def run_execute_mape():
    while True:
        time.sleep(40)
        with _meter.measure("execute_mape") as r:
            t0 = time.perf_counter()
            execute_mape()
            elapsed = time.perf_counter() - t0
        print(f"execute_mape: {elapsed:.4f}s  {r.micro_joules:.0f}µJ({r.backend})")
        log_energy("execute_mape", r.micro_joules)


def run_execute_drift():
    time.sleep(400)
    while True:
        time.sleep(3)
        with _meter.measure("execute_drift") as r:
            execute_drift()
        log_energy("execute_drift", r.micro_joules)


def run_periodic_retrain():
    while True:
        time.sleep(500)
        try:
            df = pd.read_csv(predictions_file)
            df.columns = df.columns.str.strip()
            if not df.empty:
                df.tail(1500).to_csv(drift_file, index=False)
                print("✔ Updated drift.csv with last 1500 rows")
            else:
                print("⚠️ predictions.csv is empty.")
        except FileNotFoundError:
            print("❌ predictions.csv not found.")

        with _meter.measure("periodic_retrain") as r:
            os.system(f"python \"{ROOT / 'retrain.py'}\"")
        log_energy("periodic_retrain", r.micro_joules)


def get_approach_config() -> str:
    if not os.path.exists(config_file):
        return "triage"
    with open(config_file) as f:
        return f.read().strip().lower()


approach = get_approach_config()
print(f"Running configuration: {approach}  energy_backend={_meter.backend}")

# ── aegis: hand off to the AegisML controller ─────────────────────────────
if approach == "aegis":
    import subprocess
    print("Launching AegisML controller (aegis/manage.py)…")
    subprocess.run([sys.executable, str(ROOT / "aegis" / "manage.py")])
    sys.exit(0)

# ── Triage approaches ──────────────────────────────────────────────────────
threads = []

if approach in ["triage", "switch", "switch+retrain"]:
    t1 = threading.Thread(target=run_execute_mape, daemon=True)
    threads.append(t1)
    if approach == "triage":
        t2 = threading.Thread(target=run_execute_drift, daemon=True)
        threads.append(t2)
    elif approach == "switch+retrain":
        t3 = threading.Thread(target=run_periodic_retrain, daemon=True)
        threads.append(t3)

elif approach.startswith("single"):
    print(f"Single model approach: {approach}")
    if "+retrain" in approach:
        t3 = threading.Thread(target=run_periodic_retrain, daemon=True)
        threads.append(t3)

else:
    print(f"Unknown approach '{approach}'. No management threads started.")

for t in threads:
    t.start()

threading.Event().wait()
