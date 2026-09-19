"""
tools/calibrate_effects.py — Measure effect vectors and update config/policy.json.

Reads config/hardware.json (from profile_models.py) and derives:
  - SWITCH_MODEL energy effects from relative µJ/inference differences
  - BATCH_INFERENCE energy effect from batch1 vs batch8 ratio
  - RETRAIN_CURRENT energy cost from measured retrain cost

Run after tools/profile_models.py. Updates config/policy.json with
measured values and provenance comments.

Usage:
    python3 tools/calibrate_effects.py
    python3 tools/calibrate_effects.py --dry-run   # print without writing
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

HARDWARE_JSON = ROOT / "config" / "hardware.json"
POLICY_JSON   = ROOT / "config" / "policy.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def calibrate(dry_run: bool = False) -> dict:
    """
    Derive effect vectors from hardware.json measurements.

    Returns updated policy dict (does not write unless dry_run=False).
    """
    if not HARDWARE_JSON.exists():
        print("ERROR: config/hardware.json not found. Run tools/profile_models.py first.")
        sys.exit(1)

    hw     = _load(HARDWARE_JSON)
    policy = _load(POLICY_JSON)

    profiles = hw.get("model_profiles", {})
    retrain  = hw.get("retrain_cost_uj", {})
    provenance = (
        f"calibrated from hardware.json  "
        f"backend={hw.get('energy_backend','?')}  "
        f"date={time.strftime('%Y-%m-%d')}"
    )

    # ── SWITCH_MODEL effects ──────────────────────────────────────────────
    # Baseline: lstm batch1 (most expensive) → normalise others relative to it
    lstm_uj = profiles.get("lstm", {}).get("uj_per_inference_batch1", 12000.0)

    switch_effects: dict[str, dict] = {}
    for name in ["lstm", "svm", "linear"]:
        mp   = profiles.get(name, {})
        uj   = mp.get("uj_per_inference_batch1", lstm_uj)
        ms   = mp.get("ms_per_inference_batch1", 8.0)
        rel_e = (uj - lstm_uj) / max(lstm_uj, 1.0)   # negative = cheaper
        rel_l = (ms - profiles.get("lstm", {}).get("ms_per_inference_batch1", 8.0)) / \
                max(profiles.get("lstm", {}).get("ms_per_inference_batch1", 8.0), 1.0)
        switch_effects[name] = {
            "accuracy":  0.0,           # accuracy effect is data-dependent; keep 0
            "energy":    round(rel_e, 4),
            "latency":   round(rel_l, 4),
            "_provenance": provenance,
        }
    print(f"SWITCH_MODEL effects: {switch_effects}")

    # ── BATCH_INFERENCE energy effect ─────────────────────────────────────
    lstm_b1 = profiles.get("lstm", {}).get("uj_per_inference_batch1", 12000.0)
    lstm_b8 = profiles.get("lstm", {}).get("uj_per_inference_batch8", 9000.0)
    batch_energy_effect = (lstm_b8 - lstm_b1) / max(lstm_b1, 1.0)   # negative = better
    print(f"BATCH_INFERENCE energy effect: {batch_energy_effect:.4f}")

    # ── RETRAIN_CURRENT energy cost ───────────────────────────────────────
    retrain_uj = retrain.get("lstm", 4_200_000.0)
    print(f"RETRAIN_CURRENT energy_uj: {retrain_uj:.0f}")

    # ── Write back to policy.json ─────────────────────────────────────────
    actions = policy.setdefault("actions", {})

    # SWITCH_MODEL
    if "SWITCH_MODEL" in actions:
        actions["SWITCH_MODEL"]["effect"] = switch_effects
        actions["SWITCH_MODEL"]["_effect_provenance"] = provenance

    # BATCH_INFERENCE
    if "BATCH_INFERENCE" in actions:
        actions["BATCH_INFERENCE"]["effect"]["energy"] = round(batch_energy_effect, 4)
        actions["BATCH_INFERENCE"]["_effect_provenance"] = provenance

    # RETRAIN_CURRENT
    if "RETRAIN_CURRENT" in actions:
        actions["RETRAIN_CURRENT"]["energy_uj"] = retrain_uj
        actions["RETRAIN_CURRENT"]["_energy_provenance"] = provenance

    # SWITCH_MODEL energy_uj — smallest non-lstm model µJ
    min_uj = min(
        profiles.get(n, {}).get("uj_per_inference_batch1", 500.0)
        for n in ["svm", "linear"]
    )
    if "SWITCH_MODEL" in actions:
        actions["SWITCH_MODEL"]["energy_uj"] = round(min_uj, 1)

    policy["_calibrated"] = provenance

    if dry_run:
        print("\n[dry-run] Updated policy (not written):")
        print(json.dumps(policy, indent=2)[:2000])
    else:
        POLICY_JSON.write_text(json.dumps(policy, indent=2))
        print(f"\nWritten → {POLICY_JSON}")

    return policy


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Calibrate policy.json effect vectors from hardware.json."
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Print result without writing policy.json")
    args = parser.parse_args(argv)
    calibrate(dry_run=args.dry_run)
    print("Calibration complete.")


if __name__ == "__main__":
    main()
