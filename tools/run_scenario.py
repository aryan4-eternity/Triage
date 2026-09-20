"""
tools/run_scenario.py — Deterministic scenario runner for AegisML.

Generates synthetic data for a named scenario, runs one full
MONITOR → ANALYZE → PLAN → EXECUTE cycle (headless, no real inference loop),
and prints the MAPE trace and candidate table.

Usage:
    python3 tools/run_scenario.py --scenario drift_benign
    python3 tools/run_scenario.py --scenario energy_pressure --profile energy_first
    python3 tools/run_scenario.py --list

Scenarios (ARCHITECTURE.md §8):
    normal              — no injected anomaly → OBSERVE
    drift_benign        — KL spike, accuracy holds → OBSERVE / LOWER_SAMPLING
    recoverable_drift   — shift matching archived version → REUSE_VERSION
    model_degradation   — shift + R² collapse → RETRAIN_CURRENT
    energy_pressure     — sustained energy overspend → SWITCH_MODEL / BATCH_INFERENCE
    latency_pressure    — inflated inference time → BATCH_INFERENCE / REDUCE_WINDOW
    equity_shift        — one station diverges → EQUITY_REVIEW
    weight_profile      — same incident, balanced vs accuracy_first
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aegis.core.analyze  import analyze
from aegis.core.monitor  import produce_snapshot
from aegis.core.models   import SCHEMA
from aegis.core.plan     import plan

# ── Scenario definitions ──────────────────────────────────────────────────────

def _base_window(rows: int = 1200, seed: int = 42) -> pd.DataFrame:
    """Generate a clean, low-variance prediction window."""
    rng   = np.random.default_rng(seed)
    true  = rng.normal(400.0, 20.0, rows)
    pred  = true + rng.normal(0.0, 8.0, rows)
    times = rng.uniform(0.003, 0.006, rows)
    uj    = rng.uniform(800.0, 1200.0, rows)
    stations = [f"ST_{(i % 8) + 1:02d}" for i in range(rows)]
    return pd.DataFrame({
        SCHEMA["true_value"]:      true,
        SCHEMA["predicted_value"]: pred,
        SCHEMA["model_used"]:      "lstm",
        SCHEMA["inference_time"]:  times,
        SCHEMA["energy_uj"]:       uj,
        SCHEMA["station_id"]:      stations,
        SCHEMA["energy_backend"]:  "estimator",
    })


SCENARIOS: dict[str, dict] = {
    "normal": {
        "description": "No anomaly — expect OBSERVE",
        "regime": None,
        "energy_spike": False,
        "latency_spike": False,
        "equity_station": None,
        "r2_collapse": False,
    },
    "drift_benign": {
        "description": "Distribution shift, accuracy holds — expect DRIFT_ONLY",
        "regime": (600, 1200, 2.5, 300.0),
        "energy_spike": False,
        "latency_spike": False,
        "equity_station": None,
        "r2_collapse": False,
    },
    "recoverable_drift": {
        "description": "Shift matching archived version — expect REUSE_VERSION",
        "regime": (600, 1200, 2.5, 300.0),
        "energy_spike": False,
        "latency_spike": False,
        "equity_station": None,
        "r2_collapse": False,
        "force_archived_version": True,
    },
    "model_degradation": {
        "description": "Shift + R² collapse — expect RETRAIN_CURRENT",
        "regime": (0, 1200, 2.0, 200.0),
        "energy_spike": False,
        "latency_spike": False,
        "equity_station": None,
        "r2_collapse": True,
    },
    "energy_pressure": {
        "description": "Sustained energy overspend — expect SWITCH_MODEL or BATCH_INFERENCE",
        "regime": None,
        "energy_spike": True,
        "latency_spike": False,
        "equity_station": None,
        "r2_collapse": False,
    },
    "latency_pressure": {
        "description": "Inflated inference time — expect BATCH_INFERENCE or REDUCE_WINDOW",
        "regime": None,
        "energy_spike": False,
        "latency_spike": True,
        "equity_station": None,
        "r2_collapse": False,
    },
    "equity_shift": {
        "description": "One station diverges — expect EQUITY_REVIEW + promotion block",
        "regime": None,
        "energy_spike": False,
        "latency_spike": False,
        "equity_station": "ST_07",
        "r2_collapse": False,
    },
    "weight_profile": {
        "description": "Same drift incident, balanced vs accuracy_first",
        "regime": (600, 1200, 2.5, 300.0),
        "energy_spike": False,
        "latency_spike": False,
        "equity_station": None,
        "r2_collapse": False,
        "compare_profiles": ["balanced", "accuracy_first"],
    },
}


def _build_window(cfg: dict, seed: int = 42) -> pd.DataFrame:
    """Build a synthetic prediction window for a scenario."""
    df  = _base_window(1200, seed)
    rng = np.random.default_rng(seed + 1)

    # Regime shift — applied uniformly to all stations to avoid spurious equity violations
    if cfg.get("regime"):
        s, e, scale, shift = cfg["regime"]
        s = max(0, min(s, 1199)); e = max(s + 1, min(e, 1200))
        n_regime = e - s
        orig_true = df.iloc[s:e][SCHEMA["true_value"]].values.copy()
        new_true  = orig_true * scale + shift
        df.iloc[s:e, df.columns.get_loc(SCHEMA["true_value"])] = new_true
        # Keep prediction errors at same relative level (no extra accuracy damage)
        orig_pred = df.iloc[s:e][SCHEMA["predicted_value"]].values
        pred_err  = orig_pred - orig_true  # preserve original errors
        df.iloc[s:e, df.columns.get_loc(SCHEMA["predicted_value"])] = new_true + pred_err

    # R² collapse — inject large prediction errors
    if cfg.get("r2_collapse"):
        df[SCHEMA["predicted_value"]] = (
            df[SCHEMA["true_value"]] * rng.uniform(0.3, 0.6, 1200) +
            rng.normal(0, 150.0, 1200)
        )

    # Energy spike — push µJ well above E_M so normalized > 1.0
    if cfg.get("energy_spike"):
        import json as _json
        thr_path = ROOT / "knowledge" / "thresholds.json"
        e_M = 25000.0
        if thr_path.exists():
            try:
                e_M = float(_json.loads(thr_path.read_text()).get("E_M", 25000.0))
            except Exception:
                pass
        df[SCHEMA["energy_uj"]] = rng.uniform(e_M * 1.3, e_M * 1.8, 1200)

    # Latency spike — push p95 well above the clean baseline
    if cfg.get("latency_spike"):
        df[SCHEMA["inference_time"]] = rng.uniform(0.035, 0.060, 1200)

    # Equity shift: degrade one station's predictions realistically
    # Target: gap ~0.2 (above hard_ceiling=0.15), not a catastrophic collapse
    if cfg.get("equity_station"):
        bad  = cfg["equity_station"]
        mask = df[SCHEMA["station_id"]] == bad
        n    = mask.sum()
        if n > 0:
            base_true = df.loc[mask, SCHEMA["true_value"]].values
            # Add systematic bias + increased noise to produce realistic degradation
            df.loc[mask, SCHEMA["predicted_value"]] = (
                base_true * 0.72 + rng.normal(50.0, 30.0, n)
            )

    return df


def _print_candidate_table(candidates: list) -> None:
    """Pretty-print the full candidate table including rejected actions."""
    print("\n" + "=" * 80)
    print("CANDIDATE TABLE")
    print("=" * 80)
    header = f"{'Action':<35} {'Eligible':<10} {'Utility':>8}  {'Q':>6} {'E':>6} {'R':>6}  Reason"
    print(header)
    print("-" * 80)
    for c in candidates:
        if c.eligible:
            t     = c.terms or {}
            u_str = f"{c.utility:+.4f}" if c.utility is not None else "  N/A"
            q_str = f"{t.get('Q', 0):.3f}"
            e_str = f"{t.get('E', 0):.3f}"
            r_str = f"{t.get('R', 0):.3f}"
            reason = (c.reason or "")[:40]
            print(f"  {c.action:<33} {'YES':<10} {u_str:>8}  {q_str:>6} {e_str:>6} {r_str:>6}  {reason}")
        else:
            rg = (c.rejected_by_guard or "")[:50]
            print(f"  {c.action:<33} {'NO':<10} {'':>8}  {'':>6} {'':>6} {'':>6}  GUARD: {rg}")
    print("=" * 80)


def run_scenario(
    name: str,
    seed: int = 42,
    profile: str | None = None,
    active_model: str = "lstm",
    verbose: bool = True,
) -> dict:
    """
    Run one scenario cycle and return the result dict.

    Returns:
        {scenario, incident_type, chosen_action, utility, candidates, reason}
    """
    if name not in SCENARIOS:
        raise ValueError(f"Unknown scenario '{name}'. Use --list to see options.")

    cfg    = SCENARIOS[name]
    window = _build_window(cfg, seed)

    if verbose:
        print(f"\n{'='*60}")
        print(f"SCENARIO: {name}")
        print(f"  {cfg['description']}")
        print(f"  seed={seed}  model={active_model}  profile={profile or 'balanced'}")
        print(f"{'='*60}")

    # MONITOR
    try:
        snapshot = produce_snapshot(window, active_model, scenario_tag=name)
    except Exception as exc:
        print(f"[MONITOR] error: {exc}")
        return {"scenario": name, "error": str(exc)}

    # Build 22 history snapshots so dynamic boundaries are active.
    # These are "normal operation" windows — same distribution as the scenario's
    # pre-anomaly period so the dynamic bounds tighten around normal behaviour.
    # We vary the noise seed each window to get realistic σ estimates.
    clean_history: list = []
    use_energy_history = cfg.get("energy_spike", False)

    for h_i in range(22):
        try:
            h_win = _base_window(1200, seed=seed + h_i)  # vary seed for realistic σ
            if use_energy_history:
                import json as _json
                thr_path = ROOT / "knowledge" / "thresholds.json"
                e_M = 25000.0
                if thr_path.exists():
                    try:
                        e_M = float(_json.loads(thr_path.read_text()).get("E_M", 25000.0))
                    except Exception:
                        pass
                rng_h = np.random.default_rng(seed + 200 + h_i)
                h_win[SCHEMA["energy_uj"]] = rng_h.uniform(e_M * 1.5, e_M * 2.0, len(h_win))
            h_snap = produce_snapshot(h_win, active_model)
            clean_history.append(h_snap)
        except Exception:
            break

    if verbose:
        print(f"\n[MONITOR] snap={snapshot.snapshot_id}")
        print(f"  accuracy.r2     = {snapshot.accuracy.r2:.4f}")
        print(f"  drift.kl_div    = {snapshot.drift.kl_div:.4f}")
        print(f"  energy.norm     = {snapshot.energy.normalized:.4f}")
        print(f"  latency.p95_ms  = {snapshot.latency.p95_ms:.2f}")
        print(f"  equity.gap      = {snapshot.equity.gap:.4f}")
        print(f"  equity.worst    = {snapshot.equity.worst_station}  "
              f"worst_r2={snapshot.equity.worst_r2:.4f}")

    # ANALYZE — run 3 passes to satisfy persistence=2 and let energy
    # threshold drift for energy_pressure scenario
    violation_counts: dict[str, int] = {}
    incident = analyze(snapshot, clean_history, violation_counts)
    incident = analyze(snapshot, clean_history + [snapshot], violation_counts)
    incident = analyze(snapshot, clean_history + [snapshot, snapshot], violation_counts)
    if verbose:
        print(f"\n[ANALYZE] incident={incident.incident_id}")
        print(f"  type     = {incident.type}")
        print(f"  violated = {incident.violated}")
        sev_str  = "  ".join(f"{k}={v:.2f}" for k, v in incident.severity.items() if v > 0)
        print(f"  severity = {sev_str or 'none'}")
        print(f"  reason   = {incident.classification_reason}")

    # PLAN — optionally compare two profiles
    profiles_to_run = cfg.get("compare_profiles", [profile or "balanced"])

    results  = []
    last_dec = None

    for prof in profiles_to_run:
        rng = np.random.default_rng(seed)
        dec = plan(incident, snapshot, store=None, rng=rng,
                   profile_override=prof)
        last_dec = dec

        if verbose:
            print(f"\n[PLAN] profile={prof}")
            print(f"  decision    = {dec.decision_id}")
            print(f"  chosen      = {dec.chosen_action}")
            print(f"  exploratory = {dec.exploratory}")
            print(f"  reason      = {dec.reason}")
            _print_candidate_table(dec.candidates)

        results.append({
            "profile":        prof,
            "chosen_action":  dec.chosen_action,
            "exploratory":    dec.exploratory,
            "reason":         dec.reason,
            "n_candidates":   len(dec.candidates),
            "n_eligible":     sum(1 for c in dec.candidates if c.eligible),
        })

    return {
        "scenario":      name,
        "incident_type": incident.type,
        "severity":      incident.severity,
        "results":       results,
        "candidates":    last_dec.candidates if last_dec else [],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Run AegisML demo scenarios headlessly."
    )
    parser.add_argument("--scenario", type=str, default="drift_benign",
                        help="Scenario name (default: drift_benign)")
    parser.add_argument("--seed",    type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--model",   type=str, default="lstm",
                        help="Active model (default: lstm)")
    parser.add_argument("--profile", type=str, default=None,
                        help="Weight profile override")
    parser.add_argument("--list",    action="store_true",
                        help="List all available scenarios")
    parser.add_argument("--all",     action="store_true",
                        help="Run all 8 scenarios")
    parser.add_argument("--quiet",   action="store_true",
                        help="Suppress verbose output")
    args = parser.parse_args(argv)

    if args.list:
        print("\nAvailable scenarios:")
        for name, cfg in SCENARIOS.items():
            print(f"  {name:<25} {cfg['description']}")
        return

    if args.all:
        summary = []
        for name in SCENARIOS:
            res = run_scenario(name, args.seed, args.profile, args.model,
                               verbose=not args.quiet)
            chosen = res["results"][0]["chosen_action"] if res.get("results") else "ERROR"
            summary.append(f"  {name:<25} {res['incident_type']:<22} → {chosen}")
        print("\n" + "=" * 70)
        print("ALL SCENARIOS SUMMARY")
        print("=" * 70)
        for line in summary:
            print(line)
        return

    run_scenario(
        args.scenario, args.seed, args.profile, args.model,
        verbose=not args.quiet
    )


if __name__ == "__main__":
    main()
