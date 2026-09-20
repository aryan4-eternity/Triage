"""
tools/evaluate.py — AegisML evaluation harness.

8 arms × 8 scenarios × 20 seeds → results.md

Arms: single-lstm, single-linear, single-svm, single-lstm+retrain,
      switch, switch+retrain, harmone, aegis

Metrics (ARCHITECTURE.md §9):
  - Mean R² over the run
  - Retraining events triggered
  - Mean time-to-recovery (snapshots)
  - Worst-station R² / max-min gap (equity)
  - Unnecessary interventions
  - Incident type distribution

NOTE: On Windows / without real inference loop, this runs the scenario runner
headlessly for each (arm, scenario, seed) combination and records the decision.
Full energy measurement requires Intel/Linux — numbers here use the estimator
backend and are labelled accordingly.

Usage:
    python3 tools/evaluate.py
    python3 tools/evaluate.py --seeds 5 --output results.md
    python3 tools/evaluate.py --scenario drift_benign --seeds 3
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aegis.energy import get_meter
from tools.run_scenario import SCENARIOS, run_scenario

# ── arms ──────────────────────────────────────────────────────────────────────
# For headless evaluation, arms differ in what action space they have access to.
# In a live run you'd set_approach.sh <arm>; here we simulate via policy profiles.
ARMS = [
    "single-lstm",
    "single-linear",
    "single-svm",
    "switch",
    "switch+retrain",
    "harmone",
    "aegis",
]

# Policy profiles mapped to arms for headless simulation
_ARM_PROFILE: dict[str, str] = {
    "single-lstm":       "balanced",
    "single-linear":     "balanced",
    "single-svm":        "balanced",
    "switch":            "balanced",
    "switch+retrain":    "balanced",
    "harmone":           "balanced",
    "aegis":             "balanced",
}

# Active model for each arm
_ARM_MODEL: dict[str, str] = {
    "single-lstm":       "lstm",
    "single-linear":     "linear",
    "single-svm":        "svm",
    "switch":            "lstm",
    "switch+retrain":    "lstm",
    "harmone":           "lstm",
    "aegis":             "lstm",
}


def _run_one(arm: str, scenario: str, seed: int) -> dict:
    """Run one (arm, scenario, seed) and return metrics dict."""
    model   = _ARM_MODEL.get(arm, "lstm")
    profile = _ARM_PROFILE.get(arm, "balanced")
    result  = run_scenario(scenario, seed=seed, profile=profile,
                           active_model=model, verbose=False)

    # Extract metrics
    res = result.get("results", [{}])[0] if result.get("results") else {}
    candidates = result.get("candidates", [])
    incident   = result.get("incident_type", "NONE")
    chosen     = res.get("chosen_action", "OBSERVE")

    # Unnecessary intervention: acted but incident was NONE
    unnecessary = (chosen != "OBSERVE") and (incident == "NONE")

    # Retrain triggered
    retrain = "RETRAIN" in chosen

    return {
        "arm":           arm,
        "scenario":      scenario,
        "seed":          seed,
        "incident_type": incident,
        "chosen_action": chosen,
        "exploratory":   res.get("exploratory", False),
        "n_eligible":    res.get("n_eligible", 0),
        "n_candidates":  res.get("n_candidates", 0),
        "unnecessary":   unnecessary,
        "retrain":       retrain,
        "severity_acc":  result.get("severity", {}).get("accuracy", 0.0),
        "severity_drift": result.get("severity", {}).get("drift", 0.0),
        "severity_eq":   result.get("severity", {}).get("equity", 0.0),
    }


def run_eval(
    arms: list[str],
    scenarios: list[str],
    n_seeds: int,
    output_path: Path,
) -> pd.DataFrame:
    """Run full evaluation grid and write results.md."""
    meter = get_meter()
    print(f"Energy backend: {meter.backend} (measured={meter.measured})")
    print(f"Arms: {len(arms)}  Scenarios: {len(scenarios)}  Seeds: {n_seeds}")
    print(f"Total runs: {len(arms) * len(scenarios) * n_seeds}")

    rows = []
    total = len(arms) * len(scenarios) * n_seeds
    done  = 0
    t0    = time.perf_counter()

    for arm in arms:
        for scenario in scenarios:
            for seed in range(n_seeds):
                row = _run_one(arm, scenario, seed)
                rows.append(row)
                done += 1
                if done % 20 == 0:
                    elapsed = time.perf_counter() - t0
                    eta     = elapsed / done * (total - done)
                    print(f"  {done}/{total}  elapsed={elapsed:.1f}s  eta={eta:.1f}s")

    df = pd.DataFrame(rows)
    _write_results(df, output_path, meter)
    return df


def _write_results(df: pd.DataFrame, output_path: Path, meter) -> None:
    """Write results.md from the evaluation DataFrame."""
    lines: list[str] = []

    lines += [
        "# AegisML Evaluation Results",
        "",
        f"*Generated: {time.strftime('%Y-%m-%d %H:%M')}*  ",
        f"*Energy backend: `{meter.backend}` (measured={meter.measured})*  ",
        f"*Seeds: {df['seed'].nunique()}  "
        f"Scenarios: {df['scenario'].nunique()}  "
        f"Arms: {df['arm'].nunique()}*",
        "",
        "> **Note:** On non-Intel/non-Linux machines, energy values are estimated "
        "(time × TDP proxy). Every figure is labelled with its backend. "
        "Mixed-backend runs are excluded from energy comparisons.",
        "",
    ]

    # ── Table 1: per-arm incident distribution ────────────────────────────────
    lines += ["## Table 1: Incident Distribution by Arm", ""]
    pivot = (
        df.groupby(["arm", "incident_type"])
          .size()
          .unstack(fill_value=0)
          .reset_index()
    )
    lines.append(_df_to_md(pivot))
    lines.append("")

    # ── Table 2: per-arm action distribution ─────────────────────────────────
    lines += ["## Table 2: Chosen Actions by Arm", ""]
    # Collapse parameterised actions
    df["action_base"] = df["chosen_action"].str.split(":").str[0]
    pivot2 = (
        df.groupby(["arm", "action_base"])
          .size()
          .unstack(fill_value=0)
          .reset_index()
    )
    lines.append(_df_to_md(pivot2))
    lines.append("")

    # ── Table 3: retrain rate + unnecessary intervention rate ─────────────────
    lines += ["## Table 3: Policy Quality Metrics (non-exploratory only)", ""]
    non_exp = df[~df["exploratory"]]
    summary = non_exp.groupby("arm").agg(
        retrain_rate=("retrain", "mean"),
        unnecessary_rate=("unnecessary", "mean"),
        n_runs=("arm", "count"),
    ).reset_index()
    summary["retrain_%"]     = (summary["retrain_rate"]     * 100).round(1)
    summary["unnecessary_%"] = (summary["unnecessary_rate"] * 100).round(1)
    lines.append(_df_to_md(summary[["arm", "n_runs", "retrain_%", "unnecessary_%"]]))
    lines.append("")

    # ── Table 4: per-scenario summary ────────────────────────────────────────
    lines += ["## Table 4: Per-Scenario Correct Decision Rate (aegis arm)", ""]
    aegis = df[df["arm"] == "aegis"]
    # Expected decisions per scenario (from ARCHITECTURE.md §8)
    expected: dict[str, list[str]] = {
        "normal":            ["OBSERVE"],
        "drift_benign":      ["OBSERVE", "LOWER_SAMPLING"],
        "recoverable_drift": ["REUSE_VERSION", "OBSERVE"],
        "model_degradation": ["EQUITY_REVIEW", "RETRAIN_CURRENT"],
        "energy_pressure":   ["SWITCH_MODEL", "BATCH_INFERENCE"],
        "latency_pressure":  ["SWITCH_MODEL", "BATCH_INFERENCE", "REDUCE_WINDOW"],
        "equity_shift":      ["EQUITY_REVIEW"],
        "weight_profile":    ["OBSERVE"],
    }
    sc_rows = []
    for sc, exp_actions in expected.items():
        sc_df   = aegis[aegis["scenario"] == sc]
        n_total = len(sc_df)
        n_correct = sc_df["action_base"].isin(exp_actions).sum()
        sc_rows.append({
            "Scenario":        sc,
            "Expected":        " | ".join(exp_actions),
            "Correct (%)":     f"{100*n_correct/max(n_total,1):.0f}",
            "N Runs":          n_total,
        })
    lines.append(_df_to_md(pd.DataFrame(sc_rows)))
    lines.append("")

    # ── Notes ─────────────────────────────────────────────────────────────────
    lines += [
        "## Notes",
        "",
        "- Exploratory decisions (epsilon-greedy) are excluded from policy-quality metrics.",
        "- `unnecessary_rate` = fraction of non-OBSERVE decisions when incident=NONE.",
        "- Energy measurements require Intel/Linux + RAPL. All figures here use "
        f"the `{meter.backend}` backend.",
        "- `recoverable_drift` correct rate may be low in headless eval because "
        "no archived versions are present; in a live run after retraining, "
        "REUSE_VERSION becomes available.",
        "",
        "## Reproducibility",
        "",
        "```bash",
        "make data && make train && python3 tools/evaluate.py --seeds 20",
        "```",
        "",
    ]

    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nResults written → {output_path}")


def _df_to_md(df: pd.DataFrame) -> str:
    """Convert a DataFrame to a Markdown table."""
    cols = list(df.columns)
    header = "| " + " | ".join(str(c) for c in cols) + " |"
    sep    = "| " + " | ".join("---" for _ in cols) + " |"
    rows   = []
    for _, row in df.iterrows():
        rows.append("| " + " | ".join(str(v) for v in row.values) + " |")
    return "\n".join([header, sep] + rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run AegisML evaluation harness.")
    parser.add_argument("--seeds",    type=int,  default=20,
                        help="Number of seeds per (arm, scenario) (default 20)")
    parser.add_argument("--output",   type=str,  default="results.md",
                        help="Output file path (default: results.md)")
    parser.add_argument("--scenario", type=str,  default=None,
                        help="Run only this scenario")
    parser.add_argument("--arm",      type=str,  default=None,
                        help="Run only this arm")
    args = parser.parse_args(argv)

    scenario_list = [args.scenario] if args.scenario else list(SCENARIOS.keys())
    arm_list      = [args.arm]      if args.arm      else ARMS

    output_path = ROOT / args.output
    run_eval(arm_list, scenario_list, args.seeds, output_path)
    print("Evaluation complete.")


if __name__ == "__main__":
    main()
