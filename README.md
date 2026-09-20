# Triage — Adaptive MLOps Controller

**Triage decides *what to do at all* — where switching models is one tactic among nine, chosen by a multi-objective utility with explicit eligibility guards and a safety constraint that sits outside the utility.**

---

## What Triage does

Triage is a self-adaptive MLOps controller for traffic-flow regression. It monitors six metric families in real time and selects the best action from nine heterogeneous tactics — not just switching models, but batching, version reuse, rate reduction, equity review, and more.

| Dimension | Capability |
|---|---|
| Action space | 9 tactics (observe, switch model, batch, reuse version, retrain, equity review, …) |
| Metrics monitored | Accuracy, drift, latency, energy, cost, equity |
| Decision method | Multi-objective utility + declarative eligibility guards |
| Reasoning trail | Full candidate table — every action, eligible or rejected, with reasons |
| Outcome tracking | Outcome ledger — did the intervention actually work? |
| Safety | `equity_review_pending` hard-blocks model promotion regardless of utility |

---

## Quick start

```bash
# 1 — install deps
make setup

# 2 — generate synthetic data
make data

# 3 — train models
make train

# 4 — run baselines
make base

# 5 — run Triage controller
make demo

# 6 — run a scenario headlessly
make scenario S=drift_benign

# 7 — launch dashboard
make dashboard
```

On Windows/macOS: `AEGIS_ENERGY=estimator` is auto-selected (no RAPL needed).  
On Linux/Intel: `AEGIS_ENERGY=rapl` for physically measured energy.

---

## Nine approaches

```bash
./set_approach.sh triage              # Triage multi-objective controller
./set_approach.sh switch
./set_approach.sh switch+retrain
./set_approach.sh single-lstm
./set_approach.sh single-svm
./set_approach.sh single-linear
./set_approach.sh single-lstm+retrain
./set_approach.sh single-svm+retrain
./set_approach.sh single-linear+retrain
./set_approach.sh aegis               # AegisML (same as triage)
```

---

## Energy backends

| Backend | Measured | Requires |
|---|---|---|
| `rapl` | ✅ True | Linux + Intel CPU + powercap perms |
| `codecarbon` | ❌ Estimated | `pip install codecarbon` |
| `estimator` | ❌ Estimated | Nothing — always available |

Select with `AEGIS_ENERGY=rapl|codecarbon|estimator|auto` (default: `auto`).  
Every row in `predictions.csv` carries `energy_backend`. Mixed-backend runs are refused.

---

## Repository layout

```
Triage/
├── inference.py          inference loop with model cache + EnergyMeter
├── retrain.py            model retraining on drift windows
├── mape/                 MAPE-K loop (9 baseline approaches)
├── aegis/                Triage control plane
│   ├── core/             monitor, analyze, plan, execute, learn
│   │   └── metrics/      accuracy, drift, latency, energy, equity
│   ├── energy/           portable energy backend abstraction
│   ├── actuators/        file-based actuators
│   └── store.py          SQLite knowledge store
├── dashboard/app.py      Streamlit dashboard with candidate table
├── tools/
│   ├── synth_data.py     deterministic synthetic data generator
│   ├── profile_models.py warm-cache energy/latency profiler
│   ├── run_scenario.py   headless scenario runner (8 scenarios)
│   ├── calibrate_effects.py  effect vector calibration
│   └── evaluate.py       evaluation harness (N arms × 8 scenarios × N seeds)
├── config/               policy.json, boundaries.json, hardware.json
├── tests/                32-test pytest suite
├── docs/                 design documentation
└── results.md            evaluation results
```

---

## Demo scenarios

```bash
python3 tools/run_scenario.py --list          # show all 8 scenarios
python3 tools/run_scenario.py --scenario drift_benign
python3 tools/run_scenario.py --all           # run all 8, print summary
```

| Scenario | What is injected | Expected decision |
|---|---|---|
| normal | nothing | OBSERVE |
| drift_benign | distribution shift, accuracy holds | OBSERVE (retrain not justified) |
| recoverable_drift | shift matching archived version | REUSE_VERSION |
| model_degradation | shift + R² collapse | RETRAIN_CURRENT |
| energy_pressure | sustained energy overspend | SWITCH_MODEL / BATCH_INFERENCE |
| latency_pressure | inflated inference time | SWITCH_MODEL / BATCH_INFERENCE |
| equity_shift | one station's accuracy diverges | EQUITY_REVIEW |
| weight_profile | same incident, two profiles | different actions |

---

## The candidate table

Every decision persists the full candidate table — eligible actions with utility terms AND rejected actions with guard reasons. This is what distinguishes Triage from a pipeline:

```
Action                    Eligible  Utility     Q      E      R   Reason / Guard
OBSERVE                   YES       +0.000   0.000  0.000  0.000  no violation
SWITCH_MODEL:linear       YES       -0.126   0.000  0.000  0.200  ...
RETRAIN_CURRENT           NO        —        —      —      —      GUARD: drift_window_rows 0 < 1200
REUSE_VERSION             NO        —        —      —      —      GUARD: archived_versions 0 < 1
```

---

## Safety invariant

When `equity_review_pending` is true, no utility score — however high — can promote a new model. The guard is a hard filter, not a large negative weight.

```python
# tests/test_plan.py enforces this:
def test_equity_violation_blocks_all_promoting_actions(): ...
```

---

## Evaluation

```bash
make eval           # 7 arms × 8 scenarios × 20 seeds → results.md
make eval-quick     # 3 seeds for fast iteration
```

---

## Make targets

```
make setup          create venv and install deps
make data           generate synthetic data
make train          train models
make profile        profile energy/latency
make calibrate      calibrate effect vectors from profile
make test           run 32-test pytest suite
make base           verify all baselines still work
make scenario S=X   run named scenario headlessly
make demo           start inference + Triage controller
make dashboard      start Streamlit dashboard
make eval           full evaluation
make clean          reset knowledge files
```

---

## References

- PeMS Traffic Data: California Department of Transportation, [pems.dot.ca.gov](https://pems.dot.ca.gov)
- Sculley, D. et al. (2015). *Hidden Technical Debt in Machine Learning Systems*. NeurIPS.
- Paleyes, A. et al. (2022). *Challenges in Deploying Machine Learning: a Survey of Case Studies*. ACM Computing Surveys.
