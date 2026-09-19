# AegisML — extending HarmonE

**HarmonE decides *which model* to run.
AegisML decides *what to do at all* — where switching models is one tactic
among nine, chosen by a multi-objective utility with explicit eligibility
guards and a safety constraint that sits outside the utility.**

> **Attribution:** This repo extends **HarmonE** (MIT, © 2025 Hiya Bhatt),
> a self-adaptive MLOps loop for traffic-flow regression. The MAPE-K skeleton,
> inference loop, and energy instrumentation are HarmonE's. See `NOTICE.md`.

---

## What's new in AegisML

| Dimension | HarmonE | AegisML |
|---|---|---|
| Actions | Switch between 3 models | 9 heterogeneous tactics |
| Metrics | R², energy | Accuracy, drift, latency, energy, cost, equity |
| Decision | argmax EMA score | Multi-objective utility + guards |
| Eligibility | `recovery_cycles` cooldown | Declarative guards per action |
| Reasoning | None persisted | Full candidate table (incl. rejected) |
| Outcome tracking | None | Outcome ledger; did the fix work? |
| Safety | Not modelled | `equity_review_pending` hard blocks model promotion |

---

## Quick start

```bash
# 1 — install deps
make setup

# 2 — generate synthetic data
make data

# 3 — train models
make train

# 4 — run HarmonE baselines (verify nothing is broken)
make base

# 5 — run AegisML
make demo

# 6 — run a specific scenario
make scenario S=drift_benign
```

On Windows/macOS (no Intel RAPL): set `AEGIS_ENERGY=estimator` (auto-selected).
On Linux/Intel: `AEGIS_ENERGY=rapl` for physically measured energy.

---

## Energy backends

| Backend | `measured` | Requires |
|---|---|---|
| `rapl` | ✅ True | Linux + Intel CPU + powercap perms |
| `codecarbon` | ❌ False | `pip install codecarbon` |
| `estimator` | ❌ False | Nothing — always available |

Select with `AEGIS_ENERGY=rapl|codecarbon|estimator|auto` (default: `auto`).
Every row in `predictions.csv` carries `energy_backend`.
Mixed-backend runs are refused by MONITOR.

---

## Repository layout

```
HarmonE-main/
├── inference.py          MODIFIED: model cache, serving.json, station_id, EnergyMeter
├── retrain.py            MODIFIED: --model flag, shared scaler
├── mape/                 UNCHANGED (9 HarmonE approaches, all still runnable)
├── aegis/                NEW — AegisML control plane
│   ├── core/             monitor, analyze, plan, execute, learn, boundaries, utility, guards
│   │   └── metrics/      accuracy, drift, latency, energy, equity
│   ├── energy/           portable energy backend abstraction
│   ├── actuators/        file-based actuators (model_switch, version_reuse, serving_cfg, …)
│   └── store.py          SQLite knowledge store
├── dashboard/app.py      NEW — Streamlit dashboard with candidate table
├── tools/
│   ├── synth_data.py     NEW — deterministic synthetic data generator
│   ├── profile_models.py NEW — warm-cache energy/latency profiling
│   └── calibrate_effects.py  NEW — measures effect vectors
├── config/               policy.json, boundaries.json, hardware.json
├── tests/                pytest test suite
├── docs/                 design documentation
├── NOTICE.md             attribution (read this)
└── LICENSE               MIT, © 2025 Hiya Bhatt (unchanged)
```

---

## Nine approaches (all runnable)

```bash
./set_approach.sh harmone          # original HarmonE
./set_approach.sh switch
./set_approach.sh switch+retrain
./set_approach.sh single-lstm
./set_approach.sh single-svm
./set_approach.sh single-linear
./set_approach.sh single-lstm+retrain
./set_approach.sh single-svm+retrain
./set_approach.sh single-linear+retrain
./set_approach.sh aegis            # AegisML (new)
```

---

## Key design decisions

**Local-first, scientifically required.** HarmonE's core asset is physically
measured CPU energy via Intel RAPL, which is unavailable in cloud functions.
AegisML keeps the local architecture and adds a portable estimator fallback.

**Safety as a constraint, not a weight.** When `equity_review_pending` is true,
no utility score — however high — can promote a new model. The guard is a hard
filter, not a large negative weight.

**Rejected actions are the paper figure.** Every decision persists the full
candidate table including rejected actions with their guard reasons and utility
terms. That's what distinguishes this from a pipeline.

---

## Running experiments

```bash
make eval          # 8 arms × 8 scenarios × 20 seeds → results.md
make scenario S=equity_shift
```

See `docs/ARCHITECTURE.md` §9 for the full evaluation table specification.

---

## Bugs fixed in the base (Phase 1)

| Bug | Effect | Fixed in |
|---|---|---|
| BUG-1: version reuse no-op | Archived model was never actually restored | T1.1 |
| BUG-3: model reloads every inference | Energy measurements dominated by disk I/O | T1.2 |
| BUG-2: column name mismatch | `energy_uJ` vs `energy` — worked only by accident | T1.4 |
| BUG-4: scaler leakage | Three different scalers fit on different data | T1.4 |
| BUG-5: E_M=25000 hardcoded | energy_normalized could go negative or >1 | T1.3 |

Each fix is measured with before/after results. See `results.md`.
