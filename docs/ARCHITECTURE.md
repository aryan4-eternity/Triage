# AegisML — Architecture (built on HarmonE)

**One line:** HarmonE decides *which model to run*. AegisML decides *what to do at all* — where switching models is one tactic among nine, chosen by a multi-objective utility with explicit eligibility guards and a safety constraint that sits outside the utility.

Read `docs/BASE_PROJECT_AUDIT.md` first. This document assumes you know what's in the base repo.

---

## 0. Three decisions that shape everything

**1. Local-first, and this time it's scientific, not convenience.**
The earlier AWS-heavy plan is dead. HarmonE's core asset is *physically measured CPU energy* via Intel RAPL, and RAPL is unavailable in Lambda, Fargate, and ordinary EC2. Porting to AWS would force you to replace real measurement with an estimate — trading your strongest claim for an architecture diagram. So: the energy-measured loop runs on a Linux/Intel box, and AWS is an **optional deployment arm** that carries an explicitly-labelled estimated-energy backend. If you have no Intel/Linux machine, see `docs/ENERGY.md`.

**2. Extend in place, don't rewrite.**
You are not starting a new repo. You are adding `aegis/` alongside `mape/`, adding a tenth approach (`aegis`) to `approach.conf`, and leaving all nine HarmonE approaches working untouched. This buys you six baselines for free and makes the ablation honest: same inference process, same energy meter, same data, only the Plan stage differs.

**3. Fairness becomes equity, because traffic data has no protected attributes.**
PEMS flow has no gender or age. Dropping the responsible-AI arm loses a pillar; faking one is worse. The honest substitute is **performance equity across sensor stations**: does the adaptation policy improve aggregate R² by sacrificing accuracy on a subset of stations? That's a real construct in traffic forecasting, measurable from data you already have, and it keeps the "safety is a constraint, not a weight" argument intact.

---

## 1. System context

```
┌──────────────────────────── UNCHANGED FROM HarmonE ────────────────────────┐
│  inference.py                                                              │
│    reads knowledge/model.csv + knowledge/serving.json  ◄── NEW: exec params│
│    runs LSTM | SVM | Ridge on a 5-step window                              │
│    measures energy per inference (EnergyMeter, was raw pyRAPL)             │
│    appends → knowledge/predictions.csv                                     │
└──────────────────────────────┬─────────────────────────────────────────────┘
                               │ file-mediated, append-only
                               ▼
┌──────────────────────────── AegisML CONTROL PLANE ─────────────────────────┐
│  aegis/manage.py   (approach = "aegis")                                    │
│                                                                            │
│   MONITOR ──► ANALYZE ──► PLAN ──► EXECUTE ──► LEARN                       │
│      │           │          │         │          │                         │
│  6 metric    dynamic    guards +   actuators   outcome                     │
│  families    boundaries  utility               ledger                      │
│      └───────────┴──────────┴─────────┴──────────┘                         │
│                  KNOWLEDGE (knowledge/*.json, *.csv, aegis.sqlite)         │
└──────────────────────────────┬─────────────────────────────────────────────┘
                               │
                               ▼
        knowledge/model.csv  ·  knowledge/serving.json  ·  retrain.py
        versionedMR/<model>/version_N/  ·  human review flag
                               │
                               ▼
              Streamlit dashboard (reads knowledge/, read-only)
```

Nothing above the dashed line changes structurally. You keep HarmonE's decoupling: the managed system never imports the managing system.

---

## 2. Six metric families

HarmonE monitors two. AegisML monitors six, which is what makes a *heterogeneous* action space meaningful.

| Family | Metric | Source | New? |
|---|---|---|---|
| Accuracy | R², MAE, EMA-smoothed score | `predictions.csv` | HarmonE |
| Energy | µJ/inference, normalised, EMA | `EnergyMeter` | HarmonE |
| Drift | KL divergence, PSI on the flow window | `predictions.csv` | HarmonE (KL) + PSI |
| Latency | p50/p95 `inference_time` | `predictions.csv` | **new** |
| Cost | wall-clock + energy → ₹/1k inferences | derived | **new** |
| Equity | worst-station R², max-min R² gap | `predictions.csv` + station id | **new** |

Two columns get added to `predictions.csv`: `station_id` and `energy_backend`. Both are needed for claims you'll want to make — per-station equity and honest labelling of measured-vs-estimated energy.

---

## 3. Action catalogue

HarmonE has three tactics (switch, replace-with-version, retrain). AegisML has nine. The new ones matter because they give the controller genuinely cheap options — without them, a multi-objective scorer has nothing interesting to choose between.

| Action | What it does | Actuation | Reversible | Risk |
|---|---|---|---|---|
| `OBSERVE` | nothing; schedule re-check | — | — | 0.00 |
| `SWITCH_MODEL(m)` | change active model | write `model.csv` | yes | 0.20 |
| `REUSE_VERSION(m,v)` | restore an archived version | copy artifact from `versionedMR/` | yes | 0.25 |
| `REDUCE_WINDOW` | shorten seq_length 5→3 | `serving.json` | yes | 0.30 |
| `BATCH_INFERENCE` | batch 1→8, amortise model load | `serving.json` | yes | 0.15 |
| `LOWER_SAMPLING` | log/score 1-in-k instead of all | `serving.json` | yes | 0.10 |
| `RETRAIN_CURRENT` | retrain active model on drift window | `retrain.py` | yes | 0.55 |
| `RETRAIN_ALL` | retrain all three | `retrain.py --all` | yes | 0.80 |
| `EQUITY_REVIEW` | flag disparity, block auto-promotion | `review.json` + dashboard | n/a | 0.15 |

`BATCH_INFERENCE` and `LOWER_SAMPLING` are the two that become powerful once BUG-3 is fixed — with a cached model, batching gives a large, *measurable* energy reduction at zero accuracy cost. That's your best sustainability result and it costs about 40 lines.

**Invariant:** when `equity_review_pending` is true, every action that changes the serving model (`SWITCH_MODEL`, `REUSE_VERSION`, retrains promoting a new artifact) is ineligible regardless of utility. Assert this in a test.

---

## 4. ANALYZE — boundaries

Keep HarmonE's energy integral controller. Generalise the pattern to the other families.

**Energy (kept verbatim from the base):**
```
thr ← thr + 0.95·(thr_original − energy_used)
```

**Everything else (new), rolling with hard clamps:**
```
boundary(m) = clamp(mean(m, W) ± k·std(m, W), hard_floor(m), hard_ceiling(m))
```
`W = 20` snapshots, `k = 2.0`. Cold start (<W samples) uses hard bounds and reports `mode="hard"`. Zero variance widens to `max(sd, 0.02·|mu|)`. An incident needs `persistence = 2` consecutive violations.

Severity: `clamp(|value − bound| / |hard_bound − bound|, 0, 1)`.

**Incident classification** (ordered, first match wins):
1. equity violated → `EQUITY_VIOLATION` *(safety dominates, always)*
2. accuracy violated **and** drift violated → `MODEL_DEGRADATION`
3. accuracy violated, drift fine → `ACCURACY_DROP` *(suspect pipeline/label bug — say so in the reason)*
4. drift violated, accuracy fine → `DRIFT_ONLY` *(the money case: retraining is not justified)*
5. energy violated → `ENERGY_PRESSURE`
6. latency violated → `LATENCY_PRESSURE`
7. ≥3 families → `COMPOSITE`

---

## 5. PLAN — the contribution

Three stages, in this order. Full math and worked examples in `docs/POLICY_ENGINE.md`.

**(a) Guards.** Declarative preconditions per action in `config/policy.json`. Failing candidates are dropped **with a printed reason** and still persisted. Examples:
```
REUSE_VERSION     : archived_versions >= 1 AND best_version_kl < 0.75
RETRAIN_CURRENT   : drift_window_rows >= 1200 AND cooldown_elapsed(RETRAIN, 600s)
SWITCH_MODEL(m)   : m != current AND ema[m] > ema[current] - 0.05
any promoting act : NOT equity_review_pending
```
HarmonE's `recovery_cycles` becomes one guard among many.

**(b) Effect vectors.** Each action declares its expected normalised effect per family, **calibrated offline on your hardware**, not guessed. This is what lets you compute projected benefit without a crystal ball.
```json
"SWITCH_MODEL:linear": { "accuracy": -0.12, "energy": -0.48, "latency": -0.40 },
"BATCH_INFERENCE":     { "accuracy":  0.00, "energy": -0.30, "latency": +0.10 }
```

**(c) Utility.**
```
U(a) = w_acc·Q(a) − w_e·E(a) − w_l·L(a) − w_c·C(a) − w_eq·F(a) − w_r·R(a)
```
with
```
Q(a) = Σ_families severity[f] · max(0, relief[a][f])
```
The `severity ·` multiplication is the whole trick: under `DRIFT_ONLY`, accuracy severity is 0, so `RETRAIN_CURRENT`'s big accuracy effect gets multiplied by zero and all it has left is its energy cost and risk. The controller declines to retrain — and can say exactly why.

ε-greedy exploration (HarmonE's `α = 0.1`) is **kept, but moved after the guard filter**, so exploration can never pick an ineligible or unsafe action. Log exploratory decisions with `exploratory: true` so they can be excluded from the evaluation.

Tie-break: lower risk, then lower energy, then action name. Deterministic.
Floor rule: if `max U < 0`, fall back to `OBSERVE`.

---

## 6. LEARN — outcome ledger

HarmonE never checks whether an adaptation worked. AegisML writes an `ActionOutcome` per decision and resolves it N snapshots later: did the violated family return inside its boundary, and at what actual measured energy cost?

Two uses:
1. A "was the controller right?" column on the dashboard — cheap, and it's the thing judges probe.
2. **Stretch:** a per-`(action, incident_type)` confidence `α ∈ [0.5, 1.5]`, nudged ±0.1 on resolve/fail, multiplying `Q(a)`. Thirty lines, behind `policy.learning_enabled`, and it earns you an honest "the controller learns from its own history" without any RL dependency.

---

## 7. Repo layout after extension

```
HarmonE-main/                    ← keep the name and the LICENSE
├── inference.py                 MODIFIED: model cache, serving.json, station_id, EnergyMeter
├── retrain.py                   MODIFIED: --model/--all flags, shared scaler
├── mape/                        UNTOUCHED — baselines must keep working
├── aegis/                       NEW — everything you contribute
│   ├── core/                    monitor.py analyze.py plan.py execute.py learn.py
│   │                            boundaries.py utility.py guards.py models.py
│   │                            metrics/{accuracy,drift,latency,energy,equity}.py
│   ├── energy/                  meter.py rapl.py codecarbon.py estimator.py
│   ├── actuators/               model_switch.py version_reuse.py serving_cfg.py
│   │                            retrain.py review.py
│   ├── store.py                 sqlite knowledge store
│   └── manage.py                the "aegis" approach entrypoint
├── dashboard/app.py             NEW — Streamlit
├── tools/
│   ├── synth_data.py            NEW — generator with injectable regime shifts
│   ├── profile_models.py        NEW — replaces stale load_test_models.py
│   └── calibrate_effects.py     NEW — writes measured effect vectors
├── config/policy.json           NEW — weights, effects, guards
├── config/boundaries.json       NEW
├── knowledge/serving.json       NEW — batch size, seq_length, sampling rate
├── tests/                       NEW
├── NOTICE.md                    NEW — attribution
└── LICENSE                      UNCHANGED (Hiya Bhatt, MIT)
```

**Hard rule:** `aegis/core/**` imports nothing from `mape/`, nothing from `torch`, nothing from `pyRAPL`. It's pure functions over the contracts in `docs/CONTRACTS.md`. Effects go through the actuator interfaces. This is what makes it testable without a Linux/Intel box and without the dataset.

---

## 8. Demo scenarios

Drive these from `tools/synth_data.py` so they're deterministic and reproducible.

| Scenario | Injected | Expected decision | The point |
|---|---|---|---|
| Normal | — | `OBSERVE` | baseline |
| Benign drift | distribution shift, R² holds | `OBSERVE` / `LOWER_SAMPLING` | **retrain rejected** — headline |
| Recoverable drift | shift matching an archived version | `REUSE_VERSION` | the tactic you *fixed* (BUG-1) |
| Model degradation | shift + R² collapse | `RETRAIN_CURRENT` | it retrains when justified |
| Energy pressure | sustained LSTM overspend | `SWITCH_MODEL:linear` / `BATCH_INFERENCE` | HarmonE's case, now with cheaper alternatives |
| Latency pressure | inflated inference time | `BATCH_INFERENCE` / `REDUCE_WINDOW` | dimension HarmonE can't see |
| Equity shift | one station's error diverges | `EQUITY_REVIEW` + promotion block | safety as constraint |
| Weight-profile toggle | same incident, `balanced` → `accuracy_first` | different action | the policy layer is load-bearing |

---

## 9. Evaluation — this is the paper

Run every scenario × 20 seeds against the baselines you already inherited.

**Arms:** `single-lstm`, `single-linear`, `single-svm`, `single-lstm+retrain`, `switch`, `switch+retrain`, `harmone`, `aegis`.

| Metric | Why |
|---|---|
| Mean R² over the run | did adaptation cost accuracy |
| Total measured energy (J), incl. controller overhead | the sustainability claim |
| Retraining jobs triggered | the "retrain isn't the default" claim |
| Energy spent on retraining vs inference | where the budget actually went |
| Mean time-to-recovery (snapshots) | responsiveness |
| Worst-station R² / max-min gap | the equity claim |
| Unnecessary interventions (acted, no violation persisted) | precision of the policy |
| Controller overhead (from `mape_log.csv`) | your loop must not cost more than it saves |

That last row is non-optional. HarmonE logs controller energy for exactly this reason, and a multi-objective planner is heavier than an argmax. If AegisML's overhead eats its savings, **report it** — a negative result with a clean measurement beats a positive one with a hidden confound, and reviewers will check.

Target claim shape: *"Against HarmonE on identical hardware and data, AegisML reduced retraining events by X% and total energy by Y% at ≤Z pp mean R², while reducing worst-station R² degradation by W pp."*

---

## 10. Non-goals — say these before you're asked

- Not a production system. A research prototype extending a published exemplar.
- Energy is **measured** where RAPL is available and **estimated** otherwise; every figure is labelled with its backend and mixed-backend runs are never averaged together.
- Equity here means per-station performance disparity, not demographic fairness. PEMS has no protected attributes and none are invented.
- No RL in the MVP. The transparent utility scorer is the contribution; outcome-confidence is a flagged extension.
- BUG-3's fix changes the base system's energy profile. Report before-and-after; don't quietly compare your fixed numbers against HarmonE's published ones.
