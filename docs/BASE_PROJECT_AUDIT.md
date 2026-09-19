# BASE_PROJECT_AUDIT.md — what HarmonE actually is

Read this before writing a line of AegisML code. Everything in the extension plan assumes you know what's already here, what's load-bearing, and what's broken.

---

## 0. Provenance — handle this correctly

The zip is **HarmonE**, `MIT License, Copyright (c) 2025 Hiya Bhatt`. It's a published research artifact, not code you wrote.

MIT lets you build on it commercially and privately. It requires that you **keep the license file and the copyright notice** in any substantial portion you redistribute.

What to do:
- Keep `LICENSE` in the repo, unmodified.
- Add your own `LICENSE-AEGISML` (MIT is fine) for your additions.
- Put a `NOTICE.md` at root: "AegisML extends HarmonE (Bhatt et al., MIT-licensed). The MAPE-K skeleton, inference loop, and energy instrumentation are HarmonE's. AegisML contributes the multi-objective action-selection layer, eligibility guards, cost/latency dimensions, equity monitoring, and the outcome-learning loop."
- In the paper/pitch, cite HarmonE as prior work you extend, explicitly. **Say it in the first 30 seconds to judges.** Extending a published exemplar is a strength; getting caught not disclosing it is fatal.

This also fixes a factual error from your earlier planning chat: the paper is **HarmonE**, not "Harmonica", and the arXiv ID quoted there (2601.11926) doesn't correspond to a real preprint — it was hallucinated. Get the real citation from the repo's paper link before you write anything with a bibliography.

---

## 1. What the system does

A self-adaptive MLOps loop for **univariate time-series regression** on PEMS traffic flow data (`Flow (Veh/5 Minutes)`), balancing R² against **physically measured CPU energy**.

Two processes, started separately:

```
Terminal 1:  python3 inference.py        # the managed system
Terminal 2:  python3 mape/manage.py      # the managing system
```

They communicate **only through files in `knowledge/`**. That's the whole interface. No sockets, no queues.

```
inference.py ──append──► knowledge/predictions.csv ──read──► mape/monitor.py
     ▲                                                              │
     │                                                              ▼
     └──read── knowledge/model.csv ◄──write── mape/execute.py ◄── plan ◄── analyse
```

### The three models
| Model | File | Trained by | Role |
|---|---|---|---|
| `lstm` | `models/lstm.pth` | PyTorch, 50 epochs, hidden 50 | accurate, expensive |
| `svm` | `models/svm.pkl` | `SVR(kernel="linear", C=0.05, tol=0.16)` | middle |
| `linear` | `models/linear.pkl` | `Ridge(alpha=256)` | cheap, less accurate |

Input is a 5-step rolling window (`seq_length = 5`), MinMax-scaled.

### The nine approaches
`approach.conf` + `set_approach.sh` select which threads `manage.py` starts:
- `harmone` → t1 (model switching) + t2 (drift handling)
- `switch` → t1 only
- `switch+retrain` → t1 + t3 (periodic retrain every 500s)
- `single-{lstm,svm,linear}` → no management threads
- `single-*+retrain` → t3 only

These are your **baselines for the evaluation table**. Don't rebuild them — you already have six.

---

## 2. The MAPE-K loop, stage by stage

### Monitor — `mape/monitor.py`

`monitor_mape()` reads new rows since `last_line`, then:

```
r2               = r2_score(true, pred)
energy_norm      = (mean(energy) - E_m) / (E_M - E_m)        # E_m=0, E_M=25000 µJ
model_score      = β·r2 + (1-β)·(1 - energy_norm)             # β = 0.95
ema[model]       = γ·model_score + (1-γ)·ema[model]           # γ = 0.8
```

`monitor_drift()` needs 2×1200 = 2400 samples, then KL divergence between the previous window and the current window of `true_value`, on 50-bin histograms.

### Analyse — `mape/analyse.py`

**This is where HarmonE's "dynamic boundary" lives**, and it's one line:

```python
new_energy_threshold = current_energy_threshold + 0.95 * (original_energy_threshold - used_energy)
```

An integral controller on an energy budget: underspend and your allowance grows, overspend and it shrinks. Plus `recovery_cycles = 3` after an energy violation — a cooldown that blocks switching.

Switch is needed if `ema_score < min_score (0.78)` **or** `energy_norm > current_energy_threshold`.

`analyse_drift()` fires at `KL > 0.75`, dumps the last 1200 rows to `drift.csv`, then `get_best_version()` compares that window's histogram against each archived version's training-data histogram and returns the lowest-KL one if it's under 0.75.

### Plan — `mape/plan.py`

```python
if random.random() < alpha:          # α = 0.1, ε-greedy exploration
    return random.choice(["lstm","linear","svm"])
if violation == "energy":            # pick best-EMA model that isn't the current one
elif violation == "score":           # pick argmax EMA
```

`plan_drift()` returns `{"action": "replace", "version": ...}` or `{"action": "retrain"}`.

**This is the stage you replace.** It's a single-objective argmax over three homogeneous options. Everything AegisML adds lives here.

### Execute — `mape/execute.py`
Writes a model name to `knowledge/model.csv`, or copies a version file, or shells out to `retrain.py`. Then `time.sleep(400)`.

### Knowledge — `knowledge/`
| File | Holds |
|---|---|
| `predictions.csv` | append-only inference log: true, pred, model, inference_time, energy |
| `mape_info.json` | `last_line`, `current_energy_threshold`, `ema_scores`, `recovery_cycles`, per-model versions |
| `thresholds.json` | `min_score`, `max_energy`, `β`, `γ`, `α`, `E_m`, `E_M` |
| `drift.csv` | the window that triggered drift; also the retraining set |
| `mape_log.csv` | energy cost of the controller itself |
| `model.csv` | one line: the active model |

---

## 3. Bugs and blockers, by severity

### BLOCKER-1 — `pyRAPL` won't run on your machine
`pyRAPL` reads Intel RAPL via `/sys/class/powercap/intel-rapl`. That means **Linux only, Intel only, root-ish permissions**. Not AMD, not macOS, not Windows, not WSL2 (no powercap passthrough), not Lambda, not Fargate, not standard EC2.

Both `inference.py` and `mape/manage.py` call `pyRAPL.setup()` at import — they die immediately without it.

**Fix (first task you do):** an `EnergyMeter` interface with three backends — `rapl` (real), `codecarbon` (RAPL/NVML with a fallback estimator), `model` (calibrated `time × TDP` proxy). Select by env var, record which backend produced every number, and **label it in the output**. Never present a proxy figure as measured.

### BLOCKER-2 — no dataset
PEMS data can't be redistributed, so `data/pems/` isn't in the zip. Without it, nothing runs: `train_models.py`, `inference.py` both read `data/pems/flow_data_*.csv`.

**Fix:** either register at Caltrans PeMS and export a station's 5-minute flow, or write a synthetic generator (daily + weekly seasonality + noise + injectable regime shifts). Do the generator regardless — you need deterministic, reproducible drift for the demo, and PeMS won't give you that on demand.

### BUG-1 — version reuse is silently broken
`analyse.get_best_version()` returns a path to **`data.csv`**, the version's training data. `execute.execute_drift()` treats it as a model path:

```python
model_name = os.path.basename(best_version_path).split(".")[0]   # -> "data"
model_extension = ".pth"                                          # "data" not in [linear, svm]
shutil.copy(best_version_path, "models/data.pth")                 # copies a CSV to a .pth
```

So the "reuse an archived model version" tactic — a headline claim of the base system — copies a CSV to a filename nothing ever loads. The active model is untouched.

**This is your single best opening contribution.** Fix it (return the version *directory*, resolve the model artifact by name and extension, copy that), then show a before/after: the tactic that was a no-op now actually recovers accuracy without retraining. That's a measurable result on day one.

### BUG-2 — column-name mismatch, works by accident
`inference.py` writes the header as `energy_uJ`; `monitor.py` reads `df["energy"]`. It only works because `knowledge/predictions.csv` ships pre-created with an `energy` header and `inference.py` skips header creation when the file exists. Delete that file once and the whole loop `KeyError`s.

**Fix:** one schema constant, imported by both.

### BUG-3 — the model is reloaded from disk on every single prediction
```python
for i in range(len(X_stream)):
    lstm_model = LSTMModel()
    lstm_model.load_state_dict(torch.load("models/lstm.pth"))
```
Every measured "inference energy" is dominated by object construction and disk I/O, not inference. This inflates the LSTM/linear energy gap in an uncontrolled way and is the kind of thing a reviewer finds in thirty seconds.

**Fix:** cache the loaded model, invalidate on `model.csv` mtime change. Then re-measure the energy profile of all three models — **your paper's headline numbers depend on this**. Expect the gap to shrink. Report both, honestly; "we fixed a measurement confound in the base exemplar and here's the corrected profile" is a contribution, not an embarrassment.

### BUG-4 — scaler leakage and inconsistency
`inference.py` fits a fresh `MinMaxScaler` on the entire test stream (future data leaking into the present), and it's a *different* scaler from the one `train_models.py` fit on training data. `retrain.py` fits a third on the drift window.

**Fix:** fit once on training data, persist to `artifacts/scaler.pkl`, load everywhere. Keeps predictions comparable across retrains.

### BUG-5 — `E_M = 25000 µJ` is a hardcoded, unvalidated normaliser
`energy_normalized` can go negative or above 1 depending on your hardware. Every downstream threshold inherits that.

**Fix:** calibrate `E_m`/`E_M` from a measured profiling run on your machine, write them into `thresholds.json` with provenance, and clamp to [0,1].

### BUG-6 — `time.sleep(400)` inside `execute_drift()`
Blocking sleep inside the executor rather than the scheduler. Makes the drift thread's cadence unreadable and untestable.

**Fix:** cooldowns belong in the guard layer, keyed per action, checked against timestamps.

### SMELL-1 — `tools/load_test_models.py` is stale
References `models/lstm_model.pth`, `models/lr_model.pkl`, `data/pems/flow_data_cleaned.csv` — none exist. Also `split_idx = int(len(data) * 0)`. Don't fix; delete or rewrite as your profiling script.

### SMELL-2 — imports depend on CWD
`mape/analyse.py` does `from monitor import ...`, which works only because Python puts the script's directory on `sys.path`. Meanwhile all data paths are relative to the repo root. Run it from the wrong directory and it half-works.

**Fix:** make it a package, use relative imports, resolve paths from a `ROOT` constant.

### SMELL-3 — zero tests
No test directory, no assertions. Every change you make is unverified.

---

## 4. What to keep (don't rewrite these)

- **The two-process, file-mediated architecture.** It's genuinely good for this: the managed system and managing system are decoupled and separately measurable. Keep it.
- **The nine approach configurations.** Free baselines.
- **The energy-integral threshold.** `thr += 0.95·(orig − used)` is the base system's adaptivity. Keep it as *one* boundary among several.
- **EMA scoring with γ.** Sensible smoothing. Keep it, generalise it to more metrics.
- **ε-greedy exploration.** Keep it, but move it *after* the guard layer so it can never explore into an ineligible or unsafe action.
- **Versioned model repository.** The idea is right; only the lookup is broken.
- **`pyRAPL` measurement.** Real physical measurement is a genuine strength over anything you'd get on AWS. Keep it as the primary backend.

---

## 5. What you're replacing

| HarmonE | AegisML |
|---|---|
| Plan = argmax over 3 models | Plan = argmax utility over ~9 heterogeneous tactics |
| Two metrics (R², energy) | Six families (accuracy, drift, latency, energy, cost, equity) |
| One scalar score, β-weighted | Per-family severity + projected effect vectors + weight profiles |
| Implicit eligibility | Declarative guards with printed rejection reasons |
| `recovery_cycles` as the only cooldown | Per-action cooldowns and budget guards |
| No record of why | Every decision persists its full candidate table |
| No outcome tracking | Outcome ledger; did the violation actually resolve? |
| Switching is the only cheap option | Switching, version reuse, batching, window reduction, observe |

**One-sentence delta:** *HarmonE decides **which model** to run; AegisML decides **what to do at all**, of which changing model is one option among many.*
