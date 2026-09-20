# EXTENSION_PLAN.md — how to build AegisML on top of Triage

File-by-file guidance. Pair this with `docs/TASKBOARD.md`, which sequences it.

The governing principle: **every Triage approach must still run, unchanged, at every point in the build.** If `./set_approach.sh Triage && python3 mape/manage.py` stops working, you've broken your own baselines and your evaluation table dies with them.

---

## Phase 0 — make the base run at all

You cannot extend what you can't execute. Three blockers, in order.

### 0.1 Energy abstraction

`pyRAPL.setup()` runs at import in both `inference.py` and `mape/manage.py`, and fails hard off Intel/Linux. Introduce `aegis/energy/meter.py`:

```python
class EnergyMeter(Protocol):
    backend: str                      # "rapl" | "codecarbon" | "estimator"
    measured: bool                    # True only for rapl
    def measure(self, label: str) -> ContextManager[Reading]: ...

# Reading: {"micro_joules": float, "seconds": float, "backend": str, "measured": bool}
```

Three implementations:
- `rapl.py` — wraps `pyRAPL`. Probe `/sys/class/powercap/intel-rapl` at construction; raise `EnergyBackendUnavailable` if absent.
- `codecarbon.py` — `codecarbon`'s tracker; reports RAPL where available, otherwise its own hardware-model estimate. Set `measured=False` when it falls back.
- `estimator.py` — `µJ ≈ elapsed_seconds × TDP_watts × utilisation × 1e6`, with `TDP_watts` in `config/hardware.json`. Always `measured=False`.

Selection: `AEGIS_ENERGY=rapl|codecarbon|estimator|auto` (default `auto` → first that constructs).

Then replace the two `pyRAPL` call sites. `inference.py`'s becomes:
```python
with meter.measure("inference") as r:
    prediction = model.predict(...)
energy_uj, backend = r.micro_joules, r.backend
```

**Write `backend` into `predictions.csv` as a column.** Every downstream number inherits it, and a run that mixes backends is not a valid run. Add an assertion in MONITOR: if a window contains more than one backend, refuse to score it.

### 0.2 Data

`tools/synth_data.py`, generating flow-like series:

```
flow(t) = base
        + A_day   · sin(2π t / 288)          # 288 5-min slots per day
        + A_week  · sin(2π t / 2016)
        + regime(t)                           # injectable step / scale / trend
        + ε,  ε ~ N(0, σ)
```
CLI: `--rows --seed --station-count --regime start:end:scale:shift`. Emit `flow_data_train.csv` (10%) and `flow_data_test.csv` (90%) with the same column names Triage expects, plus `station_id`.

Do this **even if you get real PeMS data**. Demos need deterministic drift at a known row index; PeMS won't give you that.

If you do get PeMS: run `tools/store_pems.py --train_ratio 0.1` as documented, then `tools/induce_drift.py` for drift. Keep both paths; report on real data, demo on synthetic, and say which is which.

### 0.3 Verify the base end to end

```bash
./cleanup.sh && python3 tools/train_models.py
./set_approach.sh Triage
python3 inference.py            # terminal 1
python3 mape/manage.py          # terminal 2
```
Let it run past 400s so the drift thread starts. Confirm: `predictions.csv` grows, `mape_info.json` EMA scores move, at least one switch appears in `model.csv`.

**Do not proceed until this works.** Everything downstream compares against it.

---

## Phase 1 — fix the base, measure the difference

These are contributions, not chores. Each one is a row in your results table.

### 1.1 BUG-1: repair version reuse

In `analyse.get_best_version()`, return the **version directory**, not `data.csv`:
```python
return {"dir": os.path.dirname(version_data_path), "kl": min_kl_div, "model": model_name}
```
In `execute.execute_drift()`, resolve the artifact properly:
```python
ext = ".pth" if model_name == "lstm" else ".pkl"
src = os.path.join(best["dir"], f"{model_name}{ext}")
shutil.copy(src, os.path.join("models", f"{model_name}{ext}"))
```
Add `tests/test_version_reuse.py`: build a fake `versionedMR/` tree, run the path, assert the *model* file was replaced and `models/data.pth` was never created.

**Measure it.** Run `Triage` on the recoverable-drift scenario before and after. Before: the tactic is a no-op and the system eventually retrains. After: accuracy recovers without a retrain. That delta is a clean, standalone result — get it on day one and you have something to show no matter what else happens.

### 1.2 BUG-3: cache the model

In `inference.py`, hold `(name, mtime, obj)`; reload only when `model.csv`'s mtime or contents change.

Then **re-profile all three models** with `tools/profile_models.py` (1000 inferences each, warm cache, report µJ/inference, ms/inference, R²). Write the results into `config/hardware.json` with the date and machine.

Expect the LSTM-vs-linear energy gap to narrow substantially, because you just removed `torch.load` from the measurement. That's uncomfortable and it's the right thing to report: *"the base exemplar's per-inference energy included model deserialisation; with a warm cache the measured gap is X instead of Y, and we use the corrected profile throughout."* Never compare your corrected numbers to Triage's published ones as if they were the same measurement.

### 1.3 BUG-2, BUG-4, BUG-5, SMELL-2

- One `SCHEMA` constant in `aegis/core/models.py`; `inference.py` and both monitors import it. Column is `energy_uj`. Migrate the shipped CSVs.
- Fit `MinMaxScaler` once in `train_models.py`, persist to `artifacts/scaler.pkl`, load in `inference.py` and `retrain.py`. Never refit on test data.
- Calibrate `E_m`/`E_M` from `profile_models.py` output; clamp `energy_normalized` to [0,1]; record provenance in `thresholds.json`.
- Make `mape/` and `aegis/` packages with `__init__.py`; resolve paths from a `ROOT = Path(__file__).resolve().parents[1]` constant.

`mape/` changes here must be behaviour-preserving. Re-run `Triage` and confirm the decision sequence on a fixed seed matches pre-change.

---

## Phase 2 — the AegisML control plane

### 2.1 Contracts and store

`aegis/core/models.py` — pydantic models from `docs/CONTRACTS.md`.
`aegis/store.py` — SQLite: `snapshots`, `incidents`, `decisions`, `candidates`, `outcomes`. SQLite over CSV because you need to query the candidate table for the dashboard and the eval harness, and because concurrent append from two processes to one CSV is a race waiting to happen. `knowledge/` JSON files stay as they are so Triage keeps working.

### 2.2 Metric extractors

`aegis/core/metrics/` — one module per family, each a pure function over a DataFrame window:
- `accuracy.py` — R², MAE, EMA (reuse Triage's β/γ formula verbatim; cite it in the docstring)
- `drift.py` — KL (Triage's) **and** PSI, so you can report both
- `latency.py` — p50/p95 from `inference_time`
- `energy.py` — normalised energy with the clamped E_m/E_M
- `equity.py` — group by `station_id`, per-station R², return worst and max-min gap

Equity needs enough rows per station. With 8 stations and a 1200-row window that's 150 each — thin for R². Either widen the equity window (use a separate, longer one) or use per-station MAE, which is stabler at small n. Make the choice explicitly and note it in the docstring.

### 2.3 ANALYZE

`boundaries.py` for the rolling-clamped boundary; keep Triage's energy integral as a special case with `mode="integral"`, calling into the same code path so the dashboard renders all boundaries uniformly. Then `analyze.py` for the 7 ordered classification rules.

### 2.4 PLAN

`guards.py`, `utility.py`, `plan.py`. Every test in `docs/POLICY_ENGINE.md` §7 must pass. The ε-greedy branch moves *after* guard filtering.

### 2.5 Actuators

`aegis/actuators/` implementing the `Actuator` protocol. Three of them just write files Triage already understands:
- `model_switch.py` → writes `knowledge/model.csv` (identical to Triage's execute)
- `version_reuse.py` → the repaired copy from 1.1
- `retrain.py` → `subprocess.run(["python3", "retrain.py", "--model", m])`
- `serving_cfg.py` → writes `knowledge/serving.json`; **new**, requires `inference.py` to read it each loop
- `review.py` → writes `knowledge/review.json`; sets `equity_review_pending`

`serving.json` shape:
```json
{ "batch_size": 1, "seq_length": 5, "sampling_rate": 1.0, "updated_at": "..." }
```
`inference.py` re-reads it on the same cadence it re-reads `model.csv`. Batching means accumulating `batch_size` windows and running one forward pass — the energy saving is real once the model is cached, and it's zero-accuracy-cost, which makes it the most interesting tactic in the catalogue.

### 2.6 The `aegis` approach

`aegis/manage.py`, mirroring `mape/manage.py`'s threading and energy-logging structure. Add `aegis` to `set_approach.sh`'s case statement. Log controller overhead to `mape_log.csv` in the same format so the two are directly comparable — this is how you show your planner's overhead honestly.

---

## Phase 3 — dashboard, evaluation, writeup

### 3.1 Dashboard (`dashboard/app.py`, Streamlit)

Six metric cards with boundary bands as shaded regions. Incident timeline. **The candidate table** — every action, eligible or not, with its guard rejection reason and its Q/E/L/C/F/R terms. Scenario buttons that rewrite the synthetic stream. A weight-profile dropdown that re-runs PLAN on the current snapshot live.

Read-only. The dashboard never writes to `knowledge/`; a third writer in a file-mediated system is how you get corrupted state mid-demo.

### 3.2 Evaluation harness (`tools/evaluate.py`)

8 arms × 8 scenarios × 20 seeds. Each run: fresh `cleanup.sh`, fixed seed, fixed row count, same energy backend. Emit `results.md` with means and standard deviations, and a per-arm energy breakdown separating inference, retraining, and controller overhead.

Runtime is the constraint. At 0.15s/inference the base loop is slow — for the harness, add `--no-sleep` to `inference.py` (real energy per inference is unchanged; only wall-clock compresses). Note that you did this; it affects latency metrics, so report latency only from unaccelerated runs.

### 3.3 Writeup

Mapping table: Triage's contribution / AegisML's delta / evidence. Architecture diagram marking clearly what's inherited and what's new. The bug-fix results as a standalone subsection — a reproducibility finding on a published artifact is publishable on its own terms.

---

## Ordering rules

1. Never break a Triage approach. Run `switch` and `Triage` after every phase.
2. Fix bugs *before* building on the affected code. Version reuse before `REUSE_VERSION`. Model cache before any energy calibration.
3. Calibrate effect vectors only after 1.2, or every number is wrong.
4. Build the eval harness before you optimise anything, so you can tell whether an optimisation helped.
5. `aegis/core/` stays importable with no torch, no pyRAPL, no dataset. If a core test needs any of those, the boundary has leaked.

## Cut list, in order

1. `RETRAIN_ALL` and `REDUCE_WINDOW` — keep in the catalogue as future work
2. Outcome-confidence learning — keep the ledger, drop the α update
3. PSI — KL alone is enough if you're short
4. AWS arm entirely — it was never load-bearing here
5. Equity → if `station_id` proves impractical, degrade to per-time-of-day performance disparity (peak vs off-peak), which is still a real equity construct and needs no new column

**Never cut:** the candidate table, the bug-fix measurements, the controller-overhead row in the results table.
