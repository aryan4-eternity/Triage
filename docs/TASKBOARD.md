# TASKBOARD.md

One task at a time. Don't start a task whose dependencies aren't checked. Tick the box and run `make test` before moving on.

Format: **files it may touch** · **acceptance** · **depends on**

**Standing rule, every task:** after finishing, run `./set_approach.sh Triage && python3 mape/manage.py` for one cycle. If a Triage approach broke, you broke your own baselines. Fix it before continuing.

---

## Phase 0 — make the base run (nothing works until this does)

- [ ] **T0.1 Energy abstraction**
  Files: `aegis/energy/{meter,rapl,codecarbon,estimator}.py`, `config/hardware.json`, `inference.py`, `mape/manage.py`
  Acceptance: `AEGIS_ENERGY=estimator python3 -c "from aegis.energy import get_meter; ..."` works on any machine; `rapl` backend raises a clear `EnergyBackendUnavailable` off Intel/Linux instead of dying at import; startup logs the selected backend.
  Depends: —

- [ ] **T0.2 Synthetic data generator**
  Files: `tools/synth_data.py`
  Acceptance: `python3 tools/synth_data.py --rows 40000 --seed 7 --stations 8` writes `data/pems/flow_data_{train,test}.csv` with columns `flow,station_id`; daily and weekly seasonality visible in a plot; `--regime 20000:24000:1.4:60` produces a detectable KL spike at that index.
  Depends: —

- [ ] **T0.3 Base system runs end to end**
  Files: none (verification only)
  Acceptance: `./cleanup.sh && python3 tools/train_models.py && ./set_approach.sh Triage`, then both processes. Past 400s: `predictions.csv` grows, EMA scores in `mape_info.json` move, at least one model switch occurs. **Record the console output — it's your before-picture.**
  Depends: T0.1, T0.2

- [ ] **T0.4 Attribution**
  Files: `NOTICE.md`, `README.md`
  Acceptance: Triage credited with its MIT copyright line; your additions scoped explicitly; the real Triage citation recorded (get it from the repo — **do not reuse the fabricated arXiv ID from the earlier planning chat**).
  Depends: —

---

## Phase 1 — fix the base, measure the difference

- [ ] **T1.1 BUG-1: repair version reuse**
  Files: `mape/analyse.py`, `mape/execute.py`, `tests/test_version_reuse.py`
  Acceptance: `get_best_version` returns the version directory; `execute_drift` copies the actual model artifact by name and extension; test builds a fake `versionedMR/` tree and asserts `models/data.pth` is never created. **Plus a before/after run on the recoverable-drift scenario showing accuracy recovery without a retrain.**
  Depends: T0.3

- [ ] **T1.2 BUG-3: cache the model**
  Files: `inference.py`
  Acceptance: model object cached, invalidated on `model.csv` mtime change; a switch still takes effect within one loop iteration. `torch.load` called once per switch, not once per inference (assert with a counter in a test).
  Depends: T0.3

- [ ] **T1.3 Profile and calibrate**
  Files: `tools/profile_models.py`, `config/hardware.json`, `knowledge/thresholds.json`; delete `tools/load_test_models.py`
  Acceptance: median µJ and ms per inference for all three models at `batch_size ∈ {1,2,4,8}`, idle power measured separately, gross and net both reported, one measured retrain cost per model. `E_m`/`E_M` written from this run with provenance. **Document the before/after energy gap from T1.2 — it will narrow.**
  Depends: T1.2

- [ ] **T1.4 Schema, scaler, paths**
  Files: `aegis/core/models.py` (SCHEMA), `inference.py`, `retrain.py`, `tools/train_models.py`, `mape/monitor.py`, `mape/__init__.py`, `aegis/__init__.py`
  Acceptance: one column-name constant used everywhere (`energy_uj`); `artifacts/scaler.pkl` fit once on train data and loaded by inference and retrain; both packages importable from any CWD. Triage decision sequence on a fixed seed matches pre-change.
  Depends: T1.3

**End of Phase 1:** you have a working, instrumented, bug-fixed base and three measured results to talk about. If the hackathon ended here you'd still have something.

---

## Phase 2 — the AegisML control plane

- [ ] **T2.1 Contracts + store**
  Files: `aegis/core/models.py`, `aegis/store.py`, `tests/test_models.py`
  Acceptance: every JSON example in `docs/CONTRACTS.md` round-trips; SQLite store persists and replays snapshots/incidents/decisions/candidates/outcomes.
  Depends: T1.4

- [ ] **T2.2 Metric extractors**
  Files: `aegis/core/metrics/{accuracy,drift,latency,energy,equity}.py`, tests
  Acceptance: KL matches `mape/monitor.py`'s output on the same window (parity test); PSI <0.1 on identical distributions, >0.25 on a 1.5σ shift; per-station R² matches a hand-computed fixture; mixed-backend window raises.
  Depends: T2.1

- [ ] **T2.3 MONITOR**
  Files: `aegis/core/monitor.py`, tests
  Acceptance: emits a valid `MetricSnapshot` from a capture window; never reads `scenario_tag`; refuses mixed backends.
  Depends: T2.2

- [ ] **T2.4 ANALYZE + boundaries**
  Files: `aegis/core/{boundaries,analyze}.py`, `config/boundaries.json`, tests
  Acceptance: all `docs/POLICY_ENGINE.md` §7 boundary tests pass, including Triage energy-integral parity and the new lower clamp; each of the 7 classification rules has a test that hits it.
  Depends: T2.3

- [ ] **T2.5 PLAN — guards + utility**
  Files: `aegis/core/{guards,utility,plan}.py`, `config/policy.json`, tests
  Acceptance: every `docs/POLICY_ENGINE.md` §7 plan test passes, including the §4 worked example reproducing U = −1.88 for `RETRAIN_CURRENT` under `balanced`.
  Depends: T2.4

- [ ] **T2.6 Actuators**
  Files: `aegis/actuators/*.py`, `inference.py` (read `serving.json`), tests
  Acceptance: `switch_model` writes the same `model.csv` Triage writes; `reuse_version` uses T1.1's repaired path; `set_serving` changes batch size and `inference.py` picks it up within one loop; `open_equity_review` writes `review.json` and sets the pending flag.
  Depends: T2.5

- [ ] **T2.7 EXECUTE + LEARN**
  Files: `aegis/core/{execute,learn}.py`, tests
  Acceptance: outcomes resolve within N snapshots, record measured `actual_energy_uj` from `mape_log.csv`, and set `snapshots_to_resolve`.
  Depends: T2.6

- [ ] **T2.8 The `aegis` approach**
  Files: `aegis/manage.py`, `set_approach.sh`, `approach.conf`
  Acceptance: `./set_approach.sh aegis` works; controller overhead logged to `mape_log.csv` in Triage's format; all nine original approaches still run.
  Depends: T2.7

- [ ] **T2.9 Effect-vector calibration**
  Files: `tools/calibrate_effects.py`, `config/policy.json`
  Acceptance: every `effect` and `energy_uj` in `policy.json` derived from T1.3's measurements or a measured run, each with a provenance comment and date. No placeholder numbers remain.
  Depends: T2.8, T1.3

**End of Phase 2:** the project is demoable headless. Everything after is amplification.

---

## Phase 3 — dashboard, evaluation, writeup

- [ ] **T3.1 Scenario runner**
  Files: `tools/run_scenario.py`
  Acceptance: `python3 tools/run_scenario.py --scenario drift_benign` prints the full MAPE trace and the candidate table; all 8 scenarios produce the decision in `ARCHITECTURE.md` §8.
  Depends: T2.9

- [ ] **T3.2 Streamlit dashboard**
  Files: `dashboard/app.py`
  Acceptance: six metric cards with boundary bands; incident timeline; **candidate table with rejected actions and guard reasons**; scenario buttons; weight-profile dropdown that re-decides live; persistent energy-backend badge. Read-only — never writes `knowledge/`.
  Depends: T3.1

- [ ] **T3.3 Evaluation harness**
  Files: `tools/evaluate.py`, `results.md`
  Acceptance: 8 arms × 8 scenarios × 20 seeds; fills `ARCHITECTURE.md` §9 with means and std devs; energy broken out into inference / retraining / controller overhead; refuses to mix energy backends; excludes `exploratory: true` decisions from policy-quality metrics.
  Depends: T3.1

- [ ] **T3.4 Backup recording**
  Acceptance: 3-minute screen recording of all 8 scenarios. **Do this before further polish.**
  Depends: T3.2

- [ ] **T3.5 Writeup**
  Files: `docs/PITCH.md`, architecture diagram
  Acceptance: Triage-vs-AegisML mapping table; bug-fix results as their own subsection; effect-vector calibration accuracy table; diagram marking inherited vs new; 3-minute script including the attribution up front and the "safety is a constraint, not a weight" line.
  Depends: T3.3

---

## Stretch (only after T3.4)

- [ ] Outcome-confidence learning (`policy.learning_enabled`) with a reported learning curve
- [ ] `RETRAIN_ALL`, `REDUCE_WINDOW` actions
- [ ] Carbon estimate derived from measured joules with a cited grid intensity
- [ ] AWS deployment arm with an explicitly-labelled estimated-energy backend
- [ ] Real PeMS data run alongside the synthetic one

---

## Cut list, in order

1. `RETRAIN_ALL`, `REDUCE_WINDOW` — stay in the catalogue as future work
2. Outcome-confidence learning — keep the ledger, drop the α update
3. PSI — KL alone suffices
4. AWS arm — never load-bearing here
5. Equity via `station_id` → degrade to peak vs off-peak disparity, still real, no new column

**Never cut:** the candidate table, the bug-fix measurements, the controller-overhead row, the energy-backend labelling.
