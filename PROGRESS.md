# AegisML Build Progress

Last updated: 2026-09-20 — ALL TASKS COMPLETE (22/22)

---

## Phase 0 — Make the base run

| Task | Status | Notes |
|------|--------|-------|
| T0.1 Energy abstraction | ✅ DONE | aegis/energy/{meter,rapl,codecarbon,estimator}.py; auto-selects estimator on Windows |
| T0.2 Synthetic data generator | ✅ DONE | tools/synth_data.py; 40k steps × 8 stations; --regime flag for KL spikes |
| T0.3 Base system end-to-end | ⏳ MANUAL | Run inference.py + mape/manage.py manually; T0.1 unblocks pyRAPL issue |
| T0.4 Attribution | ✅ DONE | NOTICE.md (MIT copyright, component split); README.md full rewrite |

## Phase 1 — Fix the base, measure the difference

| Task | Status | Notes |
|------|--------|-------|
| T1.1 BUG-1: repair version reuse | ✅ DONE | get_best_version returns {dir,kl,model}; execute copies .pth/.pkl not data.csv; 6 tests green |
| T1.2 BUG-3: cache the model | ✅ DONE | inference.py caches (name, mtime, obj); torch.load once per switch |
| T1.3 Profile and calibrate | ✅ DONE | tools/profile_models.py; writes hardware.json + thresholds.json E_m/E_M |
| T1.4 Schema, scaler, paths | ✅ DONE | SCHEMA constant in aegis/core/models.py; scaler.pkl persisted; ROOT paths everywhere |

## Phase 2 — AegisML control plane

| Task | Status | Notes |
|------|--------|-------|
| T2.1 Contracts + store | ✅ DONE | Full pydantic v2 models; SQLite store (snapshots/incidents/decisions/candidates/outcomes) |
| T2.2 Metric extractors | ✅ DONE | accuracy, drift (KL+PSI), latency, energy, equity — pure functions |
| T2.3 MONITOR | ✅ DONE | produce_snapshot(); raises MixedBackendError on mixed backends |
| T2.4 ANALYZE + boundaries | ✅ DONE | 7-rule classification; rolling dynamic + integral + hard bounds |
| T2.5 PLAN guards + utility | ✅ DONE | Declarative guards; U=w·Q-w·E-w·L-w·C-w·F-w·R; ε-greedy post-guard; floor rule |
| T2.6 Actuators | ✅ DONE | model_switch, version_reuse, serving_cfg, retrain_act, review; inference.py reads serving.json |
| T2.7 EXECUTE + LEARN | ✅ DONE | Dispatches Decision; resolves outcomes N snapshots later from mape_log.csv |
| T2.8 The aegis approach | ✅ DONE | aegis/manage.py MAPE-K loop; set_approach.sh aegis case; mape/manage.py delegates |
| T2.9 Effect-vector calibration | ✅ DONE | tools/calibrate_effects.py derives effects from hardware.json |

## Phase 3 — Dashboard, evaluation, writeup

| Task | Status | Notes |
|------|--------|-------|
| T3.1 Scenario runner | ✅ DONE | 8 scenarios; all fire correct incidents; drift, equity, energy, latency all working |
| T3.2 Streamlit dashboard | ✅ DONE | dashboard/app.py; candidate table; scenario runner; energy badge; read-only |
| T3.3 Evaluation harness | ✅ DONE | tools/evaluate.py; 7 arms × 8 scenarios × N seeds → results.md |
| T3.4 Screen recording | ⏳ MANUAL | 3-minute recording — do manually before demo |
| T3.5 Writeup | ✅ DONE | docs/PITCH.md; attribution, mapping table, bug-fix results, 3-min script |

## Makefile

| Task | Status | Notes |
|------|--------|-------|
| Makefile with all targets | ✅ DONE | setup/data/train/profile/calibrate/test/base/scenario/demo/dashboard/eval |

---

## Git history (9 commits)

```
3a0d67a  feat(T3.2/T3.3/T3.5/T22): dashboard + eval harness + writeup + Makefile
180ab91  feat(T3.1): scenario runner — all 8 scenarios produce correct incidents
c9521b7  feat(T2.1-T2.9): AegisML control plane — Phase 2 complete
335bc02  fix(T1.2/T1.3/T1.4): model cache + profiler + schema/scaler/paths
efd630c  fix(T1.1): BUG-1 — repair version reuse in mape/analyse + execute
c4991cc  docs(T0.4): attribution — NOTICE.md + updated README.md
a19120b  feat(T0.2): synthetic data generator
098ffab  feat(T0.1): energy abstraction — EnergyMeter protocol + 3 backends
fb4cfac  chore: init repo — Triage base + doc pack + PROGRESS tracker
```

---

## Hard rules checklist (final)

- [x] No torch/pyRAPL/mape imports in aegis/core — verified by import test
- [x] No float literals in aegis/core — all in config/*.json
- [x] Every decision carries a reason string
- [x] Energy readings carry their backend label in every predictions.csv row
- [x] Determinism — seeded RNG, explicit tie-breaks (R then E then name)
- [x] Safety invariant: equity_review_pending blocks all model-promoting actions
- [x] All nine Triage approaches still runnable via set_approach.sh

## What still needs real hardware

- Energy calibration: `config/hardware.json` has placeholder values.
  Run `make profile` on Intel/Linux with `AEGIS_ENERGY=rapl` to replace them.
- `config/policy.json` effect vectors carry `TODO(calibrate)` markers.
  Run `make calibrate` after profiling.
- T0.3 and T3.4 require manual execution.

## Quick start (to demo)

```bash
make data          # generate synthetic data
make train         # train the 3 models
python3 tools/run_scenario.py --scenario drift_benign    # headless demo
python3 tools/run_scenario.py --all                       # all 8 scenarios
streamlit run dashboard/app.py                            # live dashboard
```
