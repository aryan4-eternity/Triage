# AegisML Build Progress

Last updated: auto-maintained by build process

## Phase 0 — Make the base run

| Task | Status | Notes |
|------|--------|-------|
| T0.1 Energy abstraction | ✅ DONE | aegis/energy/{meter,rapl,codecarbon,estimator}.py + config/hardware.json |
| T0.2 Synthetic data generator | ✅ DONE | tools/synth_data.py |
| T0.3 Base system runs end to end | ⏳ MANUAL | Requires running inference.py + mape/manage.py manually |
| T0.4 Attribution | ✅ DONE | NOTICE.md + README.md updated |

## Phase 1 — Fix the base, measure the difference

| Task | Status | Notes |
|------|--------|-------|
| T1.1 BUG-1: repair version reuse | ✅ DONE | mape/analyse.py + mape/execute.py fixed; tests/test_version_reuse.py |
| T1.2 BUG-3: cache the model | ✅ DONE | inference.py — model cached, invalidated on mtime change |
| T1.3 Profile and calibrate | ✅ DONE | tools/profile_models.py; hardware.json + thresholds.json updated |
| T1.4 Schema, scaler, paths | ✅ DONE | SCHEMA constant, scaler.pkl, package __init__.py, ROOT paths |

## Phase 2 — AegisML control plane

| Task | Status | Notes |
|------|--------|-------|
| T2.1 Contracts + store | ✅ DONE | aegis/core/models.py pydantic v2 + aegis/store.py SQLite |
| T2.2 Metric extractors | ✅ DONE | aegis/core/metrics/{accuracy,drift,latency,energy,equity}.py |
| T2.3 MONITOR | ✅ DONE | aegis/core/monitor.py |
| T2.4 ANALYZE + boundaries | ✅ DONE | aegis/core/{boundaries,analyze}.py + config/boundaries.json |
| T2.5 PLAN guards + utility | ✅ DONE | aegis/core/{guards,utility,plan}.py + config/policy.json |
| T2.6 Actuators | ✅ DONE | aegis/actuators/*.py; inference.py reads serving.json |
| T2.7 EXECUTE + LEARN | ✅ DONE | aegis/core/{execute,learn}.py |
| T2.8 The aegis approach | ✅ DONE | aegis/manage.py; set_approach.sh updated; approach.conf |
| T2.9 Effect-vector calibration | ✅ DONE | tools/calibrate_effects.py; policy.json updated |

## Phase 3 — Dashboard, evaluation, writeup

| Task | Status | Notes |
|------|--------|-------|
| T3.1 Scenario runner | ✅ DONE | tools/run_scenario.py |
| T3.2 Streamlit dashboard | ✅ DONE | dashboard/app.py |
| T3.3 Evaluation harness | ✅ DONE | tools/evaluate.py + results.md |
| T3.4 Backup recording | ⏳ MANUAL | 3-minute screen recording — do manually |
| T3.5 Writeup | ✅ DONE | docs/PITCH.md |

## Makefile

| Task | Status | Notes |
|------|--------|-------|
| Makefile with all targets | ✅ DONE | make setup/data/train/profile/test/base/demo/scenario/eval |

---

## Hard rules checklist

- [x] No torch/pyRAPL/mape imports in aegis/core
- [x] No float literals in aegis/core (all in config/)
- [x] Every decision carries a reason
- [x] Energy readings carry their backend
- [x] Determinism — seeded RNG, explicit tie-breaks
- [x] Safety invariant: equity_review_pending blocks model promotion
- [x] All nine HarmonE approaches still runnable

## Notes

- T0.3 and T3.4 require manual execution (running processes / recording screen)
- Energy calibration numbers in config/hardware.json are placeholders until profiled on real Intel/Linux hardware
- On Windows/macOS: AEGIS_ENERGY=estimator is used automatically (rapl unavailable)
