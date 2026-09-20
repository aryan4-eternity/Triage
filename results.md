# AegisML Evaluation Results

*Generated: 2026-09-20 17:19*  
*Energy backend: `estimator` (measured=False)*  
*Seeds: 5  Scenarios: 8  Arms: 7*

> **Note:** On non-Intel/non-Linux machines, energy values are estimated (time × TDP proxy). Every figure is labelled with its backend. Mixed-backend runs are excluded from energy comparisons.

## Table 1: Incident Distribution by Arm

| arm | DRIFT_ONLY | ENERGY_PRESSURE | EQUITY_VIOLATION | LATENCY_PRESSURE | NONE |
| --- | --- | --- | --- | --- | --- |
| aegis | 15 | 4 | 13 | 4 | 4 |
| harmone | 15 | 4 | 13 | 4 | 4 |
| single-linear | 15 | 4 | 13 | 4 | 4 |
| single-lstm | 15 | 4 | 13 | 4 | 4 |
| single-svm | 15 | 4 | 13 | 4 | 4 |
| switch | 15 | 4 | 13 | 4 | 4 |
| switch+retrain | 15 | 4 | 13 | 4 | 4 |

## Table 2: Chosen Actions by Arm

| arm | EQUITY_REVIEW | OBSERVE | SWITCH_MODEL |
| --- | --- | --- | --- |
| aegis | 11 | 15 | 14 |
| harmone | 11 | 15 | 14 |
| single-linear | 11 | 18 | 11 |
| single-lstm | 11 | 15 | 14 |
| single-svm | 11 | 15 | 14 |
| switch | 11 | 15 | 14 |
| switch+retrain | 11 | 15 | 14 |

## Table 3: Policy Quality Metrics (non-exploratory only)

| arm | n_runs | retrain_% | unnecessary_% |
| --- | --- | --- | --- |
| aegis | 32 | 0.0 | 0.0 |
| harmone | 32 | 0.0 | 0.0 |
| single-linear | 32 | 0.0 | 0.0 |
| single-lstm | 32 | 0.0 | 0.0 |
| single-svm | 32 | 0.0 | 0.0 |
| switch | 32 | 0.0 | 0.0 |
| switch+retrain | 32 | 0.0 | 0.0 |

## Table 4: Per-Scenario Correct Decision Rate (aegis arm)

| Scenario | Expected | Correct (%) | N Runs |
| --- | --- | --- | --- |
| normal | OBSERVE | 60 | 5 |
| drift_benign | OBSERVE | LOWER_SAMPLING | 80 | 5 |
| recoverable_drift | REUSE_VERSION | OBSERVE | 80 | 5 |
| model_degradation | EQUITY_REVIEW | RETRAIN_CURRENT | 80 | 5 |
| energy_pressure | SWITCH_MODEL | BATCH_INFERENCE | 80 | 5 |
| latency_pressure | SWITCH_MODEL | BATCH_INFERENCE | REDUCE_WINDOW | 80 | 5 |
| equity_shift | EQUITY_REVIEW | 80 | 5 |
| weight_profile | OBSERVE | 80 | 5 |

## Notes

- Exploratory decisions (epsilon-greedy) are excluded from policy-quality metrics.
- `unnecessary_rate` = fraction of non-OBSERVE decisions when incident=NONE.
- Energy measurements require Intel/Linux + RAPL. All figures here use the `estimator` backend.
- `recoverable_drift` correct rate may be low in headless eval because no archived versions are present; in a live run after retraining, REUSE_VERSION becomes available.

## Reproducibility

```bash
make data && make train && python3 tools/evaluate.py --seeds 20
```
