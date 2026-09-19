# CONTRACTS.md — the only data shapes in AegisML

Pydantic v2 models in `aegis/core/models.py`. Nothing crosses a module boundary except these. Need a field that isn't here? Add it here first, in the same commit.

HarmonE's existing files (`knowledge/mape_info.json`, `thresholds.json`, `model.csv`) keep their current shape — the base approaches depend on them. AegisML reads them and writes its own state elsewhere.

---

## predictions.csv — MODIFIED from HarmonE

Two new columns. Both are load-bearing for claims you want to make.

```csv
true_value,predicted_value,model_used,inference_time,energy_uj,station_id,energy_backend
412.0,408.3,lstm,0.00412,1840.0,ST_03,rapl
```

- `energy_uj` — renamed from HarmonE's inconsistent `energy` / `energy_uJ` pair (BUG-2). One `SCHEMA` constant, imported by `inference.py`, `mape/monitor.py`, and `aegis/core/monitor.py`.
- `station_id` — required for equity. Synthetic generator emits it; with real PeMS it's the station identifier.
- `energy_backend` — `rapl` | `codecarbon` | `estimator`. **MONITOR refuses to score a window containing more than one value.** Mixed-backend energy numbers are not comparable and averaging them silently is how a paper gets retracted.

## serving.json — NEW

Written by actuators, read by `inference.py` each loop, same cadence as `model.csv`.

```json
{ "batch_size": 1, "seq_length": 5, "sampling_rate": 1.0, "updated_at": "2026-09-19T10:15:00Z" }
```

## review.json — NEW

```json
{ "equity_review_pending": true, "opened_at": "...", "incident_id": "inc_0042",
  "worst_station": "ST_07", "gap": 0.19, "resolved_by": null }
```

While `equity_review_pending` is true, every model-promoting action is ineligible. This is the safety invariant as a file.

---

## MetricSnapshot

Produced by MONITOR. Immutable.

```json
{
  "snapshot_id": "snap_0042",
  "ts": "2026-09-19T10:15:00Z",
  "active_model": "lstm",
  "window": { "rows": 1200, "energy_backend": "rapl", "stations": 8 },

  "accuracy": { "r2": 0.842, "mae": 18.4, "ema_score": 0.811 },
  "drift":    { "kl_div": 0.31, "psi": 0.22 },
  "latency":  { "p50_ms": 4.1, "p95_ms": 9.7 },
  "energy":   { "uj_per_inference": 1840.0, "normalized": 0.074,
                "current_threshold": 0.91, "measured": true },
  "cost":     { "inr_per_1k": 0.004, "wall_clock_s": 180.0 },
  "equity":   { "worst_station": "ST_07", "worst_r2": 0.71,
                "gap": 0.13, "per_station_r2": { "ST_01": 0.86 } },

  "serving":  { "batch_size": 1, "seq_length": 5, "sampling_rate": 1.0 },
  "scenario_tag": "drift_benign"
}
```

`scenario_tag` is simulator metadata. **The controller must never read it.** `tests/test_no_scenario_leak.py` enforces this by grepping `aegis/core/`.

`accuracy.ema_score` uses HarmonE's exact formula — `β·r2 + (1−β)·(1−energy_norm)`, then `γ`-smoothed. Keep it for comparability with the base arms, but note in the docstring that it *already* blends energy into accuracy. AegisML's utility treats the families separately, so use raw `r2` for accuracy severity and `ema_score` only where you're reproducing base behaviour. Mixing the two is a subtle double-counting bug — energy would enter the utility twice.

---

## BoundaryReport

```json
{
  "metric": "drift.kl_div",
  "value": 0.31,
  "lower": null, "upper": 0.18,
  "mode": "dynamic",
  "baseline_mean": 0.09, "baseline_std": 0.045, "k": 2.0,
  "hard_ceiling": 0.75,
  "violated": true, "severity": 0.23, "consecutive_violations": 2
}
```

`mode ∈ {dynamic, hard, integral}`. `integral` is HarmonE's energy controller (`thr += 0.95·(orig − used)`), routed through the same report type so the dashboard renders every boundary the same way.

`hard_ceiling` for KL is 0.75 — HarmonE's drift threshold, kept so drift detection stays comparable across arms.

---

## Incident

```json
{
  "incident_id": "inc_0042",
  "snapshot_id": "snap_0042",
  "type": "DRIFT_ONLY",
  "violated": ["drift.kl_div"],
  "severity": { "accuracy": 0.0, "drift": 0.23, "latency": 0.0,
                "energy": 0.0, "cost": 0.0, "equity": 0.0 },
  "boundaries": ["<BoundaryReport>"],
  "classification_reason": "drift boundary exceeded for 2 consecutive snapshots while r2 stayed inside its lower boundary"
}
```

`type ∈ {NONE, DRIFT_ONLY, ACCURACY_DROP, MODEL_DEGRADATION, ENERGY_PRESSURE, LATENCY_PRESSURE, EQUITY_VIOLATION, COMPOSITE}`

---

## ActionCandidate

One per action considered. **Rejected candidates are persisted too** — that list is the demo and the paper figure.

```json
{
  "action": "RETRAIN_CURRENT",
  "params": { "model": "lstm" },
  "eligible": true,
  "rejected_by_guard": null,
  "terms": { "Q": 0.06, "E": 0.91, "L": 0.05, "C": 0.88, "F": 0.0, "R": 0.55 },
  "weights": { "w_acc": 1.0, "w_e": 0.8, "w_l": 0.4, "w_c": 0.3, "w_eq": 1.4, "w_r": 0.4 },
  "utility": -1.24,
  "projected_effect": { "accuracy": 0.06, "drift": -0.85, "energy": 0.91 },
  "reason": "projected accuracy relief 0.06 (accuracy severity is 0) does not offset energy 0.91 and risk 0.55"
}
```

Ineligible:
```json
{ "action": "REUSE_VERSION", "eligible": false,
  "rejected_by_guard": "best_version_kl 0.82 >= max 0.75", "utility": null }
```

Parameterised actions (`SWITCH_MODEL:linear`, `SWITCH_MODEL:svm`, `REUSE_VERSION:lstm@v3`) expand to one candidate each. With three models and a few versions you get 10–15 candidates per decision. That's fine and it makes the table more convincing, not less.

---

## Decision

```json
{
  "decision_id": "dec_0042",
  "incident_id": "inc_0042",
  "chosen_action": "OBSERVE",
  "params": {},
  "candidates": ["<ActionCandidate>"],
  "policy_profile": "balanced",
  "policy_version": "2026-09-19.1",
  "exploratory": false,
  "reason": "accuracy inside boundary; drift alone does not justify retraining energy 0.91 at risk 0.55",
  "recheck_after_snapshots": 3
}
```

`exploratory: true` marks ε-greedy picks. **Exclude them when computing policy-quality metrics** in the evaluation, and say so in the paper.

---

## ActionOutcome

```json
{
  "outcome_id": "out_0042",
  "decision_id": "dec_0042",
  "action": "OBSERVE",
  "status": "SUCCESS",
  "pre":  { "drift.kl_div": 0.31, "accuracy.r2": 0.842 },
  "post": { "drift.kl_div": 0.14, "accuracy.r2": 0.838 },
  "resolved": true,
  "snapshots_to_resolve": 3,
  "actual_energy_uj": 0.0,
  "notes": "drift receded without intervention"
}
```

`status ∈ {SUCCESS, FAILED, PENDING_HUMAN, TIMED_OUT}`. `actual_energy_uj` is the **measured** cost of the action itself, from `mape_log.csv` — the ground truth against which effect vectors get recalibrated.

---

## config/policy.json

```json
{
  "version": "2026-09-19.1",
  "active_profile": "balanced",
  "persistence": 2,
  "epsilon": 0.1,
  "learning_enabled": false,
  "profiles": {
    "balanced":       { "w_acc": 1.0, "w_e": 0.8, "w_l": 0.4, "w_c": 0.3, "w_eq": 1.4, "w_r": 0.4 },
    "energy_first":   { "w_acc": 0.6, "w_e": 1.6, "w_l": 0.4, "w_c": 0.5, "w_eq": 1.4, "w_r": 0.5 },
    "accuracy_first": { "w_acc": 1.8, "w_e": 0.3, "w_l": 0.3, "w_c": 0.2, "w_eq": 1.4, "w_r": 0.2 }
  },
  "actions": {
    "OBSERVE": { "risk": 0.0, "energy_uj": 0,
      "effect": {}, "guards": [] },

    "SWITCH_MODEL": { "risk": 0.2, "energy_uj": 500, "parameterised_by": "model",
      "effect_source": "config/hardware.json#model_profiles",
      "guards": [
        { "field": "params.model", "op": "ne", "value": "$active_model" },
        { "field": "equity_review_pending", "op": "eq", "value": false }
      ] },

    "BATCH_INFERENCE": { "risk": 0.15, "energy_uj": 0,
      "effect": { "energy": -0.30, "latency": 0.10, "accuracy": 0.0 },
      "guards": [ { "field": "serving.batch_size", "op": "lt", "value": 8 } ] },

    "REUSE_VERSION": { "risk": 0.25, "energy_uj": 2000, "parameterised_by": "version",
      "effect": { "accuracy": 0.10, "drift": -0.60 },
      "guards": [
        { "field": "archived_versions", "op": "gte", "value": 1 },
        { "field": "best_version_kl", "op": "lt", "value": 0.75 },
        { "field": "equity_review_pending", "op": "eq", "value": false }
      ] },

    "RETRAIN_CURRENT": { "risk": 0.55, "energy_uj": 4200000,
      "effect": { "accuracy": 0.06, "drift": -0.85, "energy": 0.91 },
      "guards": [
        { "field": "drift_window_rows", "op": "gte", "value": 1200 },
        { "field": "cooldown_elapsed_s.RETRAIN_CURRENT", "op": "gte", "value": 600 }
      ] },

    "EQUITY_REVIEW": { "risk": 0.15, "energy_uj": 0,
      "effect": { "equity": -0.50 },
      "guards": [] }
  }
}
```

Effects are **normalised deltas**: `-0.30` means "removes 30% of that family's current distance to its boundary". Negative reduces severity. `accuracy` is inverted (positive = improvement = severity reduction) — handle that sign flip in one place in `utility.py`, with a comment, or you will get it wrong twice.

`energy_uj` per action must come from **measurement**, not guesswork. `RETRAIN_CURRENT` at 4.2 J is a placeholder until `tools/calibrate_effects.py` writes the real figure from your hardware. Every number in this file carries a provenance comment and a date.

---

## config/boundaries.json

```json
{
  "window": 20, "k": 2.0,
  "metrics": {
    "accuracy.r2":            { "direction": "lower", "hard_floor": 0.70 },
    "drift.kl_div":           { "direction": "upper", "hard_ceiling": 0.75 },
    "latency.p95_ms":         { "direction": "upper", "hard_ceiling": 50.0 },
    "energy.normalized":      { "direction": "upper", "mode": "integral",
                                "original_threshold": 1.0, "recovery_rate": 0.95 },
    "equity.gap":             { "direction": "upper", "hard_ceiling": 0.15 },
    "equity.worst_r2":        { "direction": "lower", "hard_floor": 0.60 }
  }
}
```

`hard_ceiling` for KL stays at HarmonE's 0.75 so drift detection is identical across arms. Latency and equity ceilings need calibrating from a healthy baseline run — `mean ± 3σ`, never laxer than the domain limit.

---

## Ports — `aegis/ports/`

```python
class EnergyMeter(Protocol):
    backend: str
    measured: bool
    def measure(self, label: str) -> ContextManager[Reading]: ...

class KnowledgeStore(Protocol):
    def history(self, n: int) -> list[MetricSnapshot]: ...
    def put_snapshot(self, s: MetricSnapshot) -> None: ...
    def put_incident(self, i: Incident) -> None: ...
    def put_decision(self, d: Decision) -> None: ...
    def put_outcome(self, o: ActionOutcome) -> None: ...
    def last_action_ts(self, action: str) -> datetime | None: ...
    def policy(self) -> dict: ...

class Actuator(Protocol):
    def switch_model(self, model: str) -> ActionOutcome: ...
    def reuse_version(self, model: str, version: str) -> ActionOutcome: ...
    def set_serving(self, **kw) -> ActionOutcome: ...
    def retrain(self, model: str | None) -> ActionOutcome: ...
    def open_equity_review(self, incident: Incident) -> ActionOutcome: ...
```

Two implementations of `Actuator`: `file` (writes the real `knowledge/` files HarmonE reads) and `fake` (records calls, for tests). The `file` one is the production path — there's no "local vs cloud" split here, because the whole system is local by design.
