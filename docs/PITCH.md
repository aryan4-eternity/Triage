# AegisML — Pitch & Writeup

## 0. Attribution (say this first, in the first 30 seconds)

This project extends **Triage** (MIT, copyright 2025 Hiya Bhatt), a published
self-adaptive MLOps loop for traffic-flow regression. The MAPE-K skeleton, the
inference loop, and the RAPL energy instrumentation are Triage's work. AegisML
contributes the multi-objective action-selection layer, the declarative eligibility
guards, the outcome ledger, and the equity monitoring. See `NOTICE.md`.

---

## 1. One-line pitch

> *Triage decides which model to run. AegisML decides what to do at all — where
> switching models is one tactic among nine, chosen by a multi-objective utility
> with explicit eligibility guards and a safety constraint that sits outside the
> utility.*

---

## 2. Triage vs AegisML — mapping table

| Dimension | Triage (inherited) | AegisML (contributed) |
|---|---|---|
| Action space | Switch between 3 models | 9 heterogeneous tactics |
| Metrics monitored | R², energy | Accuracy, drift, latency, energy, cost, equity |
| Decision method | argmax EMA score | Multi-objective utility + declarative guards |
| Eligibility | `recovery_cycles` cooldown | Per-action guards in `config/policy.json` |
| Reasoning trail | None persisted | Full candidate table (eligible + rejected) |
| Outcome tracking | None | Outcome ledger — did the fix work? |
| Safety | Not modelled | `equity_review_pending` hard-blocks model promotion |
| Baselines | — | All 9 original approaches still run |

---

## 3. Bug fixes as standalone results

Two bugs in Triage are reproducibility findings in their own right:

### BUG-1: Version reuse was a silent no-op

`analyse.get_best_version()` returned the path to `data.csv`.
`execute.execute_drift()` extracted `model_name = basename("data") = "data"`,
then copied the CSV to `models/data.pth` — a file nothing ever loads.

**Fix:** return the version directory; resolve artifact by name and extension.
**Result:** the "recover from archived version without retraining" tactic now
actually fires. Before: system always falls through to a full retrain.
After: accuracy recovers in the recoverable-drift scenario without a single
retraining job.

### BUG-3: Model reloaded from disk every inference

```python
for i in range(len(X_stream)):
    lstm_model = LSTMModel()
    lstm_model.load_state_dict(torch.load("models/lstm.pth"))
```

Every "inference energy" measurement included model deserialisation.
The LSTM/linear energy gap in Triage's published results is therefore
dominated by I/O, not computation.

**Fix:** cache `(name, mtime, model_object)`; reload only on mtime change.
**Result:** per-inference energy narrows substantially between models once
the measurement confound is removed. This is reported honestly as a
correction to the base exemplar's energy profile, not compared against
Triage's published numbers as if they were the same measurement.

---

## 4. The contribution — why the candidate table matters

Every MLOps pipeline shows a decision. AegisML shows *reasoning about tradeoffs*.

In the `drift_benign` scenario: KL divergence rises, accuracy stays stable.
Triage's `plan_drift` is below its threshold and reports nothing.
AegisML fires `DRIFT_ONLY`, evaluates all nine actions, and chooses `OBSERVE`
with a printed reason:

> *"drift severity 0.23 with accuracy inside boundary; RETRAIN_CURRENT projected
> relief 0.20 outweighed by energy cost 1.00 and risk 0.55."*

The rejected `RETRAIN_CURRENT` appears in the candidate table with U = -1.88.
That table is the differentiator — not a pipeline, but reasoning about tradeoffs.

---

## 5. Safety is a constraint, not a weight

In the `equity_shift` scenario: station ST_07's R² falls below its boundary
while aggregate R² = 0.84. Triage, monitoring only aggregate metrics, sees
nothing wrong and continues optimising energy by switching to a cheaper model —
plausibly making the worst station worse.

AegisML fires `EQUITY_VIOLATION`, sets `equity_review_pending = True`, and
makes every model-promoting action ineligible regardless of utility. No accuracy
score, however high, can buy its way past this guard.

**The line for judges:** *"This is the one case where utility doesn't decide.
Safety is a constraint, not a weight."*

---

## 6. Effect-vector calibration

Effect vectors in `config/policy.json` are derived from `tools/profile_models.py`
measurements (warm-cache, batch sizes 1/2/4/8, idle power measured separately).

A calibration table comparing declared effects to observed pre/post deltas
(from the outcome ledger) is available after a live run via:

```bash
python3 tools/calibrate_effects.py
```

Current values carry `TODO(calibrate)` provenance markers — replace with
measured values from an Intel/Linux run before reporting results.

---

## 7. Evaluation summary (headless, estimator backend)

From `results.md` (5 seeds × 8 scenarios × 7 arms):

- AegisML chose `EQUITY_REVIEW` on all equity-violation scenarios (0% miss rate).
- `SWITCH_MODEL:linear` was selected for both `energy_pressure` and `latency_pressure`
  — the cheapest model that satisfies the constraint, consistent with the
  "retrain isn't the default" claim.
- Unnecessary intervention rate: 0% across all arms in headless eval.
- Retraining rate: 0% — no drift scenario crossed the `drift_window_rows >= 1200`
  guard threshold in a single-window headless run.

**For full energy results:** run on Intel/Linux with `AEGIS_ENERGY=rapl`.
The dashboard shows a persistent `Energy: MEASURED (RAPL)` / `Energy: ESTIMATED`
badge so every figure is labelled with its provenance.

---

## 8. Three-minute demo script

1. *"This extends Triage — MIT licensed, copyright 2025 Hiya Bhatt — which we
   found has two reproducibility bugs. Here's the before and after."*
   → Show BUG-1 and BUG-3 fix measurements.

2. *"AegisML replaces the Plan stage. Same inference loop, same energy meter,
   nine actions instead of three."*
   → `python3 tools/run_scenario.py --scenario drift_benign`
   → Show candidate table. Point to RETRAIN_CURRENT rejected at U = -1.88.

3. *"This is the one case where utility doesn't decide."*
   → `python3 tools/run_scenario.py --scenario equity_shift`
   → Show EQUITY_REVIEW chosen; SWITCH_MODEL ineligible.

4. *"Same incident, different weight profile — the policy layer is load-bearing."*
   → `python3 tools/run_scenario.py --scenario weight_profile`

5. *"Live dashboard — read-only, never writes to knowledge/"*
   → `streamlit run dashboard/app.py`

---

## 9. What we would say if the numbers went badly

- If AegisML's controller overhead eats its energy savings: *"we measured it,
  here's the controller overhead row, and here's why it's still a contribution."*
  A negative result with a clean measurement beats a positive one with a
  hidden confound, and reviewers will check.
- Energy is measured on Intel/Linux, estimated elsewhere. Every figure is
  labelled. Mixed-backend runs are never averaged together.
- The bug fixes are publishable independently of AegisML.
  "We reproduced Triage and found two measurement confounds" is a
  reproducibility contribution on its own terms.

---

## 10. Non-goals (say these before you're asked)

- Not a production system. A research prototype extending a published exemplar.
- Energy is **measured** where RAPL is available and **estimated** otherwise.
  Every figure carries its backend label.
- Equity here means per-station performance disparity, not demographic fairness.
  PEMS traffic data has no protected attributes and none are invented.
- No reinforcement learning. The transparent utility scorer is the contribution.
  Outcome-confidence learning is flagged as an extension, default off.
- BUG-3's fix changes the energy profile. We report before-and-after;
  we do not compare our corrected numbers against Triage's published ones.
