# POLICY_ENGINE.md — how AegisML decides

This is the contribution. Everything else is plumbing around it. Read fully before editing `aegis/core/plan.py`, `utility.py`, `guards.py`, `boundaries.py`, or `analyze.py`.

**What HarmonE does here, for comparison:** collapse R² and energy into one scalar with `β`, EMA-smooth it with `γ`, and if it drops below `min_score` (or energy exceeds the threshold) pick `argmax` over three models' EMA scores. One objective, three homogeneous options, no record of why.

---

## 1. Boundaries

### Kept from HarmonE: the energy integral controller

```
thr ← thr + 0.95 · (thr_original − energy_used)
```

Genuinely elegant — underspend and your allowance grows, overspend and it shrinks. Keep it verbatim (`mode: "integral"`) so the energy arm stays comparable with the base system.

One thing to know: it has **no lower bound**. Sustained overspend drives `thr` arbitrarily negative and the energy boundary becomes permanently violated, which pins the controller into energy-pressure mode forever. HarmonE masks this with `recovery_cycles`. Clamp `thr` to `[0.1·orig, 2·orig]` and note the change — it's a small robustness fix worth a sentence in the paper.

### New: rolling boundaries for the other families

```python
def boundary(history, cfg, k):
    mu, sd = mean(history[-cfg.window:]), std(history[-cfg.window:])
    if cfg.direction == "upper":
        return None, min(mu + k*sd, cfg.hard_ceiling)
    return max(mu - k*sd, cfg.hard_floor), None
```

- **Cold start** (<`window` samples) → hard bounds only, `mode="hard"`.
- **Zero variance** (`sd < 1e-6`) → widen to `max(sd, 0.02·|mu|)`.
- **Persistence** — `persistence = 2` consecutive violations before an incident fires. HarmonE fires on a single snapshot, which combined with `recovery_cycles = 3` gives it a fire-then-freeze rhythm. Persistence is the cleaner mechanism; keep `recovery_cycles` as a per-action cooldown guard instead.

```
severity = clamp(|value − bound| / |hard_bound − bound|, 0, 1)
```

---

## 2. Guards before utility

Never rank an action that can't run. Guards are declarative in `policy.json` so the dashboard can answer "why wasn't retraining even considered?".

```python
def eligible(action, ctx):
    for g in action.guards:
        if not OPS[g.op](resolve(ctx, g.field), resolve(ctx, g.value)):
            return False, f"{g.field} {resolve(ctx, g.field)} fails {g.op} {g.value}"
    return True, None
```

`ctx` holds the snapshot plus derived facts: `active_model`, `archived_versions`, `best_version_kl`, `drift_window_rows`, `cooldown_elapsed_s.<ACTION>`, `equity_review_pending`, `serving.*`, `ema_scores`.

**Global guard.** When `equity_review_pending` is true, every model-promoting action is ineligible regardless of utility. Not a large negative weight — a hard filter. That distinction is the point of §5.

---

## 3. Utility

```
U(a) = w_acc·Q(a) − w_e·E(a) − w_l·L(a) − w_c·C(a) − w_eq·F(a) − w_r·R(a)
```

All terms in `[0,1]`, weights from the active profile.

**Q(a) — projected relief.** Not "how good is this action" but "how much of the *current* violation does it remove", weighted by how bad each violation is:

```python
Q = min(1.0, sum(severity[f] * max(0, relief(a, f)) for f in violated_families))
```

where `relief` flips sign for accuracy (positive effect = severity reduction) and takes `-effect` for every upper-direction family. Do that sign handling in **one** function with a comment, or you'll get it wrong twice.

This line is why `DRIFT_ONLY → OBSERVE`. Accuracy severity is 0, so `RETRAIN_CURRENT`'s large accuracy effect multiplies to nothing and only its energy and risk remain.

**E(a)** — action's own measured energy / most expensive action's energy. Retraining ≈ 1.0.
**L(a)** — time-to-effect: `OBSERVE` 0.0, config writes 0.05, switch 0.1, version reuse 0.2, retrain 0.9.
**C(a)** — wall-clock/operational cost, distinct from energy. Mostly tracks E but separates cleanly for retraining, which is long *and* hot.
**F(a)** — equity penalty: 0 unless the action changes the serving model while equity severity > 0; then `severity["equity"]`. `w_eq` is the largest weight in every profile — belt and braces behind the hard filter.
**R(a)** — declared operational risk from `policy.json`. Reversible and observable ⇒ low.

`argmax U`. Tie-break: lower R, then lower E, then action name. Deterministic.
Floor rule: `max U < 0` → `OBSERVE`.

### ε-greedy, relocated

HarmonE explores *before* analysis — `random.random() < α` short-circuits everything and can return the current model or an unsuitable one. In AegisML, exploration happens **after** the guard filter, over eligible candidates only:

```python
candidates = [c for c in all_candidates if c.eligible]
if rng.random() < policy.epsilon:
    chosen, exploratory = rng.choice(candidates), True
else:
    chosen, exploratory = max(candidates, key=lambda c: c.utility), False
```

Two consequences worth stating in the paper: exploration can never violate the safety filter, and `exploratory: true` decisions are excluded from policy-quality metrics so exploration doesn't contaminate the evaluation.

---

## 4. Worked example — benign drift

Snapshot: `r2 = 0.842` (lower boundary 0.79, OK), `kl_div = 0.31` (upper boundary 0.18, violated, severity 0.23), energy/latency/equity inside.

Incident: `DRIFT_ONLY`, `severity = {drift: 0.23, accuracy: 0, energy: 0, latency: 0, equity: 0}`
Profile `balanced`: `w_acc 1.0, w_e 0.8, w_l 0.4, w_c 0.3, w_eq 1.4, w_r 0.4`

| Action | Eligible | Q | E | L | C | F | R | **U** |
|---|---|---|---|---|---|---|---|---|
| OBSERVE | yes | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0.00 | **0.000** |
| LOWER_SAMPLING | yes | 0.00 | 0.00 | 0.05 | 0.00 | 0 | 0.10 | **−0.060** |
| SWITCH_MODEL:linear | yes | 0.00 | 0.00 | 0.10 | 0.02 | 0 | 0.20 | **−0.126** |
| REUSE_VERSION | no — `best_version_kl 0.82 >= 0.75` | — | — | — | — | — | — | — |
| RETRAIN_CURRENT | yes | 0.20 | 1.00 | 0.90 | 0.88 | 0 | 0.55 | **−1.88** |

Winner: `OBSERVE` (0.000).
Reason: *"drift severity 0.23 with accuracy inside its boundary; retraining's projected relief 0.20 is outweighed by measured energy cost 1.00 and risk 0.55. No archived version is close enough to reuse (KL 0.82 ≥ 0.75)."*

Compare to HarmonE on the same snapshot: KL 0.31 is below its 0.75 drift threshold, so `analyse_drift` reports no drift and nothing happens — same outcome, no reasoning, and no ability to distinguish "nothing is wrong" from "something is wrong but not worth fixing". That distinction is your contribution and this table is how you show it.

Now switch to `accuracy_first` (`w_acc 1.8, w_e 0.3, w_r 0.2`):
`U(RETRAIN) = 1.8(0.20) − 0.3(1.00) − 0.3(0.90) − 0.2(0.88) − 0.2(0.55) = −0.56` — still negative, still rejected. Correct: even an accuracy-hungry policy shouldn't retrain when accuracy isn't hurt.

Raise drift severity to 0.9 (a real regime change) and `Q` rises to 0.77: `U(RETRAIN) = 1.8(0.77) − 0.83 = +0.55` and it wins. **Demo that transition live** — it shows the policy responding to problem magnitude, not just to weights.

---

## 5. Worked example — equity violation

`equity.gap = 0.19` (ceiling 0.15, severity 0.27), `worst_r2 = 0.58` (floor 0.60, severity 0.10), aggregate `r2 = 0.84` — the model looks healthy on average while one station degrades.

Rule 1 fires: `EQUITY_VIOLATION`. The global guard sets `equity_review_pending`, making `SWITCH_MODEL`, `REUSE_VERSION`, and any promoting retrain ineligible **regardless of their utility**. Remaining: `OBSERVE`, `EQUITY_REVIEW`, `LOWER_SAMPLING`. `EQUITY_REVIEW` wins — it's the only one with a non-zero equity effect.

The line for judges: **"this is the one case where utility doesn't decide. Safety is a constraint, not a weight — a high enough accuracy score can't buy its way past it."**

This is also the scenario that justifies adding equity to a traffic-forecasting system at all: the aggregate R² is fine. HarmonE, monitoring only aggregate accuracy and energy, would see nothing wrong and keep optimising energy by switching to a cheaper model — plausibly making the worst station worse. Show that ablation. It's the cleanest argument for the whole extra metric family.

---

## 6. LEARN — outcome confidence (stretch, flagged)

```
α[a, incident_type] ← clamp(α + η·(resolved ? +1 : −1), 0.5, 1.5)    # η = 0.1
Q(a) ← α[a, incident_type] · Q(a)
```

Behind `policy.learning_enabled`, default off so the demo is reproducible. With it on, report the learning curve — "after N incidents the controller's retrain-under-drift confidence fell to 0.7" is a nice result and it's honestly derived from the outcome ledger.

Also use the ledger to **recalibrate effect vectors**: compare `projected_effect` to the observed `pre`/`post` delta and report mean absolute error per action. A calibration table in the paper ("our declared effects were accurate to ±0.08 on average") pre-empts the obvious reviewer objection that the effect vectors are made up.

---

## 7. Tests that must exist

`tests/test_plan.py`:
- `test_drift_only_rejects_retrain_under_balanced` — reproduces §4's U = −1.88
- `test_high_severity_drift_selects_retrain_under_accuracy_first`
- `test_energy_pressure_prefers_batching_over_switching_when_cached`
- `test_equity_violation_blocks_all_promoting_actions`
- `test_ineligible_actions_carry_a_guard_reason`
- `test_exploration_only_picks_eligible_actions`
- `test_exploratory_decisions_are_flagged`
- `test_decision_deterministic_across_100_runs_fixed_seed`
- `test_falls_back_to_observe_when_all_utilities_negative`
- `test_accuracy_sign_flip_handled_once` — feed an accuracy-improving effect, assert severity reduction

`tests/test_boundaries.py`:
- `test_energy_integral_matches_harmone_formula` — parity with `mape/analyse.py`
- `test_energy_threshold_clamped_below`
- `test_cold_start_uses_hard_bounds`
- `test_persistence_suppresses_single_snapshot_spike`

`tests/test_invariants.py`:
- `test_no_torch_or_pyrapl_import_in_core`
- `test_scenario_tag_never_read_in_core`
- `test_no_float_literals_in_core`
- `test_mixed_energy_backend_window_is_refused`
