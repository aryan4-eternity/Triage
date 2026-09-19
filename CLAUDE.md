# CLAUDE.md — AegisML (extending HarmonE)

Read `docs/BASE_PROJECT_AUDIT.md` before writing any code — this repo is an extension of someone else's published research artifact and you need to know what's already here. Then `ARCHITECTURE.md`. Read `docs/CONTRACTS.md` before touching any data shape, `docs/POLICY_ENGINE.md` before touching anything under `aegis/core/{plan,utility,guards,analyze,boundaries}.py`, and `docs/ENERGY.md` before touching anything that measures or reports energy.

## What this is

**HarmonE** (`mape/`, `inference.py`, `retrain.py`, MIT, © 2025 Hiya Bhatt) is a self-adaptive MLOps loop for traffic-flow regression that balances R² against physically measured CPU energy, choosing between three models.

**AegisML** (`aegis/`) extends its Plan stage: from argmax over three models to a multi-objective utility over nine heterogeneous tactics, with declarative eligibility guards, six metric families, persisted reasoning, and a safety constraint that sits outside the utility function.

Timeline is short. Optimise for a working demo plus a defensible evaluation table, not production hardening.

## Hard rules

1. **Never break a HarmonE approach.** All nine (`harmone`, `switch`, `switch+retrain`, `single-*`, `single-*+retrain`) must run unchanged at every point in the build. They are the baselines the evaluation depends on. After any change to `mape/`, `inference.py`, or `retrain.py`, run `./set_approach.sh harmone` and confirm a cycle completes.

2. **`aegis/core/**` imports no torch, no pyRAPL, no boto3, no streamlit, and nothing from `mape/`.** It is pure functions over the contracts. External effects go through `aegis/ports/` protocols implemented in `aegis/actuators/` and `aegis/energy/`. This is what keeps the contribution testable without a dataset or an Intel machine.

3. **Energy is never silently estimated.** Every reading carries its backend. Every row of `predictions.csv` carries `energy_backend`. MONITOR raises on a mixed-backend window rather than averaging. The dashboard shows a measured-vs-estimated badge. See `docs/ENERGY.md`.

4. **Every decision is explainable.** `plan()` returns the full candidate list — eligible and rejected — with per-action utility terms and, for rejected ones, the guard that eliminated them. No action is chosen without a specific human-readable `reason`. A vaguer reason string is a regression.

5. **No magic numbers in `aegis/core/`.** Weights, thresholds, effect vectors, windows, prices all live in `config/policy.json`, `config/boundaries.json`, `config/hardware.json`. If you're typing a float literal into core, stop and put it in config with a provenance comment.

6. **Determinism.** Seed numpy, random, and torch. Same snapshot + same policy + same seed ⇒ same decision. Tie-breaks explicit, never dict or set order. ε-greedy uses an injected RNG.

7. **Safety invariant:** while `equity_review_pending` is true, no code path changes the serving model. `tests/test_invariants.py` enforces it. Never weaken that test — fix the code.

8. **Measured numbers only.** Effect vectors and action energy costs come from `tools/profile_models.py` and `tools/calibrate_effects.py`, never from estimation. A placeholder is fine temporarily but must carry a `TODO(calibrate)` and be gone before any results are reported.

## Stack

Python 3.11 · pandas, numpy, scipy, scikit-learn, torch (CPU) · pydantic v2 for contracts · pyRAPL / codecarbon for energy · Streamlit for the dashboard · pytest · SQLite for the AegisML knowledge store (HarmonE's JSON/CSV files stay as they are).

## Commands

```bash
make setup                  # venv + requirements + aegis extras
make data                   # tools/synth_data.py, seeded
make train                  # tools/train_models.py
make profile                # tools/profile_models.py -> config/hardware.json
make test                   # pytest -q
make base                   # ./set_approach.sh harmone + both processes (baseline check)
make demo                   # ./set_approach.sh aegis + both processes + dashboard
make scenario S=drift_benign
make eval                   # 8 arms x 8 scenarios x 20 seeds -> results.md
```

`AEGIS_ENERGY=rapl|codecarbon|estimator|auto` selects the energy backend. Nothing else changes.

## Working agreement

- One task at a time from `docs/TASKBOARD.md`. Each names its files and its acceptance test. Don't start a task whose dependencies aren't green.
- Tests first for anything in `aegis/core/`. They're pure functions; there's no excuse.
- After each task: `make test`, then `make base`, then tick the checkbox.
- Small, reviewable diffs. Prefer editing an existing file over adding one.
- Changing a contract means editing `docs/CONTRACTS.md` in the same commit.
- Changes to `mape/` are **bug fixes only**, behaviour-preserving except where a fix is the point (BUG-1). Every such change gets a test and a before/after measurement.
- Don't build the dashboard before the headless scenario runner passes.

## When stuck or the request is ambiguous

Say so and give me two concrete options with trade-offs. Don't invent a third metric family, a new action type, or an extra config file to route around ambiguity — those leak into the demo and the evaluation table and they're expensive to remove later.

## Style

- Docstrings on every public `aegis/core/` function stating the invariant it preserves. Where you reuse a HarmonE formula, cite the source file in the docstring.
- Type-hint `aegis/core/` and `aegis/ports/`; `mypy --strict` clean on those two. Elsewhere, don't care.
- Structured log dicts, one line per MAPE stage: `stage, snapshot_id, incident_type, chosen_action, reason, energy_backend`.
- No emoji in new code or logs. HarmonE's existing emoji prints stay — don't churn files you're not otherwise touching.

## Things that will waste your time (don't)

- Porting to AWS. RAPL isn't available there, so it trades the project's strongest claim for a diagram. It's a flagged stretch goal, nothing more.
- Rewriting `mape/` into `aegis/`. You'd destroy your own baselines.
- Reinforcement learning. The transparent utility scorer is the contribution.
- Inventing demographic protected attributes for traffic data. Equity here is per-station performance disparity. See `ARCHITECTURE.md` §0.
- Comparing your post-bug-fix energy numbers against HarmonE's published ones. Different measurement, not a result.
- A React dashboard. Streamlit.
