# NOTICE

## Base system: HarmonE

AegisML extends **HarmonE**, a self-adaptive MLOps loop for traffic-flow
regression that balances R² against physically measured CPU energy.

```
MIT License
Copyright (c) 2025 Hiya Bhatt
```

The following components are HarmonE's work, reproduced and lightly modified
under the MIT licence:

- `mape/` — the MAPE-K loop (monitor, analyse, plan, execute)
- `inference.py` — the inference loop and pyRAPL energy instrumentation (base)
- `retrain.py` — model retraining on drift windows
- `tools/train_models.py`, `tools/store_pems.py`, `tools/induce_drift.py`
- `knowledge/` file schema (mape_info.json, thresholds.json, model.csv)
- The nine approach configurations in `approach.conf` / `set_approach.sh`

Bug fixes applied to the base (T1.1 version-reuse, T1.2 model-cache,
T1.3 energy calibration, T1.4 schema/scaler/paths) are documented in
`docs/BASE_PROJECT_AUDIT.md` and measured with before/after results.

## AegisML contribution

The following is original work extending HarmonE:

- `aegis/` — the AegisML control plane (monitor, analyze, plan, execute, learn)
- `aegis/energy/` — portable energy-backend abstraction
- `aegis/core/` — multi-objective utility engine, declarative guards, outcome ledger
- `aegis/actuators/` — actuator implementations
- `aegis/store.py` — SQLite knowledge store
- `dashboard/app.py` — Streamlit dashboard with candidate table
- `tools/synth_data.py` — deterministic synthetic data generator
- `tools/profile_models.py`, `tools/calibrate_effects.py`, `tools/run_scenario.py`
- `config/` — policy, boundaries, hardware configuration
- `tests/` — test suite
- `docs/` — design documentation

## Citation

If citing HarmonE, use the reference from the repository (not the arXiv ID
quoted in earlier planning materials — that ID was hallucinated and does not
correspond to a real preprint). Retrieve the correct citation from:
https://github.com/HiyaBhatt/HarmonE

## Licence

AegisML additions are released under the MIT Licence.
See `LICENSE` for the original HarmonE MIT licence (Hiya Bhatt, 2025).
