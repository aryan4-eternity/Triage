# AegisML Makefile
# Works on Linux/macOS. On Windows, use Git Bash or WSL.
# All paths relative to repo root.

PYTHON      ?= python3
VENV        ?= .venv
PIP         ?= $(VENV)/bin/pip
PYTEST      ?= $(VENV)/bin/pytest
STREAMLIT   ?= $(VENV)/bin/streamlit
AEGIS_ENERGY ?= auto

# ── Setup ──────────────────────────────────────────────────────────────────────
.PHONY: setup
setup:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "Setup complete. Activate with: source $(VENV)/bin/activate"

# ── Data ───────────────────────────────────────────────────────────────────────
.PHONY: data
data:
	$(PYTHON) tools/synth_data.py --rows 40000 --seed 7 --stations 8
	@echo "Synthetic data written to data/pems/"

# ── Training ───────────────────────────────────────────────────────────────────
.PHONY: train
train:
	$(PYTHON) tools/train_models.py
	@echo "Models trained. Scaler saved to artifacts/scaler.pkl"

# ── Profiling ──────────────────────────────────────────────────────────────────
.PHONY: profile
profile:
	AEGIS_ENERGY=$(AEGIS_ENERGY) $(PYTHON) tools/profile_models.py
	@echo "Profile written to config/hardware.json"

.PHONY: calibrate
calibrate: profile
	$(PYTHON) tools/calibrate_effects.py
	@echo "Effect vectors updated in config/policy.json"

# ── Tests ──────────────────────────────────────────────────────────────────────
.PHONY: test
test:
	AEGIS_ENERGY=estimator $(PYTEST) tests/ -v --tb=short
	@echo "Tests complete"

# ── Baseline check ─────────────────────────────────────────────────────────────
# Verifies all HarmonE approaches still work after any change.
# Run inference.py and mape/manage.py for one MAPE cycle, then check output.
.PHONY: base
base:
	@echo "Setting approach to harmone..."
	bash set_approach.sh harmone
	@echo "Running HarmonE baseline check (30s)..."
	AEGIS_ENERGY=estimator timeout 30 $(PYTHON) inference.py & \
	sleep 5 && AEGIS_ENERGY=estimator timeout 25 $(PYTHON) mape/manage.py; \
	wait
	@echo "Baseline check complete. Check knowledge/predictions.csv for new rows."

# ── Scenarios ──────────────────────────────────────────────────────────────────
.PHONY: scenario
scenario:
	$(PYTHON) tools/run_scenario.py --scenario $(S)

.PHONY: all-scenarios
all-scenarios:
	$(PYTHON) tools/run_scenario.py --all

# ── Demo (AegisML approach) ─────────────────────────────────────────────────────
.PHONY: demo
demo:
	@echo "Setting approach to aegis..."
	bash set_approach.sh aegis
	@echo "Starting inference loop in background..."
	AEGIS_ENERGY=$(AEGIS_ENERGY) $(PYTHON) inference.py &
	@echo "Starting AegisML controller..."
	AEGIS_ENERGY=$(AEGIS_ENERGY) $(PYTHON) mape/manage.py

# ── Dashboard ──────────────────────────────────────────────────────────────────
.PHONY: dashboard
dashboard:
	AEGIS_ENERGY=$(AEGIS_ENERGY) $(STREAMLIT) run dashboard/app.py

# ── Evaluation ─────────────────────────────────────────────────────────────────
.PHONY: eval
eval:
	$(PYTHON) tools/evaluate.py --seeds 20 --output results.md
	@echo "Results written to results.md"

.PHONY: eval-quick
eval-quick:
	$(PYTHON) tools/evaluate.py --seeds 3 --output results.md
	@echo "Quick eval (3 seeds) written to results.md"

# ── Cleanup ────────────────────────────────────────────────────────────────────
.PHONY: clean
clean:
	bash cleanup.sh
	@echo "Knowledge files reset"

.PHONY: clean-all
clean-all: clean
	rm -rf $(VENV) __pycache__ **/__pycache__ .pytest_cache
	rm -f knowledge/aegis.sqlite

# ── Help ───────────────────────────────────────────────────────────────────────
.PHONY: help
help:
	@echo ""
	@echo "AegisML Makefile targets:"
	@echo "  make setup          — create venv and install deps"
	@echo "  make data           — generate synthetic PEMS data"
	@echo "  make train          — train all three models"
	@echo "  make profile        — profile model energy/latency"
	@echo "  make calibrate      — calibrate effect vectors"
	@echo "  make test           — run pytest suite"
	@echo "  make base           — verify HarmonE baselines still work"
	@echo "  make scenario S=<x> — run a named scenario headlessly"
	@echo "  make all-scenarios  — run all 8 scenarios"
	@echo "  make demo           — start inference + AegisML controller"
	@echo "  make dashboard      — start Streamlit dashboard"
	@echo "  make eval           — full evaluation (20 seeds)"
	@echo "  make eval-quick     — quick evaluation (3 seeds)"
	@echo "  make clean          — reset knowledge files"
	@echo ""
	@echo "Environment variables:"
	@echo "  AEGIS_ENERGY=rapl|codecarbon|estimator|auto (default: auto)"
	@echo "  S=<scenario_name>   — for make scenario"
	@echo ""
