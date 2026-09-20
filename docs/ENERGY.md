# ENERGY.md — measuring energy honestly

Energy is Triage's primary asset and the thing that makes this project more than another MLOps pipeline. It's also the part most likely to embarrass you if handled loosely. This file is short on purpose: follow it exactly.

---

## The constraint

`pyRAPL` reads Intel RAPL through `/sys/class/powercap/intel-rapl`. That requires:

- **Linux** — not macOS, not Windows, not WSL2 (no powercap passthrough)
- **Intel CPU** — not AMD (which exposes RAPL differently and isn't supported by pyRAPL), not Apple Silicon
- **Read permission** — `sudo chmod -R a+r /sys/class/powercap/intel-rapl`, needed after each boot
- **Bare metal or a VM with powercap exposed** — not Lambda, not Fargate, not standard EC2

Check in one line:
```bash
ls /sys/class/powercap/intel-rapl && python3 -c "import pyRAPL; pyRAPL.setup(); print('ok')"
```

---

## If you don't have that machine

In order of preference:

1. **College lab machine.** Most institutional desktops are Intel + Linux. This is almost certainly your best path and worth an afternoon of asking.
2. **Dual-boot or a live USB** on an Intel laptop. RAPL works from a live session if you set the permission.
3. **A bare-metal cloud instance** — Equinix Metal, Hetzner dedicated, OVH. Cheap hourly, real RAPL. AWS `*.metal` instances also work but cost more.
4. **CodeCarbon.** Uses RAPL where available; otherwise falls back to its own hardware model. Report as estimated.
5. **Time × TDP estimator.** Last resort, always labelled.

**Do not silently substitute.** A run with estimated energy is a different experiment from one with measured energy, and the paper must say which produced each number.

---

## The abstraction

```python
class EnergyMeter(Protocol):
    backend: str          # "rapl" | "codecarbon" | "estimator"
    measured: bool        # True only for rapl
    def measure(self, label: str) -> ContextManager[Reading]: ...
```

`Reading = {micro_joules, seconds, backend, measured}`.

Selection via `AEGIS_ENERGY=rapl|codecarbon|estimator|auto` (default `auto`: try each in order, use the first that constructs). Log the selected backend once at startup, loudly.

Rules enforced in code, not just in prose:

1. Every row in `predictions.csv` carries `energy_backend`.
2. MONITOR **refuses to score a window containing more than one backend value**. Raise, don't average.
3. The evaluation harness refuses to compare arms whose runs used different backends.
4. The dashboard shows a persistent badge: `Energy: MEASURED (RAPL)` or `Energy: ESTIMATED (CodeCarbon fallback)`.

That last one costs you two lines and buys you the ability to demo on a laptop without anyone thinking you claimed more than you did.

---

## Calibration

`tools/profile_models.py`, run once per machine, after the model-cache fix (BUG-3):

- 1000 warm-cache inferences per model, 3 repeats, report median
- µJ/inference, ms/inference, R² on a held-out slice
- Same for `batch_size ∈ {1, 2, 4, 8}` — this gives you `BATCH_INFERENCE`'s real effect vector
- One full retrain per model type, measured — this gives you `RETRAIN_CURRENT`'s real energy cost

Output goes to `config/hardware.json` with CPU model, OS, backend, and date. `E_m` and `E_M` in `thresholds.json` come from this run, not from Triage's hardcoded `25000`.

Re-run it if you change machines. Numbers from two machines never go in the same table.

---

## Idle power, the subtlety that catches people

RAPL reports **package energy**, which includes idle draw. A 4ms inference on an idle-heavy package attributes baseline power to your model. With a 0.15s sleep between inferences in the base loop, that's a large fraction.

Measure idle separately — 60s of nothing, µJ/s — and report both gross and net (`gross − idle × elapsed`) per-inference energy. Use net for model comparisons, gross for total-system claims.

If you skip this, your three models will look more similar than they are and a reviewer who knows RAPL will find it immediately. It's ten lines in the profiler.

---

## What to say about carbon

If you want a carbon number: `gCO₂ = joules / 3.6e6 × grid_intensity_gCO2_per_kWh`. Use a published regional average, state the source and date, and call it an estimate derived from measured energy. Never a live carbon-intensity API in the demo path — one more network dependency that can fail on stage, for a number nobody will question.
