"""
tools/synth_data.py — Synthetic PEMS-like traffic flow generator.

Generates deterministic, reproducible flow time-series with daily and weekly
seasonality, Gaussian noise, and injectable regime shifts.

Usage:
    python3 tools/synth_data.py --rows 40000 --seed 7 --stations 8
    python3 tools/synth_data.py --rows 40000 --seed 7 --stations 8 \
        --regime 20000:24000:1.4:60

Output:
    data/pems/flow_data_train.csv   (first 10% of rows by time step)
    data/pems/flow_data_test.csv    (remaining 90%)

Both files have columns: flow, station_id
(columns match what Triage's inference.py and monitor.py expect)
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "pems"


@dataclass
class Regime:
    """A regime shift injected between start_row and end_row (time-step indices)."""
    start: int
    end: int
    scale: float   # multiplicative scale on base signal amplitude
    shift: float   # additive offset in original flow units


def parse_regime(spec: str) -> Regime:
    """Parse 'start:end:scale:shift' string into a Regime."""
    parts = spec.split(":")
    if len(parts) != 4:
        raise ValueError(
            f"Regime spec must be 'start:end:scale:shift', got: {spec!r}"
        )
    return Regime(
        start=int(parts[0]),
        end=int(parts[1]),
        scale=float(parts[2]),
        shift=float(parts[3]),
    )


def generate_flow(
    rows: int,
    seed: int,
    stations: int,
    regimes: Optional[List[Regime]] = None,
) -> pd.DataFrame:
    """
    Generate synthetic traffic flow data.

    Signal model per time-step t (before per-station noise):
        base_signal(t) = 400
                       + 120 * sin(2π·t / 288)      # daily cycle (288 × 5-min = 1 day)
                       + 40  * sin(2π·t / 2016)     # weekly cycle
                       + regime_delta(t)

    Each station draws an independent uniform base offset and per-step
    Gaussian noise so that per-station R² values differ meaningfully.

    Returns:
        DataFrame with columns [flow, station_id], rows sorted by time then station.
        Total rows = rows * stations.
    """
    rng = np.random.default_rng(seed)

    # Signal constants
    BASE = 400.0
    A_DAY = 120.0
    A_WEEK = 40.0
    SIGMA = 25.0
    SLOTS_PER_DAY = 288.0
    SLOTS_PER_WEEK = 2016.0

    t = np.arange(rows, dtype=np.float64)
    base_signal = (
        BASE
        + A_DAY * np.sin(2.0 * np.pi * t / SLOTS_PER_DAY)
        + A_WEEK * np.sin(2.0 * np.pi * t / SLOTS_PER_WEEK)
    )

    # Apply regime shifts
    if regimes:
        for reg in regimes:
            s = max(0, reg.start)
            e = min(rows, reg.end)
            if s < e:
                base_signal[s:e] = base_signal[s:e] * reg.scale + reg.shift

    # Build per-station arrays, then interleave in time-major order
    station_ids = [f"ST_{i + 1:02d}" for i in range(stations)]
    all_flow: list[float] = []
    all_station: list[str] = []
    all_t: list[int] = []

    for station_id in station_ids:
        station_offset = float(rng.uniform(-30.0, 30.0))
        noise = rng.normal(0.0, SIGMA, size=rows)
        flow = np.clip(base_signal + station_offset + noise, a_min=0.0, a_max=None)
        all_flow.extend(flow.tolist())
        all_station.extend([station_id] * rows)
        all_t.extend(range(rows))

    df = pd.DataFrame({"t": all_t, "station_id": all_station, "flow": all_flow})
    df = df.sort_values(["t", "station_id"]).reset_index(drop=True)
    return df[["flow", "station_id"]]


def split_and_save(df: pd.DataFrame, train_ratio: float = 0.1) -> None:
    """Split df into train/test by time and write CSVs."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    n = len(df)
    split = int(n * train_ratio)
    train_df = df.iloc[:split].reset_index(drop=True)
    test_df = df.iloc[split:].reset_index(drop=True)

    train_path = DATA_DIR / "flow_data_train.csv"
    test_path = DATA_DIR / "flow_data_test.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Written {len(train_df):,} rows  →  {train_path}")
    print(f"Written {len(test_df):,} rows  →  {test_path}")

    # Sanity checks
    t = pd.read_csv(train_path)
    v = pd.read_csv(test_path)
    assert list(t.columns) == ["flow", "station_id"], "train columns mismatch"
    assert list(v.columns) == ["flow", "station_id"], "test columns mismatch"
    assert t["flow"].min() >= 0.0, "negative flow values in train"
    assert v["flow"].min() >= 0.0, "negative flow values in test"
    print(f"Train flow range: [{t['flow'].min():.1f}, {t['flow'].max():.1f}]")
    print(f"Test  flow range: [{v['flow'].min():.1f}, {v['flow'].max():.1f}]")
    print(f"Stations in test: {sorted(v['station_id'].unique().tolist())}")
    print("Sanity checks passed.")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic PEMS-like traffic flow data."
    )
    parser.add_argument(
        "--rows", type=int, default=40000,
        help="Number of time steps (default: 40000)"
    )
    parser.add_argument(
        "--seed", type=int, default=7,
        help="Random seed (default: 7)"
    )
    parser.add_argument(
        "--stations", type=int, default=8,
        help="Number of sensor stations (default: 8)"
    )
    parser.add_argument(
        "--regime", type=str, default=None,
        help=(
            "Inject a regime shift as 'start:end:scale:shift'. "
            "E.g. --regime 20000:24000:1.4:60 produces a detectable KL spike "
            "starting at time-step 20000."
        ),
    )
    parser.add_argument(
        "--train-ratio", type=float, default=0.1,
        help="Fraction of rows for the training split (default: 0.1)"
    )
    args = parser.parse_args(argv)

    regimes: list[Regime] = []
    if args.regime:
        reg = parse_regime(args.regime)
        regimes.append(reg)
        print(
            f"Regime shift: rows {reg.start}–{reg.end}, "
            f"scale={reg.scale}, shift={reg.shift}"
        )

    total_rows = args.rows * args.stations
    print(
        f"Generating {args.rows:,} time steps × {args.stations} stations "
        f"= {total_rows:,} total rows  (seed={args.seed}) …"
    )

    df = generate_flow(
        rows=args.rows,
        seed=args.seed,
        stations=args.stations,
        regimes=regimes if regimes else None,
    )

    print(f"Generated {len(df):,} rows.")
    split_and_save(df, train_ratio=args.train_ratio)
    print("Done.")


if __name__ == "__main__":
    main()
