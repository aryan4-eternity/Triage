"""
dashboard/app.py — AegisML Streamlit dashboard.

READ-ONLY. Never writes to knowledge/. A third writer in a file-mediated
system is how you get corrupted state mid-demo.

Sections:
  - Energy backend badge (MEASURED / ESTIMATED)
  - Six metric cards with boundary bands
  - Incident timeline
  - Candidate table with rejected actions + guard reasons (the demo differentiator)
  - Scenario buttons (run_scenario headlessly, live update)
  - Weight-profile dropdown (re-decides on current snapshot live)

Run:
    streamlit run dashboard/app.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AegisML",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── helpers ───────────────────────────────────────────────────────────────────

def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _load_predictions() -> pd.DataFrame:
    p = ROOT / "knowledge" / "predictions.csv"
    if not p.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except Exception:
        return pd.DataFrame()


def _load_mape_info() -> dict:
    return _load_json(ROOT / "knowledge" / "mape_info.json")


def _load_thresholds() -> dict:
    return _load_json(ROOT / "knowledge" / "thresholds.json")


def _active_model() -> str:
    p = ROOT / "knowledge" / "model.csv"
    try:
        return p.read_text().strip()
    except Exception:
        return "unknown"


def _energy_backend() -> str:
    df = _load_predictions()
    if df.empty or "energy_backend" not in df.columns:
        return "estimator"
    backends = df["energy_backend"].dropna().unique()
    return backends[0] if len(backends) == 1 else "mixed"


def _equity_review() -> dict:
    return _load_json(ROOT / "knowledge" / "review.json")


def _get_latest_snapshot():
    """Try to produce a live snapshot from the current predictions window."""
    try:
        from aegis.core.monitor import produce_snapshot, read_window
        window_df, _ = read_window(
            ROOT / "knowledge" / "predictions.csv", window_size=1200
        )
        if window_df.empty:
            return None
        return produce_snapshot(window_df, _active_model())
    except Exception:
        return None


def _run_scenario_live(name: str, profile: str) -> dict:
    from tools.run_scenario import run_scenario
    return run_scenario(name, seed=42, profile=profile, verbose=False)


# ── sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🛡️ AegisML")
    st.caption("Multi-objective MLOps controller")

    # Energy backend badge
    backend = _energy_backend()
    if backend == "rapl":
        st.success("⚡ Energy: MEASURED (RAPL)")
    elif backend == "mixed":
        st.error("⚠️ Energy: MIXED BACKENDS — run is invalid")
    else:
        st.warning(f"📊 Energy: ESTIMATED ({backend})")

    st.divider()

    # Active model
    active_model = _active_model()
    st.metric("Active Model", active_model.upper())

    # Equity review status
    review = _equity_review()
    if review.get("equity_review_pending"):
        st.error(f"🔴 EQUITY REVIEW PENDING\n\nStation: {review.get('worst_station','?')}  "
                 f"Gap: {review.get('gap', 0):.3f}")

    st.divider()

    # Weight profile selector
    profile = st.selectbox(
        "Weight profile",
        ["balanced", "energy_first", "accuracy_first"],
        index=0,
    )

    st.divider()

    # Scenario buttons
    st.subheader("Run scenario")
    scenarios = [
        "normal", "drift_benign", "recoverable_drift", "model_degradation",
        "energy_pressure", "latency_pressure", "equity_shift", "weight_profile",
    ]
    selected_scenario = st.selectbox("Scenario", scenarios)
    run_btn = st.button("▶ Run scenario (headless)")

    st.divider()
    auto_refresh = st.checkbox("Auto-refresh (10s)", value=False)


# ── main content ──────────────────────────────────────────────────────────────

st.title("AegisML — Live Decision Dashboard")

# ── Section 1: Six metric cards ───────────────────────────────────────────────
st.subheader("Live Metrics")

snapshot = _get_latest_snapshot()
mape_info = _load_mape_info()
thresholds = _load_thresholds()

if snapshot:
    c1, c2, c3, c4, c5, c6 = st.columns(6)

    def _delta_color(val, boundary, direction="upper"):
        if direction == "upper":
            return "inverse" if val > boundary else "normal"
        return "inverse" if val < boundary else "normal"

    with c1:
        r2 = snapshot.accuracy.r2
        st.metric("R² Score", f"{r2:.4f}",
                  delta=f"EMA {snapshot.accuracy.ema_score:.4f}")
    with c2:
        kl = snapshot.drift.kl_div
        kl_ceil = thresholds.get("max_energy", 0.75)
        st.metric("KL Divergence", f"{kl:.4f}",
                  delta=f"PSI {snapshot.drift.psi:.4f}")
    with c3:
        p95 = snapshot.latency.p95_ms
        st.metric("p95 Latency (ms)", f"{p95:.2f}",
                  delta=f"p50 {snapshot.latency.p50_ms:.2f}")
    with c4:
        uj = snapshot.energy.uj_per_inference
        st.metric("Energy µJ/inf", f"{uj:.0f}",
                  delta=f"norm {snapshot.energy.normalized:.4f}")
    with c5:
        gap = snapshot.equity.gap
        st.metric("Equity Gap (R²)", f"{gap:.4f}",
                  delta=f"worst {snapshot.equity.worst_station}={snapshot.equity.worst_r2:.3f}")
    with c6:
        st.metric("Window Rows", f"{snapshot.window.rows:,}",
                  delta=f"backend={snapshot.window.energy_backend}")

    # Boundary bands as expander
    with st.expander("Boundary details"):
        try:
            from aegis.core.analyze import analyze
            incident = analyze(snapshot, [])
            rows = []
            for br in incident.boundaries:
                rows.append({
                    "Metric":     br.metric,
                    "Value":      round(br.value, 4),
                    "Lower":      round(br.lower, 4) if br.lower is not None else "—",
                    "Upper":      round(br.upper, 4) if br.upper is not None else "—",
                    "Mode":       br.mode,
                    "Violated":   "⚠️ YES" if br.violated else "✅ no",
                    "Severity":   round(br.severity, 4),
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
        except Exception as exc:
            st.warning(f"Could not compute boundaries: {exc}")
else:
    st.info("No live data yet — start inference.py to populate predictions.csv.")

# ── Section 2: EMA scores ─────────────────────────────────────────────────────
st.subheader("Model EMA Scores")
ema = mape_info.get("ema_scores", {})
if ema:
    cols = st.columns(len(ema))
    for col, (model, score) in zip(cols, ema.items()):
        col.metric(
            model.upper(),
            f"{score:.4f}",
            delta="active" if model == active_model else None,
        )

# ── Section 3: Predictions stream ────────────────────────────────────────────
st.subheader("Recent Predictions")
df = _load_predictions()
if not df.empty:
    st.line_chart(
        df.tail(200)[["true_value", "predicted_value"]],
        use_container_width=True,
    )
    with st.expander("Raw data (last 20 rows)"):
        st.dataframe(df.tail(20), use_container_width=True)
else:
    st.info("predictions.csv is empty or not found.")

# ── Section 4: Scenario runner + candidate table ──────────────────────────────
st.subheader("Scenario Decision & Candidate Table")
st.caption("The candidate table — rejected actions with guard reasons — is the demo differentiator.")

if run_btn:
    with st.spinner(f"Running {selected_scenario} with profile={profile}…"):
        result = _run_scenario_live(selected_scenario, profile)
    st.session_state["last_scenario_result"] = result
    st.session_state["last_scenario_name"]   = selected_scenario
    st.session_state["last_profile"]         = profile

if "last_scenario_result" in st.session_state:
    res       = st.session_state["last_scenario_result"]
    scenario  = st.session_state.get("last_scenario_name", "")
    prof_used = st.session_state.get("last_profile", "balanced")

    # Incident summary
    col_a, col_b = st.columns(2)
    with col_a:
        inc_type = res.get("incident_type", "NONE")
        color    = {"NONE": "🟢", "DRIFT_ONLY": "🟡", "ENERGY_PRESSURE": "🟡",
                    "LATENCY_PRESSURE": "🟡", "EQUITY_VIOLATION": "🔴",
                    "MODEL_DEGRADATION": "🔴", "COMPOSITE": "🔴"}.get(inc_type, "⚪")
        st.metric("Incident", f"{color} {inc_type}")
    with col_b:
        chosen = res.get("results", [{}])[0].get("chosen_action", "?")
        st.metric("Chosen Action", chosen)

    # Candidate table — the differentiator
    candidates = res.get("candidates", [])
    if candidates:
        rows = []
        for c in candidates:
            if c.eligible:
                terms = c.terms or {}
                rows.append({
                    "Action":    c.action,
                    "Eligible":  "✅ YES",
                    "Utility":   round(c.utility or 0, 4),
                    "Q":         round(terms.get("Q", 0), 3),
                    "E":         round(terms.get("E", 0), 3),
                    "L":         round(terms.get("L", 0), 3),
                    "C":         round(terms.get("C", 0), 3),
                    "F":         round(terms.get("F", 0), 3),
                    "R":         round(terms.get("R", 0), 3),
                    "Guard":     "—",
                    "Reason":    (c.reason or "")[:80],
                })
            else:
                rows.append({
                    "Action":    c.action,
                    "Eligible":  "❌ NO",
                    "Utility":   "—",
                    "Q": "—", "E": "—", "L": "—", "C": "—", "F": "—", "R": "—",
                    "Guard":     (c.rejected_by_guard or "")[:60],
                    "Reason":    "rejected",
                })

        cand_df = pd.DataFrame(rows)
        # Highlight chosen action
        def _highlight(row):
            if row["Action"] == chosen:
                return ["background-color: #1a3a1a"] * len(row)
            if row["Eligible"] == "❌ NO":
                return ["color: #888"] * len(row)
            return [""] * len(row)

        st.dataframe(
            cand_df.style.apply(_highlight, axis=1),
            use_container_width=True,
            height=min(400, 35 * (len(rows) + 1)),
        )

        st.caption(
            f"Scenario: **{scenario}** | Profile: **{prof_used}** | "
            f"{sum(1 for c in candidates if c.eligible)} eligible, "
            f"{sum(1 for c in candidates if not c.eligible)} rejected by guard"
        )

# ── Section 5: Incident timeline ─────────────────────────────────────────────
st.subheader("Incident Timeline (from SQLite store)")
try:
    from aegis.store import AegisStore
    store    = AegisStore()
    decisions = store.all_decisions()
    if decisions:
        timeline_rows = []
        for d in decisions[-20:]:   # last 20
            timeline_rows.append({
                "Decision ID":    d.decision_id[:12],
                "Chosen Action":  d.chosen_action,
                "Profile":        d.policy_profile,
                "Exploratory":    "🎲" if d.exploratory else "",
                "Reason":         (d.reason or "")[:70],
            })
        st.dataframe(pd.DataFrame(timeline_rows), use_container_width=True)
    else:
        st.info("No decisions recorded yet — run the aegis approach to populate the store.")
except Exception as exc:
    st.warning(f"Could not load store: {exc}")

# ── Section 6: Energy log ─────────────────────────────────────────────────────
with st.expander("Controller energy overhead (mape_log.csv)"):
    log_path = ROOT / "knowledge" / "mape_log.csv"
    if log_path.exists():
        log_df = pd.read_csv(log_path)
        if not log_df.empty:
            st.dataframe(log_df.tail(20), use_container_width=True)
            total_j = log_df["energy_joules"].sum()
            st.caption(f"Total controller energy: {total_j:.6f} J  "
                       f"({total_j * 1e6:.0f} µJ)")
    else:
        st.info("No mape_log.csv yet.")

# ── Auto-refresh ──────────────────────────────────────────────────────────────
if auto_refresh:
    time.sleep(10)
    st.rerun()
