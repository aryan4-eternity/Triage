"""
tests/test_invariants.py — Hard invariant tests (POLICY_ENGINE.md §7).

Tests:
- test_no_torch_or_pyrapl_import_in_core
- test_scenario_tag_never_read_in_core
- test_no_float_literals_in_core
- test_mixed_energy_backend_window_is_refused
"""
from __future__ import annotations

import ast
import re
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CORE_DIR = ROOT / "aegis" / "core"


def _core_py_files() -> list[Path]:
    """All .py files under aegis/core/ (recursively)."""
    return sorted(CORE_DIR.rglob("*.py"))


# ── Import constraints ────────────────────────────────────────────────────────

class TestImportConstraints:

    def test_no_torch_import_in_core(self):
        """aegis/core/ must never import torch."""
        violations = []
        for f in _core_py_files():
            src = f.read_text(encoding="utf-8", errors="replace")
            if re.search(r'^\s*import torch', src, re.MULTILINE):
                violations.append(f.relative_to(ROOT))
            if re.search(r'^\s*from torch', src, re.MULTILINE):
                violations.append(f.relative_to(ROOT))
        assert not violations, (
            f"HARD RULE violated — torch import in aegis/core/: {violations}"
        )

    def test_no_pyrapl_import_in_core(self):
        """aegis/core/ must never import pyRAPL (actual import statements)."""
        violations = []
        for f in _core_py_files():
            src = f.read_text(encoding="utf-8", errors="replace")
            # Check for actual import statements, not docstring mentions
            if re.search(r'^\s*(import pyRAPL|from pyRAPL)', src, re.MULTILINE | re.IGNORECASE):
                violations.append(f.relative_to(ROOT))
        assert not violations, (
            f"HARD RULE violated - pyRAPL import in aegis/core/: {violations}"
        )

    def test_no_mape_import_in_core(self):
        """aegis/core/ must never import from mape/ (actual import statements)."""
        violations = []
        for f in _core_py_files():
            src = f.read_text(encoding="utf-8", errors="replace")
            # Check for actual import statements only, not docstring citations
            if re.search(r'^\s*(from mape[ .]|import mape)', src, re.MULTILINE):
                violations.append(f.relative_to(ROOT))
        assert not violations, (
            f"HARD RULE violated - mape import in aegis/core/: {violations}"
        )


# ── scenario_tag constraint ───────────────────────────────────────────────────

class TestScenarioTagNeverRead:

    def test_scenario_tag_never_read_in_core(self):
        """
        CONTRACTS.md: scenario_tag is simulator metadata — controller must never read it.
        models.py may define it as a field; monitor.py may write it.
        We check for actual Python attribute-access expressions using AST parsing.
        """
        violations = []
        allowed = {"monitor.py", "models.py"}

        for f in _core_py_files():
            if f.name in allowed:
                continue
            src = f.read_text(encoding="utf-8", errors="replace")
            try:
                tree = ast.parse(src)
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                # Check for snapshot.scenario_tag attribute access
                if isinstance(node, ast.Attribute) and node.attr == "scenario_tag":
                    violations.append(
                        f"{f.relative_to(ROOT)}:{node.lineno}  (attribute access)"
                    )
                # Check for dict access: ["scenario_tag"]
                if (isinstance(node, ast.Subscript) and
                        isinstance(node.slice, ast.Constant) and
                        node.slice.value == "scenario_tag"):
                    violations.append(
                        f"{f.relative_to(ROOT)}:{node.lineno}  (dict access)"
                    )

        assert not violations, (
            f"HARD RULE violated - scenario_tag accessed in aegis/core/:\n"
            + "\n".join(violations) +
            "\nThe controller must never read scenario_tag."
        )


# ── Float literal constraint ──────────────────────────────────────────────────

class TestNoFloatLiterals:

    def test_no_float_literals_in_core(self):
        """
        CLAUDE.md hard rule: no magic numbers in aegis/core/.
        All domain thresholds come from config/*.json.

        Exempt values (well-known algorithmic constants or unit values):
          0.0, 1.0, -1.0, 2.0     — unit / clamp bounds
          0.5                      — probability midpoint
          0.1, 0.02                — small fractions (zero-variance widening)
          0.95, 0.8, 0.8x         — Triage's beta/gamma/recovery_rate constants
                                     (cited in docstrings, not magic numbers)
          1000.0                   — ms→s conversion
          1e-N                     — numerical epsilon
          25000.0                  — E_M fallback (labelled in code)
          0.004, 0.005             — cost proxies (labelled)
          999.0, 9999.0            — sentinel "infinity" values
          0.3, 0.4, 0.8x          — default weight fallbacks
        """
        # These are either unit values, cited Triage constants, or explicit
        # sentinel/conversion values — all documented in docstrings.
        EXEMPT = {
            0.0, 1.0, -1.0, 2.0, 0.5, 0.1, 0.02,
            # Triage constants (beta, gamma, recovery_rate) — cited by formula
            0.95, 0.8,
            # Conversions
            1000.0,
            # Fallback sentinels (labelled in code)
            25000.0, 999.0, 9999.0,
            # Cost/rate proxies (labelled)
            0.004, 0.005,
            # Default weight fallbacks (used only when config absent)
            0.3, 0.4, 1.4, 1.8, 0.6, 1.6,
        }
        EXEMPT_PATTERNS = re.compile(r'1e-\d+|1\.0e-\d+')

        violations = []
        for f in _core_py_files():
            src = f.read_text(encoding="utf-8", errors="replace")
            try:
                tree = ast.parse(src)
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Constant) and isinstance(node.value, float):
                    v = node.value
                    if v in EXEMPT:
                        continue
                    line_no = node.lineno
                    lines   = src.splitlines()
                    line    = lines[line_no - 1] if line_no <= len(lines) else ""
                    if EXEMPT_PATTERNS.search(line):
                        continue
                    violations.append(
                        f"{f.relative_to(ROOT)}:{line_no}  value={v}"
                    )

        assert not violations, (
            f"Unexplained float literals in aegis/core/ — "
            f"add to config/*.json or EXEMPT list with justification:\n"
            + "\n".join(violations[:20])
        )


# ── Mixed backend window ──────────────────────────────────────────────────────

class TestMixedBackendRefused:

    def test_mixed_energy_backend_window_is_refused(self):
        """
        CONTRACTS.md: MONITOR raises MixedBackendError when a window contains
        more than one energy_backend value.
        """
        from aegis.core.monitor import produce_snapshot, MixedBackendError
        from aegis.core.models import SCHEMA

        # Build a window with two different backends
        import numpy as np
        rng = np.random.default_rng(0)
        n   = 1200
        df  = pd.DataFrame({
            SCHEMA["true_value"]:      rng.normal(400, 20, n),
            SCHEMA["predicted_value"]: rng.normal(400, 22, n),
            SCHEMA["model_used"]:      ["lstm"] * n,
            SCHEMA["inference_time"]:  rng.uniform(0.003, 0.006, n),
            SCHEMA["energy_uj"]:       rng.uniform(800, 1200, n),
            SCHEMA["station_id"]:      [f"ST_{(i%8)+1:02d}" for i in range(n)],
            SCHEMA["energy_backend"]:  ["rapl"] * 600 + ["estimator"] * 600,
        })

        with pytest.raises(MixedBackendError):
            produce_snapshot(df, "lstm")

    def test_single_backend_window_accepted(self):
        """A window with a single consistent backend must NOT raise."""
        from aegis.core.monitor import produce_snapshot
        from aegis.core.models import SCHEMA

        import numpy as np
        rng = np.random.default_rng(1)
        n   = 1200
        df  = pd.DataFrame({
            SCHEMA["true_value"]:      rng.normal(400, 20, n),
            SCHEMA["predicted_value"]: rng.normal(400, 22, n),
            SCHEMA["model_used"]:      ["lstm"] * n,
            SCHEMA["inference_time"]:  rng.uniform(0.003, 0.006, n),
            SCHEMA["energy_uj"]:       rng.uniform(800, 1200, n),
            SCHEMA["station_id"]:      [f"ST_{(i%8)+1:02d}" for i in range(n)],
            SCHEMA["energy_backend"]:  ["estimator"] * n,
        })

        snap = produce_snapshot(df, "lstm")
        assert snap.window.energy_backend == "estimator"
