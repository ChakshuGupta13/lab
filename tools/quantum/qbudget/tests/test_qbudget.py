"""Tests for qbudget. Run with: pytest tests/"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from qbudget import budget_qiskit, budget_qsharp  # noqa: E402


# --- Q# side: deterministic, uses fixture dicts -----------------------------


def _make_estimate(ccz: int, meas: int = 0, t: int = 0) -> dict:
    return {"logicalCounts": {"cczCount": ccz, "measurementCount": meas, "tCount": t}}


def test_qsharp_aes_bp11_skeleton():
    """AES BP11: N_and=34, W=8 -> forward CCZ = 34 + 7 = 41 -> T=164. Source:
    qsharp_re_endtoend.json measurement of structural oracle skeleton."""
    fwd = _make_estimate(ccz=41, meas=41)
    full = _make_estimate(ccz=41, meas=41)  # within/apply: reverse is free
    d = budget_qsharp(
        forward_estimate=fwd,
        full_estimate=full,
        domain="AES BP11 S-box skeleton (N_and=34, W=8)",
        framework="qsharp-1.28.0",
        opt_regime="Q# Resource Estimator logical counts",
    )
    assert d["Tf"] == 164
    assert d["To"] == 164
    assert d["M"] is True
    assert d["A"] == 41
    assert d["F"] == "qsharp-1.28.0"


def test_qsharp_sm4_fermat_skeleton():
    fwd = _make_estimate(ccz=391, meas=391)
    d = budget_qsharp(
        forward_estimate=fwd,
        full_estimate=fwd,
        domain="SM4 Fermat S-box skeleton (N_and=384, W=8)",
        framework="qsharp-1.28.0",
        opt_regime="Q# Resource Estimator logical counts",
    )
    assert d["Tf"] == 1564
    assert d["To"] == 1564


def test_qsharp_sha256_ch32_skeleton():
    fwd = _make_estimate(ccz=63, meas=63)
    d = budget_qsharp(
        forward_estimate=fwd,
        full_estimate=fwd,
        domain="SHA-256 Ch32 skeleton (N_and=32, W=32)",
        framework="qsharp-1.28.0",
        opt_regime="Q# Resource Estimator logical counts",
    )
    assert d["Tf"] == 252
    assert d["To"] == 252


def test_qsharp_no_measurement_uncompute():
    """If measurementCount=0 (unitary Toffoli path), M should be False."""
    fwd = _make_estimate(ccz=10, meas=0)
    d = budget_qsharp(
        forward_estimate=fwd,
        full_estimate=fwd,
        domain="test",
        framework="qsharp-1.28.0",
        opt_regime="Q# RE",
    )
    assert d["M"] is False


def test_qsharp_estimate_accepts_json_string():
    """qsharp.estimate() in some versions returns a JSON string."""
    fwd_dict = _make_estimate(ccz=7)
    fwd_str = json.dumps(fwd_dict)
    d = budget_qsharp(
        forward_estimate=fwd_str,
        full_estimate=fwd_str,
        domain="test",
        framework="qsharp-1.28.0",
        opt_regime="Q# RE",
    )
    assert d["Tf"] == 28


# --- Qiskit side: requires qiskit; skipped if not installed -----------------


qiskit = pytest.importorskip("qiskit")


def _hand_t_circuit(n_t_gates: int) -> "qiskit.QuantumCircuit":
    """Build a circuit with exactly n_t_gates that the transpiler cannot fuse.

    Each T is on its own qubit, so adjacent-T cancellation/fusion cannot apply.
    """
    qc = qiskit.QuantumCircuit(max(n_t_gates, 1))
    for i in range(n_t_gates):
        qc.t(i)
    return qc


def test_qiskit_counts_t_gates():
    fwd = _hand_t_circuit(5)
    full = _hand_t_circuit(10)
    d = budget_qiskit(
        forward_circuit=fwd,
        full_circuit=full,
        domain="hand-built T chain",
        framework=f"qiskit-{qiskit.__version__}",
        opt_regime="qiskit transpile opt_level=1",
        measurement_uncompute_used=False,
        and_count_hint=0,
    )
    assert d["Tf"] == 5
    assert d["To"] == 10
    assert d["M"] is False
    assert d["A"] == 0


def test_qiskit_and_count_hint_default_warns():
    fwd = _hand_t_circuit(8)
    with pytest.warns(UserWarning, match="and_count_hint not supplied"):
        d = budget_qiskit(
            forward_circuit=fwd,
            full_circuit=fwd,
            domain="test",
            framework=f"qiskit-{qiskit.__version__}",
            opt_regime="qiskit transpile opt_level=1",
            measurement_uncompute_used=False,
        )
    assert d["A"] == 2  # Tf // 4 = 8 // 4


def test_disclosure_is_json_serializable():
    fwd = _hand_t_circuit(1)
    d = budget_qiskit(
        forward_circuit=fwd,
        full_circuit=fwd,
        domain="test",
        framework="qiskit",
        opt_regime="opt1",
        measurement_uncompute_used=False,
        and_count_hint=0,
    )
    assert json.dumps(d)  # raises if not serializable
