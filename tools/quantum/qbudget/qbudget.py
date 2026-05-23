"""qbudget: standardized 7-item disclosure for quantum cryptanalysis circuits.

See SPEC.md for the field definitions and the prior-art context.

Public API:
    budget_qiskit(forward_circuit, full_circuit, ...) -> dict
    budget_qsharp(forward_estimate, full_estimate, ...) -> dict
"""
from __future__ import annotations

import warnings
from typing import Any

try:
    from qiskit import QuantumCircuit, transpile  # type: ignore
except ImportError:  # pragma: no cover -- qiskit optional at import time
    QuantumCircuit = None  # type: ignore
    transpile = None  # type: ignore


CLIFFORD_T_BASIS = ["cx", "h", "t", "tdg", "s", "sdg", "x", "z"]


def _count_t(qc: "QuantumCircuit", opt_level: int) -> int:
    """Transpile to Clifford+T and count T + Tdg."""
    if transpile is None:
        raise ImportError("qiskit is required for budget_qiskit()")
    out = transpile(qc, basis_gates=CLIFFORD_T_BASIS, optimization_level=opt_level)
    ops = out.count_ops()
    return int(ops.get("t", 0)) + int(ops.get("tdg", 0))


def budget_qiskit(
    forward_circuit: "QuantumCircuit",
    full_circuit: "QuantumCircuit",
    *,
    domain: str,
    framework: str,
    opt_regime: str,
    measurement_uncompute_used: bool,
    and_count_hint: int | None = None,
    qiskit_opt_level: int = 1,
) -> dict[str, Any]:
    """Build the 7-item disclosure for a Qiskit oracle.

    Parameters
    ----------
    forward_circuit
        The compute-only half of the oracle (cipher + marker forward, no inverse).
    full_circuit
        The full oracle (cipher + marker + inverse cipher), i.e. compute-uncompute.
    domain
        Description of what the oracle computes.
    framework
        Framework identifier, e.g. ``"qiskit-1.0.2"``.
    opt_regime
        Synthesis path used, e.g. ``"qiskit transpile opt_level=1, basis Clifford+T"``.
    measurement_uncompute_used
        Whether the construction uses measurement-based AND uncomputation.
        Qiskit's Jones AND uses a unitary 4T reverse, so this is normally ``False``
        for Qiskit-only oracles. (Set ``True`` only if mid-circuit measurements
        are used as part of an AND uncompute.)
    and_count_hint
        Optional exact AND count. Strongly preferred when known from the cipher
        builder. If omitted, ``A`` defaults to ``Tf // 4`` with a warning.
    qiskit_opt_level
        Transpile optimization level for counting T-gates (default 1).
    """
    Tf = _count_t(forward_circuit, qiskit_opt_level)
    To = _count_t(full_circuit, qiskit_opt_level)
    if and_count_hint is not None:
        A = int(and_count_hint)
    else:
        warnings.warn(
            "and_count_hint not supplied; defaulting A = Tf // 4. "
            "Pass the exact AND count from your cipher builder for accuracy.",
            stacklevel=2,
        )
        A = Tf // 4

    return {
        "D": domain,
        "M": bool(measurement_uncompute_used),
        "Tf": int(Tf),
        "To": int(To),
        "F": framework,
        "O": opt_regime,
        "A": int(A),
    }


def _extract_logical_counts(estimate: Any) -> dict[str, int]:
    """Normalize qsharp.estimate() output to a dict with cczCount/measurementCount."""
    import json as _json

    if isinstance(estimate, str):
        estimate = _json.loads(estimate)
    if isinstance(estimate, dict) and "logicalCounts" in estimate:
        lc = estimate["logicalCounts"]
    elif isinstance(estimate, dict) and "cczCount" in estimate:
        lc = estimate
    else:
        raise ValueError(
            "estimate must be a qsharp.estimate() result or a dict with cczCount; "
            f"got {type(estimate).__name__}"
        )
    return {
        "cczCount": int(lc.get("cczCount", 0)),
        "measurementCount": int(lc.get("measurementCount", 0)),
        "tCount": int(lc.get("tCount", 0)),
    }


def budget_qsharp(
    forward_estimate: Any,
    full_estimate: Any,
    *,
    domain: str,
    framework: str,
    opt_regime: str,
    and_count_hint: int | None = None,
) -> dict[str, Any]:
    """Build the 7-item disclosure for a Q# oracle estimated via the Resource Estimator.

    Uses the Jones 2013 decomposition: 1 CCZ = 4 T forward, 0 T uncompute via
    measurement. Tf and To are derived as ``4 * cczCount`` from the
    Resource Estimator's ``logicalCounts``.

    ``measurement_uncompute_used`` is detected: True iff
    ``forward_estimate.measurementCount > 0`` (i.e., the forward pass invokes
    Adjoint ApplyAnd via a ``within {} apply {}`` block) OR
    ``full_estimate.measurementCount > 0``.

    Parameters
    ----------
    forward_estimate
        ``qsharp.estimate()`` result for the compute-only half (or a dict).
    full_estimate
        ``qsharp.estimate()`` result for the full oracle (or a dict).
    domain
        Description of what the oracle computes.
    framework
        Framework identifier, e.g. ``"qsharp-1.28.0"``.
    opt_regime
        Synthesis path used, e.g. ``"Q# Resource Estimator logical counts"``.
    and_count_hint
        Optional exact AND count. If omitted, ``A`` defaults to the forward
        cczCount (each ApplyAnd contributes one CCZ in Q# RE).
    """
    fwd = _extract_logical_counts(forward_estimate)
    full = _extract_logical_counts(full_estimate)
    Tf = 4 * fwd["cczCount"]
    To = 4 * full["cczCount"]
    M = (fwd["measurementCount"] > 0) or (full["measurementCount"] > 0)
    if and_count_hint is not None:
        A = int(and_count_hint)
    else:
        A = fwd["cczCount"]

    return {
        "D": domain,
        "M": bool(M),
        "Tf": int(Tf),
        "To": int(To),
        "F": framework,
        "O": opt_regime,
        "A": int(A),
    }


__all__ = ["budget_qiskit", "budget_qsharp"]
