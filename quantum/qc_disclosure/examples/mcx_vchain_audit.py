"""
Measure Qiskit-with-Jones-AND V-chain MCX T-cost vs Qiskit's default
MCXGate decomposition, and compute the equivalent Q#/ProjectQ cost
under measurement-based AND uncomputation (AND/AND-dagger).

Strategy
--------
For each n in {7, 31} (AES 8-bit and SHA-256 32-bit markers):

1. Build MCX(n) via the standard V-chain (Barenco et al. 1995,
   generalised by Jones 2013 / Gidney 2018) using Jones AND
   gates for each compute step and reverse-Jones-AND for
   uncompute. Use (n - 1) ancilla qubits.

2. Measure T-count after transpilation to Clifford+T basis at
   opt_level 0..3.

3. For small n, verify functional equivalence (statevector) of
   the V-chain MCX vs Qiskit's MCXGate.

4. Report:
   - Qiskit default MCXGate T-count (baseline)
   - Qiskit V-chain Jones-AND MCX T-count (best-practice author)
   - Implied Q#/ProjectQ MCX T-count (V-chain forward only;
     AND† is zero-cost in those frameworks)

This script does NOT execute on Q#; it instead constructs the
exact V-chain a Q# author would build (per Jones 2013, Section IV)
and uses Qiskit's T-counter to count forward T-gates only,
which is what Q#'s AND-as-zero-cost-uncompute resource estimator
would also report.

Output: results/mcx_vchain_audit.json
"""

import os
import sys
import json
import time
import numpy as np

from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit.quantum_info import Operator, Statevector
from qiskit.circuit.library import MCXGate

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from aes_bp11_sbox import lpand_jones  # noqa: E402

BASIS = ['cx', 'h', 't', 'tdg', 's', 'sdg', 'x', 'z']


# ---------------------------------------------------------------------------
# V-chain MCX with Jones AND.
# ---------------------------------------------------------------------------
def build_vchain_mcx(n_controls, jones=True, uncompute=True):
    """Build MCX(n_controls) via the standard V-chain.

    Registers:
      c : n_controls qubits (controls)
      a : n_controls - 1 ancilla qubits (initialised to |0>, restored to |0>)
      t : 1 target qubit (gets flipped iff all controls are |1>)
      gar (Jones AND garbage): n_controls - 1 qubits (only if jones=True)

    Construction (compute path):
      a[0] = c[0] AND c[1]
      a[1] = a[0] AND c[2]
      ...
      a[n-2] = a[n-3] AND c[n-1]
    Then: target ^= a[n-2]   (a single CX)
    Then (if uncompute): reverse the AND chain to restore a -> |0>.

    For jones=True, each AND uses lpand_jones (4 T forward,
    same reverse cost using a unitary reverse).
    For jones=False, each AND is a CCX (8 T after transpile).

    Returns a QuantumCircuit on n_controls + (n_controls-1) + 1 + gar qubits.
    """
    assert n_controls >= 2
    c = QuantumRegister(n_controls, 'c')
    a = QuantumRegister(n_controls - 1, 'a')
    t = QuantumRegister(1, 't')
    if jones:
        gar = QuantumRegister(n_controls - 1, 'gar')
        qc = QuantumCircuit(c, a, t, gar, name=f'V_MCX{n_controls}')
    else:
        gar = None
        qc = QuantumCircuit(c, a, t, name=f'V_MCX{n_controls}')

    # Build the forward chain into a separate sub-circuit so we can
    # take its inverse for the uncompute.
    if jones:
        fwd = QuantumCircuit(c, a, t, gar, name='fwd')
    else:
        fwd = QuantumCircuit(c, a, t, name='fwd')

    # a[0] = c[0] AND c[1]
    if jones:
        lpand_jones(fwd, c[0], c[1], a[0], gar[0])
    else:
        fwd.ccx(c[0], c[1], a[0])
    # a[k] = a[k-1] AND c[k+1]
    for k in range(1, n_controls - 1):
        if jones:
            lpand_jones(fwd, a[k - 1], c[k + 1], a[k], gar[k])
        else:
            fwd.ccx(a[k - 1], c[k + 1], a[k])

    qc.compose(fwd, inplace=True)
    # target ^= a[n-2]
    qc.cx(a[n_controls - 2], t[0])
    if uncompute:
        qc.compose(fwd.inverse(), inplace=True)

    return qc


def build_vchain_mcz(n_controls, jones=True):
    """Build MCZ(n_controls) via H-MCX-H sandwich on the target.
    This matches the MCZ marker construction in the cipher oracles."""
    qc = build_vchain_mcx(n_controls, jones=jones, uncompute=True)
    # Find the 't' qubit index
    t_idx = None
    for reg in qc.qregs:
        if reg.name == 't':
            t_idx = qc.qubits.index(reg[0])
            break
    # Wrap the MCX in H ... H to make MCZ
    sandwiched = QuantumCircuit(qc.num_qubits, name=qc.name + '_MCZ')
    sandwiched.h(t_idx)
    sandwiched.compose(qc, inplace=True)
    sandwiched.h(t_idx)
    return sandwiched


def build_default_mcx(n_controls):
    """Build MCX(n_controls) using Qiskit's library MCXGate (no Jones AND)."""
    qc = QuantumCircuit(n_controls + 1, name=f'DefMCX{n_controls}')
    qc.append(MCXGate(num_ctrl_qubits=n_controls),
              list(range(n_controls)) + [n_controls])
    return qc


# ---------------------------------------------------------------------------
# Correctness test (statevector) for small n.
# ---------------------------------------------------------------------------
def verify_vchain_correctness(n_controls, jones=False):
    """Exhaustively confirm V-chain MCX (with uncompute) matches Qiskit's
    default MCXGate on all 2^n_controls input states (target = |0> initial).
    Uses jones=False (CCX) so we can simulate cleanly without garbage."""
    v_qc = build_vchain_mcx(n_controls, jones=jones, uncompute=True)
    default_qc = build_default_mcx(n_controls)

    n_total_v = v_qc.num_qubits
    n_total_d = default_qc.num_qubits

    # For each control-pattern + target-pattern combination:
    mismatches = 0
    for cval in range(1 << n_controls):
        for tval in [0, 1]:
            # V-chain: set c bits, a=0, t=tval, gar=0
            idx_v = 0
            for i in range(n_controls):
                if (cval >> i) & 1:
                    idx_v |= (1 << i)  # c[i] at position i
            if tval:
                idx_v |= (1 << (n_controls + (n_controls - 1)))  # t after a
            sv_v = Statevector.from_int(idx_v, 2**n_total_v).evolve(v_qc)

            # Default MCX: set c bits, t=tval
            idx_d = 0
            for i in range(n_controls):
                if (cval >> i) & 1:
                    idx_d |= (1 << i)
            if tval:
                idx_d |= (1 << n_controls)
            sv_d = Statevector.from_int(idx_d, 2**n_total_d).evolve(default_qc)

            # Expected: target ^= (all controls are 1)
            all_on = (cval == (1 << n_controls) - 1)
            expected_t = tval ^ (1 if all_on else 0)

            # V-chain: ancillas and gar should be restored to 0
            # so output basis index has same c bits, a=0, t=expected_t, gar=0
            expect_idx_v = (cval) | (expected_t << (n_controls + n_controls - 1))
            expect_idx_d = (cval) | (expected_t << n_controls)

            amp_v = sv_v[expect_idx_v]
            amp_d = sv_d[expect_idx_d]
            if abs(abs(amp_v) - 1.0) > 1e-9 or abs(abs(amp_d) - 1.0) > 1e-9:
                mismatches += 1
    return mismatches


def count_t(qc, opt_level=1):
    """Transpile to Clifford+T and count t+tdg."""
    out = transpile(qc, basis_gates=BASIS, optimization_level=opt_level)
    c = out.count_ops()
    return c.get('t', 0) + c.get('tdg', 0)


def measure_mcx(n_controls):
    """Measure both forms for given n_controls. Return dict."""
    results = {'n_controls': n_controls}

    # Qiskit default MCXGate
    default_qc = build_default_mcx(n_controls)
    results['default_mcx'] = {
        'qubits': default_qc.num_qubits,
        't_opt0': count_t(default_qc, 0),
        't_opt1': count_t(default_qc, 1),
        't_opt2': count_t(default_qc, 2),
        't_opt3': count_t(default_qc, 3),
    }

    # V-chain MCX with Jones AND, with uncompute (Qiskit author's
    # best-practice unitary implementation)
    v_qc = build_vchain_mcx(n_controls, jones=True, uncompute=True)
    results['vchain_jones_with_uncompute'] = {
        'qubits': v_qc.num_qubits,
        't_opt0': count_t(v_qc, 0),
        't_opt1': count_t(v_qc, 1),
        't_opt2': count_t(v_qc, 2),
        't_opt3': count_t(v_qc, 3),
    }

    # V-chain MCX with Jones AND, FORWARD ONLY (Q# / ProjectQ
    # equivalent, since AND† is zero-T in those frameworks)
    v_fwd = build_vchain_mcx(n_controls, jones=True, uncompute=False)
    results['vchain_jones_forward_only_QSHARP_EQUIV'] = {
        'qubits': v_fwd.num_qubits,
        't_opt0': count_t(v_fwd, 0),
        't_opt1': count_t(v_fwd, 1),
    }

    return results


def main():
    print('===== V-chain MCX vs Qiskit default MCX =====')
    print()

    # Correctness check at small n (jones=False so we can simulate easily)
    print('--- Correctness check (V-chain MCX with CCX, n=2..5) ---')
    for n in [2, 3, 4, 5]:
        mm = verify_vchain_correctness(n, jones=False)
        print(f'  n={n}: mismatches = {mm} (0 = OK)')
    print()

    all_results = {
        'description': 'V-chain MCX with Jones AND vs Qiskit default MCXGate. '
                       'Implied Q#/ProjectQ MCX cost = forward-only V-chain T '
                       '(AND-dagger is 0T in those frameworks).',
        'basis_gates': BASIS,
        'measurements': [],
    }

    for n in [7, 15, 31]:  # AES 8-bit, intermediate, SHA-256 32-bit
        print(f'--- n_controls = {n} (relevant for {n+1}-bit output marker) ---')
        t0 = time.time()
        r = measure_mcx(n)
        dt = time.time() - t0
        print(f'  default MCXGate:                  '
              f'opt0={r["default_mcx"]["t_opt0"]:>5d} '
              f'opt1={r["default_mcx"]["t_opt1"]:>5d} '
              f'opt2={r["default_mcx"]["t_opt2"]:>5d} '
              f'opt3={r["default_mcx"]["t_opt3"]:>5d}')
        print(f'  V-chain Jones+uncompute (Qiskit): '
              f'opt0={r["vchain_jones_with_uncompute"]["t_opt0"]:>5d} '
              f'opt1={r["vchain_jones_with_uncompute"]["t_opt1"]:>5d} '
              f'opt2={r["vchain_jones_with_uncompute"]["t_opt2"]:>5d} '
              f'opt3={r["vchain_jones_with_uncompute"]["t_opt3"]:>5d}')
        print(f'  V-chain forward-only (Q# equiv):  '
              f'opt0={r["vchain_jones_forward_only_QSHARP_EQUIV"]["t_opt0"]:>5d} '
              f'opt1={r["vchain_jones_forward_only_QSHARP_EQUIV"]["t_opt1"]:>5d}')
        print(f'  ({dt:.1f}s)')
        print()
        all_results['measurements'].append(r)

    out_dir = os.path.join(_HERE, 'results')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'mcx_vchain_audit.json')
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f'wrote {out_path}')


if __name__ == '__main__':
    main()
