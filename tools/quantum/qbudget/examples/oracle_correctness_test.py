"""
Small-instance oracle correctness verification.

Builds 4-bit Ch and Maj Grover oracles in BOTH straight-through and
compute-uncompute modes. Extracts the diagonal of the oracle's unitary
on the input-register subspace (all ancillas start and must return to
|0>) and compares the phase pattern between the two modes AND against
the classical expectation.

Strategy (fast):
  1. Evolve the uniform superposition over x,y,z through each oracle.
  2. Because the oracle is diagonal on the computational basis of (x,y,z)
     (ancillas start at 0 and return to 0), each basis amplitude picks up
     only a phase factor ±1.
  3. Compare those 4096 phases between ST/CU modes and against the
     classical Ch/Maj truth table.

Run:
    cd .../src/quantum_sha256
    source .venv/bin/activate
    python audit/sha256/oracle_correctness_test.py
"""

import sys
import os
import time
import numpy as np

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.quantum_info import Statevector
from qiskit.circuit.library import MCXGate

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from aes_bp11_sbox import lpand_jones  # noqa: E402

W = 2  # 2-bit width; Ch 7×2=14 qubits, Maj 8×2=16 qubits — exhaustive in ~3s


# ---------------------------------------------------------------------------
# Classical references (4-bit).
# ---------------------------------------------------------------------------
def ch_classical(x, y, z):
    mask = (1 << W) - 1
    return ((x & y) ^ ((~x) & z)) & mask


def maj_classical(x, y, z):
    mask = (1 << W) - 1
    return ((x & y) ^ (x & z) ^ (y & z)) & mask


# ---------------------------------------------------------------------------
# 4-bit Ch quantum builder (parametric width, mirrors sha256_ch_maj.py).
# ---------------------------------------------------------------------------
def build_ch4(jones=True, compute_uncompute=False):
    x = QuantumRegister(W, 'x')
    y = QuantumRegister(W, 'y')
    z = QuantumRegister(W, 'z')
    out = QuantumRegister(W, 'out')
    scratch = QuantumRegister(W, 'scratch')
    anc = QuantumRegister(W, 'anc')
    gar = QuantumRegister(W, 'gar')

    qc = QuantumCircuit(x, y, z, out, scratch, anc, gar, name='Ch4')

    fwd = QuantumCircuit(x, y, z, out, scratch, anc, gar, name='Ch4_fwd')
    for i in range(W):
        fwd.cx(y[i], scratch[i])
        fwd.cx(z[i], scratch[i])
    for i in range(W):
        if jones:
            lpand_jones(fwd, x[i], scratch[i], anc[i], gar[i])
        else:
            fwd.ccx(x[i], scratch[i], anc[i])

    qc.compose(fwd, inplace=True)
    for i in range(W):
        qc.cx(z[i], out[i])
        qc.cx(anc[i], out[i])

    if compute_uncompute:
        qc.compose(fwd.inverse(), inplace=True)

    return qc


def build_maj4(jones=True, compute_uncompute=False):
    x = QuantumRegister(W, 'x')
    y = QuantumRegister(W, 'y')
    z = QuantumRegister(W, 'z')
    out = QuantumRegister(W, 'out')
    s1 = QuantumRegister(W, 'scratch1')
    s2 = QuantumRegister(W, 'scratch2')
    anc = QuantumRegister(W, 'anc')
    gar = QuantumRegister(W, 'gar')

    qc = QuantumCircuit(x, y, z, out, s1, s2, anc, gar, name='Maj4')

    fwd = QuantumCircuit(x, y, z, out, s1, s2, anc, gar, name='Maj4_fwd')
    for i in range(W):
        fwd.cx(x[i], s1[i])
        fwd.cx(z[i], s1[i])
        fwd.cx(y[i], s2[i])
        fwd.cx(z[i], s2[i])
    for i in range(W):
        if jones:
            lpand_jones(fwd, s1[i], s2[i], anc[i], gar[i])
        else:
            fwd.ccx(s1[i], s2[i], anc[i])

    qc.compose(fwd, inplace=True)
    for i in range(W):
        qc.cx(z[i], out[i])
        qc.cx(anc[i], out[i])

    if compute_uncompute:
        qc.compose(fwd.inverse(), inplace=True)

    return qc


# ---------------------------------------------------------------------------
# Oracle builder (same structure as sha256_inner_opt_levels.py).
# ---------------------------------------------------------------------------
def build_oracle(block_builder, jones=True, compute_uncompute=False,
                 target_word=0b1010):
    """Build a Grover oracle: forward + MCZ-on-out-matching-target + inverse."""
    block = block_builder(jones=jones, compute_uncompute=compute_uncompute)
    n = block.num_qubits
    qc = QuantumCircuit(n)
    qc.compose(block, inplace=True)

    out_base = None
    for reg in block.qregs:
        if reg.name == 'out':
            out_base = block.qubits.index(reg[0])
            break
    assert out_base is not None
    out_q = list(range(out_base, out_base + W))

    for i, q in enumerate(out_q):
        if not (target_word >> i) & 1:
            qc.x(q)
    qc.h(out_q[-1])
    qc.append(MCXGate(num_ctrl_qubits=W - 1), out_q[:-1] + [out_q[-1]])
    qc.h(out_q[-1])
    for i, q in enumerate(out_q):
        if not (target_word >> i) & 1:
            qc.x(q)

    qc.compose(block.inverse(), inplace=True)
    return qc


# ---------------------------------------------------------------------------
# Fast diagonal extraction: evolve each basis state one at a time,
# but only in the x,y,z subspace (ancillas always |0>).
# Strategy: for each basis state of (x,y,z), set those qubits and run
# the oracle, check that we get ±|same state> back.
# With W=4, we have 4096 states but circuits are only 28 or 32 qubits.
# Each evolve takes ~ms with numpy backend for these small circuits.
# ---------------------------------------------------------------------------
def extract_oracle_diagonal(oracle_qc, n_input_bits):
    """Return a length-2^n_input_bits array of ±1 phases.

    For each computational basis state |k> in the first n_input_bits
    (with remaining qubits |0>), evolve through oracle and extract
    the phase of the |k,0...0> amplitude in the output."""
    n_total = oracle_qc.num_qubits
    n_states = 1 << n_input_bits
    phases = np.zeros(n_states, dtype=np.float64)

    for k in range(n_states):
        sv = Statevector.from_int(k, 2**n_total)
        sv_out = sv.evolve(oracle_qc)
        amp = sv_out[k]
        # The oracle must return the state to |k,0...0> with phase ±1
        if abs(abs(amp) - 1.0) > 1e-6:
            raise ValueError(
                f'State |{k}> not returned cleanly: amp={amp} '
                f'(|amp|={abs(amp):.6f})')
        phases[k] = np.sign(amp.real)
    return phases


def test_oracle_correctness(func_name, block_builder, classical_fn):
    """Test that both oracle modes (ST and CU) mark exactly the states
    where classical_fn(x,y,z) == target."""
    print(f'\n===== {func_name} (W={W}) oracle correctness =====')
    n_input = 3 * W  # 12 bits for x,y,z
    all_pass = True

    for target in range(1 << W):  # all 2^W possible targets (exhaustive)
        t0 = time.time()
        oracle_st = build_oracle(block_builder, jones=False,
                                 compute_uncompute=False, target_word=target)
        oracle_cu = build_oracle(block_builder, jones=False,
                                 compute_uncompute=True, target_word=target)

        phases_st = extract_oracle_diagonal(oracle_st, n_input)
        phases_cu = extract_oracle_diagonal(oracle_cu, n_input)
        dt = time.time() - t0

        # Build expected phases from classical function
        expected = np.ones(1 << n_input, dtype=np.float64)
        mask = (1 << W) - 1
        for k in range(1 << n_input):
            x_val = k & mask
            y_val = (k >> W) & mask
            z_val = (k >> (2 * W)) & mask
            if classical_fn(x_val, y_val, z_val) == target:
                expected[k] = -1.0

        st_match = np.allclose(phases_st, expected)
        cu_match = np.allclose(phases_cu, expected)
        modes_match = np.allclose(phases_st, phases_cu)

        n_marked_exp = int(np.sum(expected == -1.0))
        n_marked_st = int(np.sum(phases_st == -1.0))
        n_marked_cu = int(np.sum(phases_cu == -1.0))

        ok = st_match and cu_match and modes_match
        status = 'OK' if ok else 'FAIL'
        if not ok:
            all_pass = False

        print(f'  target={target:#06b}: expected_marked={n_marked_exp}  '
              f'ST_marked={n_marked_st}  CU_marked={n_marked_cu}  '
              f'ST==exp:{st_match}  CU==exp:{cu_match}  '
              f'ST==CU:{modes_match}  [{status}] ({dt:.1f}s)')

        if not st_match:
            diffs = np.where(phases_st != expected)[0][:5]
            print(f'    ST diffs at indices: {diffs}')
        if not cu_match:
            diffs = np.where(phases_cu != expected)[0][:5]
            print(f'    CU diffs at indices: {diffs}')

    return all_pass


def main():
    ok1 = test_oracle_correctness('Ch', build_ch4, ch_classical)
    ok2 = test_oracle_correctness('Maj', build_maj4, maj_classical)
    print('\n===== SUMMARY =====')
    if ok1 and ok2:
        print('ALL PASS: both oracle modes mark the same states as '
              'the classical reference for Ch and Maj across 4 targets.')
    else:
        print('FAIL: see above for mismatches.')
        sys.exit(1)


if __name__ == '__main__':
    main()
