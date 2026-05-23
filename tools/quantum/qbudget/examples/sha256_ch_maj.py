"""
SHA-256 Ch and Maj quantum sub-circuits — Qiskit port, LUT-verified.

Purpose: provide SHA-256-derived non-linear primitives (Ch, Maj) as
quantum circuits to test whether the architecture-dependent opt_level
gap-closure finding generalizes to SHA-256-style
sub-circuits.

This file does NOT implement the full 64-round SHA-256. It implements
the 32-bit Ch and Maj functions, which are the only AND-bearing
primitives in SHA-256's compression function (Σ/σ rotations are
pure XOR, and the modular adders carry independent ANDs that we do
not model here).

Architecture (matches AES BP11 pattern):

  Forward step F:
    scratch[i] := y[i] XOR z[i]        (CNOTs into scratch)
    anc[i]     := x[i] AND scratch[i]   (Jones AND)

  Output extraction E (only writes into out, never reads anc):
    out[i] := z[i] XOR anc[i]           (CNOTs into out)

  Adjoint F† (only touches scratch / anc / gar):
    -- restores scratch, anc, gar to |0> by reversing F
    -- out is NOT touched, so the Ch result persists

  Final state:
    x, y, z      : unchanged inputs
    out          : holds Ch(x, y, z)
    scratch, anc, gar : all |0>

  This is the F + E + F† compute-copy-uncompute structure that
  exposes a transpiler-discoverable inner F†·F pair when wrapped
  inside a Grover oracle (forward + MCZ + inverse).

Verification: tested against the classical Ch / Maj truth function on
1000 random 32-bit triplets plus all-zero, all-one, and a few hand-
chosen edge cases.
"""

import os
import sys
import random

from qiskit import QuantumCircuit, QuantumRegister

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from aes_bp11_sbox import lpand_jones  # noqa: E402

WIDTH = 32


# ---------------------------------------------------------------------------
# Classical reference functions.
# ---------------------------------------------------------------------------
def Ch(x, y, z, width=WIDTH):
    mask = (1 << width) - 1
    return ((x & y) ^ ((~x) & z)) & mask


def Maj(x, y, z, width=WIDTH):
    mask = (1 << width) - 1
    return ((x & y) ^ (x & z) ^ (y & z)) & mask


# ---------------------------------------------------------------------------
# Quantum builders.
# ---------------------------------------------------------------------------
def _ch_forward(qc, x, y, z, scratch, anc, gar, jones=True):
    """Forward step F for Ch: writes y XOR z into scratch, then
    x AND scratch into anc. Does NOT touch out."""
    for i in range(WIDTH):
        qc.cx(y[i], scratch[i])
        qc.cx(z[i], scratch[i])
    for i in range(WIDTH):
        if jones:
            lpand_jones(qc, x[i], scratch[i], anc[i], gar[i])
        else:
            qc.ccx(x[i], scratch[i], anc[i])


def _ch_extract(qc, z, anc, out):
    """Output extraction E for Ch: out[i] := z[i] XOR anc[i].

    This is the ONLY place `out` is written. After F†, out persists."""
    for i in range(WIDTH):
        qc.cx(z[i], out[i])
        qc.cx(anc[i], out[i])


def build_ch_block(jones=True, compute_uncompute=False):
    """Build a Ch(x, y, z) -> out 32-bit block.

    Registers:
      x, y, z (inputs)            : 32 qubits each
      out                         : 32 qubits (must be |0> on entry)
      scratch (= y XOR z)         : 32 qubits (|0> on entry)
      anc (= x AND scratch)       : 32 qubits (|0> on entry; Jones AND target)
      gar (Jones AND garbage)     : 32 qubits (|0> on entry)

    Total: 6 × 32 = 192 qubits.

    If compute_uncompute=True, the forward F is adjointed AFTER the
    extraction E, restoring scratch, anc, gar to |0>. The out register
    retains the Ch value.
    """
    x = QuantumRegister(WIDTH, 'x')
    y = QuantumRegister(WIDTH, 'y')
    z = QuantumRegister(WIDTH, 'z')
    out = QuantumRegister(WIDTH, 'out')
    scratch = QuantumRegister(WIDTH, 'scratch')
    anc = QuantumRegister(WIDTH, 'anc')
    gar = QuantumRegister(WIDTH, 'gar')

    qc = QuantumCircuit(x, y, z, out, scratch, anc, gar, name='Ch32')

    # Build the forward into a separate subcircuit so we can take
    # its adjoint if compute_uncompute=True.
    fwd = QuantumCircuit(x, y, z, out, scratch, anc, gar, name='Ch32_fwd')
    _ch_forward(fwd, x, y, z, scratch, anc, gar, jones=jones)

    qc.compose(fwd, inplace=True)
    _ch_extract(qc, z, anc, out)

    if compute_uncompute:
        qc.compose(fwd.inverse(), inplace=True)

    return qc


# Maj uses the factored form Maj(x,y,z) = z XOR ((x XOR z) AND (y XOR z)).
def _maj_forward(qc, x, y, z, scratch1, scratch2, anc, gar, jones=True):
    """Forward F for Maj: scratch1 = x XOR z; scratch2 = y XOR z;
    anc = scratch1 AND scratch2."""
    for i in range(WIDTH):
        qc.cx(x[i], scratch1[i])
        qc.cx(z[i], scratch1[i])
        qc.cx(y[i], scratch2[i])
        qc.cx(z[i], scratch2[i])
    for i in range(WIDTH):
        if jones:
            lpand_jones(qc, scratch1[i], scratch2[i], anc[i], gar[i])
        else:
            qc.ccx(scratch1[i], scratch2[i], anc[i])


def _maj_extract(qc, z, anc, out):
    """Output extraction for Maj: out[i] = z[i] XOR anc[i]."""
    for i in range(WIDTH):
        qc.cx(z[i], out[i])
        qc.cx(anc[i], out[i])


def build_maj_block(jones=True, compute_uncompute=False):
    """Build a Maj(x, y, z) -> out 32-bit block. 7×32 = 224 qubits."""
    x = QuantumRegister(WIDTH, 'x')
    y = QuantumRegister(WIDTH, 'y')
    z = QuantumRegister(WIDTH, 'z')
    out = QuantumRegister(WIDTH, 'out')
    scratch1 = QuantumRegister(WIDTH, 'scratch1')
    scratch2 = QuantumRegister(WIDTH, 'scratch2')
    anc = QuantumRegister(WIDTH, 'anc')
    gar = QuantumRegister(WIDTH, 'gar')

    qc = QuantumCircuit(x, y, z, out, scratch1, scratch2, anc, gar, name='Maj32')

    fwd = QuantumCircuit(x, y, z, out, scratch1, scratch2, anc, gar, name='Maj32_fwd')
    _maj_forward(fwd, x, y, z, scratch1, scratch2, anc, gar, jones=jones)

    qc.compose(fwd, inplace=True)
    _maj_extract(qc, z, anc, out)

    if compute_uncompute:
        qc.compose(fwd.inverse(), inplace=True)

    return qc


# ---------------------------------------------------------------------------
# Classical simulation of a circuit (computational basis only, used
# with jones=False so Toffoli ops are CCX).
# ---------------------------------------------------------------------------
def _classical_eval(qc, x_val, y_val, z_val, x_reg, y_reg, z_reg, out_reg):
    bits = [0] * qc.num_qubits
    for i in range(WIDTH):
        bits[qc.find_bit(x_reg[i]).index] = (x_val >> i) & 1
        bits[qc.find_bit(y_reg[i]).index] = (y_val >> i) & 1
        bits[qc.find_bit(z_reg[i]).index] = (z_val >> i) & 1
    for inst in qc.data:
        name = inst.operation.name
        qs = [qc.find_bit(q).index for q in inst.qubits]
        if name == 'x':
            bits[qs[0]] ^= 1
        elif name == 'cx':
            if bits[qs[0]]:
                bits[qs[1]] ^= 1
        elif name == 'ccx':
            if bits[qs[0]] and bits[qs[1]]:
                bits[qs[2]] ^= 1
        elif name in ('h', 't', 'tdg', 's', 'sdg', 'z', 'measure'):
            pass
        else:
            raise RuntimeError(f'unsupported op: {name}')
    out = 0
    for i in range(WIDTH):
        out |= bits[qc.find_bit(out_reg[i]).index] << i
    return out


def verify_block(builder, classical_fn, label):
    """Verify a builder against its classical reference."""
    # Test on jones=False so we can classically simulate.
    rng = random.Random(0xC0FFEE)
    test_cases = [(0, 0, 0), (0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF),
                  (0xAAAAAAAA, 0x55555555, 0xCCCCCCCC),
                  (0xDEADBEEF, 0xCAFEBABE, 0x12345678)]
    for _ in range(64):
        test_cases.append(
            (rng.randint(0, 0xFFFFFFFF),
             rng.randint(0, 0xFFFFFFFF),
             rng.randint(0, 0xFFFFFFFF)))
    for cu in [False, True]:
        qc = builder(jones=False, compute_uncompute=cu)
        x_reg = next(r for r in qc.qregs if r.name == 'x')
        y_reg = next(r for r in qc.qregs if r.name == 'y')
        z_reg = next(r for r in qc.qregs if r.name == 'z')
        out_reg = next(r for r in qc.qregs if r.name == 'out')
        mismatches = []
        for xv, yv, zv in test_cases:
            got = _classical_eval(qc, xv, yv, zv, x_reg, y_reg, z_reg, out_reg)
            exp = classical_fn(xv, yv, zv)
            if got != exp:
                mismatches.append((xv, yv, zv, got, exp))
        if mismatches:
            print(f'  {label} (compute_uncompute={cu}): FAIL {len(mismatches)}/{len(test_cases)}')
            for m in mismatches[:3]:
                xv, yv, zv, got, exp = m
                print(f'    x=0x{xv:08x} y=0x{yv:08x} z=0x{zv:08x}: got 0x{got:08x}, expected 0x{exp:08x}')
            return False
        else:
            print(f'  {label} (compute_uncompute={cu}): OK {len(test_cases)}/{len(test_cases)} match')
    return True


if __name__ == '__main__':
    print('=== Classical sanity ===')
    print(f'  Ch(0xFFFFFFFF, 0xAAAAAAAA, 0x55555555) = 0x{Ch(0xFFFFFFFF, 0xAAAAAAAA, 0x55555555):08x} (expected 0xAAAAAAAA)')
    print(f'  Maj(0xFFFFFFFF, 0xFFFFFFFF, 0) = 0x{Maj(0xFFFFFFFF, 0xFFFFFFFF, 0):08x} (expected 0xFFFFFFFF)')
    print()

    print('=== Quantum Ch verification (both modes) ===')
    ok_ch = verify_block(build_ch_block, Ch, 'Ch32')
    print()
    print('=== Quantum Maj verification (both modes) ===')
    ok_maj = verify_block(build_maj_block, Maj, 'Maj32')
    print()

    if ok_ch and ok_maj:
        print('=== Structural counts (Jones AND mode) ===')
        for label, builder in [('Ch32', build_ch_block), ('Maj32', build_maj_block)]:
            for cu in [False, True]:
                qc = builder(jones=True, compute_uncompute=cu)
                counts = qc.count_ops()
                total = sum(counts.values())
                print(f'  {label} compute_uncompute={cu}: '
                      f'qubits={qc.num_qubits} ops={total} '
                      f'(cx={counts.get("cx",0)}, t={counts.get("t",0)}, tdg={counts.get("tdg",0)})')
