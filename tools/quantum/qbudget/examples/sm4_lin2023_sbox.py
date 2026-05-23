"""
SM4 S-box — Qiskit port, ground-truth-verified against published LUT.

Approach (chosen for verifiability over T-count optimality):

  1. Classical SM4 S-box implemented from first principles using
     GF(2^8) arithmetic with the SM4 irreducible polynomial
     m(x) = x^8 + x^7 + x^6 + x^5 + x^4 + x^2 + 1 (0x1F5).
     Inverse computed via Fermat: x^{-1} = x^{254} = x^{2+4+...+128}.
  2. Verified bitwise against the GB/T 32907-2016 SM4 S-box LUT
     for all 256 inputs.
  3. Quantum port: each Boolean op becomes a quantum gate (XOR -> CNOT,
     AND -> Jones AND under the same convention as the AES BP11 port).
  4. Verified by classical simulation of the Toffoli-mode circuit on
     all 256 inputs — output qubits must match SM4_SBOX[input] exactly.

T-count is NOT optimal — naive Fermat decomposition has ~7 mults of
GF(2^8) = ~448 ANDs, much more than Lin's hand-optimized 32-AND
circuit. The purpose of this file is to provide a CORRECT SM4-S-box
quantum circuit for the opt_level experiment (P3-extended for SM4),
not to reproduce Lin's exact gate counts.

Tested: see __main__ block. classical and quantum verify against the
GB/T 32907-2016 LUT for all 256 inputs.
"""

import os
import sys

from qiskit import QuantumCircuit, QuantumRegister

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from aes_bp11_sbox import lpand_jones  # noqa: E402

# ---------------------------------------------------------------------------
# Published SM4 S-box LUT (GB/T 32907-2016).
# ---------------------------------------------------------------------------
SM4_SBOX = [
    0xd6,0x90,0xe9,0xfe,0xcc,0xe1,0x3d,0xb7,0x16,0xb6,0x14,0xc2,0x28,0xfb,0x2c,0x05,
    0x2b,0x67,0x9a,0x76,0x2a,0xbe,0x04,0xc3,0xaa,0x44,0x13,0x26,0x49,0x86,0x06,0x99,
    0x9c,0x42,0x50,0xf4,0x91,0xef,0x98,0x7a,0x33,0x54,0x0b,0x43,0xed,0xcf,0xac,0x62,
    0xe4,0xb3,0x1c,0xa9,0xc9,0x08,0xe8,0x95,0x80,0xdf,0x94,0xfa,0x75,0x8f,0x3f,0xa6,
    0x47,0x07,0xa7,0xfc,0xf3,0x73,0x17,0xba,0x83,0x59,0x3c,0x19,0xe6,0x85,0x4f,0xa8,
    0x68,0x6b,0x81,0xb2,0x71,0x64,0xda,0x8b,0xf8,0xeb,0x0f,0x4b,0x70,0x56,0x9d,0x35,
    0x1e,0x24,0x0e,0x5e,0x63,0x58,0xd1,0xa2,0x25,0x22,0x7c,0x3b,0x01,0x21,0x78,0x87,
    0xd4,0x00,0x46,0x57,0x9f,0xd3,0x27,0x52,0x4c,0x36,0x02,0xe7,0xa0,0xc4,0xc8,0x9e,
    0xea,0xbf,0x8a,0xd2,0x40,0xc7,0x38,0xb5,0xa3,0xf7,0xf2,0xce,0xf9,0x61,0x15,0xa1,
    0xe0,0xae,0x5d,0xa4,0x9b,0x34,0x1a,0x55,0xad,0x93,0x32,0x30,0xf5,0x8c,0xb1,0xe3,
    0x1d,0xf6,0xe2,0x2e,0x82,0x66,0xca,0x60,0xc0,0x29,0x23,0xab,0x0d,0x53,0x4e,0x6f,
    0xd5,0xdb,0x37,0x45,0xde,0xfd,0x8e,0x2f,0x03,0xff,0x6a,0x72,0x6d,0x6c,0x5b,0x51,
    0x8d,0x1b,0xaf,0x92,0xbb,0xdd,0xbc,0x7f,0x11,0xd9,0x5c,0x41,0x1f,0x10,0x5a,0xd8,
    0x0a,0xc1,0x31,0x88,0xa5,0xcd,0x7b,0xbd,0x2d,0x74,0xd0,0x12,0xb8,0xe5,0xb4,0xb0,
    0x89,0x69,0x97,0x4a,0x0c,0x96,0x77,0x7e,0x65,0xb9,0xf1,0x09,0xc5,0x6e,0xc6,0x84,
    0x18,0xf0,0x7d,0xec,0x3a,0xdc,0x4d,0x20,0x79,0xee,0x5f,0x3e,0xd7,0xcb,0x39,0x48,
]

SM4_POLY = 0x1F5  # x^8 + x^7 + x^6 + x^5 + x^4 + x^2 + 1
SM4_AFFINE_C = 0xD3  # both Cin and Cout in our shared-affine factorisation
SM4_AFFINE_ROW0 = 0xA7  # 0b10100111 — first row of circulant matrix A,
                        # discovered by brute search to satisfy
                        # S(x) = A . MulInv(A.x XOR C) XOR C


# ---------------------------------------------------------------------------
# Classical SM4 S-box from first principles.
# ---------------------------------------------------------------------------
def gf256_mul(a, b, poly=SM4_POLY):
    """Carry-less mul + reduce in GF(2^8) with the SM4 polynomial."""
    res = 0
    for i in range(8):
        if b & (1 << i):
            res ^= a << i
    for i in range(15, 7, -1):
        if res & (1 << i):
            res ^= poly << (i - 8)
    return res & 0xFF


def gf256_inv(a, poly=SM4_POLY):
    """Multiplicative inverse via Fermat: a^254 = a^(2+4+...+128).

    a^254 = a^2 * a^4 * a^8 * a^16 * a^32 * a^64 * a^128
    """
    if a == 0:
        return 0
    powers = [a]  # powers[k] = a^(2^k)
    for _ in range(7):
        powers.append(gf256_mul(powers[-1], powers[-1], poly))
    # x^254 = product of a^(2^k) for k=1..7
    res = powers[1]
    for k in range(2, 8):
        res = gf256_mul(res, powers[k], poly)
    return res


def _build_affine_circulant(row0=SM4_AFFINE_ROW0):
    """Build the 8x8 SM4 affine matrix as a left-rotation circulant.

    A[0] = row0; A[k] = left-rotate(A[k-1]) for k = 1..7.
    Verified by brute search against the GB/T 32907-2016 LUT in the
    paper preparation; row0 = 0xA7 with polynomial 0x1F5 and
    Cin = Cout = 0xD3 is the unique solution among all 8x8 binary
    circulant matrices.
    """
    A = [row0]
    for _ in range(7):
        r = ((A[-1] << 1) & 0xFF) | (A[-1] >> 7)
        A.append(r)
    return A


SM4_AFFINE_A = _build_affine_circulant()


def sm4_affine(x, A=SM4_AFFINE_A, C=SM4_AFFINE_C):
    """Compute y = A.x XOR C with bitwise XOR/AND."""
    y = 0
    for j in range(8):
        bit = bin(A[j] & x).count('1') & 1
        y |= bit << j
    return y ^ C


def sm4_sbox_classical(x):
    """SM4 S-box: y = L(MulInv(L(x))) (with shared affine L)."""
    return sm4_affine(gf256_inv(sm4_affine(x)))


# ---------------------------------------------------------------------------
# Self-test for the classical implementation.
# ---------------------------------------------------------------------------
def _verify_classical():
    mismatches = []
    for i in range(256):
        got = sm4_sbox_classical(i)
        if got != SM4_SBOX[i]:
            mismatches.append((i, got, SM4_SBOX[i]))
    return mismatches


# ---------------------------------------------------------------------------
# Quantum SM4 S-box — naive Fermat-based decomposition.
# ---------------------------------------------------------------------------
def _affine_into(qc, src_bits, out_bits, A=SM4_AFFINE_A, C=SM4_AFFINE_C):
    """out_bits = A.src_bits XOR C. XOR-only."""
    for j in range(8):
        row = A[j]
        for i in range(8):
            if (row >> i) & 1:
                qc.cx(src_bits[i], out_bits[j])
        if (C >> j) & 1:
            qc.x(out_bits[j])


def _gf_squaring_into(qc, src_bits, out_bits, poly=SM4_POLY):
    """In char-2, squaring is linear. Build the 8x8 squaring matrix
    M_sq and apply as XOR network."""
    M = []
    for i in range(8):
        v = 1 << i
        sq = 0
        for j in range(8):
            if v & (1 << j):
                sq ^= 1 << (2 * j)
        for k in range(14, 7, -1):
            if sq & (1 << k):
                sq ^= poly << (k - 8)
        sq &= 0xFF
        M.append(sq)
    for j in range(8):
        for i in range(8):
            if (M[i] >> j) & 1:
                qc.cx(src_bits[i], out_bits[j])


def _gf_mul_into(qc, a_bits, b_bits, out_bits, scratch_anc, garbage_anc,
                 jones=True):
    """out_bits = a * b in GF(2^8). 64 ANDs + reduction XORs.

    Uses scratch_anc (64 qubits) for AND results and garbage_anc (64
    qubits) for Jones AND ancillas. Both registers must be passed by
    the caller, freshly allocated."""
    R = [[0] * 8 for _ in range(8)]
    for i in range(8):
        for j in range(8):
            v = 1 << (i + j)
            for k in range(14, 7, -1):
                if v & (1 << k):
                    v ^= SM4_POLY << (k - 8)
            R[i][j] = v & 0xFF

    idx = 0
    for i in range(8):
        for j in range(8):
            if jones:
                lpand_jones(qc, a_bits[i], b_bits[j],
                            scratch_anc[idx], garbage_anc[idx])
            else:
                qc.ccx(a_bits[i], b_bits[j], scratch_anc[idx])
            pat = R[i][j]
            for k in range(8):
                if (pat >> k) & 1:
                    qc.cx(scratch_anc[idx], out_bits[k])
            idx += 1


def build_sm4_sbox(jones=True, compute_uncompute=False):
    """Build the full SM4 S-box: x -> L(MulInv(L(x))).

    Pre-allocates all registers up front to keep qubit indices stable.

    If compute_uncompute=True, after writing the result into `out`,
    appends a programmatic adjoint of the forward computation to
    clean up all internal ancillas. This produces an `F + E + F†`
    structure analogous to aes_bp11_sbox's AES BP11 circuit, and
    is the structure that exposes a transpiler-discoverable inner
    F†·F pair when the resulting circuit is sandwiched with a Grover
    marker.
    """
    x   = QuantumRegister(8, 'x')      # input
    L1  = QuantumRegister(8, 'L1')      # = L(x)
    # P[k] holds (L1)^(2^k) for k = 1..7.
    P   = [QuantumRegister(8, f'P{k}') for k in range(1, 8)]
    # Each of the 7 multiplications uses 64 scratch ANDs + 64 Jones
    # garbage ancillas, and writes into a 'prod' register.
    prods = [QuantumRegister(8, f'pr{k}') for k in range(7)]
    sc    = [QuantumRegister(64, f'sc{k}') for k in range(7)]
    ga    = [QuantumRegister(64, f'ga{k}') for k in range(7)]
    out = QuantumRegister(8, 'out')

    qc = QuantumCircuit(x, L1, *P, *prods, *sc, *ga, out, name='SM4_SBox')

    # Save the position where the 'forward' instructions begin so we
    # can replay them in reverse later if compute_uncompute is True.
    fwd_start = len(qc.data)

    # Step 1: L1 = L(x).
    _affine_into(qc, x, L1)

    # Step 2: P[0] = L1 (copy).
    for i in range(8):
        qc.cx(L1[i], P[0][i])

    # Step 3: P[k] = P[k-1]^2 for k = 1..6 (so P[k] = L1^(2^(k+1))).
    # We need powers a^2, a^4, ..., a^128.
    # P[0] = a^2, P[1] = a^4, ..., P[6] = a^128 after these 6 squarings.
    # Actually: above, P[0]=L1 (= a^1). After squaring into P[1] we
    # get a^2. So:
    for k in range(1, 7):
        _gf_squaring_into(qc, P[k - 1], P[k])
    # Now P[0]=a, P[1]=a^2, P[2]=a^4, ..., P[6]=a^64.
    # We need a^128 too: square P[6] one more time. Use prods[6] as a
    # temporary slot for a^128 and re-use prods[0..5] for the running
    # product chain below.

    # We need the 7 powers a^2 .. a^128. That's P[1] (=a^2), P[2]
    # (=a^4), P[3] (=a^8), P[4] (=a^16), P[5] (=a^32), P[6] (=a^64),
    # and a^128 (to be computed into prods[6]).
    _gf_squaring_into(qc, P[6], prods[6])  # prods[6] = a^128

    # Step 4: running product. We want
    #     inv = a^2 * a^4 * a^8 * a^16 * a^32 * a^64 * a^128
    # = a^254.
    # Chain: prods[0] = a^2 * a^4. Then prods[k] = prods[k-1] * P[k+2]
    # for k = 1..4. Then prods[5] = prods[4] * P[6] (= a^64). Then
    # `out` = prods[5] * prods[6] (= a^128).
    # That's 7 mults of GF(2^8). Each uses sc[k] and ga[k].
    # Actually we have 7 multiplications:
    #   m0: P[1] * P[2] -> prods[0]
    #   m1: prods[0] * P[3] -> prods[1]
    #   m2: prods[1] * P[4] -> prods[2]
    #   m3: prods[2] * P[5] -> prods[3]
    #   m4: prods[3] * P[6] -> prods[4]
    #   m5: prods[4] * prods[6] -> prods[5]
    # That's only 6 muls (combining 7 powers). prods[5] = a^254.
    # We need 6 sc/ga registers; sc[6]/ga[6] are unused (allocated for
    # symmetry but harmless).
    _gf_mul_into(qc, P[1], P[2], prods[0], sc[0], ga[0], jones=jones)
    _gf_mul_into(qc, prods[0], P[3], prods[1], sc[1], ga[1], jones=jones)
    _gf_mul_into(qc, prods[1], P[4], prods[2], sc[2], ga[2], jones=jones)
    _gf_mul_into(qc, prods[2], P[5], prods[3], sc[3], ga[3], jones=jones)
    _gf_mul_into(qc, prods[3], P[6], prods[4], sc[4], ga[4], jones=jones)
    _gf_mul_into(qc, prods[4], prods[6], prods[5], sc[5], ga[5], jones=jones)

    # Step 5: out = L(prods[5]).
    _affine_into(qc, prods[5], out)

    if compute_uncompute:
        # Programmatic adjoint of the forward computation (everything
        # appended between fwd_start and the start of step 5). This
        # restores all internal ancillas (L1, P*, prods, sc, ga) to
        # |0⟩, leaving only x (input) and out (result) populated.
        # The `out` register is NOT touched by the adjoint because we
        # only re-apply the inverse of operations up to (but not
        # including) the final L-affine into `out`.
        # The final affine writes into `out` from `prods[5]`; its
        # adjoint would clear `out`, which we do NOT want. So we stop
        # the adjoint just before the final affine.
        # Locate the boundary: the final affine consists of CNOTs into
        # the `out` qubits. We rebuild by re-doing the forward up to
        # (but excluding) the final affine, then taking the adjoint of
        # THAT subcircuit and appending it.
        # To do this cleanly, capture only the gates from fwd_start to
        # just before the final affine, take their adjoint, and append.
        # Find the index where the final affine started: it is the
        # length of qc.data BEFORE we called `_affine_into(prods[5], out)`.
        # But _affine_into appends only CNOTs (and possibly Xs) onto
        # the `out` register. We saved no marker; instead we recompute.
        # Simpler approach: re-run the forward (steps 1-4) into a fresh
        # temporary circuit, take its inverse, and append.
        tmp = QuantumCircuit(*qc.qregs, name='SM4_fwd_for_adj')
        # Replay steps 1-4 (everything except the final affine).
        _affine_into(tmp, x, L1)
        for i in range(8):
            tmp.cx(L1[i], P[0][i])
        for k in range(1, 7):
            _gf_squaring_into(tmp, P[k - 1], P[k])
        _gf_squaring_into(tmp, P[6], prods[6])
        _gf_mul_into(tmp, P[1], P[2], prods[0], sc[0], ga[0], jones=jones)
        _gf_mul_into(tmp, prods[0], P[3], prods[1], sc[1], ga[1], jones=jones)
        _gf_mul_into(tmp, prods[1], P[4], prods[2], sc[2], ga[2], jones=jones)
        _gf_mul_into(tmp, prods[2], P[5], prods[3], sc[3], ga[3], jones=jones)
        _gf_mul_into(tmp, prods[3], P[6], prods[4], sc[4], ga[4], jones=jones)
        _gf_mul_into(tmp, prods[4], prods[6], prods[5], sc[5], ga[5], jones=jones)
        # Append the inverse of tmp to qc.
        qc.compose(tmp.inverse(), inplace=True)

    return qc


# ---------------------------------------------------------------------------
# Verification harness.
# ---------------------------------------------------------------------------
def _classical_eval_circuit(qc, input_byte, input_qubits, output_qubits):
    """Evaluate qc as a classical Toffoli/CNOT circuit on a given input.

    Maintains a dict of qubit-index -> bit-value. Returns the integer
    represented by output_qubits (bit i = output_qubits[i]).
    """
    bits = [0] * qc.num_qubits
    for i, q in enumerate(input_qubits):
        bits[q] = (input_byte >> i) & 1
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
            # Pure-phase or non-computational ops: no effect on
            # computational-basis simulation of Jones AND (which we
            # only verify in jones=False mode using ccx).
            pass
        else:
            raise RuntimeError(f'unsupported op for classical eval: {name}')
    out = 0
    for i, q in enumerate(output_qubits):
        out |= bits[q] << i
    return out


def verify_sm4_sbox_circuit():
    """Build the Toffoli-mode SM4 S-box and verify on all 256 inputs."""
    qc = build_sm4_sbox(jones=False)
    reg_in = next(r for r in qc.qregs if r.name == 'x')
    reg_out = next(r for r in qc.qregs if r.name == 'out')
    input_qubits = [qc.find_bit(q).index for q in reg_in]
    output_qubits = [qc.find_bit(q).index for q in reg_out]
    mismatches = []
    for inp in range(256):
        got = _classical_eval_circuit(qc, inp, input_qubits, output_qubits)
        if got != SM4_SBOX[inp]:
            mismatches.append((inp, got, SM4_SBOX[inp]))
    return mismatches


if __name__ == '__main__':
    print('=== Classical SM4 S-box self-test ===')
    cm = _verify_classical()
    if cm:
        print(f'  FAIL: {len(cm)} mismatches')
        for i, got, exp in cm[:5]:
            print(f'    0x{i:02x} -> got 0x{got:02x}, expected 0x{exp:02x}')
        sys.exit(1)
    print('  OK: all 256 inputs match GB/T 32907-2016 SM4_SBOX LUT')

    print()
    print('=== Quantum SM4 S-box (Toffoli mode) verification ===')
    qm = verify_sm4_sbox_circuit()
    if qm:
        print(f'  FAIL: {len(qm)} mismatches')
        for i, got, exp in qm[:5]:
            print(f'    0x{i:02x} -> got 0x{got:02x}, expected 0x{exp:02x}')
        sys.exit(1)
    print('  OK: all 256 inputs match SM4_SBOX LUT')

    print()
    print('=== Quantum SM4 S-box (Jones AND mode) structural counts ===')
    qc = build_sm4_sbox(jones=True)
    counts = qc.count_ops()
    print(f'  qubits      : {qc.num_qubits}')
    print(f'  total ops   : {sum(counts.values())}')
    print(f'  ops detail  : {dict(counts)}')
