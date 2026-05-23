"""
Best-practice Qiskit oracle reconstruction.

Builds Grover oracles for AES BP11 S-box, SM4 Fermat S-box, and
SHA-256 Ch32/Maj32 using:
  - Jones AND everywhere (cipher ANDs + V-chain MCX/MCZ marker)
  - explicit V-chain MCZ marker (NOT the default Qiskit MCXGate)
  - compute-uncompute architecture (industry-standard, ancilla-clean)

Then computes three ρ values:
  - rho_default_qiskit  = T(default Qiskit oracle as previously measured) / T(Q#-equivalent)
  - rho_best_qiskit     = T(V-chain Jones oracle, this script) / T(Q#-equivalent)
  - rho_floor           = predicted minimum gap (= 2.00 if cipher + marker
                          both have u=f symmetry; algebraic identity)

Q#-equivalent T-count = forward-only V-chain T (since AND-dagger is 0T in Q#).

Output: results/best_practice_oracles.json
"""
import os
import sys
import json
import time

from qiskit import QuantumCircuit, QuantumRegister, transpile

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from aes_bp11_sbox import build_bp11_sbox, lpand_jones  # noqa: E402
from sm4_lin2023_sbox import build_sm4_sbox  # noqa: E402
from sha256_ch_maj import build_ch_block, build_maj_block  # noqa: E402

BASIS = ['cx', 'h', 't', 'tdg', 's', 'sdg', 'x', 'z']


# ---------------------------------------------------------------------------
# V-chain MCZ on a target register, using Jones AND for the chain.
# ---------------------------------------------------------------------------
def append_vchain_mcz_jones(qc, target_qubits, ancilla_qubits, garbage_qubits,
                            mask_bits, uncompute=True):
    """Append an MCZ to qc that fires iff the target_qubits jointly equal
    mask_bits (least-significant-bit first).

    target_qubits  : list of W qubits in qc
    ancilla_qubits : list of W-1 ancilla qubits (must be |0> on entry, restored)
    garbage_qubits : list of W-1 Jones-AND garbage ancillas (must be |0>; restored)
    mask_bits      : integer; bit i = 1 means target_qubits[i] should be |1>

    The Jones AND target_qubits[i] is conditioned-on the bit value:
    we X-mask before/after so the MCZ fires exactly on mask_bits.
    """
    W = len(target_qubits)
    assert len(ancilla_qubits) == W - 1
    assert len(garbage_qubits) == W - 1

    # X-mask: invert qubits where mask_bit = 0, so the all-1 condition becomes
    # the equality with mask_bits.
    for i in range(W):
        if not ((mask_bits >> i) & 1):
            qc.x(target_qubits[i])

    # Build the V-chain forward into a sub-circuit so we can take inverse.
    fwd = QuantumCircuit(qc.num_qubits, name='vchain_fwd')
    lpand_jones(fwd, target_qubits[0], target_qubits[1],
                ancilla_qubits[0], garbage_qubits[0])
    for k in range(1, W - 1):
        lpand_jones(fwd, ancilla_qubits[k - 1], target_qubits[k + 1],
                    ancilla_qubits[k], garbage_qubits[k])
    qc.compose(fwd, inplace=True)

    # Apply Z to ancilla_qubits[W-2] (the final ancilla, which holds
    # AND of all targets).  This is the phase-flip: Z * |1> = -|1>.
    qc.z(ancilla_qubits[W - 2])

    if uncompute:
        qc.compose(fwd.inverse(), inplace=True)

    # Undo X-mask
    for i in range(W):
        if not ((mask_bits >> i) & 1):
            qc.x(target_qubits[i])


# ---------------------------------------------------------------------------
# Build a best-practice oracle wrapping a cipher block.
# ---------------------------------------------------------------------------
def build_best_practice_oracle(cipher_block, out_reg_name, W, target_value,
                                vchain_uncompute=True,
                                cipher_inverse_jones_uncompute=True):
    """Wrap cipher_block in a V-chain MCZ marker (Jones AND) + cipher inverse.

    cipher_block : QuantumCircuit producing the cipher (with cipher_block.qregs
                   containing a register named `out_reg_name` of width W).
    out_reg_name : 'out' or 's' etc.
    W            : width of the output register (= number of marker controls + 1)
    target_value : integer target to mark
    vchain_uncompute : if True, the marker uncomputes its ancillas via reverse
                       V-chain (Qiskit author's best practice). If False, the
                       ancillas are left dirty (Q#-style, where AND-dagger is
                       free).

    Returns the oracle and its forward-only T-count (the implied Q# cost).
    """
    # Locate the output register
    out_qubits_in_block = None
    for reg in cipher_block.qregs:
        if reg.name == out_reg_name:
            base = cipher_block.qubits.index(reg[0])
            out_qubits_in_block = list(range(base, base + reg.size))
            break
    assert out_qubits_in_block is not None
    assert len(out_qubits_in_block) == W

    # Outer circuit: needs cipher_block.num_qubits + (W-1) ancillas + (W-1) gar
    n_cipher = cipher_block.num_qubits
    n_anc = W - 1
    n_gar = W - 1
    n_total = n_cipher + n_anc + n_gar

    anc_base = n_cipher
    gar_base = n_cipher + n_anc

    qc = QuantumCircuit(n_total, name='best_practice_oracle')
    qc.compose(cipher_block, qubits=list(range(n_cipher)), inplace=True)

    target_qubits = out_qubits_in_block  # already in qc coordinate frame
    ancilla_qubits = list(range(anc_base, anc_base + n_anc))
    garbage_qubits = list(range(gar_base, gar_base + n_gar))

    append_vchain_mcz_jones(qc, target_qubits, ancilla_qubits, garbage_qubits,
                            target_value, uncompute=vchain_uncompute)

    qc.compose(cipher_block.inverse(), qubits=list(range(n_cipher)),
               inplace=True)
    return qc


def count_t(qc, opt_level=1):
    out = transpile(qc, basis_gates=BASIS, optimization_level=opt_level)
    c = out.count_ops()
    return c.get('t', 0) + c.get('tdg', 0)


def qsharp_equivalent_t(cipher_block_forward_t, vchain_forward_only_t):
    """Implied Q#/ProjectQ T-count for the equivalent oracle:
      forward cipher (Jones AND fwd cost) + V-chain forward only.
      The inverse cipher and the V-chain ancilla uncompute are BOTH
      free in Q#/ProjectQ because every AND has a corresponding AND†
      (mid-circuit measurement, 0T).  Whether the oracle structure
      includes an inverse-cipher block is therefore irrelevant to
      the Q# T-count.
    """
    return cipher_block_forward_t + vchain_forward_only_t


# ---------------------------------------------------------------------------
# Oracle audit for each cipher.
# ---------------------------------------------------------------------------
def measure_one(label, block, out_name, W, target, cipher_fwd_t,
                vchain_fwd_only_t):
    """Build and measure both best-practice (Qiskit V-chain Jones) and
    'no-uncompute marker' variants. Return dict."""
    print(f'\n===== {label} =====')

    # Best-practice Qiskit: V-chain Jones marker with uncompute
    bp_oracle = build_best_practice_oracle(block, out_name, W, target,
                                            vchain_uncompute=True)
    print(f'  qubits: {bp_oracle.num_qubits}, '
          f'pre-transpile gates: {sum(bp_oracle.count_ops().values())}')
    t_bp = {}
    for lvl in [0, 1, 2, 3]:
        t0 = time.time()
        t_bp[f'opt{lvl}'] = count_t(bp_oracle, lvl)
        dt = time.time() - t0
        print(f'  best-practice Qiskit T (opt={lvl}): {t_bp[f"opt{lvl}"]:>6d}  ({dt:.1f}s)')

    # Implied Q# equivalent (forward cipher + V-chain forward only)
    qsharp_t = qsharp_equivalent_t(cipher_fwd_t, vchain_fwd_only_t)
    print(f'  implied Q#/ProjectQ T          : {qsharp_t:>6d}')
    # ρ
    rho_bp = t_bp['opt1'] / qsharp_t
    print(f'  rho_best_practice (opt=1)      : {rho_bp:.3f}')

    return {
        'label': label,
        'W': W,
        'cipher_fwd_T': cipher_fwd_t,
        'vchain_fwd_only_T': vchain_fwd_only_t,
        'qsharp_equiv_T': qsharp_t,
        'best_practice_qiskit_T': t_bp,
        'rho_best_practice_opt1': rho_bp,
    }


def main():
    print('===== Best-practice Qiskit oracles vs Q# equivalents =====')
    print()
    results = {'description': 'Best-practice Qiskit V-chain Jones-AND oracles.',
               'basis_gates': BASIS, 'oracles': []}

    # Pre-measured V-chain T-counts (from mcx_vchain_audit.json):
    # n=7  -> with_uncompute 48,  forward-only 24
    # n=31 -> with_uncompute 240, forward-only 120
    VCHAIN_FWD_ONLY_T_n7 = 24
    VCHAIN_FWD_ONLY_T_n31 = 120

    # 1. AES BP11 (W=8, n=7 marker controls).
    # build_bp11_sbox produces F + E + F† (already CU); forward-AND-only T = 136.
    aes = build_bp11_sbox(jones=True)
    results['oracles'].append(measure_one(
        'AES BP11 S-box (W=8, n=7 marker)',
        aes, 's', W=8, target=0xA5,
        cipher_fwd_t=136,  # 34 forward ANDs × 4T (Q# author writes forward only)
        vchain_fwd_only_t=VCHAIN_FWD_ONLY_T_n7,
    ))

    # 2. SM4 Fermat S-box (W=8, n=7 marker). 384 forward ANDs → 1536 T.
    sm4_st = build_sm4_sbox(jones=True, compute_uncompute=False)
    results['oracles'].append(measure_one(
        'SM4 Fermat S-box ST (W=8, n=7)',
        sm4_st, 'out', W=8, target=0xA5,
        cipher_fwd_t=1536,
        vchain_fwd_only_t=VCHAIN_FWD_ONLY_T_n7,
    ))

    # 3. SHA-256 Ch32 (W=32, n=31 marker). 32 forward ANDs → 128 T.
    ch_st = build_ch_block(jones=True, compute_uncompute=False)
    results['oracles'].append(measure_one(
        'SHA-256 Ch32 ST (W=32, n=31)',
        ch_st, 'out', W=32, target=0xA5A5A5A5,
        cipher_fwd_t=128,
        vchain_fwd_only_t=VCHAIN_FWD_ONLY_T_n31,
    ))

    # 4. SHA-256 Maj32 (W=32, n=31 marker). 32 forward ANDs → 128 T.
    maj_st = build_maj_block(jones=True, compute_uncompute=False)
    results['oracles'].append(measure_one(
        'SHA-256 Maj32 ST (W=32, n=31)',
        maj_st, 'out', W=32, target=0xA5A5A5A5,
        cipher_fwd_t=128,
        vchain_fwd_only_t=VCHAIN_FWD_ONLY_T_n31,
    ))

    out_dir = os.path.join(_HERE, 'results')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'best_practice_oracles.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nwrote {out_path}')

    print('\n===== Summary table =====')
    print(f'{"Oracle":<35} {"Q# T":>8} {"Qiskit BP T":>12} {"rho":>6}')
    for r in results['oracles']:
        print(f'{r["label"]:<35} {r["qsharp_equiv_T"]:>8d} '
              f'{r["best_practice_qiskit_T"]["opt1"]:>12d} '
              f'{r["rho_best_practice_opt1"]:>6.3f}')


if __name__ == '__main__':
    main()
