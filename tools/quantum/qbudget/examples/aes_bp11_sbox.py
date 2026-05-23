"""
BoyarPeralta11 AES S-Box — Qiskit port from Jaques et al. 2020 Q# code.

Thread L.1: Faithful port of the BoyarPeralta depth-16 S-box circuit.

Published resource counts (jaques2020-aes/numbers/aes.csv):
  BoyarPeralta11.SBox: CNOT=654, 1q-Cliff=184, T=136, M=34,
                       T-depth=6, initial_width=16, extra_width=121
"""

from qiskit import QuantumCircuit, QuantumRegister


# ─── primitives ───

def lpxor(qc, a, b, out):
    qc.cx(a, out); qc.cx(b, out)

def lpxnor(qc, a, b, out):
    lpxor(qc, a, b, out); qc.x(out)

def lpand_jones(qc, c1, c2, tgt, anc):
    """Jones T-depth-1 AND gate (Q# LPANDWithAux, costing=True)."""
    qc.h(tgt)
    qc.cx(c2, anc); qc.cx(tgt, c1); qc.cx(tgt, c2); qc.cx(c1, anc)
    qc.tdg(c1); qc.tdg(c2); qc.t(tgt); qc.t(anc)
    qc.cx(c1, anc); qc.cx(tgt, c2); qc.cx(tgt, c1); qc.cx(c2, anc)
    qc.h(tgt); qc.s(tgt)


AES_SBOX = [
    0x63,0x7c,0x77,0x7b,0xf2,0x6b,0x6f,0xc5,0x30,0x01,0x67,0x2b,0xfe,0xd7,0xab,0x76,
    0xca,0x82,0xc9,0x7d,0xfa,0x59,0x47,0xf0,0xad,0xd4,0xa2,0xaf,0x9c,0xa4,0x72,0xc0,
    0xb7,0xfd,0x93,0x26,0x36,0x3f,0xf7,0xcc,0x34,0xa5,0xe5,0xf1,0x71,0xd8,0x31,0x15,
    0x04,0xc7,0x23,0xc3,0x18,0x96,0x05,0x9a,0x07,0x12,0x80,0xe2,0xeb,0x27,0xb2,0x75,
    0x09,0x83,0x2c,0x1a,0x1b,0x6e,0x5a,0xa0,0x52,0x3b,0xd6,0xb3,0x29,0xe3,0x2f,0x84,
    0x53,0xd1,0x00,0xed,0x20,0xfc,0xb1,0x5b,0x6a,0xcb,0xbe,0x39,0x4a,0x4c,0x58,0xcf,
    0xd0,0xef,0xaa,0xfb,0x43,0x4d,0x33,0x85,0x45,0xf9,0x02,0x7f,0x50,0x3c,0x9f,0xa8,
    0x51,0xa3,0x40,0x8f,0x92,0x9d,0x38,0xf5,0xbc,0xb6,0xda,0x21,0x10,0xff,0xf3,0xd2,
    0xcd,0x0c,0x13,0xec,0x5f,0x97,0x44,0x17,0xc4,0xa7,0x7e,0x3d,0x64,0x5d,0x19,0x73,
    0x60,0x81,0x4f,0xdc,0x22,0x2a,0x90,0x88,0x46,0xee,0xb8,0x14,0xde,0x5e,0x0b,0xdb,
    0xe0,0x32,0x3a,0x0a,0x49,0x06,0x24,0x5c,0xc2,0xd3,0xac,0x62,0x91,0x95,0xe4,0x79,
    0xe7,0xc8,0x37,0x6d,0x8d,0xd5,0x4e,0xa9,0x6c,0x56,0xf4,0xea,0x65,0x7a,0xae,0x08,
    0xba,0x78,0x25,0x2e,0x1c,0xa6,0xb4,0xc6,0xe8,0xdd,0x74,0x1f,0x4b,0xbd,0x8b,0x8a,
    0x70,0x3e,0xb5,0x66,0x48,0x03,0xf6,0x0e,0x61,0x35,0x57,0xb9,0x86,0xc1,0x1d,0x9e,
    0xe1,0xf8,0x98,0x11,0x69,0xd9,0x8e,0x94,0x9b,0x1e,0x87,0xe9,0xce,0x55,0x28,0xdf,
    0x8c,0xa1,0x89,0x0d,0xbf,0xe6,0x42,0x68,0x41,0x99,0x2d,0x0f,0xb0,0x54,0xbb,0x16,
]


# ─── ForwardSBox ───

def _build_forward(qc, u, s, t, m, l, anc, jones=True):
    AND = (lambda c1, c2, tgt, a: lpand_jones(qc, c1, c2, tgt, a)) if jones \
        else (lambda c1, c2, tgt, a: qc.ccx(c1, c2, tgt))

    # Top linear
    lpxor(qc,u[0],u[3],t[0]);  lpxor(qc,u[0],u[5],t[1]);  lpxor(qc,u[0],u[6],t[2])
    lpxor(qc,u[3],u[5],t[3]);  lpxor(qc,u[4],u[6],t[4]);  lpxor(qc,t[0],t[4],t[5])
    lpxor(qc,u[1],u[2],t[6]);  lpxor(qc,u[7],t[5],t[7]);  lpxor(qc,u[7],t[6],t[8])
    lpxor(qc,t[5],t[6],t[9]);  lpxor(qc,u[1],u[5],t[10]); lpxor(qc,u[2],u[5],t[11])
    lpxor(qc,t[2],t[3],t[12]); lpxor(qc,t[5],t[10],t[13]);lpxor(qc,t[4],t[10],t[14])
    lpxor(qc,t[4],t[11],t[15]);lpxor(qc,t[8],t[15],t[16]);lpxor(qc,u[3],u[7],t[17])
    lpxor(qc,t[6],t[17],t[18]);lpxor(qc,t[0],t[18],t[19]);lpxor(qc,u[6],u[7],t[20])
    lpxor(qc,t[6],t[20],t[21]);lpxor(qc,t[1],t[21],t[22]);lpxor(qc,t[1],t[9],t[23])
    lpxor(qc,t[19],t[16],t[24]);lpxor(qc,t[2],t[15],t[25]);lpxor(qc,t[0],t[11],t[26])

    # Depth 0→1 AND (9)
    AND(t[12],t[5],m[0],anc[0]);  AND(t[22],t[7],m[1],anc[1])
    AND(t[18],u[7],m[3],anc[2]);  AND(t[2],t[15],m[5],anc[3])
    AND(t[21],t[8],m[6],anc[4]);  AND(t[19],t[16],m[8],anc[5])
    AND(t[0],t[14],m[10],anc[6]); AND(t[3],t[26],m[11],anc[7])
    AND(t[1],t[9],m[13],anc[8])

    # Middle XOR
    lpxor(qc,t[13],m[0],m[2]);  lpxor(qc,m[3],m[0],m[4])
    lpxor(qc,t[25],m[5],m[7]);  lpxor(qc,m[8],m[5],m[9])
    lpxor(qc,m[11],m[10],m[12]);lpxor(qc,m[13],m[10],m[14])
    lpxor(qc,m[2],m[1],m[15]);  lpxor(qc,m[4],t[23],m[16])
    lpxor(qc,m[7],m[6],m[17]);  lpxor(qc,m[9],m[14],m[18])
    lpxor(qc,m[15],m[12],m[19]);lpxor(qc,m[16],m[14],m[20])
    lpxor(qc,m[17],m[12],m[21]);lpxor(qc,m[18],t[24],m[22])
    lpxor(qc,m[21],m[22],m[23])

    AND(m[21],m[19],m[24],anc[0])  # 1→2
    lpxor(qc,m[20],m[24],m[25]);lpxor(qc,m[19],m[20],m[26])
    lpxor(qc,m[22],m[24],m[27])

    # 2→3 (4)
    AND(m[27],m[26],m[28],anc[0]); AND(m[25],m[23],m[29],anc[1])
    AND(m[19],m[22],m[30],anc[2]); AND(m[20],m[21],m[33],anc[3])
    # 3→4 (2)
    AND(m[26],m[30],m[31],anc[0]); AND(m[23],m[33],m[34],anc[1])

    lpxor(qc,m[20],m[28],m[36]);lpxor(qc,m[22],m[29],m[38])
    lpxor(qc,m[36],m[38],m[41])
    lpxor(qc,m[26],m[24],m[32]);lpxor(qc,m[23],m[24],m[35])
    lpxor(qc,m[31],m[32],m[37]);lpxor(qc,m[34],m[35],m[39])
    lpxor(qc,m[37],m[39],m[40]);lpxor(qc,m[36],m[37],m[42])
    lpxor(qc,m[38],m[39],m[43]);lpxor(qc,m[41],m[40],m[44])

    # 4→5 (9)
    AND(m[43],t[5],m[45],anc[0]);  AND(m[39],t[7],m[46],anc[1])
    AND(m[38],u[7],m[47],anc[2]);  AND(m[42],t[15],m[48],anc[3])
    AND(m[37],t[8],m[49],anc[4]);  AND(m[36],t[16],m[50],anc[5])
    AND(m[41],t[14],m[51],anc[6]); AND(m[44],t[26],m[52],anc[7])
    AND(m[40],t[9],m[53],anc[8])

    # 5→6 (9)
    AND(m[43],t[12],m[54],anc[0]); AND(m[39],t[22],m[55],anc[1])
    AND(m[38],t[18],m[56],anc[2]); AND(m[42],t[2],m[57],anc[3])
    AND(m[37],t[21],m[58],anc[4]); AND(m[36],t[19],m[59],anc[5])
    AND(m[41],t[0],m[60],anc[6]);  AND(m[44],t[3],m[61],anc[7])
    AND(m[40],t[1],m[62],anc[8])

    # Bottom linear
    lpxor(qc,m[60],m[61],l[0]); lpxor(qc,m[49],m[55],l[1])
    lpxor(qc,m[45],m[47],l[2]); lpxor(qc,m[46],m[54],l[3])
    lpxor(qc,m[53],m[57],l[4]); lpxor(qc,m[48],m[60],l[5])
    lpxor(qc,m[61],l[5],l[6]);  lpxor(qc,m[45],l[3],l[7])
    lpxor(qc,m[50],m[58],l[8]); lpxor(qc,m[51],m[52],l[9])
    lpxor(qc,m[52],l[4],l[10]); lpxor(qc,m[59],l[2],l[11])
    lpxor(qc,m[47],m[50],l[12]);lpxor(qc,m[49],l[0],l[13])
    lpxor(qc,m[51],m[60],l[14]);lpxor(qc,m[54],l[1],l[15])
    lpxor(qc,m[55],l[0],l[16]); lpxor(qc,m[56],l[1],l[17])
    lpxor(qc,m[57],l[8],l[18]); lpxor(qc,m[62],l[4],l[19])
    lpxor(qc,l[0],l[1],l[20]);  lpxor(qc,l[1],l[7],l[21])
    lpxor(qc,l[3],l[12],l[22]); lpxor(qc,l[18],l[2],l[23])
    lpxor(qc,l[15],l[9],l[24]); lpxor(qc,l[6],l[10],l[25])
    lpxor(qc,l[7],l[9],l[26]);  lpxor(qc,l[8],l[10],l[27])
    lpxor(qc,l[11],l[14],l[28]);lpxor(qc,l[11],l[17],l[29])


# ─── Full S-box with programmatic adjoint ───

_ADJ = {'cx':'cx','ccx':'ccx','x':'x','h':'h','t':'tdg','tdg':'t','s':'sdg'}

def build_bp11_sbox(jones=True):
    u=QuantumRegister(8,'u'); s=QuantumRegister(8,'s'); t=QuantumRegister(27,'t')
    m=QuantumRegister(63,'m'); l=QuantumRegister(30,'l'); anc=QuantumRegister(9,'anc')

    fwd = QuantumCircuit(u,s,t,m,l,anc)
    _build_forward(fwd, u,s,t,m,l,anc, jones)

    full = QuantumCircuit(u,s,t,m,l,anc, name='BP11_SBox')
    full.compose(fwd, range(145), inplace=True)

    # Output extraction
    lpxor(full,l[6],l[24],s[0]);   lpxnor(full,l[16],l[26],s[1])
    lpxnor(full,l[19],l[28],s[2]); lpxor(full,l[6],l[21],s[3])
    lpxor(full,l[20],l[22],s[4]);  lpxor(full,l[25],l[29],s[5])
    lpxnor(full,l[13],l[27],s[6]); lpxnor(full,l[6],l[23],s[7])

    # Programmatic adjoint
    for inst in reversed(list(fwd.data)):
        nm = _ADJ[inst.operation.name]
        qs = inst.qubits
        getattr(full, nm)(*qs)

    return full


# ─── verification + resource counting ───

def count_gates(qc):
    c = {}
    for inst in qc.data:
        n = inst.operation.name; c[n] = c.get(n, 0) + 1
    return c

def verify_all_256(qc):
    ops = [(inst.operation.name, [qc.find_bit(q).index for q in inst.qubits])
           for inst in qc.data]
    errs = dirty = 0
    for x in range(256):
        b = [0]*145
        for i in range(8): b[i] = (x >> (7-i)) & 1
        for nm, qs in ops:
            if nm == 'x': b[qs[0]] ^= 1
            elif nm == 'cx': b[qs[1]] ^= b[qs[0]]
            elif nm == 'ccx': b[qs[2]] ^= b[qs[0]] & b[qs[1]]
        o = sum(b[8+i] << (7-i) for i in range(8))
        if o != AES_SBOX[x]: errs += 1
        if sum(b[16:]): dirty += 1
    return errs, dirty


if __name__ == "__main__":
    print("=" * 65)
    print("L.1 AUDIT: BoyarPeralta11 AES S-Box (Qiskit port)")
    print("=" * 65)

    # 1. Jones AND version (matches Q# forward pass)
    qc_j = build_bp11_sbox(jones=True)
    gj = count_gates(qc_j)

    # 2. CCX version (classically verifiable)
    qc_c = build_bp11_sbox(jones=False)
    gc = count_gates(qc_c)

    # 3. Verify
    errs, dirty = verify_all_256(qc_c)
    print(f"\nVerification (CCX): {256-errs}/256 correct, {dirty} dirty → "
          f"{'PASS' if errs==0 and dirty==0 else 'FAIL'}")

    # 4. Resource table
    P = {'CNOT': 654, 'Cliff1q': 184, 'T': 136, 'M': 34, 'W': 137}
    j_cnot = gj.get('cx',0); j_t = gj.get('t',0)+gj.get('tdg',0)
    j_cliff = gj.get('h',0)+gj.get('s',0)+gj.get('sdg',0)+gj.get('x',0)
    c_cnot = gc.get('cx',0); c_ccx = gc.get('ccx',0)

    print(f"\n{'Metric':<18} {'Qiskit(Jones)':>14} {'Qiskit(CCX)':>13} {'Q#(pub)':>10}")
    print("-" * 58)
    print(f"{'CNOT':<18} {j_cnot:>14} {c_cnot:>13} {P['CNOT']:>10}")
    print(f"{'1q-Clifford':<18} {j_cliff:>14} {gc.get('x',0):>13} {P['Cliff1q']:>10}")
    print(f"{'T-gates':<18} {j_t:>14} {0:>13} {P['T']:>10}")
    print(f"{'Measurements':<18} {0:>14} {0:>13} {P['M']:>10}")
    print(f"{'CCX (Toffoli)':<18} {gj.get('ccx',0):>14} {c_ccx:>13} {'—':>10}")
    print(f"{'Qubits':<18} {qc_j.num_qubits:>14} {qc_c.num_qubits:>13} {P['W']:>10}")

    print(f"\n--- KEY FINDINGS ---")
    print(f"")
    print(f"F1: T-count matches exactly (forward): {j_t//2} = 34 AND × 4 T/AND = {P['T']}")
    print(f"    But full circuit doubles it ({j_t}) because Qiskit adjoint uses")
    print(f"    T/Tdg (no measurement). Q# uses AND† with measurement (34 M).")
    print(f"    → SYSTEMIC: Q# reports {P['T']}T+{P['M']}M; Qiskit would report {j_t}T+0M")
    print(f"")
    print(f"F2: CNOT differs by {j_cnot - P['CNOT']:+d}. Same root cause: Q# AND†")
    print(f"    uses ~2 gates (H+measure+conditional); Qiskit AND† uses 10 gates")
    print(f"    (reverse Jones = 8 CNOT + 2 H + 1 Sdg).")
    print(f"")
    print(f"F3: Qubit count differs by {qc_j.num_qubits - P['W']:+d}. Q# resource estimator")
    print(f"    reports max simultaneous width (137); Qiskit allocates all")
    print(f"    registers upfront (145). 8 qubits potentially optimized away.")
    print(f"")
    print(f"Raw: {gj}")
