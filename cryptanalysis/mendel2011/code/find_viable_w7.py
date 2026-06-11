#!/usr/bin/env python3
"""Find ALL W1[7] values satisfying BOTH EXP[22] and EXP[23].

EXP[22]: (σ₀(W1[7] ⊕ xor_w7) - σ₀(W1[7])) = target_ds0_w7 = 0x33481644
EXP[23]: -(( W1[7] ⊕ xor_w7) - W1[7]) ∈ achievable_dsigma0_w8 (128 values)

Strategy:
1. σ₀ is a GF(2)-linear bijection. Preimage under σ₀ is unique.
2. EXP[22] constrains σ₀(W1[7]) → constrains W1[7] via σ₀⁻¹.
3. EXP[23] constrains W1[7] directly.
4. Enumerate candidate W1[7] satisfying EXP[22], check EXP[23].

Since P(EXP[22]) = 2^{-18}, there are ~2^{14} = 16384 solutions in [0, 2^32).
Brute-forcing 2^32 at Python speed is too slow (~hours).
Instead: we'll use the STRUCTURE of the (s⊕C)-s = target equation.

Key insight: (s⊕C) - s depends only on bits of s where C has bits.
For C = 0x74c9e9bc with 18 set bits, the carry chain propagation
makes this depend on all bits from the LSB of C upward.
But we can enumerate solutions bit-by-bit using a carry-chain approach.
"""

import sys

def rotr(x, n):
    return ((x >> n) | (x << (32-n))) & 0xFFFFFFFF

def sigma0(x):
    return rotr(x,7) ^ rotr(x,18) ^ ((x >> 3) & 0xFFFFFFFF)

# Constants
xor_w7 = 0x00cbf344  # XOR diff of W[7]
C = sigma0(xor_w7)   # = 0x74c9e9bc (σ₀ of XOR diff)
target_ds0_w7 = 0x33481644  # = -dW[15] mod 2^32

# Build achievable dσ₀(W8) set (128 values)
xor_w8 = 0x04880300
sigma0_xor_w8 = sigma0(xor_w8)  # 0x00581144, 7 non-adjacent bits
set_bits_w8 = [b for b in range(32) if (sigma0_xor_w8 >> b) & 1]
dsigma0_w8 = set()
for mask in range(1 << len(set_bits_w8)):
    a = 0
    for i, b in enumerate(set_bits_w8):
        if (mask >> i) & 1:
            a |= (1 << b)
    diff = ((a ^ sigma0_xor_w8) - a) & 0xFFFFFFFF
    dsigma0_w8.add(diff)
print(f"|dσ₀(W[8])| = {len(dsigma0_w8)}")

# Solve (s ⊕ C) - s = target for s, using carry-chain enumeration.
# Let t[i] = target bit i. For each bit position:
#   c_bit = (C >> bit) & 1
#   s_bit = (s >> bit) & 1
#   XOR bit: (s⊕C)_bit = s_bit ^ c_bit
#   Subtraction: (s⊕C) - s at bit i:
#     diff_bit = (s_bit ^ c_bit) - s_bit - borrow_in
#     borrow_out = 1 if result < 0
#
# Actually, let's think of it as addition: target = (s⊕C) + (-s) = (s⊕C) + (~s + 1)
# Or more directly: (s⊕C) - s. Let a = s⊕C, b = s. diff = a - b.
# For subtraction bit-by-bit:
#   diff_bit = a_bit - b_bit - borrow_in (mod 2)
#   borrow_out = (a_bit < b_bit + borrow_in)
#   where a_bit = s_bit ^ c_bit, b_bit = s_bit
#
# For each bit, given s_bit and borrow_in:
#   a_bit = s_bit ^ c_bit
#   temp = a_bit - s_bit - borrow_in (as signed)
#   diff_bit = temp & 1
#   borrow_out = 1 if temp < 0 else 0

def solve_exp22():
    """Find all s values where (s⊕C) - s = target, using bit-by-bit carry chain."""
    # State: list of (borrow, partial_s)
    # Start with borrow=0
    solutions = []
    states = [(0, 0)]  # (borrow_in, partial_s_value)
    
    for bit in range(32):
        c_bit = (C >> bit) & 1
        t_bit = (target_ds0_w7 >> bit) & 1
        next_states = []
        
        for borrow_in, partial_s in states:
            for s_bit in range(2):
                a_bit = s_bit ^ c_bit
                temp = a_bit - s_bit - borrow_in
                diff_bit = temp & 1
                borrow_out = 1 if temp < 0 else 0
                
                if diff_bit == t_bit:
                    new_s = partial_s | (s_bit << bit)
                    next_states.append((borrow_out, new_s))
        
        states = next_states
    
    # Filter: final borrow must be 0 (no overflow for 32-bit subtraction)
    # Actually for mod 2^32 arithmetic, we don't need borrow=0 at end.
    # The subtraction wraps around. So ALL states at bit 31 are valid.
    solutions = [s for (borrow, s) in states]
    return solutions

print("Solving EXP[22]: (s⊕C) - s = target ...")
exp22_solutions = solve_exp22()
print(f"  Found {len(exp22_solutions)} values of s = σ₀(W1[7])")

# Now invert σ₀ to get W1[7] values.
# σ₀ is linear over GF(2), so we can build the inverse matrix.
# Or we can just check: for each s, find w such that σ₀(w) = s.
# Since σ₀ is a bijection (invertible linear map), we can compute σ₀⁻¹.

# Build σ₀ matrix and invert it over GF(2)
def build_sigma0_matrix():
    """Build 32x32 GF(2) matrix for σ₀."""
    M = [[0]*32 for _ in range(32)]
    for col in range(32):
        x = 1 << col
        y = sigma0(x)
        for row in range(32):
            M[row][col] = (y >> row) & 1
    return M

def gf2_invert(M):
    """Invert a 32x32 GF(2) matrix."""
    n = 32
    # Augment with identity
    aug = [row[:] + [1 if i == j else 0 for j in range(n)] for i, row in enumerate(M)]
    
    for col in range(n):
        # Find pivot
        pivot = -1
        for row in range(col, n):
            if aug[row][col]:
                pivot = row
                break
        if pivot == -1:
            return None  # Singular
        aug[col], aug[pivot] = aug[pivot], aug[col]
        
        # Eliminate
        for row in range(n):
            if row != col and aug[row][col]:
                for j in range(2*n):
                    aug[row][j] ^= aug[col][j]
    
    return [row[n:] for row in aug]

def sigma0_inv(y):
    """Compute σ₀⁻¹(y) using precomputed inverse matrix."""
    result = 0
    for col in range(32):
        if (y >> col) & 1:
            for row in range(32):
                result ^= (INV_MATRIX[row][col] << row)
    return result

M = build_sigma0_matrix()
INV_MATRIX = gf2_invert(M)
if INV_MATRIX is None:
    print("ERROR: σ₀ is singular (unexpected)")
    sys.exit(1)

# Better implementation of σ₀⁻¹ using the inverse matrix
def sigma0_inv_fast(y):
    """Compute σ₀⁻¹(y)."""
    result = 0
    for bit in range(32):
        if (y >> bit) & 1:
            # Add column 'bit' of inverse matrix
            col_val = 0
            for row in range(32):
                col_val |= (INV_MATRIX[row][bit] << row)
            result ^= col_val
    return result

# Verify inverse
test_w = 0x278dfd29
assert sigma0_inv_fast(sigma0(test_w)) == test_w, "σ₀⁻¹ verification failed"
print("  σ₀⁻¹ verified OK")

# Convert s solutions to W1[7] values
w7_candidates = []
for s in exp22_solutions:
    w = sigma0_inv_fast(s)
    # Verify
    assert sigma0(w) == s
    assert ((sigma0(w ^ xor_w7) - sigma0(w)) & 0xFFFFFFFF) == target_ds0_w7
    w7_candidates.append(w)

print(f"  Converted to {len(w7_candidates)} W1[7] values")

# Now filter by EXP[23]
# EXP[23]: -(dW[7]) must be in dsigma0_w8 set
# dW[7] = (W1[7] ⊕ xor_w7) - W1[7]
joint_solutions = []
for w in w7_candidates:
    dw7 = ((w ^ xor_w7) - w) & 0xFFFFFFFF
    neg_dw7 = (-dw7) & 0xFFFFFFFF
    if neg_dw7 in dsigma0_w8:
        joint_solutions.append(w)

print(f"\n=== JOINT SOLUTIONS ===")
print(f"EXP[22] solutions: {len(w7_candidates)}")
print(f"EXP[22] + EXP[23] solutions: {len(joint_solutions)}")
if len(joint_solutions) > 0:
    import math
    rate = len(joint_solutions) / 2**32
    print(f"Joint rate: {len(joint_solutions)}/2^32 = 2^{{{math.log2(rate):.2f}}}")
    print(f"\nFirst 10 viable W1[7] values:")
    for i, w in enumerate(joint_solutions[:10]):
        dw7 = ((w ^ xor_w7) - w) & 0xFFFFFFFF
        ds0 = (sigma0(w ^ xor_w7) - sigma0(w)) & 0xFFFFFFFF
        print(f"  W1[7] = 0x{w:08x}  dW7={dw7:08x}  dσ₀(W7)={ds0:08x}")
    
    # Check if Table 8's value is in the set
    if 0x278dfd29 in joint_solutions:
        print(f"\n  ✓ Table 8's W1[7] = 0x278dfd29 is in the set!")
    else:
        print(f"\n  ✗ Table 8's W1[7] NOT in set (unexpected!)")
    
    # These are the W1[7] values we need to target!
    # For each: the attacker must find W[0..6] such that the step-7 computation
    # produces this specific W[7] value.
    print(f"\n=== SEARCH STRATEGY ===")
    print(f"There are {len(joint_solutions)} viable W1[7] targets.")
    print(f"For each W[0..6] (with constructed W[5,6]):")
    print(f"  - Compute pre-state at step 7")
    print(f"  - For each viable W1[7]: check if E[7] = C7 + W[7] satisfies E[7] conditions")
    print(f"  - This is a MEET-IN-THE-MIDDLE: precompute target E[7] values")
    print(f"  - Then for each W[0..6], check if C7 + w7_target gives valid E[7]")
    print(f"  Rate per target: P(E[7] conditions | specific W[7]) ≈ 2^{{-22}} (22 fixed bits)")
    print(f"  With {len(joint_solutions)} targets: P(any target works) ≈ {len(joint_solutions)} × 2^{{-22}}")
    if len(joint_solutions) > 0:
        p = len(joint_solutions) / (2**22)
        print(f"  = 2^{{{math.log2(p):.2f}}} per valid W[0..6]")
        print(f"  At 2^0 valid W[0..6] rate × 88M/s throughput:")
        print(f"  Expected time: 2^{{{-math.log2(p):.2f}}} trials / 88M/s")
        trials_needed = 1.0 / p
        time_s = trials_needed / 88e6
        print(f"  = {trials_needed:.0f} trials = {time_s:.1f}s")
else:
    print("NO JOINT SOLUTIONS! The constraints may be truly incompatible")
    print("for generic W1[7] — Table 8 must exploit additional structure.")
