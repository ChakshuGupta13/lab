// Condition propagation engine for generalized bit conditions.
// Propagates conditions through Boolean functions and modular additions.
//
// Based on DC06 Section IV-A:
//   "bits at different bit positions only interact through the carries
//    of the integer additions"
//
// Shared between SHA-1 (DC06) and SHA-256 (Mendel 2011) implementations.

#pragma once

#include "gencond.hpp"
#include "sha256.hpp"
#include <cstdint>
#include <algorithm>
#include <utility>

namespace gencond {

// ============================================================
// 1. ROTATION — trivial permutation of bit conditions
// ============================================================

template<typename W>
inline WordCondT<W> wc_rotl(const WordCondT<W>& wc, int n) {
    constexpr int BITS = WordCondT<W>::BITS;
    WordCondT<W> out;
    for (int k = 0; k < 4; ++k)
        out.allow[k] = (wc.allow[k] << n) | (wc.allow[k] >> (BITS - n));
    return out;
}

template<typename W>
inline WordCondT<W> wc_rotr(const WordCondT<W>& wc, int n) {
    return wc_rotl(wc, WordCondT<W>::BITS - n);
}

// Shift right (logical): shifted-in positions become BC_0 (both zero).
template<typename W>
inline WordCondT<W> wc_shr(const WordCondT<W>& wc, int n) {
    constexpr int BITS = WordCondT<W>::BITS;
    WordCondT<W> out;
    // After shifting, high bits are zero in both messages.
    // So allow[0] (the (0,0) pair) gets 1s in those positions,
    // and allow[1..3] get 0s.
    W hi_mask = n < BITS ? (W(~W(0)) << (BITS - n)) : W(~W(0));
    W lo_mask = ~hi_mask;
    out.allow[0] = (wc.allow[0] >> n) | hi_mask;
    out.allow[1] = (wc.allow[1] >> n) & lo_mask;
    out.allow[2] = (wc.allow[2] >> n) & lo_mask;
    out.allow[3] = (wc.allow[3] >> n) & lo_mask;
    return out;
}

// XOR of condition words (bit-independent): for each bit position,
// compute the set of allowed output pairs given conditions on two inputs.
template<typename W>
inline WordCondT<W> wc_xor(const WordCondT<W>& a, const WordCondT<W>& b) {
    WordCondT<W> out;
    for (int ok = 0; ok < 4; ++ok) {
        W bits = 0;
        for (int ik = 0; ik < 4; ++ik) {
            int jk = ik ^ ok;
            bits |= a.allow[ik] & b.allow[jk];
        }
        out.allow[ok] = bits;
    }
    return out;
}


// ============================================================
// 2. THREE-INPUT BOOLEAN FUNCTIONS (bit-independent)
// ============================================================

// For 3-input Boolean functions f(x,y,z) applied bitwise:
// Given conditions on pairs (x,x*), (y,y*), (z,z*), compute
// the condition on (f(x,y,z), f(x*,y*,z*)).
//
// A "pair index" k ∈ {0,1,2,3} encodes (val, val*):
//   k=0: (0,0)  k=1: (1,0)  k=2: (0,1)  k=3: (1,1)
// So val = k&1, val* = (k>>1)&1.

// Precomputed propagation table for a 3-input boolean function.
// table[out_pair] = set of (in_x, in_y, in_z) triples that produce out_pair.
//
// Includes precomputed lookup tables for O(1) propagation:
//   fwd_lut[16][16][16]:       forward propagate (4 KB)
//   joint_lut[16][16][16][16]: joint propagate_all (128 KB)
struct BoolPropTable {
    uint64_t achieve[4];
    uint8_t fwd_lut[16][16][16];          // propagate() LUT
    uint16_t joint_lut[16][16][16][16];   // propagate_all() LUT (packed)

    static BoolPropTable build(uint8_t truth_table) {
        BoolPropTable t{};

        // 1. Compute achieve[] masks
        for (int ix = 0; ix < 4; ++ix) {
            int x0 = ix & 1, x1 = (ix >> 1) & 1;
            for (int iy = 0; iy < 4; ++iy) {
                int y0 = iy & 1, y1 = (iy >> 1) & 1;
                for (int iz = 0; iz < 4; ++iz) {
                    int z0 = iz & 1, z1 = (iz >> 1) & 1;
                    int f0 = (truth_table >> (x0*4 + y0*2 + z0)) & 1;
                    int f1 = (truth_table >> (x1*4 + y1*2 + z1)) & 1;
                    int ok = f0 | (f1 << 1);
                    t.achieve[ok] |= 1ULL << (ix * 16 + iy * 4 + iz);
                }
            }
        }

        // 2. Build forward propagation LUT
        for (int cx = 0; cx < 16; ++cx)
            for (int cy = 0; cy < 16; ++cy)
                for (int cz = 0; cz < 16; ++cz) {
                    uint8_t result = 0;
                    for (int ok = 0; ok < 4; ++ok) {
                        for (int ix = 0; ix < 4; ++ix) {
                            if (!((cx >> ix) & 1)) continue;
                            for (int iy = 0; iy < 4; ++iy) {
                                if (!((cy >> iy) & 1)) continue;
                                for (int iz = 0; iz < 4; ++iz) {
                                    if (!((cz >> iz) & 1)) continue;
                                    if ((t.achieve[ok] >> (ix*16 + iy*4 + iz)) & 1) {
                                        result |= (1 << ok);
                                        goto fwd_next;
                                    }
                                }
                            }
                        }
                        fwd_next:;
                    }
                    t.fwd_lut[cx][cy][cz] = result;
                }

        // 3. Build joint propagation LUT
        //    Packed uint16_t: [3:0]=new_x [7:4]=new_y [11:8]=new_z [15:12]=new_out
        //    0 = contradiction (all-or-nothing: if any component is 0, all are)
        for (int cx = 0; cx < 16; ++cx)
            for (int cy = 0; cy < 16; ++cy)
                for (int cz = 0; cz < 16; ++cz)
                    for (int co = 0; co < 16; ++co) {
                        uint8_t nx = 0, ny = 0, nz = 0, no = 0;
                        for (int ok = 0; ok < 4; ++ok) {
                            if (!((co >> ok) & 1)) continue;
                            for (int ix = 0; ix < 4; ++ix) {
                                if (!((cx >> ix) & 1)) continue;
                                for (int iy = 0; iy < 4; ++iy) {
                                    if (!((cy >> iy) & 1)) continue;
                                    for (int iz = 0; iz < 4; ++iz) {
                                        if (!((cz >> iz) & 1)) continue;
                                        if ((t.achieve[ok] >> (ix*16 + iy*4 + iz)) & 1) {
                                            nx |= (1 << ix);
                                            ny |= (1 << iy);
                                            nz |= (1 << iz);
                                            no |= (1 << ok);
                                        }
                                    }
                                }
                            }
                        }
                        uint16_t packed = 0;
                        if (nx && ny && nz && no)
                            packed = nx | (uint16_t(ny) << 4) | (uint16_t(nz) << 8) | (uint16_t(no) << 12);
                        t.joint_lut[cx][cy][cz][co] = packed;
                    }

        return t;
    }

    // Forward propagate: O(1) table lookup.
    BitCond propagate(BitCond cx, BitCond cy, BitCond cz) const {
        return BitCond(fwd_lut[cx][cy][cz]);
    }

    // Joint propagation: O(1) table lookup.
    // Tightens all 4 conditions. Returns false on contradiction.
    bool propagate_all(BitCond& cx, BitCond& cy, BitCond& cz, BitCond& cout) const {
        uint16_t packed = joint_lut[cx][cy][cz][cout];
        if (!packed) return false;
        cx = BitCond(packed & 0xF);
        cy = BitCond((packed >> 4) & 0xF);
        cz = BitCond((packed >> 8) & 0xF);
        cout = BitCond((packed >> 12) & 0xF);
        return true;
    }
};

// SHA-1 / SHA-2 boolean function truth tables (8-bit):
//   f(x,y,z) where bit index = x*4 + y*2 + z
//
//   f_IF (Ch):  (x&y) ^ (~x&z)   → truth table: 0xCA
//     x=0,y=0,z=0: 0  x=0,y=0,z=1: 1  x=0,y=1,z=0: 0  x=0,y=1,z=1: 1
//     x=1,y=0,z=0: 0  x=1,y=0,z=1: 0  x=1,y=1,z=0: 1  x=1,y=1,z=1: 1
//   → 11001010 = 0xCA
//
//   f_XOR (Parity): x^y^z          → truth table: 0x96
//     00: 0  01:1  10:1  11:0  100:1  101:0  110:0  111:1
//   → 10010110 = 0x96
//
//   f_MAJ (Maj): (x&y)^(x&z)^(y&z) → truth table: 0xE8
//     00: 0  01:0  10:0  11:1  100:0  101:1  110:1  111:1
//   → 11101000 = 0xE8

constexpr uint8_t TT_IF  = 0xCA;
constexpr uint8_t TT_XOR = 0x96;
constexpr uint8_t TT_MAJ = 0xE8;

// Propagate a 3-input boolean function across a full word.
template<typename W>
inline WordCondT<W> wc_bool3(const BoolPropTable& tbl,
                          const WordCondT<W>& x, const WordCondT<W>& y, const WordCondT<W>& z) {
    constexpr int BITS = WordCondT<W>::BITS;
    WordCondT<W> out = WordCondT<W>::free();
    for (int b = 0; b < BITS; ++b) {
        out.set(b, tbl.propagate(x.get(b), y.get(b), z.get(b)));
    }
    return out;
}

// Joint propagation through a 3-input boolean function for a full word.
// Tightens all 4 word conditions (3 inputs + 1 output) simultaneously.
template<typename W>
inline bool wc_bool3_propagate(const BoolPropTable& tbl,
                                WordCondT<W>& x, WordCondT<W>& y, WordCondT<W>& z,
                                WordCondT<W>& out) {
    constexpr int BITS = WordCondT<W>::BITS;
    for (int b = 0; b < BITS; ++b) {
        BitCond cx = x.get(b), cy = y.get(b), cz = z.get(b), co = out.get(b);
        if (!tbl.propagate_all(cx, cy, cz, co)) return false;
        x.set(b, cx); y.set(b, cy); z.set(b, cz); out.set(b, co);
    }
    return true;
}


// ============================================================
// 3. MODULAR ADDITION WITH CARRY PROPAGATION
// ============================================================

// Propagate conditions through: out = a + b (mod 2^32).
//
// For each bit position (LSB to MSB), we track the set of possible
// carry-in values as conditions. A carry is a pair (c, c*) where
// c is the carry for message 1 and c* for message 2.
//
// At each bit: out_bit = a_bit XOR b_bit XOR carry_in
//              carry_out = MAJ(a_bit, b_bit, carry_in)
//
// Precomputed lookup tables replace the triple-nested enumeration loop:
//   add_fwd_lut:   [ca][cb][carry] → (out, carry_out)          — 8 KB
//   add_joint_lut: [ca][cb][cs][carry] → (na, nb, ns, nc) packed — 256 KB

namespace detail {

struct AddFwdEntry { uint8_t out_mask, carry_mask; };

struct AddFwdLUT {
    AddFwdEntry data[16][16][16];
};

inline const AddFwdLUT& get_add_fwd_lut() {
    static const AddFwdLUT lut = []() {
        AddFwdLUT t{};
        for (int ca = 0; ca < 16; ++ca)
            for (int cb = 0; cb < 16; ++cb)
                for (int cc = 0; cc < 16; ++cc) {
                    uint8_t out = 0, carry = 0;
                    for (int ia = 0; ia < 4; ++ia) {
                        if (!((ca >> ia) & 1)) continue;
                        int a0 = ia & 1, a1 = (ia >> 1) & 1;
                        for (int ib = 0; ib < 4; ++ib) {
                            if (!((cb >> ib) & 1)) continue;
                            int b0 = ib & 1, b1 = (ib >> 1) & 1;
                            for (int ic = 0; ic < 4; ++ic) {
                                if (!((cc >> ic) & 1)) continue;
                                int c0 = ic & 1, c1 = (ic >> 1) & 1;
                                out |= 1 << ((a0^b0^c0) | ((a1^b1^c1)<<1));
                                carry |= 1 << (((a0&b0)|(a0&c0)|(b0&c0)) |
                                               (((a1&b1)|(a1&c1)|(b1&c1))<<1));
                            }
                        }
                    }
                    t.data[ca][cb][cc] = {out, carry};
                }
        return t;
    }();
    return lut;
}

struct AddJointEntry { uint8_t a, b, s, carry; };

struct AddJointLUT {
    AddJointEntry data[16][16][16][16];
};

inline const AddJointLUT& get_add_joint_lut() {
    // Split static pattern: lut is too large (256KB) to return by value from a lambda.
    static AddJointLUT lut{};
    static bool init = []() {
        for (int ca = 0; ca < 16; ++ca)
            for (int cb = 0; cb < 16; ++cb)
                for (int cs = 0; cs < 16; ++cs)
                    for (int cc = 0; cc < 16; ++cc) {
                        uint8_t na = 0, nb = 0, ns = 0, nc = 0;
                        for (int ia = 0; ia < 4; ++ia) {
                            if (!((ca >> ia) & 1)) continue;
                            int a0 = ia & 1, a1 = (ia >> 1) & 1;
                            for (int ib = 0; ib < 4; ++ib) {
                                if (!((cb >> ib) & 1)) continue;
                                int b0 = ib & 1, b1 = (ib >> 1) & 1;
                                for (int ic = 0; ic < 4; ++ic) {
                                    if (!((cc >> ic) & 1)) continue;
                                    int c0 = ic & 1, c1 = (ic >> 1) & 1;
                                    int s0 = a0 ^ b0 ^ c0;
                                    int s1 = a1 ^ b1 ^ c1;
                                    int sk = s0 | (s1 << 1);
                                    if (!((cs >> sk) & 1)) continue;
                                    int co0 = (a0&b0)|(a0&c0)|(b0&c0);
                                    int co1 = (a1&b1)|(a1&c1)|(b1&c1);
                                    na |= (1 << ia);
                                    nb |= (1 << ib);
                                    ns |= (1 << sk);
                                    nc |= (1 << (co0 | (co1 << 1)));
                                }
                            }
                        }
                        lut.data[ca][cb][cs][cc] = {na, nb, ns, nc};
                    }
        return true;
    }();
    (void)init;
    return lut;
}

} // namespace detail

template<typename W>
inline WordCondT<W> wc_add(const WordCondT<W>& a, const WordCondT<W>& b) {
    static const auto& lut = detail::get_add_fwd_lut();
    WordCondT<W> out{};
    uint8_t carry = uint8_t(BC_0);
    for (int bit = 0; bit < WordCondT<W>::BITS; ++bit) {
        const auto& e = lut.data[a.get(bit)][b.get(bit)][carry];
        out.set(bit, BitCond(e.out_mask));
        carry = e.carry_mask;
    }
    return out;
}

template<typename W>
inline bool wc_add_propagate(WordCondT<W>& a, WordCondT<W>& b, WordCondT<W>& sum) {
    static const auto& lut = detail::get_add_joint_lut();
    uint8_t carry = uint8_t(BC_0);
    for (int bit = 0; bit < WordCondT<W>::BITS; ++bit) {
        const auto& e = lut.data[a.get(bit)][b.get(bit)][sum.get(bit)][carry];
        if (!e.a) return false;
        a.set(bit, BitCond(e.a));
        b.set(bit, BitCond(e.b));
        sum.set(bit, BitCond(e.s));
        carry = e.carry;
    }
    return true;
}

// N-input modular addition: out = a0 + a1 + ... + a_{n-1}
// Chains pairwise additions.
template<typename W>
inline WordCondT<W> wc_add_n(const WordCondT<W>* terms, int n) {
    if (n == 0) return WordCondT<W>::contradiction();
    WordCondT<W> acc = terms[0];
    for (int i = 1; i < n; ++i)
        acc = wc_add(acc, terms[i]);
    return acc;
}


// Joint N-input modular addition propagation at a single bit position.
//
// Given N addend BitConds, a sum BitCond, and a carry-in pair bitmask,
// enumerates all valid (addend_pair_tuple × carry_in_pair) combinations,
// filters by the sum condition, and tightens all addend + sum conditions.
//
// Carry pair encoding: bit index = c_star * n + c
//   c ∈ [0, n-1], c_star ∈ [0, n-1] (max carry value = n-1 at steady state).
//   Pair space = n². For n=6: 36 states. For n=2: 4 states (matches BitCond).
//
// Supports n ∈ [2, 8] (pair space ≤ 64, fits uint64_t).
// Returns false on contradiction.
inline bool add_n_bit_joint(BitCond* addends, int n, BitCond& sum,
                            uint64_t& carry_mask) {
    // Precompute valid pairs for each addend
    int pair_list[8][4];
    int pair_count[8];
    for (int k = 0; k < n; ++k) {
        pair_count[k] = 0;
        uint8_t mask = addends[k];
        for (int p = 0; p < 4; ++p)
            if ((mask >> p) & 1)
                pair_list[k][pair_count[k]++] = p;
        if (pair_count[k] == 0) { carry_mask = 0; return false; }
    }

    uint8_t new_a[8] = {};
    uint8_t new_s = 0;
    uint64_t new_c = 0;

    // Enumerate each valid carry-in pair
    for (int ci = 0; ci < n * n; ++ci) {
        if (!(carry_mask & (uint64_t(1) << ci))) continue;
        int cin0 = ci % n;   // carry value, message 1
        int cin1 = ci / n;   // carry value, message 2

        // Mixed-radix enumeration over addend pairs
        int digits[8] = {};
        for (;;) {
            int t0 = cin0, t1 = cin1;
            for (int k = 0; k < n; ++k) {
                int p = pair_list[k][digits[k]];
                t0 += p & 1;          // message 1 bit
                t1 += (p >> 1) & 1;   // message 2 bit
            }

            int s_pair = (t0 & 1) | ((t1 & 1) << 1);
            if ((uint8_t(sum) >> s_pair) & 1) {
                for (int k = 0; k < n; ++k)
                    new_a[k] |= uint8_t(1 << pair_list[k][digits[k]]);
                new_s |= uint8_t(1 << s_pair);
                int cout0 = t0 >> 1, cout1 = t1 >> 1;
                new_c |= uint64_t(1) << (cout1 * n + cout0);
            }

            // Increment mixed-radix counter (last addend fastest)
            int pos = n - 1;
            while (pos >= 0) {
                if (++digits[pos] < pair_count[pos]) break;
                digits[pos] = 0;
                --pos;
            }
            if (pos < 0) break;
        }
    }

    // Check for contradiction
    for (int k = 0; k < n; ++k)
        if (!new_a[k]) { carry_mask = 0; return false; }
    if (!new_s || !new_c) { carry_mask = 0; return false; }

    // Apply tightened conditions
    for (int k = 0; k < n; ++k)
        addends[k] = BitCond(new_a[k]);
    sum = BitCond(new_s);
    carry_mask = new_c;
    return true;
}

// Word-level joint N-input modular addition propagation.
// sum = addends[0] + addends[1] + ... + addends[n-1]  (mod 2^BITS)
// Tightens all addend and sum WordConds in place. Returns false on contradiction.
// Supports n ∈ [2, 8].
template<typename W>
inline bool wc_add_n_propagate(WordCondT<W>* addends, int n, WordCondT<W>& sum) {
    constexpr int BITS = WordCondT<W>::BITS;
    // Initial carry = (0,0) → bit index 0*n+0 = 0
    uint64_t carry = uint64_t(1) << 0;
    for (int bit = 0; bit < BITS; ++bit) {
        BitCond a[8];
        for (int k = 0; k < n; ++k)
            a[k] = addends[k].get(bit);
        BitCond s = sum.get(bit);
        if (!add_n_bit_joint(a, n, s, carry))
            return false;
        for (int k = 0; k < n; ++k)
            addends[k].set(bit, a[k]);
        sum.set(bit, s);
    }
    return true;
}


// ============================================================
// 4. FULL SHA-1 STEP PROPAGATION
// ============================================================

// Given conditions on the 5 input state words (A,B,C,D,E) and the
// message word W at step i, compute the output condition on A'.
//
// A' = rotl(A, 5) + f(B, C, D) + E + K + W
//
// K is a constant (both messages use the same K), so its condition
// is fully determined: BC_0 for zero bits, BC_1 for one bits.

inline WordCond wc_from_constant(uint32_t val) {
    return WordCond::from_pair(val, val);
}

inline WordCond64 wc_from_constant(uint64_t val) {
    return WordCond64::from_pair(val, val);
}

struct SHA1StepCond {
    WordCond A, B, C, D, E, W;  // input conditions
    int step;                    // step number (for K and f selection)

    WordCond propagate_A() const {
        // 1. rotl(A, 5)
        WordCond rotA = wc_rotl(A, 5);

        // 2. f(B, C, D) — depends on round
        static const BoolPropTable tbl_if  = BoolPropTable::build(TT_IF);
        static const BoolPropTable tbl_xor = BoolPropTable::build(TT_XOR);
        static const BoolPropTable tbl_maj = BoolPropTable::build(TT_MAJ);

        const BoolPropTable* tbl;
        if      (step < 20) tbl = &tbl_if;
        else if (step < 40) tbl = &tbl_xor;
        else if (step < 60) tbl = &tbl_maj;
        else                tbl = &tbl_xor;

        WordCond fBCD = wc_bool3(*tbl, B, C, D);

        // 3. K constant
        uint32_t k_val;
        if      (step < 20) k_val = 0x5A827999;
        else if (step < 40) k_val = 0x6ED9EBA1;
        else if (step < 60) k_val = 0x8F1BBCDC;
        else                k_val = 0xCA62C1D6;
        WordCond K_cond = wc_from_constant(k_val);

        // 4. Sum: rotl(A,5) + f(B,C,D) + E + K + W
        WordCond terms[5] = { rotA, fBCD, E, K_cond, W };
        return wc_add_n(terms, 5);
    }
};


// ============================================================
// 5. BACKWARD INFERENCE (tighten input conditions from output)
// ============================================================

// Given conditions on A' (output) and some inputs, tighten the input
// conditions. This is a constraint propagation step.
//
// For now, provide a consistency check: given all conditions, verify
// that the output is at least as permissive as claimed.

inline bool sha1_step_consistent(int step,
                                  const WordCond& A, const WordCond& B,
                                  const WordCond& C, const WordCond& D,
                                  const WordCond& E, const WordCond& W,
                                  const WordCond& A_out) {
    SHA1StepCond sc{A, B, C, D, E, W, step};
    WordCond computed = sc.propagate_A();
    // A_out should be a refinement of computed (every pair allowed by A_out
    // must also be allowed by computed).
    for (int b = 0; b < 32; ++b) {
        BitCond claimed = A_out.get(b);
        BitCond possible = computed.get(b);
        if (claimed & ~possible) return false;  // claimed allows pairs that aren't possible
    }
    return true;
}

// ============================================================
// 6. BITWISE NOT AND TWO'S COMPLEMENT NEGATION
// ============================================================

// NOT: flip each bit in both messages.  Bit-independent.
// Pair mapping: (0,0)↔(1,1), (1,0)↔(0,1).
template<typename W>
inline WordCondT<W> wc_not(const WordCondT<W>& wc) {
    return {{wc.allow[3], wc.allow[2], wc.allow[1], wc.allow[0]}};
}

// Two's complement negation: -x = ~x + 1.
// NOT is bit-independent; the +1 propagates carries from LSB.
template<typename W>
inline WordCondT<W> wc_neg(const WordCondT<W>& wc) {
    return wc_add(wc_not(wc), wc_from_constant(W(1)));
}

// Backward through negation: given conditions on x and -x, tighten both.
// Decomposes -x = ~x + 1 and uses wc_add_propagate on the +1 step.
// NOT is exact and self-inverse, so no precision loss there.
template<typename W>
inline bool wc_neg_propagate(WordCondT<W>& x, WordCondT<W>& neg_x) {
    WordCondT<W> not_x = wc_not(x);
    WordCondT<W> one = wc_from_constant(W(1));
    if (!wc_add_propagate(not_x, one, neg_x)) return false;
    x = wc_not(not_x);
    return true;
}


// ============================================================
// 7. SHA-256 SUB-STEP PROPAGATION (Mendel 2011 §4.3)
// ============================================================
//
// Each SHA-256 step is split into 9 sub-steps to reduce per-bit
// input count.  Bitwise sub-steps (Σ, σ, Ch, Maj) have 3 input
// bits each and can be precomputed.  Modular additions chain
// pairwise through wc_add (2 addend bits + carry = 3 bit-pairs).
//
// Trade-off: inter-sub-step relations are lost (over-approximation),
// giving ~10,000× speedup at the cost of precision.

// ---- Sub-step functions: SHA-256 Sigma/sigma ----

// Big Sigma: used on state registers (pure rotation XOR, no shift).
// Σ₀(x) = rotr2(x) XOR rotr13(x) XOR rotr22(x)
// Σ₁(x) = rotr6(x) XOR rotr11(x) XOR rotr25(x)

inline WordCond wc_Sigma0(const WordCond& x) {
    return wc_xor(wc_xor(wc_rotr(x, 2), wc_rotr(x, 13)), wc_rotr(x, 22));
}

inline WordCond wc_Sigma1(const WordCond& x) {
    return wc_xor(wc_xor(wc_rotr(x, 6), wc_rotr(x, 11)), wc_rotr(x, 25));
}

// Bidirectional Sigma: given conditions on x and Σ(x), tighten both.
// Each output bit j = x[(j+r1)%BITS] XOR x[(j+r2)%BITS] XOR x[(j+r3)%BITS].
// Uses XOR backward to jointly constrain all four conditions per bit.
template<typename W>
inline bool wc_Sigma_propagate(WordCondT<W>& x, WordCondT<W>& out, int r1, int r2, int r3) {
    constexpr int BITS = WordCondT<W>::BITS;
    static const BoolPropTable tbl_xor = BoolPropTable::build(TT_XOR);
    for (int j = 0; j < BITS; ++j) {
        int pa = (j + r1) & (BITS - 1), pb = (j + r2) & (BITS - 1), pc = (j + r3) & (BITS - 1);
        BitCond ca = x.get(pa), cb = x.get(pb), cc = x.get(pc), co = out.get(j);
        if (!tbl_xor.propagate_all(ca, cb, cc, co)) return false;
        x.set(pa, ca); x.set(pb, cb); x.set(pc, cc); out.set(j, co);
    }
    return true;
}

inline bool wc_Sigma0_propagate(WordCond& x, WordCond& out) {
    return wc_Sigma_propagate(x, out, 2, 13, 22);
}

inline bool wc_Sigma1_propagate(WordCond& x, WordCond& out) {
    return wc_Sigma_propagate(x, out, 6, 11, 25);
}

// Small sigma: used in message expansion (rotation XOR + logical shift).
// σ₀(x) = rotr7(x) XOR rotr18(x) XOR shr3(x)
// σ₁(x) = rotr17(x) XOR rotr19(x) XOR shr10(x)

inline WordCond wc_sigma0(const WordCond& x) {
    return wc_xor(wc_xor(wc_rotr(x, 7), wc_rotr(x, 18)), wc_shr(x, 3));
}

inline WordCond wc_sigma1(const WordCond& x) {
    return wc_xor(wc_xor(wc_rotr(x, 17), wc_rotr(x, 19)), wc_shr(x, 10));
}

// Bidirectional sigma (with logical shift): σ(x) = rotr_r1(x) ^ rotr_r2(x) ^ shr_s(x).
// For bit j: out[j] = x[(j+r1)%BITS] ^ x[(j+r2)%BITS] ^ (j+s < BITS ? x[j+s] : 0).
template<typename W>
inline bool wc_sigma_propagate(WordCondT<W>& x, WordCondT<W>& out, int r1, int r2, int s) {
    constexpr int BITS = WordCondT<W>::BITS;
    static const BoolPropTable tbl_xor = BoolPropTable::build(TT_XOR);
    for (int j = 0; j < BITS; ++j) {
        int pa = (j + r1) & (BITS - 1), pb = (j + r2) & (BITS - 1);
        BitCond ca = x.get(pa), cb = x.get(pb);
        BitCond cc = (j + s < BITS) ? x.get(j + s) : BC_0;
        BitCond co = out.get(j);
        if (!tbl_xor.propagate_all(ca, cb, cc, co)) return false;
        x.set(pa, ca); x.set(pb, cb);
        if (j + s < BITS) x.set(j + s, cc);
        out.set(j, co);
    }
    return true;
}

inline bool wc_sigma0_propagate(WordCond& x, WordCond& out) {
    return wc_sigma_propagate(x, out, 7, 18, 3);
}

inline bool wc_sigma1_propagate(WordCond& x, WordCond& out) {
    return wc_sigma_propagate(x, out, 17, 19, 10);
}

// ---- Sub-step functions: SHA-256 boolean functions ----

template<typename W>
inline WordCondT<W> wc_Ch(const WordCondT<W>& x, const WordCondT<W>& y, const WordCondT<W>& z) {
    static const BoolPropTable tbl = BoolPropTable::build(TT_IF);
    return wc_bool3(tbl, x, y, z);
}

template<typename W>
inline WordCondT<W> wc_Maj(const WordCondT<W>& x, const WordCondT<W>& y, const WordCondT<W>& z) {
    static const BoolPropTable tbl = BoolPropTable::build(TT_MAJ);
    return wc_bool3(tbl, x, y, z);
}

// Bidirectional Ch and Maj: tighten all 3 inputs + output jointly.
template<typename W>
inline bool wc_Ch_propagate(WordCondT<W>& x, WordCondT<W>& y, WordCondT<W>& z, WordCondT<W>& out) {
    static const BoolPropTable tbl = BoolPropTable::build(TT_IF);
    return wc_bool3_propagate(tbl, x, y, z, out);
}

template<typename W>
inline bool wc_Maj_propagate(WordCondT<W>& x, WordCondT<W>& y, WordCondT<W>& z, WordCondT<W>& out) {
    static const BoolPropTable tbl = BoolPropTable::build(TT_MAJ);
    return wc_bool3_propagate(tbl, x, y, z, out);
}

// ---- Full forward propagation for one SHA-256 alt-step ----
//
// Uses the alternative description (Eq. 2):
//   E_i = E_{i-4} + Σ₁(E_{i-1}) + Ch(E_{i-1}, E_{i-2}, E_{i-3}) + A_{i-4} + K_i + W_i
//   A_i = -A_{i-4} + Σ₀(A_{i-1}) + Maj(A_{i-1}, A_{i-2}, A_{i-3}) + E_i
//
// Inputs are WordCond conditions on each register.
// Outputs are conditions on E_i and A_i (over-approximation).

struct SHA256AltStepCond {
    // Input conditions (subscripts relative to step i):
    WordCond A_im4, A_im3, A_im2, A_im1;  // A_{i-4} .. A_{i-1}
    WordCond E_im4, E_im3, E_im2, E_im1;  // E_{i-4} .. E_{i-1}
    WordCond W_i;                           // message word
    int step;                               // for K_i constant

    // Forward-propagate E_i.
    WordCond propagate_E() const {
        // Sub-step 1: Σ₁(E_{i-1})
        WordCond sig1 = wc_Sigma1(E_im1);
        // Sub-step 2: Ch(E_{i-1}, E_{i-2}, E_{i-3})
        WordCond ch = wc_Ch(E_im1, E_im2, E_im3);
        // Sub-step 3: E_i = E_{i-4} + sig1 + ch + A_{i-4} + K_i + W_i
        WordCond k_cond = wc_from_constant(sha256::K[step]);
        WordCond t = wc_add(E_im4, sig1);
        t = wc_add(t, ch);
        t = wc_add(t, A_im4);
        t = wc_add(t, k_cond);
        return wc_add(t, W_i);
    }

    // Forward-propagate A_i (requires E_i condition, computed first).
    WordCond propagate_A(const WordCond& E_i) const {
        // Sub-step 4: Σ₀(A_{i-1})
        WordCond sig0 = wc_Sigma0(A_im1);
        // Sub-step 5: Maj(A_{i-1}, A_{i-2}, A_{i-3})
        WordCond maj = wc_Maj(A_im1, A_im2, A_im3);
        // Sub-step 6: A_i = -A_{i-4} + sig0 + maj + E_i
        WordCond neg_A = wc_neg(A_im4);
        WordCond t = wc_add(neg_A, sig0);
        t = wc_add(t, maj);
        return wc_add(t, E_i);
    }

    // Forward-propagate both E_i and A_i.
    std::pair<WordCond, WordCond> propagate() const {
        WordCond E_i = propagate_E();
        WordCond A_i = propagate_A(E_i);
        return {A_i, E_i};
    }

    // Backward from E_i: given (possibly tighter) E_i condition,
    // tighten input conditions in place via joint addition propagation.
    // Modifies member fields (E_im*, A_im4, W_i). Returns false on contradiction.
    bool backward_from_E(WordCond& E_i) {
        // Recompute intermediate sub-step results (forward direction)
        WordCond sig1 = wc_Sigma1(E_im1);
        WordCond ch = wc_Ch(E_im1, E_im2, E_im3);
        WordCond k_cond = wc_from_constant(sha256::K[step]);

        // Forward addition chain to get intermediates
        WordCond t1 = wc_add(E_im4, sig1);
        WordCond t2 = wc_add(t1, ch);
        WordCond t3 = wc_add(t2, A_im4);
        WordCond t4 = wc_add(t3, k_cond);
        // E_i is given (potentially tighter than t4 + W_i)

        // Backward: joint propagate through each addition in reverse
        if (!wc_add_propagate(t4, W_i, E_i)) return false;
        if (!wc_add_propagate(t3, k_cond, t4)) return false;
        if (!wc_add_propagate(t2, A_im4, t3)) return false;
        if (!wc_add_propagate(t1, ch, t2)) return false;
        if (!wc_add_propagate(E_im4, sig1, t1)) return false;

        // Backward through Σ₁ and Ch to tighten state registers
        if (!wc_Sigma1_propagate(E_im1, sig1)) return false;
        if (!wc_Ch_propagate(E_im1, E_im2, E_im3, ch)) return false;

        return true;
    }

    // Backward from A_i: given (possibly tighter) A_i condition,
    // tighten input conditions in place. E_i is also tightened.
    bool backward_from_A(WordCond& A_i, WordCond& E_i) {
        // Recompute intermediates
        WordCond sig0 = wc_Sigma0(A_im1);
        WordCond maj = wc_Maj(A_im1, A_im2, A_im3);
        WordCond neg_A = wc_neg(A_im4);

        // Forward addition chain
        WordCond t1 = wc_add(neg_A, sig0);
        WordCond t2 = wc_add(t1, maj);
        // A_i is given (potentially tighter than t2 + E_i)

        // Backward: joint propagate in reverse
        if (!wc_add_propagate(t2, E_i, A_i)) return false;
        if (!wc_add_propagate(t1, maj, t2)) return false;
        if (!wc_add_propagate(neg_A, sig0, t1)) return false;

        // Backward through negation to tighten A_{i-4}
        if (!wc_neg_propagate(A_im4, neg_A)) return false;

        // Backward through Σ₀ and Maj to tighten state registers
        if (!wc_Sigma0_propagate(A_im1, sig0)) return false;
        if (!wc_Maj_propagate(A_im1, A_im2, A_im3, maj)) return false;

        return true;
    }
};

// ---- Message expansion propagation (for step i >= 16) ----
//
// W_i = σ₁(W_{i-2}) + W_{i-7} + σ₀(W_{i-15}) + W_{i-16}

struct SHA256MsgExpCond {
    WordCond W_im2, W_im7, W_im15, W_im16;

    WordCond propagate() const {
        // Sub-step 7: σ₁(W_{i-2})
        WordCond s1 = wc_sigma1(W_im2);
        // Sub-step 8: σ₀(W_{i-15})
        WordCond s0 = wc_sigma0(W_im15);
        // Sub-step 9: W_i = s1 + W_{i-7} + s0 + W_{i-16}
        WordCond t = wc_add(s1, W_im7);
        t = wc_add(t, s0);
        return wc_add(t, W_im16);
    }

    // Backward: given (possibly tighter) W_i condition, tighten inputs.
    bool backward(WordCond& W_i) {
        // Recompute intermediates
        WordCond s1 = wc_sigma1(W_im2);
        WordCond s0 = wc_sigma0(W_im15);

        // Forward addition chain to get intermediates
        WordCond t1 = wc_add(s1, W_im7);
        WordCond t2 = wc_add(t1, s0);
        // W_i = t2 + W_im16

        // Backward through additions
        if (!wc_add_propagate(t2, W_im16, W_i)) return false;
        if (!wc_add_propagate(t1, s0, t2)) return false;
        if (!wc_add_propagate(s1, W_im7, t1)) return false;

        // Backward through σ functions
        if (!wc_sigma1_propagate(W_im2, s1)) return false;
        if (!wc_sigma0_propagate(W_im15, s0)) return false;

        return true;
    }
};

} // namespace gencond
