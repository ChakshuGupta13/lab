// SAT-like search engine for SHA-512 differential characteristics.
// Adapts the SHA-256 search engine to 64-bit words
// with SHA-512 rotation constants and 80-round message schedule.
//
// Three-stage search (Mendel et al. 2013, extended by Eichlseder et al. 2014):
//   Stage 1: resolve '?' → '-' and 'x' → 'u'/'n' (differential pattern)
//   Stage 2: resolve '-' → '0'/'1' for high-connectivity bits
//   Stage 3: resolve remaining '-' → '0'/'1' (conforming message pair)
//
// Includes the look-ahead branching heuristic (Algorithm 2, Eichlseder 2014).

#pragma once

#include "gencond.hpp"
#include "propagate.hpp"
#include "propagate_512.hpp"
#include "alt_step_512.hpp"
#include "twobit.hpp"
#include "sha512.hpp"
#include <cstdint>
#include <cstdio>
#include <vector>
#include <random>
#include <algorithm>
#include <numeric>
#include <atomic>
#include <chrono>

namespace eichlseder2014 {

using namespace gencond;
using namespace twobit;

// Bits per word for SHA-512.
static constexpr int BITS = 64;

// Create a WordCond64 with all 64 bits set to the same condition.
inline WordCond64 wc_uniform_64(BitCond c) {
    uint64_t mask = ~uint64_t(0);
    WordCond64 wc;
    wc.allow[0] = (c & 1) ? mask : 0;
    wc.allow[1] = (c & 2) ? mask : 0;
    wc.allow[2] = (c & 4) ? mask : 0;
    wc.allow[3] = (c & 8) ? mask : 0;
    return wc;
}

// ---- Characteristic state ----
// Holds conditions on A, E (steps -4..N-1) and W (steps 0..79).

struct CharState512 {
    int n_steps;        // number of SHA-512 steps to attack

    // Conditions indexed by step+4 offset (so A[0]=A_{-4}, A[4+i]=A_i)
    std::vector<WordCond64> A;   // size = n_steps + 4
    std::vector<WordCond64> E;   // size = n_steps + 4
    std::vector<WordCond64> W;   // size = 80 (full expansion schedule)

    CharState512() : n_steps(0) {}

    explicit CharState512(int steps) : n_steps(steps),
        A(steps + 4, WordCond64::free()),
        E(steps + 4, WordCond64::free()),
        W(80, WordCond64::free()) {}

    WordCond64& a(int step) { return A[step + 4]; }
    WordCond64& e(int step) { return E[step + 4]; }
    WordCond64& w(int step) { return W[step]; }
    const WordCond64& a(int step) const { return A[step + 4]; }
    const WordCond64& e(int step) const { return E[step + 4]; }
    const WordCond64& w(int step) const { return W[step]; }

    // Count Phase 1 undecided bits (? and x) in the attack-relevant range.
    int count_phase1_undecided() const {
        int count = 0;
        for (auto& wc : A) { count += wc.count_free(); count += wc.count_x(); }
        for (auto& wc : E) { count += wc.count_free(); count += wc.count_x(); }
        for (int i = 0; i < n_steps && i < (int)W.size(); ++i) {
            count += W[i].count_free();
            count += W[i].count_x();
        }
        return count;
    }

    // Count free (?) bits across all words.
    int count_free() const {
        int count = 0;
        for (auto& wc : A) count += wc.count_free();
        for (auto& wc : E) count += wc.count_free();
        for (auto& wc : W) count += wc.count_free();
        return count;
    }

    // Count '-' bits in the attack-relevant range.
    int count_dash() const {
        int count = 0;
        for (auto& wc : A) count += wc.count_dash();
        for (auto& wc : E) count += wc.count_dash();
        for (int i = 0; i < n_steps && i < (int)W.size(); ++i)
            count += W[i].count_dash();
        return count;
    }

    // Check for any contradiction bit.
    bool has_contradiction() const {
        for (auto& wc : A) if (!wc.is_consistent()) return true;
        for (auto& wc : E) if (!wc.is_consistent()) return true;
        for (auto& wc : W) if (!wc.is_consistent()) return true;
        return false;
    }

    // Count fully decided bits: exactly one allowed value pair.
    // Used by look-ahead scoring (Algorithm 2).
    int count_decided() const {
        auto word_decided = [](const WordCond64& wc) {
            uint64_t d0 = wc.allow[0] & ~wc.allow[1] & ~wc.allow[2] & ~wc.allow[3];
            uint64_t d1 = ~wc.allow[0] & wc.allow[1] & ~wc.allow[2] & ~wc.allow[3];
            uint64_t d2 = ~wc.allow[0] & ~wc.allow[1] & wc.allow[2] & ~wc.allow[3];
            uint64_t d3 = ~wc.allow[0] & ~wc.allow[1] & ~wc.allow[2] & wc.allow[3];
            return gc_popcount64(d0 | d1 | d2 | d3);
        };
        int count = 0;
        for (auto& wc : A) count += word_decided(wc);
        for (auto& wc : E) count += word_decided(wc);
        for (auto& wc : W) count += word_decided(wc);
        return count;
    }
};

// ---- Bit location in the characteristic ----

enum WordType : uint8_t { WT_A = 0, WT_E = 1, WT_W = 2 };

struct BitLoc {
    WordType type;
    int step;     // step index (-4..N-1 for A/E, 0..79 for W)
    int bit;      // bit position 0..63

    bool operator==(const BitLoc& o) const {
        return type == o.type && step == o.step && bit == o.bit;
    }
};

// Access a WordCond64 from CharState512 by BitLoc.
inline WordCond64& get_word(CharState512& cs, const BitLoc& loc) {
    switch (loc.type) {
        case WT_A: return cs.a(loc.step);
        case WT_E: return cs.e(loc.step);
        case WT_W: return cs.w(loc.step);
    }
    GC_UNREACHABLE();
}

inline BitCond get_cond(const CharState512& cs, const BitLoc& loc) {
    switch (loc.type) {
        case WT_A: return cs.a(loc.step).get(loc.bit);
        case WT_E: return cs.e(loc.step).get(loc.bit);
        case WT_W: return cs.w(loc.step).get(loc.bit);
    }
    GC_UNREACHABLE();
}

// ---- Decision record (for backtracking trail) ----

struct Decision {
    BitLoc loc;
    BitCond original;     // condition before this decision
    BitCond first_choice; // what we imposed first
    BitCond alt_choice;   // alternative (second) choice
    bool tried_alt;       // whether we've tried the alternative
    bool critical;        // marked as critical on double-fail
    bool stacked;         // if false, backtracking skips alternatives (dobraunig2015 Table 2)
    CharState512 snapshot; // state before this decision
};

// ---- Forward propagation ----

inline bool impose_track(WordCond64& target, const WordCond64& source, bool& changed) {
    WordCond64 before = target;
    if (!target.impose_word(source)) return false;
    if (target != before) changed = true;
    return true;
}

inline bool propagate_once(CharState512& cs, bool& changed, int* fail_step = nullptr) {
    changed = false;
    // fail_step encoding: 0..n-1 = step function, 1000+i = message expansion W[i]
    auto fail = [&](int s) -> bool { if (fail_step) *fail_step = s; return false; };

    // Forward: SHA-512 alt-step for each step
    for (int step = 0; step < cs.n_steps; ++step) {
        SHA512AltStepCond sc;
        sc.A_im4 = cs.a(step - 4);
        sc.A_im3 = cs.a(step - 3);
        sc.A_im2 = cs.a(step - 2);
        sc.A_im1 = cs.a(step - 1);
        sc.E_im4 = cs.e(step - 4);
        sc.E_im3 = cs.e(step - 3);
        sc.E_im2 = cs.e(step - 2);
        sc.E_im1 = cs.e(step - 1);
        sc.W_i = cs.w(step);
        sc.step = step;

        WordCond64 E_prop = sc.propagate_E();
        WordCond64 A_prop = sc.propagate_A(E_prop);

        WordCond64 e_before = cs.e(step);
        WordCond64 a_before = cs.a(step);
        if (!cs.e(step).impose_word(E_prop)) return fail(step);
        if (!cs.a(step).impose_word(A_prop)) return fail(step);
        if (cs.e(step) != e_before || cs.a(step) != a_before)
            changed = true;
    }

    // Message expansion: for steps 16..79
    for (int i = 16; i < 80; ++i) {
        SHA512MsgExpCond mc;
        mc.W_im2 = cs.w(i - 2);
        mc.W_im7 = cs.w(i - 7);
        mc.W_im15 = cs.w(i - 15);
        mc.W_im16 = cs.w(i - 16);
        WordCond64 W_prop = mc.propagate();

        WordCond64 w_before = cs.w(i);
        if (!cs.w(i).impose_word(W_prop)) return fail(1000 + i);
        if (cs.w(i) != w_before) changed = true;
    }

    // Backward step propagation
    for (int step = cs.n_steps - 1; step >= 0; --step) {
        SHA512AltStepCond sc;
        sc.A_im4 = cs.a(step - 4);
        sc.A_im3 = cs.a(step - 3);
        sc.A_im2 = cs.a(step - 2);
        sc.A_im1 = cs.a(step - 1);
        sc.E_im4 = cs.e(step - 4);
        sc.E_im3 = cs.e(step - 3);
        sc.E_im2 = cs.e(step - 2);
        sc.E_im1 = cs.e(step - 1);
        sc.W_i = cs.w(step);
        sc.step = step;

        WordCond64 E_i = cs.e(step);
        WordCond64 A_i = cs.a(step);

        if (!sc.backward_from_A(A_i, E_i)) return fail(step);
        if (!sc.backward_from_E(E_i)) return fail(step);

        if (!impose_track(cs.a(step - 4), sc.A_im4, changed)) return fail(step);
        if (!impose_track(cs.a(step - 3), sc.A_im3, changed)) return fail(step);
        if (!impose_track(cs.a(step - 2), sc.A_im2, changed)) return fail(step);
        if (!impose_track(cs.a(step - 1), sc.A_im1, changed)) return fail(step);
        if (!impose_track(cs.e(step - 4), sc.E_im4, changed)) return fail(step);
        if (!impose_track(cs.e(step - 3), sc.E_im3, changed)) return fail(step);
        if (!impose_track(cs.e(step - 2), sc.E_im2, changed)) return fail(step);
        if (!impose_track(cs.e(step - 1), sc.E_im1, changed)) return fail(step);
        if (!impose_track(cs.w(step), sc.W_i, changed)) return fail(step);
        if (!impose_track(cs.e(step), E_i, changed)) return fail(step);
        if (!impose_track(cs.a(step), A_i, changed)) return fail(step);
    }

    // Backward message expansion: 79 down to 16
    for (int i = 79; i >= 16; --i) {
        SHA512MsgExpCond mc;
        mc.W_im2 = cs.w(i - 2);
        mc.W_im7 = cs.w(i - 7);
        mc.W_im15 = cs.w(i - 15);
        mc.W_im16 = cs.w(i - 16);

        WordCond64 W_i = cs.w(i);
        if (!mc.backward(W_i)) return fail(1000 + i);

        if (!impose_track(cs.w(i - 2), mc.W_im2, changed)) return fail(1000 + i);
        if (!impose_track(cs.w(i - 7), mc.W_im7, changed)) return fail(1000 + i);
        if (!impose_track(cs.w(i - 15), mc.W_im15, changed)) return fail(1000 + i);
        if (!impose_track(cs.w(i - 16), mc.W_im16, changed)) return fail(1000 + i);
        if (!impose_track(cs.w(i), W_i, changed)) return fail(1000 + i);
    }

    return true;
}

// Forward declaration (defined after check_twobit).
inline bool propagate_twobit_512(CharState512& cs, bool& changed,
                                  bool value_domain = false);

// N-input backward pass for the E equation (6-input carry tracking).
// Stronger than binary-chain backward but slower (~10-50ms per call).
// Only the E equation benefits from n=6; A equation (n=4) and message
// expansion (n=4) are handled similarly.
// Targeted n-input check: E equation at a single step only.
// Cost: ~0.5 μs when most bits at that step are determined.
// Returns false if carry-level contradiction found.
inline bool ninput_check_E_step(CharState512& cs, int step, bool& changed) {
    changed = false;
    WordCond64 sig1 = wc_Sigma1_512(cs.e(step - 1));
    WordCond64 ch = wc_Ch(cs.e(step - 1), cs.e(step - 2), cs.e(step - 3));
    WordCond64 k_cond = wc_from_constant(sha512::K[step]);
    WordCond64 E_im4 = cs.e(step - 4);
    WordCond64 A_im4 = cs.a(step - 4);
    WordCond64 W_i = cs.w(step);
    WordCond64 E_i = cs.e(step);

    WordCond64 terms[6] = {E_im4, sig1, ch, A_im4, k_cond, W_i};
    if (!wc_add_n_propagate(terms, 6, E_i)) return false;

    if (!impose_track(cs.e(step - 4), terms[0], changed)) return false;
    if (!impose_track(cs.a(step - 4), terms[3], changed)) return false;
    if (!impose_track(cs.w(step), terms[5], changed)) return false;
    if (!impose_track(cs.e(step), E_i, changed)) return false;

    WordCond64 E_im1 = cs.e(step - 1);
    WordCond64 E_im2 = cs.e(step - 2);
    WordCond64 E_im3 = cs.e(step - 3);
    if (!wc_Sigma1_512_propagate(E_im1, terms[1])) return false;
    if (!wc_Ch_propagate(E_im1, E_im2, E_im3, terms[2])) return false;
    if (!impose_track(cs.e(step - 1), E_im1, changed)) return false;
    if (!impose_track(cs.e(step - 2), E_im2, changed)) return false;
    if (!impose_track(cs.e(step - 3), E_im3, changed)) return false;
    return true;
}

inline bool propagate_ninput_512(CharState512& cs, bool& changed) {
    changed = false;

    // Backward step — E equation with n=6
    for (int step = cs.n_steps - 1; step >= 0; --step) {
        WordCond64 sig1 = wc_Sigma1_512(cs.e(step - 1));
        WordCond64 ch = wc_Ch(cs.e(step - 1), cs.e(step - 2), cs.e(step - 3));
        WordCond64 k_cond = wc_from_constant(sha512::K[step]);
        WordCond64 E_im4 = cs.e(step - 4);
        WordCond64 A_im4 = cs.a(step - 4);
        WordCond64 W_i = cs.w(step);
        WordCond64 E_i = cs.e(step);

        WordCond64 terms[6] = {E_im4, sig1, ch, A_im4, k_cond, W_i};
        if (!wc_add_n_propagate(terms, 6, E_i)) return false;

        // Write back tightened conditions
        if (!impose_track(cs.e(step - 4), terms[0], changed)) return false;
        if (!impose_track(cs.a(step - 4), terms[3], changed)) return false;
        if (!impose_track(cs.w(step), terms[5], changed)) return false;
        if (!impose_track(cs.e(step), E_i, changed)) return false;

        // Propagate tightened sig1/ch back through Sigma/Ch
        WordCond64 E_im1 = cs.e(step - 1);
        WordCond64 E_im2 = cs.e(step - 2);
        WordCond64 E_im3 = cs.e(step - 3);
        if (!wc_Sigma1_512_propagate(E_im1, terms[1])) return false;
        if (!wc_Ch_propagate(E_im1, E_im2, E_im3, terms[2])) return false;
        if (!impose_track(cs.e(step - 1), E_im1, changed)) return false;
        if (!impose_track(cs.e(step - 2), E_im2, changed)) return false;
        if (!impose_track(cs.e(step - 3), E_im3, changed)) return false;
    }

    // Backward step — A equation with n=4
    for (int step = cs.n_steps - 1; step >= 0; --step) {
        WordCond64 sig0 = wc_Sigma0_512(cs.a(step - 1));
        WordCond64 maj = wc_Maj(cs.a(step - 1), cs.a(step - 2), cs.a(step - 3));
        WordCond64 neg_A = wc_neg(cs.a(step - 4));
        WordCond64 E_i = cs.e(step);
        WordCond64 A_i = cs.a(step);

        WordCond64 terms[4] = {neg_A, sig0, maj, E_i};
        if (!wc_add_n_propagate(terms, 4, A_i)) return false;

        WordCond64 A_im4_new = cs.a(step - 4);
        if (!wc_neg_propagate(A_im4_new, terms[0])) return false;
        if (!impose_track(cs.a(step - 4), A_im4_new, changed)) return false;
        if (!impose_track(cs.e(step), terms[3], changed)) return false;
        if (!impose_track(cs.a(step), A_i, changed)) return false;

        WordCond64 A_im1 = cs.a(step - 1);
        WordCond64 A_im2 = cs.a(step - 2);
        WordCond64 A_im3 = cs.a(step - 3);
        if (!wc_Sigma0_512_propagate(A_im1, terms[1])) return false;
        if (!wc_Maj_propagate(A_im1, A_im2, A_im3, terms[2])) return false;
        if (!impose_track(cs.a(step - 1), A_im1, changed)) return false;
        if (!impose_track(cs.a(step - 2), A_im2, changed)) return false;
        if (!impose_track(cs.a(step - 3), A_im3, changed)) return false;
    }

    // Backward message expansion with n=4
    for (int i = 79; i >= 16; --i) {
        WordCond64 s1 = wc_sigma1_512(cs.w(i - 2));
        WordCond64 s0 = wc_sigma0_512(cs.w(i - 15));
        WordCond64 W_im7 = cs.w(i - 7);
        WordCond64 W_im16 = cs.w(i - 16);
        WordCond64 W_i = cs.w(i);

        WordCond64 terms[4] = {s1, W_im7, s0, W_im16};
        if (!wc_add_n_propagate(terms, 4, W_i)) return false;

        WordCond64 W_im2 = cs.w(i - 2);
        WordCond64 W_im15 = cs.w(i - 15);
        if (!wc_sigma1_512_propagate(W_im2, terms[0])) return false;
        if (!wc_sigma0_512_propagate(W_im15, terms[2])) return false;
        if (!impose_track(cs.w(i - 2), W_im2, changed)) return false;
        if (!impose_track(cs.w(i - 7), terms[1], changed)) return false;
        if (!impose_track(cs.w(i - 15), W_im15, changed)) return false;
        if (!impose_track(cs.w(i - 16), terms[3], changed)) return false;
        if (!impose_track(cs.w(i), W_i, changed)) return false;
    }

    return true;
}

inline bool propagate_full(CharState512& cs, int* fail_step = nullptr,
                          bool use_ninput = false, bool use_value_twobit = false) {
    for (int iter = 0; iter < 50; ++iter) {
        bool changed = false;
        if (!propagate_once(cs, changed, fail_step)) return false;
        if (!changed) {
            // Bitwise converged — try twobit deductions
            bool tb_changed = false;
            if (!propagate_twobit_512(cs, tb_changed, use_value_twobit)) {
                if (fail_step) *fail_step = -1;  // twobit: no step info
                return false;
            }
            if (!tb_changed) {
                if (!use_ninput) return true;  // two-layer fixpoint
                // Twobit also converged — try n-input deductions
                bool ni_changed = false;
                if (!propagate_ninput_512(cs, ni_changed)) {
                    if (fail_step) *fail_step = -2;  // n-input failure
                    return false;
                }
                if (!ni_changed) return true;  // three-layer fixpoint
                // N-input made deductions — re-enter from binary chain
                continue;
            }
            // Twobit made deductions — re-enter bitwise loop
            continue;
        }
    }
    return true;
}

// ---- Two-bit consistency check ----

// Helpers for addition two-bit conditions (diff-domain XOR equations).
inline int bc_diff_status_512(uint8_t c) {
    if ((c & 0x6) == 0 && c != 0) return 0;  // {0, 1, -}: no diff
    if ((c & 0x9) == 0 && c != 0) return 1;  // {u, n, x}: has diff
    return -1;  // indeterminate
}

inline XorTerm make_xor_term_512(int word_id, int bit, uint8_t cond) {
    int ds = bc_diff_status_512(cond);
    return XorTerm{{word_id, bit}, ds >= 0, (ds == 1) ? 1 : 0};
}

inline XorTerm make_xor_known_512(int val) {
    return XorTerm{{0, 0}, true, val};
}

// Addition two-bit conditions from modular addition equations at each bit j.
// At bit j, if carry diff is 0, the multi-input addition satisfies:
//   Δsum[j] = Δa₁[j] ⊕ Δa₂[j] ⊕ ... ⊕ Δaₖ[j]
// Carry diff = 0 at bit 0 always, and at bit j>0 if ALL addend diffs
// at bits 0..j-1 are 0. Sigma/sigma are linear (XOR): their output is
// inlined as individual bit terms. Ch/Maj outputs are nonlinear: treated
// as known constants when their diff is determined, else opaque.
inline bool extract_addition_twobit_512(const CharState512& cs, TwoBitTracker& tracker) {
    int n = cs.n_steps;
    auto a_word_id = [](int step) { return step + 4; };
    auto e_word_id = [&](int step) { return (n + 4) + step + 4; };
    auto w_word_id = [&](int i) { return 2 * (n + 4) + i; };

    static const BoolPropTable tbl_ch = BoolPropTable::build(TT_IF);
    static const BoolPropTable tbl_maj = BoolPropTable::build(TT_MAJ);

    // E equation per step (expanded, bit j, carry diff = 0):
    //   E_i = A_{i-4} + E_{i-4} + Σ₁(E_{i-1}) + Ch(E_{i-1},E_{i-2},E_{i-3}) + K_i + W_i
    //   ΔE_i[j] ⊕ ΔE_{i-4}[j] ⊕ ΔΣ₁(E_{i-1})[j] ⊕ ΔCh[j] ⊕ ΔA_{i-4}[j] ⊕ ΔW_i[j] = 0
    //   Σ₁ inlined: ΔE_{i-1}[(j+14)%64] ⊕ ΔE_{i-1}[(j+18)%64] ⊕ ΔE_{i-1}[(j+41)%64]
    //   K_i has no diff (constant).
    for (int step = 0; step < n; ++step) {
        for (int j = 0; j < BITS; ++j) {
            XorTerm terms[8];
            int n_opaque = 0;
            int p1 = (j + 14) & 63, p2 = (j + 18) & 63, p3 = (j + 41) & 63;

            terms[0] = make_xor_term_512(e_word_id(step), j, cs.e(step).get(j));
            terms[1] = make_xor_term_512(e_word_id(step - 4), j, cs.e(step - 4).get(j));
            terms[2] = make_xor_term_512(e_word_id(step - 1), p1, cs.e(step - 1).get(p1));
            terms[3] = make_xor_term_512(e_word_id(step - 1), p2, cs.e(step - 1).get(p2));
            terms[4] = make_xor_term_512(e_word_id(step - 1), p3, cs.e(step - 1).get(p3));
            terms[5] = make_xor_term_512(a_word_id(step - 4), j, cs.a(step - 4).get(j));
            terms[6] = make_xor_term_512(w_word_id(step), j, cs.w(step).get(j));

            // Ch output: nonlinear intermediate
            uint8_t ch_out = tbl_ch.propagate(
                BitCond(cs.e(step - 1).get(j)),
                BitCond(cs.e(step - 2).get(j)),
                BitCond(cs.e(step - 3).get(j)));
            int ch_ds = bc_diff_status_512(ch_out);
            if (ch_ds >= 0) {
                terms[7] = make_xor_known_512(ch_ds);
            } else {
                terms[7] = make_xor_known_512(0);
                n_opaque++;
            }

            if (n_opaque == 0) {
                bool contra = false;
                auto conds = extract_xor_twobit(terms, 8, 0, &contra);
                if (contra) return false;
                for (auto& c : conds)
                    if (!tracker.add(c)) return false;
            }

            // Carry chain: all addend diffs at bit j must be 0 to continue.
            // Addends: E_{i-4}, Σ₁(E_{i-1}), Ch, A_{i-4}, W_i
            bool chain_ok = true;
            if (!terms[1].known || terms[1].val != 0) chain_ok = false;
            if (chain_ok) {
                if (!terms[2].known || !terms[3].known || !terms[4].known)
                    chain_ok = false;
                else if ((terms[2].val ^ terms[3].val ^ terms[4].val) != 0)
                    chain_ok = false;
            }
            if (chain_ok && ch_ds != 0) chain_ok = false;
            if (chain_ok && (!terms[5].known || terms[5].val != 0)) chain_ok = false;
            if (chain_ok && (!terms[6].known || terms[6].val != 0)) chain_ok = false;
            if (chain_ok)
                continue;
            break;
        }
    }

    // A equation per step:
    //   A_i = E_i - A_{i-4} + Σ₀(A_{i-1}) + Maj(A_{i-1},A_{i-2},A_{i-3})
    //   ΔA_i[j] ⊕ ΔA_{i-4}[j] ⊕ ΔΣ₀(A_{i-1})[j] ⊕ ΔMaj[j] ⊕ ΔE_i[j] = 0
    //   Σ₀ inlined: ΔA_{i-1}[(j+28)%64] ⊕ ΔA_{i-1}[(j+34)%64] ⊕ ΔA_{i-1}[(j+39)%64]
    for (int step = 0; step < n; ++step) {
        for (int j = 0; j < BITS; ++j) {
            XorTerm terms[7];
            int n_opaque = 0;
            int p1 = (j + 28) & 63, p2 = (j + 34) & 63, p3 = (j + 39) & 63;

            terms[0] = make_xor_term_512(a_word_id(step), j, cs.a(step).get(j));
            terms[1] = make_xor_term_512(a_word_id(step - 4), j, cs.a(step - 4).get(j));
            terms[2] = make_xor_term_512(a_word_id(step - 1), p1, cs.a(step - 1).get(p1));
            terms[3] = make_xor_term_512(a_word_id(step - 1), p2, cs.a(step - 1).get(p2));
            terms[4] = make_xor_term_512(a_word_id(step - 1), p3, cs.a(step - 1).get(p3));
            terms[5] = make_xor_term_512(e_word_id(step), j, cs.e(step).get(j));

            uint8_t maj_out = tbl_maj.propagate(
                BitCond(cs.a(step - 1).get(j)),
                BitCond(cs.a(step - 2).get(j)),
                BitCond(cs.a(step - 3).get(j)));
            int maj_ds = bc_diff_status_512(maj_out);
            if (maj_ds >= 0) {
                terms[6] = make_xor_known_512(maj_ds);
            } else {
                terms[6] = make_xor_known_512(0);
                n_opaque++;
            }

            if (n_opaque == 0) {
                bool contra = false;
                auto conds = extract_xor_twobit(terms, 7, 0, &contra);
                if (contra) return false;
                for (auto& c : conds)
                    if (!tracker.add(c)) return false;
            }

            // Carry chain: addends are -A_{i-4}, Σ₀(A_{i-1}), Maj, E_i.
            bool chain_ok = true;
            if (!terms[1].known || terms[1].val != 0) chain_ok = false;
            if (chain_ok) {
                if (!terms[2].known || !terms[3].known || !terms[4].known)
                    chain_ok = false;
                else if ((terms[2].val ^ terms[3].val ^ terms[4].val) != 0)
                    chain_ok = false;
            }
            if (chain_ok && maj_ds != 0) chain_ok = false;
            if (chain_ok && (!terms[5].known || terms[5].val != 0)) chain_ok = false;
            if (chain_ok)
                continue;
            break;
        }
    }

    // W expansion: W_i = σ₁(W_{i-2}) + W_{i-7} + σ₀(W_{i-15}) + W_{i-16}
    // σ₁(x) = rotr(x,19) ^ rotr(x,61) ^ (x >> 6)
    // σ₀(x) = rotr(x,1) ^ rotr(x,8) ^ (x >> 7)
    for (int i = 16; i < 80; ++i) {
        for (int j = 0; j < BITS; ++j) {
            XorTerm terms[9];
            int nt = 0;

            terms[nt++] = make_xor_term_512(w_word_id(i), j, cs.w(i).get(j));

            // σ₁(W_{i-2}): rotr19 ^ rotr61 ^ shr6
            int q1 = (j + 19) & 63, q2 = (j + 61) & 63, q3 = j + 6;
            terms[nt++] = make_xor_term_512(w_word_id(i - 2), q1, cs.w(i - 2).get(q1));
            terms[nt++] = make_xor_term_512(w_word_id(i - 2), q2, cs.w(i - 2).get(q2));
            if (q3 < BITS)
                terms[nt++] = make_xor_term_512(w_word_id(i - 2), q3, cs.w(i - 2).get(q3));
            else
                terms[nt++] = make_xor_known_512(0);

            terms[nt++] = make_xor_term_512(w_word_id(i - 7), j, cs.w(i - 7).get(j));

            // σ₀(W_{i-15}): rotr1 ^ rotr8 ^ shr7
            int r1 = (j + 1) & 63, r2 = (j + 8) & 63, r3 = j + 7;
            terms[nt++] = make_xor_term_512(w_word_id(i - 15), r1, cs.w(i - 15).get(r1));
            terms[nt++] = make_xor_term_512(w_word_id(i - 15), r2, cs.w(i - 15).get(r2));
            if (r3 < BITS)
                terms[nt++] = make_xor_term_512(w_word_id(i - 15), r3, cs.w(i - 15).get(r3));
            else
                terms[nt++] = make_xor_known_512(0);

            terms[nt++] = make_xor_term_512(w_word_id(i - 16), j, cs.w(i - 16).get(j));

            bool contra = false;
            auto conds = extract_xor_twobit(terms, nt, 0, &contra);
            if (contra) return false;
            for (auto& c : conds)
                if (!tracker.add(c)) return false;

            // Carry chain: all addend diffs at j must be 0.
            bool chain_ok = true;
            if (!terms[1].known || !terms[2].known || !terms[3].known)
                chain_ok = false;
            else if ((terms[1].val ^ terms[2].val ^ terms[3].val) != 0)
                chain_ok = false;
            if (chain_ok && (!terms[4].known || terms[4].val != 0)) chain_ok = false;
            if (chain_ok) {
                if (!terms[5].known || !terms[6].known || !terms[7].known)
                    chain_ok = false;
                else if ((terms[5].val ^ terms[6].val ^ terms[7].val) != 0)
                    chain_ok = false;
            }
            if (chain_ok && (!terms[8].known || terms[8].val != 0)) chain_ok = false;
            if (chain_ok)
                continue;
            break;
        }
    }

    return true;
}

inline bool check_twobit(const CharState512& cs) {
    int n = cs.n_steps;
    int total_words = (n + 4) + (n + 4) + 80;
    TwoBitTracker tracker(total_words, BITS);

    auto a_word_id = [](int step) { return step + 4; };
    auto e_word_id = [&](int step) { return (n + 4) + step + 4; };
    auto w_word_id = [&](int i) { return 2 * (n + 4) + i; };

    // NOTE: Maj and Ch two-bit conditions are OMITTED here.
    // extract_maj_twobit/extract_ch_twobit assume diff-domain linearity
    // ("a has diff + output no diff → b_diff == c_diff"), but Maj and Ch
    // are nonlinear. Counterexample: a=u(1→0), b='-'(1→1), c=n(0→1) →
    // Maj(1,1,0)=1, Maj(0,1,1)=1 → output no diff, but b_diff ≠ c_diff.
    // Including them causes false rejections of valid solutions in Phase 3.
    // Only Sigma conditions (linear XOR) are extracted below.

    // Σ₀(A_{i-1}): rotr28 ^ rotr34 ^ rotr39
    for (int step = 0; step < n; ++step) {
        uint8_t in_conds[BITS], out_conds[BITS];
        WordCond64 sig0_out = wc_Sigma0_512(cs.a(step - 1));
        for (int b = 0; b < BITS; ++b) {
            in_conds[b] = cs.a(step - 1).get(b);
            out_conds[b] = sig0_out.get(b);
        }
        auto conds = extract_sigma_twobit(a_word_id(step - 1), 28, 34, 39, false,
                                           in_conds, out_conds, BITS);
        for (auto& c : conds) {
            if (!tracker.add(c)) return false;
        }
    }

    // Σ₁(E_{i-1}): rotr14 ^ rotr18 ^ rotr41
    for (int step = 0; step < n; ++step) {
        uint8_t in_conds[BITS], out_conds[BITS];
        WordCond64 sig1_out = wc_Sigma1_512(cs.e(step - 1));
        for (int b = 0; b < BITS; ++b) {
            in_conds[b] = cs.e(step - 1).get(b);
            out_conds[b] = sig1_out.get(b);
        }
        auto conds = extract_sigma_twobit(e_word_id(step - 1), 14, 18, 41, false,
                                           in_conds, out_conds, BITS);
        for (auto& c : conds) {
            if (!tracker.add(c)) return false;
        }
    }

    // σ₀(W_{i-15}): rotr1 ^ rotr8 ^ shr7
    for (int i = 16; i < 80; ++i) {
        uint8_t in_conds[BITS], out_conds[BITS];
        WordCond64 s0_out = wc_sigma0_512(cs.w(i - 15));
        for (int b = 0; b < BITS; ++b) {
            in_conds[b] = cs.w(i - 15).get(b);
            out_conds[b] = s0_out.get(b);
        }
        auto conds = extract_sigma_twobit(w_word_id(i - 15), 1, 8, 7, true,
                                           in_conds, out_conds, BITS);
        for (auto& c : conds) {
            if (!tracker.add(c)) return false;
        }
    }

    // σ₁(W_{i-2}): rotr19 ^ rotr61 ^ shr6
    for (int i = 16; i < 80; ++i) {
        uint8_t in_conds[BITS], out_conds[BITS];
        WordCond64 s1_out = wc_sigma1_512(cs.w(i - 2));
        for (int b = 0; b < BITS; ++b) {
            in_conds[b] = cs.w(i - 2).get(b);
            out_conds[b] = s1_out.get(b);
        }
        auto conds = extract_sigma_twobit(w_word_id(i - 2), 19, 61, 6, true,
                                           in_conds, out_conds, BITS);
        for (auto& c : conds) {
            if (!tracker.add(c)) return false;
        }
    }

    // Addition two-bit conditions (cross-word diff-domain XOR equations).
    if (!extract_addition_twobit_512(cs, tracker)) return false;

    return true;
}

// ---- Backward-constrained Sigma output reconstruction ----
// After bitwise convergence, the intermediate Sigma output is lost.
// Reconstruct it from the addition chain backward to get value-domain
// constraints on the Sigma input bits.

inline bool backward_sigma0_out_512(const CharState512& cs, int step,
                                     WordCond64& sig0_out) {
    sig0_out = wc_Sigma0_512(cs.a(step - 1));

    static const BoolPropTable tbl_maj = BoolPropTable::build(TT_MAJ);
    WordCond64 maj;
    for (int b = 0; b < BITS; ++b)
        maj.set(b, BitCond(tbl_maj.propagate(
            BitCond(cs.a(step - 1).get(b)),
            BitCond(cs.a(step - 2).get(b)),
            BitCond(cs.a(step - 3).get(b)))));

    WordCond64 neg_a4 = wc_neg(cs.a(step - 4));
    WordCond64 t1 = wc_add(neg_a4, sig0_out);
    WordCond64 t2 = wc_add(t1, maj);

    WordCond64 A_i = cs.A[step + 4];
    WordCond64 E_i = cs.E[step + 4];

    if (!wc_add_propagate(t2, E_i, A_i)) return false;
    if (!wc_add_propagate(t1, maj, t2)) return false;
    if (!wc_add_propagate(neg_a4, sig0_out, t1)) return false;

    return true;
}

inline bool backward_sigma1_out_512(const CharState512& cs, int step,
                                     WordCond64& sig1_out) {
    sig1_out = wc_Sigma1_512(cs.e(step - 1));

    static const BoolPropTable tbl_ch = BoolPropTable::build(TT_IF);
    WordCond64 ch;
    for (int b = 0; b < BITS; ++b)
        ch.set(b, BitCond(tbl_ch.propagate(
            BitCond(cs.e(step - 1).get(b)),
            BitCond(cs.e(step - 2).get(b)),
            BitCond(cs.e(step - 3).get(b)))));

    WordCond64 k_cond = wc_from_constant(sha512::K[step]);

    WordCond64 E_im4 = cs.E[step - 4 + 4];
    WordCond64 t1 = wc_add(E_im4, sig1_out);
    WordCond64 t2 = wc_add(t1, ch);
    WordCond64 A_im4 = cs.A[step - 4 + 4];
    WordCond64 t3 = wc_add(t2, A_im4);
    WordCond64 t4 = wc_add(t3, k_cond);

    WordCond64 E_i = cs.E[step + 4];
    WordCond64 W_i = cs.W[step];

    if (!wc_add_propagate(t4, W_i, E_i)) return false;
    if (!wc_add_propagate(t3, k_cond, t4)) return false;
    if (!wc_add_propagate(t2, A_im4, t3)) return false;
    if (!wc_add_propagate(t1, ch, t2)) return false;
    if (!wc_add_propagate(E_im4, sig1_out, t1)) return false;

    return true;
}

// ---- Two-bit deduction (propagation, not just check) ----
// Builds the same Union-Find as check_twobit, then forces undecided
// members of each equivalence class to match any decided anchor.
// This is the key mechanism that makes look-ahead scoring work:
// one 0/1 decision cascades through Sigma/addition-linked bits.

inline bool propagate_twobit_512(CharState512& cs, bool& changed,
                                  bool value_domain) {
    int n = cs.n_steps;
    int total_words = (n + 4) + (n + 4) + 80;
    TwoBitTracker tracker(total_words, BITS);

    auto a_word_id = [](int step) { return step + 4; };
    auto e_word_id = [&](int step) { return (n + 4) + step + 4; };
    auto w_word_id = [&](int i) { return 2 * (n + 4) + i; };
    int w_base = 2 * (n + 4);

    // --- Build the Union-Find (same extraction as check_twobit) ---
    // Maj/Ch omitted (nonlinear → false constraints). See check_twobit.

    // Σ₀(A_{i-1}): rotr28 ^ rotr34 ^ rotr39
    for (int step = 0; step < n; ++step) {
        uint8_t in_conds[BITS], out_conds[BITS];
        WordCond64 sig0_out = wc_Sigma0_512(cs.a(step - 1));
        for (int b = 0; b < BITS; ++b) {
            in_conds[b] = cs.a(step - 1).get(b);
            out_conds[b] = sig0_out.get(b);
        }
        for (auto& c : extract_sigma_twobit(a_word_id(step - 1), 28, 34, 39, false,
                                              in_conds, out_conds, BITS))
            if (!tracker.add(c)) return false;
    }

    // Σ₁(E_{i-1}): rotr14 ^ rotr18 ^ rotr41
    for (int step = 0; step < n; ++step) {
        uint8_t in_conds[BITS], out_conds[BITS];
        WordCond64 sig1_out = wc_Sigma1_512(cs.e(step - 1));
        for (int b = 0; b < BITS; ++b) {
            in_conds[b] = cs.e(step - 1).get(b);
            out_conds[b] = sig1_out.get(b);
        }
        for (auto& c : extract_sigma_twobit(e_word_id(step - 1), 14, 18, 41, false,
                                              in_conds, out_conds, BITS))
            if (!tracker.add(c)) return false;
    }

    // σ₀(W_{i-15}): rotr1 ^ rotr8 ^ shr7
    for (int i = 16; i < 80; ++i) {
        uint8_t in_conds[BITS], out_conds[BITS];
        WordCond64 s0_out = wc_sigma0_512(cs.w(i - 15));
        for (int b = 0; b < BITS; ++b) {
            in_conds[b] = cs.w(i - 15).get(b);
            out_conds[b] = s0_out.get(b);
        }
        for (auto& c : extract_sigma_twobit(w_word_id(i - 15), 1, 8, 7, true,
                                              in_conds, out_conds, BITS))
            if (!tracker.add(c)) return false;
    }

    // σ₁(W_{i-2}): rotr19 ^ rotr61 ^ shr6
    for (int i = 16; i < 80; ++i) {
        uint8_t in_conds[BITS], out_conds[BITS];
        WordCond64 s1_out = wc_sigma1_512(cs.w(i - 2));
        for (int b = 0; b < BITS; ++b) {
            in_conds[b] = cs.w(i - 2).get(b);
            out_conds[b] = s1_out.get(b);
        }
        for (auto& c : extract_sigma_twobit(w_word_id(i - 2), 19, 61, 6, true,
                                              in_conds, out_conds, BITS))
            if (!tracker.add(c)) return false;
    }

    // Addition two-bit conditions (cross-word diff-domain XOR equations).
    if (!extract_addition_twobit_512(cs, tracker)) return false;

    // --- Extract deductions from equivalence classes ---
    int total_nodes = total_words * BITS;

    auto node_cond = [&](int node_id) -> BitCond {
        int word = node_id / BITS, bit = node_id % BITS;
        if (word < n + 4)          return cs.A[word].get(bit);
        if (word < 2 * (n + 4))    return cs.E[word - (n + 4)].get(bit);
        return cs.W[word - w_base].get(bit);
    };

    auto impose_node = [&](int node_id, BitCond cond) -> bool {
        int word = node_id / BITS, bit = node_id % BITS;
        WordCond64* wc;
        if (word < n + 4)          wc = &cs.A[word];
        else if (word < 2*(n+4))   wc = &cs.E[word - (n + 4)];
        else                       wc = &cs.W[word - w_base];
        BitCond cur = wc->get(bit);
        BitCond meet = bc_meet(cur, cond);
        if (bc_is_contradiction(meet)) return false;
        if (meet != cur) { wc->set(bit, meet); changed = true; }
        return true;
    };

    auto flip_parity = [](BitCond c) -> BitCond {
        if (c == BC_DASH) return BC_X;
        if (c == BC_X)    return BC_DASH;
        return c;
    };

    auto to_diff_category = [](BitCond c) -> BitCond {
        uint8_t v = static_cast<uint8_t>(c);
        if ((v & 0x6) == 0 && v != 0) return BC_DASH;  // {0, 1, -}: no diff
        if ((v & 0x9) == 0 && v != 0) return BC_X;     // {u, n, x}: has diff
        return c;  // multi-valued: no useful deduction
    };

    // First pass: find anchor condition for each equivalence class root.
    struct AnchorInfo { BitCond cond; uint8_t parity; };
    std::vector<int8_t> root_has_anchor(total_nodes, 0);
    std::vector<AnchorInfo> root_anchor(total_nodes, {BC_FREE, 0});

    for (int node_id = 0; node_id < total_nodes; ++node_id) {
        BitCond c = node_cond(node_id);
        BitCond cat = to_diff_category(c);
        if (cat != BC_DASH && cat != BC_X) continue;

        auto [root, par] = tracker.find(node_id);
        if (root_has_anchor[root]) continue;

        root_has_anchor[root] = 1;
        root_anchor[root] = {cat, par};
    }

    // Second pass: impose implied condition on all nodes in anchored classes.
    for (int node_id = 0; node_id < total_nodes; ++node_id) {
        auto [root, par] = tracker.find(node_id);
        if (!root_has_anchor[root]) continue;

        auto& ai = root_anchor[root];
        uint8_t rel_par = par ^ ai.parity;
        BitCond implied = (rel_par == 0) ? ai.cond : flip_parity(ai.cond);
        if (!impose_node(node_id, implied)) return false;
    }

    // ---- Value-domain twobit (Task 21) ----
    // Independent Union-Find tracking first-copy VALUE equivalences
    // through Sigma XOR equations.  Backward-constrained Sigma output
    // provides the target parity that forward-only computation cannot.
    if (value_domain) {
        TwoBitTracker val_tracker(total_words, BITS);

        // Σ₀(A_{i-1}): backward-constrained output from A equation
        for (int step = 0; step < n; ++step) {
            WordCond64 sig0_bwd;
            if (backward_sigma0_out_512(cs, step, sig0_bwd)) {
                uint8_t in_conds[BITS], out_conds[BITS];
                for (int b = 0; b < BITS; ++b) {
                    in_conds[b] = cs.a(step - 1).get(b);
                    out_conds[b] = sig0_bwd.get(b);
                }
                for (auto& c : extract_sigma_twobit_value(
                        a_word_id(step - 1), 28, 34, 39, false,
                        in_conds, out_conds, BITS))
                    if (!val_tracker.add(c)) return false;
            }
        }

        // Σ₁(E_{i-1}): backward-constrained output from E equation
        for (int step = 0; step < n; ++step) {
            WordCond64 sig1_bwd;
            if (backward_sigma1_out_512(cs, step, sig1_bwd)) {
                uint8_t in_conds[BITS], out_conds[BITS];
                for (int b = 0; b < BITS; ++b) {
                    in_conds[b] = cs.e(step - 1).get(b);
                    out_conds[b] = sig1_bwd.get(b);
                }
                for (auto& c : extract_sigma_twobit_value(
                        e_word_id(step - 1), 14, 18, 41, false,
                        in_conds, out_conds, BITS))
                    if (!val_tracker.add(c)) return false;
            }
        }

        // σ₀(W_{i-15}) and σ₁(W_{i-2}): forward-only (no backward chain)
        for (int i = 16; i < 80; ++i) {
            {
                uint8_t in_conds[BITS], out_conds[BITS];
                WordCond64 s0_out = wc_sigma0_512(cs.w(i - 15));
                for (int b = 0; b < BITS; ++b) {
                    in_conds[b] = cs.w(i - 15).get(b);
                    out_conds[b] = s0_out.get(b);
                }
                for (auto& c : extract_sigma_twobit_value(
                        w_word_id(i - 15), 1, 8, 7, true,
                        in_conds, out_conds, BITS))
                    if (!val_tracker.add(c)) return false;
            }
            {
                uint8_t in_conds[BITS], out_conds[BITS];
                WordCond64 s1_out = wc_sigma1_512(cs.w(i - 2));
                for (int b = 0; b < BITS; ++b) {
                    in_conds[b] = cs.w(i - 2).get(b);
                    out_conds[b] = s1_out.get(b);
                }
                for (auto& c : extract_sigma_twobit_value(
                        w_word_id(i - 2), 19, 61, 6, true,
                        in_conds, out_conds, BITS))
                    if (!val_tracker.add(c)) return false;
            }
        }

        // Value-domain deduction: anchor on known-value nodes, impose on class.
        // val=0 → constrain first copy to 0 (mask 0x5 = BC_5).
        // val=1 → constrain first copy to 1 (mask 0xA = BC_A).
        struct ValAnchor { int value; uint8_t parity; };
        std::vector<int8_t> root_has_val(total_nodes, 0);
        std::vector<ValAnchor> root_val(total_nodes, {0, 0});

        for (int node_id = 0; node_id < total_nodes; ++node_id) {
            BitCond c = node_cond(node_id);
            int vs = twobit::bc_value_status(static_cast<uint8_t>(c));
            if (vs < 0) continue;

            auto [root, par] = val_tracker.find(node_id);
            if (root_has_val[root]) continue;

            root_has_val[root] = 1;
            root_val[root] = {vs, par};
        }

        for (int node_id = 0; node_id < total_nodes; ++node_id) {
            auto [root, par] = val_tracker.find(node_id);
            if (!root_has_val[root]) continue;

            auto& va = root_val[root];
            uint8_t rel_par = par ^ va.parity;
            int implied_val = (rel_par == 0) ? va.value : (1 - va.value);
            BitCond imposed = (implied_val == 0) ? BitCond(0x5) : BitCond(0xA);
            if (!impose_node(node_id, imposed)) return false;
        }
    }

    return true;
}

// ---- Bit selection ----

struct ConnectivityMap {
    int n_steps;
    std::vector<int> a_conn, e_conn, w_conn;

    int get(const BitLoc& loc) const {
        switch (loc.type) {
            case WT_A: return a_conn[(loc.step + 4) * BITS + loc.bit];
            case WT_E: return e_conn[(loc.step + 4) * BITS + loc.bit];
            case WT_W: return w_conn[loc.step * BITS + loc.bit];
        }
        return 0;
    }
};

// Per-bit activity scores for conflict-driven decision ordering (EVSIDS-style).
// Paper §3.3: "favoring bits involved in recent conflicts."
// Indexed identically to ConnectivityMap for consistency.
struct ActivityMap {
    int n_steps;
    std::vector<double> a_act, e_act, w_act;
    double decay = 0.95;        // multiplicative decay per decision
    double bump_value = 1.0;    // current bump increment (grows with rescaling)

    void init(int steps) {
        n_steps = steps;
        a_act.assign((steps + 4) * BITS, 0.0);
        e_act.assign((steps + 4) * BITS, 0.0);
        w_act.assign(80 * BITS, 0.0);
        bump_value = 1.0;
    }

    double get(const BitLoc& loc) const {
        switch (loc.type) {
            case WT_A: return a_act[(loc.step + 4) * BITS + loc.bit];
            case WT_E: return e_act[(loc.step + 4) * BITS + loc.bit];
            case WT_W: return w_act[loc.step * BITS + loc.bit];
        }
        return 0.0;
    }

    void bump(const BitLoc& loc) {
        switch (loc.type) {
            case WT_A: a_act[(loc.step + 4) * BITS + loc.bit] += bump_value; break;
            case WT_E: e_act[(loc.step + 4) * BITS + loc.bit] += bump_value; break;
            case WT_W: w_act[loc.step * BITS + loc.bit] += bump_value; break;
        }
    }

    // Call once per decision. Instead of decaying all scores (O(n)),
    // we scale up bump_value — equivalent but O(1).
    void decay_step() {
        bump_value /= decay;
        // Rescale when bump_value gets too large to avoid overflow
        if (bump_value > 1e100) {
            double scale = 1.0 / bump_value;
            for (auto& v : a_act) v *= scale;
            for (auto& v : e_act) v *= scale;
            for (auto& v : w_act) v *= scale;
            bump_value = 1.0;
        }
    }
};

// Bump activity for all undecided bits involved in the equation that caused
// a contradiction. fail_step encoding: 0..n-1 = step function, 1000+i = W expansion.
// Also bumps the decision bit itself.
inline void bump_conflict_bits(ActivityMap& activity, const CharState512& cs,
                               int fail_step, const BitLoc& decision_loc) {
    auto bump_word_dashes = [&](WordType wt, int step, const WordCond64& wc) {
        uint64_t dm = wc.dash_mask() | wc.free_mask();
        while (dm) {
            int b = gc_ctz64(dm);
            activity.bump({wt, step, b});
            dm &= dm - 1;
        }
    };

    // Always bump the decision bit
    activity.bump(decision_loc);

    if (fail_step >= 0 && fail_step < cs.n_steps) {
        // Step function: inputs are A[s-4..s-1], E[s-4..s-1], W[s], A[s], E[s]
        int s = fail_step;
        for (int d = -4; d <= 0; ++d)
            bump_word_dashes(WT_A, s + d, cs.a(s + d));
        for (int d = -4; d <= 0; ++d)
            bump_word_dashes(WT_E, s + d, cs.e(s + d));
        bump_word_dashes(WT_W, s, cs.w(s));
    } else if (fail_step >= 1000) {
        // Message expansion: W[i] = σ₁(W[i-2]) + W[i-7] + σ₀(W[i-15]) + W[i-16]
        int i = fail_step - 1000;
        bump_word_dashes(WT_W, i, cs.w(i));
        if (i >= 2) bump_word_dashes(WT_W, i - 2, cs.w(i - 2));
        if (i >= 7) bump_word_dashes(WT_W, i - 7, cs.w(i - 7));
        if (i >= 15) bump_word_dashes(WT_W, i - 15, cs.w(i - 15));
        if (i >= 16) bump_word_dashes(WT_W, i - 16, cs.w(i - 16));
    }
    // fail_step == -1 (twobit) or -2 (n-input/unknown): bump decision bit only
}

inline ConnectivityMap compute_connectivity(const CharState512& cs) {
    int n = cs.n_steps;
    ConnectivityMap cm;
    cm.n_steps = n;
    cm.a_conn.assign((n + 4) * BITS, 0);
    cm.e_conn.assign((n + 4) * BITS, 0);
    cm.w_conn.assign(80 * BITS, 0);

    int w_base = 2 * (n + 4);
    auto inc = [&](const BitId& id) {
        if (id.word >= w_base) {
            cm.w_conn[(id.word - w_base) * BITS + id.bit]++;
        } else if (id.word >= n + 4) {
            cm.e_conn[(id.word - (n + 4)) * BITS + id.bit]++;
        } else {
            cm.a_conn[id.word * BITS + id.bit]++;
        }
    };

    auto count_cond = [&](const TwoBitCond& c) {
        inc(c.a);
        inc(c.b);
    };

    auto a_word_id = [](int step) { return step + 4; };
    auto e_word_id = [&](int step) { return (n + 4) + step + 4; };
    auto w_word_id = [&](int i) { return 2 * (n + 4) + i; };

    // NOTE: Maj/Ch conditions omitted (nonlinear → false constraints).
    // See check_twobit comment for details.

    // Σ₀(A_{i-1}): 28, 34, 39 (not shift)
    for (int step = 0; step < n; ++step) {
        uint8_t in_conds[BITS], out_conds[BITS];
        WordCond64 sig0_out = wc_Sigma0_512(cs.a(step-1));
        for (int b = 0; b < BITS; ++b) { in_conds[b] = cs.a(step-1).get(b); out_conds[b] = sig0_out.get(b); }
        for (auto& c : extract_sigma_twobit(a_word_id(step-1), 28, 34, 39, false, in_conds, out_conds, BITS))
            count_cond(c);
    }

    // Σ₁(E_{i-1}): 14, 18, 41 (not shift)
    for (int step = 0; step < n; ++step) {
        uint8_t in_conds[BITS], out_conds[BITS];
        WordCond64 sig1_out = wc_Sigma1_512(cs.e(step-1));
        for (int b = 0; b < BITS; ++b) { in_conds[b] = cs.e(step-1).get(b); out_conds[b] = sig1_out.get(b); }
        for (auto& c : extract_sigma_twobit(e_word_id(step-1), 14, 18, 41, false, in_conds, out_conds, BITS))
            count_cond(c);
    }

    // σ₀(W_{i-15}): 1, 8, 7 (shift)
    for (int i = 16; i < 80; ++i) {
        uint8_t in_conds[BITS], out_conds[BITS];
        WordCond64 s0_out = wc_sigma0_512(cs.w(i-15));
        for (int b = 0; b < BITS; ++b) { in_conds[b] = cs.w(i-15).get(b); out_conds[b] = s0_out.get(b); }
        for (auto& c : extract_sigma_twobit(w_word_id(i-15), 1, 8, 7, true, in_conds, out_conds, BITS))
            count_cond(c);
    }

    // σ₁(W_{i-2}): 19, 61, 6 (shift)
    for (int i = 16; i < 80; ++i) {
        uint8_t in_conds[BITS], out_conds[BITS];
        WordCond64 s1_out = wc_sigma1_512(cs.w(i-2));
        for (int b = 0; b < BITS; ++b) { in_conds[b] = cs.w(i-2).get(b); out_conds[b] = s1_out.get(b); }
        for (auto& c : extract_sigma_twobit(w_word_id(i-2), 19, 61, 6, true, in_conds, out_conds, BITS))
            count_cond(c);
    }

    // Addition two-bit conditions (same equations as extract_addition_twobit_512).
    // We extract them into a temporary tracker just to count connectivity.
    {
        static const BoolPropTable tbl_ch_conn = BoolPropTable::build(TT_IF);
        static const BoolPropTable tbl_maj_conn = BoolPropTable::build(TT_MAJ);

        // E equation
        for (int step = 0; step < n; ++step) {
            for (int j = 0; j < BITS; ++j) {
                XorTerm terms[8];
                int n_opaque = 0;
                int p1=(j+14)&63, p2=(j+18)&63, p3=(j+41)&63;
                terms[0] = make_xor_term_512(e_word_id(step), j, cs.e(step).get(j));
                terms[1] = make_xor_term_512(e_word_id(step-4), j, cs.e(step-4).get(j));
                terms[2] = make_xor_term_512(e_word_id(step-1), p1, cs.e(step-1).get(p1));
                terms[3] = make_xor_term_512(e_word_id(step-1), p2, cs.e(step-1).get(p2));
                terms[4] = make_xor_term_512(e_word_id(step-1), p3, cs.e(step-1).get(p3));
                terms[5] = make_xor_term_512(a_word_id(step-4), j, cs.a(step-4).get(j));
                terms[6] = make_xor_term_512(w_word_id(step), j, cs.w(step).get(j));
                uint8_t ch_out = tbl_ch_conn.propagate(
                    BitCond(cs.e(step-1).get(j)), BitCond(cs.e(step-2).get(j)), BitCond(cs.e(step-3).get(j)));
                int ch_ds = bc_diff_status_512(ch_out);
                if (ch_ds >= 0) terms[7] = make_xor_known_512(ch_ds);
                else { terms[7] = make_xor_known_512(0); n_opaque++; }
                if (n_opaque == 0) {
                    auto conds = extract_xor_twobit(terms, 8, 0, nullptr);
                    for (auto& c : conds) count_cond(c);
                }
                bool chain_ok = true;
                if (!terms[1].known || terms[1].val != 0) chain_ok = false;
                if (chain_ok && (!terms[2].known || !terms[3].known || !terms[4].known)) chain_ok = false;
                else if (chain_ok && (terms[2].val ^ terms[3].val ^ terms[4].val) != 0) chain_ok = false;
                if (chain_ok && ch_ds != 0) chain_ok = false;
                if (chain_ok && (!terms[5].known || terms[5].val != 0)) chain_ok = false;
                if (chain_ok && (!terms[6].known || terms[6].val != 0)) chain_ok = false;
                if (chain_ok) continue;
                break;
            }
        }
        // A equation
        for (int step = 0; step < n; ++step) {
            for (int j = 0; j < BITS; ++j) {
                XorTerm terms[7];
                int n_opaque = 0;
                int p1=(j+28)&63, p2=(j+34)&63, p3=(j+39)&63;
                terms[0] = make_xor_term_512(a_word_id(step), j, cs.a(step).get(j));
                terms[1] = make_xor_term_512(a_word_id(step-4), j, cs.a(step-4).get(j));
                terms[2] = make_xor_term_512(a_word_id(step-1), p1, cs.a(step-1).get(p1));
                terms[3] = make_xor_term_512(a_word_id(step-1), p2, cs.a(step-1).get(p2));
                terms[4] = make_xor_term_512(a_word_id(step-1), p3, cs.a(step-1).get(p3));
                terms[5] = make_xor_term_512(e_word_id(step), j, cs.e(step).get(j));
                uint8_t maj_out = tbl_maj_conn.propagate(
                    BitCond(cs.a(step-1).get(j)), BitCond(cs.a(step-2).get(j)), BitCond(cs.a(step-3).get(j)));
                int maj_ds = bc_diff_status_512(maj_out);
                if (maj_ds >= 0) terms[6] = make_xor_known_512(maj_ds);
                else { terms[6] = make_xor_known_512(0); n_opaque++; }
                if (n_opaque == 0) {
                    auto conds = extract_xor_twobit(terms, 7, 0, nullptr);
                    for (auto& c : conds) count_cond(c);
                }
                bool chain_ok = true;
                if (!terms[1].known || terms[1].val != 0) chain_ok = false;
                if (chain_ok && (!terms[2].known || !terms[3].known || !terms[4].known)) chain_ok = false;
                else if (chain_ok && (terms[2].val ^ terms[3].val ^ terms[4].val) != 0) chain_ok = false;
                if (chain_ok && maj_ds != 0) chain_ok = false;
                if (chain_ok && (!terms[5].known || terms[5].val != 0)) chain_ok = false;
                if (chain_ok) continue;
                break;
            }
        }
        // W expansion
        for (int i = 16; i < 80; ++i) {
            for (int j = 0; j < BITS; ++j) {
                XorTerm terms[9];
                int nt = 0;
                terms[nt++] = make_xor_term_512(w_word_id(i), j, cs.w(i).get(j));
                int q1=(j+19)&63, q2=(j+61)&63, q3=j+6;
                terms[nt++] = make_xor_term_512(w_word_id(i-2), q1, cs.w(i-2).get(q1));
                terms[nt++] = make_xor_term_512(w_word_id(i-2), q2, cs.w(i-2).get(q2));
                if (q3 < BITS) terms[nt++] = make_xor_term_512(w_word_id(i-2), q3, cs.w(i-2).get(q3));
                else terms[nt++] = make_xor_known_512(0);
                terms[nt++] = make_xor_term_512(w_word_id(i-7), j, cs.w(i-7).get(j));
                int r1=(j+1)&63, r2=(j+8)&63, r3=j+7;
                terms[nt++] = make_xor_term_512(w_word_id(i-15), r1, cs.w(i-15).get(r1));
                terms[nt++] = make_xor_term_512(w_word_id(i-15), r2, cs.w(i-15).get(r2));
                if (r3 < BITS) terms[nt++] = make_xor_term_512(w_word_id(i-15), r3, cs.w(i-15).get(r3));
                else terms[nt++] = make_xor_known_512(0);
                terms[nt++] = make_xor_term_512(w_word_id(i-16), j, cs.w(i-16).get(j));
                auto conds = extract_xor_twobit(terms, nt, 0, nullptr);
                for (auto& c : conds) count_cond(c);
                bool chain_ok = true;
                if (!terms[1].known || !terms[2].known || !terms[3].known) chain_ok = false;
                else if ((terms[1].val ^ terms[2].val ^ terms[3].val) != 0) chain_ok = false;
                if (chain_ok && (!terms[4].known || terms[4].val != 0)) chain_ok = false;
                if (chain_ok && (!terms[5].known || !terms[6].known || !terms[7].known)) chain_ok = false;
                else if (chain_ok && (terms[5].val ^ terms[6].val ^ terms[7].val) != 0) chain_ok = false;
                if (chain_ok && (!terms[8].known || terms[8].val != 0)) chain_ok = false;
                if (chain_ok) continue;
                break;
            }
        }
    }

    return cm;
}

// Stage 1 bits: W '?' and 'x' bits (message expansion differential).
// Resolving W first has the highest propagation impact because message
// expansion constrains W[16..79], which in turn constrains A/E through
// the step function. Matches the paper's Stage 1.
inline std::vector<BitLoc> collect_stage1_bits(const CharState512& cs) {
    std::vector<BitLoc> bits;
    int n = std::min(cs.n_steps, (int)cs.W.size());
    for (int i = 0; i < n; ++i) {
        uint64_t mask = cs.W[i].free_mask() | cs.W[i].x_mask();
        while (mask) {
            int b = gc_ctz64(mask);
            bits.push_back({WT_W, i, b});
            mask &= mask - 1;
        }
    }
    return bits;
}

// Stage 2 bits: A/E '?' and 'x' bits (state update differential).
// Resolved after Stage 1. Matches the paper's Stage 2.
// Paper §4.3: "We pick decision bits more often from A, since this results
// in sparser characteristics for A."  A bits are included twice to give them
// ~2:1 selection odds over E bits in the randomized look-ahead.
inline std::vector<BitLoc> collect_stage2_bits(const CharState512& cs) {
    std::vector<BitLoc> bits;
    auto scan = [&](WordType wt, const std::vector<WordCond64>& words,
                    int step_offset) {
        for (int i = 0; i < (int)words.size(); ++i) {
            int step = i + step_offset;
            uint64_t mask = words[i].free_mask() | words[i].x_mask();
            while (mask) {
                int b = gc_ctz64(mask);
                bits.push_back({wt, step, b});
                mask &= mask - 1;
            }
        }
    };
    scan(WT_A, cs.A, -4);
    scan(WT_A, cs.A, -4);  // A-bias: include A bits twice
    scan(WT_E, cs.E, -4);
    return bits;
}

// Collect all Phase 1 undecided bits ('?' and 'x') across A, E, W.
// Used for counting and best-solution tracking.
inline std::vector<BitLoc> collect_phase1_bits(const CharState512& cs) {
    auto s1 = collect_stage1_bits(cs);
    auto s2 = collect_stage2_bits(cs);
    s1.insert(s1.end(), s2.begin(), s2.end());
    return s1;
}

// Collect W[0..15] dash bits for message-first strategy.
inline std::vector<BitLoc> collect_msg_dash_bits(const CharState512& cs) {
    std::vector<BitLoc> bits;
    int limit = std::min(16, cs.n_steps);
    for (int i = 0; i < limit; ++i) {
        uint64_t dm = cs.W[i].dash_mask();
        while (dm) {
            int b = gc_ctz64(dm);
            bits.push_back({WT_W, i, b});
            dm &= dm - 1;
        }
    }
    return bits;
}

// ---- Backjumping support (non-chronological backtracking) ----
// When both choices of a decision fail, identify which EARLIER decision
// caused the conflict and jump there directly, skipping irrelevant levels.
// Adapted from the SHA-256 search engine for 64-bit words.

// Encode a BitLoc to a flat index for the bit_level array.
// Layout: A words [0..(n+4)), E words [(n+4)..2(n+4)), W words [2(n+4)..2(n+4)+80)
// Each word occupies 64 consecutive positions.
inline int encode_bitloc_512(const BitLoc& loc, int n_steps) {
    switch (loc.type) {
        case WT_A: return (loc.step + 4) * 64 + loc.bit;
        case WT_E: return ((n_steps + 4) + loc.step + 4) * 64 + loc.bit;
        case WT_W: return (2 * (n_steps + 4) + loc.step) * 64 + loc.bit;
    }
    GC_UNREACHABLE();
}

inline int bit_level_total_512(int n_steps) {
    return (2 * (n_steps + 4) + 80) * 64;
}

// Compute backjump target from the step where propagation failed.
// Checks all input registers of the failing equation for stamp levels.
// fail_step encoding: 0..n-1 = step function, 1000+i = message expansion W[i].
// Returns the highest bit_level strictly below trail_size, or -1 if no target.
inline int compute_backjump_from_step_512(
    const std::vector<int>& bit_level, int fail_step,
    int trail_size, int n_steps)
{
    if (fail_step < -1) return -1;

    int n_total = (int)bit_level.size();
    int best = -1;

    auto check_word = [&](WordType wt, int step) {
        for (int b = 0; b < 64; ++b) {
            BitLoc loc{wt, step, b};
            int idx = encode_bitloc_512(loc, n_steps);
            if (idx >= 0 && idx < n_total) {
                int lv = bit_level[idx];
                if (lv >= 0 && lv < trail_size)
                    best = std::max(best, lv);
            }
        }
    };

    if (fail_step >= 0 && fail_step < n_steps) {
        // Step function inputs: A[s-4..s-1], E[s-4..s-1], W[s], A[s], E[s]
        int s = fail_step;
        check_word(WT_A, s - 4); check_word(WT_A, s - 3);
        check_word(WT_A, s - 2); check_word(WT_A, s - 1);
        check_word(WT_E, s - 4); check_word(WT_E, s - 3);
        check_word(WT_E, s - 2); check_word(WT_E, s - 1);
        check_word(WT_W, s);
        check_word(WT_A, s); check_word(WT_E, s);
    } else if (fail_step >= 1000) {
        // Message expansion: W_i = σ₁(W_{i-2}) + W_{i-7} + σ₀(W_{i-15}) + W_{i-16}
        int i = fail_step - 1000;
        check_word(WT_W, i - 2); check_word(WT_W, i - 7);
        check_word(WT_W, i - 15); check_word(WT_W, i - 16);
        check_word(WT_W, i);
    }
    // fail_step == -1: twobit failure, no step info → returns -1

    return best;
}

// Stamp newly changed bits with the decision level that caused them.
// Compares current state vs snapshot; updates bit_level[encoded] = level
// for any bit whose condition changed.
inline void stamp_changed_bits_512(const CharState512& cs, const CharState512& snap,
                                    std::vector<int>& bit_level, int level, int n_steps) {
    auto do_word = [&](const WordCond64& cur, const WordCond64& old, int base_idx) {
        uint64_t diff = 0;
        for (int k = 0; k < 4; ++k) diff |= (cur.allow[k] ^ old.allow[k]);
        while (diff) {
            int b = gc_ctz64(diff);
            if (base_idx + b < (int)bit_level.size())
                bit_level[base_idx + b] = level;
            diff &= diff - 1;
        }
    };
    int n4 = n_steps + 4;
    for (int i = 0; i < n4; ++i)
        do_word(cs.A[i], snap.A[i], i * 64);
    for (int i = 0; i < n4; ++i)
        do_word(cs.E[i], snap.E[i], (n4 + i) * 64);
    for (int i = 0; i < 80; ++i)
        do_word(cs.W[i], snap.W[i], (2 * n4 + i) * 64);
}

// Collect Phase 2 candidate bits: '-' bits with nonzero connectivity.
// When activity map is provided, sort by (-connectivity, -activity) to
// prioritize bits involved in recent conflicts (EVSIDS-style).
inline std::vector<BitLoc> collect_phase2_bits(const CharState512& cs,
                                                const ConnectivityMap& conn,
                                                const ActivityMap* activity = nullptr) {
    std::vector<std::pair<std::pair<int, double>, BitLoc>> scored;
    auto scan = [&](WordType wt, const std::vector<WordCond64>& words,
                    int step_offset) {
        for (int i = 0; i < (int)words.size(); ++i) {
            int step = i + step_offset;
            uint64_t dm = words[i].dash_mask();
            while (dm) {
                int b = gc_ctz64(dm);
                BitLoc loc{wt, step, b};
                int c = conn.get(loc);
                if (c > 0) {
                    double act = activity ? -activity->get(loc) : 0.0;
                    scored.push_back({{-c, act}, loc});
                }
                dm &= dm - 1;
            }
        }
    };
    scan(WT_A, cs.A, -4);
    scan(WT_E, cs.E, -4);
    // W[16+] are expansion-derived: W_i = σ₁(W_{i-2}) + W_{i-7} +
    // σ₀(W_{i-15}) + W_{i-16}.  Deciding them directly before their inputs
    // are resolved causes spurious contradictions.  Only include W[0..15]
    // (primary message words) in Phase 2 candidates.
    {
        int w_limit = std::min(16, (int)cs.W.size());
        for (int i = 0; i < w_limit; ++i) {
            uint64_t dm = cs.W[i].dash_mask();
            while (dm) {
                int b = gc_ctz64(dm);
                BitLoc loc{WT_W, i, b};
                int c = conn.get(loc);
                if (c > 0) {
                    double act = activity ? -activity->get(loc) : 0.0;
                    scored.push_back({{-c, act}, loc});
                }
                dm &= dm - 1;
            }
        }
    }

    std::sort(scored.begin(), scored.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });

    std::vector<BitLoc> bits;
    bits.reserve(scored.size());
    for (auto& [_, loc] : scored)
        bits.push_back(loc);
    return bits;
}

// Collect remaining dash bits (zero connectivity).
inline std::vector<BitLoc> collect_free_dash_bits(const CharState512& cs,
                                                   const ConnectivityMap& conn) {
    std::vector<BitLoc> bits;
    auto scan_zero_conn = [&](WordType wt, const std::vector<WordCond64>& words,
                              int step_offset) {
        for (int i = 0; i < (int)words.size(); ++i) {
            int step = i + step_offset;
            uint64_t dm = words[i].dash_mask();
            while (dm) {
                int b = gc_ctz64(dm);
                BitLoc loc{wt, step, b};
                if (conn.get(loc) == 0)
                    bits.push_back(loc);
                dm &= dm - 1;
            }
        }
    };
    scan_zero_conn(WT_A, cs.A, -4);
    scan_zero_conn(WT_E, cs.E, -4);
    scan_zero_conn(WT_W, cs.W, 0);
    return bits;
}

// ---- Ordered sub-phase guessing (dobraunig2015 Appendix B) ----
//
// For dash resolution (Phase II-IV in dobraunig2015 / Stage 3 in eichlseder2014),
// instead of random/look-ahead selection from a large pool, the paper uses:
//   1. Sub-phased word targeting: specific word ranges per sub-phase
//   2. LSB→MSB ordering: within each word, fix bit 0 first, then bit 1, etc.
//   3. No look-ahead in dash phases (LA only in Phase I)
//
// This enables carry chain cascade through modular addition: fixing bit k
// determines the carry into bit k+1, which helps propagation resolve more bits.
//
// Sub-phase definitions for 38-step SFS (adapted from dobraunig2015 39-step
// by shifting step ranges -1):
//   Phase II:  A[8..11], E[8..11]  — core of local collision
//   Phase III: E[12..23]           — extended active region
//   Phase IV:  W[8..11]            — message words in active region
//   Phase V:   all remaining '-'   — cleanup (fallback)

struct SubPhaseDef {
    const char* name;
    // Word specifications: (word_type, step_lo, step_hi) inclusive
    struct WordRange { WordType wt; int lo; int hi; };
    std::vector<WordRange> ranges;
};

// Collect dash bits for a sub-phase, ordered LSB→MSB within each word.
// Words are processed in the order listed in the sub-phase definition.
inline std::vector<BitLoc> collect_ordered_subphase_bits(
    const CharState512& cs, const SubPhaseDef& phase) {
    std::vector<BitLoc> bits;
    for (auto& [wt, lo, hi] : phase.ranges) {
        for (int step = lo; step <= hi; ++step) {
            const auto& wc = (wt == WT_A) ? cs.a(step)
                           : (wt == WT_E) ? cs.e(step)
                           : cs.w(step);
            for (int b = 0; b < 64; ++b) {
                if (wc.get(b) == BC_DASH)
                    bits.push_back({wt, step, b});
            }
        }
    }
    return bits;
}

// Collect ALL remaining dash bits, ordered LSB→MSB (fallback Phase V).
inline std::vector<BitLoc> collect_ordered_all_dash(const CharState512& cs) {
    std::vector<BitLoc> bits;
    // A words first, then E, then W — each ordered by step then bit
    for (int step = -4; step < cs.n_steps; ++step) {
        for (int b = 0; b < 64; ++b)
            if (cs.a(step).get(b) == BC_DASH)
                bits.push_back({WT_A, step, b});
    }
    for (int step = -4; step < cs.n_steps; ++step) {
        for (int b = 0; b < 64; ++b)
            if (cs.e(step).get(b) == BC_DASH)
                bits.push_back({WT_E, step, b});
    }
    for (int i = 0; i < (int)cs.W.size(); ++i) {
        for (int b = 0; b < 64; ++b)
            if (cs.w(i).get(b) == BC_DASH)
                bits.push_back({WT_W, i, b});
    }
    return bits;
}

// Default sub-phases for 38-step SFS.
// Derived from dobraunig2015 39-step (Phase II: A9..12/E9..12,
// Phase III: E13..24, Phase IV: W9..12) shifted by -1 step.
inline std::vector<SubPhaseDef> default_subphases_38sfs() {
    return {
        {"PhII:A8-11,E8-11", {{WT_A, 8, 11}, {WT_E, 8, 11}}},
        {"PhIII:E12-23",     {{WT_E, 12, 23}}},
        {"PhIV:W8-11",       {{WT_W, 8, 11}}},
    };
}

// ---- Look-ahead branching heuristic (Algorithm 2) ----

struct LookAheadDecision {
    BitLoc loc;
    BitCond first_choice;
    BitCond alt_choice;
};

inline std::pair<BitCond, BitCond> phase_choices(
    BitCond cur, int phase, std::mt19937& rng)
{
    if (phase == 1) {
        // Paper Table 1: ? → always try '-' first (probability 1), 'x' as alt.
        // This biases toward sparser characteristics, making Phase 2/3 easier.
        if (bc_is_free(cur))
            return std::make_pair(BC_DASH, BC_X);
        // x → u/n with equal probability (paper Table 1).
        return (rng() & 1) ? std::make_pair(BC_U, BC_N)
                           : std::make_pair(BC_N, BC_U);
    }
    return (rng() & 1) ? std::make_pair(BC_0, BC_1)
                       : std::make_pair(BC_1, BC_0);
}

inline LookAheadDecision look_ahead_pick(
    const CharState512& cs,
    const std::vector<BitLoc>& candidates,
    int s_max,
    int phase,
    std::mt19937& rng)
{
    int n_cand = (int)candidates.size();
    if (n_cand == 1) {
        auto [f, a] = phase_choices(get_cond(cs, candidates[0]), phase, rng);
        return {candidates[0], f, a};
    }

    int n_eval = std::min(s_max, n_cand);

    std::vector<int> order(n_cand);
    std::iota(order.begin(), order.end(), 0);
    std::shuffle(order.begin(), order.end(), rng);

    int decided_before = cs.count_decided();
    int best_order_pos = 0;
    int best_score = -1;
    BitCond best_first{}, best_alt{};

    std::vector<bool> skip(n_cand, false);

    int evaluated = 0;
    for (int pos = 0; pos < n_cand && evaluated < n_eval; ++pos) {
        int idx = order[pos];
        if (skip[idx]) continue;

        ++evaluated;

        const BitLoc& loc = candidates[idx];
        auto [test_val, alt_val] = phase_choices(
            get_cond(cs, loc), phase, rng);

        CharState512 test = cs;
        get_word(test, loc).impose(loc.bit, test_val);
        bool ok = propagate_full(test) && !test.has_contradiction();

        if (!ok) {
            return {loc, alt_val, test_val};
        }

        int score = test.count_decided() - decided_before;
        if (score > best_score) {
            best_score = score;
            best_order_pos = pos;
            best_first = test_val;
            best_alt = alt_val;
        }

        for (int other = pos + 1; other < n_cand; ++other) {
            int other_idx = order[other];
            if (skip[other_idx]) continue;
            if (get_cond(cs, candidates[other_idx]) !=
                get_cond(test, candidates[other_idx]))
                skip[other_idx] = true;
        }
    }

    int best_idx = order[best_order_pos];
    return {candidates[best_idx], best_first, best_alt};
}

inline LookAheadDecision select_decision(
    const CharState512& cs,
    const std::vector<BitLoc>& candidates,
    int phase,
    bool use_look_ahead,
    int la_s_max,
    int window,
    std::mt19937& rng)
{
    if (candidates.empty()) return {{WT_A, 0, 0}, BC_DASH, BC_X};
    int limit = (window > 0) ? std::min((int)candidates.size(), window)
                             : (int)candidates.size();

    if (use_look_ahead && limit > 1) {
        std::vector<BitLoc> subset(candidates.begin(),
                                   candidates.begin() + limit);
        return look_ahead_pick(cs, subset, la_s_max, phase, rng);
    }

    int idx = std::uniform_int_distribution<int>(0, limit - 1)(rng);
    auto [first, alt] = phase_choices(
        get_cond(cs, candidates[idx]), phase, rng);
    return {candidates[idx], first, alt};
}

// ---- Search engine ----

struct SearchConfig {
    int max_contradictions = 1000;
    int max_decisions = 100000;
    int max_restarts = 10000;     // prevent infinite restart loops
    bool verbose = false;
    bool phase1_only = false;
    int phase2_window = 16;
    bool msg_first = false;
    bool use_look_ahead = true;   // default ON for SHA-512
    int la_s_max = 16;            // paper's optimal value
    bool use_backjump = false;    // non-chronological backtracking
    int max_backjump_skip = 32;   // max levels to skip per backjump (0=unlimited)
    bool ordered_guess = false;   // dobraunig2015: LSB→MSB ordered guessing in dash phases
    std::vector<SubPhaseDef> subphases; // ordered sub-phases; empty = use default_subphases_38sfs()
    bool use_stacking = false;    // dobraunig2015 Table 2: ?→- not stacked (p=0), x→u/n and -→0/1 stacked (p=1)
    bool use_ninput = false;      // n-input in propagate_full convergence loop (expensive — P2 only)
    int ninput_interval = 0;      // periodic standalone n-input check every K decisions (0=off)
    std::vector<int> ninput_hot_steps; // targeted E-equation n-input at these steps (every decision)
    bool use_activity = false;    // conflict-driven activity scoring (EVSIDS-style)
    double activity_decay = 0.95; // decay factor per decision

    // Per-restart callback: called at end of each restart with
    //   (restart_id, decisions_this_restart, contradictions_this_restart,
    //    best_p1_this_restart, total_decisions, total_contradictions,
    //    p1_reached_zero, best_dash_this_restart)
    // best_dash tracks Phase 2/3 progress (lower = more resolved).
    // p1_reached_zero indicates whether Phase 1 completed in this restart.
    // Set to nullptr to disable.
    void (*restart_callback)(int, int, int, int, int, int, bool, int) = nullptr;

    // External stop flag. If non-null, search checks this periodically
    // and exits gracefully when it becomes true.
    const std::atomic<bool>* stop_flag = nullptr;

    // Wall-clock deadline. If non-null, search exits when deadline is passed.
    // Checked every 200 decisions (low overhead).
    const std::chrono::steady_clock::time_point* wall_deadline = nullptr;
};

struct SearchResult {
    bool found;
    int restarts;
    int total_decisions;
    int total_contradictions;
    int total_backjump_skips = 0;  // levels skipped by backjumping
    int best_dash = INT_MAX;       // best dash count seen (Phase 2/3 progress)
    CharState512 solution;
};

inline SearchResult search(CharState512 initial, SearchConfig config, std::mt19937& rng) {
    SearchResult result{false, 0, 0, 0, 0, INT_MAX, initial};
    int best_p1 = INT_MAX;

    // Resolve ordered sub-phases for dash guessing
    const auto active_subphases = config.ordered_guess
        ? (config.subphases.empty() ? default_subphases_38sfs() : config.subphases)
        : std::vector<SubPhaseDef>{};

    int dec_at_restart_start = 0;
    int ctr_at_restart_start = 0;
    int best_p1_this_restart = INT_MAX;
    bool p1_reached_zero = false;
    int best_dash_this_restart = INT_MAX;

    auto restart = [&]() {
        if (config.restart_callback) {
            int dec_this = result.total_decisions - dec_at_restart_start;
            int ctr_this = result.total_contradictions - ctr_at_restart_start;
            config.restart_callback(result.restarts, dec_this, ctr_this,
                                    best_p1_this_restart,
                                    result.total_decisions,
                                    result.total_contradictions,
                                    p1_reached_zero,
                                    best_dash_this_restart);
        }
        ++result.restarts;
        dec_at_restart_start = result.total_decisions;
        ctr_at_restart_start = result.total_contradictions;
        best_p1_this_restart = INT_MAX;
        p1_reached_zero = false;
        best_dash_this_restart = INT_MAX;
    };

    // Conflict-driven activity scoring (persists across restarts)
    ActivityMap activity;
    if (config.use_activity) {
        activity.init(initial.n_steps);
        activity.decay = config.activity_decay;
    }

    while (result.total_decisions < config.max_decisions &&
           result.restarts < config.max_restarts) {
        // Check stop conditions
        if (config.stop_flag && config.stop_flag->load(std::memory_order_relaxed))
            break;
        if (config.wall_deadline && std::chrono::steady_clock::now() >= *config.wall_deadline)
            break;

        CharState512 cs = initial;
        std::vector<Decision> trail;
        int contradictions_this_run = 0;
        int last_ninput_check = result.total_decisions;  // total_decisions at last n-input check

        // Backjump: decision-level stamps for each bit
        int n_total_bits = bit_level_total_512(cs.n_steps);
        std::vector<int> dl;
        if (config.use_backjump) dl.assign(n_total_bits, -1);

        // Initial propagation (bitwise-only for SHA-512)
        if (!propagate_full(cs)) {
            restart();
            continue;
        }

        auto conn = compute_connectivity(cs);
        int conn_refresh_counter = 0;
        bool phase2_bootstrapped = false;

        while (result.total_decisions < config.max_decisions) {
            if (++conn_refresh_counter >= 200) {
                conn = compute_connectivity(cs);
                conn_refresh_counter = 0;
                // Check stop conditions periodically (every 200 decisions)
                if (config.stop_flag &&
                    config.stop_flag->load(std::memory_order_relaxed))
                    break;
                if (config.wall_deadline &&
                    std::chrono::steady_clock::now() >= *config.wall_deadline)
                    break;
            }

            // Stage 1: W undecided bits first (highest propagation impact)
            auto s1_bits = collect_stage1_bits(cs);
            // Stage 2: A/E undecided bits
            auto s2_bits = s1_bits.empty() ? collect_stage2_bits(cs)
                                           : std::vector<BitLoc>{};

            BitLoc chosen;
            BitCond first_choice, alt_choice;

            if (!s1_bits.empty()) {
                // Stage 1: resolve W message expansion differential
                auto sel = select_decision(cs, s1_bits, 1,
                    config.use_look_ahead, config.la_s_max, 0, rng);
                chosen = sel.loc;
                first_choice = sel.first_choice;
                alt_choice = sel.alt_choice;
            } else if (!s2_bits.empty()) {
                // Stage 2: resolve A/E state differential
                auto sel = select_decision(cs, s2_bits, 1,
                    config.use_look_ahead, config.la_s_max, 0, rng);
                chosen = sel.loc;
                first_choice = sel.first_choice;
                alt_choice = sel.alt_choice;
            } else {
                // Stages 1+2 complete: all ?/x bits resolved.
                if (config.phase1_only) {
                    result.found = true;
                    result.solution = cs;
                    return result;
                }

                // Phase boundary re-propagation: Phase 1 resolved all ?/x bits,
                // exposing new twobit deductions. Re-propagate and refresh
                // connectivity once at the Phase 1→2 boundary.
                if (!phase2_bootstrapped) {
                    phase2_bootstrapped = true;
                    if (!propagate_full(cs) || cs.has_contradiction()
                        || !check_twobit(cs))
                        break;
                    conn = compute_connectivity(cs);
                    conn_refresh_counter = 0;
                }

                // Message-first: resolve W[0..15] dashes before A/E dashes.
                // Each W bit propagates through the step function carry chain,
                // resolving multiple dashes per assignment.
                auto msg_dashes = config.msg_first ? collect_msg_dash_bits(cs)
                                                   : std::vector<BitLoc>{};
                if (!msg_dashes.empty()) {
                    auto sel = select_decision(cs, msg_dashes, 2,
                        config.use_look_ahead, config.la_s_max, 0, rng);
                    chosen = sel.loc;
                    first_choice = sel.first_choice;
                    alt_choice = sel.alt_choice;
                } else if (config.ordered_guess) {
                    // Ordered sub-phase guessing (dobraunig2015):
                    // Try each sub-phase in order; within each, pick the
                    // first (LSB) unresolved dash bit. No look-ahead.
                    const auto& sps = active_subphases;
                    std::vector<BitLoc> bits;
                    int sp_idx = 0;
                    for (; sp_idx < (int)sps.size(); ++sp_idx) {
                        bits = collect_ordered_subphase_bits(cs, sps[sp_idx]);
                        if (!bits.empty()) break;
                    }
                    if (bits.empty()) {
                        // All sub-phases exhausted — fallback to all remaining
                        bits = collect_ordered_all_dash(cs);
                    }
                    if (bits.empty()) {
                        if (!cs.has_contradiction() && check_twobit(cs)) {
                            result.found = true;
                            result.solution = cs;
                            return result;
                        }
                        break;
                    }
                    // Pick first bit (LSB of first word in sub-phase)
                    chosen = bits[0];
                    auto [f, a] = phase_choices(BC_DASH, 2, rng);
                    first_choice = f;
                    alt_choice = a;
                } else {
                    auto p2_bits = collect_phase2_bits(cs, conn,
                        config.use_activity ? &activity : nullptr);
                    if (p2_bits.empty()) {
                        auto free_bits = collect_free_dash_bits(cs, conn);
                        if (free_bits.empty()) {
                            if (!cs.has_contradiction() && check_twobit(cs)) {
                                result.found = true;
                                result.solution = cs;
                                return result;
                            }
                            break;
                        }

                        auto sel = select_decision(cs, free_bits, 3,
                            config.use_look_ahead, config.la_s_max, 0, rng);
                        chosen = sel.loc;
                        first_choice = sel.first_choice;
                        alt_choice = sel.alt_choice;
                    } else {
                        auto sel = select_decision(cs, p2_bits, 2,
                            config.use_look_ahead, config.la_s_max,
                            config.phase2_window, rng);
                        chosen = sel.loc;
                        first_choice = sel.first_choice;
                        alt_choice = sel.alt_choice;
                    }
                }
            }

            Decision dec;
            dec.loc = chosen;
            dec.original = get_cond(cs, chosen);
            dec.first_choice = first_choice;
            dec.alt_choice = alt_choice;
            dec.tried_alt = false;
            dec.critical = false;
            // Stacking: ?→- (BC_FREE→BC_DASH/BC_X) has p=0 (not stacked)
            // x→u/n and -→0/1 have p=1 (always stacked)
            dec.stacked = !config.use_stacking || (dec.original != BC_FREE);
            dec.snapshot = cs;
            trail.push_back(dec);

            ++result.total_decisions;
            if (config.use_activity) activity.decay_step();

            get_word(cs, chosen).impose(chosen.bit, first_choice);

            bool phase1_done = s1_bits.empty() && s2_bits.empty();
            bool do_twobit = phase1_done ||
                             (s1_bits.empty() && (int)s2_bits.size() <= 32) ||
                             result.total_decisions % 50 == 0;
            int fail_step_1 = -2;
            bool need_fail_step = config.use_backjump || config.use_activity;
            bool ok = propagate_full(cs, need_fail_step ? &fail_step_1 : nullptr)
                       && !cs.has_contradiction()
                       && (!do_twobit || check_twobit(cs));

            // Targeted n-input hot-step check (every decision, ~0.5 μs per step)
            if (ok && !config.ninput_hot_steps.empty()) {
                for (int hs : config.ninput_hot_steps) {
                    bool ni_changed = false;
                    if (!ninput_check_E_step(cs, hs, ni_changed)) {
                        ok = false; break;
                    }
                    if (ni_changed) {
                        ok = propagate_full(cs) && !cs.has_contradiction();
                        if (!ok) break;
                    }
                }
            }

            // Periodic full n-input validation on first choice
            bool ninput_due = config.ninput_interval > 0 &&
                (result.total_decisions - last_ninput_check >= config.ninput_interval);
            if (ok && ninput_due) {
                bool ni_changed = false;
                if (!propagate_ninput_512(cs, ni_changed)) {
                    ok = false;
                } else if (ni_changed) {
                    ok = propagate_full(cs) && !cs.has_contradiction();
                }
                if (ok) last_ninput_check = result.total_decisions;
            }

            if (ok) {
                // Stamp bits changed by this successful decision
                if (config.use_backjump)
                    stamp_changed_bits_512(cs, trail.back().snapshot, dl,
                                           (int)trail.size() - 1, cs.n_steps);
            } else {
                cs = trail.back().snapshot;
                trail.back().tried_alt = true;
                get_word(cs, chosen).impose(chosen.bit, alt_choice);

                int fail_step_2 = -2;
                ok = propagate_full(cs, need_fail_step ? &fail_step_2 : nullptr)
                      && !cs.has_contradiction()
                      && (!do_twobit || check_twobit(cs));

                // Targeted n-input hot-step check on alt choice
                if (ok && !config.ninput_hot_steps.empty()) {
                    for (int hs : config.ninput_hot_steps) {
                        bool ni_changed = false;
                        if (!ninput_check_E_step(cs, hs, ni_changed)) {
                            ok = false; break;
                        }
                        if (ni_changed) {
                            ok = propagate_full(cs) && !cs.has_contradiction();
                            if (!ok) break;
                        }
                    }
                }

                // Periodic full n-input validation on alt choice
                // (fires if first choice failed n-input or interval elapsed since last success)
                if (ok && ninput_due) {
                    bool ni_changed = false;
                    if (!propagate_ninput_512(cs, ni_changed)) {
                        ok = false;
                    } else if (ni_changed) {
                        ok = propagate_full(cs) && !cs.has_contradiction();
                    }
                    if (ok) last_ninput_check = result.total_decisions;
                }

                if (ok) {
                    // Stamp bits changed by alt choice
                    if (config.use_backjump)
                        stamp_changed_bits_512(cs, trail.back().snapshot, dl,
                                               (int)trail.size() - 1, cs.n_steps);
                } else {
                    trail.back().critical = true;
                    ++contradictions_this_run;
                    ++result.total_contradictions;

                    // Bump activity for conflict bits (use snapshot — cs is
                    // corrupted by the second propagation attempt)
                    if (config.use_activity) {
                        const auto& snap = trail.back().snapshot;
                        bump_conflict_bits(activity, snap, fail_step_1, chosen);
                        bump_conflict_bits(activity, snap, fail_step_2, chosen);
                    }

                    // Compute backjump target from both fail steps
                    int last_fail = fail_step_2;
                    if (config.use_backjump && fail_step_1 >= 0) {
                        int t1 = compute_backjump_from_step_512(
                            dl, fail_step_1, (int)trail.size(), cs.n_steps);
                        int t2 = compute_backjump_from_step_512(
                            dl, fail_step_2, (int)trail.size(), cs.n_steps);
                        if (t1 >= 0 || t2 >= 0)
                            last_fail = (t1 >= t2) ? fail_step_1 : fail_step_2;
                    }

                    bool resolved = false;
                    while (!trail.empty()) {
                        // Non-chronological backjump: skip irrelevant levels
                        if (config.use_backjump && last_fail >= 0) {
                            int target = compute_backjump_from_step_512(
                                dl, last_fail, (int)trail.size(), cs.n_steps);
                            if (target >= 0 && target < (int)trail.size() - 1) {
                                int n_skipped = (int)trail.size() - 1 - target;
                                // Cap skip distance to prevent cascade destruction
                                if (config.max_backjump_skip > 0 &&
                                    n_skipped > config.max_backjump_skip)
                                    n_skipped = config.max_backjump_skip;
                                int actual_target = (int)trail.size() - 1 - n_skipped;
                                result.total_backjump_skips += n_skipped;
                                while ((int)trail.size() > actual_target + 1)
                                    trail.pop_back();
                                // Clear stale stamps from popped levels
                                for (int idx = 0; idx < n_total_bits; ++idx)
                                    if (dl[idx] >= (int)trail.size()) dl[idx] = -1;
                            }
                        }

                        Decision& top = trail.back();
                        // Stacking: non-stacked decisions (p=0) skip alternatives
                        if (!top.stacked) {
                            trail.pop_back();
                            continue;
                        }
                        if (!top.tried_alt) {
                            cs = top.snapshot;
                            top.tried_alt = true;
                            get_word(cs, top.loc).impose(top.loc.bit, top.alt_choice);

                            last_fail = -2;
                            ok = propagate_full(cs, need_fail_step ? &last_fail : nullptr)
                                  && !cs.has_contradiction()
                                  && (!do_twobit || check_twobit(cs));

                            // Periodic n-input validation on backtrack-alt
                            // (skip — backtrack alternatives should not trigger periodic check;
                            //  the check will fire again after enough new decisions)

                            if (ok) {
                                if (config.use_backjump) {
                                    int lv = (int)trail.size() - 1;
                                    for (int idx = 0; idx < n_total_bits; ++idx)
                                        if (dl[idx] == lv) dl[idx] = -1;
                                    stamp_changed_bits_512(cs, top.snapshot, dl,
                                                           lv, cs.n_steps);
                                }
                                resolved = true;
                                break;
                            }
                            top.critical = true;
                            ++contradictions_this_run;
                            ++result.total_contradictions;
                            if (config.use_activity)
                                bump_conflict_bits(activity, top.snapshot, last_fail, top.loc);
                        }
                        trail.pop_back();
                    }

                    if (!resolved) break;

                    if (contradictions_this_run >= config.max_contradictions)
                        break;
                }
            }

            if (config.verbose && result.total_decisions % 1000 == 0) {
                auto s1 = collect_stage1_bits(cs);
                auto s2 = collect_stage2_bits(cs);
                auto p2 = collect_phase2_bits(cs, conn);
                auto p3 = collect_free_dash_bits(cs, conn);
                auto msg = collect_msg_dash_bits(cs);
                std::printf("[%d dec, %d ctr] Msg=%zu S1=%zu S2=%zu Ph2=%zu Ph3=%zu\n",
                            result.total_decisions, result.total_contradictions,
                            msg.size(), s1.size(), s2.size(), p2.size(), p3.size());
            }

            if (result.total_decisions % 100 == 0) {
                auto s1 = collect_stage1_bits(cs);
                auto s2 = collect_stage2_bits(cs);
                int p1_count = (int)s1.size() + (int)s2.size();
                if (p1_count < best_p1) {
                    best_p1 = p1_count;
                    result.solution = cs;
                }
                if (p1_count < best_p1_this_restart)
                    best_p1_this_restart = p1_count;
                if (p1_count == 0) {
                    p1_reached_zero = true;
                    int dash = cs.count_dash();
                    if (dash < best_dash_this_restart)
                        best_dash_this_restart = dash;
                    if (dash < result.best_dash) {
                        result.best_dash = dash;
                        result.solution = cs;
                    }
                }
            }
        }

        restart();
    }

    // Final callback for the interrupted restart (stop_flag or deadline exit)
    bool was_interrupted = (config.stop_flag && config.stop_flag->load(std::memory_order_relaxed))
                        || (config.wall_deadline && std::chrono::steady_clock::now() >= *config.wall_deadline);
    if (config.restart_callback && was_interrupted) {
        int dec_this = result.total_decisions - dec_at_restart_start;
        int ctr_this = result.total_contradictions - ctr_at_restart_start;
        if (dec_this > 0) {
            config.restart_callback(result.restarts, dec_this, ctr_this,
                                    best_p1_this_restart,
                                    result.total_decisions,
                                    result.total_contradictions,
                                    p1_reached_zero,
                                    best_dash_this_restart);
        }
    }

    return result;
}

} // namespace eichlseder2014
