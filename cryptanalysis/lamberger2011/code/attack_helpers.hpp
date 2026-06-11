// attack_helpers.hpp — shared attack primitives for the 46-step boomerang.
//
// Provides: split_input, alpha variant generation, match_alpha,
//           msg_modify + its consistency helpers.
//
// All functions are parameterized by differential characteristic data
// (StepDiffs, Characteristic) so they work with any characteristic,
// not just Table 1.

#pragma once

#include "sha256.hpp"
#include "differentials.hpp"
#include <vector>
#include <random>

namespace attack46 {

using sha256::Word;
using sha256::State;
using sha256::Sigma_0;
using sha256::Sigma_1;
using sha256::Maj;
using sha256::Ch;
using sha256::compress;
using sha256::inverse_compress_range;
using sha256::msg_expand;
using sha256::W_to_set_A;
using sha256::rand_word;
using sha256::feed_forward;

// ---------------------------------------------------------------------------
// Input splitting (paper convention: [0..15] = message, [16..23] = IV)
// ---------------------------------------------------------------------------

inline void split_input(const Word src[24], State& iv, Word M[16]) {
    for (int i = 0; i < 16; ++i) M[i] = src[i];
    for (int i = 0; i < 8; ++i)  iv[i] = src[i + 16];
}

// ---------------------------------------------------------------------------
// Alpha variant generation
// ---------------------------------------------------------------------------
//
// The backward characteristic's IV diff (alpha) has NAF bits in certain
// registers. Flipping signs on "free" NAF bits yields additional valid
// backward characteristics. Which bits are free is determined empirically
// (piece 5e); this function takes the specification as parameters.
//
// NAF bit specification: position, original sign (+1 or -1), flippable flag.

struct NafBit {
    int position;   // bit position (0-31)
    int sign;       // original sign: +1 or -1
    bool flippable; // true if this bit can be freely sign-flipped
};

struct AlphaVariant {
    State alpha;  // IV diff: P* - P
};

// Build all alpha variants by enumerating sign flips on the specified
// NAF bits for registers E (index 4) and H (index 7).
// Other registers are taken from base_alpha unchanged.
inline std::vector<AlphaVariant> generate_alpha_variants(
    const State& base_alpha,
    const NafBit* e_naf, int n_e_naf,
    const NafBit* h_naf, int n_h_naf)
{
    // Count flippable bits
    int n_e_flip = 0, n_h_flip = 0;
    for (int i = 0; i < n_e_naf; i++) if (e_naf[i].flippable) n_e_flip++;
    for (int i = 0; i < n_h_naf; i++) if (h_naf[i].flippable) n_h_flip++;

    int n_variants = (1 << n_e_flip) * (1 << n_h_flip);
    std::vector<AlphaVariant> variants;
    variants.reserve(n_variants);

    for (int e_mask = 0; e_mask < (1 << n_e_flip); e_mask++) {
        // Build E register from NAF
        Word dE = 0;
        int flip_idx = 0;
        for (int i = 0; i < n_e_naf; i++) {
            int sign = e_naf[i].sign;
            if (e_naf[i].flippable) {
                if (e_mask & (1 << flip_idx)) sign = -sign;
                flip_idx++;
            }
            Word bit = Word(1) << e_naf[i].position;
            dE += (sign > 0) ? bit : -bit;
        }

        for (int h_mask = 0; h_mask < (1 << n_h_flip); h_mask++) {
            Word dH = 0;
            flip_idx = 0;
            for (int i = 0; i < n_h_naf; i++) {
                int sign = h_naf[i].sign;
                if (h_naf[i].flippable) {
                    if (h_mask & (1 << flip_idx)) sign = -sign;
                    flip_idx++;
                }
                Word bit = Word(1) << h_naf[i].position;
                dH += (sign > 0) ? bit : -bit;
            }

            AlphaVariant v;
            v.alpha = base_alpha;
            v.alpha[4] = dE;
            v.alpha[7] = dH;
            variants.push_back(v);
        }
    }
    return variants;
}

// Table 1 NAF specification (convenience function).
inline std::vector<AlphaVariant> generate_table1_alpha_variants() {
    // E NAF: +2^9, -2^23, -2^28 (all flippable)
    static constexpr NafBit E_NAF[] = {
        {9,  +1, true},
        {23, -1, true},
        {28, -1, true},
    };
    // H NAF: -2^2 (fixed), -2^11 (flip), -2^16 (flip), -2^22 (flip), -2^28 (fixed)
    static constexpr NafBit H_NAF[] = {
        {2,  -1, false},
        {11, -1, true},
        {16, -1, true},
        {22, -1, true},
        {28, -1, false},
    };
    return generate_alpha_variants(
        diff46::TABLE1.alpha,
        E_NAF, 3,
        H_NAF, 5);
}

// Check if a backward alpha matches any variant. Returns variant index or -1.
// Quick-rejects on registers that are constant across all variants.
inline int match_alpha(const State& alpha,
                       const State& base_alpha,
                       const std::vector<AlphaVariant>& variants) {
    // A and C must match base (typically 0), B and D must match base (typically +4)
    if (alpha[0] != base_alpha[0] || alpha[2] != base_alpha[2]) return -1;
    if (alpha[1] != base_alpha[1] || alpha[3] != base_alpha[3]) return -1;
    // F and G are fixed across variants
    if (alpha[5] != base_alpha[5]) return -1;
    if (alpha[6] != base_alpha[6]) return -1;
    // Only E and H vary; scan variants
    for (int i = 0; i < (int)variants.size(); i++) {
        if (alpha[4] == variants[i].alpha[4] &&
            alpha[7] == variants[i].alpha[7])
            return i;
    }
    return -1;
}

// ---------------------------------------------------------------------------
// Message modification (Wang-style, forward characteristic steps 21-29)
// ---------------------------------------------------------------------------

// Consistency condition LHS: C_A - C_E = dSig0(A) + dMaj(A,B,C) - dD.
// This is W-independent and must equal the RHS at each step.
inline Word consistency_lhs(const State& s0, const State& s2) {
    return (Sigma_0(s2[0]) - Sigma_0(s0[0]))
         + (Maj(s2[0], s2[1], s2[2]) - Maj(s0[0], s0[1], s0[2]))
         - (s2[3] - s0[3]);
}

// Consistency condition RHS from the characteristic's state diffs.
inline Word consistency_rhs(int step, const diff46::StepDiffs& sd) {
    return sd.d[step + 1][0] - sd.d[step + 1][4];
}

// Check whether candidate A_{step+1} satisfies next-step consistency.
inline bool check_next_consistency(Word A_cand, int step,
                                   const State& s0, const State& s2,
                                   const diff46::StepDiffs& sd,
                                   int last_mod_step) {
    int next = step + 1;
    if (next > last_mod_step) return true;

    Word dA_next = sd.d[next][0];
    Word A2_cand = A_cand + dA_next;
    Word dSig0 = Sigma_0(A2_cand) - Sigma_0(A_cand);
    Word dMaj  = Maj(A2_cand, s2[0], s2[1]) - Maj(A_cand, s0[0], s0[1]);
    Word dD    = s2[2] - s0[2];
    return dSig0 + dMaj - dD == consistency_rhs(next, sd);
}

// Precheck: does the initial state pair satisfy consistency at the first step?
inline bool precheck_consistency(const State& X, const State& Y,
                                 int first_mod_step,
                                 const diff46::StepDiffs& sd) {
    return consistency_lhs(X, Y) == consistency_rhs(first_mod_step, sd);
}

// Attempt message modification for the specified step range.
// s0_in, s2_in: states at first_step (BEFORE that step executes).
// W0_out, W2_out: output arrays; indices [first_step..last_step] are set.
// sd: a2 state diffs from the characteristic.
// Returns true on success.
inline bool msg_modify(const State& s0_in, const State& s2_in,
                       Word* W0_out, Word* W2_out,
                       std::mt19937& rng,
                       const diff46::StepDiffs& sd,
                       int first_step = 21, int last_step = 29) {
    State s0 = s0_in, s2 = s2_in;
    for (int step = first_step; step <= last_step; ++step) {
        if (consistency_lhs(s0, s2) != consistency_rhs(step, sd))
            return false;
        Word dSig0 = Sigma_0(s2[0]) - Sigma_0(s0[0]);
        Word dMaj  = Maj(s2[0], s2[1], s2[2]) - Maj(s0[0], s0[1], s0[2]);
        Word dSig1 = Sigma_1(s2[4]) - Sigma_1(s0[4]);
        Word dCh   = Ch(s2[4], s2[5], s2[6]) - Ch(s0[4], s0[5], s0[6]);
        Word dH    = s2[7] - s0[7];
        Word C_A   = dSig0 + dMaj + dSig1 + dCh + dH;
        Word target_dA = sd.d[step + 1][0];
        Word dW = target_dA - C_A;
        Word A_chosen;
        if (step < last_step) {
            bool found = false;
            for (int tries = 0; tries < (1 << 20); ++tries) {
                Word A_cand = rand_word(rng);
                if (check_next_consistency(A_cand, step, s0, s2, sd, last_step)) {
                    A_chosen = A_cand; found = true; break;
                }
            }
            if (!found) return false;
        } else {
            A_chosen = rand_word(rng);
        }
        W0_out[step] = W_to_set_A(s0, step, A_chosen);
        W2_out[step] = W0_out[step] + dW;
        compress(s0, step, W0_out);
        compress(s2, step, W2_out);
    }
    return true;
}

} // namespace attack46
