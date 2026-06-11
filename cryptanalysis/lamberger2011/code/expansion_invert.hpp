// expansion_invert.hpp — Modular backward substitution for SHA-256 expansion.
//
// Given target W[21..29] (from msg_modify) and free W[16..20], M[14], M[15],
// solve for M[0..13] such that natural expansion of M[0..15] produces exactly
// the given W[16..29].
//
// Cost: 14 iterations, each ~1 sigma + 2 subtracts.  O(1), exact, no
// approximation.
//
// Free parameters (7 words = 224 bits):
//   W[16..20]  — unconstrained (msg_modify starts at step 21)
//   M[14], M[15] — unconstrained (no expansion equation targets them)

#pragma once

#include "sha256.hpp"

namespace attack46 {

using sha256::Word;

// Solve M[0..15] from target expansion words.
//
// W_target[14]: W_target[i - 16] = target W[i] for i in [16..29].
//               Indices 0..4 are free (W[16..20]), indices 5..13 are
//               the msg_modify targets (W[21..29]).
// M14, M15:    free message words.
// M_out[16]:   solved message.  M_out[14] = M14, M_out[15] = M15.
//
// After calling, sha256::msg_expand(M_out) will produce W[16..29]
// matching W_target exactly.
inline void invert_expansion(const Word W_target[14],
                             Word M14, Word M15,
                             Word M_out[16]) {
    M_out[14] = M14;
    M_out[15] = M15;

    // Backward sweep: i from 29 down to 16.
    // At each step, M[i-16] is the single unknown.
    // W[i] = sigma1(W[i-2]) + W[i-7] + sigma0(W[i-15]) + W[i-16]
    // => M[i-16] = W[i] - sigma1(W[i-2]) - W[i-7] - sigma0(M[i-15])
    for (int i = 29; i >= 16; i--) {
        // W[i-2]: either in W_target (if i-2 >= 16) or already-solved M
        Word wi2 = (i - 2 >= 16) ? W_target[i - 2 - 16] : M_out[i - 2];

        // W[i-7]: either in W_target (if i-7 >= 16) or already-solved M
        Word wi7 = (i - 7 >= 16) ? W_target[i - 7 - 16] : M_out[i - 7];

        // W[i-15] = M[i-15]: always in M_out (i-15 ranges 1..14)
        Word mi15 = M_out[i - 15];

        M_out[i - 16] = W_target[i - 16]
                       - sha256::sigma_1(wi2)
                       - wi7
                       - sha256::sigma_0(mi15);
    }
}

// Convenience wrapper: separate target arrays for the two regions.
//   W_free_16_20[5]  — random free words for expansion slots 16..20
//   W_target_21_29[9] — msg_modify targets for slots 21..29
inline void invert_expansion_split(const Word W_free_16_20[5],
                                   const Word W_target_21_29[9],
                                   Word M14, Word M15,
                                   Word M_out[16]) {
    Word W_target[14];
    for (int i = 0; i < 5; i++)  W_target[i]     = W_free_16_20[i];
    for (int i = 0; i < 9; i++)  W_target[i + 5]  = W_target_21_29[i];
    invert_expansion(W_target, M14, M15, M_out);
}

// SIMD version: processes SIMD_WIDTH lanes in parallel.
// Eliminates scalar-per-lane loop + AoS-to-SoA transpose.
// Requires sha256_simd.hpp to be included first (defines u32xN, SIMD_WIDTH).
#ifdef SIMD_WIDTH
inline void invert_expansion_split_simd(const u32xN W_free_16_20[5],
                                        const u32xN W_target_21_29[9],
                                        u32xN M14, u32xN M15,
                                        u32xN M_out[16]) {
    u32xN W[14];
    for (int i = 0; i < 5; i++)  W[i]     = W_free_16_20[i];
    for (int i = 0; i < 9; i++)  W[i + 5] = W_target_21_29[i];

    M_out[14] = M14;
    M_out[15] = M15;

    for (int i = 29; i >= 16; i--) {
        u32xN wi2  = (i - 2 >= 16) ? W[i - 2 - 16] : M_out[i - 2];
        u32xN wi7  = (i - 7 >= 16) ? W[i - 7 - 16] : M_out[i - 7];
        u32xN mi15 = M_out[i - 15];

        M_out[i - 16] = simd_sub(simd_sub(simd_sub(W[i - 16],
                            sha256_simd::sigma_1(wi2)), wi7),
                            sha256_simd::sigma_0(mi15));
    }
}
#endif // SIMD_WIDTH

} // namespace attack46
