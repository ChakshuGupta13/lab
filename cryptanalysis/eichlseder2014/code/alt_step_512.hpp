// Alternative SHA-512 description: two independent SHA-1-like recurrences.
//
// Identical structure to alt_step.hpp (SHA-256) but with 64-bit words
// and SHA-512 rotation constants.
//
// Two recurrences:
//   E_i = E_{i-4} + Σ₁(E_{i-1}) + Ch(E_{i-1}, E_{i-2}, E_{i-3}) + A_{i-4} + K_i + W_i
//   A_i = -A_{i-4} + Σ₀(A_{i-1}) + Maj(A_{i-1}, A_{i-2}, A_{i-3}) + E_i
//
// Derived identities:
//   CDE:  E_i = A_i + A_{i-4} - Σ₀(A_{i-1}) - Maj(A_{i-1}, A_{i-2}, A_{i-3})
//   W_i from state: W_i = E_i - E_{i-4} - Σ₁(E_{i-1}) - Ch(E_{i-1}, E_{i-2}, E_{i-3}) - A_{i-4} - K_i

#pragma once

#include "sha512.hpp"
#include <cstdint>

namespace sha512 {

// ---- Alternative state representation ----

struct AltState {
    Word A[4];  // A[0]=A_{i-4}, A[1]=A_{i-3}, A[2]=A_{i-2}, A[3]=A_{i-1}
    Word E[4];  // E[0]=E_{i-4}, E[1]=E_{i-3}, E[2]=E_{i-2}, E[3]=E_{i-1}
};


// ---- Conversion between standard and alternative state ----

inline AltState state_to_alt(const State& s) {
    return {{s[3], s[2], s[1], s[0]},    // d, c, b, a
            {s[7], s[6], s[5], s[4]}};   // h, g, f, e
}

inline State alt_to_state(const AltState& as) {
    return {as.A[3], as.A[2], as.A[1], as.A[0],
            as.E[3], as.E[2], as.E[1], as.E[0]};
}


// ---- Forward step ----

inline void alt_step(AltState& s, int step, const Word* W) {
    Word E_new = s.E[0] + Sigma_1(s.E[3]) + Ch(s.E[3], s.E[2], s.E[1])
               + s.A[0] + K[step] + W[step];
    Word A_new = -s.A[0] + Sigma_0(s.A[3]) + Maj(s.A[3], s.A[2], s.A[1])
               + E_new;

    s.A[0] = s.A[1]; s.A[1] = s.A[2]; s.A[2] = s.A[3]; s.A[3] = A_new;
    s.E[0] = s.E[1]; s.E[1] = s.E[2]; s.E[2] = s.E[3]; s.E[3] = E_new;
}

inline void alt_step_range(AltState& s, int first, int last, const Word* W) {
    for (int i = first; i <= last; ++i)
        alt_step(s, i, W);
}


// ---- Inverse step ----

inline void alt_inverse_step(AltState& s, int step, const Word* W) {
    Word A_old = -s.A[3] + Sigma_0(s.A[2]) + Maj(s.A[2], s.A[1], s.A[0])
               + s.E[3];
    Word E_old = s.E[3] - Sigma_1(s.E[2]) - Ch(s.E[2], s.E[1], s.E[0])
               - A_old - K[step] - W[step];

    s.A[3] = s.A[2]; s.A[2] = s.A[1]; s.A[1] = s.A[0]; s.A[0] = A_old;
    s.E[3] = s.E[2]; s.E[2] = s.E[1]; s.E[1] = s.E[0]; s.E[0] = E_old;
}

inline void alt_inverse_step_range(AltState& s, int first, int last, const Word* W) {
    for (int i = last; i >= first; --i)
        alt_inverse_step(s, i, W);
}


// ---- Derived identities ----

inline Word cde(Word A_i, Word A_im1, Word A_im2, Word A_im3, Word A_im4) {
    return A_i + A_im4 - Sigma_0(A_im1) - Maj(A_im1, A_im2, A_im3);
}

inline Word W_from_state(Word E_i, Word E_im1, Word E_im2, Word E_im3, Word E_im4,
                         Word A_im4, int step) {
    return E_i - E_im4 - Sigma_1(E_im1) - Ch(E_im1, E_im2, E_im3) - A_im4 - K[step];
}

} // namespace sha512
