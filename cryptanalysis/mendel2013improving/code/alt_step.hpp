// Alternative SHA-256 description: two independent SHA-1-like recurrences.
//
// From Mendel-Nad-Schläffer 2011 (Eq. 2), also used by Lamberger-Mendel 2011
// and implicitly by Sanadhya-Sarkar 2008 (CDE).
//
// Standard SHA-256 updates 8 registers (a,b,c,d,e,f,g,h) per step.
// The alternative description observes that only registers a and e are
// freshly computed; the rest are shifts (b=a_{i-1}, c=a_{i-2}, etc.).
//
// State: (A[0..3], E[0..3]) where
//   A[0] = oldest (A_{step-4}),  A[3] = newest (A_{step-1})
//   E[0] = oldest (E_{step-4}),  E[3] = newest (E_{step-1})
//
// Two recurrences:
//   E_i = E_{i-4} + Σ₁(E_{i-1}) + Ch(E_{i-1}, E_{i-2}, E_{i-3}) + A_{i-4} + K_i + W_i
//   A_i = -A_{i-4} + Σ₀(A_{i-1}) + Maj(A_{i-1}, A_{i-2}, A_{i-3}) + E_i
//
// Derived identities (rearrangements):
//   CDE:  E_i = A_i + A_{i-4} - Σ₀(A_{i-1}) - Maj(A_{i-1}, A_{i-2}, A_{i-3})
//   W_i from state: W_i = E_i - E_{i-4} - Σ₁(E_{i-1}) - Ch(E_{i-1}, E_{i-2}, E_{i-3}) - A_{i-4} - K_i
//
// Mapping to standard state (a,b,c,d,e,f,g,h):
//   a = A[3], b = A[2], c = A[1], d = A[0]
//   e = E[3], f = E[2], g = E[1], h = E[0]

#pragma once

#include "sha256.hpp"
#include <cstdint>

namespace sha256 {

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

    // Shift: discard oldest, append newest
    s.A[0] = s.A[1]; s.A[1] = s.A[2]; s.A[2] = s.A[3]; s.A[3] = A_new;
    s.E[0] = s.E[1]; s.E[1] = s.E[2]; s.E[2] = s.E[3]; s.E[3] = E_new;
}

inline void alt_step_range(AltState& s, int first, int last, const Word* W) {
    for (int i = first; i <= last; ++i)
        alt_step(s, i, W);
}


// ---- Inverse step ----
// Given state AFTER step i, recover state BEFORE step i.

inline void alt_inverse_step(AltState& s, int step, const Word* W) {
    // s contains the state after step i:
    //   A[3]=A_i, A[2]=A_{i-1}, A[1]=A_{i-2}, A[0]=A_{i-3}
    //   E[3]=E_i, E[2]=E_{i-1}, E[1]=E_{i-2}, E[0]=E_{i-3}
    //
    // Recover A_{i-4} from the A equation:
    //   A_i = -A_{i-4} + Σ₀(A_{i-1}) + Maj(A_{i-1}, A_{i-2}, A_{i-3}) + E_i
    //   → A_{i-4} = -A_i + Σ₀(A_{i-1}) + Maj(A_{i-1}, A_{i-2}, A_{i-3}) + E_i
    Word A_old = -s.A[3] + Sigma_0(s.A[2]) + Maj(s.A[2], s.A[1], s.A[0])
               + s.E[3];

    // Recover E_{i-4} from the E equation:
    //   E_i = E_{i-4} + Σ₁(E_{i-1}) + Ch(E_{i-1}, E_{i-2}, E_{i-3}) + A_{i-4} + K_i + W_i
    //   → E_{i-4} = E_i - Σ₁(E_{i-1}) - Ch(E_{i-1}, E_{i-2}, E_{i-3}) - A_{i-4} - K_i - W_i
    Word E_old = s.E[3] - Sigma_1(s.E[2]) - Ch(s.E[2], s.E[1], s.E[0])
               - A_old - K[step] - W[step];

    // Unshift: prepend oldest, discard newest
    s.A[3] = s.A[2]; s.A[2] = s.A[1]; s.A[1] = s.A[0]; s.A[0] = A_old;
    s.E[3] = s.E[2]; s.E[2] = s.E[1]; s.E[1] = s.E[0]; s.E[0] = E_old;
}

inline void alt_inverse_step_range(AltState& s, int first, int last, const Word* W) {
    for (int i = last; i >= first; --i)
        alt_inverse_step(s, i, W);
}


// ---- Derived identities (useful across papers) ----

// Cross Dependence Equation (Sanadhya-Sarkar 2008, Eq. 1):
//   E_i = A_i + A_{i-4} - Σ₀(A_{i-1}) - Maj(A_{i-1}, A_{i-2}, A_{i-3})
//
// Given A values at five positions (i, i-1, i-2, i-3, i-4), compute E_i.
inline Word cde(Word A_i, Word A_im1, Word A_im2, Word A_im3, Word A_im4) {
    return A_i + A_im4 - Sigma_0(A_im1) - Maj(A_im1, A_im2, A_im3);
}

// W derivation from alt-state values (Lamberger-Mendel 2011, message modification):
//   W_i = E_i - E_{i-4} - Σ₁(E_{i-1}) - Ch(E_{i-1}, E_{i-2}, E_{i-3}) - A_{i-4} - K_i
//
// Requires values from both before AND after the step.
// Caller provides the individual register values.
inline Word W_from_state(Word E_i, Word E_im1, Word E_im2, Word E_im3, Word E_im4,
                         Word A_im4, int step) {
    return E_i - E_im4 - Sigma_1(E_im1) - Ch(E_im1, E_im2, E_im3) - A_im4 - K[step];
}

} // namespace sha256
