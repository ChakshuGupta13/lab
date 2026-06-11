// Carry-chain Markov model for modular addition differentials.
//
// For x uniform in [0, 2^N), x' = x + Δ (mod 2^N):
//   P(carry into bit b) follows a Markov chain:
//     p[0] = 0
//     p[b+1] = (1 + p[b])/2  if Δ[b] = 1
//     p[b+1] = p[b]/2        if Δ[b] = 0
//
//   P(x[b]=v, x'[b]=v') = (1/2) · P(carry_b = v ⊕ v' ⊕ Δ[b])
//
// Applies to any ARX construction (SHA-256, SHA-512, SHA-1, etc.).
// Reference: analysis in Zhang et al. 2026 β-condition cost model.

#pragma once

#include <cstdint>
#include <cmath>

namespace carry_chain {

// Carry-chain model for 32-bit modular addition.
struct CarryChain32 {
    double p[33]; // p[b] = P(carry into bit b), b = 0..32

    // Compute carry probabilities for x' = x + delta, x uniform in [0, 2^32).
    void compute(uint32_t delta) {
        p[0] = 0.0;
        for (int b = 0; b < 32; ++b) {
            if ((delta >> b) & 1)
                p[b+1] = (1.0 + p[b]) / 2.0;
            else
                p[b+1] = p[b] / 2.0;
        }
    }

    // P(x[b]=v, x'[b]=v') where x' = x + delta, x uniform.
    double p_joint(int b, int v, int vp, uint32_t delta) const {
        int delta_b = (delta >> b) & 1;
        int need_carry = v ^ vp ^ delta_b;
        return 0.5 * ((need_carry == 1) ? p[b] : (1.0 - p[b]));
    }

    // P(signed-difference condition c holds at bit b).
    // Conditions: '0' (both 0), '1' (both 1), 'n' (0→1), 'u' (1→0), '=' (any).
    double p_cond(int b, char c, uint32_t delta) const {
        switch (c) {
            case '1': return p_joint(b, 1, 1, delta);
            case '0': return p_joint(b, 0, 0, delta);
            case 'n': return p_joint(b, 0, 1, delta);
            case 'u': return p_joint(b, 1, 0, delta);
            default:  return 1.0;
        }
    }

    // Cost in bits = -log2(P(condition c at bit b)).
    double cost(int b, char c, uint32_t delta) const {
        double pr = p_cond(b, c, delta);
        return (pr > 0) ? -std::log2(pr) : 99.0;
    }
};

// Carry-chain model for 64-bit modular addition (SHA-512).
struct CarryChain64 {
    double p[65]; // p[b] = P(carry into bit b), b = 0..64

    void compute(uint64_t delta) {
        p[0] = 0.0;
        for (int b = 0; b < 64; ++b) {
            if ((delta >> b) & 1)
                p[b+1] = (1.0 + p[b]) / 2.0;
            else
                p[b+1] = p[b] / 2.0;
        }
    }

    double p_joint(int b, int v, int vp, uint64_t delta) const {
        int delta_b = (delta >> b) & 1;
        int need_carry = v ^ vp ^ delta_b;
        return 0.5 * ((need_carry == 1) ? p[b] : (1.0 - p[b]));
    }

    double p_cond(int b, char c, uint64_t delta) const {
        switch (c) {
            case '1': return p_joint(b, 1, 1, delta);
            case '0': return p_joint(b, 0, 0, delta);
            case 'n': return p_joint(b, 0, 1, delta);
            case 'u': return p_joint(b, 1, 0, delta);
            default:  return 1.0;
        }
    }

    double cost(int b, char c, uint64_t delta) const {
        double pr = p_cond(b, c, delta);
        return (pr > 0) ? -std::log2(pr) : 99.0;
    }
};

} // namespace carry_chain
