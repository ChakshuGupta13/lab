// SHA-512 condition propagation: Sigma/sigma functions and step propagation.
// Uses the templated generic infrastructure from propagate.hpp with
// SHA-512-specific rotation constants.

#pragma once

#include "propagate.hpp"
#include "sha512.hpp"

namespace gencond {

// ============================================================
// SHA-512 Sigma / sigma (forward propagation)
// ============================================================

// Big Sigma (state update, pure rotations):
//   Σ₀(x) = rotr28(x) ^ rotr34(x) ^ rotr39(x)
//   Σ₁(x) = rotr14(x) ^ rotr18(x) ^ rotr41(x)

inline WordCond64 wc_Sigma0_512(const WordCond64& x) {
    return wc_xor(wc_xor(wc_rotr(x, 28), wc_rotr(x, 34)), wc_rotr(x, 39));
}

inline WordCond64 wc_Sigma1_512(const WordCond64& x) {
    return wc_xor(wc_xor(wc_rotr(x, 14), wc_rotr(x, 18)), wc_rotr(x, 41));
}

// Small sigma (message expansion, rotations + shift):
//   σ₀(x) = rotr1(x) ^ rotr8(x) ^ shr7(x)
//   σ₁(x) = rotr19(x) ^ rotr61(x) ^ shr6(x)

inline WordCond64 wc_sigma0_512(const WordCond64& x) {
    return wc_xor(wc_xor(wc_rotr(x, 1), wc_rotr(x, 8)), wc_shr(x, 7));
}

inline WordCond64 wc_sigma1_512(const WordCond64& x) {
    return wc_xor(wc_xor(wc_rotr(x, 19), wc_rotr(x, 61)), wc_shr(x, 6));
}

// ============================================================
// SHA-512 Sigma / sigma (bidirectional propagation)
// ============================================================

inline bool wc_Sigma0_512_propagate(WordCond64& x, WordCond64& out) {
    return wc_Sigma_propagate(x, out, 28, 34, 39);
}

inline bool wc_Sigma1_512_propagate(WordCond64& x, WordCond64& out) {
    return wc_Sigma_propagate(x, out, 14, 18, 41);
}

inline bool wc_sigma0_512_propagate(WordCond64& x, WordCond64& out) {
    return wc_sigma_propagate(x, out, 1, 8, 7);
}

inline bool wc_sigma1_512_propagate(WordCond64& x, WordCond64& out) {
    return wc_sigma_propagate(x, out, 19, 61, 6);
}

// ============================================================
// SHA-512 full alt-step propagation (forward + backward)
// ============================================================

struct SHA512AltStepCond {
    WordCond64 A_im4, A_im3, A_im2, A_im1;
    WordCond64 E_im4, E_im3, E_im2, E_im1;
    WordCond64 W_i;
    int step;

    // Forward-propagate E_i.
    WordCond64 propagate_E() const {
        WordCond64 sig1 = wc_Sigma1_512(E_im1);
        WordCond64 ch = wc_Ch(E_im1, E_im2, E_im3);
        WordCond64 k_cond = wc_from_constant(sha512::K[step]);
        WordCond64 t = wc_add(E_im4, sig1);
        t = wc_add(t, ch);
        t = wc_add(t, A_im4);
        t = wc_add(t, k_cond);
        return wc_add(t, W_i);
    }

    // Forward-propagate A_i (requires E_i condition, computed first).
    WordCond64 propagate_A(const WordCond64& E_i) const {
        WordCond64 sig0 = wc_Sigma0_512(A_im1);
        WordCond64 maj = wc_Maj(A_im1, A_im2, A_im3);
        WordCond64 neg_A = wc_neg(A_im4);
        WordCond64 t = wc_add(neg_A, sig0);
        t = wc_add(t, maj);
        return wc_add(t, E_i);
    }

    std::pair<WordCond64, WordCond64> propagate() const {
        WordCond64 E_i = propagate_E();
        WordCond64 A_i = propagate_A(E_i);
        return {A_i, E_i};
    }

    // Backward from E_i.
    bool backward_from_E(WordCond64& E_i) {
        WordCond64 sig1 = wc_Sigma1_512(E_im1);
        WordCond64 ch = wc_Ch(E_im1, E_im2, E_im3);
        WordCond64 k_cond = wc_from_constant(sha512::K[step]);

        WordCond64 t1 = wc_add(E_im4, sig1);
        WordCond64 t2 = wc_add(t1, ch);
        WordCond64 t3 = wc_add(t2, A_im4);
        WordCond64 t4 = wc_add(t3, k_cond);

        if (!wc_add_propagate(t4, W_i, E_i)) return false;
        if (!wc_add_propagate(t3, k_cond, t4)) return false;
        if (!wc_add_propagate(t2, A_im4, t3)) return false;
        if (!wc_add_propagate(t1, ch, t2)) return false;
        if (!wc_add_propagate(E_im4, sig1, t1)) return false;

        if (!wc_Sigma1_512_propagate(E_im1, sig1)) return false;
        if (!wc_Ch_propagate(E_im1, E_im2, E_im3, ch)) return false;

        return true;
    }

    // Backward from A_i.
    bool backward_from_A(WordCond64& A_i, WordCond64& E_i) {
        WordCond64 sig0 = wc_Sigma0_512(A_im1);
        WordCond64 maj = wc_Maj(A_im1, A_im2, A_im3);
        WordCond64 neg_A = wc_neg(A_im4);

        WordCond64 t1 = wc_add(neg_A, sig0);
        WordCond64 t2 = wc_add(t1, maj);

        if (!wc_add_propagate(t2, E_i, A_i)) return false;
        if (!wc_add_propagate(t1, maj, t2)) return false;
        if (!wc_add_propagate(neg_A, sig0, t1)) return false;

        if (!wc_neg_propagate(A_im4, neg_A)) return false;
        if (!wc_Sigma0_512_propagate(A_im1, sig0)) return false;
        if (!wc_Maj_propagate(A_im1, A_im2, A_im3, maj)) return false;

        return true;
    }
};

// ============================================================
// SHA-512 message expansion propagation
// ============================================================

struct SHA512MsgExpCond {
    WordCond64 W_im2, W_im7, W_im15, W_im16;

    WordCond64 propagate() const {
        WordCond64 s1 = wc_sigma1_512(W_im2);
        WordCond64 s0 = wc_sigma0_512(W_im15);
        WordCond64 t = wc_add(s1, W_im7);
        t = wc_add(t, s0);
        return wc_add(t, W_im16);
    }

    bool backward(WordCond64& W_i) {
        WordCond64 s1 = wc_sigma1_512(W_im2);
        WordCond64 s0 = wc_sigma0_512(W_im15);

        WordCond64 t1 = wc_add(s1, W_im7);
        WordCond64 t2 = wc_add(t1, s0);

        if (!wc_add_propagate(t2, W_im16, W_i)) return false;
        if (!wc_add_propagate(t1, s0, t2)) return false;
        if (!wc_add_propagate(s1, W_im7, t1)) return false;

        if (!wc_sigma1_512_propagate(W_im2, s1)) return false;
        if (!wc_sigma0_512_propagate(W_im15, s0)) return false;

        return true;
    }
};

} // namespace gencond
