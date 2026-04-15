// Probabilistic 24-step collision attack on SHA-256
// Implements Section 7 of Sanadhya & Sarkar, "New Collision Attacks Against Up To 24-step SHA-2" (2008)
// Uses Column I of the 9-step SS local collision (Steps 10-18, u=1)
// Includes the guess-then-determine algorithm from Appendix B (SHA-256)

#include "sha256.hpp"
#include <chrono>

using namespace sha256;

// C_i from Eq. 17
Word C_calc(int i, const Word* a, const Word* e) {
    return e[i + 5] - Sigma_1(e[i + 4]) - Ch(e[i + 4], e[i + 3], e[i + 2])
         - 2 * a[i + 1] - K[i + 5] + Sigma_0(a[i]);
}

// Phi_i from Eq. 17
Word phi_calc(const State& s, int i) {
    return Sigma_0(s[0]) + Maj(s[0], s[1], s[2])
         + Sigma_1(s[4]) + Ch(s[4], s[5], s[6]) + s[7] + K[i + 1];
}



// Guess-then-determine for D = -W1 + sigma_0(W1), Appendix B.1
// Returns true if solution found, writes result to W1_out
bool guess_then_determine(Word D, Word& W1_out) {
    for (Word lo = 0; lo <= 0x7FFF; ++lo) {
        Word W1 = lo;
        Word X = D + W1;
        Word Y = (W1 >> 3) ^ (W1 >> 7);
        Word bits_25_18 = (X ^ Y) & 0xFF;
        W1 |= rotr(bits_25_18, 14);

        Word saved1 = W1;
        for (Word c0 = 0; c0 < 2; ++c0) {
            W1 = saved1;
            X = (D >> 19) + (bits_25_18 >> 1) + c0;
            Y = (W1 >> 5) ^ (bits_25_18 >> 4);
            Word bits_29_26 = (X ^ Y) & 0xF;
            W1 |= rotr(bits_29_26, 6);

            Word saved2 = W1;
            for (Word c1 = 0; c1 < 2; ++c1) {
                W1 = saved2;
                X = (D >> 23) + (bits_25_18 >> 5) + c1;
                Y = (W1 >> 9) ^ bits_29_26;
                W1 |= rotr(((X ^ Y) & 0x3), 2);

                Word saved3 = W1;
                for (Word c2 = 0; c2 < 2; ++c2) {
                    W1 = saved3;
                    X = (D >> 8) + (W1 >> 8) + c2;
                    Y = (W1 >> 11) ^ bits_29_26;
                    W1 |= rotr(((X ^ Y) & 0x7), 17);

                    if (D == sigma_0(W1) - W1) {
                        W1_out = W1;
                        return true;
                    }
                }
            }
        }
    }
    return false;
}



int main() {
    std::random_device rd;
    std::mt19937 rng(rd());

    // Constants from Table 4 (24-step, i=10)
    constexpr Word del_1 = 0x00006000;
    constexpr Word del_2 = 0xff006001;
    constexpr Word alpha = 0x32b308b2;
    constexpr Word lamda = 0x051f9f7f;
    constexpr Word gamma = 0x98e3923b;
    constexpr Word mu    = 0xfbe05f81;
    constexpr Word u     = 1;

    // Fixed register values from Table 3 (i=10)
    Word a[24]{}, e[24]{}, phi[24]{};
    a[8] = alpha; a[9] = alpha; a[10] = Word(-1); a[11] = ~alpha; a[12] = ~alpha;
    e[8]  = gamma; e[9] = gamma + 1; e[10] = Word(-1); e[11] = mu;
    e[12] = lamda; e[13] = lamda - 1; e[14] = Word(-1); e[15] = Word(-1);
    e[16] = Word(-1) - u;

    // Differential schedule
    Word dW[24]{};
    dW[10] = 1; dW[11] = Word(-1); dW[12] = del_1; dW[13] = del_2;
    dW[17] = 1; dW[18] = Word(-1);

    std::printf("Searching for 24-step SHA-256 collision...\n");
    auto t_start = std::chrono::steady_clock::now();

    Word W[24]{};
    Word C[24]{};
    State s{};
    uint64_t trials = 0;

    do {
        // Inner loop: find W0, W1 via guess-then-determine (Section 7, steps 1-5)
        bool w1_found = false;
        do {
            s = IV;
            W[0] = rand_word(rng);
            a[2] = rand_word(rng);
            a[3] = rand_word(rng);

            compress(s, 0, W);
            a[0] = s[0]; e[0] = s[4];
            phi[0] = phi_calc(s, 0);

            // Determine a4..a7 from CDE (reverse direction)
            a[7] = e[11] - a[11] + Sigma_0(a[10]) + Maj(a[10], a[9], a[8]);
            a[6] = e[10] - a[10] + Sigma_0(a[9])  + Maj(a[9], a[8], a[7]);
            a[5] = e[9]  - a[9]  + Sigma_0(a[8])  + Maj(a[8], a[7], a[6]);
            a[4] = e[8]  - a[8]  + Sigma_0(a[7])  + Maj(a[7], a[6], a[5]);

            // e6, e7 via CDE
            e[7] = a[7] + a[3] - Sigma_0(a[6]) - Maj(a[6], a[5], a[4]);
            e[6] = a[6] + a[2] - Sigma_0(a[5]) - Maj(a[5], a[4], a[3]);

            C[4] = C_calc(4, a, e);

            // Fixed values for W14, W16 (Eqs. 14-15)
            W[16] = e[16] - Sigma_1(e[15]) - Ch(e[15], e[14], e[13]) - a[12] - e[12] - K[16];
            W[14] = e[14] - Sigma_1(e[13]) - Ch(e[13], e[12], e[11]) - a[10] - e[10] - K[14];

            // D from Eq. 19
            Word D = W[16] - (sigma_1(W[14]) + C[4] + Maj(a[4], a[3], a[2]) - phi[0] + W[0]);

            // Guess-then-determine: solve D = -W1 + sigma_0(W1)
            w1_found = guess_then_determine(D, W[1]);
            ++trials;
        } while (!w1_found);

        // Steps 6-9 of the procedure in Section 7
        compress(s, 1, W);
        a[1] = s[0]; e[1] = s[4];
        phi[1] = phi_calc(s, 1);

        W[2] = W_to_set_A(s, 2, a[2]);
        compress(s, 2, W);
        a[2] = s[0]; e[2] = s[4];
        phi[2] = phi_calc(s, 2);

        W[3] = W_to_set_A(s, 3, a[3]);
        compress(s, 3, W);
        a[3] = s[0]; e[3] = s[4];

        W[15] = e[15] - Sigma_1(e[14]) - Ch(e[14], e[13], e[12]) - a[11] - e[11] - K[15];
        C[5] = C_calc(5, a, e);

        // Compute W17, W18 (Eq. 18)
        W[17] = sigma_1(W[15]) + C[5] - W[2] + Maj(a[5], a[4], a[3]) - phi[1] + sigma_0(W[2]) + W[1];
        C[6] = C_calc(6, a, e);
        W[18] = sigma_1(W[16]) + C[6] - W[3] + Maj(a[6], a[5], a[4]) - phi[2] + sigma_0(W[3]) + W[2];

        // Step 11: check sigma_1 conditions
    } while ((sigma_1(W[17] + 1) - sigma_1(W[17])) != Word(-del_1)
          || (sigma_1(W[18] - 1) - sigma_1(W[18])) != Word(-del_2));

    auto t_end = std::chrono::steady_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count();
    std::printf("Found after %llu trials (%.2f s)\n\n", trials, ms / 1000.0);

    // Build remaining message words (steps 4-13)
    for (int i = 4; i < 13; ++i) {
        W[i] = W_to_set_A(s, i, a[i]);
        compress(s, i, W);
    }
    W[13] = e[13] - Sigma_1(e[12]) - Ch(e[12], e[11], e[10]) - a[9] - e[9] - K[13];
    compress_range(s, 13, 15, W);

    // Verify message expansion consistency
    std::printf("Message expansion check:\n");
    bool exp_ok = true;
    for (int i = 16; i <= 18; ++i) {
        Word expected = msg_expand(i, W);
        if (expected != W[i]) {
            std::printf("  W[%d]: MISMATCH (got %08x, expected %08x)\n", i, W[i], expected);
            exp_ok = false;
        }
    }
    if (exp_ok) std::printf("  W[16..18]: OK\n");

    // Expand W[19..23]
    for (int i = 19; i < 24; ++i)
        W[i] = msg_expand(i, W);
    compress_range(s, 16, 23, W);

    // Build second message
    Word Wd[24]{};
    for (int i = 0; i < 24; ++i)
        Wd[i] = W[i] + dW[i];

    // Verify expansion of W'
    std::printf("W' expansion check:\n");
    bool exp_ok2 = true;
    for (int i = 16; i <= 23; ++i) {
        Word expected = msg_expand(i, Wd);
        if (expected != Wd[i]) {
            std::printf("  W'[%d]: MISMATCH (got %08x, expected %08x)\n", i, Wd[i], expected);
            exp_ok2 = false;
        }
    }
    if (exp_ok2) std::printf("  W'[16..23]: OK\n");

    // Compress second message
    State sd = IV;
    compress_range(sd, 0, 23, Wd);

    // Print messages
    std::printf("\nW:\n");
    print_words(W, 24);
    std::printf("\nW':\n");
    print_words(Wd, 24);

    // Verify collision
    bool collision = (s == sd);
    std::printf("\n24-step collision: %s\n", collision ? "CONFIRMED" : "FAILED");
    return collision ? 0 : 1;
}
