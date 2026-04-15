// Deterministic 22-step collision attack on SHA-256
// Implements Table 15 of Sanadhya & Sarkar, "New Collision Attacks Against Up To 24-step SHA-2" (2008)
// Uses Column II of the 9-step SS local collision (Steps 7-15)

#include "sha256.hpp"

using namespace sha256;

int main() {
    // --- RNG setup ---
    std::random_device rd;
    std::mt19937 rng(rd());

    // --- First message: 16 free words, of which W0, W1, W14, W15 are random ---
    std::array<Word, 22> W{};
    W[0]  = rand_word(rng);
    W[1]  = rand_word(rng);
    W[14] = rand_word(rng);
    W[15] = rand_word(rng);

    // --- Differential schedule (Column II, local collision Steps 7-15) ---
    std::array<Word, 16> dW{};
    dW[7]  =  1;          // perturbation
    dW[15] = static_cast<Word>(-1); // cancellation

    // DELTA = sigma_1(W15) - sigma_1(W15 - 1)  [Section 4.1]
    const Word DELTA = sigma_1(W[15]) - sigma_1(W[15] - 1);
    std::printf("DELTA:\t\t\t\t%08x\n\n", DELTA);

    // --- Build first message following Table 15 ---
    State s = IV;

    // Steps 0-1: free words, just run compression
    compress(s, 0, W.data());
    compress(s, 1, W.data());

    // Step 2: a2 = DELTA - 1 + MAJ(-1, -2, DELTA - 3)
    W[2] = W_to_set_A(s, 2, DELTA - 1 + Maj(Word(-1), Word(-2), DELTA - 3));
    compress(s, 2, W.data());

    // Step 3: a3 = DELTA - 3
    W[3] = W_to_set_A(s, 3, DELTA - 3);
    compress(s, 3, W.data());

    // Steps 4-7: a = {-2, -1, -1, -1}
    W[4] = W_to_set_A(s, 4, Word(-2));  compress(s, 4, W.data());
    W[5] = W_to_set_A(s, 5, Word(-1));  compress(s, 5, W.data());
    W[6] = W_to_set_A(s, 6, Word(-1));  compress(s, 6, W.data());
    W[7] = W_to_set_A(s, 7, Word(-1));  compress(s, 7, W.data());

    // Compute dW[8] using Eq. 7: after step 7, s[4]=e7, s[5]=f7, s[6]=g7
    dW[8] = Word(-1) - (Ch(s[4] + 1, s[5], s[6]) - Ch(s[4], s[5], s[6]))
                      - (Sigma_1(s[4] + 1) - Sigma_1(s[4]));

    // Steps 8-9: a = {0, 0}
    W[8] = W_to_set_A(s, 8, 0);  compress(s, 8, W.data());
    W[9] = W_to_set_A(s, 9, 0);  compress(s, 9, W.data());

    // Compute dW[10] using Eq. 8: after step 9, s[4]=e9, s[5]=f9, s[6]=g9
    dW[10] = Sigma_1(s[4]) - Ch(s[4] - 1, s[5] - 1, s[6] + 1)
           + Ch(s[4], s[5], s[6]) - Sigma_1(s[4] - 1);

    // Steps 10-13: e = -1
    W[10] = W_to_set_E(s, 10, Word(-1));  compress(s, 10, W.data());
    W[11] = W_to_set_E(s, 11, Word(-1));  compress(s, 11, W.data());
    W[12] = W_to_set_E(s, 12, Word(-1));  compress(s, 12, W.data());
    W[13] = W_to_set_E(s, 13, Word(-1));  compress(s, 13, W.data());

    // --- Build second message ---
    std::array<Word, 22> Wd{};
    std::printf("W\t\tdW\t\tW'\n");
    for (int i = 0; i < 16; ++i) {
        Wd[i] = W[i] + dW[i];
        std::printf("%08x\t%08x\t%08x\n", W[i], dW[i], Wd[i]);
    }

    // --- Finish compression for first message (steps 14-15, then expand 16-21) ---
    compress(s, 14, W.data());
    compress(s, 15, W.data());

    for (int i = 16; i < 22; ++i) {
        W[i] = msg_expand(i, W.data());
        Wd[i] = W[i];  // zero differential by the proof in Section 4
    }
    for (int i = 16; i < 22; ++i)
        compress(s, i, W.data());

    // --- Compress second message ---
    State sd = IV;
    for (int i = 0; i < 22; ++i)
        compress(sd, i, Wd.data());

    // --- Print and verify ---
    std::printf("\n22-step registers for W:\n");
    for (int i = 0; i < 8; ++i) std::printf("  %08x\n", s[i]);

    std::printf("\n22-step registers for W':\n");
    for (int i = 0; i < 8; ++i) std::printf("  %08x\n", sd[i]);

    bool collision = (s == sd);
    std::printf("\nCollision: %s\n", collision ? "CONFIRMED" : "FAILED");
    return collision ? 0 : 1;
}
