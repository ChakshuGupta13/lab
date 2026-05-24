// SHA-512 common primitives for collision / differential attacks
// Header-only. Covers: types, constants, step functions (forward + inverse),
// message expansion, register-targeting helpers, and IVs for all truncated
// variants (SHA-384, SHA-512/224, SHA-512/256).

#pragma once

#include <array>
#include <cstdint>
#include <cstdio>
#include <random>

namespace sha512 {

using Word  = uint64_t;
using State = std::array<Word, 8>; // a, b, c, d, e, f, g, h

// --- Bitwise primitives ---

constexpr Word rotr(Word x, unsigned n) { return (x >> n) | (x << (64 - n)); }

// Small sigma (message expansion)
constexpr Word sigma_0(Word x) { return rotr(x, 1)  ^ rotr(x, 8)  ^ (x >> 7);  }
constexpr Word sigma_1(Word x) { return rotr(x, 19) ^ rotr(x, 61) ^ (x >> 6);  }

// Big Sigma (state update)
constexpr Word Sigma_0(Word x) { return rotr(x, 28) ^ rotr(x, 34) ^ rotr(x, 39); }
constexpr Word Sigma_1(Word x) { return rotr(x, 14) ^ rotr(x, 18) ^ rotr(x, 41); }

// Boolean functions
constexpr Word Ch (Word x, Word y, Word z) { return (x & y) ^ (~x & z); }
constexpr Word Maj(Word x, Word y, Word z) { return (x & y) ^ (x & z) ^ (y & z); }

// --- Constants ---

constexpr std::array<Word, 80> K = {
    0x428a2f98d728ae22ULL, 0x7137449123ef65cdULL, 0xb5c0fbcfec4d3b2fULL, 0xe9b5dba58189dbbcULL,
    0x3956c25bf348b538ULL, 0x59f111f1b605d019ULL, 0x923f82a4af194f9bULL, 0xab1c5ed5da6d8118ULL,
    0xd807aa98a3030242ULL, 0x12835b0145706fbeULL, 0x243185be4ee4b28cULL, 0x550c7dc3d5ffb4e2ULL,
    0x72be5d74f27b896fULL, 0x80deb1fe3b1696b1ULL, 0x9bdc06a725c71235ULL, 0xc19bf174cf692694ULL,
    0xe49b69c19ef14ad2ULL, 0xefbe4786384f25e3ULL, 0x0fc19dc68b8cd5b5ULL, 0x240ca1cc77ac9c65ULL,
    0x2de92c6f592b0275ULL, 0x4a7484aa6ea6e483ULL, 0x5cb0a9dcbd41fbd4ULL, 0x76f988da831153b5ULL,
    0x983e5152ee66dfabULL, 0xa831c66d2db43210ULL, 0xb00327c898fb213fULL, 0xbf597fc7beef0ee4ULL,
    0xc6e00bf33da88fc2ULL, 0xd5a79147930aa725ULL, 0x06ca6351e003826fULL, 0x142929670a0e6e70ULL,
    0x27b70a8546d22ffcULL, 0x2e1b21385c26c926ULL, 0x4d2c6dfc5ac42aedULL, 0x53380d139d95b3dfULL,
    0x650a73548baf63deULL, 0x766a0abb3c77b2a8ULL, 0x81c2c92e47edaee6ULL, 0x92722c851482353bULL,
    0xa2bfe8a14cf10364ULL, 0xa81a664bbc423001ULL, 0xc24b8b70d0f89791ULL, 0xc76c51a30654be30ULL,
    0xd192e819d6ef5218ULL, 0xd69906245565a910ULL, 0xf40e35855771202aULL, 0x106aa07032bbd1b8ULL,
    0x19a4c116b8d2d0c8ULL, 0x1e376c085141ab53ULL, 0x2748774cdf8eeb99ULL, 0x34b0bcb5e19b48a8ULL,
    0x391c0cb3c5c95a63ULL, 0x4ed8aa4ae3418acbULL, 0x5b9cca4f7763e373ULL, 0x682e6ff3d6b2b8a3ULL,
    0x748f82ee5defb2fcULL, 0x78a5636f43172f60ULL, 0x84c87814a1f0ab72ULL, 0x8cc702081a6439ecULL,
    0x90befffa23631e28ULL, 0xa4506cebde82bde9ULL, 0xbef9a3f7b2c67915ULL, 0xc67178f2e372532bULL,
    0xca273eceea26619cULL, 0xd186b8c721c0c207ULL, 0xeada7dd6cde0eb1eULL, 0xf57d4f7fee6ed178ULL,
    0x06f067aa72176fbaULL, 0x0a637dc5a2c898a6ULL, 0x113f9804bef90daeULL, 0x1b710b35131c471bULL,
    0x28db77f523047d84ULL, 0x32caab7b40c72493ULL, 0x3c9ebe0a15c9bebcULL, 0x431d67c49c100d4cULL,
    0x4cc5d4becb3e42b6ULL, 0x597f299cfc657e2aULL, 0x5fcb6fab3ad6faecULL, 0x6c44198c4a475817ULL,
};

// --- Initial Values ---

// SHA-512 (FIPS 180-4, Section 5.3.5)
constexpr State IV_512 = {
    0x6a09e667f3bcc908ULL, 0xbb67ae8584caa73bULL,
    0x3c6ef372fe94f82bULL, 0xa54ff53a5f1d36f1ULL,
    0x510e527fade682d1ULL, 0x9b05688c2b3e6c1fULL,
    0x1f83d9abfb41bd6bULL, 0x5be0cd19137e2179ULL,
};

// SHA-384 (FIPS 180-4, Section 5.3.4)
constexpr State IV_384 = {
    0xcbbb9d5dc1059ed8ULL, 0x629a292a367cd507ULL,
    0x9159015a3070dd17ULL, 0x152fecd8f70e5939ULL,
    0x67332667ffc00b31ULL, 0x8eb44a8768581511ULL,
    0xdb0c2e0d64f98fa7ULL, 0x47b5481dbefa4fa4ULL,
};

// SHA-512/224 (FIPS 180-4, Section 5.3.6.1)
constexpr State IV_512_224 = {
    0x8c3d37c819544da2ULL, 0x73e1996689dcd4d6ULL,
    0x1dfab7ae32ff9c82ULL, 0x679dd514582f9fcfULL,
    0x0f6d2b697bd44da8ULL, 0x77e36f7304c48942ULL,
    0x3f9d85a86a1d36c8ULL, 0x1112e6ad91d692a1ULL,
};

// SHA-512/256 (FIPS 180-4, Section 5.3.6.2)
constexpr State IV_512_256 = {
    0x22312194fc2bf72cULL, 0x9f555fa3c84c64c2ULL,
    0x2393b86b6f53b151ULL, 0x963877195940eabdULL,
    0x96283ee2a88effe3ULL, 0xbe5e1e2553863992ULL,
    0x2b0199fc2c85b8aaULL, 0x0eb72ddc81c52ca2ULL,
};

// --- Forward compression step ---

inline void compress(State& s, int step, const Word* W) {
    Word t1 = s[7] + Sigma_1(s[4]) + Ch(s[4], s[5], s[6]) + K[step] + W[step];
    Word t2 = Sigma_0(s[0]) + Maj(s[0], s[1], s[2]);
    s[7] = s[6]; s[6] = s[5]; s[5] = s[4]; s[4] = s[3] + t1;
    s[3] = s[2]; s[2] = s[1]; s[1] = s[0]; s[0] = t1 + t2;
}

inline void compress_range(State& s, int first, int last, const Word* W) {
    for (int i = first; i <= last; ++i)
        compress(s, i, W);
}

// --- Inverse compression step ---

inline void inverse_compress(State& s, int step, const Word* W) {
    Word a_prev = s[1];
    Word b_prev = s[2];
    Word c_prev = s[3];
    Word e_prev = s[5];
    Word f_prev = s[6];
    Word g_prev = s[7];

    Word t2 = Sigma_0(a_prev) + Maj(a_prev, b_prev, c_prev);
    Word t1 = s[0] - t2;
    Word d_prev = s[4] - t1;
    Word h_prev = t1 - Sigma_1(e_prev) - Ch(e_prev, f_prev, g_prev) - K[step] - W[step];

    s[0] = a_prev; s[1] = b_prev; s[2] = c_prev; s[3] = d_prev;
    s[4] = e_prev; s[5] = f_prev; s[6] = g_prev; s[7] = h_prev;
}

inline void inverse_compress_range(State& s, int first_step, int last_step, const Word* W) {
    for (int i = last_step; i >= first_step; --i)
        inverse_compress(s, i, W);
}

// --- Message expansion ---

constexpr Word msg_expand(int i, const Word* W) {
    return sigma_1(W[i - 2]) + W[i - 7] + sigma_0(W[i - 15]) + W[i - 16];
}

// --- Register-targeting helpers ---

inline Word W_to_set_A(const State& s, int step, Word a) {
    return a - Sigma_0(s[0]) - Maj(s[0], s[1], s[2])
             - Sigma_1(s[4]) - Ch(s[4], s[5], s[6]) - s[7] - K[step];
}

inline Word W_to_set_E(const State& s, int step, Word e) {
    return e - s[3] - Sigma_1(s[4]) - Ch(s[4], s[5], s[6]) - s[7] - K[step];
}

// --- Davies-Meyer feed-forward ---

inline State feed_forward(const State& final_state, const State& iv) {
    State out;
    for (int i = 0; i < 8; ++i)
        out[i] = final_state[i] + iv[i];
    return out;
}

// --- Full compression function (expand + compress + feed-forward) ---

inline State sha512_compress(const State& iv, const Word* M, int num_steps) {
    std::array<Word, 80> W{};
    for (int i = 0; i < 16; ++i) W[i] = M[i];
    for (int i = 16; i < num_steps; ++i)
        W[i] = sigma_1(W[i-2]) + W[i-7] + sigma_0(W[i-15]) + W[i-16];

    State s = iv;
    compress_range(s, 0, num_steps - 1, W.data());
    return feed_forward(s, iv);
}

// --- Utility ---

inline Word rand_word(std::mt19937_64& rng) {
    return std::uniform_int_distribution<uint64_t>{}(rng);
}

inline void print_state(const State& s) {
    for (int i = 0; i < 8; ++i)
        std::printf("  %016llx\n", (unsigned long long)s[i]);
}

inline void print_words(const Word* W, int n) {
    for (int i = 0; i < n; ++i) {
        std::printf("0x%016llx, ", (unsigned long long)W[i]);
        if ((i + 1) % 4 == 0) std::printf("\n");
    }
    if (n % 4 != 0) std::printf("\n");
}

} // namespace sha512
