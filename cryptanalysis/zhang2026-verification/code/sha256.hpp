// SHA-256 common primitives for collision / differential attacks
// Header-only. Covers: types, constants, step functions (forward + inverse),
// message expansion, and register-targeting helpers.

#pragma once

#include <array>
#include <cstdint>
#include <cstdio>
#include <random>

namespace sha256 {

using Word  = uint32_t;
using State = std::array<Word, 8>; // a, b, c, d, e, f, g, h

// --- Bitwise primitives ---

constexpr Word rotr(Word x, unsigned n) { return (x >> n) | (x << (32 - n)); }

// Small sigma (message expansion)
constexpr Word sigma_0(Word x) { return rotr(x, 7)  ^ rotr(x, 18) ^ (x >> 3);  }
constexpr Word sigma_1(Word x) { return rotr(x, 17) ^ rotr(x, 19) ^ (x >> 10); }

// Big Sigma (state update)
constexpr Word Sigma_0(Word x) { return rotr(x, 2)  ^ rotr(x, 13) ^ rotr(x, 22); }
constexpr Word Sigma_1(Word x) { return rotr(x, 6)  ^ rotr(x, 11) ^ rotr(x, 25); }

// Boolean functions
constexpr Word Ch (Word x, Word y, Word z) { return (x & y) ^ (~x & z); }
constexpr Word Maj(Word x, Word y, Word z) { return (x & y) ^ (x & z) ^ (y & z); }

// --- Constants ---

constexpr std::array<Word, 64> K = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
};

constexpr State IV = {
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
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
// Given state AFTER step i, recover state BEFORE step i (requires W[step]).
//
// Forward:  (a,b,c,d,e,f,g,h) -> (t1+t2, a, b, c, d+t1, e, f, g)
// Inverse:  from output state s_out, recover s_in:
//   a_{i-1} = s_out[1],  b_{i-1} = s_out[2],  c_{i-1} = s_out[3]
//   e_{i-1} = s_out[5],  f_{i-1} = s_out[6],  g_{i-1} = s_out[7]
//   t2 = Sigma_0(a_{i-1}) + Maj(a_{i-1}, b_{i-1}, c_{i-1})
//   t1 = s_out[0] - t2
//   d_{i-1} = s_out[4] - t1
//   h_{i-1} = t1 - Sigma_1(e_{i-1}) - Ch(e_{i-1}, f_{i-1}, g_{i-1}) - K[step] - W[step]

inline void inverse_compress(State& s, int step, const Word* W) {
    // s is the state AFTER step; we recover the state BEFORE step
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

// --- Register-targeting helpers (Sanadhya-Sarkar, Table 15) ---
// Return W_i that forces register a_i or e_i to a desired value.

inline Word W_to_set_A(const State& s, int step, Word a) {
    return a - Sigma_0(s[0]) - Maj(s[0], s[1], s[2])
             - Sigma_1(s[4]) - Ch(s[4], s[5], s[6]) - s[7] - K[step];
}

inline Word W_to_set_E(const State& s, int step, Word e) {
    return e - s[3] - Sigma_1(s[4]) - Ch(s[4], s[5], s[6]) - s[7] - K[step];
}

// --- Utility ---

inline Word rand_word(std::mt19937& rng) {
    return std::uniform_int_distribution<uint32_t>{}(rng);
}

inline void print_state(const State& s) {
    for (int i = 0; i < 8; ++i)
        std::printf("  %08x\n", s[i]);
}

inline void print_words(const Word* W, int n) {
    for (int i = 0; i < n; ++i) {
        std::printf("0x%08x, ", W[i]);
        if ((i + 1) % 8 == 0) std::printf("\n");
    }
    if (n % 8 != 0) std::printf("\n");
}

// Davies-Meyer feed-forward: output = final_state + IV
inline State feed_forward(const State& final_state, const State& iv) {
    State out;
    for (int i = 0; i < 8; ++i)
        out[i] = final_state[i] + iv[i];
    return out;
}

} // namespace sha256
