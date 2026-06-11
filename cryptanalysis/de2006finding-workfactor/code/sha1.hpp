// SHA-1 common primitives for collision / differential attacks
// Header-only. Covers: types, constants, step functions (forward + inverse),
// message expansion, and full compression.

#pragma once

#include <array>
#include <cstdint>
#include <cstdio>
#include <random>

namespace sha1 {

using Word  = uint32_t;
using State = std::array<Word, 5>; // A, B, C, D, E

// --- Bitwise primitives ---

constexpr Word rotl(Word x, unsigned n) { return (x << n) | (x >> (32 - n)); }

// Round functions
constexpr Word f_IF (Word b, Word c, Word d) { return (b & c) ^ (~b & d); }       // Ch
constexpr Word f_XOR(Word b, Word c, Word d) { return b ^ c ^ d; }                // Parity
constexpr Word f_MAJ(Word b, Word c, Word d) { return (b & c) ^ (b & d) ^ (c & d); } // Maj

inline Word f(int step, Word b, Word c, Word d) {
    if (step < 20) return f_IF(b, c, d);
    if (step < 40) return f_XOR(b, c, d);
    if (step < 60) return f_MAJ(b, c, d);
    return f_XOR(b, c, d);
}

// --- Constants ---

constexpr Word K_val(int step) {
    if (step < 20) return 0x5A827999;
    if (step < 40) return 0x6ED9EBA1;
    if (step < 60) return 0x8F1BBCDC;
    return 0xCA62C1D6;
}

constexpr std::array<Word, 80> K = [] {
    std::array<Word, 80> k{};
    for (int i = 0; i < 80; ++i) k[i] = K_val(i);
    return k;
}();

constexpr State IV = {
    0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0
};

// --- Message expansion ---
// W[0..15] = M[0..15] (input), W[16..79] = expand.

inline void expand_message(Word* W, int steps = 80) {
    for (int i = 16; i < steps; ++i)
        W[i] = rotl(W[i-3] ^ W[i-8] ^ W[i-14] ^ W[i-16], 1);
}

// --- Forward compression step ---
// State = (A, B, C, D, E)
// After one step:
//   A' = rotl(A,5) + f(B,C,D) + E + K[i] + W[i]
//   B' = A
//   C' = rotl(B, 30)
//   D' = C
//   E' = D

inline void compress(State& s, int step, const Word* W) {
    Word temp = rotl(s[0], 5) + f(step, s[1], s[2], s[3]) + s[4] + K[step] + W[step];
    s[4] = s[3];
    s[3] = s[2];
    s[2] = rotl(s[1], 30);
    s[1] = s[0];
    s[0] = temp;
}

inline void compress_range(State& s, int first, int last, const Word* W) {
    for (int i = first; i <= last; ++i)
        compress(s, i, W);
}

// --- Inverse compression step ---
// Given state AFTER step i, recover state BEFORE step i.
// Forward: (A,B,C,D,E) -> (temp, A, rotl(B,30), C, D)
// Inverse:
//   A_prev = s[1]
//   B_prev = rotl(s[2], 2)   (undo rotl(B,30): rotl(x,2) = rotr(x,30))
//   C_prev = s[3]
//   D_prev = s[4]
//   E_prev = s[0] - rotl(A_prev,5) - f(B_prev, C_prev, D_prev) - K[step] - W[step]
//   [where B,C,D in f() are pre-step values]

inline void inverse_compress(State& s, int step, const Word* W) {
    Word A_prev = s[1];
    Word B_prev = rotl(s[2], 2);  // undo rotl(30) by rotating 2 more
    Word C_prev = s[3];
    Word D_prev = s[4];
    Word E_prev = s[0] - rotl(A_prev, 5) - f(step, B_prev, C_prev, D_prev) - K[step] - W[step];

    s[0] = A_prev;
    s[1] = B_prev;
    s[2] = C_prev;
    s[3] = D_prev;
    s[4] = E_prev;
}

inline void inverse_compress_range(State& s, int first_step, int last_step, const Word* W) {
    for (int i = last_step; i >= first_step; --i)
        inverse_compress(s, i, W);
}

// --- Full hash (N steps, with feed-forward) ---

inline State hash_block(const State& iv, const Word* M, int steps) {
    Word W[80];
    for (int i = 0; i < 16; ++i) W[i] = M[i];
    expand_message(W, steps);

    State s = iv;
    compress_range(s, 0, steps - 1, W);

    // Davies-Meyer feed-forward
    for (int i = 0; i < 5; ++i) s[i] += iv[i];
    return s;
}

// Two-block hash (for collision verification)
inline State hash_two_blocks(const State& iv, const Word* M0, const Word* M1, int steps) {
    State h1 = hash_block(iv, M0, steps);
    return hash_block(h1, M1, steps);
}

// --- Utility ---

inline Word rand_word(std::mt19937& rng) {
    return std::uniform_int_distribution<uint32_t>{}(rng);
}

inline void print_state(const State& s) {
    for (int i = 0; i < 5; ++i)
        std::printf("  %08X\n", s[i]);
}

inline void print_words(const Word* W, int n) {
    for (int i = 0; i < n; ++i) {
        std::printf("%08X ", W[i]);
        if ((i + 1) % 4 == 0) std::printf("\n");
    }
    if (n % 4 != 0) std::printf("\n");
}

} // namespace sha1
