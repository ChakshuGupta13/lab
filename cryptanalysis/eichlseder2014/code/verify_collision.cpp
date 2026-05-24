// Verify the 38-step SHA-512 semi-free-start collision from
// Eichlseder, Mendel & Schläffer, "Branching Heuristics in Differential
// Collision Search with Applications to SHA-512" (FSE 2014), Table 2.
//
// Self-contained: includes a minimal SHA-512 implementation (38-step only).
// Build and usage: see README.md in this directory.

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>

using Word  = uint64_t;
using State = std::array<Word, 8>;  // a, b, c, d, e, f, g, h

// --- SHA-512 primitives ---

constexpr Word rotr(Word x, unsigned n) { return (x >> n) | (x << (64 - n)); }

constexpr Word sigma_0(Word x) { return rotr(x, 1)  ^ rotr(x, 8)  ^ (x >> 7);  }
constexpr Word sigma_1(Word x) { return rotr(x, 19) ^ rotr(x, 61) ^ (x >> 6);  }
constexpr Word Sigma_0(Word x) { return rotr(x, 28) ^ rotr(x, 34) ^ rotr(x, 39); }
constexpr Word Sigma_1(Word x) { return rotr(x, 14) ^ rotr(x, 18) ^ rotr(x, 41); }
constexpr Word Ch (Word x, Word y, Word z) { return (x & y) ^ (~x & z); }
constexpr Word Maj(Word x, Word y, Word z) { return (x & y) ^ (x & z) ^ (y & z); }

// SHA-512 round constants (first 38 used)
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

// SHA-512 IV (unused here — SFS collision uses arbitrary h₀)
// constexpr State IV = {
//     0x6a09e667f3bcc908ULL, 0xbb67ae8584caa73bULL,
//     0x3c6ef372fe94f82bULL, 0xa54ff53a5f1d36f1ULL,
//     0x510e527fade682d1ULL, 0x9b05688c2b3e6c1fULL,
//     0x1f83d9abfb41bd6bULL, 0x5be0cd19137e2179ULL,
// };

inline void compress_step(State& s, int step, const Word* W) {
    Word t1 = s[7] + Sigma_1(s[4]) + Ch(s[4], s[5], s[6]) + K[step] + W[step];
    Word t2 = Sigma_0(s[0]) + Maj(s[0], s[1], s[2]);
    s[7] = s[6]; s[6] = s[5]; s[5] = s[4]; s[4] = s[3] + t1;
    s[3] = s[2]; s[2] = s[1]; s[1] = s[0]; s[0] = t1 + t2;
}

// Run num_steps of SHA-512 compression and add Davies-Meyer feed-forward.
State sha512_compress(const State& iv, const Word* M, int num_steps) {
    // Message expansion
    std::array<Word, 80> W{};
    for (int i = 0; i < 16; ++i) W[i] = M[i];
    for (int i = 16; i < num_steps; ++i)
        W[i] = sigma_1(W[i-2]) + W[i-7] + sigma_0(W[i-15]) + W[i-16];

    State s = iv;
    for (int i = 0; i < num_steps; ++i)
        compress_step(s, i, W.data());

    // Davies-Meyer feed-forward
    for (int i = 0; i < 8; ++i) s[i] += iv[i];
    return s;
}

// --- Table 2 data ---

// h₀ (chaining input / IV for semi-free-start)
constexpr State H0 = {
    0xe8626f53a3771964ULL, 0x2ae427b8c5065790ULL,
    0xc8fd5a1628fc3337ULL, 0x0f362d297f82f987ULL,
    0x89166a0c022ffc40ULL, 0xc2c49c30e629239fULL,
    0xd1fa8bd692843025ULL, 0xad4bba64c797e6ecULL,
};

// m (16 × 64-bit words)
constexpr std::array<Word, 16> M1 = {
    0x610519a88f0d2809ULL, 0x3addc83f01c8b179ULL,
    0x84afa7a2772c6141ULL, 0xad539854e64c9cceULL,
    0x85450b73549b2085ULL, 0x7296b5291f31c0d9ULL,
    0xfc978d9624e2c2ccULL, 0xfffffffffffffffeULL,
    0x92114cb9d2f4cd9bULL, 0x34a3198b79871212ULL,
    0xcca7f43154e38081ULL, 0xac0598a589168fe1ULL,
    0xf32ae6a0070a8d2eULL, 0x755aa5cada87e894ULL,
    0x4b9bd7df3c94b667ULL, 0x65291f2b80cc8c51ULL,
};

// m* (16 × 64-bit words)
constexpr std::array<Word, 16> M2 = {
    0x610519a88f0d2809ULL, 0x3addc83f01c8b179ULL,
    0x84afa7a2772c6141ULL, 0xad539854e64c9cceULL,
    0x85450b73549b2085ULL, 0x7296b5291f31c0d9ULL,
    0xfc978d9624e2c2ccULL, 0x0000000000000001ULL,
    0x92114cb9d2f4cd9cULL, 0x34a3198b79871212ULL,
    0xcca8143154e38079ULL, 0xac0598a589168fe1ULL,
    0xf32ae6a0070a8d2eULL, 0x755aa5cada87e894ULL,
    0x4b9bd7df3c94b667ULL, 0x65291f2b80cc8c50ULL,
};

// Expected hash output h₁
constexpr State H1_EXPECTED = {
    0x946a28eedc3b2ff6ULL, 0xc4573d0a13ea6268ULL,
    0x11f07b04b06900ddULL, 0x897c606e4053bbe4ULL,
    0x2406aae9d58504b4ULL, 0x89b237932b061ba8ULL,
    0x663402cb4bb1972cULL, 0xd99c062dce945423ULL,
};

void print_state(const char* label, const State& s) {
    std::printf("%s: ", label);
    for (int i = 0; i < 8; ++i)
        std::printf("%016llx ", (unsigned long long)s[i]);
    std::printf("\n");
}

int main() {
    constexpr int STEPS = 38;
    int failures = 0;

    std::printf("=== Eichlseder-Mendel-Schlaffer 2014, Table 2 ===\n");
    std::printf("38-step SHA-512 semi-free-start collision\n\n");

    // Verify message difference
    std::printf("--- Message difference (XOR) ---\n");
    for (int i = 0; i < 16; ++i) {
        Word diff = M1[i] ^ M2[i];
        if (diff != 0)
            std::printf("  W[%2d]: %016llx ^ %016llx = %016llx\n",
                        i, (unsigned long long)M1[i],
                        (unsigned long long)M2[i],
                        (unsigned long long)diff);
    }
    std::printf("\n");

    // Compress both messages
    State h1 = sha512_compress(H0, M1.data(), STEPS);
    State h2 = sha512_compress(H0, M2.data(), STEPS);

    print_state("h₀ (IV) ", H0);
    print_state("h(m)    ", h1);
    print_state("h(m*)   ", h2);
    print_state("expected", H1_EXPECTED);
    std::printf("\n");

    // Check h(m) == expected
    bool h1_ok = (h1 == H1_EXPECTED);
    std::printf("h(m) matches Table 2 h₁:  %s\n", h1_ok ? "YES" : "NO");
    if (!h1_ok) ++failures;

    // Check collision: h(m) == h(m*)
    bool collision = (h1 == h2);
    std::printf("h(m) == h(m*) (collision): %s\n", collision ? "YES" : "NO");
    if (!collision) ++failures;

    if (failures == 0)
        std::printf("\n*** 38-step SHA-512 SFS collision CONFIRMED ***\n");
    else
        std::printf("\n*** VERIFICATION FAILED (%d checks) ***\n", failures);

    return failures;
}
