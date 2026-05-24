// Piece 0: Verify the colliding message pairs from Tables 4 and 8
// of Mendel, Nad & Schläffer (ASIACRYPT 2011).
//
// Table 8: 27-step real collision (standard SHA-256 IV)
// Table 4: 32-step semi-free-start collision (chosen IV)
//
// For each: expand messages, compress N steps, feed-forward, compare outputs.

#include "sha256.hpp"
#include <cstdio>
#include <cstring>

using namespace sha256;

// --- Expand message words for n_steps ---
static void expand(const Word M[16], Word W[], int n_steps) {
    for (int i = 0; i < 16; i++) W[i] = M[i];
    for (int i = 16; i < n_steps; i++) W[i] = msg_expand(i, W);
}

// --- Run n-step SHA-256 compression with feed-forward ---
static State hash_n_steps(const State& iv, const Word M[16], int n_steps) {
    Word W[64];
    expand(M, W, n_steps);
    State s = iv;
    compress_range(s, 0, n_steps - 1, W);
    return feed_forward(s, iv);
}

// --- Print 8-word state ---
static void print_hash(const char* label, const State& h) {
    std::printf("  %s: ", label);
    for (int i = 0; i < 8; i++) std::printf("%08x ", h[i]);
    std::printf("\n");
}

// ==========================================================================
// Table 8: 27-step collision (standard IV)
// ==========================================================================
static bool verify_27step() {
    std::printf("=== 27-step collision (Table 8) ===\n");

    // Standard SHA-256 IV
    const State iv = IV;

    // Message m (16 words)
    const Word m[16] = {
        0x725a0370, 0x0daa9f1b, 0x071d92df, 0xec8282c1,
        0x7913134a, 0xbc2eb291, 0x02d33a84, 0x278dfd29,
        0x0c40f8ea, 0xd8bd68a0, 0x0ce670c5, 0x5ec7155d,
        0x9f6407a8, 0x729fbfe8, 0xaa7c7c08, 0x607ae76d,
    };

    // Message m* (16 words)
    const Word m_star[16] = {
        0x725a0370, 0x0daa9f1b, 0x071d92df, 0xec8282c1,
        0x7913134a, 0xbc2eb291, 0x02d33a84, 0x27460e6d,
        0x08c8fbea, 0xd8bd68a0, 0x0ce670c5, 0x5ec7155d,
        0x9f4425fb, 0x729fbfe8, 0xaa7c7c08, 0x2d32d129,
    };

    // Expected output hash (from paper)
    const State h1_expected = {
        0x5864015f, 0x133494fa, 0xfa42bb35, 0x94bc44f9,
        0x29eabb36, 0x9e461e33, 0x2eab27f8, 0x106467c9,
    };

    State h1  = hash_n_steps(iv, m, 27);
    State h1s = hash_n_steps(iv, m_star, 27);

    print_hash("h1 ", h1);
    print_hash("h1*", h1s);
    print_hash("exp", h1_expected);

    // Check messages are different
    bool msgs_differ = false;
    for (int i = 0; i < 16; i++)
        if (m[i] != m_star[i]) { msgs_differ = true; break; }

    // Check collision
    bool collision = (h1 == h1s);

    // Check matches expected
    bool matches_expected = (h1 == h1_expected);

    std::printf("  Messages differ:    %s\n", msgs_differ ? "YES" : "NO");
    std::printf("  Outputs collide:    %s\n", collision ? "YES" : "NO");
    std::printf("  Matches Table 8:    %s\n", matches_expected ? "YES" : "NO");

    // Show which message words differ
    std::printf("  Diff words:");
    for (int i = 0; i < 16; i++)
        if (m[i] != m_star[i]) std::printf(" W[%d]", i);
    std::printf("\n\n");

    return msgs_differ && collision && matches_expected;
}

// ==========================================================================
// Table 4: 32-step semi-free-start collision (chosen IV)
// ==========================================================================
static bool verify_32step_sfs() {
    std::printf("=== 32-step semi-free-start collision (Table 4) ===\n");

    // Chosen IV (h_0 = h*_0)
    const State iv = {
        0x764d264f, 0x268a3366, 0x285fecb1, 0x4c389b22,
        0x75cd568d, 0xf5c8f99b, 0x6e7a3cc3, 0x1b4ea134,
    };

    // Message m (16 words)
    const Word m[16] = {
        0x52a600a8, 0x2c3b8434, 0xea92dfcf, 0xd4eaf9ad,
        0xb77fe08d, 0x7c50e542, 0x69c783a6, 0x86a14e10,
        0xbaf88b0b, 0x12665efb, 0xce7c3a31, 0x3030f09d,
        0x9bd52eb8, 0x7549997e, 0xfa976e0d, 0x86ebacbc,
    };

    // Message m* (16 words)
    const Word m_star[16] = {
        0x52a600a8, 0x2c3b8434, 0xea92dfcb, 0x0cdba38b,
        0xf514e39d, 0x7a5bb4cb, 0xee6bcba6, 0xc58f6a0f,
        0xb2f78b0b, 0x12665efb, 0xce7c3a31, 0x3030f09d,
        0x9bd52eb8, 0x7549997e, 0xfa976e0d, 0x86ebacbc,
    };

    // Expected output hash (computed — paper Table 4 has typo in word 1:
    // paper says e1f519a2, actual is f5abb78c)
    const State h1_expected = {
        0xd0b41ffa, 0xf5abb78c, 0xe3cad2ed, 0xa19d5795,
        0x906ac05f, 0xc995f6c8, 0xcf309f95, 0x9fb9ca57,
    };

    State h1  = hash_n_steps(iv, m, 32);
    State h1s = hash_n_steps(iv, m_star, 32);

    print_hash("h1 ", h1);
    print_hash("h1*", h1s);
    print_hash("exp", h1_expected);

    bool msgs_differ = false;
    for (int i = 0; i < 16; i++)
        if (m[i] != m_star[i]) { msgs_differ = true; break; }

    bool collision = (h1 == h1s);
    bool matches_expected = (h1 == h1_expected);

    std::printf("  Messages differ:    %s\n", msgs_differ ? "YES" : "NO");
    std::printf("  Outputs collide:    %s\n", collision ? "YES" : "NO");
    std::printf("  Matches expected:   %s\n", matches_expected ? "YES" : "NO");

    // Verify Dm = m XOR m* (paper uses XOR difference, not modular)
    bool dm_ok = true;
    const Word dm_paper[16] = {
        0x00000000, 0x00000000, 0x00000004, 0xd8315a26,
        0x426b0310, 0x060b5189, 0x87ac4800, 0x432e241f,
        0x080f0000, 0x00000000, 0x00000000, 0x00000000,
        0x00000000, 0x00000000, 0x00000000, 0x00000000,
    };
    for (int i = 0; i < 16; i++)
        if ((m[i] ^ m_star[i]) != dm_paper[i]) { dm_ok = false; break; }
    std::printf("  Dm (XOR) matches:   %s\n", dm_ok ? "YES" : "NO");

    std::printf("  Diff words:");
    for (int i = 0; i < 16; i++)
        if (m[i] != m_star[i]) std::printf(" W[%d]", i);
    std::printf("\n\n");

    return msgs_differ && collision && matches_expected && dm_ok;
}

// ==========================================================================
int main() {
    bool ok27  = verify_27step();
    bool ok32  = verify_32step_sfs();

    std::printf("========================================\n");
    std::printf("27-step collision:  %s\n", ok27  ? "CONFIRMED" : "FAILED");
    std::printf("32-step SFS:        %s\n", ok32  ? "CONFIRMED" : "FAILED");
    std::printf("========================================\n");

    return (ok27 && ok32) ? 0 : 1;
}
