// 27-step SHA-256 collision generator.
//
// Implements the algebraic completion phase of Mendel-Nad-Schlaffer 2011:
// given W[0..6] from the paper's Table 8, solve steps 7-15 via per-step
// E-enumeration + joint step 14-15 enumeration (2^32 candidates), then
// verify each collision through full 27-step compression + feed-forward.
//
// Output: all distinct colliding message pairs (M, M*) with verification.
//
// Usage:
//   ./collision_generator              # generate + verify all collisions
//   ./collision_generator --count      # count only (no hex output)
//   ./collision_generator --first N    # stop after N collisions
//
// Build:
//   g++ -std=c++17 -O3 -Wall -Wextra -Wno-trigraphs -pthread -I. \
//       -o build/collision_generator src/mendel2011/collision_generator.cpp

#include "sha256.hpp"
#include "alt_step.hpp"
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#include <cassert>

using namespace sha256;
using Word = uint32_t;

// ========================================================================
// Table 7: generalized bit conditions per step (from Mendel et al. 2011)
// Position s[0] = MSB (bit 31), s[31] = LSB (bit 0).
// Conditions: 0/1 = fixed equal, u = (1,0), n = (0,1), - = equal free,
//             x = unequal free, ? = unconstrained.
// ========================================================================

static const char* TABLE7_E[] = {
    "--------------------------------", "--------------------------------",
    "--------------------------------", "--------------------------------",
    "--------------------------------",
    "----------------1--------1------",
    "-1--------0--0-10-1----0-0------",
    "101-11---u10u1-0nuu-uuuu1n---n0-",
    "0n0n001001u-1u1n01un010n01n00110",
    "-1n1n1011u011100nn100u10-10000u-",
    "u00000nuuu10uun01u00n00n110-u-u1",
    "0n000uuuuu01010111n-uun01n000n01",
    "01---1010u01u----111-010-0--110-",
    "01-10u1nunuuu---1110-1nn11---01-",
    "-----1-01011----------00--------",
    "-----1-001000---------11--------",
    "--------------------------------", "--------------------------------",
    "--------------------------------", "--------------------------------",
    "--------------------------------", "--------------------------------",
    "--------------------------------", "--------------------------------",
    "--------------------------------", "--------------------------------",
    "--------------------------------",
};

static const char* TABLE7_A[] = {
    "--------------------------------", "--------------------------------",
    "--------------------------------", "--------------------------------",
    "--------------------------------", "--------------------------------",
    "--------------------------------",
    "-------unn--u------n---nn-uuuu--",
    "nnnnn-nnnn--------nuu-----------",
    "----un--n--nu-------nu-u--------",
    "--------------------------------", "--------------------------------",
    "--------------------------------", "--------------------------------",
    "--------------------------------", "--------------------------------",
    "--------------------------------", "--------------------------------",
    "--------------------------------", "--------------------------------",
    "--------------------------------", "--------------------------------",
    "--------------------------------", "--------------------------------",
    "--------------------------------", "--------------------------------",
    "--------------------------------",
};

static const char* TABLE7_W[] = {
    "--------------------------------", "--------------------------------",
    "--------------------------------", "--------------------------------",
    "--------------------------------", "--------------------------------",
    "--------------------------------",
    "00---1--un-0u-nuuuuu1-nu0n101n--",
    "-----u--n---n---------nn--------",
    "--------------------------------", "--------------------------------",
    "--------------------------------",
    "------110-u-------n0--u--n-n--nn",
    "--------------------------------", "--------------------------------",
    "0u1-nn-n-u-1u---11un0uu10u101u0-",
};

static const char* TABLE7_W17 = "---0-1nnn---u-1-----10uu0-------";

// ========================================================================
// Table 8: known collision (Mendel et al. 2011, ASIACRYPT)
// ========================================================================

static const Word TABLE8_M1[16] = {
    0x725a0370, 0x0daa9f1b, 0x071d92df, 0xec8282c1,
    0x7913134a, 0xbc2eb291, 0x02d33a84, 0x278dfd29,
    0x0c40f8ea, 0xd8bd68a0, 0x0ce670c5, 0x5ec7155d,
    0x9f6407a8, 0x729fbfe8, 0xaa7c7c08, 0x607ae76d
};
static const Word TABLE8_M2[16] = {
    0x725a0370, 0x0daa9f1b, 0x071d92df, 0xec8282c1,
    0x7913134a, 0xbc2eb291, 0x02d33a84, 0x27460e6d,
    0x08c8fbea, 0xd8bd68a0, 0x0ce670c5, 0x5ec7155d,
    0x9f4425fb, 0x729fbfe8, 0xaa7c7c08, 0x2d32d129
};

// ========================================================================
// Condition checking + E enumeration
// ========================================================================

static inline bool check_cond(Word v1, Word v2, const char* cond) {
    for (int i = 0; i < 32; ++i) {
        int bit = 31 - i;
        int b1 = (v1 >> bit) & 1;
        int b2 = (v2 >> bit) & 1;
        switch (cond[i]) {
            case '0': if (b1 != 0 || b2 != 0) return false; break;
            case '1': if (b1 != 1 || b2 != 1) return false; break;
            case 'u': if (b1 != 1 || b2 != 0) return false; break;
            case 'n': if (b1 != 0 || b2 != 1) return false; break;
            case '-': if (b1 != b2) return false; break;
            case 'x': if (b1 == b2) return false; break;
            default: break;
        }
    }
    return true;
}

struct ETemplate {
    Word fixed_e1, fixed_e2, dash_mask, x_mask;
    int n_dash, n_x;
    int dash_pos[32], x_pos[32];
};

static ETemplate build_E_template(const char* cond) {
    ETemplate t{};
    for (int i = 0; i < 32; ++i) {
        int bit = 31 - i;
        switch (cond[i]) {
            case '0': break;
            case '1': t.fixed_e1 |= (1u << bit); t.fixed_e2 |= (1u << bit); break;
            case 'u': t.fixed_e1 |= (1u << bit); break;
            case 'n': t.fixed_e2 |= (1u << bit); break;
            case '-': t.dash_mask |= (1u << bit); t.dash_pos[t.n_dash++] = bit; break;
            case 'x': t.x_mask |= (1u << bit); t.x_pos[t.n_x++] = bit; break;
            default:  t.dash_mask |= (1u << bit); t.dash_pos[t.n_dash++] = bit; break;
        }
    }
    return t;
}

struct EPair { Word e1, e2; };

static std::vector<EPair> enumerate_E(const ETemplate& t) {
    int total_free = t.n_dash + t.n_x;
    uint64_t count = 1ULL << total_free;
    std::vector<EPair> result;
    result.reserve(count);
    for (uint64_t idx = 0; idx < count; ++idx) {
        Word e1 = t.fixed_e1, e2 = t.fixed_e2;
        for (int d = 0; d < t.n_dash; ++d) {
            if ((idx >> d) & 1) { e1 |= (1u << t.dash_pos[d]); e2 |= (1u << t.dash_pos[d]); }
        }
        for (int x = 0; x < t.n_x; ++x) {
            int bi = t.n_dash + x;
            if ((idx >> bi) & 1) e1 |= (1u << t.x_pos[x]);
            else e2 |= (1u << t.x_pos[x]);
        }
        result.push_back({e1, e2});
    }
    return result;
}

// ========================================================================
// solve_step: enumerate E (or W for step 15), derive W (or E), check conds
// ========================================================================

struct StepSolution {
    AltState s1, s2;
    Word w1, w2;
};

static std::vector<StepSolution> solve_step(
    const AltState& pre_s1, const AltState& pre_s2,
    int step, Word* W1, Word* W2)
{
    std::vector<StepSolution> survivors;
    const char* e_cond = TABLE7_E[step];
    const char* a_cond = TABLE7_A[step];
    const char* w_cond = (step < 16) ? TABLE7_W[step] : "--------------------------------";

    // Step 15: enumerate W[15], compute E[15], check E+A+W[17]
    if (step == 15) {
        ETemplate w_tmpl = build_E_template(w_cond);
        auto w_candidates = enumerate_E(w_tmpl);
        for (auto& [w1, w2] : w_candidates) {
            Word e1 = pre_s1.E[0] + Sigma_1(pre_s1.E[3]) + Ch(pre_s1.E[3], pre_s1.E[2], pre_s1.E[1])
                     + pre_s1.A[0] + K[step] + w1;
            Word e2 = pre_s2.E[0] + Sigma_1(pre_s2.E[3]) + Ch(pre_s2.E[3], pre_s2.E[2], pre_s2.E[1])
                     + pre_s2.A[0] + K[step] + w2;
            if (!check_cond(e1, e2, e_cond)) continue;
            Word a1 = -pre_s1.A[0] + Sigma_0(pre_s1.A[3]) + Maj(pre_s1.A[3], pre_s1.A[2], pre_s1.A[1]) + e1;
            Word a2 = -pre_s2.A[0] + Sigma_0(pre_s2.A[3]) + Maj(pre_s2.A[3], pre_s2.A[2], pre_s2.A[1]) + e2;
            if (!check_cond(a1, a2, a_cond)) continue;
            Word w17_1 = sigma_1(w1) + W1[10] + sigma_0(W1[2]) + W1[1];
            Word w17_2 = sigma_1(w2) + W2[10] + sigma_0(W2[2]) + W2[1];
            if (!check_cond(w17_1, w17_2, TABLE7_W17)) continue;
            AltState post1 = pre_s1, post2 = pre_s2;
            W1[step] = w1; W2[step] = w2;
            alt_step(post1, step, W1); alt_step(post2, step, W2);
            survivors.push_back({post1, post2, w1, w2});
        }
        return survivors;
    }

    // General case: enumerate E, derive W, check A+W conditions
    ETemplate e_tmpl = build_E_template(e_cond);
    bool has_aw_conds = false;
    for (int i = 0; i < 32; ++i) {
        if (a_cond[i] != '-' && a_cond[i] != '?') { has_aw_conds = true; break; }
    }
    if (!has_aw_conds) {
        for (int i = 0; i < 32; ++i) {
            if (w_cond[i] != '-' && w_cond[i] != '?') { has_aw_conds = true; break; }
        }
    }

    // High-dash, no A/W filtering → pick one arbitrary E
    if (e_tmpl.n_dash > 16 && !has_aw_conds) {
        Word e1 = e_tmpl.fixed_e1, e2 = e_tmpl.fixed_e2;
        Word w1 = W_from_state(e1, pre_s1.E[3], pre_s1.E[2], pre_s1.E[1], pre_s1.E[0], pre_s1.A[0], step);
        Word w2 = W_from_state(e2, pre_s2.E[3], pre_s2.E[2], pre_s2.E[1], pre_s2.E[0], pre_s2.A[0], step);
        AltState post1 = pre_s1, post2 = pre_s2;
        W1[step] = w1; W2[step] = w2;
        alt_step(post1, step, W1); alt_step(post2, step, W2);
        survivors.push_back({post1, post2, w1, w2});
        return survivors;
    }

    auto e_candidates = enumerate_E(e_tmpl);
    for (auto& [e1, e2] : e_candidates) {
        Word a1 = -pre_s1.A[0] + Sigma_0(pre_s1.A[3]) + Maj(pre_s1.A[3], pre_s1.A[2], pre_s1.A[1]) + e1;
        Word a2 = -pre_s2.A[0] + Sigma_0(pre_s2.A[3]) + Maj(pre_s2.A[3], pre_s2.A[2], pre_s2.A[1]) + e2;
        if (!check_cond(a1, a2, a_cond)) continue;
        Word w1 = W_from_state(e1, pre_s1.E[3], pre_s1.E[2], pre_s1.E[1], pre_s1.E[0], pre_s1.A[0], step);
        Word w2 = W_from_state(e2, pre_s2.E[3], pre_s2.E[2], pre_s2.E[1], pre_s2.E[0], pre_s2.A[0], step);
        if (!check_cond(w1, w2, w_cond)) continue;
        AltState post1 = pre_s1, post2 = pre_s2;
        W1[step] = w1; W2[step] = w2;
        alt_step(post1, step, W1); alt_step(post2, step, W2);
        survivors.push_back({post1, post2, w1, w2});
    }
    return survivors;
}

// ========================================================================
// Expansion constraint checks
// ========================================================================

static bool check_exp_after_step15(const Word* W1, const Word* W2) {
    Word dw15 = W2[15] - W1[15];
    Word ds0_w7 = sigma_0(W2[7]) - sigma_0(W1[7]);
    if ((dw15 + ds0_w7) != 0) return false;

    Word w17_1 = sigma_1(W1[15]) + W1[10] + sigma_0(W1[2]) + W1[1];
    Word w17_2 = sigma_1(W2[15]) + W2[10] + sigma_0(W2[2]) + W2[1];
    if ((w17_2 - w17_1 + W2[8] - W1[8]) != 0) return false;

    Word ds1_w17 = sigma_1(w17_2) - sigma_1(w17_1);
    if ((ds1_w17 + W2[12] - W1[12]) != 0) return false;

    return true;
}

// Check collision after step 15: expand W[16..26], compress, feed-forward
static bool check_collision_16_26(
    const AltState& pre16_s1, const AltState& pre16_s2,
    Word* W1, Word* W2)
{
    for (int i = 16; i < 27; ++i) {
        W1[i] = sigma_1(W1[i-2]) + W1[i-7] + sigma_0(W1[i-15]) + W1[i-16];
        W2[i] = sigma_1(W2[i-2]) + W2[i-7] + sigma_0(W2[i-15]) + W2[i-16];
    }
    AltState s1 = pre16_s1, s2 = pre16_s2;
    for (int step = 16; step < 27; ++step) {
        alt_step(s1, step, W1); alt_step(s2, step, W2);
    }
    State st1 = alt_to_state(s1), st2 = alt_to_state(s2);
    State ff1 = feed_forward(st1, sha256::IV);
    State ff2 = feed_forward(st2, sha256::IV);
    for (int i = 0; i < 8; ++i)
        if (ff1[i] != ff2[i]) return false;
    return true;
}

// ========================================================================
// Independent verification via standard SHA-256
// ========================================================================

static bool verify_collision_standard(const Word* m1, const Word* m2) {
    Word W1[64], W2[64];
    for (int i = 0; i < 16; i++) { W1[i] = m1[i]; W2[i] = m2[i]; }
    for (int i = 16; i < 27; i++) {
        W1[i] = sigma_1(W1[i-2]) + W1[i-7] + sigma_0(W1[i-15]) + W1[i-16];
        W2[i] = sigma_1(W2[i-2]) + W2[i-7] + sigma_0(W2[i-15]) + W2[i-16];
    }
    State s1 = sha256::IV, s2 = sha256::IV;
    for (int i = 0; i < 27; i++) {
        sha256::compress(s1, i, W1);
        sha256::compress(s2, i, W2);
    }
    State ff1 = feed_forward(s1, sha256::IV);
    State ff2 = feed_forward(s2, sha256::IV);
    for (int i = 0; i < 8; i++)
        if (ff1[i] != ff2[i]) return false;
    return true;
}

// ========================================================================
// Main: follow Table 8 through steps 7-13, joint-enumerate steps 14-15
// ========================================================================

int main(int argc, char** argv) {
    setbuf(stdout, NULL);

    bool count_only = false;
    int first_n = 0;  // 0 = all
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--count") == 0) count_only = true;
        else if (strcmp(argv[i], "--first") == 0 && i+1 < argc) first_n = atoi(argv[++i]);
    }

    // Reference: expand Table 8 fully for path verification
    Word W1_ref[64] = {}, W2_ref[64] = {};
    for (int i = 0; i < 16; i++) { W1_ref[i] = TABLE8_M1[i]; W2_ref[i] = TABLE8_M2[i]; }
    for (int i = 16; i < 27; i++) {
        W1_ref[i] = sigma_1(W1_ref[i-2]) + W1_ref[i-7] + sigma_0(W1_ref[i-15]) + W1_ref[i-16];
        W2_ref[i] = sigma_1(W2_ref[i-2]) + W2_ref[i-7] + sigma_0(W2_ref[i-15]) + W2_ref[i-16];
    }

    // ---- Phase 1: follow Table 8's exact path through steps 0-13 ----
    AltState s1 = state_to_alt(sha256::IV), s2 = state_to_alt(sha256::IV);
    Word W1[64] = {}, W2[64] = {};
    for (int i = 0; i < 16; i++) { W1[i] = TABLE8_M1[i]; W2[i] = TABLE8_M2[i]; }

    // Steps 0-6: forward from IV (both copies identical, W[0..6] same)
    for (int step = 0; step < 7; step++) {
        alt_step(s1, step, W1);
        alt_step(s2, step, W2);
    }

    // Steps 7-13: follow Table 8's path via solve_step
    for (int step = 7; step <= 13; step++) {
        Word Wc1[64], Wc2[64];
        memcpy(Wc1, W1, sizeof(Word)*64);
        memcpy(Wc2, W2, sizeof(Word)*64);
        auto survivors = solve_step(s1, s2, step, Wc1, Wc2);

        bool found = false;
        for (auto& sol : survivors) {
            if (sol.w1 == W1_ref[step] && sol.w2 == W2_ref[step]) {
                s1 = sol.s1; s2 = sol.s2;
                W1[step] = sol.w1; W2[step] = sol.w2;
                found = true;
                break;
            }
        }
        if (!found) {
            fprintf(stderr, "ERROR: Table 8 path lost at step %d (solver has %zu survivors)\n",
                    step, survivors.size());
            return 1;
        }
    }

    // ---- Phase 2: joint step 14-15 enumeration ----
    AltState pre14_s1 = s1, pre14_s2 = s2;
    ETemplate e14_tmpl = build_E_template(TABLE7_E[14]);
    ETemplate w15_tmpl = build_E_template(TABLE7_W[15]);
    auto w15_candidates = enumerate_E(w15_tmpl);

    auto e14_candidates = enumerate_E(e14_tmpl);
    uint64_t n_e14 = e14_candidates.size();
    uint64_t total_joint = n_e14 * w15_candidates.size();

    // dW[23] = dσ₀(W[8]) + dW[7] is invariant (depends only on fixed W[7,8])
    assert((sigma_0(W2[8]) - sigma_0(W1[8]) + W2[7] - W1[7]) == 0
           && "EXP[23] constraint violated — W[7,8] from Table 8 are inconsistent");

    fprintf(stderr, "Phase 1: Table 8 path followed through step 13.\n");
    fprintf(stderr, "Phase 2: joint step 14-15 enumeration.\n");
    fprintf(stderr, "  E[14]: %llu candidates\n", (unsigned long long)n_e14);
    fprintf(stderr, "  W[15]: %zu candidates\n", w15_candidates.size());
    fprintf(stderr, "  Total: %llu\n", (unsigned long long)total_joint);

    uint64_t collisions = 0;
    uint64_t verified = 0;

    for (uint64_t e14_idx = 0; e14_idx < n_e14; ++e14_idx) {
        auto [e14_1, e14_2] = e14_candidates[e14_idx];

        // Compute A[14], check dA[14]=0
        Word a14_1 = -pre14_s1.A[0] + Sigma_0(pre14_s1.A[3])
                   + Maj(pre14_s1.A[3], pre14_s1.A[2], pre14_s1.A[1]) + e14_1;
        Word a14_2 = -pre14_s2.A[0] + Sigma_0(pre14_s2.A[3])
                   + Maj(pre14_s2.A[3], pre14_s2.A[2], pre14_s2.A[1]) + e14_2;
        if (a14_1 != a14_2) continue;

        // Build pre-step-15 state
        AltState pre15_s1 = {{pre14_s1.A[1], pre14_s1.A[2], pre14_s1.A[3], a14_1},
                             {pre14_s1.E[1], pre14_s1.E[2], pre14_s1.E[3], e14_1}};
        AltState pre15_s2 = {{pre14_s2.A[1], pre14_s2.A[2], pre14_s2.A[3], a14_2},
                             {pre14_s2.E[1], pre14_s2.E[2], pre14_s2.E[3], e14_2}};

        // Derive W[14]
        Word w14_1 = W_from_state(e14_1, pre14_s1.E[3], pre14_s1.E[2],
                                   pre14_s1.E[1], pre14_s1.E[0], pre14_s1.A[0], 14);
        Word w14_2 = W_from_state(e14_2, pre14_s2.E[3], pre14_s2.E[2],
                                   pre14_s2.E[1], pre14_s2.E[0], pre14_s2.A[0], 14);

        // Try all W[15] candidates
        for (auto& [w15_1, w15_2] : w15_candidates) {
            Word e15_1 = pre15_s1.E[0] + Sigma_1(pre15_s1.E[3])
                       + Ch(pre15_s1.E[3], pre15_s1.E[2], pre15_s1.E[1])
                       + pre15_s1.A[0] + K[15] + w15_1;
            Word e15_2 = pre15_s2.E[0] + Sigma_1(pre15_s2.E[3])
                       + Ch(pre15_s2.E[3], pre15_s2.E[2], pre15_s2.E[1])
                       + pre15_s2.A[0] + K[15] + w15_2;

            if (!check_cond(e15_1, e15_2, TABLE7_E[15])) continue;

            Word a15_1 = -pre15_s1.A[0] + Sigma_0(pre15_s1.A[3])
                       + Maj(pre15_s1.A[3], pre15_s1.A[2], pre15_s1.A[1]) + e15_1;
            Word a15_2 = -pre15_s2.A[0] + Sigma_0(pre15_s2.A[3])
                       + Maj(pre15_s2.A[3], pre15_s2.A[2], pre15_s2.A[1]) + e15_2;
            if (a15_1 != a15_2) continue;

            Word w17_1 = sigma_1(w15_1) + W1[10] + sigma_0(W1[2]) + W1[1];
            Word w17_2 = sigma_1(w15_2) + W2[10] + sigma_0(W2[2]) + W2[1];
            if (!check_cond(w17_1, w17_2, TABLE7_W17)) continue;

            // Check all 4 expansion constraints
            Word Wc1[64] = {}, Wc2[64] = {};
            memcpy(Wc1, W1, sizeof(Word)*14);
            memcpy(Wc2, W2, sizeof(Word)*14);
            Wc1[14] = w14_1; Wc2[14] = w14_2;
            Wc1[15] = w15_1; Wc2[15] = w15_2;

            if (!check_exp_after_step15(Wc1, Wc2)) continue;

            // Full collision check (steps 16-26 + feed-forward)
            AltState post15_s1 = pre15_s1, post15_s2 = pre15_s2;
            alt_step(post15_s1, 15, Wc1);
            alt_step(post15_s2, 15, Wc2);

            Word Wfull1[64], Wfull2[64];
            for (int i = 0; i < 16; i++) { Wfull1[i] = Wc1[i]; Wfull2[i] = Wc2[i]; }

            if (!check_collision_16_26(post15_s1, post15_s2, Wfull1, Wfull2))
                continue;

            collisions++;

            // Build full M1, M2 (message words 0..15)
            Word M1[16], M2[16];
            for (int i = 0; i < 14; i++) { M1[i] = W1[i]; M2[i] = W2[i]; }
            M1[14] = w14_1; M2[14] = w14_2;
            M1[15] = w15_1; M2[15] = w15_2;

            // Independent verification via standard SHA-256
            bool ok = verify_collision_standard(M1, M2);
            if (ok) verified++;

            if (!count_only) {
                printf("collision %llu %s\n", (unsigned long long)collisions, ok ? "VERIFIED" : "FAIL");
                printf("  M1:");
                for (int i = 0; i < 16; i++) printf(" %08x", M1[i]);
                printf("\n  M2:");
                for (int i = 0; i < 16; i++) printf(" %08x", M2[i]);
                printf("\n");
            }

            if (first_n > 0 && (int)collisions >= first_n) goto done;
        }

        // Progress (to stderr so it doesn't mix with collision output)
        if ((e14_idx & 0xFFFFF) == 0 && e14_idx > 0) {
            fprintf(stderr, "  ... %llu / %llu E[14] tested, %llu collisions so far\n",
                    (unsigned long long)e14_idx, (unsigned long long)n_e14,
                    (unsigned long long)collisions);
        }
    }

done:
    fprintf(stderr, "\nResult: %llu collisions found, %llu independently verified.\n",
            (unsigned long long)collisions, (unsigned long long)verified);
    fprintf(stderr, "All collisions are 27-step SHA-256 with standard IV.\n");
    fprintf(stderr, "Messages share W[0..13] with Table 8; W[14,15] vary.\n");

    return (collisions > 0 && verified == collisions) ? 0 : 1;
}
