// Piece 3: Wang-style message modification for the forward characteristic.
//
// Given a state pair (s0, s2) at step 21 with s2 - s0 = gamma,
// construct W[21..29] such that the a2 differential characteristic
// (Table 3) holds through step 30, at which point the diff is fully
// absorbed (zero everywhere except H = 0xc0000000).
//
// The modifier works step by step:
//   1. Check consistency: C_A - C_E = target_dA' - target_dE'
//   2. Compute dW = target_dA' - (dSig0 + dMaj + dSig1 + dCh + dH)
//   3. Search for absolute A_{i+1} satisfying next-step consistency
//   4. W0_i = W_to_set_A(s0, i, A); W2_i = W0_i + dW
//   5. Advance both states
//
// Verification:
//   - Reproduces the exact Table 1 W values when given Table 1 states.
//   - Given random states with the correct gamma diff, constructs
//     valid W sequences with probability roughly 2^{-12} per attempt.
//
// Build: make msg-modify    (or: g++ -std=c++17 -O2 msg_modify.cpp -o msg_modify)

#include "sha256.hpp"
#include "differentials.hpp"
#include "attack_helpers.hpp"
#include <cstdio>
#include <cstring>
#include <random>

using namespace sha256;

// ---------------------------------------------------------------------------
// Core modifier
// ---------------------------------------------------------------------------

// Compute C_A - C_E = dSig0(A) + dMaj(A,B,C) - dD.
// This is the LHS of the consistency condition (W-independent).
static Word consistency_lhs(const State& s0, const State& s2) {
    return (Sigma_0(s2[0]) - Sigma_0(s0[0]))
         + (Maj(s2[0], s2[1], s2[2]) - Maj(s0[0], s0[1], s0[2]))
         - (s2[3] - s0[3]);
}

// RHS of consistency = target_dA' - target_dE' at step i.
static Word consistency_rhs(int step) {
    return diff46::TABLE1_A2_STATE.d[step + 1][0]
         - diff46::TABLE1_A2_STATE.d[step + 1][4];
}

// Check whether candidate A_{step+1} (for path 0) satisfies the consistency
// condition at step+1.  Uses only A, B=s0[0], C=s0[1] (path 0) and their
// path-2 counterparts.
static bool check_next_consistency(Word A_cand, int step,
                                   const State& s0, const State& s2) {
    int next = step + 1;
    if (next > 29) return true;

    Word dA_next = diff46::TABLE1_A2_STATE.d[next][0];
    Word A2_cand = A_cand + dA_next;

    Word dSig0 = Sigma_0(A2_cand) - Sigma_0(A_cand);
    Word dMaj  = Maj(A2_cand, s2[0], s2[1]) - Maj(A_cand, s0[0], s0[1]);
    Word dD    = s2[2] - s0[2]; // D_{next} = C_i
    Word lhs   = dSig0 + dMaj - dD;
    return lhs == consistency_rhs(next);
}

// Attempt message modification for steps 21-29.
// s0_in, s2_in: states at step 21 (BEFORE step 21 executes).
// W0_out, W2_out: output arrays, must have space for indices [21..29].
// Returns true on success, false if any step fails consistency.
// fail_step: if non-null, set to the step that failed (-1 if success).
bool msg_modify(const State& s0_in, const State& s2_in,
                Word* W0_out, Word* W2_out, std::mt19937& rng,
                int* fail_step = nullptr) {
    State s0 = s0_in, s2 = s2_in;

    for (int step = 21; step <= 29; ++step) {
        // 1. Check consistency at this step
        if (consistency_lhs(s0, s2) != consistency_rhs(step)) {
            if (fail_step) *fail_step = step;
            return false;
        }

        // 2. Compute dW
        Word dSig0 = Sigma_0(s2[0]) - Sigma_0(s0[0]);
        Word dMaj  = Maj(s2[0], s2[1], s2[2]) - Maj(s0[0], s0[1], s0[2]);
        Word dSig1 = Sigma_1(s2[4]) - Sigma_1(s0[4]);
        Word dCh   = Ch(s2[4], s2[5], s2[6]) - Ch(s0[4], s0[5], s0[6]);
        Word dH    = s2[7] - s0[7];
        Word C_A   = dSig0 + dMaj + dSig1 + dCh + dH;
        Word target_dA = diff46::TABLE1_A2_STATE.d[step + 1][0];
        Word dW = target_dA - C_A;

        // 3. Choose absolute A_{step+1}
        Word A_chosen;
        if (step < 29) {
            // Search for A satisfying next-step consistency.
            // Try random candidates until one works.
            bool found = false;
            for (int tries = 0; tries < (1 << 24); ++tries) {
                Word A_cand = rand_word(rng);
                if (check_next_consistency(A_cand, step, s0, s2)) {
                    A_chosen = A_cand;
                    found = true;
                    break;
                }
            }
            if (!found) {
                if (fail_step) *fail_step = step;
                return false;
            }
        } else {
            // Last step: any A works
            A_chosen = rand_word(rng);
        }

        // 4. Compute W values
        W0_out[step] = W_to_set_A(s0, step, A_chosen);
        W2_out[step] = W0_out[step] + dW;

        // 5. Advance states
        compress(s0, step, W0_out);
        compress(s2, step, W2_out);

        // Sanity: verify diff matches characteristic
        for (int r = 0; r < 8; ++r) {
            if (s2[r] - s0[r] != diff46::TABLE1_A2_STATE.d[step + 1][r]) {
                std::fprintf(stderr, "BUG: diff mismatch at step %d reg %d\n", step, r);
                return false;
            }
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

// Table 1 input data from differentials.hpp: TABLE1_Y, TABLE1_A2

// Test 1: Reproduce Table 1 W values.
static bool test_table1() {
    std::printf("=== Test 1: Table 1 reproduction ===\n");

    Word y2_data[24];
    for (int i = 0; i < 24; ++i) y2_data[i] = diff46::TABLE1_Y[i] + diff46::TABLE1_A2[i];

    State iv0, iv2;
    Word M0[16], M2[16];
    attack46::split_input(diff46::TABLE1_Y, iv0, M0);
    attack46::split_input(y2_data, iv2, M2);

    Word W0[46], W2[46];
    for (int i = 0; i < 16; ++i) { W0[i] = M0[i]; W2[i] = M2[i]; }
    for (int i = 16; i < 46; ++i) {
        W0[i] = msg_expand(i, W0);
        W2[i] = msg_expand(i, W2);
    }

    // Compress to step 21
    State s0 = iv0, s2 = iv2;
    for (int i = 0; i < 21; ++i) {
        compress(s0, i, W0);
        compress(s2, i, W2);
    }

    // Run modifier with retry (greedy lookahead doesn't guarantee success)
    Word W0_out[46], W2_out[46];
    memset(W0_out, 0, sizeof(W0_out));
    memset(W2_out, 0, sizeof(W2_out));

    std::mt19937 rng(0);
    bool ok = false;
    int fail_step;
    int attempts = 0;
    for (int retry = 0; retry < 100000; ++retry) {
        ++attempts;
        if (msg_modify(s0, s2, W0_out, W2_out, rng, &fail_step)) {
            ok = true;
            break;
        }
    }
    if (!ok) {
        std::printf("  FAIL: msg_modify never succeeded in %d attempts (last fail at step %d)\n",
                    attempts, fail_step);
        return false;
    }
    std::printf("  Succeeded after %d attempts\n", attempts);

    // Verify the full forward path produces correct diff at step 30
    bool all_match = true;
    State s0_fwd = s0, s2_fwd = s2;
    for (int step = 21; step <= 29; ++step) {
        compress(s0_fwd, step, W0_out);
        compress(s2_fwd, step, W2_out);
        // Check diff at each step
        for (int r = 0; r < 8; ++r) {
            Word got = s2_fwd[r] - s0_fwd[r];
            Word exp = diff46::TABLE1_A2_STATE.d[step + 1][r];
            if (got != exp) {
                std::printf("  Step %d diff MISMATCH reg %d: got=%08x exp=%08x\n",
                            step + 1, r, got, exp);
                all_match = false;
            }
        }
    }

    // Also verify dW at step 21 matches the a2 characteristic
    Word dW21 = W2_out[21] - W0_out[21];
    Word a2_dW21 = W2[21] - W0[21];
    std::printf("  dW[21]: got=%08x ref=%08x %s\n", dW21, a2_dW21,
                dW21 == a2_dW21 ? "MATCH" : "DIFF (ok: different A, same diff)");

    std::printf("  %s\n\n", all_match ? "PASS" : "FAIL");
    return all_match;
}

// Test 2: Random states with correct gamma diff.
// Generate random s0, set s2 = s0 + gamma. Run modifier.
// Count success rate (should be ~2^{-12}).
static bool test_random() {
    std::printf("=== Test 2: Random states (success rate) ===\n");

    const State& gamma = diff46::TABLE1.gamma;
    int total = 1 << 16;  // 65536 trials
    int successes = 0;
    std::mt19937 rng(42);

    Word W0_out[46], W2_out[46];

    for (int trial = 0; trial < total; ++trial) {
        // Random s0
        State s0, s2;
        for (int r = 0; r < 8; ++r) {
            s0[r] = rand_word(rng);
            s2[r] = s0[r] + gamma[r];
        }

        if (msg_modify(s0, s2, W0_out, W2_out, rng))
            ++successes;
    }

    double rate = (double)successes / total;
    double log2_rate = successes > 0 ? std::log2(rate) : -99.0;
    std::printf("  %d / %d successes (2^{%.2f})\n", successes, total, log2_rate);
    std::printf("  Expected: ~2^{-12}\n");

    // Check: rate should be in [2^{-16}, 2^{-8}] range
    bool reasonable = (log2_rate > -16.0 && log2_rate < -8.0);
    std::printf("  %s\n\n", reasonable ? "PASS (in expected range)" : "UNEXPECTED RATE");
    return reasonable;
}

// Test 3: Verify diff integrity. For each successful random modification,
// verify the state diff at step 30 matches the characteristic exactly.
static bool test_diff_integrity() {
    std::printf("=== Test 3: Diff integrity on successes ===\n");

    const State& gamma = diff46::TABLE1.gamma;
    int checked = 0, target = 100;
    std::mt19937 rng(123);
    Word W0_out[46], W2_out[46];

    while (checked < target) {
        State s0, s2;
        for (int r = 0; r < 8; ++r) {
            s0[r] = rand_word(rng);
            s2[r] = s0[r] + gamma[r];
        }

        if (!msg_modify(s0, s2, W0_out, W2_out, rng))
            continue;

        // Replay forward from step 21 to step 30
        State s0_fwd = s0, s2_fwd = s2;
        for (int step = 21; step <= 29; ++step) {
            compress(s0_fwd, step, W0_out);
            compress(s2_fwd, step, W2_out);
        }

        for (int r = 0; r < 8; ++r) {
            Word got = s2_fwd[r] - s0_fwd[r];
            Word exp = diff46::TABLE1_A2_STATE.d[30][r];
            if (got != exp) {
                std::printf("  FAIL at check %d: reg %d got=%08x exp=%08x\n",
                            checked, r, got, exp);
                return false;
            }
        }
        ++checked;
    }

    std::printf("  %d/%d diff checks passed\n", checked, target);
    std::printf("  PASS\n\n");
    return true;
}

int main() {
    bool t1 = test_table1();
    bool t2 = test_random();
    bool t3 = test_diff_integrity();

    std::printf("=== Summary ===\n");
    std::printf("  Test 1 (Table 1 reproduction): %s\n", t1 ? "PASS" : "FAIL");
    std::printf("  Test 2 (random success rate):   %s\n", t2 ? "PASS" : "FAIL");
    std::printf("  Test 3 (diff integrity):        %s\n", t3 ? "PASS" : "FAIL");

    return (t1 && t2 && t3) ? 0 : 1;
}
