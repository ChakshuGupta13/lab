// Carry-chain Markov model for β-condition cost prediction (Zhang et al. 2026).
//
// For modular addition x' = x + Δ (mod 2^32), x uniform:
//   P(carry into bit b) = p[b] via Markov chain:
//     p[0] = 0
//     p[b+1] = (1 + p[b])/2  if Δ[b] = 1
//     p[b+1] = p[b]/2        if Δ[b] = 0
//
//   P(x[b]=v, x'[b]=v') = (1/2) · P(carry_b = v ⊕ v' ⊕ Δ[b])
//
// Key insight: the paper's β model assumes each diff condition costs 1 bit
// (P = 1/2), which requires carries to be deterministic. In the unconstrained
// regime, carries equilibrate at P = 1/2, making each diff condition cost 2 bits.
// The constrained regime (TAB entry) partially determines carries, giving
// intermediate cost (empirically ~1.71 bits for 36-step).
//
// This program:
// 1. Validates the carry-chain model against direct sampling (fix Δ, sample x)
// 2. Shows that averaging carry-chain predictions over random Δ_mod reproduces
//    the 2.00 bits/diff-cond measured empirically for the unconstrained case
// 3. Computes SFS-pair-specific predictions (one data point of the constrained regime)

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <random>
#include <vector>
#include <string>
#include <algorithm>
#include "sha256.hpp"
#include "carry_chain.hpp"
#include "table8_char.hpp"

using namespace sha256;
using namespace zhang2026;

// CarryChain now in common/carry_chain.hpp
using CarryChain = carry_chain::CarryChain32;

// ═══════════════════════════════════════════════════════════════════════════════
// Part 2: SHA-256 37-step computation (reused from scratch_beta37_parallel.cpp)
// ═══════════════════════════════════════════════════════════════════════════════

static constexpr int STEPS = 37;
static constexpr int OFF = 4;

struct FullState {
    Word A[STEPS + 4];
    Word E[STEPS + 4];
    Word W[64];
};

static FullState compute_full(const State& cv, const Word* msg) {
    FullState fs{};
    fs.A[0] = cv[3]; fs.A[1] = cv[2]; fs.A[2] = cv[1]; fs.A[3] = cv[0];
    fs.E[0] = cv[7]; fs.E[1] = cv[6]; fs.E[2] = cv[5]; fs.E[3] = cv[4];

    for (int i = 0; i < 16; ++i) fs.W[i] = msg[i];
    for (int i = 16; i < STEPS; ++i)
        fs.W[i] = sigma_1(fs.W[i-2]) + fs.W[i-7] + sigma_0(fs.W[i-15]) + fs.W[i-16];

    for (int i = 0; i < STEPS; ++i) {
        int ai = i + OFF;
        Word t1 = fs.E[ai-4] + Sigma_1(fs.E[ai-1])
                  + Ch(fs.E[ai-1], fs.E[ai-2], fs.E[ai-3]) + K[i] + fs.W[i];
        Word t2 = Sigma_0(fs.A[ai-1]) + Maj(fs.A[ai-1], fs.A[ai-2], fs.A[ai-3]);
        fs.A[ai] = t1 + t2;
        fs.E[ai] = fs.A[ai-4] + t1;
    }
    return fs;
}

static void apply_diff_37(const Word* M, Word* Mp) {
    for (int i = 0; i < 16; ++i) Mp[i] = M[i];
    for (int w : {6, 7, 9, 14}) {
        const char* pat = TABLE8_DW[w - T8_FIRST_W];
        Word mask = 0;
        for (int b = 0; b < 32; ++b) {
            char c = pat[31 - b];
            if (c == 'n' || c == 'u') mask |= (1u << b);
        }
        Mp[w] = M[w] ^ mask;
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Part 3: β condition descriptors (diff conditions only for carry-chain model)
// ═══════════════════════════════════════════════════════════════════════════════

struct DiffCond {
    int reg;    // 1=E, 2=W
    int step;   // register step index
    int bit;
    char cond;  // 'n', 'u', '0', '1'
    std::string label;
};

static std::vector<DiffCond> build_diff_conditions() {
    std::vector<DiffCond> conds;

    // E16..E21
    for (int step = 16; step <= 21; ++step) {
        int idx = step - T8_FIRST_STEP;
        const char* pat = TABLE8_DE[idx];
        for (int b = 0; b < 32; ++b) {
            char c = pat[31 - b];
            if (c == '=') continue;
            char buf[64];
            snprintf(buf, sizeof(buf), "E%d[%d]='%c'", step, b, c);
            conds.push_back({1, step, b, c, buf});
        }
    }

    // W22, W23
    for (int w : {22, 23}) {
        const char* pat = TABLE8_DW[w - T8_FIRST_W];
        for (int b = 0; b < 32; ++b) {
            char c = pat[31 - b];
            if (c == '=') continue;
            char buf[64];
            snprintf(buf, sizeof(buf), "W%d[%d]='%c'", w, b, c);
            conds.push_back({2, w, b, c, buf});
        }
    }

    return conds;
}

static inline bool check_cond(char c, Word x, Word xp, int b) {
    int bx  = (x  >> b) & 1;
    int bxp = (xp >> b) & 1;
    switch (c) {
        case 'n': return bx == 0 && bxp == 1;
        case 'u': return bx == 1 && bxp == 0;
        case '0': return bx == 0 && bxp == 0;
        case '1': return bx == 1 && bxp == 1;
        default:  return true;
    }
}

// Get register value from FullState
static inline Word get_reg(const FullState& fs, int reg, int step) {
    if (reg == 1) return fs.E[step + OFF];
    if (reg == 2) return fs.W[step];
    return fs.A[step + OFF];
}

// ═══════════════════════════════════════════════════════════════════════════════
// Part 4: Direct model validation (fix Δ, sample uniform x)
// ═══════════════════════════════════════════════════════════════════════════════

static void validate_carry_chain() {
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("VALIDATION 1: Carry-chain model vs direct sampling\n");
    printf("  Method: fix Δ, sample uniform x, check x' = x + Δ\n");
    printf("═══════════════════════════════════════════════════════════════\n\n");

    const uint64_t N = 10'000'000;
    std::mt19937 rng(12345);
    std::uniform_int_distribution<uint32_t> dist;

    // Test with several representative Δ values
    uint32_t deltas[] = {
        0x00000001,  // single low bit
        0x80000000,  // single high bit
        0x0000FF00,  // byte in middle
        0xA8BCF700,  // complex pattern (from 36-step example)
        0xFFFFFFFF,  // all ones (-1)
        0x00406800,  // sparse pattern
    };

    for (uint32_t delta : deltas) {
        CarryChain cc;
        cc.compute(delta);

        // For each of the 4 condition types, test at bit 15 (middle) and bit 3 (low)
        struct Test { char cond; int bit; };
        Test tests[] = {{'1', 15}, {'0', 15}, {'n', 15}, {'u', 15},
                        {'1',  3}, {'0',  3}, {'n',  3}, {'u',  3},
                        {'1', 28}, {'0', 28}};

        printf("Δ = 0x%08X:\n", delta);
        printf("  %-12s %8s %8s %8s\n", "Condition", "Predict", "Actual", "Error");

        for (auto& t : tests) {
            double predicted = cc.p_cond(t.bit, t.cond, delta);

            uint64_t pass = 0;
            for (uint64_t i = 0; i < N; ++i) {
                uint32_t x = dist(rng);
                uint32_t xp = x + delta;
                if (check_cond(t.cond, x, xp, t.bit)) ++pass;
            }
            double actual = (double)pass / (double)N;
            double err = fabs(predicted - actual);

            printf("  [%d]='%c'    %8.6f %8.6f %8.6f%s\n",
                   t.bit, t.cond, predicted, actual, err,
                   (err > 0.005) ? " *** MISMATCH ***" : "");
        }
        printf("\n");
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Part 5: SFS pair analysis
// ═══════════════════════════════════════════════════════════════════════════════

static void sfs_pair_analysis(const std::vector<DiffCond>& conds) {
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("SFS PAIR: Δ_mod extraction and carry-chain prediction\n");
    printf("═══════════════════════════════════════════════════════════════\n\n");

    // Table 9 SFS pair
    State cv = {0x0f4252be, 0xe2d5d87a, 0x56b40fde, 0x6ab7e678,
                0x3dfb0dcf, 0x6ac73c9d, 0x587ec5f5, 0x67bbd7dc};
    Word M[16] = {0x5a69a8e4, 0x45ff466e, 0xafec8126, 0x2d74afe7,
                  0x54780c76, 0x94b9dae7, 0x675ce76b, 0x107ffeb9,
                  0xbe7baa67, 0x2653bae8, 0x45b576c8, 0x0de40fc1,
                  0x2d9ea187, 0x26b93c1b, 0x31f1ac39, 0x24de0094};
    Word Mp[16];
    apply_diff_37(M, Mp);
    FullState fs  = compute_full(cv, M);
    FullState fsp = compute_full(cv, Mp);

    // Extract Δ_mod at each relevant register
    printf("Modular differences (Δ_mod = register' - register):\n");
    struct RegInfo { int reg; int step; const char* name; };
    RegInfo regs[] = {
        {1, 16, "E16"}, {1, 17, "E17"}, {1, 18, "E18"},
        {1, 19, "E19"}, {1, 20, "E20"}, {1, 21, "E21"},
        {2, 22, "W22"}, {2, 23, "W23"},
    };

    for (auto& ri : regs) {
        Word v  = get_reg(fs, ri.reg, ri.step);
        Word vp = get_reg(fsp, ri.reg, ri.step);
        uint32_t delta_mod = vp - v;
        uint32_t delta_xor = v ^ vp;
        int hw_mod = __builtin_popcount(delta_mod);
        int hw_xor = __builtin_popcount(delta_xor);
        printf("  %s: Δ_mod = 0x%08X (HW=%d), Δ_xor = 0x%08X (HW=%d)\n",
               ri.name, delta_mod, hw_mod, delta_xor, hw_xor);
    }
    printf("\n");

    // Per-condition carry-chain prediction
    printf("Per-condition carry-chain prediction (SFS pair Δ_mod):\n");
    printf("  %-5s %-20s %6s %8s %8s %8s\n",
           "Index", "Condition", "Reg", "Δ_mod", "P(pred)", "cost");
    printf("  %-5s %-20s %6s %8s %8s %8s\n",
           "-----", "--------------------", "------", "--------", "--------", "--------");

    double sum_predicted = 0;
    for (int i = 0; i < (int)conds.size(); ++i) {
        auto& dc = conds[i];
        Word v  = get_reg(fs, dc.reg, dc.step);
        Word vp = get_reg(fsp, dc.reg, dc.step);
        uint32_t delta_mod = vp - v;

        CarryChain cc;
        cc.compute(delta_mod);
        double pred = cc.p_cond(dc.bit, dc.cond, delta_mod);
        double cost = (pred > 0) ? -log2(pred) : 99.0;
        sum_predicted += cost;

        const char* rname = (dc.reg == 1) ? "E" : "W";
        printf("  β[%2d] %-20s %s%-4d 0x%08X %8.6f %8.3f\n",
               i, dc.label.c_str(), rname, dc.step, delta_mod, pred, cost);
    }
    printf("\n  SFS pair predicted β = %.1f bits (paper: %d, unconstrained: ~%.0f)\n\n",
           sum_predicted, (int)conds.size(), conds.size() * 2.0);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Part 6: Monte Carlo validation (random CV+M → carry-chain vs actual)
// ═══════════════════════════════════════════════════════════════════════════════

static void monte_carlo_validation(const std::vector<DiffCond>& conds) {
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("VALIDATION 2: Carry-chain prediction vs actual (random CV+M)\n");
    printf("  For each trial: compute Δ_mod at register, apply carry chain,\n");
    printf("  predict P(condition), compare to actual pass/fail.\n");
    printf("═══════════════════════════════════════════════════════════════\n\n");

    const uint64_t N = 500'000;
    std::mt19937 rng(42);
    std::uniform_int_distribution<uint32_t> dist;

    int nc = (int)conds.size();
    std::vector<double> sum_predicted(nc, 0.0);
    std::vector<uint64_t> actual_pass(nc, 0);

    // Also collect Δ_mod statistics per register
    struct RegStat {
        double sum_hw = 0;
        uint64_t count = 0;
    };
    // Map (reg, step) → stat
    std::vector<RegStat> reg_stats(8); // E16..E21, W22, W23
    auto reg_idx = [](int reg, int step) -> int {
        if (reg == 1) return step - 16;      // E16→0, E21→5
        if (reg == 2) return step - 22 + 6;  // W22→6, W23→7
        return -1;
    };

    for (uint64_t t = 0; t < N; ++t) {
        State cv;
        for (int j = 0; j < 8; ++j) cv[j] = dist(rng);
        Word M[16], Mp[16];
        for (int j = 0; j < 16; ++j) M[j] = dist(rng);
        apply_diff_37(M, Mp);

        FullState fs  = compute_full(cv, M);
        FullState fsp = compute_full(cv, Mp);

        // Compute Δ_mod at each register (cache per trial)
        uint32_t delta_cache[8]; // E16..E21, W22, W23
        bool delta_computed[8] = {};
        CarryChain cc_cache[8];

        for (int i = 0; i < nc; ++i) {
            auto& dc = conds[i];
            int ridx = reg_idx(dc.reg, dc.step);

            if (!delta_computed[ridx]) {
                Word v  = get_reg(fs, dc.reg, dc.step);
                Word vp = get_reg(fsp, dc.reg, dc.step);
                delta_cache[ridx] = vp - v;
                cc_cache[ridx].compute(delta_cache[ridx]);
                delta_computed[ridx] = true;
                reg_stats[ridx].sum_hw += __builtin_popcount(delta_cache[ridx]);
                reg_stats[ridx].count++;
            }

            // Carry-chain prediction
            double pred = cc_cache[ridx].p_cond(dc.bit, dc.cond, delta_cache[ridx]);
            sum_predicted[i] += pred;

            // Actual check
            Word v  = get_reg(fs, dc.reg, dc.step);
            Word vp = get_reg(fsp, dc.reg, dc.step);
            if (check_cond(dc.cond, v, vp, dc.bit))
                actual_pass[i]++;
        }
    }

    // Print Δ_mod statistics
    printf("Δ_mod Hamming weight statistics (across %llu trials):\n", (unsigned long long)N);
    const char* reg_names[] = {"E16", "E17", "E18", "E19", "E20", "E21", "W22", "W23"};
    for (int r = 0; r < 8; ++r) {
        if (reg_stats[r].count > 0) {
            printf("  %s: avg HW(Δ_mod) = %.1f / 32\n",
                   reg_names[r], reg_stats[r].sum_hw / reg_stats[r].count);
        }
    }
    printf("\n");

    // Per-condition comparison
    printf("Per-condition comparison: predicted vs actual\n");
    printf("  %-5s %-20s %8s %8s %8s %8s %8s\n",
           "Index", "Condition", "P(pred)", "P(act)", "bits_p", "bits_a", "error");
    printf("  %-5s %-20s %8s %8s %8s %8s %8s\n",
           "-----", "--------------------", "--------", "--------", "--------", "--------", "--------");

    double total_pred_bits = 0, total_act_bits = 0;
    double max_err = 0;
    for (int i = 0; i < nc; ++i) {
        double p_pred = sum_predicted[i] / N;
        double p_act  = (double)actual_pass[i] / N;
        double bits_p = (p_pred > 0) ? -log2(p_pred) : 99.0;
        double bits_a = (p_act  > 0) ? -log2(p_act)  : 99.0;
        double err = fabs(bits_p - bits_a);
        max_err = std::max(max_err, err);
        total_pred_bits += bits_p;
        total_act_bits  += bits_a;

        printf("  β[%2d] %-20s %8.6f %8.6f %8.3f %8.3f %8.4f%s\n",
               i, conds[i].label.c_str(), p_pred, p_act, bits_p, bits_a, err,
               (err > 0.05) ? " ***" : "");
    }

    printf("\n");
    printf("Aggregate:\n");
    printf("  Predicted β_eff = %.2f bits\n", total_pred_bits);
    printf("  Actual β_eff    = %.2f bits\n", total_act_bits);
    printf("  Max per-cond error = %.4f bits\n", max_err);
    printf("  Avg predicted bits/diff-cond = %.3f\n", total_pred_bits / nc);
    printf("  Avg actual bits/diff-cond    = %.3f\n", total_act_bits / nc);
    printf("  Paper assumption: 1.000 bits/diff-cond\n\n");
}

// ═══════════════════════════════════════════════════════════════════════════════
// Part 7: Theoretical analysis — average cost for uniform Δ
// ═══════════════════════════════════════════════════════════════════════════════

static void theoretical_analysis() {
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("THEORY: Expected cost per condition for random Δ_mod\n");
    printf("═══════════════════════════════════════════════════════════════\n\n");

    // Two metrics:
    //   -log2(E[P(cond|Δ)]): correct for attack complexity (joint probability)
    //   E[-log2(P(cond|Δ))]: per-Δ cost averaged — inflated by Jensen's inequality
    // The attack sees the PRODUCT of probabilities across conditions and starting
    // points, so -log2(E[P]) is the right metric.

    const uint64_t N = 1'000'000;
    std::mt19937 rng(999);
    std::uniform_int_distribution<uint32_t> dist;

    struct TestCase { char cond; int bit; };
    TestCase cases[] = {
        {'1', 1}, {'1', 3}, {'1', 7}, {'1', 15}, {'1', 24}, {'1', 31},
        {'n', 1}, {'n', 3}, {'n', 7}, {'n', 15}, {'n', 24}, {'n', 31},
        {'0', 7}, {'0', 15}, {'u', 7}, {'u', 15},
    };

    printf("  %-12s %10s %10s %12s\n",
           "Condition", "E[P]", "-log2(E[P])", "E[-log2(P)]");
    printf("  %-12s %10s %10s %12s\n",
           "------------", "----------", "----------", "------------");

    for (auto& tc : cases) {
        double sum_cost = 0;
        double sum_p = 0;
        int valid = 0;
        CarryChain cc;
        for (uint64_t t = 0; t < N; ++t) {
            uint32_t delta = dist(rng);
            cc.compute(delta);
            double p = cc.p_cond(tc.bit, tc.cond, delta);
            sum_p += p;
            if (p > 1e-30) { sum_cost += -log2(p); ++valid; }
        }
        double avg_p = sum_p / N;
        double cost_correct = -log2(avg_p);
        double cost_jensen = (valid > 0) ? sum_cost / valid : 99.0;
        char buf[32];
        snprintf(buf, sizeof(buf), "[%d]='%c'", tc.bit, tc.cond);
        printf("  %-12s %10.6f %10.4f %12.4f\n",
               buf, avg_p, cost_correct, cost_jensen);
    }

    printf("\n  -log2(E[P]) ≈ 2.0 for all positions (the attack-relevant metric).\n");
    printf("  E[-log2(P)|P>0] can be < 2.0 at low bits (P=0 cases excluded).\n");
    printf("  Unconditional E[-log2(P)] is +\u221e when P(P=0)>0 (low bit positions).\n");
    printf("  The attack's complexity depends on E[P], not E[-log2(P)].\n\n");
}

// ═══════════════════════════════════════════════════════════════════════════════
// Main
// ═══════════════════════════════════════════════════════════════════════════════

int main() {
    printf("Carry-Chain Markov Model for β-Condition Cost\n");
    printf("Zhang et al. 2026, 37-Step SHA-256\n");
    printf("══════════════════════════════════════════════════════════════\n\n");

    // Build diff conditions (37 total, the ones affected by carry chains)
    auto conds = build_diff_conditions();
    printf("Diff conditions in β: %d\n\n", (int)conds.size());

    // Part 1: Direct model validation
    validate_carry_chain();

    // Part 2: SFS pair analysis
    sfs_pair_analysis(conds);

    // Part 3: Monte Carlo (random CV+M)
    monte_carlo_validation(conds);

    // Part 4: Theoretical analysis
    theoretical_analysis();

    // Summary
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("SUMMARY\n");
    printf("═══════════════════════════════════════════════════════════════\n\n");
    printf("The carry-chain Markov model explains the β discrepancy:\n\n");
    printf("  Regime           | bits/diff | β_diff | β_total | Mechanism\n");
    printf("  -----------------+-----------+--------+---------+-----------\n");
    printf("  Paper (ideal)    |    1.00   |   37   |   95    | carries determined\n");
    printf("  SFS pair (model) |    1.07   |   39.6 |   97.6  | small residual carry\n");
    printf("  Constrained (36s)|   ~1.71   |   ~63  |   ~79   | partial constraint\n");
    printf("  Unconstrained    |   ~2.00   |   ~74  |  ~132   | carry equilibrium\n");
    printf("\n");
    printf("The paper's 1-bit-per-condition is achieved when Δ_mod → 0 at the\n");
    printf("β registers. The SFS pair nearly achieves this (E17,E20,E21 have\n");
    printf("Δ_mod=0). Residual cost from Δ_mod ≠ 0: E16(+0.01), E18(+1.72),\n");
    printf("W22(+0.48), W23(+0.42) — totaling 2.6 extra bits over the ideal 37.\n");
    printf("\n");
    printf("For random starting points (unconstrained), Δ_mod is random with\n");
    printf("avg HW ≈ 16/32, pushing carries to equilibrium (P=1/2) and making\n");
    printf("each diff condition cost 2 bits instead of 1.\n");
    printf("\n");
    printf("The constrained regime (TAB entry) partially determines Δ_mod,\n");
    printf("giving intermediate cost. The exact value depends on the TAB\n");
    printf("entry's Δ_mod distribution (not measured here — requires pipeline).\n");
    printf("\nDone.\n");

    return 0;
}
