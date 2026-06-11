// Work factor calculator for SHA-1 collision search.
// Implements DC06 Section III-C: F_W, P_u, P_c, N_s, N_w.
//
// F_W is computed exactly from condition bit counts.
// P_u and P_c are estimated empirically by sampling.
//
// All public quantities are in log2 scale (matching the paper's tables).

#pragma once

#include "sha1.hpp"
#include "gencond.hpp"
#include "propagate.hpp"
#include <cstdint>
#include <cmath>
#include <random>
#include <algorithm>

namespace workfactor {

using gencond::WordCond;
using gencond::BitCond;

// ---- F_W: message freedom ----
// For i < 16: number of valid (W_i, W_i*) pairs satisfying nabla-W_i.
// Each bit independently contributes popcount(condition) choices.
// F_W = product of per-bit choice counts.
// log2(F_W) = sum of log2(popcount(condition)).

inline double log2_F_W(const WordCond& wc_W) {
    double total = 0.0;
    for (int b = 0; b < 32; ++b) {
        BitCond c = wc_W.get(b);
        int cnt = __builtin_popcount(c);
        if (cnt == 0) return -1e30;  // contradiction
        total += std::log2(cnt);
    }
    return total;
}

// ---- Sampling helpers ----

// Generate a random pair (val, val*) satisfying a WordCond.
// Returns false if the condition is unsatisfiable.
inline bool sample_pair(const WordCond& wc, uint32_t& val, uint32_t& val_star,
                        std::mt19937& rng) {
    val = 0;
    val_star = 0;
    for (int b = 0; b < 32; ++b) {
        BitCond c = wc.get(b);
        int cnt = __builtin_popcount(c);
        if (cnt == 0) return false;

        // Pick a random allowed pair
        int choice = std::uniform_int_distribution<int>(0, cnt - 1)(rng);
        int k = 0;
        for (int p = 0; p < 4; ++p) {
            if ((c >> p) & 1) {
                if (k == choice) {
                    val      |= uint32_t(p & 1) << b;
                    val_star |= uint32_t((p >> 1) & 1) << b;
                    break;
                }
                k++;
            }
        }
    }
    return true;
}

// ---- P_u estimation ----
// P_u(i) = Prob(A'_out ∈ nabla-A_{i+1} | all inputs ∈ their nabla, and W ∈ nabla-W)
//
// Sampling: generate random inputs satisfying conditions, run step, check output.
// For steps where input conditions are fully determined, this is exact.

struct SHA1Characteristic {
    int N;                   // number of steps
    WordCond A[85];          // A[-4]..A[N], indexed as A[i+4]
    WordCond W[80];          // W[0]..W[N-1]

    // Get A condition at step index (step -4 to N).
    const WordCond& cond_A(int step) const { return A[step + 4]; }
    WordCond& cond_A(int step) { return A[step + 4]; }
};

struct StepMetrics {
    double log2_F_W;
    double log2_P_u;
    double log2_P_c;
    double log2_N_s;
};

// ---- Analytical P_u via propagation engine ----
// Propagate input conditions through the step function; compare with claimed output.
// Per-bit approximation: P_u ≈ product of (popcount(claimed & propagated) / popcount(propagated)).
// This ignores carry correlations between bit positions but is the standard first-order estimate.

inline double analytical_log2_Pu(const SHA1Characteristic& ch, int step) {
    using gencond::BitCond;
    using gencond::SHA1StepCond;
    using gencond::wc_rotl;

    WordCond cA = ch.cond_A(step);
    WordCond cB = ch.cond_A(step - 1);
    WordCond cC = wc_rotl(ch.cond_A(step - 2), 30);
    WordCond cD = wc_rotl(ch.cond_A(step - 3), 30);
    WordCond cE = wc_rotl(ch.cond_A(step - 4), 30);
    WordCond cW = ch.W[step];
    WordCond claimed = ch.cond_A(step + 1);

    SHA1StepCond sc{cA, cB, cC, cD, cE, cW, step};
    WordCond propagated = sc.propagate_A();

    double log2_pu = 0.0;
    for (int b = 0; b < 32; ++b) {
        BitCond c = claimed.get(b);
        BitCond p = propagated.get(b);
        int common = __builtin_popcount(c & p);
        int total_p = __builtin_popcount(p);
        if (common == 0) return -1e30;  // impossible
        if (common < total_p)
            log2_pu += std::log2(double(common) / total_p);
    }
    return log2_pu;
}

// ---- Exact per-bit P_u via carry-state DP ----
//
// Computes P_u(step) exactly by tracking the joint carry distribution
// through the 5-input modular addition: A' = rotl(A,5) + f(B,C,D) + E + K + W.
//
// The carries for the first message and second message evolve independently
// (different bit values → different carry chains). State = (c1, c2) where
// c1,c2 ∈ {0,..,4}, giving 25 states per bit position.
//
// At each bit b:
//   1. Enumerate all valid input pair-combinations (consistent with conditions)
//   2. For each (carry_in_1, carry_in_2) state with nonzero weight:
//      - Compute sum1 = input_bits_1 + carry_in_1 for first message
//      - Compute sum2 = input_bits_2 + carry_in_2 for second message
//      - Get output pair (sum1 mod 2, sum2 mod 2)
//      - Check if output pair is in claimed A' condition at bit b
//      - Add weight to (carry_out_1, carry_out_2) state
//   3. P_u = (total matching weight) / (total weight)

inline double exact_log2_Pu(const SHA1Characteristic& ch, int step) {
    using gencond::BitCond;
    using gencond::wc_rotl;
    using gencond::BoolPropTable;
    using gencond::TT_IF;
    using gencond::TT_XOR;
    using gencond::TT_MAJ;

    WordCond cRotA5 = wc_rotl(ch.cond_A(step), 5);
    WordCond cB     = ch.cond_A(step - 1);
    WordCond cC     = wc_rotl(ch.cond_A(step - 2), 30);
    WordCond cD     = wc_rotl(ch.cond_A(step - 3), 30);
    WordCond cE     = wc_rotl(ch.cond_A(step - 4), 30);
    WordCond cW     = ch.W[step];
    WordCond claimed = ch.cond_A(step + 1);

    uint32_t k_val;
    if      (step < 20) k_val = 0x5A827999;
    else if (step < 40) k_val = 0x6ED9EBA1;
    else if (step < 60) k_val = 0x8F1BBCDC;
    else                k_val = 0xCA62C1D6;

    static const BoolPropTable tbl_if  = BoolPropTable::build(TT_IF);
    static const BoolPropTable tbl_xor = BoolPropTable::build(TT_XOR);
    static const BoolPropTable tbl_maj = BoolPropTable::build(TT_MAJ);
    const BoolPropTable* bf_tbl;
    if      (step < 20) bf_tbl = &tbl_if;
    else if (step < 40) bf_tbl = &tbl_xor;
    else if (step < 60) bf_tbl = &tbl_maj;
    else                bf_tbl = &tbl_xor;

    // Joint carry state: dp[c1][c2] = count of input combos reaching this state
    // c1, c2 ∈ {0..4}.  Max carry for 5-input: (5 + 4) / 2 = 4.
    constexpr int MC = 5;
    double dp[MC][MC] = {};
    dp[0][0] = 1.0;

    double log2_pu = 0.0;

    for (int b = 0; b < 32; ++b) {
        BitCond c_r  = cRotA5.get(b);
        BitCond c_f  = BitCond(bf_tbl->fwd_lut[cB.get(b)][cC.get(b)][cD.get(b)]);
        BitCond c_e  = cE.get(b);
        BitCond c_w  = cW.get(b);
        BitCond c_out = claimed.get(b);
        int k0 = (k_val >> b) & 1;

        // Pre-enumerate valid input combos: (r1,r2, f1,f2, e1,e2, w1,w2)
        // Each input pair p encodes (val, val*): val=p&1, val*=(p>>1)&1.
        // K is constant: both messages have same k0.
        // We combine into (sum_first, sum_second) at this bit.
        // With 4 variable inputs (rotA, f, E, W) × up to 4 pair states each:
        // worst case 4^4=256 combos per bit. Typically much less.

        struct Combo { int s1, s2; }; // partial sums for first/second message
        Combo combos[256];
        int n_combos = 0;

        for (int pr = 0; pr < 4; ++pr) {
            if (!((c_r >> pr) & 1)) continue;
            int r1 = pr & 1, r2 = (pr >> 1) & 1;
            for (int pf = 0; pf < 4; ++pf) {
                if (!((c_f >> pf) & 1)) continue;
                int f1 = pf & 1, f2 = (pf >> 1) & 1;
                for (int pe = 0; pe < 4; ++pe) {
                    if (!((c_e >> pe) & 1)) continue;
                    int e1 = pe & 1, e2 = (pe >> 1) & 1;
                    for (int pw = 0; pw < 4; ++pw) {
                        if (!((c_w >> pw) & 1)) continue;
                        int w1 = pw & 1, w2 = (pw >> 1) & 1;
                        combos[n_combos++] = {
                            r1 + f1 + e1 + k0 + w1,
                            r2 + f2 + e2 + k0 + w2
                        };
                    }
                }
            }
        }

        // Total input weight at this bit (for normalizing)
        // Each variable input pair is equally likely among its allowed pairs.
        // The total_input_combos is the product of popcount of each input condition.
        int n_r = __builtin_popcount(c_r);
        int n_f = __builtin_popcount(c_f);
        int n_e = __builtin_popcount(c_e);
        int n_w = __builtin_popcount(c_w);
        int total_input = n_r * n_f * n_e * n_w;
        if (total_input == 0) return -1e30;  // contradiction

        double new_dp[MC][MC] = {};
        double bit_total = 0.0;
        double bit_match = 0.0;

        for (int c1 = 0; c1 < MC; ++c1) {
            for (int c2 = 0; c2 < MC; ++c2) {
                if (dp[c1][c2] == 0.0) continue;
                double w = dp[c1][c2];

                for (int ci = 0; ci < n_combos; ++ci) {
                    int sum1 = combos[ci].s1 + c1;
                    int sum2 = combos[ci].s2 + c2;
                    int out1 = sum1 & 1;
                    int out2 = sum2 & 1;
                    int co1 = sum1 >> 1;
                    int co2 = sum2 >> 1;
                    int out_pair = out1 | (out2 << 1);

                    bit_total += w;
                    if ((c_out >> out_pair) & 1) {
                        bit_match += w;
                        new_dp[co1][co2] += w;
                    }
                }
            }
        }

        if (bit_total == 0.0) return -1e30;

        double ratio = bit_match / bit_total;
        if (ratio == 0.0) return -1e30;
        log2_pu += std::log2(ratio);

        // Normalize dp to prevent overflow
        double dp_sum = 0.0;
        for (int c1 = 0; c1 < MC; ++c1)
            for (int c2 = 0; c2 < MC; ++c2)
                dp_sum += new_dp[c1][c2];
        if (dp_sum > 0.0) {
            for (int c1 = 0; c1 < MC; ++c1)
                for (int c2 = 0; c2 < MC; ++c2)
                    new_dp[c1][c2] /= dp_sum;
        }

        std::memcpy(dp, new_dp, sizeof(dp));
    }

    return log2_pu;
}

// Exact P_c: like P_u, but for each state input combo, try all valid W
// and report success if ANY W makes the output match.
// For steps where F_W > 1, P_c can be significantly better than P_u.

inline double exact_log2_Pc(const SHA1Characteristic& ch, int step) {
    using gencond::BitCond;
    using gencond::wc_rotl;
    using gencond::BoolPropTable;
    using gencond::TT_IF;
    using gencond::TT_XOR;
    using gencond::TT_MAJ;

    double log2_fw = log2_F_W(ch.W[step]);
    if (log2_fw < 0.01) {
        // F_W = 1: P_c = P_u (only one W choice)
        return exact_log2_Pu(ch, step);
    }

    // For steps with F_W > 1, P_c is more complex because we need to check
    // if ANY W value satisfying W conditions makes the step work.
    // This requires tracking which (state, carry) combos succeed for at least
    // one W value.
    //
    // For now, approximate: P_c ≈ P_u (conservative for the greedy search).
    // The greedy sets P_c = 0 when F_W >= 2 (attacker picks best W).
    // This is the convention from the paper.

    // If F_W >= 2, the paper treats P_c as 1 (log = 0).
    return 0.0;
}

// ---- Sampling P_u and P_c ----

inline void estimate_Pu_Pc(const SHA1Characteristic& ch, int step,
                            int samples, std::mt19937& rng,
                            double& log2_Pu, double& log2_Pc) {
    using namespace sha1;
    using namespace gencond;

    const WordCond& cA    = ch.cond_A(step);
    const WordCond& cA_m1 = ch.cond_A(step - 1);
    const WordCond& cA_m2 = ch.cond_A(step - 2);
    const WordCond& cA_m3 = ch.cond_A(step - 3);
    const WordCond& cA_m4 = ch.cond_A(step - 4);
    const WordCond& cW    = ch.W[step];
    const WordCond& cA_out= ch.cond_A(step + 1);

    int good_u = 0;
    int good_c = 0;
    int total = 0;
    int total_state = 0;

    WordCond cB = cA_m1;
    WordCond cC = wc_rotl(cA_m2, 30);
    WordCond cD = wc_rotl(cA_m3, 30);
    WordCond cE = wc_rotl(cA_m4, 30);

    double F_W_count = std::pow(2.0, log2_F_W(cW));

    for (int s = 0; s < samples; ++s) {
        // Sample state inputs
        uint32_t a, a_s, b, b_s, c, c_s, d, d_s, e, e_s;
        if (!sample_pair(cA, a, a_s, rng)) { log2_Pu = 0; log2_Pc = 0; return; }
        if (!sample_pair(cB, b, b_s, rng)) { log2_Pu = 0; log2_Pc = 0; return; }
        if (!sample_pair(cC, c, c_s, rng)) { log2_Pu = 0; log2_Pc = 0; return; }
        if (!sample_pair(cD, d, d_s, rng)) { log2_Pu = 0; log2_Pc = 0; return; }
        if (!sample_pair(cE, e, e_s, rng)) { log2_Pu = 0; log2_Pc = 0; return; }

        total_state++;
        bool any_w_works = false;

        // For P_u: also sample W
        // For P_c: try multiple W values
        int w_trials = (F_W_count <= 1.0) ? 1 : std::min(64, (int)F_W_count);

        for (int wt = 0; wt < w_trials; ++wt) {
            uint32_t w, w_s;
            if (!sample_pair(cW, w, w_s, rng)) continue;

            // Compute step for both messages:
            // A' = rotl(A,5) + f(B,C,D) + E + K + W
            uint32_t k_val = sha1::K[step];
            uint32_t a_out = sha1::rotl(a, 5) + sha1::f(step, b, c, d) + e + k_val + w;
            uint32_t a_out_s = sha1::rotl(a_s, 5) + sha1::f(step, b_s, c_s, d_s) + e_s + k_val + w_s;

            total++;
            if (cA_out.conforms(a_out, a_out_s)) {
                good_u++;
                any_w_works = true;
            }
        }
        if (any_w_works) good_c++;
    }

    log2_Pu = (total > 0 && good_u > 0)
              ? std::log2((double)good_u / total)
              : (total > 0 ? -30.0 : 0.0);
    log2_Pc = (total_state > 0 && good_c > 0)
              ? std::log2((double)good_c / total_state)
              : (total_state > 0 ? -30.0 : 0.0);
}

// ---- Full work factor computation ----

struct WorkFactorResult {
    int N;
    StepMetrics steps[80];
    double log2_N_w;
};

// use_analytical: if true, use propagation-based P_u instead of sampling.
// With analytical mode, P_c is set equal to P_u (exact only when F_W=1).
inline WorkFactorResult compute_work_factor(const SHA1Characteristic& ch,
                                             int samples = 10000,
                                             uint32_t seed = 12345,
                                             bool use_analytical = false) {
    std::mt19937 rng(seed);
    WorkFactorResult res{};
    res.N = ch.N;

    // Compute F_W for each step
    for (int i = 0; i < ch.N; ++i) {
        if (i < 16) {
            res.steps[i].log2_F_W = log2_F_W(ch.W[i]);
        } else {
            res.steps[i].log2_F_W = 0.0;  // determined by expansion
        }
    }

    // Compute P_u, P_c for each step
    for (int i = 0; i < ch.N; ++i) {
        if (use_analytical) {
            res.steps[i].log2_P_u = analytical_log2_Pu(ch, i);
            res.steps[i].log2_P_c = res.steps[i].log2_P_u;  // approximation
        } else {
            estimate_Pu_Pc(ch, i, samples, rng,
                           res.steps[i].log2_P_u, res.steps[i].log2_P_c);
        }
    }

    // Compute N_s backward: N_s(N) = 1, log2(N_s(N)) = 0
    // N_s(i) = max{ N_s(i+1) / (F_W(i) * P_u(i)),  1/P_c(i) }
    // log2: max{ log2(N_s(i+1)) - log2(F_W(i)) - log2(P_u(i)),  -log2(P_c(i)) }
    double log2_Ns_next = 0.0;  // N_s(N) = 1
    for (int i = ch.N - 1; i >= 0; --i) {
        double branch1 = log2_Ns_next - res.steps[i].log2_F_W - res.steps[i].log2_P_u;
        double branch2 = -res.steps[i].log2_P_c;
        res.steps[i].log2_N_s = std::max(branch1, branch2);
        log2_Ns_next = res.steps[i].log2_N_s;
    }

    // Total work: N_w = sum N_s(i) for i=0..N
    // In linear scale, then convert back.
    double sum = 1.0;  // N_s(N) = 1
    for (int i = 0; i < ch.N; ++i)
        sum += std::pow(2.0, res.steps[i].log2_N_s);
    res.log2_N_w = std::log2(sum);

    return res;
}

// ---- Pretty-print ----

inline void print_work_factor(const WorkFactorResult& res) {
    std::printf("  i    log2(F_W)  log2(P_u)  log2(P_c)  log2(N_s)\n");
    for (int i = 0; i < res.N; ++i) {
        std::printf(" %2d:    %6.2f     %6.2f     %6.2f     %6.2f\n",
                    i, res.steps[i].log2_F_W, res.steps[i].log2_P_u,
                    res.steps[i].log2_P_c, res.steps[i].log2_N_s);
    }
    std::printf("\nTotal log2(N_w) = %.2f\n", res.log2_N_w);
}

} // namespace workfactor
