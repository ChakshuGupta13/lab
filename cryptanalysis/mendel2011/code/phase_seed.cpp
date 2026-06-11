// Phase-seeded CaDiCaL: bias VSIDS polarity toward Table 8 values.
//
// Zero-overhead technique: solver.phase(lit) sets the initial polarity
// preference for a variable. VSIDS still controls variable ordering.
// Tests whether "search near known solution" helps at harder n-fixed levels.
//
// Build:
//   g++ -std=c++17 -O2 -Wall -Wextra -Wno-trigraphs -I. \
//       -Ideps/cadical/src -Ldeps/cadical/build -o build/scratch_phase_seed \
//       src/mendel2011/scratch/scratch_phase_seed.cpp -lcadical -pthread
//
// Usage: ./build/scratch_phase_seed [n_fixed] [conflict_limit] [seed]

#include "sha256.hpp"
#include "gencond.hpp"
#include "cnf_encoder.hpp"
#include "starting_points.hpp"
#include "cadical.hpp"
#include <cstdio>
#include <chrono>
#include <cstdlib>
#include <climits>

using Word = sha256::Word;
using namespace mendel2011;

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

static bool verify_collision(const Word* m1, const Word* m2, int steps) {
    Word W1[64] = {}, W2[64] = {};
    for (int i = 0; i < 16; i++) { W1[i] = m1[i]; W2[i] = m2[i]; }
    for (int i = 16; i < steps; i++) {
        W1[i] = sha256::sigma_1(W1[i-2]) + W1[i-7] + sha256::sigma_0(W1[i-15]) + W1[i-16];
        W2[i] = sha256::sigma_1(W2[i-2]) + W2[i-7] + sha256::sigma_0(W2[i-15]) + W2[i-16];
    }
    sha256::State s1 = sha256::IV, s2 = sha256::IV;
    for (int i = 0; i < steps; i++) {
        sha256::compress(s1, i, W1);
        sha256::compress(s2, i, W2);
    }
    for (int r = 0; r < 8; r++)
        if (s1[r] != s2[r]) return false;
    return true;
}

struct RunResult {
    int status;
    double time_s;
    long long conflicts;
    long long decisions;
    bool valid;
    int ham_m1;  // Hamming distance of found M1 from TABLE8_M1 (free words only)
    int ham_m2;  // Hamming distance of found M2 from TABLE8_M2 (free words only)
};

// Extended run that can block Table 8 and report what solution was found
RunResult run_test_extended(bool phase_seed, int n_fixed, int64_t conflict_limit, int seed, bool block_table8) {
    RunResult r = {};

    CharState sp = starting_point_27();
    cnf::SHA256Encoder enc(27);
    enc.encode_iv(sha256::IV);

    std::vector<gencond::WordCond> A, E, W;
    int n_a = 27 + 4;
    A.resize(n_a); E.resize(n_a); W.resize(27);
    for (int i = 0; i < n_a; i++) { A[i] = sp.A[i]; E[i] = sp.E[i]; }
    for (int i = 0; i < 27; i++) W[i] = sp.W[i];

    enc.encode_starting_point(A, E, W);
    enc.encode_steps();
    enc.encode_msg_expansion();
    enc.encode_collision();
    enc.encode_message(TABLE8_M1, TABLE8_M2, n_fixed);

    CaDiCaL::Solver solver;
    if (seed != 0) solver.set("seed", seed);
    enc.add_to_solver(solver);
    solver.limit("conflicts", static_cast<int>(std::min(conflict_limit, (int64_t)INT_MAX)));

    if (block_table8) {
        // Add clause: at least one free message bit must differ from Table 8
        // This blocks the exact Table 8 assignment for the unfixed words
        const auto& vm = enc.vars;
        for (int i = n_fixed; i < 16; i++) {
            for (int j = 0; j < 32; j++) {
                bool b1 = (TABLE8_M1[i] >> j) & 1;
                // If Table 8 bit is 1, add negative literal (force at least one to be 0)
                solver.add(b1 ? -vm.w_f[i][j] : vm.w_f[i][j]);
            }
        }
        solver.add(0); // terminate clause
    }

    if (phase_seed) {
        const auto& vm = enc.vars;
        for (int i = 0; i < 16; i++) {
            for (int j = 0; j < 32; j++) {
                bool b1 = (TABLE8_M1[i] >> j) & 1;
                bool b2 = (TABLE8_M2[i] >> j) & 1;
                solver.phase(b1 ? vm.w_f[i][j] : -vm.w_f[i][j]);
                solver.phase(b2 ? vm.w_g[i][j] : -vm.w_g[i][j]);
            }
        }
        // Also phase-seed state registers
        Word W1[64], W2[64];
        for (int i = 0; i < 16; i++) { W1[i] = TABLE8_M1[i]; W2[i] = TABLE8_M2[i]; }
        for (int i = 16; i < 27; i++) {
            W1[i] = sha256::sigma_1(W1[i-2]) + W1[i-7] + sha256::sigma_0(W1[i-15]) + W1[i-16];
            W2[i] = sha256::sigma_1(W2[i-2]) + W2[i-7] + sha256::sigma_0(W2[i-15]) + W2[i-16];
        }
        sha256::State s1 = sha256::IV, s2 = sha256::IV;
        for (int step = 0; step < 27; step++) {
            sha256::compress(s1, step, W1);
            sha256::compress(s2, step, W2);
            for (int j = 0; j < 32; j++) {
                solver.phase(((s1[0] >> j) & 1) ? vm.a_f[step+4][j] : -vm.a_f[step+4][j]);
                solver.phase(((s2[0] >> j) & 1) ? vm.a_g[step+4][j] : -vm.a_g[step+4][j]);
                solver.phase(((s1[4] >> j) & 1) ? vm.e_f[step+4][j] : -vm.e_f[step+4][j]);
                solver.phase(((s2[4] >> j) & 1) ? vm.e_g[step+4][j] : -vm.e_g[step+4][j]);
            }
        }
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    r.status = solver.solve();
    auto t1 = std::chrono::high_resolution_clock::now();
    r.time_s = std::chrono::duration<double>(t1 - t0).count();

    r.conflicts = solver.get_statistic_value("conflicts");
    r.decisions = solver.get_statistic_value("decisions");

    if (r.status == 10) {
        Word m1[16], m2[16];
        enc.extract_message([&](int v) { return solver.val(v); }, m1, m2, 16);
        r.valid = verify_collision(m1, m2, 27);
        // Hamming distance from Table 8 (free words only)
        r.ham_m1 = 0; r.ham_m2 = 0;
        for (int i = n_fixed; i < 16; i++) {
            r.ham_m1 += __builtin_popcount(m1[i] ^ TABLE8_M1[i]);
            r.ham_m2 += __builtin_popcount(m2[i] ^ TABLE8_M2[i]);
        }
    }
    return r;
}

int main(int argc, char* argv[]) {
    int n_fixed = 10;
    int64_t conflict_limit = 10000000LL;
    int seed = 0;
    int mode_filter = -1;  // -1 = all, 0-3 = specific

    if (argc > 1) n_fixed = std::atoi(argv[1]);
    if (argc > 2) conflict_limit = std::atoll(argv[2]);
    if (argc > 3) seed = std::atoi(argv[3]);
    if (argc > 4) mode_filter = std::atoi(argv[4]);

    printf("=== Phase-Seed Diagnostic: %d-fixed, limit %lld, seed %d ===\n\n", n_fixed, conflict_limit, seed);
    printf("%-30s | %8s | %10s | %10s | %8s | %s\n", "Mode", "Result", "Conflicts", "Decisions", "Time", "Ham(M1/M2)");
    printf("-------------------------------+----------+------------+------------+---------+-----------\n");

    struct TestCase { const char* name; bool phase; bool block; };
    TestCase cases[] = {
        {"Plain",                     false, false},
        {"Phase-seed (Table 8)",      true,  false},
        {"Phase-seed + block T8",     true,  true },
        {"Plain + block T8",          false, true },
    };

    for (int i = 0; i < 4; i++) {
        if (mode_filter >= 0 && i != mode_filter) continue;
        auto& tc = cases[i];
        auto r = run_test_extended(tc.phase, n_fixed, conflict_limit, seed, tc.block);
        const char* status = (r.status == 10) ? "SAT" : (r.status == 20) ? "UNSAT" : "UNKNOWN";
        printf("%-30s | %5s %2s | %10lld | %10lld | %6.1fs | %d/%d\n",
               tc.name, status, r.valid ? "ok" : "",
               r.conflicts, r.decisions, r.time_s, r.ham_m1, r.ham_m2);
    }

    return 0;
}
