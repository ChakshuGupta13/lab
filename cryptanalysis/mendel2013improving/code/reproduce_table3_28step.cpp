// 28-step head-to-head: Table 1 (sparse SP) vs Table 3 (full characteristic).
//
// Standard SHA-256 IV (real collision). Block Table 6's known solution to
// force the SAT solver to find a NEW collision.
//
// Build and usage: see README.md in this directory.

#include "sha256.hpp"
#include "gencond.hpp"
#include "sha256_starting_points.hpp"          // sparse SP (Table 1)
#include "starting_points.hpp" // full SP (Table 3)
#include "cnf_encoder.hpp"
#include "cadical.hpp"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <chrono>
#include <climits>

using Word = uint32_t;
using namespace gencond;
using namespace mendel2011;  // CharState, WordCond, search.hpp types
using namespace cnf;         // SHA256Encoder

// Table 6 — 28-step real collision (paper-published, to block out).
static const Word TABLE6_M1[16] = {
    0x14c48440, 0xb3c3277f, 0xad69812d, 0xc3d4dffa,
    0x7eae690b, 0x7f9fe027, 0x832aece8, 0x9a489458,
    0x1607a45c, 0xdb81bdc8, 0x8786e031, 0xd8f22801,
    0x72b6be5e, 0x45a2652f, 0xf3fbb17a, 0x2ce70f52
};
static const Word TABLE6_M2[16] = {
    0x14c48440, 0xb3c3277f, 0xad69812d, 0xc3d4dffa,
    0x7eae690b, 0x7f9fe027, 0x832aece8, 0x9a489458,
    0xe6b2f4fc, 0xd759b930, 0x8786e031, 0xd8f22801,
    0x72b6be5e, 0x47e26dbf, 0xf3fbb17a, 0x2ce70f52
};

static bool verify_collision(const Word* m1, const Word* m2, int n_steps) {
    Word W1[64], W2[64];
    for (int i = 0; i < 16; i++) { W1[i] = m1[i]; W2[i] = m2[i]; }
    for (int i = 16; i < n_steps; i++) {
        W1[i] = sha256::sigma_1(W1[i-2]) + W1[i-7] + sha256::sigma_0(W1[i-15]) + W1[i-16];
        W2[i] = sha256::sigma_1(W2[i-2]) + W2[i-7] + sha256::sigma_0(W2[i-15]) + W2[i-16];
    }
    sha256::State s1 = sha256::IV, s2 = sha256::IV;
    for (int i = 0; i < n_steps; i++) { sha256::compress(s1, i, W1); sha256::compress(s2, i, W2); }
    for (int i = 0; i < 8; i++) if (s1[i] != s2[i]) return false;
    return true;
}

// Count slack over the slots the encoder actually consumes:
//   A, E: indices [-4 .. n_steps-1]  (vector indices 0 .. n_steps+3)
//   W:    indices [0  .. n_steps-1]  (vector indices 0 .. n_steps-1)
// Phantom W[n_steps..63] entries in CharState are not encoded.
static void count_slack(const CharState& cs, const char* label, int n_steps) {
    int q=0, x=0, dash=0, conc=0, un=0;
    auto count_field = [&](const WordCond& wc) {
        for (int b = 0; b < 32; b++) {
            BitCond c = wc.get(b);
            switch (c) {
                case BC_FREE: q++; break;
                case BC_X: x++; break;
                case BC_DASH: dash++; break;
                case BC_0: case BC_1: conc++; break;
                case BC_U: case BC_N: un++; break;
                default: break;
            }
        }
    };
    for (int i = 0; i < n_steps + 4; ++i) count_field(cs.A[i]);
    for (int i = 0; i < n_steps + 4; ++i) count_field(cs.E[i]);
    for (int i = 0; i < n_steps; ++i)     count_field(cs.W[i]);
    printf("%-30s ?=%d x=%d u/n=%d -=%d 0/1=%d\n", label, q, x, un, dash, conc);
}

struct Run {
    const char* label;
    CharState sp;
    int64_t conflict_limit;
    int n_fixed;
};

static void run_one(const Run& r) {
    printf("\n=== %s ===\n", r.label);
    SHA256Encoder enc(28);
    enc.encode_iv(sha256::IV);  // 28-step is a REAL collision, fix the IV
    std::vector<WordCond> A(r.sp.A.begin(), r.sp.A.begin() + 32);   // -4..27
    std::vector<WordCond> E(r.sp.E.begin(), r.sp.E.begin() + 32);
    std::vector<WordCond> W(r.sp.W.begin(), r.sp.W.begin() + 28);
    enc.encode_starting_point(A, E, W);
    enc.encode_steps();
    enc.encode_msg_expansion();
    enc.encode_collision();
    enc.encode_message(TABLE6_M1, TABLE6_M2, r.n_fixed);
    printf("Raw CNF: %d vars, %d clauses\n", enc.vars.next_var, enc.db.num_clauses());

    CaDiCaL::Solver solver;
    solver.set("factor", 0);
    enc.add_to_solver(solver);

    // Block Table 6's known solution to force NEW collision discovery.
    const auto& vm = enc.vars;
    for (int i = r.n_fixed; i < 16; i++) {
        for (int j = 0; j < 32; j++) {
            bool b = (TABLE6_M1[i] >> j) & 1;
            solver.add(b ? -vm.w_f[i][j] : vm.w_f[i][j]);
        }
    }
    solver.add(0);

    // Phase-seed toward Table 6 (guides solver toward the blocked solution's neighborhood).
    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 32; j++) {
            solver.phase(((TABLE6_M1[i] >> j) & 1) ? vm.w_f[i][j] : -vm.w_f[i][j]);
            if (vm.w_g[i][j] != vm.w_f[i][j])
                solver.phase(((TABLE6_M2[i] >> j) & 1) ? vm.w_g[i][j] : -vm.w_g[i][j]);
        }
    }

    solver.limit("conflicts", (int)std::min(r.conflict_limit, (int64_t)INT_MAX));
    auto t0 = std::chrono::high_resolution_clock::now();
    int status = solver.solve();
    auto t1 = std::chrono::high_resolution_clock::now();
    double dt = std::chrono::duration<double>(t1 - t0).count();
    long long confl = (long long)solver.get_statistic_value("conflicts");
    const char* st = (status == 10 ? "SAT!" : (status == 20 ? "UNSAT" : "TIMEOUT"));
    printf("Result: %s in %.1fs, %lld conflicts\n", st, dt, confl);

    if (status == 10) {
        Word m1[16], m2[16];
        for (int i = 0; i < 16; i++) {
            uint32_t v1 = 0, v2 = 0;
            for (int j = 0; j < 32; j++) {
                if (solver.val(vm.w_f[i][j]) > 0) v1 |= (1u << j);
                if (solver.val(vm.w_g[i][j]) > 0) v2 |= (1u << j);
            }
            m1[i] = v1; m2[i] = v2;
        }
        bool ok = verify_collision(m1, m2, 28);
        printf("Verification: %s\n", ok ? "OK" : "FAIL");
        printf("M1: ");
        for (int i = 0; i < 16; i++) printf("%08x ", m1[i]);
        printf("\nM2: ");
        for (int i = 0; i < 16; i++) printf("%08x ", m2[i]);
        printf("\n");
    }
}

int main(int argc, char* argv[]) {
    int64_t cl = (argc > 1) ? std::atoll(argv[1]) : 5000000LL;
    int n_fixed = (argc > 2) ? std::atoi(argv[2]) : 0;
    std::setvbuf(stdout, nullptr, _IOLBF, 0);

    printf("=== 28-step: Table 1 (sparse SP) vs Table 3 (full characteristic) ===\n");
    printf("conflict_limit=%lld  n_fixed=%d\n\n", (long long)cl, n_fixed);

    CharState t1 = mendel2013::starting_point_28();
    CharState t3 = mendel2013::starting_point_28_full();
    count_slack(t1, "Table 1 (sparse SP):", 28);
    count_slack(t3, "Table 3 (full):",      28);

    Run runs[] = {
        {"Table 1 (sparse SP) baseline", t1, cl, n_fixed},
        {"Table 3 (full characteristic)", t3, cl, n_fixed},
    };
    for (auto& r : runs) run_one(r);
    return 0;
}
