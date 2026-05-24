// 31-step head-to-head: Table 1 (sparse SP) vs Table 4 (full characteristic).
//
// SFS = Semi-Free-Start. SKIP encode_iv() — solver chooses IV consistent
// with the SP's DASH conditions on A[-4..-1] / E[-4..-1].
//
// Paper Table 7 example uses h_0 = 532f13f5 6a28c3c0 e301fab5 0 0 0 0 0
// (chosen to demonstrate the last-5-chaining-words = 0 case). The SAT
// run picks ANY valid h_0; the resulting (m, m*) is independently
// verified as an SFS collision via direct SHA-256.
//
// Build and usage: see README.md in this directory.

#include "sha256.hpp"
#include "gencond.hpp"
#include "sha256_starting_points.hpp"
#include "starting_points.hpp"
#include "cnf_encoder.hpp"
#include "cadical.hpp"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <chrono>
#include <climits>

using Word = uint32_t;
using namespace gencond;
using namespace mendel2011;
using namespace cnf;

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

static void encode_sfs(SHA256Encoder& enc, const CharState& sp, int n_steps) {
    // SFS: NO encode_iv(). DASH conditions on IV positions force equality
    // between the two copies; the value is solver-chosen.
    std::vector<WordCond> A(sp.A.begin(), sp.A.begin() + (n_steps + 4));
    std::vector<WordCond> E(sp.E.begin(), sp.E.begin() + (n_steps + 4));
    std::vector<WordCond> W(sp.W.begin(), sp.W.begin() + n_steps);
    enc.encode_starting_point(A, E, W);
    enc.encode_steps();
    enc.encode_msg_expansion();
    enc.encode_collision();
}

static bool verify_sfs(const Word IV_chosen[8], const Word* m1, const Word* m2, int n_steps) {
    Word W1[64] = {}, W2[64] = {};
    for (int i = 0; i < 16; i++) { W1[i] = m1[i]; W2[i] = m2[i]; }
    for (int i = 16; i < n_steps; i++) {
        W1[i] = sha256::sigma_1(W1[i-2]) + W1[i-7] + sha256::sigma_0(W1[i-15]) + W1[i-16];
        W2[i] = sha256::sigma_1(W2[i-2]) + W2[i-7] + sha256::sigma_0(W2[i-15]) + W2[i-16];
    }
    sha256::State s1, s2;
    for (int r = 0; r < 8; r++) { s1[r] = IV_chosen[r]; s2[r] = IV_chosen[r]; }
    for (int i = 0; i < n_steps; i++) {
        sha256::compress(s1, i, W1);
        sha256::compress(s2, i, W2);
    }
    for (int r = 0; r < 8; r++) if (s1[r] != s2[r]) return false;
    return true;
}

static void solve_one(const CharState& sp, int64_t conflict_limit, const char* label) {
    printf("\n=== %s ===\n", label);
    SHA256Encoder enc(31);
    encode_sfs(enc, sp, 31);
    printf("Raw CNF: %d vars, %d clauses\n", enc.vars.next_var, enc.db.num_clauses());

    CaDiCaL::Solver solver;
    solver.set("factor", 0);
    enc.add_to_solver(solver);
    const auto& vm = enc.vars;

    solver.limit("conflicts", (int)std::min(conflict_limit, (int64_t)INT_MAX));
    auto t0 = std::chrono::high_resolution_clock::now();
    int status = solver.solve();
    auto t1 = std::chrono::high_resolution_clock::now();
    double dt = std::chrono::duration<double>(t1 - t0).count();
    long long confl = (long long)solver.get_statistic_value("conflicts");
    const char* st = (status == 10 ? "SAT!" : (status == 20 ? "UNSAT" : "TIMEOUT"));
    printf("Result: %s in %.1fs, %lld conflicts\n", st, dt, confl);

    if (status == 10) {
        // Extract chosen IV: IV[0]=A_{-1}, IV[1]=A_{-2}, IV[2]=A_{-3}, IV[3]=A_{-4},
        //                   IV[4]=E_{-1}, IV[5]=E_{-2}, IV[6]=E_{-3}, IV[7]=E_{-4}.
        Word IV_x[8];
        for (int r = 0; r < 4; r++) {
            uint32_t va = 0, ve = 0;
            int step = -1 - r;
            for (int j = 0; j < 32; j++) {
                if (solver.val(vm.a_f[step + 4][j]) > 0) va |= (1u << j);
                if (solver.val(vm.e_f[step + 4][j]) > 0) ve |= (1u << j);
            }
            IV_x[r] = va;
            IV_x[r + 4] = ve;
        }
        Word m1[16], m2[16];
        for (int i = 0; i < 16; i++) {
            uint32_t v1 = 0, v2 = 0;
            for (int j = 0; j < 32; j++) {
                if (solver.val(vm.w_f[i][j]) > 0) v1 |= (1u << j);
                if (solver.val(vm.w_g[i][j]) > 0) v2 |= (1u << j);
            }
            m1[i] = v1; m2[i] = v2;
        }
        bool ok = verify_sfs(IV_x, m1, m2, 31);
        printf("Verification: %s\n", ok ? "OK" : "FAIL");
        printf("IV: ");
        for (int i = 0; i < 8; i++) printf("%08x ", IV_x[i]);
        printf("\nM1: ");
        for (int i = 0; i < 16; i++) printf("%08x ", m1[i]);
        printf("\nM2: ");
        for (int i = 0; i < 16; i++) printf("%08x ", m2[i]);
        printf("\n");
    }
}

int main(int argc, char* argv[]) {
    int64_t cl = (argc > 1) ? std::atoll(argv[1]) : 5000000LL;
    std::setvbuf(stdout, nullptr, _IOLBF, 0);

    printf("=== 31-step SFS: Table 1 (sparse SP) vs Table 4 (full characteristic) ===\n");
    printf("conflict_limit=%lld\n\n", (long long)cl);

    CharState t1 = mendel2013::starting_point_31();
    CharState t4 = mendel2013::starting_point_31_full();
    count_slack(t1, "Table 1 (sparse SP):", 31);
    count_slack(t4, "Table 4 (full):",      31);

    solve_one(t1, cl, "Table 1 (sparse SP)");
    solve_one(t4, cl, "Table 4 (full characteristic)");
    return 0;
}
