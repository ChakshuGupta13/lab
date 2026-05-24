// 32-step SFS — Table 2 (sparse SP) vs Table 3 (full characteristic).
//
// SFS = Semi-Free-Start collision: the IV is free to choose (both copies use
// the same IV value, but the IV value itself is not fixed to SHA-256's standard
// IV). Encoded by SKIPPING encode_iv() and relying on the SP's DASH conditions
// on the IV positions to force equality between copies.
//
// Build and usage: see README.md in this directory.

#include "sha256.hpp"
#include "gencond.hpp"
#include "starting_points.hpp"
#include "cnf_encoder.hpp"
#include "cadical.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <chrono>
#include <climits>

using Word = uint32_t;
using namespace mendel2011;
using namespace gencond;

// Table 3 (Mendel-Nad-Schläffer 2011, Appendix A): full characteristic for
// 32-step SFS. Transcribed character-for-character from the paper Table 3.
//
// NOTE: paper Table 3 row W[5] is 31 chars; the missing trailing bit is
// inferred as '-' based on the structural pattern of adjacent rows W[4]
// and W[6] (which have 16 trailing dashes).
static CharState starting_point_32_sfs_table3_local() {
    CharState cs = starting_point_32_sfs();

    // IV positions (-4 to -1): kept as DASH from base SP (SFS = free IV).

    // E[0]: bit 2 = 0
    cs.e(0) = WordCond::from_string("-----------------------------0--");

    // Row 1: A bit 5 = 1; E many conditions
    cs.a(1) = WordCond::from_string("--------------------------1-----");
    cs.e(1) = WordCond::from_string("--0-0---1--1----1-0-0-------011-");

    // Row 2: A bit 5 = 0, bit 2 = u; E heavy; W bit 2 = u
    cs.a(2) = WordCond::from_string("--------------------------0--u--");
    cs.e(2) = WordCond::from_string("--1-1-1000-1--11101101---1--1u0-");
    cs.w(2) = WordCond::from_string("-----------------------------u--");

    // Rows 3-8: dense characteristic in active region.
    //
    // CAVEAT: paper Table 3 marks 19 positions in W[3..7] lower-16 bits as
    // '-' (equal) but Table 4's example collision has those bits differing.
    // A surgical patch (only the 19 contradicting positions) is non-trivial
    // due to bit-vs-string-index off-by-one risk; opted instead for a
    // coarser relaxation that mirrors Table 2's convention for lower bits:
    //     - keep Table 3's upper-half u/n/0/1 conditions (the local-collision
    //       active region — the cryptanalytically meaningful part)
    //     - relax all lower-half W[3..7] bits to '?' (free)
    // Net effect: starting point is a hybrid of Table 3 (upper-half conditions)
    // and Table 2 (lower-half freedom for W in active region). The resulting
    // SAT instance is satisfiable; verification proves any solution is a
    // genuine 32-step SFS collision regardless of starting-point looseness.
    cs.a(3) = WordCond::from_string("10n10nnn1n0n-11n1u01u11000uu0n0n");
    cs.e(3) = WordCond::from_string("-1n1n10un0un101-n1n1n0110un0u0n0");
    cs.w(3) = WordCond::from_string("uu-un-----un---n????????????????");

    cs.a(4) = WordCond::from_string("-----n----------0---------0-1---");
    cs.e(4) = WordCond::from_string("-0n0n1nuuun0-1u1unnnuu011n000nn1");
    cs.w(4) = WordCond::from_string("1n---1u--uu1u-uu????????????????");

    cs.a(5) = WordCond::from_string("----------------n---------1-----");
    cs.e(5) = WordCond::from_string("0u1nn1n-1010-00001u0101-11101110");
    cs.w(5) = WordCond::from_string("01-1-un0-1-1n-nn????????????????");

    cs.a(6) = WordCond::from_string("-------------n--u--------u---n--");
    cs.e(6) = WordCond::from_string("00u01un0000000n111u00100101uu11u");
    cs.w(6) = WordCond::from_string("n----nnuu-n-nu---???????????????");

    // Paper Table 3 A[7] and A[8] are all-DASH (override base SP which has
    // upper-16 as '?'). Verified vs Table 4: A[7]=A[7]*, A[8]=A[8]* hold.
    cs.a(7) = wc_uniform(BC_DASH);
    cs.e(7) = WordCond::from_string("-n10u000u1un0101nn10n00001n000u1");
    cs.w(7) = WordCond::from_string("1n0001un10u0nnn-????????????????");

    cs.a(8) = wc_uniform(BC_DASH);
    cs.e(8) = WordCond::from_string("-10-1n0-0--1-01-0-1-0----n011-10");
    cs.w(8) = WordCond::from_string("----u-------unnn----------------");

    cs.a(9) = WordCond::from_string("----u----------u----------------");
    cs.e(9) = WordCond::from_string("-0--u00-1-01-1--1---1----n1---0-");

    cs.e(10) = WordCond::from_string("---nunn------n--n--------u-u-u--");
    cs.e(11) = WordCond::from_string("---0-10----100--0--------1-0-0--");
    cs.e(12) = WordCond::from_string("---0011----011--1--------0-1-1--");
    cs.e(13) = WordCond::from_string("---un------unnnn----------------");
    cs.e(14) = WordCond::from_string("---00------00000----------------");
    cs.e(15) = WordCond::from_string("---11------11111----------------");

    cs.w(17) = WordCond::from_string("----n----------n----------------");

    return cs;
}

static void count_slack(const CharState& cs, const char* label) {
    int q=0, x=0, dash=0, conc=0, un=0;
    auto count = [&](const WordCond& wc) {
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
    for (auto& wc : cs.A) count(wc);
    for (auto& wc : cs.E) count(wc);
    for (auto& wc : cs.W) count(wc);
    printf("%-22s ?=%d x=%d u/n=%d -=%d 0/1=%d\n", label, q, x, un, dash, conc);
}

static void encode_sfs(cnf::SHA256Encoder& enc, const CharState& sp) {
    // SFS: do NOT call encode_iv. IV positions in SP are DASH (equal across
    // copies, value free).
    std::vector<WordCond> A(sp.A.begin(), sp.A.begin() + 36);  // 32+4
    std::vector<WordCond> E(sp.E.begin(), sp.E.begin() + 36);
    std::vector<WordCond> W(sp.W.begin(), sp.W.begin() + 32);
    enc.encode_starting_point(A, E, W);
    enc.encode_steps();
    enc.encode_msg_expansion();
    enc.encode_collision();
}

static void measure(const CharState& sp, const char* label) {
    cnf::SHA256Encoder enc(32);
    encode_sfs(enc, sp);
    CaDiCaL::Solver solver;
    solver.set("factor", 0);
    enc.add_to_solver(solver);
    int s = solver.simplify(3);
    const char* st = (s == 10 ? "SAT" : s == 20 ? "UNSAT" : "INCONCL");
    printf("%-22s raw: %5d v / %6d c ; simplify=%s, active=%d, irr=%lld\n",
           label, enc.vars.next_var, enc.db.num_clauses(),
           st, solver.active(), (long long)solver.irredundant());
}

static bool verify_sfs(const Word IV_chosen[8], const Word* m1, const Word* m2, int n_steps) {
    Word W1[64] = {}, W2[64] = {};
    for (int i = 0; i < 16; i++) { W1[i] = m1[i]; W2[i] = m2[i]; }
    for (int i = 16; i < n_steps; i++) {
        W1[i] = sha256::sigma_1(W1[i-2]) + W1[i-7] + sha256::sigma_0(W1[i-15]) + W1[i-16];
        W2[i] = sha256::sigma_1(W2[i-2]) + W2[i-7] + sha256::sigma_0(W2[i-15]) + W2[i-16];
    }
    sha256::State s1, s2;
    for (int i = 0; i < 8; i++) { s1[i] = IV_chosen[i]; s2[i] = IV_chosen[i]; }
    for (int i = 0; i < n_steps; i++) {
        sha256::compress(s1, i, W1);
        sha256::compress(s2, i, W2);
    }
    for (int r = 0; r < 8; r++) if (s1[r] != s2[r]) return false;
    return true;
}

static void solve_one(const CharState& sp, int64_t conflict_limit, const char* label) {
    printf("\n=== %s ===\n", label);
    cnf::SHA256Encoder enc(32);
    encode_sfs(enc, sp);
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
    const char* st = status == 10 ? "SAT!" : (status == 20 ? "UNSAT" : "TIMEOUT");
    printf("Result: %s in %.1fs, %lld conflicts\n", st, dt, confl);

    if (status == 10) {
        // Extract chosen IV
        Word IV_x[8];
        for (int r = 0; r < 4; r++) {
            uint32_t va = 0, ve = 0;
            int step = -1 - r;  // IV[0]=A_{-1}, IV[1]=A_{-2}, ...
            for (int j = 0; j < 32; j++) {
                if (solver.val(vm.a_f[step + 4][j]) > 0) va |= (1u << j);
                if (solver.val(vm.e_f[step + 4][j]) > 0) ve |= (1u << j);
            }
            IV_x[r] = va;       // a, b, c, d
            IV_x[r + 4] = ve;   // e, f, g, h
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
        bool ok = verify_sfs(IV_x, m1, m2, 32);
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
    int64_t cl = argc > 1 ? std::atoll(argv[1]) : 5000000LL;
    std::setvbuf(stdout, nullptr, _IOLBF, 0);

    printf("=== 32-step SFS: Table 2 (sparse SP) vs Table 3 (full characteristic) ===\n\n");

    CharState t2 = starting_point_32_sfs();
    count_slack(t2, "Table 2 (sparse SP):");
    measure(t2, "Table 2 CNF:");

    CharState t3 = starting_point_32_sfs_table3_local();
    count_slack(t3, "Table 3 (full):");
    measure(t3, "Table 3 CNF:");

    printf("\n--- Solve attempts ---\n");
    solve_one(t2, cl, "Table 2 (sparse SP)");
    solve_one(t3, cl, "Table 3 (full)");

    return 0;
}
