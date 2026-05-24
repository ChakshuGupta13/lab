// Head-to-head Table 6 (sparse SP) vs Table 7 (full characteristic) at
// a given n_fixed (number of message words pinned to Table 8 prefix).
// Same conflict limit, same blocking of the Table 8 known solution,
// measures time to first NEW collision (or timeout).
//
// Build and usage: see README.md in this directory.

#include "gencond.hpp"
#include "sha256.hpp"
#include "starting_points.hpp"
#include "cnf_encoder.hpp"
#include "cadical.hpp"
#include <cstdio>
#include <vector>
#include <chrono>
#include <cstdlib>
#include <climits>
#include <cstring>

using namespace gencond;
using namespace mendel2011;
using Word = uint32_t;

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

static CharState starting_point_27_table7_local() {
    CharState cs = starting_point_27();
    cs.e(5)  = WordCond::from_string("----------------1--------1------");
    cs.e(6)  = WordCond::from_string("-1--------0--0-10-1----0-0------");
    cs.a(7)  = WordCond::from_string("-------unn--u------n---nn-uuuu--");
    cs.e(7)  = WordCond::from_string("101-11---u10u1-0nuu-uuuu1n---n0-");
    cs.w(7)  = WordCond::from_string("00---1--un-0u-nuuuuu1-nu0n101n--");
    cs.a(8)  = WordCond::from_string("nnnnn-nnnn--------nuu-----------");
    cs.e(8)  = WordCond::from_string("0n0n001001u-1u1n01un010n01n00110");
    cs.w(8)  = WordCond::from_string("-----u--n---n---------nn--------");
    cs.a(9)  = WordCond::from_string("----un--n--nu-------nu-u--------");
    cs.e(9)  = WordCond::from_string("-1n1n1011u011100nn100u10-10000u-");
    cs.e(10) = WordCond::from_string("u00000nuuu10uun01u00n00n110-u-u1");
    cs.e(11) = WordCond::from_string("0n000uuuuu01010111n-uun01n000n01");
    cs.e(12) = WordCond::from_string("01---1010u01u----111-010-0--110-");
    cs.w(12) = WordCond::from_string("------110-u-------n0--u--n-n--nn");
    cs.e(13) = WordCond::from_string("01-10u1nunuuu---1110-1nn11---01-");
    cs.e(14) = WordCond::from_string("-----1-01011----------00--------");
    cs.e(15) = WordCond::from_string("-----1-001000---------11--------");
    cs.w(15) = WordCond::from_string("0u1-nn-n-u-1u---11un0uu10u101u0-");
    cs.w(17) = WordCond::from_string("---0-1nnn---u-1-----10uu0-------");
    return cs;
}

static bool verify_collision(const Word* m1, const Word* m2) {
    Word W1[64], W2[64];
    for (int i = 0; i < 16; i++) { W1[i] = m1[i]; W2[i] = m2[i]; }
    for (int i = 16; i < 27; i++) {
        W1[i] = sha256::sigma_1(W1[i-2]) + W1[i-7] + sha256::sigma_0(W1[i-15]) + W1[i-16];
        W2[i] = sha256::sigma_1(W2[i-2]) + W2[i-7] + sha256::sigma_0(W2[i-15]) + W2[i-16];
    }
    sha256::State s1 = sha256::IV, s2 = sha256::IV;
    for (int i = 0; i < 27; i++) { sha256::compress(s1, i, W1); sha256::compress(s2, i, W2); }
    bool eq = true;
    for (int i = 0; i < 8; i++) if (s1[i] != s2[i]) { eq = false; break; }
    return eq;
}

struct Run {
    const char* label;
    bool use_table7;
    int n_fixed;
    int64_t conflict_limit;
};

static void run_one(const Run& r) {
    printf("\n=== %s ===\n", r.label);
    CharState sp = r.use_table7 ? starting_point_27_table7_local() : starting_point_27();

    cnf::SHA256Encoder enc(27);
    enc.encode_iv(sha256::IV);
    std::vector<WordCond> A(sp.A.begin(), sp.A.begin() + 31);
    std::vector<WordCond> E(sp.E.begin(), sp.E.begin() + 31);
    std::vector<WordCond> W(sp.W.begin(), sp.W.begin() + 27);
    enc.encode_starting_point(A, E, W);
    enc.encode_steps();
    enc.encode_msg_expansion();
    enc.encode_collision();
    enc.encode_message(TABLE8_M1, TABLE8_M2, r.n_fixed);
    printf("Raw CNF: %d vars, %d clauses\n", enc.vars.next_var, enc.db.num_clauses());

    CaDiCaL::Solver solver;
    solver.set("factor", 0);
    enc.add_to_solver(solver);

    // Block Table 8 (force a new solution)
    const auto& vm = enc.vars;
    for (int i = r.n_fixed; i < 16; i++) {
        for (int j = 0; j < 32; j++) {
            bool b = (TABLE8_M1[i] >> j) & 1;
            solver.add(b ? -vm.w_f[i][j] : vm.w_f[i][j]);
        }
    }
    solver.add(0);

    // Phase-seed from Table 8 (same as v2)
    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 32; j++) {
            solver.phase(((TABLE8_M1[i] >> j) & 1) ? vm.w_f[i][j] : -vm.w_f[i][j]);
            if (vm.w_g[i][j] != vm.w_f[i][j])
                solver.phase(((TABLE8_M2[i] >> j) & 1) ? vm.w_g[i][j] : -vm.w_g[i][j]);
        }
    }

    solver.limit("conflicts", (int)std::min(r.conflict_limit, (int64_t)INT_MAX));
    auto t0 = std::chrono::high_resolution_clock::now();
    int status = solver.solve();
    auto t1 = std::chrono::high_resolution_clock::now();
    double dt = std::chrono::duration<double>(t1 - t0).count();
    long long confl = (long long)solver.get_statistic_value("conflicts");

    const char* st = status == 10 ? "SAT!" : (status == 20 ? "UNSAT" : "TIMEOUT");
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
        bool ok = verify_collision(m1, m2);
        printf("Verification: %s\n", ok ? "OK" : "FAIL");
        printf("M1: ");
        for (int i = 0; i < 16; i++) printf("%08x ", m1[i]);
        printf("\nM2: ");
        for (int i = 0; i < 16; i++) printf("%08x ", m2[i]);
        printf("\n");
    }
}

int main(int argc, char* argv[]) {
    int64_t cl = argc > 1 ? std::atoll(argv[1]) : 2000000LL;
    int n_fixed = argc > 2 ? std::atoi(argv[2]) : 7;
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    Run runs[] = {
        {"Table 6 (baseline)", false, n_fixed, cl},
        {"Table 7 (tightened)", true,  n_fixed, cl},
    };
    for (auto& r : runs) run_one(r);
    return 0;
}
