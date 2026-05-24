// Benchmark the Table-3 (full characteristic) starting point with CaDiCaL.
// For SFS: skip encode_iv (the SP enforces IV equality via DASH conditions on
// A[-4..-1] and E[-4..-1]; H0 itself is free).
//
// Build and usage: see README.md in this directory.
// Default conflict_limit = 100_000_000 (100M).

#include "cnf_encoder_512.hpp"
#include "starting_points_512.hpp"
#include "cadical.hpp"
#include <chrono>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstdint>

int main(int argc, char** argv) {
    long long conflict_limit = (argc > 1) ? atoll(argv[1]) : 100'000'000LL;

    cnf512::SHA512Encoder enc(38);
    auto sp = eichlseder2014::starting_point_38_sfs();
    enc.encode_starting_point(sp);
    enc.encode_steps();
    enc.encode_msg_expansion();
    enc.encode_collision();
    enc.print_stats();

    CaDiCaL::Solver solver;
    enc.add_to_solver(solver);

    printf("\nSimplifying (3 rounds)...\n");
    auto t0 = std::chrono::high_resolution_clock::now();
    int simp_status = solver.simplify(3);
    auto t1 = std::chrono::high_resolution_clock::now();
    double simp_dt = std::chrono::duration<double>(t1 - t0).count();
    printf("Simplify: status=%d in %.2fs\n", simp_status, simp_dt);
    printf("  Active vars: %d\n", solver.active());
    printf("  Irredundant clauses: %lld\n", (long long)solver.irredundant());

    if (simp_status == 20) {
        printf("\nUNSAT during simplify — SP is infeasible!\n");
        return 2;
    }
    if (simp_status == 10) {
        printf("\nSAT during simplify — collision found in preprocessing!\n");
        return 0;
    }

    printf("\nSolving (limit %lld conflicts)...\n", conflict_limit);
    int cap = (conflict_limit > INT_MAX) ? INT_MAX : (int)conflict_limit;
    solver.limit("conflicts", cap);
    auto s0 = std::chrono::high_resolution_clock::now();
    int status = solver.solve();
    auto s1 = std::chrono::high_resolution_clock::now();
    double solve_dt = std::chrono::duration<double>(s1 - s0).count();
    long long confl = (long long)solver.get_statistic_value("conflicts");

    printf("\nResult: %s in %.1fs, %lld conflicts (%.0f conflicts/s)\n",
           status == 10 ? "SAT" : (status == 20 ? "UNSAT" : "TIMEOUT"),
           solve_dt, confl, solve_dt > 0 ? confl / solve_dt : 0.0);

    if (status == 10) {
        // Extract H0 (both copies' state at step -1..-4)
        auto extract_word = [&](const std::vector<int>& bits) {
            uint64_t w = 0;
            for (int j = 0; j < 64; j++) if (solver.val(bits[j]) > 0) w |= (uint64_t(1) << j);
            return w;
        };
        sha512::State H0_f{}, H0_g{};
        for (int r = 0; r < 4; r++) {
            H0_f[r]   = extract_word(enc.vars.A(-1 - r, false));
            H0_g[r]   = extract_word(enc.vars.A(-1 - r, true));
            H0_f[4+r] = extract_word(enc.vars.E(-1 - r, false));
            H0_g[4+r] = extract_word(enc.vars.E(-1 - r, true));
        }
        printf("\nH0 (copy 1):");
        for (int i = 0; i < 8; i++) printf(" %016llx", (unsigned long long)H0_f[i]);
        printf("\nH0 (copy 2):");
        for (int i = 0; i < 8; i++) printf(" %016llx", (unsigned long long)H0_g[i]);
        printf("\n  Equal? %s\n", H0_f == H0_g ? "YES (true SFS collision)" : "NO");

        // Extract messages
        uint64_t M1[16], M2[16];
        enc.extract_message([&](int v){ return solver.val(v); }, M1, M2, 16);
        printf("\nCollision messages:\n");
        printf("  M1:");
        for (int i = 0; i < 16; i++) {
            if (i % 4 == 0) printf("\n    ");
            printf(" %016llx", (unsigned long long)M1[i]);
        }
        printf("\n  M2:");
        for (int i = 0; i < 16; i++) {
            if (i % 4 == 0) printf("\n    ");
            printf(" %016llx", (unsigned long long)M2[i]);
        }
        printf("\n  Diff words: ");
        int n_diff = 0;
        for (int i = 0; i < 16; i++) if (M1[i] != M2[i]) {
            printf("W[%d]=%016llx ", i, (unsigned long long)(M1[i] ^ M2[i]));
            n_diff++;
        }
        printf("(%d nonzero)\n", n_diff);
    }
    return status == 10 ? 0 : 1;
}
