// CNF encoder for SHA-512 step-reduced semi-free-start collision attacks.
// Direct port of the SHA-256 (32-bit) CNF encoder from the mendel2011
// reproduction package to SHA-512 (64-bit) word width.
// SHA-512 (64-bit). Changes are mechanical:
//   - Word size 32 → 64
//   - Rotation constants updated for SHA-512 Sigma/sigma functions
//   - K table: sha256::K[64] → sha512::K[80]
//   - WordCond → WordCond64
//
// Architecture (unchanged from SHA-256 encoder):
//   VarMap: allocates Boolean variables for all state registers
//   ClauseDB: builds clauses for Tseitin gates and adders
//   SHA512Encoder: top-level encoder
//
// Usage:
//   SHA512Encoder enc(38);
//   enc.encode_starting_point(starting_point_38_sfs());  // SFS: no encode_iv
//   enc.encode_steps();
//   enc.encode_msg_expansion();
//   enc.encode_collision();
//   enc.add_to_solver(solver);

#pragma once

#include "sha512.hpp"
#include "gencond.hpp"
#include "search_512.hpp"
#include <vector>
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdint>
#include <string>

namespace cnf512 {

using gencond::BitCond;
using gencond::WordCond64;

// ---- Variable allocator and mapper ----

struct VarMap {
    int next_var = 1;
    int n_steps;

    std::vector<std::vector<int>> a_f, a_g, e_f, e_g;
    std::vector<std::vector<int>> w_f, w_g;

    struct StepAux {
        std::vector<int> sigma1_f, sigma1_g;
        std::vector<int> ch_f, ch_g;
        std::vector<int> t1a_f, t1a_g;
        std::vector<int> t1a_carry_f, t1a_carry_g;
        std::vector<int> t1b_f, t1b_g;
        std::vector<int> t1b_carry_f, t1b_carry_g;
        std::vector<int> t1c_f, t1c_g;
        std::vector<int> t1c_carry_f, t1c_carry_g;
        std::vector<int> enew_carry_f, enew_carry_g;
        std::vector<int> sigma0_f, sigma0_g;
        std::vector<int> maj_f, maj_g;
        std::vector<int> t2a_f, t2a_g;
        std::vector<int> t2a_carry_f, t2a_carry_g;
        std::vector<int> anew_carry_f, anew_carry_g;
        std::vector<int> kw_f, kw_g;
        std::vector<int> kw_carry_f, kw_carry_g;
    };
    std::vector<StepAux> step_aux;

    struct MsgExpAux {
        std::vector<int> s0_f, s0_g;
        std::vector<int> s1_f, s1_g;
        std::vector<int> me_a_f, me_a_g;
        std::vector<int> me_a_carry_f, me_a_carry_g;
        std::vector<int> me_b_f, me_b_g;
        std::vector<int> me_b_carry_f, me_b_carry_g;
        std::vector<int> me_c_carry_f, me_c_carry_g;
    };
    std::vector<MsgExpAux> msg_aux;

    int alloc(int n = 1) {
        int v = next_var;
        next_var += n;
        return v;
    }

    // 64-bit word: 64 variables
    std::vector<int> alloc_word() {
        std::vector<int> w(64);
        for (int j = 0; j < 64; j++) w[j] = alloc();
        return w;
    }

    // 65 carries for a 64-bit adder (carry[0] = 0 forced externally)
    std::vector<int> alloc_carries() {
        std::vector<int> c(65);
        for (int j = 0; j < 65; j++) c[j] = alloc();
        return c;
    }

    void init(int steps) {
        n_steps = steps;
        int n_a = steps + 4;

        a_f.resize(n_a); a_g.resize(n_a);
        e_f.resize(n_a); e_g.resize(n_a);
        for (int i = 0; i < n_a; i++) {
            a_f[i] = alloc_word(); a_g[i] = alloc_word();
            e_f[i] = alloc_word(); e_g[i] = alloc_word();
        }

        w_f.resize(steps); w_g.resize(steps);
        for (int i = 0; i < steps; i++) {
            w_f[i] = alloc_word(); w_g[i] = alloc_word();
        }

        step_aux.resize(steps);
        for (int i = 0; i < steps; i++) {
            auto& sa = step_aux[i];
            sa.sigma1_f = alloc_word(); sa.sigma1_g = alloc_word();
            sa.ch_f = alloc_word(); sa.ch_g = alloc_word();
            sa.t1a_f = alloc_word(); sa.t1a_g = alloc_word();
            sa.t1a_carry_f = alloc_carries(); sa.t1a_carry_g = alloc_carries();
            sa.t1b_f = alloc_word(); sa.t1b_g = alloc_word();
            sa.t1b_carry_f = alloc_carries(); sa.t1b_carry_g = alloc_carries();
            sa.kw_f = alloc_word(); sa.kw_g = alloc_word();
            sa.kw_carry_f = alloc_carries(); sa.kw_carry_g = alloc_carries();
            sa.t1c_f = alloc_word(); sa.t1c_g = alloc_word();
            sa.t1c_carry_f = alloc_carries(); sa.t1c_carry_g = alloc_carries();
            sa.enew_carry_f = alloc_carries(); sa.enew_carry_g = alloc_carries();
            sa.sigma0_f = alloc_word(); sa.sigma0_g = alloc_word();
            sa.maj_f = alloc_word(); sa.maj_g = alloc_word();
            sa.t2a_f = alloc_word(); sa.t2a_g = alloc_word();
            sa.t2a_carry_f = alloc_carries(); sa.t2a_carry_g = alloc_carries();
            sa.anew_carry_f = alloc_carries(); sa.anew_carry_g = alloc_carries();
        }

        if (steps > 16) {
            msg_aux.resize(steps - 16);
            for (int i = 16; i < steps; i++) {
                auto& ma = msg_aux[i - 16];
                ma.s0_f = alloc_word(); ma.s0_g = alloc_word();
                ma.s1_f = alloc_word(); ma.s1_g = alloc_word();
                ma.me_a_f = alloc_word(); ma.me_a_g = alloc_word();
                ma.me_a_carry_f = alloc_carries(); ma.me_a_carry_g = alloc_carries();
                ma.me_b_f = alloc_word(); ma.me_b_g = alloc_word();
                ma.me_b_carry_f = alloc_carries(); ma.me_b_carry_g = alloc_carries();
                ma.me_c_carry_f = alloc_carries(); ma.me_c_carry_g = alloc_carries();
            }
        }
    }

    const std::vector<int>& A(int step, bool copy2) const {
        return copy2 ? a_g[step + 4] : a_f[step + 4];
    }
    const std::vector<int>& E(int step, bool copy2) const {
        return copy2 ? e_g[step + 4] : e_f[step + 4];
    }
    const std::vector<int>& W(int step, bool copy2) const {
        return copy2 ? w_g[step] : w_f[step];
    }

    int total_vars() const { return next_var - 1; }
};

// ---- Clause emitter ----

struct ClauseDB {
    std::vector<std::vector<int>> clauses;

    void add(std::initializer_list<int> lits) { clauses.push_back(lits); }
    void add(const std::vector<int>& lits)    { clauses.push_back(lits); }
    void unit(int lit)                        { clauses.push_back({lit}); }
    void fix(int var, bool val)               { unit(val ? var : -var); }

    void equiv(int a, int b) {
        add({-a, b});
        add({a, -b});
    }

    void xor2(int a, int b) {
        add({a, b});
        add({-a, -b});
    }

    // 3-input XOR: out = a XOR b XOR c (8 clauses)
    void xor3(int out, int a, int b, int c) {
        add({ out,  a,  b, -c});
        add({ out,  a, -b,  c});
        add({ out, -a,  b,  c});
        add({ out, -a, -b, -c});
        add({-out,  a,  b,  c});
        add({-out,  a, -b, -c});
        add({-out, -a,  b, -c});
        add({-out, -a, -b,  c});
    }

    void ch_gate(int out, int x, int y, int z) {
        add({-x, -y,  out});
        add({-x,  y, -out});
        add({ x, -z,  out});
        add({ x,  z, -out});
    }

    void maj_gate(int out, int x, int y, int z) {
        add({-x, -y,  out});
        add({-x, -z,  out});
        add({-y, -z,  out});
        add({ x,  y, -out});
        add({ x,  z, -out});
        add({ y,  z, -out});
    }

    void full_adder(int sum, int cout, int a, int b, int cin) {
        xor3(sum, a, b, cin);
        maj_gate(cout, a, b, cin);
    }

    // 64-bit ripple-carry adder
    void adder64(const std::vector<int>& result,
                 const std::vector<int>& carry,
                 const std::vector<int>& A,
                 const std::vector<int>& B) {
        assert(result.size() == 64 && carry.size() == 65);
        assert(A.size() == 64 && B.size() == 64);
        for (int j = 0; j < 64; j++) {
            full_adder(result[j], carry[j + 1], A[j], B[j], carry[j]);
        }
    }

    // Allocate a constant 64-bit word.
    std::vector<int> make_const_word(uint64_t val, int& next_var) {
        std::vector<int> vars(64);
        for (int j = 0; j < 64; j++) {
            vars[j] = next_var++;
            fix(vars[j], (val >> j) & 1);
        }
        return vars;
    }

    // 3-way XOR with optional shift (vs rotation) on the third input.
    // SHA-512: sigma_0 = ROTR(1) ^ ROTR(8) ^ SHR(7)
    //          sigma_1 = ROTR(19) ^ ROTR(61) ^ SHR(6)
    //          Sigma_0 = ROTR(28) ^ ROTR(34) ^ ROTR(39)  (no shift)
    //          Sigma_1 = ROTR(14) ^ ROTR(18) ^ ROTR(41)  (no shift)
    void sigma_xor3(const std::vector<int>& out,
                    const std::vector<int>& x,
                    int r1, int r2, int r3,
                    bool shift3 = false) {
        for (int j = 0; j < 64; j++) {
            int a = x[(j + r1) % 64];
            int b = x[(j + r2) % 64];
            int c;
            if (shift3 && (j + r3 >= 64)) {
                // Shifted-out bit = 0 → out = a XOR b
                add({-out[j], -a, -b});
                add({-out[j],  a,  b});
                add({ out[j], -a,  b});
                add({ out[j],  a, -b});
                continue;
            } else {
                c = x[(j + r3) % 64];
            }
            xor3(out[j], a, b, c);
        }
    }

    int num_clauses() const { return (int)clauses.size(); }
};

// ---- Top-level encoder ----

struct SHA512Encoder {
    int n_steps;
    VarMap vars;
    ClauseDB db;

    explicit SHA512Encoder(int steps) : n_steps(steps) {
        vars.init(steps);
    }

    // Encode IV as unit clauses for both copies (standard SHA-512).
    // For SFS attacks, skip this — IV equality is enforced via DASH conditions.
    void encode_iv(const sha512::State& iv) {
        for (int r = 0; r < 4; r++) {
            int step = -1 - r;
            for (int j = 0; j < 64; j++) {
                bool bit = (iv[r] >> j) & 1;
                db.fix(vars.A(step, false)[j], bit);
                int ag = vars.A(step, true)[j];
                if (ag != vars.A(step, false)[j]) db.fix(ag, bit);
            }
        }
        for (int r = 0; r < 4; r++) {
            int step = -1 - r;
            for (int j = 0; j < 64; j++) {
                bool bit = (iv[4 + r] >> j) & 1;
                db.fix(vars.E(step, false)[j], bit);
                int eg = vars.E(step, true)[j];
                if (eg != vars.E(step, false)[j]) db.fix(eg, bit);
            }
        }
    }

    // Encode a single bit-pair condition.
    void encode_condition(int var_f, int var_g, BitCond cond) {
        if (var_f == var_g) {
            switch (cond) {
                case BitCond::BC_DASH: return;
                case BitCond::BC_FREE: return;
                case BitCond::BC_0: db.fix(var_f, false); return;
                case BitCond::BC_1: db.fix(var_f, true);  return;
                default: return;
            }
        }
        switch (cond) {
            case BitCond::BC_0: db.fix(var_f, false); db.fix(var_g, false); break;
            case BitCond::BC_1: db.fix(var_f, true);  db.fix(var_g, true);  break;
            case BitCond::BC_U: db.fix(var_f, true);  db.fix(var_g, false); break;
            case BitCond::BC_N: db.fix(var_f, false); db.fix(var_g, true);  break;
            case BitCond::BC_DASH: db.equiv(var_f, var_g); break;
            case BitCond::BC_X:    db.xor2(var_f, var_g); break;
            case BitCond::BC_FREE: break;
            case BitCond::BC_HASH: db.add({var_f}); db.add({-var_f}); break;
            default:
                encode_multival_condition(var_f, var_g, cond);
                break;
        }
    }

    void encode_multival_condition(int vf, int vg, BitCond cond) {
        uint8_t mask = static_cast<uint8_t>(cond);
        if (!(mask & 1)) db.add({ vf,  vg});  // disallow (0,0)
        if (!(mask & 2)) db.add({-vf,  vg});  // disallow (1,0)
        if (!(mask & 4)) db.add({ vf, -vg});  // disallow (0,1)
        if (!(mask & 8)) db.add({-vf, -vg});  // disallow (1,1)
    }

    // Encode starting point from CharState512.
    // Walks A (steps -4..n-1), E (steps -4..n-1), W (steps 0..n-1).
    void encode_starting_point(const eichlseder2014::CharState512& cs) {
        for (int step = -4; step < n_steps; ++step) {
            const auto& wa = cs.a(step);
            const auto& we = cs.e(step);
            for (int j = 0; j < 64; j++) {
                BitCond ca = wa.get(j);
                if (ca != BitCond::BC_FREE)
                    encode_condition(vars.a_f[step+4][j], vars.a_g[step+4][j], ca);
                BitCond ce = we.get(j);
                if (ce != BitCond::BC_FREE)
                    encode_condition(vars.e_f[step+4][j], vars.e_g[step+4][j], ce);
            }
        }
        for (int step = 0; step < n_steps; ++step) {
            const auto& ww = cs.w(step);
            for (int j = 0; j < 64; j++) {
                BitCond cw = ww.get(j);
                if (cw != BitCond::BC_FREE)
                    encode_condition(vars.w_f[step][j], vars.w_g[step][j], cw);
            }
        }
    }

    // Encode one copy of one SHA-512 step.
    // SHA-512 step: same structure as SHA-256, with different rotation constants.
    //   Sigma_1(E_{i-1}) = ROTR(14) XOR ROTR(18) XOR ROTR(41)
    //   Sigma_0(A_{i-1}) = ROTR(28) XOR ROTR(34) XOR ROTR(39)
    void encode_step_copy(int step, bool copy2) {
        auto& sa = vars.step_aux[step];
        auto& A_prev1 = vars.A(step - 1, copy2);
        auto& A_prev2 = vars.A(step - 2, copy2);
        auto& A_prev3 = vars.A(step - 3, copy2);
        auto& A_prev4 = vars.A(step - 4, copy2);
        auto& E_prev1 = vars.E(step - 1, copy2);
        auto& E_prev2 = vars.E(step - 2, copy2);
        auto& E_prev3 = vars.E(step - 3, copy2);
        auto& E_prev4 = vars.E(step - 4, copy2);
        auto& A_new = vars.A(step, copy2);
        auto& E_new = vars.E(step, copy2);
        auto& W_step = vars.W(step, copy2);

        auto& sigma1 = copy2 ? sa.sigma1_g : sa.sigma1_f;
        auto& ch     = copy2 ? sa.ch_g     : sa.ch_f;
        auto& sigma0 = copy2 ? sa.sigma0_g : sa.sigma0_f;
        auto& maj    = copy2 ? sa.maj_g    : sa.maj_f;
        auto& t1a    = copy2 ? sa.t1a_g    : sa.t1a_f;
        auto& t1a_c  = copy2 ? sa.t1a_carry_g : sa.t1a_carry_f;
        auto& t1b    = copy2 ? sa.t1b_g    : sa.t1b_f;
        auto& t1b_c  = copy2 ? sa.t1b_carry_g : sa.t1b_carry_f;
        auto& kw     = copy2 ? sa.kw_g     : sa.kw_f;
        auto& kw_c   = copy2 ? sa.kw_carry_g  : sa.kw_carry_f;
        auto& t1c    = copy2 ? sa.t1c_g    : sa.t1c_f;
        auto& t1c_c  = copy2 ? sa.t1c_carry_g : sa.t1c_carry_f;
        auto& enew_c = copy2 ? sa.enew_carry_g : sa.enew_carry_f;
        auto& t2a    = copy2 ? sa.t2a_g    : sa.t2a_f;
        auto& t2a_c  = copy2 ? sa.t2a_carry_g : sa.t2a_carry_f;
        auto& anew_c = copy2 ? sa.anew_carry_g : sa.anew_carry_f;

        // Sigma_1(E_{i-1}) = ROTR(14) XOR ROTR(18) XOR ROTR(41)
        db.sigma_xor3(sigma1, E_prev1, 14, 18, 41);

        // Ch(E_{i-1}, E_{i-2}, E_{i-3})
        for (int j = 0; j < 64; j++)
            db.ch_gate(ch[j], E_prev1[j], E_prev2[j], E_prev3[j]);

        // T1 = h + Sigma1 + Ch + K_i + W_i
        db.fix(t1a_c[0], false);
        db.adder64(t1a, t1a_c, E_prev4, sigma1);

        db.fix(t1b_c[0], false);
        db.adder64(t1b, t1b_c, t1a, ch);

        auto k_vars = db.make_const_word(sha512::K[step], vars.next_var);
        db.fix(kw_c[0], false);
        db.adder64(kw, kw_c, k_vars, W_step);

        db.fix(t1c_c[0], false);
        db.adder64(t1c, t1c_c, t1b, kw);

        // E_new = A_{i-4} + T1
        db.fix(enew_c[0], false);
        db.adder64(E_new, enew_c, A_prev4, t1c);

        // Sigma_0(A_{i-1}) = ROTR(28) XOR ROTR(34) XOR ROTR(39)
        db.sigma_xor3(sigma0, A_prev1, 28, 34, 39);

        // Maj(A_{i-1}, A_{i-2}, A_{i-3})
        for (int j = 0; j < 64; j++)
            db.maj_gate(maj[j], A_prev1[j], A_prev2[j], A_prev3[j]);

        // A_new = T1 + (Sigma0 + Maj)
        db.fix(t2a_c[0], false);
        db.adder64(t2a, t2a_c, sigma0, maj);

        db.fix(anew_c[0], false);
        db.adder64(A_new, anew_c, t1c, t2a);
    }

    // Encode message expansion for step >= 16.
    // SHA-512: sigma_0 = ROTR(1) XOR ROTR(8) XOR SHR(7)
    //          sigma_1 = ROTR(19) XOR ROTR(61) XOR SHR(6)
    void encode_msg_exp_copy(int step, bool copy2) {
        assert(step >= 16 && step < n_steps);
        auto& ma = vars.msg_aux[step - 16];
        auto& W_out = vars.W(step, copy2);
        auto& W_m2  = vars.W(step - 2, copy2);
        auto& W_m7  = vars.W(step - 7, copy2);
        auto& W_m15 = vars.W(step - 15, copy2);
        auto& W_m16 = vars.W(step - 16, copy2);

        auto& s1     = copy2 ? ma.s1_g       : ma.s1_f;
        auto& s0     = copy2 ? ma.s0_g       : ma.s0_f;
        auto& me_a   = copy2 ? ma.me_a_g     : ma.me_a_f;
        auto& me_a_c = copy2 ? ma.me_a_carry_g : ma.me_a_carry_f;
        auto& me_b   = copy2 ? ma.me_b_g     : ma.me_b_f;
        auto& me_b_c = copy2 ? ma.me_b_carry_g : ma.me_b_carry_f;
        auto& me_c_c = copy2 ? ma.me_c_carry_g : ma.me_c_carry_f;

        // sigma_1(W_{i-2}) = ROTR(19) XOR ROTR(61) XOR SHR(6)
        db.sigma_xor3(s1, W_m2, 19, 61, 6, true);

        // sigma_0(W_{i-15}) = ROTR(1) XOR ROTR(8) XOR SHR(7)
        db.sigma_xor3(s0, W_m15, 1, 8, 7, true);

        // W_i = s1 + W_{i-7} + s0 + W_{i-16}
        db.fix(me_a_c[0], false);
        db.adder64(me_a, me_a_c, s1, W_m7);

        db.fix(me_b_c[0], false);
        db.adder64(me_b, me_b_c, me_a, s0);

        db.fix(me_c_c[0], false);
        db.adder64(W_out, me_c_c, me_b, W_m16);
    }

    void encode_steps() {
        for (int i = 0; i < n_steps; i++) {
            encode_step_copy(i, false);
            encode_step_copy(i, true);
        }
    }

    void encode_msg_expansion() {
        for (int i = 16; i < n_steps; i++) {
            encode_msg_exp_copy(i, false);
            encode_msg_exp_copy(i, true);
        }
    }

    // Collision: final 4 (A, E) pairs equal between copies.
    void encode_collision() {
        for (int r = 0; r < 4; r++) {
            int step = n_steps - 1 - r;
            for (int j = 0; j < 64; j++) {
                int af = vars.A(step, false)[j], ag = vars.A(step, true)[j];
                int ef = vars.E(step, false)[j], eg = vars.E(step, true)[j];
                if (af != ag) db.equiv(af, ag);
                if (ef != eg) db.equiv(ef, eg);
            }
        }
    }

    // Encode concrete message words (both copies) as unit clauses.
    void encode_message(const uint64_t* m1, const uint64_t* m2, int n_words) {
        for (int i = 0; i < n_words && i < n_steps; i++) {
            for (int j = 0; j < 64; j++) {
                db.fix(vars.w_f[i][j], (m1[i] >> j) & 1);
                db.fix(vars.w_g[i][j], (m2[i] >> j) & 1);
            }
        }
    }

    // Encode concrete state at step = -1..-4 (both copies). For SFS sanity tests.
    void encode_state(const sha512::State& h0_f, const sha512::State& h0_g) {
        for (int r = 0; r < 4; r++) {
            int step = -1 - r;
            for (int j = 0; j < 64; j++) {
                db.fix(vars.A(step, false)[j], (h0_f[r] >> j) & 1);
                db.fix(vars.A(step, true)[j],  (h0_g[r] >> j) & 1);
                db.fix(vars.E(step, false)[j], (h0_f[4+r] >> j) & 1);
                db.fix(vars.E(step, true)[j],  (h0_g[4+r] >> j) & 1);
            }
        }
    }

    void write_dimacs(const char* path) const {
        FILE* f = fopen(path, "w");
        if (!f) { perror(path); return; }
        fprintf(f, "p cnf %d %d\n", vars.total_vars(), db.num_clauses());
        for (auto& cl : db.clauses) {
            for (int lit : cl) fprintf(f, "%d ", lit);
            fprintf(f, "0\n");
        }
        fclose(f);
    }

    template<typename Solver>
    void add_to_solver(Solver& solver) const {
        solver.resize(vars.total_vars());
        for (auto& cl : db.clauses) {
            for (int lit : cl) solver.add(lit);
            solver.add(0);
        }
    }

    template<typename F>
    void extract_message(const F& val_fn, uint64_t* m1, uint64_t* m2, int n_words) const {
        for (int i = 0; i < n_words && i < n_steps; i++) {
            m1[i] = 0; m2[i] = 0;
            for (int j = 0; j < 64; j++) {
                if (val_fn(vars.w_f[i][j]) > 0) m1[i] |= (uint64_t(1) << j);
                if (val_fn(vars.w_g[i][j]) > 0) m2[i] |= (uint64_t(1) << j);
            }
        }
    }

    void print_stats() const {
        printf("CNF stats: %d variables, %d clauses\n",
               vars.total_vars(), db.num_clauses());
        printf("  Primary: A(%d×2×64=%d) E(%d×2×64=%d) W(%d×2×64=%d)\n",
               n_steps + 4, (n_steps + 4) * 2 * 64,
               n_steps + 4, (n_steps + 4) * 2 * 64,
               n_steps, n_steps * 2 * 64);
    }
};

} // namespace cnf512
