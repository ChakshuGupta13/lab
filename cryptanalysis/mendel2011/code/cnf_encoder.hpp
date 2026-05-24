// CNF encoder for SHA-256 step-reduced collision attacks.
// Encodes two parallel copies of SHA-256 (m, m') with:
//   - Standard SHA-256 step function constraints (Tseitin)
//   - Message expansion constraints
//   - Starting point conditions (from gencond.hpp CharState)
//   - IV constraints (standard or free for SFS)
//   - Collision condition: H(m) = H(m')
//
// Architecture:
//   VarMap: allocates Boolean variables for all state registers
//   ClauseEmitter: builds clauses for Tseitin gates and adders
//   SHA256Encoder: top-level encoder combining all components
//
// Usage:
//   SHA256Encoder enc(27);        // 27-step attack
//   enc.encode_iv(sha256::IV);    // standard IV (both copies)
//   enc.encode_starting_point(starting_point_27());
//   enc.encode_steps();           // SHA-256 step constraints
//   enc.encode_collision();       // output equality
//   enc.write_dimacs("27step.cnf");
//   // Or: enc.add_to_solver(solver); for API-based use.

#pragma once

#include "sha256.hpp"
#include "gencond.hpp"
#include <vector>
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdint>
#include <string>

namespace cnf {

// ---- Variable allocator and mapper ----

struct VarMap {
    int next_var = 1;
    int n_steps;

    // Primary variables: two copies (f = first block, g = second block)
    // Index: [copy][step_offset][bit], where step_offset is relative
    // For A: indices -4..n_steps-1 (offset +4), for E: same
    // For W: indices 0..n_steps-1

    // a_f[step+4][bit], a_g[step+4][bit]  (step = -4..n_steps-1)
    // e_f[step+4][bit], e_g[step+4][bit]
    // w_f[step][bit], w_g[step][bit]       (step = 0..n_steps-1)
    std::vector<std::vector<int>> a_f, a_g, e_f, e_g;
    std::vector<std::vector<int>> w_f, w_g;

    // Auxiliary variables for step function intermediates
    // Per step: Sigma1, Ch, T1_carries, Sigma0, Maj, T2, A_carries
    // Each for both copies
    struct StepAux {
        // Sigma1(E_{i-1}): 32 output bits (XOR of 3 rotations — no aux needed,
        // but for Tseitin we allocate output variables)
        std::vector<int> sigma1_f, sigma1_g;   // 32 bits each
        // Ch(E_{i-1}, E_{i-2}, E_{i-3}): 32 output bits
        std::vector<int> ch_f, ch_g;
        // T1 = h + Sigma1 + Ch + K + W: 32-bit adder with carries
        // We compute T1 as a chain of additions, each with carry variables
        // T1a = h + Sigma1 (32 carries)
        // T1b = T1a + Ch (32 carries)
        // T1c = T1b + K_i (no variables — K is constant, fold into clause)
        // Actually: simpler to use a single 5-input addition with carry chain
        // Let's use sequential 2-input adders to keep clauses simple.
        std::vector<int> t1a_f, t1a_g;        // h + Sigma1 result (32)
        std::vector<int> t1a_carry_f, t1a_carry_g; // 33 carries (c0=0)
        std::vector<int> t1b_f, t1b_g;        // t1a + Ch result
        std::vector<int> t1b_carry_f, t1b_carry_g;
        std::vector<int> t1c_f, t1c_g;        // t1b + K+W result
        std::vector<int> t1c_carry_f, t1c_carry_g;
        // E_new = d + T1: carries
        std::vector<int> enew_carry_f, enew_carry_g;
        // Sigma0(A_{i-1}): 32 output bits
        std::vector<int> sigma0_f, sigma0_g;
        // Maj(A_{i-1}, A_{i-2}, A_{i-3}): 32 output bits
        std::vector<int> maj_f, maj_g;
        // A_new = T1 + Sigma0 + Maj: cascaded adders
        std::vector<int> t2a_f, t2a_g;        // Sigma0 + Maj result
        std::vector<int> t2a_carry_f, t2a_carry_g;
        std::vector<int> anew_carry_f, anew_carry_g; // T1c + t2a
        // K+W combined: 32-bit sum + carries (K is constant per step)
        std::vector<int> kw_f, kw_g;
        std::vector<int> kw_carry_f, kw_carry_g;
    };
    std::vector<StepAux> step_aux;

    // Message expansion auxiliaries (for step >= 16)
    struct MsgExpAux {
        std::vector<int> s0_f, s0_g;   // sigma_0 output (32 bits)
        std::vector<int> s1_f, s1_g;   // sigma_1 output
        // W_i = s1 + W_{i-7} + s0 + W_{i-16}: cascaded adders
        std::vector<int> me_a_f, me_a_g;     // s1 + W_{i-7}
        std::vector<int> me_a_carry_f, me_a_carry_g;
        std::vector<int> me_b_f, me_b_g;     // me_a + s0
        std::vector<int> me_b_carry_f, me_b_carry_g;
        std::vector<int> me_c_carry_f, me_c_carry_g; // me_b + W_{i-16} = W_i
    };
    std::vector<MsgExpAux> msg_aux; // indexed by step - 16

    int alloc(int n = 1) {
        int v = next_var;
        next_var += n;
        return v;
    }

    std::vector<int> alloc_word() {
        std::vector<int> w(32);
        for (int j = 0; j < 32; j++) w[j] = alloc();
        return w;
    }

    // 33 carries for a 32-bit adder (carry[0] = 0 forced externally)
    std::vector<int> alloc_carries() {
        std::vector<int> c(33);
        for (int j = 0; j < 33; j++) c[j] = alloc();
        return c;
    }

    void init(int steps) {
        n_steps = steps;
        int n_a = steps + 4; // a[-4]..a[steps-1]

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

        // Step auxiliaries
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

        // Message expansion auxiliaries
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

    // Access helpers (step is absolute: -4..n_steps-1)
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

    void add(std::initializer_list<int> lits) {
        clauses.push_back(lits);
    }

    void add(const std::vector<int>& lits) {
        clauses.push_back(lits);
    }

    // Unit clause
    void unit(int lit) { clauses.push_back({lit}); }

    // x = constant bit
    void fix(int var, bool val) { unit(val ? var : -var); }

    // Equivalence: a <=> b  (2 clauses)
    void equiv(int a, int b) {
        add({-a, b});
        add({a, -b});
    }

    // XOR: a XOR b (a != b)  (2 clauses)
    void xor2(int a, int b) {
        add({a, b});
        add({-a, -b});
    }

    // 3-input XOR: out = a XOR b XOR c  (8 clauses)
    // Blocks the 8 odd-parity assignments of {out, a, b, c}.
    void xor3(int out, int a, int b, int c) {
        add({ out,  a,  b, -c});   // blocks (0,0,0,1)
        add({ out,  a, -b,  c});   // blocks (0,0,1,0)
        add({ out, -a,  b,  c});   // blocks (0,1,0,0)
        add({ out, -a, -b, -c});   // blocks (0,1,1,1)
        add({-out,  a,  b,  c});   // blocks (1,0,0,0)
        add({-out,  a, -b, -c});   // blocks (1,0,1,1)
        add({-out, -a,  b, -c});   // blocks (1,1,0,1)
        add({-out, -a, -b,  c});   // blocks (1,1,1,0)
    }

    // Ch(x,y,z) = (x AND y) XOR (NOT x AND z) = x ? y : z
    // out = Ch(x,y,z)  (4 clauses)
    void ch_gate(int out, int x, int y, int z) {
        // out = (x & y) | (~x & z)
        // If x=1: out=y.  If x=0: out=z.
        add({-x, -y,  out});  // x=1, y=1 → out=1
        add({-x,  y, -out});  // x=1, y=0 → out=0
        add({ x, -z,  out});  // x=0, z=1 → out=1
        add({ x,  z, -out});  // x=0, z=0 → out=0
    }

    // Maj(x,y,z) = (x AND y) XOR (x AND z) XOR (y AND z)
    // out = Maj(x,y,z)  (4 clauses: majority vote)
    void maj_gate(int out, int x, int y, int z) {
        // out=1 iff at least 2 of {x,y,z} are 1
        add({-x, -y,  out});  // x=1, y=1 → out=1
        add({-x, -z,  out});  // x=1, z=1 → out=1
        add({-y, -z,  out});  // y=1, z=1 → out=1
        add({ x,  y, -out});  // x=0, y=0 → out=0
        add({ x,  z, -out});  // x=0, z=0 → out=0
        add({ y,  z, -out});  // y=0, z=0 → out=0
    }

    // Full adder: (sum, carry_out) = a + b + carry_in
    // sum = a XOR b XOR cin
    // cout = (a AND b) OR (a AND cin) OR (b AND cin) = Maj(a, b, cin)
    void full_adder(int sum, int cout, int a, int b, int cin) {
        xor3(sum, a, b, cin);
        maj_gate(cout, a, b, cin);
    }

    // 32-bit ripple-carry adder: result[0..31] = A[0..31] + B[0..31]
    // carry[0] must be forced to 0 externally (or 1 for subtraction)
    // carry[32] is the overflow (ignored for mod 2^32)
    void adder32(const std::vector<int>& result,
                 const std::vector<int>& carry,
                 const std::vector<int>& A,
                 const std::vector<int>& B) {
        assert(result.size() == 32 && carry.size() == 33);
        assert(A.size() == 32 && B.size() == 32);
        for (int j = 0; j < 32; j++) {
            full_adder(result[j], carry[j + 1], A[j], B[j], carry[j]);
        }
    }

    // Add a constant word (K) to a variable word.
    // For each bit of K: if K_bit=0, the adder input is just carry propagation.
    // We still need full clauses, but can simplify where K bits are known.
    // For simplicity, allocate temp variables for K and fix them.
    // Returns the temp variable IDs.
    std::vector<int> make_const_word(uint32_t val, int& next_var) {
        std::vector<int> vars(32);
        for (int j = 0; j < 32; j++) {
            vars[j] = next_var++;
            fix(vars[j], (val >> j) & 1);
        }
        return vars;
    }

    // Sigma/sigma functions: XOR of rotated copies
    // Sigma_0(x) = ROTR(x,2) XOR ROTR(x,13) XOR ROTR(x,22)
    void sigma_xor3(const std::vector<int>& out,
                    const std::vector<int>& x,
                    int r1, int r2, int r3,
                    bool shift3 = false) {
        // out[j] = x[(j+r1)%32] XOR x[(j+r2)%32] XOR x[(j+r3)%32]
        // For shift (sigma_0, sigma_1): third term is x >> r3 (not rotation)
        for (int j = 0; j < 32; j++) {
            int a = x[(j + r1) % 32];
            int b = x[(j + r2) % 32];
            int c;
            if (shift3 && (j + r3 >= 32)) {
                // Shifted-out bit is 0 → XOR with 0 = identity
                // out[j] = a XOR b (2 clauses, via xor3 with dummy? No — use equiv.)
                // Actually: out = a XOR b is just xor3 with c = FALSE.
                // Encode as: out = a XOR b (even parity with 2 vars)
                // 4 clauses: (¬out∨¬a∨¬b)(¬out∨a∨b)(out∨¬a∨b)(out∨a∨¬b)
                add({-out[j], -a, -b});
                add({-out[j],  a,  b});
                add({ out[j], -a,  b});
                add({ out[j],  a, -b});
                continue;
            } else {
                c = x[(j + r3) % 32];
            }
            xor3(out[j], a, b, c);
        }
    }

    int num_clauses() const { return (int)clauses.size(); }
};

// ---- Top-level encoder ----

struct SHA256Encoder {
    int n_steps;
    VarMap vars;
    ClauseDB db;

    explicit SHA256Encoder(int steps) : n_steps(steps) {
        vars.init(steps);
    }

    // Encode IV as unit clauses for both copies
    void encode_iv(const sha256::State& iv) {
        // State = [a,b,c,d, e,f,g,h] = [A_{-1},A_{-2},A_{-3},A_{-4}, E_{-1},...,E_{-4}]
        // IV[0]=a=A_{-1}, IV[1]=b=A_{-2}, IV[2]=c=A_{-3}, IV[3]=d=A_{-4}
        for (int r = 0; r < 4; r++) {
            int step = -1 - r;  // -1, -2, -3, -4
            for (int j = 0; j < 32; j++) {
                bool bit = (iv[r] >> j) & 1;
                db.fix(vars.A(step, false)[j], bit);
                int ag = vars.A(step, true)[j];
                if (ag != vars.A(step, false)[j]) db.fix(ag, bit);
            }
        }
        // IV[4]=e=E_{-1}, IV[5]=f=E_{-2}, IV[6]=g=E_{-3}, IV[7]=h=E_{-4}
        for (int r = 0; r < 4; r++) {
            int step = -1 - r;
            for (int j = 0; j < 32; j++) {
                bool bit = (iv[4 + r] >> j) & 1;
                db.fix(vars.E(step, false)[j], bit);
                int eg = vars.E(step, true)[j];
                if (eg != vars.E(step, false)[j]) db.fix(eg, bit);
            }
        }
    }

    // Encode starting point conditions from gencond CharState
    // Maps BitCond values to CNF constraints on (var_f, var_g) pairs
    void encode_condition(int var_f, int var_g, gencond::BitCond cond) {
        using BC = gencond::BitCond;
        // When variables are aliased (DASH-word optimization), skip redundant constraints
        if (var_f == var_g) {
            switch (cond) {
                case BC::BC_DASH: return; // tautology: x == x
                case BC::BC_FREE: return;
                case BC::BC_0: db.fix(var_f, false); return;
                case BC::BC_1: db.fix(var_f, true); return;
                default: return; // X/U/N would be contradictions on aliased vars
            }
        }
        switch (cond) {
            case BC::BC_0: db.fix(var_f, false); db.fix(var_g, false); break;
            case BC::BC_1: db.fix(var_f, true);  db.fix(var_g, true);  break;
            case BC::BC_U:    db.fix(var_f, true);  db.fix(var_g, false); break;
            case BC::BC_N:    db.fix(var_f, false); db.fix(var_g, true);  break;
            case BC::BC_DASH: db.equiv(var_f, var_g); break;
            case BC::BC_X:    db.xor2(var_f, var_g); break;  // f != g
            case BC::BC_FREE: break; // no constraint
            case BC::BC_HASH: db.add({var_f}); db.add({-var_f}); break; // contradiction
            default:
                // Multi-valued conditions (3,5,7,A,B,C,D,E)
                // Encode as disjunction of allowed pairs
                encode_multival_condition(var_f, var_g, cond);
                break;
        }
    }

    void encode_multival_condition(int vf, int vg, gencond::BitCond cond) {
        uint8_t mask = static_cast<uint8_t>(cond);
        // Bits: bit0=(0,0), bit1=(1,0), bit2=(0,1), bit3=(1,1)
        // At least one must hold → disjunction
        std::vector<int> clause;
        // Introduce indicator variables for each allowed pair
        // Actually, we can encode directly via clauses on (vf, vg):
        // Disallow pairs where mask bit is 0.
        if (!(mask & 1)) db.add({ vf,  vg});  // disallow (0,0)
        if (!(mask & 2)) db.add({-vf,  vg});  // disallow (1,0)
        if (!(mask & 4)) db.add({ vf, -vg});  // disallow (0,1)
        if (!(mask & 8)) db.add({-vf, -vg});  // disallow (1,1)
    }

    // Encode starting point from a CharState-like structure
    // A_conds[step+4] for step -4..n_steps-1
    // E_conds[step+4] for step -4..n_steps-1
    // W_conds[step] for step 0..n_steps-1
    void encode_starting_point(
        const std::vector<gencond::WordCond>& A_conds,
        const std::vector<gencond::WordCond>& E_conds,
        const std::vector<gencond::WordCond>& W_conds)
    {
        for (int i = 0; i < (int)A_conds.size(); i++) {
            for (int j = 0; j < 32; j++) {
                auto c = A_conds[i].get(j);
                if (c != gencond::BitCond::BC_FREE)
                    encode_condition(vars.a_f[i][j], vars.a_g[i][j], c);
            }
        }
        for (int i = 0; i < (int)E_conds.size(); i++) {
            for (int j = 0; j < 32; j++) {
                auto c = E_conds[i].get(j);
                if (c != gencond::BitCond::BC_FREE)
                    encode_condition(vars.e_f[i][j], vars.e_g[i][j], c);
            }
        }
        for (int i = 0; i < (int)W_conds.size(); i++) {
            for (int j = 0; j < 32; j++) {
                auto c = W_conds[i].get(j);
                if (c != gencond::BitCond::BC_FREE)
                    encode_condition(vars.w_f[i][j], vars.w_g[i][j], c);
            }
        }
    }

    // Encode one copy of one SHA-256 step
    void encode_step_copy(int step, bool copy2) {
        auto& sa = vars.step_aux[step];
        auto& A_prev1 = vars.A(step - 1, copy2);
        auto& A_prev2 = vars.A(step - 2, copy2);
        auto& A_prev3 = vars.A(step - 3, copy2);
        auto& A_prev4 = vars.A(step - 4, copy2); // d register
        auto& E_prev1 = vars.E(step - 1, copy2);
        auto& E_prev2 = vars.E(step - 2, copy2);
        auto& E_prev3 = vars.E(step - 3, copy2);
        auto& E_prev4 = vars.E(step - 4, copy2); // h register
        auto& A_new = vars.A(step, copy2);
        auto& E_new = vars.E(step, copy2);
        auto& W_step = vars.W(step, copy2);

        auto& sigma1 = copy2 ? sa.sigma1_g : sa.sigma1_f;
        auto& ch = copy2 ? sa.ch_g : sa.ch_f;
        auto& sigma0 = copy2 ? sa.sigma0_g : sa.sigma0_f;
        auto& maj = copy2 ? sa.maj_g : sa.maj_f;

        auto& t1a = copy2 ? sa.t1a_g : sa.t1a_f;
        auto& t1a_c = copy2 ? sa.t1a_carry_g : sa.t1a_carry_f;
        auto& t1b = copy2 ? sa.t1b_g : sa.t1b_f;
        auto& t1b_c = copy2 ? sa.t1b_carry_g : sa.t1b_carry_f;
        auto& kw = copy2 ? sa.kw_g : sa.kw_f;
        auto& kw_c = copy2 ? sa.kw_carry_g : sa.kw_carry_f;
        auto& t1c = copy2 ? sa.t1c_g : sa.t1c_f;
        auto& t1c_c = copy2 ? sa.t1c_carry_g : sa.t1c_carry_f;
        auto& enew_c = copy2 ? sa.enew_carry_g : sa.enew_carry_f;
        auto& t2a = copy2 ? sa.t2a_g : sa.t2a_f;
        auto& t2a_c = copy2 ? sa.t2a_carry_g : sa.t2a_carry_f;
        auto& anew_c = copy2 ? sa.anew_carry_g : sa.anew_carry_f;

        // 1. Sigma_1(E_{i-1}): ROTR(6) XOR ROTR(11) XOR ROTR(25)
        db.sigma_xor3(sigma1, E_prev1, 6, 11, 25);

        // 2. Ch(E_{i-1}, E_{i-2}, E_{i-3})
        for (int j = 0; j < 32; j++)
            db.ch_gate(ch[j], E_prev1[j], E_prev2[j], E_prev3[j]);

        // 3. Compute T1 = h + Sigma1 + Ch + K_i + W_i
        //    Using cascaded 2-input adders:
        //    t1a = h + Sigma1
        //    t1b = t1a + Ch
        //    kw  = K_i + W_i  (K_i is constant)
        //    t1c = t1b + kw
        db.fix(t1a_c[0], false);
        db.adder32(t1a, t1a_c, E_prev4, sigma1);

        db.fix(t1b_c[0], false);
        db.adder32(t1b, t1b_c, t1a, ch);

        // K+W: allocate constant K bits
        auto k_vars = db.make_const_word(sha256::K[step], vars.next_var);
        db.fix(kw_c[0], false);
        db.adder32(kw, kw_c, k_vars, W_step);

        db.fix(t1c_c[0], false);
        db.adder32(t1c, t1c_c, t1b, kw);

        // 4. E_new = d + T1  (d = A_{i-4} in standard notation,
        //    but in State notation: s[3]+t1 → s[4]. Here d = old s[3] = A_{i-4})
        //    Wait — in compress(): t1 = s[7]+Sigma1+Ch+K+W, e_new = s[3]+t1
        //    s[7] is E_{i-4} (h), s[3] is A_{i-4} (d) — but s[4]=e_new.
        //    Actually s[4] is E_{i-1} in the shifted notation... Let me re-check.
        //
        //    Standard compress:
        //      t1 = s[7] + Sigma1(s[4]) + Ch(s[4],s[5],s[6]) + K + W
        //      t2 = Sigma0(s[0]) + Maj(s[0],s[1],s[2])
        //      new_s = [t1+t2, s[0], s[1], s[2], s[3]+t1, s[4], s[5], s[6]]
        //    So: new_a = t1+t2, new_e = s[3]+t1
        //
        //    In alt_step notation:
        //      A_i = new s[0], E_i = new s[4]
        //      A_{i-1} = old s[0], A_{i-2} = old s[1], A_{i-3} = old s[2], A_{i-4} = old s[3]
        //      E_{i-1} = old s[4], E_{i-2} = old s[5], E_{i-3} = old s[6], E_{i-4} = old s[7]
        //    So: t1 = E_{i-4} + Sigma1(E_{i-1}) + Ch(E_{i-1},E_{i-2},E_{i-3}) + K + W
        //        E_i = A_{i-4} + t1
        //        A_i = t1 + Sigma0(A_{i-1}) + Maj(A_{i-1},A_{i-2},A_{i-3})
        //    This matches our encoding above (E_prev4 = E_{i-4} = h, A_prev4 = d)
        //    BUT: t1 uses E_prev4 as h. Let me re-verify...
        //    compress: t1 = s[7] + ... where s[7] = E_{i-4} ← WRONG? No:
        //      Before step i: s = [A_{i-1}, A_{i-2}, A_{i-3}, A_{i-4}, E_{i-1}, E_{i-2}, E_{i-3}, E_{i-4}]
        //      s[7] = E_{i-4}, s[3] = A_{i-4} ✓
        //    So t1 = E_{i-4} + Sigma1(E_{i-1}) + Ch(...) + K + W → correct above

        db.fix(enew_c[0], false);
        db.adder32(E_new, enew_c, A_prev4, t1c);

        // 5. Sigma_0(A_{i-1}): ROTR(2) XOR ROTR(13) XOR ROTR(22)
        db.sigma_xor3(sigma0, A_prev1, 2, 13, 22);

        // 6. Maj(A_{i-1}, A_{i-2}, A_{i-3})
        for (int j = 0; j < 32; j++)
            db.maj_gate(maj[j], A_prev1[j], A_prev2[j], A_prev3[j]);

        // 7. A_new = T1 + Sigma0 + Maj = T1 + (Sigma0 + Maj)
        db.fix(t2a_c[0], false);
        db.adder32(t2a, t2a_c, sigma0, maj);

        db.fix(anew_c[0], false);
        db.adder32(A_new, anew_c, t1c, t2a);
    }

    // Encode message expansion for step >= 16
    void encode_msg_exp_copy(int step, bool copy2) {
        assert(step >= 16 && step < n_steps);
        auto& ma = vars.msg_aux[step - 16];
        auto& W_out = vars.W(step, copy2);
        auto& W_m2  = vars.W(step - 2, copy2);
        auto& W_m7  = vars.W(step - 7, copy2);
        auto& W_m15 = vars.W(step - 15, copy2);
        auto& W_m16 = vars.W(step - 16, copy2);

        auto& s1 = copy2 ? ma.s1_g : ma.s1_f;
        auto& s0 = copy2 ? ma.s0_g : ma.s0_f;
        auto& me_a = copy2 ? ma.me_a_g : ma.me_a_f;
        auto& me_a_c = copy2 ? ma.me_a_carry_g : ma.me_a_carry_f;
        auto& me_b = copy2 ? ma.me_b_g : ma.me_b_f;
        auto& me_b_c = copy2 ? ma.me_b_carry_g : ma.me_b_carry_f;
        auto& me_c_c = copy2 ? ma.me_c_carry_g : ma.me_c_carry_f;

        // sigma_1(W_{i-2}): ROTR(17) XOR ROTR(19) XOR SHR(10)
        db.sigma_xor3(s1, W_m2, 17, 19, 10, true);

        // sigma_0(W_{i-15}): ROTR(7) XOR ROTR(18) XOR SHR(3)
        db.sigma_xor3(s0, W_m15, 7, 18, 3, true);

        // W_i = s1 + W_{i-7} + s0 + W_{i-16}
        db.fix(me_a_c[0], false);
        db.adder32(me_a, me_a_c, s1, W_m7);

        db.fix(me_b_c[0], false);
        db.adder32(me_b, me_b_c, me_a, s0);

        db.fix(me_c_c[0], false);
        db.adder32(W_out, me_c_c, me_b, W_m16);
    }

    // Encode all steps (both copies)
    void encode_steps() {
        for (int i = 0; i < n_steps; i++) {
            encode_step_copy(i, false);
            encode_step_copy(i, true);
        }
    }

    // Encode steps starting from step `from` (both copies).
    // Steps 0..from-1 are assumed to be constant-folded via encode_constant_prefix.
    void encode_steps_from(int from) {
        for (int i = from; i < n_steps; i++) {
            encode_step_copy(i, false);
            encode_step_copy(i, true);
        }
    }

    // Encode message expansion (both copies)
    void encode_msg_expansion() {
        for (int i = 16; i < n_steps; i++) {
            encode_msg_exp_copy(i, false);
            encode_msg_exp_copy(i, true);
        }
    }

    // Encode message expansion starting from step `from`.
    void encode_msg_expansion_from(int from) {
        int start = std::max(from, 16);
        for (int i = start; i < n_steps; i++) {
            encode_msg_exp_copy(i, false);
            encode_msg_exp_copy(i, true);
        }
    }

    // Constant-fold a prefix of steps where state is fully determined.
    // Given W[0..prefix_steps-1] values (both copies), compute the state
    // through those steps and fix the resulting state variables as unit clauses.
    // This eliminates step-function encoding for those steps entirely.
    //
    // Prerequisites: IV must be encoded first. W[0..prefix_steps-1] must be
    // identical in both copies (DASH) for this to be valid.
    void encode_constant_prefix(const uint32_t* w_vals, int prefix_steps) {
        // Compute state through prefix steps
        sha256::State state = sha256::IV;
        uint32_t W[64];
        for (int i = 0; i < prefix_steps; i++) W[i] = w_vals[i];
        // Expand if needed
        for (int i = 16; i < prefix_steps; i++) {
            W[i] = sha256::sigma_1(W[i-2]) + W[i-7] + sha256::sigma_0(W[i-15]) + W[i-16];
        }
        for (int i = 0; i < prefix_steps; i++) {
            sha256::compress(state, i, W);
            // Fix A[i] and E[i] in both copies
            // state = [a,b,c,d,e,f,g,h] after step i
            // A_i = state[0], E_i = state[4]
            int idx = i + 4;
            for (int j = 0; j < 32; j++) {
                db.fix(vars.a_f[idx][j], (state[0] >> j) & 1);
                db.fix(vars.a_g[idx][j], (state[0] >> j) & 1);
                db.fix(vars.e_f[idx][j], (state[4] >> j) & 1);
                db.fix(vars.e_g[idx][j], (state[4] >> j) & 1);
            }
        }
        // Fix W variables for the prefix
        for (int i = 0; i < prefix_steps && i < n_steps; i++) {
            for (int j = 0; j < 32; j++) {
                db.fix(vars.w_f[i][j], (W[i] >> j) & 1);
                db.fix(vars.w_g[i][j], (W[i] >> j) & 1);
            }
        }
    }

    // Encode collision condition: final hash values equal
    // After n_steps: state = [A_{n-1}, A_{n-2}, A_{n-3}, A_{n-4},
    //                         E_{n-1}, E_{n-2}, E_{n-3}, E_{n-4}]
    // Hash = IV + state (mod 2^32 per word)
    // Collision: IV + state_f = IV + state_g → state_f = state_g
    // So for real collision: A_f[i] = A_g[i] and E_f[i] = E_g[i] for i in {n-1..n-4}
    void encode_collision() {
        for (int r = 0; r < 4; r++) {
            int step = n_steps - 1 - r;
            for (int j = 0; j < 32; j++) {
                int af = vars.A(step, false)[j], ag = vars.A(step, true)[j];
                int ef = vars.E(step, false)[j], eg = vars.E(step, true)[j];
                if (af != ag) db.equiv(af, ag);
                if (ef != eg) db.equiv(ef, eg);
            }
        }
    }

    // Encode concrete message values as unit clauses (for verification)
    void encode_message(const uint32_t* m1, const uint32_t* m2, int n_words) {
        for (int i = 0; i < n_words && i < n_steps; i++) {
            for (int j = 0; j < 32; j++) {
                db.fix(vars.w_f[i][j], (m1[i] >> j) & 1);
                db.fix(vars.w_g[i][j], (m2[i] >> j) & 1);
            }
        }
    }

    // Write DIMACS CNF to file
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

    // Add all clauses to a CaDiCaL solver via API
    template<typename Solver>
    void add_to_solver(Solver& solver) const {
        solver.resize(vars.total_vars());
        for (auto& cl : db.clauses) {
            for (int lit : cl) solver.add(lit);
            solver.add(0);
        }
    }

    // Extract message words from SAT solution
    template<typename F>
    void extract_message(const F& val_fn, uint32_t* m1, uint32_t* m2, int n_words) const {
        for (int i = 0; i < n_words && i < n_steps; i++) {
            m1[i] = 0; m2[i] = 0;
            for (int j = 0; j < 32; j++) {
                if (val_fn(vars.w_f[i][j]) > 0) m1[i] |= (1u << j);
                if (val_fn(vars.w_g[i][j]) > 0) m2[i] |= (1u << j);
            }
        }
    }

    void print_stats() const {
        printf("CNF stats: %d variables, %d clauses\n",
               vars.total_vars(), db.num_clauses());
        printf("  Primary: A(%d×2×32=%d) E(%d×2×32=%d) W(%d×2×32=%d)\n",
               n_steps + 4, (n_steps + 4) * 2 * 32,
               n_steps + 4, (n_steps + 4) * 2 * 32,
               n_steps, n_steps * 2 * 32);
    }
};

} // namespace cnf
