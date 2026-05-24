// Embedded 2-watched-literal BCP engine for SHA-256 collision search.
//
// Provides incremental Boolean Constraint Propagation over the Tseitin
// CNF encoding of the dual-copy SHA-256 step function + differential
// conditions. Catches carry-chain contradictions that the bitwise
// GnD propagator misses.
//
// Architecture:
//   1. Build CNF once from starting_point via cnf_encoder.hpp
//   2. Propagate IV + starting-point unit clauses → base state
//   3. Checkpoint base state
//   4. Per GnD decision: assert literal, run BCP
//   5. On GnD restart: restore to base checkpoint
//
// The 2-watched-literal scheme is standard (Moskewicz et al., Chaff 2001).

#pragma once

#include "cnf_encoder.hpp"
#include "search.hpp"  // BitLoc, CharState, WordType
#include "gencond.hpp"
#include <vector>
#include <cstdint>
#include <cassert>

namespace mendel2011 {

class EmbeddedBCP {
public:
    // Literal encoding: var v (1-based) → positive literal = 2*v,
    //                                     negative literal = 2*v+1
    // This avoids sign manipulation and gives direct indexing.
    static int lit(int var, bool positive) {
        return positive ? 2 * var : 2 * var + 1;
    }
    static int neg(int l) { return l ^ 1; }
    static int var_of(int l) { return l >> 1; }
    static bool sign(int l) { return (l & 1) == 0; } // true = positive

    enum : int8_t { UNDEF = 0, TRUE_VAL = 1, FALSE_VAL = -1 };

private:
    int n_vars_ = 0;

    // Clause storage: flat array with clause headers
    struct ClauseRef {
        int start;  // index into lit_pool_
        int size;
    };
    std::vector<int> lit_pool_;       // all clause literals, contiguous
    std::vector<ClauseRef> clauses_;  // clause metadata

    // Per-variable assignment
    std::vector<int8_t> assign_;      // 0=undef, +1=true, -1=false

    // Watch lists: for each literal, list of clause indices watching it
    std::vector<std::vector<int>> watches_;

    // Propagation trail
    std::vector<int> trail_;          // assigned literals in order
    int prop_head_ = 0;              // next literal to propagate

    // Checkpoint: trail size after base propagation
    int base_trail_size_ = 0;
    int base_clause_count_ = 0;
    int base_litpool_size_ = 0;

    bool conflict_ = false;

public:
    EmbeddedBCP() = default;

    // Initialize from a ClauseDB (from cnf_encoder.hpp)
    void init(int n_vars, const cnf::ClauseDB& db) {
        n_vars_ = n_vars;
        assign_.assign(n_vars + 1, UNDEF);  // 1-indexed
        watches_.assign(2 * (n_vars + 1), {});
        trail_.clear();
        trail_.reserve(n_vars);
        lit_pool_.clear();
        clauses_.clear();
        prop_head_ = 0;
        conflict_ = false;
        base_clause_count_ = 0;
        base_litpool_size_ = 0;

        // Load clauses
        clauses_.reserve(db.clauses.size());
        lit_pool_.reserve(db.clauses.size() * 4);  // rough estimate

        for (auto& cl : db.clauses) {
            if (cl.empty()) {
                conflict_ = true;
                continue;
            }

            int start = (int)lit_pool_.size();
            for (int dimacs_lit : cl) {
                // Convert DIMACS literal to internal literal
                int v = dimacs_lit > 0 ? dimacs_lit : -dimacs_lit;
                bool pos = dimacs_lit > 0;
                lit_pool_.push_back(lit(v, pos));
            }
            int sz = (int)cl.size();
            int ci = (int)clauses_.size();
            clauses_.push_back({start, sz});

            if (sz == 1) {
                // Unit clause: enqueue immediately
                int l = lit_pool_[start];
                enqueue(l);
            } else {
                // Watch first two literals
                watches_[lit_pool_[start]].push_back(ci);
                watches_[lit_pool_[start + 1]].push_back(ci);
            }
        }
    }

    // Run BCP to fixpoint. Returns false if conflict detected.
    bool propagate() {
        while (prop_head_ < (int)trail_.size()) {
            if (conflict_) return false;
            int p = trail_[prop_head_++];
            // p was set to true → ¬p is false → update clauses watching ¬p
            int false_lit = neg(p);

            auto& wlist = watches_[false_lit];
            int i = 0, j = 0;
            while (i < (int)wlist.size()) {
                int ci = wlist[i];
                auto& cr = clauses_[ci];
                int* lits = &lit_pool_[cr.start];
                int sz = cr.size;

                // Ensure the false literal is at position 1 (swap if at 0)
                if (lits[0] == false_lit) {
                    std::swap(lits[0], lits[1]);
                }
                assert(sz >= 2 && lits[1] == false_lit);

                // If lits[0] is already true, clause is satisfied
                if (value_of(lits[0]) == TRUE_VAL) {
                    wlist[j++] = ci;
                    ++i;
                    continue;
                }

                // Look for a new watch literal (not false)
                bool found = false;
                for (int k = 2; k < sz; k++) {
                    if (value_of(lits[k]) != FALSE_VAL) {
                        // Swap lits[1] and lits[k], watch lits[1]
                        std::swap(lits[1], lits[k]);
                        watches_[lits[1]].push_back(ci);
                        found = true;
                        break;
                    }
                }

                if (found) {
                    // Don't copy this watch entry (removed from false_lit's list)
                    ++i;
                    continue;
                }

                // No replacement found — clause is unit or conflict
                wlist[j++] = ci;
                ++i;

                if (value_of(lits[0]) == FALSE_VAL) {
                    // Conflict: all literals false
                    conflict_ = true;
                    // Copy remaining watches
                    while (i < (int)wlist.size()) wlist[j++] = wlist[i++];
                    wlist.resize(j);
                    return false;
                }

                // Unit: lits[0] is the only unset literal
                enqueue(lits[0]);
            }
            wlist.resize(j);
        }
        return !conflict_;
    }

    // Save current state as base checkpoint (call after initial propagation)
    void checkpoint_base() {
        base_trail_size_ = (int)trail_.size();
        base_clause_count_ = (int)clauses_.size();
        base_litpool_size_ = (int)lit_pool_.size();
    }

    // Restore to base checkpoint (for GnD restart)
    void restore_base() {
        // Undo all assignments after base
        while ((int)trail_.size() > base_trail_size_) {
            int l = trail_.back();
            trail_.pop_back();
            assign_[var_of(l)] = UNDEF;
        }
        prop_head_ = base_trail_size_;
        conflict_ = false;

        // Remove dynamic clauses (DASH/X binary clauses added after base)
        // from watch lists, then truncate clause/literal pools.
        for (int ci = base_clause_count_; ci < (int)clauses_.size(); ci++) {
            auto& cr = clauses_[ci];
            int* lits = &lit_pool_[cr.start];
            remove_watch(lits[0], ci);
            remove_watch(lits[1], ci);
        }
        clauses_.resize(base_clause_count_);
        lit_pool_.resize(base_litpool_size_);
        // Note: watch lists for base clauses are self-correcting — watched
        // literals that were moved during propagation after the checkpoint
        // are still valid watches (they may watch non-optimal literals, but
        // never incorrect ones). This is a standard property of 2WL on
        // backtrack.
    }

    // Assert a literal (from GnD decision). Returns false if conflict.
    bool assert_literal(int dimacs_lit) {
        if (conflict_) return false;
        int v = dimacs_lit > 0 ? dimacs_lit : -dimacs_lit;
        bool pos = dimacs_lit > 0;
        int l = lit(v, pos);

        int8_t cur = assign_[v];
        if (cur != UNDEF) {
            // Already assigned — check consistency
            bool cur_pos = (cur == TRUE_VAL);
            if (cur_pos != pos) { conflict_ = true; return false; }
            return true;  // already set to same value
        }

        enqueue(l);
        return propagate();
    }

    // Assert a (BitLoc, BitCond) decision using the VarMap
    // Returns false if BCP detects conflict
    bool assert_decision(const BitLoc& loc, gencond::BitCond cond,
                         const cnf::VarMap& vars) {
        using BC = gencond::BitCond;
        int vf = -1, vg = -1;

        // Map BitLoc to SAT variable pair (var_f, var_g)
        switch (loc.type) {
            case WT_A:
                vf = vars.a_f[loc.step + 4][loc.bit];
                vg = vars.a_g[loc.step + 4][loc.bit];
                break;
            case WT_E:
                vf = vars.e_f[loc.step + 4][loc.bit];
                vg = vars.e_g[loc.step + 4][loc.bit];
                break;
            case WT_W:
                vf = vars.w_f[loc.step][loc.bit];
                vg = vars.w_g[loc.step][loc.bit];
                break;
            default:
                return true; // unknown register, skip
        }

        // Map BitCond to unit assertions on (vf, vg)
        switch (cond) {
            case BC::BC_0:
                if (!assert_literal(-vf)) return false;
                return assert_literal(-vg);
            case BC::BC_1:
                if (!assert_literal(vf)) return false;
                return assert_literal(vg);
            case BC::BC_U:
                if (!assert_literal(vf)) return false;
                return assert_literal(-vg);
            case BC::BC_N:
                if (!assert_literal(-vf)) return false;
                return assert_literal(vg);
            case BC::BC_DASH:
                return assert_equiv(vf, vg);
            case BC::BC_X:
                return assert_xor(vf, vg);
            case BC::BC_FREE:
                return true; // no constraint
            case BC::BC_HASH:
                conflict_ = true;
                return false;
            default:
                return true;
        }
    }

    // Dynamically add a binary clause (for Phase 1 DASH/X decisions).
    // If either literal is already determined, this may become a unit
    // propagation or a conflict.
    bool add_binary_clause(int dimacs_a, int dimacs_b) {
        if (conflict_) return false;
        int la = (dimacs_a > 0) ? lit(dimacs_a, true) : lit(-dimacs_a, false);
        int lb = (dimacs_b > 0) ? lit(dimacs_b, true) : lit(-dimacs_b, false);

        int start = (int)lit_pool_.size();
        lit_pool_.push_back(la);
        lit_pool_.push_back(lb);
        int ci = (int)clauses_.size();
        clauses_.push_back({start, 2});

        // Always set up watches — needed for correctness after restore_base()
        watches_[la].push_back(ci);
        watches_[lb].push_back(ci);

        int8_t va = value_of(la), vb = value_of(lb);

        if (va == FALSE_VAL && vb == FALSE_VAL) {
            conflict_ = true;
            return false;
        }
        if (va == FALSE_VAL) {
            enqueue(lb);
            return propagate();
        }
        if (vb == FALSE_VAL) {
            enqueue(la);
            return propagate();
        }
        return true;
    }

    // Assert a DASH condition: vf <=> vg (2 binary clauses)
    bool assert_equiv(int dimacs_vf, int dimacs_vg) {
        if (!add_binary_clause(-dimacs_vf, dimacs_vg)) return false;
        return add_binary_clause(dimacs_vf, -dimacs_vg);
    }

    // Assert an X (XOR) condition: vf != vg (2 binary clauses)
    bool assert_xor(int dimacs_vf, int dimacs_vg) {
        if (!add_binary_clause(dimacs_vf, dimacs_vg)) return false;
        return add_binary_clause(-dimacs_vf, -dimacs_vg);
    }

    bool has_conflict() const { return conflict_; }
    int trail_size() const { return (int)trail_.size(); }
    int base_size() const { return base_trail_size_; }
    int num_vars() const { return n_vars_; }
    int num_clauses() const { return (int)clauses_.size(); }

    // Get the current value of a DIMACS variable
    int8_t value(int dimacs_var) const {
        return assign_[dimacs_var > 0 ? dimacs_var : -dimacs_var];
    }

    // Stats: number of variables fixed after base propagation
    int base_fixed() const {
        int count = 0;
        for (int v = 1; v <= n_vars_; v++)
            if (assign_[v] != UNDEF) count++;
        return count;
    }

private:
    int8_t value_of(int l) const {
        int8_t v = assign_[var_of(l)];
        if (v == UNDEF) return UNDEF;
        // If literal is positive and var is true → TRUE_VAL
        // If literal is negative and var is false → TRUE_VAL
        return sign(l) ? v : -v;
    }

    void enqueue(int l) {
        int v = var_of(l);
        int8_t val = sign(l) ? TRUE_VAL : FALSE_VAL;
        if (assign_[v] != UNDEF) {
            if (assign_[v] != val) conflict_ = true;
            return;
        }
        assign_[v] = val;
        trail_.push_back(l);
    }

    // Remove clause ci from the watch list of literal l.
    // Used by restore_base() to clean up dynamic clauses.
    void remove_watch(int l, int ci) {
        auto& wl = watches_[l];
        for (int i = 0; i < (int)wl.size(); i++) {
            if (wl[i] == ci) {
                wl[i] = wl.back();
                wl.pop_back();
                return;
            }
        }
    }
};

// Helper: build an EmbeddedBCP from a CharState (starting point)
// Encodes IV, starting-point conditions, step function, and collision.
// Runs initial BCP and checkpoints the base state.
// Returns false if the base state is already UNSAT (should not happen
// for a valid starting point).
inline bool build_bcp_engine(EmbeddedBCP& bcp, const cnf::VarMap*& out_vars,
                             const CharState& initial, int n_steps) {
    static thread_local cnf::SHA256Encoder* encoder = nullptr;
    // Rebuild encoder each time (starting point may differ)
    delete encoder;
    encoder = new cnf::SHA256Encoder(n_steps);

    encoder->encode_iv(sha256::IV);

    std::vector<gencond::WordCond> W_conds(initial.W.begin(),
                                           initial.W.begin() + n_steps);
    encoder->encode_starting_point(initial.A, initial.E, W_conds);
    encoder->encode_steps();
    encoder->encode_collision();

    bcp.init(encoder->vars.total_vars(), encoder->db);
    out_vars = &encoder->vars;

    if (!bcp.propagate()) return false;  // base is UNSAT
    bcp.checkpoint_base();
    return true;
}

} // namespace mendel2011
