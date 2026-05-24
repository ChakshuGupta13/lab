// SAT-like search engine for SHA-256 differential characteristics.
// Implements the two-phase search from Mendel-Nad-Schläffer 2011, §5.
//
// Phase 1: resolve '?' → '-' and 'x' → 'u'/'n' (characteristic search)
// Phase 2: resolve '-' → '0'/'1' for high-connectivity bits (message search)
// Phases interleave dynamically based on backtracking needs.
//
// Core loop: decision → propagation → consistency check → backtrack on
// contradiction, with critical-bit tracking and restart-from-scratch.

#pragma once

#include "gencond.hpp"
#include "propagate.hpp"
#include "alt_step.hpp"
#include "twobit.hpp"
#include "wordwise.hpp"
#include <cstdint>
#include <cstdio>
#include <vector>
#include <random>
#include <algorithm>
#include <numeric>
#include <unordered_set>

namespace mendel2011 {

using namespace gencond;
using namespace sha256;
using namespace twobit;

// Create a WordCond with all 32 bits set to the same condition.
inline WordCond wc_uniform(BitCond c) {
    uint32_t mask = 0xFFFFFFFF;
    WordCond wc;
    wc.allow[0] = (c & 1) ? mask : 0;
    wc.allow[1] = (c & 2) ? mask : 0;
    wc.allow[2] = (c & 4) ? mask : 0;
    wc.allow[3] = (c & 8) ? mask : 0;
    return wc;
}

// ---- Characteristic state ----
// Holds conditions on A, E (steps -4..N-1) and W (steps 0..63).

struct CharState {
    int n_steps;        // number of SHA-256 steps to attack

    // Conditions indexed by step+4 offset (so A[0]=A_{-4}, A[4+i]=A_i)
    std::vector<WordCond> A;   // size = n_steps + 4
    std::vector<WordCond> E;   // size = n_steps + 4
    std::vector<WordCond> W;   // size = 64 (full expansion schedule)

    CharState() : n_steps(0) {}

    explicit CharState(int steps) : n_steps(steps),
        A(steps + 4, WordCond::free()),
        E(steps + 4, WordCond::free()),
        W(64, WordCond::free()) {}

    // Access by alt-step index: step i uses A_{i-4}..A_{i-1}, etc.
    // Array index = step + 4 for A/E (step -4 → index 0).
    WordCond& a(int step) { return A[step + 4]; }
    WordCond& e(int step) { return E[step + 4]; }
    WordCond& w(int step) { return W[step]; }
    const WordCond& a(int step) const { return A[step + 4]; }
    const WordCond& e(int step) const { return E[step + 4]; }
    const WordCond& w(int step) const { return W[step]; }

    // Count total '?' and 'x' bits (Phase 1 undecided).
    // Count Phase 1 undecided bits (? and x) in the attack-relevant range.
    int count_phase1_undecided() const {
        int count = 0;
        for (auto& wc : A) { count += wc.count_free(); count += __builtin_popcount(wc.x_mask()); }
        for (auto& wc : E) { count += wc.count_free(); count += __builtin_popcount(wc.x_mask()); }
        for (int i = 0; i < n_steps && i < (int)W.size(); ++i) {
            count += W[i].count_free();
            count += __builtin_popcount(W[i].x_mask());
        }
        return count;
    }

    // Count '-' bits in the attack-relevant range.
    int count_dash() const {
        int count = 0;
        for (auto& wc : A) count += wc.count_dash();
        for (auto& wc : E) count += wc.count_dash();
        for (int i = 0; i < n_steps && i < (int)W.size(); ++i)
            count += W[i].count_dash();
        return count;
    }

    // Check for any contradiction bit.
    bool has_contradiction() const {
        for (auto& wc : A) if (!wc.is_consistent()) return true;
        for (auto& wc : E) if (!wc.is_consistent()) return true;
        for (auto& wc : W) if (!wc.is_consistent()) return true;
        return false;
    }

    // Count fully decided bits: {BC_0, BC_1, BC_U, BC_N} have exactly
    // one allowed value pair.  Used by look-ahead scoring (Algorithm 2,
    // Eichlseder et al. 2014).
    int count_decided() const {
        auto word_decided = [](const WordCond& wc) {
            uint32_t d0 = wc.allow[0] & ~wc.allow[1] & ~wc.allow[2] & ~wc.allow[3];
            uint32_t d1 = ~wc.allow[0] & wc.allow[1] & ~wc.allow[2] & ~wc.allow[3];
            uint32_t d2 = ~wc.allow[0] & ~wc.allow[1] & wc.allow[2] & ~wc.allow[3];
            uint32_t d3 = ~wc.allow[0] & ~wc.allow[1] & ~wc.allow[2] & wc.allow[3];
            return __builtin_popcount(d0 | d1 | d2 | d3);
        };
        int count = 0;
        for (auto& wc : A) count += word_decided(wc);
        for (auto& wc : E) count += word_decided(wc);
        for (auto& wc : W) count += word_decided(wc);
        return count;
    }
};

// ---- Bit location in the characteristic ----

enum WordType : uint8_t { WT_A = 0, WT_E = 1, WT_W = 2 };

struct BitLoc {
    WordType type;
    int step;     // step index (-4..N-1 for A/E, 0..63 for W)
    int bit;      // bit position 0..31

    bool operator==(const BitLoc& o) const {
        return type == o.type && step == o.step && bit == o.bit;
    }
};

// Access a WordCond from CharState by BitLoc.
inline WordCond& get_word(CharState& cs, const BitLoc& loc) {
    switch (loc.type) {
        case WT_A: return cs.a(loc.step);
        case WT_E: return cs.e(loc.step);
        case WT_W: return cs.w(loc.step);
    }
    __builtin_unreachable();
}

inline BitCond get_cond(const CharState& cs, const BitLoc& loc) {
    switch (loc.type) {
        case WT_A: return cs.a(loc.step).get(loc.bit);
        case WT_E: return cs.e(loc.step).get(loc.bit);
        case WT_W: return cs.w(loc.step).get(loc.bit);
    }
    __builtin_unreachable();
}

// ---- Decision record (for backtracking trail) ----

struct Decision {
    BitLoc loc;
    BitCond original;     // condition before this decision
    BitCond first_choice; // what we imposed first
    BitCond alt_choice;   // alternative (second) choice
    bool tried_alt;       // whether we've tried the alternative
    bool critical;        // marked as critical on double-fail
    CharState snapshot;   // state before this decision
};

// ---- Backjumping support (§5.1 step 8) ----
// Targeted non-chronological backtracking: when both choices of a decision
// fail, identify which EARLIER decision caused the conflict and jump there
// directly, skipping intermediate decisions that are irrelevant.

// Encode a BitLoc to a flat index for the bit_level array.
// Layout: A words [0..(n+4)), E words [(n+4)..2(n+4)), W words [2(n+4)..2(n+4)+64)
// Each word occupies 32 consecutive positions.
inline int encode_bitloc(const BitLoc& loc, int n_steps) {
    switch (loc.type) {
        case WT_A: return (loc.step + 4) * 32 + loc.bit;
        case WT_E: return ((n_steps + 4) + loc.step + 4) * 32 + loc.bit;
        case WT_W: return (2 * (n_steps + 4) + loc.step) * 32 + loc.bit;
    }
    __builtin_unreachable();
}

inline int bit_level_total(int n_steps) {
    return (2 * (n_steps + 4) + 64) * 32;
}

// Find all contradiction (#) bits in the characteristic.
inline std::vector<BitLoc> find_contradiction_locs(const CharState& cs) {
    std::vector<BitLoc> locs;
    auto scan = [&](WordType wt, const std::vector<WordCond>& words, int step_offset) {
        for (int i = 0; i < (int)words.size(); ++i) {
            uint32_t contra = ~(words[i].allow[0] | words[i].allow[1] |
                                words[i].allow[2] | words[i].allow[3]);
            while (contra) {
                int b = __builtin_ctz(contra);
                locs.push_back({wt, i + step_offset, b});
                contra &= contra - 1;
            }
        }
    };
    scan(WT_A, cs.A, -4);
    scan(WT_E, cs.E, -4);
    scan(WT_W, cs.W, 0);
    return locs;
}

// Get the dependency words for a bit location (one level of SHA-256
// step function or message expansion inputs).
// Returns empty for IV registers (A/E with step < 0) and W[0..15].
struct WordLoc { WordType type; int step; };

inline std::vector<WordLoc> get_dependency_words(const BitLoc& loc, int n_steps) {
    std::vector<WordLoc> deps;
    switch (loc.type) {
        case WT_A:
            // A_i = -A_{i-4} + Σ₀(A_{i-1}) + Maj(A_{i-1},A_{i-2},A_{i-3}) + E_i
            if (loc.step >= 0 && loc.step < n_steps) {
                deps.push_back({WT_A, loc.step - 4});
                deps.push_back({WT_A, loc.step - 1});
                deps.push_back({WT_A, loc.step - 2});
                deps.push_back({WT_A, loc.step - 3});
                deps.push_back({WT_E, loc.step});
            }
            break;
        case WT_E:
            // E_i = E_{i-4} + Σ₁(E_{i-1}) + Ch(E_{i-1},E_{i-2},E_{i-3}) + A_{i-4} + K_i + W_i
            if (loc.step >= 0 && loc.step < n_steps) {
                deps.push_back({WT_E, loc.step - 4});
                deps.push_back({WT_E, loc.step - 1});
                deps.push_back({WT_E, loc.step - 2});
                deps.push_back({WT_E, loc.step - 3});
                deps.push_back({WT_A, loc.step - 4});
                deps.push_back({WT_W, loc.step});
            }
            break;
        case WT_W:
            // W_i = σ₁(W_{i-2}) + W_{i-7} + σ₀(W_{i-15}) + W_{i-16}
            if (loc.step >= 16) {
                deps.push_back({WT_W, loc.step - 2});
                deps.push_back({WT_W, loc.step - 7});
                deps.push_back({WT_W, loc.step - 15});
                deps.push_back({WT_W, loc.step - 16});
            }
            break;
    }
    return deps;
}

// Compute backjump target level from contradiction locations.
// Checks the conflict bits' own levels and their step-function dependency
// levels. Returns the highest bit_level found strictly below trail_size.
// Returns -1 if no target found (restart needed).
inline int compute_backjump_level(
    const std::vector<BitLoc>& conflicts,
    const std::vector<int>& bit_level,
    int trail_size,
    int n_steps)
{
    int best = -1;
    for (const auto& conflict : conflicts) {
        // Check the conflict bit's own level
        int conf_idx = encode_bitloc(conflict, n_steps);
        if (conf_idx >= 0 && conf_idx < (int)bit_level.size()) {
            int lv = bit_level[conf_idx];
            if (lv >= 0 && lv < trail_size)
                best = std::max(best, lv);
        }
        // Check dependency word bits
        for (const auto& dep : get_dependency_words(conflict, n_steps)) {
            for (int b = 0; b < 32; ++b) {
                BitLoc dep_bit{dep.type, dep.step, b};
                int idx = encode_bitloc(dep_bit, n_steps);
                if (idx >= 0 && idx < (int)bit_level.size()) {
                    int lv = bit_level[idx];
                    if (lv >= 0 && lv < trail_size)
                        best = std::max(best, lv);
                }
            }
        }
    }
    return best;
}

// Compute backjump target from the step where propagation failed.
// Checks all input registers of the failing equation for stamp levels.
// fail_step encoding: 0..n-1 = step function, 1000+i = message expansion W[i].
// Returns the highest bit_level strictly below trail_size, or -1 if no target.
inline int compute_backjump_from_step(
    const std::vector<int>& bit_level, int fail_step,
    int trail_size, int n_steps)
{
    if (fail_step < -1) return -1;  // no info

    int n_total = (int)bit_level.size();
    int best = -1;

    auto check_word = [&](WordType wt, int step) {
        for (int b = 0; b < 32; ++b) {
            BitLoc loc{wt, step, b};
            int idx = encode_bitloc(loc, n_steps);
            if (idx >= 0 && idx < n_total) {
                int lv = bit_level[idx];
                if (lv >= 0 && lv < trail_size)
                    best = std::max(best, lv);
            }
        }
    };

    if (fail_step >= 0 && fail_step < n_steps) {
        // Step function: A_i = f(A_{i-4}..A_{i-1}, E_{i-4}..E_{i-1}, W_i)
        //                E_i = g(E_{i-4}..E_{i-1}, A_{i-4}, W_i)
        int s = fail_step;
        check_word(WT_A, s - 4); check_word(WT_A, s - 3);
        check_word(WT_A, s - 2); check_word(WT_A, s - 1);
        check_word(WT_E, s - 4); check_word(WT_E, s - 3);
        check_word(WT_E, s - 2); check_word(WT_E, s - 1);
        check_word(WT_W, s);
        check_word(WT_A, s); check_word(WT_E, s);
    } else if (fail_step >= 1000) {
        // Message expansion: W_i = σ₁(W_{i-2}) + W_{i-7} + σ₀(W_{i-15}) + W_{i-16}
        int i = fail_step - 1000;
        check_word(WT_W, i - 2); check_word(WT_W, i - 7);
        check_word(WT_W, i - 15); check_word(WT_W, i - 16);
        check_word(WT_W, i);
    }
    // fail_step == -1: twobit failure, no step info → returns -1

    return best;
}

// G6: Mark all trail decisions whose bits are inputs to the failing step as critical.
// Paper §5.1: "in step 7 and 8 of the backtracking we can also mark more than
// one bit as critical."  Each decision on an input register of the failing step
// contributed to the conflict — mark them all.
// Also inserts into critical_bits set (G3) if provided.
inline int mark_step_inputs_critical(
    std::vector<Decision>& trail, int fail_step, int n_steps,
    std::unordered_set<int>* critical_bits = nullptr)
{
    if (fail_step < -1) return 0;
    int marked = 0;

    // Collect the (type, step) pairs for all input registers of the failing equation
    struct RegId { WordType type; int step; };
    std::vector<RegId> inputs;
    if (fail_step >= 0 && fail_step < n_steps) {
        int s = fail_step;
        for (int d = -4; d <= -1; ++d) {
            inputs.push_back({WT_A, s + d});
            inputs.push_back({WT_E, s + d});
        }
        inputs.push_back({WT_A, s});
        inputs.push_back({WT_E, s});
        inputs.push_back({WT_W, s});
    } else if (fail_step >= 1000) {
        int i = fail_step - 1000;
        inputs.push_back({WT_W, i - 2});
        inputs.push_back({WT_W, i - 7});
        inputs.push_back({WT_W, i - 15});
        inputs.push_back({WT_W, i - 16});
        inputs.push_back({WT_W, i});
    }

    // Mark any trail decision that targets one of these input registers
    for (auto& dec : trail) {
        if (dec.critical) continue;  // already marked
        for (auto& inp : inputs) {
            if (dec.loc.type == inp.type && dec.loc.step == inp.step) {
                dec.critical = true;
                ++marked;
                if (critical_bits)
                    critical_bits->insert(encode_bitloc(dec.loc, n_steps));
                break;
            }
        }
    }
    return marked;
}

// Stamp newly changed bits with the decision level that caused them.
// Compares current state vs snapshot; updates bit_level[encoded] = level
// for any bit whose condition changed.
inline void stamp_changed_bits(const CharState& cs, const CharState& snap,
                                std::vector<int>& bit_level, int level, int n_steps) {
    auto do_word = [&](const WordCond& cur, const WordCond& old, int base_idx) {
        uint32_t diff = 0;
        for (int k = 0; k < 4; ++k) diff |= (cur.allow[k] ^ old.allow[k]);
        while (diff) {
            int b = __builtin_ctz(diff);
            bit_level[base_idx + b] = level;
            diff &= diff - 1;
        }
    };
    int n4 = n_steps + 4;
    for (int i = 0; i < n4; ++i) do_word(cs.A[i], snap.A[i], i * 32);
    for (int i = 0; i < n4; ++i) do_word(cs.E[i], snap.E[i], (n4 + i) * 32);
    for (int i = 0; i < 64; ++i) do_word(cs.W[i], snap.W[i], (2 * n4 + i) * 32);
}

// ---- Forward propagation ----
// Propagate conditions through all steps, tightening the characteristic.
// Returns false if a contradiction is detected.

// Helper: impose and track whether change occurred.
inline bool impose_track(WordCond& target, const WordCond& source, bool& changed) {
    WordCond before = target;
    if (!target.impose_word(source)) return false;
    if (target != before) changed = true;
    return true;
}

// Forward declaration: twobit propagation (defined after check_twobit).
inline bool propagate_twobit(CharState& cs, bool& changed);

inline bool propagate_once(CharState& cs, bool& changed, int* fail_step = nullptr) {
    changed = false;
    // fail(s): report failing step and return false.  Step function: s = step
    // index (0..n-1).  Message expansion: s = 1000 + W index (1016..1063).
    auto fail = [&](int s) -> bool { if (fail_step) *fail_step = s; return false; };

    // Forward: for each step, propagate E and A conditions
    for (int step = 0; step < cs.n_steps; ++step) {
        SHA256AltStepCond sc;
        sc.A_im4 = cs.a(step - 4);
        sc.A_im3 = cs.a(step - 3);
        sc.A_im2 = cs.a(step - 2);
        sc.A_im1 = cs.a(step - 1);
        sc.E_im4 = cs.e(step - 4);
        sc.E_im3 = cs.e(step - 3);
        sc.E_im2 = cs.e(step - 2);
        sc.E_im1 = cs.e(step - 1);
        sc.W_i = cs.w(step);
        sc.step = step;

        WordCond E_prop = sc.propagate_E();
        WordCond A_prop = sc.propagate_A(E_prop);

        // Track changes
        WordCond e_before = cs.e(step);
        WordCond a_before = cs.a(step);
        if (!cs.e(step).impose_word(E_prop)) return fail(step);
        if (!cs.a(step).impose_word(A_prop)) return fail(step);
        if (cs.e(step) != e_before || cs.a(step) != a_before)
            changed = true;
    }

    // Message expansion: for steps 16..63, propagate W
    for (int i = 16; i < 64; ++i) {
        SHA256MsgExpCond mc;
        mc.W_im2 = cs.w(i - 2);
        mc.W_im7 = cs.w(i - 7);
        mc.W_im15 = cs.w(i - 15);
        mc.W_im16 = cs.w(i - 16);
        WordCond W_prop = mc.propagate();

        WordCond w_before = cs.w(i);
        if (!cs.w(i).impose_word(W_prop)) return fail(1000 + i);
        if (cs.w(i) != w_before) changed = true;
    }

    // Backward step propagation: from last step to first,
    // use output conditions (E_i, A_i) to tighten inputs.
    for (int step = cs.n_steps - 1; step >= 0; --step) {
        SHA256AltStepCond sc;
        sc.A_im4 = cs.a(step - 4);
        sc.A_im3 = cs.a(step - 3);
        sc.A_im2 = cs.a(step - 2);
        sc.A_im1 = cs.a(step - 1);
        sc.E_im4 = cs.e(step - 4);
        sc.E_im3 = cs.e(step - 3);
        sc.E_im2 = cs.e(step - 2);
        sc.E_im1 = cs.e(step - 1);
        sc.W_i = cs.w(step);
        sc.step = step;

        WordCond E_i = cs.e(step);
        WordCond A_i = cs.a(step);

        // Backward from A_i (tightens E_i and A/E inputs)
        if (!sc.backward_from_A(A_i, E_i)) return fail(step);
        // Backward from E_i (tightens E inputs, A_{i-4}, W_i)
        if (!sc.backward_from_E(E_i)) return fail(step);

        // Impose tightened conditions back to CharState
        if (!impose_track(cs.a(step - 4), sc.A_im4, changed)) return fail(step);
        if (!impose_track(cs.a(step - 3), sc.A_im3, changed)) return fail(step);
        if (!impose_track(cs.a(step - 2), sc.A_im2, changed)) return fail(step);
        if (!impose_track(cs.a(step - 1), sc.A_im1, changed)) return fail(step);
        if (!impose_track(cs.e(step - 4), sc.E_im4, changed)) return fail(step);
        if (!impose_track(cs.e(step - 3), sc.E_im3, changed)) return fail(step);
        if (!impose_track(cs.e(step - 2), sc.E_im2, changed)) return fail(step);
        if (!impose_track(cs.e(step - 1), sc.E_im1, changed)) return fail(step);
        if (!impose_track(cs.w(step), sc.W_i, changed)) return fail(step);
        if (!impose_track(cs.e(step), E_i, changed)) return fail(step);
        if (!impose_track(cs.a(step), A_i, changed)) return fail(step);
    }

    // Backward message expansion: from step 63 down to 16,
    // use W_i conditions to tighten earlier message words.
    for (int i = 63; i >= 16; --i) {
        SHA256MsgExpCond mc;
        mc.W_im2 = cs.w(i - 2);
        mc.W_im7 = cs.w(i - 7);
        mc.W_im15 = cs.w(i - 15);
        mc.W_im16 = cs.w(i - 16);

        WordCond W_i = cs.w(i);
        if (!mc.backward(W_i)) return fail(1000 + i);

        if (!impose_track(cs.w(i - 2), mc.W_im2, changed)) return fail(1000 + i);
        if (!impose_track(cs.w(i - 7), mc.W_im7, changed)) return fail(1000 + i);
        if (!impose_track(cs.w(i - 15), mc.W_im15, changed)) return fail(1000 + i);
        if (!impose_track(cs.w(i - 16), mc.W_im16, changed)) return fail(1000 + i);
        if (!impose_track(cs.w(i), W_i, changed)) return fail(1000 + i);
    }

    return true;
}

// ---- Word-completion propagation ----
// When all inputs to a SHA-256 step become singleton (each bit is exactly one
// of {0,1,u,n}), compute the step function concretely for both copies and
// impose the result on E_i and A_i.  This provides MAXIMUM propagation:
// a fully determined step produces 64 determined output bits.
//
// Trigger condition: all 9 input words (A_{i-4..i-1}, E_{i-4..i-1}, W_i)
// are singleton.  In practice, this fires sequentially: once W[0..6] are
// resolved, steps 0..6 become fully determined in order (since IV is known).
//
// This addresses the Phase 2 stall: without word-completion, carry-chain
// propagation produces zero deductions because intermediate additions have
// free inputs.  With word-completion, each resolved W word immediately
// determines the entire step output.

// Check if a WordCond is fully singleton (each bit has exactly one allowed pair).
inline bool wc_is_singleton(const WordCond& wc) {
    uint32_t multi = (wc.allow[0] & wc.allow[1]) | (wc.allow[0] & wc.allow[2]) |
                     (wc.allow[0] & wc.allow[3]) | (wc.allow[1] & wc.allow[2]) |
                     (wc.allow[1] & wc.allow[3]) | (wc.allow[2] & wc.allow[3]);
    return multi == 0 && wc.is_consistent();
}

// Extract concrete value for copy 1 from a singleton WordCond.
inline uint32_t wc_extract_val1(const WordCond& wc) {
    return wc.allow[1] | wc.allow[3];  // pairs (1,0) or (1,1)
}

// Extract concrete value for copy 2 from a singleton WordCond.
inline uint32_t wc_extract_val2(const WordCond& wc) {
    return wc.allow[2] | wc.allow[3];  // pairs (0,1) or (1,1)
}

// Word-completion propagation pass.  For each step, if all inputs are
// singleton, compute the step function and impose results.
// Returns false on contradiction; sets changed=true if any deductions made.
inline bool word_complete_propagate(CharState& cs, bool& changed) {
    for (int step = 0; step < cs.n_steps; ++step) {
        // Check if all 9 inputs are singleton
        if (!wc_is_singleton(cs.a(step - 4))) continue;
        if (!wc_is_singleton(cs.a(step - 3))) continue;
        if (!wc_is_singleton(cs.a(step - 2))) continue;
        if (!wc_is_singleton(cs.a(step - 1))) continue;
        if (!wc_is_singleton(cs.e(step - 4))) continue;
        if (!wc_is_singleton(cs.e(step - 3))) continue;
        if (!wc_is_singleton(cs.e(step - 2))) continue;
        if (!wc_is_singleton(cs.e(step - 1))) continue;
        if (!wc_is_singleton(cs.w(step)))     continue;

        // Skip if outputs are already singleton (nothing to gain)
        if (wc_is_singleton(cs.e(step)) && wc_is_singleton(cs.a(step))) continue;

        // Extract concrete values for both copies
        uint32_t A_im4_1 = wc_extract_val1(cs.a(step - 4));
        uint32_t A_im3_1 = wc_extract_val1(cs.a(step - 3));
        uint32_t A_im2_1 = wc_extract_val1(cs.a(step - 2));
        uint32_t A_im1_1 = wc_extract_val1(cs.a(step - 1));
        uint32_t E_im4_1 = wc_extract_val1(cs.e(step - 4));
        uint32_t E_im3_1 = wc_extract_val1(cs.e(step - 3));
        uint32_t E_im2_1 = wc_extract_val1(cs.e(step - 2));
        uint32_t E_im1_1 = wc_extract_val1(cs.e(step - 1));
        uint32_t W_1     = wc_extract_val1(cs.w(step));

        uint32_t A_im4_2 = wc_extract_val2(cs.a(step - 4));
        uint32_t A_im3_2 = wc_extract_val2(cs.a(step - 3));
        uint32_t A_im2_2 = wc_extract_val2(cs.a(step - 2));
        uint32_t A_im1_2 = wc_extract_val2(cs.a(step - 1));
        uint32_t E_im4_2 = wc_extract_val2(cs.e(step - 4));
        uint32_t E_im3_2 = wc_extract_val2(cs.e(step - 3));
        uint32_t E_im2_2 = wc_extract_val2(cs.e(step - 2));
        uint32_t E_im1_2 = wc_extract_val2(cs.e(step - 1));
        uint32_t W_2     = wc_extract_val2(cs.w(step));

        // Compute step function for copy 1
        uint32_t E_new_1 = E_im4_1 + sha256::Sigma_1(E_im1_1)
                         + sha256::Ch(E_im1_1, E_im2_1, E_im3_1)
                         + A_im4_1 + sha256::K[step] + W_1;
        uint32_t A_new_1 = -A_im4_1 + sha256::Sigma_0(A_im1_1)
                         + sha256::Maj(A_im1_1, A_im2_1, A_im3_1) + E_new_1;

        // Compute step function for copy 2
        uint32_t E_new_2 = E_im4_2 + sha256::Sigma_1(E_im1_2)
                         + sha256::Ch(E_im1_2, E_im2_2, E_im3_2)
                         + A_im4_2 + sha256::K[step] + W_2;
        uint32_t A_new_2 = -A_im4_2 + sha256::Sigma_0(A_im1_2)
                         + sha256::Maj(A_im1_2, A_im2_2, A_im3_2) + E_new_2;

        // Impose computed results
        WordCond E_result = WordCond::from_pair(E_new_1, E_new_2);
        WordCond A_result = WordCond::from_pair(A_new_1, A_new_2);

        if (!impose_track(cs.e(step), E_result, changed)) return false;
        if (!impose_track(cs.a(step), A_result, changed)) return false;
    }

    // Message expansion word-completion: W[i] = σ₁(W[i-2]) + W[i-7] + σ₀(W[i-15]) + W[i-16]
    // When all 4 inputs are singleton, compute W[i] concretely.
    for (int i = 16; i < cs.n_steps; ++i) {
        if (!wc_is_singleton(cs.w(i - 2)))  continue;
        if (!wc_is_singleton(cs.w(i - 7)))  continue;
        if (!wc_is_singleton(cs.w(i - 15))) continue;
        if (!wc_is_singleton(cs.w(i - 16))) continue;
        if (wc_is_singleton(cs.w(i)))       continue;  // already determined

        uint32_t w1 = sha256::sigma_1(wc_extract_val1(cs.w(i - 2)))
                    + wc_extract_val1(cs.w(i - 7))
                    + sha256::sigma_0(wc_extract_val1(cs.w(i - 15)))
                    + wc_extract_val1(cs.w(i - 16));
        uint32_t w2 = sha256::sigma_1(wc_extract_val2(cs.w(i - 2)))
                    + wc_extract_val2(cs.w(i - 7))
                    + sha256::sigma_0(wc_extract_val2(cs.w(i - 15)))
                    + wc_extract_val2(cs.w(i - 16));

        WordCond W_result = WordCond::from_pair(w1, w2);
        if (!impose_track(cs.w(i), W_result, changed)) return false;
    }

    // Backward word-completion: derive W[i] from known step outputs.
    // E_new = E[-4] + Σ₁(E[-1]) + Ch(E[-1],E[-2],E[-3]) + A[-4] + K + W
    // => W = E_new - E[-4] - Σ₁(E[-1]) - Ch(E[-1],E[-2],E[-3]) - A[-4] - K
    // Fires when E[i], A[-4..-1], E[-4..-1] are all singleton but W[i] is not.
    for (int step = 0; step < cs.n_steps; ++step) {
        if (wc_is_singleton(cs.w(step))) continue;  // W already known
        if (!wc_is_singleton(cs.e(step))) continue;  // need E[step] known
        if (!wc_is_singleton(cs.a(step - 4))) continue;
        if (!wc_is_singleton(cs.a(step - 3))) continue;
        if (!wc_is_singleton(cs.a(step - 2))) continue;
        if (!wc_is_singleton(cs.a(step - 1))) continue;
        if (!wc_is_singleton(cs.e(step - 4))) continue;
        if (!wc_is_singleton(cs.e(step - 3))) continue;
        if (!wc_is_singleton(cs.e(step - 2))) continue;
        if (!wc_is_singleton(cs.e(step - 1))) continue;

        // Compute W = E_new - E[-4] - Σ₁(E[-1]) - Ch(E[-1],E[-2],E[-3]) - A[-4] - K
        uint32_t E_1 = wc_extract_val1(cs.e(step));
        uint32_t W_1 = E_1 - wc_extract_val1(cs.e(step - 4))
                     - sha256::Sigma_1(wc_extract_val1(cs.e(step - 1)))
                     - sha256::Ch(wc_extract_val1(cs.e(step - 1)),
                                  wc_extract_val1(cs.e(step - 2)),
                                  wc_extract_val1(cs.e(step - 3)))
                     - wc_extract_val1(cs.a(step - 4))
                     - sha256::K[step];

        uint32_t E_2 = wc_extract_val2(cs.e(step));
        uint32_t W_2 = E_2 - wc_extract_val2(cs.e(step - 4))
                     - sha256::Sigma_1(wc_extract_val2(cs.e(step - 1)))
                     - sha256::Ch(wc_extract_val2(cs.e(step - 1)),
                                  wc_extract_val2(cs.e(step - 2)),
                                  wc_extract_val2(cs.e(step - 3)))
                     - wc_extract_val2(cs.a(step - 4))
                     - sha256::K[step];

        WordCond W_result = WordCond::from_pair(W_1, W_2);
        if (!impose_track(cs.w(step), W_result, changed)) return false;
    }

    return true;
}

// ---- Wordwise propagation pass ----
// Applies modular integer difference constraints (Alamgir et al. 2024 §5.2)
// to resolve x bits that bitwise propagation cannot determine.
//
// For each SHA-256 alt-step equation and message expansion equation,
// computes intermediate conditions (Σ, σ, Ch, Maj) from current state,
// then applies wordwise_propagate_and_apply to the addition chain.
// Any resolved x→u/n deductions are imposed back to CharState.
// Returns false on contradiction; sets changed=true if any deductions made.

// Helper: check if all bits in a WordCond have well-defined modular
// difference contribution. Returns true if every bit is one of
// {0, 1, -, u, n, x} (the six conditions with known delta).
// Conditions like ?, 3, 5, 7, A, B, C, D, E have multiple possible
// delta values at a single bit position, making wordwise unsound.
inline bool wc_delta_defined(const WordCond& wc) {
    // Valid conditions and their allow[3..0] bit patterns:
    //   0=0001  1=1000  -=1001  u=0010  n=0100  x=0110
    // Invalid if any bit allows both an equal pair AND a diff pair:
    //   ?=1111  3=0011  5=0101  7=0111  A=1010  B=1011  C=1100  D=1101  E=1110
    //
    // A condition is well-defined iff:
    //   (allow[0] | allow[3]) and (allow[1] | allow[2]) don't overlap
    //   at the SAME bit position. Equivalently: no bit is simultaneously
    //   allowed as equal-pair AND as diff-pair.
    uint32_t equal = wc.allow[0] | wc.allow[3];  // bits that can be equal
    uint32_t diff  = wc.allow[1] | wc.allow[2];  // bits that can differ
    return (equal & diff) == 0;
}

inline bool wordwise_pass(CharState& cs, bool& changed) {
    using namespace wordwise;
    using gencond::wc_Sigma0;
    using gencond::wc_Sigma1;
    using gencond::wc_sigma0;
    using gencond::wc_sigma1;
    using gencond::wc_Ch;
    using gencond::wc_Maj;
    using gencond::wc_neg;
    using gencond::wc_from_constant;

    // Fast global check: wordwise can only make deductions if there are
    // x bits somewhere in the characteristic.  In Phase 2 (all ? and x
    // resolved to - and 0/1/u/n), wordwise only checks differential
    // arithmetic, and differentials don't change when '-' → '0' or '1'
    // (both have Δ=0).  So wordwise adds no value in Phase 2.
    {
        uint32_t any_x = 0;
        for (size_t i = 0; i < cs.A.size(); ++i) any_x |= cs.A[i].x_mask();
        if (!any_x) {
            for (size_t i = 0; i < cs.E.size(); ++i) any_x |= cs.E[i].x_mask();
        }
        if (!any_x) {
            for (size_t i = 0; i < cs.W.size(); ++i) any_x |= cs.W[i].x_mask();
        }
        if (!any_x) return true;  // no x bits → wordwise cannot deduce anything
    }

    // E equation per step:
    //   E_i = E_{i-4} + Σ₁(E_{i-1}) + Ch(E_{i-1},E_{i-2},E_{i-3}) + A_{i-4} + K_i + W_i
    for (int step = 0; step < cs.n_steps; ++step) {
        // Early skip: if any input state word has undetermined bits (not
        // in {0,1,-,u,n,x}), the equation's modular diff is ill-defined.
        if (!wc_delta_defined(cs.e(step - 4)) || !wc_delta_defined(cs.e(step - 1)) ||
            !wc_delta_defined(cs.e(step - 2)) || !wc_delta_defined(cs.e(step - 3)) ||
            !wc_delta_defined(cs.a(step - 4)) || !wc_delta_defined(cs.w(step)) ||
            !wc_delta_defined(cs.e(step)))
            continue;

        WordCond sig1 = wc_Sigma1(cs.e(step - 1));
        WordCond ch = wc_Ch(cs.e(step - 1), cs.e(step - 2), cs.e(step - 3));
        WordCond k_cond = wc_from_constant(sha256::K[step]);

        WordCond addends[6] = {
            cs.e(step - 4), sig1, ch, cs.a(step - 4), k_cond, cs.w(step)
        };
        WordCond sum = cs.e(step);

        int deductions = 0;
        if (!wordwise_propagate_and_apply(addends, 6, sum, true, &deductions))
            return false;

        if (deductions > 0) {
            changed = true;
            if (!impose_track(cs.e(step - 4), addends[0], changed)) return false;
            if (!impose_track(cs.a(step - 4), addends[3], changed)) return false;
            if (!impose_track(cs.w(step), addends[5], changed)) return false;
            if (!impose_track(cs.e(step), sum, changed)) return false;
        }
    }

    // A equation per step:
    //   A_i = -A_{i-4} + Σ₀(A_{i-1}) + Maj(A_{i-1},A_{i-2},A_{i-3}) + E_i
    for (int step = 0; step < cs.n_steps; ++step) {
        if (!wc_delta_defined(cs.a(step - 4)) || !wc_delta_defined(cs.a(step - 1)) ||
            !wc_delta_defined(cs.a(step - 2)) || !wc_delta_defined(cs.a(step - 3)) ||
            !wc_delta_defined(cs.e(step)) || !wc_delta_defined(cs.a(step)))
            continue;

        WordCond neg_a = wc_neg(cs.a(step - 4));
        WordCond sig0 = wc_Sigma0(cs.a(step - 1));
        WordCond maj = wc_Maj(cs.a(step - 1), cs.a(step - 2), cs.a(step - 3));

        WordCond addends[4] = { neg_a, sig0, maj, cs.e(step) };
        WordCond sum = cs.a(step);

        int deductions = 0;
        if (!wordwise_propagate_and_apply(addends, 4, sum, true, &deductions))
            return false;

        if (deductions > 0) {
            changed = true;
            if (!impose_track(cs.e(step), addends[3], changed)) return false;
            if (!impose_track(cs.a(step), sum, changed)) return false;
        }
    }

    // Message expansion for steps 16..63:
    //   W_i = σ₁(W_{i-2}) + W_{i-7} + σ₀(W_{i-15}) + W_{i-16}
    for (int i = 16; i < 64; ++i) {
        if (!wc_delta_defined(cs.w(i - 2)) || !wc_delta_defined(cs.w(i - 7)) ||
            !wc_delta_defined(cs.w(i - 15)) || !wc_delta_defined(cs.w(i - 16)) ||
            !wc_delta_defined(cs.w(i)))
            continue;

        WordCond s1 = wc_sigma1(cs.w(i - 2));
        WordCond s0 = wc_sigma0(cs.w(i - 15));

        WordCond addends[4] = { s1, cs.w(i - 7), s0, cs.w(i - 16) };
        WordCond sum = cs.w(i);

        int deductions = 0;
        if (!wordwise_propagate_and_apply(addends, 4, sum, true, &deductions))
            return false;

        if (deductions > 0) {
            changed = true;
            if (!impose_track(cs.w(i - 7), addends[1], changed)) return false;
            if (!impose_track(cs.w(i - 16), addends[3], changed)) return false;
            if (!impose_track(cs.w(i), sum, changed)) return false;
        }
    }

    return true;
}

inline bool propagate_full(CharState& cs, int* fail_step = nullptr,
                           bool skip_twobit_prop = false) {
    for (int iter = 0; iter < 50; ++iter) {
        // Bitwise fixpoint
        bool changed = false;
        if (!propagate_once(cs, changed, fail_step)) return false;
        // Word-completion: fires when all inputs to a step are singleton.
        // Cheap check (mostly skips), but when it fires, produces 64 bits.
        bool wc_changed = false;
        if (!word_complete_propagate(cs, wc_changed)) return false;
        if (wc_changed) changed = true;
        if (!changed) {
            if (skip_twobit_prop) return true;  // bitwise-only fixpoint
            // Bitwise converged — try twobit deductions
            bool tb_changed = false;
            if (!propagate_twobit(cs, tb_changed)) {
                if (fail_step) *fail_step = -1;  // twobit: no step info
                return false;
            }
            if (!tb_changed) return true;  // true fixpoint
            // Twobit made deductions — re-enter bitwise loop
            continue;
        }
    }
    return true;
}

// propagate_full + wordwise: runs bitwise fixpoint then applies wordwise
// modular constraints to resolve x bits.  Use when x bits may exist
// (Phase 1) and you want the extra deduction power.  Significantly more
// expensive than bitwise-only propagate_full on some states.
inline bool propagate_full_ww(CharState& cs) {
    for (int iter = 0; iter < 50; ++iter) {
        bool changed = false;
        if (!propagate_once(cs, changed)) return false;
        if (!changed) {
            // Bitwise converged — try twobit, then wordwise
            bool tb_changed = false;
            if (!propagate_twobit(cs, tb_changed)) return false;
            if (tb_changed) continue;
            bool ww_changed = false;
            if (!wordwise_pass(cs, ww_changed)) return false;
            if (!ww_changed) return true;
            continue;
        }
    }
    return true;
}

// ---- Two-bit consistency check ----
// Extract two-bit conditions from the current characteristic and check
// for contradictions using the Union-Find tracker.

// Helper: get diff status of a single bit condition.
// Returns: 0 if no diff (-, 0, 1), 1 if has diff (x, u, n), -1 if unknown.
inline int bc_diff_status(uint8_t c) {
    if ((c & 0x6) == 0 && c != 0) return 0;  // {0, 1, -}: no diff
    if ((c & 0x9) == 0 && c != 0) return 1;  // {u, n, x}: has diff
    return -1;  // {?, 3, 5, 7, A, B, C, D, E, #}: indeterminate
}

// Helper: build an XorTerm from a CharState word at a bit position.
inline XorTerm make_xor_term(int word_id, int bit, uint8_t cond) {
    int ds = bc_diff_status(cond);
    return XorTerm{{word_id, bit}, ds >= 0, (ds == 1) ? 1 : 0};
}

// Helper: build a known-constant XorTerm (intermediate with determined diff).
inline XorTerm make_xor_known(int val) {
    return XorTerm{{0, 0}, true, val};
}

// Extract addition twobit conditions from modular addition equations.
// At bit j, if carry diff is known to be 0, the multi-input addition
// satisfies: Δsum[j] = Δa₁[j] ⊕ Δa₂[j] ⊕ ... ⊕ Δaₖ[j].
// Carry diff = 0 at bit 0 always, and at bit j>0 if ALL addend diffs
// at bits 0..j-1 are 0 (no diff → carry chain identical in both messages).
//
// Sigma/sigma are LINEAR (XOR), so their contribution is inlined:
//   Σ₁(E_{i-1})[j] → E_{i-1}[(j+6)%32] ⊕ E_{i-1}[(j+11)%32] ⊕ E_{i-1}[(j+25)%32]
// This avoids treating Sigma outputs as opaque intermediates and lets all
// terms (except Ch/Maj) be direct CharState bits → more 2-unknown extractions.
//
// Ch and Maj outputs are nonlinear: treated as known constants when their
// diff is determined, else counted as opaque unknowns.
inline bool extract_addition_twobit(const CharState& cs, TwoBitTracker& tracker) {
    int n = cs.n_steps;
    auto a_word_id = [](int step) { return step + 4; };
    auto e_word_id = [&](int step) { return (n + 4) + step + 4; };
    auto w_word_id = [&](int i) { return 2 * (n + 4) + i; };

    static const BoolPropTable tbl_ch = BoolPropTable::build(TT_IF);
    static const BoolPropTable tbl_maj = BoolPropTable::build(TT_MAJ);

    // E equation per step (expanded, bit j, carry diff = 0):
    //   ΔE_i[j] ⊕ ΔE_{i-4}[j] ⊕ ΔE_{i-1}[(j+6)%32] ⊕ ΔE_{i-1}[(j+11)%32]
    //   ⊕ ΔE_{i-1}[(j+25)%32] ⊕ ΔCh_out[j] ⊕ ΔA_{i-4}[j] ⊕ ΔW_i[j] = 0
    for (int step = 0; step < n; ++step) {
        for (int j = 0; j < 32; ++j) {
            XorTerm terms[8];
            int n_opaque = 0;
            int p1 = (j + 6) & 31, p2 = (j + 11) & 31, p3 = (j + 25) & 31;

            terms[0] = make_xor_term(e_word_id(step), j, cs.e(step).get(j));
            terms[1] = make_xor_term(e_word_id(step - 4), j, cs.e(step - 4).get(j));
            terms[2] = make_xor_term(e_word_id(step - 1), p1, cs.e(step - 1).get(p1));
            terms[3] = make_xor_term(e_word_id(step - 1), p2, cs.e(step - 1).get(p2));
            terms[4] = make_xor_term(e_word_id(step - 1), p3, cs.e(step - 1).get(p3));
            terms[5] = make_xor_term(a_word_id(step - 4), j, cs.a(step - 4).get(j));
            terms[6] = make_xor_term(w_word_id(step), j, cs.w(step).get(j));

            // Ch output: nonlinear intermediate
            uint8_t ch_out = tbl_ch.propagate(
                BitCond(cs.e(step - 1).get(j)),
                BitCond(cs.e(step - 2).get(j)),
                BitCond(cs.e(step - 3).get(j)));
            int ch_ds = bc_diff_status(ch_out);
            if (ch_ds >= 0) {
                terms[7] = make_xor_known(ch_ds);
            } else {
                terms[7] = make_xor_known(0);
                n_opaque++;
            }

            if (n_opaque == 0) {
                bool contra = false;
                auto conds = extract_xor_twobit(terms, 8, 0, &contra);
                if (contra) return false;
                for (auto& c : conds)
                    if (!tracker.add(c)) return false;
            }

            // Carry chain: continue to bit j+1 if ALL addend diffs at j are 0.
            // Addends (original, not expanded): E_{i-4}, Σ₁(E_{i-1}), Ch, A_{i-4}, W_i
            // Σ₁ diff = XOR of its 3 components (terms[2..4]).
            bool chain_ok = true;
            // E_{i-4}
            if (!terms[1].known || terms[1].val != 0) chain_ok = false;
            // Σ₁ output: XOR of 3 rotated components
            if (chain_ok) {
                if (!terms[2].known || !terms[3].known || !terms[4].known)
                    chain_ok = false;
                else if ((terms[2].val ^ terms[3].val ^ terms[4].val) != 0)
                    chain_ok = false;
            }
            // Ch output
            if (chain_ok && ch_ds != 0) chain_ok = false;
            // A_{i-4}, W_i
            if (chain_ok && (!terms[5].known || terms[5].val != 0)) chain_ok = false;
            if (chain_ok && (!terms[6].known || terms[6].val != 0)) chain_ok = false;
            if (chain_ok)
                continue;
            break;
        }
    }

    // A equation per step (expanded, bit j, carry diff = 0):
    //   ΔA_i[j] ⊕ ΔA_{i-4}[j] ⊕ ΔA_{i-1}[(j+2)%32] ⊕ ΔA_{i-1}[(j+13)%32]
    //   ⊕ ΔA_{i-1}[(j+22)%32] ⊕ ΔMaj_out[j] ⊕ ΔE_i[j] = 0
    // Negation note: Δ(-x)[j] = Δx[j] when all Δx[b]=0 for b<j (carry=0).
    for (int step = 0; step < n; ++step) {
        for (int j = 0; j < 32; ++j) {
            XorTerm terms[7];
            int n_opaque = 0;
            int p1 = (j + 2) & 31, p2 = (j + 13) & 31, p3 = (j + 22) & 31;

            terms[0] = make_xor_term(a_word_id(step), j, cs.a(step).get(j));
            terms[1] = make_xor_term(a_word_id(step - 4), j, cs.a(step - 4).get(j));
            terms[2] = make_xor_term(a_word_id(step - 1), p1, cs.a(step - 1).get(p1));
            terms[3] = make_xor_term(a_word_id(step - 1), p2, cs.a(step - 1).get(p2));
            terms[4] = make_xor_term(a_word_id(step - 1), p3, cs.a(step - 1).get(p3));
            terms[5] = make_xor_term(e_word_id(step), j, cs.e(step).get(j));

            uint8_t maj_out = tbl_maj.propagate(
                BitCond(cs.a(step - 1).get(j)),
                BitCond(cs.a(step - 2).get(j)),
                BitCond(cs.a(step - 3).get(j)));
            int maj_ds = bc_diff_status(maj_out);
            if (maj_ds >= 0) {
                terms[6] = make_xor_known(maj_ds);
            } else {
                terms[6] = make_xor_known(0);
                n_opaque++;
            }

            if (n_opaque == 0) {
                bool contra = false;
                auto conds = extract_xor_twobit(terms, 7, 0, &contra);
                if (contra) return false;
                for (auto& c : conds)
                    if (!tracker.add(c)) return false;
            }

            // Carry chain: addends are -A_{i-4}, Σ₀(A_{i-1}), Maj, E_i.
            // Σ₀ diff = XOR of terms[2..4].
            // Negation: Δ(-A_{i-4})[j] = ΔA_{i-4}[j] when lower bits all 0.
            bool chain_ok = true;
            // A_{i-4} (also validates negation since diff must be 0)
            if (!terms[1].known || terms[1].val != 0) chain_ok = false;
            // Σ₀ output
            if (chain_ok) {
                if (!terms[2].known || !terms[3].known || !terms[4].known)
                    chain_ok = false;
                else if ((terms[2].val ^ terms[3].val ^ terms[4].val) != 0)
                    chain_ok = false;
            }
            // Maj output, E_i
            if (chain_ok && maj_ds != 0) chain_ok = false;
            if (chain_ok && (!terms[5].known || terms[5].val != 0)) chain_ok = false;
            if (chain_ok)
                continue;
            break;
        }
    }

    // W expansion: W_i = σ₁(W_{i-2}) + W_{i-7} + σ₀(W_{i-15}) + W_{i-16}
    // σ₁: rotr17 ^ rotr19 ^ shr10     σ₀: rotr7 ^ rotr18 ^ shr3
    // Expanded (bit j, carry diff = 0):
    //   ΔW_i[j] ⊕ ΔW_{i-2}[(j+17)%32] ⊕ ΔW_{i-2}[(j+19)%32] ⊕ shift₁[j]
    //   ⊕ ΔW_{i-7}[j] ⊕ ΔW_{i-15}[(j+7)%32] ⊕ ΔW_{i-15}[(j+18)%32] ⊕ shift₀[j]
    //   ⊕ ΔW_{i-16}[j] = 0
    // shift terms: shr10 → W_{i-2}[j+10] if j+10<32, else 0 (known no-diff)
    //              shr3  → W_{i-15}[j+3] if j+3<32, else 0
    for (int i = 16; i < 64; ++i) {
        for (int j = 0; j < 32; ++j) {
            XorTerm terms[9];
            int nt = 0;

            terms[nt++] = make_xor_term(w_word_id(i), j, cs.w(i).get(j));

            // σ₁(W_{i-2}): rotr17 ^ rotr19 ^ shr10
            int q1 = (j + 17) & 31, q2 = (j + 19) & 31, q3 = j + 10;
            terms[nt++] = make_xor_term(w_word_id(i - 2), q1, cs.w(i - 2).get(q1));
            terms[nt++] = make_xor_term(w_word_id(i - 2), q2, cs.w(i - 2).get(q2));
            if (q3 < 32)
                terms[nt++] = make_xor_term(w_word_id(i - 2), q3, cs.w(i - 2).get(q3));
            else
                terms[nt++] = make_xor_known(0);  // shifted out → 0

            terms[nt++] = make_xor_term(w_word_id(i - 7), j, cs.w(i - 7).get(j));

            // σ₀(W_{i-15}): rotr7 ^ rotr18 ^ shr3
            int r1 = (j + 7) & 31, r2 = (j + 18) & 31, r3 = j + 3;
            terms[nt++] = make_xor_term(w_word_id(i - 15), r1, cs.w(i - 15).get(r1));
            terms[nt++] = make_xor_term(w_word_id(i - 15), r2, cs.w(i - 15).get(r2));
            if (r3 < 32)
                terms[nt++] = make_xor_term(w_word_id(i - 15), r3, cs.w(i - 15).get(r3));
            else
                terms[nt++] = make_xor_known(0);

            terms[nt++] = make_xor_term(w_word_id(i - 16), j, cs.w(i - 16).get(j));

            // No opaque intermediates — all terms are CharState bits or known
            bool contra = false;
            auto conds = extract_xor_twobit(terms, nt, 0, &contra);
            if (contra) return false;
            for (auto& c : conds)
                if (!tracker.add(c)) return false;

            // Carry chain: addends are σ₁(W_{i-2}), W_{i-7}, σ₀(W_{i-15}), W_{i-16}.
            // σ₁ diff = XOR of terms[1..3], σ₀ diff = XOR of terms[5..7].
            // Layout: [0]=W_i, [1..3]=σ₁ comps, [4]=W_{i-7}, [5..7]=σ₀ comps, [8]=W_{i-16}
            bool chain_ok = true;
            // σ₁ output
            if (!terms[1].known || !terms[2].known || !terms[3].known)
                chain_ok = false;
            else if ((terms[1].val ^ terms[2].val ^ terms[3].val) != 0)
                chain_ok = false;
            // W_{i-7}
            if (chain_ok && (!terms[4].known || terms[4].val != 0)) chain_ok = false;
            // σ₀ output
            if (chain_ok) {
                if (!terms[5].known || !terms[6].known || !terms[7].known)
                    chain_ok = false;
                else if ((terms[5].val ^ terms[6].val ^ terms[7].val) != 0)
                    chain_ok = false;
            }
            // W_{i-16}
            if (chain_ok && (!terms[8].known || terms[8].val != 0)) chain_ok = false;
            if (chain_ok)
                continue;
            break;
        }
    }

    return true;
}

inline bool check_twobit(const CharState& cs) {
    // Number words: A steps as 0..n+3, E steps as n+4..2n+7, W as 2n+8..2n+71
    int n = cs.n_steps;
    int total_words = (n + 4) + (n + 4) + 64;
    TwoBitTracker tracker(total_words);

    auto a_word_id = [](int step) { return step + 4; };
    auto e_word_id = [&](int step) { return (n + 4) + step + 4; };
    auto w_word_id = [&](int i) { return 2 * (n + 4) + i; };

    static const BoolPropTable tbl_ch = BoolPropTable::build(TT_IF);
    static const BoolPropTable tbl_maj = BoolPropTable::build(TT_MAJ);
    (void)tbl_ch; (void)tbl_maj;

    // Extract from Maj(A_{i-1}, A_{i-2}, A_{i-3}) for each step
    for (int step = 0; step < n; ++step) {
        for (int b = 0; b < 32; ++b) {
            uint8_t ca1 = cs.a(step - 1).get(b);
            uint8_t ca2 = cs.a(step - 2).get(b);
            uint8_t ca3 = cs.a(step - 3).get(b);
            // Compute output condition of Maj at this bit
            uint8_t out = tbl_maj.propagate(BitCond(ca1), BitCond(ca2), BitCond(ca3));
            auto conds = extract_maj_twobit(
                a_word_id(step - 1), a_word_id(step - 2), a_word_id(step - 3), b,
                ca1, ca2, ca3, out);
            for (auto& c : conds) {
                if (!tracker.add(c)) return false;
            }
        }
    }

    // Extract from Ch(E_{i-1}, E_{i-2}, E_{i-3}) for each step
    for (int step = 0; step < n; ++step) {
        for (int b = 0; b < 32; ++b) {
            uint8_t ce1 = cs.e(step - 1).get(b);
            uint8_t ce2 = cs.e(step - 2).get(b);
            uint8_t ce3 = cs.e(step - 3).get(b);
            uint8_t out = tbl_ch.propagate(BitCond(ce1), BitCond(ce2), BitCond(ce3));
            auto conds = extract_ch_twobit(
                e_word_id(step - 1), e_word_id(step - 2), e_word_id(step - 3), b,
                ce1, ce2, ce3, out);
            for (auto& c : conds) {
                if (!tracker.add(c)) return false;
            }
        }
    }

    // Extract from Σ₀(A_{i-1}) for each step
    // Σ₀ = rotr2 ^ rotr13 ^ rotr22
    for (int step = 0; step < n; ++step) {
        uint8_t in_conds[32], out_conds[32];
        WordCond sig0_out = wc_Sigma0(cs.a(step - 1));
        for (int b = 0; b < 32; ++b) {
            in_conds[b] = cs.a(step - 1).get(b);
            out_conds[b] = sig0_out.get(b);
        }
        auto conds = extract_sigma_twobit(a_word_id(step - 1), 2, 13, 22, false,
                                           in_conds, out_conds);
        for (auto& c : conds) {
            if (!tracker.add(c)) return false;
        }
    }

    // Extract from Σ₁(E_{i-1}) for each step
    // Σ₁ = rotr6 ^ rotr11 ^ rotr25
    for (int step = 0; step < n; ++step) {
        uint8_t in_conds[32], out_conds[32];
        WordCond sig1_out = wc_Sigma1(cs.e(step - 1));
        for (int b = 0; b < 32; ++b) {
            in_conds[b] = cs.e(step - 1).get(b);
            out_conds[b] = sig1_out.get(b);
        }
        auto conds = extract_sigma_twobit(e_word_id(step - 1), 6, 11, 25, false,
                                           in_conds, out_conds);
        for (auto& c : conds) {
            if (!tracker.add(c)) return false;
        }
    }

    // Extract from σ₀(W_{i-15}) in message expansion: W_i = σ₁(W_{i-2}) + W_{i-7} + σ₀(W_{i-15}) + W_{i-16}
    // σ₀ = rotr7 ^ rotr18 ^ shr3 (is_shift3 = true)
    for (int i = 16; i < 64; ++i) {
        uint8_t in_conds[32], out_conds[32];
        WordCond s0_out = wc_sigma0(cs.w(i - 15));
        for (int b = 0; b < 32; ++b) {
            in_conds[b] = cs.w(i - 15).get(b);
            out_conds[b] = s0_out.get(b);
        }
        auto conds = extract_sigma_twobit(w_word_id(i - 15), 7, 18, 3, true,
                                           in_conds, out_conds);
        for (auto& c : conds) {
            if (!tracker.add(c)) return false;
        }
    }

    // Extract from σ₁(W_{i-2}) in message expansion
    // σ₁ = rotr17 ^ rotr19 ^ shr10 (is_shift3 = true)
    for (int i = 16; i < 64; ++i) {
        uint8_t in_conds[32], out_conds[32];
        WordCond s1_out = wc_sigma1(cs.w(i - 2));
        for (int b = 0; b < 32; ++b) {
            in_conds[b] = cs.w(i - 2).get(b);
            out_conds[b] = s1_out.get(b);
        }
        auto conds = extract_sigma_twobit(w_word_id(i - 2), 17, 19, 10, true,
                                           in_conds, out_conds);
        for (auto& c : conds) {
            if (!tracker.add(c)) return false;
        }
    }

    // Extract from modular additions (cross-word diff-domain constraints)
    if (!extract_addition_twobit(cs, tracker)) return false;

    return true;
}

// ---- Two-bit propagation (§4.4 turned into deductions) ----
// Builds the same Union-Find as check_twobit, then for each equivalence class,
// if any member has a decided condition, forces all other class members to the
// value implied by parity.  This cascades fixings through cross-equation
// two-bit relations that bitwise propagation cannot see.
//
// Two-bit parity semantics:
//   REL_EQUAL (parity 0): both bits have the same "value category"
//     equal-pair ↔ equal-pair:  - ↔ -,  0 ↔ 0,  1 ↔ 1
//     diff-pair  ↔ diff-pair:   x ↔ x,  u ↔ u,  n ↔ n
//   REL_UNEQUAL (parity 1): opposite categories
//     equal ↔ diff:  - ↔ x,  0 ↔ u,  1 ↔ n   (and vice versa)
//
// Returns false on contradiction, sets changed=true if any deductions made.

inline bool propagate_twobit(CharState& cs, bool& changed) {
    int n = cs.n_steps;
    int total_words = (n + 4) + (n + 4) + 64;
    TwoBitTracker tracker(total_words);

    auto a_word_id = [](int step) { return step + 4; };
    auto e_word_id = [&](int step) { return (n + 4) + step + 4; };
    auto w_word_id = [&](int i) { return 2 * (n + 4) + i; };
    int w_base = 2 * (n + 4);

    static const BoolPropTable tbl_ch = BoolPropTable::build(TT_IF);
    static const BoolPropTable tbl_maj = BoolPropTable::build(TT_MAJ);

    // --- Build the Union-Find (same extraction as check_twobit) ---
    for (int step = 0; step < n; ++step) {
        for (int b = 0; b < 32; ++b) {
            uint8_t ca1 = cs.a(step-1).get(b), ca2 = cs.a(step-2).get(b), ca3 = cs.a(step-3).get(b);
            uint8_t out = tbl_maj.propagate(BitCond(ca1), BitCond(ca2), BitCond(ca3));
            for (auto& c : extract_maj_twobit(a_word_id(step-1), a_word_id(step-2), a_word_id(step-3), b, ca1, ca2, ca3, out))
                if (!tracker.add(c)) return false;
        }
    }
    for (int step = 0; step < n; ++step) {
        for (int b = 0; b < 32; ++b) {
            uint8_t ce1 = cs.e(step-1).get(b), ce2 = cs.e(step-2).get(b), ce3 = cs.e(step-3).get(b);
            uint8_t out = tbl_ch.propagate(BitCond(ce1), BitCond(ce2), BitCond(ce3));
            for (auto& c : extract_ch_twobit(e_word_id(step-1), e_word_id(step-2), e_word_id(step-3), b, ce1, ce2, ce3, out))
                if (!tracker.add(c)) return false;
        }
    }
    for (int step = 0; step < n; ++step) {
        uint8_t in_c[32], out_c[32];
        WordCond s0 = wc_Sigma0(cs.a(step-1));
        for (int b = 0; b < 32; ++b) { in_c[b] = cs.a(step-1).get(b); out_c[b] = s0.get(b); }
        for (auto& c : extract_sigma_twobit(a_word_id(step-1), 2, 13, 22, false, in_c, out_c))
            if (!tracker.add(c)) return false;
    }
    for (int step = 0; step < n; ++step) {
        uint8_t in_c[32], out_c[32];
        WordCond s1 = wc_Sigma1(cs.e(step-1));
        for (int b = 0; b < 32; ++b) { in_c[b] = cs.e(step-1).get(b); out_c[b] = s1.get(b); }
        for (auto& c : extract_sigma_twobit(e_word_id(step-1), 6, 11, 25, false, in_c, out_c))
            if (!tracker.add(c)) return false;
    }
    for (int i = 16; i < 64; ++i) {
        uint8_t in_c[32], out_c[32];
        WordCond s0 = wc_sigma0(cs.w(i-15));
        for (int b = 0; b < 32; ++b) { in_c[b] = cs.w(i-15).get(b); out_c[b] = s0.get(b); }
        for (auto& c : extract_sigma_twobit(w_word_id(i-15), 7, 18, 3, true, in_c, out_c))
            if (!tracker.add(c)) return false;
    }
    for (int i = 16; i < 64; ++i) {
        uint8_t in_c[32], out_c[32];
        WordCond s1 = wc_sigma1(cs.w(i-2));
        for (int b = 0; b < 32; ++b) { in_c[b] = cs.w(i-2).get(b); out_c[b] = s1.get(b); }
        for (auto& c : extract_sigma_twobit(w_word_id(i-2), 17, 19, 10, true, in_c, out_c))
            if (!tracker.add(c)) return false;
    }

    // Addition diff-domain twobit (cross-word constraints)
    if (!extract_addition_twobit(cs, tracker)) return false;

    // --- Extract deductions from equivalence classes ---
    // For efficiency: collect all nodes, group by root. For each group,
    // if any member has a decided condition, force all undecided members.

    // Helper: read condition for a node_id
    int total_nodes = total_words * 32;
    auto node_cond = [&](int node_id) -> BitCond {
        int word = node_id / 32, bit = node_id % 32;
        if (word < n + 4)          return cs.A[word].get(bit);
        if (word < 2 * (n + 4))    return cs.E[word - (n + 4)].get(bit);
        return cs.W[word - w_base].get(bit);
    };

    // Helper: impose condition on a node_id, return false on contradiction
    auto impose_node = [&](int node_id, BitCond cond) -> bool {
        int word = node_id / 32, bit = node_id % 32;
        WordCond* wc;
        if (word < n + 4)          wc = &cs.A[word];
        else if (word < 2*(n+4))   wc = &cs.E[word - (n + 4)];
        else                       wc = &cs.W[word - w_base];
        BitCond cur = wc->get(bit);
        BitCond meet = bc_meet(cur, cond);
        if (bc_is_contradiction(meet)) return false;
        if (meet != cur) { wc->set(bit, meet); changed = true; }
        return true;
    };

    // Transform a decided condition through parity:
    //   parity 0 (equal): keep as-is
    //   parity 1 (unequal): swap diff category
    auto flip_parity = [](BitCond c) -> BitCond {
        // Twobit tracks diff-parity only: '-' ↔ 'x'
        if (c == BC_DASH) return BC_X;
        if (c == BC_X)    return BC_DASH;
        return c;  // shouldn't happen after generalization
    };

    // Generalize a concrete condition to its diff category.
    // Twobit tracks whether bits differ between messages (diff parity),
    // NOT specific values or signs.  '0','1' → '-' (no diff);
    // 'u','n' → 'x' (has diff).  '-' and 'x' are already categories.
    auto to_diff_category = [](BitCond c) -> BitCond {
        uint8_t v = static_cast<uint8_t>(c);
        if ((v & 0x6) == 0 && v != 0) return BC_DASH;  // '0','1','-' → no diff
        if ((v & 0x9) == 0 && v != 0) return BC_X;     // 'u','n','x' → has diff
        return c;  // multi-valued: no useful deduction
    };

    // Scan all nodes: for each group, propagate the strongest decided member
    // First pass: find the "anchor" condition for each root
    // Direct-indexed: root_anchor[root_id] stores anchor info.
    struct AnchorInfo { BitCond cond; uint8_t parity; };
    std::vector<int8_t> root_has_anchor(total_nodes, 0);  // 0 = no anchor
    std::vector<AnchorInfo> root_anchor(total_nodes, {BC_FREE, 0});

    // A condition is useful as anchor if it resolves to a definite diff category.
    for (int node_id = 0; node_id < total_nodes; ++node_id) {
        BitCond c = node_cond(node_id);
        BitCond cat = to_diff_category(c);
        if (cat != BC_DASH && cat != BC_X) continue;  // not useful as anchor

        auto [root, par] = tracker.find(node_id);
        if (root_has_anchor[root]) continue;  // already have an anchor

        root_has_anchor[root] = 1;
        root_anchor[root] = {cat, par};
    }

    // Second pass: for each node, if its root has an anchor, compute the
    // implied condition and impose it (O(1) lookup per node)
    for (int node_id = 0; node_id < total_nodes; ++node_id) {
        auto [root, par] = tracker.find(node_id);
        if (!root_has_anchor[root]) continue;

        auto& ai = root_anchor[root];

        // Relative parity between this node and the anchor
        uint8_t rel_par = par ^ ai.parity;

        // Compute implied condition
        BitCond implied = (rel_par == 0) ? ai.cond : flip_parity(ai.cond);
        if (!impose_node(node_id, implied)) return false;
    }

    return true;
}

// ---- Bit selection ----

// Connectivity scoring: count two-bit conditions per bit location (§5.1).
// Returns a flat array indexed by node_id, where each BitLoc maps to one node.
// Uses the same word_id scheme as check_twobit.
struct ConnectivityMap {
    int n_steps;
    // Flat arrays: a[(step+4)*32 + bit], e[(step+4)*32 + bit], w[step*32 + bit]
    std::vector<int> a_conn, e_conn, w_conn;

    int get(const BitLoc& loc) const {
        switch (loc.type) {
            case WT_A: return a_conn[(loc.step + 4) * 32 + loc.bit];
            case WT_E: return e_conn[(loc.step + 4) * 32 + loc.bit];
            case WT_W: return w_conn[loc.step * 32 + loc.bit];
        }
        return 0;
    }
};

inline ConnectivityMap compute_connectivity(const CharState& cs) {
    int n = cs.n_steps;
    ConnectivityMap cm;
    cm.n_steps = n;
    cm.a_conn.assign((n + 4) * 32, 0);
    cm.e_conn.assign((n + 4) * 32, 0);
    cm.w_conn.assign(64 * 32, 0);

    // Map check_twobit word_ids to connectivity arrays
    // a_word_id(step) = step + 4  → a_conn index = (step+4)*32 + bit
    // e_word_id(step) = (n+4) + step + 4
    // w_word_id(i) = 2*(n+4) + i
    int w_base = 2 * (n + 4);
    auto inc = [&](const BitId& id) {
        if (id.word >= w_base) {
            cm.w_conn[(id.word - w_base) * 32 + id.bit]++;
        } else if (id.word >= n + 4) {
            cm.e_conn[(id.word - (n + 4)) * 32 + id.bit]++;
        } else {
            cm.a_conn[id.word * 32 + id.bit]++;
        }
    };

    auto count_cond = [&](const TwoBitCond& c) {
        inc(c.a);
        inc(c.b);
    };

    static const BoolPropTable tbl_ch = BoolPropTable::build(TT_IF);
    static const BoolPropTable tbl_maj = BoolPropTable::build(TT_MAJ);

    auto a_word_id = [](int step) { return step + 4; };
    auto e_word_id = [&](int step) { return (n + 4) + step + 4; };
    auto w_word_id = [&](int i) { return 2 * (n + 4) + i; };

    // Maj two-bit conditions
    for (int step = 0; step < n; ++step) {
        for (int b = 0; b < 32; ++b) {
            uint8_t ca1 = cs.a(step-1).get(b), ca2 = cs.a(step-2).get(b), ca3 = cs.a(step-3).get(b);
            uint8_t out = tbl_maj.propagate(BitCond(ca1), BitCond(ca2), BitCond(ca3));
            for (auto& c : extract_maj_twobit(a_word_id(step-1), a_word_id(step-2), a_word_id(step-3), b, ca1, ca2, ca3, out))
                count_cond(c);
        }
    }

    // Ch two-bit conditions
    for (int step = 0; step < n; ++step) {
        for (int b = 0; b < 32; ++b) {
            uint8_t ce1 = cs.e(step-1).get(b), ce2 = cs.e(step-2).get(b), ce3 = cs.e(step-3).get(b);
            uint8_t out = tbl_ch.propagate(BitCond(ce1), BitCond(ce2), BitCond(ce3));
            for (auto& c : extract_ch_twobit(e_word_id(step-1), e_word_id(step-2), e_word_id(step-3), b, ce1, ce2, ce3, out))
                count_cond(c);
        }
    }

    // Σ₀ two-bit conditions
    for (int step = 0; step < n; ++step) {
        uint8_t in_conds[32], out_conds[32];
        WordCond sig0_out = wc_Sigma0(cs.a(step-1));
        for (int b = 0; b < 32; ++b) { in_conds[b] = cs.a(step-1).get(b); out_conds[b] = sig0_out.get(b); }
        for (auto& c : extract_sigma_twobit(a_word_id(step-1), 2, 13, 22, false, in_conds, out_conds))
            count_cond(c);
    }

    // Σ₁ two-bit conditions
    for (int step = 0; step < n; ++step) {
        uint8_t in_conds[32], out_conds[32];
        WordCond sig1_out = wc_Sigma1(cs.e(step-1));
        for (int b = 0; b < 32; ++b) { in_conds[b] = cs.e(step-1).get(b); out_conds[b] = sig1_out.get(b); }
        for (auto& c : extract_sigma_twobit(e_word_id(step-1), 6, 11, 25, false, in_conds, out_conds))
            count_cond(c);
    }

    // σ₀(W_{i-15}) two-bit conditions (message expansion)
    for (int i = 16; i < 64; ++i) {
        uint8_t in_conds[32], out_conds[32];
        WordCond s0_out = wc_sigma0(cs.w(i-15));
        for (int b = 0; b < 32; ++b) { in_conds[b] = cs.w(i-15).get(b); out_conds[b] = s0_out.get(b); }
        for (auto& c : extract_sigma_twobit(w_word_id(i-15), 7, 18, 3, true, in_conds, out_conds))
            count_cond(c);
    }

    // σ₁(W_{i-2}) two-bit conditions (message expansion)
    for (int i = 16; i < 64; ++i) {
        uint8_t in_conds[32], out_conds[32];
        WordCond s1_out = wc_sigma1(cs.w(i-2));
        for (int b = 0; b < 32; ++b) { in_conds[b] = cs.w(i-2).get(b); out_conds[b] = s1_out.get(b); }
        for (auto& c : extract_sigma_twobit(w_word_id(i-2), 17, 19, 10, true, in_conds, out_conds))
            count_cond(c);
    }

    return cm;
}

// Collect all Phase 1 undecided bits ('?' and 'x').
// Only scans W[0..n_steps-1] — words beyond the attack scope are
// expansion-derived and resolved automatically once inputs are fixed.
inline std::vector<BitLoc> collect_phase1_bits(const CharState& cs) {
    std::vector<BitLoc> bits;
    auto scan = [&](WordType wt, const std::vector<WordCond>& words,
                    int step_offset, int limit = -1) {
        int n = (limit >= 0) ? std::min(limit, (int)words.size())
                             : (int)words.size();
        for (int i = 0; i < n; ++i) {
            int step = i + step_offset;
            uint32_t free = words[i].free_mask();
            uint32_t xm = words[i].x_mask();
            uint32_t mask = free | xm;
            while (mask) {
                int b = __builtin_ctz(mask);
                bits.push_back({wt, step, b});
                mask &= mask - 1;
            }
        }
    };
    scan(WT_A, cs.A, -4);
    scan(WT_E, cs.E, -4);
    scan(WT_W, cs.W, 0, cs.n_steps);  // W[0..n_steps-1] only
    return bits;
}

// Collect W[0..15] dash bits for message-first search strategy.
// Prioritizes message bits before Phase 1 because fixing message values
// naturally resolves Phase 1 (?/x) bits through step-function propagation.
// Without this, Phase 1 randomly fixes diff patterns that are almost always
// incompatible with any valid collision, making Phase 2 impossible.
inline std::vector<BitLoc> collect_msg_dash_bits(const CharState& cs) {
    std::vector<BitLoc> bits;
    int limit = std::min(16, cs.n_steps);
    for (int i = 0; i < limit; ++i) {
        uint32_t dm = cs.W[i].dash_mask();
        while (dm) {
            int b = __builtin_ctz(dm);
            bits.push_back({WT_W, i, b});
            dm &= dm - 1;
        }
    }
    return bits;
}

// Collect Stage 1 bits (Mendel-Nad-Schläffer 2013, §5.2):
// W-only '?' and 'x' bits, sorted with LATER steps first.
//
// Rationale: message expansion W[16+] is derived from W[0..15] via
// σ₀/σ₁ operations. Resolving W bits first (biased toward later steps)
// means expansion constraints produce maximum forward propagation,
// determining most of the state before A/E decisions are needed.
// This reduces the effective search space for subsequent stages.
//
// Used in 3-stage strategy: Stage 1 (W-only) → Stage 2 (A/E) → Stage 3 (pair).
// Backward-compatible: can replace collect_phase1_bits() in existing search.
inline std::vector<BitLoc> collect_stage1_bits(const CharState& cs) {
    std::vector<BitLoc> bits;
    int limit = std::min(cs.n_steps, (int)cs.W.size());
    for (int i = 0; i < limit; ++i) {
        uint32_t free = cs.W[i].free_mask();
        uint32_t xm = cs.W[i].x_mask();
        uint32_t mask = free | xm;
        while (mask) {
            int b = __builtin_ctz(mask);
            bits.push_back({WT_W, i, b});
            mask &= mask - 1;
        }
    }
    // Sort later steps first (higher step index = earlier in decision queue).
    // Within same step, random order is fine.
    std::sort(bits.begin(), bits.end(),
              [](const BitLoc& a, const BitLoc& b) {
                  return a.step > b.step;
              });
    return bits;
}

// Collect Stage 2 bits (Mendel-Nad-Schläffer 2013, §5.2):
// A/E only '?' and 'x' bits, with A-biased ordering.
//
// Rationale: after Stage 1 resolves all W bits, A/E bits remain.
// A-bias means A bits are decided before E bits. Since A feeds into
// the next step's E via the step function recurrence, sparser A
// characteristics cascade into sparser E, making Stage 3 easier.
inline std::vector<BitLoc> collect_stage2_bits(const CharState& cs) {
    std::vector<BitLoc> bits;
    // Collect A bits first
    for (int i = 0; i < (int)cs.A.size(); ++i) {
        int step = i - 4;
        uint32_t free = cs.A[i].free_mask();
        uint32_t xm = cs.A[i].x_mask();
        uint32_t mask = free | xm;
        while (mask) {
            int b = __builtin_ctz(mask);
            bits.push_back({WT_A, step, b});
            mask &= mask - 1;
        }
    }
    // Then E bits
    for (int i = 0; i < (int)cs.E.size(); ++i) {
        int step = i - 4;
        uint32_t free = cs.E[i].free_mask();
        uint32_t xm = cs.E[i].x_mask();
        uint32_t mask = free | xm;
        while (mask) {
            int b = __builtin_ctz(mask);
            bits.push_back({WT_E, step, b});
            mask &= mask - 1;
        }
    }
    return bits;
}

// Collect Phase 2 candidate bits: '-' bits with nonzero connectivity (§5.1).
// The paper (§5.1): "Let U' be the set of all '-' with many two-bit conditions."
// "Pick randomly a bit in U'." — uniform random from qualifying set, no sorting.
// G4: min_conn parameter controls "many" threshold (default 0 = any connectivity).
inline std::vector<BitLoc> collect_phase2_bits(const CharState& cs,
                                                const ConnectivityMap& conn,
                                                int min_conn = 0) {
    std::vector<BitLoc> bits;
    int threshold = std::max(1, min_conn);  // at least 1 (always require conn > 0)
    auto scan = [&](WordType wt, const std::vector<WordCond>& words,
                    int step_offset) {
        for (int i = 0; i < (int)words.size(); ++i) {
            int step = i + step_offset;
            uint32_t dm = words[i].dash_mask();
            while (dm) {
                int b = __builtin_ctz(dm);
                BitLoc loc{wt, step, b};
                if (conn.get(loc) >= threshold)
                    bits.push_back(loc);
                dm &= dm - 1;
            }
        }
    };
    scan(WT_A, cs.A, -4);
    scan(WT_E, cs.E, -4);
    // W[16+] are expansion-derived (§4.3): W_i = σ₁(W_{i-2}) + W_{i-7} +
    // σ₀(W_{i-15}) + W_{i-16}.  They are NOT free variables — deciding them
    // before their inputs are resolved causes spurious contradictions.
    // Only include W[0..15] (primary message words) in Phase 2.
    {
        int w_limit = std::min(16, (int)cs.W.size());
        for (int i = 0; i < w_limit; ++i) {
            uint32_t dm = cs.W[i].dash_mask();
            while (dm) {
                int b = __builtin_ctz(dm);
                BitLoc loc{WT_W, i, b};
                if (conn.get(loc) >= threshold)
                    bits.push_back(loc);
                dm &= dm - 1;
            }
        }
    }
    return bits;
}

// Collect remaining dash bits for Phase 3: bits not covered by Phase 2.
// Includes all A/E/W dash bits with zero connectivity (unconstrained).
inline std::vector<BitLoc> collect_free_dash_bits(const CharState& cs,
                                                   const ConnectivityMap& conn) {
    std::vector<BitLoc> bits;
    auto scan_zero_conn = [&](WordType wt, const std::vector<WordCond>& words,
                              int step_offset) {
        for (int i = 0; i < (int)words.size(); ++i) {
            int step = i + step_offset;
            uint32_t dm = words[i].dash_mask();
            while (dm) {
                int b = __builtin_ctz(dm);
                BitLoc loc{wt, step, b};
                if (conn.get(loc) == 0)
                    bits.push_back(loc);
                dm &= dm - 1;
            }
        }
    };
    scan_zero_conn(WT_A, cs.A, -4);
    scan_zero_conn(WT_E, cs.E, -4);
    scan_zero_conn(WT_W, cs.W, 0);
    return bits;
}

// ---- Unit propagation (§4.5 "Complete Condition Check") ----
// For each candidate bit, tentatively impose each possible value.
// If one leads to contradiction (via propagation), force the other.
// Handles ? → -/x, x → u/n, and - → 0/1 transitions.
//
// Parameters:
//   bits      — candidate bits to check
//   max_bits  — skip if more candidates than this (too expensive)
//   do_twobit — include two-bit consistency in contradiction test
//
// Returns: -1 on contradiction (both choices fail), 0 if no change,
//          >0 = number of forced moves.

inline int unit_propagate_bits(CharState& cs, const std::vector<BitLoc>& bits,
                               int max_bits = 64, bool do_twobit = false) {
    if (bits.empty() || (int)bits.size() > max_bits) return 0;

    int resolved = 0;
    for (size_t i = 0; i < bits.size(); ++i) {
        const auto& loc = bits[i];
        BitCond cur = get_cond(cs, loc);
        BitCond choice_a, choice_b;

        if (bc_is_free(cur)) {
            choice_a = BC_DASH; choice_b = BC_X;
        } else if (bc_is_x(cur)) {
            choice_a = BC_U; choice_b = BC_N;
        } else if (cur == BC_DASH) {
            choice_a = BC_0; choice_b = BC_1;
        } else {
            continue;  // already decided
        }

        CharState test_a = cs;
        get_word(test_a, loc).impose(loc.bit, choice_a);
        bool ok_a = propagate_full(test_a) && !test_a.has_contradiction()
                    && (!do_twobit || check_twobit(test_a));

        CharState test_b = cs;
        get_word(test_b, loc).impose(loc.bit, choice_b);
        bool ok_b = propagate_full(test_b) && !test_b.has_contradiction()
                    && (!do_twobit || check_twobit(test_b));

        if (!ok_a && !ok_b) return -1;  // contradiction: no valid choice
        if (!ok_a) {
            cs = test_b;
            ++resolved;
        } else if (!ok_b) {
            cs = test_a;
            ++resolved;
        }
    }
    return resolved;
}

// Legacy wrapper: unit propagation over Phase 1 bits only.
inline int unit_propagate(CharState& cs) {
    return unit_propagate_bits(cs, collect_phase1_bits(cs));
}

// ---- Look-ahead branching heuristic ----
// Algorithm 2 from Eichlseder-Mendel-Schläffer 2014 (FSE).
// Randomly sample s_max candidates, tentatively propagate each,
// select the one that determines the most variables (or first
// candidate that causes a contradiction).

struct LookAheadDecision {
    BitLoc loc;
    BitCond first_choice;
    BitCond alt_choice;
};

// Determine phase-appropriate choices for a bit's current condition.
// Phase 1: paper §5.1 step 2 mandates '-' first for '?', 'x' only on
// backtrack (step 6).  For 'x' bits, sign is random per paper.
inline std::pair<BitCond, BitCond> phase_choices(
    BitCond cur, int phase, std::mt19937& rng)
{
    // phase=0: auto-detect from condition
    if (phase == 0) {
        if (bc_is_free(cur) || cur == BC_X) phase = 1;
        else phase = 2;  // dash → 0/1
    }
    if (phase == 1) {
        if (bc_is_free(cur))
            return {BC_DASH, BC_X};  // paper: '-' first, 'x' on backtrack
        return (rng() & 1) ? std::make_pair(BC_U, BC_N)
                           : std::make_pair(BC_N, BC_U);
    }
    return (rng() & 1) ? std::make_pair(BC_0, BC_1)
                       : std::make_pair(BC_1, BC_0);
}

// Look-ahead branching: evaluate s_max random candidates, pick best.
inline LookAheadDecision look_ahead_pick(
    const CharState& cs,
    const std::vector<BitLoc>& candidates,
    int s_max,
    int phase,
    std::mt19937& rng)
{
    int n_cand = (int)candidates.size();
    if (n_cand == 1) {
        auto [f, a] = phase_choices(get_cond(cs, candidates[0]), phase, rng);
        return {candidates[0], f, a};
    }

    int n_eval = std::min(s_max, n_cand);

    // Shuffled indices for random sampling
    std::vector<int> order(n_cand);
    std::iota(order.begin(), order.end(), 0);
    std::shuffle(order.begin(), order.end(), rng);

    int decided_before = cs.count_decided();
    int best_order_pos = 0;
    int best_score = -1;
    BitCond best_first{}, best_alt{};

    // Track candidates implicitly scored (determined by a prior
    // candidate's propagation — Algorithm 2 step 5).
    std::vector<bool> skip(n_cand, false);

    int evaluated = 0;
    for (int pos = 0; pos < n_cand && evaluated < n_eval; ++pos) {
        int idx = order[pos];
        if (skip[idx]) continue;

        ++evaluated;

        const BitLoc& loc = candidates[idx];
        auto [test_val, alt_val] = phase_choices(
            get_cond(cs, loc), phase, rng);

        // Tentatively impose and propagate
        CharState test = cs;
        get_word(test, loc).impose(loc.bit, test_val);
        bool ok = propagate_full(test) && !test.has_contradiction();

        if (!ok) {
            // Contradiction: select immediately, non-contradicting
            // value becomes first_choice.
            return {loc, alt_val, test_val};
        }

        int score = test.count_decided() - decided_before;
        if (score > best_score) {
            best_score = score;
            best_order_pos = pos;
            best_first = test_val;
            best_alt = alt_val;
        }

        // Mark candidates determined by this propagation
        for (int other = pos + 1; other < n_cand; ++other) {
            int other_idx = order[other];
            if (skip[other_idx]) continue;
            if (get_cond(cs, candidates[other_idx]) !=
                get_cond(test, candidates[other_idx]))
                skip[other_idx] = true;
        }
    }

    int best_idx = order[best_order_pos];
    return {candidates[best_idx], best_first, best_alt};
}

// Unified decision selection: random or look-ahead.
inline LookAheadDecision select_decision(
    const CharState& cs,
    const std::vector<BitLoc>& candidates,
    int phase,
    bool use_look_ahead,
    int la_s_max,
    int window,
    std::mt19937& rng)
{
    if (candidates.empty()) return {{WT_A, 0, 0}, BC_DASH, BC_X};
    int limit = (window > 0) ? std::min((int)candidates.size(), window)
                             : (int)candidates.size();

    if (use_look_ahead && limit > 1) {
        std::vector<BitLoc> subset(candidates.begin(),
                                   candidates.begin() + limit);
        return look_ahead_pick(cs, subset, la_s_max, phase, rng);
    }

    int idx = std::uniform_int_distribution<int>(0, limit - 1)(rng);
    auto [first, alt] = phase_choices(
        get_cond(cs, candidates[idx]), phase, rng);
    return {candidates[idx], first, alt};
}

// ---- Search engine ----

// B2': include here (after BitLoc/CharState/WordType are declared) so that
// search() below can use EmbeddedBCP. embedded_bcp.hpp opens its own
// `namespace mendel2011 {`, so we must close ours, include, then reopen.
// embedded_bcp.hpp re-includes search.hpp which is a no-op (#pragma once),
// and the names defined above are still visible to it.
} // namespace mendel2011
#include "embedded_bcp.hpp"
namespace mendel2011 {

struct SearchConfig {
    int max_contradictions = 1000;  // restart threshold
    int max_decisions = 100000;     // max decisions before giving up on this run
    bool verbose = false;
    bool phase1_only = false;       // stop after resolving all '?' and 'x' bits
    int phase2_window = 16;         // top-k connectivity window for Phase 2 selection
    bool msg_first = false;         // prioritize W[0..15] dash bits (before Phase 1/2) [non-paper]
    bool use_look_ahead = false;    // enable look-ahead branching heuristic
    int la_s_max = 16;              // max look-ahead candidates per decision
    bool skip_unit_prop = false;    // skip unit_prop_fixpoint (for fast A/B testing)
    bool skip_twobit_prop = false; // skip propagate_twobit inside propagate_full
    bool skip_twobit_check = false; // skip check_twobit (for fast A/B testing)
    bool use_backjump = false;      // A11: non-chronological backjumping
    bool use_regression = false;    // G5: Phase 2→1 regression instead of restart (neutral, keep off)
    bool use_critical_memory = true; // G3: remember critical bits across restarts, bias value selection
    bool use_multibit_critical = true; // G6: mark all inputs of failing step as critical (not just decision bit)
    int phase2_min_conn = 0;        // G4: minimum connectivity for Phase 2 candidates (paper: "many")
    bool combined_phase1 = false;   // merge W[0..15] dash bits into Phase 1 candidate pool
    bool use_3stage = false;        // Mendel 2013 §5.2: 3-stage strategy (Stage 1=W-only, Stage 2=A/E, Stage 3=pair)
    bool msg_lsb_first = true;      // resolve W[0..15] dashes in LSB-first order (word 0 bit 0 first)
    // EmbeddedBCP integration. When true, each main-decision impose is
    // forwarded to a 2-watched-literal BCP engine over the Tseitin CNF of
    // the dual-copy SHA-256 step function. A BCP conflict triggers an
    // immediate hard restart.
    bool use_embedded_bcp = false;
};

struct SearchResult {
    bool found;
    int restarts;
    int total_decisions;
    int total_contradictions;
    int total_forced_moves;  // bits resolved by unit propagation (§4.5)
    int total_backjump_skips;  // decision levels skipped by backjumping
    int total_regressions;     // G5: Phase 2→1 regression count
    int total_critical_hits;   // G3: decisions where critical memory swapped value
    CharState solution;      // best state found (min Phase 1 bits; valid even if found==false)
    CharState final_state;   // state at budget exhaustion (for Phase 2 plateau analysis)
};

// §4.5 Complete Condition Check — fixpoint loop.
// Repeatedly runs unit_propagate on the most constrained undecided bits until
// no more forced moves are found.  This cascades: a forced move changes the
// state, which may create new forced moves in other bits.
// Returns total forced moves (≥0) or -1 on contradiction.
inline int unit_prop_fixpoint(CharState& cs, const ConnectivityMap& conn) {
    int total = 0;
    for (int iter = 0; iter < 20; ++iter) {
        auto p1 = collect_phase1_bits(cs);
        int up = 0;
        if (!p1.empty() && (int)p1.size() <= 256) {
            // G1 (§4.5): "applying it only to bits which are restricted by
            // two-bit conditions" — filter to conn > 0 before sorting.
            p1.erase(std::remove_if(p1.begin(), p1.end(),
                [&](const BitLoc& loc) { return conn.get(loc) == 0; }),
                p1.end());
            if (p1.empty()) { break; }
            std::sort(p1.begin(), p1.end(),
                      [&](const BitLoc& a, const BitLoc& b) {
                          return conn.get(a) > conn.get(b);
                      });
            int limit = ((int)p1.size() <= 64) ? (int)p1.size() : 16;
            if ((int)p1.size() > limit) p1.resize(limit);
            up = unit_propagate_bits(cs, p1, limit, true);
        } else if (p1.empty()) {
            auto p2 = collect_phase2_bits(cs, conn);
            if ((int)p2.size() > 16) p2.resize(16);
            up = unit_propagate_bits(cs, p2, 16, true);
        } else {
            break;  // too many Phase 1 bits
        }
        if (up < 0) return -1;
        if (up == 0) break;
        total += up;
    }
    return total;
}

inline SearchResult search(CharState initial, SearchConfig config, std::mt19937& rng) {
    SearchResult result{false, 0, 0, 0, 0, 0, 0, 0, initial, initial};
    int best_p1 = INT_MAX;

    // G3: Critical bit memory — persists across restarts.
    // Records flat bit indices where both choices failed. On re-encounter,
    // swap first/alt choices to try the other value first.
    // Paper §5.1: "remembering critical bits during the search"
    std::unordered_set<int> critical_bits;

    auto restart = [&]() {
        ++result.restarts;
    };

    // EmbeddedBCP engine (forward declaration; instantiation gated by
    // config.use_embedded_bcp). Defined in embedded_bcp.hpp; deliberately
    // included AFTER search.hpp's own type declarations to avoid a circular
    // dependency (embedded_bcp.hpp includes search.hpp for BitLoc/CharState).
    // The instance lives for the duration of this search() call.
    EmbeddedBCP bcp;
    const cnf::VarMap* bcp_vars = nullptr;
    if (config.use_embedded_bcp) {
        bool base_ok = build_bcp_engine(bcp, bcp_vars, initial, initial.n_steps);
        if (!base_ok) {
            // Initial starting point is BCP-UNSAT — no valid completion exists.
            // Mirror the early-return for !propagate_full(initial).
            result.final_state = initial;
            return result;
        }
    }

    while (result.total_decisions < config.max_decisions) {
        // B2': rewind BCP to the IV + starting-point fixpoint at every restart
        // epoch. On the first iteration this is a no-op (base just built).
        if (config.use_embedded_bcp) bcp.restore_base();

        // Start fresh from initial
        CharState cs = initial;
        std::vector<Decision> trail;
        int contradictions_this_run = 0;

        // A11: decision-level stamps for backjumping
        int n_total_bits = bit_level_total(initial.n_steps);
        std::vector<int> dl;
        if (config.use_backjump) dl.assign(n_total_bits, -1);

        // Initial propagation (bitwise + twobit fixpoint)
        if (!propagate_full(cs, nullptr, config.skip_twobit_prop)) {
            // Guard against infinite restart loop when initial state is
            // inconsistent (total_decisions never increments).
            restart();
            if (result.restarts > 100) { result.final_state = cs; return result; }
            continue;
        }

        // Precompute connectivity map for Phase 2 bit selection
        auto conn = compute_connectivity(cs);
        int conn_refresh_counter = 0;
        bool phase2_bootstrapped = false;  // M2: one-shot re-propagation

        while (result.total_decisions < config.max_decisions) {
            // Refresh connectivity periodically (every 200 decisions)
            if (++conn_refresh_counter >= 200) {
                conn = compute_connectivity(cs);
                conn_refresh_counter = 0;
            }

            // Check which phase we're in
            auto p1_bits = collect_phase1_bits(cs);

            // Combined Phase 1: merge W[0..15] dash bits into Phase 1 pool.
            // This implements the paper's "combined search" (§5.1):
            // "we already determine some message bits during the search for
            // a differential characteristic."
            bool using_combined = false;
            if (config.combined_phase1 && !p1_bits.empty()) {
                auto msg_dashes = collect_msg_dash_bits(cs);
                p1_bits.insert(p1_bits.end(), msg_dashes.begin(), msg_dashes.end());
                using_combined = !msg_dashes.empty();
            }

            BitLoc chosen;
            BitCond first_choice, alt_choice;

            // 3-stage strategy (Mendel 2013 §5.2):
            // Stage 1: W-only ?/x bits (later steps first)
            // Stage 2: A/E ?/x bits (A-biased)
            // Stage 3: confirming pair (same as existing Phase 2/3)
            auto s1_bits = config.use_3stage ? collect_stage1_bits(cs)
                                             : std::vector<BitLoc>{};
            auto s2_bits = config.use_3stage ? collect_stage2_bits(cs)
                                             : std::vector<BitLoc>{};

            if (config.use_3stage && !s1_bits.empty()) {
                // Stage 1: resolve W-only ?/x bits, later steps first
                auto sel = select_decision(cs, s1_bits, 1,
                    config.use_look_ahead, config.la_s_max, 0, rng);
                chosen = sel.loc;
                first_choice = sel.first_choice;
                alt_choice = sel.alt_choice;
            } else if (config.use_3stage && !s2_bits.empty()) {
                // Stage 2: resolve A/E ?/x bits, A-biased
                auto sel = select_decision(cs, s2_bits, 1,
                    config.use_look_ahead, config.la_s_max, 0, rng);
                chosen = sel.loc;
                first_choice = sel.first_choice;
                alt_choice = sel.alt_choice;
            } else {
                // Not in 3-stage mode (or Stages 1+2 exhausted) — fall through
                // to original Phase 1/msg-first/Phase 2 logic.

                // Message-first strategy: when enabled, fix W[0..15] dash bits
                // BEFORE Phase 1 to avoid incompatible diff patterns.
                auto msg_bits = config.msg_first ? collect_msg_dash_bits(cs)
                                                 : std::vector<BitLoc>{};

            if (config.msg_first && !msg_bits.empty()) {
                auto sel = select_decision(cs, msg_bits, 2,
                    config.use_look_ahead, config.la_s_max, 0, rng);
                chosen = sel.loc;
                first_choice = sel.first_choice;
                alt_choice = sel.alt_choice;
            } else if (!p1_bits.empty()) {
                // Phase 1 (possibly with merged W[0..15] dashes).
                // Use phase=0 (auto-detect) for combined mode so dash bits
                // get Phase 2 choices (0/1) while ?/x get Phase 1 choices.
                auto sel = select_decision(cs, p1_bits,
                    using_combined ? 0 : 1,
                    config.use_look_ahead, config.la_s_max, 0, rng);
                chosen = sel.loc;
                first_choice = sel.first_choice;
                alt_choice = sel.alt_choice;
            } else {
                // Phase 1 complete — all '?' and 'x' resolved
                if (config.phase1_only) {
                    result.found = true;
                    result.solution = cs;
                    result.final_state = cs;
                    return result;
                }

                // M2: Re-propagate at phase boundary.  Phase 1 resolved all
                // '?'/'x' bits, exposing new twobit deductions.
                // Also refresh connectivity since the condition landscape changed.
                if (!phase2_bootstrapped) {
                    phase2_bootstrapped = true;
                    if (!propagate_full(cs, nullptr, config.skip_twobit_prop) || cs.has_contradiction()) break;
                    conn = compute_connectivity(cs);
                    conn_refresh_counter = 0;
                }

                // Phase 2: message-first, then high-connectivity dashes.
                //
                // Key insight: assigning W[0..15] dash bits first lets the
                // step-function propagation cascade through E and A registers.
                // With known IV, each W[k] bit propagates through the step
                // function's carry chain, resolving ~2-3 dashes per bit
                // assigned (data-dependent on carry resolution).
                // Without this, the search scatters assignments across all
                // ~2500 dashes, getting zero propagation per bit because
                // addition carry chains need full word knowledge.
                auto msg_dashes = collect_msg_dash_bits(cs);
                if (!msg_dashes.empty()) {
                    // Assign W[0..15] dashes before state register dashes.
                    if (config.msg_lsb_first) {
                        // LSB-first: pick first candidate (word 0 bit 0, etc.)
                        // This maximizes carry-chain propagation and triggers
                        // word-completion once full words are resolved.
                        chosen = msg_dashes[0];
                        auto [fc, ac] = phase_choices(get_cond(cs, chosen), 2, rng);
                        first_choice = fc;
                        alt_choice = ac;
                    } else {
                        auto sel = select_decision(cs, msg_dashes, 2,
                            config.use_look_ahead, config.la_s_max, 0, rng);
                        chosen = sel.loc;
                        first_choice = sel.first_choice;
                        alt_choice = sel.alt_choice;
                    }
                } else {
                    // All W[0..15] resolved — pick from remaining dashes
                    auto p2_bits = collect_phase2_bits(cs, conn, config.phase2_min_conn);
                    if (p2_bits.empty()) {
                        // Phase 2 complete — all high-connectivity dashes resolved.
                        // Phase 3: assign remaining zero-connectivity dash bits.
                        auto free_bits = collect_free_dash_bits(cs, conn);
                        if (free_bits.empty()) {
                            // All bits determined — check full consistency
                            if (!cs.has_contradiction() && check_twobit(cs)) {
                                result.found = true;
                                result.solution = cs;
                                result.final_state = cs;
                                return result;
                            }
                            // Inconsistent — backtrack or restart
                            break;
                        }

                        auto sel = select_decision(cs, free_bits, 3,
                            config.use_look_ahead, config.la_s_max, 0, rng);
                        chosen = sel.loc;
                        first_choice = sel.first_choice;
                        alt_choice = sel.alt_choice;
                    } else {
                        auto sel = select_decision(cs, p2_bits, 2,
                            config.use_look_ahead, config.la_s_max,
                            config.phase2_window, rng);
                        chosen = sel.loc;
                        first_choice = sel.first_choice;
                        alt_choice = sel.alt_choice;
                    }
                }
            }
            } // end else: non-3-stage fallback

            // G3: If this bit was previously critical, swap first/alt to try
            // a different value first.  Paper: "prevent that critical bits
            // result in a contradiction again."
            if (config.use_critical_memory &&
                critical_bits.count(encode_bitloc(chosen, cs.n_steps))) {
                std::swap(first_choice, alt_choice);
                ++result.total_critical_hits;
            }

            // Save state and make decision
            Decision dec;
            dec.loc = chosen;
            dec.original = get_cond(cs, chosen);
            dec.first_choice = first_choice;
            dec.alt_choice = alt_choice;
            dec.tried_alt = false;
            dec.critical = false;
            dec.snapshot = cs;
            trail.push_back(dec);

            ++result.total_decisions;

            // Impose first choice
            get_word(cs, chosen).impose(chosen.bit, first_choice);

            // B2': BCP pre-filter on the main-decision first_choice. Per
            // path_b_design.md §C+§D-bis, this runs BEFORE the expensive
            // propagate_full so a BCP-detectable conflict (~100–1000 unit
            // propagations) short-circuits the heavier GnD propagation.
            // BCP is NOT called on the alt path or in backtrack pops; on
            // BCP-UNSAT we restart immediately (the stale-state cascade
            // makes within-basin recovery impossible — see Adversary W1).
            if (config.use_embedded_bcp) {
                if (!bcp.assert_decision(chosen, first_choice, *bcp_vars)) {
                    // Sound per §F: the BCP-asserted superset (current
                    // trail + stale popped assertions + new decision) is
                    // UNSAT. Restart-on-conflict cannot prune any valid
                    // solution. NOTE: just `break;` — the outer-loop tail
                    // already calls restart(). Calling restart() here too
                    // would double-count restarts (Adversary C1).
                    break;
                }
            }

            // Propagate and check.
            // Two-bit check (§4.5): the paper says "apply this check only rarely
            // and only to specific conditions during the search" because it's
            // expensive.  Apply it: always in Phase 2, and periodically in Phase 1
            // (every 50 decisions or when nearing completion).
            bool do_twobit = !config.skip_twobit_check &&
                            (p1_bits.empty() ||              // Phase 2: always
                             (int)p1_bits.size() <= 32 ||    // Ph1 endgame
                             result.total_decisions % 50 == 0); // periodic
            int fail_step_1 = -1;
            bool ok = propagate_full(cs, config.use_backjump ? &fail_step_1 : nullptr, config.skip_twobit_prop)
                       && !cs.has_contradiction()
                       && (!do_twobit || check_twobit(cs));

            // §4.5 Complete Condition Check — fixpoint loop over forced moves
            if (ok && !config.skip_unit_prop) {
                int up_total = unit_prop_fixpoint(cs, conn);
                if (up_total < 0) ok = false;
                else result.total_forced_moves += up_total;
            }

            if (ok) {
                // A11: stamp bits changed by this successful decision
                if (config.use_backjump)
                    stamp_changed_bits(cs, trail.back().snapshot, dl,
                                       (int)trail.size() - 1, cs.n_steps);
            } else {
                // Try alternative
                cs = trail.back().snapshot;
                trail.back().tried_alt = true;
                get_word(cs, chosen).impose(chosen.bit, alt_choice);

                int fail_step_2 = -1;
                ok = propagate_full(cs, config.use_backjump ? &fail_step_2 : nullptr, config.skip_twobit_prop)
                      && !cs.has_contradiction()
                      && (!do_twobit || check_twobit(cs));

                // §4.5 Complete Condition Check
                if (ok && !config.skip_unit_prop) {
                    int up_total = unit_prop_fixpoint(cs, conn);
                    if (up_total < 0) ok = false;
                    else result.total_forced_moves += up_total;
                }

                if (ok) {
                    // A11: stamp bits changed by alt choice
                    if (config.use_backjump)
                        stamp_changed_bits(cs, trail.back().snapshot, dl,
                                           (int)trail.size() - 1, cs.n_steps);
                } else {
                    // Both choices failed — mark as critical and backtrack
                    trail.back().critical = true;
                    // G3: record in cross-restart critical memory
                    if (config.use_critical_memory)
                        critical_bits.insert(encode_bitloc(trail.back().loc, cs.n_steps));
                    // G6: mark all inputs of failing step as critical
                    if (config.use_multibit_critical) {
                        int fs = (fail_step_2 >= 0) ? fail_step_2 : fail_step_1;
                        mark_step_inputs_critical(trail, fs, cs.n_steps,
                            config.use_critical_memory ? &critical_bits : nullptr);
                    }
                    ++contradictions_this_run;
                    ++result.total_contradictions;

                    // A11: use max of both conflict targets for backjumping
                    int last_fail = fail_step_2;
                    if (config.use_backjump && fail_step_1 >= 0) {
                        int t1 = compute_backjump_from_step(
                            dl, fail_step_1, (int)trail.size(), cs.n_steps);
                        int t2 = compute_backjump_from_step(
                            dl, fail_step_2, (int)trail.size(), cs.n_steps);
                        // Conservative: if either target is valid, use max
                        if (t1 >= 0 || t2 >= 0)
                            last_fail = (t1 >= t2) ? fail_step_1 : fail_step_2;
                    }

                    // Backtrack: pop this decision and find an earlier one we can retry
                    // G2 (§5.1 step 8): "jump back until the critical bit can be
                    // resolved" — prefer jumping to the nearest untried critical
                    // decision, which is more targeted than LIFO popping.
                    bool resolved = false;

                    // G2: find nearest untried critical decision (if any)
                    if (config.use_multibit_critical) {
                        for (int ti = (int)trail.size() - 1; ti >= 0; --ti) {
                            if (trail[ti].critical && !trail[ti].tried_alt) {
                                // Pop trail down to this critical decision
                                while ((int)trail.size() > ti + 1)
                                    trail.pop_back();
                                if (config.use_backjump) {
                                    for (int idx = 0; idx < n_total_bits; ++idx)
                                        if (dl[idx] >= (int)trail.size()) dl[idx] = -1;
                                }
                                break;
                            }
                        }
                    }

                    while (!trail.empty()) {
                        // A11: non-chronological backjump — skip irrelevant levels
                        if (config.use_backjump && last_fail >= 0) {
                            int target = compute_backjump_from_step(
                                dl, last_fail, (int)trail.size(), cs.n_steps);
                            if (target >= 0 && target < (int)trail.size() - 1) {
                                int n_skipped = (int)trail.size() - 1 - target;
                                result.total_backjump_skips += n_skipped;
                                while ((int)trail.size() > target + 1)
                                    trail.pop_back();
                                // Clear stale stamps from popped levels
                                for (int idx = 0; idx < n_total_bits; ++idx)
                                    if (dl[idx] >= (int)trail.size()) dl[idx] = -1;
                            }
                        }

                        Decision& top = trail.back();
                        if (!top.tried_alt) {
                            // Try alt choice on this earlier decision
                            cs = top.snapshot;
                            top.tried_alt = true;
                            get_word(cs, top.loc).impose(top.loc.bit, top.alt_choice);

                            last_fail = -1;
                            ok = propagate_full(cs, config.use_backjump ? &last_fail : nullptr, config.skip_twobit_prop)
                                  && !cs.has_contradiction()
                                  && (!do_twobit || check_twobit(cs));
                            if (ok) {
                                if (config.use_backjump) {
                                    // Clear stale stamps from original choice
                                    // before re-stamping with alt's changes
                                    int lv = (int)trail.size() - 1;
                                    for (int idx = 0; idx < n_total_bits; ++idx)
                                        if (dl[idx] == lv) dl[idx] = -1;
                                    stamp_changed_bits(cs, top.snapshot, dl,
                                                       lv, cs.n_steps);
                                }
                                resolved = true;
                                break;
                            }
                            // Also failed — mark critical and continue popping
                            top.critical = true;
                            // G3: record in cross-restart critical memory
                            if (config.use_critical_memory)
                                critical_bits.insert(encode_bitloc(top.loc, cs.n_steps));
                            // G6: mark all inputs of failing step as critical
                            if (config.use_multibit_critical && last_fail >= 0)
                                mark_step_inputs_critical(trail, last_fail, cs.n_steps,
                                    config.use_critical_memory ? &critical_bits : nullptr);
                            ++contradictions_this_run;
                            ++result.total_contradictions;
                        }
                        trail.pop_back();
                    }

                    if (!resolved) break;  // ran out of trail, restart

                    if (contradictions_this_run >= config.max_contradictions) {
                        if (!config.use_regression) break;  // old behavior: restart

                        // G5: Phase 2→1 regression (§5.1 step 9)
                        // "If necessary jump back to phase 1"
                        // Pop trail to an untried Phase 1 decision instead of restarting.
                        bool regressed = false;
                        while (!trail.empty()) {
                            Decision& top = trail.back();
                            bool is_phase1 = (top.original == BC_FREE
                                           || top.original == BC_X);
                            if (is_phase1 && !top.tried_alt) {
                                cs = top.snapshot;
                                top.tried_alt = true;
                                get_word(cs, top.loc).impose(top.loc.bit, top.alt_choice);
                                int fs = -1;
                                if (propagate_full(cs, config.use_backjump ? &fs : nullptr, config.skip_twobit_prop)
                                    && !cs.has_contradiction()) {
                                    // Clear stale stamps from popped levels + current
                                    if (config.use_backjump) {
                                        for (int idx = 0; idx < n_total_bits; ++idx)
                                            if (dl[idx] >= (int)trail.size() - 1)
                                                dl[idx] = -1;
                                        stamp_changed_bits(cs, top.snapshot, dl,
                                                           (int)trail.size() - 1, cs.n_steps);
                                    }
                                    contradictions_this_run = 0;
                                    phase2_bootstrapped = false;
                                    conn = compute_connectivity(cs);
                                    conn_refresh_counter = 0;
                                    ++result.total_regressions;
                                    regressed = true;
                                    break;
                                }
                                top.critical = true;
                                if (config.use_critical_memory)
                                    critical_bits.insert(encode_bitloc(top.loc, cs.n_steps));
                                // G6: mark all inputs of failing step as critical
                                if (config.use_multibit_critical && fs >= 0)
                                    mark_step_inputs_critical(trail, fs, cs.n_steps,
                                        config.use_critical_memory ? &critical_bits : nullptr);
                                ++result.total_contradictions;
                            }
                            trail.pop_back();
                        }
                        if (!regressed) break;  // trail exhausted, restart
                        continue;  // resume search in Phase 1
                    }
                }
            }

            if (config.verbose && result.total_decisions % 1000 == 0) {
                auto p1 = collect_phase1_bits(cs);
                auto p2 = collect_phase2_bits(cs, conn);
                auto p3 = collect_free_dash_bits(cs, conn);
                auto msg = collect_msg_dash_bits(cs);
                std::printf("[%d decisions, %d contradictions, %d forced, %d regressions] Msg=%zu Ph1=%zu Ph2=%zu Ph3=%zu\n",
                            result.total_decisions, result.total_contradictions,
                            result.total_forced_moves, result.total_regressions,
                            msg.size(), p1.size(), p2.size(), p3.size());
            }

            // Track best state (fewest Phase 1 bits), checked periodically
            if (result.total_decisions % 100 == 0) {
                int p1_count = (int)collect_phase1_bits(cs).size();
                if (p1_count < best_p1) {
                    best_p1 = p1_count;
                    result.solution = cs;
                }
            }
        }

        result.final_state = cs;
        restart();
    }

    return result;
}

} // namespace mendel2011
