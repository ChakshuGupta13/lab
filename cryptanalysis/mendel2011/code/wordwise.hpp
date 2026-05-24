// Wordwise propagation for modular addition equations.
//
// Refines differential conditions on words involved in modular
// addition by exploiting modular integer difference constraints.
//
// Given an equation delta_A1 + delta_A2 + ... + delta_An = C (mod 2^32)
// where some differential conditions have unknown signs (x or ? bits),
// this module determines which signs are forced by the constraint.
//
// Algorithm from Alamgir, Nejati & Bright 2024, Section 5.2
// ("Wordwise Propagation"), building on Mendel et al. 2011/2013
// and Eichlseder 2013.
//
// Operates on WordCond from gencond.hpp. Does NOT depend on any
// SAT solver — usable from custom search (mendel2011/) or external
// solver callbacks alike.

#pragma once

#include "gencond.hpp"
#include <cstdint>
#include <algorithm>

namespace wordwise {

using gencond::BitCond;
using gencond::WordCond;
using gencond::BC_0;
using gencond::BC_1;
using gencond::BC_U;
using gencond::BC_N;
using gencond::BC_X;
using gencond::BC_DASH;
using gencond::BC_FREE;
using gencond::BC_HASH;

// Maximum number of unknown variables per subproblem before we skip it.
// Brute-force limit: 2^MAX_SUBPROBLEM_VARS iterations.
static constexpr int MAX_SUBPROBLEM_VARS = 10;

// Maximum number of input words in a single addition equation.
static constexpr int MAX_ADDENDS = 8;

// ============================================================
// Variable extraction from WordCond
// ============================================================

// A "wordwise variable" is a bit position in a specific addend word
// whose differential condition is x or ? (i.e., the modular difference
// contribution at that position is not fully determined).
//
// For condition 'x' at bit b: contribution is ±2^b (1 binary variable)
// For condition '?' at bit b: contribution is {-2^b, 0, +2^b} — but
//   per the paper's heuristic, '?' in auxiliary words is assumed to be '-'.
//   For primary words (A, E, W), '?' contributes 2 unknowns which is
//   expensive. In practice the caller decides whether to include '?' bits.

// Per-bit contribution info for one addend.
struct BitInfo {
    int32_t fixed_contrib;  // known contribution: +1 for u, -1 for n, 0 for 0/1/-
    bool is_variable;       // true if x (sign unknown)
    // If is_variable: the variable contributes either +2^b or -2^b.
    // We normalize: let v ∈ {0,1}, contribution = (2v - 1) · 2^b.
    // After normalization (adding 2^b offset), contribution = v · 2^{b+1}.
};

// Extract the modular difference information from a WordCond.
// Returns: fixed part of delta (sum of known contributions, mod 2^32),
//          and a bitmask of "variable" positions (x bits).
//
// For x bits: the true contribution is ±2^b. We normalize by adding 2^b
// to both sides so the variable becomes v · 2^{b+1} with v ∈ {0,1}.
// The offset (sum of 2^b for all x positions) is returned separately.
struct WordDiffInfo {
    uint32_t fixed_delta;   // sum of known contributions (mod 2^32)
    uint32_t x_mask;        // bitmask of positions with 'x' condition
    uint32_t offset;        // normalization offset = sum of 2^b for each x bit
};

inline WordDiffInfo extract_diff_info(const WordCond& wc) {
    WordDiffInfo info{0, 0, 0};
    for (int b = 0; b < 32; ++b) {
        BitCond c = wc.get(b);
        if (c == BC_U) {
            // u: (1,0), contribution = +2^b
            info.fixed_delta += (1u << b);
        } else if (c == BC_N) {
            // n: (0,1), contribution = -2^b
            info.fixed_delta -= (1u << b);
        } else if (c == BC_X) {
            // x: unknown sign, contribution = ±2^b
            // After normalization: contribution = v · 2^{b+1}, offset += 2^b
            info.x_mask |= (1u << b);
            info.offset += (1u << b);
        }
        // For -, 0, 1: contribution is 0 (no difference)
        // For ?: we treat as - (the heuristic from §5.2)
    }
    return info;
}

// ============================================================
// Subproblem decomposition via carry analysis
// ============================================================

// After extracting diff info from all addends in an equation
//   delta_A1 + delta_A2 + ... + delta_An = target  (mod 2^32)
//
// we have:
//   (sum of fixed_deltas + sum of offsets) + variable_part = target
//
// where variable_part is a sum of terms v_k · 2^{b_k+1} with v_k ∈ {0,1}.
//
// This reduces to a bitvector addition problem. We decompose it
// into independent subproblems by scanning LSB to MSB, grouping
// columns that are connected through carries.

// A subproblem: bit range [lo, hi) with a set of variables.
struct Subproblem {
    int lo;             // starting bit position (inclusive)
    int hi;             // ending bit position (exclusive)
    int nvars;          // number of binary variables in this range
    // Variable locations: for each variable, (addend_index, bit_position)
    // in the original equation. Stored in a flat array.
    struct VarLoc {
        uint8_t addend;
        uint8_t bit;
    };
    VarLoc vars[MAX_SUBPROBLEM_VARS + 1];  // +1 for overflow check
    uint32_t target_bits;  // target value in this bit range (shifted to start at bit 0)
};

// Result of wordwise propagation: which variables were determined.
struct PropResult {
    bool contradiction;     // true if the equation is unsatisfiable
    int deductions;         // number of variables determined
    // Deduced values stored per addend as a mask update:
    // For each addend, a (refine_mask, refine_sign) pair:
    //   refine_mask: positions where x was resolved to u or n
    //   refine_sign: subset of refine_mask where x → u (positive)
    //                complement within mask: x → n (negative)
    struct Refinement {
        uint32_t mask;   // positions resolved
        uint32_t sign;   // positive (u) positions within mask
    };
    Refinement refinements[MAX_ADDENDS];
    int n_addends;

    PropResult() : contradiction(false), deductions(0), n_addends(0) {
        for (auto& r : refinements) { r.mask = 0; r.sign = 0; }
    }
};

// ============================================================
// Core algorithm
// ============================================================

// Propagate constraints on a modular addition equation:
//   wc[0] + wc[1] + ... + wc[n-1] = sum_wc  (mod 2^32)
//
// where wc[i] are the differential conditions on the addends and
// sum_wc is the differential condition on the sum.
//
// The function determines which 'x' bits in the addends are forced
// to be 'u' or 'n' by the equation, and returns the refinements.
//
// If assume_free_is_dash is true (the heuristic from §5.2), any '?'
// condition in any word is treated as '-' for the purpose of computing
// the modular difference. This makes delta calculable more frequently.
//
// The sum_wc is always included in the analysis (its x bits may also
// be resolved). Internally the equation is rewritten as:
//   wc[0] + wc[1] + ... + wc[n-1] - sum_wc = 0
// which is equivalent to:
//   wc[0] + wc[1] + ... + wc[n-1] + (-sum_wc) = 0

inline PropResult wordwise_propagate(
    const WordCond* addends, int n_addends,
    const WordCond& sum_wc,
    bool assume_free_is_dash = true)
{
    PropResult result;
    result.n_addends = n_addends;

    // We treat the equation as: sum(addends) - sum_wc = 0.
    // Equivalently: sum(addends) + neg(sum_wc) = 0.
    //
    // Step 1: Extract diff info for each term.
    // We need n_addends + 1 terms (the +1 is for -sum_wc).

    if (n_addends + 1 > MAX_ADDENDS) return result;  // too many addends

    WordDiffInfo infos[MAX_ADDENDS];
    for (int i = 0; i < n_addends; ++i) {
        WordCond wc = addends[i];
        if (assume_free_is_dash) {
            // Replace ? with - for computing delta
            uint32_t free = wc.free_mask();
            if (free) {
                for (int b = 0; b < 32; ++b) {
                    if (free & (1u << b)) wc.set(b, BC_DASH);
                }
            }
        }
        infos[i] = extract_diff_info(wc);
    }

    // For the sum: we need -sum_wc's diff info.
    // If sum_wc has u at bit b → -sum_wc has n at bit b (contribution -(+2^b) = -2^b, but
    //   we're computing delta for -sum_wc, which is -(sum - sum*) = sum* - sum.
    //   If sum condition is u: (sum_b=1, sum*_b=0) → delta_sum = +2^b → -delta_sum = -2^b → n contribution.
    // Actually: the negated term's delta is -delta_sum = -(sum - sum*) = sum* - sum.
    // Simpler: just negate the fixed_delta, and for x bits the sign is also flipped.
    // But sign of x is unknown anyway (it's ±), so x stays x.
    {
        WordCond wc = sum_wc;
        if (assume_free_is_dash) {
            uint32_t free = wc.free_mask();
            if (free) {
                for (int b = 0; b < 32; ++b) {
                    if (free & (1u << b)) wc.set(b, BC_DASH);
                }
            }
        }
        WordDiffInfo si = extract_diff_info(wc);
        // Negate: -(fixed_delta), x_mask stays same, offset stays same
        // (because normalization of ±2^b is symmetric)
        infos[n_addends].fixed_delta = (uint32_t)(-(int32_t)si.fixed_delta);
        infos[n_addends].x_mask = si.x_mask;
        infos[n_addends].offset = si.offset;
    }

    int total_terms = n_addends + 1;

    // Step 2: Check if there are any variables to solve.
    uint32_t any_vars = 0;
    for (int i = 0; i < total_terms; ++i) any_vars |= infos[i].x_mask;
    if (any_vars == 0) {
        // No unknowns. Just check consistency.
        uint32_t total_fixed = 0;
        for (int i = 0; i < total_terms; ++i) total_fixed += infos[i].fixed_delta;
        if (total_fixed != 0) result.contradiction = true;
        return result;
    }

    // Step 3: Compute the RHS of the normalized equation.
    // After normalization: sum(fixed_deltas) + sum(v_k · 2^{b_k+1} - 2^{b_k}) = 0
    // → sum(v_k · 2^{b_k+1}) = sum(offsets) - sum(fixed_deltas)
    // → variable_part = target  (mod 2^32)
    uint32_t sum_fixed = 0, sum_offset = 0;
    for (int i = 0; i < total_terms; ++i) {
        sum_fixed += infos[i].fixed_delta;
        sum_offset += infos[i].offset;
    }
    uint32_t target = sum_offset - sum_fixed;

    // Step 4: Collect all variables and their bit positions.
    // Each variable v_k contributes 2^{b_k+1} to the variable_part
    // (because after normalization, ±2^b → v·2^{b+1} - 2^b, and we've
    //  already subtracted the 2^b offsets).
    //
    // Wait — let me re-derive the normalization carefully.
    // For an x bit at position b in addend i:
    //   True contribution: d_b · 2^b where d_b ∈ {+1, -1}
    //   Let v = (d_b + 1) / 2 ∈ {0, 1}  (v=1 means d=+1 i.e. u; v=0 means d=-1 i.e. n)
    //   Then d_b = 2v - 1, so contribution = (2v-1)·2^b = v·2^{b+1} - 2^b
    //   The -2^b part is the offset (already accounted for).
    //   The variable part is v · 2^{b+1}.
    //
    // So each variable v_k at bit position b_k contributes v_k · 2^{b_k + 1}
    // to the "variable_part" bitvector sum, which must equal target.

    // Build list of variables with their effective bit position (b+1 mod 32).
    struct VarDesc {
        uint8_t addend_idx;
        uint8_t orig_bit;     // position in the original word
        uint8_t eff_bit;      // effective bit = (orig_bit + 1) & 31
    };

    VarDesc all_vars[MAX_ADDENDS * 32];
    int nvar = 0;
    for (int i = 0; i < total_terms; ++i) {
        uint32_t xm = infos[i].x_mask;
        while (xm) {
            int b = __builtin_ctz(xm);
            xm &= xm - 1;
            // Skip bit 31: contribution is v · 2^{32} ≡ 0 (mod 2^32),
            // so the variable is unconstrained by the equation.
            // (The offset for bit 31 is still included in sum_offset.)
            if (b == 31) continue;
            if (nvar >= MAX_ADDENDS * 32) break;
            all_vars[nvar++] = {(uint8_t)i, (uint8_t)b, (uint8_t)(b + 1)};
        }
    }

    // Step 5: Group variables by effective bit position and decompose
    // into independent subproblems via carry analysis.
    //
    // Scan from bit 0 to bit 31. At each bit position, count how many
    // variables contribute (their eff_bit equals this position). These
    // variables plus the target bits form a column. If the column can
    // overflow (i.e., sum of variable values + existing carry can produce
    // a carry-out), the next column is dependent and must be included
    // in the same subproblem.

    // Count variables per effective bit position.
    int vars_at[32] = {};
    for (int k = 0; k < nvar; ++k)
        vars_at[all_vars[k].eff_bit]++;

    // Sort variables by effective bit for easy grouping.
    // Simple bucket sort since eff_bit ∈ [0,31].
    int bucket_start[32];
    bucket_start[0] = 0;
    for (int b = 1; b < 32; ++b)
        bucket_start[b] = bucket_start[b-1] + vars_at[b-1];
    VarDesc sorted[MAX_ADDENDS * 32];
    int bucket_pos[32];
    for (int b = 0; b < 32; ++b) bucket_pos[b] = bucket_start[b];
    for (int k = 0; k < nvar; ++k) {
        int eb = all_vars[k].eff_bit;
        sorted[bucket_pos[eb]++] = all_vars[k];
    }

    // Decompose into subproblems.
    // A subproblem spans a contiguous range of bit positions.
    // We track the maximum possible carry into the next column.

    Subproblem subs[32];
    int nsubs = 0;

    int bit = 0;
    while (bit < 32) {
        // Skip bits with no variables and no carry potential
        if (vars_at[bit] == 0) {
            // Check if target has bits set here — if not, skip.
            // Actually we need to track carry from previous subproblems,
            // but subproblems are independent, so we start fresh.
            ++bit;
            continue;
        }

        Subproblem& sp = subs[nsubs];
        sp.lo = bit;
        sp.nvars = 0;

        // max_carry tracks the maximum possible carry-out from this column.
        // At each column, the sum = (number of variable 1-bits) + (carry_in_max)
        //                          + (target bit contribution from fixed terms).
        // But we're doing it simpler: track max possible unsigned sum at each position.
        int max_carry = 0;

        while (bit < 32) {
            int nv = vars_at[bit];
            // Max sum at this bit position: nv (all vars = 1) + max_carry
            int max_sum = nv + max_carry;

            // Add variables at this bit position to the subproblem
            for (int j = bucket_start[bit]; j < bucket_start[bit] + nv; ++j) {
                if (sp.nvars < MAX_SUBPROBLEM_VARS + 1) {
                    sp.vars[sp.nvars] = {sorted[j].addend_idx, sorted[j].orig_bit};
                    sp.nvars++;
                }
            }

            // Can this column produce a carry?
            // carry_out_max = (max_sum) / 2 (integer division)
            max_carry = max_sum >> 1;

            ++bit;

            // If no carry possible and no variables at next bit, this subproblem is done
            if (max_carry == 0) break;
            // Also break if we've wrapped around or there are no more bits
            if (bit >= 32) break;
        }

        sp.hi = bit;

        // Extract the target bits for this subproblem range
        if (sp.hi <= 32 && sp.lo < sp.hi) {
            int width = sp.hi - sp.lo;
            if (width < 32)
                sp.target_bits = (target >> sp.lo) & ((1u << width) - 1);
            else
                sp.target_bits = target;  // full 32-bit
        }

        nsubs++;
    }

    // Step 5b: Verify that all target bits outside subproblem ranges are zero.
    // Bits not covered by any subproblem have no variables, so the variable
    // part contributes 0 there. If the target bit is nonzero, contradiction.
    {
        uint32_t covered = 0;
        for (int si = 0; si < nsubs; ++si) {
            for (int b = subs[si].lo; b < subs[si].hi && b < 32; ++b)
                covered |= (1u << b);
        }
        if (target & ~covered) {
            result.contradiction = true;
            return result;
        }
    }

    // Step 6: Solve each subproblem by brute force.
    for (int si = 0; si < nsubs; ++si) {
        const Subproblem& sp = subs[si];
        if (sp.nvars == 0) continue;
        if (sp.nvars > MAX_SUBPROBLEM_VARS) continue;  // too expensive, skip

        int width = sp.hi - sp.lo;
        uint32_t width_mask = (width < 32) ? ((1u << width) - 1) : 0xFFFFFFFF;

        // For each variable, precompute which bit it contributes to
        // within the subproblem (relative to sp.lo).
        uint8_t var_relbit[MAX_SUBPROBLEM_VARS];
        for (int v = 0; v < sp.nvars; ++v) {
            int eb = (sp.vars[v].bit + 1) & 31;
            var_relbit[v] = (uint8_t)(eb - sp.lo);
            // Note: eb >= sp.lo is guaranteed by construction
        }

        // Track which solutions each variable takes.
        // seen_0[v]: variable v was 0 in at least one solution
        // seen_1[v]: variable v was 1 in at least one solution
        uint32_t seen_0 = 0, seen_1 = 0;
        int n_solutions = 0;

        // Enumerate all 2^nvars assignments
        uint32_t limit = 1u << sp.nvars;
        for (uint32_t assign = 0; assign < limit; ++assign) {
            // Compute the sum of variable contributions within the subproblem window
            uint32_t sum = 0;
            for (int v = 0; v < sp.nvars; ++v) {
                if (assign & (1u << v))
                    sum += (1u << var_relbit[v]);
            }

            if ((sum & width_mask) == sp.target_bits) {
                n_solutions++;
                for (int v = 0; v < sp.nvars; ++v) {
                    if (assign & (1u << v))
                        seen_1 |= (1u << v);
                    else
                        seen_0 |= (1u << v);
                }
                // Early exit: if all variables have been seen as both 0 and 1,
                // no further deductions possible.
                if ((seen_0 & seen_1) == (limit - 1)) break;
            }
        }

        if (n_solutions == 0) {
            result.contradiction = true;
            return result;
        }

        // Step 7: Variables that appear with only one value → deduced.
        for (int v = 0; v < sp.nvars; ++v) {
            bool can_be_0 = (seen_0 >> v) & 1;
            bool can_be_1 = (seen_1 >> v) & 1;

            if (can_be_0 && can_be_1) continue;  // both possible
            if (!can_be_0 && !can_be_1) { result.contradiction = true; return result; }

            // v = 1 means d = +1 (u); v = 0 means d = -1 (n).
            bool is_positive = can_be_1;
            uint8_t addend = sp.vars[v].addend;
            uint8_t bit = sp.vars[v].bit;

            // Record refinement
            if (addend < (uint8_t)n_addends) {
                // It's one of the input addends
                result.refinements[addend].mask |= (1u << bit);
                if (is_positive)
                    result.refinements[addend].sign |= (1u << bit);
                result.deductions++;
            } else {
                // It's the negated sum word. The sign is flipped:
                // In the negated term, v=1 means original delta was negative → n.
                // v=0 means original delta was positive → u.
                // We don't store sum refinements in result.refinements; the caller
                // handles the sum word separately. We use a special index.
                // Actually, let's store it if there's room (n_addends < MAX_ADDENDS).
                // Convention: refinements[n_addends] is for the sum word, but with flipped sign.
                if (n_addends < MAX_ADDENDS) {
                    result.refinements[n_addends].mask |= (1u << bit);
                    if (!is_positive)  // flip: negated term v=0 → original is u
                        result.refinements[n_addends].sign |= (1u << bit);
                    result.deductions++;
                }
            }
        }
    }

    return result;
}

// ============================================================
// Convenience: apply PropResult refinements to WordCond arrays
// ============================================================

// Apply the deduced refinements to the addend WordConds and sum WordCond.
// Returns false if any refinement causes a contradiction (should not
// happen if wordwise_propagate returned contradiction=false).
inline bool apply_refinements(
    WordCond* addends, int n_addends,
    WordCond& sum_wc,
    const PropResult& res)
{
    for (int i = 0; i < n_addends; ++i) {
        uint32_t m = res.refinements[i].mask;
        uint32_t s = res.refinements[i].sign;
        while (m) {
            int b = __builtin_ctz(m);
            m &= m - 1;
            BitCond refined = (s & (1u << b)) ? BC_U : BC_N;
            if (!addends[i].impose(b, refined)) return false;
        }
    }
    // Apply sum refinements (stored at index n_addends)
    if (n_addends < MAX_ADDENDS) {
        uint32_t m = res.refinements[n_addends].mask;
        uint32_t s = res.refinements[n_addends].sign;
        while (m) {
            int b = __builtin_ctz(m);
            m &= m - 1;
            BitCond refined = (s & (1u << b)) ? BC_U : BC_N;
            if (!sum_wc.impose(b, refined)) return false;
        }
    }
    return true;
}

// ============================================================
// All-in-one: propagate and apply
// ============================================================

// Convenience wrapper: propagate wordwise constraints and immediately
// apply refinements. Returns false on contradiction, true otherwise.
// The number of deductions made is stored in *n_deductions if non-null.
inline bool wordwise_propagate_and_apply(
    WordCond* addends, int n_addends,
    WordCond& sum_wc,
    bool assume_free_is_dash = true,
    int* n_deductions = nullptr)
{
    PropResult res = wordwise_propagate(addends, n_addends, sum_wc,
                                         assume_free_is_dash);
    if (n_deductions) *n_deductions = res.deductions;
    if (res.contradiction) return false;
    if (res.deductions == 0) return true;
    return apply_refinements(addends, n_addends, sum_wc, res);
}

} // namespace wordwise
