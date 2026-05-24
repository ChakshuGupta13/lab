// Two-bit condition tracker for SHA-256 differential characteristic search.
// Based on Mendel-Nad-Schläffer 2011, §4.4–4.5.
//
// Two-bit conditions are linear constraints over GF(2) of the form:
//   A_{i,j} = A_{k,l}   (equality)
//   A_{i,j} ≠ A_{k,l}   (inequality)
//
// These arise from Boolean functions (Ch, Maj) and Sigma rotations:
// - Σ functions link different bit positions within the same word
// - f functions link the same bit position across different words
// - Combined: creates cycles that may contradict
//
// A single contradicting cycle makes the entire characteristic impossible.
//
// Implementation: Union-Find with parity (aka weighted quick-union).
// Each node is a (word_id, bit_pos) pair, flattened to a single integer.
// The parity tracks whether a node has the same or different value as
// its root — equivalent to GF(2) Gaussian elimination for 2-variable
// equations.
//
// Shared infrastructure: usable by any SHA-2 attack in the repo.

#pragma once

#include <cstdint>
#include <vector>
#include <utility>

namespace twobit {

// A bit identity: which word (register at some step) and which bit position.
struct BitId {
    int word;   // word identifier (caller-defined numbering)
    int bit;    // bit position 0..31

    bool operator==(const BitId& o) const { return word == o.word && bit == o.bit; }
    bool operator!=(const BitId& o) const { return !(*this == o); }
};

// Relation between two bits.
enum Relation : uint8_t {
    REL_EQUAL   = 0,  // bits must have same value
    REL_UNEQUAL = 1,  // bits must have different values
};

// A single two-bit condition.
struct TwoBitCond {
    BitId a, b;
    Relation rel;
};

// Union-Find with parity for detecting contradictions in two-bit conditions.
//
// Nodes are integers 0..n-1.  Each node has a parent and a parity bit
// (0 = same value as parent, 1 = different value from parent).
// The parity to root = XOR of parities along the path.
//
// Adding an equality constraint: merge(a, b, 0)
// Adding an inequality constraint: merge(a, b, 1)
// Contradiction: a and b are already in the same component but the
// required parity disagrees with the existing one.

class TwoBitTracker {
public:
    // Construct with a given number of bit nodes.
    // word_count: number of distinct words
    // bits_per_word: bit width (32 for SHA-256, 64 for SHA-512)
    // Total nodes = word_count * bits_per_word.
    explicit TwoBitTracker(int word_count, int bits_per_word = 32)
        : n_words_(word_count)
        , bits_per_word_(bits_per_word)
        , n_nodes_(word_count * bits_per_word)
        , parent_(n_nodes_)
        , rank_(n_nodes_, 0)
        , parity_(n_nodes_, 0)  // parity to parent: 0 = same
    {
        for (int i = 0; i < n_nodes_; ++i)
            parent_[i] = i;
    }

    // Flatten a BitId to a node index.
    int node(BitId id) const { return id.word * bits_per_word_ + id.bit; }
    int node(int word, int bit) const { return word * bits_per_word_ + bit; }

    // Find root of node x, returning (root, parity_to_root).
    // Path compression maintains correct parities.
    std::pair<int, uint8_t> find(int x) {
        if (parent_[x] == x) return {x, 0};
        auto [root, p] = find(parent_[x]);
        parity_[x] ^= p;  // accumulate parity through path
        parent_[x] = root; // path compression
        return {root, parity_[x]};
    }

    // Add a two-bit condition: bits a and b must have relation rel.
    // Returns false if this creates a contradiction.
    bool add(BitId a, BitId b, Relation rel) {
        return merge(node(a), node(b), static_cast<uint8_t>(rel));
    }

    // Add from a TwoBitCond struct.
    bool add(const TwoBitCond& c) {
        return add(c.a, c.b, c.rel);
    }

    // Check if two bits are in the same component and what their relation is.
    // Returns: {connected, relation} where relation is valid only if connected.
    std::pair<bool, Relation> query(BitId a, BitId b) {
        auto [ra, pa] = find(node(a));
        auto [rb, pb] = find(node(b));
        if (ra != rb) return {false, REL_EQUAL};
        return {true, Relation(pa ^ pb)};
    }

    // Number of connected components.
    int component_count() {
        int count = 0;
        for (int i = 0; i < n_nodes_; ++i) {
            auto [r, p] = find(i);
            if (r == i) ++count;
        }
        return count;
    }

    // Number of constraints added successfully (non-redundant).
    int constraint_count() const { return n_constraints_; }

    // Number of words and nodes.
    int word_count() const { return n_words_; }
    int bits_per_word() const { return bits_per_word_; }
    int node_count() const { return n_nodes_; }

    // Reset to initial state (all nodes independent).
    void reset() {
        for (int i = 0; i < n_nodes_; ++i) {
            parent_[i] = i;
            rank_[i] = 0;
            parity_[i] = 0;
        }
        n_constraints_ = 0;
    }

    // Number of bits in the same component as a given bit (component size).
    int component_size(BitId id) {
        auto [root, _p] = find(node(id));
        int count = 0;
        for (int i = 0; i < n_nodes_; ++i) {
            auto [r, p] = find(i);
            if (r == root) ++count;
        }
        return count;
    }

    // Count total two-bit conditions involving a given bit.
    // (This counts edges in the original constraint graph, not UF edges.)
    // For Phase 2 bit selection heuristic ("pick - with many two-bit conditions").
    // Requires the caller to maintain an adjacency count externally —
    // the UF structure doesn't directly track original edges.

private:
    int n_words_;
    int bits_per_word_;
    int n_nodes_;
    std::vector<int> parent_;
    std::vector<int> rank_;
    std::vector<uint8_t> parity_;  // parity to parent (0=same, 1=different)
    int n_constraints_ = 0;

    // Union by rank with parity.
    // rel = 0 means x and y should have same value.
    // rel = 1 means x and y should have different values.
    // Returns false on contradiction.
    bool merge(int x, int y, uint8_t rel) {
        auto [rx, px] = find(x);
        auto [ry, py] = find(y);

        if (rx == ry) {
            // Already connected — check consistency
            uint8_t actual_rel = px ^ py;
            return actual_rel == rel;
        }

        // Union by rank
        ++n_constraints_;
        // Required: parity from rx to ry should be px ^ rel ^ py
        uint8_t edge_parity = px ^ rel ^ py;

        if (rank_[rx] < rank_[ry]) {
            parent_[rx] = ry;
            parity_[rx] = edge_parity;
        } else if (rank_[rx] > rank_[ry]) {
            parent_[ry] = rx;
            parity_[ry] = edge_parity;
        } else {
            parent_[ry] = rx;
            parity_[ry] = edge_parity;
            rank_[rx]++;
        }
        return true;
    }
};

// ---- Two-bit condition extraction from Boolean functions ----
//
// Given conditions on inputs and output of a Boolean function at a bit
// position, extract implied two-bit conditions between inputs.
//
// For Maj(x,y,z): if output diff is 0 ('-') and input x has diff ('x','u','n'):
//   y and z must be equal (otherwise output diff would be nonzero).
// Symmetrically for y diff or z diff.
//
// For Ch(x,y,z): if output diff is 0 ('-') and x has diff:
//   y must equal z (the diff in x selects between y and z; for output=0
//   they must be the same).
// If x has no diff ('-') and x=0: output = z, so z determines output diff.
// If x has no diff ('-') and x=1: output = y, so y determines output diff.
// More complex conditions arise when x is '?' or multi-valued.

// Extract two-bit conditions from Maj(a,b,c) = output at bit position j.
// Returns conditions as a vector (may be empty if nothing can be deduced).
//
// Maj is non-linear: the diff-domain constraint depends on actual VALUES.
// Truth table for Δa=1, ΔMaj=0 (constraint on Δb, Δc):
//   all three same value (a=b=c): forced Δb=Δc=0 → REL_EQUAL
//   b_val != c_val:               forced Δb≠Δc → REL_UNEQUAL
//   b_val == c_val != a_val:      multiple valid options → no constraint
// Symmetric for b_diff and c_diff cases.
inline std::vector<TwoBitCond> extract_maj_twobit(
    int word_a, int word_b, int word_c, int bit,
    uint8_t cond_a, uint8_t cond_b, uint8_t cond_c, uint8_t cond_out)
{
    std::vector<TwoBitCond> result;

    // Output has no diff: '-' (0x9), '0' (0x1), '1' (0x8) — no diff pairs allowed
    bool out_no_diff = (cond_out & 0x6) == 0 && cond_out != 0;
    if (!out_no_diff) return result;

    // First-copy value: 0 if only pairs with x=0 (bits 0,1 of mask),
    // 1 if only pairs with x=1 (bits 2,3 of mask), -1 if unknown.
    // Encoding: bit0=(0,0), bit1=(0,1), bit2=(1,0), bit3=(1,1).
    auto get_val = [](uint8_t c) -> int {
        bool allows_0 = (c & 0x3) != 0;  // pairs with first-copy=0
        bool allows_1 = (c & 0xC) != 0;  // pairs with first-copy=1
        if (allows_0 && !allows_1) return 0;
        if (allows_1 && !allows_0) return 1;
        return -1;
    };

    BitId ba{word_a, bit}, bb{word_b, bit}, bc{word_c, bit};
    bool a_diff = (cond_a & 0x6) != 0 && (cond_a & 0x9) == 0;
    bool b_diff = (cond_b & 0x6) != 0 && (cond_b & 0x9) == 0;
    bool c_diff = (cond_c & 0x6) != 0 && (cond_c & 0x9) == 0;

    if (a_diff) {
        int vb = get_val(cond_b), vc = get_val(cond_c);
        if (vb >= 0 && vc >= 0) {
            if (vb != vc) {
                result.push_back({bb, bc, REL_UNEQUAL});
            } else {
                int va = get_val(cond_a);
                if (va >= 0 && va == vb)
                    result.push_back({bb, bc, REL_EQUAL});
            }
        }
    }
    if (b_diff) {
        int va = get_val(cond_a), vc = get_val(cond_c);
        if (va >= 0 && vc >= 0) {
            if (va != vc) {
                result.push_back({ba, bc, REL_UNEQUAL});
            } else {
                int vb = get_val(cond_b);
                if (vb >= 0 && vb == va)
                    result.push_back({ba, bc, REL_EQUAL});
            }
        }
    }
    if (c_diff) {
        int va = get_val(cond_a), vb = get_val(cond_b);
        if (va >= 0 && vb >= 0) {
            if (va != vb) {
                result.push_back({ba, bb, REL_UNEQUAL});
            } else {
                int vc = get_val(cond_c);
                if (vc >= 0 && vc == va)
                    result.push_back({ba, bb, REL_EQUAL});
            }
        }
    }

    return result;
}

// Extract two-bit conditions from Ch(x,y,z) = output at bit position j.
//
// Ch is non-linear: with Δx=1 and ΔCh=0, the constraint on (Δy, Δz) is
// UNDETERMINED for all value combinations (truth table shows every case
// admits both Δy=Δz and Δy≠Δz). No diff-domain two-bit constraint between
// y and z can be derived from Ch.
inline std::vector<TwoBitCond> extract_ch_twobit(
    [[maybe_unused]] int word_x, [[maybe_unused]] int word_y,
    [[maybe_unused]] int word_z, [[maybe_unused]] int bit,
    [[maybe_unused]] uint8_t cond_x, [[maybe_unused]] uint8_t cond_y,
    [[maybe_unused]] uint8_t cond_z, [[maybe_unused]] uint8_t cond_out)
{
    // No valid diff-domain two-bit conditions can be extracted from Ch.
    return {};
}

// Extract two-bit conditions from Σ functions (XOR of rotated copies).
// For Σ₀: bits j, (j+2)%32, (j+13)%32, (j+22)%32 of the same word are linked.
// If the word at the input has a difference at one of these positions,
// and the output bit's diff status is known, the other input bits must
// satisfy specific XOR relations.
//
// For a general 3-way XOR: out = a ^ b ^ c
//   diff(out) = diff(a) ⊕ diff(b) ⊕ diff(c)
//
// This is a linear relation in the diff domain, so:
//   out no-diff (even parity): diff(a) + diff(b) + diff(c) = 0 mod 2
//   out has-diff (odd parity):  diff(a) + diff(b) + diff(c) = 1 mod 2
//
// Even parity: known diff → others UNEQUAL, known no-diff → others EQUAL
// Odd parity:  known diff → others EQUAL, known no-diff → others UNEQUAL

// Given word_id and the three rotation amounts for a Sigma function,
// extract two-bit conditions at each output bit position.
// bits: word width (32 for SHA-256, 64 for SHA-512).
inline std::vector<TwoBitCond> extract_sigma_twobit(
    int word_id, int rot1, int rot2, int rot3, bool is_shift3,
    const uint8_t* conds,       // per-bit condition on the input word
    const uint8_t* out_conds,   // per-bit condition on the output word
    int bits = 32)
{
    std::vector<TwoBitCond> result;

    for (int j = 0; j < bits; ++j) {
        uint8_t co = out_conds[j];
        // Output has definite diff status?
        // No-diff: '-' (0x9), '0' (0x1), '1' (0x8) — (co & 0x6) == 0
        // Has-diff: 'x' (0x6), 'u' (0x2), 'n' (0x4) — (co & 0x9) == 0
        bool out_no_diff = (co & 0x6) == 0 && co != 0;
        bool out_has_diff = (co & 0x9) == 0 && co != 0;
        if (!out_no_diff && !out_has_diff) continue;

        // The three input bit positions that contribute to output bit j
        int p1 = (j + rot1) & (bits - 1);
        int p2 = (j + rot2) & (bits - 1);
        int p3 = is_shift3 ? (j + rot3) : ((j + rot3) & (bits - 1));

        // For shift: if p3 >= bits, that position is fixed to 0 (no diff possible)
        bool p3_valid = (p3 < bits);

        uint8_t c1 = conds[p1];
        uint8_t c2 = conds[p2];
        uint8_t c3 = p3_valid ? conds[p3] : 0x9;  // fixed 0 → no diff

        bool d1 = (c1 & 0x6) != 0 && (c1 & 0x9) == 0;  // definitely has diff
        bool d2 = (c2 & 0x6) != 0 && (c2 & 0x9) == 0;
        bool d3 = (c3 & 0x6) != 0 && (c3 & 0x9) == 0;
        bool n1 = (c1 & 0x6) == 0 && c1 != 0;  // definitely no diff
        bool n2 = (c2 & 0x6) == 0 && c2 != 0;
        bool n3 = (c3 & 0x6) == 0 && c3 != 0;

        // Even parity (out_no_diff): diff→UNEQUAL, no-diff→EQUAL
        // Odd parity (out_has_diff): diff→EQUAL, no-diff→UNEQUAL
        Relation rel_if_diff   = out_no_diff ? REL_UNEQUAL : REL_EQUAL;
        Relation rel_if_nodiff = out_no_diff ? REL_EQUAL   : REL_UNEQUAL;

        BitId b1{word_id, p1}, b2{word_id, p2};
        BitId b3_id{word_id, p3};

        if (d1 && p3_valid) {
            result.push_back({b2, b3_id, rel_if_diff});
        } else if (n1 && p3_valid) {
            result.push_back({b2, b3_id, rel_if_nodiff});
        }
        if (d2 && p3_valid) {
            result.push_back({b1, b3_id, rel_if_diff});
        } else if (n2 && p3_valid) {
            result.push_back({b1, b3_id, rel_if_nodiff});
        }
        if (d3) {
            result.push_back({b1, b2, rel_if_diff});
        } else if (n3) {
            result.push_back({b1, b2, rel_if_nodiff});
        }
    }

    return result;
}

// ---- Value-domain helpers ----

// Returns 0 if first-copy value is definitely 0, 1 if definitely 1, -1 if unknown.
// Pair encoding: bit 0→(0,0), bit 1→(1,0), bit 2→(0,1), bit 3→(1,1).
// First copy (x) is 0 when only pairs with x=0 are allowed (mask bits 0,2 = 0x5),
// and 1 when only pairs with x=1 are allowed (mask bits 1,3 = 0xA).
inline int bc_value_status(uint8_t c) {
    bool allows_x0 = (c & 0x5) != 0;
    bool allows_x1 = (c & 0xA) != 0;
    if (allows_x0 && !allows_x1) return 0;
    if (allows_x1 && !allows_x0) return 1;
    return -1;
}

// Value-domain sigma twobit extraction.
// For XOR equation: out_val[j] = in_val[p1] ⊕ in_val[p2] ⊕ in_val[p3]
// If out_val[j] is known AND exactly 2 of the 3 inputs are value-unknown,
// the two unknowns have a definite EQUAL/UNEQUAL value relation.
inline std::vector<TwoBitCond> extract_sigma_twobit_value(
    int word_id, int rot1, int rot2, int rot3, bool is_shift3,
    const uint8_t* conds,
    const uint8_t* out_conds,
    int bits = 32)
{
    std::vector<TwoBitCond> result;

    for (int j = 0; j < bits; ++j) {
        int out_val = bc_value_status(out_conds[j]);
        if (out_val < 0) continue;

        int p1 = (j + rot1) & (bits - 1);
        int p2 = (j + rot2) & (bits - 1);
        int p3 = is_shift3 ? (j + rot3) : ((j + rot3) & (bits - 1));
        bool p3_valid = (p3 < bits);

        int v1 = bc_value_status(conds[p1]);
        int v2 = bc_value_status(conds[p2]);
        int v3 = p3_valid ? bc_value_status(conds[p3]) : 0;  // shift: fixed 0

        bool u1 = (v1 < 0);
        bool u2 = (v2 < 0);
        bool u3 = p3_valid ? (v3 < 0) : false;

        int n_unk = u1 + u2 + u3;
        if (n_unk != 2) continue;

        int residual = out_val;
        if (!u1) residual ^= v1;
        if (!u2) residual ^= v2;
        if (!u3) residual ^= v3;

        Relation rel = (residual == 0) ? REL_EQUAL : REL_UNEQUAL;

        BitId unk_bits[2];
        int idx = 0;
        if (u1) unk_bits[idx++] = {word_id, p1};
        if (u2) unk_bits[idx++] = {word_id, p2};
        if (u3) unk_bits[idx++] = {word_id, p3};

        result.push_back({unk_bits[0], unk_bits[1], rel});
    }

    return result;
}

// ---- Two-bit conditions from GF(2) XOR equations ----
//
// A k-variable XOR equation:  x₁ ⊕ x₂ ⊕ ... ⊕ xₖ = target_parity
// where each xᵢ is the diff status (0=no diff, 1=has diff) of a bit.
//
// Each term is either:
//   known=true, val=0/1: contributes a constant to the parity
//   known=false: an unknown variable with a BitId for the Union-Find
//
// If exactly 2 unknowns remain after absorbing knowns:
//   x_a ⊕ x_b = residual_parity  →  twobit edge (EQUAL or UNEQUAL)
// If 1 unknown: unit deduction (not extracted here; bitwise prop handles it)
// If 0 unknown and residual ≠ 0: contradiction (set *contradiction = true)
// If 0 unknown and residual = 0: consistent (nothing to extract)
//
// Primary use: modular addition at bit j with known carry diff.
// The multi-input addition s = a₁ + a₂ + ... + aₖ satisfies:
//   Δs[j] = Δa₁[j] ⊕ ... ⊕ Δaₖ[j]  when Δcarry_in[j] = 0
// which holds at bit 0 always, and at bit j>0 when ALL addend diffs
// at bits 0..j-1 are 0 (making carry diffs 0 by induction).

struct XorTerm {
    BitId id;       // valid only when known=false
    bool known;     // true if diff status is determined
    int val;        // 0 or 1 if known
};

inline std::vector<TwoBitCond> extract_xor_twobit(
    const XorTerm* terms, int n_terms, int target_parity,
    bool* contradiction = nullptr)
{
    std::vector<TwoBitCond> result;
    if (contradiction) *contradiction = false;

    int n_unknown = 0;
    int residual = target_parity;
    BitId unk[2];

    for (int i = 0; i < n_terms; ++i) {
        if (terms[i].known) {
            residual ^= terms[i].val;
        } else {
            if (n_unknown < 2) unk[n_unknown] = terms[i].id;
            n_unknown++;
        }
    }

    if (n_unknown == 2) {
        // x_a ⊕ x_b = residual
        // residual=0 → EQUAL, residual=1 → UNEQUAL
        Relation rel = (residual == 0) ? REL_EQUAL : REL_UNEQUAL;
        result.push_back({unk[0], unk[1], rel});
    } else if (n_unknown == 0 && residual != 0) {
        // All terms known but XOR equation violated → contradiction
        if (contradiction) *contradiction = true;
    }
    // n_unknown 1: unit deduction (handled by bitwise propagation)
    // n_unknown >= 3: insufficient info for pairwise constraint

    return result;
}

} // namespace twobit
