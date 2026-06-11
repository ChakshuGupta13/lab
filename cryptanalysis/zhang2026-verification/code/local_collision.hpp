// Word-level local collision search for SHA-2 message expansion.
// Implements Zhang et al. 2026, Section 3 / Algorithm 1.
//
// The model operates on 1-bit variables ∆W_i (whether message word i has a
// nonzero difference) and flag_i (whether expansion step i requires a
// cancellation condition). The 27 valid transitions from Table 2 encode
// all feasible (∆W_{i-2}, ∆W_{i-7}, ∆W_{i-15}, ∆W_{i-16}) → (∆W_i, flag_i).
//
// Works for both SHA-256 and SHA-512: both use the same expansion recurrence
// W_i = σ1(W_{i-2}) + W_{i-7} + σ0(W_{i-15}) + W_{i-16} with identical index
// offsets. The σ constants differ but the word-level binary model is the same.

#pragma once

#include <cstdint>
#include <vector>

namespace local_collision {

// A valid transition for the message expansion at step i (i >= 16).
// Input: 4 bits packed as (∆W_{i-2}, ∆W_{i-7}, ∆W_{i-15}, ∆W_{i-16}).
// Output: (∆W_i, flag_i).
struct Transition {
    uint8_t input;   // 4-bit: bits [3..0] = (dW_{i-2}, dW_{i-7}, dW_{i-15}, dW_{i-16})
    uint8_t dW;      // 0 or 1: whether W_i has a difference
    uint8_t flag;    // 0 or 1: whether a cancellation condition is needed
};

// All 27 valid transitions from Table 2.
// Encoding: input bits [3]=∆W_{i-2}, [2]=∆W_{i-7}, [1]=∆W_{i-15}, [0]=∆W_{i-16}
inline constexpr Transition TABLE2[27] = {
    // 0 active inputs → no difference, no cancellation
    {0b0000, 0, 0},
    // 1 active input → difference propagates, no cancellation
    {0b0001, 1, 0}, {0b0010, 1, 0}, {0b0100, 1, 0}, {0b1000, 1, 0},
    // 2 active inputs → cancel (flag=1) or reinforce (dW=1)
    {0b0011, 0, 1}, {0b0011, 1, 0},
    {0b0101, 0, 1}, {0b0101, 1, 0},
    {0b1001, 0, 1}, {0b1001, 1, 0},
    {0b0110, 0, 1}, {0b0110, 1, 0},
    {0b1010, 0, 1}, {0b1010, 1, 0},
    {0b1100, 0, 1}, {0b1100, 1, 0},
    // 3 active inputs → cancel or reinforce
    {0b0111, 0, 1}, {0b0111, 1, 0},
    {0b1011, 0, 1}, {0b1011, 1, 0},
    {0b1101, 0, 1}, {0b1101, 1, 0},
    {0b1110, 0, 1}, {0b1110, 1, 0},
    // 4 active inputs → cancel or reinforce
    {0b1111, 0, 1}, {0b1111, 1, 0},
};

// For a given 4-bit input pattern, return the valid (dW, flag) pairs.
// Rule (from paper):
//   popcount == 0 → {(0,0)}
//   popcount == 1 → {(1,0)}
//   popcount >= 2 → {(0,1), (1,0)}
inline std::vector<std::pair<uint8_t, uint8_t>> valid_outputs(uint8_t input) {
    int pc = __builtin_popcount(input & 0xF);
    if (pc == 0) return {{0, 0}};
    if (pc == 1) return {{1, 0}};
    return {{0, 1}, {1, 0}};
}

// Pack 4 individual ∆W bits into the input format.
// Order: dW_{i-2} is bit 3, dW_{i-7} is bit 2, dW_{i-15} is bit 1, dW_{i-16} is bit 0.
inline constexpr uint8_t pack_input(uint8_t dw_i2, uint8_t dw_i7, uint8_t dw_i15, uint8_t dw_i16) {
    return ((dw_i2 & 1) << 3) | ((dw_i7 & 1) << 2) | ((dw_i15 & 1) << 1) | (dw_i16 & 1);
}

// --- Algorithm 1: DFS-based local collision search ---

// A found local collision: the set of active word indices and cancellation steps.
struct LocalCollision {
    int R, V, K;                     // attack steps, start step, span
    std::vector<int> active_words;   // indices where ∆W_i = 1
    std::vector<int> cancel_steps;   // expansion steps where flag_i = 1
    int obj;                         // |active_words| + |cancel_steps|
};

// Search for all local collisions with OBJ ≤ max_obj.
// R = total attack steps (0..R-1), V = start step (∆W_V = 1),
// K = span (∆W_{V+K-1} = 1), max_obj = upper bound on objective.
//
// DFS over free ∆W positions. For i >= 16, expansion constraints apply.
// For i < 16 within [V+1..V+K-2], choice is free (0 or 1).
// Outside [V..V+K-1], ∆W_i = 0.
inline std::vector<LocalCollision> search_local_collisions(int R, int V, int K, int max_obj) {
    if (V + K > R) return {};

    std::vector<LocalCollision> results;

    // State: dW[0..R-1] and flag[16..R-1]
    std::vector<uint8_t> dW(R, 0);
    std::vector<uint8_t> flag(R, 0);

    // Fixed constraints
    dW[V] = 1;
    if (V + K - 1 < R) dW[V + K - 1] = 1;

    // Free positions: indices in [V+1..V+K-2] that we branch on.
    // For i >= 16, the expansion constraint determines valid (dW[i], flag[i])
    // given dW[i-2], dW[i-7], dW[i-15], dW[i-16]. So for i >= 16, the "free"
    // choice is which valid output to pick (if popcount >= 2: cancel or reinforce).
    //
    // Strategy: process positions left-to-right. At each position:
    //   - If i < 16 and i is in the free interior: try dW=0 and dW=1
    //   - If i >= 16: compute input from already-assigned dW values,
    //     get valid outputs, try each that's consistent with constraints
    //   - If i is outside [V..V+K-1]: dW must be 0
    //   - If i == V or i == V+K-1: dW must be 1

    // We process expansion steps 16..R-1 in order. But we also need to
    // decide free message words (i < 16, V < i < V+K-1). We interleave:
    // process all positions from 0 to R-1 left-to-right.

    // Running OBJ for pruning
    int obj_so_far = 0;
    // Count initial fixed bits
    obj_so_far += 1; // dW[V] = 1
    if (V + K - 1 != V) obj_so_far += 1; // dW[V+K-1] = 1

    // DFS stack: position index + branch index
    // At each position, track which branch we're trying
    struct Frame {
        int pos;        // which ∆W index
        int branch;     // which option we're currently trying
        int n_options;  // total options at this position
        uint8_t saved_dW;
        uint8_t saved_flag;
        int obj_delta;  // how much this frame added to obj
    };

    // Determine positions we need to decide (skip fixed-0 outside span, skip V and V+K-1)
    std::vector<int> decision_positions;
    for (int i = 0; i < R; ++i) {
        if (i == V || i == V + K - 1) continue;        // fixed to 1
        if (i < V || i >= V + K) {
            // Outside span: must be 0. For i >= 16, need to check expansion is consistent.
            if (i >= 16) decision_positions.push_back(i); // must verify constraint
            // For i < 16 outside span: already 0, no constraint to check
            continue;
        }
        // Inside span interior: free or expansion-constrained
        decision_positions.push_back(i);
    }

    // Recursive DFS (depth <= ~60, safe for stack)
    int n_dp = (int)decision_positions.size();

    struct DFS {
        int R, V, K, max_obj;
        std::vector<uint8_t>& dW;
        std::vector<uint8_t>& flag;
        std::vector<int>& decision_positions;
        int n_dp;
        std::vector<LocalCollision>& results;

        void run(int dp_idx, int obj_so_far) {
            if (dp_idx == n_dp) {
                // Verify expansion constraint at forced endpoints (V and V+K-1)
                // if they are in the expanded region (>= 16).
                for (int ep : {V, V + K - 1}) {
                    if (ep < 16 || ep >= R) continue;
                    uint8_t inp = pack_input(
                        ep >= 2  ? dW[ep - 2]  : 0,
                        ep >= 7  ? dW[ep - 7]  : 0,
                        ep >= 15 ? dW[ep - 15] : 0,
                        ep >= 16 ? dW[ep - 16] : 0
                    );
                    auto opts = valid_outputs(inp);
                    bool endpoint_ok = false;
                    for (auto [dw_val, fl_val] : opts) {
                        if (dw_val == dW[ep]) { endpoint_ok = true; break; }
                    }
                    if (!endpoint_ok) return; // expansion forbids forced dW at endpoint
                }

                // All positions assigned and endpoints verified. Record solution.
                LocalCollision lc;
                lc.R = R; lc.V = V; lc.K = K; lc.obj = obj_so_far;
                for (int i = 0; i < R; ++i)
                    if (dW[i]) lc.active_words.push_back(i);
                for (int i = 16; i < R; ++i)
                    if (flag[i]) lc.cancel_steps.push_back(i);
                // Note: endpoint dW=1 is the "reinforce" option (flag=0),
                // so no flag cost is added for expansion-constrained endpoints.
                results.push_back(lc);
                return;
            }

            int i = decision_positions[dp_idx];
            bool in_span = (i >= V && i < V + K);
            bool is_expanded = (i >= 16);

            if (!is_expanded) {
                // i < 16, in span interior: free choice 0 or 1
                for (uint8_t val = 0; val <= 1; ++val) {
                    int delta = val;
                    if (obj_so_far + delta > max_obj) continue;
                    dW[i] = val;
                    run(dp_idx + 1, obj_so_far + delta);
                }
                dW[i] = 0; // restore
            } else {
                // i >= 16: expansion constraint
                uint8_t inp = pack_input(
                    i >= 2  ? dW[i - 2]  : 0,
                    i >= 7  ? dW[i - 7]  : 0,
                    i >= 15 ? dW[i - 15] : 0,
                    i >= 16 ? dW[i - 16] : 0
                );
                auto opts = valid_outputs(inp);

                for (auto [dw_val, fl_val] : opts) {
                    // Outside span: dW must be 0
                    if (!in_span && dw_val != 0) continue;
                    // Inside span: any valid output
                    // But also: outside span, flag must be 0 (no cancellation
                    // if we're not carrying differences — but if inp has active
                    // bits and we're outside span, we MUST cancel to get dW=0)
                    // Actually the constraint is just dW[i]=0 outside span.
                    // If dw_val=0 and fl_val=1, that means cancellation happened
                    // (which is fine, but we're outside the span so there shouldn't
                    // be active inputs... unless the span's last word feeds forward).
                    // Wait: outside span means ∆W_i = 0. If inp has active bits,
                    // then cancellation is required (dW=0, flag=1). But flag=1
                    // means an extra condition. This IS possible and adds to OBJ.

                    int delta = dw_val + fl_val;
                    if (obj_so_far + delta > max_obj) continue;

                    dW[i] = dw_val;
                    flag[i] = fl_val;
                    run(dp_idx + 1, obj_so_far + delta);
                }
                dW[i] = 0;
                flag[i] = 0;
            }
        }
    };

    DFS dfs{R, V, K, max_obj, dW, flag, decision_positions, n_dp, results};
    dfs.run(0, obj_so_far);
    return results;
}

// Convenience: find minimum OBJ for given (R, V, K), searching down from max_obj.
inline int find_min_obj(int R, int V, int K, int max_obj = 30) {
    for (int c = 1; c <= max_obj; ++c) {
        auto res = search_local_collisions(R, V, K, c);
        if (!res.empty()) return c;
    }
    return -1; // no solution found
}

} // namespace local_collision
