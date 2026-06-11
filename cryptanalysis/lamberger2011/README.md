# Lamberger–Mendel 2011 — Message Modification Analysis

Structural analysis of the message-modification phase for:

> Lamberger, M. & Mendel, F. (2011).
> *Higher-Order Differential Attack on Reduced SHA-256.*
> LNCS vol. 6733, pp. 167–182.
> [ePrint 2011/037](https://eprint.iacr.org/2011/037)

## What this is

The paper's 46-step second-order differential on the SHA-256 compression
function claims 2⁴⁶ complexity, of which message modification (steps
21–29) accounts for the dominant savings factor of 2¹³². This directory
documents an independent analysis of *how* that savings is achieved,
decomposing the 66 Table 3 conditions into two structurally distinct
layers.

## Key findings

### 1. Two-layer constraint decomposition

The 66 forward conditions split into:
- **Consistency layer** (~12 bits): nonlinear, W-independent constraints
  on ΔA' − ΔE'. These cannot be satisfied by choosing W; they constrain
  the state difference independently of the message.
- **Bit-condition layer** (cheap, constructive): given a state satisfying
  the consistency layer, the remaining bit conditions are Wang-style
  (25–31 free A bits per step, deterministic correction).

This decomposition is not stated in the paper.

### 2. Per-step consistency costs

Measured exhaustively by scanning all 2³² values at each step:

| Step | Cost (bits) |
|------|-------------|
| 21   | 6.00        |
| 22   | 3.83        |
| 23   | 1.00        |
| 24   | 1.00        |
| 25–28| 0           |

Total consistency cost: ~12 bits. The paper gives only the aggregate
"message modification saves 2¹³²."

### 3. Revised complexity: 2⁴⁷·⁶ (vs claimed 2⁴⁶)

The expansion linearity analysis reduces to 3 critical inputs:
- **W[6]**: absorbed into the backward characteristic.
- **W[21]** (CTRL): absorbed into msg_modify at cost 2⁻²·¹⁹.
- **W[30]** (FIXED): uncontrollable, cost 2⁻¹¹·⁶². This is not
  accounted for in the paper's 2⁴⁶ estimate.

Net: ~2⁴⁷·⁶ after accounting for W[30], offset by the paper's 2¹⁰
additional characteristics.

### 4. Greedy selection failure

First-hit W[21] provably dead-ends: choosing the first W[21] satisfying
step 21's consistency condition gives 0/2³² success at step 22. Joint
optimization across steps 21–22 is required. The paper does not discuss
this failure mode.

### 5. Pitfall catalog

Six experimentally verified misconceptions about message modification in
second-order differentials are documented in
[MSG-MOD-ANALYSIS.md](MSG-MOD-ANALYSIS.md).

## Contents

| File | Description |
|------|-------------|
| `MSG-MOD-ANALYSIS.md` | Full 508-line analysis: algebraic structure, 12 experiments consolidated, per-step costs, expansion linearity, pitfalls |
| `msg_modify.cpp` | Wang-style forward message modification implementation (steps 21–29) |
| `expansion_invert.hpp` | Backward substitution: given W[21..29], solve M[0..13] from expansion recurrence |
| `differentials.hpp` | Table 1 differential characteristic (47 state diffs, 46 msg diffs) as constexpr data |
| `attack_helpers.hpp` | Shared primitives: alpha matching, consistency checking |
| `sha256.hpp` | Self-contained SHA-256 step function |

## Build

```bash
cd code
g++ -std=c++17 -O2 -Wall -Wextra -o msg_modify msg_modify.cpp
./msg_modify    # runs message modification + verification
```

## Status

The message-modification analysis is complete. The full 46-step attack
(which requires an automated characteristic search — the paper's
actual contribution) is not reproduced here; the search engine remains
an open problem. See the research repository for the full attack loop
infrastructure.
