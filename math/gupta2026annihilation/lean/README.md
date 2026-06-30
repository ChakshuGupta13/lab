# Lean formalization — Theorem 1 + Theorem 2 + Corollaries 1–3

The two files in this directory formalize the algebraic content of the
paper's Results section against Mathlib v4.30.0-rc2.

## `CaroWeiAnnihilation.lean` — Theorem 1

- `TxGraffitiC1.pointwise` — the pointwise inequality
  `1/(k+1) + k/(Δ(Δ+1)) ≥ 2/(Δ+1)` over ℚ, for naturals `k ≤ Δ` with `1 ≤ Δ`.
  Proved by clearing denominators and factoring as `(Δ-k)(Δ-k-1) ≥ 0`.

- `TxGraffitiC1.annihilation_caroWei` — the main inequality
  `|H| ≤ (Δ+1)/2 · Σ 1/(d_i+1)` over ℚ, given degrees `d : Fin n → ℕ` with
  `d i ≤ Δ` and the head degree-sum bound `Σ_{i ∈ H} d i ≤ Δ(n − |H|)`.

The hypothesis `Σ_{i ∈ H} d i ≤ Δ(n − |H|)` is exactly what the paper proves
from "Σ_H d ≤ m" (definition of the annihilation number) and the handshake
lemma "Σ d_i = 2m"; everything past that bound is the algebraic content the
Lean file checks.

## `MainAndCorollaries.lean` — Theorem 2 + Corollaries 1–3

The graph-theoretic content (the Caro-Wei bound `α ≥ W`, Favaron's
`α ≥ R`, Pepper's `α ≤ a`, and `a = α` for paths/cycles) is captured as
hypotheses; the algebraic chains that combine them are kernel-checked here.

- `TxGraffitiC1.vehicleToα_ge3` — From Theorem 1's vehicle `a ≤ (Δ+1)/2 · W`
  and Caro-Wei `W ≤ α`, deduce `a ≤ (Δ-1)·α` for `Δ ≥ 3`. The arithmetic
  step uses `(Δ+1)/2 ≤ Δ-1`.

- `TxGraffitiC1.main` — Theorem 2's algebraic core: given the head bound
  `(Δ-1)α ≥ a` and Favaron's `α ≥ R`, conclude `Δα ≥ a + R` via
  `Δα = (Δ-1)α + α`.

- `TxGraffitiC1.sharpness_iff` — Corollary 1: `Δα = a + R` if and only if
  both summand inequalities are equalities, that is, `(Δ-1)α = a` and `α = R`.

- `TxGraffitiC1.K4_attains_sharpness` — K_4 witness: with `Δ=3, α=1, a=2, R=1`,
  both equality conditions hold.

- `TxGraffitiC1.dominated_by_max` — Corollary 2: `(a + R)/Δ ≤ max(R, W)`
  for `Δ ≥ 3`. Uses `(Δ+3)/2 ≤ Δ` together with the vehicle.

- `TxGraffitiC1.bracketing` — Corollary 3: from the three classical bounds
  on `α` and the vehicle, the bracket
  `max(R, W) ≤ α ≤ a ≤ (Δ+1)/2 · max(R, W)` holds for every `Δ ≥ 1`.

- `TxGraffitiC1.K_DeltaPlus1_attains_bracket` — K_{Δ+1} (odd Δ) witness:
  with `α=1, a=(Δ+1)/2, R=W=1`, the upper bracket is tight.

## Axiom guarantee

All nine theorems depend only on Lean 4 / Mathlib defaults
`[propext, Classical.choice, Quot.sound]`. No `sorry`, no `admit`, no
`native_decide`.

## How to verify

Both files import Mathlib and are self-contained against any Lean 4 project
with Mathlib at version `v4.30.0-rc2` (or compatible).

A concrete recipe in this repository (using the Mathlib already pinned by
the lean-decoder-correctness project):

```
cd domains/quantum/src/lean-decoder-correctness/lean
lake env lean ../../../../math/src/txgraffiti-c1/lean/CaroWeiAnnihilation.lean
lake env lean ../../../../math/src/txgraffiti-c1/lean/MainAndCorollaries.lean
```

Each invocation should exit `0` with no diagnostics.

To check axiom dependencies, append `#print axioms` lines and re-run:

```
cat ../../../../math/src/txgraffiti-c1/lean/MainAndCorollaries.lean \
    <(echo; echo '#print axioms TxGraffitiC1.vehicleToα_ge3'; \
      echo '#print axioms TxGraffitiC1.main'; \
      echo '#print axioms TxGraffitiC1.sharpness_iff'; \
      echo '#print axioms TxGraffitiC1.K4_attains_sharpness'; \
      echo '#print axioms TxGraffitiC1.dominated_by_max'; \
      echo '#print axioms TxGraffitiC1.bracketing'; \
      echo '#print axioms TxGraffitiC1.K_DeltaPlus1_attains_bracket') \
  | lake env lean /dev/stdin
```

Each should print `[propext, Classical.choice, Quot.sound]`.
