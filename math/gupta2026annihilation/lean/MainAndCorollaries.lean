/-
  Theorem 2 + Corollaries 1–3 of "An annihilation-number Caro-Wei bound"
  — Lean formalization.

  This file builds on `CaroWeiAnnihilation.lean` (which formalizes
  Theorem 1, the annihilation Caro-Wei vehicle `a ≤ (Δ+1)/2 · W`).
  Here we formalize the algebraic content of:

  * Theorem 2 (Resolution of Conjecture 1): `Δα ≥ a + R` for Δ ≥ 2.
  * Corollary 1 (Sharpness): equality iff `(Δ-1)α = a` and `α = R`,
    attained at K_4.
  * Corollary 2 (Domination): `(a + R)/Δ ≤ max(R, W)` for Δ ≥ 3.
  * Corollary 3 (Bracketing): `max(R, W) ≤ α ≤ a ≤ (Δ+1)/2 · max(R, W)`,
    attained at K_{Δ+1} for every odd Δ ≥ 1.

  The graph-theoretic content (e.g. "for paths and cycles, `a = α`"; the
  Caro-Wei bound `α ≥ W`; Favaron's bound `α ≥ R`; Pepper's bound
  `α ≤ a`) is captured as hypotheses; the algebraic chains that combine
  these into the theorems are kernel-checked here.

  Axiom guarantee: only `propext`, `Classical.choice`, `Quot.sound`.
  No `sorry`, `admit`, or `native_decide`.
-/
import Mathlib

namespace TxGraffitiC1

/--
**Vehicle-to-α reduction (Δ ≥ 3).** From Theorem 1's vehicle
`a ≤ (Δ+1)/2 · W` and the Caro-Wei bound `W ≤ α`, conclude
`a ≤ (Δ-1)·α`. The arithmetic step uses `(Δ+1)/2 ≤ Δ-1`, which holds
iff `Δ ≥ 3`. This is the Δ ≥ 3 half of the "head bound `(Δ-1)α ≥ a`"
that drives Theorem 2.
-/
theorem vehicleToα_ge3
    (Δ : ℕ) (a α W : ℚ) (hΔ : 3 ≤ Δ)
    (h_vehicle : a ≤ ((Δ : ℚ) + 1) / 2 * W)
    (h_W_nonneg : 0 ≤ W) (h_W_le_α : W ≤ α) :
    a ≤ ((Δ : ℚ) - 1) * α := by
  have hΔ_Q : (3 : ℚ) ≤ (Δ : ℚ) := by exact_mod_cast hΔ
  have h_ratio : ((Δ : ℚ) + 1) / 2 ≤ (Δ : ℚ) - 1 := by linarith
  have h_ratio_nn : (0 : ℚ) ≤ ((Δ : ℚ) + 1) / 2 := by linarith
  have h_α_nn : (0 : ℚ) ≤ α := le_trans h_W_nonneg h_W_le_α
  calc a ≤ ((Δ : ℚ) + 1) / 2 * W := h_vehicle
    _ ≤ ((Δ : ℚ) + 1) / 2 * α := mul_le_mul_of_nonneg_left h_W_le_α h_ratio_nn
    _ ≤ ((Δ : ℚ) - 1) * α := mul_le_mul_of_nonneg_right h_ratio h_α_nn

/--
**Theorem 2 (Resolution of Conjecture 1, algebraic core).** Given the
head bound `(Δ-1)α ≥ a` and Favaron's `α ≥ R`, the main inequality
`Δα ≥ a + R` follows from `Δα = (Δ-1)α + α`.

The hypothesis `(Δ-1)α ≥ a` is established in the paper proof by a
case split on Δ:
* Δ = 2: `a = α` (direct degree-sequence calculation on paths/cycles),
  hence `(Δ-1)α = α = a`.
* Δ ≥ 3: see `vehicleToα_ge3` above.
-/
theorem main
    (Δ : ℕ) (α a R : ℚ) (_hΔ : 2 ≤ Δ)
    (h_vehicleToα : a ≤ ((Δ : ℚ) - 1) * α)
    (h_favaron : R ≤ α) :
    a + R ≤ (Δ : ℚ) * α := by
  have h_sum : (Δ : ℚ) * α = ((Δ : ℚ) - 1) * α + α := by ring
  linarith

/--
**Corollary 1 (Sharpness).** Equality `Δα = a + R` holds if and only
if both summand inequalities are equalities, that is,
`(Δ-1)α = a` and `α = R`.
-/
theorem sharpness_iff
    (Δ α a R : ℚ)
    (h_vehicleToα : a ≤ (Δ - 1) * α) (h_favaron : R ≤ α) :
    Δ * α = a + R ↔ ((Δ - 1) * α = a ∧ α = R) := by
  have h_sum : Δ * α = (Δ - 1) * α + α := by ring
  refine ⟨?_, ?_⟩
  · intro h
    refine ⟨?_, ?_⟩ <;> linarith
  · rintro ⟨h1, h2⟩
    linarith

/--
**K_4 attains the Sharpness equality.** With Δ = 3, α = 1, a = 2, R = 1,
both equality conditions `(Δ-1)α = a` and `α = R` hold, giving
`Δα = a + R = 3`.
-/
theorem K4_attains_sharpness :
    ((3 : ℚ) - 1) * 1 = 2 ∧ (1 : ℚ) = 1 ∧ (3 : ℚ) * 1 = 2 + 1 := by
  refine ⟨?_, ?_, ?_⟩ <;> norm_num

/--
**Corollary 2 (Domination by `max(R, W)`).** For Δ ≥ 3,
`(a + R)/Δ ≤ max(R, W)`. The auxiliary fact `(Δ+3)/2 ≤ Δ` holds for
Δ ≥ 3, and combined with Theorem 1 and `R ≤ max(R, W)`, gives
`a + R ≤ Δ · max(R, W)`.
-/
theorem dominated_by_max
    (Δ : ℕ) (a R W : ℚ) (hΔ : 3 ≤ Δ)
    (h_vehicle : a ≤ ((Δ : ℚ) + 1) / 2 * W)
    (h_W_nonneg : 0 ≤ W) :
    (a + R) / (Δ : ℚ) ≤ max R W := by
  have hΔ_Q : (3 : ℚ) ≤ (Δ : ℚ) := by exact_mod_cast hΔ
  have hΔ_pos : (0 : ℚ) < (Δ : ℚ) := by linarith
  have h_aux : ((Δ : ℚ) + 3) / 2 ≤ (Δ : ℚ) := by linarith
  have h_ratio_nn : (0 : ℚ) ≤ ((Δ : ℚ) + 1) / 2 := by linarith
  have hWmax : W ≤ max R W := le_max_right _ _
  have hRmax : R ≤ max R W := le_max_left _ _
  have h_max_nn : (0 : ℚ) ≤ max R W := le_trans h_W_nonneg hWmax
  have step1 : a ≤ ((Δ : ℚ) + 1) / 2 * max R W := by
    calc a ≤ ((Δ : ℚ) + 1) / 2 * W := h_vehicle
      _ ≤ ((Δ : ℚ) + 1) / 2 * max R W :=
          mul_le_mul_of_nonneg_left hWmax h_ratio_nn
  have h_ring : ((Δ : ℚ) + 1) / 2 * max R W + max R W
              = ((Δ : ℚ) + 3) / 2 * max R W := by ring
  have step2 : a + R ≤ ((Δ : ℚ) + 3) / 2 * max R W := by linarith
  have step3 : a + R ≤ (Δ : ℚ) * max R W :=
    le_trans step2 (mul_le_mul_of_nonneg_right h_aux h_max_nn)
  rw [div_le_iff₀ hΔ_pos]
  linarith

/--
**Corollary 3 (Bracketing).** Given the three classical bounds on α
(`R ≤ α`, `W ≤ α`, `α ≤ a`) and Theorem 1's vehicle, the bracket
`max(R, W) ≤ α ≤ a ≤ (Δ+1)/2 · max(R, W)` holds for every Δ ≥ 1.
-/
theorem bracketing
    (Δ : ℕ) (α a R W : ℚ) (hΔ : 1 ≤ Δ)
    (h_R_le_α : R ≤ α) (h_W_le_α : W ≤ α) (h_α_le_a : α ≤ a)
    (h_vehicle : a ≤ ((Δ : ℚ) + 1) / 2 * W) :
    max R W ≤ α ∧ α ≤ a ∧ a ≤ ((Δ : ℚ) + 1) / 2 * max R W := by
  refine ⟨max_le h_R_le_α h_W_le_α, h_α_le_a, ?_⟩
  have hΔ_Q : (1 : ℚ) ≤ (Δ : ℚ) := by exact_mod_cast hΔ
  have h_ratio_nn : (0 : ℚ) ≤ ((Δ : ℚ) + 1) / 2 := by linarith
  calc a ≤ ((Δ : ℚ) + 1) / 2 * W := h_vehicle
    _ ≤ ((Δ : ℚ) + 1) / 2 * max R W :=
        mul_le_mul_of_nonneg_left (le_max_right _ _) h_ratio_nn

/--
**K_{Δ+1} (Δ odd) attains the Bracketing sharpness.** For odd Δ
written as `2k+1`, the complete graph K_{Δ+1} has α = 1,
a = (Δ+1)/2, R = W = 1, so `a = (Δ+1)/2 · max(R, W)` is exact.
-/
theorem K_DeltaPlus1_attains_bracket (k : ℕ) :
    let Δ : ℚ := 2 * k + 1
    let a : ℚ := (Δ + 1) / 2
    let R : ℚ := 1
    let W : ℚ := 1
    a = (Δ + 1) / 2 * max R W := by
  simp [max_self]

end TxGraffitiC1
