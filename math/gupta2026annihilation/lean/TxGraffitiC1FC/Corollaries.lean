import Mathlib

/-!
# Corollaries 1–3 (algebraic layer)

The paper's three corollaries are algebraic consequences of the two summand
bounds (`a ≤ (Δ−1)·α` and `R ≤ α`) and the vehicle `a ≤ (Δ+1)/2·W`. They are
stated here over `ℚ`, parametric in the invariants, exactly as in the paper —
the graph-theoretic inputs they consume are the ones proved concretely over
`SimpleGraph` elsewhere in this project (`annih_le_predMaxDegree_indepNum`,
`caroWei_le_indepNum`, `residue_le_indepNum`, and Pepper's `α ≤ a`).

Kept as a self-contained algebraic module so the arithmetic is auditable in
isolation. Axioms: only `propext`, `Classical.choice`, `Quot.sound`.
-/

namespace TxGraffitiC1FC

/-- **Corollary 1 (Sharpness).** Equality `Δα = a + R` holds iff both summand
inequalities are equalities, `(Δ−1)α = a` and `α = R`. -/
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

/-- **K₄ attains the Sharpness equality.** With `Δ = 3, α = 1, a = 2, R = 1`,
both equality conditions hold, giving `Δα = a + R = 3`. -/
theorem K4_attains_sharpness :
    ((3 : ℚ) - 1) * 1 = 2 ∧ (1 : ℚ) = 1 ∧ (3 : ℚ) * 1 = 2 + 1 := by
  refine ⟨?_, ?_, ?_⟩ <;> norm_num

/-- **Corollary 2 (Domination by `max(R, W)`).** For `Δ ≥ 3`,
`(a + R)/Δ ≤ max(R, W)`, from the vehicle and `R ≤ max(R, W)`. -/
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

/-- **Corollary 3 (Bracketing).** Given the three classical bounds on `α`
(`R ≤ α`, `W ≤ α`, `α ≤ a`) and the vehicle, the bracket
`max(R, W) ≤ α ≤ a ≤ (Δ+1)/2·max(R, W)` holds for every `Δ ≥ 1`. -/
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

/-- **K_{Δ+1} (Δ odd) attains the Bracketing sharpness.** For odd `Δ = 2k+1`,
`K_{Δ+1}` has `α = 1, a = (Δ+1)/2, R = W = 1`, so `a = (Δ+1)/2·max(R, W)`. -/
theorem K_DeltaPlus1_attains_bracket (k : ℕ) :
    let Δ : ℚ := 2 * k + 1
    let a : ℚ := (Δ + 1) / 2
    let R : ℚ := 1
    let W : ℚ := 1
    a = (Δ + 1) / 2 * max R W := by
  simp [max_self]

end TxGraffitiC1FC
