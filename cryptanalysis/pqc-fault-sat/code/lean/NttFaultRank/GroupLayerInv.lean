/-
B6 — `groupLayer L z` is bijective when `z ≠ 0` and `(2 : K) ≠ 0`.

  The 2×2 butterfly `(a, b) ↦ (a + z·b, a − z·b)` has inverse
  `(x, y) ↦ ((x+y)/2, (x−y)/(2z))`. Lifted to `Fin 2 → Fin L → K`,
  this gives an explicit inverse `groupLayerInv L z` that we verify
  composes both ways to the identity.

  Used by Theorem 2's `e₀, e₁ ∈ ker` direction (post-fault layers
  must be bijective so the propagation is well-defined).
-/

import NttFaultRank.GroupLayer

namespace NttFaultRank

variable {K : Type*} [Field K]

section GroupLayerInv
variable (L : ℕ)

/-- Inverse of one Cooley-Tukey butterfly group, when `z ≠ 0` and `2 ≠ 0`.
    Sends `(x, y) ↦ ((x+y)/2, (x−y)/(2z))`. -/
def groupLayerInv (z : K) : (Fin 2 → Fin L → K) →ₗ[K] (Fin 2 → Fin L → K) where
  toFun v := ![fun j => (v 0 j + v 1 j) / 2, fun j => (v 0 j - v 1 j) / (2 * z)]
  map_add' x y := by
    ext b j
    rcases fin2_cases b with rfl | rfl
    · simp; ring
    · simp; ring
  map_smul' c v := by
    ext b j
    rcases fin2_cases b with rfl | rfl
    · simp; ring
    · simp; ring

@[simp] lemma groupLayerInv_apply_zero (z : K) (v : Fin 2 → Fin L → K) (j : Fin L) :
    groupLayerInv L z v 0 j = (v 0 j + v 1 j) / 2 := by simp [groupLayerInv]

@[simp] lemma groupLayerInv_apply_one (z : K) (v : Fin 2 → Fin L → K) (j : Fin L) :
    groupLayerInv L z v 1 j = (v 0 j - v 1 j) / (2 * z) := by simp [groupLayerInv]

/-- `groupLayerInv ∘ groupLayer = id`. Requires `z ≠ 0` and `(2 : K) ≠ 0`. -/
lemma groupLayerInv_comp_groupLayer {z : K} (hz : z ≠ 0) (h2 : (2 : K) ≠ 0) :
    (groupLayerInv L z).comp (groupLayer L z) = LinearMap.id := by
  apply LinearMap.ext
  intro v
  funext b j
  rcases fin2_cases b with rfl | rfl
  · simp
    field_simp
    ring
  · simp
    have h2z : (2 : K) * z ≠ 0 := mul_ne_zero h2 hz
    field_simp
    ring

/-- `groupLayer ∘ groupLayerInv = id`. Requires `z ≠ 0` and `(2 : K) ≠ 0`. -/
lemma groupLayer_comp_groupLayerInv {z : K} (hz : z ≠ 0) (h2 : (2 : K) ≠ 0) :
    (groupLayer L z).comp (groupLayerInv L z) = LinearMap.id := by
  apply LinearMap.ext
  intro v
  funext b j
  rcases fin2_cases b with rfl | rfl
  · simp
    field_simp
    ring
  · simp
    have h2z : (2 : K) * z ≠ 0 := mul_ne_zero h2 hz
    field_simp
    ring

/-- **B6 — `groupLayer L z` is bijective when `z ≠ 0` and `(2 : K) ≠ 0`.** -/
theorem groupLayer_bijective {z : K} (hz : z ≠ 0) (h2 : (2 : K) ≠ 0) :
    Function.Bijective (groupLayer L z) := by
  refine ⟨?_, ?_⟩
  · -- injectivity from left inverse
    intro x y hxy
    have := congrArg (groupLayerInv L z) hxy
    have hid := groupLayerInv_comp_groupLayer L hz h2
    have hx : (groupLayerInv L z).comp (groupLayer L z) x = x := by
      rw [hid]; rfl
    have hy : (groupLayerInv L z).comp (groupLayer L z) y = y := by
      rw [hid]; rfl
    simp only [LinearMap.comp_apply] at hx hy
    rw [← hx, ← hy, this]
  · -- surjectivity from right inverse
    intro w
    refine ⟨groupLayerInv L z w, ?_⟩
    have hid := groupLayer_comp_groupLayerInv L hz h2
    have : (groupLayer L z).comp (groupLayerInv L z) w = w := by
      rw [hid]; rfl
    simpa [LinearMap.comp_apply] using this

end GroupLayerInv

end NttFaultRank

