/-
Gentleman-Sande (GS) butterfly layer.

The GS butterfly maps `(a, b) ↦ (a + b, ζ(b − a))`. This file defines:
- `gsGroupLayer L z`: one GS butterfly group with twiddle `z`.
- `gsFullLayer G L zs`: G independent GS butterfly groups.
- `gsFullLayer' G L zs`: flat-form on `Fin (G*(2*L)) → K`.
- `gsLayerOn n G L h zs`: lifted to `Fin n → K`.

Properties: bijectivity when `z ≠ 0` and `2 ≠ 0`, per-group rank = L.
-/

import NttFaultRank.FlatLayer
import NttFaultRank.LayerOn
import Mathlib.Tactic.LinearCombination

namespace NttFaultRank

variable {K : Type*} [Field K]

section GsGroupLayer
variable (L : ℕ)

/-- One GS butterfly group: `(a, b) ↦ (a + b, ζ(b − a))` applied at each
    of `L` coordinate pairs. -/
def gsGroupLayer (z : K) : (Fin 2 → Fin L → K) →ₗ[K] (Fin 2 → Fin L → K) where
  toFun v := ![fun j => v 0 j + v 1 j, fun j => z * (v 1 j - v 0 j)]
  map_add' x y := by ext b j; fin_cases b <;> simp <;> ring
  map_smul' c v := by ext b j; fin_cases b <;> simp <;> ring

@[simp] lemma gsGroupLayer_apply_zero (z : K) (v : Fin 2 → Fin L → K) (j : Fin L) :
    gsGroupLayer L z v 0 j = v 0 j + v 1 j := by simp [gsGroupLayer]

@[simp] lemma gsGroupLayer_apply_one (z : K) (v : Fin 2 → Fin L → K) (j : Fin L) :
    gsGroupLayer L z v 1 j = z * (v 1 j - v 0 j) := by simp [gsGroupLayer]

/-- Difference of two GS group layers with different twiddles. -/
def gsGroupLayerDiff (z z' : K) :
    (Fin 2 → Fin L → K) →ₗ[K] (Fin 2 → Fin L → K) :=
  gsGroupLayer L z - gsGroupLayer L z'

@[simp] lemma gsGroupLayerDiff_apply_zero (z z' : K) (v : Fin 2 → Fin L → K) (j : Fin L) :
    gsGroupLayerDiff L z z' v 0 j = 0 := by
  simp [gsGroupLayerDiff, LinearMap.sub_apply]

@[simp] lemma gsGroupLayerDiff_apply_one (z z' : K) (v : Fin 2 → Fin L → K) (j : Fin L) :
    gsGroupLayerDiff L z z' v 1 j = (z - z') * (v 1 j - v 0 j) := by
  simp [gsGroupLayerDiff, LinearMap.sub_apply]; ring

/-- GS perturbation direction: `(0, w)` where `w j = v 1 j − v 0 j`. -/
def gsEmbedDiff (z : K) : (Fin L → K) →ₗ[K] (Fin 2 → Fin L → K) where
  toFun w := ![fun _ => (0 : K), fun j => z * w j]
  map_add' x y := by ext b j; fin_cases b <;> simp <;> ring
  map_smul' c w := by ext b j; fin_cases b <;> simp <;> ring

/-- The b-minus-a extraction: `v ↦ (j ↦ v 1 j − v 0 j)`. -/
def bMinusAProj : (Fin 2 → Fin L → K) →ₗ[K] (Fin L → K) where
  toFun v := fun j => v 1 j - v 0 j
  map_add' x y := by funext j; simp; ring
  map_smul' c v := by funext j; simp; ring

/-- The GS group-layer diff factors as `gsEmbedDiff ∘ bMinusAProj`. -/
lemma gsGroupLayerDiff_eq_comp (z z' : K) :
    gsGroupLayerDiff L z z' = (gsEmbedDiff L (z - z')).comp (bMinusAProj L) := by
  apply LinearMap.ext; intro v; funext b j
  rcases fin2_cases b with rfl | rfl
  · simp [gsEmbedDiff, bMinusAProj]
  · simp [gsEmbedDiff, bMinusAProj]

/-- `gsEmbedDiff L z` is injective when `z ≠ 0`. -/
lemma gsEmbedDiff_injective {z : K} (hz : z ≠ 0) :
    Function.Injective (gsEmbedDiff L z) := by
  intro w₁ w₂ h
  funext j
  have h1 : gsEmbedDiff L z w₁ 1 j = gsEmbedDiff L z w₂ 1 j := by rw [h]
  simp [gsEmbedDiff] at h1
  rcases h1 with h1 | h1
  · exact h1
  · exact absurd h1 hz

/-- `bMinusAProj L` is surjective: given `w`, take `v = ![0, w]`. -/
lemma bMinusAProj_surjective :
    Function.Surjective (bMinusAProj (K := K) L) := by
  intro w
  refine ⟨![0, w], ?_⟩
  funext j; simp [bMinusAProj]

/-- **Per-group GS rank = L.** When `z ≠ z'`, the GS group-layer difference
    has rank `L`. -/
theorem gsGroupLayerDiff_rank_eq {z z' : K} (hne : z ≠ z') :
    Module.finrank K (LinearMap.range (gsGroupLayerDiff L z z')) = L := by
  rw [gsGroupLayerDiff_eq_comp L z z']
  have hrange : LinearMap.range ((gsEmbedDiff L (z - z')).comp (bMinusAProj L))
              = LinearMap.range (gsEmbedDiff L (z - z')) := by
    rw [LinearMap.range_comp, LinearMap.range_eq_top.mpr (bMinusAProj_surjective L)]
    simp
  rw [hrange]
  have hzne : z - z' ≠ 0 := sub_ne_zero.mpr hne
  rw [LinearMap.finrank_range_of_inj (gsEmbedDiff_injective L hzne)]
  exact Module.finrank_fin_fun (R := K)

end GsGroupLayer

section GsFullLayer
variable (G L : ℕ)

/-- Full GS layer with G independent butterfly groups. -/
def gsFullLayer (zs : Fin G → K) :
    (Fin G → Fin 2 → Fin L → K) →ₗ[K] (Fin G → Fin 2 → Fin L → K) where
  toFun v := fun g => gsGroupLayer L (zs g) (v g)
  map_add' x y := by funext g; simp [map_add]
  map_smul' c v := by funext g; simp [map_smul]

/-- Flat-form GS layer on `Fin (G*(2*L)) → K`. -/
def gsFullLayer' (zs : Fin G → K) :
    (Fin (G * (2 * L)) → K) →ₗ[K] (Fin (G * (2 * L)) → K) :=
  (reshape K G L).symm.toLinearMap.comp <|
    (gsFullLayer G L zs).comp (reshape K G L).toLinearMap

/-- Flat-form GS layer difference. -/
def gsFullLayerDiff' (zs zs' : Fin G → K) :
    (Fin (G * (2 * L)) → K) →ₗ[K] (Fin (G * (2 * L)) → K) :=
  gsFullLayer' G L zs - gsFullLayer' G L zs'

end GsFullLayer

section GsLayerOn
variable (n G L : ℕ)

/-- GS layer lifted to `Fin n → K` via the proof `G * (2 * L) = n`.
    Mirrors `layerOn` from LayerOn.lean but with GS butterfly. -/
def gsLayerOn (h : G * (2 * L) = n) (zs : Fin G → K) :
    (Fin n → K) →ₗ[K] (Fin n → K) :=
  (funReindex (K := K) n G L h).symm.toLinearMap.comp <|
    (gsFullLayer' G L zs).comp (funReindex (K := K) n G L h).toLinearMap

/-- GS layer difference on `Fin n → K`. -/
def gsLayerOnDiff (h : G * (2 * L) = n) (zs zs' : Fin G → K) :
    (Fin n → K) →ₗ[K] (Fin n → K) :=
  gsLayerOn n G L h zs - gsLayerOn n G L h zs'

end GsLayerOn

/-! ### Bijectivity of GS layer -/

/-- GS butterfly on a pair is bijective when `z ≠ 0` and `2 ≠ 0`. -/
theorem gsGroupLayer_bijective (L : ℕ) {z : K} (hz : z ≠ 0) (h2 : (2 : K) ≠ 0) :
    Function.Bijective (gsGroupLayer L z) := by
  refine ⟨?_, ?_⟩
  · -- Injective: recover v from gsGroupLayer z v.
    -- a_out = v0 + v1, b_out = z*(v1 - v0).
    -- v0 = (a_out - b_out/z) / 2, v1 = (a_out + b_out/z) / 2.
    intro v₁ v₂ h
    funext s j
    have h0 : gsGroupLayer L z v₁ 0 j = gsGroupLayer L z v₂ 0 j := congrFun (congrFun h 0) j
    have h1 : gsGroupLayer L z v₁ 1 j = gsGroupLayer L z v₂ 1 j := congrFun (congrFun h 1) j
    simp only [gsGroupLayer_apply_zero] at h0
    simp only [gsGroupLayer_apply_one] at h1
    -- h0: v₁ 0 j + v₁ 1 j = v₂ 0 j + v₂ 1 j
    -- h1: z * (v₁ 1 j - v₁ 0 j) = z * (v₂ 1 j - v₂ 0 j)
    have h1' : v₁ 1 j - v₁ 0 j = v₂ 1 j - v₂ 0 j :=
      mul_left_cancel₀ hz h1
    rcases fin2_cases s with rfl | rfl
    · -- Goal: v₁ 0 j = v₂ 0 j. From h0 and h1': (a+b = a'+b') ∧ (b-a = b'-a') ⇒ 2a = 2a'.
      have h2a : (2 : K) * v₁ 0 j = (2 : K) * v₂ 0 j := by linear_combination h0 - h1'
      exact mul_left_cancel₀ h2 h2a
    · have h2b : (2 : K) * v₁ 1 j = (2 : K) * v₂ 1 j := by linear_combination h0 + h1'
      exact mul_left_cancel₀ h2 h2b
  · -- Surjective: given w, find v such that gsGroupLayer z v = w.
    intro w
    -- v 0 j = (w 0 j) / 2 - (w 1 j) / (2z)
    -- v 1 j = (w 0 j) / 2 + (w 1 j) / (2z)
    -- More precisely: a + b = w0, z(b-a) = w1 => b-a = w1/z => a = (w0 - w1/z)/2, b = (w0 + w1/z)/2
    refine ⟨![fun j => (w 0 j - z⁻¹ * w 1 j) / 2,
              fun j => (w 0 j + z⁻¹ * w 1 j) / 2], ?_⟩
    funext s j
    rcases fin2_cases s with rfl | rfl
    · simp; field_simp; ring
    · simp; field_simp; ring

/-- The GS full layer (indexed form) is bijective. -/
theorem gsFullLayer_bijective {G L : ℕ} {zs : Fin G → K}
    (hzs : ∀ g, zs g ≠ 0) (h2 : (2 : K) ≠ 0) :
    Function.Bijective (gsFullLayer G L zs) := by
  constructor
  · intro v₁ v₂ h
    funext g
    have hg : gsFullLayer G L zs v₁ g = gsFullLayer G L zs v₂ g := by rw [h]
    exact (gsGroupLayer_bijective L (hzs g) h2).1 hg
  · intro w
    choose v hv using fun g => (gsGroupLayer_bijective L (hzs g) h2).2 (w g)
    exact ⟨v, funext hv⟩

/-- The flat-form GS layer is bijective. -/
theorem gsFullLayer'_bijective {G L : ℕ} {zs : Fin G → K}
    (hzs : ∀ g, zs g ≠ 0) (h2 : (2 : K) ≠ 0) :
    Function.Bijective (gsFullLayer' G L zs) := by
  exact (reshape K G L).symm.bijective.comp
    ((gsFullLayer_bijective hzs h2).comp (reshape K G L).bijective)

/-- The lifted GS layer is bijective. -/
theorem gsLayerOn_bijective {n G L : ℕ} {h : G * (2 * L) = n} {zs : Fin G → K}
    (hzs : ∀ g, zs g ≠ 0) (h2 : (2 : K) ≠ 0) :
    Function.Bijective (gsLayerOn n G L h zs) := by
  show Function.Bijective
    ((funReindex (K := K) n G L h).symm.toLinearMap.comp
      ((gsFullLayer' G L zs).comp (funReindex (K := K) n G L h).toLinearMap))
  exact (funReindex (K := K) n G L h).symm.bijective.comp
    ((gsFullLayer'_bijective hzs h2).comp (funReindex (K := K) n G L h).bijective)

/-! ### GS layer agreement when b = a (kernel cascade) -/

/-- If the b-side equals the a-side at group `g₀` (i.e. `v g₀ 1 j = v g₀ 0 j`
    for all j), then two GS group layers with different twiddles agree on that
    input. This is because the perturbation is `δ(b-a) = 0`. -/
lemma gsGroupLayer_agree_of_b_eq_a (L : ℕ) {z z' : K}
    {v : Fin 2 → Fin L → K} (hba : ∀ j, v 1 j = v 0 j) :
    gsGroupLayer L z v = gsGroupLayer L z' v := by
  funext s j
  rcases fin2_cases s with rfl | rfl
  · simp
  · simp [hba j, sub_self]

/-- Full-layer agreement: if twiddles agree outside `g₀`, and `v g₀ 1 = v g₀ 0`,
    then the two full GS layers agree. -/
lemma gsFullLayer_agree_of_b_eq_a {G L : ℕ} {zs zs' : Fin G → K}
    {g₀ : Fin G} (hagree : ∀ g ≠ g₀, zs g = zs' g)
    {w : Fin G → Fin 2 → Fin L → K} (hba : ∀ j, w g₀ 1 j = w g₀ 0 j) :
    gsFullLayer G L zs w = gsFullLayer G L zs' w := by
  funext g
  show gsGroupLayer L (zs g) (w g) = gsGroupLayer L (zs' g) (w g)
  by_cases hg : g = g₀
  · subst hg; exact gsGroupLayer_agree_of_b_eq_a L hba
  · rw [hagree g hg]

end NttFaultRank
