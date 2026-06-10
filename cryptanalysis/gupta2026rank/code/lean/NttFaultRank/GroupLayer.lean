/-
B4/B5 — `groupLayer`: one Cooley-Tukey butterfly group (L disjoint
butterflies at coordinate pairs `(0, j)` ↔ `(1, j)` with shared twiddle).
The per-group difference `groupLayerDiff L z` has rank exactly `L`.

This is the per-layer content of paper-Theorem 1: a fault at layer ℓ
controls one twiddle that drives `L = n/2^(ℓ+1)` butterflies, giving
combined rank `L`. Once we package this with the disjoint-group structure
in the next file, full Theorem 1 (per-layer rank = L) follows immediately.

Proof strategy: factor `groupLayerDiff L z` as `embedDiff L z ∘ proj 1`:
- `proj 1` extracts the b-side of the vector — surjective onto `Fin L → K`.
- `embedDiff L z` sends `w` to `(z·w, −z·w)` — injective when `z ≠ 0`.
The range of the composition equals the range of `embedDiff L z` (because
`proj 1` is surjective), and injectivity gives `finrank range = L`.
-/

import NttFaultRank.PairButterfly

namespace NttFaultRank

open Module

variable {K : Type*} [Field K]

section GroupLayer
variable (L : ℕ)

/-- One Cooley-Tukey butterfly group with twiddle `z`: applies a butterfly
    `(a, b) ↦ (a + z·b, a − z·b)` at each of `L` coordinate pairs. -/
def groupLayer (z : K) : (Fin 2 → Fin L → K) →ₗ[K] (Fin 2 → Fin L → K) where
  toFun v := ![fun j => v 0 j + z * v 1 j, fun j => v 0 j - z * v 1 j]
  map_add' x y := by
    ext b j
    fin_cases b <;> simp <;> ring
  map_smul' c v := by
    ext b j
    fin_cases b <;> simp <;> ring

@[simp] lemma groupLayer_apply_zero (z : K) (v : Fin 2 → Fin L → K) (j : Fin L) :
    groupLayer L z v 0 j = v 0 j + z * v 1 j := by simp [groupLayer]

@[simp] lemma groupLayer_apply_one (z : K) (v : Fin 2 → Fin L → K) (j : Fin L) :
    groupLayer L z v 1 j = v 0 j - z * v 1 j := by simp [groupLayer]

/-- Difference between the unfaulted layer and the faulted (twiddle=0) layer. -/
def groupLayerDiff (z : K) : (Fin 2 → Fin L → K) →ₗ[K] (Fin 2 → Fin L → K) :=
  groupLayer L z - groupLayer L 0

@[simp] lemma groupLayerDiff_apply_zero (z : K) (v : Fin 2 → Fin L → K) (j : Fin L) :
    groupLayerDiff L z v 0 j = z * v 1 j := by
  simp [groupLayerDiff, LinearMap.sub_apply]

@[simp] lemma groupLayerDiff_apply_one (z : K) (v : Fin 2 → Fin L → K) (j : Fin L) :
    groupLayerDiff L z v 1 j = -(z * v 1 j) := by
  simp [groupLayerDiff, LinearMap.sub_apply]

/-- The "embed" map: `w ↦ (b, j) ↦ if b = 0 then z·w j else −(z·w j)`.
    Captures the action of `groupLayerDiff L z` modulo the projection
    that extracts the b-side input. -/
def embedDiff (z : K) : (Fin L → K) →ₗ[K] (Fin 2 → Fin L → K) where
  toFun w := ![fun j => z * w j, fun j => -(z * w j)]
  map_add' x y := by ext b j; fin_cases b <;> simp <;> ring
  map_smul' c w := by ext b j; fin_cases b <;> simp <;> ring

@[simp] lemma embedDiff_apply_zero (z : K) (w : Fin L → K) (j : Fin L) :
    embedDiff L z w 0 j = z * w j := by simp [embedDiff]

@[simp] lemma embedDiff_apply_one (z : K) (w : Fin L → K) (j : Fin L) :
    embedDiff L z w 1 j = -(z * w j) := by simp [embedDiff]

/-- The factorisation: `groupLayerDiff L z = embedDiff L z ∘ proj 1`. -/
lemma groupLayerDiff_eq_comp (z : K) :
    groupLayerDiff L z =
      (embedDiff L z).comp (LinearMap.proj (R := K) (φ := fun _ : Fin 2 => Fin L → K) 1) := by
  apply LinearMap.ext
  intro v
  funext b j
  simp only [LinearMap.comp_apply, LinearMap.proj_apply]
  rcases fin2_cases b with rfl | rfl
  · simp
  · simp

/-- `embedDiff L z` is injective when `z ≠ 0`. -/
lemma embedDiff_injective {z : K} (hz : z ≠ 0) :
    Function.Injective (embedDiff L z) := by
  intro w₁ w₂ h
  funext j
  have h0 : embedDiff L z w₁ 0 j = embedDiff L z w₂ 0 j := by rw [h]
  rw [embedDiff_apply_zero, embedDiff_apply_zero] at h0
  exact mul_left_cancel₀ hz h0

/-- The projection `proj 1 : (Fin 2 → Fin L → K) →ₗ[K] (Fin L → K)` is surjective. -/
lemma proj_one_surjective :
    Function.Surjective
      (LinearMap.proj (R := K) (φ := fun _ : Fin 2 => Fin L → K) 1) := by
  intro w
  refine ⟨![0, w], ?_⟩
  simp [LinearMap.proj_apply]

/-- **B5 — Per-group-layer rank = L.** When `z ≠ 0`, the difference between
    the unfaulted and faulted (twiddle-zeroed) group layer has rank `L`. -/
theorem groupLayerDiff_rank_eq {z : K} (hz : z ≠ 0) :
    finrank K (LinearMap.range (groupLayerDiff L z)) = L := by
  have hcomp := groupLayerDiff_eq_comp L z
  -- range (embed ∘ proj 1) = range embed (proj is surjective)
  have hrange : LinearMap.range (groupLayerDiff L z) = LinearMap.range (embedDiff L z) := by
    rw [hcomp, LinearMap.range_comp]
    rw [LinearMap.range_eq_top.mpr (proj_one_surjective L)]
    simp
  rw [hrange]
  -- injective ⇒ finrank range = finrank domain = L
  have hinj := embedDiff_injective L hz
  rw [LinearMap.finrank_range_of_inj hinj]
  exact Module.finrank_fin_fun (R := K)

/-- `groupLayer L z - groupLayer L z' = groupLayerDiff L (z - z')`:
    the difference of two group layers with different twiddles equals the
    zero-referenced diff applied to the twiddle difference. -/
lemma groupLayer_sub_eq_groupLayerDiff (z z' : K) :
    groupLayer L z - groupLayer L z' = groupLayerDiff L (z - z') := by
  apply LinearMap.ext; intro v; funext b j
  rcases fin2_cases b with rfl | rfl
  · simp [groupLayerDiff, LinearMap.sub_apply]; ring
  · simp [groupLayerDiff, LinearMap.sub_apply]; ring

end GroupLayer

end NttFaultRank

