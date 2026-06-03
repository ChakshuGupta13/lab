/-
B7c/d/e — `fullLayer`: a Cooley-Tukey layer with G independent groups,
each carrying its own twiddle. Acts on `Fin G → Fin 2 → Fin L → K`,
which is `Fin n → K` reshaped (n = G·2·L).

Each group is independent, so `fullLayer zs` is pointwise `groupLayer L (zs g)`.
Rank and bijectivity transport from the per-group results in `GroupLayer.lean`
and `GroupLayerInv.lean` via the universal property of pi types.

We work in the indexed form `Fin G → Fin 2 → Fin L → K` rather than reshape
to flat `Fin n → K`. The two are linearly isomorphic, and the indexed form
is far cleaner for the algebraic arguments. A flat-form view, if needed
for Theorem 2 statement, is one `LinearEquiv` away.
-/

import NttFaultRank.GroupLayer
import NttFaultRank.GroupLayerInv

namespace NttFaultRank

variable {K : Type*} [Field K]

section FullLayer
variable (G L : ℕ)

/-- One Cooley-Tukey layer with G independent butterfly groups, each driven
    by its own twiddle `zs g`. -/
def fullLayer (zs : Fin G → K) :
    (Fin G → Fin 2 → Fin L → K) →ₗ[K] (Fin G → Fin 2 → Fin L → K) where
  toFun v := fun g => groupLayer L (zs g) (v g)
  map_add' x y := by funext g; simp [map_add]
  map_smul' c v := by funext g; simp [map_smul]

@[simp] lemma fullLayer_apply (zs : Fin G → K) (v : Fin G → Fin 2 → Fin L → K) (g : Fin G) :
    fullLayer G L zs v g = groupLayer L (zs g) (v g) := rfl

/-- Difference of `fullLayer` evaluated at two twiddle vectors that may differ
    in arbitrary positions. -/
def fullLayerDiff (zs zs' : Fin G → K) :
    (Fin G → Fin 2 → Fin L → K) →ₗ[K] (Fin G → Fin 2 → Fin L → K) :=
  fullLayer G L zs - fullLayer G L zs'

@[simp] lemma fullLayerDiff_apply (zs zs' : Fin G → K)
    (v : Fin G → Fin 2 → Fin L → K) (g : Fin G) :
    fullLayerDiff G L zs zs' v g
      = groupLayer L (zs g) (v g) - groupLayer L (zs' g) (v g) := by
  simp [fullLayerDiff, LinearMap.sub_apply]

/-- If `zs` and `zs'` agree everywhere except at `g₀`, then `fullLayerDiff`
    is supported on coordinate `g₀`. -/
lemma fullLayerDiff_apply_eq_zero_of_ne {zs zs' : Fin G → K} {g₀ g : Fin G}
    (hagree : ∀ g ≠ g₀, zs g = zs' g) (hg : g ≠ g₀)
    (v : Fin G → Fin 2 → Fin L → K) :
    fullLayerDiff G L zs zs' v g = 0 := by
  simp [hagree g hg]

/-! ### B7d — rank of a single-group-fault layer diff -/

/-- The map "extract group g" `(Fin G → Fin 2 → Fin L → K) →ₗ[K]
    (Fin 2 → Fin L → K)`. Surjective. -/
abbrev projGroup (g : Fin G) :
    (Fin G → Fin 2 → Fin L → K) →ₗ[K] (Fin 2 → Fin L → K) :=
  LinearMap.proj g

/-- The map "embed at group g, zero elsewhere"
    `(Fin 2 → Fin L → K) →ₗ[K] (Fin G → Fin 2 → Fin L → K)`. -/
def embedGroup (g₀ : Fin G) :
    (Fin 2 → Fin L → K) →ₗ[K] (Fin G → Fin 2 → Fin L → K) where
  toFun w := fun g => if g = g₀ then w else 0
  map_add' x y := by funext g; by_cases h : g = g₀ <;> simp [h]
  map_smul' c x := by funext g; by_cases h : g = g₀ <;> simp [h]

@[simp] lemma embedGroup_apply_self (g₀ : Fin G) (w : Fin 2 → Fin L → K) :
    embedGroup G L g₀ w g₀ = w := by simp [embedGroup]

lemma embedGroup_apply_other {g₀ g : Fin G} (h : g ≠ g₀) (w : Fin 2 → Fin L → K) :
    embedGroup G L g₀ w g = 0 := by simp [embedGroup, h]

/-- When `zs` and `zs'` agree everywhere except at `g₀`, the difference
    `fullLayer zs - fullLayer zs'` factors as `embedGroup ∘ groupDiff ∘ projGroup`. -/
lemma fullLayerDiff_eq_comp {zs zs' : Fin G → K} (g₀ : Fin G)
    (hagree : ∀ g ≠ g₀, zs g = zs' g) :
    fullLayerDiff G L zs zs' =
      (embedGroup G L g₀).comp
        ((groupLayer L (zs g₀) - groupLayer L (zs' g₀)).comp (projGroup G L g₀)) := by
  apply LinearMap.ext
  intro v
  funext g
  simp only [LinearMap.comp_apply, fullLayerDiff_apply, projGroup, LinearMap.proj_apply]
  by_cases h : g = g₀
  · subst h
    simp
  · rw [embedGroup_apply_other G L h, hagree g h]
    simp [LinearMap.sub_apply]

/-- The range of `fullLayer zs - fullLayer zs'` (zs differs from zs' only at g₀)
    equals the embed-image of the per-group difference range. -/
lemma fullLayerDiff_range {zs zs' : Fin G → K} (g₀ : Fin G)
    (hagree : ∀ g ≠ g₀, zs g = zs' g) :
    LinearMap.range (fullLayerDiff G L zs zs') =
      Submodule.map (embedGroup G L g₀)
        (LinearMap.range (groupLayer L (zs g₀) - groupLayer L (zs' g₀))) := by
  rw [fullLayerDiff_eq_comp G L g₀ hagree]
  rw [LinearMap.range_comp, LinearMap.range_comp]
  congr 1
  rw [LinearMap.range_eq_top.mpr]
  · simp
  · -- projGroup g₀ is surjective: given w, pick (fun g => if g = g₀ then w else 0)
    intro w
    refine ⟨embedGroup G L g₀ w, ?_⟩
    simp [projGroup]

/-- `embedGroup g₀` is injective. -/
lemma embedGroup_injective (g₀ : Fin G) :
    Function.Injective (embedGroup G L g₀ : _ →ₗ[K] _) := by
  intro x y h
  have := congrArg (fun w => w g₀) h
  simpa using this

/-- **B7d — Single-group-fault full-layer rank = L.** When `zs` and `zs'`
    agree everywhere except at `g₀`, `zs g₀ ≠ 0`, and `zs' g₀ = 0`, the
    full-layer difference has rank `L`. -/
theorem fullLayerDiff_rank_eq {zs zs' : Fin G → K} (g₀ : Fin G)
    (hagree : ∀ g ≠ g₀, zs g = zs' g)
    (hz : zs g₀ ≠ 0) (hz' : zs' g₀ = 0) :
    Module.finrank K (LinearMap.range (fullLayerDiff G L zs zs')) = L := by
  rw [fullLayerDiff_range G L g₀ hagree]
  have hgrp : groupLayer L (zs g₀) - groupLayer L (zs' g₀) = groupLayerDiff L (zs g₀) := by
    rw [hz', groupLayerDiff]
  rw [hgrp]
  -- embedGroup is injective ⇒ finrank of image of any subspace = finrank of source.
  have := (Submodule.equivMapOfInjective (embedGroup G L g₀)
              (embedGroup_injective G L g₀)
              (LinearMap.range (groupLayerDiff L (zs g₀)))).finrank_eq
  rw [← this]
  exact groupLayerDiff_rank_eq L hz

/-- Inverse of `fullLayer`: componentwise `groupLayerInv`. -/
def fullLayerInv (zs : Fin G → K) :
    (Fin G → Fin 2 → Fin L → K) →ₗ[K] (Fin G → Fin 2 → Fin L → K) where
  toFun v := fun g => groupLayerInv L (zs g) (v g)
  map_add' x y := by funext g; simp [map_add]
  map_smul' c v := by funext g; simp [map_smul]

@[simp] lemma fullLayerInv_apply (zs : Fin G → K) (v : Fin G → Fin 2 → Fin L → K) (g : Fin G) :
    fullLayerInv G L zs v g = groupLayerInv L (zs g) (v g) := rfl

lemma fullLayerInv_comp_fullLayer {zs : Fin G → K}
    (hz : ∀ g, zs g ≠ 0) (h2 : (2 : K) ≠ 0) :
    (fullLayerInv G L zs).comp (fullLayer G L zs) = LinearMap.id := by
  apply LinearMap.ext
  intro v
  funext g
  have hg := groupLayerInv_comp_groupLayer L (hz g) h2
  have : (groupLayerInv L (zs g)).comp (groupLayer L (zs g)) (v g) = v g := by
    rw [hg]; rfl
  simpa [LinearMap.comp_apply] using this

lemma fullLayer_comp_fullLayerInv {zs : Fin G → K}
    (hz : ∀ g, zs g ≠ 0) (h2 : (2 : K) ≠ 0) :
    (fullLayer G L zs).comp (fullLayerInv G L zs) = LinearMap.id := by
  apply LinearMap.ext
  intro v
  funext g
  have hg := groupLayer_comp_groupLayerInv L (hz g) h2
  have : (groupLayer L (zs g)).comp (groupLayerInv L (zs g)) (v g) = v g := by
    rw [hg]; rfl
  simpa [LinearMap.comp_apply] using this

/-- **B7e — `fullLayer zs` is bijective when every twiddle is nonzero
    and `(2 : K) ≠ 0`.** -/
theorem fullLayer_bijective {zs : Fin G → K} (hz : ∀ g, zs g ≠ 0)
    (h2 : (2 : K) ≠ 0) :
    Function.Bijective (fullLayer G L zs) := by
  refine ⟨?_, ?_⟩
  · intro x y hxy
    have hid := fullLayerInv_comp_fullLayer G L hz h2
    have hx : (fullLayerInv G L zs).comp (fullLayer G L zs) x = x := by rw [hid]; rfl
    have hy : (fullLayerInv G L zs).comp (fullLayer G L zs) y = y := by rw [hid]; rfl
    simp only [LinearMap.comp_apply] at hx hy
    rw [← hx, ← hy, hxy]
  · intro w
    refine ⟨fullLayerInv G L zs w, ?_⟩
    have hid := fullLayer_comp_fullLayerInv G L hz h2
    have : (fullLayer G L zs).comp (fullLayerInv G L zs) w = w := by rw [hid]; rfl
    simpa [LinearMap.comp_apply] using this

end FullLayer

end NttFaultRank

