/-
Phase B — Flat-form Cooley-Tukey layer.

Defines `fullLayer'` acting on `Fin (G*(2*L)) → K` by conjugating the
indexed-form `fullLayer` with `reshape`. The conjugation transports
rank and bijectivity from `FullLayer.lean` immediately, because
`LinearEquiv` preserves both `finrank` and `Function.Bijective`.
-/

import NttFaultRank.FullLayer
import NttFaultRank.Reshape

namespace NttFaultRank

variable {K : Type*} [Field K]

section FlatLayer
variable (G L : ℕ)

/-- One Cooley-Tukey layer on the flat vector space `Fin (G*2*L) → K`. -/
def fullLayer' (zs : Fin G → K) :
    (Fin (G * (2 * L)) → K) →ₗ[K] (Fin (G * (2 * L)) → K) :=
  (reshape K G L).symm.toLinearMap.comp <|
    (fullLayer G L zs).comp (reshape K G L).toLinearMap

/-- Difference of `fullLayer'` at two twiddle vectors. -/
def fullLayerDiff' (zs zs' : Fin G → K) :
    (Fin (G * (2 * L)) → K) →ₗ[K] (Fin (G * (2 * L)) → K) :=
  fullLayer' G L zs - fullLayer' G L zs'

/-- The flat-form difference is `reshape⁻¹ ∘ (indexed diff) ∘ reshape`. -/
lemma fullLayerDiff'_eq_conj (zs zs' : Fin G → K) :
    fullLayerDiff' G L zs zs' =
      (reshape K G L).symm.toLinearMap.comp
        ((fullLayerDiff G L zs zs').comp (reshape K G L).toLinearMap) := by
  apply LinearMap.ext
  intro v
  simp [fullLayerDiff', fullLayer', fullLayerDiff, LinearMap.sub_apply,
        LinearMap.comp_apply]

/-! ### B3' — rank transport -/

/-- The range of the flat-form diff equals the image, under
    `reshape.symm`, of the indexed-form diff's range. -/
lemma range_fullLayerDiff' (zs zs' : Fin G → K) :
    LinearMap.range (fullLayerDiff' G L zs zs') =
      Submodule.map (reshape K G L).symm.toLinearMap
        (LinearMap.range (fullLayerDiff G L zs zs')) := by
  rw [fullLayerDiff'_eq_conj]
  rw [LinearMap.range_comp]
  congr 1
  rw [LinearMap.range_comp]
  rw [LinearMap.range_eq_top.mpr (reshape K G L).surjective]
  simp

/-- **B3' — Single-group-fault flat-layer rank = L.** -/
theorem fullLayerDiff'_rank_eq {zs zs' : Fin G → K} (g₀ : Fin G)
    (hagree : ∀ g ≠ g₀, zs g = zs' g)
    (hz : zs g₀ ≠ 0) (hz' : zs' g₀ = 0) :
    Module.finrank K (LinearMap.range (fullLayerDiff' G L zs zs')) = L := by
  rw [range_fullLayerDiff' G L zs zs']
  -- A LinearEquiv preserves finrank of any subspace via its underlying map.
  have hinj := (reshape K G L).symm.injective
  rw [← ((reshape K G L).symm.submoduleMap
          (LinearMap.range (fullLayerDiff G L zs zs'))).finrank_eq] at *
  -- Recover the inner rank via FullLayer.fullLayerDiff_rank_eq.
  exact fullLayerDiff_rank_eq G L g₀ hagree hz hz'

/-- **B3'-gen — Generalized flat-layer rank = L when twiddles differ at g₀.** -/
theorem fullLayerDiff'_rank_eq_gen {zs zs' : Fin G → K} (g₀ : Fin G)
    (hagree : ∀ g ≠ g₀, zs g = zs' g)
    (hne : zs g₀ ≠ zs' g₀) :
    Module.finrank K (LinearMap.range (fullLayerDiff' G L zs zs')) = L := by
  rw [range_fullLayerDiff' G L zs zs']
  rw [← ((reshape K G L).symm.submoduleMap
          (LinearMap.range (fullLayerDiff G L zs zs'))).finrank_eq] at *
  exact fullLayerDiff_rank_eq_gen G L g₀ hagree hne

/-! ### B2' — flat-layer bijectivity -/

/-- **B2' — Flat-form `fullLayer'` is bijective when twiddles are nonzero.** -/
theorem fullLayer'_bijective {zs : Fin G → K} (hz : ∀ g, zs g ≠ 0)
    (h2 : (2 : K) ≠ 0) :
    Function.Bijective (fullLayer' G L zs) := by
  show Function.Bijective
    ((reshape K G L).symm ∘ (fullLayer G L zs) ∘ (reshape K G L))
  exact (reshape K G L).symm.bijective.comp
    ((fullLayer_bijective G L hz h2).comp (reshape K G L).bijective)

end FlatLayer

end NttFaultRank

