/-
Phase C — Multi-layer NTT composition.

To compose layers of different shapes `(G_ℓ, L_ℓ)` into one operator,
every layer must act on the same ambient type `Fin n → K`. We lift
`fullLayer'` along a proof `G * (2 * L) = n` using `Fin.cast`.

C2 (`layerOn`) is the type-arithmetic landmine called out in the plan;
this file tests it in isolation before continuing.
-/

import NttFaultRank.FlatLayer

namespace NttFaultRank

variable {K : Type*} [Field K]

section LayerOn
variable (n G L : ℕ)

/-- `Fin n ≃ Fin (G * (2 * L))` when `n = G * (2 * L)`. -/
def finCastEquiv (h : G * (2 * L) = n) : Fin n ≃ Fin (G * (2 * L)) :=
  Fin.castOrderIso h.symm |>.toEquiv

/-- Function-level reindexing `(Fin n → K) ≃ₗ[K] (Fin (G*(2*L)) → K)`. -/
def funReindex (h : G * (2 * L) = n) :
    (Fin n → K) ≃ₗ[K] (Fin (G * (2 * L)) → K) where
  toFun v := fun i => v ((finCastEquiv n G L h).symm i)
  invFun w := fun i => w (finCastEquiv n G L h i)
  left_inv v := by funext i; simp [finCastEquiv]
  right_inv w := by funext i; simp [finCastEquiv]
  map_add' _ _ := rfl
  map_smul' _ _ := rfl

/-- `layerOn` — one Cooley-Tukey layer lifted to act on `Fin n → K`
    when `n = G * (2 * L)`. -/
def layerOn (h : G * (2 * L) = n) (zs : Fin G → K) :
    (Fin n → K) →ₗ[K] (Fin n → K) :=
  (funReindex n G L h).symm.toLinearMap.comp <|
    (fullLayer' G L zs).comp (funReindex n G L h).toLinearMap

/-- Difference of two `layerOn`s. -/
def layerOnDiff (h : G * (2 * L) = n) (zs zs' : Fin G → K) :
    (Fin n → K) →ₗ[K] (Fin n → K) :=
  layerOn n G L h zs - layerOn n G L h zs'

/-! ### Rank and bijectivity transport through `funReindex` -/

lemma layerOnDiff_eq_conj (h : G * (2 * L) = n) (zs zs' : Fin G → K) :
    layerOnDiff n G L h zs zs' =
      (funReindex n G L h).symm.toLinearMap.comp
        ((fullLayerDiff' G L zs zs').comp (funReindex n G L h).toLinearMap) := by
  apply LinearMap.ext
  intro v
  simp [layerOnDiff, layerOn, fullLayerDiff', LinearMap.sub_apply, LinearMap.comp_apply]

lemma range_layerOnDiff (h : G * (2 * L) = n) (zs zs' : Fin G → K) :
    LinearMap.range (layerOnDiff n G L h zs zs') =
      Submodule.map (funReindex n G L h).symm.toLinearMap
        (LinearMap.range (fullLayerDiff' G L zs zs')) := by
  rw [layerOnDiff_eq_conj]
  rw [LinearMap.range_comp, LinearMap.range_comp]
  rw [LinearMap.range_eq_top.mpr (funReindex n G L h).surjective]
  simp

theorem layerOnDiff_rank_eq {h : G * (2 * L) = n} {zs zs' : Fin G → K}
    (g₀ : Fin G) (hagree : ∀ g ≠ g₀, zs g = zs' g)
    (hz : zs g₀ ≠ 0) (hz' : zs' g₀ = 0) :
    Module.finrank K (LinearMap.range (layerOnDiff n G L h zs zs')) = L := by
  rw [range_layerOnDiff n G L h zs zs']
  rw [← ((funReindex n G L h).symm.submoduleMap
            (LinearMap.range (fullLayerDiff' G L zs zs'))).finrank_eq] at *
  exact fullLayerDiff'_rank_eq G L g₀ hagree hz hz'

/-- Generalized: rank = L when twiddles differ at g₀. -/
theorem layerOnDiff_rank_eq_gen {h : G * (2 * L) = n} {zs zs' : Fin G → K}
    (g₀ : Fin G) (hagree : ∀ g ≠ g₀, zs g = zs' g)
    (hne : zs g₀ ≠ zs' g₀) :
    Module.finrank K (LinearMap.range (layerOnDiff n G L h zs zs')) = L := by
  rw [range_layerOnDiff n G L h zs zs']
  rw [← ((funReindex n G L h).symm.submoduleMap
            (LinearMap.range (fullLayerDiff' G L zs zs'))).finrank_eq] at *
  exact fullLayerDiff'_rank_eq_gen G L g₀ hagree hne

theorem layerOn_bijective {h : G * (2 * L) = n} {zs : Fin G → K}
    (hz : ∀ g, zs g ≠ 0) (h2 : (2 : K) ≠ 0) :
    Function.Bijective (layerOn n G L h zs) := by
  show Function.Bijective
    ((funReindex n G L h).symm ∘ (fullLayer' G L zs) ∘ (funReindex n G L h))
  exact (funReindex n G L h).symm.bijective.comp
    ((fullLayer'_bijective G L hz h2).comp (funReindex n G L h).bijective)

end LayerOn

end NttFaultRank

