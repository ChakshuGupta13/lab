/-
Phase F4 — Per-layer perturbation rank.

`P_ℓ(Z, F) := layer Z ℓ - layer (faultedTwiddles Z F) ℓ` is the
single-layer twiddle perturbation that appears as the middle factor of
each `teleTerm`. When the fault at layer `ℓ` is a single non-zero
twiddle being zeroed out, the range of `P_ℓ` has finrank `spec.L ℓ`
(the butterfly length).

This is essentially a wrapper around `layerOnDiff_rank_eq` from
`LayerOn.lean`, packaged at the `LayerSpec` level for use in F5.
-/

import NttFaultRank.Invertibility

namespace NttFaultRank

variable {K : Type*} [Field K]

namespace LayerSpec
variable (spec : LayerSpec)

/-- The layer-`ℓ` perturbation: difference of the unfaulted and faulted
    layer maps. -/
def perturbation (Z : spec.Twiddles K) (F : spec.FaultSet) (ℓ : Fin spec.N) :
    (Fin spec.n → K) →ₗ[K] (Fin spec.n → K) :=
  spec.layer Z ℓ - spec.layer (spec.faultedTwiddles Z F) ℓ

lemma perturbation_eq_layerOnDiff (Z : spec.Twiddles K) (F : spec.FaultSet)
    (ℓ : Fin spec.N) :
    spec.perturbation Z F ℓ =
      layerOnDiff spec.n (spec.G ℓ) (spec.L ℓ) (spec.hGL ℓ)
        (Z ℓ) ((spec.faultedTwiddles Z F) ℓ) := by
  rfl

/-- **Headline F4.** Under "one fault per layer" with all `Z` twiddles
    nonzero, the layer-`ℓ` perturbation has rank `spec.L ℓ`. -/
theorem perturbation_rank_eq
    {Z : spec.Twiddles K} (hZ : ∀ ℓ g, Z ℓ g ≠ 0)
    {F : spec.FaultSet} (hF : spec.OneFaultPerLayer F) (ℓ : Fin spec.N) :
    Module.finrank K (LinearMap.range (spec.perturbation Z F ℓ)) = spec.L ℓ := by
  -- Extract the unique fault index at layer ℓ.
  obtain ⟨g₀, hg₀⟩ := Finset.card_eq_one.mp (hF ℓ)
  rw [spec.perturbation_eq_layerOnDiff Z F ℓ]
  refine layerOnDiff_rank_eq (n := spec.n) (G := spec.G ℓ) (L := spec.L ℓ)
    (h := spec.hGL ℓ) (zs := Z ℓ) (zs' := (spec.faultedTwiddles Z F) ℓ)
    g₀ ?_ ?_ ?_
  · -- agreement off g₀
    intro g hg
    unfold faultedTwiddles
    simp [hg₀, hg]
  · -- Z ℓ g₀ ≠ 0
    exact hZ ℓ g₀
  · -- faultedTwiddles Z F ℓ g₀ = 0
    unfold faultedTwiddles
    simp [hg₀]

/-! ### Generalized perturbation: arbitrary replacement twiddles -/

/-- Generalized perturbation: difference of unfaulted and generalized-faulted layer. -/
def perturbationGen (Z Z' : spec.Twiddles K) (F : spec.FaultSet) (ℓ : Fin spec.N) :
    (Fin spec.n → K) →ₗ[K] (Fin spec.n → K) :=
  spec.layer Z ℓ - spec.layer (spec.faultedTwiddlesGen Z Z' F) ℓ

/-- **Headline F4-gen.** Under "one fault per layer", when faulted twiddles
    differ from original (`Z ℓ g ≠ Z' ℓ g` at faulted groups), the layer-`ℓ`
    perturbation has rank `spec.L ℓ`. -/
theorem perturbation_rank_eq_gen
    {Z Z' : spec.Twiddles K} {F : spec.FaultSet}
    (hδ : ∀ ℓ g, g ∈ F ℓ → Z ℓ g ≠ Z' ℓ g)
    (hF : spec.OneFaultPerLayer F) (ℓ : Fin spec.N) :
    Module.finrank K (LinearMap.range (spec.perturbationGen Z Z' F ℓ)) = spec.L ℓ := by
  obtain ⟨g₀, hg₀⟩ := Finset.card_eq_one.mp (hF ℓ)
  show Module.finrank K (LinearMap.range
    (layerOnDiff spec.n (spec.G ℓ) (spec.L ℓ) (spec.hGL ℓ)
      (Z ℓ) ((spec.faultedTwiddlesGen Z Z' F) ℓ))) = spec.L ℓ
  refine layerOnDiff_rank_eq_gen (n := spec.n) (G := spec.G ℓ) (L := spec.L ℓ)
    (h := spec.hGL ℓ) (zs := Z ℓ) (zs' := (spec.faultedTwiddlesGen Z Z' F) ℓ)
    g₀ ?_ ?_
  · -- agreement off g₀
    intro g hg
    unfold faultedTwiddlesGen
    simp [hg₀, hg]
  · -- Z ℓ g₀ ≠ faultedTwiddlesGen Z Z' F ℓ g₀
    unfold faultedTwiddlesGen
    have hg₀_mem : g₀ ∈ F ℓ := by rw [hg₀]; exact Finset.mem_singleton_self g₀
    rw [if_pos hg₀_mem]
    exact hδ ℓ g₀ hg₀_mem

end LayerSpec

end NttFaultRank
