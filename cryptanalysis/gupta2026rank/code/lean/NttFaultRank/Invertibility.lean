/-
Phase F2 — Invertibility of layer chains.

Builds a `LinearEquiv` for any composition of NTT layers whose twiddles
are all nonzero. The recursion is on a `List (Fin spec.N)` of layer
indices rather than a varying-bound `Fin.foldl`, sidestepping the
dependent-type friction that blocked earlier `chainAt offset len`
attempts.

Deliverables (for F5):
  * `layerEquiv Z ℓ hZ h2` — one layer as `LinearEquiv` when its
    twiddles are nonzero.
  * `chainList Z ls` — `LinearMap` composition of layers in `ls`,
    applying the head FIRST (so `(ℓ :: ls).chainList = chainList ls ∘ layer ℓ`).
  * `chainListEquiv Z h2 ls hZ` — `LinearEquiv` form of the chain when
    every twiddle in `ls` is nonzero.
  * `chainListEquiv_toLinearMap` — the equiv's underlying linear map
    equals `chainList`.
-/

import NttFaultRank.Telescope

namespace NttFaultRank

variable {K : Type*} [Field K]

namespace LayerSpec
variable (spec : LayerSpec)

/-- One NTT layer as a `LinearEquiv`, given nonzero twiddles and
    `(2 : K) ≠ 0`. -/
noncomputable def layerEquiv (Z : spec.Twiddles K) (ℓ : Fin spec.N)
    (hZ : ∀ g, Z ℓ g ≠ 0) (h2 : (2 : K) ≠ 0) :
    (Fin spec.n → K) ≃ₗ[K] (Fin spec.n → K) :=
  LinearEquiv.ofBijective (spec.layer Z ℓ)
    (layerOn_bijective spec.n (spec.G ℓ) (spec.L ℓ) hZ h2)

@[simp] lemma layerEquiv_toLinearMap (Z : spec.Twiddles K) (ℓ : Fin spec.N)
    (hZ : ∀ g, Z ℓ g ≠ 0) (h2 : (2 : K) ≠ 0) :
    (spec.layerEquiv Z ℓ hZ h2).toLinearMap = spec.layer Z ℓ := rfl

/-- `chainList Z ls` composes the layers indexed by `ls`, applying the
    head of the list FIRST. -/
def chainList (Z : spec.Twiddles K) : List (Fin spec.N) →
    ((Fin spec.n → K) →ₗ[K] (Fin spec.n → K))
  | [] => LinearMap.id
  | ℓ :: ls => (chainList Z ls).comp (spec.layer Z ℓ)

@[simp] lemma chainList_nil (Z : spec.Twiddles K) :
    chainList spec Z [] = LinearMap.id := rfl

@[simp] lemma chainList_cons (Z : spec.Twiddles K) (ℓ : Fin spec.N)
    (ls : List (Fin spec.N)) :
    chainList spec Z (ℓ :: ls) = (chainList spec Z ls).comp (spec.layer Z ℓ) :=
  rfl

/-- `LinearEquiv` form of `chainList`. Built by structural recursion on
    the list; the input `hZ` certifies that every layer used has
    nonzero twiddles. -/
noncomputable def chainListEquiv (Z : spec.Twiddles K) (h2 : (2 : K) ≠ 0) :
    (ls : List (Fin spec.N)) → (∀ ℓ ∈ ls, ∀ g, Z ℓ g ≠ 0) →
    ((Fin spec.n → K) ≃ₗ[K] (Fin spec.n → K))
  | [], _ => LinearEquiv.refl K _
  | ℓ :: ls, hZ =>
    (spec.layerEquiv Z ℓ (hZ ℓ List.mem_cons_self) h2).trans
      (chainListEquiv Z h2 ls
        (fun ℓ' hℓ' => hZ ℓ' (List.mem_cons_of_mem _ hℓ')))

@[simp] lemma chainListEquiv_nil (Z : spec.Twiddles K) (h2 : (2 : K) ≠ 0)
    (hZ : ∀ ℓ ∈ ([] : List (Fin spec.N)), ∀ g, Z ℓ g ≠ 0) :
    chainListEquiv spec Z h2 [] hZ = LinearEquiv.refl K _ := rfl

lemma chainListEquiv_cons (Z : spec.Twiddles K) (h2 : (2 : K) ≠ 0)
    (ℓ : Fin spec.N) (ls : List (Fin spec.N))
    (hZ : ∀ ℓ' ∈ (ℓ :: ls), ∀ g, Z ℓ' g ≠ 0) :
    chainListEquiv spec Z h2 (ℓ :: ls) hZ =
      (spec.layerEquiv Z ℓ (hZ ℓ List.mem_cons_self) h2).trans
        (chainListEquiv spec Z h2 ls
          (fun ℓ' hℓ' => hZ ℓ' (List.mem_cons_of_mem _ hℓ'))) := rfl

/-- **Headline F2.** The `LinearEquiv` form of a layer chain has
    `chainList` as its underlying linear map. -/
theorem chainListEquiv_toLinearMap (Z : spec.Twiddles K) (h2 : (2 : K) ≠ 0) :
    ∀ (ls : List (Fin spec.N)) (hZ : ∀ ℓ ∈ ls, ∀ g, Z ℓ g ≠ 0),
      (chainListEquiv spec Z h2 ls hZ).toLinearMap = chainList spec Z ls
  | [], _ => rfl
  | ℓ :: ls, hZ => by
    rw [chainListEquiv_cons spec Z h2 ℓ ls hZ, chainList_cons spec Z ℓ ls]
    ext v
    have ih := chainListEquiv_toLinearMap Z h2 ls
      (fun ℓ' hℓ' => hZ ℓ' (List.mem_cons_of_mem _ hℓ'))
    -- (A.trans B) v = B (A v); chainList_cons puts the head map on the right.
    simp [LinearEquiv.trans_apply, layerEquiv, ← ih]

end LayerSpec

end NttFaultRank
