/-
Phase A — Reshape: `(Fin (G * (2 * L)) → K) ≃ₗ[K] (Fin G → Fin 2 → Fin L → K)`.

Defined directly as a `LinearEquiv` rather than composed from
`piCongrLeft`/`piCurry` chains, which require sigma↔prod gymnastics.
-/

import Mathlib.LinearAlgebra.Pi
import Mathlib.Logic.Equiv.Fin.Basic

namespace NttFaultRank

variable (K : Type*) [Semiring K] (G L : ℕ)

/-- The coordinate equivalence `Fin (G * (2 * L)) ≃ Fin G × Fin 2 × Fin L`. -/
def coordEquiv : Fin (G * (2 * L)) ≃ Fin G × (Fin 2 × Fin L) :=
  finProdFinEquiv.symm.trans
    ((Equiv.refl (Fin G)).prodCongr finProdFinEquiv.symm)

/-- Reshape `(Fin (G*2*L) → K) ≃ₗ[K] (Fin G → Fin 2 → Fin L → K)`. -/
def reshape :
    (Fin (G * (2 * L)) → K) ≃ₗ[K] (Fin G → Fin 2 → Fin L → K) where
  toFun v := fun g s j => v ((coordEquiv G L).symm (g, s, j))
  invFun w := fun i =>
    let p := coordEquiv G L i
    w p.1 p.2.1 p.2.2
  left_inv v := by
    funext i
    show v ((coordEquiv G L).symm (coordEquiv G L i)) = v i
    rw [Equiv.symm_apply_apply]
  right_inv w := by
    funext g s j
    simp only [Equiv.apply_symm_apply]
  map_add' x y := by funext g s j; rfl
  map_smul' c v := by funext g s j; rfl

@[simp] lemma reshape_apply (v : Fin (G * (2 * L)) → K) (g : Fin G) (s : Fin 2) (j : Fin L) :
    reshape K G L v g s j = v ((coordEquiv G L).symm (g, s, j)) := rfl

@[simp] lemma reshape_symm_apply (w : Fin G → Fin 2 → Fin L → K)
    (i : Fin (G * (2 * L))) :
    (reshape K G L).symm w i =
      w (coordEquiv G L i).1 (coordEquiv G L i).2.1 (coordEquiv G L i).2.2 := rfl

end NttFaultRank
