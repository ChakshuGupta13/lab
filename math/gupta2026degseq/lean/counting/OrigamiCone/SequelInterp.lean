import Mathlib

namespace OrigamiCone.Sequel

open Polynomial

theorem interp_principle {D lo : ℕ} (F : ℕ → ℕ → ℚ)
    (hrow : ∀ a, lo ≤ a → ∃ p : ℚ[X], p.natDegree ≤ D ∧
      ∀ b, lo ≤ b → F a b = p.eval (b : ℚ))
    (hcol : ∀ b, lo ≤ b → ∃ q : ℚ[X], q.natDegree ≤ D ∧
      ∀ a, lo ≤ a → F a b = q.eval (a : ℚ)) :
    ∃ (g L : Fin (D + 1) → ℚ[X]),
      (∀ i, (g i).natDegree ≤ D) ∧ (∀ i, (L i).natDegree ≤ D) ∧
      ∀ a b, lo ≤ a → lo ≤ b →
        F a b = ∑ i, (g i).eval (b : ℚ) * (L i).eval (a : ℚ) := by
  classical

  set x : Fin (D + 1) → ℚ := fun i => (lo : ℚ) + ((i : ℕ) : ℚ) with hx
  have hinj : Set.InjOn x (↑(Finset.univ : Finset (Fin (D + 1)))) := by
    intro i _ j _ hij
    simp only [hx] at hij
    have hcast : ((i : ℕ) : ℚ) = ((j : ℕ) : ℚ) := by linarith
    have : (i : ℕ) = (j : ℕ) := by exact_mod_cast hcast
    exact Fin.ext this

  choose g hgdeg hgeval using fun i : Fin (D + 1) =>
    hrow (lo + (i : ℕ)) (Nat.le_add_right lo (i : ℕ))
  refine ⟨g, fun i => Lagrange.basis Finset.univ x i, hgdeg, ?_, ?_⟩
  ·
    intro i
    rw [Lagrange.natDegree_basis hinj (Finset.mem_univ i), Finset.card_univ,
        Fintype.card_fin]
    omega
  · intro a b ha hb
    obtain ⟨q, hqdeg, hqeval⟩ := hcol b hb

    have hdeg : q.degree < ((Finset.univ : Finset (Fin (D + 1))).card : WithBot ℕ) := by
      rw [Finset.card_univ, Fintype.card_fin]
      calc q.degree ≤ (q.natDegree : WithBot ℕ) := degree_le_natDegree
        _ ≤ (D : WithBot ℕ) := by exact_mod_cast hqdeg
        _ < ((D + 1 : ℕ) : WithBot ℕ) := by exact_mod_cast Nat.lt_succ_self D
    have hnode : ∀ i ∈ (Finset.univ : Finset (Fin (D + 1))),
        eval (x i) q = (g i).eval (b : ℚ) := by
      intro i _
      have hxi : x i = ((lo + (i : ℕ) : ℕ) : ℚ) := by simp only [hx]; push_cast; ring
      rw [hxi, ← hqeval (lo + (i : ℕ)) (Nat.le_add_right _ _)]
      exact hgeval i b hb
    have hq : q = Lagrange.interpolate Finset.univ x (fun i => (g i).eval (b : ℚ)) :=
      Lagrange.eq_interpolate_of_eval_eq _ hinj hdeg hnode
    rw [hqeval a ha, hq, Lagrange.interpolate_apply, eval_finset_sum]
    simp only [eval_mul, eval_C]

end OrigamiCone.Sequel
