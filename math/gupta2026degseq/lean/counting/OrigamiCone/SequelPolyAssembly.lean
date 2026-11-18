import OrigamiCone.SequelInterp

namespace OrigamiCone.Sequel

open Polynomial Finset

theorem thm_poly_abstract {D lo : ℕ} (F : ℕ → ℕ → ℚ)
    (hrow : ∀ a, lo ≤ a → ∃ p : ℚ[X], p.natDegree ≤ D ∧
      ∀ b, lo ≤ b → F a b = p.eval (b : ℚ))
    (hcol : ∀ b, lo ≤ b → ∃ q : ℚ[X], q.natDegree ≤ D ∧
      ∀ a, lo ≤ a → F a b = q.eval (a : ℚ))
    (hsym : ∀ a b, lo ≤ a → lo ≤ b → F a b = F b a) :
    ∃ (g L : Fin (D + 1) → ℚ[X]),
      (∀ i, (g i).natDegree ≤ D) ∧ (∀ i, (L i).natDegree ≤ D) ∧
      (∀ a b, lo ≤ a → lo ≤ b →
        F a b = ∑ i, (g i).eval (b : ℚ) * (L i).eval (a : ℚ)) ∧
      (∀ a b, lo ≤ a → lo ≤ b →
        (∑ i, (g i).eval (b : ℚ) * (L i).eval (a : ℚ)) =
          (∑ i, (g i).eval (a : ℚ) * (L i).eval (b : ℚ))) := by
  obtain ⟨g, L, hgdeg, hLdeg, hFP⟩ := interp_principle F hrow hcol
  refine ⟨g, L, hgdeg, hLdeg, hFP, ?_⟩
  intro a b ha hb

  rw [← hFP a b ha hb, hsym a b ha hb, hFP b a hb ha]

theorem thm_poly_value_symmetry {D lo : ℕ} (F : ℕ → ℕ → ℚ)
    (hrow : ∀ a, lo ≤ a → ∃ p : ℚ[X], p.natDegree ≤ D ∧
      ∀ b, lo ≤ b → F a b = p.eval (b : ℚ))
    (hcol : ∀ b, lo ≤ b → ∃ q : ℚ[X], q.natDegree ≤ D ∧
      ∀ a, lo ≤ a → F a b = q.eval (a : ℚ))
    (hsym : ∀ a b, lo ≤ a → lo ≤ b → F a b = F b a) :
    ∃ (g L : Fin (D + 1) → ℚ[X]),
      (∀ a b, lo ≤ a → lo ≤ b →
        (∑ i, (g i).eval (b : ℚ) * (L i).eval (a : ℚ)) =
          (∑ i, (g i).eval (a : ℚ) * (L i).eval (b : ℚ))) := by
  obtain ⟨g, L, _, _, _, hPsym⟩ := thm_poly_abstract F hrow hcol hsym
  exact ⟨g, L, hPsym⟩

private lemma infinite_nat_image_of_le (lo : ℕ) :
    Set.Infinite {x : ℚ | ∃ n : ℕ, lo ≤ n ∧ x = (n : ℚ)} := by
  apply Set.Infinite.mono (s := Set.range (fun n : ℕ => ((lo + n : ℕ) : ℚ)))
  · rintro x ⟨n, rfl⟩
    exact ⟨lo + n, Nat.le_add_right lo n, rfl⟩
  · exact Set.infinite_range_of_injective (fun n m h => by exact_mod_cast (by
      have : lo + n = lo + m := by exact_mod_cast h
      omega))

theorem thm_poly_polynomial_symmetry {D lo : ℕ} (F : ℕ → ℕ → ℚ)
    (hrow : ∀ a, lo ≤ a → ∃ p : ℚ[X], p.natDegree ≤ D ∧
      ∀ b, lo ≤ b → F a b = p.eval (b : ℚ))
    (hcol : ∀ b, lo ≤ b → ∃ q : ℚ[X], q.natDegree ≤ D ∧
      ∀ a, lo ≤ a → F a b = q.eval (a : ℚ))
    (hsym : ∀ a b, lo ≤ a → lo ≤ b → F a b = F b a) :
    ∃ (g L : Fin (D + 1) → ℚ[X]),
      (∀ i, (g i).natDegree ≤ D) ∧ (∀ i, (L i).natDegree ≤ D) ∧
      (∀ a b, lo ≤ a → lo ≤ b →
        F a b = ∑ i, (g i).eval (b : ℚ) * (L i).eval (a : ℚ)) ∧
      (∀ a b : ℚ,
        (∑ i, (g i).eval b * (L i).eval a) =
          (∑ i, (g i).eval a * (L i).eval b)) := by
  obtain ⟨g, L, hgdeg, hLdeg, hFP, hPsym_hi⟩ := thm_poly_abstract F hrow hcol hsym
  refine ⟨g, L, hgdeg, hLdeg, hFP, ?_⟩

  have hStep1 : ∀ a : ℕ, lo ≤ a → ∀ b : ℚ,
      (∑ i, (g i).eval b * (L i).eval (a : ℚ)) =
        (∑ i, (g i).eval (a : ℚ) * (L i).eval b) := by
    intro aN haN b

    set Δ : ℚ[X] :=
      ∑ i, ((L i).eval (aN : ℚ)) • (g i) - ∑ i, ((g i).eval (aN : ℚ)) • (L i)
    have hΔroot : ∀ x : ℚ, (∃ n : ℕ, lo ≤ n ∧ x = (n : ℚ)) → IsRoot Δ x := by
      rintro x ⟨bN, hbN, rfl⟩

      show Δ.eval ((bN : ℕ) : ℚ) = 0
      simp only [Δ, eval_sub, eval_finset_sum, eval_smul, smul_eq_mul]
      have hPS := hPsym_hi aN bN haN hbN

      have hL : (∑ i, (L i).eval (aN : ℚ) * (g i).eval ((bN : ℕ) : ℚ)) =
                (∑ i, (g i).eval ((bN : ℕ) : ℚ) * (L i).eval (aN : ℚ)) := by
        apply Finset.sum_congr rfl; intros; ring
      rw [hL, hPS]; ring

    have hΔzero : Δ = 0 :=
      eq_zero_of_infinite_isRoot Δ ((infinite_nat_image_of_le lo).mono hΔroot)

    have heval := congrArg (Polynomial.eval b) hΔzero
    simp only [Δ, eval_sub, eval_finset_sum, eval_smul, smul_eq_mul, eval_zero] at heval

    have hL : (∑ i, (L i).eval (aN : ℚ) * (g i).eval b) =
              (∑ i, (g i).eval b * (L i).eval (aN : ℚ)) := by
      apply Finset.sum_congr rfl; intros; ring
    linarith

  intro a b
  set Δ' : ℚ[X] :=
    ∑ i, ((g i).eval b) • (L i) - ∑ i, ((L i).eval b) • (g i)
  have hΔ'root : ∀ x : ℚ, (∃ n : ℕ, lo ≤ n ∧ x = (n : ℚ)) → IsRoot Δ' x := by
    rintro x ⟨aN, haN, rfl⟩
    show Δ'.eval ((aN : ℕ) : ℚ) = 0
    simp only [Δ', eval_sub, eval_finset_sum, eval_smul, smul_eq_mul]
    have hS1 := hStep1 aN haN b

    have hL : (∑ i, (L i).eval b * (g i).eval ((aN : ℕ) : ℚ)) =
              (∑ i, (g i).eval ((aN : ℕ) : ℚ) * (L i).eval b) := by
      apply Finset.sum_congr rfl; intros; ring
    linarith
  have hΔ'zero : Δ' = 0 :=
    eq_zero_of_infinite_isRoot Δ' ((infinite_nat_image_of_le lo).mono hΔ'root)
  have heval := congrArg (Polynomial.eval a) hΔ'zero
  simp only [Δ', eval_sub, eval_finset_sum, eval_smul, smul_eq_mul, eval_zero] at heval

  have hL : (∑ i, (L i).eval b * (g i).eval a) =
            (∑ i, (g i).eval a * (L i).eval b) := by
    apply Finset.sum_congr rfl; intros; ring
  linarith

end OrigamiCone.Sequel
