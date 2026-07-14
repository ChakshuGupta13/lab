import OrigamiCone.SequelInterp

/-!
# Sequel meta-theorem: bivariate polynomial assembly for `thm:poly`

Standalone formalisation of the assembly step of the sequel paper's
`Theorem thm:poly` (Polynomiality on the high region), which states:

> For each `d ≥ 3`, `E_d` agrees on `H_d = {m, n ≥ d − 1}` with a single
> **symmetric** bivariate polynomial `p_d(m, n)` of per-axis degree `≤ d − 2`.

The paper's proof combines four pieces:
1. **Per-axis polynomiality**: for fixed `m ≥ 2`, `E_d(m, ·)` is eventually a
   polynomial in `n` (from `SequelRatGF` + `SequelPoles*` + `SequelFrozen` +
   `SequelQuotient` — the peripheral spectrum of `T₀^triv` is `{1}`).
2. **Bivariate interpolation**: per-axis polynomiality upgrades to a single
   bivariate polynomial on the high region (`SequelInterp.interp_principle`).
3. **Symmetry**: `E_d(m, n) = E_d(n, m)` (transposition preserves height
   functions and extrema), so `p_d` is symmetric.
4. **Degree pinning**: `d − 2` from `SequelWalk.abs_numMin_sub_numMax_le_one`
   + `SequelBinom.card_turnPatterns` (`lem:binom` bound on per-axis walk count).

This module formalises step (2) → step (3) — the assembly from per-axis
polynomiality + F-symmetry to bivariate polynomial + polynomial-level symmetry.
The step-(1) bridge (§8 chain → per-axis polynomiality) needs a formal `E_d`,
which the sequel Lean modules do not carry, and is left as the abstract
`hrow`, `hcol` hypotheses. Step (4) (degree pinning) is a separate assembly
using `lem:splitedge`.

Contents:

* `thm_poly_abstract`: the paper's `thm:poly` main assembly — per-axis
  polynomiality + F-symmetry on `{a, b ≥ lo}` produces (i) a bivariate-
  polynomial witness in factored Lagrange form, (ii) F agrees with the witness
  on the high region, (iii) the witness evaluated at `(a, b)` equals its
  evaluation at `(b, a)` on the high region (value-level symmetry).
* `thm_poly_value_symmetry`: projection of `thm_poly_abstract` at the value
  symmetry conjunct, for direct downstream use.
* `thm_poly_polynomial_symmetry`: **strengthened** — the value equality
  extends to ALL `(a, b) : ℚ × ℚ` (not just the grid), via double bivariate
  polynomial uniqueness (`Polynomial.eq_zero_of_infinite_isRoot`). This is
  the paper's `thm:poly` symmetry claim in full: the interpolating bivariate
  polynomial IS symmetric as a function on all of `ℚ²`, hence its underlying
  polynomial coefficients are invariant under swap.

Scope: abstract in `F` — any `F : ℕ → ℕ → ℚ` satisfying the hypotheses
receives the conclusion. A downstream instantiation with a formal `E_d` for
degree-`d` origami-flip-graph vertex counts specialises this to the paper's
`thm:poly`.

No `sorry`; check with `#print axioms OrigamiCone.Sequel.thm_poly_polynomial_symmetry`.
-/

namespace OrigamiCone.Sequel

open Polynomial Finset

/-- **`thm:poly` (abstract assembly)**. If `F : ℕ → ℕ → ℚ` is per-axis polynomial
of degree `≤ D` on `{a, b ≥ lo}` in each variable AND symmetric there, then `F`
agrees on `{a, b ≥ lo}` with a bivariate polynomial (given in factored Lagrange
form) that also evaluates symmetrically on the high region.

The factored form `∑ᵢ (g i).eval b · (L i).eval a` has degree `≤ D` in each
variable (the `g i` are the row polynomials at the `D + 1` interpolation nodes,
each of degree `≤ D` in the column variable; the `L i` are the Lagrange basis
polynomials at those nodes, each of degree exactly `D` in the row variable).
Value-level symmetry (`P(a, b) = P(b, a)` for `a, b ≥ lo`) follows from
`F(a, b) = F(b, a)` and the interpolation identity. -/
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
  -- P(a, b) = F(a, b) = F(b, a) = P(b, a) on the high region.
  rw [← hFP a b ha hb, hsym a b ha hb, hFP b a hb ha]

/-- **`thm:poly` value-level symmetry**. Under the hypotheses of
`thm_poly_abstract`, the bivariate polynomial witness `P` agrees with its
transpose `(a, b) ↦ P(b, a)` on the high region. This is the paper's
`thm:poly` symmetry statement, at the value level (not polynomial-identity
level — that stronger claim needs bivariate polynomial uniqueness). -/
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

/-- **Set of ℕ-image ℚ-values above `lo` is infinite.** Auxiliary lemma: the
image of `{n : ℕ | lo ≤ n}` under the injection `n ↦ ((n : ℕ) : ℚ)` is an
infinite subset of `ℚ`. Used to apply `Polynomial.eq_zero_of_infinite_isRoot`
to polynomials vanishing on the high region of natural numbers. -/
private lemma infinite_nat_image_of_le (lo : ℕ) :
    Set.Infinite {x : ℚ | ∃ n : ℕ, lo ≤ n ∧ x = (n : ℚ)} := by
  apply Set.Infinite.mono (s := Set.range (fun n : ℕ => ((lo + n : ℕ) : ℚ)))
  · rintro x ⟨n, rfl⟩
    exact ⟨lo + n, Nat.le_add_right lo n, rfl⟩
  · exact Set.infinite_range_of_injective (fun n m h => by exact_mod_cast (by
      have : lo + n = lo + m := by exact_mod_cast h
      omega))

/-- **`thm:poly` polynomial-level symmetry**. Under the hypotheses of
`thm_poly_abstract`, the bivariate polynomial witness `P(a, b) = ∑ᵢ (g i)(b) ·
(L i)(a)` satisfies `P(a, b) = P(b, a)` for ALL `a, b : ℚ` — not just on the
high region. Proof uses `Polynomial.eq_zero_of_infinite_isRoot` twice: fix
`a ≥ lo`, then `b ↦ P(a, b) - P(b, a)` is a polynomial in `b` of degree `≤ D`
vanishing on `{b ≥ lo}` (infinite), hence identically zero as a polynomial in
`ℚ[Y]`; then fix `b : ℚ`, the residual polynomial in `a` of degree `≤ D`
vanishes on `{a ≥ lo}` (again infinite), hence identically zero. So
`P(a, b) = P(b, a)` as ℚ-values everywhere. -/
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
  -- STEP 1: for each fixed a : ℕ with lo ≤ a, the residual polynomial in b
  -- vanishes at every b ≥ lo (b : ℕ), and is a polynomial of natDegree ≤ D,
  -- hence identically zero in ℚ[Y]. Then evaluate at arbitrary b : ℚ.
  have hStep1 : ∀ a : ℕ, lo ≤ a → ∀ b : ℚ,
      (∑ i, (g i).eval b * (L i).eval (a : ℚ)) =
        (∑ i, (g i).eval (a : ℚ) * (L i).eval b) := by
    intro aN haN b
    -- Δ_a(Y) := ∑ᵢ ((L i)(aN) · g i) - ∑ᵢ ((g i)(aN) · L i),  a polynomial in Y.
    set Δ : ℚ[X] :=
      ∑ i, ((L i).eval (aN : ℚ)) • (g i) - ∑ i, ((g i).eval (aN : ℚ)) • (L i)
    have hΔroot : ∀ x : ℚ, (∃ n : ℕ, lo ≤ n ∧ x = (n : ℚ)) → IsRoot Δ x := by
      rintro x ⟨bN, hbN, rfl⟩
      -- Show Δ.eval (bN : ℚ) = 0 by connecting to hPsym_hi at (aN, bN).
      show Δ.eval ((bN : ℕ) : ℚ) = 0
      simp only [Δ, eval_sub, eval_finset_sum, eval_smul, smul_eq_mul]
      have hPS := hPsym_hi aN bN haN hbN
      -- hPS : ∑ (g i)(bN) * (L i)(aN) = ∑ (g i)(aN) * (L i)(bN)
      -- Goal: ∑ (L i)(aN) * (g i)(bN) - ∑ (g i)(aN) * (L i)(bN) = 0
      -- Rewrite first sum via mul_comm inside → hPS's LHS, then apply hPS.
      have hL : (∑ i, (L i).eval (aN : ℚ) * (g i).eval ((bN : ℕ) : ℚ)) =
                (∑ i, (g i).eval ((bN : ℕ) : ℚ) * (L i).eval (aN : ℚ)) := by
        apply Finset.sum_congr rfl; intros; ring
      rw [hL, hPS]; ring
    -- Δ vanishes on {(n : ℚ) | lo ≤ n} — an infinite subset of ℚ — hence Δ = 0.
    have hΔzero : Δ = 0 :=
      eq_zero_of_infinite_isRoot Δ ((infinite_nat_image_of_le lo).mono hΔroot)
    -- Evaluate the zero polynomial at b : ℚ.
    have heval := congrArg (Polynomial.eval b) hΔzero
    simp only [Δ, eval_sub, eval_finset_sum, eval_smul, smul_eq_mul, eval_zero] at heval
    -- heval : ∑ (L i)(aN) * (g i)(b) - ∑ (g i)(aN) * (L i)(b) = 0
    -- Goal: ∑ (g i)(b) * (L i)(aN) = ∑ (g i)(aN) * (L i)(b)
    have hL : (∑ i, (L i).eval (aN : ℚ) * (g i).eval b) =
              (∑ i, (g i).eval b * (L i).eval (aN : ℚ)) := by
      apply Finset.sum_congr rfl; intros; ring
    linarith
  -- STEP 2: for each fixed b : ℚ, the residual polynomial in X vanishes at every
  -- a ≥ lo (a : ℕ) — again infinite — hence identically zero.
  intro a b
  set Δ' : ℚ[X] :=
    ∑ i, ((g i).eval b) • (L i) - ∑ i, ((L i).eval b) • (g i)
  have hΔ'root : ∀ x : ℚ, (∃ n : ℕ, lo ≤ n ∧ x = (n : ℚ)) → IsRoot Δ' x := by
    rintro x ⟨aN, haN, rfl⟩
    show Δ'.eval ((aN : ℕ) : ℚ) = 0
    simp only [Δ', eval_sub, eval_finset_sum, eval_smul, smul_eq_mul]
    have hS1 := hStep1 aN haN b
    -- hS1 : ∑ (g i)(b) * (L i)(aN) = ∑ (g i)(aN) * (L i)(b)
    -- Goal: ∑ (g i)(b) * (L i)(aN) - ∑ (L i)(b) * (g i)(aN) = 0
    -- Reorder factors in the second sum via mul_comm.
    have hL : (∑ i, (L i).eval b * (g i).eval ((aN : ℕ) : ℚ)) =
              (∑ i, (g i).eval ((aN : ℕ) : ℚ) * (L i).eval b) := by
      apply Finset.sum_congr rfl; intros; ring
    linarith
  have hΔ'zero : Δ' = 0 :=
    eq_zero_of_infinite_isRoot Δ' ((infinite_nat_image_of_le lo).mono hΔ'root)
  have heval := congrArg (Polynomial.eval a) hΔ'zero
  simp only [Δ', eval_sub, eval_finset_sum, eval_smul, smul_eq_mul, eval_zero] at heval
  -- heval : ∑ (g i)(b) * (L i)(a) - ∑ (L i)(b) * (g i)(a) = 0
  -- Goal: ∑ (g i)(b) * (L i)(a) = ∑ (g i)(a) * (L i)(b)
  have hL : (∑ i, (L i).eval b * (g i).eval a) =
            (∑ i, (g i).eval a * (L i).eval b) := by
    apply Finset.sum_congr rfl; intros; ring
  linarith

end OrigamiCone.Sequel
