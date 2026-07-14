import Mathlib

/-!
# Sequel meta-theorem: the interpolation principle (`lem:interp`)

Standalone formalisation of the **Interpolation Principle** of the sequel paper

> *Degree-$d$ vertex counts of the $m \times n$ origami flip graph:
> a polynomial meta-theorem.*

In the sequel, per-axis polynomiality (for each fixed `m` the count is a
polynomial in `n` of degree `≤ D`, and symmetrically) is upgraded to a single
*bivariate* polynomial on the high region by `Lemma lem:interp`:

> If `F : {d-1,…}² → ℚ` is, for each fixed first coordinate, a polynomial of
> degree `≤ D` in the second, and for each fixed second coordinate a polynomial
> of degree `≤ D` in the first, then `F` agrees on `{m,n ≥ d-1}` with a single
> bivariate polynomial of degree `≤ D` in each variable.

This is the step that makes `Theorem thm:poly` (polynomiality on the high region)
unconditional once per-axis polynomiality is in hand.

## The Lean statement

`interp_principle` produces the bivariate polynomial in *factored Lagrange form*:
fixing the `D+1` interpolation nodes `a = lo, lo+1, …, lo+D`, there are row
polynomials `g i` (the `F(lo+i, ·)`, each of degree `≤ D` in the column variable)
and the Lagrange basis polynomials `L i` for the nodes (each of degree exactly
`D = card-1` in the row variable), such that
`F a b = ∑ i, (g i).eval b · (L i).eval a` on the whole high region.
The right-hand side is the single bivariate polynomial: its dependence on the
column variable `b` runs through the `g i` (degree `≤ D`), and on the row variable
`a` through the `L i` (degree `≤ D`), so it has degree `≤ D` in each variable —
exactly the paper's conclusion, written as an explicit separable sum.

The proof is the paper's: build `P(a,b) = ∑ g_i(b) L_i(a)`; for each fixed `b`,
the column polynomial `q_b` (degree `≤ D`) agrees with the interpolant of the
node values `g_i(b)` at the `D+1` nodes (`Lagrange.eq_interpolate_of_eval_eq`), so
they coincide, giving `F(a,b) = q_b(a) = ∑ g_i(b) L_i(a)`.

No `sorry`; check with `#print axioms OrigamiCone.Sequel.interp_principle`.
-/

namespace OrigamiCone.Sequel

open Polynomial

/-- **Interpolation principle** (`Lemma lem:interp`). If `F` is separately
polynomial of degree `≤ D` in each variable on the high region `{a,b ≥ lo}`, then
`F` agrees there with a single bivariate polynomial of degree `≤ D` in each
variable, exhibited in factored Lagrange form `∑ i, (g i).eval b · (L i).eval a`
with `g i, L i` of degree `≤ D`. -/
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
  -- node values x i = lo + i, distinct in ℚ
  set x : Fin (D + 1) → ℚ := fun i => (lo : ℚ) + ((i : ℕ) : ℚ) with hx
  have hinj : Set.InjOn x (↑(Finset.univ : Finset (Fin (D + 1)))) := by
    intro i _ j _ hij
    simp only [hx] at hij
    have hcast : ((i : ℕ) : ℚ) = ((j : ℕ) : ℚ) := by linarith
    have : (i : ℕ) = (j : ℕ) := by exact_mod_cast hcast
    exact Fin.ext this
  -- row polynomials at the nodes lo + i
  choose g hgdeg hgeval using fun i : Fin (D + 1) =>
    hrow (lo + (i : ℕ)) (Nat.le_add_right lo (i : ℕ))
  refine ⟨g, fun i => Lagrange.basis Finset.univ x i, hgdeg, ?_, ?_⟩
  · -- each Lagrange basis polynomial has degree card-1 = D
    intro i
    rw [Lagrange.natDegree_basis hinj (Finset.mem_univ i), Finset.card_univ,
        Fintype.card_fin]
    omega
  · intro a b ha hb
    obtain ⟨q, hqdeg, hqeval⟩ := hcol b hb
    -- q is the interpolant of the node values b ↦ g_i(b)
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
