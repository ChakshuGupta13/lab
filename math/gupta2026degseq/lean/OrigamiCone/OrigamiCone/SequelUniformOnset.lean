import OrigamiCone.SequelEd
import OrigamiCone.SequelUniformOnsetProof

/-!
# `thm:poly` at `Ed`, unconditional modulo the paper's Lemma 8.5

`SequelEd.Ed_thm_poly_of_perAxis` (the specialisation of the abstract
`thm_poly_polynomial_symmetry` at `F := fun m n => (Ed d m n : ℚ)`)
takes per-axis polynomiality `hrow` and `hcol` at a common threshold
`lo` as explicit hypotheses.  The paper's `Lemma 8.5` (`lem:uniform`,
Uniform onset) supplies exactly `hrow` at `Ed` at an `m`-free threshold
`N_d ≤ 2d + 4`; symmetry `Ed_symm` then supplies `hcol` from the same
witness.

This module packages that final reduction:

1. `Ed_uniform_onset` — the paper's `Lemma 8.5` as a named
   theorem.  Proved from the finer axiom `Ed_decomposition`
   (`SequelUniformOnsetProof`, substrate 5/5) using substrates 1–4.
2. `Ed_thm_poly_unconditional` — the paper's `Theorem thm:poly`
   (existence half) at `Ed`, in factored-Lagrange form, resting on
   the single axiom `Ed_decomposition`.  Every other step in the
   reduction (per-axis polynomiality along columns from symmetry;
   the bivariate interpolation principle; polynomial-level symmetry
   on all of `ℚ²`; the sum-of-composition-polynomials assembly) is
   kernel-checked in the substrate.

The sole `pending`-substrate obligation is `Ed_decomposition` — the
paper's `Lemma 8.5` proof body (frozen-contraction bijection between
`d`-extremum height functions and `(type C, run-length vector)` pairs).
Check with `#print axioms OrigamiCone.Sequel.Ed_thm_poly_unconditional`.
-/

namespace OrigamiCone.Sequel

open Polynomial

/-- **Uniform onset** (paper `main.tex`, `Lemma 8.5`, `lem:uniform`).

For every `d`, there is a threshold `N_d`, depending only on `d`, such
that for every `m ≥ 2` the row function `Ed d m ·` agrees on
`{n ≥ N_d}` with a single polynomial in `n` of degree at most `d`.

**Now a THEOREM** (formerly an axiom in this module): proved from the
finer axiom `Ed_decomposition` (paper's `Lemma 8.5` proof body — the
frozen-contraction bijection) using the substrate modules 1–4.  See
`SequelUniformOnsetProof` for the derivation. -/
theorem Ed_uniform_onset (d : ℕ) :
    ∃ N : ℕ, ∀ m : ℕ, 2 ≤ m →
      ∃ p : Polynomial ℚ, p.natDegree ≤ d ∧
        ∀ n : ℕ, N ≤ n → (Ed d m n : ℚ) = p.eval (n : ℚ) :=
  Ed_uniform_onset_of_decomposition d

/-- **`thm:poly` at `Ed`, existence half, unconditional modulo
`Ed_decomposition`** (paper `Theorem thm:poly`).

For each `d`, there is a threshold `lo`, depending only on `d`, and
factored-Lagrange data `g, L : Fin (d + 1) → Polynomial ℚ` such that
`Ed d` agrees on `{a, b ≥ lo}` with a bivariate polynomial of degree
at most `d` in each variable, symmetric on all of `ℚ²`.

The paper's `Lemma 8.5` (`Ed_uniform_onset`, now proved from
`Ed_decomposition`) supplies per-axis
polynomiality along rows at an `m`-free threshold, and symmetry
(`Ed_symm`) transports it to columns at the same threshold; the
bivariate interpolation principle (`SequelInterp.interp_principle`,
kernel-checked) then produces the factored-Lagrange witness and
`SequelPolyAssembly.thm_poly_polynomial_symmetry` lifts symmetry to
`ℚ²`.  All steps of this reduction are kernel-checked; only
`Ed_decomposition` is axiomatic. -/
theorem Ed_thm_poly_unconditional (d : ℕ) :
    ∃ (lo : ℕ) (g L : Fin (d + 1) → Polynomial ℚ),
      1 ≤ lo ∧
      (∀ i, (g i).natDegree ≤ d) ∧ (∀ i, (L i).natDegree ≤ d) ∧
      (∀ a b, lo ≤ a → lo ≤ b →
        (Ed d a b : ℚ) = ∑ i, (g i).eval (b : ℚ) * (L i).eval (a : ℚ)) ∧
      (∀ a b : ℚ,
        (∑ i, (g i).eval b * (L i).eval a) =
          (∑ i, (g i).eval a * (L i).eval b)) := by
  -- Invoke the axiom once; the same `N` witnesses both `hrow` and `hcol`.
  obtain ⟨N, hpol⟩ := Ed_uniform_onset d
  -- Bump the threshold to `max N 2` so `Ed_symm`'s `1 ≤ m,n` hypotheses
  -- along `lo ≤ a` / `lo ≤ b` are discharged along with `1 ≤ lo`.
  let lo : ℕ := max N 2
  have hlo1 : 1 ≤ lo := by
    have : (2 : ℕ) ≤ lo := le_max_right _ _
    omega
  have hlo2 : ∀ x, lo ≤ x → 2 ≤ x := fun x hx =>
    le_trans (le_max_right _ _) hx
  have hloN : ∀ x, lo ≤ x → N ≤ x := fun x hx =>
    le_trans (le_max_left _ _) hx
  -- `hrow` at threshold `lo`, degree `d`.
  have hrow : ∀ a, lo ≤ a → ∃ p : Polynomial ℚ, p.natDegree ≤ d ∧
      ∀ b, lo ≤ b → (Ed d a b : ℚ) = p.eval (b : ℚ) := by
    intro a ha
    obtain ⟨p, hpdeg, hpev⟩ := hpol a (hlo2 a ha)
    exact ⟨p, hpdeg, fun b hb => hpev b (hloN b hb)⟩
  -- `hcol` at threshold `lo`, degree `d`, via `Ed_symm`: the column
  -- function `a ↦ Ed d a b` at fixed `b` equals `a ↦ Ed d b a` on
  -- `{a ≥ lo}` (both `a, b ≥ 2` from `lo ≥ 2`), which is the row
  -- function at `m = b` supplied by the axiom.
  have hcol : ∀ b, lo ≤ b → ∃ q : Polynomial ℚ, q.natDegree ≤ d ∧
      ∀ a, lo ≤ a → (Ed d a b : ℚ) = q.eval (a : ℚ) := by
    intro b hb
    obtain ⟨q, hqdeg, hqev⟩ := hpol b (hlo2 b hb)
    refine ⟨q, hqdeg, fun a ha => ?_⟩
    have ha2 : 1 ≤ a := le_trans hlo1 ha
    have hb2 : 1 ≤ b := le_trans hlo1 hb
    have hsym : Ed d a b = Ed d b a := Ed_symm d ha2 hb2
    calc (Ed d a b : ℚ)
        = (Ed d b a : ℚ) := by exact_mod_cast hsym
      _ = q.eval (a : ℚ) := hqev a (hloN a ha)
  -- Assemble via `Ed_thm_poly_of_perAxis` at `D := d`, `lo := lo`.
  obtain ⟨g, L, hgdeg, hLdeg, hagree, hsymm⟩ :=
    Ed_thm_poly_of_perAxis d d lo hlo1 hrow hcol
  exact ⟨lo, g, L, hlo1, hgdeg, hLdeg, hagree, hsymm⟩

end OrigamiCone.Sequel
