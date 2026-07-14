import Mathlib

/-!
# Sequel meta-theorem: the separable case of the dimension bound (G2)

Standalone formalisation of the **proven** part of the dimension bound
(`Conjecture G2`) from the sequel paper

> *Degree-$d$ vertex counts of the $m \times n$ origami flip graph:
> a polynomial meta-theorem.*

The sequel reduces its meta-theorem to a single conjecture, the **dimension
bound** (`conj:G2`): for `d ≥ 5` and the balanced split, every *multi-edge* cell
of the parameter polytope has parametric dimension at most `d - 3`. The paper
proves this for all **separable** configurations (`Theorem thm:sepcase`); only
the non-separable residual is left open. This module formalises that proven
separable case.

## The mathematical core

A separable configuration factors as `E(i,j) = φ(i) + ψ(j)`, a pair of
one-dimensional `±1` walks (`Lemma lem:sep`). Record it by four positive
integers:

* `ρ`  — number of rows carrying a minimum   (rows of `φ`-minima),
* `γ`  — number of columns carrying a minimum (columns of `ψ`-minima),
* `ρ'` — number of rows carrying a maximum,
* `γ'` — number of columns carrying a maximum,

so the split is `a = ρ·γ` minima and `b = ρ'·γ'` maxima, of total degree
`d = a + b`. The family's contribution to the count has total degree
`D = (ρ + ρ' − 2) + (γ + γ' − 2)` (`Lemma lem:dimid`, a product of two
one-dimensional walk counts of degrees `ρ+ρ'−2` and `γ+γ'−2`).

The **Dimension Identity** (`dim_identity`/`dim_codim`) is
`(d − 2) − D = (ρ−1)(γ−1) + (ρ'−1)(γ'−1) ≥ 0`,
so `D ≤ d − 2` always (`dim_le`), and equality forces both products to vanish.
The **Separable Case** theorem (`sep_dim_bound`, `thm:sepcase`) then shows that,
for `d ≥ 5`, a separable family of maximal degree `d − 2` must be *single-edge*
(all minima on one boundary row/column and all maxima on the opposite one); every
other separable family has degree `≤ d − 3`.

## Faithfulness / scope

* The `±1`-walk constraint `|ρ − ρ'| ≤ 1` (and `|γ − γ'| ≤ 1`) is taken as a
  hypothesis here. In the paper it is `Lemma lem:binom`: a `±1` walk with `a`
  strict local minima and `b` strict local maxima exists only when `|a − b| ≤ 1`.
  Formalising `lem:binom` itself (the run-length / composition bijection giving
  the count `(1+[a=b])C(n-2,d-2)`) is deferred; here we record the one
  consequence the separable-case proof uses.
* `D` is the *total degree* a separable family contributes (`Lemma lem:dimid`);
  the identification of `D` with the parametric dimension of the corresponding
  Ehrhart cell (via positivity of leading coefficients) is the surrounding
  Barvinok–Woods layer and is not re-derived here.
* `SingleEdge` encodes "all apexes on one side" in the separable language: minima
  and maxima collinear in rows (`ρ = ρ' = 1`, the top/bottom edges) or in columns
  (`γ = γ' = 1`, the left/right edges).

This is the arithmetic heart of `thm:sepcase`; the geometric content it rests on
(`lem:sep`, `lem:dimid`, `lem:binom`) is cited above and proved in the paper.

No `sorry`; check with `#print axioms sep_dim_bound`.
-/

namespace OrigamiCone.Sequel

/-- The **Dimension Identity** (`Lemma lem:dimid`), additive form. For a separable
configuration with `ρ, γ` minima rows/columns and `ρ', γ'` maxima rows/columns,
the degree `d = ρ·γ + ρ'·γ'` equals the contributed degree
`D = (ρ+ρ'−2) + (γ+γ'−2)` plus `2` plus the codimension
`(ρ−1)(γ−1) + (ρ'−1)(γ'−1)`. -/
lemma dim_identity (ρ γ ρ' γ' : ℕ)
    (hρ : 1 ≤ ρ) (hγ : 1 ≤ γ) (hρ' : 1 ≤ ρ') (hγ' : 1 ≤ γ') :
    ρ * γ + ρ' * γ'
      = ((ρ + ρ' - 2) + (γ + γ' - 2)) + 2
        + ((ρ - 1) * (γ - 1) + (ρ' - 1) * (γ' - 1)) := by
  have h2ρ : 2 ≤ ρ + ρ' := by omega
  have h2γ : 2 ≤ γ + γ' := by omega
  zify [hρ, hγ, hρ', hγ', h2ρ, h2γ]
  ring

/-- The **codimension form** of the Dimension Identity, exactly as stated in the
paper: `(d − 2) − D = (ρ−1)(γ−1) + (ρ'−1)(γ'−1)`. -/
lemma dim_codim (ρ γ ρ' γ' : ℕ)
    (hρ : 1 ≤ ρ) (hγ : 1 ≤ γ) (hρ' : 1 ≤ ρ') (hγ' : 1 ≤ γ') :
    (ρ * γ + ρ' * γ' - 2) - ((ρ + ρ' - 2) + (γ + γ' - 2))
      = (ρ - 1) * (γ - 1) + (ρ' - 1) * (γ' - 1) := by
  have hid := dim_identity ρ γ ρ' γ' hρ hγ hρ' hγ'
  omega

/-- **Dimension bound (separable, always).** The contributed degree never exceeds
`d − 2`: a separable family has total degree at most `d − 2`. This is the `≥ 0`
half of the Dimension Identity. -/
lemma dim_le (ρ γ ρ' γ' : ℕ)
    (hρ : 1 ≤ ρ) (hγ : 1 ≤ γ) (hρ' : 1 ≤ ρ') (hγ' : 1 ≤ γ') :
    (ρ + ρ' - 2) + (γ + γ' - 2) ≤ ρ * γ + ρ' * γ' - 2 := by
  have hid := dim_identity ρ γ ρ' γ' hρ hγ hρ' hγ'
  omega

/-- A separable family is **single-edge** when its minima and maxima are both
row-collinear (`ρ = ρ' = 1`, top and bottom edges) or both column-collinear
(`γ = γ' = 1`, left and right edges). Argument order matches the paper: `ρ, γ,
ρ', γ'`. -/
def SingleEdge (ρ γ ρ' γ' : ℕ) : Prop := (ρ = 1 ∧ ρ' = 1) ∨ (γ = 1 ∧ γ' = 1)

/-- **Separable case of the dimension bound** (`Theorem thm:sepcase`). For
`d = ρ·γ + ρ'·γ' ≥ 5`, a separable family attaining the maximal degree
`D = d − 2` is single-edge. Equivalently (with `dim_le`), every non-single-edge
separable family has total degree at most `d − 3`, so `Conjecture G2` holds for
all separable configurations.

The `±1`-walk hypotheses `hrow : |ρ − ρ'| ≤ 1` and `hcol : |γ − γ'| ≤ 1` are the
consequence of `Lemma lem:binom` used in the paper's proof. -/
theorem sep_dim_bound (ρ γ ρ' γ' : ℕ)
    (hρ : 1 ≤ ρ) (hγ : 1 ≤ γ) (hρ' : 1 ≤ ρ') (hγ' : 1 ≤ γ')
    (hrow : ρ ≤ ρ' + 1 ∧ ρ' ≤ ρ + 1)
    (hcol : γ ≤ γ' + 1 ∧ γ' ≤ γ + 1)
    (hd : 5 ≤ ρ * γ + ρ' * γ')
    (hmax : (ρ + ρ' - 2) + (γ + γ' - 2) = ρ * γ + ρ' * γ' - 2) :
    SingleEdge ρ γ ρ' γ' := by
  -- The Dimension Identity turns maximal degree into a vanishing codimension.
  have hid := dim_identity ρ γ ρ' γ' hρ hγ hρ' hγ'
  have hP1 : (ρ - 1) * (γ - 1) = 0 := by omega
  have hP2 : (ρ' - 1) * (γ' - 1) = 0 := by omega
  rcases Nat.mul_eq_zero.mp hP1 with h1 | h1
  · -- ρ = 1 : minima share one row.
    have hρ1 : ρ = 1 := by omega
    rcases Nat.mul_eq_zero.mp hP2 with h2 | h2
    · -- ρ' = 1 : maxima share one row too — single-edge (top/bottom).
      exact Or.inl ⟨hρ1, by omega⟩
    · -- γ' = 1, ρ' ≥ 2 : the walk constraints cap d ≤ 4, contradicting d ≥ 5.
      exfalso
      have hγ'1 : γ' = 1 := by omega
      subst hρ1; subst hγ'1
      simp only [one_mul, mul_one] at hd
      omega
  · -- γ = 1 : minima share one column.
    have hγ1 : γ = 1 := by omega
    rcases Nat.mul_eq_zero.mp hP2 with h2 | h2
    · -- ρ' = 1, γ ≥ ... : the walk constraints cap d ≤ 4, contradicting d ≥ 5.
      exfalso
      have hρ'1 : ρ' = 1 := by omega
      subst hγ1; subst hρ'1
      simp only [one_mul, mul_one] at hd
      omega
    · -- γ' = 1 : maxima share one column too — single-edge (left/right).
      exact Or.inr ⟨hγ1, by omega⟩

end OrigamiCone.Sequel
