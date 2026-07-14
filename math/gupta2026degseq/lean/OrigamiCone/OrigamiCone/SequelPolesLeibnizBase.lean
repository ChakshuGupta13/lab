import Mathlib

/-!
# Sequel: PowerSeries Leibniz identification — base case + form definition

Begin the formalisation of step 3 of the paper's `lem:poles` chain — the
PowerSeries Leibniz identification linking the abstract `d`-fold convolutional
sum `RseqMat T A d` (from `SequelPolesArbD`) to the `[x^d]` coefficient of the
matrix-coefficient polynomial `(T + x·A)^N`.

This module is the **first installment** of a multi-turn campaign:

* **This module (base)**: definitions (`Tmx`, `leibnizFactor`, `LeibnizForm`),
  basic coefficient and degree bounds for `Tmx`, the `d = 0` matrix
  coefficient identity, the `d = 0` Leibniz form identity, the matched
  identification at `d = 0`, and the vacuous case `N < d`.
* **Next turn**: the inductive step on `N`, showing
  `((Tmx)^N).coeff d = LeibnizForm T A d N` for general `d` by induction on
  `N` (uses `coeff_mul` + antidiagonal split into "current factor is `T`" vs
  "current factor is `A`").
* **Turn after**: the identification `LeibnizForm T A d N = RseqMat T A d (N - d)`
  by induction on `d` matching the recursive convolutional structure.

## Definitions

* `Tmx T A := C T + X · C A` is the matrix-coefficient polynomial `T + x A`
  in `Polynomial (Matrix ι ι R)`.
* `leibnizFactor T A d g`, for `g : Fin (d+1) → ℕ`, is the ordered product
  `T^{g 0} · A · T^{g 1} · A · ⋯ · A · T^{g d}` (the first `d` blocks
  carry an `A`-multiplier, the last is just `T^{g d}`). Built via
  `List.ofFn` + `List.prod` because matrix multiplication is non-commutative
  (so `Finset.prod`, which requires `CommMonoid`, is unavailable).
* `LeibnizForm T A d N` is the sum of `leibnizFactor T A d g` over all
  compositions `g : Fin (d+1) → ℕ` of `N - d` (i.e., `∑ g i = N - d`),
  with the convention that `LeibnizForm = 0` when `N < d` (no valid
  composition). The total length of each `leibnizFactor` is
  `d (A's) + ∑ g i = d + (N - d) = N`, matching `(Tmx)^N`.

## Theorems (this turn)

* `Tmx_coeff_zero`, `Tmx_coeff_one`, `Tmx_coeff_higher` : `Tmx` is degree-≤-1.
* `Tmx_natDegree_le_one` : matches the coefficient lemmas at the degree level.
* `Tmx_pow_natDegree_le` : `((Tmx)^N).natDegree ≤ N`. Induction on `N`.
* `Tmx_pow_coeff_vacuous` : `((Tmx)^N).coeff d = 0` for `N < d`. Direct from
  the degree bound.
* `leibnizFactor_zero` : at `d = 0` the factor collapses to `T^{g 0}`.
* `LeibnizForm_zero` : `LeibnizForm T A 0 N = T^N`.
* `Tmx_pow_coeff_zero` : `((Tmx)^N).coeff 0 = T^N`. Induction on `N` using
  `coeff_mul` + `antidiagonal_zero`.
* `Tmx_pow_coeff_eq_LeibnizForm_at_zero` (**main d=0 identification**) :
  `((Tmx)^N).coeff 0 = LeibnizForm T A 0 N`. Direct from the two preceding
  lemmas. Combined with `Tmx_pow_coeff_vacuous` and `LeibnizForm`'s `N < d`
  branch, this also handles the `d ≥ 1, N < d` vacuous case at the level of
  matching both sides to `0`.
* `Tmx_pow_coeff_eq_LeibnizForm_vacuous` (**vacuous d ≥ 1, N < d identification**) :
  for `N < d`, `((Tmx)^N).coeff d = LeibnizForm T A d N` because both sides
  collapse to `0` (the LHS by the degree bound, the RHS by the `if`-branch).

## Scope

* This is the **base layer** of the PowerSeries Leibniz identification. The
  inductive step on `N` for general `d` (turn 2) and the matching with
  `RseqMat T A d` (turn 3) are not formalised here.
* Per the discipline: imports `Mathlib` only (no Sequel imports yet — the
  matching with `RseqMat` comes only in the final turn).
* NOT added to root aggregator `OrigamiCone.lean`.

No `sorry`; check with `#print axioms
OrigamiCone.Sequel.Tmx_pow_coeff_eq_LeibnizForm_at_zero`.
-/

namespace OrigamiCone.Sequel

open Polynomial Matrix Finset

variable {R : Type*} [CommRing R] {ι : Type*} [Fintype ι] [DecidableEq ι]

/-- **Matrix-coefficient polynomial `T + x·A`**, the building block of the
Leibniz identification. Lives in `Polynomial (Matrix ι ι R)` (matrix-valued
coefficients). Its `N`-th power has `[x^d]` coefficient equal to the d-fold
convolutional sum of `T` and `A` factors. -/
noncomputable def Tmx (T A : Matrix ι ι R) : Polynomial (Matrix ι ι R) :=
  C T + Polynomial.X * C A

/-- The `0`-coefficient of `Tmx T A` is `T` (only `C T` contributes). -/
lemma Tmx_coeff_zero (T A : Matrix ι ι R) : (Tmx T A).coeff 0 = T := by
  show (C T + Polynomial.X * C A).coeff 0 = T
  rw [Polynomial.coeff_add]; simp

/-- The `1`-coefficient of `Tmx T A` is `A` (only `X * C A` contributes). -/
lemma Tmx_coeff_one (T A : Matrix ι ι R) : (Tmx T A).coeff 1 = A := by
  show (C T + Polynomial.X * C A).coeff 1 = A
  rw [Polynomial.coeff_add]; simp

/-- All coefficients of `Tmx T A` beyond degree `1` are zero. -/
lemma Tmx_coeff_higher (T A : Matrix ι ι R) (i : ℕ) (hi : 2 ≤ i) :
    (Tmx T A).coeff i = 0 := by
  show (C T + Polynomial.X * C A).coeff i = 0
  rcases Nat.exists_eq_add_of_le (show 1 ≤ i from by omega) with ⟨j, hj⟩
  subst hj
  rw [show 1 + j = j + 1 from by omega]
  rw [Polynomial.coeff_add, Polynomial.coeff_C]
  simp [Polynomial.coeff_C, show j ≠ 0 from by omega]

/-- `Tmx T A` has `natDegree ≤ 1`. Used downstream for the degree bound on
`(Tmx T A)^N`. -/
lemma Tmx_natDegree_le_one (T A : Matrix ι ι R) : (Tmx T A).natDegree ≤ 1 := by
  unfold Tmx
  have hC : (C T : Polynomial (Matrix ι ι R)).natDegree = 0 :=
    Polynomial.natDegree_C T
  have hXA : ((Polynomial.X : Polynomial (Matrix ι ι R)) * C A).natDegree ≤ 1 := by
    refine le_trans Polynomial.natDegree_mul_le ?_
    have hX : (Polynomial.X : Polynomial (Matrix ι ι R)).natDegree ≤ 1 :=
      Polynomial.natDegree_X_le
    have hCa : (C A : Polynomial (Matrix ι ι R)).natDegree = 0 :=
      Polynomial.natDegree_C A
    omega
  refine le_trans (Polynomial.natDegree_add_le _ _) ?_
  omega

/-- `((Tmx T A)^N).natDegree ≤ N`. Induction on `N` using
`Polynomial.natDegree_mul_le` + `Tmx_natDegree_le_one`. -/
lemma Tmx_pow_natDegree_le (T A : Matrix ι ι R) (N : ℕ) :
    ((Tmx T A) ^ N).natDegree ≤ N := by
  induction N with
  | zero => simp
  | succ N IH =>
    rw [pow_succ]
    refine le_trans Polynomial.natDegree_mul_le ?_
    have h := Tmx_natDegree_le_one T A
    omega

/-- **Vacuous case**: for `N < d`, the `d`-coefficient of `(Tmx T A)^N` is
zero. The `d`-fold convolutional sum needs at least `d` factors of `Tmx`
to provide `d` `A`'s, so fewer factors give nothing. Direct from
`Polynomial.coeff_eq_zero_of_natDegree_lt` + `Tmx_pow_natDegree_le`. -/
lemma Tmx_pow_coeff_vacuous (T A : Matrix ι ι R) (d N : ℕ) (h : N < d) :
    ((Tmx T A) ^ N).coeff d = 0 := by
  apply Polynomial.coeff_eq_zero_of_natDegree_lt
  have := Tmx_pow_natDegree_le T A N
  omega

/-- **Leibniz factor**: for a composition `g : Fin (d+1) → ℕ`, build the
ordered product `T^{g 0} · A · T^{g 1} · A · ⋯ · A · T^{g d}`. Uses
`List.ofFn` + `List.prod` because matrix multiplication is non-commutative
(so `Finset.prod` over `Fin (d+1)`, which requires `CommMonoid`, is
unavailable). The first `d` blocks carry an `A`-multiplier; the last is bare
`T^{g d}`. -/
noncomputable def leibnizFactor (T A : Matrix ι ι R) (d : ℕ)
    (g : Fin (d + 1) → ℕ) : Matrix ι ι R :=
  (List.ofFn (fun i : Fin (d + 1) =>
    if (i : ℕ) < d then T ^ g i * A else T ^ g i)).prod

/-- **Leibniz form**: sum of `leibnizFactor T A d g` over all compositions
of `N - d` into `d + 1` parts. With convention `LeibnizForm = 0` when
`N < d` (no valid composition; `Nat.antidiagonalTuple (d+1) (N - d)` would
collapse to `{const 0}` via `Nat`-truncation, giving the wrong term).

The total length of each `leibnizFactor` is `d (A's) + ∑ g i = d + (N - d)
= N`, matching the length of `(Tmx T A)^N`. -/
noncomputable def LeibnizForm (T A : Matrix ι ι R) (d N : ℕ) : Matrix ι ι R :=
  if N < d then 0 else
    ∑ g ∈ Finset.Nat.antidiagonalTuple (d + 1) (N - d), leibnizFactor T A d g

/-- At `d = 0`, the factor reduces to `T^{g 0}` (no `A` insertions). -/
lemma leibnizFactor_zero (T A : Matrix ι ι R) (g : Fin 1 → ℕ) :
    leibnizFactor T A 0 g = T ^ g 0 := by
  unfold leibnizFactor
  rw [List.ofFn_succ, List.ofFn_zero, List.prod_cons, List.prod_nil, mul_one]
  simp

/-- `LeibnizForm T A 0 N = T^N`. The unique composition of `N` into a single
part is `g = fun _ => N`, giving the single factor `T^N`. -/
lemma LeibnizForm_zero (T A : Matrix ι ι R) (N : ℕ) :
    LeibnizForm T A 0 N = T ^ N := by
  unfold LeibnizForm
  rw [if_neg (by omega : ¬ N < 0)]
  rw [show N - 0 = N from rfl]
  have hsingleton : Finset.Nat.antidiagonalTuple 1 N = {fun _ : Fin 1 => N} := by
    ext g
    rw [Finset.Nat.mem_antidiagonalTuple, Finset.mem_singleton]
    refine ⟨?_, ?_⟩
    · intro hg
      ext i
      have hi : i = 0 := Subsingleton.elim i 0
      subst hi
      rw [Fin.sum_univ_one] at hg
      exact hg
    · intro hg
      subst hg
      simp
  rw [hsingleton, Finset.sum_singleton, leibnizFactor_zero]

/-- `((Tmx T A)^N).coeff 0 = T^N`. Induction on `N` using `coeff_mul` and the
fact that the antidiagonal of `0` is a singleton. -/
lemma Tmx_pow_coeff_zero (T A : Matrix ι ι R) (N : ℕ) :
    ((Tmx T A) ^ N).coeff 0 = T ^ N := by
  induction N with
  | zero => simp
  | succ N IH =>
    rw [pow_succ, Polynomial.coeff_mul, Finset.antidiagonal_zero,
        Finset.sum_singleton]
    rw [IH, Tmx_coeff_zero, pow_succ]

/-- **Main `d = 0` identification** (this turn's headline):
`((Tmx T A)^N).coeff 0 = LeibnizForm T A 0 N`. Both sides equal `T^N`.

The general `d`-coefficient identification `((Tmx T A)^N).coeff d
= LeibnizForm T A d N` will be proved in the next turn by induction on `N`
using `coeff_mul` + antidiagonal split.

The further identification `LeibnizForm T A d N = RseqMat T A d (N - d)`
(matching the abstract `d`-fold convolutional sum from `SequelPolesArbD`)
will be proved in the turn after, by induction on `d` matching the recursive
convolutional structure. -/
theorem Tmx_pow_coeff_eq_LeibnizForm_at_zero (T A : Matrix ι ι R) (N : ℕ) :
    ((Tmx T A) ^ N).coeff 0 = LeibnizForm T A 0 N := by
  rw [Tmx_pow_coeff_zero, LeibnizForm_zero]

/-- **Vacuous-case identification**: for `N < d`, both `((Tmx T A)^N).coeff d`
and `LeibnizForm T A d N` are zero. This handles half of the general
identification; the `N ≥ d` non-vacuous case is the substantive content of
the next turn. -/
theorem Tmx_pow_coeff_eq_LeibnizForm_vacuous (T A : Matrix ι ι R) (d N : ℕ)
    (h : N < d) :
    ((Tmx T A) ^ N).coeff d = LeibnizForm T A d N := by
  rw [Tmx_pow_coeff_vacuous T A d N h]
  unfold LeibnizForm
  rw [if_pos h]

end OrigamiCone.Sequel
