import Mathlib
import OrigamiCone.SequelPolesArbDScalar

/-!
# Sequel: arbitrary-`d` `lem:poles` (polynomial-power recurrence form)

Standalone formalisation of the **polynomial-power-recurrence form** of the
scalar sandwich recurrence from `SequelPolesArbDScalar`. The `d = 0` case (so
the power is `d + 1 = 1`) is the standard linear-recurrence statement with
characteristic polynomial `T.charpoly^(d+1)`:

    `∑ k ∈ range ((T.charpoly^(d+1)).natDegree + 1),
      (T.charpoly^(d+1)).coeff k * scalarSeq u (RseqMat T A d) v (n+k) = 0`.

This is the canonical form needed to bridge to rational-GF results: a scalar
sequence satisfying a linear recurrence with characteristic polynomial `q`
has generating function with denominator dividing the reciprocal of `q`. The
paper's `lem:poles` then concludes that the GF poles lie among the reciprocal
roots of `T.charpoly^(d+1)` — i.e. reciprocal eigenvalues of `T`, with pole
order at each `1/λ` bounded by `μ(λ) · (d + 1)` where `μ(λ)` is the
algebraic multiplicity of `λ` in `T.charpoly` (which collapses to the simple
bound `d + 1` when every eigenvalue is simple).

## How the bridge works

The scalar sandwich recurrence proved in `SequelPolesArbDScalar` is in the
*iterated* form:

    `scalarCharActIter T.charpoly (scalarSeq u (RseqMat T A d) v) (d+1) n = 0`.

To get the standard linear-recurrence form, we need

    `scalarCharActIter p c d = scalarCharAct (p^d) c`     (as functions).

This is a polynomial-arithmetic identity: applying the charpoly action `d`
times is the same as applying the `d`-th power's charpoly action once. The
key step is the **composition lemma**:

    `scalarCharAct (p * q) c = scalarCharAct p (scalarCharAct q c)`.

Together with the recursive definition of `scalarCharActIter` and the
identity `p^(d+1) = p * p^d`, the iter form unwinds to the power form by
induction on `d`.

## Composition lemma proof strategy

The composition lemma is proved by induction on `p` via
`Polynomial.induction_on'`:

* **Additive case** `p = p1 + p2`: linearity in `p` (`scalarCharAct_add`) +
  distributivity of multiplication over addition.
* **Monomial case** `p = monomial i a`: decompose `monomial i a = C a * X^i`,
  then apply the `C_mul` factoring lemma and the `X^i * q` shift lemma:
  * `scalarCharAct (C a * p) c n = a * scalarCharAct p c n`
    (`scalarCharAct_C_mul`).
  * `scalarCharAct (X^i * q) c n = scalarCharAct q c (n + i)`
    (`scalarCharAct_X_pow_mul`).

The `X^i * q` lemma is the heart of the proof: it uses the coefficient
shift `(X^i * q).coeff (k + i) = q.coeff k` (Mathlib's
`Polynomial.coeff_X_pow_mul`) to reindex the sum.

## Theorems

* `scalarCharAct_eq_sum_range` : range-extension — the sum over
  `range (p.natDegree + 1)` equals the sum over any larger range `N ≥
  p.natDegree + 1` (extra coefficients are zero).
* `scalarCharAct_add` : linearity in `p` (additive).
* `scalarCharAct_monomial` : `scalarCharAct (monomial k a) c n = a * c (n+k)`.
* `scalarCharAct_C_mul` : factor `C a` out: `scalarCharAct (C a * p) c n = a *
  scalarCharAct p c n`.
* `scalarCharAct_X_pow_mul` : `X^i` shift: `scalarCharAct (X^i * q) c n =
  scalarCharAct q c (n + i)`.
* `scalarCharAct_monomial_mul` : combined: `scalarCharAct (monomial i a * q) c n
  = a * scalarCharAct q c (n + i)`.
* `scalarCharAct_mul` (**composition**): `scalarCharAct (p * q) c n =
  scalarCharAct p (scalarCharAct q c) n`.
* `scalarCharActIter_eq_pow` : `scalarCharActIter p c d n = scalarCharAct (p^d)
  c n`. Induction on `d` using `scalarCharAct_mul` at the step.
* `RseqMat_sandwich_polypow_recurrence` (**main**): the polynomial-power form
  of the scalar sandwich recurrence —
  `scalarCharAct (T.charpoly^(d+1)) (scalarSeq u (RseqMat T A d) v) n = 0`.
  Direct corollary: rewrite via `scalarCharActIter_eq_pow` (backwards) and
  apply `RseqMat_sandwich_recurrence`.

## Scope

* The composition lemma + the iter-to-power identity + the corollary are
  proved end-to-end (no `sorry`).
* The rational-GF bridge from a linear recurrence to a rational generating
  function with denominator dividing the reciprocal of the characteristic
  polynomial is **standard** from this recurrence form (via `LinearRecurrence`-
  style arguments); not formalised here. Step 4 in `SequelPolesArbDScalar`'s
  docstring "chain of reasoning" is now closed at the recurrence-statement
  level; the rational-GF step itself remains standard / downstream.
* The PowerSeries Leibniz identification (step 3 — `[x^d]` of the resolvent
  sandwich equals the scalar sandwich at the `+d`-shifted exponent) is still
  disclaimed.
* **Discipline deviation**: like `SequelPolesArbD` and `SequelPolesArbDScalar`,
  this module imports a Sequel module (`OrigamiCone.SequelPolesArbDScalar`,
  which transitively imports the rest of the chain). Same DRY justification
  and no parallel-session edit risk.
* Per the discipline: NOT added to root aggregator `OrigamiCone.lean`.

No `sorry`; check with `#print axioms
OrigamiCone.Sequel.RseqMat_sandwich_polypow_recurrence`.
-/

namespace OrigamiCone.Sequel

open Polynomial Finset Matrix

variable {R : Type*} [CommRing R]

/-- **Range extension**: `scalarCharAct p c n = ∑ k ∈ range N, p.coeff k * c (n+k)`
for any `N > p.natDegree`. Extra terms have coefficient zero
(`Polynomial.coeff_eq_zero_of_natDegree_lt`). -/
lemma scalarCharAct_eq_sum_range (p : Polynomial R) (c : ℕ → R) (n N : ℕ)
    (hN : p.natDegree < N) :
    scalarCharAct p c n = ∑ k ∈ Finset.range N, p.coeff k * c (n + k) := by
  unfold scalarCharAct
  refine Finset.sum_subset ?_ ?_
  · intro x hx; rw [Finset.mem_range] at hx ⊢; omega
  · intro x _ hxNot
    rw [Finset.mem_range] at hxNot
    have hxlt : p.natDegree < x := by omega
    rw [Polynomial.coeff_eq_zero_of_natDegree_lt hxlt, zero_mul]

/-- **Linearity** (additive) of `scalarCharAct` in `p`. -/
lemma scalarCharAct_add (p q : Polynomial R) (c : ℕ → R) (n : ℕ) :
    scalarCharAct (p + q) c n = scalarCharAct p c n + scalarCharAct q c n := by
  set N := max p.natDegree q.natDegree + 1 with hN_def
  have hpq : (p + q).natDegree ≤ max p.natDegree q.natDegree :=
    Polynomial.natDegree_add_le _ _
  rw [scalarCharAct_eq_sum_range (p + q) c n N (by omega)]
  rw [scalarCharAct_eq_sum_range p c n N (by
    have := le_max_left p.natDegree q.natDegree; omega)]
  rw [scalarCharAct_eq_sum_range q c n N (by
    have := le_max_right p.natDegree q.natDegree; omega)]
  rw [← Finset.sum_add_distrib]
  refine Finset.sum_congr rfl fun k _ => ?_
  rw [Polynomial.coeff_add, add_mul]

/-- **Monomial evaluation**: `scalarCharAct (monomial k a) c n = a * c (n+k)`. -/
lemma scalarCharAct_monomial (k : ℕ) (a : R) (c : ℕ → R) (n : ℕ) :
    scalarCharAct ((Polynomial.monomial k a : Polynomial R)) c n = a * c (n + k) := by
  rcases eq_or_ne a 0 with ha | ha
  · subst ha
    simp [scalarCharAct, Polynomial.monomial_zero_right]
  · have hND : (Polynomial.monomial k a).natDegree = k :=
      Polynomial.natDegree_monomial_eq k ha
    rw [scalarCharAct_eq_sum_range _ _ _ (k + 1) (by rw [hND]; omega)]
    rw [Finset.sum_eq_single k]
    · rw [Polynomial.coeff_monomial]; simp
    · intro b _ hbk
      rw [Polynomial.coeff_monomial]
      simp [hbk.symm]
    · intro hkNot; exfalso; apply hkNot; simp

/-- **Constant factor**: `scalarCharAct (C a * p) c n = a * scalarCharAct p c n`.
The constant `a` factors out cleanly. -/
lemma scalarCharAct_C_mul (a : R) (p : Polynomial R) (c : ℕ → R) (n : ℕ) :
    scalarCharAct ((C a : Polynomial R) * p) c n = a * scalarCharAct p c n := by
  have hND : ((C a : Polynomial R) * p).natDegree ≤ p.natDegree :=
    Polynomial.natDegree_C_mul_le a p
  rw [scalarCharAct_eq_sum_range (C a * p) c n (p.natDegree + 1) (by omega)]
  unfold scalarCharAct
  rw [Finset.mul_sum]
  refine Finset.sum_congr rfl fun k _ => ?_
  rw [Polynomial.coeff_C_mul]; ring

/-- **`X^i` shift**: `scalarCharAct (X^i * q) c n = scalarCharAct q c (n + i)`.
Reindexes the sum via the coefficient identity
`(X^i * q).coeff (k + i) = q.coeff k`. -/
lemma scalarCharAct_X_pow_mul (q : Polynomial R) (i : ℕ) (c : ℕ → R) (n : ℕ) :
    scalarCharAct ((X : Polynomial R)^i * q) c n
      = scalarCharAct q c (n + i) := by
  have hbound : ((X : Polynomial R)^i * q).natDegree < i + q.natDegree + 1 := by
    have h1 : ((X : Polynomial R)^i * q).natDegree ≤
        ((X : Polynomial R)^i).natDegree + q.natDegree :=
      Polynomial.natDegree_mul_le
    have h2 : ((X : Polynomial R)^i).natDegree ≤ i :=
      Polynomial.natDegree_X_pow_le i
    omega
  rw [scalarCharAct_eq_sum_range _ c n (i + q.natDegree + 1) hbound]
  unfold scalarCharAct
  have hsplit :
      (∑ k ∈ Finset.range (i + q.natDegree + 1),
          ((X : Polynomial R)^i * q).coeff k * c (n + k))
        = (∑ k ∈ Finset.range i, ((X : Polynomial R)^i * q).coeff k * c (n + k))
          + (∑ k ∈ Finset.range (q.natDegree + 1),
              ((X : Polynomial R)^i * q).coeff (i + k) * c (n + (i + k))) := by
    rw [show i + q.natDegree + 1 = i + (q.natDegree + 1) from by ring]
    exact Finset.sum_range_add _ _ _
  rw [hsplit]
  have hzero :
      (∑ k ∈ Finset.range i,
          ((X : Polynomial R)^i * q).coeff k * c (n + k)) = 0 := by
    refine Finset.sum_eq_zero ?_
    intro k hk
    rw [Finset.mem_range] at hk
    rw [Polynomial.coeff_X_pow_mul']
    simp [show ¬ (i ≤ k) from by omega]
  rw [hzero, zero_add]
  refine Finset.sum_congr rfl fun k _ => ?_
  rw [show i + k = k + i from Nat.add_comm i k]
  rw [Polynomial.coeff_X_pow_mul q i k]
  congr 2; omega

/-- **Monomial-times-polynomial factor**: combines `C_mul` and `X^i shift`. -/
lemma scalarCharAct_monomial_mul (i : ℕ) (a : R) (q : Polynomial R) (c : ℕ → R)
    (n : ℕ) :
    scalarCharAct ((Polynomial.monomial i a : Polynomial R) * q) c n
      = a * scalarCharAct q c (n + i) := by
  rw [show (Polynomial.monomial i a : Polynomial R) * q = C a * (X^i * q) from by
    rw [show (Polynomial.monomial i a : Polynomial R) = C a * X^i from
        Polynomial.C_mul_X_pow_eq_monomial.symm]
    ring]
  rw [scalarCharAct_C_mul, scalarCharAct_X_pow_mul]

/-- **Composition lemma** (the key result):
`scalarCharAct (p * q) c n = scalarCharAct p (scalarCharAct q c) n`. Proved by
induction on `p` via `Polynomial.induction_on'`: the additive case is
linearity (`scalarCharAct_add`) plus distributivity; the monomial case
combines `scalarCharAct_monomial_mul` and `scalarCharAct_monomial`. -/
lemma scalarCharAct_mul (p q : Polynomial R) (c : ℕ → R) (n : ℕ) :
    scalarCharAct (p * q) c n = scalarCharAct p (scalarCharAct q c) n := by
  induction p using Polynomial.induction_on' with
  | add p1 p2 hp1 hp2 =>
    rw [add_mul, scalarCharAct_add, scalarCharAct_add, hp1, hp2]
  | monomial i a =>
    rw [scalarCharAct_monomial_mul, scalarCharAct_monomial]

/-- **Iter-to-power identity**: `scalarCharActIter p c d n = scalarCharAct (p^d)
c n`. Proved by induction on `d` using `scalarCharAct_mul` at the step
(`pow_succ' p d : p^(d+1) = p * p^d`). -/
lemma scalarCharActIter_eq_pow (p : Polynomial R) (c : ℕ → R) (d n : ℕ) :
    scalarCharActIter p c d n = scalarCharAct (p^d) c n := by
  induction d generalizing n with
  | zero =>
    show c n = scalarCharAct (p^0) c n
    rw [pow_zero]
    unfold scalarCharAct
    simp [Polynomial.natDegree_one]
  | succ d IH =>
    show scalarCharAct p (scalarCharActIter p c d) n
      = scalarCharAct (p^(d+1)) c n
    rw [pow_succ', scalarCharAct_mul]
    congr 1
    funext m
    exact IH m

variable {ι : Type*} [Fintype ι] [DecidableEq ι]

/-- **`p^{d+1}`-linear-recurrence form** of `lem:poles` at arbitrary `d`
(scalar sandwich version). The scalar sandwich `c(n) := u ⬝ᵥ (RseqMat T A d n)
*ᵥ v` satisfies the **standard linear-recurrence form** with characteristic
polynomial `T.charpoly^(d+1)`:

    `scalarCharAct (T.charpoly^(d+1)) c n = 0`     for all `n`,

which expands to
    `∑ k ∈ Finset.range ((T.charpoly^(d+1)).natDegree + 1),
        (T.charpoly^(d+1)).coeff k * c (n + k) = 0`.

This is the form connectable to standard rational-GF theory: a scalar
sequence satisfying a linear recurrence with characteristic polynomial `q`
has generating function with denominator dividing the reciprocal of `q`,
hence poles at reciprocal roots of `q`. Applied here: poles of the GF of
`c(n)` lie among reciprocal eigenvalues of `T`, with pole order at each
`1/λ` bounded by `μ(λ) · (d + 1)` (and simply by `d + 1` when every eigenvalue
is simple) — the paper's `lem:poles` conclusion. The rational-GF bridge
itself is standard from the recurrence and not re-formalised here.

Proved directly: rewrite via `scalarCharActIter_eq_pow` (backwards) and
apply `SequelPolesArbDScalar.RseqMat_sandwich_recurrence`. -/
theorem RseqMat_sandwich_polypow_recurrence (T A : Matrix ι ι R) (u v : ι → R)
    (d n : ℕ) :
    scalarCharAct (T.charpoly^(d+1)) (scalarSeq u (RseqMat T A d) v) n = 0 := by
  rw [← scalarCharActIter_eq_pow]
  exact RseqMat_sandwich_recurrence T A u v d n

end OrigamiCone.Sequel
