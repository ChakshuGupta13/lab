import OrigamiCone.SequelEdRatGF

/-!
# Pole location for `cnPolyGF m` (Task D, paper §8.1 step 7)

Bridge from the custom `revCharpoly` (formal-power-series form used by
`SequelRatGF.transfer_GF_rational`) to Mathlib's `Matrix.charpolyRev`
(polynomial form `det(1 - X · M.map C)`).  The identification gives the
paper's `lem:poles` denominator characterisation:

  `cnPolyGF m = P / (transferMatrix m).charpolyRev`,

i.e. the denominator of the rational generating function is
`det(I - z · T_m(x))`, whose roots (as functions of `x`) are the
reciprocals of the eigenvalues of `T_m(x)`.

## Contents

* `revCharpoly_eq_coe_charpolyReverse` — abstract identity connecting the
  `PowerSeries`-valued `revCharpoly` to the `Polynomial`-valued
  `T.charpoly.reverse`.  Just uses `Polynomial.coeff_reverse` and
  `Polynomial.revAt`.
* `revCharpoly_eq_coe_charpolyRev` — corollary via
  `Matrix.reverse_charpoly`: `revCharpoly T = (T.charpolyRev : R⟦z⟧)`.
* `cnPolyGF_denominator_charpolyRev` — Task D headline: the denominator of
  `cnPolyGF m` (viewed via the `SequelEdRatGF.cnPolyGF_rational` product)
  is exactly `(transferMatrix m).charpolyRev = det(1 - z · T_m(x))`.

The eigenvalue-inversion interpretation ("roots are reciprocals of nonzero
eigenvalues") is the standard consequence of `T.charpolyRev = ∏(1 - z λ_i)`
which is well-known when `T.charpoly = ∏(X - λ_i)`; we do not re-derive
it here.

All lemmas axiom-clean `[propext, Classical.choice, Quot.sound]`.  No
`sorry`, no `native_decide`.

Unblocks Task E (per-axis polynomiality discharge for `thm:poly`).
-/

namespace OrigamiCone.Sequel

open Matrix Polynomial PowerSeries

/-- **Bridge from custom `revCharpoly` to `Polynomial.reverse`**: the
formal-power-series `revCharpoly T` (with coefficients supported on
`[0, T.charpoly.natDegree]` in reversed order) is the coercion of the
polynomial `T.charpoly.reverse` into `PowerSeries R`.

Follows from `Polynomial.coeff_reverse` (which uses `Polynomial.revAt`)
plus vanishing of `charpoly.coeff` above the natDegree. -/
lemma revCharpoly_eq_coe_charpolyReverse
    {R : Type*} [CommRing R] {ι : Type*} [Fintype ι] [DecidableEq ι]
    (T : Matrix ι ι R) :
    revCharpoly T = (T.charpoly.reverse : PowerSeries R) := by
  unfold revCharpoly
  refine PowerSeries.ext fun n => ?_
  rw [PowerSeries.coeff_mk, Polynomial.coeff_coe, Polynomial.coeff_reverse,
      Polynomial.revAt]
  by_cases h : n ≤ T.charpoly.natDegree
  · rw [if_pos h]
    show _ = T.charpoly.coeff
      (if n ≤ T.charpoly.natDegree then T.charpoly.natDegree - n else n)
    rw [if_pos h]
  · push_neg at h
    rw [if_neg (not_le.mpr h)]
    show 0 = T.charpoly.coeff
      (if n ≤ T.charpoly.natDegree then T.charpoly.natDegree - n else n)
    rw [if_neg (not_le.mpr h)]
    exact (T.charpoly.coeff_eq_zero_of_natDegree_lt h).symm

/-- **Bridge to Mathlib's `charpolyRev`**: `revCharpoly T` (as
`PowerSeries R`) equals `T.charpolyRev = det(1 - X · T.map C)` coerced.

The identification uses `Matrix.reverse_charpoly : T.charpoly.reverse =
T.charpolyRev`. -/
lemma revCharpoly_eq_coe_charpolyRev
    {R : Type*} [CommRing R] {ι : Type*} [Fintype ι] [DecidableEq ι]
    (T : Matrix ι ι R) :
    revCharpoly T = (T.charpolyRev : PowerSeries R) := by
  rw [revCharpoly_eq_coe_charpolyReverse, T.reverse_charpoly]

/-- **Task D headline**: the denominator of the rational GF `cnPolyGF m` is
the coercion of `(transferMatrix m).charpolyRev`, whose value in the paper
is `det(1 - z · T_m(x))`.

For every `n ≥ (transferMatrix m).charpoly.natDegree`, the coefficient of
`z^n` in `(transferMatrix m).charpolyRev · cnPolyGF m` vanishes — i.e.,
the product is a polynomial of degree strictly less than the matrix's
characteristic-polynomial degree, so `cnPolyGF m` has denominator
`(transferMatrix m).charpolyRev` in the standard rational form. -/
theorem cnPolyGF_denominator_charpolyRev
    (m : ℕ) (hm : 0 < m) (n : ℕ)
    (hn : (transferMatrix m).charpoly.natDegree ≤ n) :
    (PowerSeries.coeff (R := Polynomial ℤ) n)
      (((transferMatrix m).charpolyRev : PowerSeries (Polynomial ℤ)) * cnPolyGF m)
      = 0 := by
  rw [← revCharpoly_eq_coe_charpolyRev]
  exact cnPolyGF_rational m hm n hn

end OrigamiCone.Sequel
