import OrigamiCone.SequelEdMatrixPower
import OrigamiCone.SequelRatGF

/-!
# Rational generating function for `cnPoly` (Task C, paper §8.1 step 6)

Instantiation of the abstract `SequelRatGF.transfer_GF_rational` identity at
the concrete quadruple

  `T := transferMatrix m,   u := leftBdyVecNorm m,   v := rightBdyVec m`.

Combined with Task B's `cnPoly_eq_matrix_power`

  `cnPoly m (n + 2) = leftBdyVecNorm m ⬝ᵥ ((transferMatrix m)^n *ᵥ rightBdyVec m)`

this yields the paper's `lem:ratGF` conclusion: the generating function

  `∑_n c_{m,n+2}(x) z^n`

is rational in `z` — its product with the reverse characteristic polynomial
of `T_m(x)` vanishes at every coefficient beyond `T_m.charpoly.natDegree`.

Equivalently, `cnPolyGF m = P / revCharpoly (T_m)` in `Polynomial ℤ[x][[z]]`
for some polynomial `P` of degree strictly less than
`(T_m).charpoly.natDegree` — the standard rational form for a linear-recurrent
sequence over `Polynomial ℤ[x]`.

## Contents

* `cnPolyGF m` — the paper's `∑_n c_{m,n+2}(x) z^n`, as a formal power series.
* `cnPolyGF_eq_transferGF` — bridge to the abstract `transferGF` API of
  `SequelRatGF`.
* `cnPolyGF_rational` — headline: `revCharpoly (T_m) * cnPolyGF m` has zero
  coefficient at every `n ≥ (T_m).charpoly.natDegree`.

All lemmas axiom-clean `[propext, Classical.choice, Quot.sound]`.  No `sorry`,
no `native_decide`.

Unblocks Task D (pole location via `T_0` spectrum), Task E (`thm:poly` step
discharge).
-/

namespace OrigamiCone.Sequel

open Matrix Polynomial PowerSeries

/-- **The paper's `∑_n c_{m,n+2}(x) z^n` power series**, shifted so `n = 0`
corresponds to the two-column base grid.  Formally, `cnPolyGF m` is the
`Polynomial ℤ`-valued formal power series in `z` whose `n`-th coefficient is
`cnPoly m (n + 2)`. -/
noncomputable def cnPolyGF (m : ℕ) : PowerSeries (Polynomial ℤ) :=
  PowerSeries.mk fun n => cnPoly m (n + 2)

/-- **Bridge**: `cnPolyGF m` equals the abstract `transferGF` at
`(leftBdyVecNorm m, rightBdyVec m, transferMatrix m)`.

Follows pointwise from B.3's `cnPoly_eq_matrix_power`: the `n`-th coefficient
of `cnPolyGF m` is `cnPoly m (n + 2)` by definition, and the `n`-th
coefficient of the RHS is `leftBdyVecNorm m ⬝ᵥ ((transferMatrix m)^n *ᵥ
rightBdyVec m)`, which equals `cnPoly m (n + 2)` by
`cnPoly_eq_matrix_power`. -/
lemma cnPolyGF_eq_transferGF (m : ℕ) (hm : 0 < m) :
    cnPolyGF m
      = transferGF (leftBdyVecNorm m) (rightBdyVec m) (transferMatrix m) := by
  unfold cnPolyGF transferGF
  refine PowerSeries.ext fun n => ?_
  rw [PowerSeries.coeff_mk, PowerSeries.coeff_mk]
  exact cnPoly_eq_matrix_power m n hm

/-- **Task C headline: rationality of `∑_n c_{m,n+2}(x) z^n`** (paper's
`lem:ratGF` conclusion).

For `n ≥ (transferMatrix m).charpoly.natDegree`, the coefficient of `z^n` in
`revCharpoly (transferMatrix m) * cnPolyGF m` vanishes.  Equivalently, the
product is a polynomial of degree strictly less than the transfer matrix's
characteristic-polynomial degree, so `cnPolyGF m` is a rational function of
`z` with denominator `revCharpoly (transferMatrix m)`.

Proof: bridge to `transferGF` via `cnPolyGF_eq_transferGF`, then apply the
abstract `SequelRatGF.transfer_GF_rational`. -/
theorem cnPolyGF_rational (m : ℕ) (hm : 0 < m) (n : ℕ)
    (hn : (transferMatrix m).charpoly.natDegree ≤ n) :
    (PowerSeries.coeff (R := Polynomial ℤ) n)
        (revCharpoly (transferMatrix m) * cnPolyGF m) = 0 := by
  rw [cnPolyGF_eq_transferGF m hm]
  exact transfer_GF_rational (transferMatrix m) (leftBdyVecNorm m) (rightBdyVec m) n hn

end OrigamiCone.Sequel
