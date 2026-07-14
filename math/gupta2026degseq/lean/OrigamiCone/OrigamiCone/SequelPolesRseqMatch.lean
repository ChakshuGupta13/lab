import Mathlib
import OrigamiCone.SequelPolesLeibniz
import OrigamiCone.SequelPolesArbD

/-!
# Sequel: PowerSeries Leibniz identification — closing the campaign

Fifth and final installment (turn 3/3) of the multi-turn campaign formalising
step 3 of the paper's `lem:poles` chain. This module closes the matching

```
LeibnizForm T A d N = RseqMat T A d (N - d)        (for d ≤ N),
```

and consequently the full step-3 identity:

```
((T + x·A) ^ N).coeff d = RseqMat T A d (N - d)    (for d ≤ N),
```

connecting the matrix-coefficient `[x^d]` of `(T + x·A)^N` to the abstract
`d`-fold convolutional sum `RseqMat T A d (·)` from `SequelPolesArbD`.

## Campaign chain (turns 1–3)

* **Turn 1 (`SequelPolesLeibnizBase`, 9deb844)** : `Tmx`, `leibnizFactor`,
  `LeibnizForm` definitions + `d = 0` and `N < d` cases.
* **Turn 2a (`SequelPolesLeibnizStep`, 7c200ef)** : split lemma +
  polynomial-side step `Tmx_pow_succ_coeff_succ`.
* **Turn 2b (`SequelPolesLeibnizRec`, 471af16)** : Leibniz-side step
  `LeibnizForm_succ_succ`.
* **Turn 2c (`SequelPolesLeibniz`, 33eae62)** : matrix-coefficient closure
  `Tmx_pow_coeff_eq_LeibnizForm` (all `d, N`).
* **This module (turn 3)** : convolutional partition of `LeibnizForm`
  + matching identity with `RseqMat` (for `d ≤ N`) + capstone
  `Tmx_pow_coeff_eq_RseqMat`.

## Hypothesis `d ≤ N`

The identification `LeibnizForm T A d N = RseqMat T A d (N - d)` only holds
for `d ≤ N`. For `N < d`, the LHS is `0` (vacuous: no composition of `N - d`
into `d + 1` parts exists, by `LeibnizForm`'s `if N < d then 0`), but the
RHS `RseqMat T A d 0` = (by recursion) `A^d`, which is nonzero in general.
The campaign's matrix-coefficient identity
`((Tmx T A)^N).coeff d = LeibnizForm T A d N` (turn 2c) provides the
naturally vacuous LHS via the degree bound; the `RseqMat`-matched form is
the natural shape only in the `d ≤ N` regime.

## Theorems

* `LeibnizForm_succ_eq_conv` (intermediate): for `d + 1 ≤ N`,
  `LeibnizForm T A (d + 1) N = ∑ j ∈ range (N - d), T^j · A
  · LeibnizForm T A d (N - 1 - j)`. **Convolutional partition** of the
  Leibniz form along the head exponent `g 0`. Proved by induction on
  `k := N - (d + 1)` (peeling one `LeibnizForm_succ_succ` recurrence at
  each step, starting from `N = d + 1` where the inner
  `LeibnizForm T A (d + 1) d` is vacuously zero).
* `LeibnizForm_eq_RseqMat` (**main turn 3 result**): for `d ≤ N`,
  `LeibnizForm T A d N = RseqMat T A d (N - d)`. By induction on `d`; the
  step uses the convolutional partition + the recursive definition of
  `RseqMat` + the inductive hypothesis at each summand index.
* `Tmx_pow_coeff_eq_RseqMat` (**campaign capstone**): for `d ≤ N`,
  `((T + x·A) ^ N).coeff d = RseqMat T A d (N - d)`. Combines turn 2c's
  `Tmx_pow_coeff_eq_LeibnizForm` and this module's
  `LeibnizForm_eq_RseqMat`.

## Scope

* Imports `Mathlib`, `OrigamiCone.SequelPolesLeibniz` (transitively the
  full Leibniz chain), and `OrigamiCone.SequelPolesArbD` (for `RseqMat`
  and `convolveMat`). Same cross-Sequel-import discipline (same-campaign,
  no parallel-session race).
* No `sorry`. Axioms: `[propext, Classical.choice, Quot.sound]` baseline.
* NOT added to root aggregator `OrigamiCone.lean`.

Check axioms with
`#print axioms OrigamiCone.Sequel.Tmx_pow_coeff_eq_RseqMat`.
-/

namespace OrigamiCone.Sequel

open Polynomial Matrix Finset

variable {R : Type*} [CommRing R] {ι : Type*} [Fintype ι] [DecidableEq ι]

/-- **Convolutional partition of `LeibnizForm`**: for `d + 1 ≤ N`,
`LeibnizForm T A (d + 1) N` equals the convolutional sum

```
∑ j ∈ range (N - d), T^j · A · LeibnizForm T A d (N - 1 - j),
```

partitioning the antidiagonal-tuple sum by the head exponent `g 0`.

Proof: induction on `k := N - (d + 1)`. At `k = 0` (`N = d + 1`), the
single-index sum reduces to `A · LeibnizForm T A d d = A · A^d`, matching
the LHS `LeibnizForm T A (d + 1) (d + 1) = A^(d + 1)` via
`LeibnizForm_succ_succ` at `e := d, N := d` (where the `T·_` part vanishes
because `LeibnizForm T A (d + 1) d = 0` from the vacuous `N < d` branch).

For the inductive step `k → k + 1`, expand `LeibnizForm T A (d + 1)
(N + 1)` via `LeibnizForm_succ_succ`, apply the IH to the
`T · LeibnizForm T A (d + 1) N` summand, then realign indices: the
left sum gains a `T` factor (shifting `T^j → T^(j+1)`), and the `A
· LeibnizForm T A d N` summand becomes the new `j = 0` term. -/
theorem LeibnizForm_succ_eq_conv (T A : Matrix ι ι R) (d N : ℕ) (hN : d + 1 ≤ N) :
    LeibnizForm T A (d + 1) N
      = ∑ j ∈ Finset.range (N - d), T ^ j * A * LeibnizForm T A d (N - 1 - j) := by
  obtain ⟨k, hk⟩ : ∃ k, N = d + 1 + k := ⟨N - (d + 1), by omega⟩
  subst hk
  induction k with
  | zero =>
    simp only [Nat.add_zero]
    rw [show d + 1 - d = 1 from by omega]
    rw [Finset.sum_range_one]
    rw [show d + 1 - 1 - 0 = d from by omega]
    rw [pow_zero, one_mul]
    rw [LeibnizForm_succ_succ T A d d (le_refl d)]
    have hZ : LeibnizForm T A (d + 1) d = 0 := by
      unfold LeibnizForm
      rw [if_pos (by omega : d < d + 1)]
    rw [hZ, mul_zero, zero_add]
  | succ k IH =>
    have hLE : d ≤ d + 1 + k := by omega
    rw [show d + 1 + (k + 1) = (d + 1 + k) + 1 from by omega]
    rw [LeibnizForm_succ_succ T A d (d + 1 + k) hLE]
    rw [IH (by omega)]
    rw [Finset.mul_sum]
    rw [show d + 1 + k + 1 - d = k + 2 from by omega]
    rw [show d + 1 + k + 1 - 1 = d + 1 + k from by omega]
    rw [show k + 2 = (k + 1) + 1 from rfl]
    rw [Finset.sum_range_succ'
          (fun j => T ^ j * A * LeibnizForm T A d (d + 1 + k - j)) (k + 1)]
    rw [pow_zero, one_mul]
    rw [show d + 1 + k - 0 = d + 1 + k from rfl]
    rw [show d + 1 + k - d = k + 1 from by omega]
    congr 1
    refine Finset.sum_congr rfl ?_
    intro j _
    rw [show d + 1 + k - (j + 1) = d + 1 + k - 1 - j from by omega]
    rw [show d + 1 + k - 1 = d + k from by omega]
    rw [pow_succ']
    rw [mul_assoc T (T ^ j) A, mul_assoc T (T ^ j * A) _]

/-- **Main turn 3 result**: for `d ≤ N`,
`LeibnizForm T A d N = RseqMat T A d (N - d)`.

Closes the matching between the explicit Leibniz form (sum over compositions
of `N - d` into `d + 1` parts) and the abstract `d`-fold convolutional sum
`RseqMat T A d (·)` from `SequelPolesArbD`.

Proof: induction on `d`, generalising `N`.

* `d = 0`: `LeibnizForm T A 0 N = T^N` from `LeibnizForm_zero`;
  `RseqMat T A 0 (N - 0) = T^N` by definition of `RseqMat` at `d = 0`
  (and `N - 0 = N`).
* `d + 1`: rewrite the LHS via `LeibnizForm_succ_eq_conv` into the
  convolutional partition `∑ j ∈ range (N - d), T^j · A · LeibnizForm T A d
  (N - 1 - j)`; unfold the RHS `RseqMat T A (d + 1) (N - (d + 1))` to
  `convolveMat T A (RseqMat T A d) (N - (d + 1)) = ∑ j ∈ range (N - d),
  T^j · A · RseqMat T A d (N - (d + 1) - j)`; apply the IH to each
  summand index (using `N - 1 - j ≥ d` for `j < N - d`). -/
theorem LeibnizForm_eq_RseqMat (T A : Matrix ι ι R) (d N : ℕ) (hN : d ≤ N) :
    LeibnizForm T A d N = RseqMat T A d (N - d) := by
  induction d generalizing N with
  | zero =>
    rw [LeibnizForm_zero, Nat.sub_zero]
    rfl
  | succ d IH =>
    rw [LeibnizForm_succ_eq_conv T A d N hN]
    show ∑ j ∈ range (N - d), T ^ j * A * LeibnizForm T A d (N - 1 - j)
        = convolveMat T A (RseqMat T A d) (N - (d + 1))
    unfold convolveMat
    rw [show N - (d + 1) + 1 = N - d from by omega]
    refine Finset.sum_congr rfl ?_
    intro j hj
    rw [Finset.mem_range] at hj
    rw [IH (N - 1 - j) (by omega)]
    rw [show N - 1 - j - d = N - (d + 1) - j from by omega]

/-- **Campaign capstone**: for `d ≤ N`, the `[x^d]` coefficient of `(T + x·A)^N`
equals the abstract `d`-fold convolutional sum `RseqMat T A d (N - d)`.

```
((T + x·A) ^ N).coeff d = RseqMat T A d (N - d)        (for d ≤ N).
```

Direct corollary of `Tmx_pow_coeff_eq_LeibnizForm` (turn 2c) and
`LeibnizForm_eq_RseqMat` (turn 3, above).

This closes step 3 of the paper's `lem:poles` chain in Lean. Combined with
`SequelPolesArbD.RseqMat_p_dPlus1_recurrence` (the `(d + 1)`-fold charpoly
annihilation theorem), the matrix-level Cayley–Hamilton recurrence for the
power-series coefficient `((T + x·A) ^ N).coeff d` is now end-to-end
formalised. -/
theorem Tmx_pow_coeff_eq_RseqMat (T A : Matrix ι ι R) (d N : ℕ) (hN : d ≤ N) :
    ((Tmx T A) ^ N).coeff d = RseqMat T A d (N - d) := by
  rw [Tmx_pow_coeff_eq_LeibnizForm, LeibnizForm_eq_RseqMat T A d N hN]

end OrigamiCone.Sequel
