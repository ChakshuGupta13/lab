import Mathlib
import OrigamiCone.SequelPolesLeibnizBase

/-!
# Sequel: PowerSeries Leibniz identification — split + polynomial-coefficient step

Second installment (turn 2a/3) of the multi-turn campaign formalising step 3
of the paper's `lem:poles` chain. The campaign target is the identification

```
((T + x·A)^N).coeff d = LeibnizForm T A d N = RseqMat T A d (N - d)
```

in `Polynomial (Matrix ι ι R)`.

* **Previous turn (`SequelPolesLeibnizBase`)** : definitions (`Tmx`,
  `leibnizFactor`, `LeibnizForm`), degree bound on `Tmx`, vacuous case
  `N < d`, base case `d = 0` identification.
* **This module (turn 2a)** : structural identities used by the inductive
  step on `N` for general `d`.
    - `leibnizFactor_split` peels the first block of a `leibnizFactor` at
      arity `e + 1` into `T^{g 0} · A · (rest as a leibnizFactor at arity
      `e` over the tail tuple). This is the dependent-type-friendly form
      (arbitrary `g : Fin (e + 1 + 1) → ℕ` rather than `Fin.cons k h`,
      avoiding the elaboration trap of substituting `Fin.cons` inside
      `List.ofFn`).
    - `leibnizFactor_cons_zero` and `leibnizFactor_cons_succ` are the
      immediate corollaries used in the bijection-based sum splits.
    - `Tmx_pow_succ_coeff_succ` is the **polynomial-side** step: for
      `d = e + 1 ≥ 1`, `((Tmx)^{N+1}).coeff (e+1) = T · ((Tmx)^N).coeff
      (e+1) + A · ((Tmx)^N).coeff e`. Direct from `coeff_mul` +
      antidiagonal-as-range expansion + `Tmx_coeff_higher` to kill `j ≥ 2`
      terms.
* **Next turn (turn 2b)** : the recursive identity
  `LeibnizForm T A (e+1) (N+1) = T · LeibnizForm T A (e+1) N + A ·
  LeibnizForm T A e N` via bijection-based splitting of
  `antidiagonalTuple (e+2) (N - e)` by `g 0 = 0` vs `g 0 ≥ 1`.
* **Turn 2c** : combine `Tmx_pow_succ_coeff_succ` with `LeibnizForm_succ` to
  prove `((Tmx)^N).coeff d = LeibnizForm T A d N` by induction on `N`
  (parametric in `d`).
* **Turn 3** : the identification `LeibnizForm T A d N = RseqMat T A d
  (N - d)` matching the abstract `d`-fold convolutional sum from
  `SequelPolesArbD`.

## Theorems

* `leibnizFactor_split` : `leibnizFactor T A (e+1) g = T^{g 0} · A ·
  leibnizFactor T A e (Fin.tail g)`. The arbitrary-`g` formulation is
  essential — substituting `Fin.cons k h` directly into the `List.ofFn`
  defeats Lean's congruence machinery.
* `leibnizFactor_cons_zero` : the `g 0 = 0` specialisation.
* `leibnizFactor_cons_succ` : the `g 0 = k + 1` specialisation, exhibiting
  the `T` factor that the bijection-based sum split will pull out.
* `Tmx_pow_succ_coeff_succ` : the polynomial-coefficient identity for
  `d = e + 1`. The `d = 0` analogue is already
  `Tmx_pow_coeff_zero` (combined with the `pow_succ'` unfold).

## Scope

* Imports `Mathlib` and `OrigamiCone.SequelPolesLeibnizBase`. Per the
  cross-Sequel-import discipline disclosed in `SequelPolesArbD`, this is
  acceptable when the imported module was built and committed in the same
  campaign as the importing module (no parallel-session race).
* No `sorry`. Axioms: `[propext, Classical.choice, Quot.sound]` baseline.
* NOT added to root aggregator `OrigamiCone.lean`.

Check axioms with
`#print axioms OrigamiCone.Sequel.Tmx_pow_succ_coeff_succ`.
-/

namespace OrigamiCone.Sequel

open Polynomial Matrix Finset

variable {R : Type*} [CommRing R] {ι : Type*} [Fintype ι] [DecidableEq ι]

/-- **Split lemma** (the key infrastructure): `leibnizFactor T A (e+1) g` peels
into `T^{g 0} · A · leibnizFactor T A e (Fin.tail g)`.

Stated for arbitrary `g : Fin (e + 1 + 1) → ℕ` rather than the `Fin.cons k h`
specialisation. The arbitrary form is necessary to side-step Lean's
elaboration trap: directly substituting `Fin.cons k h` inside the
`List.ofFn` definition of `leibnizFactor` blocks the rewrite chain on
dependent-type residuals from `List.ofFn_succ` interacting with the
`if (i : ℕ) < d` predicate. The arbitrary-`g` form lets us peel
`List.ofFn_succ` cleanly, then handle the body's `i.succ` index via
`Fin.val_succ` + `by_cases` on the predicate. -/
theorem leibnizFactor_split (T A : Matrix ι ι R) (e : ℕ)
    (g : Fin (e + 1 + 1) → ℕ) :
    leibnizFactor T A (e + 1) g
      = T ^ g 0 * A * leibnizFactor T A e (Fin.tail g) := by
  show (List.ofFn (fun i : Fin (e + 1 + 1) =>
        if (i : ℕ) < e + 1 then T ^ g i * A else T ^ g i)).prod
      = T ^ g 0 * A * (List.ofFn (fun i : Fin (e + 1) =>
        if (i : ℕ) < e then T ^ Fin.tail g i * A else T ^ Fin.tail g i)).prod
  rw [List.ofFn_succ, List.prod_cons]
  simp only [Fin.val_zero, Nat.zero_lt_succ, if_true]
  have key :
      (fun i : Fin (e + 1) =>
          if (i.succ : ℕ) < e + 1 then T ^ g i.succ * A else T ^ g i.succ)
      = (fun i : Fin (e + 1) =>
          if (i : ℕ) < e then T ^ Fin.tail g i * A else T ^ Fin.tail g i) := by
    funext i
    rw [Fin.val_succ]
    by_cases hi : (i : ℕ) < e
    · have : (i : ℕ) + 1 < e + 1 := by omega
      simp [this, hi, Fin.tail]
    · have : ¬ ((i : ℕ) + 1 < e + 1) := by omega
      simp [this, hi, Fin.tail]
  rw [key]

/-- **`g 0 = 0` corollary**: when the head exponent is zero, the leading
`T^0 = 1` collapses and a bare `A` factor exposes the tail. -/
theorem leibnizFactor_cons_zero (T A : Matrix ι ι R) (e : ℕ)
    (h : Fin (e + 1) → ℕ) :
    leibnizFactor T A (e + 1) (Fin.cons 0 h) = A * leibnizFactor T A e h := by
  rw [leibnizFactor_split]
  simp [Fin.cons_zero, Fin.tail_cons]

/-- **`g 0 = k + 1` corollary**: when the head exponent is positive, peel a
single `T` off the front. Used by the bijection that maps the `g 0 ≥ 1`
subset of `antidiagonalTuple (e+2) (N - e)` bijectively onto
`antidiagonalTuple (e+2) (N - e - 1)` via `g 0 ↦ g 0 - 1`. -/
theorem leibnizFactor_cons_succ (T A : Matrix ι ι R) (e k : ℕ)
    (h : Fin (e + 1) → ℕ) :
    leibnizFactor T A (e + 1) (Fin.cons (k + 1) h)
      = T * leibnizFactor T A (e + 1) (Fin.cons k h) := by
  rw [leibnizFactor_split T A e (Fin.cons (k + 1) h)]
  rw [leibnizFactor_split T A e (Fin.cons k h)]
  simp only [Fin.cons_zero, Fin.tail_cons]
  rw [pow_succ', mul_assoc T (T ^ k) A]
  rw [mul_assoc T (T ^ k * A) _]

/-- **Polynomial-side step** (for `d = e + 1 ≥ 1`): peel a `Tmx` off the
front of `(Tmx)^{N+1}` and read the `(e+1)`-coefficient. Only `j ∈ {0, 1}`
survive the antidiagonal sum because `Tmx` has degree `≤ 1`.

The `d = 0` analogue is already `Tmx_pow_coeff_zero` (where `(Tmx)^N.coeff 0
= T^N`); the recurrence there collapses to a single `T` factor since
`antidiagonal 0 = {(0,0)}`. -/
theorem Tmx_pow_succ_coeff_succ (T A : Matrix ι ι R) (N e : ℕ) :
    ((Tmx T A) ^ (N + 1)).coeff (e + 1) =
      T * ((Tmx T A) ^ N).coeff (e + 1) + A * ((Tmx T A) ^ N).coeff e := by
  rw [pow_succ', Polynomial.coeff_mul]
  rw [Finset.Nat.sum_antidiagonal_eq_sum_range_succ_mk
        (fun x => (Tmx T A).coeff x.1 * ((Tmx T A) ^ N).coeff x.2) (e + 1)]
  rw [Finset.sum_range_succ'
        (fun k => (Tmx T A).coeff k * ((Tmx T A) ^ N).coeff (e + 1 - k)) (e + 1)]
  rw [Tmx_coeff_zero]
  rw [Finset.sum_range_succ'
        (fun k => (Tmx T A).coeff (k + 1) * ((Tmx T A) ^ N).coeff (e + 1 - (k + 1))) e]
  rw [zero_add, Tmx_coeff_one, show e + 1 - 1 = e from rfl]
  rw [Finset.sum_eq_zero (fun j _ => by
        rw [Tmx_coeff_higher T A (j + 1 + 1) (by omega), zero_mul])]
  rw [zero_add, show e + 1 - 0 = e + 1 from rfl, add_comm]

end OrigamiCone.Sequel
