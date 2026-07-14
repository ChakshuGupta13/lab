import Mathlib
import OrigamiCone.SequelPolesLeibnizRec

/-!
# Sequel: PowerSeries Leibniz identification — closing the matrix-coefficient identity

Fourth installment (turn 2c/3) of the multi-turn campaign formalising step 3
of the paper's `lem:poles` chain. This module closes the
**matrix-coefficient identification** for general `d`:

```
((T + x·A)^N).coeff d = LeibnizForm T A d N    (in Polynomial (Matrix ι ι R))
```

* **Turn 1 (`SequelPolesLeibnizBase`, 9deb844)** : `Tmx`, `leibnizFactor`,
  `LeibnizForm` definitions + `d = 0` and `N < d` cases.
* **Turn 2a (`SequelPolesLeibnizStep`, 7c200ef)** : split lemma +
  polynomial-side step `Tmx_pow_succ_coeff_succ`.
* **Turn 2b (`SequelPolesLeibnizRec`, 471af16)** : Leibniz-side step
  `LeibnizForm_succ_succ`.
* **This module (turn 2c)** : compose the polynomial-side and Leibniz-side
  step lemmas via induction on `N` (parametric in `d`) to close the
  identification for **all `d, N`**.
* **Turn 3 (future)** : match `LeibnizForm T A d N = RseqMat T A d (N - d)`
  by induction on `d`, completing the Leibniz step of the paper's
  `lem:poles` chain.

## Theorem

`Tmx_pow_coeff_eq_LeibnizForm` (closes turn 2c):
`((Tmx T A) ^ N).coeff d = LeibnizForm T A d N` for all `d, N : ℕ`.

Proof outline:

* Induction on `N`, generalising `d`.
* `N = 0`: split on `d = 0` vs `d > 0`. For `d = 0`, both sides are `1`
  (LHS via `Polynomial.coeff_one` + `if_pos`, RHS via `LeibnizForm_zero`
  + `pow_zero`). For `d > 0`, both sides are `0` (LHS via
  `Polynomial.coeff_one` + `if_neg`, RHS via the `N < d` vacuous branch).
* `N + 1`: split on `d = 0` vs `d = e + 1`.
  - `d = 0`: apply `Tmx_pow_coeff_zero` (LHS = `T^(N+1)`) + `LeibnizForm_zero`
    (RHS = `T^(N+1)`).
  - `d = e + 1`: split on `N < e` vs `N ≥ e`.
    - `N < e`: apply `Tmx_pow_coeff_eq_LeibnizForm_vacuous` (both sides 0).
    - `N ≥ e`: apply the matched step lemmas
      `Tmx_pow_succ_coeff_succ` (LHS recurrence) and
      `LeibnizForm_succ_succ` (RHS recurrence; the `e ≤ N` hypothesis is
      satisfied). Close with two applications of the inductive hypothesis
      at `(e + 1, N)` and `(e, N)`.

## Scope

* Imports `Mathlib` and `OrigamiCone.SequelPolesLeibnizRec` (which
  transitively imports `SequelPolesLeibnizStep` and `SequelPolesLeibnizBase`).
  Same cross-Sequel-import discipline (same-campaign, no parallel-session
  race).
* No `sorry`. Axioms: `[propext, Classical.choice, Quot.sound]` baseline.
* NOT added to root aggregator `OrigamiCone.lean`.

Check axioms with
`#print axioms OrigamiCone.Sequel.Tmx_pow_coeff_eq_LeibnizForm`.
-/

namespace OrigamiCone.Sequel

open Polynomial Matrix Finset

variable {R : Type*} [CommRing R] {ι : Type*} [Fintype ι] [DecidableEq ι]

/-- **Matrix-coefficient identification** (closes turn 2c of the campaign):

```
((T + x·A) ^ N).coeff d = LeibnizForm T A d N         (in Polynomial (Matrix ι ι R)).
```

The `[x^d]` coefficient of `(T + x·A)^N` is the formal Leibniz sum over all
compositions `g : Fin (d + 1) → ℕ` of `N - d` of the products
`T^{g 0} · A · T^{g 1} · A · ⋯ · A · T^{g d}` (the empty sum, equal to
`0`, when `N < d` — no valid composition exists).

Proof: induction on `N`, generalising `d`.

* `N = 0`: both sides reduce to `if d = 0 then 1 else 0`.
* `N + 1`: case-split on `d`. At `d = 0`, both sides equal `T^(N + 1)`
  (`Tmx_pow_coeff_zero` and `LeibnizForm_zero`). At `d = e + 1`, case-split
  on `N < e` (both sides vacuously zero by
  `Tmx_pow_coeff_eq_LeibnizForm_vacuous`) vs `N ≥ e` (apply the matched
  step lemmas `Tmx_pow_succ_coeff_succ` and `LeibnizForm_succ_succ`, then
  the inductive hypothesis at `(e + 1, N)` and `(e, N)`). -/
theorem Tmx_pow_coeff_eq_LeibnizForm (T A : Matrix ι ι R) (d N : ℕ) :
    ((Tmx T A) ^ N).coeff d = LeibnizForm T A d N := by
  induction N generalizing d with
  | zero =>
    rw [pow_zero]
    rcases Nat.eq_zero_or_pos d with hd | hd
    · subst hd
      rw [Polynomial.coeff_one]
      simp [LeibnizForm_zero]
    · rw [Polynomial.coeff_one, if_neg (Nat.ne_of_gt hd)]
      unfold LeibnizForm
      rw [if_pos hd]
  | succ N IH =>
    rcases Nat.eq_zero_or_pos d with hd | hd
    · subst hd
      rw [Tmx_pow_coeff_zero, LeibnizForm_zero]
    · obtain ⟨e, rfl⟩ : ∃ e, d = e + 1 := ⟨d - 1, by omega⟩
      by_cases hNe : N < e
      · exact Tmx_pow_coeff_eq_LeibnizForm_vacuous T A (e + 1) (N + 1) (by omega)
      · rw [Tmx_pow_succ_coeff_succ T A N e]
        rw [LeibnizForm_succ_succ T A e N (by omega)]
        rw [IH (e + 1), IH e]

end OrigamiCone.Sequel
