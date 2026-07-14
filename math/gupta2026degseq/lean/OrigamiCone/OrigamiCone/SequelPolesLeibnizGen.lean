import OrigamiCone.SequelPolesLeibnizBase

/-!
# Sequel: `T + x·B(x)` Leibniz generalization (Task E.γ.b)

The existing `SequelPolesLeibniz` chain formalizes the Leibniz identification
for `Tmx T A = C T + X * C A` (constant matrix `A`).  This module generalizes
the setup to **polynomial** matrix `B(x)`, matching the paper's proof of
`lem:poles` and the "Onset" degree-bound in `lem:uniform`.

## Definition

* `TmxGen T B` — the matrix-coefficient polynomial `T + X * B` for a polynomial
  matrix `B : Polynomial (Matrix ι ι R)`.  Specializes to `Tmx T A` when
  `B = C A`.

## Theorems

* `TmxGen_natDegree_le` — `(TmxGen T B).natDegree ≤ 1 + B.natDegree`.
* `TmxGen_pow_natDegree_le` — `((TmxGen T B)^N).natDegree ≤ N * (1 + B.natDegree)`.
* **`TmxGen_pow_coeff_vacuous`** — `((TmxGen T B)^N).coeff d = 0` when
  `d > N * (1 + B.natDegree)`.  This is the pole-location structural fact: the
  `[x^d]`-slice is a finite polynomial in the generating variable, so its
  denominator (after summing `∑_N z^N ·`) has bounded pole order.
* `coeff_pow_of_low_coeff_eq` — general polynomial power stability lemma:
  low-degree coefficient agreement lifts to the `N`-th power.
* **`TmxGen_pow_coeff_truncation_invariant`** — `((TmxGen T B)^N).coeff d`
  depends only on `B.coeff 0, ..., B.coeff (d-1)`.  This packages the paper's
  observation that the `[x^d]`-slice depends only on the "low-degree" part of
  `B(x) = ∑_j x^{j-1} B_j`, namely `B_1, ..., B_d`.
* `TmxGen_of_constant` — `TmxGen T (C A) = Tmx T A` (bridge to the existing
  `Tmx T A` framework).

## Substrate

Imports `SequelPolesLeibnizBase` (for `Tmx T A`).  Standalone.

No `sorry`.  Axioms: `[propext, Classical.choice, Quot.sound]` baseline.
-/

namespace OrigamiCone.Sequel

open Polynomial

variable {R : Type*} [CommRing R] {ι : Type*} [Fintype ι] [DecidableEq ι]

/-- **Generalized `Tmx`**: `T + x · B(x)` for a polynomial matrix `B`. -/
noncomputable def TmxGen (T : Matrix ι ι R) (B : Polynomial (Matrix ι ι R)) :
    Polynomial (Matrix ι ι R) :=
  C T + Polynomial.X * B

/-- `TmxGen T (C A) = Tmx T A` (the polynomial `B = C A` case reduces to
`Tmx`). -/
lemma TmxGen_of_constant (T A : Matrix ι ι R) :
    TmxGen T (C A) = Tmx T A := rfl

/-- `(TmxGen T B).natDegree ≤ 1 + B.natDegree`. -/
lemma TmxGen_natDegree_le (T : Matrix ι ι R) (B : Polynomial (Matrix ι ι R)) :
    (TmxGen T B).natDegree ≤ 1 + B.natDegree := by
  unfold TmxGen
  have hC : (C T : Polynomial (Matrix ι ι R)).natDegree = 0 :=
    Polynomial.natDegree_C T
  have hXB : ((Polynomial.X : Polynomial (Matrix ι ι R)) * B).natDegree ≤ 1 + B.natDegree := by
    refine le_trans Polynomial.natDegree_mul_le ?_
    have hX : (Polynomial.X : Polynomial (Matrix ι ι R)).natDegree ≤ 1 :=
      Polynomial.natDegree_X_le
    omega
  refine le_trans (Polynomial.natDegree_add_le _ _) ?_
  omega

/-- `((TmxGen T B)^N).natDegree ≤ N * (1 + B.natDegree)`. -/
lemma TmxGen_pow_natDegree_le (T : Matrix ι ι R) (B : Polynomial (Matrix ι ι R)) (N : ℕ) :
    ((TmxGen T B) ^ N).natDegree ≤ N * (1 + B.natDegree) := by
  induction N with
  | zero => simp
  | succ N IH =>
    rw [pow_succ]
    refine le_trans Polynomial.natDegree_mul_le ?_
    have h := TmxGen_natDegree_le T B
    have hmul : N * (1 + B.natDegree) + (1 + B.natDegree) = (N + 1) * (1 + B.natDegree) := by
      ring
    omega

/-- **Vacuous case**: for `d > N * (1 + B.natDegree)`, the `d`-coefficient of
`(TmxGen T B)^N` is zero.  This is the pole-location structural fact — the
`[x^d]`-slice is a finite polynomial in `N`, so `∑_N z^N · [x^d]((TmxGen T B)^N)`
is a rational function of `z` of bounded denominator degree. -/
theorem TmxGen_pow_coeff_vacuous (T : Matrix ι ι R) (B : Polynomial (Matrix ι ι R))
    (d N : ℕ) (h : N * (1 + B.natDegree) < d) :
    ((TmxGen T B) ^ N).coeff d = 0 := by
  apply Polynomial.coeff_eq_zero_of_natDegree_lt
  have := TmxGen_pow_natDegree_le T B N
  omega

/-- **Low-coefficient stability of polynomial powers.**  If two polynomials
`P` and `Q` agree on all coefficients up to degree `d`, then their `N`-th
powers also agree on coefficients up to degree `d`.  Proved by induction on
`N`; the step case uses `Polynomial.coeff_mul` to split the `(N+1)`-power
into a sum over pairs `(j, k)` with `j + k ≤ d`, and applies the IH plus the
hypothesis to the two factors. -/
lemma coeff_pow_of_low_coeff_eq {S : Type*} [Semiring S] (P Q : Polynomial S) (d N : ℕ)
    (hPQ : ∀ i ≤ d, P.coeff i = Q.coeff i) :
    ∀ i ≤ d, (P^N).coeff i = (Q^N).coeff i := by
  induction N with
  | zero => intro i _; rfl
  | succ N IH =>
    intro i hi
    rw [pow_succ, pow_succ]
    rw [Polynomial.coeff_mul, Polynomial.coeff_mul]
    apply Finset.sum_congr rfl
    intro ⟨j, k⟩ hjk
    simp only [Finset.mem_antidiagonal] at hjk
    have hj : j ≤ d := by omega
    have hk : k ≤ d := by omega
    rw [IH j hj, hPQ k hk]

/-- **Truncation invariance of `TmxGen^N.coeff d`**: the `[x^d]`-coefficient
of `(TmxGen T B)^N` depends only on the coefficients of `B` at degrees
`< d`.  Concretely, if `B_1` and `B_2` agree on `.coeff j` for all `j < d`,
then `((TmxGen T B_1)^N).coeff d = ((TmxGen T B_2)^N).coeff d`.

This is the key "finite support" fact: the paper's expansion of
`(I - z T_m(x))^{-1}` about `T_0 = TmxGen T 0` in `T_m(x) = T_0 + x B(x)`
depends only on finitely many `B_j` (namely `B_1, ..., B_d`), even if `B`
has arbitrary degree.  So the `[x^d]`-slice is a finite matrix polynomial in
`T` and `B_1, ..., B_d`. -/
theorem TmxGen_pow_coeff_truncation_invariant (T : Matrix ι ι R)
    (B_1 B_2 : Polynomial (Matrix ι ι R)) (d N : ℕ)
    (h_agree : ∀ j < d, B_1.coeff j = B_2.coeff j) :
    ((TmxGen T B_1)^N).coeff d = ((TmxGen T B_2)^N).coeff d := by
  have h_TmxGen_agree : ∀ i ≤ d, (TmxGen T B_1).coeff i = (TmxGen T B_2).coeff i := by
    intro i hi
    unfold TmxGen
    rcases Nat.eq_zero_or_pos i with hi0 | hi_pos
    · subst hi0
      simp
    · obtain ⟨k, rfl⟩ : ∃ k, i = k + 1 := ⟨i - 1, by omega⟩
      have hk : k < d := by omega
      simp only [Polynomial.coeff_add, Polynomial.coeff_C,
        show (k + 1) ≠ 0 from Nat.succ_ne_zero k, if_false, Polynomial.coeff_X_mul, zero_add]
      exact h_agree k hk
  exact coeff_pow_of_low_coeff_eq (TmxGen T B_1) (TmxGen T B_2) d N h_TmxGen_agree d le_rfl

end OrigamiCone.Sequel
