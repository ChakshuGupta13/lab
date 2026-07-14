import Mathlib
import OrigamiCone.SequelPolesArbD

/-!
# Sequel: arbitrary-`d` `lem:poles` (scalar sandwich form)

Standalone formalisation of the **scalar sandwich form** of the arbitrary-`d`
`lem:poles` theorem from the sequel paper

> *Degree-$d$ vertex counts of the $m \times n$ origami flip graph:
> a polynomial meta-theorem.*

`SequelPolesArbD.RseqMat_p_dPlus1_recurrence` proved the matrix-level
recurrence: `charActIter T (RseqMat T A d) (d+1) n = 0` (the `(d+1)`-fold
charpoly action of `T` annihilates the `d`-fold matrix-valued convolutional
sum). This module bridges to the **scalar form** by sandwiching with row/column
boundary vectors `u, v`. The resulting scalar sequence

    `c(n) := u ⬝ᵥ (RseqMat T A d n) *ᵥ v`

satisfies the scalar linear recurrence whose characteristic polynomial is
`T.charpoly^(d+1)` — which is the form needed to bridge to the GF rational-
poles conclusion of `lem:poles` via `SequelRatGF.transfer_GF_rational`.

## Bridge structure

The bridge consists of two compatibility lemmas (sandwich commutes with
charpoly action) and an iterated form (commutes with the iterated action):

* `scalarSeq u S v n := u ⬝ᵥ (S n *ᵥ v)` — scalar sandwich of a matrix-
  valued sequence.
* `scalarCharAct p c n := ∑_k p.coeff k * c (n+k)` — scalar charpoly action
  on a scalar sequence (the canonical scalar analogue of `charActMat`).
* `scalarCharActIter p c d n` — `d`-fold iterate of `scalarCharAct`.
* `scalarSeq_charActMat` : single-step bridge
  `scalarSeq u (charActMat T S) v n = scalarCharAct T.charpoly (scalarSeq u S v) n`.
  Direct from `Matrix.sum_mulVec`, `dotProduct_sum`, `Matrix.smul_mulVec`,
  `dotProduct_smul`.
* `scalarSeq_charActIter` : iterated bridge
  `scalarSeq u (charActIter T S d) v n = scalarCharActIter T.charpoly (scalarSeq u S v) d n`.
  Induction on `d` using the single-step bridge.
* `RseqMat_sandwich_recurrence` (**main**) : the scalar `(d+1)`-fold charpoly
  action annihilates the scalar sandwich of `RseqMat T A d`:
  `scalarCharActIter T.charpoly (scalarSeq u (RseqMat T A d) v) (d+1) n = 0`.
  Direct corollary: apply `scalarSeq_charActIter` in reverse, then apply
  `RseqMat_p_dPlus1_recurrence` to get a sandwich of `0`, which evaluates
  to `0` via `Matrix.zero_mulVec` + `dotProduct_zero`.

## Connection to the paper's `lem:poles`

The paper states `lem:poles` in terms of the GF of `E_d(m, ·)`:

> The poles of `∑_n E_d(m,·) z^n` lie among the reciprocal eigenvalues of
> `T_0`, so the period of the quasi-polynomial `E_d(m,·)` divides the lcm
> of the multiplicative orders of the unit-circle eigenvalues of `T_0`.

The chain of reasoning, now mostly formalised in this Sequel chain:

1. **Matrix-level recurrence**: `charActIter T (RseqMat T A d) (d+1) ≡ 0`
   (`SequelPolesArbD.RseqMat_p_dPlus1_recurrence`).
2. **Scalar sandwich**: the scalar sequence `c(n) := u ⬝ᵥ (RseqMat T A d n) *ᵥ v`
   satisfies `scalarCharActIter T.charpoly c (d+1) ≡ 0` (this module's
   `RseqMat_sandwich_recurrence`).
3. **PowerSeries identification** (NOT formalised; disclosed): `c(n)` equals
   `[x^d](u^⊤ T_m(x)^{n+d} v)` (with the `+d` Leibniz shift documented in
   `SequelPolesConv`/`SequelPolesArbD`).
4. **Rational-GF bridge** (`SequelRatGF.transfer_GF_rational` style; partially
   formalised at `d=0` in `SequelPoles.poles_at_x_zero`; the general bridge
   from a linear recurrence to a rational GF with denominator dividing the
   reversed characteristic polynomial is standard).
5. **Pole localisation**: the GF of `c(n)` has poles only among the reciprocal
   roots of `T.charpoly^(d+1)`, i.e. reciprocal eigenvalues of `T` with
   multiplicity at most `d+1`. This is the paper's `lem:poles` conclusion.

Steps 1 and 2 (the matrix and scalar recurrences) are now formalised
end-to-end. Steps 3, 4, 5 are downstream / standard.

## Theorems

* `scalarSeq`, `scalarCharAct`, `scalarCharActIter` : definitions.
* `scalarSeq_charActMat` : single-step bridge.
* `scalarSeq_charActIter` : iterated bridge.
* `RseqMat_sandwich_recurrence` (**main**) : scalar `(d+1)`-fold charpoly
  action annihilates the scalar sandwich of `RseqMat T A d`.

## Scope

* All scalar primitives + the bridge + the main theorem are proved end-to-end
  (no `sorry`).
* The connection to the paper's `[x^d]` coefficient (step 3 above) is the
  PowerSeries Leibniz identification, disclaimed in
  `SequelPolesConv`/`SequelPolesArbD` and still downstream.
* The pole-localisation conclusion (step 5 above) is the standard rational-GF
  bridge from a linear recurrence; partially formalised at `d=0` in
  `SequelPoles.poles_at_x_zero`.
* **Discipline deviation**: like `SequelPolesArbD`, this module imports a
  Sequel module (`OrigamiCone.SequelPolesArbD`, which in turn transitively
  imports `OrigamiCone.SequelPolesIter`). Justified by the same DRY argument
  (~250 lines of primitives + ~80 lines of matrix-level induction otherwise).
  No parallel-session edit risk: both dependencies were built and committed
  in the same session as this module.
* Per the discipline: NOT added to root aggregator `OrigamiCone.lean`.

No `sorry`; check with
`#print axioms OrigamiCone.Sequel.RseqMat_sandwich_recurrence`.
-/

namespace OrigamiCone.Sequel

open Matrix Polynomial

variable {R : Type*} [CommRing R] {ι : Type*} [Fintype ι] [DecidableEq ι]

/-- **Scalar sandwich** of a matrix-valued sequence: `u ⬝ᵥ (S n *ᵥ v)`. -/
noncomputable def scalarSeq (u : ι → R) (S : ℕ → Matrix ι ι R) (v : ι → R)
    (n : ℕ) : R :=
  u ⬝ᵥ (S n *ᵥ v)

/-- **Scalar charpoly action** on a scalar sequence: the canonical scalar
analogue of `charActMat`. `∑_k p.coeff k * c (n+k)`. -/
noncomputable def scalarCharAct (p : Polynomial R) (c : ℕ → R) (n : ℕ) : R :=
  ∑ k ∈ Finset.range (p.natDegree + 1), p.coeff k * c (n + k)

/-- **Iterated scalar charpoly action**: the `d`-fold composition of
`scalarCharAct p`. -/
noncomputable def scalarCharActIter (p : Polynomial R) (c : ℕ → R) :
    ℕ → ℕ → R
  | 0, n => c n
  | d + 1, n => scalarCharAct p (scalarCharActIter p c d) n

/-- **Single-step sandwich bridge**: the scalar sandwich of `charActMat T S`
equals the scalar charpoly action applied to the scalar sandwich of `S`.
Direct from `Matrix.sum_mulVec` + `dotProduct_sum` + `Matrix.smul_mulVec` +
`dotProduct_smul`. -/
lemma scalarSeq_charActMat (T : Matrix ι ι R) (S : ℕ → Matrix ι ι R)
    (u v : ι → R) (n : ℕ) :
    scalarSeq u (charActMat T S) v n
      = scalarCharAct T.charpoly (scalarSeq u S v) n := by
  unfold scalarSeq charActMat scalarCharAct
  rw [Matrix.sum_mulVec, dotProduct_sum]
  refine Finset.sum_congr rfl ?_
  intro k _
  rw [Matrix.smul_mulVec, dotProduct_smul, smul_eq_mul]

/-- **Iterated sandwich bridge**: the scalar sandwich of `charActIter T S d`
equals the iterated scalar charpoly action applied to the scalar sandwich of
`S`. Proof by induction on `d`, using `scalarSeq_charActMat` at the step. -/
lemma scalarSeq_charActIter (T : Matrix ι ι R) (S : ℕ → Matrix ι ι R)
    (u v : ι → R) (d n : ℕ) :
    scalarSeq u (charActIter T S d) v n
      = scalarCharActIter T.charpoly (scalarSeq u S v) d n := by
  induction d generalizing n with
  | zero => rfl
  | succ d IH =>
    show scalarSeq u (charActMat T (charActIter T S d)) v n
      = scalarCharAct T.charpoly (scalarCharActIter T.charpoly (scalarSeq u S v) d) n
    rw [scalarSeq_charActMat]
    congr 1
    funext m
    exact IH m

/-- **`p^{d+1}`-recurrence on the scalar sandwich of the `d`-fold convolutional
sum** (`lem:poles` at arbitrary `d`, scalar sandwich form). For all `d` and `n`,

    `scalarCharActIter T.charpoly (scalarSeq u (RseqMat T A d) v) (d+1) n = 0`.

That is, the scalar sequence `c(n) := u ⬝ᵥ (RseqMat T A d n) *ᵥ v` satisfies
the iterated `(d+1)`-fold scalar charpoly action of `T.charpoly`. Together
with the PowerSeries Leibniz identification (still downstream) and the
rational-GF bridge (`SequelRatGF`-style), this gives the paper's pole
localisation: the GF of `c(n)` has poles only at reciprocal eigenvalues of
`T` with multiplicity at most `d+1`.

Proof: rewrite the LHS via `scalarSeq_charActIter` (backwards) to obtain
`scalarSeq u (charActIter T (RseqMat T A d) (d+1)) v n`; apply
`RseqMat_p_dPlus1_recurrence` to reduce the inner `charActIter ...` to the
zero matrix; then `Matrix.zero_mulVec` + `dotProduct_zero` finish. -/
theorem RseqMat_sandwich_recurrence (T A : Matrix ι ι R) (u v : ι → R) (d n : ℕ) :
    scalarCharActIter T.charpoly (scalarSeq u (RseqMat T A d) v) (d + 1) n = 0 := by
  rw [← scalarSeq_charActIter]
  -- Goal: scalarSeq u (charActIter T (RseqMat T A d) (d+1)) v n = 0
  unfold scalarSeq
  rw [RseqMat_p_dPlus1_recurrence]
  -- Goal: u ⬝ᵥ ((0 : Matrix ι ι R) *ᵥ v) = 0
  rw [Matrix.zero_mulVec, dotProduct_zero]

end OrigamiCone.Sequel
