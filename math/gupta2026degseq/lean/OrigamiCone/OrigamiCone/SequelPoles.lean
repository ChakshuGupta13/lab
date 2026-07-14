import Mathlib

/-!
# Sequel meta-theorem: specialisation primitive for the transfer-matrix poles (`lem:poles`)

Standalone formalisation of the **specialisation principle** underlying
`Lemma lem:poles` of the sequel paper

> *Degree-$d$ vertex counts of the $m \times n$ origami flip graph:
> a polynomial meta-theorem.*

`Lemma lem:poles` is the second ingredient of the period-`1` (polynomiality)
mechanism. With `T_m(x) = T_0 + x B(x)` where `T_0 := T_m(0)` is the
extremum-free transfer matrix, the paper claims:

> The poles of $\sum_n E_d(m,\cdot)\,z^n = [x^d]\bigl(\mathbf u^\top(I-zT_m(x))^{-1}\mathbf v\bigr)$
> lie among the reciprocal eigenvalues of $T_0$, so the period of the
> quasi-polynomial $E_d(m,\cdot)$ divides the lcm of the multiplicative orders of
> the unit-circle eigenvalues of $T_0$.

The argument expands
$(I-zT_0-zx B(x))^{-1} = \sum_{k\ge0}[(I-zT_0)^{-1}\,zx B(x)]^k (I-zT_0)^{-1}$
and observes that, for each fixed `d`, `[x^d]` of the sum is a finite combination
of powers of `(I-zT_0)^{-1}` (times constant matrices), so its denominator
divides a power of `det(I-zT_0)` — equivalently, of `T_0.charpoly` after reversal.

This module proves the **specialisation principle**: the abstract statement
underlying the `[x^0]` case of the argument. For any commutative-ring
homomorphism `f : R →+* S` applied entrywise to a matrix `T : Matrix ι ι R`, the
recurrence of `SequelRatGF.transfer_recurrence` specialises faithfully — both
the scalar sequence `c n := u ⬝ᵥ T^n *ᵥ v` and the recurrence coefficients
(`T.charpoly`) push through `f`. Specialising at the constant-evaluation
homomorphism `Polynomial.evalRingHom 0` recovers the `[x^0]` slice of the
generating function and shows that the constant-in-`x` part of `c n` satisfies
the (smaller) recurrence whose characteristic polynomial is `T_0.charpoly`:

* `pow_map_ringHom` : `(T.map f)^n = (T^n).map f` for any ringhom `f`;
* `mulVec_map_ringHom` : `T.map f *ᵥ (f ∘ v) = f ∘ (T *ᵥ v)`;
* `dotProduct_map_ringHom` : `(f ∘ u) ⬝ᵥ (f ∘ v) = f (u ⬝ᵥ v)`;
* `transfer_recurrence_map` : the recurrence
  `∑ k, T.charpoly.coeff k * (u ⬝ᵥ T^(k+n) *ᵥ v) = 0` (`SequelRatGF.transfer_recurrence`)
  commutes with `f` entrywise — applying `f` everywhere gives the same recurrence
  with `T` replaced by `T.map f`, `u` by `f ∘ u`, `v` by `f ∘ v`, and the
  characteristic polynomial replaced by `(T.map f).charpoly` (equivalently, by
  `T.charpoly.map f` via `Matrix.charpoly_map`);
* `poles_at_x_zero` (`lem:poles` at the `[x^0]` slice, **recurrence-complete**):
  for `T : Matrix ι ι (Polynomial R)`, `u, v : ι → Polynomial R`, with
  `T_0 := T.map (Polynomial.evalRingHom 0)` and `u_0, v_0` defined analogously,
  the sequence `c_n^{(0)} := (u ⋝ᵥ T^n *ᵥ v).eval 0` satisfies the linear
  recurrence whose characteristic polynomial is `T_0.charpoly` — the recurrence
  half of `lem:poles` at `d = 0`. Composing with `SequelRatGF.transfer_GF_rational`
  upgrades this recurrence to the rational-GF conclusion ($C^{(0)}(z)$ has poles
  only among reciprocal roots of `T_0.charpoly`, i.e. reciprocal eigenvalues of
  `T_0`); that composition is not performed in this module but is mechanical.

Scope: the `[x^0]` slice (`d = 0`) is proved as a recurrence statement. The
full `[x^d]` case for `d > 0` is the substantive Neumann-series content of
`lem:poles`: it requires expanding `(I - zT_0 - zxB(x))^{-1}` as a power series
in `x` and tracking how each `[x^d]` is a finite combination of powers of
`(I-zT_0)^{-1}`. That expansion is a multi-hundred-line linear-algebra argument
in Lean and is **not** formalised here. The `[x^0]` case + the specialisation
principle establish the structural pattern (`T_0`'s charpoly determines the
recurrence — hence the denominator, via the rational-GF bridge of
`SequelRatGF` — for the constant-in-`x` slice); the higher `[x^d]` cases
follow the same template applied to derivatives / power-series coefficients of
the resolvent.

This module is the second step of the transfer-matrix chain
(`lem:ratGF → lem:poles → lem:quotient`). It is fully self-contained: the
Cayley-Hamilton recurrence is re-proved here directly (rather than imported from
`SequelRatGF`) because the alias form triggers a kernel `whnf` timeout when the
image matrix `T.map f` is fed into the conclusion of `transfer_recurrence`.
`lem:quotient` (peripheral spectrum `{1}` on the colour-rotation quotient) is
not formalised in this module.

No `sorry`; check with `#print axioms OrigamiCone.Sequel.poles_at_x_zero`.
-/

namespace OrigamiCone.Sequel

open Matrix

/-- **Ringhom-power commutativity**: applying a ringhom entrywise commutes with
matrix powers. -/
theorem pow_map_ringHom {R S : Type*} [CommRing R] [CommRing S] (f : R →+* S)
    {ι : Type*} [Fintype ι] [DecidableEq ι] (T : Matrix ι ι R) (k : ℕ) :
    (T.map f) ^ k = (T ^ k).map f := by
  induction k with
  | zero => simp [Matrix.map_one]
  | succ k IH => rw [pow_succ, pow_succ, IH, Matrix.map_mul]

/-- **Ringhom-mulVec commutativity**: applying a ringhom entrywise to a matrix
and a vector commutes with their matrix-vector product. -/
theorem mulVec_map_ringHom {R S : Type*} [CommRing R] [CommRing S] (f : R →+* S)
    {ι : Type*} [Fintype ι] [DecidableEq ι]
    (T : Matrix ι ι R) (v : ι → R) :
    T.map f *ᵥ (fun i => f (v i)) = fun i => f ((T *ᵥ v) i) := by
  ext i
  show ∑ j, (T.map f) i j * f (v j) = f (∑ j, T i j * v j)
  rw [map_sum]
  refine Finset.sum_congr rfl ?_
  intro j _
  show (T.map f) i j * f (v j) = f (T i j * v j)
  rw [Matrix.map_apply, map_mul]

/-- **Ringhom-dotProduct commutativity**: applying a ringhom entrywise to two
vectors commutes with their dot product. -/
theorem dotProduct_map_ringHom {R S : Type*} [CommRing R] [CommRing S] (f : R →+* S)
    {ι : Type*} [Fintype ι]
    (u v : ι → R) :
    (fun i => f (u i)) ⬝ᵥ (fun i => f (v i)) = f (u ⬝ᵥ v) := by
  show ∑ i, f (u i) * f (v i) = f (∑ i, u i * v i)
  rw [map_sum]
  refine Finset.sum_congr rfl ?_
  intro i _
  rw [map_mul]

/-- **Transfer recurrence under specialisation**. The transfer recurrence
re-proved for `T.map f`, `f ∘ u`, `f ∘ v`, where `f` is any ringhom. Direct
re-derivation rather than alias of `SequelRatGF.transfer_recurrence`: the
alias form triggers a kernel `whnf` timeout (the conclusions are defeq but
unfolding `Matrix.charpoly.coeff` through both forms exhausts heartbeats).
The proof is identical in structure to `transfer_recurrence`. -/
theorem transfer_recurrence_map {R S : Type*} [CommRing R] [CommRing S]
    (f : R →+* S) {ι : Type*} [Fintype ι] [DecidableEq ι]
    (T : Matrix ι ι R) (u v : ι → R) (n : ℕ) :
    ∑ k ∈ Finset.range ((T.map f).charpoly.natDegree + 1),
      (T.map f).charpoly.coeff k *
        ((fun i => f (u i)) ⬝ᵥ (((T.map f) ^ (k + n)) *ᵥ (fun i => f (v i)))) = 0 := by
  set Tf := T.map f
  -- Cayley-Hamilton at the matrix level, multiplied through by Tf^n.
  have hCH : (Polynomial.aeval Tf) Tf.charpoly = 0 := Matrix.aeval_self_charpoly _
  have hExpand : (Polynomial.aeval Tf) Tf.charpoly
      = ∑ k ∈ Finset.range (Tf.charpoly.natDegree + 1),
          Tf.charpoly.coeff k • Tf ^ k := Polynomial.aeval_eq_sum_range _
  rw [hExpand] at hCH
  have hMat : ∑ k ∈ Finset.range (Tf.charpoly.natDegree + 1),
      Tf.charpoly.coeff k • Tf ^ (k + n) = 0 := by
    calc ∑ k ∈ Finset.range (Tf.charpoly.natDegree + 1),
            Tf.charpoly.coeff k • Tf ^ (k + n)
        = ∑ k ∈ Finset.range (Tf.charpoly.natDegree + 1),
            Tf.charpoly.coeff k • (Tf ^ k * Tf ^ n) := by
          refine Finset.sum_congr rfl ?_
          intro k _; rw [pow_add]
      _ = (∑ k ∈ Finset.range (Tf.charpoly.natDegree + 1),
            Tf.charpoly.coeff k • Tf ^ k) * Tf ^ n := by
          rw [Finset.sum_mul]
          refine Finset.sum_congr rfl ?_
          intro k _; exact (Matrix.smul_mul _ _ _).symm
      _ = 0 * Tf ^ n := by rw [hCH]
      _ = 0 := Matrix.zero_mul _
  have hDot : (fun i => f (u i)) ⬝ᵥ ((∑ k ∈ Finset.range (Tf.charpoly.natDegree + 1),
      Tf.charpoly.coeff k • Tf ^ (k + n)) *ᵥ (fun i => f (v i))) = 0 := by
    rw [hMat, Matrix.zero_mulVec, dotProduct_zero]
  rw [Matrix.sum_mulVec, dotProduct_sum] at hDot
  convert hDot using 1
  refine Finset.sum_congr rfl ?_
  intro k _
  rw [Matrix.smul_mulVec, dotProduct_smul, smul_eq_mul]

/-- **Poles at the `[x^0]` slice** (`lem:poles`, the `d = 0` case,
recurrence-complete). Let `T : Matrix ι ι (Polynomial R)` with extremum-free
part `T_0 := T.map (Polynomial.evalRingHom 0)`. Then the constant-in-`x` slice
of the scalar transfer sequence — `c_n^{(0)} := (u ⋝ᵥ T^n *ᵥ v).eval 0` (equal
to `u_0 ⋝ᵥ T_0^n *ᵥ v_0` by `pow_map_ringHom` + `mulVec_map_ringHom` +
`dotProduct_map_ringHom`) — satisfies the linear recurrence whose characteristic
polynomial is `T_0.charpoly`. Composing with `SequelRatGF.transfer_GF_rational`
upgrades this to: the generating function of `c_n^{(0)}` has poles only among
the reciprocal roots of `T_0.charpoly` (i.e., reciprocal eigenvalues of `T_0`)
— the `d = 0` instance of `lem:poles`. -/
theorem poles_at_x_zero {R : Type*} [CommRing R] {ι : Type*}
    [Fintype ι] [DecidableEq ι]
    (T : Matrix ι ι (Polynomial R)) (u v : ι → Polynomial R) (n : ℕ) :
    ∑ k ∈ Finset.range
        ((T.map (Polynomial.evalRingHom 0)).charpoly.natDegree + 1),
      (T.map (Polynomial.evalRingHom 0)).charpoly.coeff k *
        ((fun i => (u i).eval 0) ⬝ᵥ
          (((T.map (Polynomial.evalRingHom 0)) ^ (k + n)) *ᵥ (fun i => (v i).eval 0))) = 0 := by
  -- Re-derive in-place rather than call `transfer_recurrence_map` to avoid the
  -- same kernel `whnf` timeout that motivates the latter's standalone proof.
  set T0 := T.map (Polynomial.evalRingHom 0)
  have hCH : (Polynomial.aeval T0) T0.charpoly = 0 := Matrix.aeval_self_charpoly _
  have hExpand : (Polynomial.aeval T0) T0.charpoly
      = ∑ k ∈ Finset.range (T0.charpoly.natDegree + 1),
          T0.charpoly.coeff k • T0 ^ k := Polynomial.aeval_eq_sum_range _
  rw [hExpand] at hCH
  have hMat : ∑ k ∈ Finset.range (T0.charpoly.natDegree + 1),
      T0.charpoly.coeff k • T0 ^ (k + n) = 0 := by
    calc ∑ k ∈ Finset.range (T0.charpoly.natDegree + 1),
            T0.charpoly.coeff k • T0 ^ (k + n)
        = ∑ k ∈ Finset.range (T0.charpoly.natDegree + 1),
            T0.charpoly.coeff k • (T0 ^ k * T0 ^ n) := by
          refine Finset.sum_congr rfl ?_
          intro k _; rw [pow_add]
      _ = (∑ k ∈ Finset.range (T0.charpoly.natDegree + 1),
            T0.charpoly.coeff k • T0 ^ k) * T0 ^ n := by
          rw [Finset.sum_mul]
          refine Finset.sum_congr rfl ?_
          intro k _; exact (Matrix.smul_mul _ _ _).symm
      _ = 0 * T0 ^ n := by rw [hCH]
      _ = 0 := Matrix.zero_mul _
  have hDot : (fun i => (u i).eval 0) ⬝ᵥ
      ((∑ k ∈ Finset.range (T0.charpoly.natDegree + 1),
          T0.charpoly.coeff k • T0 ^ (k + n)) *ᵥ (fun i => (v i).eval 0)) = 0 := by
    rw [hMat, Matrix.zero_mulVec, dotProduct_zero]
  rw [Matrix.sum_mulVec, dotProduct_sum] at hDot
  convert hDot using 1
  refine Finset.sum_congr rfl ?_
  intro k _
  rw [Matrix.smul_mulVec, dotProduct_smul, smul_eq_mul]

end OrigamiCone.Sequel
