import Mathlib

namespace OrigamiCone.Sequel.CoefficientOnset

noncomputable def denominator (order : Nat) : Polynomial Rat :=
  (1 - Polynomial.X) ^ order

theorem denominator_degree (order : Nat) : (denominator order).natDegree = order := by
  have linearDegree : (1 - (Polynomial.X : Polynomial Rat)).natDegree = 1 := by
    rw [Polynomial.natDegree_sub]
    simpa only [Polynomial.C_1] using Polynomial.natDegree_X_sub_C (1 : Rat)
  rw [denominator, Polynomial.natDegree_pow, linearDegree, mul_one]

theorem denominator_nonzero (order : Nat) : denominator order ≠ 0 := by
  intro zero
  have equality := congrArg (Polynomial.eval (0 : Rat)) zero
  simp [denominator] at equality

theorem quotient_degree_bound (numerator : Polynomial Rat) (order : Nat)
    (orderPositive : 0 < order) (numeratorBound : numerator.natDegree ≤ 2 * order - 1) :
    (numerator / denominator order).natDegree ≤ order - 1 := by
  have remainderBound : (numerator % denominator order).natDegree < order := by
    simpa only [denominator_degree] using Polynomial.natDegree_mod_lt numerator
      (q := denominator order) (by rw [denominator_degree]; omega)
  by_cases zero : numerator / denominator order = 0
  · simp [zero]
  · have productDegree :
        (denominator order * (numerator / denominator order)).natDegree =
          order + (numerator / denominator order).natDegree := by
      rw [Polynomial.natDegree_mul (denominator_nonzero order) zero, denominator_degree]
    have smaller : (numerator % denominator order).natDegree <
        (denominator order * (numerator / denominator order)).natDegree := by
      rw [productDegree]
      omega
    have sumDegree := Polynomial.natDegree_add_eq_left_of_natDegree_lt smaller
    rw [EuclideanDomain.div_add_mod, productDegree] at sumDegree
    omega

theorem hilbert_degree_bound (numerator : Polynomial Rat) (order : Nat)
    (orderPositive : 0 < order) :
    (Polynomial.hilbertPoly numerator order).natDegree ≤ order - 1 := by
  cases order with
  | zero => omega
  | succ previous =>
    rw [Polynomial.hilbertPoly_succ]
    apply Polynomial.natDegree_sum_le_of_forall_le
    intro index _
    exact (Polynomial.natDegree_smul_le _ _).trans (by
      rw [Polynomial.natDegree_preHilbertPoly]
      omega)

theorem rational_coefficients_polynomial_from (series : PowerSeries Rat)
    (numerator : Polynomial Rat) (order : Nat) (orderPositive : 0 < order)
    (numeratorBound : numerator.natDegree ≤ 2 * order - 1)
    (identity : (1 - (PowerSeries.X : PowerSeries Rat)) ^ order * series = numerator) :
    ∃ polynomial : Polynomial Rat, polynomial.natDegree ≤ order - 1 ∧
      ∀ width : Nat, order ≤ width ->
        PowerSeries.coeff width series = polynomial.eval (width : Rat) := by
  let divisor := denominator order
  let quotient := numerator / divisor
  let remainder := numerator % divisor
  have quotientBound : quotient.natDegree ≤ order - 1 :=
    quotient_degree_bound numerator order orderPositive numeratorBound
  have remainderBound : remainder.natDegree < order := by
    simpa only [divisor, denominator_degree] using Polynomial.natDegree_mod_lt numerator
      (q := divisor) (by
        change (denominator order).natDegree ≠ 0
        rw [denominator_degree]
        omega)
  let inverse : PowerSeries Rat := (PowerSeries.invOneSubPow Rat order).val
  have inverseLaw : inverse * (1 - (PowerSeries.X : PowerSeries Rat)) ^ order = 1 := by
    change (PowerSeries.invOneSubPow Rat order).val *
      (1 - (PowerSeries.X : PowerSeries Rat)) ^ order = 1
    rw [← PowerSeries.invOneSubPow_inv_eq_one_sub_pow Rat order]
    exact (PowerSeries.invOneSubPow Rat order).val_inv
  have division : divisor * quotient + remainder = numerator :=
    EuclideanDomain.div_add_mod numerator divisor
  have divisionSeries :
      (1 - (PowerSeries.X : PowerSeries Rat)) ^ order * (quotient : PowerSeries Rat) +
        (remainder : PowerSeries Rat) = (numerator : PowerSeries Rat) := by
    have mapped := congrArg (Polynomial.coeToPowerSeries.ringHom) division
    simpa only [map_add, map_mul, Polynomial.coeToPowerSeries.ringHom_apply,
      divisor, denominator, Polynomial.coe_pow, Polynomial.coe_sub, Polynomial.coe_one,
      Polynomial.coe_X] using mapped
  have representation : series =
      (quotient : PowerSeries Rat) + (remainder : PowerSeries Rat) * inverse := by
    calc
      series = inverse * ((1 - (PowerSeries.X : PowerSeries Rat)) ^ order * series) := by
        rw [← mul_assoc, inverseLaw, one_mul]
      _ = inverse * (numerator : PowerSeries Rat) := by rw [identity]
      _ = inverse * ((1 - (PowerSeries.X : PowerSeries Rat)) ^ order *
          (quotient : PowerSeries Rat) + (remainder : PowerSeries Rat)) := by rw [divisionSeries]
      _ = _ := by
        rw [mul_add, ← mul_assoc, inverseLaw, one_mul, mul_comm inverse (remainder : PowerSeries Rat)]
  refine ⟨Polynomial.hilbertPoly remainder order, hilbert_degree_bound remainder order orderPositive, ?_⟩
  intro width widthValid
  rw [representation, map_add, Polynomial.coeff_coe,
    Polynomial.coeff_eq_zero_of_natDegree_lt (by omega), zero_add]
  exact Polynomial.coeff_mul_invOneSubPow_eq_hilbertPoly_eval order (by omega)

#print axioms denominator_degree
#print axioms denominator_nonzero
#print axioms quotient_degree_bound
#print axioms hilbert_degree_bound
#print axioms rational_coefficients_polynomial_from

end OrigamiCone.Sequel.CoefficientOnset
