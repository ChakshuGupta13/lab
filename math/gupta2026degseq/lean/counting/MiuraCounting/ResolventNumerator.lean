import Mathlib

namespace OrigamiCone.Sequel.ResolventNumerator

open Polynomial Matrix

variable {Scalar Index : Type*} [CommRing Scalar]

noncomputable def constantVector (values : Index -> Scalar) : Index -> Polynomial Scalar :=
  fun index => Polynomial.C (values index)

noncomputable def constantMatrix (matrix : Matrix Index Index Scalar) :
    Matrix Index Index (Polynomial Scalar) :=
  fun source target => Polynomial.C (matrix source target)

def BoundedVector (values : Index -> Polynomial Scalar) (bound : Nat) : Prop :=
  ∀ index, (values index).natDegree ≤ bound

theorem bounded_constant (values : Index -> Scalar) :
    BoundedVector (constantVector values) 0 := by
  intro index
  simp [constantVector]

theorem bounded_add {left right : Index -> Polynomial Scalar} {bound : Nat}
    (leftBound : BoundedVector left bound) (rightBound : BoundedVector right bound) :
    BoundedVector (left + right) bound := by
  intro index
  exact Polynomial.natDegree_add_le_of_degree_le (leftBound index) (rightBound index)

theorem bounded_scale {values : Index -> Polynomial Scalar} {valueBound scalarBound : Nat}
    (polynomial : Polynomial Scalar) (polynomialBound : polynomial.natDegree ≤ scalarBound)
    (valuesBound : BoundedVector values valueBound) :
    BoundedVector (polynomial • values) (scalarBound + valueBound) := by
  intro index
  exact Polynomial.natDegree_mul_le.trans (Nat.add_le_add polynomialBound (valuesBound index))

theorem bounded_sum {Family : Type*} [Fintype Family]
    (values : Family -> Index -> Polynomial Scalar) (bound : Nat)
    (bounds : ∀ member, BoundedVector (values member) bound) :
    BoundedVector (∑ member, values member) bound := by
  intro index
  simp only [Finset.sum_apply]
  exact Polynomial.natDegree_sum_le_of_forall_le _ _ (fun member _ => bounds member index)

variable [Fintype Index]

theorem bounded_matrix_mulVec (matrix : Matrix Index Index (Polynomial Scalar))
    (values : Index -> Polynomial Scalar) (matrixBound valueBound : Nat)
    (matrixBounds : ∀ source target, (matrix source target).natDegree ≤ matrixBound)
    (valuesBound : BoundedVector values valueBound) :
    BoundedVector (matrix.mulVec values) (matrixBound + valueBound) := by
  intro source
  apply Polynomial.natDegree_sum_le_of_forall_le
  intro target _
  exact Polynomial.natDegree_mul_le.trans
    (Nat.add_le_add (matrixBounds source target) (valuesBound target))

theorem bounded_constant_mulVec (matrix : Matrix Index Index Scalar)
    {values : Index -> Polynomial Scalar} {bound : Nat}
    (valuesBound : BoundedVector values bound) :
    BoundedVector ((constantMatrix matrix).mulVec values) bound := by
  simpa only [zero_add] using bounded_matrix_mulVec (constantMatrix matrix) values 0 bound
    (fun _ _ => by simp [constantMatrix]) valuesBound

variable [DecidableEq Index]

noncomputable def linearNumerator (complement : Matrix Index Index Scalar) :
    Matrix Index Index (Polynomial Scalar) :=
  1 - (Polynomial.X : Polynomial Scalar) • constantMatrix complement

omit [Fintype Index] in
theorem linearNumerator_degree (complement : Matrix Index Index Scalar) :
    ∀ source target, (linearNumerator complement source target).natDegree ≤ 1 := by
  intro source target
  have identityBound : ((1 : Matrix Index Index (Polynomial Scalar)) source target).natDegree ≤ 1 := by
    by_cases same : source = target <;> simp [Matrix.one_apply, same]
  have shiftedBound : (Polynomial.X * Polynomial.C (complement source target)).natDegree ≤ 1 := by
    exact Polynomial.natDegree_mul_le.trans (by
      simpa only [Polynomial.natDegree_C, add_zero] using
        (Polynomial.natDegree_X_le (R := Scalar)))
  exact (Polynomial.natDegree_sub_le _ _).trans (max_le identityBound shiftedBound)

omit [DecidableEq Index] in
theorem constant_mulVec (matrix : Matrix Index Index Scalar) (values : Index -> Scalar) :
    (constantMatrix matrix).mulVec (constantVector values) =
      constantVector (matrix.mulVec values) := by
  funext source
  simp [Matrix.mulVec, dotProduct, constantMatrix, constantVector, map_sum, map_mul]

theorem linearNumerator_fixed (complement : Matrix Index Index Scalar)
    (values : Index -> Scalar) (annihilated : complement.mulVec values = 0) :
    (linearNumerator complement).mulVec (constantVector values) = constantVector values := by
  have constantZero : constantVector (0 : Index -> Scalar) = 0 := by
    funext index
    exact map_zero Polynomial.C
  rw [linearNumerator, Matrix.sub_mulVec, Matrix.one_mulVec, Matrix.smul_mulVec,
    constant_mulVec, annihilated, constantZero, smul_zero, sub_zero]

noncomputable def rightNumerator (complement : Matrix Index Index Scalar)
    (positiveTransfer : Nat -> Matrix Index Index Scalar)
    (boundary : Nat -> Index -> Scalar) : Nat -> Index -> Polynomial Scalar
  | 0 => 0
  | degree + 1 =>
    (1 - (Polynomial.X : Polynomial Scalar)) ^ degree •
        (linearNumerator complement).mulVec (constantVector (boundary (degree + 1))) +
      (Polynomial.X : Polynomial Scalar) • (linearNumerator complement).mulVec
        (∑ factor : Fin degree,
          (1 - (Polynomial.X : Polynomial Scalar)) ^ factor.val •
            (constantMatrix (positiveTransfer (factor.val + 1))).mulVec
              (rightNumerator complement positiveTransfer boundary (degree - factor.val)))
termination_by degree => degree
decreasing_by omega

theorem rightNumerator_one (complement : Matrix Index Index Scalar)
    (positiveTransfer : Nat -> Matrix Index Index Scalar)
    (boundary : Nat -> Index -> Scalar)
    (singleFrozen : complement.mulVec (boundary 1) = 0) :
    rightNumerator complement positiveTransfer boundary 1 = constantVector (boundary 1) := by
  simp [rightNumerator, linearNumerator_fixed complement (boundary 1) singleFrozen]

theorem denominator_power_degree (exponent : Nat) :
    ((1 - (Polynomial.X : Polynomial Scalar)) ^ exponent).natDegree ≤ exponent := by
  have linearBound : (1 - (Polynomial.X : Polynomial Scalar)).natDegree ≤ 1 :=
    (Polynomial.natDegree_sub_le _ _).trans
      (max_le (by simp) Polynomial.natDegree_X_le)
  exact Polynomial.natDegree_pow_le.trans (by
    simpa only [mul_one] using Nat.mul_le_mul_left exponent linearBound)

theorem rightNumerator_degree (complement : Matrix Index Index Scalar)
    (positiveTransfer : Nat -> Matrix Index Index Scalar)
    (boundary : Nat -> Index -> Scalar)
    (singleFrozen : complement.mulVec (boundary 1) = 0) (degree : Nat) :
    0 < degree -> BoundedVector (rightNumerator complement positiveTransfer boundary degree)
      (2 * degree - 2) := by
  induction degree using Nat.strong_induction_on with
  | h degree inductionHypothesis =>
    intro positive
    cases degree with
    | zero => omega
    | succ previous =>
      by_cases first : previous = 0
      · subst previous
        rw [rightNumerator_one complement positiveTransfer boundary singleFrozen]
        exact bounded_constant _
      · have previousPositive : 0 < previous := Nat.pos_of_ne_zero first
        rw [rightNumerator]
        apply bounded_add
        · have coreBound := bounded_matrix_mulVec (linearNumerator complement)
            (constantVector (boundary (previous + 1))) 1 0
            (linearNumerator_degree complement) (bounded_constant _)
          have termBound := bounded_scale ((1 - (Polynomial.X : Polynomial Scalar)) ^ previous)
            (denominator_power_degree previous) coreBound
          intro state
          exact (termBound state).trans (by omega)
        · have summandBound : ∀ factor : Fin previous,
              BoundedVector
                ((1 - (Polynomial.X : Polynomial Scalar)) ^ factor.val •
                  (constantMatrix (positiveTransfer (factor.val + 1))).mulVec
                    (rightNumerator complement positiveTransfer boundary (previous - factor.val)))
                (2 * previous - 2) := by
            intro factor
            have remainingPositive : 0 < previous - factor.val := by omega
            have remainingLess : previous - factor.val < previous + 1 := by omega
            have smallerBound := inductionHypothesis (previous - factor.val)
              remainingLess remainingPositive
            have matrixBound := bounded_constant_mulVec
              (positiveTransfer (factor.val + 1)) smallerBound
            have termBound := bounded_scale
              ((1 - (Polynomial.X : Polynomial Scalar)) ^ factor.val)
              (denominator_power_degree factor.val) matrixBound
            intro state
            exact (termBound state).trans (by have := factor.isLt; omega)
          have sumBound := bounded_sum _ (2 * previous - 2) summandBound
          have matrixBound := bounded_matrix_mulVec (linearNumerator complement) _ 1
            (2 * previous - 2) (linearNumerator_degree complement) sumBound
          have termBound := bounded_scale (Polynomial.X : Polynomial Scalar)
            Polynomial.natDegree_X_le matrixBound
          intro state
          exact (termBound state).trans (by omega)

omit [DecidableEq Index] in
theorem constant_vecMul (values : Index -> Scalar) (matrix : Matrix Index Index Scalar) :
    Matrix.vecMul (constantVector values) (constantMatrix matrix) =
      constantVector (Matrix.vecMul values matrix) := by
  funext target
  simp [Matrix.vecMul, dotProduct, constantMatrix, constantVector, map_sum, map_mul]

theorem left_linearNumerator (complement : Matrix Index Index Scalar)
    (left : Index -> Scalar) (annihilated : Matrix.vecMul left complement = 0)
    (values : Index -> Polynomial Scalar) :
    dotProduct (constantVector left) ((linearNumerator complement).mulVec values) =
      dotProduct (constantVector left) values := by
  have constantZero : constantVector (0 : Index -> Scalar) = 0 := by
    funext index
    exact map_zero Polynomial.C
  have fixed : Matrix.vecMul (constantVector left) (linearNumerator complement) =
      constantVector left := by
    rw [linearNumerator, Matrix.vecMul_sub, Matrix.vecMul_one, Matrix.vecMul_smul,
      constant_vecMul, annihilated, constantZero, smul_zero, sub_zero]
  rw [Matrix.dotProduct_mulVec, fixed]

omit [DecidableEq Index] in
theorem bounded_dot (left : Index -> Scalar) {values : Index -> Polynomial Scalar}
    {bound : Nat} (valuesBound : BoundedVector values bound) :
    (dotProduct (constantVector left) values).natDegree ≤ bound := by
  apply Polynomial.natDegree_sum_le_of_forall_le
  intro index _
  exact (Polynomial.natDegree_C_mul_le _ _).trans (valuesBound index)

theorem rightNumerator_left_degree (complement : Matrix Index Index Scalar)
    (positiveTransfer : Nat -> Matrix Index Index Scalar)
    (boundary : Nat -> Index -> Scalar) (left : Index -> Scalar)
    (singleRight : complement.mulVec (boundary 1) = 0)
    (singleLeft : Matrix.vecMul left complement = 0) (degree : Nat) (degreeValid : 2 ≤ degree) :
    (dotProduct (constantVector left)
      (rightNumerator complement positiveTransfer boundary degree)).natDegree ≤ 2 * degree - 3 := by
  cases degree with
  | zero => omega
  | succ previous =>
    have previousPositive : 0 < previous := by omega
    rw [rightNumerator, dotProduct_add, dotProduct_smul, dotProduct_smul,
      left_linearNumerator complement left singleLeft,
      left_linearNumerator complement left singleLeft]
    simp only [smul_eq_mul]
    apply Polynomial.natDegree_add_le_of_degree_le
    · have innerBound := bounded_dot left (bounded_constant (boundary (previous + 1)))
      exact Polynomial.natDegree_mul_le.trans
        ((Nat.add_le_add (denominator_power_degree previous) innerBound).trans (by omega))
    · have summandBound : ∀ factor : Fin previous,
          BoundedVector
            ((1 - (Polynomial.X : Polynomial Scalar)) ^ factor.val •
              (constantMatrix (positiveTransfer (factor.val + 1))).mulVec
                (rightNumerator complement positiveTransfer boundary (previous - factor.val)))
            (2 * previous - 2) := by
        intro factor
        have remainingPositive : 0 < previous - factor.val := by omega
        have smallerBound := rightNumerator_degree complement positiveTransfer boundary
          singleRight (previous - factor.val) remainingPositive
        have matrixBound := bounded_constant_mulVec (positiveTransfer (factor.val + 1)) smallerBound
        have termBound := bounded_scale ((1 - (Polynomial.X : Polynomial Scalar)) ^ factor.val)
          (denominator_power_degree factor.val) matrixBound
        intro state
        exact (termBound state).trans (by have := factor.isLt; omega)
      have sumBound := bounded_sum _ (2 * previous - 2) summandBound
      have innerBound := bounded_dot left sumBound
      exact Polynomial.natDegree_mul_le.trans
        ((Nat.add_le_add Polynomial.natDegree_X_le innerBound).trans (by omega))

noncomputable def fullNumerator (complement : Matrix Index Index Scalar)
    (positiveTransfer : Nat -> Matrix Index Index Scalar)
    (left right : Nat -> Index -> Scalar) (degree : Nat) : Polynomial Scalar :=
  Polynomial.X ^ 2 * ∑ boundary : Fin (degree - 1),
    (1 - (Polynomial.X : Polynomial Scalar)) ^ boundary.val *
      dotProduct (constantVector (left (boundary.val + 1)))
        (rightNumerator complement positiveTransfer right (degree - 1 - boundary.val))

theorem fullNumerator_degree (complement : Matrix Index Index Scalar)
    (positiveTransfer : Nat -> Matrix Index Index Scalar)
    (left right : Nat -> Index -> Scalar)
    (singleRight : complement.mulVec (right 1) = 0)
    (singleLeft : Matrix.vecMul (left 1) complement = 0) (degree : Nat)
    (degreeValid : 3 ≤ degree) :
    (fullNumerator complement positiveTransfer left right degree).natDegree ≤ 2 * degree - 3 := by
  have sumBound :
      (∑ boundary : Fin (degree - 1),
        (1 - (Polynomial.X : Polynomial Scalar)) ^ boundary.val *
          dotProduct (constantVector (left (boundary.val + 1)))
            (rightNumerator complement positiveTransfer right
              (degree - 1 - boundary.val))).natDegree ≤ 2 * degree - 5 := by
    apply Polynomial.natDegree_sum_le_of_forall_le
    intro boundary _
    by_cases first : boundary.val = 0
    · have remainingValid : 2 ≤ degree - 1 := by omega
      have innerBound := rightNumerator_left_degree complement positiveTransfer right (left 1)
        singleRight singleLeft (degree - 1) remainingValid
      simpa only [first, pow_zero, one_mul, zero_add, Nat.sub_zero] using
        innerBound.trans (by omega)
    · have remainingPositive : 0 < degree - 1 - boundary.val := by have := boundary.isLt; omega
      have valuesBound := rightNumerator_degree complement positiveTransfer right singleRight
        (degree - 1 - boundary.val) remainingPositive
      have innerBound := bounded_dot (left (boundary.val + 1)) valuesBound
      exact Polynomial.natDegree_mul_le.trans
        ((Nat.add_le_add (denominator_power_degree boundary.val) innerBound).trans (by
          have := boundary.isLt
          omega))
  exact Polynomial.natDegree_mul_le.trans
    ((Nat.add_le_add (Polynomial.natDegree_X_pow_le 2) sumBound).trans (by omega))

#print axioms bounded_constant
#print axioms bounded_add
#print axioms bounded_scale
#print axioms bounded_sum
#print axioms bounded_matrix_mulVec
#print axioms bounded_constant_mulVec
#print axioms linearNumerator_degree
#print axioms constant_mulVec
#print axioms linearNumerator_fixed
#print axioms rightNumerator_one
#print axioms denominator_power_degree
#print axioms rightNumerator_degree
#print axioms constant_vecMul
#print axioms left_linearNumerator
#print axioms bounded_dot
#print axioms rightNumerator_left_degree
#print axioms fullNumerator_degree

end OrigamiCone.Sequel.ResolventNumerator
