import MiuraCounting.ConcreteRotation
import MiuraCounting.ConcreteEndpointCancellation
import MiuraCounting.ResolventNumerator
import OrigamiCone.SequelEdMatrixPower

namespace OrigamiCone.Sequel.ConcreteTransfer

theorem transfer_polynomial_preserves_rotation {rows : Nat}
    (values : TransferState rows -> Polynomial Int)
    (invariant : ∀ source slope, values (rotateState source slope) = values source) :
    ∀ source slope,
      Matrix.mulVec (transferMatrix rows) values (rotateState source slope) =
        Matrix.mulVec (transferMatrix rows) values source := by
  intro source slope
  change (∑ target, transferMatrix rows (rotateState source slope) target * values target) =
    ∑ target, transferMatrix rows source target * values target
  calc
    _ = ∑ target, transferMatrix rows (rotateState source slope) (rotateState target slope) *
        values (rotateState target slope) :=
      (Equiv.sum_comp (rotateStateEquiv slope) (fun target =>
        transferMatrix rows (rotateState source slope) target * values target)).symm
    _ = _ := Finset.sum_congr rfl (fun target _ => by
      rw [transferMatrix_rotate, invariant target slope])

noncomputable def propagatedRight (rows width : Nat) : TransferState rows -> Polynomial Int :=
  Matrix.mulVec ((transferMatrix rows) ^ width) (rightBdyVec rows)

theorem propagatedRight_zero (rows : Nat) : propagatedRight rows 0 = rightBdyVec rows := by
  simp [propagatedRight]

theorem propagatedRight_succ (rows width : Nat) :
    propagatedRight rows (width + 1) =
      Matrix.mulVec (transferMatrix rows) (propagatedRight rows width) := by
  simp only [propagatedRight, pow_succ', Matrix.mulVec_mulVec]

theorem propagatedRight_rotation (rows width : Nat) :
    ∀ source slope, propagatedRight rows width (rotateState source slope) =
      propagatedRight rows width source := by
  induction width with
  | zero => simpa only [propagatedRight_zero] using rightBdyVec_rotate
  | succ width inductionHypothesis =>
    rw [propagatedRight_succ]
    exact transfer_polynomial_preserves_rotation _ inductionHypothesis

noncomputable def propagatedCoefficient (rows degree width : Nat) : TransferState rows -> Rat :=
  fun source => ((propagatedRight rows width source).coeff degree : Rat)

theorem propagatedCoefficient_succ (rows degree width : Nat) (source : TransferState rows) :
    propagatedCoefficient rows degree (width + 1) source =
      ∑ split ∈ Finset.antidiagonal degree,
        Matrix.mulVec (transferCoefficient rows split.1)
          (propagatedCoefficient rows split.2 width) source := by
  unfold propagatedCoefficient
  rw [propagatedRight_succ]
  change (((∑ target, transferMatrix rows source target * propagatedRight rows width target).coeff
    degree : Int) : Rat) = _
  rw [Polynomial.finset_sum_coeff]
  push_cast
  simp_rw [Polynomial.coeff_mul]
  push_cast
  simp only [Matrix.mulVec, dotProduct, transferCoefficient]
  exact Finset.sum_comm

theorem propagatedCoefficient_zero {rows : Nat} (rowsPositive : 0 < rows) (width : Nat) :
    propagatedCoefficient rows 0 width = 0 := by
  induction width with
  | zero =>
    funext source
    simp only [propagatedCoefficient, propagatedRight_zero,
      (endpoint_constant_zero rowsPositive source).2, Int.cast_zero, Pi.zero_apply]
  | succ width inductionHypothesis =>
    funext source
    rw [propagatedCoefficient_succ]
    simp only [Finset.antidiagonal_zero, Finset.sum_singleton, inductionHypothesis,
      Matrix.mulVec_zero, Pi.zero_apply]

noncomputable def rightCoefficientSeries (rows degree : Nat) :
    TransferState rows -> PowerSeries Rat :=
  fun source => PowerSeries.mk fun width => propagatedCoefficient rows degree width source

theorem rightCoefficientSeries_initial (rows degree : Nat) (source : TransferState rows) :
    PowerSeries.coeff 0 (rightCoefficientSeries rows degree source) =
      ((rightBdyVec rows source).coeff degree : Rat) := by
  simp only [rightCoefficientSeries, PowerSeries.coeff_mk, propagatedCoefficient,
    propagatedRight_zero]

theorem rightCoefficientSeries_rotation (rows degree : Nat)
    (source : TransferState rows) (slope : ZMod 3) :
    rightCoefficientSeries rows degree (rotateState source slope) =
      rightCoefficientSeries rows degree source := by
  apply PowerSeries.ext
  intro width
  simp only [rightCoefficientSeries, PowerSeries.coeff_mk, propagatedCoefficient,
    propagatedRight_rotation]

theorem rightCoefficientSeries_zero {rows : Nat} (rowsPositive : 0 < rows) :
    rightCoefficientSeries rows 0 = 0 := by
  funext source
  apply PowerSeries.ext
  intro width
  simp only [rightCoefficientSeries, PowerSeries.coeff_mk,
    propagatedCoefficient_zero rowsPositive width, Pi.zero_apply, map_zero]

noncomputable def leftBoundaryCoefficient (rows degree : Nat) : TransferState rows -> Rat :=
  fun source => ((leftBdyVec rows source).coeff degree : Rat) / 3

noncomputable def rightBoundaryCoefficient (rows degree : Nat) : TransferState rows -> Rat :=
  fun source => ((rightBdyVec rows source).coeff degree : Rat)

theorem concrete_numerator_degree {rows : Nat} (rowsValid : 2 ≤ rows)
    (degree : Nat) (degreeValid : 3 ≤ degree) :
    (ResolventNumerator.fullNumerator (1 - frozenProjection rows)
      (transferCoefficient rows) (leftBoundaryCoefficient rows) (rightBoundaryCoefficient rows)
      degree).natDegree ≤ 2 * degree - 3 := by
  apply ResolventNumerator.fullNumerator_degree
  · exact right_single_complement_zero rowsValid
  · exact left_single_complement_zero rowsValid
  · exact degreeValid

noncomputable def seriesConstantMatrix {Index : Type*} (matrix : Matrix Index Index Rat) :
    Matrix Index Index (PowerSeries Rat) :=
  fun source target => PowerSeries.C (matrix source target)

theorem series_mulVec_coeff {Index : Type*} [Fintype Index]
    (matrix : Matrix Index Index Rat) (values : Index -> PowerSeries Rat)
    (width : Nat) (source : Index) :
    PowerSeries.coeff width ((seriesConstantMatrix matrix).mulVec values source) =
      matrix.mulVec (fun target => PowerSeries.coeff width (values target)) source := by
  simp [seriesConstantMatrix, Matrix.mulVec, dotProduct, map_sum]

theorem rightCoefficientSeries_equation (rows degree : Nat) :
    rightCoefficientSeries rows degree = fun source =>
      PowerSeries.C (((rightBdyVec rows source).coeff degree : Int) : Rat) +
        PowerSeries.X * ∑ split ∈ Finset.antidiagonal degree,
          (seriesConstantMatrix (transferCoefficient rows split.1)).mulVec
            (rightCoefficientSeries rows split.2) source := by
  funext source
  apply PowerSeries.ext
  intro width
  cases width with
  | zero => simp [rightCoefficientSeries_initial]
  | succ width =>
    simpa only [rightCoefficientSeries, PowerSeries.coeff_mk, map_add,
      PowerSeries.coeff_C, Nat.succ_ne_zero, if_false, zero_add,
      PowerSeries.coeff_succ_X_mul, map_sum, series_mulVec_coeff] using
        propagatedCoefficient_succ rows degree width source

theorem series_zeroTransfer_projection {rows : Nat} (rowsValid : 2 ≤ rows)
    (values : TransferState rows -> PowerSeries Rat)
    (invariant : ∀ source slope, values (rotateState source slope) = values source) :
    (seriesConstantMatrix (zeroTransferMatrix rows)).mulVec values =
      (seriesConstantMatrix (frozenProjection rows)).mulVec values := by
  funext source
  apply PowerSeries.ext
  intro width
  rw [series_mulVec_coeff, series_mulVec_coeff]
  have coefficientInvariant : ∀ source slope,
      PowerSeries.coeff width (values (rotateState source slope)) =
        PowerSeries.coeff width (values source) := by
    intro state slope
    rw [invariant state slope]
  simpa only [frozenProjection, Matrix.mulVec_diagonal, ite_mul, one_mul, zero_mul] using
    zeroTransfer_action rowsValid (fun target => PowerSeries.coeff width (values target))
      coefficientInvariant source

noncomputable def positiveCoefficientConvolution (rows degree : Nat) :
    TransferState rows -> PowerSeries Rat :=
  ∑ factor : Fin degree,
    (seriesConstantMatrix (transferCoefficient rows (factor.val + 1))).mulVec
      (rightCoefficientSeries rows (degree - factor.val))

theorem rightCoefficientSeries_positive {rows : Nat} (rowsPositive : 0 < rows) (degree : Nat) :
    rightCoefficientSeries rows (degree + 1) = fun source =>
      PowerSeries.C (rightBoundaryCoefficient rows (degree + 1) source) +
        PowerSeries.X *
          ((seriesConstantMatrix (zeroTransferMatrix rows)).mulVec
              (rightCoefficientSeries rows (degree + 1)) source +
            positiveCoefficientConvolution rows degree source) := by
  funext source
  have splitSum :
      (∑ split ∈ Finset.antidiagonal (degree + 1),
        (seriesConstantMatrix (transferCoefficient rows split.1)).mulVec
          (rightCoefficientSeries rows split.2) source) =
        (seriesConstantMatrix (zeroTransferMatrix rows)).mulVec
            (rightCoefficientSeries rows (degree + 1)) source +
          positiveCoefficientConvolution rows degree source := by
    rw [Finset.Nat.sum_antidiagonal_eq_sum_range_succ_mk]
    rw [Finset.sum_range_succ', Finset.sum_range_succ]
    simp only [Nat.add_sub_add_right, Nat.sub_self, rightCoefficientSeries_zero rowsPositive,
      Matrix.mulVec_zero, Pi.zero_apply, add_zero, Nat.sub_zero]
    unfold positiveCoefficientConvolution
    rw [Finset.sum_apply, Fin.sum_univ_eq_sum_range (fun factor =>
      (seriesConstantMatrix (transferCoefficient rows (factor + 1))).mulVec
        (rightCoefficientSeries rows (degree - factor)) source) degree]
    exact add_comm _ _
  have equation := congrFun (rightCoefficientSeries_equation rows (degree + 1)) source
  rw [splitSum] at equation
  exact equation

theorem series_frozenProjection_action {rows : Nat}
    (values : TransferState rows -> PowerSeries Rat) (source : TransferState rows) :
    (seriesConstantMatrix (frozenProjection rows)).mulVec values source =
      if frozenState source then values source else 0 := by
  apply PowerSeries.ext
  intro width
  rw [series_mulVec_coeff]
  by_cases frozen : frozenState source <;>
    simp [frozenProjection, Matrix.mulVec_diagonal, frozen]

theorem series_complement_action {rows : Nat}
    (values : TransferState rows -> PowerSeries Rat) (source : TransferState rows) :
    (seriesConstantMatrix (1 - frozenProjection rows)).mulVec values source =
      if frozenState source then 0 else values source := by
  apply PowerSeries.ext
  intro width
  rw [series_mulVec_coeff]
  by_cases frozen : frozenState source <;>
    simp [frozenProjection, Matrix.sub_mulVec, Matrix.mulVec_diagonal, frozen]

noncomputable def seriesLinearNumerator (rows : Nat) :
    Matrix (TransferState rows) (TransferState rows) (PowerSeries Rat) :=
  1 - (PowerSeries.X : PowerSeries Rat) • seriesConstantMatrix (1 - frozenProjection rows)

theorem seriesLinearNumerator_action {rows : Nat}
    (values : TransferState rows -> PowerSeries Rat) (source : TransferState rows) :
    (seriesLinearNumerator rows).mulVec values source =
      if frozenState source then values source else (1 - PowerSeries.X) * values source := by
  simp only [seriesLinearNumerator, Matrix.sub_mulVec, Matrix.one_mulVec, Matrix.smul_mulVec,
    Pi.sub_apply, Pi.smul_apply, smul_eq_mul, series_complement_action]
  by_cases frozen : frozenState source <;> simp only [frozen, if_true, if_false] <;> ring

theorem rightCoefficientSeries_reduced {rows : Nat} (rowsValid : 2 ≤ rows) (degree : Nat) :
    (1 - (PowerSeries.X : PowerSeries Rat)) • rightCoefficientSeries rows (degree + 1) =
      (seriesLinearNumerator rows).mulVec (fun source =>
        PowerSeries.C (rightBoundaryCoefficient rows (degree + 1) source) +
          PowerSeries.X * positiveCoefficientConvolution rows degree source) := by
  funext source
  have rowsPositive : 0 < rows := by omega
  have equation := congrFun (rightCoefficientSeries_positive rowsPositive degree) source
  have projected := congrFun (series_zeroTransfer_projection rowsValid
    (rightCoefficientSeries rows (degree + 1)) (rightCoefficientSeries_rotation rows (degree + 1))) source
  rw [projected, series_frozenProjection_action] at equation
  rw [seriesLinearNumerator_action]
  change (1 - PowerSeries.X) * rightCoefficientSeries rows (degree + 1) source = _
  by_cases frozen : frozenState source
  · simp only [frozen, if_true] at equation ⊢
    linear_combination equation
  · simp only [frozen, if_false, zero_add] at equation ⊢
    rw [equation]

noncomputable def seriesConstantVector {Index : Type*} (values : Index -> Rat) :
    Index -> PowerSeries Rat :=
  fun index => PowerSeries.C (values index)

noncomputable def polynomialSeriesVector {Index : Type*} (values : Index -> Polynomial Rat) :
    Index -> PowerSeries Rat :=
  fun index => Polynomial.coeToPowerSeries.ringHom (values index)

theorem polynomial_series_sum {Family : Type*} (members : Finset Family)
    (values : Family -> Polynomial Rat) :
    ((∑ member ∈ members, values member : Polynomial Rat) : PowerSeries Rat) =
      ∑ member ∈ members, (values member : PowerSeries Rat) :=
  map_sum Polynomial.coeToPowerSeries.ringHom values members

theorem polynomial_numerator_series_step (rows degree : Nat) :
    polynomialSeriesVector
        (ResolventNumerator.rightNumerator (1 - frozenProjection rows) (transferCoefficient rows)
          (rightBoundaryCoefficient rows) (degree + 1)) =
      (1 - (PowerSeries.X : PowerSeries Rat)) ^ degree •
          (seriesLinearNumerator rows).mulVec
            (seriesConstantVector (rightBoundaryCoefficient rows (degree + 1))) +
        (PowerSeries.X : PowerSeries Rat) • (seriesLinearNumerator rows).mulVec
          (∑ factor : Fin degree,
            (1 - (PowerSeries.X : PowerSeries Rat)) ^ factor.val •
              (seriesConstantMatrix (transferCoefficient rows (factor.val + 1))).mulVec
                (polynomialSeriesVector
                  (ResolventNumerator.rightNumerator (1 - frozenProjection rows)
                    (transferCoefficient rows) (rightBoundaryCoefficient rows)
                    (degree - factor.val)))) := by
  funext source
  simp [polynomialSeriesVector, ResolventNumerator.rightNumerator,
    ResolventNumerator.linearNumerator, ResolventNumerator.constantVector,
    ResolventNumerator.constantMatrix, seriesConstantVector, seriesConstantMatrix,
    seriesLinearNumerator, Matrix.mulVec, dotProduct, polynomial_series_sum, Matrix.one_apply,
    Polynomial.coeToPowerSeries.ringHom_apply, smul_eq_mul, apply_ite]

theorem rightCoefficientSeries_numerator_identity {rows : Nat} (rowsValid : 2 ≤ rows)
    (degree : Nat) :
    (1 - (PowerSeries.X : PowerSeries Rat)) ^ degree • rightCoefficientSeries rows degree =
      polynomialSeriesVector
        (ResolventNumerator.rightNumerator (1 - frozenProjection rows) (transferCoefficient rows)
          (rightBoundaryCoefficient rows) degree) := by
  induction degree using Nat.strong_induction_on with
  | h degree inductionHypothesis =>
    cases degree with
    | zero =>
      have rowsPositive : 0 < rows := by omega
      funext source
      simp [rightCoefficientSeries_zero rowsPositive, ResolventNumerator.rightNumerator,
        polynomialSeriesVector]
    | succ previous =>
      have convolutionIdentity :
          (1 - (PowerSeries.X : PowerSeries Rat)) ^ previous •
              positiveCoefficientConvolution rows previous =
            ∑ factor : Fin previous,
              (1 - (PowerSeries.X : PowerSeries Rat)) ^ factor.val •
                (seriesConstantMatrix (transferCoefficient rows (factor.val + 1))).mulVec
                  (polynomialSeriesVector
                    (ResolventNumerator.rightNumerator (1 - frozenProjection rows)
                      (transferCoefficient rows) (rightBoundaryCoefficient rows)
                      (previous - factor.val))) := by
        unfold positiveCoefficientConvolution
        rw [Finset.smul_sum]
        apply Finset.sum_congr rfl
        intro factor _
        have powerFactor : (1 - (PowerSeries.X : PowerSeries Rat)) ^ previous =
            (1 - PowerSeries.X) ^ factor.val * (1 - PowerSeries.X) ^ (previous - factor.val) := by
          rw [← pow_add]
          congr 1
          have := factor.isLt
          omega
        rw [← Matrix.mulVec_smul, powerFactor, ← smul_smul,
          inductionHypothesis (previous - factor.val) (by omega), Matrix.mulVec_smul]
      have reduced := rightCoefficientSeries_reduced rowsValid previous
      change (1 - (PowerSeries.X : PowerSeries Rat)) • rightCoefficientSeries rows (previous + 1) =
        (seriesLinearNumerator rows).mulVec
          (seriesConstantVector (rightBoundaryCoefficient rows (previous + 1)) +
            (PowerSeries.X : PowerSeries Rat) • positiveCoefficientConvolution rows previous) at reduced
      calc
        _ = (1 - (PowerSeries.X : PowerSeries Rat)) ^ previous •
            ((1 - (PowerSeries.X : PowerSeries Rat)) • rightCoefficientSeries rows (previous + 1)) := by
          rw [smul_smul, ← pow_succ]
        _ = (1 - (PowerSeries.X : PowerSeries Rat)) ^ previous •
            (seriesLinearNumerator rows).mulVec
              (seriesConstantVector (rightBoundaryCoefficient rows (previous + 1)) +
                (PowerSeries.X : PowerSeries Rat) • positiveCoefficientConvolution rows previous) := by
          rw [reduced]
        _ = (1 - (PowerSeries.X : PowerSeries Rat)) ^ previous •
              (seriesLinearNumerator rows).mulVec
                (seriesConstantVector (rightBoundaryCoefficient rows (previous + 1))) +
            (PowerSeries.X : PowerSeries Rat) • (seriesLinearNumerator rows).mulVec
              ((1 - (PowerSeries.X : PowerSeries Rat)) ^ previous •
                positiveCoefficientConvolution rows previous) := by
          rw [Matrix.mulVec_add, Matrix.mulVec_smul, smul_add, Matrix.mulVec_smul]
          congr 1
          exact smul_comm _ _ _
        _ = _ := by
          rw [convolutionIdentity, ← polynomial_numerator_series_step]

theorem counted_coefficient_identity {rows : Nat} (rowsPositive : 0 < rows)
    (degree width : Nat) :
    (Ed degree rows (width + 2) : Rat) =
      ∑ split ∈ Finset.antidiagonal degree,
        dotProduct (leftBoundaryCoefficient rows split.1)
          (propagatedCoefficient rows split.2 width) := by
  have countIdentity :
      (3 : Polynomial Int) * cnPoly rows (width + 2) =
        ∑ source, leftBdyVec rows source * propagatedRight rows width source := by
    rw [cnPoly_eq_matrix_power rows width rowsPositive]
    exact three_times_normalized_sandwich rowsPositive (propagatedRight rows width)
      (propagatedRight_rotation rows width)
  have coefficients := congrArg
    (fun polynomial : Polynomial Int => ((polynomial.coeff degree : Int) : Rat)) countIdentity
  simp only [Polynomial.coeff_ofNat_mul, Polynomial.finset_sum_coeff,
    Polynomial.coeff_mul] at coefficients
  push_cast at coefficients
  have averaged :
      (3 : Rat) *
          (∑ split ∈ Finset.antidiagonal degree,
            dotProduct (leftBoundaryCoefficient rows split.1)
              (propagatedCoefficient rows split.2 width)) =
        ∑ source, ∑ split ∈ Finset.antidiagonal degree,
          ((leftBdyVec rows source).coeff split.1 : Rat) *
            ((propagatedRight rows width source).coeff split.2 : Rat) := by
    simp only [dotProduct, leftBoundaryCoefficient, propagatedCoefficient, Finset.mul_sum]
    rw [Finset.sum_comm]
    apply Finset.sum_congr rfl
    intro source _
    apply Finset.sum_congr rfl
    intro split _
    ring
  have result : ((cnPoly rows (width + 2)).coeff degree : Rat) =
      ∑ split ∈ Finset.antidiagonal degree,
        dotProduct (leftBoundaryCoefficient rows split.1)
          (propagatedCoefficient rows split.2 width) := by
    linarith [coefficients, averaged]
  rw [← Ed_eq_cnPoly_coeff] at result
  simpa only [Int.cast_natCast] using result

noncomputable def degreeCountWidthSeries (rows degree : Nat) : PowerSeries Rat :=
  PowerSeries.mk fun width => if 2 ≤ width then (Ed degree rows width : Rat) else 0

theorem degreeCountWidthSeries_identity {rows : Nat} (rowsPositive : 0 < rows) (degree : Nat) :
    degreeCountWidthSeries rows degree =
      PowerSeries.X ^ 2 * ∑ split ∈ Finset.antidiagonal degree,
        dotProduct (seriesConstantVector (leftBoundaryCoefficient rows split.1))
          (rightCoefficientSeries rows split.2) := by
  apply PowerSeries.ext
  intro width
  rw [PowerSeries.coeff_X_pow_mul']
  simp only [degreeCountWidthSeries, PowerSeries.coeff_mk]
  by_cases valid : 2 ≤ width
  · simp only [valid, if_true, map_sum, dotProduct, seriesConstantVector,
      PowerSeries.coeff_C_mul, rightCoefficientSeries, PowerSeries.coeff_mk]
    have shifted : width - 2 + 2 = width := by omega
    simpa only [shifted, dotProduct] using counted_coefficient_identity rowsPositive degree (width - 2)
  · simp only [valid, if_false]

theorem degreeCountWidthSeries_positive {rows : Nat} (rowsPositive : 0 < rows)
    (degree : Nat) (degreePositive : 0 < degree) :
    degreeCountWidthSeries rows degree =
      PowerSeries.X ^ 2 * ∑ boundary : Fin (degree - 1),
        dotProduct (seriesConstantVector (leftBoundaryCoefficient rows (boundary.val + 1)))
          (rightCoefficientSeries rows (degree - 1 - boundary.val)) := by
  cases degree with
  | zero => omega
  | succ previous =>
    rw [degreeCountWidthSeries_identity rowsPositive]
    apply congrArg (fun tail : PowerSeries Rat => PowerSeries.X ^ 2 * tail)
    have leftZero : seriesConstantVector (leftBoundaryCoefficient rows 0) = 0 := by
      funext source
      simp [seriesConstantVector, leftBoundaryCoefficient,
        (endpoint_constant_zero rowsPositive source).1]
    rw [Finset.Nat.sum_antidiagonal_eq_sum_range_succ_mk,
      Finset.sum_range_succ', Finset.sum_range_succ]
    simp only [Nat.add_sub_add_right, Nat.sub_self, rightCoefficientSeries_zero rowsPositive,
      dotProduct_zero, leftZero, zero_dotProduct, add_zero, Nat.sub_zero, Nat.add_sub_cancel]
    exact (Fin.sum_univ_eq_sum_range (fun boundary =>
      dotProduct (seriesConstantVector (leftBoundaryCoefficient rows (boundary + 1)))
        (rightCoefficientSeries rows (previous - boundary))) previous).symm

theorem full_numerator_series_map (rows degree : Nat) :
    ((ResolventNumerator.fullNumerator (1 - frozenProjection rows) (transferCoefficient rows)
        (leftBoundaryCoefficient rows) (rightBoundaryCoefficient rows) degree : Polynomial Rat) :
        PowerSeries Rat) =
      PowerSeries.X ^ 2 * ∑ boundary : Fin (degree - 1),
        (1 - (PowerSeries.X : PowerSeries Rat)) ^ boundary.val *
          dotProduct (seriesConstantVector (leftBoundaryCoefficient rows (boundary.val + 1)))
            (polynomialSeriesVector
              (ResolventNumerator.rightNumerator (1 - frozenProjection rows)
                (transferCoefficient rows) (rightBoundaryCoefficient rows)
                (degree - 1 - boundary.val))) := by
  simp [ResolventNumerator.fullNumerator, ResolventNumerator.constantVector, polynomial_series_sum,
    seriesConstantVector, polynomialSeriesVector, dotProduct,
    Polynomial.coeToPowerSeries.ringHom_apply]

theorem degreeCountWidthSeries_numerator_identity {rows : Nat} (rowsValid : 2 ≤ rows)
    (degree : Nat) (degreePositive : 0 < degree) :
    (1 - (PowerSeries.X : PowerSeries Rat)) ^ (degree - 1) * degreeCountWidthSeries rows degree =
      (ResolventNumerator.fullNumerator (1 - frozenProjection rows) (transferCoefficient rows)
        (leftBoundaryCoefficient rows) (rightBoundaryCoefficient rows) degree : Polynomial Rat) := by
  have rowsPositive : 0 < rows := by omega
  rw [degreeCountWidthSeries_positive rowsPositive degree degreePositive, full_numerator_series_map]
  have sumIdentity :
      (1 - (PowerSeries.X : PowerSeries Rat)) ^ (degree - 1) *
          (∑ boundary : Fin (degree - 1),
            dotProduct (seriesConstantVector (leftBoundaryCoefficient rows (boundary.val + 1)))
              (rightCoefficientSeries rows (degree - 1 - boundary.val))) =
        ∑ boundary : Fin (degree - 1),
          (1 - (PowerSeries.X : PowerSeries Rat)) ^ boundary.val *
            dotProduct (seriesConstantVector (leftBoundaryCoefficient rows (boundary.val + 1)))
              (polynomialSeriesVector
                (ResolventNumerator.rightNumerator (1 - frozenProjection rows)
                  (transferCoefficient rows) (rightBoundaryCoefficient rows)
                  (degree - 1 - boundary.val))) := by
    rw [Finset.mul_sum]
    apply Finset.sum_congr rfl
    intro boundary _
    have powerFactor : (1 - (PowerSeries.X : PowerSeries Rat)) ^ (degree - 1) =
        (1 - PowerSeries.X) ^ boundary.val *
          (1 - PowerSeries.X) ^ (degree - 1 - boundary.val) := by
      rw [← pow_add]
      congr 1
      have := boundary.isLt
      omega
    calc
      _ = dotProduct (seriesConstantVector (leftBoundaryCoefficient rows (boundary.val + 1)))
          ((1 - (PowerSeries.X : PowerSeries Rat)) ^ (degree - 1) •
            rightCoefficientSeries rows (degree - 1 - boundary.val)) := by
        rw [dotProduct_smul, smul_eq_mul]
      _ = dotProduct (seriesConstantVector (leftBoundaryCoefficient rows (boundary.val + 1)))
          ((1 - (PowerSeries.X : PowerSeries Rat)) ^ boundary.val •
            polynomialSeriesVector
              (ResolventNumerator.rightNumerator (1 - frozenProjection rows)
                (transferCoefficient rows) (rightBoundaryCoefficient rows)
                (degree - 1 - boundary.val))) := by
        rw [powerFactor, ← smul_smul, rightCoefficientSeries_numerator_identity rowsValid]
      _ = _ := by rw [dotProduct_smul, smul_eq_mul]
  calc
    _ = PowerSeries.X ^ 2 *
        ((1 - (PowerSeries.X : PowerSeries Rat)) ^ (degree - 1) *
          (∑ boundary : Fin (degree - 1),
            dotProduct (seriesConstantVector (leftBoundaryCoefficient rows (boundary.val + 1)))
              (rightCoefficientSeries rows (degree - 1 - boundary.val)))) := by ring
    _ = _ := by rw [sumIdentity]

theorem degreeCountWidthSeries_bounded_numerator {rows : Nat} (rowsValid : 2 ≤ rows)
    (degree : Nat) (degreeValid : 3 ≤ degree) :
    ∃ numerator : Polynomial Rat, numerator.natDegree ≤ 2 * degree - 3 ∧
      (1 - (PowerSeries.X : PowerSeries Rat)) ^ (degree - 1) *
        degreeCountWidthSeries rows degree = numerator := by
  refine ⟨ResolventNumerator.fullNumerator (1 - frozenProjection rows) (transferCoefficient rows)
    (leftBoundaryCoefficient rows) (rightBoundaryCoefficient rows) degree,
      concrete_numerator_degree rowsValid degree degreeValid, ?_⟩
  exact degreeCountWidthSeries_numerator_identity rowsValid degree (by omega)

#print axioms transfer_polynomial_preserves_rotation
#print axioms propagatedRight_zero
#print axioms propagatedRight_succ
#print axioms propagatedRight_rotation
#print axioms propagatedCoefficient_succ
#print axioms propagatedCoefficient_zero
#print axioms rightCoefficientSeries_initial
#print axioms rightCoefficientSeries_rotation
#print axioms rightCoefficientSeries_zero
#print axioms concrete_numerator_degree
#print axioms series_mulVec_coeff
#print axioms rightCoefficientSeries_equation
#print axioms series_zeroTransfer_projection
#print axioms rightCoefficientSeries_positive
#print axioms series_frozenProjection_action
#print axioms series_complement_action
#print axioms seriesLinearNumerator_action
#print axioms rightCoefficientSeries_reduced
#print axioms polynomial_series_sum
#print axioms polynomial_numerator_series_step
#print axioms rightCoefficientSeries_numerator_identity
#print axioms counted_coefficient_identity
#print axioms degreeCountWidthSeries_identity
#print axioms degreeCountWidthSeries_positive
#print axioms full_numerator_series_map
#print axioms degreeCountWidthSeries_numerator_identity
#print axioms degreeCountWidthSeries_bounded_numerator

end OrigamiCone.Sequel.ConcreteTransfer
