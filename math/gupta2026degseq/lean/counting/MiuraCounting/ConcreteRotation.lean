import MiuraCounting.ConcreteFrozenTransfer

namespace OrigamiCone.Sequel.ConcreteTransfer

theorem rotateState_zero {rows : Nat} (source : TransferState rows) :
    rotateState source 0 = source := by
  apply Subtype.ext
  apply Prod.ext <;> funext row <;> simp [rotateState]

theorem rotateState_add {rows : Nat} (source : TransferState rows)
    (first second : ZMod 3) :
    rotateState (rotateState source first) second = rotateState source (first + second) := by
  apply Subtype.ext
  apply Prod.ext <;> funext row <;> simp [rotateState, add_assoc]

def rotateStateEquiv {rows : Nat} (slope : ZMod 3) : TransferState rows ≃ TransferState rows where
  toFun source := rotateState source slope
  invFun source := rotateState source (-slope)
  left_inv source := by
    change rotateState (rotateState source slope) (-slope) = source
    rw [rotateState_add, add_neg_cancel, rotateState_zero]
  right_inv source := by
    change rotateState (rotateState source (-slope)) slope = source
    rw [rotateState_add, neg_add_cancel, rotateState_zero]

theorem rotation_invariant_all {rows : Nat} {Value : Type}
    (values : TransferState rows -> Value)
    (invariant : ∀ source, values (rotateState source 1) = values source) :
    ∀ source slope, values (rotateState source slope) = values source := by
  intro source slope
  fin_cases slope
  · change values (rotateState source (0 : ZMod 3)) = values source
    rw [rotateState_zero]
  · exact invariant source
  · have twice := invariant (rotateState source 1)
    rw [rotateState_add] at twice
    exact twice.trans (invariant source)

theorem column_extremum_shift {rows : Nat} (left middle right : PathColouring rows)
    (slope : ZMod 3) (row : Fin rows) :
    IsColExtremum (fun index => left index + slope)
      (fun index => middle index + slope) (fun index => right index + slope) row ↔
      IsColExtremum left middle right row := by
  simp [IsColExtremum]

theorem column_count_shift {rows : Nat} (left middle right : PathColouring rows)
    (slope : ZMod 3) :
    columnExtremaCount (fun index => left index + slope)
      (fun index => middle index + slope) (fun index => right index + slope) =
      columnExtremaCount left middle right := by
  unfold columnExtremaCount
  apply congrArg Finset.card
  ext row
  simp only [Finset.mem_filter, Finset.mem_univ, true_and, column_extremum_shift]

theorem boundary_extremum_shift {rows : Nat} (own adjacent : PathColouring rows)
    (slope : ZMod 3) (row : Fin rows) :
    IsBoundaryExtremum (fun index => own index + slope)
      (fun index => adjacent index + slope) row ↔ IsBoundaryExtremum own adjacent row := by
  simp [IsBoundaryExtremum]

theorem boundary_count_shift {rows : Nat} (own adjacent : PathColouring rows)
    (slope : ZMod 3) :
    (Finset.univ.filter (IsBoundaryExtremum (fun index => own index + slope)
      (fun index => adjacent index + slope))).card =
      (Finset.univ.filter (IsBoundaryExtremum own adjacent)).card := by
  apply congrArg Finset.card
  ext row
  simp only [Finset.mem_filter, Finset.mem_univ, true_and, boundary_extremum_shift]

theorem transferMatrix_rotate {rows : Nat} (source target : TransferState rows)
    (slope : ZMod 3) :
    transferMatrix rows (rotateState source slope) (rotateState target slope) =
      transferMatrix rows source target := by
  have matching : (rotateState source slope).val.2 = (rotateState target slope).val.1 ↔
      source.val.2 = target.val.1 := by
    constructor
    · intro equality
      funext row
      exact add_right_cancel (congrFun equality row)
    · intro equality
      funext row
      exact congrArg (fun value => value + slope) (congrFun equality row)
  have sameCount : columnExtremaCount (rotateState source slope).val.1
      (rotateState source slope).val.2 (rotateState target slope).val.2 =
      columnExtremaCount source.val.1 source.val.2 target.val.2 :=
    column_count_shift source.val.1 source.val.2 target.val.2 slope
  unfold transferMatrix
  simp only [matching, sameCount]

theorem leftBdyVec_rotate {rows : Nat} (source : TransferState rows) (slope : ZMod 3) :
    leftBdyVec rows (rotateState source slope) = leftBdyVec rows source := by
  unfold leftBdyVec
  rw [show (Finset.univ.filter (IsBoundaryExtremum (rotateState source slope).val.1
      (rotateState source slope).val.2)).card =
      (Finset.univ.filter (IsBoundaryExtremum source.val.1 source.val.2)).card from
    boundary_count_shift source.val.1 source.val.2 slope]

theorem rightBdyVec_rotate {rows : Nat} (source : TransferState rows) (slope : ZMod 3) :
    rightBdyVec rows (rotateState source slope) = rightBdyVec rows source := by
  unfold rightBdyVec
  rw [show (Finset.univ.filter (IsBoundaryExtremum (rotateState source slope).val.2
      (rotateState source slope).val.1)).card =
      (Finset.univ.filter (IsBoundaryExtremum source.val.2 source.val.1)).card from
    boundary_count_shift source.val.2 source.val.1 slope]

noncomputable def transferCoefficient (rows degree : Nat) :
    Matrix (TransferState rows) (TransferState rows) Rat :=
  fun source target => ((transferMatrix rows source target).coeff degree : Rat)

theorem transferCoefficient_rotate {rows : Nat} (degree : Nat)
    (source target : TransferState rows) (slope : ZMod 3) :
    transferCoefficient rows degree (rotateState source slope) (rotateState target slope) =
      transferCoefficient rows degree source target := by
  exact congrArg (fun entry : Polynomial Int => (entry.coeff degree : Rat))
    (transferMatrix_rotate source target slope)

theorem transferCoefficient_preserves_rotation {rows : Nat} (degree : Nat)
    (values : TransferState rows -> Rat)
    (invariant : ∀ source slope, values (rotateState source slope) = values source) :
    ∀ source slope,
      Matrix.mulVec (transferCoefficient rows degree) values (rotateState source slope) =
        Matrix.mulVec (transferCoefficient rows degree) values source := by
  intro source slope
  change (∑ target, transferCoefficient rows degree (rotateState source slope) target *
    values target) = ∑ target, transferCoefficient rows degree source target * values target
  calc
    _ = ∑ target, transferCoefficient rows degree (rotateState source slope)
        (rotateState target slope) * values (rotateState target slope) :=
      (Equiv.sum_comp (rotateStateEquiv slope) (fun target =>
        transferCoefficient rows degree (rotateState source slope) target * values target)).symm
    _ = _ := Finset.sum_congr rfl (fun target _ => by
      rw [transferCoefficient_rotate, invariant target slope])

theorem frozenState_rotate {rows : Nat} (source : TransferState rows) (slope : ZMod 3) :
    frozenState (rotateState source slope) ↔ frozenState source := by
  constructor
  · rintro ⟨increment, nonzero, properties⟩
    refine ⟨increment, nonzero, ?_⟩
    intro row
    have equality := properties row
    change source.val.2 row + slope = source.val.1 row + slope + increment at equality
    linear_combination equality
  · rintro ⟨increment, nonzero, properties⟩
    refine ⟨increment, nonzero, ?_⟩
    intro row
    have equality := properties row
    change source.val.2 row + slope = source.val.1 row + slope + increment
    linear_combination equality

theorem zeroTransfer_generator_action {rows : Nat} (rowsValid : 2 ≤ rows)
    (values : TransferState rows -> Rat)
    (invariant : ∀ source, values (rotateState source 1) = values source)
    (source : TransferState rows) :
    Matrix.mulVec (zeroTransferMatrix rows) values source =
      if frozenState source then values source else 0 := by
  exact zeroTransfer_action rowsValid values (rotation_invariant_all values invariant) source

theorem normalized_rotate_iff {rows : Nat} (rowsPositive : 0 < rows)
    (source : TransferState rows) (slope : ZMod 3) :
    IsNormalizedLeft (rotateState source slope) ↔
      slope = -source.val.1 ⟨0, rowsPositive⟩ := by
  constructor
  · rintro ⟨positive, equality⟩
    change source.val.1 ⟨0, rowsPositive⟩ + slope = 0 at equality
    linear_combination equality
  · intro equality
    refine ⟨rowsPositive, ?_⟩
    change source.val.1 ⟨0, rowsPositive⟩ + slope = 0
    rw [equality]
    ring

theorem sum_rotated_left_normalized {rows : Nat} (rowsPositive : 0 < rows)
    (source : TransferState rows) :
    ∑ slope : ZMod 3, leftBdyVecNorm rows (rotateState source slope) = leftBdyVec rows source := by
  have summand : ∀ slope : ZMod 3, leftBdyVecNorm rows (rotateState source slope) =
      if slope = -source.val.1 ⟨0, rowsPositive⟩ then leftBdyVec rows source else 0 := by
    intro slope
    unfold leftBdyVecNorm
    simp only [normalized_rotate_iff rowsPositive, leftBdyVec_rotate]
  simp_rw [summand]
  simp

theorem three_times_normalized_sandwich {rows : Nat} (rowsPositive : 0 < rows)
    (values : TransferState rows -> Polynomial Int)
    (invariant : ∀ source slope, values (rotateState source slope) = values source) :
    (3 : Polynomial Int) * (∑ source, leftBdyVecNorm rows source * values source) =
      ∑ source, leftBdyVec rows source * values source := by
  have reindex : ∀ slope : ZMod 3,
      (∑ source, leftBdyVecNorm rows (rotateState source slope) * values source) =
        ∑ source, leftBdyVecNorm rows source * values source := by
    intro slope
    calc
      _ = ∑ source, leftBdyVecNorm rows (rotateState source slope) *
          values (rotateState source slope) := Finset.sum_congr rfl (fun source _ => by
            rw [invariant source slope])
      _ = _ := Equiv.sum_comp (rotateStateEquiv slope)
        (fun source => leftBdyVecNorm rows source * values source)
  calc
    _ = ∑ _slope : ZMod 3, ∑ source, leftBdyVecNorm rows source * values source := by
      simp [nsmul_eq_mul]
    _ = ∑ slope : ZMod 3, ∑ source,
        leftBdyVecNorm rows (rotateState source slope) * values source :=
      Finset.sum_congr rfl (fun slope _ => (reindex slope).symm)
    _ = ∑ source, (∑ slope : ZMod 3,
        leftBdyVecNorm rows (rotateState source slope)) * values source := by
      rw [Finset.sum_comm]
      exact Finset.sum_congr rfl (fun source _ => (Finset.sum_mul _ _ _).symm)
    _ = _ := by simp_rw [sum_rotated_left_normalized rowsPositive]

#print axioms rotateState_zero
#print axioms rotateState_add
#print axioms rotation_invariant_all
#print axioms column_extremum_shift
#print axioms column_count_shift
#print axioms boundary_extremum_shift
#print axioms boundary_count_shift
#print axioms transferMatrix_rotate
#print axioms leftBdyVec_rotate
#print axioms rightBdyVec_rotate
#print axioms transferCoefficient_rotate
#print axioms transferCoefficient_preserves_rotation
#print axioms frozenState_rotate
#print axioms zeroTransfer_generator_action
#print axioms normalized_rotate_iff
#print axioms sum_rotated_left_normalized
#print axioms three_times_normalized_sandwich

end OrigamiCone.Sequel.ConcreteTransfer
