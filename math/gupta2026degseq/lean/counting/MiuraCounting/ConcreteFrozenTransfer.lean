import OrigamiCone.SequelEdTransfer
import OrigamiCone.SequelCascade
import OrigamiCone.SequelFrozen

namespace OrigamiCone.Sequel.ConcreteTransfer

def extendColumn {rows : Nat} (column : PathColouring rows) (index : Nat) : ZMod 3 :=
  if bound : index < rows then column ⟨index, bound⟩ else 0

theorem extendColumn_proper {rows : Nat} (column : PathColouring rows)
    (proper : IsPathProperColouring column) {index : Nat}
    (bound : index + 1 < rows) :
    extendColumn column index ≠ extendColumn column (index + 1) := by
  have currentBound : index < rows := by omega
  simpa only [extendColumn, dif_pos currentBound, dif_pos bound] using
    proper ⟨index, currentBound⟩ bound

theorem zero_column_iff_frozen {rows : Nat} (rowsValid : 2 ≤ rows)
    (left middle right : PathColouring rows)
    (leftAdjacent : PathAdjacent left middle)
    (rightAdjacent : PathAdjacent middle right) :
    columnExtremaCount left middle right = 0 ↔
      ∃ slope : ZMod 3, slope ≠ 0 ∧
        (∀ row, left row = middle row - slope) ∧
        (∀ row, right row = middle row + slope) := by
  constructor
  · intro countZero
    let leftExtended := extendColumn left
    let middleExtended := extendColumn middle
    let rightExtended := extendColumn right
    let alpha := fun index => leftExtended index - middleExtended index
    let gamma := fun index => rightExtended index - middleExtended index
    let step := fun index => middleExtended (index + 1) - middleExtended index
    have alphaNonzero : ∀ index, index < rows → alpha index ≠ 0 := by
      intro index bound
      apply sub_ne_zero.mpr
      simpa only [leftExtended, middleExtended, extendColumn, dif_pos bound] using
        leftAdjacent.2.2 ⟨index, bound⟩
    have gammaNonzero : ∀ index, index < rows → gamma index ≠ 0 := by
      intro index bound
      apply sub_ne_zero.mpr
      simpa only [rightExtended, middleExtended, extendColumn, dif_pos bound] using
        (rightAdjacent.2.2 ⟨index, bound⟩).symm
    have leftProper : ∀ index, index + 1 < rows →
        leftExtended index ≠ leftExtended (index + 1) := by
      intro index bound
      exact extendColumn_proper left leftAdjacent.1 bound
    have middleProper : ∀ index, index + 1 < rows →
        middleExtended index ≠ middleExtended (index + 1) := by
      intro index bound
      exact extendColumn_proper middle leftAdjacent.2.1 bound
    have rightProper : ∀ index, index + 1 < rows →
        rightExtended index ≠ rightExtended (index + 1) := by
      intro index bound
      exact extendColumn_proper right rightAdjacent.2.1 bound
    have stepNonzero : ∀ index, index + 1 < rows → step index ≠ 0 := by
      intro index bound
      exact sub_ne_zero.mpr (middleProper index bound).symm
    have alphaProper : ∀ index, index + 1 < rows →
        alpha (index + 1) ≠ alpha index - step index := by
      intro index bound equality
      apply leftProper index bound
      dsimp only [alpha, step] at equality
      linear_combination -equality
    have gammaProper : ∀ index, index + 1 < rows →
        gamma (index + 1) ≠ gamma index - step index := by
      intro index bound equality
      apply rightProper index bound
      dsimp only [gamma, step] at equality
      linear_combination -equality
    have noColumnExtremum : ∀ row, ¬ IsColExtremum left middle right row := by
      intro row extremum
      have member : row ∈ Finset.univ.filter (IsColExtremum left middle right) :=
        Finset.mem_filter.mpr ⟨Finset.mem_univ row, extremum⟩
      have empty := Finset.card_eq_zero.mp countZero
      rw [empty] at member
      simp at member
    have noOffsetExtremum : ∀ index, index < rows →
        ¬ IsExtremum alpha gamma step rows index := by
      intro index bound extremum
      apply noColumnExtremum ⟨index, bound⟩
      refine ⟨?_, ?_, ?_⟩
      · have equality := extremum.1
        dsimp only [alpha, gamma] at equality
        have same : leftExtended index = rightExtended index := by
          linear_combination equality
        simpa only [leftExtended, rightExtended, extendColumn, dif_pos bound] using same
      · intro positive
        change 0 < index at positive
        have previousBound : index - 1 < rows := by omega
        have previousNext : index - 1 + 1 = index := by omega
        have equality := extremum.2.1 positive
        dsimp only [step, alpha] at equality
        rw [previousNext] at equality
        have same : middleExtended (index - 1) = leftExtended index := by
          linear_combination -equality
        simpa only [middleExtended, leftExtended, extendColumn,
          dif_pos previousBound, dif_pos bound] using same
      · intro nextBound
        change index + 1 < rows at nextBound
        have equality := extremum.2.2 nextBound
        dsimp only [step, alpha] at equality
        have same : middleExtended (index + 1) = leftExtended index := by
          linear_combination equality
        simpa only [middleExtended, leftExtended, extendColumn,
          dif_pos nextBound, dif_pos bound] using same
    have rainbowOffsets := cascade alpha gamma step rows alphaNonzero gammaNonzero
      stepNonzero alphaProper gammaProper noOffsetExtremum
    have rainbow : ∀ index,
        rightExtended index - middleExtended index =
          -(leftExtended index - middleExtended index) := by
      intro index
      by_cases bound : index < rows
      · have residueRule : ∀ first second : ZMod 3,
            first ≠ 0 → second ≠ 0 → first ≠ second → second = -first := by
          decide
        exact residueRule _ _ (alphaNonzero index bound) (gammaNonzero index bound)
          (rainbowOffsets index bound)
      · simp only [leftExtended, middleExtended, rightExtended, extendColumn,
          dif_neg bound, sub_self, neg_zero]
    have alphaStep : ∀ index, index + 1 < rows → alpha (index + 1) = alpha index := by
      intro index bound
      exact rainbow_imp_const_slope leftExtended middleExtended rightExtended index
        (middleProper index bound).symm (leftProper index bound).symm
        (rightProper index bound).symm rainbow
    have alphaConstant : ∀ index, index < rows → alpha index = alpha 0 := by
      intro index bound
      induction index with
      | zero => rfl
      | succ index inductionHypothesis =>
          exact (alphaStep index bound).trans (inductionHypothesis (by omega))
    refine ⟨-alpha 0, neg_ne_zero.mpr (alphaNonzero 0 (by omega)), ?_, ?_⟩
    · intro row
      have equality := alphaConstant row.val row.isLt
      dsimp only [alpha] at equality ⊢
      have same : leftExtended row.val = middleExtended row.val -
          -(leftExtended 0 - middleExtended 0) := by
        linear_combination equality
      simpa only [leftExtended, middleExtended, extendColumn, dif_pos row.isLt] using same
    · intro row
      have equality := alphaConstant row.val row.isLt
      have rowRainbow := rainbow row.val
      dsimp only [alpha] at equality ⊢
      have same : rightExtended row.val = middleExtended row.val +
          -(leftExtended 0 - middleExtended 0) := by
        linear_combination rowRainbow - equality
      simpa only [rightExtended, middleExtended, extendColumn, dif_pos row.isLt] using same
  · rintro ⟨slope, slopeNonzero, leftValues, rightValues⟩
    have noExtremum : ∀ row, ¬ IsColExtremum left middle right row := by
      intro row extremum
      have different : ∀ value increment : ZMod 3, increment ≠ 0 →
          value - increment ≠ value + increment := by
        decide
      apply different (middle row) slope slopeNonzero
      rw [← leftValues row, ← rightValues row]
      exact extremum.1
    exact Finset.card_eq_zero.mpr
      (Finset.filter_eq_empty_iff.mpr (fun row _ => noExtremum row))

def rotateState {rows : Nat} (source : TransferState rows) (slope : ZMod 3) :
    TransferState rows :=
  ⟨(fun row => source.val.1 row + slope, fun row => source.val.2 row + slope), by
    refine ⟨?_, ?_, ?_⟩
    · intro row adjacent equality
      exact source.property.1 row adjacent (add_right_cancel equality)
    · intro row adjacent equality
      exact source.property.2.1 row adjacent (add_right_cancel equality)
    · intro row equality
      exact source.property.2.2 row (add_right_cancel equality)⟩

theorem transfer_constant_entry {rows : Nat} (source target : TransferState rows) :
    (transferMatrix rows source target).coeff 0 =
      if source.val.2 = target.val.1 ∧
        columnExtremaCount source.val.1 source.val.2 target.val.2 = 0 then
        (1 : Int) else 0 := by
  classical
  by_cases matching : source.val.2 = target.val.1
  · rw [transferMatrix_apply_eq source target matching]
    simp only [Polynomial.coeff_X_pow, matching, true_and, eq_comm]
  · rw [transferMatrix_apply_ne source target matching]
    simp only [Polynomial.coeff_zero, matching, false_and, if_false]

theorem transfer_constant_entry_frozen {rows : Nat} (rowsValid : 2 ≤ rows)
    (source target : TransferState rows) :
    (transferMatrix rows source target).coeff 0 =
      if source.val.2 = target.val.1 ∧
        (∃ slope : ZMod 3, slope ≠ 0 ∧
          (∀ row, source.val.1 row = source.val.2 row - slope) ∧
          (∀ row, target.val.2 row = source.val.2 row + slope)) then
        (1 : Int) else 0 := by
  classical
  rw [transfer_constant_entry]
  by_cases matching : source.val.2 = target.val.1
  · have rightAdjacent : PathAdjacent source.val.2 target.val.2 := by
      rw [matching]
      exact target.property
    simp only [zero_column_iff_frozen rowsValid source.val.1 source.val.2 target.val.2
      source.property rightAdjacent]
  · simp only [matching, false_and, if_false]

def frozenState {rows : Nat} (source : TransferState rows) : Prop :=
  ∃ slope : ZMod 3, slope ≠ 0 ∧
    ∀ row, source.val.2 row = source.val.1 row + slope

instance {rows : Nat} (source : TransferState rows) : Decidable (frozenState source) := by
  unfold frozenState
  infer_instance

theorem transfer_constant_entry_of_frozen {rows : Nat} (rowsValid : 2 ≤ rows)
    (source target : TransferState rows) (slope : ZMod 3)
    (slopeNonzero : slope ≠ 0)
    (sourceFrozen : ∀ row, source.val.2 row = source.val.1 row + slope) :
    (transferMatrix rows source target).coeff 0 =
      if target = rotateState source slope then (1 : Int) else 0 := by
  classical
  rw [transfer_constant_entry_frozen rowsValid]
  by_cases sameTarget : target = rotateState source slope
  · subst target
    have matching : source.val.2 = (rotateState source slope).val.1 :=
      funext sourceFrozen
    have leftValues : ∀ row, source.val.1 row = source.val.2 row - slope := by
      intro row
      rw [sourceFrozen row]
      ring
    rw [if_pos ⟨matching, slope, slopeNonzero, leftValues, fun _ => rfl⟩,
      if_pos rfl]
  · rw [if_neg sameTarget]
    apply if_neg
    rintro ⟨matching, otherSlope, otherNonzero, leftValues, rightValues⟩
    apply sameTarget
    have sameSlope : otherSlope = slope := by
      have firstEquality := leftValues ⟨0, by omega⟩
      have secondEquality := sourceFrozen ⟨0, by omega⟩
      linear_combination firstEquality + secondEquality
    apply Subtype.ext
    apply Prod.ext
    · exact matching.symm.trans (funext sourceFrozen)
    · funext row
      change target.val.2 row = source.val.2 row + slope
      rw [rightValues row, sameSlope]

theorem transfer_constant_entry_of_not_frozen {rows : Nat} (rowsValid : 2 ≤ rows)
    (source target : TransferState rows) (sourceNotFrozen : ¬ frozenState source) :
    (transferMatrix rows source target).coeff 0 = 0 := by
  classical
  rw [transfer_constant_entry_frozen rowsValid]
  apply if_neg
  rintro ⟨matching, slope, slopeNonzero, leftValues, rightValues⟩
  apply sourceNotFrozen
  refine ⟨slope, slopeNonzero, ?_⟩
  intro row
  have equality := leftValues row
  linear_combination -equality

noncomputable def zeroTransferMatrix (rows : Nat) :
    Matrix (TransferState rows) (TransferState rows) Rat :=
  fun source target => ((transferMatrix rows source target).coeff 0 : Rat)

theorem zeroTransfer_action {rows : Nat} (rowsValid : 2 ≤ rows)
    (values : TransferState rows -> Rat)
    (rotationInvariant : ∀ source slope, values (rotateState source slope) = values source)
    (source : TransferState rows) :
    Matrix.mulVec (zeroTransferMatrix rows) values source =
      if frozenState source then values source else 0 := by
  classical
  change (∑ target, (((transferMatrix rows source target).coeff 0 : Int) : Rat) *
    values target) = if frozenState source then values source else 0
  by_cases sourceIsFrozen : frozenState source
  · rw [if_pos sourceIsFrozen]
    obtain ⟨slope, slopeNonzero, sourceFrozen⟩ := sourceIsFrozen
    have entries : ∀ target,
        (((transferMatrix rows source target).coeff 0 : Int) : Rat) =
          if target = rotateState source slope then 1 else 0 := by
      intro target
      rw [transfer_constant_entry_of_frozen rowsValid source target slope
        slopeNonzero sourceFrozen]
      split_ifs <;> rfl
    simp_rw [entries]
    simpa [ite_mul] using rotationInvariant source slope
  · rw [if_neg sourceIsFrozen]
    simp_rw [transfer_constant_entry_of_not_frozen rowsValid source _ sourceIsFrozen]
    simp

#print axioms extendColumn_proper
#print axioms zero_column_iff_frozen
#print axioms transfer_constant_entry
#print axioms transfer_constant_entry_frozen
#print axioms transfer_constant_entry_of_frozen
#print axioms transfer_constant_entry_of_not_frozen
#print axioms zeroTransfer_action

end OrigamiCone.Sequel.ConcreteTransfer
