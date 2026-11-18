import MiuraCounting.ConcreteBoundaryBlocks
import MiuraCounting.ConcreteFrozenTransfer

namespace OrigamiCone.Sequel.ConcreteTransfer

def stateColumns {rows : Nat} (source : TransferState rows) : Fin 2 -> PathColouring rows :=
  fun column => if column.val = 0 then source.val.1 else source.val.2

theorem stateColumns_proper {rows : Nat} (source : TransferState rows) :
    ∀ column, IsPathProperColouring (stateColumns source column) := by
  intro column
  unfold stateColumns
  split_ifs
  · exact source.property.1
  · exact source.property.2.1

theorem stateColumns_adjacent {rows : Nat} (source : TransferState rows) :
    ∀ index : Nat, ∀ bound : index + 1 < 2, ∀ row : Fin rows,
      stateColumns source ⟨index, by omega⟩ row ≠
        stateColumns source ⟨index + 1, bound⟩ row := by
  intro index bound row
  have indexZero : index = 0 := by omega
  subst index
  exact source.property.2.2 row

noncomputable def stateLift {rows : Nat} (source : TransferState rows) :
    OrigamiCone.Cell rows 2 -> Int := pathToHeight (stateColumns source)

theorem stateLift_isHeight {rows : Nat} (source : TransferState rows) :
    OrigamiCone.IsHeight (stateLift source) :=
  pathToHeight_isHeight (stateColumns source)
    (stateColumns_proper source) (stateColumns_adjacent source)

theorem stateLift_color {rows : Nat} (rowsPositive : 0 < rows)
    (source : TransferState rows) (column : Fin 2) (row : Fin rows) :
    ((stateLift source (row, column) : Int) : ZMod 3) =
      stateColumns source column row - source.val.1 ⟨0, rowsPositive⟩ := by
  exact hCol_pathToHeight (stateColumns source) (stateColumns_proper source)
    (stateColumns_adjacent source) row.val row.isLt column.val column.isLt

theorem stateLift_left_extremum {rows : Nat} (rowsPositive : 0 < rows)
    (source : TransferState rows) (row : Fin rows) :
    OrigamiCone.IsStrictLocalExtremum (stateLift source) (row, ⟨0, by decide⟩) ↔
      IsBoundaryExtremum source.val.1 source.val.2 row := by
  rw [IsStrictLocalExtremum_iff_IsBoundaryExtremum_left (stateLift_isHeight source)
    row (by decide) (by decide)]
  simp [IsBoundaryExtremum, hCol, stateLift_color rowsPositive source, stateColumns]

theorem stateLift_right_extremum {rows : Nat} (rowsPositive : 0 < rows)
    (source : TransferState rows) (row : Fin rows) :
    OrigamiCone.IsStrictLocalExtremum (stateLift source) (row, ⟨1, by decide⟩) ↔
      IsBoundaryExtremum source.val.2 source.val.1 row := by
  rw [IsStrictLocalExtremum_iff_IsBoundaryExtremum_right (stateLift_isHeight source)
    row (by decide)]
  simp [IsBoundaryExtremum, hCol, stateLift_color rowsPositive source, stateColumns]

theorem one_left_boundary_frozen {rows : Nat} (rowsValid : 2 ≤ rows)
    (source : TransferState rows)
    (oneBoundary : (Finset.univ.filter
      (IsBoundaryExtremum source.val.1 source.val.2)).card = 1) :
    frozenState source := by
  have rowsPositive : 0 < rows := by omega
  have sameSet : (Finset.univ.filter (fun row : Fin rows =>
      OrigamiCone.IsStrictLocalExtremum (stateLift source) (row, ⟨0, by decide⟩))) =
      Finset.univ.filter (IsBoundaryExtremum source.val.1 source.val.2) := by
    ext row
    simp only [Finset.mem_filter, Finset.mem_univ, true_and,
      stateLift_left_extremum rowsPositive source]
  have oneLifted : (Finset.univ.filter (fun row : Fin rows =>
      OrigamiCone.IsStrictLocalExtremum (stateLift source) (row, ⟨0, by decide⟩))).card = 1 := by
    rw [sameSet]
    exact oneBoundary
  obtain ⟨slope, slopeUnit, slopeValues⟩ :=
    unique_first_boundary_frozen (stateLift source) (stateLift_isHeight source)
      (by decide) oneLifted
  refine ⟨(slope : ZMod 3), ?_, ?_⟩
  · rcases slopeUnit with rfl | rfl <;> decide
  · intro row
    have equality := congrArg (fun value : Int => (value : ZMod 3)) (slopeValues row)
    push_cast at equality
    rw [stateLift_color rowsPositive source _ row,
      stateLift_color rowsPositive source _ row] at equality
    change source.val.2 row - source.val.1 ⟨0, rowsPositive⟩ =
      source.val.1 row - source.val.1 ⟨0, rowsPositive⟩ + (slope : ZMod 3) at equality
    linear_combination equality

theorem one_right_boundary_frozen {rows : Nat} (rowsValid : 2 ≤ rows)
    (source : TransferState rows)
    (oneBoundary : (Finset.univ.filter
      (IsBoundaryExtremum source.val.2 source.val.1)).card = 1) :
    frozenState source := by
  have rowsPositive : 0 < rows := by omega
  have sameSet : (Finset.univ.filter (fun row : Fin rows =>
      OrigamiCone.IsStrictLocalExtremum (stateLift source) (row, ⟨1, by decide⟩))) =
      Finset.univ.filter (IsBoundaryExtremum source.val.2 source.val.1) := by
    ext row
    simp only [Finset.mem_filter, Finset.mem_univ, true_and,
      stateLift_right_extremum rowsPositive source]
  have oneLifted : (Finset.univ.filter (fun row : Fin rows =>
      OrigamiCone.IsStrictLocalExtremum (stateLift source) (row, ⟨1, by decide⟩))).card = 1 := by
    rw [sameSet]
    exact oneBoundary
  obtain ⟨slope, slopeUnit, slopeValues⟩ :=
    unique_last_boundary_frozen (stateLift source) (stateLift_isHeight source)
      (by decide) oneLifted
  refine ⟨-(slope : ZMod 3), ?_, ?_⟩
  · rcases slopeUnit with rfl | rfl <;> decide
  · intro row
    have equality := congrArg (fun value : Int => (value : ZMod 3)) (slopeValues row)
    push_cast at equality
    rw [stateLift_color rowsPositive source _ row,
      stateLift_color rowsPositive source _ row] at equality
    change source.val.1 row - source.val.1 ⟨0, rowsPositive⟩ =
      source.val.2 row - source.val.1 ⟨0, rowsPositive⟩ + (slope : ZMod 3) at equality
    linear_combination -equality

theorem left_boundary_single_zero {rows : Nat} (rowsValid : 2 ≤ rows)
    (source : TransferState rows) (notFrozen : ¬ frozenState source) :
    (leftBdyVec rows source).coeff 1 = 0 := by
  have notOne : (Finset.univ.filter
      (IsBoundaryExtremum source.val.1 source.val.2)).card ≠ 1 :=
    fun one => notFrozen (one_left_boundary_frozen rowsValid source one)
  simp only [leftBdyVec, Polynomial.coeff_monomial]
  split_ifs with equality
  · exact False.elim (notOne (by omega))
  · rfl

theorem right_boundary_single_zero {rows : Nat} (rowsValid : 2 ≤ rows)
    (source : TransferState rows) (notFrozen : ¬ frozenState source) :
    (rightBdyVec rows source).coeff 1 = 0 := by
  have notOne : (Finset.univ.filter
      (IsBoundaryExtremum source.val.2 source.val.1)).card ≠ 1 :=
    fun one => notFrozen (one_right_boundary_frozen rowsValid source one)
  simp only [rightBdyVec, Polynomial.coeff_monomial]
  split_ifs with equality
  · exact False.elim (notOne (by omega))
  · rfl

noncomputable def frozenProjection (rows : Nat) :
    Matrix (TransferState rows) (TransferState rows) Rat :=
  Matrix.diagonal fun source => if frozenState source then 1 else 0

noncomputable def leftSingleBoundary (rows : Nat) : TransferState rows -> Rat :=
  fun source => ((leftBdyVec rows source).coeff 1 : Rat) / 3

noncomputable def rightSingleBoundary (rows : Nat) : TransferState rows -> Rat :=
  fun source => ((rightBdyVec rows source).coeff 1 : Rat)

theorem left_single_projection {rows : Nat} (rowsValid : 2 ≤ rows) :
    Matrix.vecMul (leftSingleBoundary rows) (frozenProjection rows) =
      leftSingleBoundary rows := by
  funext source
  unfold frozenProjection
  rw [Matrix.vecMul_diagonal]
  by_cases active : frozenState source
  · simp only [active, if_true, mul_one]
  · have zero := left_boundary_single_zero rowsValid source active
    simp only [active, if_false, mul_zero, leftSingleBoundary, zero, Int.cast_zero,
      zero_div]

theorem right_single_projection {rows : Nat} (rowsValid : 2 ≤ rows) :
    Matrix.mulVec (frozenProjection rows) (rightSingleBoundary rows) =
      rightSingleBoundary rows := by
  funext source
  unfold frozenProjection
  rw [Matrix.mulVec_diagonal]
  by_cases active : frozenState source
  · simp only [active, if_true, one_mul]
  · have zero := right_boundary_single_zero rowsValid source active
    simp only [active, if_false, zero_mul, rightSingleBoundary, zero, Int.cast_zero]

theorem left_single_complement_zero {rows : Nat} (rowsValid : 2 ≤ rows) :
    Matrix.vecMul (leftSingleBoundary rows) (1 - frozenProjection rows) = 0 := by
  simp only [Matrix.vecMul_sub, Matrix.vecMul_one, left_single_projection rowsValid,
    sub_self]

theorem right_single_complement_zero {rows : Nat} (rowsValid : 2 ≤ rows) :
    Matrix.mulVec (1 - frozenProjection rows) (rightSingleBoundary rows) = 0 := by
  simp only [Matrix.sub_mulVec, Matrix.one_mulVec, right_single_projection rowsValid,
    sub_self]

theorem endpoint_counts_positive {rows : Nat} (rowsPositive : 0 < rows)
    (source : TransferState rows) :
    0 < (Finset.univ.filter (IsBoundaryExtremum source.val.1 source.val.2)).card ∧
      0 < (Finset.univ.filter (IsBoundaryExtremum source.val.2 source.val.1)).card := by
  obtain ⟨leftActive, rightActive⟩ := boundary_columns_active
    (stateLift source) (stateLift_isHeight source) rowsPositive (by decide)
  obtain ⟨leftRow, leftExtremum⟩ := leftActive
  obtain ⟨rightRow, rightExtremum⟩ := rightActive
  constructor
  · exact Finset.card_pos.mpr ⟨leftRow, Finset.mem_filter.mpr ⟨Finset.mem_univ _,
      (stateLift_left_extremum rowsPositive source leftRow).mp leftExtremum⟩⟩
  · exact Finset.card_pos.mpr ⟨rightRow, Finset.mem_filter.mpr ⟨Finset.mem_univ _,
      (stateLift_right_extremum rowsPositive source rightRow).mp rightExtremum⟩⟩

theorem endpoint_constant_zero {rows : Nat} (rowsPositive : 0 < rows)
    (source : TransferState rows) :
    (leftBdyVec rows source).coeff 0 = 0 ∧ (rightBdyVec rows source).coeff 0 = 0 := by
  obtain ⟨leftPositive, rightPositive⟩ := endpoint_counts_positive rowsPositive source
  constructor
  · simp only [leftBdyVec, Polynomial.coeff_monomial]
    split_ifs with zeroDegree
    · have empty : (Finset.univ.filter
          (IsBoundaryExtremum source.val.1 source.val.2)).card = 0 := by
        simpa only [eq_comm] using zeroDegree
      exact False.elim ((Nat.ne_of_gt leftPositive) empty)
    · rfl
  · simp only [rightBdyVec, Polynomial.coeff_monomial]
    split_ifs with zeroDegree
    · have empty : (Finset.univ.filter
          (IsBoundaryExtremum source.val.2 source.val.1)).card = 0 := by
        simpa only [eq_comm] using zeroDegree
      exact False.elim ((Nat.ne_of_gt rightPositive) empty)
    · rfl

theorem normalized_left_constant_zero {rows : Nat} (rowsPositive : 0 < rows)
    (source : TransferState rows) : (leftBdyVecNorm rows source).coeff 0 = 0 := by
  by_cases normalized : IsNormalizedLeft source
  · rw [leftBdyVecNorm_of_normalized normalized]
    exact (endpoint_constant_zero rowsPositive source).1
  · rw [leftBdyVecNorm_of_not_normalized normalized]
    simp

#print axioms stateColumns_proper
#print axioms stateColumns_adjacent
#print axioms stateLift_isHeight
#print axioms stateLift_color
#print axioms stateLift_left_extremum
#print axioms stateLift_right_extremum
#print axioms one_left_boundary_frozen
#print axioms one_right_boundary_frozen
#print axioms left_boundary_single_zero
#print axioms right_boundary_single_zero
#print axioms left_single_projection
#print axioms right_single_projection
#print axioms left_single_complement_zero
#print axioms right_single_complement_zero
#print axioms endpoint_counts_positive
#print axioms endpoint_constant_zero
#print axioms normalized_left_constant_zero

end OrigamiCone.Sequel.ConcreteTransfer
