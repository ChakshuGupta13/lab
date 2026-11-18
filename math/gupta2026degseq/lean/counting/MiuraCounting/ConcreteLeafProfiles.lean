import MiuraCounting.ConcreteJointNormalForm

namespace OrigamiCone.Sequel.LeafProfiles

open OrigamiCone FrozenExpansion

def rowLine {rows width : Nat} (height : Cell rows width -> Int) (row : Fin rows) : Cell 1 width -> Int :=
  fun point => height (row, point.2)

theorem rowLine_isHeight {rows width : Nat} (height : Cell rows width -> Int)
    (heightValid : IsHeight height) (row : Fin rows) : IsHeight (rowLine height row) := by
  intro first second adjacent
  change |height (row, first.2) - height (row, second.2)| = 1
  apply heightValid
  have firstZero : first.1.val = 0 := by have := first.1.isLt; omega
  have secondZero : second.1.val = 0 := by have := second.1.isLt; omega
  simpa [adj, gdist, firstZero, secondZero] using adjacent

theorem rowLine_distance {width : Nat} (first second : Fin width) :
    gdist ((0 : Fin 1), first) ((0 : Fin 1), second) = |(second.val : Int) - first.val| := by
  simp [gdist, abs_sub_comm]

theorem rowLine_cone_min {rows width : Nat} (height : Cell rows width -> Int)
    (heightValid : IsHeight height) (row : Fin rows) (apex : Fin width)
    (unique : ∀ column : Fin width, IsStrictLocalMin (rowLine height row) ((0 : Fin 1), column) -> column = apex) :
    ∀ column : Fin width, height (row, column) = height (row, apex) + |(column.val : Int) - apex.val| := by
  have uniquePoint : ∀ point : Cell 1 width, IsStrictLocalMin (rowLine height row) point ->
      point = ((0 : Fin 1), apex) := by
    rintro ⟨index, column⟩ minimum
    have indexZero : index = (0 : Fin 1) := Subsingleton.elim _ _
    subst index
    exact Prod.ext rfl (unique column minimum)
  intro column
  simpa only [rowLine, rowLine_distance] using
    cone_min (rowLine_isHeight height heightValid row) uniquePoint ((0 : Fin 1), column)

theorem rowLine_cone_max {rows width : Nat} (height : Cell rows width -> Int)
    (heightValid : IsHeight height) (row : Fin rows) (apex : Fin width)
    (unique : ∀ column : Fin width, IsStrictLocalMax (rowLine height row) ((0 : Fin 1), column) -> column = apex) :
    ∀ column : Fin width, height (row, column) = height (row, apex) - |(column.val : Int) - apex.val| := by
  have uniquePoint : ∀ point : Cell 1 width, IsStrictLocalMax (rowLine height row) point ->
      point = ((0 : Fin 1), apex) := by
    rintro ⟨index, column⟩ maximum
    have indexZero : index = (0 : Fin 1) := Subsingleton.elim _ _
    subst index
    exact Prod.ext rfl (unique column maximum)
  intro column
  simpa only [rowLine, rowLine_distance] using
    cone_max (rowLine_isHeight height heightValid row) uniquePoint ((0 : Fin 1), column)

theorem rowLine_increment_lift {rows width : Nat} (height : Cell rows width -> Int)
    (row : Fin rows) (column : Fin width) (step : Int)
    (horizontal : ∀ other : Fin width, adj ((0 : Fin 1), column) ((0 : Fin 1), other) ->
      height (row, other) = height (row, column) + step)
    (vertical : ∀ other : Fin rows, ((row.val : Int) - other.val).natAbs = 1 ->
      height (other, column) = height (row, column) + step) :
    ∀ point : Cell rows width, adj (row, column) point -> height point = height (row, column) + step := by
  rintro ⟨otherRow, otherColumn⟩ adjacent
  by_cases sameRow : otherRow = row
  · subst otherRow
    exact horizontal otherColumn (by simpa [adj, gdist] using adjacent)
  · have sameColumn : otherColumn = column := by
      apply Fin.ext
      have differentRows : otherRow.val ≠ row.val := fun equality => sameRow (Fin.ext equality)
      simp only [adj, gdist] at adjacent
      omega
    subst otherColumn
    apply vertical otherRow
    simp only [adj, gdist] at adjacent
    omega

theorem rowLine_minimum_lift {rows width : Nat} (height : Cell rows width -> Int)
    (row : Fin rows) (column : Fin width)
    (vertical : ∀ other : Fin rows, ((row.val : Int) - other.val).natAbs = 1 ->
      height (other, column) = height (row, column) + 1)
    (minimum : IsStrictLocalMin (rowLine height row) ((0 : Fin 1), column)) :
    IsStrictLocalMin height (row, column) :=
  rowLine_increment_lift height row column 1 (fun other adjacent => minimum ((0 : Fin 1), other) adjacent) vertical

theorem rowLine_maximum_lift {rows width : Nat} (height : Cell rows width -> Int)
    (row : Fin rows) (column : Fin width)
    (vertical : ∀ other : Fin rows, ((row.val : Int) - other.val).natAbs = 1 ->
      height (other, column) = height (row, column) - 1)
    (maximum : IsStrictLocalMax (rowLine height row) ((0 : Fin 1), column)) :
    IsStrictLocalMax height (row, column) :=
  rowLine_increment_lift height row column (-1) (fun other adjacent => maximum ((0 : Fin 1), other) adjacent) vertical

theorem row_leaf_minimum_profile {rows width : Nat} (height : Cell rows width -> Int)
    (heightValid : IsHeight height) (row : Fin rows) (apex : Fin width)
    (unique : ∀ column : Fin width, IsStrictLocalExtremum height (row, column) -> column = apex)
    (transverse : ∀ column : Fin width, ∀ other : Fin rows,
      ((row.val : Int) - other.val).natAbs = 1 -> height (other, column) = height (row, column) + 1) :
    ∀ column : Fin width, height (row, column) = height (row, apex) + |(column.val : Int) - apex.val| := by
  apply rowLine_cone_min height heightValid row apex
  intro column minimum
  exact unique column (Or.inr (rowLine_minimum_lift height row column (transverse column) minimum))

theorem row_leaf_maximum_profile {rows width : Nat} (height : Cell rows width -> Int)
    (heightValid : IsHeight height) (row : Fin rows) (apex : Fin width)
    (unique : ∀ column : Fin width, IsStrictLocalExtremum height (row, column) -> column = apex)
    (transverse : ∀ column : Fin width, ∀ other : Fin rows,
      ((row.val : Int) - other.val).natAbs = 1 -> height (other, column) = height (row, column) - 1) :
    ∀ column : Fin width, height (row, column) = height (row, apex) - |(column.val : Int) - apex.val| := by
  apply rowLine_cone_max height heightValid row apex
  intro column maximum
  exact unique column (Or.inl (rowLine_maximum_lift height row column (transverse column) maximum))

theorem adjacent_frozen_row_offset {rows width : Nat} (height : Cell rows width -> Int)
    (heightValid : IsHeight height) (widthPositive : 0 < width)
    (row other : Fin rows) (frozen : frozenColumn (swapH height) other)
    (adjacent : ((row.val : Int) - other.val).natAbs = 1) (first second : Fin width) :
    height (other, first) - height (row, first) = height (other, second) - height (row, second) := by
  by_cases before : row.val < other.val
  · have previous : (⟨other.val - 1, by have := frozen.1; have := other.isLt; omega⟩ : Fin rows) = row := by
      apply Fin.ext
      change other.val - 1 = row.val
      omega
    have firstStep := frozenSlope_eq (swapH height) (swapH_isHeight heightValid) widthPositive other frozen first
    have secondStep := frozenSlope_eq (swapH height) (swapH_isHeight heightValid) widthPositive other frozen second
    simpa only [swapH, previous] using firstStep.trans secondStep.symm
  · have next : (⟨other.val + 1, frozen.2.1⟩ : Fin rows) = row := by
      apply Fin.ext
      change other.val + 1 = row.val
      omega
    have firstStep := frozenSlope_eq_right (swapH height) (swapH_isHeight heightValid) widthPositive other frozen first
    have secondStep := frozenSlope_eq_right (swapH height) (swapH_isHeight heightValid) widthPositive other frozen second
    simp only [swapH, next] at firstStep secondStep
    change height (row, first) - height (other, first) = _ at firstStep
    change height (row, second) - height (other, second) = _ at secondStep
    linarith

theorem row_transverse_minimum {rows width : Nat} (height : Cell rows width -> Int)
    (heightValid : IsHeight height) (row : Fin rows) (apex : Fin width)
    (minimum : IsStrictLocalMin height (row, apex))
    (neighborsFrozen : ∀ other : Fin rows, ((row.val : Int) - other.val).natAbs = 1 ->
      frozenColumn (swapH height) other) :
    ∀ column : Fin width, ∀ other : Fin rows, ((row.val : Int) - other.val).natAbs = 1 ->
      height (other, column) = height (row, column) + 1 := by
  intro column other adjacent
  have offset := adjacent_frozen_row_offset height heightValid (by have := apex.isLt; omega)
    row other (neighborsFrozen other adjacent) adjacent column apex
  have gridAdjacent : adj (row, apex) (other, apex) := by
    simp only [adj, gdist]
    omega
  have value := minimum (other, apex) gridAdjacent
  linarith

theorem row_transverse_maximum {rows width : Nat} (height : Cell rows width -> Int)
    (heightValid : IsHeight height) (row : Fin rows) (apex : Fin width)
    (maximum : IsStrictLocalMax height (row, apex))
    (neighborsFrozen : ∀ other : Fin rows, ((row.val : Int) - other.val).natAbs = 1 ->
      frozenColumn (swapH height) other) :
    ∀ column : Fin width, ∀ other : Fin rows, ((row.val : Int) - other.val).natAbs = 1 ->
      height (other, column) = height (row, column) - 1 := by
  intro column other adjacent
  have offset := adjacent_frozen_row_offset height heightValid (by have := apex.isLt; omega)
    row other (neighborsFrozen other adjacent) adjacent column apex
  have gridAdjacent : adj (row, apex) (other, apex) := by
    simp only [adj, gdist]
    omega
  have value := maximum (other, apex) gridAdjacent
  linarith

theorem row_leaf_minimum_from_frozen {rows width : Nat} (height : Cell rows width -> Int)
    (heightValid : IsHeight height) (row : Fin rows) (apex : Fin width)
    (minimum : IsStrictLocalMin height (row, apex))
    (unique : ∀ column : Fin width, IsStrictLocalExtremum height (row, column) -> column = apex)
    (neighborsFrozen : ∀ other : Fin rows, ((row.val : Int) - other.val).natAbs = 1 ->
      frozenColumn (swapH height) other) :
    ∀ column : Fin width, height (row, column) = height (row, apex) + |(column.val : Int) - apex.val| :=
  row_leaf_minimum_profile height heightValid row apex unique
    (row_transverse_minimum height heightValid row apex minimum neighborsFrozen)

theorem row_leaf_maximum_from_frozen {rows width : Nat} (height : Cell rows width -> Int)
    (heightValid : IsHeight height) (row : Fin rows) (apex : Fin width)
    (maximum : IsStrictLocalMax height (row, apex))
    (unique : ∀ column : Fin width, IsStrictLocalExtremum height (row, column) -> column = apex)
    (neighborsFrozen : ∀ other : Fin rows, ((row.val : Int) - other.val).natAbs = 1 ->
      frozenColumn (swapH height) other) :
    ∀ column : Fin width, height (row, column) = height (row, apex) - |(column.val : Int) - apex.val| :=
  row_leaf_maximum_profile height heightValid row apex unique
    (row_transverse_maximum height heightValid row apex maximum neighborsFrozen)

theorem row_leaf_signed_profile {rows width : Nat} (height : Cell rows width -> Int)
    (heightValid : IsHeight height) (rowsValid : 2 ≤ rows) (row : Fin rows) (apex : Fin width)
    (extremum : IsStrictLocalExtremum height (row, apex))
    (unique : ∀ column : Fin width, IsStrictLocalExtremum height (row, column) -> column = apex)
    (neighborsInactive : ∀ other : Fin rows, ((row.val : Int) - other.val).natAbs = 1 ->
      ¬ activeColumn (swapH height) other) :
    ∃ sign : Int, (sign = 1 ∨ sign = -1) ∧
      (∀ point : Cell rows width, adj (row, apex) point -> height point = height (row, apex) + sign) ∧
      (∀ column : Fin width, height (row, column) = height (row, apex) + sign * |(column.val : Int) - apex.val|) ∧
      ∀ column : Fin width, ∀ other : Fin rows, ((row.val : Int) - other.val).natAbs = 1 ->
        height (other, column) = height (row, column) + sign := by
  have neighborsFrozen : ∀ other : Fin rows, ((row.val : Int) - other.val).natAbs = 1 ->
      frozenColumn (swapH height) other := by
    intro other adjacent
    exact (column_active_or_frozen (swapH height) (swapH_isHeight heightValid)
      (by have := apex.isLt; omega) rowsValid other).resolve_left (neighborsInactive other adjacent)
  rcases extremum with maximum | minimum
  · refine ⟨-1, Or.inr rfl, maximum, ?_, ?_⟩
    · intro column
      simpa only [neg_one_mul, sub_eq_add_neg] using
        row_leaf_maximum_from_frozen height heightValid row apex maximum unique neighborsFrozen column
    · exact row_transverse_maximum height heightValid row apex maximum neighborsFrozen
  · refine ⟨1, Or.inl rfl, minimum, ?_, ?_⟩
    · intro column
      simpa only [one_mul] using
        row_leaf_minimum_from_frozen height heightValid row apex minimum unique neighborsFrozen column
    · exact row_transverse_minimum height heightValid row apex minimum neighborsFrozen

theorem same_sign_profiles_not_frozen_between {rows width : Nat} (height : Cell rows width -> Int)
    (heightValid : IsHeight height) (rowsPositive : 0 < rows)
    (firstRow secondRow : Fin rows) (firstApex secondApex column : Fin width) (sign : Int)
    (unitSign : sign = 1 ∨ sign = -1)
    (firstProfile : ∀ position : Fin width,
      height (firstRow, position) = height (firstRow, firstApex) + sign * |(position.val : Int) - firstApex.val|)
    (secondProfile : ∀ position : Fin width,
      height (secondRow, position) = height (secondRow, secondApex) + sign * |(position.val : Int) - secondApex.val|)
    (afterFirst : firstApex.val < column.val) (beforeSecond : column.val < secondApex.val) :
    ¬ frozenColumn height column := by
  intro frozen
  let previous : Fin width := ⟨column.val - 1, by have := column.isLt; omega⟩
  have previousValue : previous.val = column.val - 1 := rfl
  have firstValue := firstProfile column
  have firstPrevious := firstProfile previous
  have secondValue := secondProfile column
  have secondPrevious := secondProfile previous
  rw [abs_of_nonneg (show 0 ≤ (column.val : Int) - firstApex.val from by omega)] at firstValue
  rw [abs_of_nonneg (show 0 ≤ (previous.val : Int) - firstApex.val from by omega)] at firstPrevious
  rw [abs_of_nonpos (show (column.val : Int) - secondApex.val ≤ 0 from by omega)] at secondValue
  rw [abs_of_nonpos (show (previous.val : Int) - secondApex.val ≤ 0 from by omega)] at secondPrevious
  have firstStep := frozenSlope_eq height heightValid rowsPositive column frozen firstRow
  have secondStep := frozenSlope_eq height heightValid rowsPositive column frozen secondRow
  change height (firstRow, column) - height (firstRow, previous) = _ at firstStep
  change height (secondRow, column) - height (secondRow, previous) = _ at secondStep
  rcases unitSign with rfl | rfl <;>
    simp only [one_mul, neg_one_mul] at firstValue firstPrevious secondValue secondPrevious <;> omega

theorem same_sign_profiles_active_between {rows width : Nat} (height : Cell rows width -> Int)
    (heightValid : IsHeight height) (rowsPositive : 0 < rows) (widthValid : 2 ≤ width)
    (firstRow secondRow : Fin rows) (firstApex secondApex column : Fin width) (sign : Int)
    (unitSign : sign = 1 ∨ sign = -1)
    (firstProfile : ∀ position : Fin width,
      height (firstRow, position) = height (firstRow, firstApex) + sign * |(position.val : Int) - firstApex.val|)
    (secondProfile : ∀ position : Fin width,
      height (secondRow, position) = height (secondRow, secondApex) + sign * |(position.val : Int) - secondApex.val|)
    (afterFirst : firstApex.val < column.val) (beforeSecond : column.val < secondApex.val) :
    activeColumn height column :=
  (column_active_or_frozen height heightValid rowsPositive widthValid column).resolve_right
    (same_sign_profiles_not_frozen_between height heightValid rowsPositive firstRow secondRow
      firstApex secondApex column sign unitSign firstProfile secondProfile afterFirst beforeSecond)

theorem row_transverse_column_profile_sign {rows width : Nat} (height : Cell rows width -> Int)
    (row columnApex : Fin rows) (column : Fin width) (rowSign columnSign : Int)
    (columnUnit : columnSign = 1 ∨ columnSign = -1)
    (transverse : ∀ other : Fin rows, ((row.val : Int) - other.val).natAbs = 1 ->
      height (other, column) = height (row, column) + rowSign)
    (columnProfile : ∀ other : Fin rows,
      height (other, column) = height (columnApex, column) + columnSign * |(other.val : Int) - columnApex.val|)
    (differentRows : columnApex ≠ row) : rowSign = -columnSign := by
  have differentValues : columnApex.val ≠ row.val := fun equality => differentRows (Fin.ext equality)
  by_cases before : row.val < columnApex.val
  · let next : Fin rows := ⟨row.val + 1, by have := columnApex.isLt; omega⟩
    have nextValue : next.val = row.val + 1 := rfl
    have adjacent : ((row.val : Int) - next.val).natAbs = 1 := by omega
    have step := transverse next adjacent
    have current := columnProfile row
    have following := columnProfile next
    rw [abs_of_nonpos (show (row.val : Int) - columnApex.val ≤ 0 from by omega)] at current
    rw [abs_of_nonpos (show (next.val : Int) - columnApex.val ≤ 0 from by omega)] at following
    rcases columnUnit with rfl | rfl <;> simp only [one_mul, neg_one_mul] at current following <;> omega
  · let previous : Fin rows := ⟨row.val - 1, by have := row.isLt; omega⟩
    have previousValue : previous.val = row.val - 1 := rfl
    have adjacent : ((row.val : Int) - previous.val).natAbs = 1 := by omega
    have step := transverse previous adjacent
    have current := columnProfile row
    have preceding := columnProfile previous
    rw [abs_of_nonneg (show 0 ≤ (row.val : Int) - columnApex.val from by omega)] at current
    rw [abs_of_nonneg (show 0 ≤ (previous.val : Int) - columnApex.val from by omega)] at preceding
    rcases columnUnit with rfl | rfl <;> simp only [one_mul, neg_one_mul] at current preceding <;> omega

theorem interior_transverse_column_apex {rows width : Nat} (height : Cell rows width -> Int)
    (row columnApex : Fin rows) (column : Fin width) (rowSign columnSign : Int)
    (interior : 0 < row.val ∧ row.val + 1 < rows) (columnUnit : columnSign = 1 ∨ columnSign = -1)
    (transverse : ∀ other : Fin rows, ((row.val : Int) - other.val).natAbs = 1 ->
      height (other, column) = height (row, column) + rowSign)
    (columnProfile : ∀ other : Fin rows,
      height (other, column) = height (columnApex, column) + columnSign * |(other.val : Int) - columnApex.val|) :
    columnApex = row := by
  by_contra different
  have opposite := row_transverse_column_profile_sign height row columnApex column rowSign columnSign
    columnUnit transverse columnProfile different
  by_cases before : row.val < columnApex.val
  · let previous : Fin rows := ⟨row.val - 1, by have := row.isLt; omega⟩
    have previousValue : previous.val = row.val - 1 := rfl
    have adjacent : ((row.val : Int) - previous.val).natAbs = 1 := by omega
    have step := transverse previous adjacent
    have current := columnProfile row
    have preceding := columnProfile previous
    rw [abs_of_nonpos (show (row.val : Int) - columnApex.val ≤ 0 from by omega)] at current
    rw [abs_of_nonpos (show (previous.val : Int) - columnApex.val ≤ 0 from by omega)] at preceding
    rcases columnUnit with rfl | rfl <;> simp only [one_mul, neg_one_mul] at current preceding <;> omega
  · let next : Fin rows := ⟨row.val + 1, interior.2⟩
    have nextValue : next.val = row.val + 1 := rfl
    have adjacent : ((row.val : Int) - next.val).natAbs = 1 := by omega
    have step := transverse next adjacent
    have current := columnProfile row
    have following := columnProfile next
    rw [abs_of_nonneg (show 0 ≤ (row.val : Int) - columnApex.val from by omega)] at current
    rw [abs_of_nonneg (show 0 ≤ (next.val : Int) - columnApex.val from by omega)] at following
    rcases columnUnit with rfl | rfl <;> simp only [one_mul, neg_one_mul] at current following <;> omega

theorem interior_row_column_leaf_coincide {rows width : Nat} (height : Cell rows width -> Int)
    (heightValid : IsHeight height) (rowsValid : 2 ≤ rows) (row : Fin rows) (apex : Fin width)
    (extremum : IsStrictLocalExtremum height (row, apex))
    (unique : ∀ position : Fin width, IsStrictLocalExtremum height (row, position) -> position = apex)
    (neighborsInactive : ∀ other : Fin rows, ((row.val : Int) - other.val).natAbs = 1 ->
      ¬ activeColumn (swapH height) other)
    (interior : 0 < row.val ∧ row.val + 1 < rows) (columnApex : Fin rows) (column : Fin width)
    (columnExtremum : IsStrictLocalExtremum height (columnApex, column)) (columnSign : Int)
    (columnUnit : columnSign = 1 ∨ columnSign = -1)
    (columnProfile : ∀ other : Fin rows,
      height (other, column) = height (columnApex, column) + columnSign * |(other.val : Int) - columnApex.val|) :
    (columnApex, column) = (row, apex) := by
  obtain ⟨rowSign, _, _, _, transverse⟩ :=
    row_leaf_signed_profile height heightValid rowsValid row apex extremum unique neighborsInactive
  have sameRow := interior_transverse_column_apex height row columnApex column rowSign columnSign
    interior columnUnit (transverse column) columnProfile
  have sameColumn : column = apex := by
    apply unique column
    simpa only [sameRow] using columnExtremum
  exact Prod.ext sameRow sameColumn

#print axioms rowLine_isHeight
#print axioms rowLine_distance
#print axioms rowLine_cone_min
#print axioms rowLine_cone_max
#print axioms rowLine_increment_lift
#print axioms rowLine_minimum_lift
#print axioms rowLine_maximum_lift
#print axioms row_leaf_minimum_profile
#print axioms row_leaf_maximum_profile
#print axioms adjacent_frozen_row_offset
#print axioms row_transverse_minimum
#print axioms row_transverse_maximum
#print axioms row_leaf_minimum_from_frozen
#print axioms row_leaf_maximum_from_frozen
#print axioms row_leaf_signed_profile
#print axioms same_sign_profiles_not_frozen_between
#print axioms same_sign_profiles_active_between
#print axioms row_transverse_column_profile_sign
#print axioms interior_transverse_column_apex
#print axioms interior_row_column_leaf_coincide

end OrigamiCone.Sequel.LeafProfiles
