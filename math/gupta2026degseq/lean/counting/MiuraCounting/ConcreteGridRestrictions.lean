import MiuraCounting.ConcreteJointNormalForm

namespace OrigamiCone.Sequel.GridRestrictions

open OrigamiCone

def embedRows {rows width length : Nat} (start : Nat) (within : start + length ≤ rows)
    (point : Cell length width) : Cell rows width :=
  (⟨start + point.1.val, by have := point.1.isLt; omega⟩, point.2)

def restrictRows {rows width length : Nat} (height : Cell rows width -> Int)
    (start : Nat) (within : start + length ≤ rows) : Cell length width -> Int :=
  fun point => height (embedRows start within point)

theorem embedRows_distance {rows width length : Nat} (start : Nat) (within : start + length ≤ rows)
    (first second : Cell length width) :
    gdist (embedRows start within first) (embedRows start within second) = gdist first second := by
  unfold embedRows gdist
  dsimp only []
  omega

theorem embedRows_injective {rows width length : Nat} (start : Nat) (within : start + length ≤ rows) :
    Function.Injective (embedRows (width := width) start within) := by
  intro first second equality
  have rowEquality := congrArg (fun point : Cell rows width => point.1.val) equality
  have columnEquality : first.2 = second.2 := congrArg (fun point : Cell rows width => point.2) equality
  refine Prod.ext ?_ columnEquality
  apply Fin.ext
  change start + first.1.val = start + second.1.val at rowEquality
  omega

theorem restrictRows_isHeight {rows width length : Nat} (height : Cell rows width -> Int)
    (heightValid : IsHeight height) (start : Nat) (within : start + length ≤ rows) :
    IsHeight (restrictRows height start within) := by
  intro first second adjacent
  change |height (embedRows start within first) - height (embedRows start within second)| = 1
  apply heightValid
  change gdist (embedRows start within first) (embedRows start within second) = 1
  rw [embedRows_distance]
  exact adjacent

theorem embedRows_neighbors {rows width length : Nat} (start : Nat) (within : start + length ≤ rows)
    (point : Cell length width) (lowerSafe : start = 0 ∨ 0 < point.1.val)
    (upperSafe : start + length = rows ∨ point.1.val + 1 < length)
    (neighbor : Cell rows width) (adjacent : adj (embedRows start within point) neighbor) :
    ∃ original : Cell length width, adj point original ∧ embedRows start within original = neighbor := by
  have bounds : start ≤ neighbor.1.val ∧ neighbor.1.val < start + length := by
    have pointWithin := point.1.isLt
    have neighborWithin := neighbor.1.isLt
    have distance := adjacent
    unfold adj gdist embedRows at distance
    dsimp only [] at distance
    omega
  let original : Cell length width :=
    (⟨neighbor.1.val - start, by omega⟩, neighbor.2)
  have recovered : embedRows start within original = neighbor := by
    apply Prod.ext
    · apply Fin.ext
      change start + (neighbor.1.val - start) = neighbor.1.val
      omega
    · rfl
  refine ⟨original, ?_, recovered⟩
  change gdist point original = 1
  rw [← embedRows_distance start within point original, recovered]
  exact adjacent

theorem restrictRows_minimum_lift {rows width length : Nat} (height : Cell rows width -> Int)
    (start : Nat) (within : start + length ≤ rows) (point : Cell length width)
    (lowerSafe : start = 0 ∨ 0 < point.1.val) (upperSafe : start + length = rows ∨ point.1.val + 1 < length)
    (minimum : IsStrictLocalMin (restrictRows height start within) point) :
    IsStrictLocalMin height (embedRows start within point) := by
  intro neighbor adjacent
  obtain ⟨original, originalAdjacent, recovered⟩ :=
    embedRows_neighbors start within point lowerSafe upperSafe neighbor adjacent
  have value := minimum original originalAdjacent
  change height (embedRows start within original) = height (embedRows start within point) + 1 at value
  rwa [recovered] at value

theorem restrictRows_maximum_lift {rows width length : Nat} (height : Cell rows width -> Int)
    (start : Nat) (within : start + length ≤ rows) (point : Cell length width)
    (lowerSafe : start = 0 ∨ 0 < point.1.val) (upperSafe : start + length = rows ∨ point.1.val + 1 < length)
    (maximum : IsStrictLocalMax (restrictRows height start within) point) :
    IsStrictLocalMax height (embedRows start within point) := by
  intro neighbor adjacent
  obtain ⟨original, originalAdjacent, recovered⟩ :=
    embedRows_neighbors start within point lowerSafe upperSafe neighbor adjacent
  have value := maximum original originalAdjacent
  change height (embedRows start within original) = height (embedRows start within point) - 1 at value
  rwa [recovered] at value

theorem restrictRows_extremum_lift {rows width length : Nat} (height : Cell rows width -> Int)
    (start : Nat) (within : start + length ≤ rows) (point : Cell length width)
    (lowerSafe : start = 0 ∨ 0 < point.1.val) (upperSafe : start + length = rows ∨ point.1.val + 1 < length)
    (extremum : IsStrictLocalExtremum (restrictRows height start within) point) :
    IsStrictLocalExtremum height (embedRows start within point) := by
  rcases extremum with maximum | minimum
  · exact Or.inl (restrictRows_maximum_lift height start within point lowerSafe upperSafe maximum)
  · exact Or.inr (restrictRows_minimum_lift height start within point lowerSafe upperSafe minimum)

theorem not_extremum_opposite_neighbors {rows width : Nat} (height : Cell rows width -> Int)
    (point first second : Cell rows width) (firstAdjacent : adj point first) (secondAdjacent : adj point second)
    (sign : Int) (unitSign : sign = 1 ∨ sign = -1)
    (firstValue : height first = height point + sign) (secondValue : height second = height point - sign) :
    ¬ IsStrictLocalExtremum height point := by
  rintro (maximum | minimum)
  · have firstStep := maximum first firstAdjacent
    have secondStep := maximum second secondAdjacent
    rcases unitSign with rfl | rfl <;> omega
  · have firstStep := minimum first firstAdjacent
    have secondStep := minimum second secondAdjacent
    rcases unitSign with rfl | rfl <;> omega

theorem row_cut_right_not_extremum {rows width length : Nat} (height : Cell rows width -> Int)
    (start : Nat) (within : start + length ≤ rows) (lengthValid : 2 ≤ length) (widthValid : 2 ≤ width)
    (row : Fin rows) (apex : Fin width) (localRow : Fin length) (atRow : start + localRow.val = row.val)
    (apexBeforeRight : apex.val < width - 1) (sign : Int) (unitSign : sign = 1 ∨ sign = -1)
    (profile : ∀ column : Fin width, height (row, column) = height (row, apex) + sign * |(column.val : Int) - apex.val|)
    (transverse : ∀ column : Fin width, ∀ other : Fin rows, ((row.val : Int) - other.val).natAbs = 1 ->
      height (other, column) = height (row, column) + sign) :
    ¬ IsStrictLocalExtremum (restrictRows height start within)
      (localRow, ⟨width - 1, by omega⟩) := by
  let right : Fin width := ⟨width - 1, by omega⟩
  let preceding : Fin width := ⟨width - 2, by omega⟩
  obtain ⟨neighbor, adjacent⟩ := exists_neighbor (m := length) (n := 1) (by omega) (localRow, (0 : Fin 1))
  have localAdjacent : ((localRow.val : Int) - neighbor.1.val).natAbs = 1 := by
    have zero := neighbor.2.isLt
    unfold adj gdist at adjacent
    dsimp only [] at adjacent
    omega
  let fullRow : Fin rows := ⟨start + neighbor.1.val, by have := neighbor.1.isLt; omega⟩
  have fullAdjacent : ((row.val : Int) - fullRow.val).natAbs = 1 := by
    change ((row.val : Int) - (start + neighbor.1.val : Nat)).natAbs = 1
    omega
  have pointRow : (⟨start + localRow.val, by have := localRow.isLt; omega⟩ : Fin rows) = row := Fin.ext atRow
  refine not_extremum_opposite_neighbors (restrictRows height start within)
    (localRow, right) (neighbor.1, right) (localRow, preceding) ?_ ?_ sign unitSign ?_ ?_
  · simp only [adj, gdist]
    omega
  · simp only [adj, gdist]
    dsimp only [right, preceding]
    omega
  · change height (fullRow, right) =
      height (⟨start + localRow.val, by have := localRow.isLt; omega⟩, right) + sign
    rw [pointRow]
    exact transverse right fullRow fullAdjacent
  · change height (⟨start + localRow.val, by have := localRow.isLt; omega⟩, preceding) =
      height (⟨start + localRow.val, by have := localRow.isLt; omega⟩, right) - sign
    rw [pointRow]
    have rightValue := profile right
    have precedingValue := profile preceding
    have rightIndex : right.val = width - 1 := rfl
    have precedingIndex : preceding.val = width - 2 := rfl
    rw [abs_of_nonneg (show 0 ≤ (right.val : Int) - apex.val from by omega)] at rightValue
    rw [abs_of_nonneg (show 0 ≤ (preceding.val : Int) - apex.val from by omega)] at precedingValue
    rcases unitSign with rfl | rfl <;> simp only [one_mul, neg_one_mul] at rightValue precedingValue <;> omega

theorem right_extremum_above_cut {rows width : Nat} (height : Cell rows width -> Int)
    (heightValid : IsHeight height) (widthValid : 2 ≤ width) (cut : Fin rows) (cutPositive : 0 < cut.val)
    (apex : Fin width) (apexBeforeRight : apex.val < width - 1) (sign : Int) (unitSign : sign = 1 ∨ sign = -1)
    (profile : ∀ column : Fin width, height (cut, column) = height (cut, apex) + sign * |(column.val : Int) - apex.val|)
    (transverse : ∀ column : Fin width, ∀ other : Fin rows, ((cut.val : Int) - other.val).natAbs = 1 ->
      height (other, column) = height (cut, column) + sign) :
    ∃ row : Fin rows, row.val < cut.val ∧
      IsStrictLocalExtremum height (row, ⟨width - 1, by omega⟩) := by
  have within : 0 + (cut.val + 1) ≤ rows := by have := cut.isLt; omega
  let restricted := restrictRows height 0 within
  have valid := restrictRows_isHeight height heightValid 0 within
  obtain ⟨row, extremum⟩ := lastColumn_active restricted valid (by omega) widthValid
  let cutRow : Fin (cut.val + 1) := ⟨cut.val, by omega⟩
  have noCut := row_cut_right_not_extremum height 0 within (by omega) widthValid cut apex cutRow
    (by change 0 + cut.val = cut.val; omega) apexBeforeRight sign unitSign profile transverse
  have different : row.val ≠ cut.val := by
    intro equality
    apply noCut
    have sameRow : row = cutRow := Fin.ext equality
    simpa only [sameRow] using extremum
  have lifted := restrictRows_extremum_lift height 0 within (row, ⟨width - 1, by omega⟩)
    (Or.inl rfl) (Or.inr (by change row.val + 1 < cut.val + 1; have := row.isLt; omega)) extremum
  refine ⟨⟨row.val, by have := row.isLt; have := cut.isLt; omega⟩, ?_, ?_⟩
  · change row.val < cut.val
    have := row.isLt
    omega
  · simpa only [embedRows, Nat.zero_add] using lifted

theorem right_extremum_below_cut {rows width : Nat} (height : Cell rows width -> Int)
    (heightValid : IsHeight height) (widthValid : 2 ≤ width) (cut : Fin rows) (belowCut : cut.val + 1 < rows)
    (apex : Fin width) (apexBeforeRight : apex.val < width - 1) (sign : Int) (unitSign : sign = 1 ∨ sign = -1)
    (profile : ∀ column : Fin width, height (cut, column) = height (cut, apex) + sign * |(column.val : Int) - apex.val|)
    (transverse : ∀ column : Fin width, ∀ other : Fin rows, ((cut.val : Int) - other.val).natAbs = 1 ->
      height (other, column) = height (cut, column) + sign) :
    ∃ row : Fin rows, cut.val < row.val ∧
      IsStrictLocalExtremum height (row, ⟨width - 1, by omega⟩) := by
  have within : cut.val + (rows - cut.val) ≤ rows := by omega
  let restricted := restrictRows height cut.val within
  have valid := restrictRows_isHeight height heightValid cut.val within
  obtain ⟨row, extremum⟩ := lastColumn_active restricted valid (by omega) widthValid
  let cutRow : Fin (rows - cut.val) := ⟨0, by omega⟩
  have noCut := row_cut_right_not_extremum height cut.val within (by omega) widthValid cut apex cutRow
    (by change cut.val + 0 = cut.val; omega) apexBeforeRight sign unitSign profile transverse
  have rowPositive : 0 < row.val := by
    by_contra notPositive
    apply noCut
    have sameRow : row = cutRow := Fin.ext (by change row.val = 0; omega)
    simpa only [sameRow] using extremum
  have lifted := restrictRows_extremum_lift height cut.val within (row, ⟨width - 1, by omega⟩)
    (Or.inr rowPositive) (Or.inl (by omega)) extremum
  refine ⟨⟨cut.val + row.val, by have := row.isLt; omega⟩, ?_, ?_⟩
  · change cut.val < cut.val + row.val
    omega
  · exact lifted

#print axioms embedRows_distance
#print axioms embedRows_injective
#print axioms restrictRows_isHeight
#print axioms embedRows_neighbors
#print axioms restrictRows_minimum_lift
#print axioms restrictRows_maximum_lift
#print axioms restrictRows_extremum_lift
#print axioms not_extremum_opposite_neighbors
#print axioms row_cut_right_not_extremum
#print axioms right_extremum_above_cut
#print axioms right_extremum_below_cut

end OrigamiCone.Sequel.GridRestrictions
