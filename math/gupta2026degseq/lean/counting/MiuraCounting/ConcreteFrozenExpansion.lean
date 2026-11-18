import OrigamiCone.SequelEdContractExtremum
import OrigamiCone.SequelEdFrozenBridge

namespace OrigamiCone.Sequel.FrozenExpansion

open OrigamiCone

def expandAfter {rows width : Nat} (height : Cell rows width -> Int)
    (mark : Fin width) (slope : Int) : Cell rows (width + 1) -> Int := fun point =>
  if before : point.2.val ≤ mark.val then
    height (point.1, ⟨point.2.val, by have := mark.isLt; omega⟩)
  else
    height (point.1, ⟨point.2.val - 1, by have := point.2.isLt; omega⟩) + slope

theorem expandAfter_left {rows width : Nat} (height : Cell rows width -> Int)
    (mark : Fin width) (slope : Int) (point : Cell rows (width + 1))
    (before : point.2.val ≤ mark.val) :
    expandAfter height mark slope point =
      height (point.1, ⟨point.2.val, by have := mark.isLt; omega⟩) := by
  simp only [expandAfter, before, dite_true]

theorem expandAfter_right {rows width : Nat} (height : Cell rows width -> Int)
    (mark : Fin width) (slope : Int) (point : Cell rows (width + 1))
    (after : ¬ point.2.val ≤ mark.val) :
    expandAfter height mark slope point =
      height (point.1, ⟨point.2.val - 1, by have := point.2.isLt; omega⟩) + slope := by
  simp only [expandAfter, after, dite_false]

theorem inserted_frozen {rows width : Nat} (height : Cell rows width -> Int)
    (mark : Fin width) (interior : mark.val + 1 < width) (slope : Int)
    (rightStep : ∀ row : Fin rows,
      height (row, ⟨mark.val + 1, interior⟩) - height (row, mark) = slope) :
    frozenColumn (expandAfter height mark slope) ⟨mark.val + 1, by omega⟩ := by
  refine ⟨by change 0 < mark.val + 1; omega,
    by change mark.val + 1 + 1 < width + 1; omega, ?_⟩
  intro row
  change expandAfter height mark slope (row, ⟨mark.val, by omega⟩) +
      expandAfter height mark slope (row, ⟨mark.val + 2, by omega⟩) =
    2 * expandAfter height mark slope (row, ⟨mark.val + 1, by omega⟩)
  have leftValue : expandAfter height mark slope (row, ⟨mark.val, by omega⟩) =
      height (row, mark) := by simp [expandAfter]
  have middleValue : expandAfter height mark slope (row, ⟨mark.val + 1, by omega⟩) =
      height (row, mark) + slope := by
    simp [expandAfter, show ¬ mark.val + 1 ≤ mark.val from by omega]
  have rightValue : expandAfter height mark slope (row, ⟨mark.val + 2, by omega⟩) =
      height (row, ⟨mark.val + 1, interior⟩) + slope := by
    simp [expandAfter, show ¬ mark.val + 2 ≤ mark.val from by omega]
  rw [leftValue, middleValue, rightValue]
  have step := rightStep row
  linarith

theorem inserted_slope {rows width : Nat} (height : Cell rows width -> Int)
    (rowsPositive : 0 < rows) (mark : Fin width) (slope : Int)
    (frozen : frozenColumn (expandAfter height mark slope) ⟨mark.val + 1, by have := mark.isLt; omega⟩) :
    frozenSlope (expandAfter height mark slope) rowsPositive ⟨mark.val + 1, by have := mark.isLt; omega⟩
      frozen = slope := by
  unfold frozenSlope
  simp [expandAfter, show ¬ mark.val + 1 ≤ mark.val from by omega]

theorem contract_inserted {rows width : Nat} (height : Cell rows width -> Int)
    (rowsPositive : 0 < rows) (mark : Fin width) (slope : Int)
    (frozen : frozenColumn (expandAfter height mark slope) ⟨mark.val + 1, by have := mark.isLt; omega⟩) :
    contractAt (expandAfter height mark slope) rowsPositive
      ⟨mark.val + 1, by have := mark.isLt; omega⟩ frozen = height := by
  funext point
  by_cases before : point.2.val ≤ mark.val
  · rw [contractAt_left _ _ _ _ point (by change point.2.val < mark.val + 1; omega)]
    rw [expandAfter_left _ _ _ _ (by change point.2.val ≤ mark.val; exact before)]
    apply congrArg height
    apply Prod.ext
    · rfl
    · apply Fin.ext
      rfl
  · rw [contractAt_right _ _ _ _ point (by change ¬ point.2.val < mark.val + 1; omega)]
    rw [expandAfter_right _ _ _ _ (by change ¬ point.2.val + 1 ≤ mark.val; omega),
      inserted_slope height rowsPositive mark slope frozen]
    have restored : (⟨point.2.val + 1 - 1, by have := point.2.isLt; omega⟩ : Fin width) = point.2 := by
      apply Fin.ext
      change point.2.val + 1 - 1 = point.2.val
      omega
    rw [restored]
    have samePoint : (point.1, point.2) = point := by cases point; rfl
    rw [samePoint]
    ring

theorem expandAfter_isHeight {rows width : Nat} (height : Cell rows width -> Int)
    (heightValid : IsHeight height) (mark : Fin width) (slope : Int)
    (unitSlope : slope = 1 ∨ slope = -1) : IsHeight (expandAfter height mark slope) := by
  have seam : ∀ source target : Cell rows (width + 1), adj source target ->
      source.2.val ≤ mark.val -> ¬ target.2.val ≤ mark.val ->
      |expandAfter height mark slope source - expandAfter height mark slope target| = 1 := by
    intro source target adjacent sourceBefore targetAfter
    have sameRow : source.1 = target.1 := by
      apply Fin.ext
      simp only [adj, gdist] at adjacent
      omega
    have sameColumn : source.2.val = target.2.val - 1 := by
      simp only [adj, gdist] at adjacent
      omega
    rw [expandAfter_left _ _ _ source sourceBefore, expandAfter_right _ _ _ target targetAfter]
    have samePoint :
      ((source.1, ⟨source.2.val, by have := mark.isLt; omega⟩) : Cell rows width) =
          (target.1, ⟨target.2.val - 1, by have := target.2.isLt; omega⟩) :=
      Prod.ext sameRow (Fin.ext sameColumn)
    rw [samePoint]
    have difference : height (target.1, ⟨target.2.val - 1, by have := target.2.isLt; omega⟩) -
        (height (target.1, ⟨target.2.val - 1, by have := target.2.isLt; omega⟩) + slope) = -slope := by ring
    rw [difference]
    rcases unitSlope with rfl | rfl <;> decide
  intro source target adjacent
  by_cases sourceBefore : source.2.val ≤ mark.val
  · by_cases targetBefore : target.2.val ≤ mark.val
    · rw [expandAfter_left _ _ _ source sourceBefore, expandAfter_left _ _ _ target targetBefore]
      apply heightValid
      simpa only [adj, gdist, Prod.fst, Prod.snd, Fin.val_mk] using adjacent
    · exact seam source target adjacent sourceBefore targetBefore
  · by_cases targetBefore : target.2.val ≤ mark.val
    · rw [abs_sub_comm]
      exact seam target source (adj_comm.mp adjacent) targetBefore sourceBefore
    · rw [expandAfter_right _ _ _ source sourceBefore, expandAfter_right _ _ _ target targetBefore]
      have projectedAdjacent : adj
          ((source.1, ⟨source.2.val - 1, by have := source.2.isLt; omega⟩) : Cell rows width)
          (target.1, ⟨target.2.val - 1, by have := target.2.isLt; omega⟩) := by
        simp only [adj, gdist] at adjacent ⊢
        omega
      calc
        _ = |height (source.1, ⟨source.2.val - 1, by have := source.2.isLt; omega⟩) -
            height (target.1, ⟨target.2.val - 1, by have := target.2.isLt; omega⟩)| := by
          congr 1
          ring
        _ = 1 := heightValid _ _ projectedAdjacent

theorem expandAfter_canonical {rows width : Nat} (height : Cell rows width -> Int)
    (heightCanonical : IsCanonicalHeight height) (mark : Fin width) (slope : Int)
    (unitSlope : slope = 1 ∨ slope = -1) :
    IsCanonicalHeight (expandAfter height mark slope) := by
  refine ⟨expandAfter_isHeight height heightCanonical.1 mark slope unitSlope, ?_⟩
  intro point rowZero columnZero
  rw [expandAfter_left _ _ _ point (by rw [columnZero]; omega)]
  exact heightCanonical.2 _ rowZero (by simpa only [Prod.snd, Fin.val_mk] using columnZero)

theorem retained_frozen {rows width : Nat} (height : Cell rows width -> Int)
    (heightValid : IsHeight height) (rowsPositive : 0 < rows)
    (mark : Fin width) (markFrozen : frozenColumn height mark) :
    frozenColumn (expandAfter height mark (frozenSlope height rowsPositive mark markFrozen))
      ⟨mark.val, by have := mark.isLt; omega⟩ := by
  let slope := frozenSlope height rowsPositive mark markFrozen
  refine ⟨markFrozen.1, by change mark.val + 1 < width + 1; have := mark.isLt; omega, ?_⟩
  intro row
  change expandAfter height mark slope (row, ⟨mark.val - 1, by have := markFrozen.1; omega⟩) +
      expandAfter height mark slope (row, ⟨mark.val + 1, by have := mark.isLt; omega⟩) =
    2 * expandAfter height mark slope (row, ⟨mark.val, by have := mark.isLt; omega⟩)
  have leftValue : expandAfter height mark slope
      (row, ⟨mark.val - 1, by have := markFrozen.1; omega⟩) =
        height (row, ⟨mark.val - 1, by have := markFrozen.1; omega⟩) := by
    simp [expandAfter]
  have middleValue : expandAfter height mark slope (row, ⟨mark.val, by have := mark.isLt; omega⟩) =
      height (row, mark) := by simp [expandAfter]
  have rightValue : expandAfter height mark slope (row, ⟨mark.val + 1, by have := mark.isLt; omega⟩) =
      height (row, mark) + slope := by
    simp [expandAfter, show ¬ mark.val + 1 ≤ mark.val from by omega]
  rw [leftValue, middleValue, rightValue]
  have step := frozenSlope_eq height heightValid rowsPositive mark markFrozen row
  change height (row, mark) - height (row, ⟨mark.val - 1, by have := markFrozen.1; omega⟩) = slope at step
  linarith

theorem expandAfter_numExtrema {rows width : Nat} (height : Cell rows width -> Int)
    (heightValid : IsHeight height) (rowsPositive : 0 < rows)
    (mark : Fin width) (markFrozen : frozenColumn height mark) :
    numExtrema (expandAfter height mark (frozenSlope height rowsPositive mark markFrozen)) =
      numExtrema height := by
  let slope := frozenSlope height rowsPositive mark markFrozen
  have frozen := inserted_frozen height mark markFrozen.2.1 slope
    (frozenSlope_eq_right height heightValid rowsPositive mark markFrozen)
  have expandedValid := expandAfter_isHeight height heightValid mark slope
    (frozenSlope_pm height heightValid rowsPositive mark markFrozen)
  have recovered := contract_inserted height rowsPositive mark slope frozen
  have count := contractAt_numExtrema_eq (expandAfter height mark slope) expandedValid
    rowsPositive ⟨mark.val + 1, by have := mark.isLt; omega⟩ frozen
  rw [recovered] at count
  exact count.symm

theorem expand_contract_at {rows width : Nat} (height : Cell rows (width + 1) -> Int)
    (heightValid : IsHeight height) (rowsPositive : 0 < rows)
    (column : Fin (width + 1)) (columnFrozen : frozenColumn height column) :
    expandAfter (contractAt height rowsPositive column columnFrozen)
      ⟨column.val, by have := columnFrozen.2.1; omega⟩
      (frozenSlope height rowsPositive column columnFrozen) = height := by
  funext point
  by_cases before : point.2.val ≤ column.val
  · rw [expandAfter_left _ _ _ _ before]
    by_cases strictlyBefore : point.2.val < column.val
    · rw [contractAt_left _ _ _ _ _ strictlyBefore]
      apply congrArg height
      apply Prod.ext
      · rfl
      · apply Fin.ext
        rfl
    · rw [contractAt_right _ _ _ _ _ strictlyBefore]
      have sameColumn : point.2.val = column.val := by omega
      change height (point.1, ⟨point.2.val + 1, by have := columnFrozen.2.1; omega⟩) -
        frozenSlope height rowsPositive column columnFrozen = height point
      have nextColumn :
          (⟨point.2.val + 1, by have := columnFrozen.2.1; omega⟩ : Fin (width + 1)) =
            ⟨column.val + 1, columnFrozen.2.1⟩ := Fin.ext (by
              change point.2.val + 1 = column.val + 1
              omega)
      have samePoint : point = (point.1, column) := Prod.ext rfl (Fin.ext sameColumn)
      have currentValue := congrArg height samePoint
      have nextValue :
          height (point.1, ⟨point.2.val + 1, by have := columnFrozen.2.1; omega⟩) =
            height (point.1, ⟨column.val + 1, columnFrozen.2.1⟩) := by
        apply congrArg height
        exact Prod.ext rfl nextColumn
      have step := frozenSlope_eq_right height heightValid rowsPositive column columnFrozen point.1
      linarith
  · rw [expandAfter_right _ _ _ _ before]
    rw [contractAt_right _ _ _ _ _ (by change ¬ point.2.val - 1 < column.val; omega)]
    have restored : (⟨point.2.val - 1 + 1, by have := point.2.isLt; omega⟩ : Fin (width + 1)) =
        point.2 := by
      apply Fin.ext
      change point.2.val - 1 + 1 = point.2.val
      omega
    rw [restored]
    have samePoint : (point.1, point.2) = point := by cases point; rfl
    rw [samePoint]
    ring

theorem contracted_marker_frozen {rows width : Nat} (height : Cell rows (width + 1) -> Int)
    (heightValid : IsHeight height) (rowsPositive : 0 < rows)
    (column : Fin (width + 1)) (columnFrozen : frozenColumn height column)
    (nextFrozen : frozenColumn height ⟨column.val + 1, columnFrozen.2.1⟩) :
    frozenColumn (contractAt height rowsPositive column columnFrozen)
      ⟨column.val, by have := columnFrozen.2.1; omega⟩ := by
  refine ⟨columnFrozen.1, by
    change column.val + 1 < width
    have nextInterior := nextFrozen.2.1
    change column.val + 1 + 1 < width + 1 at nextInterior
    omega, ?_⟩
  intro row
  change contractAt height rowsPositive column columnFrozen
        (row, ⟨column.val - 1, by have := columnFrozen.1; have := columnFrozen.2.1; omega⟩) +
      contractAt height rowsPositive column columnFrozen
        (row, ⟨column.val + 1, by have := nextFrozen.2.1; change column.val + 1 + 1 < width + 1 at this; omega⟩) =
    2 * contractAt height rowsPositive column columnFrozen
      (row, ⟨column.val, by have := columnFrozen.2.1; omega⟩)
  rw [contractAt_left _ _ _ _ _ (by change column.val - 1 < column.val; have := columnFrozen.1; omega),
    contractAt_right _ _ _ _ _ (by change ¬ column.val + 1 < column.val; omega),
    contractAt_right _ _ _ _ _ (by change ¬ column.val < column.val; omega)]
  have nextRelation :
      height (row, column) +
        height (row, ⟨column.val + 2, by have := nextFrozen.2.1; change column.val + 1 + 1 < width + 1 at this; omega⟩) =
          2 * height (row, ⟨column.val + 1, columnFrozen.2.1⟩) := by
    simpa using nextFrozen.2.2 row
  have leftStep := frozenSlope_eq height heightValid rowsPositive column columnFrozen row
  have rightStep := frozenSlope_eq_right height heightValid rowsPositive column columnFrozen row
  linarith

def expandRows {rows width : Nat} (height : Cell rows width -> Int)
    (mark : Fin rows) (slope : Int) : Cell (rows + 1) width -> Int :=
  swapH (expandAfter (swapH height) mark slope)

theorem expandRows_isHeight {rows width : Nat} (height : Cell rows width -> Int)
    (heightValid : IsHeight height) (mark : Fin rows) (slope : Int)
    (unitSlope : slope = 1 ∨ slope = -1) : IsHeight (expandRows height mark slope) := by
  exact swapH_isHeight (expandAfter_isHeight (swapH height) (swapH_isHeight heightValid)
    mark slope unitSlope)

theorem expandRows_canonical {rows width : Nat} (height : Cell rows width -> Int)
    (heightCanonical : IsCanonicalHeight height) (mark : Fin rows) (slope : Int)
    (unitSlope : slope = 1 ∨ slope = -1) : IsCanonicalHeight (expandRows height mark slope) := by
  exact swapH_isCanonical (expandAfter_canonical (swapH height) (swapH_isCanonical heightCanonical)
    mark slope unitSlope)

theorem expandRows_numExtrema {rows width : Nat} (height : Cell rows width -> Int)
    (heightValid : IsHeight height) (widthPositive : 0 < width)
    (mark : Fin rows) (markFrozen : frozenColumn (swapH height) mark) :
    numExtrema (expandRows height mark (frozenSlope (swapH height) widthPositive mark markFrozen)) =
      numExtrema height := by
  unfold expandRows
  rw [swapH_numExtrema, expandAfter_numExtrema (swapH height) (swapH_isHeight heightValid)
    widthPositive mark markFrozen, swapH_numExtrema]

theorem expandAxes_commute {rows width : Nat} (height : Cell rows width -> Int)
    (rowMark : Fin rows) (columnMark : Fin width) (rowSlope columnSlope : Int) :
    expandRows (expandAfter height columnMark columnSlope) rowMark rowSlope =
      expandAfter (expandRows height rowMark rowSlope) columnMark columnSlope := by
  funext point
  rcases point with ⟨row, column⟩
  by_cases rowBefore : row.val ≤ rowMark.val
  · by_cases columnBefore : column.val ≤ columnMark.val
    · simp [expandRows, expandAfter, swapH, rowBefore, columnBefore]
    · simp [expandRows, expandAfter, swapH, rowBefore, columnBefore]
  · by_cases columnBefore : column.val ≤ columnMark.val
    · simp [expandRows, expandAfter, swapH, rowBefore, columnBefore]
    · simp [expandRows, expandAfter, swapH, rowBefore, columnBefore]
      ring

theorem expandAfter_maximum_iff {rows width : Nat} (height : Cell rows width -> Int)
    (heightValid : IsHeight height) (rowsPositive : 0 < rows) (mark : Fin width) (slope : Int)
    (unitSlope : slope = 1 ∨ slope = -1)
    (insertedFrozen : frozenColumn (expandAfter height mark slope)
      ⟨mark.val + 1, by have := mark.isLt; omega⟩) (point : Cell rows width) :
    IsStrictLocalMax height point ↔
      IsStrictLocalMax (expandAfter height mark slope)
        (cellExpand ⟨mark.val + 1, by have := mark.isLt; omega⟩ point
          insertedFrozen.1 insertedFrozen.2.1) := by
  have expandedValid := expandAfter_isHeight height heightValid mark slope unitSlope
  have recovered := contract_inserted height rowsPositive mark slope insertedFrozen
  have relation :
      IsStrictLocalMax (contractAt (expandAfter height mark slope) rowsPositive
        ⟨mark.val + 1, by have := mark.isLt; omega⟩ insertedFrozen) point ↔
      IsStrictLocalMax (expandAfter height mark slope)
        (cellExpand ⟨mark.val + 1, by have := mark.isLt; omega⟩ point
          insertedFrozen.1 insertedFrozen.2.1) :=
    ⟨h_max_of_contractAt_isSLM _ expandedValid rowsPositive _ insertedFrozen point,
      contractAt_isSLM_of_h_max _ expandedValid rowsPositive _ insertedFrozen point⟩
  simpa only [recovered] using relation

theorem expandAfter_minimum_iff {rows width : Nat} (height : Cell rows width -> Int)
    (heightValid : IsHeight height) (rowsPositive : 0 < rows) (mark : Fin width) (slope : Int)
    (unitSlope : slope = 1 ∨ slope = -1)
    (insertedFrozen : frozenColumn (expandAfter height mark slope)
      ⟨mark.val + 1, by have := mark.isLt; omega⟩) (point : Cell rows width) :
    IsStrictLocalMin height point ↔
      IsStrictLocalMin (expandAfter height mark slope)
        (cellExpand ⟨mark.val + 1, by have := mark.isLt; omega⟩ point
          insertedFrozen.1 insertedFrozen.2.1) := by
  have expandedValid := expandAfter_isHeight height heightValid mark slope unitSlope
  have recovered := contract_inserted height rowsPositive mark slope insertedFrozen
  have relation :
      IsStrictLocalMin (contractAt (expandAfter height mark slope) rowsPositive
        ⟨mark.val + 1, by have := mark.isLt; omega⟩ insertedFrozen) point ↔
      IsStrictLocalMin (expandAfter height mark slope)
        (cellExpand ⟨mark.val + 1, by have := mark.isLt; omega⟩ point
          insertedFrozen.1 insertedFrozen.2.1) :=
    ⟨h_min_of_contractAt_isSLm _ expandedValid rowsPositive _ insertedFrozen point,
      contractAt_isSLm_of_h_min _ expandedValid rowsPositive _ insertedFrozen point⟩
  simpa only [recovered] using relation

theorem rowFrozen_expandAfter_iff {rows width : Nat} (height : Cell rows width -> Int)
    (columnMark : Fin width) (slope : Int) (rowMark : Fin rows) :
    frozenColumn (swapH (expandAfter height columnMark slope)) rowMark ↔
      frozenColumn (swapH height) rowMark := by
  constructor
  · rintro ⟨positive, interior, relations⟩
    refine ⟨positive, interior, ?_⟩
    intro column
    by_cases before : column.val ≤ columnMark.val
    · have keptBound : column.val < width + 1 := by have := column.isLt; omega
      have relation := relations ⟨column.val, keptBound⟩
      simpa only [swapH, Prod.swap, expandAfter, Fin.val_mk, before, dite_true] using relation
    · have keptBound : column.val + 1 < width + 1 := by have := column.isLt; omega
      have after : ¬ column.val + 1 ≤ columnMark.val := by omega
      have relation := relations ⟨column.val + 1, keptBound⟩
      have shifted :
          (swapH height) (column, ⟨rowMark.val - 1, by omega⟩) + slope +
            ((swapH height) (column, ⟨rowMark.val + 1, interior⟩) + slope) =
          2 * ((swapH height) (column, rowMark) + slope) := by
        simpa only [swapH, Prod.swap, expandAfter, Fin.val_mk, after, dite_false, Nat.add_sub_cancel] using relation
      linarith
  · rintro ⟨positive, interior, relations⟩
    refine ⟨positive, interior, ?_⟩
    intro column
    by_cases before : column.val ≤ columnMark.val
    · have oldBound : column.val < width := by have := columnMark.isLt; omega
      have relation := relations ⟨column.val, oldBound⟩
      simpa only [swapH, Prod.swap, expandAfter, Fin.val_mk, before, dite_true] using relation
    · have oldBound : column.val - 1 < width := by have := column.isLt; omega
      have relation := relations ⟨column.val - 1, oldBound⟩
      change expandAfter height columnMark slope
          (⟨rowMark.val - 1, by omega⟩, column) +
        expandAfter height columnMark slope (⟨rowMark.val + 1, interior⟩, column) =
          2 * expandAfter height columnMark slope (rowMark, column)
      simp only [expandAfter, before, dite_false]
      change height (⟨rowMark.val - 1, by omega⟩, ⟨column.val - 1, oldBound⟩) +
          height (⟨rowMark.val + 1, interior⟩, ⟨column.val - 1, oldBound⟩) =
            2 * height (rowMark, ⟨column.val - 1, oldBound⟩) at relation
      linarith

theorem columnFrozen_expandRows_iff {rows width : Nat} (height : Cell rows width -> Int)
    (rowMark : Fin rows) (slope : Int) (columnMark : Fin width) :
    frozenColumn (expandRows height rowMark slope) columnMark ↔ frozenColumn height columnMark := by
  change frozenColumn (swapH (expandAfter (swapH height) rowMark slope)) columnMark ↔
    frozenColumn (swapH (swapH height)) columnMark
  exact rowFrozen_expandAfter_iff (swapH height) rowMark slope columnMark

def AdditivelySeparable {rows width : Nat} (height : Cell rows width -> Int) : Prop :=
  ∃ rowValues : Fin rows -> Int, ∃ columnValues : Fin width -> Int,
    ∀ row column, height (row, column) = rowValues row + columnValues column

theorem separable_swap_iff {rows width : Nat} (height : Cell rows width -> Int) :
    AdditivelySeparable (swapH height) ↔ AdditivelySeparable height := by
  constructor
  · rintro ⟨rowValues, columnValues, equations⟩
    refine ⟨columnValues, rowValues, ?_⟩
    intro row column
    simpa [swapH, add_comm] using equations column row
  · rintro ⟨rowValues, columnValues, equations⟩
    refine ⟨columnValues, rowValues, ?_⟩
    intro row column
    simpa [swapH, add_comm] using equations column row

theorem expandAfter_separable_iff {rows width : Nat} (height : Cell rows width -> Int)
    (mark : Fin width) (slope : Int) :
    AdditivelySeparable (expandAfter height mark slope) ↔ AdditivelySeparable height := by
  constructor
  · rintro ⟨rowValues, columnValues, equations⟩
    refine ⟨rowValues, fun column =>
      if before : column.val ≤ mark.val then
        columnValues ⟨column.val, by have := column.isLt; omega⟩
      else columnValues ⟨column.val + 1, by have := column.isLt; omega⟩ - slope, ?_⟩
    intro row column
    by_cases before : column.val ≤ mark.val
    · simp only [before, dite_true]
      have relation := equations row ⟨column.val, by have := column.isLt; omega⟩
      simpa [expandAfter, before] using relation
    · simp only [before, dite_false]
      have after : ¬ column.val + 1 ≤ mark.val := by omega
      have relation := equations row ⟨column.val + 1, by have := column.isLt; omega⟩
      have shifted : height (row, column) + slope =
          rowValues row + columnValues ⟨column.val + 1, by have := column.isLt; omega⟩ := by
        simpa [expandAfter, after] using relation
      linarith
  · rintro ⟨rowValues, columnValues, equations⟩
    refine ⟨rowValues, fun column =>
      if before : column.val ≤ mark.val then
        columnValues ⟨column.val, by have := mark.isLt; omega⟩
      else columnValues ⟨column.val - 1, by have := column.isLt; omega⟩ + slope, ?_⟩
    intro row column
    by_cases before : column.val ≤ mark.val
    · simp only [before, dite_true]
      rw [expandAfter_left _ _ _ _ before]
      exact equations row _
    · simp only [before, dite_false]
      rw [expandAfter_right _ _ _ _ before, equations]
      ring

theorem expandRows_separable_iff {rows width : Nat} (height : Cell rows width -> Int)
    (mark : Fin rows) (slope : Int) :
    AdditivelySeparable (expandRows height mark slope) ↔ AdditivelySeparable height := by
  unfold expandRows
  rw [separable_swap_iff, expandAfter_separable_iff, separable_swap_iff]

theorem rowFrozen_contractAt_iff {rows width : Nat} (height : Cell rows (width + 1) -> Int)
    (heightValid : IsHeight height) (rowsPositive : 0 < rows)
    (column : Fin (width + 1)) (columnFrozen : frozenColumn height column) (rowMark : Fin rows) :
    frozenColumn (swapH (contractAt height rowsPositive column columnFrozen)) rowMark ↔
      frozenColumn (swapH height) rowMark := by
  have relation := rowFrozen_expandAfter_iff (contractAt height rowsPositive column columnFrozen)
    ⟨column.val, by have := columnFrozen.2.1; omega⟩
    (frozenSlope height rowsPositive column columnFrozen) rowMark
  rw [expand_contract_at height heightValid rowsPositive column columnFrozen] at relation
  exact relation.symm

theorem contractAt_separable_iff {rows width : Nat} (height : Cell rows (width + 1) -> Int)
    (heightValid : IsHeight height) (rowsPositive : 0 < rows)
    (column : Fin (width + 1)) (columnFrozen : frozenColumn height column) :
    AdditivelySeparable (contractAt height rowsPositive column columnFrozen) ↔
      AdditivelySeparable height := by
  have relation := expandAfter_separable_iff (contractAt height rowsPositive column columnFrozen)
    ⟨column.val, by have := columnFrozen.2.1; omega⟩
    (frozenSlope height rowsPositive column columnFrozen)
  rw [expand_contract_at height heightValid rowsPositive column columnFrozen] at relation
  exact relation.symm

def deleteColumn {rows width : Nat} (height : Cell rows (width + 1) -> Int)
    (column : Fin (width + 1)) (slope : Int) : Cell rows width -> Int := fun point =>
  if before : point.2.val < column.val then
    height (point.1, ⟨point.2.val, by have := point.2.isLt; omega⟩)
  else height (point.1, ⟨point.2.val + 1, by have := point.2.isLt; omega⟩) - slope

def deleteRow {rows width : Nat} (height : Cell (rows + 1) width -> Int)
    (row : Fin (rows + 1)) (slope : Int) : Cell rows width -> Int :=
  swapH (deleteColumn (swapH height) row slope)

theorem deleteColumn_eq_contractAt {rows width : Nat} (height : Cell rows (width + 1) -> Int)
    (rowsPositive : 0 < rows) (column : Fin (width + 1))
    (columnFrozen : frozenColumn height column) :
    deleteColumn height column (frozenSlope height rowsPositive column columnFrozen) =
      contractAt height rowsPositive column columnFrozen := by
  funext point
  unfold deleteColumn contractAt
  split_ifs <;> rfl

theorem deleteAxes_commute {rows width : Nat} (height : Cell (rows + 1) (width + 1) -> Int)
    (row : Fin (rows + 1)) (column : Fin (width + 1)) (rowSlope columnSlope : Int) :
    deleteRow (deleteColumn height column columnSlope) row rowSlope =
      deleteColumn (deleteRow height row rowSlope) column columnSlope := by
  funext point
  rcases point with ⟨pointRow, pointColumn⟩
  by_cases rowBefore : pointRow.val < row.val
  · by_cases columnBefore : pointColumn.val < column.val
    · simp [deleteRow, deleteColumn, swapH, rowBefore, columnBefore]
    · simp [deleteRow, deleteColumn, swapH, rowBefore, columnBefore]
  · by_cases columnBefore : pointColumn.val < column.val
    · simp [deleteRow, deleteColumn, swapH, rowBefore, columnBefore]
    · simp [deleteRow, deleteColumn, swapH, rowBefore, columnBefore]
      ring

theorem rowSlope_deleteColumn {rows width : Nat} (height : Cell rows (width + 1) -> Int)
    (widthPositive : 0 < width) (column : Fin (width + 1)) (columnPositive : 0 < column.val)
    (columnSlope : Int) (row : Fin rows)
    (originalFrozen : frozenColumn (swapH height) row)
    (contractedFrozen : frozenColumn (swapH (deleteColumn height column columnSlope)) row) :
    frozenSlope (swapH (deleteColumn height column columnSlope)) widthPositive row contractedFrozen =
      frozenSlope (swapH height) (by omega) row originalFrozen := by
  simp [frozenSlope, swapH, deleteColumn, columnPositive]

theorem columnSlope_deleteRow {rows width : Nat} (height : Cell (rows + 1) width -> Int)
    (rowsPositive : 0 < rows) (row : Fin (rows + 1)) (rowPositive : 0 < row.val)
    (rowSlope : Int) (column : Fin width)
    (originalFrozen : frozenColumn height column)
    (contractedFrozen : frozenColumn (deleteRow height row rowSlope) column) :
    frozenSlope (deleteRow height row rowSlope) rowsPositive column contractedFrozen =
      frozenSlope height (by omega) column originalFrozen := by
  simp [frozenSlope, deleteRow, swapH, deleteColumn, rowPositive]

noncomputable def contractRows {rows width : Nat} (height : Cell (rows + 1) width -> Int)
    (widthPositive : 0 < width) (row : Fin (rows + 1))
    (rowFrozen : frozenColumn (swapH height) row) : Cell rows width -> Int :=
  swapH (contractAt (swapH height) widthPositive row rowFrozen)

theorem contractRows_eq_deleteRow {rows width : Nat} (height : Cell (rows + 1) width -> Int)
    (widthPositive : 0 < width) (row : Fin (rows + 1))
    (rowFrozen : frozenColumn (swapH height) row) :
    contractRows height widthPositive row rowFrozen =
      deleteRow height row (frozenSlope (swapH height) widthPositive row rowFrozen) := by
  unfold contractRows deleteRow
  rw [deleteColumn_eq_contractAt]

theorem columnFrozen_contractRows_iff {rows width : Nat} (height : Cell (rows + 1) width -> Int)
    (heightValid : IsHeight height) (widthPositive : 0 < width)
    (row : Fin (rows + 1)) (rowFrozen : frozenColumn (swapH height) row) (column : Fin width) :
    frozenColumn (contractRows height widthPositive row rowFrozen) column ↔
      frozenColumn height column := by
  change frozenColumn (swapH (contractAt (swapH height) widthPositive row rowFrozen)) column ↔
    frozenColumn (swapH (swapH height)) column
  exact rowFrozen_contractAt_iff (swapH height) (swapH_isHeight heightValid) widthPositive
    row rowFrozen column

theorem contractAxes_commute {rows width : Nat} (height : Cell (rows + 1) (width + 1) -> Int)
    (heightValid : IsHeight height) (rowsPositive : 0 < rows) (widthPositive : 0 < width)
    (row : Fin (rows + 1)) (column : Fin (width + 1))
    (rowFrozen : frozenColumn (swapH height) row) (columnFrozen : frozenColumn height column) :
    ∃ rowAfter : frozenColumn
        (swapH (contractAt height (by omega) column columnFrozen)) row,
      ∃ columnAfter : frozenColumn (contractRows height (by omega) row rowFrozen) column,
        contractRows (contractAt height (by omega) column columnFrozen) widthPositive row rowAfter =
          contractAt (contractRows height (by omega) row rowFrozen) rowsPositive column columnAfter := by
  have rowAfter := (rowFrozen_contractAt_iff height heightValid (by omega)
    column columnFrozen row).mpr rowFrozen
  have columnAfter := (columnFrozen_contractRows_iff height heightValid (by omega)
    row rowFrozen column).mpr columnFrozen
  refine ⟨rowAfter, columnAfter, ?_⟩
  have rowSlope :
      frozenSlope (swapH (contractAt height (by omega) column columnFrozen)) widthPositive row rowAfter =
        frozenSlope (swapH height) (by omega) row rowFrozen := by
    simp [frozenSlope, swapH, contractAt, columnFrozen.1]
  have columnSlope :
      frozenSlope (contractRows height (by omega) row rowFrozen) rowsPositive column columnAfter =
        frozenSlope height (by omega) column columnFrozen := by
    simp [contractRows, frozenSlope, swapH, contractAt, rowFrozen.1]
  rw [contractRows_eq_deleteRow (contractAt height (by omega) column columnFrozen)
      widthPositive row rowAfter, rowSlope,
    ← deleteColumn_eq_contractAt (contractRows height (by omega) row rowFrozen)
      rowsPositive column columnAfter, columnSlope]
  calc
    _ = deleteRow
        (deleteColumn height column (frozenSlope height (by omega) column columnFrozen)) row
        (frozenSlope (swapH height) (by omega) row rowFrozen) := by
      rw [deleteColumn_eq_contractAt]
    _ = deleteColumn
        (deleteRow height row (frozenSlope (swapH height) (by omega) row rowFrozen)) column
        (frozenSlope height (by omega) column columnFrozen) := deleteAxes_commute _ _ _ _ _
    _ = _ := by rw [← contractRows_eq_deleteRow height (by omega) row rowFrozen]

def keptColumn {width : Nat} (deleted : Fin (width + 1)) (column : Fin width) : Fin (width + 1) :=
  if before : column.val < deleted.val then
    ⟨column.val, by have := column.isLt; omega⟩
  else ⟨column.val + 1, by have := column.isLt; omega⟩

theorem cellExpand_keptColumn {rows width : Nat} (deleted : Fin (width + 1))
    (positive : 0 < deleted.val) (interior : deleted.val + 1 < width + 1)
    (row : Fin rows) (column : Fin width) :
    cellExpand deleted (row, column) positive interior = (row, keptColumn deleted column) := by
  unfold cellExpand keptColumn
  split_ifs <;> rfl

theorem activeColumn_contractAt_iff {rows width : Nat} (height : Cell rows (width + 1) -> Int)
    (heightValid : IsHeight height) (rowsPositive : 0 < rows)
    (deleted : Fin (width + 1)) (deletedFrozen : frozenColumn height deleted) (column : Fin width) :
    activeColumn (contractAt height rowsPositive deleted deletedFrozen) column ↔
      activeColumn height (keptColumn deleted column) := by
  unfold activeColumn
  apply exists_congr
  intro row
  have relation := contractAt_isSLE_iff height heightValid rowsPositive deleted deletedFrozen (row, column)
  simpa only [cellExpand_keptColumn] using relation

theorem frozenColumn_contractAt_interior_iff {rows width : Nat}
    (height : Cell rows (width + 1) -> Int) (heightValid : IsHeight height) (rowsPositive : 0 < rows)
    (deleted : Fin (width + 1)) (deletedFrozen : frozenColumn height deleted)
    (column : Fin width) (columnPositive : 0 < column.val) (columnInterior : column.val + 1 < width) :
    frozenColumn (contractAt height rowsPositive deleted deletedFrozen) column ↔
      frozenColumn height (keptColumn deleted column) := by
  have keptInterior : 0 < (keptColumn deleted column).val ∧
      (keptColumn deleted column).val + 1 < width + 1 := by
    unfold keptColumn
    split_ifs
    · change 0 < column.val ∧ column.val + 1 < width + 1
      omega
    · change 0 < column.val + 1 ∧ column.val + 1 + 1 < width + 1
      omega
  rw [frozenColumn_iff_inactive _ (contractAt_isHeight height heightValid rowsPositive deleted deletedFrozen)
      column columnPositive columnInterior rowsPositive,
    frozenColumn_iff_inactive height heightValid (keptColumn deleted column)
      keptInterior.1 keptInterior.2 rowsPositive,
    activeColumn_contractAt_iff height heightValid rowsPositive deleted deletedFrozen column]

theorem frozenColumn_contractAt_iff {rows width : Nat}
    (height : Cell rows (width + 1) -> Int) (heightValid : IsHeight height) (rowsPositive : 0 < rows)
    (deleted : Fin (width + 1)) (deletedFrozen : frozenColumn height deleted) (column : Fin width) :
    frozenColumn (contractAt height rowsPositive deleted deletedFrozen) column ↔
      frozenColumn height (keptColumn deleted column) := by
  by_cases positive : 0 < column.val
  · by_cases interior : column.val + 1 < width
    · exact frozenColumn_contractAt_interior_iff height heightValid rowsPositive
        deleted deletedFrozen column positive interior
    · have last : column.val + 1 = width := by have := column.isLt; omega
      have after : ¬ column.val < deleted.val := by
        have deletedInterior := deletedFrozen.2.1
        omega
      have keptLast : (keptColumn deleted column).val + 1 = width + 1 := by
        simp only [keptColumn, after, dite_false]
        omega
      constructor
      · intro frozen
        have bound := frozen.2.1
        omega
      · intro frozen
        have bound := frozen.2.1
        omega
  · have first : column.val = 0 := by omega
    have before : column.val < deleted.val := by have := deletedFrozen.1; omega
    have keptFirst : (keptColumn deleted column).val = 0 := by
      simp only [keptColumn, before, dite_true]
      exact first
    constructor
    · intro frozen
      have bound := frozen.1
      omega
    · intro frozen
      have bound := frozen.1
      omega

theorem adjacent_frozen_slopes_equal {rows width : Nat} (height : Cell rows (width + 1) -> Int)
    (heightValid : IsHeight height) (rowsPositive : 0 < rows)
    (column : Fin (width + 1)) (columnFrozen : frozenColumn height column)
    (nextFrozen : frozenColumn height ⟨column.val + 1, columnFrozen.2.1⟩) :
    frozenSlope height rowsPositive ⟨column.val + 1, columnFrozen.2.1⟩ nextFrozen =
      frozenSlope height rowsPositive column columnFrozen := by
  have sharedStep := frozenSlope_eq_right height heightValid rowsPositive column columnFrozen
    ⟨0, rowsPositive⟩
  simpa [frozenSlope] using sharedStep

theorem contract_adjacent_equal {rows width : Nat} (height : Cell rows (width + 1) -> Int)
    (heightValid : IsHeight height) (rowsPositive : 0 < rows)
    (column : Fin (width + 1)) (columnFrozen : frozenColumn height column)
    (nextFrozen : frozenColumn height ⟨column.val + 1, columnFrozen.2.1⟩) :
    contractAt height rowsPositive column columnFrozen =
      contractAt height rowsPositive ⟨column.val + 1, columnFrozen.2.1⟩ nextFrozen := by
  have sameSlope := adjacent_frozen_slopes_equal height heightValid rowsPositive column
    columnFrozen nextFrozen
  funext point
  by_cases before : point.2.val < column.val
  · rw [contractAt_left _ _ _ _ point before,
      contractAt_left _ _ _ _ point (by change point.2.val < column.val + 1; omega)]
  · by_cases atColumn : point.2.val = column.val
    · rw [contractAt_right _ _ _ _ point before,
        contractAt_left _ _ _ _ point (by change point.2.val < column.val + 1; omega)]
      have step := frozenSlope_eq_right height heightValid rowsPositive column columnFrozen point.1
      have restored :
          height (point.1, ⟨column.val + 1, columnFrozen.2.1⟩) -
            frozenSlope height rowsPositive column columnFrozen = height (point.1, column) := by
        linarith
      simpa only [atColumn] using restored
    · rw [contractAt_right _ _ _ _ point before,
        contractAt_right _ _ _ _ point (by change ¬ point.2.val < column.val + 1; omega), sameSlope]

  theorem deleteColumns_commute {rows width : Nat} (height : Cell rows (width + 2) -> Int)
      (lower upper : Fin (width + 2)) (ordered : lower.val < upper.val)
      (lowerSlope upperSlope : Int) :
      deleteColumn (deleteColumn height lower lowerSlope)
          ⟨upper.val - 1, by have := upper.isLt; omega⟩ upperSlope =
        deleteColumn (deleteColumn height upper upperSlope)
          ⟨lower.val, by have := upper.isLt; omega⟩ lowerSlope := by
    funext point
    by_cases before : point.2.val < lower.val
    · have beforeUpper : point.2.val < upper.val := by omega
      have beforeShifted : point.2.val < upper.val - 1 := by omega
      simp [deleteColumn, before, beforeUpper, beforeShifted]
    · by_cases between : point.2.val < upper.val - 1
      · have nextBefore : point.2.val + 1 < upper.val := by omega
        simp [deleteColumn, before, between, nextBefore]
      · have nextAfterUpper : ¬ point.2.val + 1 < upper.val := by omega
        have nextAfterLower : ¬ point.2.val + 1 < lower.val := by omega
        simp [deleteColumn, before, between, nextAfterUpper, nextAfterLower]
        ring

  theorem lowerSlope_deleteUpper {rows width : Nat} (height : Cell rows (width + 2) -> Int)
      (rowsPositive : 0 < rows) (lower upper : Fin (width + 2)) (ordered : lower.val < upper.val)
      (upperSlope : Int) (lowerFrozen : frozenColumn height lower)
      (contractedFrozen : frozenColumn (deleteColumn height upper upperSlope)
        ⟨lower.val, by have := upper.isLt; omega⟩) :
      frozenSlope (deleteColumn height upper upperSlope) rowsPositive
          ⟨lower.val, by have := upper.isLt; omega⟩ contractedFrozen =
        frozenSlope height rowsPositive lower lowerFrozen := by
    have previousBefore : lower.val - 1 < upper.val := by omega
    simp [frozenSlope, deleteColumn, ordered, previousBefore]

  theorem upperSlope_deleteLower {rows width : Nat} (height : Cell rows (width + 2) -> Int)
      (rowsPositive : 0 < rows) (lower upper : Fin (width + 2))
      (separated : lower.val + 1 < upper.val) (lowerSlope : Int)
      (upperFrozen : frozenColumn height upper)
      (contractedFrozen : frozenColumn (deleteColumn height lower lowerSlope)
        ⟨upper.val - 1, by have := upper.isLt; omega⟩) :
      frozenSlope (deleteColumn height lower lowerSlope) rowsPositive
          ⟨upper.val - 1, by have := upper.isLt; omega⟩ contractedFrozen =
        frozenSlope height rowsPositive upper upperFrozen := by
    have currentAfter : ¬ upper.val - 1 < lower.val := by omega
    have previousAfter : ¬ upper.val - 1 - 1 < lower.val := by omega
    have currentIndex : upper.val - 1 + 1 = upper.val := by omega
    have previousIndex : upper.val - 1 - 1 + 1 = upper.val - 1 := by omega
    simp only [frozenSlope, deleteColumn, currentAfter, previousAfter, dite_false,
      currentIndex, previousIndex]
    ring

#print axioms expandAfter_left
#print axioms expandAfter_right
#print axioms inserted_frozen
#print axioms inserted_slope
#print axioms contract_inserted
#print axioms expandAfter_isHeight
#print axioms expandAfter_canonical
#print axioms retained_frozen
#print axioms expandAfter_numExtrema
#print axioms expand_contract_at
#print axioms contracted_marker_frozen
#print axioms expandRows_isHeight
#print axioms expandRows_canonical
#print axioms expandRows_numExtrema
#print axioms expandAxes_commute
#print axioms expandAfter_maximum_iff
#print axioms expandAfter_minimum_iff
#print axioms rowFrozen_expandAfter_iff
#print axioms columnFrozen_expandRows_iff
#print axioms separable_swap_iff
#print axioms expandAfter_separable_iff
#print axioms expandRows_separable_iff
#print axioms rowFrozen_contractAt_iff
#print axioms contractAt_separable_iff
#print axioms deleteColumn_eq_contractAt
#print axioms deleteAxes_commute
#print axioms rowSlope_deleteColumn
#print axioms columnSlope_deleteRow
#print axioms contractRows_eq_deleteRow
#print axioms columnFrozen_contractRows_iff
#print axioms contractAxes_commute
#print axioms cellExpand_keptColumn
#print axioms activeColumn_contractAt_iff
#print axioms frozenColumn_contractAt_interior_iff
#print axioms frozenColumn_contractAt_iff
#print axioms adjacent_frozen_slopes_equal
#print axioms contract_adjacent_equal
#print axioms deleteColumns_commute
#print axioms lowerSlope_deleteUpper
#print axioms upperSlope_deleteLower

end OrigamiCone.Sequel.FrozenExpansion
