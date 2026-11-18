import OrigamiCone.Dual
import MiuraCounting.ConcreteOccupiedBlocks
import MiuraCounting.ConcreteBoundaryComparison

namespace OrigamiCone.Sequel.GridNegation

open OrigamiCone FrozenExpansion RunLengthFibers OccupiedBlocks

def negH {rows width : Nat} (height : Cell rows width -> Int) : Cell rows width -> Int :=
  fun point => -height point

theorem negH_isHeight {rows width : Nat} (height : Cell rows width -> Int)
    (heightValid : IsHeight height) : IsHeight (negH height) := isHeight_neg heightValid

theorem negH_extremum_iff {rows width : Nat} (height : Cell rows width -> Int)
    (point : Cell rows width) : IsStrictLocalExtremum (negH height) point ↔
      IsStrictLocalExtremum height point := by
  change (IsStrictLocalMax (fun point => -height point) point ∨
    IsStrictLocalMin (fun point => -height point) point) ↔ _
  rw [strictMax_neg_iff, strictMin_neg_iff]
  exact or_comm

theorem negH_active_iff {rows width : Nat} (height : Cell rows width -> Int)
    (column : Fin width) : activeColumn (negH height) column ↔ activeColumn height column := by
  unfold activeColumn
  exact exists_congr (fun row => negH_extremum_iff height (row, column))

theorem negH_runStart_iff {rows width : Nat} (height : Cell rows width -> Int)
    (column : Fin width) : frozenRunStart (negH height) column ↔ frozenRunStart height column := by
  simp only [frozenRunStart, negH_active_iff]

theorem negH_numFrozenRuns {rows width : Nat} (height : Cell rows width -> Int) :
    numFrozenRuns (negH height) = numFrozenRuns height := by
  classical
  unfold numFrozenRuns
  congr 1
  ext column
  simp only [Finset.mem_filter, negH_runStart_iff]

theorem negH_runPrefixCount {rows width : Nat} (height : Cell rows width -> Int)
    (column : Fin width) : runPrefixCount (negH height) column = runPrefixCount height column := by
  classical
  unfold runPrefixCount
  congr 1
  ext start
  simp only [Finset.mem_filter, negH_runStart_iff]

noncomputable def negBlockEquiv {rows width : Nat} (height : Cell rows width -> Int) :
    Fin (numFrozenRuns height + 1) ≃ Fin (numFrozenRuns (negH height) + 1) where
  toFun := fun block => ⟨block.val, lt_of_lt_of_eq block.isLt
    (congrArg (fun count : Nat => count + 1) (negH_numFrozenRuns height).symm)⟩
  invFun := fun block => ⟨block.val, lt_of_lt_of_eq block.isLt
    (congrArg (fun count : Nat => count + 1) (negH_numFrozenRuns height))⟩
  left_inv := fun _ => Fin.ext rfl
  right_inv := fun _ => Fin.ext rfl

def negExtremumEquiv {rows width : Nat} (height : Cell rows width -> Int) :
    Extrema height ≃ Extrema (negH height) :=
  Equiv.subtypeEquivRight (fun point => (negH_extremum_iff height point).symm)

theorem neg_columnOf {rows width : Nat} (height : Cell rows width -> Int) (point : Extrema height) :
    columnOf (negH height) (negExtremumEquiv height point) = negBlockEquiv height (columnOf height point) := by
  apply Fin.ext
  change runPrefixCount (negH height) point.val.2 = runPrefixCount height point.val.2
  exact negH_runPrefixCount height point.val.2

theorem neg_rowOf {rows width : Nat} (height : Cell rows width -> Int) (point : Extrema height) :
    rowOf (negH height) (negExtremumEquiv height point) =
      negBlockEquiv (swapH height) (rowOf height point) := by
  apply Fin.ext
  change runPrefixCount (negH (swapH height)) point.val.1 = runPrefixCount (swapH height) point.val.1
  exact negH_runPrefixCount (swapH height) point.val.1

noncomputable def negVertexEquiv {rows width : Nat} (height : Cell rows width -> Int) :
    GraphVertex height ≃ GraphVertex (negH height) where
  toFun := fun vertex => match vertex with
    | BlockGraph.Vertex.rowBlock row => BlockGraph.Vertex.rowBlock (negBlockEquiv (swapH height) row)
    | BlockGraph.Vertex.columnBlock column => BlockGraph.Vertex.columnBlock (negBlockEquiv height column)
    | BlockGraph.Vertex.extremum point => BlockGraph.Vertex.extremum (negExtremumEquiv height point)
  invFun := fun vertex => match vertex with
    | BlockGraph.Vertex.rowBlock row => BlockGraph.Vertex.rowBlock ((negBlockEquiv (swapH height)).symm row)
    | BlockGraph.Vertex.columnBlock column => BlockGraph.Vertex.columnBlock ((negBlockEquiv height).symm column)
    | BlockGraph.Vertex.extremum point => BlockGraph.Vertex.extremum ((negExtremumEquiv height).symm point)
  left_inv := by
    intro vertex
    cases vertex with
    | rowBlock row => exact congrArg BlockGraph.Vertex.rowBlock ((negBlockEquiv (swapH height)).symm_apply_apply row)
    | columnBlock column => exact congrArg BlockGraph.Vertex.columnBlock ((negBlockEquiv height).symm_apply_apply column)
    | extremum point => exact congrArg BlockGraph.Vertex.extremum ((negExtremumEquiv height).symm_apply_apply point)
  right_inv := by
    intro vertex
    cases vertex with
    | rowBlock row => exact congrArg BlockGraph.Vertex.rowBlock ((negBlockEquiv (swapH height)).apply_symm_apply row)
    | columnBlock column => exact congrArg BlockGraph.Vertex.columnBlock ((negBlockEquiv height).apply_symm_apply column)
    | extremum point => exact congrArg BlockGraph.Vertex.extremum ((negExtremumEquiv height).apply_symm_apply point)

noncomputable def negGraphIso {rows width : Nat} (height : Cell rows width -> Int) :
    graph height ≃g graph (negH height) where
  toEquiv := negVertexEquiv height
  map_rel_iff' := by
    intro first second
    cases first with
    | rowBlock row =>
      cases second with
      | rowBlock _ => rfl
      | columnBlock _ => rfl
      | extremum point =>
        change (rowOf (negH height) (negExtremumEquiv height point) =
          negBlockEquiv (swapH height) row) ↔ rowOf height point = row
        rw [neg_rowOf]
        exact (negBlockEquiv (swapH height)).injective.eq_iff
    | columnBlock column =>
      cases second with
      | rowBlock _ => rfl
      | columnBlock _ => rfl
      | extremum point =>
        change (columnOf (negH height) (negExtremumEquiv height point) =
          negBlockEquiv height column) ↔ columnOf height point = column
        rw [neg_columnOf]
        exact (negBlockEquiv height).injective.eq_iff
    | extremum point =>
      cases second with
      | rowBlock row =>
        change (rowOf (negH height) (negExtremumEquiv height point) =
          negBlockEquiv (swapH height) row) ↔ rowOf height point = row
        rw [neg_rowOf]
        exact (negBlockEquiv (swapH height)).injective.eq_iff
      | columnBlock column =>
        change (columnOf (negH height) (negExtremumEquiv height point) =
          negBlockEquiv height column) ↔ columnOf height point = column
        rw [neg_columnOf]
        exact (negBlockEquiv height).injective.eq_iff
      | extremum _ => rfl

theorem neg_row_degree {rows width : Nat} (height : Cell rows width -> Int) (point : Extrema height) :
    (graph (negH height)).degree (BlockGraph.Vertex.rowBlock (rowOf (negH height) (negExtremumEquiv height point))) =
      (graph height).degree (BlockGraph.Vertex.rowBlock (rowOf height point)) := by
  rw [neg_rowOf]
  exact (negGraphIso height).degree_eq (BlockGraph.Vertex.rowBlock (rowOf height point))

theorem neg_column_degree {rows width : Nat} (height : Cell rows width -> Int) (point : Extrema height) :
    (graph (negH height)).degree (BlockGraph.Vertex.columnBlock (columnOf (negH height) (negExtremumEquiv height point))) =
      (graph height).degree (BlockGraph.Vertex.columnBlock (columnOf height point)) := by
  rw [neg_columnOf]
  exact (negGraphIso height).degree_eq (BlockGraph.Vertex.columnBlock (columnOf height point))

theorem negative_sign_becomes_minimum {rows width : Nat} (height : Cell rows width -> Int)
    (point : Extrema height) (negative : extremumSign height point = -1) :
    extremumSign (negH height) (negExtremumEquiv height point) = 1 := by
  have notMinimum : ¬ IsStrictLocalMin height point.val := by
    intro minimum
    have positive : extremumSign height point = 1 := by
      simp only [extremumSign, if_pos minimum]
    omega
  have maximum := point.property.resolve_right notMinimum
  have minimum : IsStrictLocalMin (negH height) point.val := strictMin_neg_iff.mpr maximum
  change (if IsStrictLocalMin (negH height) point.val then 1 else -1) = 1
  rw [if_pos minimum]

theorem negH_separable_iff {rows width : Nat} (height : Cell rows width -> Int) :
    AdditivelySeparable (negH height) ↔ AdditivelySeparable height := by
  constructor
  · rintro ⟨rowPart, columnPart, profile⟩
    refine ⟨fun row => -rowPart row, fun column => -columnPart column, ?_⟩
    intro row column
    have value := profile row column
    change -height (row, column) = rowPart row + columnPart column at value
    change height (row, column) = -rowPart row + -columnPart column
    omega
  · rintro ⟨rowPart, columnPart, profile⟩
    refine ⟨fun row => -rowPart row, fun column => -columnPart column, ?_⟩
    intro row column
    have value := profile row column
    change -height (row, column) = -rowPart row + -columnPart column
    omega

theorem negative_boundary_profiles_separable {rows width : Nat}
    (height : Cell rows width -> Int) (heightValid : IsHeight height)
    (rowsValid : 2 ≤ rows) (widthValid : 2 ≤ width)
    (firstInward : ∀ row : Fin rows, height (row, ⟨1, by omega⟩) =
      height (row, ⟨0, by omega⟩) - 1)
    (lastInward : ∀ row : Fin rows, height (row, ⟨width - 2, by omega⟩) =
      height (row, ⟨width - 1, by omega⟩) + 1)
    (firstLeaves : ∀ point : Extrema height, point.val.2.val = 0 ->
      (graph height).degree (BlockGraph.Vertex.rowBlock (rowOf height point)) = 1)
    (lastLeaves : ∀ point : Extrema height, point.val.2.val + 1 = width ->
      (graph height).degree (BlockGraph.Vertex.rowBlock (rowOf height point)) = 1) :
    AdditivelySeparable height := by
  apply (negH_separable_iff height).mp
  apply BoundaryComparison.boundary_leaf_profiles_separable (negH height)
    (negH_isHeight height heightValid) rowsValid widthValid
  · intro row
    have value := firstInward row
    change -height (row, ⟨1, by omega⟩) = -height (row, ⟨0, by omega⟩) + 1
    omega
  · intro row
    have value := lastInward row
    change -height (row, ⟨width - 2, by omega⟩) =
      -height (row, ⟨width - 1, by omega⟩) - 1
    omega
  · intro point atFirst
    let original := (negExtremumEquiv height).symm point
    have leaf := firstLeaves original atFirst
    have degree := neg_row_degree height original
    rw [(negExtremumEquiv height).apply_symm_apply point] at degree
    exact degree.trans leaf
  · intro point atLast
    let original := (negExtremumEquiv height).symm point
    have leaf := lastLeaves original atLast
    have degree := neg_row_degree height original
    rw [(negExtremumEquiv height).apply_symm_apply point] at degree
    exact degree.trans leaf

#print axioms negH_isHeight
#print axioms negH_extremum_iff
#print axioms negH_active_iff
#print axioms negH_runStart_iff
#print axioms negH_numFrozenRuns
#print axioms negH_runPrefixCount
#print axioms neg_columnOf
#print axioms neg_rowOf
#print axioms negGraphIso
#print axioms neg_row_degree
#print axioms neg_column_degree
#print axioms negative_sign_becomes_minimum
#print axioms negH_separable_iff
#print axioms negative_boundary_profiles_separable

end OrigamiCone.Sequel.GridNegation
