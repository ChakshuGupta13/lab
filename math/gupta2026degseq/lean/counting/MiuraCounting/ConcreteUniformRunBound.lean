import MiuraCounting.ConcreteOppositeAxisTrees
import MiuraCounting.ConcreteTreeCases

namespace OrigamiCone.Sequel.UniformRunBound

open OrigamiCone FrozenExpansion OccupiedBlocks

abbrev RowLeaf {rows width : Nat} (height : Cell rows width -> Int) (point : Extrema height) : Prop :=
  (graph height).degree (BlockGraph.Vertex.rowBlock (rowOf height point)) = 1

abbrev ColumnLeaf {rows width : Nat} (height : Cell rows width -> Int) (point : Extrema height) : Prop :=
  (graph height).degree (BlockGraph.Vertex.columnBlock (columnOf height point)) = 1

abbrev OnlyRows {rows width : Nat} (height : Cell rows width -> Int)
    (component : (graph height).ConnectedComponent) : Prop :=
  ∀ vertex : component, (graph height).degree vertex.val = 1 ->
    ∃ row, vertex.val = BlockGraph.Vertex.rowBlock row

abbrev OnlyColumns {rows width : Nat} (height : Cell rows width -> Int)
    (component : (graph height).ConnectedComponent) : Prop :=
  ∀ vertex : component, (graph height).degree vertex.val = 1 ->
    ∃ column, vertex.val = BlockGraph.Vertex.columnBlock column

inductive LeafPattern {rows width : Nat} (height : Cell rows width -> Int)
    (component : (graph height).ConnectedComponent) : Prop where
  | mixed (rowPoint columnPoint : Extrema height)
      (rowMember : BlockGraph.Vertex.extremum rowPoint ∈ component)
      (columnMember : BlockGraph.Vertex.extremum columnPoint ∈ component)
      (rowLeaf : RowLeaf height rowPoint) (columnLeaf : ColumnLeaf height columnPoint)
  | rowOnly (first second : Extrema height) (different : first ≠ second)
      (firstMember : BlockGraph.Vertex.extremum first ∈ component)
      (secondMember : BlockGraph.Vertex.extremum second ∈ component)
      (firstLeaf : RowLeaf height first) (secondLeaf : RowLeaf height second)
      (onlyRows : OnlyRows height component)
  | columnOnly (first second : Extrema height) (different : first ≠ second)
      (firstMember : BlockGraph.Vertex.extremum first ∈ component)
      (secondMember : BlockGraph.Vertex.extremum second ∈ component)
      (firstLeaf : ColumnLeaf height first) (secondLeaf : ColumnLeaf height second)
      (onlyColumns : OnlyColumns height component)

theorem row_vertex_witness {rows width : Nat} (height : Cell rows width -> Int)
    (component : (graph height).ConnectedComponent) (vertex : component)
    (leaf : (graph height).degree vertex.val = 1)
    (isRow : ∃ row, vertex.val = BlockGraph.Vertex.rowBlock row) :
    ∃ point : Extrema height, BlockGraph.Vertex.extremum point ∈ component ∧
      RowLeaf height point ∧ vertex.val = BlockGraph.Vertex.rowBlock (rowOf height point) := by
  obtain ⟨point, member, kind⟩ := TreeStars.component_leaf_witness height component vertex leaf
  obtain ⟨row, atRow⟩ := isRow
  have atPoint : vertex.val = BlockGraph.Vertex.rowBlock (rowOf height point) := by
    rcases kind with rowKind | columnKind
    · exact rowKind
    · have collision := atRow.symm.trans columnKind
      cases collision
  refine ⟨point, member, ?_, atPoint⟩
  change (graph height).degree (BlockGraph.Vertex.rowBlock (rowOf height point)) = 1
  rw [← atPoint]
  exact leaf

theorem column_vertex_witness {rows width : Nat} (height : Cell rows width -> Int)
    (component : (graph height).ConnectedComponent) (vertex : component)
    (leaf : (graph height).degree vertex.val = 1)
    (isColumn : ∃ column, vertex.val = BlockGraph.Vertex.columnBlock column) :
    ∃ point : Extrema height, BlockGraph.Vertex.extremum point ∈ component ∧
      ColumnLeaf height point ∧ vertex.val = BlockGraph.Vertex.columnBlock (columnOf height point) := by
  obtain ⟨point, member, kind⟩ := TreeStars.component_leaf_witness height component vertex leaf
  obtain ⟨column, atColumn⟩ := isColumn
  have atPoint : vertex.val = BlockGraph.Vertex.columnBlock (columnOf height point) := by
    rcases kind with rowKind | columnKind
    · have collision := atColumn.symm.trans rowKind
      cases collision
    · exact columnKind
  refine ⟨point, member, ?_, atPoint⟩
  change (graph height).degree (BlockGraph.Vertex.columnBlock (columnOf height point)) = 1
  rw [← atPoint]
  exact leaf

theorem tree_leaf_pattern {rows width : Nat} (height : Cell rows width -> Int)
    (heightValid : IsHeight height) (rowsValid : 2 ≤ rows) (widthValid : 2 ≤ width)
    (component : (graph height).ConnectedComponent) (tree : component.toSimpleGraph.IsTree) :
    LeafPattern height component := by
  classical
  by_cases onlyRows : OnlyRows height component
  · obtain ⟨firstVertex, secondVertex, differentVertices, firstMember, secondMember, firstLeaf, secondLeaf⟩ :=
      component_tree_two_leaves height heightValid rowsValid widthValid component tree
    let first : component := ⟨firstVertex, firstMember⟩
    let second : component := ⟨secondVertex, secondMember⟩
    obtain ⟨firstPoint, firstPointMember, firstPointLeaf, firstIdentity⟩ :=
      row_vertex_witness height component first firstLeaf (onlyRows first firstLeaf)
    obtain ⟨secondPoint, secondPointMember, secondPointLeaf, secondIdentity⟩ :=
      row_vertex_witness height component second secondLeaf (onlyRows second secondLeaf)
    have differentPoints : firstPoint ≠ secondPoint := by
      intro equality
      apply differentVertices
      exact firstIdentity.trans ((congrArg
        (fun point : Extrema height => BlockGraph.Vertex.rowBlock (rowOf height point)) equality).trans
          secondIdentity.symm)
    exact LeafPattern.rowOnly firstPoint secondPoint differentPoints firstPointMember secondPointMember
      firstPointLeaf secondPointLeaf onlyRows
  · by_cases onlyColumns : OnlyColumns height component
    · obtain ⟨firstVertex, secondVertex, differentVertices, firstMember, secondMember, firstLeaf, secondLeaf⟩ :=
        component_tree_two_leaves height heightValid rowsValid widthValid component tree
      let first : component := ⟨firstVertex, firstMember⟩
      let second : component := ⟨secondVertex, secondMember⟩
      obtain ⟨firstPoint, firstPointMember, firstPointLeaf, firstIdentity⟩ :=
        column_vertex_witness height component first firstLeaf (onlyColumns first firstLeaf)
      obtain ⟨secondPoint, secondPointMember, secondPointLeaf, secondIdentity⟩ :=
        column_vertex_witness height component second secondLeaf (onlyColumns second secondLeaf)
      have differentPoints : firstPoint ≠ secondPoint := by
        intro equality
        apply differentVertices
        exact firstIdentity.trans ((congrArg
          (fun point : Extrema height => BlockGraph.Vertex.columnBlock (columnOf height point)) equality).trans
            secondIdentity.symm)
      exact LeafPattern.columnOnly firstPoint secondPoint differentPoints firstPointMember secondPointMember
        firstPointLeaf secondPointLeaf onlyColumns
    · unfold OnlyRows at onlyRows
      push_neg at onlyRows
      obtain ⟨columnVertex, columnLeaf, notRow⟩ := onlyRows
      have isColumn : ∃ column, columnVertex.val = BlockGraph.Vertex.columnBlock column := by
        rcases degree_one_is_block height columnVertex.val columnLeaf with isRow | isColumn
        · obtain ⟨row, equality⟩ := isRow
          exact (notRow row equality).elim
        · exact isColumn
      unfold OnlyColumns at onlyColumns
      push_neg at onlyColumns
      obtain ⟨rowVertex, rowLeaf, notColumn⟩ := onlyColumns
      have isRow : ∃ row, rowVertex.val = BlockGraph.Vertex.rowBlock row := by
        rcases degree_one_is_block height rowVertex.val rowLeaf with isRow | isColumn
        · exact isRow
        · obtain ⟨column, equality⟩ := isColumn
          exact (notColumn column equality).elim
      obtain ⟨rowPoint, rowMember, rowPointLeaf, _⟩ :=
        row_vertex_witness height component rowVertex rowLeaf isRow
      obtain ⟨columnPoint, columnMember, columnPointLeaf, _⟩ :=
        column_vertex_witness height component columnVertex columnLeaf isColumn
      exact LeafPattern.mixed rowPoint columnPoint rowMember columnMember rowPointLeaf columnPointLeaf

theorem same_component_reachable {rows width : Nat} (height : Cell rows width -> Int)
    (component : (graph height).ConnectedComponent) (first second : Extrema height)
    (firstMember : BlockGraph.Vertex.extremum first ∈ component)
    (secondMember : BlockGraph.Vertex.extremum second ∈ component) :
    (graph height).Reachable (BlockGraph.Vertex.extremum first) (BlockGraph.Vertex.extremum second) :=
  SimpleGraph.ConnectedComponent.exact (firstMember.trans secondMember.symm)

theorem different_components_not_reachable {rows width : Nat} (height : Cell rows width -> Int)
    (firstComponent secondComponent : (graph height).ConnectedComponent)
    (different : firstComponent ≠ secondComponent) (first second : Extrema height)
    (firstMember : BlockGraph.Vertex.extremum first ∈ firstComponent)
    (secondMember : BlockGraph.Vertex.extremum second ∈ secondComponent) :
    ¬ (graph height).Reachable (BlockGraph.Vertex.extremum first) (BlockGraph.Vertex.extremum second) := by
  intro reachable
  exact different (firstMember.symm.trans ((SimpleGraph.ConnectedComponent.sound reachable).trans secondMember))

theorem run_violation_separable_or_four {rows width : Nat} (height : Cell rows width -> Int)
    (heightValid : IsHeight height) (rowsValid : 2 ≤ rows) (widthValid : 2 ≤ width)
    (violation : numExtrema height ≤ numFrozenRuns (swapH height) + numFrozenRuns height) :
    AdditivelySeparable height ∨ numExtrema height ≤ 4 := by
  classical
  obtain ⟨firstComponent, secondComponent, different, firstTree, secondTree⟩ :=
    run_violation_two_trees height violation
  have firstPattern := tree_leaf_pattern height heightValid rowsValid widthValid firstComponent firstTree
  have secondPattern := tree_leaf_pattern height heightValid rowsValid widthValid secondComponent secondTree
  have separated := different_components_not_reachable height firstComponent secondComponent different
  have reverseSeparated := different_components_not_reachable height secondComponent firstComponent
    (fun equality => different equality.symm)
  cases firstPattern with
  | mixed firstRow firstColumn firstRowMember firstColumnMember firstRowLeaf firstColumnLeaf =>
    cases secondPattern with
    | mixed secondRow secondColumn secondRowMember secondColumnMember secondRowLeaf secondColumnLeaf =>
      left
      exact two_mixed_components_separable height heightValid rowsValid widthValid
        firstRow firstColumn secondRow secondColumn firstRowLeaf firstColumnLeaf secondRowLeaf secondColumnLeaf
        (same_component_reachable height firstComponent firstRow firstColumn firstRowMember firstColumnMember)
        (same_component_reachable height secondComponent secondRow secondColumn secondRowMember secondColumnMember)
        (separated firstRow secondRow firstRowMember secondRowMember)
    | rowOnly secondFirst secondLast secondDifferent secondFirstMember secondLastMember secondFirstLeaf secondLastLeaf secondOnlyRows =>
      left
      exact TreeCases.mixed_with_two_row_leaves_separable height heightValid rowsValid widthValid
        firstRow firstColumn secondFirst secondLast firstRowLeaf firstColumnLeaf secondFirstLeaf secondLastLeaf
        secondDifferent
        (same_component_reachable height firstComponent firstRow firstColumn firstRowMember firstColumnMember)
        (same_component_reachable height secondComponent secondFirst secondLast secondFirstMember secondLastMember)
        (separated firstRow secondFirst firstRowMember secondFirstMember) violation
    | columnOnly secondFirst secondLast secondDifferent secondFirstMember secondLastMember secondFirstLeaf secondLastLeaf secondOnlyColumns =>
      left
      exact TreeCases.mixed_with_two_column_leaves_separable height heightValid rowsValid widthValid
        firstRow firstColumn secondFirst secondLast firstRowLeaf firstColumnLeaf secondFirstLeaf secondLastLeaf
        secondDifferent
        (same_component_reachable height firstComponent firstRow firstColumn firstRowMember firstColumnMember)
        (same_component_reachable height secondComponent secondFirst secondLast secondFirstMember secondLastMember)
        (separated firstRow secondFirst firstRowMember secondFirstMember) violation
  | rowOnly firstFirst firstLast firstDifferent firstFirstMember firstLastMember firstFirstLeaf firstLastLeaf firstOnlyRows =>
    cases secondPattern with
    | mixed secondRow secondColumn secondRowMember secondColumnMember secondRowLeaf secondColumnLeaf =>
      left
      exact TreeCases.mixed_with_two_row_leaves_separable height heightValid rowsValid widthValid
        secondRow secondColumn firstFirst firstLast secondRowLeaf secondColumnLeaf firstFirstLeaf firstLastLeaf
        firstDifferent
        (same_component_reachable height secondComponent secondRow secondColumn secondRowMember secondColumnMember)
        (same_component_reachable height firstComponent firstFirst firstLast firstFirstMember firstLastMember)
        (reverseSeparated secondRow firstFirst secondRowMember firstFirstMember) violation
    | rowOnly secondFirst secondLast secondDifferent secondFirstMember secondLastMember secondFirstLeaf secondLastLeaf secondOnlyRows =>
      left
      exact SameAxisTrees.two_row_only_components_separable height heightValid rowsValid widthValid
        firstComponent secondComponent different firstTree secondTree firstOnlyRows secondOnlyRows
        firstFirst secondFirst firstFirstMember secondFirstMember firstFirstLeaf secondFirstLeaf
    | columnOnly secondFirst secondLast secondDifferent secondFirstMember secondLastMember secondFirstLeaf secondLastLeaf secondOnlyColumns =>
      right
      exact OppositeAxisTrees.opposite_leaf_pairs_card_le_four height heightValid rowsValid widthValid
        firstFirst firstLast secondFirst secondLast firstDifferent secondDifferent
        firstFirstLeaf firstLastLeaf secondFirstLeaf secondLastLeaf
        (separated firstFirst secondFirst firstFirstMember secondFirstMember)
        (separated firstFirst secondLast firstFirstMember secondLastMember)
        (separated firstLast secondFirst firstLastMember secondFirstMember)
        (separated firstLast secondLast firstLastMember secondLastMember)
  | columnOnly firstFirst firstLast firstDifferent firstFirstMember firstLastMember firstFirstLeaf firstLastLeaf firstOnlyColumns =>
    cases secondPattern with
    | mixed secondRow secondColumn secondRowMember secondColumnMember secondRowLeaf secondColumnLeaf =>
      left
      exact TreeCases.mixed_with_two_column_leaves_separable height heightValid rowsValid widthValid
        secondRow secondColumn firstFirst firstLast secondRowLeaf secondColumnLeaf firstFirstLeaf firstLastLeaf
        firstDifferent
        (same_component_reachable height secondComponent secondRow secondColumn secondRowMember secondColumnMember)
        (same_component_reachable height firstComponent firstFirst firstLast firstFirstMember firstLastMember)
        (reverseSeparated secondRow firstFirst secondRowMember firstFirstMember) violation
    | rowOnly secondFirst secondLast secondDifferent secondFirstMember secondLastMember secondFirstLeaf secondLastLeaf secondOnlyRows =>
      right
      exact OppositeAxisTrees.opposite_leaf_pairs_card_le_four height heightValid rowsValid widthValid
        secondFirst secondLast firstFirst firstLast secondDifferent firstDifferent
        secondFirstLeaf secondLastLeaf firstFirstLeaf firstLastLeaf
        (reverseSeparated secondFirst firstFirst secondFirstMember firstFirstMember)
        (reverseSeparated secondFirst firstLast secondFirstMember firstLastMember)
        (reverseSeparated secondLast firstFirst secondLastMember firstFirstMember)
        (reverseSeparated secondLast firstLast secondLastMember firstLastMember)
    | columnOnly secondFirst secondLast secondDifferent secondFirstMember secondLastMember secondFirstLeaf secondLastLeaf secondOnlyColumns =>
      left
      exact SameAxisTrees.two_column_only_components_separable height heightValid rowsValid widthValid
        firstComponent secondComponent different firstTree secondTree firstOnlyColumns secondOnlyColumns
        firstFirst secondFirst firstFirstMember secondFirstMember firstFirstLeaf secondFirstLeaf

theorem nonseparable_run_bound {rows width : Nat} (height : Cell rows width -> Int)
    (heightValid : IsHeight height) (rowsValid : 2 ≤ rows) (widthValid : 2 ≤ width)
    (enoughExtrema : 5 ≤ numExtrema height) (nonseparable : ¬ AdditivelySeparable height) :
    numFrozenRuns (swapH height) + numFrozenRuns height + 1 ≤ numExtrema height := by
  by_contra tooManyRuns
  have violation : numExtrema height ≤ numFrozenRuns (swapH height) + numFrozenRuns height := by omega
  rcases run_violation_separable_or_four height heightValid rowsValid widthValid violation with separable | few
  · exact nonseparable separable
  · omega

#print axioms row_vertex_witness
#print axioms column_vertex_witness
#print axioms tree_leaf_pattern
#print axioms same_component_reachable
#print axioms different_components_not_reachable
#print axioms run_violation_separable_or_four
#print axioms nonseparable_run_bound

end OrigamiCone.Sequel.UniformRunBound
