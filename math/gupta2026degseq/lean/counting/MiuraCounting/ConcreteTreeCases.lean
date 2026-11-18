import MiuraCounting.ConcreteOccupiedBlocks
import MiuraCounting.ConcreteGridRestrictions
import MiuraCounting.ConcreteGridReflection
import MiuraCounting.ConcreteBlockTranspose

namespace OrigamiCone.Sequel.TreeCases

open OrigamiCone FrozenExpansion RunLengthFibers OccupiedBlocks

open scoped BigOperators

theorem degree_le_two_of_four_leaves {Vertex : Type*} [Fintype Vertex] [DecidableEq Vertex]
    (graph : SimpleGraph Vertex) [DecidableRel graph.Adj]
    (positive : ∀ vertex : Vertex, 0 < graph.degree vertex)
    (surplus : graph.edgeFinset.card + 2 ≤ Fintype.card Vertex)
    (fewLeaves : (Finset.univ.filter (fun vertex => graph.degree vertex = 1)).card ≤ 4) :
    ∀ vertex : Vertex, graph.degree vertex ≤ 2 := by
  let leaves := Finset.univ.filter (fun vertex => graph.degree vertex = 1)
  have leavesBound : leaves.card ≤ 4 := fewLeaves
  have leafCard : leaves.card = ∑ vertex : Vertex, if graph.degree vertex = 1 then 1 else 0 := by
    change (Finset.univ.filter (fun vertex => graph.degree vertex = 1)).card = _
    rw [Finset.card_eq_sum_ones, Finset.sum_filter]
  have pointwise : ∀ vertex : Vertex,
      graph.degree vertex + (if graph.degree vertex = 1 then 1 else 0) = 2 + (graph.degree vertex - 2) := by
    intro vertex
    have bound := positive vertex
    by_cases leaf : graph.degree vertex = 1
    · rw [if_pos leaf]
      omega
    · rw [if_neg leaf]
      omega
  have total : (∑ vertex : Vertex, (graph.degree vertex + (if graph.degree vertex = 1 then 1 else 0))) =
      ∑ vertex : Vertex, (2 + (graph.degree vertex - 2)) :=
    Finset.sum_congr rfl (fun vertex _ => pointwise vertex)
  simp only [Finset.sum_add_distrib, Finset.sum_const, Finset.card_univ, smul_eq_mul] at total
  rw [← leafCard] at total
  have handshake := graph.sum_degrees_eq_twice_card_edges
  intro vertex
  have slackBound : graph.degree vertex - 2 ≤ ∑ point : Vertex, (graph.degree point - 2) :=
    Finset.single_le_sum (fun point _ => Nat.zero_le (graph.degree point - 2)) (Finset.mem_univ vertex)
  omega

theorem interior_double_leaf_four_leaves {rows width : Nat} (height : Cell rows width -> Int)
    (heightValid : IsHeight height) (rowsValid : 2 ≤ rows) (widthValid : 2 ≤ width)
    (point : Extrema height)
    (rowLeaf : (OccupiedBlocks.graph height).degree (BlockGraph.Vertex.rowBlock (rowOf height point)) = 1)
    (columnLeaf : (OccupiedBlocks.graph height).degree (BlockGraph.Vertex.columnBlock (columnOf height point)) = 1)
    (interior : 0 < point.val.1.val ∧ point.val.1.val + 1 < rows) :
    (Finset.univ.filter (fun vertex : GraphVertex height => (OccupiedBlocks.graph height).degree vertex = 1)).card ≤ 4 := by
  let firstRow : GraphVertex height :=
    BlockGraph.Vertex.rowBlock (blockRank (swapH height) ⟨0, by omega⟩)
  let lastRow : GraphVertex height :=
    BlockGraph.Vertex.rowBlock (blockRank (swapH height) ⟨rows - 1, by omega⟩)
  let pointRow : GraphVertex height := BlockGraph.Vertex.rowBlock (rowOf height point)
  let pointColumn : GraphVertex height := BlockGraph.Vertex.columnBlock (columnOf height point)
  let allowed : Finset (GraphVertex height) := {firstRow, lastRow, pointRow, pointColumn}
  have subset : Finset.univ.filter (fun vertex : GraphVertex height => (OccupiedBlocks.graph height).degree vertex = 1) ⊆ allowed := by
    intro vertex member
    have leaf := (Finset.mem_filter.mp member).2
    rcases degree_one_is_block height vertex leaf with ⟨row, rfl⟩ | ⟨column, rfl⟩
    · obtain ⟨edge, belongs, _⟩ := BlockGraph.row_leaf_unique (rowOf height) (columnOf height) row leaf
      have edgeLeaf : (OccupiedBlocks.graph height).degree (BlockGraph.Vertex.rowBlock (rowOf height edge)) = 1 := by
        rw [belongs]
        exact leaf
      by_cases atFirst : edge.val.1.val = 0
      · have rowEquality : rowOf height edge = blockRank (swapH height) ⟨0, by omega⟩ :=
          congrArg (blockRank (swapH height)) (Fin.ext atFirst)
        have vertexEquality : BlockGraph.Vertex.rowBlock row = firstRow :=
          congrArg BlockGraph.Vertex.rowBlock (belongs.symm.trans rowEquality)
        simp only [vertexEquality, allowed, Finset.mem_insert, true_or]
      · by_cases atLast : edge.val.1.val + 1 = rows
        · have rowEquality : rowOf height edge = blockRank (swapH height) ⟨rows - 1, by omega⟩ :=
          congrArg (blockRank (swapH height)) (Fin.ext (show edge.val.1.val = rows - 1 from by omega))
          have vertexEquality : BlockGraph.Vertex.rowBlock row = lastRow :=
            congrArg BlockGraph.Vertex.rowBlock (belongs.symm.trans rowEquality)
          simp only [vertexEquality, allowed, Finset.mem_insert, true_or, or_true]
        · have inside : 0 < edge.val.1.val ∧ edge.val.1.val + 1 < rows := by have := edge.val.1.isLt; omega
          have equality := interior_row_column_leaf_equal height heightValid rowsValid widthValid edge point edgeLeaf columnLeaf inside
          have vertexEquality : BlockGraph.Vertex.rowBlock row = pointRow := by
            change BlockGraph.Vertex.rowBlock row = BlockGraph.Vertex.rowBlock (rowOf height point)
            rw [← equality, belongs]
          simp only [vertexEquality, allowed, Finset.mem_insert, true_or, or_true]
    · obtain ⟨edge, belongs, _⟩ := BlockGraph.column_leaf_unique (rowOf height) (columnOf height) column leaf
      have edgeLeaf : (OccupiedBlocks.graph height).degree (BlockGraph.Vertex.columnBlock (columnOf height edge)) = 1 := by
        rw [belongs]
        exact leaf
      have equality := interior_row_column_leaf_equal height heightValid rowsValid widthValid point edge rowLeaf edgeLeaf interior
      have vertexEquality : BlockGraph.Vertex.columnBlock column = pointColumn := by
        change BlockGraph.Vertex.columnBlock column = BlockGraph.Vertex.columnBlock (columnOf height point)
        rw [equality, belongs]
      simp only [vertexEquality, allowed, Finset.mem_insert, Finset.mem_singleton, or_true]
  have four : allowed.card ≤ 4 := by
    have firstBound := Finset.card_insert_le firstRow ({lastRow, pointRow, pointColumn} : Finset (GraphVertex height))
    have lastBound := Finset.card_insert_le lastRow ({pointRow, pointColumn} : Finset (GraphVertex height))
    have pointBound := Finset.card_insert_le pointRow ({pointColumn} : Finset (GraphVertex height))
    have singleton := Finset.card_singleton pointColumn
    change ({firstRow, lastRow, pointRow, pointColumn} : Finset (GraphVertex height)).card ≤ 4
    omega
  exact (Finset.card_le_card subset).trans four

theorem interior_double_leaf_degree_le_two {rows width : Nat} (height : Cell rows width -> Int)
    (heightValid : IsHeight height) (rowsValid : 2 ≤ rows) (widthValid : 2 ≤ width)
    (point : Extrema height)
    (rowLeaf : (OccupiedBlocks.graph height).degree (BlockGraph.Vertex.rowBlock (rowOf height point)) = 1)
    (columnLeaf : (OccupiedBlocks.graph height).degree (BlockGraph.Vertex.columnBlock (columnOf height point)) = 1)
    (interior : 0 < point.val.1.val ∧ point.val.1.val + 1 < rows)
    (violation : numExtrema height ≤ numFrozenRuns (swapH height) + numFrozenRuns height) :
    ∀ vertex : GraphVertex height, (OccupiedBlocks.graph height).degree vertex ≤ 2 := by
  apply degree_le_two_of_four_leaves (OccupiedBlocks.graph height)
  · intro vertex
    obtain ⟨neighbor, adjacent⟩ := graph_no_isolated_vertices height heightValid rowsValid widthValid vertex
    change 0 < ((OccupiedBlocks.graph height).neighborFinset vertex).card
    exact Finset.card_pos.mpr ⟨neighbor, by simpa only [SimpleGraph.mem_neighborFinset] using adjacent⟩
  · rw [card_graph_edges, card_graph_vertices]
    omega
  · exact interior_double_leaf_four_leaves height heightValid rowsValid widthValid point rowLeaf columnLeaf interior

theorem interior_double_leaf_column_card_le_two {rows width : Nat} (height : Cell rows width -> Int)
    (heightValid : IsHeight height) (rowsValid : 2 ≤ rows) (widthValid : 2 ≤ width)
    (point : Extrema height)
    (rowLeaf : (OccupiedBlocks.graph height).degree (BlockGraph.Vertex.rowBlock (rowOf height point)) = 1)
    (columnLeaf : (OccupiedBlocks.graph height).degree (BlockGraph.Vertex.columnBlock (columnOf height point)) = 1)
    (interior : 0 < point.val.1.val ∧ point.val.1.val + 1 < rows)
    (violation : numExtrema height ≤ numFrozenRuns (swapH height) + numFrozenRuns height)
    (column : Fin (numFrozenRuns height + 1)) :
    (Finset.univ.filter (fun edge : Extrema height => columnOf height edge = column)).card ≤ 2 := by
  have bound := interior_double_leaf_degree_le_two height heightValid rowsValid widthValid point
    rowLeaf columnLeaf interior violation (BlockGraph.Vertex.columnBlock column)
  simpa only [OccupiedBlocks.graph, BlockGraph.degree_columnBlock] using bound

theorem opposite_profiles_right_same_block {rows width : Nat} (height : Cell rows width -> Int)
    (heightValid : IsHeight height) (rowsPositive : 0 < rows) (widthValid : 2 ≤ width)
    (firstRow secondRow : Fin rows) (apex : Fin width) (firstValue secondValue sign : Int)
    (unitSign : sign = 1 ∨ sign = -1)
    (firstProfile : ∀ column : Fin width, height (firstRow, column) = firstValue + sign * column.val)
    (secondProfile : ∀ column : Fin width, height (secondRow, column) = secondValue - sign * |(column.val : Int) - apex.val|)
    (apexActive : activeColumn height apex) :
    blockRank height apex = blockRank height ⟨width - 1, by omega⟩ := by
  apply blockRank_eq_of_active_interval height apex ⟨width - 1, by omega⟩
    (show apex.val ≤ width - 1 from by have := apex.isLt; omega)
  intro column lower _
  by_cases atApex : column = apex
  · rw [atApex]
    exact apexActive
  · apply (column_active_or_frozen height heightValid rowsPositive widthValid column).resolve_right
    intro frozen
    let previous : Fin width := ⟨column.val - 1, by have := column.isLt; omega⟩
    have previousValue : previous.val = column.val - 1 := rfl
    have different : column.val ≠ apex.val := fun equality => atApex (Fin.ext equality)
    have firstCurrent := firstProfile column
    have firstPrevious := firstProfile previous
    have secondCurrent := secondProfile column
    have secondPrevious := secondProfile previous
    rw [abs_of_nonneg (show 0 ≤ (column.val : Int) - apex.val from by omega)] at secondCurrent
    rw [abs_of_nonneg (show 0 ≤ (previous.val : Int) - apex.val from by omega)] at secondPrevious
    have firstStep := frozenSlope_eq height heightValid rowsPositive column frozen firstRow
    have secondStep := frozenSlope_eq height heightValid rowsPositive column frozen secondRow
    change height (firstRow, column) - height (firstRow, previous) = _ at firstStep
    change height (secondRow, column) - height (secondRow, previous) = _ at secondStep
    rcases unitSign with rfl | rfl <;>
      simp only [one_mul, neg_one_mul] at firstCurrent firstPrevious secondCurrent secondPrevious <;> omega

theorem member_of_card_le_two {Element : Type*} [DecidableEq Element] (set : Finset Element)
    (first second point : Element) (firstMember : first ∈ set) (secondMember : second ∈ set)
    (different : first ≠ second) (bound : set.card ≤ 2) (member : point ∈ set) :
    point = first ∨ point = second := by
  have pairSubset : ({first, second} : Finset Element) ⊆ set := by
    intro element belongs
    simp only [Finset.mem_insert, Finset.mem_singleton] at belongs
    rcases belongs with rfl | rfl
    · exact firstMember
    · exact secondMember
  have pairCard : ({first, second} : Finset Element).card = 2 := by simp [different]
  have equality : set = {first, second} :=
    (Finset.eq_of_subset_of_card_le pairSubset (by omega)).symm
  simpa only [equality, Finset.mem_insert, Finset.mem_singleton] using member

theorem left_row_leaf_far_block {rows width : Nat} (height : Cell rows width -> Int)
    (heightValid : IsHeight height) (rowsValid : 2 ≤ rows) (widthValid : 2 ≤ width)
    (point outer : Extrema height) (pointLeft : point.val.2.val = 0)
    (pointLeaf : (OccupiedBlocks.graph height).degree (BlockGraph.Vertex.rowBlock (rowOf height point)) = 1)
    (outerLeaf : (OccupiedBlocks.graph height).degree (BlockGraph.Vertex.rowBlock (rowOf height outer)) = 1)
    (disconnected : ¬ (OccupiedBlocks.graph height).Reachable
      (BlockGraph.Vertex.extremum point) (BlockGraph.Vertex.extremum outer)) :
    columnOf height outer = blockRank height ⟨width - 1, by omega⟩ := by
  have pointProfile := (row_leaf_fixed_sign_profile height heightValid rowsValid point pointLeaf).1
  have outerProfile := (row_leaf_fixed_sign_profile height heightValid rowsValid outer outerLeaf).1
  have opposite := row_leaves_opposite_sign height heightValid rowsValid widthValid point outer
    pointLeaf outerLeaf disconnected
  have signFlip : extremumSign height outer = -extremumSign height point := by omega
  apply opposite_profiles_right_same_block height heightValid (by omega) widthValid point.val.1 outer.val.1
    outer.val.2 (height point.val) (height outer.val) (extremumSign height point) (extremumSign_unit height point)
  · intro column
    have value := pointProfile column
    have distance : |(column.val : Int) - point.val.2.val| = column.val := by rw [pointLeft]; simp
    rw [distance] at value
    exact value
  · intro column
    have value := outerProfile column
    rw [signFlip] at value
    simpa only [neg_mul, sub_eq_add_neg] using value
  · exact extremum_column_active height outer

theorem left_double_leaf_outer_apices {rows width : Nat} (height : Cell rows width -> Int)
    (heightValid : IsHeight height) (rowsValid : 2 ≤ rows) (widthValid : 2 ≤ width)
    (point top bottom : Extrema height) (pointLeft : point.val.2.val = 0)
    (pointRowLeaf : (OccupiedBlocks.graph height).degree (BlockGraph.Vertex.rowBlock (rowOf height point)) = 1)
    (pointColumnLeaf : (OccupiedBlocks.graph height).degree (BlockGraph.Vertex.columnBlock (columnOf height point)) = 1)
    (interior : 0 < point.val.1.val ∧ point.val.1.val + 1 < rows)
    (topRow : top.val.1.val = 0) (bottomRow : bottom.val.1.val + 1 = rows)
    (topLeaf : (OccupiedBlocks.graph height).degree (BlockGraph.Vertex.rowBlock (rowOf height top)) = 1)
    (bottomLeaf : (OccupiedBlocks.graph height).degree (BlockGraph.Vertex.rowBlock (rowOf height bottom)) = 1)
    (topDisconnected : ¬ (OccupiedBlocks.graph height).Reachable
      (BlockGraph.Vertex.extremum point) (BlockGraph.Vertex.extremum top))
    (bottomDisconnected : ¬ (OccupiedBlocks.graph height).Reachable
      (BlockGraph.Vertex.extremum point) (BlockGraph.Vertex.extremum bottom))
    (violation : numExtrema height ≤ numFrozenRuns (swapH height) + numFrozenRuns height) :
    top.val.2.val = width - 1 ∧ bottom.val.2.val = width - 1 := by
  let right : Fin width := ⟨width - 1, by omega⟩
  let candidates := Finset.univ.filter (fun edge : Extrema height => columnOf height edge = blockRank height right)
  have topMember : top ∈ candidates := Finset.mem_filter.mpr ⟨Finset.mem_univ _,
    left_row_leaf_far_block height heightValid rowsValid widthValid point top pointLeft pointRowLeaf topLeaf topDisconnected⟩
  have bottomMember : bottom ∈ candidates := Finset.mem_filter.mpr ⟨Finset.mem_univ _,
    left_row_leaf_far_block height heightValid rowsValid widthValid point bottom pointLeft pointRowLeaf bottomLeaf bottomDisconnected⟩
  have different : top ≠ bottom := by
    intro equality
    have sameRow := congrArg (fun edge : Extrema height => edge.val.1.val) equality
    change top.val.1.val = bottom.val.1.val at sameRow
    omega
  have cap := interior_double_leaf_column_card_le_two height heightValid rowsValid widthValid point
    pointRowLeaf pointColumnLeaf interior violation (blockRank height right)
  have profile := row_leaf_fixed_sign_profile height heightValid rowsValid point pointRowLeaf
  have beforeRight : point.val.2.val < width - 1 := by omega
  obtain ⟨upper, upperBefore, upperExtremum⟩ := GridRestrictions.right_extremum_above_cut height heightValid
    widthValid point.val.1 interior.1 point.val.2 beforeRight (extremumSign height point)
    (extremumSign_unit height point) profile.1 profile.2
  obtain ⟨lower, lowerAfter, lowerExtremum⟩ := GridRestrictions.right_extremum_below_cut height heightValid
    widthValid point.val.1 interior.2 point.val.2 beforeRight (extremumSign height point)
    (extremumSign_unit height point) profile.1 profile.2
  let upperPoint : Extrema height := ⟨(upper, right), upperExtremum⟩
  let lowerPoint : Extrema height := ⟨(lower, right), lowerExtremum⟩
  have upperMember : upperPoint ∈ candidates := Finset.mem_filter.mpr ⟨Finset.mem_univ _, rfl⟩
  have lowerMember : lowerPoint ∈ candidates := Finset.mem_filter.mpr ⟨Finset.mem_univ _, rfl⟩
  constructor
  · rcases member_of_card_le_two candidates top bottom upperPoint topMember bottomMember different cap upperMember
      with equality | equality
    · have sameColumn := congrArg (fun edge : Extrema height => edge.val.2.val) equality
      change width - 1 = top.val.2.val at sameColumn
      exact sameColumn.symm
    · have sameRow := congrArg (fun edge : Extrema height => edge.val.1.val) equality
      change upper.val = bottom.val.1.val at sameRow
      omega
  · rcases member_of_card_le_two candidates top bottom lowerPoint topMember bottomMember different cap lowerMember
      with equality | equality
    · have sameRow := congrArg (fun edge : Extrema height => edge.val.1.val) equality
      change lower.val = top.val.1.val at sameRow
      omega
    · have sameColumn := congrArg (fun edge : Extrema height => edge.val.2.val) equality
      change width - 1 = bottom.val.2.val at sameColumn
      exact sameColumn.symm

theorem separated_column_leaf_saturated {rows width : Nat} (height : Cell rows width -> Int)
    (heightValid : IsHeight height) (rowsValid : 2 ≤ rows) (widthValid : 2 ≤ width)
    (first second : Extrema height)
    (columnLeaf : (OccupiedBlocks.graph height).degree (BlockGraph.Vertex.columnBlock (columnOf height first)) = 1)
    (rowLeaf : (OccupiedBlocks.graph height).degree (BlockGraph.Vertex.rowBlock (rowOf height second)) = 1)
    (disconnected : ¬ (OccupiedBlocks.graph height).Reachable
      (BlockGraph.Vertex.extremum first) (BlockGraph.Vertex.extremum second)) :
    height second.val - height first.val = extremumSign height first * gdist first.val second.val := by
  have reverse := separated_mixed_leaf_saturated height heightValid rowsValid widthValid second first rowLeaf columnLeaf
    (fun path => disconnected path.symm)
  have opposite := separated_mixed_leaves_opposite_sign height heightValid rowsValid widthValid second first rowLeaf columnLeaf
    (fun path => disconnected path.symm)
  rw [opposite, gdist_comm second.val first.val] at reverse
  nlinarith

theorem left_point_two_far_corners_separable {rows width : Nat} (height : Cell rows width -> Int)
    (heightValid : IsHeight height) (point top bottom : Cell rows width)
    (pointLeft : point.2.val = 0) (topRow : top.1.val = 0) (bottomRow : bottom.1.val + 1 = rows)
    (topRight : top.2.val + 1 = width) (bottomRight : bottom.2.val + 1 = width)
    (sign : Int) (unitSign : sign = 1 ∨ sign = -1)
    (topSaturated : height top - height point = sign * gdist point top)
    (bottomSaturated : height bottom - height point = sign * gdist point bottom) :
    AdditivelySeparable height := by
  apply Saturation.distance_cone_separable height point sign
  intro cell
  by_cases above : cell.1.val ≤ point.1.val
  · apply Saturation.between_saturated_signed height heightValid point top cell sign unitSign topSaturated
    apply Saturation.rectangle_geodesic
    · omega
    · have within := cell.2.isLt
      omega
  · apply Saturation.between_saturated_signed height heightValid point bottom cell sign unitSign bottomSaturated
    apply Saturation.rectangle_geodesic
    · have within := cell.1.isLt
      omega
    · have within := cell.2.isLt
      omega

theorem left_double_leaf_separable {rows width : Nat} (height : Cell rows width -> Int)
    (heightValid : IsHeight height) (rowsValid : 2 ≤ rows) (widthValid : 2 ≤ width)
    (point top bottom : Extrema height) (pointLeft : point.val.2.val = 0)
    (pointRowLeaf : (OccupiedBlocks.graph height).degree (BlockGraph.Vertex.rowBlock (rowOf height point)) = 1)
    (pointColumnLeaf : (OccupiedBlocks.graph height).degree (BlockGraph.Vertex.columnBlock (columnOf height point)) = 1)
    (interior : 0 < point.val.1.val ∧ point.val.1.val + 1 < rows)
    (topRow : top.val.1.val = 0) (bottomRow : bottom.val.1.val + 1 = rows)
    (topLeaf : (OccupiedBlocks.graph height).degree (BlockGraph.Vertex.rowBlock (rowOf height top)) = 1)
    (bottomLeaf : (OccupiedBlocks.graph height).degree (BlockGraph.Vertex.rowBlock (rowOf height bottom)) = 1)
    (topDisconnected : ¬ (OccupiedBlocks.graph height).Reachable
      (BlockGraph.Vertex.extremum point) (BlockGraph.Vertex.extremum top))
    (bottomDisconnected : ¬ (OccupiedBlocks.graph height).Reachable
      (BlockGraph.Vertex.extremum point) (BlockGraph.Vertex.extremum bottom))
    (violation : numExtrema height ≤ numFrozenRuns (swapH height) + numFrozenRuns height) :
    AdditivelySeparable height := by
  have apices := left_double_leaf_outer_apices height heightValid rowsValid widthValid point top bottom pointLeft
    pointRowLeaf pointColumnLeaf interior topRow bottomRow topLeaf bottomLeaf topDisconnected bottomDisconnected violation
  apply left_point_two_far_corners_separable height heightValid point.val top.val bottom.val pointLeft topRow bottomRow
    (by omega) (by omega) (extremumSign height point) (extremumSign_unit height point)
  · exact separated_column_leaf_saturated height heightValid rowsValid widthValid point top pointColumnLeaf topLeaf topDisconnected
  · exact separated_column_leaf_saturated height heightValid rowsValid widthValid point bottom pointColumnLeaf bottomLeaf bottomDisconnected

theorem opposite_profiles_left_same_block {rows width : Nat} (height : Cell rows width -> Int)
    (heightValid : IsHeight height) (rowsPositive : 0 < rows) (widthValid : 2 ≤ width)
    (firstRow secondRow : Fin rows) (apex : Fin width) (firstValue secondValue sign : Int)
    (unitSign : sign = 1 ∨ sign = -1)
    (firstProfile : ∀ column : Fin width, height (firstRow, column) = firstValue - sign * column.val)
    (secondProfile : ∀ column : Fin width, height (secondRow, column) = secondValue - sign * |(column.val : Int) - apex.val|)
    (apexActive : activeColumn height apex) :
    blockRank height ⟨0, by omega⟩ = blockRank height apex := by
  apply blockRank_eq_of_active_interval height ⟨0, by omega⟩ apex (Nat.zero_le _)
  intro column _ upper
  by_cases atApex : column = apex
  · rw [atApex]
    exact apexActive
  · apply (column_active_or_frozen height heightValid rowsPositive widthValid column).resolve_right
    intro frozen
    let next : Fin width := ⟨column.val + 1, frozen.2.1⟩
    have nextValue : next.val = column.val + 1 := rfl
    have different : column.val ≠ apex.val := fun equality => atApex (Fin.ext equality)
    have firstCurrent := firstProfile column
    have firstNext := firstProfile next
    have secondCurrent := secondProfile column
    have secondNext := secondProfile next
    rw [abs_of_nonpos (show (column.val : Int) - apex.val ≤ 0 from by omega)] at secondCurrent
    rw [abs_of_nonpos (show (next.val : Int) - apex.val ≤ 0 from by omega)] at secondNext
    have firstStep := frozenSlope_eq_right height heightValid rowsPositive column frozen firstRow
    have secondStep := frozenSlope_eq_right height heightValid rowsPositive column frozen secondRow
    change height (firstRow, next) - height (firstRow, column) = _ at firstStep
    change height (secondRow, next) - height (secondRow, column) = _ at secondStep
    rcases unitSign with rfl | rfl <;>
      simp only [one_mul, neg_one_mul] at firstCurrent firstNext secondCurrent secondNext <;> omega

theorem right_row_leaf_far_block {rows width : Nat} (height : Cell rows width -> Int)
    (heightValid : IsHeight height) (rowsValid : 2 ≤ rows) (widthValid : 2 ≤ width)
    (point outer : Extrema height) (pointRight : point.val.2.val + 1 = width)
    (pointLeaf : (OccupiedBlocks.graph height).degree (BlockGraph.Vertex.rowBlock (rowOf height point)) = 1)
    (outerLeaf : (OccupiedBlocks.graph height).degree (BlockGraph.Vertex.rowBlock (rowOf height outer)) = 1)
    (disconnected : ¬ (OccupiedBlocks.graph height).Reachable
      (BlockGraph.Vertex.extremum point) (BlockGraph.Vertex.extremum outer)) :
    columnOf height outer = blockRank height ⟨0, by omega⟩ := by
  have pointProfile := (row_leaf_fixed_sign_profile height heightValid rowsValid point pointLeaf).1
  have outerProfile := (row_leaf_fixed_sign_profile height heightValid rowsValid outer outerLeaf).1
  have opposite := row_leaves_opposite_sign height heightValid rowsValid widthValid point outer
    pointLeaf outerLeaf disconnected
  have signFlip : extremumSign height outer = -extremumSign height point := by omega
  symm
  apply opposite_profiles_left_same_block height heightValid (by omega) widthValid point.val.1 outer.val.1
    outer.val.2 (height point.val + extremumSign height point * point.val.2.val) (height outer.val)
    (extremumSign height point) (extremumSign_unit height point)
  · intro column
    have value := pointProfile column
    rw [abs_of_nonpos (show (column.val : Int) - point.val.2.val ≤ 0 from by have := column.isLt; omega)] at value
    nlinarith
  · intro column
    have value := outerProfile column
    rw [signFlip] at value
    simpa only [neg_mul, sub_eq_add_neg] using value
  · exact extremum_column_active height outer

theorem right_point_two_far_corners_separable {rows width : Nat} (height : Cell rows width -> Int)
    (heightValid : IsHeight height) (point top bottom : Cell rows width)
    (pointRight : point.2.val + 1 = width) (topRow : top.1.val = 0) (bottomRow : bottom.1.val + 1 = rows)
    (topLeft : top.2.val = 0) (bottomLeft : bottom.2.val = 0)
    (sign : Int) (unitSign : sign = 1 ∨ sign = -1)
    (topSaturated : height top - height point = sign * gdist point top)
    (bottomSaturated : height bottom - height point = sign * gdist point bottom) :
    AdditivelySeparable height := by
  apply Saturation.distance_cone_separable height point sign
  intro cell
  by_cases above : cell.1.val ≤ point.1.val
  · apply Saturation.between_saturated_signed height heightValid point top cell sign unitSign topSaturated
    apply Saturation.rectangle_geodesic
    · omega
    · have within := cell.2.isLt
      omega
  · apply Saturation.between_saturated_signed height heightValid point bottom cell sign unitSign bottomSaturated
    apply Saturation.rectangle_geodesic
    · have within := cell.1.isLt
      omega
    · have within := cell.2.isLt
      omega

theorem right_double_leaf_outer_apices {rows width : Nat} (height : Cell rows width -> Int)
    (heightValid : IsHeight height) (rowsValid : 2 ≤ rows) (widthValid : 2 ≤ width)
    (point top bottom : Extrema height) (pointRight : point.val.2.val + 1 = width)
    (pointRowLeaf : (OccupiedBlocks.graph height).degree (BlockGraph.Vertex.rowBlock (rowOf height point)) = 1)
    (pointColumnLeaf : (OccupiedBlocks.graph height).degree (BlockGraph.Vertex.columnBlock (columnOf height point)) = 1)
    (interior : 0 < point.val.1.val ∧ point.val.1.val + 1 < rows)
    (topRow : top.val.1.val = 0) (bottomRow : bottom.val.1.val + 1 = rows)
    (topLeaf : (OccupiedBlocks.graph height).degree (BlockGraph.Vertex.rowBlock (rowOf height top)) = 1)
    (bottomLeaf : (OccupiedBlocks.graph height).degree (BlockGraph.Vertex.rowBlock (rowOf height bottom)) = 1)
    (topDisconnected : ¬ (OccupiedBlocks.graph height).Reachable
      (BlockGraph.Vertex.extremum point) (BlockGraph.Vertex.extremum top))
    (bottomDisconnected : ¬ (OccupiedBlocks.graph height).Reachable
      (BlockGraph.Vertex.extremum point) (BlockGraph.Vertex.extremum bottom))
    (violation : numExtrema height ≤ numFrozenRuns (swapH height) + numFrozenRuns height) :
    top.val.2.val = 0 ∧ bottom.val.2.val = 0 := by
  let left : Fin width := ⟨0, by omega⟩
  let candidates := Finset.univ.filter (fun edge : Extrema height => columnOf height edge = blockRank height left)
  have topMember : top ∈ candidates := Finset.mem_filter.mpr ⟨Finset.mem_univ _,
    right_row_leaf_far_block height heightValid rowsValid widthValid point top pointRight pointRowLeaf topLeaf topDisconnected⟩
  have bottomMember : bottom ∈ candidates := Finset.mem_filter.mpr ⟨Finset.mem_univ _,
    right_row_leaf_far_block height heightValid rowsValid widthValid point bottom pointRight pointRowLeaf bottomLeaf bottomDisconnected⟩
  have different : top ≠ bottom := by
    intro equality
    have sameRow := congrArg (fun edge : Extrema height => edge.val.1.val) equality
    change top.val.1.val = bottom.val.1.val at sameRow
    omega
  have cap := interior_double_leaf_column_card_le_two height heightValid rowsValid widthValid point
    pointRowLeaf pointColumnLeaf interior violation (blockRank height left)
  have profile := row_leaf_fixed_sign_profile height heightValid rowsValid point pointRowLeaf
  have apexPositive : 0 < point.val.2.val := by omega
  obtain ⟨upper, upperBefore, upperExtremum⟩ := GridReflection.left_extremum_above_cut height heightValid
    widthValid point.val.1 interior.1 point.val.2 apexPositive (extremumSign height point)
    (extremumSign_unit height point) profile.1 profile.2
  obtain ⟨lower, lowerAfter, lowerExtremum⟩ := GridReflection.left_extremum_below_cut height heightValid
    widthValid point.val.1 interior.2 point.val.2 apexPositive (extremumSign height point)
    (extremumSign_unit height point) profile.1 profile.2
  let upperPoint : Extrema height := ⟨(upper, left), upperExtremum⟩
  let lowerPoint : Extrema height := ⟨(lower, left), lowerExtremum⟩
  have upperMember : upperPoint ∈ candidates := Finset.mem_filter.mpr ⟨Finset.mem_univ _, rfl⟩
  have lowerMember : lowerPoint ∈ candidates := Finset.mem_filter.mpr ⟨Finset.mem_univ _, rfl⟩
  constructor
  · rcases member_of_card_le_two candidates top bottom upperPoint topMember bottomMember different cap upperMember
      with equality | equality
    · have sameColumn := congrArg (fun edge : Extrema height => edge.val.2.val) equality
      change 0 = top.val.2.val at sameColumn
      exact sameColumn.symm
    · have sameRow := congrArg (fun edge : Extrema height => edge.val.1.val) equality
      change upper.val = bottom.val.1.val at sameRow
      omega
  · rcases member_of_card_le_two candidates top bottom lowerPoint topMember bottomMember different cap lowerMember
      with equality | equality
    · have sameRow := congrArg (fun edge : Extrema height => edge.val.1.val) equality
      change lower.val = top.val.1.val at sameRow
      omega
    · have sameColumn := congrArg (fun edge : Extrema height => edge.val.2.val) equality
      change 0 = bottom.val.2.val at sameColumn
      exact sameColumn.symm

theorem right_double_leaf_separable {rows width : Nat} (height : Cell rows width -> Int)
    (heightValid : IsHeight height) (rowsValid : 2 ≤ rows) (widthValid : 2 ≤ width)
    (point top bottom : Extrema height) (pointRight : point.val.2.val + 1 = width)
    (pointRowLeaf : (OccupiedBlocks.graph height).degree (BlockGraph.Vertex.rowBlock (rowOf height point)) = 1)
    (pointColumnLeaf : (OccupiedBlocks.graph height).degree (BlockGraph.Vertex.columnBlock (columnOf height point)) = 1)
    (interior : 0 < point.val.1.val ∧ point.val.1.val + 1 < rows)
    (topRow : top.val.1.val = 0) (bottomRow : bottom.val.1.val + 1 = rows)
    (topLeaf : (OccupiedBlocks.graph height).degree (BlockGraph.Vertex.rowBlock (rowOf height top)) = 1)
    (bottomLeaf : (OccupiedBlocks.graph height).degree (BlockGraph.Vertex.rowBlock (rowOf height bottom)) = 1)
    (topDisconnected : ¬ (OccupiedBlocks.graph height).Reachable
      (BlockGraph.Vertex.extremum point) (BlockGraph.Vertex.extremum top))
    (bottomDisconnected : ¬ (OccupiedBlocks.graph height).Reachable
      (BlockGraph.Vertex.extremum point) (BlockGraph.Vertex.extremum bottom))
    (violation : numExtrema height ≤ numFrozenRuns (swapH height) + numFrozenRuns height) :
    AdditivelySeparable height := by
  have apices := right_double_leaf_outer_apices height heightValid rowsValid widthValid point top bottom pointRight
    pointRowLeaf pointColumnLeaf interior topRow bottomRow topLeaf bottomLeaf topDisconnected bottomDisconnected violation
  apply right_point_two_far_corners_separable height heightValid point.val top.val bottom.val pointRight topRow bottomRow
    apices.1 apices.2 (extremumSign height point) (extremumSign_unit height point)
  · exact separated_column_leaf_saturated height heightValid rowsValid widthValid point top pointColumnLeaf topLeaf topDisconnected
  · exact separated_column_leaf_saturated height heightValid rowsValid widthValid point bottom pointColumnLeaf bottomLeaf bottomDisconnected

theorem mixed_with_outer_row_leaves_separable {rows width : Nat} (height : Cell rows width -> Int)
    (heightValid : IsHeight height) (rowsValid : 2 ≤ rows) (widthValid : 2 ≤ width)
    (rowPoint columnPoint top bottom : Extrema height)
    (pointRowLeaf : (OccupiedBlocks.graph height).degree (BlockGraph.Vertex.rowBlock (rowOf height rowPoint)) = 1)
    (pointColumnLeaf : (OccupiedBlocks.graph height).degree (BlockGraph.Vertex.columnBlock (columnOf height columnPoint)) = 1)
    (topRow : top.val.1.val = 0) (bottomRow : bottom.val.1.val + 1 = rows)
    (topLeaf : (OccupiedBlocks.graph height).degree (BlockGraph.Vertex.rowBlock (rowOf height top)) = 1)
    (bottomLeaf : (OccupiedBlocks.graph height).degree (BlockGraph.Vertex.rowBlock (rowOf height bottom)) = 1)
    (topDisconnected : ¬ (OccupiedBlocks.graph height).Reachable
      (BlockGraph.Vertex.extremum rowPoint) (BlockGraph.Vertex.extremum top))
    (bottomDisconnected : ¬ (OccupiedBlocks.graph height).Reachable
      (BlockGraph.Vertex.extremum rowPoint) (BlockGraph.Vertex.extremum bottom))
    (violation : numExtrema height ≤ numFrozenRuns (swapH height) + numFrozenRuns height) :
    AdditivelySeparable height := by
  have differentTop : rowPoint.val.1.val ≠ top.val.1.val := by
    intro equality
    apply topDisconnected
    apply extrema_reachable_of_same_rowBlock height rowPoint top
    exact congrArg (blockRank (swapH height)) (Fin.ext equality)
  have differentBottom : rowPoint.val.1.val ≠ bottom.val.1.val := by
    intro equality
    apply bottomDisconnected
    apply extrema_reachable_of_same_rowBlock height rowPoint bottom
    exact congrArg (blockRank (swapH height)) (Fin.ext equality)
  have interior : 0 < rowPoint.val.1.val ∧ rowPoint.val.1.val + 1 < rows := by
    have within := rowPoint.val.1.isLt
    omega
  have samePoint := interior_row_column_leaf_equal height heightValid rowsValid widthValid rowPoint columnPoint
    pointRowLeaf pointColumnLeaf interior
  have columnLeaf : (OccupiedBlocks.graph height).degree (BlockGraph.Vertex.columnBlock (columnOf height rowPoint)) = 1 := by
    rw [samePoint]
    exact pointColumnLeaf
  have boundary := (separated_mixed_leaf_boundaries height heightValid rowsValid widthValid top rowPoint
    topLeaf columnLeaf (fun path => topDisconnected path.symm)).2
  rcases boundary with atLeft | atRight
  · exact left_double_leaf_separable height heightValid rowsValid widthValid rowPoint top bottom atLeft
      pointRowLeaf columnLeaf interior topRow bottomRow topLeaf bottomLeaf topDisconnected bottomDisconnected violation
  · exact right_double_leaf_separable height heightValid rowsValid widthValid rowPoint top bottom atRight
      pointRowLeaf columnLeaf interior topRow bottomRow topLeaf bottomLeaf topDisconnected bottomDisconnected violation

theorem mixed_with_two_row_leaves_separable {rows width : Nat} (height : Cell rows width -> Int)
    (heightValid : IsHeight height) (rowsValid : 2 ≤ rows) (widthValid : 2 ≤ width)
    (rowPoint columnPoint first second : Extrema height)
    (pointRowLeaf : (OccupiedBlocks.graph height).degree (BlockGraph.Vertex.rowBlock (rowOf height rowPoint)) = 1)
    (pointColumnLeaf : (OccupiedBlocks.graph height).degree (BlockGraph.Vertex.columnBlock (columnOf height columnPoint)) = 1)
    (firstLeaf : (OccupiedBlocks.graph height).degree (BlockGraph.Vertex.rowBlock (rowOf height first)) = 1)
    (secondLeaf : (OccupiedBlocks.graph height).degree (BlockGraph.Vertex.rowBlock (rowOf height second)) = 1)
    (different : first ≠ second)
    (mixedConnected : (OccupiedBlocks.graph height).Reachable
      (BlockGraph.Vertex.extremum rowPoint) (BlockGraph.Vertex.extremum columnPoint))
    (outerConnected : (OccupiedBlocks.graph height).Reachable
      (BlockGraph.Vertex.extremum first) (BlockGraph.Vertex.extremum second))
    (disconnected : ¬ (OccupiedBlocks.graph height).Reachable
      (BlockGraph.Vertex.extremum rowPoint) (BlockGraph.Vertex.extremum first))
    (violation : numExtrema height ≤ numFrozenRuns (swapH height) + numFrozenRuns height) :
    AdditivelySeparable height := by
  have secondDisconnected : ¬ (OccupiedBlocks.graph height).Reachable
      (BlockGraph.Vertex.extremum rowPoint) (BlockGraph.Vertex.extremum second) :=
    fun path => disconnected (path.trans outerConnected.symm)
  have firstBoundary := (separated_mixed_leaf_boundaries height heightValid rowsValid widthValid first columnPoint
    firstLeaf pointColumnLeaf (fun path => disconnected (mixedConnected.trans path.symm))).1
  have secondBoundary := (separated_mixed_leaf_boundaries height heightValid rowsValid widthValid second columnPoint
    secondLeaf pointColumnLeaf (fun path => secondDisconnected (mixedConnected.trans path.symm))).1
  have differentRows : first.val.1.val ≠ second.val.1.val := by
    intro equality
    apply different
    exact (row_leaf_unique_extremum height first firstLeaf second
      (congrArg (blockRank (swapH height)) (Fin.ext equality.symm))).symm
  by_cases firstTop : first.val.1.val = 0
  · have secondBottom : second.val.1.val + 1 = rows := by omega
    exact mixed_with_outer_row_leaves_separable height heightValid rowsValid widthValid rowPoint columnPoint first second
      pointRowLeaf pointColumnLeaf firstTop secondBottom firstLeaf secondLeaf disconnected secondDisconnected violation
  · have firstBottom : first.val.1.val + 1 = rows := by omega
    have secondTop : second.val.1.val = 0 := by omega
    exact mixed_with_outer_row_leaves_separable height heightValid rowsValid widthValid rowPoint columnPoint second first
      pointRowLeaf pointColumnLeaf secondTop firstBottom secondLeaf firstLeaf secondDisconnected disconnected violation

theorem mixed_with_two_column_leaves_separable {rows width : Nat} (height : Cell rows width -> Int)
    (heightValid : IsHeight height) (rowsValid : 2 ≤ rows) (widthValid : 2 ≤ width)
    (rowPoint columnPoint first second : Extrema height)
    (pointRowLeaf : (OccupiedBlocks.graph height).degree (BlockGraph.Vertex.rowBlock (rowOf height rowPoint)) = 1)
    (pointColumnLeaf : (OccupiedBlocks.graph height).degree (BlockGraph.Vertex.columnBlock (columnOf height columnPoint)) = 1)
    (firstLeaf : (OccupiedBlocks.graph height).degree (BlockGraph.Vertex.columnBlock (columnOf height first)) = 1)
    (secondLeaf : (OccupiedBlocks.graph height).degree (BlockGraph.Vertex.columnBlock (columnOf height second)) = 1)
    (different : first ≠ second)
    (mixedConnected : (OccupiedBlocks.graph height).Reachable
      (BlockGraph.Vertex.extremum rowPoint) (BlockGraph.Vertex.extremum columnPoint))
    (outerConnected : (OccupiedBlocks.graph height).Reachable
      (BlockGraph.Vertex.extremum first) (BlockGraph.Vertex.extremum second))
    (disconnected : ¬ (OccupiedBlocks.graph height).Reachable
      (BlockGraph.Vertex.extremum rowPoint) (BlockGraph.Vertex.extremum first))
    (violation : numExtrema height ≤ numFrozenRuns (swapH height) + numFrozenRuns height) :
    AdditivelySeparable height := by
  have transposedViolation : numExtrema (swapH height) ≤
      numFrozenRuns (swapH (swapH height)) + numFrozenRuns (swapH height) := by
    rw [swapH_numExtrema]
    change numExtrema height ≤ numFrozenRuns height + numFrozenRuns (swapH height)
    omega
  have transposed := mixed_with_two_row_leaves_separable (swapH height) (swapH_isHeight heightValid) widthValid rowsValid
    (BlockTranspose.transposePoint height columnPoint) (BlockTranspose.transposePoint height rowPoint)
    (BlockTranspose.transposePoint height first) (BlockTranspose.transposePoint height second)
    ((BlockTranspose.transpose_column_degree height columnPoint).trans pointColumnLeaf)
    ((BlockTranspose.transpose_row_degree height rowPoint).trans pointRowLeaf)
    ((BlockTranspose.transpose_column_degree height first).trans firstLeaf)
    ((BlockTranspose.transpose_column_degree height second).trans secondLeaf)
    (fun equality => different ((BlockTranspose.transposeExtremumEquiv height).symm.injective equality))
    ((BlockTranspose.transpose_reachable_iff height columnPoint rowPoint).mpr mixedConnected.symm)
    ((BlockTranspose.transpose_reachable_iff height first second).mpr outerConnected)
    (fun path => disconnected (mixedConnected.trans
      ((BlockTranspose.transpose_reachable_iff height columnPoint first).mp path))) transposedViolation
  exact (separable_swap_iff height).mp transposed

#print axioms degree_le_two_of_four_leaves
#print axioms interior_double_leaf_four_leaves
#print axioms interior_double_leaf_degree_le_two
#print axioms interior_double_leaf_column_card_le_two
#print axioms opposite_profiles_right_same_block
#print axioms member_of_card_le_two
#print axioms left_row_leaf_far_block
#print axioms left_double_leaf_outer_apices
#print axioms separated_column_leaf_saturated
#print axioms left_point_two_far_corners_separable
#print axioms left_double_leaf_separable
#print axioms opposite_profiles_left_same_block
#print axioms right_row_leaf_far_block
#print axioms right_point_two_far_corners_separable
#print axioms right_double_leaf_outer_apices
#print axioms right_double_leaf_separable
#print axioms mixed_with_outer_row_leaves_separable
#print axioms mixed_with_two_row_leaves_separable
#print axioms mixed_with_two_column_leaves_separable

end OrigamiCone.Sequel.TreeCases
