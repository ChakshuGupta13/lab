import MiuraCounting.ConcreteBlockGraph
import MiuraCounting.ConcreteLeafProfiles
import MiuraCounting.ConcreteTreeInventory
import MiuraCounting.ConcreteSaturation

namespace OrigamiCone.Sequel.OccupiedBlocks

open OrigamiCone FrozenExpansion RunLengthFibers

noncomputable def blockRank {rows width : Nat} (height : Cell rows width -> Int) (column : Fin width) :
    Fin (numFrozenRuns height + 1) :=
  ⟨runPrefixCount height column, Nat.lt_succ_of_le (runPrefixCount_le_numFrozenRuns height column)⟩

abbrev Extrema {rows width : Nat} (height : Cell rows width -> Int) :=
  {point : Cell rows width // IsStrictLocalExtremum height point}

noncomputable def rowOf {rows width : Nat} (height : Cell rows width -> Int) (point : Extrema height) :
    Fin (numFrozenRuns (swapH height) + 1) := blockRank (swapH height) point.val.1

noncomputable def columnOf {rows width : Nat} (height : Cell rows width -> Int) (point : Extrema height) :
    Fin (numFrozenRuns height + 1) := blockRank height point.val.2

abbrev GraphVertex {rows width : Nat} (height : Cell rows width -> Int) :=
  BlockGraph.Vertex (Fin (numFrozenRuns (swapH height) + 1)) (Fin (numFrozenRuns height + 1)) (Extrema height)

noncomputable def graph {rows width : Nat} (height : Cell rows width -> Int) : SimpleGraph (GraphVertex height) :=
  BlockGraph.incidenceGraph (rowOf height) (columnOf height)

noncomputable instance {rows width : Nat} (height : Cell rows width -> Int) : DecidableRel (graph height).Adj := by
  change DecidableRel (BlockGraph.incidenceGraph (rowOf height) (columnOf height)).Adj
  infer_instance

theorem blockRank_next_of_active {rows width : Nat} (height : Cell rows width -> Int)
    (column : Fin width) (nextWithin : column.val + 1 < width)
    (nextActive : activeColumn height ⟨column.val + 1, nextWithin⟩) :
    blockRank height column = blockRank height ⟨column.val + 1, nextWithin⟩ := by
  apply Fin.ext
  change runPrefixCount height column = runPrefixCount height ⟨column.val + 1, nextWithin⟩
  exact (runPrefixCount_next_eq_of_not_start height column nextWithin (fun start => start.1 nextActive)).symm

theorem blockRank_adjacent_active {rows width : Nat} (height : Cell rows width -> Int)
    (first second : Fin width) (firstActive : activeColumn height first) (secondActive : activeColumn height second)
    (adjacent : ((first.val : Int) - second.val).natAbs = 1) : blockRank height first = blockRank height second := by
  by_cases before : first.val < second.val
  · have within : first.val + 1 < width := by have := second.isLt; omega
    have nextSame : (⟨first.val + 1, within⟩ : Fin width) = second := by
      apply Fin.ext
      change first.val + 1 = second.val
      omega
    have relation := blockRank_next_of_active height first within (by rw [nextSame]; exact secondActive)
    simpa only [nextSame] using relation
  · have within : second.val + 1 < width := by have := first.isLt; omega
    have nextSame : (⟨second.val + 1, within⟩ : Fin width) = first := by
      apply Fin.ext
      change second.val + 1 = first.val
      omega
    have relation := blockRank_next_of_active height second within (by rw [nextSame]; exact firstActive)
    simpa only [nextSame] using relation.symm

theorem card_extrema {rows width : Nat} (height : Cell rows width -> Int) :
    Fintype.card (Extrema height) = numExtrema height := by
  classical
  let equivalence : Extrema height ≃ (Finset.univ.filter (IsStrictLocalExtremum height)) :=
    Equiv.subtypeEquivRight (by intro point; simp)
  rw [Fintype.card_congr equivalence, Fintype.card_coe]
  rfl

theorem card_graph_vertices {rows width : Nat} (height : Cell rows width -> Int) :
    Fintype.card (GraphVertex height) = numFrozenRuns (swapH height) + numFrozenRuns height + 2 + numExtrema height := by
  rw [BlockGraph.card_vertices]
  simp only [Fintype.card_fin, card_extrema]
  omega

theorem card_graph_edges {rows width : Nat} (height : Cell rows width -> Int) :
    (graph height).edgeFinset.card = 2 * numExtrema height := by
  classical
  calc
    _ = 2 * Fintype.card (Extrema height) := BlockGraph.card_edges (rowOf height) (columnOf height)
    _ = _ := congrArg (fun value : Nat => 2 * value) (card_extrema height)

theorem extremum_row_active {rows width : Nat} (height : Cell rows width -> Int) (point : Extrema height) :
    activeColumn (swapH height) point.val.1 :=
  ⟨point.val.2, (swapH_strictExtremum_iff (h := height) (point.val.2, point.val.1)).mpr point.property⟩

theorem extremum_column_active {rows width : Nat} (height : Cell rows width -> Int) (point : Extrema height) :
    activeColumn height point.val.2 := ⟨point.val.1, point.property⟩

theorem row_leaf_unique_extremum {rows width : Nat} (height : Cell rows width -> Int)
    (point : Extrema height) (leaf : (graph height).degree (BlockGraph.Vertex.rowBlock (rowOf height point)) = 1) :
    ∀ other : Extrema height, rowOf height other = rowOf height point -> other = point := by
  classical
  obtain ⟨uniqueEdge, _, unique⟩ :=
    BlockGraph.row_leaf_unique (rowOf height) (columnOf height) (rowOf height point) leaf
  have pointEquality := unique point rfl
  intro other belongs
  exact (unique other belongs).trans pointEquality.symm

theorem row_leaf_unique_column {rows width : Nat} (height : Cell rows width -> Int)
    (point : Extrema height) (leaf : (graph height).degree (BlockGraph.Vertex.rowBlock (rowOf height point)) = 1) :
    ∀ column : Fin width, IsStrictLocalExtremum height (point.val.1, column) -> column = point.val.2 := by
  intro column extremum
  let candidate : Extrema height := ⟨(point.val.1, column), extremum⟩
  have equality := row_leaf_unique_extremum height point leaf candidate rfl
  exact congrArg Prod.snd (congrArg Subtype.val equality)

theorem row_leaf_neighbors_inactive {rows width : Nat} (height : Cell rows width -> Int)
    (point : Extrema height) (leaf : (graph height).degree (BlockGraph.Vertex.rowBlock (rowOf height point)) = 1) :
    ∀ other : Fin rows, ((point.val.1.val : Int) - other.val).natAbs = 1 ->
      ¬ activeColumn (swapH height) other := by
  intro other adjacent active
  obtain ⟨column, extremum⟩ := active
  let candidate : Extrema height := ⟨(other, column),
    (swapH_strictExtremum_iff (h := height) (column, other)).mp extremum⟩
  have sameBlock : rowOf height candidate = rowOf height point :=
    blockRank_adjacent_active (swapH height) other point.val.1 ⟨column, extremum⟩
      (extremum_row_active height point) (by omega)
  have equality := row_leaf_unique_extremum height point leaf candidate sameBlock
  have sameRow : other = point.val.1 := congrArg Prod.fst (congrArg Subtype.val equality)
  have values := congrArg Fin.val sameRow
  omega

theorem row_leaf_profile {rows width : Nat} (height : Cell rows width -> Int)
    (heightValid : IsHeight height) (rowsValid : 2 ≤ rows) (point : Extrema height)
    (leaf : (graph height).degree (BlockGraph.Vertex.rowBlock (rowOf height point)) = 1) :
    ∃ sign : Int, (sign = 1 ∨ sign = -1) ∧
      (∀ neighbor : Cell rows width, adj point.val neighbor -> height neighbor = height point.val + sign) ∧
      (∀ column : Fin width, height (point.val.1, column) =
        height point.val + sign * |(column.val : Int) - point.val.2.val|) ∧
      ∀ column : Fin width, ∀ other : Fin rows, ((point.val.1.val : Int) - other.val).natAbs = 1 ->
        height (other, column) = height (point.val.1, column) + sign :=
  LeafProfiles.row_leaf_signed_profile height heightValid rowsValid point.val.1 point.val.2 point.property
    (row_leaf_unique_column height point leaf) (row_leaf_neighbors_inactive height point leaf)

theorem prefix_runStart_eq_previous_add_one {rows width : Nat} (height : Cell rows width -> Int)
    (column : Fin width) (positive : 0 < column.val) (isStart : frozenRunStart height column) :
    runPrefixCount height column =
      runPrefixCount height ⟨column.val - 1, by have := column.isLt; omega⟩ + 1 := by
  classical
  let previous : Fin width := ⟨column.val - 1, by have := column.isLt; omega⟩
  have setsEqual : Finset.univ.filter (fun start => frozenRunStart height start ∧ start.val ≤ column.val) =
      insert column (Finset.univ.filter (fun start => frozenRunStart height start ∧ start.val ≤ previous.val)) := by
    ext start
    simp only [Finset.mem_filter, Finset.mem_univ, true_and, Finset.mem_insert]
    constructor
    · rintro ⟨startValid, before⟩
      by_cases same : start = column
      · exact Or.inl same
      · have different : start.val ≠ column.val := fun equality => same (Fin.ext equality)
        exact Or.inr ⟨startValid, show start.val ≤ column.val - 1 from by omega⟩
    · rintro (same | ⟨startValid, before⟩)
      · rw [same]
        exact ⟨isStart, Nat.le_refl _⟩
      · have preceding : start.val ≤ column.val - 1 := before
        exact ⟨startValid, by omega⟩
  have notMember : column ∉ Finset.univ.filter
      (fun start => frozenRunStart height start ∧ start.val ≤ previous.val) := by
    intro member
    have before : column.val ≤ column.val - 1 := (Finset.mem_filter.mp member).2.2
    omega
  unfold runPrefixCount
  rw [setsEqual, Finset.card_insert_of_notMem notMember]

theorem prefix_last_eq_runs {rows width : Nat} (height : Cell rows width -> Int) (widthPositive : 0 < width) :
    runPrefixCount height ⟨width - 1, by omega⟩ = numFrozenRuns height := by
  classical
  unfold runPrefixCount numFrozenRuns
  apply congrArg Finset.card
  ext start
  simp only [Finset.mem_filter, Finset.mem_univ, true_and]
  constructor
  · exact fun member => member.1
  · intro isStart
    exact ⟨isStart, show start.val ≤ width - 1 from by have := start.isLt; omega⟩

theorem blockRank_realized_active {rows width : Nat} (height : Cell rows width -> Int)
    (heightValid : IsHeight height) (rowsPositive : 0 < rows) (widthValid : 2 ≤ width)
    (rank : Fin (numFrozenRuns height + 1)) :
    ∃ column : Fin width, activeColumn height column ∧ blockRank height column = rank := by
  by_cases earlier : rank.val < numFrozenRuns height
  · obtain ⟨start, isStart, atRank⟩ := runRank_realized height ⟨rank.val, earlier⟩
    have frozen := (column_active_or_frozen height heightValid rowsPositive widthValid start).resolve_left isStart.1
    let previous : Fin width := ⟨start.val - 1, by have := start.isLt; omega⟩
    have previousActive : activeColumn height previous := by
      rcases isStart.2 with atZero | ⟨_, active⟩
      · have positive := frozen.1
        omega
      · exact active
    have recurrence := prefix_runStart_eq_previous_add_one height start frozen.1 isStart
    refine ⟨previous, previousActive, ?_⟩
    apply Fin.ext
    change runPrefixCount height previous = rank.val
    change runPrefixCount height start = runPrefixCount height previous + 1 at recurrence
    change runPrefixCount height start = rank.val + 1 at atRank
    omega
  · let last : Fin width := ⟨width - 1, by omega⟩
    refine ⟨last, lastColumn_active height heightValid rowsPositive widthValid, ?_⟩
    apply Fin.ext
    change runPrefixCount height last = rank.val
    have lastRank := prefix_last_eq_runs height (by omega)
    change runPrefixCount height last = numFrozenRuns height at lastRank
    have bound := rank.isLt
    omega

theorem columnOf_surjective {rows width : Nat} (height : Cell rows width -> Int)
    (heightValid : IsHeight height) (rowsPositive : 0 < rows) (widthValid : 2 ≤ width) :
    Function.Surjective (columnOf height) := by
  intro rank
  obtain ⟨column, active, atRank⟩ := blockRank_realized_active height heightValid rowsPositive widthValid rank
  obtain ⟨row, extremum⟩ := active
  exact ⟨⟨(row, column), extremum⟩, atRank⟩

theorem rowOf_surjective {rows width : Nat} (height : Cell rows width -> Int)
    (heightValid : IsHeight height) (rowsValid : 2 ≤ rows) (widthPositive : 0 < width) :
    Function.Surjective (rowOf height) := by
  intro rank
  obtain ⟨row, active, atRank⟩ :=
    blockRank_realized_active (swapH height) (swapH_isHeight heightValid) widthPositive rowsValid rank
  obtain ⟨column, extremum⟩ := active
  exact ⟨⟨(row, column), (swapH_strictExtremum_iff (h := height) (column, row)).mp extremum⟩, atRank⟩

theorem graph_no_isolated_vertices {rows width : Nat} (height : Cell rows width -> Int)
    (heightValid : IsHeight height) (rowsValid : 2 ≤ rows) (widthValid : 2 ≤ width) (vertex : GraphVertex height) :
    ∃ other, (graph height).Adj vertex other :=
  BlockGraph.no_isolated_vertices (rowOf height) (columnOf height)
    (rowOf_surjective height heightValid rowsValid (by omega))
    (columnOf_surjective height heightValid (by omega) widthValid) vertex

theorem blockRank_eq_of_active_interval {rows width : Nat} (height : Cell rows width -> Int)
    (first last : Fin width) (ordered : first.val ≤ last.val)
    (activeInterval : ∀ column : Fin width, first.val ≤ column.val -> column.val ≤ last.val ->
      activeColumn height column) : blockRank height first = blockRank height last := by
  have along : ∀ index : Nat, first.val ≤ index -> index ≤ last.val -> ∀ within : index < width,
      blockRank height first = blockRank height ⟨index, within⟩ := by
    intro index lower
    induction index, lower using Nat.le_induction with
    | base => intro _ within; rfl
    | succ index lower inductionHypothesis =>
      intro upper within
      have previousWithin : index < width := by omega
      have active := activeInterval ⟨index + 1, within⟩ (Nat.le_succ_of_le lower) upper
      exact (inductionHypothesis (Nat.le_of_succ_le upper) previousWithin).trans
        (blockRank_next_of_active height ⟨index, previousWithin⟩ within active)
  exact along last.val ordered (Nat.le_refl _) last.isLt

theorem same_sign_profiles_ordered_same_block {rows width : Nat} (height : Cell rows width -> Int)
    (heightValid : IsHeight height) (rowsPositive : 0 < rows) (widthValid : 2 ≤ width)
    (firstRow secondRow : Fin rows) (firstApex secondApex : Fin width) (sign : Int)
    (unitSign : sign = 1 ∨ sign = -1)
    (firstProfile : ∀ position : Fin width,
      height (firstRow, position) = height (firstRow, firstApex) + sign * |(position.val : Int) - firstApex.val|)
    (secondProfile : ∀ position : Fin width,
      height (secondRow, position) = height (secondRow, secondApex) + sign * |(position.val : Int) - secondApex.val|)
    (firstActive : activeColumn height firstApex) (secondActive : activeColumn height secondApex)
    (ordered : firstApex.val ≤ secondApex.val) : blockRank height firstApex = blockRank height secondApex := by
  apply blockRank_eq_of_active_interval height firstApex secondApex ordered
  intro column lower upper
  by_cases atFirst : column = firstApex
  · rw [atFirst]
    exact firstActive
  · by_cases atSecond : column = secondApex
    · rw [atSecond]
      exact secondActive
    · have firstDifferent : column.val ≠ firstApex.val := fun equality => atFirst (Fin.ext equality)
      have secondDifferent : column.val ≠ secondApex.val := fun equality => atSecond (Fin.ext equality)
      exact LeafProfiles.same_sign_profiles_active_between height heightValid rowsPositive widthValid
        firstRow secondRow firstApex secondApex column sign unitSign firstProfile secondProfile (by omega) (by omega)

theorem same_sign_profiles_same_block {rows width : Nat} (height : Cell rows width -> Int)
    (heightValid : IsHeight height) (rowsPositive : 0 < rows) (widthValid : 2 ≤ width)
    (firstRow secondRow : Fin rows) (firstApex secondApex : Fin width) (sign : Int)
    (unitSign : sign = 1 ∨ sign = -1)
    (firstProfile : ∀ position : Fin width,
      height (firstRow, position) = height (firstRow, firstApex) + sign * |(position.val : Int) - firstApex.val|)
    (secondProfile : ∀ position : Fin width,
      height (secondRow, position) = height (secondRow, secondApex) + sign * |(position.val : Int) - secondApex.val|)
    (firstActive : activeColumn height firstApex) (secondActive : activeColumn height secondApex) :
    blockRank height firstApex = blockRank height secondApex := by
  rcases Nat.le_total firstApex.val secondApex.val with ordered | reverse
  · exact same_sign_profiles_ordered_same_block height heightValid rowsPositive widthValid firstRow secondRow
      firstApex secondApex sign unitSign firstProfile secondProfile firstActive secondActive ordered
  · exact (same_sign_profiles_ordered_same_block height heightValid rowsPositive widthValid secondRow firstRow
      secondApex firstApex sign unitSign secondProfile firstProfile secondActive firstActive reverse).symm

theorem extrema_reachable_of_same_rowBlock {rows width : Nat} (height : Cell rows width -> Int)
    (first second : Extrema height) (sameBlock : rowOf height first = rowOf height second) :
    (graph height).Reachable (BlockGraph.Vertex.extremum first) (BlockGraph.Vertex.extremum second) := by
  have firstJoin : (graph height).Adj (BlockGraph.Vertex.extremum first)
      (BlockGraph.Vertex.rowBlock (rowOf height first)) := rfl
  have secondJoin : (graph height).Adj (BlockGraph.Vertex.rowBlock (rowOf height first))
      (BlockGraph.Vertex.extremum second) := sameBlock.symm
  exact firstJoin.reachable.trans secondJoin.reachable

theorem extrema_reachable_of_same_columnBlock {rows width : Nat} (height : Cell rows width -> Int)
    (first second : Extrema height) (sameBlock : columnOf height first = columnOf height second) :
    (graph height).Reachable (BlockGraph.Vertex.extremum first) (BlockGraph.Vertex.extremum second) := by
  have firstJoin : (graph height).Adj (BlockGraph.Vertex.extremum first)
      (BlockGraph.Vertex.columnBlock (columnOf height first)) := rfl
  have secondJoin : (graph height).Adj (BlockGraph.Vertex.columnBlock (columnOf height first))
      (BlockGraph.Vertex.extremum second) := sameBlock.symm
  exact firstJoin.reachable.trans secondJoin.reachable

theorem same_sign_profiles_reachable {rows width : Nat} (height : Cell rows width -> Int)
    (heightValid : IsHeight height) (rowsPositive : 0 < rows) (widthValid : 2 ≤ width)
    (first second : Extrema height) (sign : Int) (unitSign : sign = 1 ∨ sign = -1)
    (firstProfile : ∀ column : Fin width, height (first.val.1, column) =
      height first.val + sign * |(column.val : Int) - first.val.2.val|)
    (secondProfile : ∀ column : Fin width, height (second.val.1, column) =
      height second.val + sign * |(column.val : Int) - second.val.2.val|) :
    (graph height).Reachable (BlockGraph.Vertex.extremum first) (BlockGraph.Vertex.extremum second) :=
  extrema_reachable_of_same_columnBlock height first second
    (same_sign_profiles_same_block height heightValid rowsPositive widthValid first.val.1 second.val.1
      first.val.2 second.val.2 sign unitSign firstProfile secondProfile
      (extremum_column_active height first) (extremum_column_active height second))

noncomputable def extremumSign {rows width : Nat} (height : Cell rows width -> Int) (point : Extrema height) : Int :=
  if IsStrictLocalMin height point.val then 1 else -1

theorem extremumSign_unit {rows width : Nat} (height : Cell rows width -> Int) (point : Extrema height) :
    extremumSign height point = 1 ∨ extremumSign height point = -1 := by
  by_cases minimum : IsStrictLocalMin height point.val <;> simp [extremumSign, minimum]

theorem extremumSign_increments {rows width : Nat} (height : Cell rows width -> Int) (point : Extrema height) :
    ∀ neighbor : Cell rows width, adj point.val neighbor -> height neighbor = height point.val + extremumSign height point := by
  intro neighbor adjacent
  by_cases minimum : IsStrictLocalMin height point.val
  · simpa only [extremumSign, if_pos minimum] using minimum neighbor adjacent
  · have maximum := point.property.resolve_right minimum
    simpa only [extremumSign, if_neg minimum] using maximum neighbor adjacent

theorem row_leaf_fixed_sign_profile {rows width : Nat} (height : Cell rows width -> Int)
    (heightValid : IsHeight height) (rowsValid : 2 ≤ rows) (point : Extrema height)
    (leaf : (graph height).degree (BlockGraph.Vertex.rowBlock (rowOf height point)) = 1) :
    (∀ column : Fin width, height (point.val.1, column) = height point.val +
      extremumSign height point * |(column.val : Int) - point.val.2.val|) ∧
      ∀ column : Fin width, ∀ other : Fin rows, ((point.val.1.val : Int) - other.val).natAbs = 1 ->
        height (other, column) = height (point.val.1, column) + extremumSign height point := by
  obtain ⟨sign, _, increments, profile, transverse⟩ := row_leaf_profile height heightValid rowsValid point leaf
  have widthPositive : 0 < width := by have := point.val.2.isLt; omega
  have nontrivialGrid : 2 ≤ rows * width := by nlinarith
  obtain ⟨neighbor, adjacent⟩ := exists_neighbor nontrivialGrid point.val
  have actual := extremumSign_increments height point neighbor adjacent
  have provided := increments neighbor adjacent
  have sameSign : sign = extremumSign height point := by omega
  rw [sameSign] at profile transverse
  exact ⟨profile, transverse⟩

theorem row_leaves_opposite_sign {rows width : Nat} (height : Cell rows width -> Int)
    (heightValid : IsHeight height) (rowsValid : 2 ≤ rows) (widthValid : 2 ≤ width)
    (first second : Extrema height)
    (firstLeaf : (graph height).degree (BlockGraph.Vertex.rowBlock (rowOf height first)) = 1)
    (secondLeaf : (graph height).degree (BlockGraph.Vertex.rowBlock (rowOf height second)) = 1)
    (disconnected : ¬ (graph height).Reachable (BlockGraph.Vertex.extremum first) (BlockGraph.Vertex.extremum second)) :
    extremumSign height first = -extremumSign height second := by
  have unequal : extremumSign height first ≠ extremumSign height second := by
    intro sameSign
    apply disconnected
    apply same_sign_profiles_reachable height heightValid (by omega) widthValid first second
      (extremumSign height first) (extremumSign_unit height first)
      (row_leaf_fixed_sign_profile height heightValid rowsValid first firstLeaf).1
    rw [sameSign]
    exact (row_leaf_fixed_sign_profile height heightValid rowsValid second secondLeaf).1
  have firstUnit := extremumSign_unit height first
  have secondUnit := extremumSign_unit height second
  omega

theorem run_violation_two_trees {rows width : Nat} (height : Cell rows width -> Int)
    (violation : numExtrema height ≤ numFrozenRuns (swapH height) + numFrozenRuns height) :
    ∃ first second : (graph height).ConnectedComponent, first ≠ second ∧
      first.toSimpleGraph.IsTree ∧ second.toSimpleGraph.IsTree := by
  classical
  apply TreeInventory.two_tree_components (graph height)
  have vertices := card_graph_vertices height
  have edges := card_graph_edges height
  omega

theorem tree_two_leaves {Vertex : Type*} [Fintype Vertex] [DecidableEq Vertex]
    (treeGraph : SimpleGraph Vertex) [DecidableRel treeGraph.Adj] (tree : treeGraph.IsTree)
    (positive : ∀ vertex : Vertex, 0 < treeGraph.degree vertex) :
    ∃ first second : Vertex, first ≠ second ∧ treeGraph.degree first = 1 ∧ treeGraph.degree second = 1 := by
  have lower : ∀ vertex : Vertex, (2 : Nat) ≤ treeGraph.degree vertex + 1 := by
    intro vertex
    have bound := positive vertex
    omega
  have surplus : (∑ vertex : Vertex, treeGraph.degree vertex) + 2 ≤ ∑ _vertex : Vertex, (2 : Nat) := by
    have treeCount := tree.card_edgeFinset
    have handshake := treeGraph.sum_degrees_eq_twice_card_edges
    simp only [Finset.sum_const, Finset.card_univ, smul_eq_mul]
    omega
  obtain ⟨first, second, different, firstCount, secondCount⟩ :=
    TreeInventory.two_tree_indices (fun _vertex : Vertex => 2) (fun vertex : Vertex => treeGraph.degree vertex) lower surplus
  exact ⟨first, second, different, by omega, by omega⟩

theorem component_tree_two_leaves {rows width : Nat} (height : Cell rows width -> Int)
    (heightValid : IsHeight height) (rowsValid : 2 ≤ rows) (widthValid : 2 ≤ width)
    (component : (graph height).ConnectedComponent) (tree : component.toSimpleGraph.IsTree) :
    ∃ first second : GraphVertex height, first ≠ second ∧ first ∈ component ∧ second ∈ component ∧
      (graph height).degree first = 1 ∧ (graph height).degree second = 1 := by
  classical
  letI : Fintype component := Fintype.ofFinite _
  letI : DecidableRel component.toSimpleGraph.Adj := fun first second =>
    inferInstanceAs (Decidable ((graph height).Adj first.val second.val))
  have positive : ∀ vertex : component, 0 < component.toSimpleGraph.degree vertex := by
    intro vertex
    obtain ⟨neighbor, adjacent⟩ := graph_no_isolated_vertices height heightValid rowsValid widthValid vertex.val
    have inside := component.mem_supp_of_adj_mem_supp vertex.property adjacent
    change 0 < (component.toSimpleGraph.neighborFinset vertex).card
    apply Finset.card_pos.mpr
    refine ⟨⟨neighbor, inside⟩, ?_⟩
    simpa only [SimpleGraph.mem_neighborFinset] using adjacent
  obtain ⟨first, second, different, firstLeaf, secondLeaf⟩ := tree_two_leaves component.toSimpleGraph tree positive
  have firstDegree := TreeInventory.component_degree (graph height) component first
  have secondDegree := TreeInventory.component_degree (graph height) component second
  exact ⟨first.val, second.val, (fun equality => different (Subtype.ext equality)), first.property, second.property,
    firstDegree.symm.trans firstLeaf, secondDegree.symm.trans secondLeaf⟩

theorem degree_one_is_block {rows width : Nat} (height : Cell rows width -> Int) (vertex : GraphVertex height)
    (leaf : (graph height).degree vertex = 1) :
    (∃ row, vertex = BlockGraph.Vertex.rowBlock row) ∨ (∃ column, vertex = BlockGraph.Vertex.columnBlock column) := by
  cases vertex with
  | rowBlock row => exact Or.inl ⟨row, rfl⟩
  | columnBlock column => exact Or.inr ⟨column, rfl⟩
  | extremum edge =>
    have two := BlockGraph.degree_extremum (rowOf height) (columnOf height) edge
    change (graph height).degree (BlockGraph.Vertex.extremum edge) = 2 at two
    omega

theorem column_leaf_unique_extremum {rows width : Nat} (height : Cell rows width -> Int)
    (point : Extrema height)
    (leaf : (graph height).degree (BlockGraph.Vertex.columnBlock (columnOf height point)) = 1) :
    ∀ other : Extrema height, columnOf height other = columnOf height point -> other = point := by
  classical
  obtain ⟨uniqueEdge, _, unique⟩ :=
    BlockGraph.column_leaf_unique (rowOf height) (columnOf height) (columnOf height point) leaf
  have pointEquality := unique point rfl
  intro other belongs
  exact (unique other belongs).trans pointEquality.symm

theorem column_leaf_unique_row {rows width : Nat} (height : Cell rows width -> Int)
    (point : Extrema height)
    (leaf : (graph height).degree (BlockGraph.Vertex.columnBlock (columnOf height point)) = 1) :
    ∀ row : Fin rows, IsStrictLocalExtremum height (row, point.val.2) -> row = point.val.1 := by
  intro row extremum
  let candidate : Extrema height := ⟨(row, point.val.2), extremum⟩
  have equality := column_leaf_unique_extremum height point leaf candidate rfl
  exact congrArg Prod.fst (congrArg Subtype.val equality)

theorem column_leaf_neighbors_inactive {rows width : Nat} (height : Cell rows width -> Int)
    (point : Extrema height)
    (leaf : (graph height).degree (BlockGraph.Vertex.columnBlock (columnOf height point)) = 1) :
    ∀ other : Fin width, ((point.val.2.val : Int) - other.val).natAbs = 1 -> ¬ activeColumn height other := by
  intro other adjacent active
  obtain ⟨row, extremum⟩ := active
  let candidate : Extrema height := ⟨(row, other), extremum⟩
  have sameBlock : columnOf height candidate = columnOf height point :=
    blockRank_adjacent_active height other point.val.2 ⟨row, extremum⟩
      (extremum_column_active height point) (by omega)
  have equality := column_leaf_unique_extremum height point leaf candidate sameBlock
  have sameColumn : other = point.val.2 := congrArg Prod.snd (congrArg Subtype.val equality)
  have values := congrArg Fin.val sameColumn
  omega

theorem column_leaf_profile {rows width : Nat} (height : Cell rows width -> Int)
    (heightValid : IsHeight height) (widthValid : 2 ≤ width) (point : Extrema height)
    (leaf : (graph height).degree (BlockGraph.Vertex.columnBlock (columnOf height point)) = 1) :
    ∃ sign : Int, (sign = 1 ∨ sign = -1) ∧
      (∀ row : Fin rows, height (row, point.val.2) =
        height point.val + sign * |(row.val : Int) - point.val.1.val|) ∧
      ∀ row : Fin rows, ∀ other : Fin width, ((point.val.2.val : Int) - other.val).natAbs = 1 ->
        height (row, other) = height (row, point.val.2) + sign := by
  obtain ⟨sign, unitSign, _, profile, transverse⟩ :=
    LeafProfiles.row_leaf_signed_profile (swapH height) (swapH_isHeight heightValid) widthValid point.val.2 point.val.1
      ((swapH_strictExtremum_iff (h := height) (point.val.2, point.val.1)).mpr point.property)
      (fun row extremum => column_leaf_unique_row height point leaf row
        ((swapH_strictExtremum_iff (h := height) (point.val.2, row)).mp extremum))
      (column_leaf_neighbors_inactive height point leaf)
  exact ⟨sign, unitSign, profile, transverse⟩

theorem horizontal_neighbor {width : Nat} (widthValid : 2 ≤ width) (column : Fin width) :
    ∃ other : Fin width, ((column.val : Int) - other.val).natAbs = 1 := by
  by_cases atZero : column.val = 0
  · refine ⟨⟨1, by omega⟩, ?_⟩
    change ((column.val : Int) - 1).natAbs = 1
    omega
  · refine ⟨⟨column.val - 1, by have := column.isLt; omega⟩, ?_⟩
    change ((column.val : Int) - (column.val - 1 : Nat)).natAbs = 1
    omega

theorem column_leaf_fixed_sign_profile {rows width : Nat} (height : Cell rows width -> Int)
    (heightValid : IsHeight height) (widthValid : 2 ≤ width) (point : Extrema height)
    (leaf : (graph height).degree (BlockGraph.Vertex.columnBlock (columnOf height point)) = 1) :
    (∀ row : Fin rows, height (row, point.val.2) =
      height point.val + extremumSign height point * |(row.val : Int) - point.val.1.val|) ∧
      ∀ row : Fin rows, ∀ other : Fin width, ((point.val.2.val : Int) - other.val).natAbs = 1 ->
        height (row, other) = height (row, point.val.2) + extremumSign height point := by
  obtain ⟨sign, _, profile, transverse⟩ := column_leaf_profile height heightValid widthValid point leaf
  obtain ⟨other, adjacent⟩ := horizontal_neighbor widthValid point.val.2
  have localStep := transverse point.val.1 other adjacent
  change height (point.val.1, other) = height point.val + sign at localStep
  have gridAdjacent : adj point.val (point.val.1, other) := by
    simp only [adj, gdist]
    omega
  have actual := extremumSign_increments height point (point.val.1, other) gridAdjacent
  have sameSign : sign = extremumSign height point := by omega
  rw [sameSign] at profile transverse
  exact ⟨profile, transverse⟩

theorem interior_row_column_leaf_equal {rows width : Nat} (height : Cell rows width -> Int)
    (heightValid : IsHeight height) (rowsValid : 2 ≤ rows) (widthValid : 2 ≤ width)
    (first second : Extrema height)
    (rowLeaf : (graph height).degree (BlockGraph.Vertex.rowBlock (rowOf height first)) = 1)
    (columnLeaf : (graph height).degree (BlockGraph.Vertex.columnBlock (columnOf height second)) = 1)
    (interior : 0 < first.val.1.val ∧ first.val.1.val + 1 < rows) : first = second := by
  obtain ⟨sign, unitSign, profile, _⟩ := column_leaf_profile height heightValid widthValid second columnLeaf
  apply Subtype.ext
  exact (LeafProfiles.interior_row_column_leaf_coincide height heightValid rowsValid first.val.1 first.val.2
    first.property (row_leaf_unique_column height first rowLeaf) (row_leaf_neighbors_inactive height first rowLeaf)
    interior second.val.1 second.val.2 second.property sign unitSign profile).symm

theorem interior_column_row_leaf_equal {rows width : Nat} (height : Cell rows width -> Int)
    (heightValid : IsHeight height) (rowsValid : 2 ≤ rows) (widthValid : 2 ≤ width)
    (first second : Extrema height)
    (rowLeaf : (graph height).degree (BlockGraph.Vertex.rowBlock (rowOf height first)) = 1)
    (columnLeaf : (graph height).degree (BlockGraph.Vertex.columnBlock (columnOf height second)) = 1)
    (interior : 0 < second.val.2.val ∧ second.val.2.val + 1 < width) : first = second := by
  have firstProfile := (row_leaf_fixed_sign_profile height heightValid rowsValid first rowLeaf).1
  have secondProfile := (column_leaf_fixed_sign_profile height heightValid widthValid second columnLeaf).2
  have sameColumn : first.val.2 = second.val.2 :=
    LeafProfiles.interior_transverse_column_apex (swapH height) second.val.2 first.val.2 first.val.1
      (extremumSign height second) (extremumSign height first) interior (extremumSign_unit height first)
      (secondProfile first.val.1) firstProfile
  have sameRow : first.val.1 = second.val.1 := by
    apply column_leaf_unique_row height second columnLeaf first.val.1
    have extremum := first.property
    change IsStrictLocalExtremum height (first.val.1, first.val.2) at extremum
    rwa [sameColumn] at extremum
  exact Subtype.ext (Prod.ext sameRow sameColumn)

theorem separated_mixed_leaf_boundaries {rows width : Nat} (height : Cell rows width -> Int)
    (heightValid : IsHeight height) (rowsValid : 2 ≤ rows) (widthValid : 2 ≤ width)
    (first second : Extrema height)
    (rowLeaf : (graph height).degree (BlockGraph.Vertex.rowBlock (rowOf height first)) = 1)
    (columnLeaf : (graph height).degree (BlockGraph.Vertex.columnBlock (columnOf height second)) = 1)
    (disconnected : ¬ (graph height).Reachable (BlockGraph.Vertex.extremum first) (BlockGraph.Vertex.extremum second)) :
    (first.val.1.val = 0 ∨ first.val.1.val + 1 = rows) ∧
      (second.val.2.val = 0 ∨ second.val.2.val + 1 = width) := by
  have different : first ≠ second := by
    intro equality
    subst second
    exact disconnected (SimpleGraph.Walk.nil : (graph height).Walk _ _).reachable
  constructor
  · by_contra notBoundary
    have within := first.val.1.isLt
    exact different (interior_row_column_leaf_equal height heightValid rowsValid widthValid first second
      rowLeaf columnLeaf (by omega))
  · by_contra notBoundary
    have within := second.val.2.isLt
    exact different (interior_column_row_leaf_equal height heightValid rowsValid widthValid first second
      rowLeaf columnLeaf (by omega))

theorem separated_mixed_leaves_opposite_sign {rows width : Nat} (height : Cell rows width -> Int)
    (heightValid : IsHeight height) (rowsValid : 2 ≤ rows) (widthValid : 2 ≤ width)
    (first second : Extrema height)
    (rowLeaf : (graph height).degree (BlockGraph.Vertex.rowBlock (rowOf height first)) = 1)
    (columnLeaf : (graph height).degree (BlockGraph.Vertex.columnBlock (columnOf height second)) = 1)
    (disconnected : ¬ (graph height).Reachable (BlockGraph.Vertex.extremum first) (BlockGraph.Vertex.extremum second)) :
    extremumSign height first = -extremumSign height second := by
  have differentRows : second.val.1 ≠ first.val.1 := by
    intro equality
    apply disconnected
    apply extrema_reachable_of_same_rowBlock height first second
    change blockRank (swapH height) first.val.1 = blockRank (swapH height) second.val.1
    rw [equality]
  exact LeafProfiles.row_transverse_column_profile_sign height first.val.1 second.val.1 second.val.2
    (extremumSign height first) (extremumSign height second) (extremumSign_unit height second)
    ((row_leaf_fixed_sign_profile height heightValid rowsValid first rowLeaf).2 second.val.2)
    (column_leaf_fixed_sign_profile height heightValid widthValid second columnLeaf).1 differentRows

theorem same_sign_mixed_leaves_equal {rows width : Nat} (height : Cell rows width -> Int)
    (heightValid : IsHeight height) (rowsValid : 2 ≤ rows) (widthValid : 2 ≤ width)
    (first second : Extrema height)
    (rowLeaf : (graph height).degree (BlockGraph.Vertex.rowBlock (rowOf height first)) = 1)
    (columnLeaf : (graph height).degree (BlockGraph.Vertex.columnBlock (columnOf height second)) = 1)
    (sameSign : extremumSign height first = extremumSign height second) : first = second := by
  by_cases sameRow : second.val.1 = first.val.1
  · have sameColumn : second.val.2 = first.val.2 := by
      apply row_leaf_unique_column height first rowLeaf second.val.2
      have extremum := second.property
      change IsStrictLocalExtremum height (second.val.1, second.val.2) at extremum
      rwa [sameRow] at extremum
    exact Subtype.ext (Prod.ext sameRow sameColumn).symm
  · have opposite := LeafProfiles.row_transverse_column_profile_sign height first.val.1 second.val.1 second.val.2
      (extremumSign height first) (extremumSign height second) (extremumSign_unit height second)
      ((row_leaf_fixed_sign_profile height heightValid rowsValid first rowLeaf).2 second.val.2)
      (column_leaf_fixed_sign_profile height heightValid widthValid second columnLeaf).1 sameRow
    have unitSign := extremumSign_unit height second
    omega

theorem separated_mixed_leaf_saturated {rows width : Nat} (height : Cell rows width -> Int)
    (heightValid : IsHeight height) (rowsValid : 2 ≤ rows) (widthValid : 2 ≤ width)
    (first second : Extrema height)
    (rowLeaf : (graph height).degree (BlockGraph.Vertex.rowBlock (rowOf height first)) = 1)
    (columnLeaf : (graph height).degree (BlockGraph.Vertex.columnBlock (columnOf height second)) = 1)
    (disconnected : ¬ (graph height).Reachable (BlockGraph.Vertex.extremum first) (BlockGraph.Vertex.extremum second)) :
    height second.val - height first.val = extremumSign height first * gdist first.val second.val := by
  have firstProfile := (row_leaf_fixed_sign_profile height heightValid rowsValid first rowLeaf).1 second.val.2
  have secondProfile := (column_leaf_fixed_sign_profile height heightValid widthValid second columnLeaf).1 first.val.1
  have opposite := separated_mixed_leaves_opposite_sign height heightValid rowsValid widthValid
    first second rowLeaf columnLeaf disconnected
  have signFlip : extremumSign height second = -extremumSign height first := by omega
  rw [signFlip] at secondProfile
  have distance : gdist first.val second.val =
      |(first.val.1.val : Int) - second.val.1.val| + |(second.val.2.val : Int) - first.val.2.val| := by
    simp [gdist, abs_sub_comm]
  rw [distance]
  nlinarith

theorem separated_double_leaves_separable {rows width : Nat} (height : Cell rows width -> Int)
    (heightValid : IsHeight height) (rowsValid : 2 ≤ rows) (widthValid : 2 ≤ width)
    (first second : Extrema height)
    (firstRowLeaf : (graph height).degree (BlockGraph.Vertex.rowBlock (rowOf height first)) = 1)
    (firstColumnLeaf : (graph height).degree (BlockGraph.Vertex.columnBlock (columnOf height first)) = 1)
    (secondRowLeaf : (graph height).degree (BlockGraph.Vertex.rowBlock (rowOf height second)) = 1)
    (secondColumnLeaf : (graph height).degree (BlockGraph.Vertex.columnBlock (columnOf height second)) = 1)
    (disconnected : ¬ (graph height).Reachable (BlockGraph.Vertex.extremum first) (BlockGraph.Vertex.extremum second)) :
    AdditivelySeparable height := by
  have firstBoundary := separated_mixed_leaf_boundaries height heightValid rowsValid widthValid
    first second firstRowLeaf secondColumnLeaf disconnected
  have secondBoundary := separated_mixed_leaf_boundaries height heightValid rowsValid widthValid
    second first secondRowLeaf firstColumnLeaf (fun path => disconnected path.symm)
  have differentRows : first.val.1.val ≠ second.val.1.val := by
    intro equality
    apply disconnected
    apply extrema_reachable_of_same_rowBlock height first second
    exact congrArg (blockRank (swapH height)) (Fin.ext equality)
  have differentColumns : first.val.2.val ≠ second.val.2.val := by
    intro equality
    apply disconnected
    apply extrema_reachable_of_same_columnBlock height first second
    exact congrArg (blockRank height) (Fin.ext equality)
  apply Saturation.opposite_endpoints_separable height heightValid first.val second.val (extremumSign height first)
    (extremumSign_unit height first)
    (separated_mixed_leaf_saturated height heightValid rowsValid widthValid first second
      firstRowLeaf secondColumnLeaf disconnected)
  · have firstWithin := first.val.1.isLt
    have secondWithin := second.val.1.isLt
    omega
  · have firstWithin := first.val.2.isLt
    have secondWithin := second.val.2.isLt
    omega

theorem two_mixed_components_separable {rows width : Nat} (height : Cell rows width -> Int)
    (heightValid : IsHeight height) (rowsValid : 2 ≤ rows) (widthValid : 2 ≤ width)
    (firstRow firstColumn secondRow secondColumn : Extrema height)
    (firstRowLeaf : (graph height).degree (BlockGraph.Vertex.rowBlock (rowOf height firstRow)) = 1)
    (firstColumnLeaf : (graph height).degree (BlockGraph.Vertex.columnBlock (columnOf height firstColumn)) = 1)
    (secondRowLeaf : (graph height).degree (BlockGraph.Vertex.rowBlock (rowOf height secondRow)) = 1)
    (secondColumnLeaf : (graph height).degree (BlockGraph.Vertex.columnBlock (columnOf height secondColumn)) = 1)
    (firstConnected : (graph height).Reachable (BlockGraph.Vertex.extremum firstRow) (BlockGraph.Vertex.extremum firstColumn))
    (secondConnected : (graph height).Reachable (BlockGraph.Vertex.extremum secondRow) (BlockGraph.Vertex.extremum secondColumn))
    (disconnected : ¬ (graph height).Reachable (BlockGraph.Vertex.extremum firstRow) (BlockGraph.Vertex.extremum secondRow)) :
    AdditivelySeparable height := by
  have rowSigns := row_leaves_opposite_sign height heightValid rowsValid widthValid firstRow secondRow
    firstRowLeaf secondRowLeaf disconnected
  have firstCross := separated_mixed_leaves_opposite_sign height heightValid rowsValid widthValid
    secondRow firstColumn secondRowLeaf firstColumnLeaf (fun path => disconnected (firstConnected.trans path.symm))
  have secondCross := separated_mixed_leaves_opposite_sign height heightValid rowsValid widthValid
    firstRow secondColumn firstRowLeaf secondColumnLeaf (fun path => disconnected (path.trans secondConnected.symm))
  have firstEqual := same_sign_mixed_leaves_equal height heightValid rowsValid widthValid firstRow firstColumn
    firstRowLeaf firstColumnLeaf (by omega)
  have secondEqual := same_sign_mixed_leaves_equal height heightValid rowsValid widthValid secondRow secondColumn
    secondRowLeaf secondColumnLeaf (by omega)
  exact separated_double_leaves_separable height heightValid rowsValid widthValid firstRow secondRow firstRowLeaf
    (by rw [firstEqual]; exact firstColumnLeaf) secondRowLeaf
    (by rw [secondEqual]; exact secondColumnLeaf) disconnected

#print axioms blockRank_next_of_active
#print axioms blockRank_adjacent_active
#print axioms card_extrema
#print axioms card_graph_vertices
#print axioms card_graph_edges
#print axioms extremum_row_active
#print axioms extremum_column_active
#print axioms row_leaf_unique_extremum
#print axioms row_leaf_unique_column
#print axioms row_leaf_neighbors_inactive
#print axioms row_leaf_profile
#print axioms prefix_runStart_eq_previous_add_one
#print axioms prefix_last_eq_runs
#print axioms blockRank_realized_active
#print axioms columnOf_surjective
#print axioms rowOf_surjective
#print axioms graph_no_isolated_vertices
#print axioms blockRank_eq_of_active_interval
#print axioms same_sign_profiles_ordered_same_block
#print axioms same_sign_profiles_same_block
#print axioms extrema_reachable_of_same_rowBlock
#print axioms extrema_reachable_of_same_columnBlock
#print axioms same_sign_profiles_reachable
#print axioms extremumSign_unit
#print axioms extremumSign_increments
#print axioms row_leaf_fixed_sign_profile
#print axioms row_leaves_opposite_sign
#print axioms run_violation_two_trees
#print axioms tree_two_leaves
#print axioms component_tree_two_leaves
#print axioms degree_one_is_block
#print axioms column_leaf_unique_extremum
#print axioms column_leaf_unique_row
#print axioms column_leaf_neighbors_inactive
#print axioms column_leaf_profile
#print axioms horizontal_neighbor
#print axioms column_leaf_fixed_sign_profile
#print axioms interior_row_column_leaf_equal
#print axioms interior_column_row_leaf_equal
#print axioms separated_mixed_leaf_boundaries
#print axioms separated_mixed_leaves_opposite_sign
#print axioms same_sign_mixed_leaves_equal
#print axioms separated_mixed_leaf_saturated
#print axioms separated_double_leaves_separable
#print axioms two_mixed_components_separable

end OrigamiCone.Sequel.OccupiedBlocks
