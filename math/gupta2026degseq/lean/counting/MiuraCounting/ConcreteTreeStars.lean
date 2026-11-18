import MiuraCounting.ConcreteOccupiedBlocks

namespace OrigamiCone.Sequel.TreeStars

open OrigamiCone FrozenExpansion OccupiedBlocks

theorem farthest_is_leaf {Vertex : Type*} [Fintype Vertex]
    (graph : SimpleGraph Vertex) [DecidableRel graph.Adj]
    (tree : graph.IsTree) (center farthest : Vertex)
    (maximal : ∀ vertex, graph.dist center vertex ≤ graph.dist center farthest)
    (different : center ≠ farthest) : graph.degree farthest = 1 := by
  classical
  obtain ⟨path, pathValid, pathLength⟩ :=
    tree.isConnected.exists_path_of_dist center farthest
  apply SimpleGraph.degree_eq_one_iff_existsUnique_adj.mpr
  refine ⟨path.penultimate,
    (path.adj_penultimate (SimpleGraph.Walk.not_nil_of_ne different)).symm, ?_⟩
  intro neighbor near
  obtain ⟨otherPath, otherValid, otherLength⟩ :=
    tree.isConnected.exists_path_of_dist center neighbor
  have endpointAbsent : farthest ∉ otherPath.support := by
    intro member
    have equality := tree.IsAcyclic.path_concat pathValid otherValid near member
    have lengths := congrArg (fun walk : graph.Walk center neighbor => walk.length) equality
    simp only [SimpleGraph.Walk.length_concat, pathLength, otherLength] at lengths
    have bound := maximal neighbor
    omega
  have neighborMember := tree.IsAcyclic.mem_support_of_ne_mem_support_of_adj_of_isPath
    pathValid otherValid near endpointAbsent
  have equality := tree.IsAcyclic.path_concat otherValid pathValid near.symm neighborMember
  have penultimate := congrArg (fun walk : graph.Walk center farthest => walk.penultimate) equality
  simpa only [SimpleGraph.Walk.penultimate_concat] using penultimate.symm

theorem leaf_distance_bound {Vertex : Type*} [Fintype Vertex]
    (graph : SimpleGraph Vertex) [DecidableRel graph.Adj]
    (tree : graph.IsTree) (center : Vertex) (bound : Nat)
    (leavesBounded : ∀ vertex, graph.degree vertex = 1 -> graph.dist center vertex ≤ bound) :
    ∀ vertex, graph.dist center vertex ≤ bound := by
  classical
  haveI : Nonempty Vertex := ⟨center⟩
  obtain ⟨farthest, maximal⟩ := Finite.exists_max (fun vertex => graph.dist center vertex)
  by_cases same : center = farthest
  · intro vertex
    have distance := maximal vertex
    rw [← same, SimpleGraph.dist_self] at distance
    omega
  · have leaf := farthest_is_leaf graph tree center farthest maximal same
    intro vertex
    exact (maximal vertex).trans (leavesBounded farthest leaf)

theorem tree_is_star {Vertex : Type*} [Fintype Vertex]
    (graph : SimpleGraph Vertex) [DecidableRel graph.Adj]
    (tree : graph.IsTree) (center : Vertex)
    (leavesAdjacent : ∀ vertex, graph.degree vertex = 1 ->
      vertex = center ∨ graph.Adj center vertex) :
    ∀ vertex, vertex = center ∨ graph.Adj center vertex := by
  classical
  have allBounded : ∀ vertex, graph.dist center vertex ≤ 1 := by
    apply leaf_distance_bound graph tree center 1
    intro vertex leaf
    rcases leavesAdjacent vertex leaf with same | near
    · rw [same, SimpleGraph.dist_self]
      omega
    · exact le_of_eq (SimpleGraph.dist_eq_one_iff_adj.mpr near)
  intro vertex
  by_cases same : vertex = center
  · exact Or.inl same
  · right
    apply SimpleGraph.dist_eq_one_iff_adj.mp
    have positive := tree.isConnected.pos_dist_of_ne (fun equality => same equality.symm)
    have bounded := allBounded vertex
    omega

theorem column_eq_of_short_walk {RowBlock ColumnBlock Extremum : Type*}
    (rowOf : Extremum -> RowBlock) (columnOf : Extremum -> ColumnBlock)
    (center other : ColumnBlock)
    (walk : (BlockGraph.incidenceGraph rowOf columnOf).Walk
      (BlockGraph.Vertex.columnBlock center) (BlockGraph.Vertex.columnBlock other))
    (short : walk.length ≤ 2) : center = other := by
  cases walk with
  | nil => rfl
  | @cons _ middle _ near tail =>
    cases middle with
    | rowBlock row => cases near
    | columnBlock column => cases near
    | extremum edge =>
      change columnOf edge = center at near
      cases tail with
      | @cons _ after _ nextNear rest =>
        have restLength : rest.length = 0 := by
          simp only [SimpleGraph.Walk.length_cons] at short
          omega
        have endpoint := rest.eq_of_length_eq_zero restLength
        cases endpoint
        change columnOf edge = other at nextNear
        exact near.symm.trans nextNear

theorem extremum_column_eq_of_short_walk {RowBlock ColumnBlock Extremum : Type*}
    (rowOf : Extremum -> RowBlock) (columnOf : Extremum -> ColumnBlock)
    (center : ColumnBlock) (point : Extremum)
    (walk : (BlockGraph.incidenceGraph rowOf columnOf).Walk
      (BlockGraph.Vertex.columnBlock center) (BlockGraph.Vertex.extremum point))
    (short : walk.length ≤ 2) : columnOf point = center := by
  cases walk with
  | @cons _ middle _ near tail =>
    cases middle with
    | rowBlock row => cases near
    | columnBlock column => cases near
    | extremum edge =>
      change columnOf edge = center at near
      cases tail with
      | nil => exact near
      | @cons _ after _ nextNear rest =>
        have restLength : rest.length = 0 := by
          simp only [SimpleGraph.Walk.length_cons] at short
          omega
        have endpoint := rest.eq_of_length_eq_zero restLength
        cases endpoint
        cases nextNear

theorem component_column_center {rows width : Nat} (height : Cell rows width -> Int)
    (component : (OccupiedBlocks.graph height).ConnectedComponent)
    (tree : component.toSimpleGraph.IsTree)
    (center : Fin (numFrozenRuns height + 1))
    (centerMember : BlockGraph.Vertex.columnBlock center ∈ component)
    (leavesAtCenter : ∀ vertex : component,
      (OccupiedBlocks.graph height).degree vertex.val = 1 ->
      ∃ point : Extrema height,
        vertex.val = BlockGraph.Vertex.rowBlock (rowOf height point) ∧
          columnOf height point = center) :
    (∀ point : Extrema height, BlockGraph.Vertex.extremum point ∈ component ->
      columnOf height point = center) ∧
    (∀ column, BlockGraph.Vertex.columnBlock column ∈ component -> column = center) := by
  classical
  letI : Fintype component := Fintype.ofFinite _
  let root : component := ⟨BlockGraph.Vertex.columnBlock center, centerMember⟩
  have leavesBounded : ∀ vertex : component, component.toSimpleGraph.degree vertex = 1 ->
      component.toSimpleGraph.dist root vertex ≤ 2 := by
    intro vertex leaf
    have ambientLeaf : (OccupiedBlocks.graph height).degree vertex.val = 1 := by
      obtain ⟨neighbor, near, unique⟩ :=
        SimpleGraph.degree_eq_one_iff_existsUnique_adj.mp leaf
      apply SimpleGraph.degree_eq_one_iff_existsUnique_adj.mpr
      refine ⟨neighbor.val, near, ?_⟩
      intro other otherNear
      let otherVertex : component := ⟨other,
        component.mem_supp_of_adj_mem_supp vertex.property otherNear⟩
      exact congrArg Subtype.val (unique otherVertex otherNear)
    obtain ⟨point, atRow, atCenter⟩ := leavesAtCenter vertex ambientLeaf
    have centerAdjacent : (OccupiedBlocks.graph height).Adj
        (BlockGraph.Vertex.columnBlock center) (BlockGraph.Vertex.extremum point) := atCenter
    let edgeVertex : component := ⟨BlockGraph.Vertex.extremum point,
      component.mem_supp_of_adj_mem_supp centerMember centerAdjacent⟩
    have firstAdjacent : component.toSimpleGraph.Adj root edgeVertex := centerAdjacent
    have secondAdjacent : component.toSimpleGraph.Adj edgeVertex vertex := by
      change (OccupiedBlocks.graph height).Adj (BlockGraph.Vertex.extremum point) vertex.val
      rw [atRow]
      rfl
    let walk : component.toSimpleGraph.Walk root vertex :=
      .cons firstAdjacent (.cons secondAdjacent .nil)
    exact (component.toSimpleGraph.dist_le walk).trans (show walk.length ≤ 2 from by rfl)
  have allBounded := leaf_distance_bound component.toSimpleGraph tree root 2 leavesBounded
  have shortWalk : ∀ vertex : component,
      ∃ walk : (OccupiedBlocks.graph height).Walk
        (BlockGraph.Vertex.columnBlock center) vertex.val, walk.length ≤ 2 := by
    intro vertex
    obtain ⟨walk, length⟩ := tree.isConnected.exists_walk_length_eq_dist root vertex
    refine ⟨walk.map component.toSimpleGraph_hom, ?_⟩
    rw [SimpleGraph.Walk.length_map, length]
    exact allBounded vertex
  constructor
  · intro point member
    obtain ⟨walk, short⟩ := shortWalk ⟨BlockGraph.Vertex.extremum point, member⟩
    exact extremum_column_eq_of_short_walk (rowOf height) (columnOf height)
      center point walk short
  · intro column member
    obtain ⟨walk, short⟩ := shortWalk ⟨BlockGraph.Vertex.columnBlock column, member⟩
    exact (column_eq_of_short_walk (rowOf height) (columnOf height)
      center column walk short).symm

theorem component_extrema_unique {rows width : Nat} (height : Cell rows width -> Int)
    (component : (OccupiedBlocks.graph height).ConnectedComponent)
    (tree : component.toSimpleGraph.IsTree) (first second : Extrema height)
    (firstMember : BlockGraph.Vertex.extremum first ∈ component)
    (secondMember : BlockGraph.Vertex.extremum second ∈ component)
    (sameRow : rowOf height first = rowOf height second)
    (sameColumn : columnOf height first = columnOf height second) : first = second := by
  classical
  let firstVertex : component := ⟨BlockGraph.Vertex.extremum first, firstMember⟩
  let secondVertex : component := ⟨BlockGraph.Vertex.extremum second, secondMember⟩
  have firstColumnAdjacent : (OccupiedBlocks.graph height).Adj
      (BlockGraph.Vertex.extremum first) (BlockGraph.Vertex.columnBlock (columnOf height first)) := rfl
  have firstRowAdjacent : (OccupiedBlocks.graph height).Adj
      (BlockGraph.Vertex.extremum first) (BlockGraph.Vertex.rowBlock (rowOf height first)) := rfl
  let columnVertex : component := ⟨BlockGraph.Vertex.columnBlock (columnOf height first),
    component.mem_supp_of_adj_mem_supp firstMember firstColumnAdjacent⟩
  let rowVertex : component := ⟨BlockGraph.Vertex.rowBlock (rowOf height first),
    component.mem_supp_of_adj_mem_supp firstMember firstRowAdjacent⟩
  have firstStart : component.toSimpleGraph.Adj columnVertex firstVertex := rfl
  have firstEnd : component.toSimpleGraph.Adj firstVertex rowVertex := rfl
  have secondStart : component.toSimpleGraph.Adj columnVertex secondVertex := sameColumn.symm
  have secondEnd : component.toSimpleGraph.Adj secondVertex rowVertex := sameRow.symm
  let firstPath : component.toSimpleGraph.Walk columnVertex rowVertex :=
    .cons firstStart (.cons firstEnd .nil)
  let secondPath : component.toSimpleGraph.Walk columnVertex rowVertex :=
    .cons secondStart (.cons secondEnd .nil)
  have firstValid : firstPath.IsPath := by
    simp [firstPath, SimpleGraph.Walk.cons_isPath_iff, firstVertex, columnVertex, rowVertex]
  have secondValid : secondPath.IsPath := by
    simp [secondPath, SimpleGraph.Walk.cons_isPath_iff, secondVertex, columnVertex, rowVertex]
  have equality := (tree.existsUnique_path columnVertex rowVertex).unique firstValid secondValid
  have middle := congrArg
    (fun walk : component.toSimpleGraph.Walk columnVertex rowVertex => walk.snd) equality
  have vertices := congrArg Subtype.val middle
  change BlockGraph.Vertex.extremum first = BlockGraph.Vertex.extremum second at vertices
  exact BlockGraph.Vertex.extremum.inj vertices

theorem component_center_row_leaf {rows width : Nat} (height : Cell rows width -> Int)
    (component : (OccupiedBlocks.graph height).ConnectedComponent)
    (tree : component.toSimpleGraph.IsTree) (center : Fin (numFrozenRuns height + 1))
    (centered : ∀ point : Extrema height, BlockGraph.Vertex.extremum point ∈ component ->
      columnOf height point = center)
    (point : Extrema height) (pointMember : BlockGraph.Vertex.extremum point ∈ component) :
    (OccupiedBlocks.graph height).degree (BlockGraph.Vertex.rowBlock (rowOf height point)) = 1 := by
  classical
  apply SimpleGraph.degree_eq_one_iff_existsUnique_adj.mpr
  refine ⟨BlockGraph.Vertex.extremum point, rfl, ?_⟩
  intro vertex near
  cases vertex with
  | rowBlock row => cases near
  | columnBlock column => cases near
  | extremum other =>
    have sameRow : rowOf height other = rowOf height point := near
    have pointRowAdjacent : (OccupiedBlocks.graph height).Adj
        (BlockGraph.Vertex.extremum point) (BlockGraph.Vertex.rowBlock (rowOf height point)) := rfl
    have rowMember := component.mem_supp_of_adj_mem_supp pointMember pointRowAdjacent
    have otherMember := component.mem_supp_of_adj_mem_supp rowMember near
    have sameColumn := (centered point pointMember).trans (centered other otherMember).symm
    have equality := component_extrema_unique height component tree point other
      pointMember otherMember sameRow.symm sameColumn
    exact congrArg BlockGraph.Vertex.extremum equality.symm

theorem component_leaf_witness {rows width : Nat} (height : Cell rows width -> Int)
    (component : (OccupiedBlocks.graph height).ConnectedComponent) (vertex : component)
    (leaf : (OccupiedBlocks.graph height).degree vertex.val = 1) :
    ∃ point : Extrema height, BlockGraph.Vertex.extremum point ∈ component ∧
      (vertex.val = BlockGraph.Vertex.rowBlock (rowOf height point) ∨
        vertex.val = BlockGraph.Vertex.columnBlock (columnOf height point)) := by
  classical
  rcases degree_one_is_block height vertex.val leaf with ⟨row, atRow⟩ | ⟨column, atColumn⟩
  · have rowLeaf : (OccupiedBlocks.graph height).degree (BlockGraph.Vertex.rowBlock row) = 1 := by
      rw [← atRow]
      exact leaf
    obtain ⟨point, belongs, _⟩ := BlockGraph.row_leaf_unique (rowOf height) (columnOf height) row rowLeaf
    have atPoint : vertex.val = BlockGraph.Vertex.rowBlock (rowOf height point) := by
      rw [belongs]
      exact atRow
    have near : (OccupiedBlocks.graph height).Adj vertex.val (BlockGraph.Vertex.extremum point) := by
      rw [atPoint]
      rfl
    exact ⟨point, component.mem_supp_of_adj_mem_supp vertex.property near, Or.inl atPoint⟩
  · have columnLeaf : (OccupiedBlocks.graph height).degree (BlockGraph.Vertex.columnBlock column) = 1 := by
      rw [← atColumn]
      exact leaf
    obtain ⟨point, belongs, _⟩ := BlockGraph.column_leaf_unique (rowOf height) (columnOf height) column columnLeaf
    have atPoint : vertex.val = BlockGraph.Vertex.columnBlock (columnOf height point) := by
      rw [belongs]
      exact atColumn
    have near : (OccupiedBlocks.graph height).Adj vertex.val (BlockGraph.Vertex.extremum point) := by
      rw [atPoint]
      rfl
    exact ⟨point, component.mem_supp_of_adj_mem_supp vertex.property near, Or.inr atPoint⟩

theorem one_sided_component_star {rows width : Nat} (height : Cell rows width -> Int)
    (heightValid : IsHeight height) (rowsValid : 2 ≤ rows) (widthValid : 2 ≤ width)
    (component : (OccupiedBlocks.graph height).ConnectedComponent)
    (tree : component.toSimpleGraph.IsTree)
    (onlyRows : ∀ vertex : component, (OccupiedBlocks.graph height).degree vertex.val = 1 ->
      ∃ row, vertex.val = BlockGraph.Vertex.rowBlock row)
    (base opposite : Extrema height)
    (baseMember : BlockGraph.Vertex.extremum base ∈ component)
    (baseLeaf : (OccupiedBlocks.graph height).degree (BlockGraph.Vertex.rowBlock (rowOf height base)) = 1)
    (oppositeLeaf : (OccupiedBlocks.graph height).degree (BlockGraph.Vertex.rowBlock (rowOf height opposite)) = 1)
    (outside : BlockGraph.Vertex.extremum opposite ∉ component) :
    (∀ point : Extrema height, BlockGraph.Vertex.extremum point ∈ component ->
      columnOf height point = columnOf height base) ∧
    (∀ point : Extrema height, BlockGraph.Vertex.extremum point ∈ component ->
      (OccupiedBlocks.graph height).degree (BlockGraph.Vertex.rowBlock (rowOf height point)) = 1) ∧
    (∀ point : Extrema height, BlockGraph.Vertex.extremum point ∈ component ->
      extremumSign height point = extremumSign height base) := by
  classical
  have separated : ∀ point : Extrema height, BlockGraph.Vertex.extremum point ∈ component ->
      ¬ (OccupiedBlocks.graph height).Reachable
        (BlockGraph.Vertex.extremum point) (BlockGraph.Vertex.extremum opposite) := by
    intro point member reachable
    apply outside
    change (OccupiedBlocks.graph height).connectedComponentMk (BlockGraph.Vertex.extremum opposite) = component
    exact (SimpleGraph.ConnectedComponent.sound reachable).symm.trans member
  have baseSign := row_leaves_opposite_sign height heightValid rowsValid widthValid
    base opposite baseLeaf oppositeLeaf (separated base baseMember)
  have leavesAtCenter : ∀ vertex : component,
      (OccupiedBlocks.graph height).degree vertex.val = 1 ->
      ∃ point : Extrema height, vertex.val = BlockGraph.Vertex.rowBlock (rowOf height point) ∧
        columnOf height point = columnOf height base := by
    intro vertex leaf
    obtain ⟨point, member, kind⟩ := component_leaf_witness height component vertex leaf
    obtain ⟨row, atRow⟩ := onlyRows vertex leaf
    have atPoint : vertex.val = BlockGraph.Vertex.rowBlock (rowOf height point) := by
      rcases kind with rowKind | columnKind
      · exact rowKind
      · have collision := atRow.symm.trans columnKind
        cases collision
    have pointLeaf : (OccupiedBlocks.graph height).degree
        (BlockGraph.Vertex.rowBlock (rowOf height point)) = 1 := by
      rw [← atPoint]
      exact leaf
    have pointSign := row_leaves_opposite_sign height heightValid rowsValid widthValid
      point opposite pointLeaf oppositeLeaf (separated point member)
    have sameSign := pointSign.trans baseSign.symm
    have pointProfile := (row_leaf_fixed_sign_profile height heightValid rowsValid point pointLeaf).1
    have baseProfile := (row_leaf_fixed_sign_profile height heightValid rowsValid base baseLeaf).1
    refine ⟨point, atPoint, ?_⟩
    exact same_sign_profiles_same_block height heightValid (by omega) widthValid
      point.val.1 base.val.1 point.val.2 base.val.2 (extremumSign height point)
      (extremumSign_unit height point) pointProfile (by simpa only [sameSign] using baseProfile)
      (extremum_column_active height point) (extremum_column_active height base)
  have centerAdjacent : (OccupiedBlocks.graph height).Adj
      (BlockGraph.Vertex.extremum base) (BlockGraph.Vertex.columnBlock (columnOf height base)) := rfl
  have centerMember := component.mem_supp_of_adj_mem_supp baseMember centerAdjacent
  have centered := (component_column_center height component tree (columnOf height base)
    centerMember leavesAtCenter).1
  have rowLeaves : ∀ point : Extrema height, BlockGraph.Vertex.extremum point ∈ component ->
      (OccupiedBlocks.graph height).degree (BlockGraph.Vertex.rowBlock (rowOf height point)) = 1 := by
    intro point member
    exact component_center_row_leaf height component tree (columnOf height base) centered point member
  refine ⟨centered, rowLeaves, ?_⟩
  intro point member
  exact (row_leaves_opposite_sign height heightValid rowsValid widthValid point opposite
    (rowLeaves point member) oppositeLeaf (separated point member)).trans baseSign.symm

#print axioms farthest_is_leaf
#print axioms leaf_distance_bound
#print axioms tree_is_star
#print axioms column_eq_of_short_walk
#print axioms extremum_column_eq_of_short_walk
#print axioms component_column_center
#print axioms component_extrema_unique
#print axioms component_center_row_leaf
#print axioms component_leaf_witness
#print axioms one_sided_component_star

end OrigamiCone.Sequel.TreeStars
