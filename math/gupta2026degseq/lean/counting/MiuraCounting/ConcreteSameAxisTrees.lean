import MiuraCounting.ConcreteTreeStars
import MiuraCounting.ConcreteBoundaryComparison
import MiuraCounting.ConcreteBoundaryBlocks
import MiuraCounting.ConcreteGridReflection
import MiuraCounting.ConcreteGridNegation
import MiuraCounting.ConcreteBlockTranspose

namespace OrigamiCone.Sequel.SameAxisTrees

open OrigamiCone FrozenExpansion OccupiedBlocks

theorem opposite_leaf_outer_blocks {rows width : Nat} (height : Cell rows width -> Int)
    (heightValid : IsHeight height) (rowsValid : 2 ≤ rows) (widthValid : 2 ≤ width)
    (first second : Extrema height)
    (firstLeaf : (OccupiedBlocks.graph height).degree (BlockGraph.Vertex.rowBlock (rowOf height first)) = 1)
    (secondLeaf : (OccupiedBlocks.graph height).degree (BlockGraph.Vertex.rowBlock (rowOf height second)) = 1)
    (opposite : extremumSign height first = -extremumSign height second)
    (ordered : first.val.2.val ≤ second.val.2.val) :
    blockRank height ⟨0, by omega⟩ = columnOf height first ∧
      columnOf height second = blockRank height ⟨width - 1, by omega⟩ := by
  have firstProfile := (row_leaf_fixed_sign_profile height heightValid rowsValid first firstLeaf).1
  have secondProfile := (row_leaf_fixed_sign_profile height heightValid rowsValid second secondLeaf).1
  have secondSign : extremumSign height second = -extremumSign height first := by omega
  rw [secondSign] at secondProfile
  constructor
  · apply blockRank_eq_of_active_interval height ⟨0, by omega⟩ first.val.2 (Nat.zero_le _)
    intro column _ upper
    by_cases atApex : column = first.val.2
    · rw [atApex]
      exact extremum_column_active height first
    · apply (column_active_or_frozen height heightValid (by omega) widthValid column).resolve_right
      intro frozen
      let next : Fin width := ⟨column.val + 1, frozen.2.1⟩
      have nextValue : next.val = column.val + 1 := rfl
      have different : column.val ≠ first.val.2.val := fun equality => atApex (Fin.ext equality)
      have firstCurrent := firstProfile column
      have firstNext := firstProfile next
      have secondCurrent := secondProfile column
      have secondNext := secondProfile next
      rw [abs_of_nonpos (show (column.val : Int) - first.val.2.val ≤ 0 from by omega)] at firstCurrent
      rw [abs_of_nonpos (show (next.val : Int) - first.val.2.val ≤ 0 from by omega)] at firstNext
      rw [abs_of_nonpos (show (column.val : Int) - second.val.2.val ≤ 0 from by omega)] at secondCurrent
      rw [abs_of_nonpos (show (next.val : Int) - second.val.2.val ≤ 0 from by omega)] at secondNext
      have firstStep := frozenSlope_eq_right height heightValid (by omega) column frozen first.val.1
      have secondStep := frozenSlope_eq_right height heightValid (by omega) column frozen second.val.1
      change height (first.val.1, next) - height (first.val.1, column) = _ at firstStep
      change height (second.val.1, next) - height (second.val.1, column) = _ at secondStep
      rcases extremumSign_unit height first with signCase | signCase <;>
        rw [signCase] at firstCurrent firstNext secondCurrent secondNext <;>
        simp only [one_mul, neg_one_mul, neg_neg] at firstCurrent firstNext secondCurrent secondNext <;> omega
  · apply blockRank_eq_of_active_interval height second.val.2 ⟨width - 1, by omega⟩
      (show second.val.2.val ≤ width - 1 from by have := second.val.2.isLt; omega)
    intro column lower _
    by_cases atApex : column = second.val.2
    · rw [atApex]
      exact extremum_column_active height second
    · apply (column_active_or_frozen height heightValid (by omega) widthValid column).resolve_right
      intro frozen
      let previous : Fin width := ⟨column.val - 1, by have := column.isLt; omega⟩
      have previousValue : previous.val = column.val - 1 := rfl
      have different : column.val ≠ second.val.2.val := fun equality => atApex (Fin.ext equality)
      have firstCurrent := firstProfile column
      have firstPrevious := firstProfile previous
      have secondCurrent := secondProfile column
      have secondPrevious := secondProfile previous
      rw [abs_of_nonneg (show 0 ≤ (column.val : Int) - first.val.2.val from by omega)] at firstCurrent
      rw [abs_of_nonneg (show 0 ≤ (previous.val : Int) - first.val.2.val from by omega)] at firstPrevious
      rw [abs_of_nonneg (show 0 ≤ (column.val : Int) - second.val.2.val from by omega)] at secondCurrent
      rw [abs_of_nonneg (show 0 ≤ (previous.val : Int) - second.val.2.val from by omega)] at secondPrevious
      have firstStep := frozenSlope_eq height heightValid (by omega) column frozen first.val.1
      have secondStep := frozenSlope_eq height heightValid (by omega) column frozen second.val.1
      change height (first.val.1, column) - height (first.val.1, previous) = _ at firstStep
      change height (second.val.1, column) - height (second.val.1, previous) = _ at secondStep
      rcases extremumSign_unit height first with signCase | signCase <;>
        rw [signCase] at firstCurrent firstPrevious secondCurrent secondPrevious <;>
        simp only [one_mul, neg_one_mul, neg_neg] at firstCurrent firstPrevious secondCurrent secondPrevious <;> omega

theorem first_boundary_signed_offset {rows width : Nat} (height : Cell rows width -> Int)
    (heightValid : IsHeight height) (widthValid : 2 ≤ width) (sign : Int)
    (allSigns : ∀ point : Extrema height, point.val.2.val = 0 -> extremumSign height point = sign) :
    ∀ row : Fin rows, height (row, ⟨1, by omega⟩) = height (row, ⟨0, by omega⟩) + sign := by
  intro row
  let own := ConcreteTransfer.heightColumn height (⟨0, by omega⟩ : Fin width)
  let adjacent := ConcreteTransfer.heightColumn height (⟨1, by omega⟩ : Fin width)
  have offsets := ConcreteTransfer.heightColumn_offsets height heightValid
    (⟨0, by omega⟩ : Fin width) ⟨1, by omega⟩ (by norm_num)
  obtain ⟨first, last, before, after, inside, constantBlock, maximalBefore, maximalAfter⟩ :=
    ConcreteTransfer.maximal_constant_block (fun index => adjacent index - own index)
      rows row.val row.isLt
  obtain ⟨index, afterFirst, beforeLast, boundaryExtremum⟩ :=
    ConcreteTransfer.boundary_block_contains_extremum own adjacent rows first last (by omega)
      inside (ConcreteTransfer.heightColumn_steps height heightValid _)
      (ConcreteTransfer.heightColumn_steps height heightValid _) offsets
      (adjacent row.val - own row.val) constantBlock maximalBefore maximalAfter
  have indexBound : index < rows := by omega
  let point : Extrema height := ⟨(⟨index, indexBound⟩, ⟨0, by omega⟩),
    ConcreteTransfer.first_boundary_row_extremum height heightValid widthValid index indexBound boundaryExtremum⟩
  have sameSign := allSigns point rfl
  have near : adj point.val (⟨index, indexBound⟩, (⟨1, by omega⟩ : Fin width)) := by
    simp [point, adj, gdist]
  have increment := extremumSign_increments height point _ near
  rw [sameSign] at increment
  have equality := constantBlock index afterFirst beforeLast
  dsimp only [own, adjacent] at equality
  simp only [ConcreteTransfer.heightColumn, dif_pos indexBound, dif_pos row.isLt] at equality
  have rowEta : (⟨row.val, row.isLt⟩ : Fin rows) = row := Fin.ext rfl
  rw [rowEta] at equality
  change height (⟨index, indexBound⟩, ⟨1, by omega⟩) =
    height (⟨index, indexBound⟩, ⟨0, by omega⟩) + sign at increment
  omega

theorem last_boundary_signed_offset {rows width : Nat} (height : Cell rows width -> Int)
    (heightValid : IsHeight height) (widthValid : 2 ≤ width) (sign : Int)
    (allSigns : ∀ point : Extrema height, point.val.2.val + 1 = width ->
      extremumSign height point = sign) :
    ∀ row : Fin rows, height (row, ⟨width - 2, by omega⟩) =
      height (row, ⟨width - 1, by omega⟩) + sign := by
  classical
  have reflectedSigns : ∀ point : Extrema (GridReflection.reflectH height),
      point.val.2.val = 0 -> extremumSign (GridReflection.reflectH height) point = sign := by
    intro point firstColumn
    let original : Extrema height := ⟨colRefl point.val,
      (GridReflection.reflectH_extremum_iff height point.val).mp point.property⟩
    have lastColumn : original.val.2.val + 1 = width := by
      change width - (point.val.2.val + 1) + 1 = width
      omega
    have originalSign := allSigns original lastColumn
    have sameMinimum := GridReflection.reflectH_minimum_iff height point.val
    change (if IsStrictLocalMin (GridReflection.reflectH height) point.val then 1 else -1) = sign
    change (if IsStrictLocalMin height (colRefl point.val) then 1 else -1) = sign at originalSign
    simpa only [sameMinimum] using originalSign
  have reflected := first_boundary_signed_offset (GridReflection.reflectH height)
    (GridReflection.reflectH_isHeight height heightValid) widthValid sign reflectedSigns
  intro row
  have values := reflected row
  have zeroReversed : Fin.rev (⟨0, by omega⟩ : Fin width) = ⟨width - 1, by omega⟩ := Fin.ext rfl
  have oneReversed : Fin.rev (⟨1, by omega⟩ : Fin width) = ⟨width - 2, by omega⟩ := Fin.ext rfl
  change height (row, Fin.rev (⟨1, by omega⟩ : Fin width)) =
    height (row, Fin.rev (⟨0, by omega⟩ : Fin width)) + sign at values
  simpa only [zeroReversed, oneReversed] using values

theorem first_minimum_block_singleton {rows width : Nat} (height : Cell rows width -> Int)
    (heightValid : IsHeight height) (rowsPositive : 0 < rows) (widthValid : 2 ≤ width)
    (allSigns : ∀ point : Extrema height,
      columnOf height point = blockRank height ⟨0, by omega⟩ -> extremumSign height point = 1) :
    ∀ column : Fin width, blockRank height column = blockRank height ⟨0, by omega⟩ -> column.val = 0 := by
  have increments := first_boundary_signed_offset height heightValid widthValid 1 (by
    intro point firstColumn
    apply allSigns point
    apply congrArg (blockRank height)
    exact Fin.ext firstColumn)
  have inactive : ¬ activeColumn height (⟨1, by omega⟩ : Fin width) := by
    rintro ⟨row, isExtremum⟩
    let point : Extrema height := ⟨(row, ⟨1, by omega⟩), isExtremum⟩
    have sameBlock : columnOf height point = blockRank height ⟨0, by omega⟩ := by
      have relation := blockRank_next_of_active height (⟨0, by omega⟩ : Fin width)
        (by change 0 + 1 < width; omega) ⟨row, isExtremum⟩
      exact relation.symm
    have sign := allSigns point sameBlock
    have near : adj point.val (row, (⟨0, by omega⟩ : Fin width)) := by
      simp [point, adj, gdist]
    have actual := extremumSign_increments height point _ near
    rw [sign] at actual
    have inward := increments row
    change height (row, ⟨0, by omega⟩) = height (row, ⟨1, by omega⟩) + 1 at actual
    omega
  have start : frozenRunStart height (⟨1, by omega⟩ : Fin width) := by
    refine ⟨inactive, Or.inr ⟨by norm_num, ?_⟩⟩
    exact firstColumn_active height heightValid rowsPositive widthValid
  intro column belongs
  by_contra positive
  have strict := RunLengthFibers.runPrefixCount_lt_of_start height
    (first := (⟨0, by omega⟩ : Fin width)) (second := (⟨1, by omega⟩ : Fin width))
    (by norm_num) start
  have monotone := RunLengthFibers.runPrefixCount_mono height
    (first := (⟨1, by omega⟩ : Fin width)) (second := column) (by
      change 1 ≤ column.val
      omega)
  have ranks := congrArg Fin.val belongs
  change RunLengthFibers.runPrefixCount height column =
    RunLengthFibers.runPrefixCount height ⟨0, by omega⟩ at ranks
  omega

theorem blockRank_active_interval {rows width : Nat} (height : Cell rows width -> Int)
  (first last : Fin width)
    (firstActive : activeColumn height first)
    (sameRank : blockRank height first = blockRank height last) :
    ∀ column : Fin width, first.val ≤ column.val -> column.val ≤ last.val ->
      activeColumn height column := by
  classical
  intro column lower upper
  by_contra inactive
  have strictlyAfter : first.val < column.val := by
    by_contra notAfter
    have same : column = first := Fin.ext (by omega)
    exact inactive (same.symm ▸ firstActive)
  let candidates := Finset.univ.filter (fun point : Fin width =>
    first.val < point.val ∧ point.val ≤ column.val ∧ ¬ activeColumn height point)
  have nonempty : candidates.Nonempty :=
    ⟨column, Finset.mem_filter.mpr ⟨Finset.mem_univ _, strictlyAfter, le_refl _, inactive⟩⟩
  let start := candidates.min' nonempty
  have properties := (Finset.mem_filter.mp (Finset.min'_mem candidates nonempty)).2
  have positive : 0 < start.val := by omega
  let previous : Fin width := ⟨start.val - 1, by have := start.isLt; omega⟩
  have previousActive : activeColumn height previous := by
    by_contra previousInactive
    by_cases atFirst : previous = first
    · exact previousInactive (atFirst.symm ▸ firstActive)
    · have previousMember : previous ∈ candidates := by
        apply Finset.mem_filter.mpr
        refine ⟨Finset.mem_univ _, ?_, ?_, previousInactive⟩
        · have different : previous.val ≠ first.val := fun equality => atFirst (Fin.ext equality)
          change first.val < start.val - 1
          change start.val - 1 ≠ first.val at different
          omega
        · change start.val - 1 ≤ column.val
          omega
      have minimal := Finset.min'_le candidates previous previousMember
      change start.val ≤ start.val - 1 at minimal
      omega
  have isStart : frozenRunStart height start :=
    ⟨properties.2.2, Or.inr ⟨positive, previousActive⟩⟩
  have strict : RunLengthFibers.runPrefixCount height first <
      RunLengthFibers.runPrefixCount height start :=
    RunLengthFibers.runPrefixCount_lt_of_start height
      (first := first) (second := start) properties.1 isStart
  have monotone := RunLengthFibers.runPrefixCount_mono height
    (first := start) (second := last) (properties.2.1.trans upper)
  have ranks := congrArg Fin.val sameRank
  change RunLengthFibers.runPrefixCount height first =
    RunLengthFibers.runPrefixCount height last at ranks
  omega

theorem last_maximum_block_singleton {rows width : Nat} (height : Cell rows width -> Int)
    (heightValid : IsHeight height) (rowsPositive : 0 < rows) (widthValid : 2 ≤ width)
    (allSigns : ∀ point : Extrema height,
      columnOf height point = blockRank height ⟨width - 1, by omega⟩ ->
        extremumSign height point = -1) :
    ∀ point : Extrema height,
      columnOf height point = blockRank height ⟨width - 1, by omega⟩ ->
        point.val.2.val + 1 = width := by
  have increments := last_boundary_signed_offset height heightValid widthValid (-1) (by
    intro point lastColumn
    apply allSigns point
    apply congrArg (blockRank height)
    apply Fin.ext
    change point.val.2.val = width - 1
    omega)
  let previous : Fin width := ⟨width - 2, by omega⟩
  let last : Fin width := ⟨width - 1, by omega⟩
  have lastActive : activeColumn height last :=
    lastColumn_active height heightValid rowsPositive widthValid
  have inactive : ¬ activeColumn height previous := by
    rintro ⟨row, isExtremum⟩
    let point : Extrema height := ⟨(row, previous), isExtremum⟩
    have sameBlock : columnOf height point = blockRank height last := by
      apply blockRank_adjacent_active height previous last ⟨row, isExtremum⟩ lastActive
      dsimp only [previous, last]
      omega
    have sign := allSigns point sameBlock
    have near : adj point.val (row, last) := by
      simp only [adj, gdist]
      dsimp only [point, previous, last]
      omega
    have actual := extremumSign_increments height point _ near
    rw [sign] at actual
    have inward := increments row
    change height (row, last) = height (row, previous) + (-1) at actual
    change height (row, previous) = height (row, last) + (-1) at inward
    omega
  intro point belongs
  by_contra notLast
  have pointBound := point.val.2.isLt
  have allActive := blockRank_active_interval height point.val.2 last
    (extremum_column_active height point) belongs
  apply inactive
  apply allActive previous
  · change point.val.2.val ≤ width - 2
    omega
  · change width - 2 ≤ width - 1
    omega

theorem ordered_components_boundary_data {rows width : Nat} (height : Cell rows width -> Int)
    (heightValid : IsHeight height) (rowsValid : 2 ≤ rows) (widthValid : 2 ≤ width)
    (firstComponent secondComponent : (OccupiedBlocks.graph height).ConnectedComponent)
    (different : firstComponent ≠ secondComponent)
    (firstTree : firstComponent.toSimpleGraph.IsTree)
    (secondTree : secondComponent.toSimpleGraph.IsTree)
    (firstOnlyRows : ∀ vertex : firstComponent, (OccupiedBlocks.graph height).degree vertex.val = 1 ->
      ∃ row, vertex.val = BlockGraph.Vertex.rowBlock row)
    (secondOnlyRows : ∀ vertex : secondComponent, (OccupiedBlocks.graph height).degree vertex.val = 1 ->
      ∃ row, vertex.val = BlockGraph.Vertex.rowBlock row)
    (first second : Extrema height)
    (firstMember : BlockGraph.Vertex.extremum first ∈ firstComponent)
    (secondMember : BlockGraph.Vertex.extremum second ∈ secondComponent)
    (firstLeaf : (OccupiedBlocks.graph height).degree (BlockGraph.Vertex.rowBlock (rowOf height first)) = 1)
    (secondLeaf : (OccupiedBlocks.graph height).degree (BlockGraph.Vertex.rowBlock (rowOf height second)) = 1)
    (ordered : first.val.2.val ≤ second.val.2.val) :
    extremumSign height first = -extremumSign height second ∧
    (∀ point : Extrema height, point.val.2.val = 0 ->
      extremumSign height point = extremumSign height first) ∧
    (∀ point : Extrema height, point.val.2.val + 1 = width ->
      extremumSign height point = extremumSign height second) ∧
    (∀ point : Extrema height, point.val.2.val = 0 ->
      (OccupiedBlocks.graph height).degree (BlockGraph.Vertex.rowBlock (rowOf height point)) = 1) ∧
    (∀ point : Extrema height, point.val.2.val + 1 = width ->
      (OccupiedBlocks.graph height).degree (BlockGraph.Vertex.rowBlock (rowOf height point)) = 1) := by
  have firstOutside : BlockGraph.Vertex.extremum first ∉ secondComponent := by
    intro member
    exact different (firstMember.symm.trans member)
  have secondOutside : BlockGraph.Vertex.extremum second ∉ firstComponent := by
    intro member
    exact different (member.symm.trans secondMember)
  have disconnected : ¬ (OccupiedBlocks.graph height).Reachable
      (BlockGraph.Vertex.extremum first) (BlockGraph.Vertex.extremum second) := by
    intro reachable
    apply secondOutside
    exact (SimpleGraph.ConnectedComponent.sound reachable).symm.trans firstMember
  have opposite := row_leaves_opposite_sign height heightValid rowsValid widthValid
    first second firstLeaf secondLeaf disconnected
  have firstStar := TreeStars.one_sided_component_star height heightValid rowsValid widthValid
    firstComponent firstTree firstOnlyRows first second firstMember firstLeaf secondLeaf secondOutside
  have secondStar := TreeStars.one_sided_component_star height heightValid rowsValid widthValid
    secondComponent secondTree secondOnlyRows second first secondMember secondLeaf firstLeaf firstOutside
  have blocks := opposite_leaf_outer_blocks height heightValid rowsValid widthValid
    first second firstLeaf secondLeaf opposite ordered
  have firstBlockMember : ∀ point : Extrema height,
      columnOf height point = blockRank height ⟨0, by omega⟩ ->
      BlockGraph.Vertex.extremum point ∈ firstComponent := by
    intro point belongs
    have same := belongs.trans blocks.1
    have reachable := extrema_reachable_of_same_columnBlock height first point same.symm
    exact (SimpleGraph.ConnectedComponent.sound reachable).symm.trans firstMember
  have lastBlockMember : ∀ point : Extrema height,
      columnOf height point = blockRank height ⟨width - 1, by omega⟩ ->
      BlockGraph.Vertex.extremum point ∈ secondComponent := by
    intro point belongs
    have same := belongs.trans blocks.2.symm
    have reachable := extrema_reachable_of_same_columnBlock height second point same.symm
    exact (SimpleGraph.ConnectedComponent.sound reachable).symm.trans secondMember
  have firstBoundaryMember : ∀ point : Extrema height, point.val.2.val = 0 ->
      BlockGraph.Vertex.extremum point ∈ firstComponent := by
    intro point atFirst
    apply firstBlockMember point
    exact congrArg (blockRank height) (Fin.ext atFirst)
  have lastBoundaryMember : ∀ point : Extrema height, point.val.2.val + 1 = width ->
      BlockGraph.Vertex.extremum point ∈ secondComponent := by
    intro point atLast
    apply lastBlockMember point
    apply congrArg (blockRank height)
    apply Fin.ext
    change point.val.2.val = width - 1
    omega
  refine ⟨opposite, ?_, ?_, ?_, ?_⟩
  · intro point atFirst
    exact firstStar.2.2 point (firstBoundaryMember point atFirst)
  · intro point atLast
    exact secondStar.2.2 point (lastBoundaryMember point atLast)
  · intro point atFirst
    exact firstStar.2.1 point (firstBoundaryMember point atFirst)
  · intro point atLast
    exact secondStar.2.1 point (lastBoundaryMember point atLast)

theorem ordered_components_separable {rows width : Nat} (height : Cell rows width -> Int)
    (heightValid : IsHeight height) (rowsValid : 2 ≤ rows) (widthValid : 2 ≤ width)
    (firstComponent secondComponent : (OccupiedBlocks.graph height).ConnectedComponent)
    (different : firstComponent ≠ secondComponent)
    (firstTree : firstComponent.toSimpleGraph.IsTree)
    (secondTree : secondComponent.toSimpleGraph.IsTree)
    (firstOnlyRows : ∀ vertex : firstComponent, (OccupiedBlocks.graph height).degree vertex.val = 1 ->
      ∃ row, vertex.val = BlockGraph.Vertex.rowBlock row)
    (secondOnlyRows : ∀ vertex : secondComponent, (OccupiedBlocks.graph height).degree vertex.val = 1 ->
      ∃ row, vertex.val = BlockGraph.Vertex.rowBlock row)
    (first second : Extrema height)
    (firstMember : BlockGraph.Vertex.extremum first ∈ firstComponent)
    (secondMember : BlockGraph.Vertex.extremum second ∈ secondComponent)
    (firstLeaf : (OccupiedBlocks.graph height).degree (BlockGraph.Vertex.rowBlock (rowOf height first)) = 1)
    (secondLeaf : (OccupiedBlocks.graph height).degree (BlockGraph.Vertex.rowBlock (rowOf height second)) = 1)
    (ordered : first.val.2.val ≤ second.val.2.val) : AdditivelySeparable height := by
  obtain ⟨opposite, firstSigns, lastSigns, firstLeaves, lastLeaves⟩ :=
    ordered_components_boundary_data height heightValid rowsValid widthValid firstComponent secondComponent
      different firstTree secondTree firstOnlyRows secondOnlyRows first second firstMember secondMember
      firstLeaf secondLeaf ordered
  rcases extremumSign_unit height first with minimum | maximum
  · have secondMaximum : extremumSign height second = -1 := by omega
    have firstInward := first_boundary_signed_offset height heightValid widthValid 1 (by
      intro point atFirst
      exact (firstSigns point atFirst).trans minimum)
    have lastInward := last_boundary_signed_offset height heightValid widthValid (-1) (by
      intro point atLast
      exact (lastSigns point atLast).trans secondMaximum)
    exact BoundaryComparison.boundary_leaf_profiles_separable height heightValid rowsValid widthValid
      firstInward lastInward firstLeaves lastLeaves
  · have secondMinimum : extremumSign height second = 1 := by omega
    have firstInward := first_boundary_signed_offset height heightValid widthValid (-1) (by
      intro point atFirst
      exact (firstSigns point atFirst).trans maximum)
    have lastInward := last_boundary_signed_offset height heightValid widthValid 1 (by
      intro point atLast
      exact (lastSigns point atLast).trans secondMinimum)
    exact GridNegation.negative_boundary_profiles_separable height heightValid rowsValid widthValid
      firstInward lastInward firstLeaves lastLeaves

theorem two_row_only_components_separable {rows width : Nat} (height : Cell rows width -> Int)
    (heightValid : IsHeight height) (rowsValid : 2 ≤ rows) (widthValid : 2 ≤ width)
    (firstComponent secondComponent : (OccupiedBlocks.graph height).ConnectedComponent)
    (different : firstComponent ≠ secondComponent)
    (firstTree : firstComponent.toSimpleGraph.IsTree)
    (secondTree : secondComponent.toSimpleGraph.IsTree)
    (firstOnlyRows : ∀ vertex : firstComponent, (OccupiedBlocks.graph height).degree vertex.val = 1 ->
      ∃ row, vertex.val = BlockGraph.Vertex.rowBlock row)
    (secondOnlyRows : ∀ vertex : secondComponent, (OccupiedBlocks.graph height).degree vertex.val = 1 ->
      ∃ row, vertex.val = BlockGraph.Vertex.rowBlock row)
    (first second : Extrema height)
    (firstMember : BlockGraph.Vertex.extremum first ∈ firstComponent)
    (secondMember : BlockGraph.Vertex.extremum second ∈ secondComponent)
    (firstLeaf : (OccupiedBlocks.graph height).degree (BlockGraph.Vertex.rowBlock (rowOf height first)) = 1)
    (secondLeaf : (OccupiedBlocks.graph height).degree (BlockGraph.Vertex.rowBlock (rowOf height second)) = 1) :
    AdditivelySeparable height := by
  rcases Nat.le_total first.val.2.val second.val.2.val with ordered | reverse
  · exact ordered_components_separable height heightValid rowsValid widthValid firstComponent secondComponent
      different firstTree secondTree firstOnlyRows secondOnlyRows first second firstMember secondMember
      firstLeaf secondLeaf ordered
  · exact ordered_components_separable height heightValid rowsValid widthValid secondComponent firstComponent
      (fun equality => different equality.symm) secondTree firstTree secondOnlyRows firstOnlyRows
      second first secondMember firstMember secondLeaf firstLeaf reverse

def componentTransportIso {Source Target : Type*}
    {sourceGraph : SimpleGraph Source} {targetGraph : SimpleGraph Target}
    (iso : sourceGraph ≃g targetGraph) (component : sourceGraph.ConnectedComponent) :
    component.toSimpleGraph ≃g (iso.connectedComponentEquiv component).toSimpleGraph where
  toEquiv := {
    toFun := fun vertex => ⟨iso vertex.val,
      SimpleGraph.ConnectedComponent.iso_image_comp_eq_map_iff_eq_comp.mpr vertex.property⟩
    invFun := fun vertex => ⟨iso.symm vertex.val,
      SimpleGraph.ConnectedComponent.iso_inv_image_comp_eq_iff_eq_map.mpr vertex.property⟩
    left_inv := fun vertex => Subtype.ext (iso.symm_apply_apply vertex.val)
    right_inv := fun vertex => Subtype.ext (iso.apply_symm_apply vertex.val) }
  map_rel_iff' := by
    intro first second
    change targetGraph.Adj (iso first.val) (iso second.val) ↔ sourceGraph.Adj first.val second.val
    exact iso.map_rel_iff

theorem two_column_only_components_separable {rows width : Nat} (height : Cell rows width -> Int)
    (heightValid : IsHeight height) (rowsValid : 2 ≤ rows) (widthValid : 2 ≤ width)
    (firstComponent secondComponent : (OccupiedBlocks.graph height).ConnectedComponent)
    (different : firstComponent ≠ secondComponent)
    (firstTree : firstComponent.toSimpleGraph.IsTree)
    (secondTree : secondComponent.toSimpleGraph.IsTree)
    (firstOnlyColumns : ∀ vertex : firstComponent, (OccupiedBlocks.graph height).degree vertex.val = 1 ->
      ∃ column, vertex.val = BlockGraph.Vertex.columnBlock column)
    (secondOnlyColumns : ∀ vertex : secondComponent, (OccupiedBlocks.graph height).degree vertex.val = 1 ->
      ∃ column, vertex.val = BlockGraph.Vertex.columnBlock column)
    (first second : Extrema height)
    (firstMember : BlockGraph.Vertex.extremum first ∈ firstComponent)
    (secondMember : BlockGraph.Vertex.extremum second ∈ secondComponent)
    (firstLeaf : (OccupiedBlocks.graph height).degree (BlockGraph.Vertex.columnBlock (columnOf height first)) = 1)
    (secondLeaf : (OccupiedBlocks.graph height).degree (BlockGraph.Vertex.columnBlock (columnOf height second)) = 1) :
    AdditivelySeparable height := by
  classical
  let iso := BlockTranspose.transposeGraphIso height
  have transportTree : ∀ component : (OccupiedBlocks.graph height).ConnectedComponent,
      component.toSimpleGraph.IsTree -> (iso.connectedComponentEquiv component).toSimpleGraph.IsTree := by
    intro component tree
    refine ⟨(iso.connectedComponentEquiv component).connected_toSimpleGraph, ?_⟩
    exact tree.IsAcyclic.comap (componentTransportIso iso component).symm
      (componentTransportIso iso component).symm.injective
  have transportLeaves : ∀ component : (OccupiedBlocks.graph height).ConnectedComponent,
      (∀ vertex : component, (OccupiedBlocks.graph height).degree vertex.val = 1 ->
        ∃ column, vertex.val = BlockGraph.Vertex.columnBlock column) ->
      ∀ vertex : iso.connectedComponentEquiv component,
        (OccupiedBlocks.graph (swapH height)).degree vertex.val = 1 ->
          ∃ row, vertex.val = BlockGraph.Vertex.rowBlock row := by
    intro component onlyColumns vertex leaf
    let original : component := (componentTransportIso iso component).symm vertex
    have mapped : iso original.val = vertex.val :=
      congrArg Subtype.val ((componentTransportIso iso component).apply_symm_apply vertex)
    have degree := iso.degree_eq original.val
    rw [mapped] at degree
    have originalLeaf : (OccupiedBlocks.graph height).degree original.val = 1 :=
      degree.symm.trans leaf
    obtain ⟨column, atColumn⟩ := onlyColumns original originalLeaf
    have atRow := congrArg iso atColumn
    change iso original.val = BlockGraph.Vertex.rowBlock column at atRow
    exact ⟨column, mapped.symm.trans atRow⟩
  have firstMapped : BlockGraph.Vertex.extremum (BlockTranspose.transposePoint height first) ∈
      iso.connectedComponentEquiv firstComponent := by
    change (OccupiedBlocks.graph (swapH height)).connectedComponentMk
      (iso (BlockGraph.Vertex.extremum first)) = firstComponent.map iso
    exact SimpleGraph.ConnectedComponent.iso_image_comp_eq_map_iff_eq_comp.mpr firstMember
  have secondMapped : BlockGraph.Vertex.extremum (BlockTranspose.transposePoint height second) ∈
      iso.connectedComponentEquiv secondComponent := by
    change (OccupiedBlocks.graph (swapH height)).connectedComponentMk
      (iso (BlockGraph.Vertex.extremum second)) = secondComponent.map iso
    exact SimpleGraph.ConnectedComponent.iso_image_comp_eq_map_iff_eq_comp.mpr secondMember
  apply (separable_swap_iff height).mp
  exact two_row_only_components_separable (swapH height) (swapH_isHeight heightValid) widthValid rowsValid
    (iso.connectedComponentEquiv firstComponent) (iso.connectedComponentEquiv secondComponent)
    (fun equality => different (iso.connectedComponentEquiv.injective equality))
    (transportTree firstComponent firstTree) (transportTree secondComponent secondTree)
    (transportLeaves firstComponent firstOnlyColumns) (transportLeaves secondComponent secondOnlyColumns)
    (BlockTranspose.transposePoint height first) (BlockTranspose.transposePoint height second)
    firstMapped secondMapped ((BlockTranspose.transpose_column_degree height first).trans firstLeaf)
    ((BlockTranspose.transpose_column_degree height second).trans secondLeaf)

#print axioms opposite_leaf_outer_blocks
#print axioms first_boundary_signed_offset
#print axioms last_boundary_signed_offset
#print axioms first_minimum_block_singleton
#print axioms blockRank_active_interval
#print axioms last_maximum_block_singleton
#print axioms ordered_components_boundary_data
#print axioms ordered_components_separable
#print axioms two_row_only_components_separable
#print axioms componentTransportIso
#print axioms two_column_only_components_separable

end OrigamiCone.Sequel.SameAxisTrees
