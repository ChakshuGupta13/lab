import OrigamiCone.SequelEdBoundary
import OrigamiCone.ColReflectTransport

namespace OrigamiCone.Sequel.ConcreteTransfer

theorem unit_offset_change (first second adjacentFirst adjacentSecond : Int)
    (ownStep : second - first = 1 ∨ second - first = -1)
    (adjacentStep : adjacentSecond - adjacentFirst = 1 ∨
      adjacentSecond - adjacentFirst = -1)
    (firstOffset : adjacentFirst - first = 1 ∨ adjacentFirst - first = -1)
    (secondOffset : adjacentSecond - second = 1 ∨ adjacentSecond - second = -1)
    (offsetChanged : adjacentFirst - first ≠ adjacentSecond - second) :
    first = second + (adjacentSecond - second) ∧
      second = first + (adjacentFirst - first) := by
  rcases ownStep with ownPositive | ownNegative <;>
    rcases adjacentStep with adjacentPositive | adjacentNegative <;>
    rcases firstOffset with firstPositive | firstNegative <;>
    rcases secondOffset with secondPositive | secondNegative <;> omega

theorem boundary_block_contains_extremum (own adjacent : Nat -> Int)
    (rows first last : Nat) (ordered : first ≤ last) (lastBound : last < rows)
    (ownSteps : ∀ index, index + 1 < rows →
      own (index + 1) - own index = 1 ∨ own (index + 1) - own index = -1)
    (adjacentSteps : ∀ index, index + 1 < rows →
      adjacent (index + 1) - adjacent index = 1 ∨
        adjacent (index + 1) - adjacent index = -1)
    (offsets : ∀ index, index < rows →
      adjacent index - own index = 1 ∨ adjacent index - own index = -1)
    (offset : Int)
    (constantBlock : ∀ index, first ≤ index → index ≤ last →
      adjacent index - own index = offset)
    (maximalBefore : first = 0 ∨ adjacent (first - 1) - own (first - 1) ≠ offset)
    (maximalAfter : last + 1 = rows ∨ adjacent (last + 1) - own (last + 1) ≠ offset) :
    ∃ index, first ≤ index ∧ index ≤ last ∧
      (isMaxRow (rows - 1) own adjacent index ∨
        isMinRow (rows - 1) own adjacent index) := by
  have entering : 0 < first → own (first - 1) = own first + offset := by
    intro positive
    have previousNext : first - 1 + 1 = first := by omega
    have ownStep := ownSteps (first - 1) (by omega)
    have adjacentStep := adjacentSteps (first - 1) (by omega)
    rw [previousNext] at ownStep adjacentStep
    have changed : adjacent (first - 1) - own (first - 1) ≠
        adjacent first - own first := by
      rw [constantBlock first (le_refl _) ordered]
      exact maximalBefore.resolve_left (by omega)
    have identity := (unit_offset_change (own (first - 1)) (own first)
      (adjacent (first - 1)) (adjacent first) ownStep adjacentStep
      (offsets (first - 1) (by omega)) (offsets first (by omega)) changed).1
    rw [constantBlock first (le_refl _) ordered] at identity
    exact identity
  have leaving : last + 1 < rows → own (last + 1) = own last + offset := by
    intro nextBound
    have changed : adjacent last - own last ≠
        adjacent (last + 1) - own (last + 1) := by
      intro equality
      apply maximalAfter.resolve_left (by omega)
      exact equality.symm.trans (constantBlock last ordered (le_refl _))
    have identity := (unit_offset_change (own last) (own (last + 1))
      (adjacent last) (adjacent (last + 1)) (ownSteps last nextBound)
      (adjacentSteps last nextBound) (offsets last lastBound)
      (offsets (last + 1) nextBound) changed).2
    rw [constantBlock last ordered (le_refl _)] at identity
    exact identity
  obtain ⟨index, indexBound, segmentExtremum⟩ :=
    boundary_walk_abstract (last - first)
      (fun index => own (first + index)) (fun index => adjacent (first + index))
      (fun index bound => ownSteps (first + index) (by omega))
      (fun index bound => offsets (first + index) (by omega))
      (fun index bound => adjacentSteps (first + index) (by omega))
  have segmentBefore : 0 < index →
      own (first + (index - 1)) = adjacent (first + index) := by
    intro positive
    rcases segmentExtremum with maximum | minimum
    · exact (maximum.1 positive).trans maximum.2.2.symm
    · exact (minimum.1 positive).trans minimum.2.2.symm
  have segmentAfter : index < last - first →
      own (first + (index + 1)) = adjacent (first + index) := by
    intro inside
    rcases segmentExtremum with maximum | minimum
    · exact (maximum.2.1 inside).trans maximum.2.2.symm
    · exact (minimum.2.1 inside).trans minimum.2.2.symm
  have before : 0 < first + index →
      own (first + index - 1) = adjacent (first + index) := by
    intro positive
    by_cases inside : 0 < index
    · have aligned : first + (index - 1) = first + index - 1 := by omega
      rw [← aligned]
      exact segmentBefore inside
    · have indexZero : index = 0 := by omega
      simp only [indexZero, Nat.add_zero] at positive ⊢
      have identity := entering positive
      have sameOffset := constantBlock first (le_refl _) ordered
      omega
  have after : first + index + 1 < rows →
      own (first + index + 1) = adjacent (first + index) := by
    intro nextBound
    by_cases inside : index < last - first
    · exact segmentAfter inside
    · have atLast : first + index = last := by omega
      rw [atLast] at nextBound ⊢
      have identity := leaving nextBound
      have sameOffset := constantBlock last ordered (le_refl _)
      omega
  refine ⟨first + index, by omega, by omega, ?_⟩
  rcases offsets (first + index) (by omega) with positive | negative
  · right
    refine ⟨?_, ?_, by omega⟩
    · intro bound
      have identity := before bound
      omega
    · intro bound
      have identity := after (by omega)
      omega
  · left
    refine ⟨?_, ?_, by omega⟩
    · intro bound
      have identity := before bound
      omega
    · intro bound
      have identity := after (by omega)
      omega

theorem maximal_constant_block {Value : Type} [DecidableEq Value]
    (values : Nat -> Value) (rows index : Nat) (indexBound : index < rows) :
    ∃ first last, first ≤ index ∧ index ≤ last ∧ last < rows ∧
      (∀ position, first ≤ position → position ≤ last → values position = values index) ∧
      (first = 0 ∨ values (first - 1) ≠ values index) ∧
      (last + 1 = rows ∨ values (last + 1) ≠ values index) := by
  let starts := (Finset.range (index + 1)).filter (fun first =>
    ∀ position ∈ Finset.Icc first index, values position = values index)
  let ends := (Finset.range rows).filter (fun last => index ≤ last ∧
    ∀ position ∈ Finset.Icc index last, values position = values index)
  have startsNonempty : starts.Nonempty := by
    refine ⟨index, Finset.mem_filter.mpr ⟨Finset.mem_range.mpr (by omega), ?_⟩⟩
    intro position member
    have bounds := Finset.mem_Icc.mp member
    have same : position = index := by omega
    rw [same]
  have endsNonempty : ends.Nonempty := by
    refine ⟨index, Finset.mem_filter.mpr ⟨Finset.mem_range.mpr indexBound,
      le_refl _, ?_⟩⟩
    intro position member
    have bounds := Finset.mem_Icc.mp member
    have same : position = index := by omega
    rw [same]
  let first := starts.min' startsNonempty
  let last := ends.max' endsNonempty
  have firstMember : first ∈ starts := Finset.min'_mem starts startsNonempty
  have lastMember : last ∈ ends := Finset.max'_mem ends endsNonempty
  have firstProperties := Finset.mem_filter.mp firstMember
  have lastProperties := Finset.mem_filter.mp lastMember
  have firstBound : first ≤ index := by
    have bound := Finset.mem_range.mp firstProperties.1
    omega
  have lastBound : last < rows := Finset.mem_range.mp lastProperties.1
  refine ⟨first, last, firstBound, lastProperties.2.1, lastBound, ?_, ?_, ?_⟩
  · intro position afterFirst beforeLast
    by_cases beforeIndex : position ≤ index
    · exact firstProperties.2 position (Finset.mem_Icc.mpr ⟨afterFirst, beforeIndex⟩)
    · exact lastProperties.2.2 position
        (Finset.mem_Icc.mpr ⟨by omega, beforeLast⟩)
  · by_cases atStart : first = 0
    · exact Or.inl atStart
    · right
      intro sameValue
      have previousMember : first - 1 ∈ starts := by
        refine Finset.mem_filter.mpr ⟨Finset.mem_range.mpr (by omega), ?_⟩
        intro position member
        have bounds := Finset.mem_Icc.mp member
        by_cases previous : position = first - 1
        · rw [previous]
          exact sameValue
        · exact firstProperties.2 position (Finset.mem_Icc.mpr ⟨by omega, bounds.2⟩)
      have minimal : first ≤ first - 1 := Finset.min'_le starts _ previousMember
      omega
  · by_cases atEnd : last + 1 = rows
    · exact Or.inl atEnd
    · right
      intro sameValue
      have nextMember : last + 1 ∈ ends := by
        refine Finset.mem_filter.mpr ⟨Finset.mem_range.mpr (by omega), ?_, ?_⟩
        · omega
        · intro position member
          have bounds := Finset.mem_Icc.mp member
          by_cases next : position = last + 1
          · rw [next]
            exact sameValue
          · exact lastProperties.2.2 position
              (Finset.mem_Icc.mpr ⟨bounds.1, by omega⟩)
      have maximal : last + 1 ≤ last := Finset.le_max' ends _ nextMember
      omega

theorem unique_boundary_extremum_constant_offset (own adjacent : Nat -> Int)
    (rows center : Nat)
    (ownSteps : ∀ index, index + 1 < rows →
      own (index + 1) - own index = 1 ∨ own (index + 1) - own index = -1)
    (adjacentSteps : ∀ index, index + 1 < rows →
      adjacent (index + 1) - adjacent index = 1 ∨
        adjacent (index + 1) - adjacent index = -1)
    (offsets : ∀ index, index < rows →
      adjacent index - own index = 1 ∨ adjacent index - own index = -1)
    (uniqueExtremum : ∀ index, index < rows →
      (isMaxRow (rows - 1) own adjacent index ∨
        isMinRow (rows - 1) own adjacent index) → index = center) :
    ∀ index, index < rows →
      adjacent index - own index = adjacent center - own center := by
  intro index indexBound
  obtain ⟨first, last, firstBound, lastBound, lastInside,
    constantBlock, maximalBefore, maximalAfter⟩ :=
    maximal_constant_block (fun position => adjacent position - own position)
      rows index indexBound
  obtain ⟨extremum, afterFirst, beforeLast, isExtremum⟩ :=
    boundary_block_contains_extremum own adjacent rows first last (by omega)
      lastInside ownSteps adjacentSteps offsets (adjacent index - own index)
      constantBlock maximalBefore maximalAfter
  have sameCenter := uniqueExtremum extremum (by omega) isExtremum
  have sameOffset := constantBlock extremum afterFirst beforeLast
  rw [sameCenter] at sameOffset
  exact sameOffset.symm

def heightColumn {rows width : Nat} (height : OrigamiCone.Cell rows width -> Int)
    (column : Fin width) (index : Nat) : Int :=
  if bound : index < rows then height (⟨index, bound⟩, column) else 0

theorem heightColumn_steps {rows width : Nat}
    (height : OrigamiCone.Cell rows width -> Int) (heightValid : OrigamiCone.IsHeight height)
    (column : Fin width) :
    ∀ index, index + 1 < rows →
      heightColumn height column (index + 1) - heightColumn height column index = 1 ∨
        heightColumn height column (index + 1) - heightColumn height column index = -1 := by
  intro index nextBound
  have currentBound : index < rows := by omega
  have adjacentCells : OrigamiCone.adj
      ((⟨index, currentBound⟩, column) : OrigamiCone.Cell rows width)
      (⟨index + 1, nextBound⟩, column) := by
    unfold OrigamiCone.adj OrigamiCone.gdist
    dsimp only
    omega
  have unit := heightValid _ _ adjacentCells
  simp only [heightColumn, dif_pos nextBound, dif_pos currentBound]
  rcases abs_cases (height (⟨index, currentBound⟩, column) -
      height (⟨index + 1, nextBound⟩, column)) with ⟨identity, sign⟩ | ⟨identity, sign⟩ <;>
    omega

theorem heightColumn_offsets {rows width : Nat}
    (height : OrigamiCone.Cell rows width -> Int) (heightValid : OrigamiCone.IsHeight height)
    (column adjacent : Fin width)
    (columnsAdjacent : |(column.val : Int) - adjacent.val| = 1) :
    ∀ index, index < rows →
      heightColumn height adjacent index - heightColumn height column index = 1 ∨
        heightColumn height adjacent index - heightColumn height column index = -1 := by
  intro index bound
  have adjacentCells : OrigamiCone.adj
      ((⟨index, bound⟩, column) : OrigamiCone.Cell rows width)
      (⟨index, bound⟩, adjacent) := by
    unfold OrigamiCone.adj OrigamiCone.gdist
    simpa only [sub_self, Int.natAbs_zero, Nat.zero_add, Int.natCast_natAbs]
      using columnsAdjacent
  have unit := heightValid _ _ adjacentCells
  simp only [heightColumn, dif_pos bound]
  rcases abs_cases (height (⟨index, bound⟩, column) -
      height (⟨index, bound⟩, adjacent)) with ⟨identity, sign⟩ | ⟨identity, sign⟩ <;>
    omega

theorem first_boundary_row_extremum {rows width : Nat}
    (height : OrigamiCone.Cell rows width -> Int) (heightValid : OrigamiCone.IsHeight height)
    (widthValid : 2 ≤ width) (index : Nat) (indexBound : index < rows)
    (rowExtremum :
      isMaxRow (rows - 1) (heightColumn height ⟨0, by omega⟩)
        (heightColumn height ⟨1, by omega⟩) index ∨
      isMinRow (rows - 1) (heightColumn height ⟨0, by omega⟩)
        (heightColumn height ⟨1, by omega⟩) index) :
    OrigamiCone.IsStrictLocalExtremum height (⟨index, indexBound⟩, ⟨0, by omega⟩) := by
  have before : 0 < index →
      heightColumn height ⟨0, by omega⟩ (index - 1) =
        heightColumn height ⟨1, by omega⟩ index := by
    intro positive
    rcases rowExtremum with maximum | minimum
    · exact (maximum.1 positive).trans maximum.2.2.symm
    · exact (minimum.1 positive).trans minimum.2.2.symm
  have after : index + 1 < rows →
      heightColumn height ⟨0, by omega⟩ (index + 1) =
        heightColumn height ⟨1, by omega⟩ index := by
    intro nextBound
    rcases rowExtremum with maximum | minimum
    · exact (maximum.2.1 (by omega)).trans maximum.2.2.symm
    · exact (minimum.2.1 (by omega)).trans minimum.2.2.symm
  have allNeighbors : ∀ neighbor,
      OrigamiCone.adj ((⟨index, indexBound⟩, ⟨0, by omega⟩) : OrigamiCone.Cell rows width)
        neighbor → height neighbor = height (⟨index, indexBound⟩, ⟨1, by omega⟩) := by
    intro neighbor near
    rcases adj_col0_cases ⟨index, indexBound⟩ (by omega) neighbor near with
      ⟨next, rfl⟩ | ⟨previous, rfl⟩ | ⟨horizontal, rfl⟩
    · change index + 1 < rows at next
      have identity := after next
      simpa only [heightColumn, dif_pos next, dif_pos indexBound] using identity
    · change 0 < index at previous
      have previousBound : index - 1 < rows := by omega
      have identity := before previous
      simpa only [heightColumn, dif_pos previousBound, dif_pos indexBound] using identity
    · rfl
  have offset := heightColumn_offsets height heightValid
    ⟨0, by omega⟩ ⟨1, by omega⟩ (by norm_num) index indexBound
  simp only [heightColumn, dif_pos indexBound] at offset
  rcases offset with positive | negative
  · right
    intro neighbor near
    rw [allNeighbors neighbor near]
    omega
  · left
    intro neighbor near
    rw [allNeighbors neighbor near]
    omega

theorem unique_first_boundary_frozen {rows width : Nat}
    (height : OrigamiCone.Cell rows width -> Int) (heightValid : OrigamiCone.IsHeight height)
    (widthValid : 2 ≤ width)
    (oneExtremum : (Finset.univ.filter (fun row : Fin rows =>
      OrigamiCone.IsStrictLocalExtremum height (row, ⟨0, by omega⟩))).card = 1) :
    ∃ slope : Int, (slope = 1 ∨ slope = -1) ∧
      ∀ row : Fin rows, height (row, ⟨1, by omega⟩) =
        height (row, ⟨0, by omega⟩) + slope := by
  obtain ⟨center, singleton⟩ := Finset.card_eq_one.mp oneExtremum
  let own := heightColumn height (⟨0, by omega⟩ : Fin width)
  let adjacent := heightColumn height (⟨1, by omega⟩ : Fin width)
  have offsets := heightColumn_offsets height heightValid
    (⟨0, by omega⟩ : Fin width) ⟨1, by omega⟩ (by norm_num)
  have uniqueExtremum : ∀ index, index < rows →
      (isMaxRow (rows - 1) own adjacent index ∨
        isMinRow (rows - 1) own adjacent index) → index = center.val := by
    intro index indexBound rowExtremum
    have actual := first_boundary_row_extremum height heightValid widthValid
      index indexBound rowExtremum
    have member : (⟨index, indexBound⟩ : Fin rows) ∈
        Finset.univ.filter (fun row : Fin rows =>
          OrigamiCone.IsStrictLocalExtremum height (row, ⟨0, by omega⟩)) :=
      Finset.mem_filter.mpr ⟨Finset.mem_univ _, actual⟩
    rw [singleton] at member
    exact congrArg Fin.val (Finset.mem_singleton.mp member)
  have sameOffsets := unique_boundary_extremum_constant_offset own adjacent rows center.val
    (heightColumn_steps height heightValid _) (heightColumn_steps height heightValid _)
    offsets uniqueExtremum
  refine ⟨adjacent center.val - own center.val, offsets center.val center.isLt, ?_⟩
  intro row
  have equality := sameOffsets row.val row.isLt
  dsimp only [own, adjacent] at equality ⊢
  simp only [heightColumn, dif_pos row.isLt, dif_pos center.isLt] at equality ⊢
  linear_combination equality

theorem columnReflection_extremum {rows width : Nat}
    (height : OrigamiCone.Cell rows width -> Int) (point : OrigamiCone.Cell rows width) :
    OrigamiCone.IsStrictLocalExtremum (fun cell => height (OrigamiCone.colRefl cell)) point ↔
      OrigamiCone.IsStrictLocalExtremum height (OrigamiCone.colRefl point) := by
  have transport : ∀ difference : Int,
      (∀ neighbor, OrigamiCone.adj point neighbor →
        height (OrigamiCone.colRefl neighbor) = height (OrigamiCone.colRefl point) + difference) ↔
      (∀ neighbor, OrigamiCone.adj (OrigamiCone.colRefl point) neighbor →
        height neighbor = height (OrigamiCone.colRefl point) + difference) := by
    intro difference
    constructor
    · intro before neighbor near
      have reflectedNear : OrigamiCone.adj point (OrigamiCone.colRefl neighbor) := by
        apply (OrigamiCone.colRefl_adj point (OrigamiCone.colRefl neighbor)).mp
        rw [OrigamiCone.colRefl_involutive]
        exact near
      have identity := before (OrigamiCone.colRefl neighbor) reflectedNear
      simpa only [OrigamiCone.colRefl_involutive] using identity
    · intro after neighbor near
      exact after (OrigamiCone.colRefl neighbor)
        ((OrigamiCone.colRefl_adj point neighbor).mpr near)
  unfold OrigamiCone.IsStrictLocalExtremum OrigamiCone.IsStrictLocalMax
    OrigamiCone.IsStrictLocalMin
  simp only [sub_eq_add_neg]
  exact or_congr (transport (-1)) (transport 1)

theorem unique_last_boundary_frozen {rows width : Nat}
    (height : OrigamiCone.Cell rows width -> Int) (heightValid : OrigamiCone.IsHeight height)
    (widthValid : 2 ≤ width)
    (oneExtremum : (Finset.univ.filter (fun row : Fin rows =>
      OrigamiCone.IsStrictLocalExtremum height (row, ⟨width - 1, by omega⟩))).card = 1) :
    ∃ slope : Int, (slope = 1 ∨ slope = -1) ∧
      ∀ row : Fin rows, height (row, ⟨width - 2, by omega⟩) =
        height (row, ⟨width - 1, by omega⟩) + slope := by
  let reflected := fun cell : OrigamiCone.Cell rows width => height (OrigamiCone.colRefl cell)
  have reflectedValid : OrigamiCone.IsHeight reflected := by
    intro first second near
    exact heightValid (OrigamiCone.colRefl first) (OrigamiCone.colRefl second)
      ((OrigamiCone.colRefl_adj first second).mpr near)
  have zeroReversed : Fin.rev (⟨0, by omega⟩ : Fin width) = ⟨width - 1, by omega⟩ := by
    apply Fin.ext
    change width - (0 + 1) = width - 1
    omega
  have oneReversed : Fin.rev (⟨1, by omega⟩ : Fin width) = ⟨width - 2, by omega⟩ := by
    apply Fin.ext
    change width - (1 + 1) = width - 2
    omega
  have sameExtrema : (Finset.univ.filter (fun row : Fin rows =>
      OrigamiCone.IsStrictLocalExtremum reflected (row, ⟨0, by omega⟩))) =
      Finset.univ.filter (fun row : Fin rows =>
        OrigamiCone.IsStrictLocalExtremum height (row, ⟨width - 1, by omega⟩)) := by
    ext row
    simp only [Finset.mem_filter, Finset.mem_univ, true_and, reflected,
      columnReflection_extremum]
    change OrigamiCone.IsStrictLocalExtremum height
      (row, Fin.rev (⟨0, by omega⟩ : Fin width)) ↔ _
    rw [zeroReversed]
  have reflectedOne : (Finset.univ.filter (fun row : Fin rows =>
      OrigamiCone.IsStrictLocalExtremum reflected (row, ⟨0, by omega⟩))).card = 1 := by
    rw [sameExtrema]
    exact oneExtremum
  obtain ⟨slope, slopeUnit, slopeValues⟩ :=
    unique_first_boundary_frozen reflected reflectedValid widthValid reflectedOne
  refine ⟨slope, slopeUnit, ?_⟩
  intro row
  have identity := slopeValues row
  change height (row, Fin.rev (⟨1, by omega⟩ : Fin width)) =
    height (row, Fin.rev (⟨0, by omega⟩ : Fin width)) + slope at identity
  rw [zeroReversed, oneReversed] at identity
  exact identity

#print axioms unit_offset_change
#print axioms boundary_block_contains_extremum
#print axioms maximal_constant_block
#print axioms unique_boundary_extremum_constant_offset
#print axioms heightColumn_steps
#print axioms heightColumn_offsets
#print axioms first_boundary_row_extremum
#print axioms unique_first_boundary_frozen
#print axioms columnReflection_extremum
#print axioms unique_last_boundary_frozen

end OrigamiCone.Sequel.ConcreteTransfer
