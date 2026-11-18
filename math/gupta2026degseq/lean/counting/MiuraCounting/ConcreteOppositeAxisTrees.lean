import MiuraCounting.ConcreteSameAxisTrees

namespace OrigamiCone.Sequel.OppositeAxisTrees

open OrigamiCone FrozenExpansion OccupiedBlocks

theorem step_toward_first_stays_between {rows width : Nat}
    (first last point neighbor : Cell rows width)
    (between : gdist first point + gdist point last = gdist first last)
    (near : adj point neighbor)
    (closer : gdist neighbor first = gdist point first - 1) :
    gdist first neighbor + gdist neighbor last = gdist first last := by
  have lower := gdist_triangle first neighbor last
  have upper := gdist_triangle neighbor point last
  have firstSymmetry := gdist_comm first point
  have neighborSymmetry := gdist_comm first neighbor
  have edgeSymmetry := gdist_comm neighbor point
  change gdist point neighbor = 1 at near
  omega

theorem saturated_geodesic_extremum_endpoints {rows width : Nat}
    (height : Cell rows width -> Int) (heightValid : IsHeight height)
    (first last point : Cell rows width) (sign : Int)
    (unitSign : sign = 1 ∨ sign = -1)
    (saturated : height last - height first = sign * gdist first last)
    (between : gdist first point + gdist point last = gdist first last)
    (extremum : IsStrictLocalExtremum height point) : point = first ∨ point = last := by
  by_contra neither
  have notFirst : point ≠ first := fun equality => neither (Or.inl equality)
  have notLast : point ≠ last := fun equality => neither (Or.inr equality)
  obtain ⟨before, beforeNear, beforeDistance⟩ := exists_step_toward notFirst
  obtain ⟨after, afterNear, afterDistance⟩ := exists_step_toward notLast
  have beforeBetween := step_toward_first_stays_between first last point before
    between beforeNear beforeDistance
  have reverseBetween : gdist last point + gdist point first = gdist last first := by
    rw [gdist_comm last point, gdist_comm point first, gdist_comm last first]
    omega
  have afterReverse := step_toward_first_stays_between last first point after
    reverseBetween afterNear afterDistance
  have afterBetween : gdist first after + gdist after last = gdist first last := by
    rw [gdist_comm first after, gdist_comm after last, gdist_comm first last]
    omega
  have pointValue := Saturation.between_saturated_signed height heightValid
    first last point sign unitSign saturated between
  have beforeValue := Saturation.between_saturated_signed height heightValid
    first last before sign unitSign saturated beforeBetween
  have afterValue := Saturation.between_saturated_signed height heightValid
    first last after sign unitSign saturated afterBetween
  have beforeFirst : gdist first before = gdist first point - 1 := by
    rw [gdist_comm first before, gdist_comm first point]
    exact beforeDistance
  have afterFirst : gdist first after = gdist first point + 1 := by
    omega
  rw [beforeFirst] at beforeValue
  rw [afterFirst] at afterValue
  rcases extremum with maximum | minimum
  · have beforeStep := maximum before beforeNear
    have afterStep := maximum after afterNear
    rcases unitSign with signCase | signCase <;>
      rw [signCase] at pointValue beforeValue afterValue <;> nlinarith
  · have beforeStep := minimum before beforeNear
    have afterStep := minimum after afterNear
    rcases unitSign with signCase | signCase <;>
      rw [signCase] at pointValue beforeValue afterValue <;> nlinarith

theorem separated_leaf_rectangle_extremum {rows width : Nat}
    (height : Cell rows width -> Int) (heightValid : IsHeight height)
    (rowsValid : 2 ≤ rows) (widthValid : 2 ≤ width)
    (rowPoint columnPoint point : Extrema height)
    (rowLeaf : (OccupiedBlocks.graph height).degree (BlockGraph.Vertex.rowBlock (rowOf height rowPoint)) = 1)
    (columnLeaf : (OccupiedBlocks.graph height).degree (BlockGraph.Vertex.columnBlock (columnOf height columnPoint)) = 1)
    (disconnected : ¬ (OccupiedBlocks.graph height).Reachable
      (BlockGraph.Vertex.extremum rowPoint) (BlockGraph.Vertex.extremum columnPoint))
    (rowBetween : min rowPoint.val.1.val columnPoint.val.1.val ≤ point.val.1.val ∧
      point.val.1.val ≤ max rowPoint.val.1.val columnPoint.val.1.val)
    (columnBetween : min rowPoint.val.2.val columnPoint.val.2.val ≤ point.val.2.val ∧
      point.val.2.val ≤ max rowPoint.val.2.val columnPoint.val.2.val) :
    point = rowPoint ∨ point = columnPoint := by
  have saturated := separated_mixed_leaf_saturated height heightValid rowsValid widthValid
    rowPoint columnPoint rowLeaf columnLeaf disconnected
  have between := Saturation.rectangle_geodesic rowPoint.val columnPoint.val point.val rowBetween columnBetween
  rcases saturated_geodesic_extremum_endpoints height heightValid rowPoint.val columnPoint.val point.val
      (extremumSign height rowPoint) (extremumSign_unit height rowPoint) saturated between point.property with
      atRow | atColumn
  · exact Or.inl (Subtype.ext atRow)
  · exact Or.inr (Subtype.ext atColumn)

theorem additional_extremum_strictly_inside {rows width : Nat}
    (height : Cell rows width -> Int) (heightValid : IsHeight height)
    (rowsValid : 2 ≤ rows) (widthValid : 2 ≤ width)
    (top bottom left right point : Extrema height)
    (topBoundary : top.val.1.val = 0) (bottomBoundary : bottom.val.1.val + 1 = rows)
    (leftBoundary : left.val.2.val = 0) (rightBoundary : right.val.2.val + 1 = width)
    (topLeaf : (OccupiedBlocks.graph height).degree (BlockGraph.Vertex.rowBlock (rowOf height top)) = 1)
    (bottomLeaf : (OccupiedBlocks.graph height).degree (BlockGraph.Vertex.rowBlock (rowOf height bottom)) = 1)
    (leftLeaf : (OccupiedBlocks.graph height).degree (BlockGraph.Vertex.columnBlock (columnOf height left)) = 1)
    (rightLeaf : (OccupiedBlocks.graph height).degree (BlockGraph.Vertex.columnBlock (columnOf height right)) = 1)
    (topLeft : ¬ (OccupiedBlocks.graph height).Reachable (BlockGraph.Vertex.extremum top) (BlockGraph.Vertex.extremum left))
    (topRight : ¬ (OccupiedBlocks.graph height).Reachable (BlockGraph.Vertex.extremum top) (BlockGraph.Vertex.extremum right))
    (bottomLeft : ¬ (OccupiedBlocks.graph height).Reachable (BlockGraph.Vertex.extremum bottom) (BlockGraph.Vertex.extremum left))
    (bottomRight : ¬ (OccupiedBlocks.graph height).Reachable (BlockGraph.Vertex.extremum bottom) (BlockGraph.Vertex.extremum right))
    (notTop : point ≠ top) (notBottom : point ≠ bottom)
    (notLeft : point ≠ left) (notRight : point ≠ right) :
    (min left.val.1.val right.val.1.val < point.val.1.val ∧
      point.val.1.val < max left.val.1.val right.val.1.val) ∧
    (min top.val.2.val bottom.val.2.val < point.val.2.val ∧
      point.val.2.val < max top.val.2.val bottom.val.2.val) := by
  have pointRowBound := point.val.1.isLt
  have pointColumnBound := point.val.2.isLt
  have outsideTopLeft : ¬ (point.val.1.val ≤ left.val.1.val ∧ point.val.2.val ≤ top.val.2.val) := by
    rintro ⟨rowBound, columnBound⟩
    rcases separated_leaf_rectangle_extremum height heightValid rowsValid widthValid top left point
        topLeaf leftLeaf topLeft (by omega) (by omega) with sameTop | sameLeft
    · exact notTop sameTop
    · exact notLeft sameLeft
  have outsideTopRight : ¬ (point.val.1.val ≤ right.val.1.val ∧ top.val.2.val ≤ point.val.2.val) := by
    rintro ⟨rowBound, columnBound⟩
    rcases separated_leaf_rectangle_extremum height heightValid rowsValid widthValid top right point
        topLeaf rightLeaf topRight (by omega) (by omega) with sameTop | sameRight
    · exact notTop sameTop
    · exact notRight sameRight
  have outsideBottomLeft : ¬ (left.val.1.val ≤ point.val.1.val ∧ point.val.2.val ≤ bottom.val.2.val) := by
    rintro ⟨rowBound, columnBound⟩
    rcases separated_leaf_rectangle_extremum height heightValid rowsValid widthValid bottom left point
        bottomLeaf leftLeaf bottomLeft (by omega) (by omega) with sameBottom | sameLeft
    · exact notBottom sameBottom
    · exact notLeft sameLeft
  have outsideBottomRight : ¬ (right.val.1.val ≤ point.val.1.val ∧ bottom.val.2.val ≤ point.val.2.val) := by
    rintro ⟨rowBound, columnBound⟩
    rcases separated_leaf_rectangle_extremum height heightValid rowsValid widthValid bottom right point
        bottomLeaf rightLeaf bottomRight (by omega) (by omega) with sameBottom | sameRight
    · exact notBottom sameBottom
    · exact notRight sameRight
  omega

theorem blockRank_eq_between {rows width : Nat} (height : Cell rows width -> Int)
    (first last column : Fin width) (sameRank : blockRank height first = blockRank height last)
    (between : min first.val last.val ≤ column.val ∧ column.val ≤ max first.val last.val) :
    blockRank height column = blockRank height first := by
  apply Fin.ext
  have ranks := congrArg Fin.val sameRank
  change RunLengthFibers.runPrefixCount height first = RunLengthFibers.runPrefixCount height last at ranks
  change RunLengthFibers.runPrefixCount height column = RunLengthFibers.runPrefixCount height first
  rcases Nat.le_total first.val last.val with ordered | reverse
  · have lower := RunLengthFibers.runPrefixCount_mono height (first := first) (second := column) (by omega)
    have upper := RunLengthFibers.runPrefixCount_mono height (first := column) (second := last) (by omega)
    omega
  · have lower := RunLengthFibers.runPrefixCount_mono height (first := last) (second := column) (by omega)
    have upper := RunLengthFibers.runPrefixCount_mono height (first := column) (second := first) (by omega)
    omega

theorem opposite_boundary_leaves_cover_extrema {rows width : Nat}
    (height : Cell rows width -> Int) (heightValid : IsHeight height)
    (rowsValid : 2 ≤ rows) (widthValid : 2 ≤ width)
    (top bottom left right : Extrema height)
    (topBoundary : top.val.1.val = 0) (bottomBoundary : bottom.val.1.val + 1 = rows)
    (leftBoundary : left.val.2.val = 0) (rightBoundary : right.val.2.val + 1 = width)
    (topLeaf : (OccupiedBlocks.graph height).degree (BlockGraph.Vertex.rowBlock (rowOf height top)) = 1)
    (bottomLeaf : (OccupiedBlocks.graph height).degree (BlockGraph.Vertex.rowBlock (rowOf height bottom)) = 1)
    (leftLeaf : (OccupiedBlocks.graph height).degree (BlockGraph.Vertex.columnBlock (columnOf height left)) = 1)
    (rightLeaf : (OccupiedBlocks.graph height).degree (BlockGraph.Vertex.columnBlock (columnOf height right)) = 1)
    (topLeft : ¬ (OccupiedBlocks.graph height).Reachable (BlockGraph.Vertex.extremum top) (BlockGraph.Vertex.extremum left))
    (topRight : ¬ (OccupiedBlocks.graph height).Reachable (BlockGraph.Vertex.extremum top) (BlockGraph.Vertex.extremum right))
    (bottomLeft : ¬ (OccupiedBlocks.graph height).Reachable (BlockGraph.Vertex.extremum bottom) (BlockGraph.Vertex.extremum left))
    (bottomRight : ¬ (OccupiedBlocks.graph height).Reachable (BlockGraph.Vertex.extremum bottom) (BlockGraph.Vertex.extremum right)) :
    ∀ point : Extrema height, point = top ∨ point = bottom ∨ point = left ∨ point = right := by
  have topLeftSign := separated_mixed_leaves_opposite_sign height heightValid rowsValid widthValid
    top left topLeaf leftLeaf topLeft
  have topRightSign := separated_mixed_leaves_opposite_sign height heightValid rowsValid widthValid
    top right topLeaf rightLeaf topRight
  have bottomLeftSign := separated_mixed_leaves_opposite_sign height heightValid rowsValid widthValid
    bottom left bottomLeaf leftLeaf bottomLeft
  have rowSigns : extremumSign height top = extremumSign height bottom :=
    topLeftSign.trans bottomLeftSign.symm
  have columnSigns : extremumSign height left = extremumSign height right := by omega
  have topProfile := (row_leaf_fixed_sign_profile height heightValid rowsValid top topLeaf).1
  have bottomProfile := (row_leaf_fixed_sign_profile height heightValid rowsValid bottom bottomLeaf).1
  have sameColumnBlock : columnOf height top = columnOf height bottom :=
    same_sign_profiles_same_block height heightValid (by omega) widthValid top.val.1 bottom.val.1
      top.val.2 bottom.val.2 (extremumSign height top) (extremumSign_unit height top) topProfile
      (by simpa only [rowSigns] using bottomProfile)
      (extremum_column_active height top) (extremum_column_active height bottom)
  have leftProfile := (column_leaf_fixed_sign_profile height heightValid widthValid left leftLeaf).1
  have rightProfile := (column_leaf_fixed_sign_profile height heightValid widthValid right rightLeaf).1
  have sameRowBlock : rowOf height left = rowOf height right :=
    same_sign_profiles_same_block (swapH height) (swapH_isHeight heightValid) (by omega) rowsValid
      left.val.2 right.val.2 left.val.1 right.val.1 (extremumSign height left)
      (extremumSign_unit height left) leftProfile (by simpa only [columnSigns] using rightProfile)
      (extremum_row_active height left) (extremum_row_active height right)
  intro point
  by_contra additional
  have notTop : point ≠ top := fun equality => additional (Or.inl equality)
  have notBottom : point ≠ bottom := fun equality => additional (Or.inr (Or.inl equality))
  have notLeft : point ≠ left := fun equality => additional (Or.inr (Or.inr (Or.inl equality)))
  have notRight : point ≠ right := fun equality => additional (Or.inr (Or.inr (Or.inr equality)))
  have inside := additional_extremum_strictly_inside height heightValid rowsValid widthValid
    top bottom left right point topBoundary bottomBoundary leftBoundary rightBoundary
    topLeaf bottomLeaf leftLeaf rightLeaf topLeft topRight bottomLeft bottomRight
    notTop notBottom notLeft notRight
  have pointRowBlock : rowOf height point = rowOf height left :=
    blockRank_eq_between (swapH height) left.val.1 right.val.1 point.val.1 sameRowBlock
      ⟨Nat.le_of_lt inside.1.1, Nat.le_of_lt inside.1.2⟩
  have pointColumnBlock : columnOf height point = columnOf height top :=
    blockRank_eq_between height top.val.2 bottom.val.2 point.val.2 sameColumnBlock
      ⟨Nat.le_of_lt inside.2.1, Nat.le_of_lt inside.2.2⟩
  apply topLeft
  exact (extrema_reachable_of_same_columnBlock height top point pointColumnBlock.symm).trans
    (extrema_reachable_of_same_rowBlock height point left pointRowBlock)

theorem card_le_four_of_cover {Point : Type*} [Fintype Point]
    (first second third fourth : Point)
    (cover : ∀ point, point = first ∨ point = second ∨ point = third ∨ point = fourth) :
    Fintype.card Point ≤ 4 := by
  classical
  have subset : (Finset.univ : Finset Point) ⊆ {first, second, third, fourth} := by
    intro point _
    simpa only [Finset.mem_insert, Finset.mem_singleton] using cover point
  have total := Finset.card_le_card subset
  have firstBound := Finset.card_insert_le first ({second, third, fourth} : Finset Point)
  have secondBound := Finset.card_insert_le second ({third, fourth} : Finset Point)
  have thirdBound := Finset.card_insert_le third ({fourth} : Finset Point)
  simp only [Finset.card_univ, Finset.card_singleton] at total thirdBound
  omega

theorem opposite_leaf_pairs_card_le_four {rows width : Nat}
    (height : Cell rows width -> Int) (heightValid : IsHeight height)
    (rowsValid : 2 ≤ rows) (widthValid : 2 ≤ width)
    (firstRow secondRow firstColumn secondColumn : Extrema height)
    (differentRows : firstRow ≠ secondRow) (differentColumns : firstColumn ≠ secondColumn)
    (firstRowLeaf : (OccupiedBlocks.graph height).degree (BlockGraph.Vertex.rowBlock (rowOf height firstRow)) = 1)
    (secondRowLeaf : (OccupiedBlocks.graph height).degree (BlockGraph.Vertex.rowBlock (rowOf height secondRow)) = 1)
    (firstColumnLeaf : (OccupiedBlocks.graph height).degree (BlockGraph.Vertex.columnBlock (columnOf height firstColumn)) = 1)
    (secondColumnLeaf : (OccupiedBlocks.graph height).degree (BlockGraph.Vertex.columnBlock (columnOf height secondColumn)) = 1)
    (firstFirst : ¬ (OccupiedBlocks.graph height).Reachable (BlockGraph.Vertex.extremum firstRow) (BlockGraph.Vertex.extremum firstColumn))
    (firstSecond : ¬ (OccupiedBlocks.graph height).Reachable (BlockGraph.Vertex.extremum firstRow) (BlockGraph.Vertex.extremum secondColumn))
    (secondFirst : ¬ (OccupiedBlocks.graph height).Reachable (BlockGraph.Vertex.extremum secondRow) (BlockGraph.Vertex.extremum firstColumn))
    (secondSecond : ¬ (OccupiedBlocks.graph height).Reachable (BlockGraph.Vertex.extremum secondRow) (BlockGraph.Vertex.extremum secondColumn)) :
    numExtrema height ≤ 4 := by
  classical
  have firstBounds := separated_mixed_leaf_boundaries height heightValid rowsValid widthValid
    firstRow firstColumn firstRowLeaf firstColumnLeaf firstFirst
  have secondBounds := separated_mixed_leaf_boundaries height heightValid rowsValid widthValid
    secondRow secondColumn secondRowLeaf secondColumnLeaf secondSecond
  have rowValuesDifferent : firstRow.val.1.val ≠ secondRow.val.1.val := by
    intro equality
    have sameRowBlock : rowOf height secondRow = rowOf height firstRow :=
      congrArg (blockRank (swapH height)) (Fin.ext equality.symm)
    exact differentRows (row_leaf_unique_extremum height firstRow firstRowLeaf secondRow sameRowBlock).symm
  have columnValuesDifferent : firstColumn.val.2.val ≠ secondColumn.val.2.val := by
    intro equality
    have sameColumnBlock : columnOf height secondColumn = columnOf height firstColumn :=
      congrArg (blockRank height) (Fin.ext equality.symm)
    exact differentColumns
      (column_leaf_unique_extremum height firstColumn firstColumnLeaf secondColumn sameColumnBlock).symm
  rw [← card_extrema height]
  rcases firstBounds.1 with firstTop | firstBottom <;>
    rcases firstBounds.2 with firstLeft | firstRight
  · have secondBottom : secondRow.val.1.val + 1 = rows := by omega
    have secondRight : secondColumn.val.2.val + 1 = width := by omega
    exact card_le_four_of_cover firstRow secondRow firstColumn secondColumn
      (opposite_boundary_leaves_cover_extrema height heightValid rowsValid widthValid
        firstRow secondRow firstColumn secondColumn firstTop secondBottom firstLeft secondRight
        firstRowLeaf secondRowLeaf firstColumnLeaf secondColumnLeaf firstFirst firstSecond secondFirst secondSecond)
  · have secondBottom : secondRow.val.1.val + 1 = rows := by omega
    have secondLeft : secondColumn.val.2.val = 0 := by omega
    exact card_le_four_of_cover firstRow secondRow secondColumn firstColumn
      (opposite_boundary_leaves_cover_extrema height heightValid rowsValid widthValid
        firstRow secondRow secondColumn firstColumn firstTop secondBottom secondLeft firstRight
        firstRowLeaf secondRowLeaf secondColumnLeaf firstColumnLeaf firstSecond firstFirst secondSecond secondFirst)
  · have secondTop : secondRow.val.1.val = 0 := by omega
    have secondRight : secondColumn.val.2.val + 1 = width := by omega
    exact card_le_four_of_cover secondRow firstRow firstColumn secondColumn
      (opposite_boundary_leaves_cover_extrema height heightValid rowsValid widthValid
        secondRow firstRow firstColumn secondColumn secondTop firstBottom firstLeft secondRight
        secondRowLeaf firstRowLeaf firstColumnLeaf secondColumnLeaf secondFirst secondSecond firstFirst firstSecond)
  · have secondTop : secondRow.val.1.val = 0 := by omega
    have secondLeft : secondColumn.val.2.val = 0 := by omega
    exact card_le_four_of_cover secondRow firstRow secondColumn firstColumn
      (opposite_boundary_leaves_cover_extrema height heightValid rowsValid widthValid
        secondRow firstRow secondColumn firstColumn secondTop firstBottom secondLeft firstRight
        secondRowLeaf firstRowLeaf secondColumnLeaf firstColumnLeaf secondSecond secondFirst firstSecond firstFirst)

#print axioms step_toward_first_stays_between
#print axioms saturated_geodesic_extremum_endpoints
#print axioms separated_leaf_rectangle_extremum
#print axioms additional_extremum_strictly_inside
#print axioms blockRank_eq_between
#print axioms opposite_boundary_leaves_cover_extrema
#print axioms card_le_four_of_cover
#print axioms opposite_leaf_pairs_card_le_four

end OrigamiCone.Sequel.OppositeAxisTrees
