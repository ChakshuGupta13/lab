import MiuraCounting.ConcreteOccupiedBlocks
import MiuraCounting.ConcreteSaturation

namespace OrigamiCone.Sequel.BoundaryComparison

open OrigamiCone

theorem line_neighbors_exhausted {rows : Nat} (center first second : Cell rows 1)
    (firstNear : adj center first) (secondNear : adj center second)
    (different : first ≠ second) :
    ∀ other : Cell rows 1, adj center other -> other = first ∨ other = second := by
  intro other otherNear
  by_cases same : other = first
  · exact Or.inl same
  · right
    apply Prod.ext
    · apply Fin.ext
      have differentRows : first.1.val ≠ second.1.val := by
        intro equality
        exact different (Prod.ext (Fin.ext equality) (Subsingleton.elim _ _))
      have otherRow : other.1.val ≠ first.1.val := by
        intro equality
        exact same (Prod.ext (Fin.ext equality) (Subsingleton.elim _ _))
      have centerColumn := center.2.isLt
      have firstColumn := first.2.isLt
      have secondColumn := second.2.isLt
      have otherColumn := other.2.isLt
      simp only [adj, gdist] at firstNear secondNear otherNear
      omega
    · exact Subsingleton.elim _ _

theorem equal_neighbor {rows : Nat} (leftWalk rightWalk : Cell rows 1 -> Int)
    (leftValid : IsHeight leftWalk) (rightValid : IsHeight rightWalk)
    (ordered : ∀ point, rightWalk point ≤ leftWalk point)
    (minimumCoupling : ∀ point, IsStrictLocalMin leftWalk point ->
      ∀ other, adj point other ->
        leftWalk other - leftWalk point = rightWalk other - rightWalk point)
    (maximumCoupling : ∀ point, IsStrictLocalMax rightWalk point ->
      ∀ other, adj point other ->
        leftWalk other - leftWalk point = rightWalk other - rightWalk point)
    (center neighbor : Cell rows 1) (same : leftWalk center = rightWalk center)
    (near : adj center neighbor) : leftWalk neighbor = rightWalk neighbor := by
  classical
  by_contra different
  have leftStep := (abs_eq (by norm_num : (0 : Int) ≤ 1)).mp
    (leftValid center neighbor near)
  have rightStep := (abs_eq (by norm_num : (0 : Int) ≤ 1)).mp
    (rightValid center neighbor near)
  have neighborOrder := ordered neighbor
  have rising : leftWalk neighbor = leftWalk center + 1 := by omega
  have falling : rightWalk neighbor = rightWalk center - 1 := by omega
  by_cases minimum : IsStrictLocalMin leftWalk center
  · have coupling := minimumCoupling center minimum neighbor near
    omega
  · simp only [IsStrictLocalMin, not_forall] at minimum
    obtain ⟨other, otherNear, notRising⟩ := minimum
    have otherLeftStep := (abs_eq (by norm_num : (0 : Int) ≤ 1)).mp
      (leftValid center other otherNear)
    have otherRightStep := (abs_eq (by norm_num : (0 : Int) ≤ 1)).mp
      (rightValid center other otherNear)
    have otherOrder := ordered other
    have otherFalling : rightWalk other = rightWalk center - 1 := by omega
    have distinct : neighbor ≠ other := by
      intro equality
      rw [← equality] at notRising
      exact notRising rising
    have maximum : IsStrictLocalMax rightWalk center := by
      intro next nextNear
      rcases line_neighbors_exhausted center neighbor other near otherNear distinct
          next nextNear with nextNeighbor | nextOther
      · rw [nextNeighbor]
        exact falling
      · rw [nextOther]
        exact otherFalling
    have coupling := maximumCoupling center maximum neighbor near
    omega

theorem ordered_walks_equal {rows : Nat} (leftWalk rightWalk : Cell rows 1 -> Int)
    (leftValid : IsHeight leftWalk) (rightValid : IsHeight rightWalk)
    (ordered : ∀ point, rightWalk point ≤ leftWalk point)
    (minimumCoupling : ∀ point, IsStrictLocalMin leftWalk point ->
      ∀ other, adj point other ->
        leftWalk other - leftWalk point = rightWalk other - rightWalk point)
    (maximumCoupling : ∀ point, IsStrictLocalMax rightWalk point ->
      ∀ other, adj point other ->
        leftWalk other - leftWalk point = rightWalk other - rightWalk point)
    (start : Cell rows 1) (startEqual : leftWalk start = rightWalk start) :
    leftWalk = rightWalk := by
  have propagate : ∀ budget : Nat, ∀ point, gdist point start ≤ budget ->
      leftWalk point = rightWalk point := by
    intro budget
    induction budget with
    | zero =>
      intro point bound
      have distanceZero : gdist point start = 0 := by
        have nonnegative := gdist_nonneg point start
        omega
      have pointEqual := gdist_eq_zero.mp distanceZero
      rw [pointEqual]
      exact startEqual
    | succ budget inductionHypothesis =>
      intro point bound
      by_cases pointEqual : point = start
      · rw [pointEqual]
        exact startEqual
      · obtain ⟨toward, near, distance⟩ := exists_step_toward pointEqual
        have towardEqual := inductionHypothesis toward (by omega)
        have reverseNear : adj toward point := by
          change gdist toward point = 1
          rw [gdist_comm]
          exact near
        exact equal_neighbor leftWalk rightWalk leftValid rightValid ordered
          minimumCoupling maximumCoupling toward point towardEqual reverseNear
  funext point
  exact propagate (gdist point start).toNat point (by
    have nonnegative := gdist_nonneg point start
    omega)

def columnLine {rows width : Nat} (height : Cell rows width -> Int)
    (column : Fin width) : Cell rows 1 -> Int := fun point => height (point.1, column)

theorem columnLine_isHeight {rows width : Nat} (height : Cell rows width -> Int)
    (heightValid : IsHeight height) (column : Fin width) :
    IsHeight (columnLine height column) := by
  intro first second near
  change |height (first.1, column) - height (second.1, column)| = 1
  apply heightValid
  have firstColumn : first.2.val = 0 := by have := first.2.isLt; omega
  have secondColumn : second.2.val = 0 := by have := second.2.isLt; omega
  simpa [adj, gdist, firstColumn, secondColumn] using near

theorem columnLine_increment_lift {rows width : Nat} (height : Cell rows width -> Int)
    (row : Fin rows) (column : Fin width) (step : Int)
    (vertical : ∀ other : Fin rows, adj ((row, 0) : Cell rows 1) (other, 0) ->
      height (other, column) = height (row, column) + step)
    (horizontal : ∀ other : Fin width, ((column.val : Int) - other.val).natAbs = 1 ->
      height (row, other) = height (row, column) + step) :
    ∀ point, adj (row, column) point -> height point = height (row, column) + step := by
  rintro ⟨otherRow, otherColumn⟩ near
  by_cases sameColumn : otherColumn = column
  · subst otherColumn
    exact vertical otherRow (by simpa [adj, gdist] using near)
  · have sameRow : otherRow = row := by
      apply Fin.ext
      have differentColumns : otherColumn.val ≠ column.val := by
        intro equality
        exact sameColumn (Fin.ext equality)
      simp only [adj, gdist] at near
      omega
    subst otherRow
    apply horizontal otherColumn
    simp only [adj, gdist] at near
    omega

theorem columnLine_minimum_lift {rows width : Nat} (height : Cell rows width -> Int)
    (row : Fin rows) (column : Fin width)
    (horizontal : ∀ other : Fin width, ((column.val : Int) - other.val).natAbs = 1 ->
      height (row, other) = height (row, column) + 1)
    (minimum : IsStrictLocalMin (columnLine height column) (row, 0)) :
    IsStrictLocalMin height (row, column) :=
  columnLine_increment_lift height row column 1
    (fun other near => minimum (other, 0) near) horizontal

theorem columnLine_maximum_lift {rows width : Nat} (height : Cell rows width -> Int)
    (row : Fin rows) (column : Fin width)
    (horizontal : ∀ other : Fin width, ((column.val : Int) - other.val).natAbs = 1 ->
      height (row, other) = height (row, column) - 1)
    (maximum : IsStrictLocalMax (columnLine height column) (row, 0)) :
    IsStrictLocalMax height (row, column) :=
  columnLine_increment_lift height row column (-1)
    (fun other near => maximum (other, 0) near) horizontal

theorem row_leaf_column_differences_equal {rows width : Nat}
    (height : Cell rows width -> Int) (heightValid : IsHeight height)
    (rowsValid : 2 ≤ rows) (point : OccupiedBlocks.Extrema height)
    (leaf : (OccupiedBlocks.graph height).degree
      (BlockGraph.Vertex.rowBlock (OccupiedBlocks.rowOf height point)) = 1)
    (first second : Fin width) (other : Fin rows)
    (near : ((point.val.1.val : Int) - other.val).natAbs = 1) :
    height (other, first) - height (point.val.1, first) =
      height (other, second) - height (point.val.1, second) := by
  have transverse := (OccupiedBlocks.row_leaf_fixed_sign_profile height heightValid
    rowsValid point leaf).2
  have firstIncrement := transverse first other near
  have secondIncrement := transverse second other near
  omega

theorem boundary_leaf_profiles_separable {rows width : Nat}
    (height : Cell rows width -> Int) (heightValid : IsHeight height)
    (rowsValid : 2 ≤ rows) (widthValid : 2 ≤ width)
    (firstInward : ∀ row : Fin rows, height (row, ⟨1, by omega⟩) =
      height (row, ⟨0, by omega⟩) + 1)
    (lastInward : ∀ row : Fin rows, height (row, ⟨width - 2, by omega⟩) =
      height (row, ⟨width - 1, by omega⟩) - 1)
    (firstLeaves : ∀ point : OccupiedBlocks.Extrema height, point.val.2.val = 0 ->
      (OccupiedBlocks.graph height).degree
        (BlockGraph.Vertex.rowBlock (OccupiedBlocks.rowOf height point)) = 1)
    (lastLeaves : ∀ point : OccupiedBlocks.Extrema height, point.val.2.val + 1 = width ->
      (OccupiedBlocks.graph height).degree
        (BlockGraph.Vertex.rowBlock (OccupiedBlocks.rowOf height point)) = 1) :
    FrozenExpansion.AdditivelySeparable height := by
  let first : Fin width := ⟨0, by omega⟩
  let last : Fin width := ⟨width - 1, by omega⟩
  let leftWalk := columnLine height first
  let rightWalk : Cell rows 1 -> Int := fun point =>
    height (point.1, last) - (width - 1 : Nat)
  have leftValid : IsHeight leftWalk := columnLine_isHeight height heightValid first
  have rightValid : IsHeight rightWalk := by
    intro center neighbor near
    have original := columnLine_isHeight height heightValid last center neighbor near
    change |(height (center.1, last) - (width - 1 : Nat)) -
      (height (neighbor.1, last) - (width - 1 : Nat))| = 1
    have cancel : (height (center.1, last) - (width - 1 : Nat)) -
        (height (neighbor.1, last) - (width - 1 : Nat)) =
        height (center.1, last) - height (neighbor.1, last) := by ring
    rw [cancel]
    exact original
  have ordered : ∀ point, rightWalk point ≤ leftWalk point := by
    intro point
    have bound := Saturation.height_distance_bound height heightValid
      (point.1, first) (point.1, last)
    have distance : gdist (point.1, first) (point.1, last) = (width - 1 : Nat) := by
      simp only [gdist]
      dsimp only [first, last]
      omega
    rw [distance] at bound
    have upper := (abs_le.mp bound).1
    change height (point.1, last) - (width - 1 : Nat) ≤ height (point.1, first)
    omega
  have liftMinimum : ∀ point, IsStrictLocalMin leftWalk point ->
      IsStrictLocalMin height (point.1, first) := by
    intro point minimum
    apply columnLine_minimum_lift height point.1 first
    · intro other near
      have otherBound := other.isLt
      have atOne : other = (⟨1, by omega⟩ : Fin width) := by
        apply Fin.ext
        change other.val = 1
        dsimp only [first] at near
        omega
      rw [atOne]
      exact firstInward point.1
    · change IsStrictLocalMin leftWalk (point.1, 0)
      have pointEqual : ((point.1, 0) : Cell rows 1) = point :=
        Prod.ext rfl (Subsingleton.elim _ _)
      rw [pointEqual]
      exact minimum
  have liftMaximum : ∀ point, IsStrictLocalMax rightWalk point ->
      IsStrictLocalMax height (point.1, last) := by
    intro point maximum
    apply columnLine_maximum_lift height point.1 last
    · intro other near
      have otherBound := other.isLt
      have atPrevious : other = (⟨width - 2, by omega⟩ : Fin width) := by
        apply Fin.ext
        change other.val = width - 2
        dsimp only [last] at near
        omega
      rw [atPrevious]
      exact lastInward point.1
    · intro neighbor near
      have pointEqual : ((point.1, 0) : Cell rows 1) = point :=
        Prod.ext rfl (Subsingleton.elim _ _)
      rw [pointEqual] at near
      have relation := maximum neighbor near
      change height (neighbor.1, last) - (width - 1 : Nat) =
        height (point.1, last) - (width - 1 : Nat) - 1 at relation
      change height (neighbor.1, last) = height (point.1, last) - 1
      omega
  have minimumCoupling : ∀ point, IsStrictLocalMin leftWalk point ->
      ∀ other, adj point other ->
        leftWalk other - leftWalk point = rightWalk other - rightWalk point := by
    intro point minimum other near
    let extremum : OccupiedBlocks.Extrema height :=
      ⟨(point.1, first), Or.inr (liftMinimum point minimum)⟩
    have leaf := firstLeaves extremum rfl
    have rowNear : ((point.1.val : Int) - other.1.val).natAbs = 1 := by
      have pointColumn := point.2.isLt
      have otherColumn := other.2.isLt
      simp only [adj, gdist] at near
      omega
    have relation := row_leaf_column_differences_equal height heightValid rowsValid
      extremum leaf first last other.1 rowNear
    change height (other.1, first) - height (point.1, first) =
      height (other.1, last) - height (point.1, last) at relation
    change height (other.1, first) - height (point.1, first) =
      (height (other.1, last) - (width - 1 : Nat)) -
        (height (point.1, last) - (width - 1 : Nat))
    omega
  have maximumCoupling : ∀ point, IsStrictLocalMax rightWalk point ->
      ∀ other, adj point other ->
        leftWalk other - leftWalk point = rightWalk other - rightWalk point := by
    intro point maximum other near
    let extremum : OccupiedBlocks.Extrema height :=
      ⟨(point.1, last), Or.inl (liftMaximum point maximum)⟩
    have boundary : extremum.val.2.val + 1 = width := by dsimp [extremum, last]; omega
    have leaf := lastLeaves extremum boundary
    have rowNear : ((point.1.val : Int) - other.1.val).natAbs = 1 := by
      have pointColumn := point.2.isLt
      have otherColumn := other.2.isLt
      simp only [adj, gdist] at near
      omega
    have relation := row_leaf_column_differences_equal height heightValid rowsValid
      extremum leaf first last other.1 rowNear
    change height (other.1, first) - height (point.1, first) =
      height (other.1, last) - height (point.1, last) at relation
    change height (other.1, first) - height (point.1, first) =
      (height (other.1, last) - (width - 1 : Nat)) -
        (height (point.1, last) - (width - 1 : Nat))
    omega
  haveI : Nonempty (Cell rows 1) := ⟨(⟨0, by omega⟩, 0)⟩
  obtain ⟨start, startMinimum⟩ := Finite.exists_min leftWalk
  have minimum : IsStrictLocalMin leftWalk start := by
    intro neighbor near
    have step := (abs_eq (by norm_num : (0 : Int) ≤ 1)).mp
      (leftValid start neighbor near)
    have lower := startMinimum neighbor
    omega
  let extremum : OccupiedBlocks.Extrema height :=
    ⟨(start.1, first), Or.inr (liftMinimum start minimum)⟩
  have leaf := firstLeaves extremum rfl
  have profile := (OccupiedBlocks.row_leaf_fixed_sign_profile height heightValid
    rowsValid extremum leaf).1 last
  have sign : OccupiedBlocks.extremumSign height extremum = 1 := by
    change (if IsStrictLocalMin height (start.1, first) then 1 else -1) = 1
    rw [if_pos (liftMinimum start minimum)]
  rw [sign, one_mul] at profile
  have distance : |(last.val : Int) - first.val| = (width - 1 : Nat) := by
    change |((width - 1 : Nat) : Int) - 0| = (width - 1 : Nat)
    rw [sub_zero, abs_of_nonneg (Nat.cast_nonneg _)]
  change height (start.1, last) = height (start.1, first) +
    |(last.val : Int) - first.val| at profile
  rw [distance] at profile
  have startEqual : leftWalk start = rightWalk start := by
    change height (start.1, first) = height (start.1, last) - (width - 1 : Nat)
    omega
  have walksEqual := ordered_walks_equal leftWalk rightWalk leftValid rightValid
    ordered minimumCoupling maximumCoupling start startEqual
  refine ⟨fun row => height (row, first), fun column => (column.val : Int), ?_⟩
  intro row column
  have endpoints := congrFun walksEqual ((row, 0) : Cell rows 1)
  change height (row, first) = height (row, last) - (width - 1 : Nat) at endpoints
  have saturated : height (row, last) - height (row, first) =
      gdist (row, first) (row, last) := by
    simp only [gdist]
    dsimp only [first, last] at endpoints ⊢
    omega
  have between : gdist (row, first) (row, column) + gdist (row, column) (row, last) =
      gdist (row, first) (row, last) := by
    have columnBound := column.isLt
    simp only [gdist]
    dsimp only [first, last]
    omega
  have value := Saturation.between_saturated_rise height heightValid
    (row, first) (row, last) (row, column) saturated between
  have columnDistance : gdist (row, first) (row, column) = (column.val : Int) := by
    simp [gdist, first]
  exact value.trans (congrArg (height (row, first) + ·) columnDistance)

#print axioms line_neighbors_exhausted
#print axioms equal_neighbor
#print axioms ordered_walks_equal
#print axioms columnLine_isHeight
#print axioms columnLine_increment_lift
#print axioms columnLine_minimum_lift
#print axioms columnLine_maximum_lift
#print axioms row_leaf_column_differences_equal
#print axioms boundary_leaf_profiles_separable

end OrigamiCone.Sequel.BoundaryComparison
