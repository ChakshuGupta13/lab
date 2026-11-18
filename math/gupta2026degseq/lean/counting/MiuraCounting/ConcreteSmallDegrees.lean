import MiuraCounting.ConcreteDegreeRemainder
import OrigamiCone.Degree3

namespace OrigamiCone.Sequel.SmallDegrees

open OrigamiCone SingleSideFamilies SideDecomposition DegreeRemainder

theorem boundary_cone_singleSide {rows width : Nat} (rowsValid : 2 ≤ rows) (widthValid : 2 ≤ width)
    (apex : Cell rows width) (level : Int) (boundary : IsEndpoint apex.1 ∨ IsEndpoint apex.2) :
    IsSingleSide (coneC apex level) := by
  have endpoints (point : Cell rows width) (minimum : IsStrictLocalMin (coneC apex level) point) :
      PathEnd apex.1 point.1 ∧ PathEnd apex.2 point.2 := by
    apply (Dq_strictMax_iff rowsValid widthValid apex point).mp
    intro other adjacent
    have step := minimum other adjacent
    simp only [coneC, Dq] at step ⊢
    omega
  rcases boundary with rowBoundary | columnBoundary
  · rcases rowBoundary with rowZero | rowLast
    · refine ⟨Side.bottom, ?_⟩
      intro point minimum
      have far := (endpoints point minimum).1
      change point.1.val + 1 = rows
      rcases far with ⟨_, different⟩ | ⟨last, _⟩
      · exact False.elim (different rowZero)
      · exact last
    · refine ⟨Side.top, ?_⟩
      intro point minimum
      have far := (endpoints point minimum).1
      change point.1.val = 0
      rcases far with ⟨zero, _⟩ | ⟨_, different⟩
      · exact zero
      · exact False.elim (different rowLast)
  · rcases columnBoundary with columnZero | columnLast
    · refine ⟨Side.right, ?_⟩
      intro point minimum
      have far := (endpoints point minimum).2
      change point.2.val + 1 = width
      rcases far with ⟨_, different⟩ | ⟨last, _⟩
      · exact False.elim (different columnZero)
      · exact last
    · refine ⟨Side.left, ?_⟩
      intro point minimum
      have far := (endpoints point minimum).2
      change point.2.val = 0
      rcases far with ⟨zero, _⟩ | ⟨_, different⟩
      · exact zero
      · exact False.elim (different columnLast)

theorem boundary_negative_cone_singleSide {rows width : Nat}
    (apex : Cell rows width) (level : Int) (boundary : IsEndpoint apex.1 ∨ IsEndpoint apex.2) :
    IsSingleSide (fun point => -(coneC apex level) point) := by
  have someSide : ∃ side : Side, OnSide side apex := by
    rcases boundary with rowBoundary | columnBoundary
    · rcases rowBoundary with zero | last
      · exact ⟨Side.top, zero⟩
      · exact ⟨Side.bottom, last⟩
    · rcases columnBoundary with zero | last
      · exact ⟨Side.left, zero⟩
      · exact ⟨Side.right, last⟩
  obtain ⟨side, onSide⟩ := someSide
  refine ⟨side, ?_⟩
  intro point minimum
  have maximum : IsStrictLocalMax (coneC apex level) point := by
    simpa only [strictMin_neg_iff] using minimum
  have same := coneC_unique_max apex level point maximum
  simpa only [same] using onSide

theorem three_extrema_singleSide {rows width : Nat} (rowsValid : 2 ≤ rows) (widthValid : 2 ≤ width)
    (height : Cell rows width -> Int) (heightValid : IsHeight height) (counts : numExtrema height = 3) :
    IsSingleSide height := by
  have sizeBound : 2 ≤ rows * width := by nlinarith
  have degreeThree : (neighbors height).ncard = 3 := (degree_eq_extrema heightValid sizeBound).trans counts
  rcases (degree_three_iff rowsValid widthValid heightValid).mp degreeThree with
    ⟨apex, level, boundary, rfl⟩ | ⟨apex, level, boundary, rfl⟩
  · apply boundary_cone_singleSide rowsValid widthValid apex level
    exact boundary.elim (fun conditions => Or.inl conditions.1) (fun conditions => Or.inr conditions.2)
  · apply boundary_negative_cone_singleSide apex level
    exact boundary.elim (fun conditions => Or.inl conditions.1) (fun conditions => Or.inr conditions.2)

theorem Ed_three (rows width : Nat) (rowsValid : 2 ≤ rows) (widthValid : 2 ≤ width) :
    Ed 3 rows width = 4 * (rows + width - 4) := by
  letI : IsEmpty (CountedRemainderHeights rows width 3) := ⟨fun height =>
    height.property (three_extrema_singleSide rowsValid widthValid height.val.val
      height.val.property.1.1 height.val.property.2)⟩
  rw [Ed_singleSide_remainder rows width 3 rowsValid widthValid (by decide), Nat.card_of_isEmpty]
  norm_num
  omega

#print axioms boundary_cone_singleSide
#print axioms boundary_negative_cone_singleSide
#print axioms three_extrema_singleSide
#print axioms Ed_three

end OrigamiCone.Sequel.SmallDegrees
