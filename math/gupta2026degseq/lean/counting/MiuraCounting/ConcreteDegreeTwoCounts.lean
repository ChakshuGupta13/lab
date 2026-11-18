import MiuraCounting.ConcreteSideDecomposition
import OrigamiCone.CornerCount

namespace OrigamiCone.Sequel.DegreeTwoCounts

open OrigamiCone SideDecomposition

def normalizedCornerHeight {rows width : Nat} (rowsPositive : 0 < rows) (widthPositive : 0 < width)
    (apex : Cell rows width) : Cell rows width -> Int :=
  coneC apex (gdist apex (⟨0, rowsPositive⟩, ⟨0, widthPositive⟩))

theorem normalizedCornerHeight_canonical {rows width : Nat} (rowsPositive : 0 < rows)
    (widthPositive : 0 < width) (apex : Cell rows width) :
    IsCanonicalHeight (normalizedCornerHeight rowsPositive widthPositive apex) := by
  refine ⟨coneC_isHeight _ _, ?_⟩
  intro point rowZero columnZero
  have root : point = (⟨0, rowsPositive⟩, ⟨0, widthPositive⟩) :=
    Prod.ext (Fin.ext rowZero) (Fin.ext columnZero)
  simp only [root, normalizedCornerHeight, coneC, sub_self]

def cornerCountedHeight (rows width : Nat) (rowsValid : 2 ≤ rows) (widthValid : 2 ≤ width)
    (apex : {point : Cell rows width // IsGridCorner point}) : CountedCanonicalHeights rows width 2 := by
  let height := normalizedCornerHeight (by omega : 0 < rows) (by omega : 0 < width) apex.val
  have canonical : IsCanonicalHeight height := normalizedCornerHeight_canonical _ _ apex.val
  have sizeBound : 2 ≤ rows * width := by nlinarith
  have degree : (neighbors height).ncard = 2 := by
    apply (degree_two_iff_corner rowsValid widthValid canonical.1).mpr
    exact ⟨apex.val, _, apex.property.1, apex.property.2, rfl⟩
  exact ⟨height, canonical, (degree_eq_extrema canonical.1 sizeBound).symm.trans degree⟩

noncomputable def cornerHeightEquiv (rows width : Nat) (rowsValid : 2 ≤ rows) (widthValid : 2 ≤ width) :
    {point : Cell rows width // IsGridCorner point} ≃ CountedCanonicalHeights rows width 2 :=
  Equiv.ofBijective (cornerCountedHeight rows width rowsValid widthValid) (by
    constructor
    · intro first second equality
      apply Subtype.ext
      have sameHeight := congrArg (fun height : CountedCanonicalHeights rows width 2 => height.val) equality
      change normalizedCornerHeight _ _ first.val = normalizedCornerHeight _ _ second.val at sameHeight
      have maximum : IsStrictLocalMax (normalizedCornerHeight (by omega : 0 < rows)
          (by omega : 0 < width) first.val) first.val := coneC_max_at _ _
      rw [sameHeight] at maximum
      exact coneC_unique_max second.val _ first.val maximum
    · intro height
      have sizeBound : 2 ≤ rows * width := by nlinarith
      have degree : (neighbors height.val).ncard = 2 :=
        (degree_eq_extrema height.property.1.1 sizeBound).trans height.property.2
      obtain ⟨apex, level, rowEnd, columnEnd, profile⟩ :=
        (degree_two_iff_corner rowsValid widthValid height.property.1.1).mp degree
      refine ⟨⟨apex, rowEnd, columnEnd⟩, ?_⟩
      apply Subtype.ext
      let root : Cell rows width := (⟨0, by omega⟩, ⟨0, by omega⟩)
      have atRoot := height.property.1.2 root rfl rfl
      have rootProfile := congrFun profile root
      simp only [coneC] at rootProfile
      have levelEquality : level = gdist apex root := by omega
      change normalizedCornerHeight _ _ apex = height.val
      rw [profile]
      change coneC apex (gdist apex root) = coneC apex level
      rw [levelEquality])

theorem Ed_two (rows width : Nat) (rowsValid : 2 ≤ rows) (widthValid : 2 ≤ width) :
    Ed 2 rows width = 4 := by
  have corners : Nat.card {point : Cell rows width // IsGridCorner point} = 4 := by
    simpa only [Nat.card_eq_fintype_card, Fintype.card_subtype] using card_isGridCorner rowsValid widthValid
  rw [Ed_eq_natCard]
  exact (Nat.card_congr (cornerHeightEquiv rows width rowsValid widthValid)).symm.trans corners

def cornerGradient {rows width : Nat} (rowDirection columnDirection : Bool) : Cell rows width -> Int :=
  fun point => (if rowDirection then (point.1.val : Int) else -(point.1.val : Int)) +
    (if columnDirection then (point.2.val : Int) else -(point.2.val : Int))

def gradientApex (rows width : Nat) (rowsValid : 2 ≤ rows) (widthValid : 2 ≤ width)
    (rowDirection columnDirection : Bool) : Cell rows width :=
  (if rowDirection then ⟨rows - 1, by omega⟩ else ⟨0, by omega⟩,
    if columnDirection then ⟨width - 1, by omega⟩ else ⟨0, by omega⟩)

theorem gradientApex_corner (rows width : Nat) (rowsValid : 2 ≤ rows) (widthValid : 2 ≤ width)
    (rowDirection columnDirection : Bool) :
    IsGridCorner (gradientApex rows width rowsValid widthValid rowDirection columnDirection) := by
  cases rowDirection <;> cases columnDirection <;> simp [gradientApex, IsGridCorner, IsEndpoint] <;> omega

theorem normalized_gradientApex_profile (rows width : Nat) (rowsValid : 2 ≤ rows) (widthValid : 2 ≤ width)
    (rowDirection columnDirection : Bool) :
    normalizedCornerHeight (by omega : 0 < rows) (by omega : 0 < width)
      (gradientApex rows width rowsValid widthValid rowDirection columnDirection) =
        cornerGradient rowDirection columnDirection := by
  funext point
  have rowBound := point.1.isLt
  have columnBound := point.2.isLt
  have rowAbs : |((rows - 1 : Nat) : Int) - (point.1.val : Int)| =
      ((rows - 1 : Nat) : Int) - (point.1.val : Int) := abs_of_nonneg (by omega)
  have columnAbs : |((width - 1 : Nat) : Int) - (point.2.val : Int)| =
      ((width - 1 : Nat) : Int) - (point.2.val : Int) := abs_of_nonneg (by omega)
  cases rowDirection <;> cases columnDirection <;>
    simp [normalizedCornerHeight, gradientApex, cornerGradient, coneC, gdist, rowAbs, columnAbs] <;> omega

theorem corner_enumeration (rows width : Nat) (rowsValid : 2 ≤ rows) (widthValid : 2 ≤ width)
    (apex : Cell rows width) (corner : IsGridCorner apex) :
    ∃ rowDirection columnDirection : Bool,
      apex = gradientApex rows width rowsValid widthValid rowDirection columnDirection := by
  refine ⟨decide (apex.1.val ≠ 0), decide (apex.2.val ≠ 0), ?_⟩
  apply Prod.ext
  · apply Fin.ext
    rcases corner.1 with zero | last
    · simp [gradientApex, zero]
    · have notZero : apex.1.val ≠ 0 := by omega
      simp [gradientApex, notZero]
      omega
  · apply Fin.ext
    rcases corner.2 with zero | last
    · simp [gradientApex, zero]
    · have notZero : apex.2.val ≠ 0 := by omega
      simp [gradientApex, notZero]
      omega

theorem gradient_canonical_and_two (rows width : Nat) (rowsValid : 2 ≤ rows) (widthValid : 2 ≤ width)
    (rowDirection columnDirection : Bool) :
    IsCanonicalHeight (cornerGradient (rows := rows) (width := width) rowDirection columnDirection) ∧
      numExtrema (cornerGradient (rows := rows) (width := width) rowDirection columnDirection) = 2 := by
  let height := cornerCountedHeight rows width rowsValid widthValid
    ⟨gradientApex rows width rowsValid widthValid rowDirection columnDirection,
      gradientApex_corner rows width rowsValid widthValid rowDirection columnDirection⟩
  have canonical := height.property.1
  have counts := height.property.2
  change IsCanonicalHeight (normalizedCornerHeight _ _
    (gradientApex rows width rowsValid widthValid rowDirection columnDirection)) at canonical
  change numExtrema (normalizedCornerHeight _ _
    (gradientApex rows width rowsValid widthValid rowDirection columnDirection)) = 2 at counts
  rw [normalized_gradientApex_profile] at canonical counts
  exact ⟨canonical, counts⟩

theorem degree_two_iff_gradient (rows width : Nat) (rowsValid : 2 ≤ rows) (widthValid : 2 ≤ width)
    (height : Cell rows width -> Int) (canonical : IsCanonicalHeight height) :
    numExtrema height = 2 ↔ ∃ rowDirection columnDirection : Bool,
      height = cornerGradient rowDirection columnDirection := by
  constructor
  · intro counts
    obtain ⟨apex, same⟩ := (cornerHeightEquiv rows width rowsValid widthValid).surjective ⟨height, canonical, counts⟩
    have sameHeight := congrArg (fun value : CountedCanonicalHeights rows width 2 => value.val) same
    change normalizedCornerHeight _ _ apex.val = height at sameHeight
    obtain ⟨rowDirection, columnDirection, sameApex⟩ := corner_enumeration rows width rowsValid widthValid apex.val apex.property
    refine ⟨rowDirection, columnDirection, ?_⟩
    rw [← sameHeight, sameApex, normalized_gradientApex_profile]
  · rintro ⟨rowDirection, columnDirection, rfl⟩
    exact (gradient_canonical_and_two rows width rowsValid widthValid rowDirection columnDirection).2

theorem cornerGradient_injective (rows width : Nat) (rowsValid : 2 ≤ rows) (widthValid : 2 ≤ width) :
    Function.Injective (fun directions : Bool × Bool =>
      cornerGradient (rows := rows) (width := width) directions.1 directions.2) := by
  rintro ⟨firstRow, firstColumn⟩ ⟨secondRow, secondColumn⟩ equality
  apply Prod.ext
  · have value := congrFun equality ((⟨1, by omega⟩, ⟨0, by omega⟩) : Cell rows width)
    cases firstRow <;> cases secondRow
    · rfl
    · norm_num [cornerGradient] at value
    · norm_num [cornerGradient] at value
    · rfl
  · have value := congrFun equality ((⟨0, by omega⟩, ⟨1, by omega⟩) : Cell rows width)
    cases firstColumn <;> cases secondColumn
    · rfl
    · norm_num [cornerGradient] at value
    · norm_num [cornerGradient] at value
    · rfl

#print axioms normalizedCornerHeight
#print axioms normalizedCornerHeight_canonical
#print axioms cornerCountedHeight
#print axioms cornerHeightEquiv
#print axioms Ed_two
#print axioms cornerGradient
#print axioms gradientApex
#print axioms gradientApex_corner
#print axioms normalized_gradientApex_profile
#print axioms corner_enumeration
#print axioms gradient_canonical_and_two
#print axioms degree_two_iff_gradient
#print axioms cornerGradient_injective

end OrigamiCone.Sequel.DegreeTwoCounts
