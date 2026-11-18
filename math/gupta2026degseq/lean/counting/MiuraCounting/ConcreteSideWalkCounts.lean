import MiuraCounting.ConcreteRemainderPolynomials
import MiuraCounting.ConcreteSignedPathWalks
import OrigamiCone.Envelope
import OrigamiCone.RowReflectTransport

namespace OrigamiCone.Sequel.SideWalkCounts

open OrigamiCone FrozenExpansion JointReduction JointFibers JointTypePartition SplitPolynomials SignedHeightCounts SingleSideFamilies SeparableProducts SeparableRunBound RemainderPolynomials

theorem topSide_profile {rows width : Nat} (height : Cell rows width -> Int)
    (heightValid : IsHeight height) (rowsPositive : 0 < rows)
    (topSide : MinimaOnSide height Side.top) :
    ∀ point, height point = (point.1.val : Int) + height (⟨0, rowsPositive⟩, point.2) := by
  intro point
  obtain ⟨apex, minimum, attained⟩ := exists_descent_to_min heightValid point
  have apexTop : apex.1.val = 0 := topSide apex minimum
  let boundary : Cell rows width := (⟨0, rowsPositive⟩, point.2)
  have boundaryBound := envelope_le heightValid apex boundary
  have verticalBound := envelope_le heightValid boundary point
  have distanceSplit : gdist apex point = (point.1.val : Int) + gdist apex boundary := by
    unfold gdist
    dsimp only [boundary]
    omega
  have verticalDistance : gdist boundary point = (point.1.val : Int) := by
    simp [boundary, gdist]
  rw [distanceSplit] at attained
  rw [verticalDistance] at verticalBound
  change height point = (point.1.val : Int) + height boundary
  omega

def rampWalk (length : Nat) : CanonicalWalk length :=
  ⟨fun point => (point.2.val : Int), by
    constructor
    · intro first second near
      have firstZero : first.1.val = 0 := by have := first.1.isLt; omega
      have secondZero : second.1.val = 0 := by have := second.1.isLt; omega
      change |(first.2.val : Int) - second.2.val| = 1
      simpa [adj, gdist, firstZero, secondZero] using near
    · intro point _ columnZero
      change (point.2.val : Int) = 0
      exact_mod_cast columnZero⟩

theorem rampWalk_minimum_iff {length : Nat} (point : Cell 1 length) :
    IsStrictLocalMin (rampWalk length).val point ↔ point.2.val = 0 := by
  constructor
  · intro minimum
    by_contra notZero
    have previousBound : point.2.val - 1 < length := by have := point.2.isLt; omega
    have near : adj point (point.1, ⟨point.2.val - 1, previousBound⟩) := by
      simp only [adj, gdist]
      omega
    have relation := minimum _ near
    change ((point.2.val - 1 : Nat) : Int) = (point.2.val : Int) + 1 at relation
    omega
  · intro zero neighbor near
    have pointZero : point.1.val = 0 := by have := point.1.isLt; omega
    have neighborZero : neighbor.1.val = 0 := by have := neighbor.1.isLt; omega
    change (neighbor.2.val : Int) = (point.2.val : Int) + 1
    simp only [adj, gdist] at near
    omega

theorem rampWalk_maximum_iff {length : Nat} (point : Cell 1 length) :
    IsStrictLocalMax (rampWalk length).val point ↔ point.2.val + 1 = length := by
  constructor
  · intro maximum
    by_contra notLast
    have nextBound : point.2.val + 1 < length := by have := point.2.isLt; omega
    have near : adj point (point.1, ⟨point.2.val + 1, nextBound⟩) := by
      simp only [adj, gdist]
      omega
    have relation := maximum _ near
    change ((point.2.val + 1 : Nat) : Int) = (point.2.val : Int) - 1 at relation
    omega
  · intro last neighbor near
    have pointZero : point.1.val = 0 := by have := point.1.isLt; omega
    have neighborZero : neighbor.1.val = 0 := by have := neighbor.1.isLt; omega
    have neighborBound := neighbor.2.isLt
    change (neighbor.2.val : Int) = (point.2.val : Int) - 1
    simp only [adj, gdist] at near
    omega

theorem rampWalk_signed_counts (length : Nat) (lengthPositive : 0 < length) :
    SignedExtrema.count (rampWalk length).val SignedExtrema.Kind.minimum = 1 ∧
      SignedExtrema.count (rampWalk length).val SignedExtrema.Kind.maximum = 1 := by
  let minima : SignedExtrema.Sites (rampWalk length).val SignedExtrema.Kind.minimum ≃ PUnit.{1} := {
    toFun := fun _ => PUnit.unit
    invFun := fun _ => ⟨((0 : Fin 1), ⟨0, lengthPositive⟩), (rampWalk_minimum_iff _).mpr rfl⟩
    left_inv := by
      intro point
      apply Subtype.ext
      apply Prod.ext
      · exact Subsingleton.elim _ _
      · apply Fin.ext
        change 0 = point.val.2.val
        exact ((rampWalk_minimum_iff point.val).mp point.property).symm
    right_inv := fun _ => Subsingleton.elim _ _ }
  let maxima : SignedExtrema.Sites (rampWalk length).val SignedExtrema.Kind.maximum ≃ PUnit.{1} := {
    toFun := fun _ => PUnit.unit
    invFun := fun _ => ⟨((0 : Fin 1), ⟨length - 1, by omega⟩), (rampWalk_maximum_iff _).mpr (by
      change length - 1 + 1 = length
      omega)⟩
    left_inv := by
      intro point
      apply Subtype.ext
      apply Prod.ext
      · exact Subsingleton.elim _ _
      · apply Fin.ext
        have last := (rampWalk_maximum_iff point.val).mp point.property
        change length - 1 = point.val.2.val
        omega
    right_inv := fun _ => Subsingleton.elim _ _ }
  exact ⟨(Nat.card_congr minima).trans (by simp), (Nat.card_congr maxima).trans (by simp)⟩

abbrev CanonicalTopHeights (rows width : Nat) :=
  {height : Cell rows width -> Int // IsCanonicalHeight height ∧ MinimaOnSide height Side.top}

def topSideWalkEquiv (rows width : Nat) (rowsPositive : 0 < rows) :
    CanonicalTopHeights rows width ≃ CanonicalWalk width where
  toFun := fun height => columnRestriction height.val height.property.1 rowsPositive
  invFun := fun walk => ⟨productHeight (rampWalk rows) walk, productHeight_canonical (rampWalk rows) walk, by
    intro point minimum
    have rowMinimum := ((product_kind_iff (rampWalk rows) walk SignedExtrema.Kind.minimum point).mp minimum).1
    exact (rampWalk_minimum_iff ((0 : Fin 1), point.1)).mp rowMinimum⟩
  left_inv := by
    intro height
    apply Subtype.ext
    funext point
    change (point.1.val : Int) + height.val (⟨0, rowsPositive⟩, point.2) = height.val point
    exact (topSide_profile height.val height.property.1.1 rowsPositive height.property.2 point).symm
  right_inv := by
    intro walk
    apply Subtype.ext
    funext point
    change (0 : Int) + walk.val (0, point.2) = walk.val point
    rw [zero_add]
    exact congrArg walk.val (pathPoint_eta point)

theorem topExtension_signed_counts (rows : Nat) (rowsPositive : 0 < rows) {width : Nat} (walk : CanonicalWalk width) :
    SignedExtrema.count (productHeight (rampWalk rows) walk) SignedExtrema.Kind.minimum =
        SignedExtrema.count walk.val SignedExtrema.Kind.minimum ∧
      SignedExtrema.count (productHeight (rampWalk rows) walk) SignedExtrema.Kind.maximum =
        SignedExtrema.count walk.val SignedExtrema.Kind.maximum := by
  have counts := rampWalk_signed_counts rows rowsPositive
  constructor
  · rw [product_signed_count, counts.1, one_mul]
  · rw [product_signed_count, counts.2, one_mul]

theorem topExtension_numExtrema (rows : Nat) (rowsPositive : 0 < rows) {width : Nat}
    (widthValid : 2 ≤ width) (walk : CanonicalWalk width) :
    numExtrema (productHeight (rampWalk rows) walk) = numExtrema walk.val := by
  have gridTotal := numExtrema_eq_signed_counts (productHeight (rampWalk rows) walk) (by nlinarith)
  have pathTotal := numExtrema_eq_signed_counts walk.val (by simpa using widthValid)
  have counts := topExtension_signed_counts rows rowsPositive walk
  omega

theorem topSideWalk_extrema (rows width : Nat) (rowsPositive : 0 < rows) (widthValid : 2 ≤ width)
    (height : CanonicalTopHeights rows width) :
    numExtrema height.val = numExtrema ((topSideWalkEquiv rows width rowsPositive) height).val := by
  let equivalence := topSideWalkEquiv rows width rowsPositive
  let walk := equivalence height
  have reconstruction := congrArg (fun value : CanonicalTopHeights rows width => value.val)
    (equivalence.symm_apply_apply height)
  change productHeight (rampWalk rows) walk = height.val at reconstruction
  have counts := topExtension_numExtrema rows rowsPositive widthValid walk
  rw [reconstruction] at counts
  exact counts

noncomputable def countedTopSideEquiv (rows width degree : Nat) (rowsPositive : 0 < rows) (widthValid : 2 ≤ width) :
    {height : CanonicalTopHeights rows width // numExtrema height.val = degree} ≃
      {walk : CanonicalWalk width // numExtrema walk.val = degree} :=
  (topSideWalkEquiv rows width rowsPositive).subtypeEquiv (by
    intro height
    rw [topSideWalk_extrema rows width rowsPositive widthValid height])

theorem topSide_extrema_card (rows junctions turnCount : Nat) (rowsPositive : 0 < rows) :
    Nat.card {height : CanonicalTopHeights rows (junctions + 2) // numExtrema height.val = turnCount + 2} =
      2 * junctions.choose turnCount :=
  (Nat.card_congr (countedTopSideEquiv rows (junctions + 2) (turnCount + 2) rowsPositive (by omega))).trans
    (SignedPathWalks.canonicalWalk_extrema_card junctions turnCount)

def reflectRowsH {rows width : Nat} (height : Cell rows width -> Int) : Cell rows width -> Int :=
  fun point => height (rowRefl point)

theorem reflectRowsH_isHeight {rows width : Nat} (height : Cell rows width -> Int)
    (heightValid : IsHeight height) : IsHeight (reflectRowsH height) := by
  intro first second near
  exact heightValid (rowRefl first) (rowRefl second) ((rowRefl_adj first second).mpr near)

theorem reflectRowsH_minimum_iff {rows width : Nat} (height : Cell rows width -> Int) (point : Cell rows width) :
    IsStrictLocalMin (reflectRowsH height) point ↔ IsStrictLocalMin height (rowRefl point) := by
  constructor
  · intro minimum neighbor near
    have reflectedNear : adj point (rowRefl neighbor) := by
      apply (rowRefl_adj point (rowRefl neighbor)).mp
      simpa only [rowRefl_involutive] using near
    have value := minimum (rowRefl neighbor) reflectedNear
    simpa only [reflectRowsH, rowRefl_involutive] using value
  · intro minimum neighbor near
    exact minimum (rowRefl neighbor) ((rowRefl_adj point neighbor).mpr near)

theorem bottomSide_profile {rows width : Nat} (height : Cell rows width -> Int)
    (heightValid : IsHeight height) (rowsPositive : 0 < rows)
    (bottomSide : MinimaOnSide height Side.bottom) :
    ∀ point, height point = height (⟨0, rowsPositive⟩, point.2) - (point.1.val : Int) := by
  have reflectedTop : MinimaOnSide (reflectRowsH height) Side.top := by
    intro point minimum
    have originalMinimum := (reflectRowsH_minimum_iff height point).mp minimum
    have boundary := bottomSide (rowRefl point) originalMinimum
    have within := point.1.isLt
    change rows - (point.1.val + 1) + 1 = rows at boundary
    change point.1.val = 0
    omega
  have reverseProfile : ∀ point : Cell rows width,
      height point = ((rows - (point.1.val + 1) : Nat) : Int) +
        height (⟨rows - 1, by omega⟩, point.2) := by
    intro point
    have value := topSide_profile (reflectRowsH height) (reflectRowsH_isHeight height heightValid)
      rowsPositive reflectedTop (rowRefl point)
    change height (rowRefl (rowRefl point)) = ((rows - (point.1.val + 1) : Nat) : Int) +
      height (⟨rows - 1, by omega⟩, point.2) at value
    simpa only [rowRefl_involutive] using value
  intro point
  have value := reverseProfile point
  have atFirst := reverseProfile (⟨0, rowsPositive⟩, point.2)
  change height (⟨0, rowsPositive⟩, point.2) = ((rows - 1 : Nat) : Int) +
    height (⟨rows - 1, by omega⟩, point.2) at atFirst
  have within := point.1.isLt
  omega

abbrev CanonicalBottomHeights (rows width : Nat) :=
  {height : Cell rows width -> Int // IsCanonicalHeight height ∧ MinimaOnSide height Side.bottom}

def bottomSideWalkEquiv (rows width : Nat) (rowsPositive : 0 < rows) :
    CanonicalBottomHeights rows width ≃ CanonicalWalk width where
  toFun := fun height => columnRestriction height.val height.property.1 rowsPositive
  invFun := fun walk =>
    ⟨productHeight (negateWalk (rampWalk rows)) walk, productHeight_canonical (negateWalk (rampWalk rows)) walk, by
      intro point minimum
      have rowMinimum := ((product_kind_iff (negateWalk (rampWalk rows)) walk SignedExtrema.Kind.minimum point).mp minimum).1
      have rowMaximum : IsStrictLocalMax (rampWalk rows).val ((0 : Fin 1), point.1) := strictMin_neg_iff.mp rowMinimum
      exact (rampWalk_maximum_iff ((0 : Fin 1), point.1)).mp rowMaximum⟩
  left_inv := by
    intro height
    apply Subtype.ext
    funext point
    change -(point.1.val : Int) + height.val (⟨0, rowsPositive⟩, point.2) = height.val point
    have value := bottomSide_profile height.val height.property.1.1 rowsPositive height.property.2 point
    omega
  right_inv := by
    intro walk
    apply Subtype.ext
    funext point
    change -(0 : Int) + walk.val (0, point.2) = walk.val point
    rw [neg_zero, zero_add]
    exact congrArg walk.val (pathPoint_eta point)

theorem bottomExtension_signed_counts (rows : Nat) (rowsPositive : 0 < rows) {width : Nat} (walk : CanonicalWalk width) :
    SignedExtrema.count (productHeight (negateWalk (rampWalk rows)) walk) SignedExtrema.Kind.minimum =
        SignedExtrema.count walk.val SignedExtrema.Kind.minimum ∧
      SignedExtrema.count (productHeight (negateWalk (rampWalk rows)) walk) SignedExtrema.Kind.maximum =
        SignedExtrema.count walk.val SignedExtrema.Kind.maximum := by
  have counts := rampWalk_signed_counts rows rowsPositive
  have flipped := negateWalk_signed_counts (rampWalk rows)
  constructor
  · rw [product_signed_count, flipped.1, counts.2, one_mul]
  · rw [product_signed_count, flipped.2, counts.1, one_mul]

theorem bottomExtension_numExtrema (rows : Nat) (rowsPositive : 0 < rows) {width : Nat}
    (widthValid : 2 ≤ width) (walk : CanonicalWalk width) :
    numExtrema (productHeight (negateWalk (rampWalk rows)) walk) = numExtrema walk.val := by
  have gridTotal := numExtrema_eq_signed_counts (productHeight (negateWalk (rampWalk rows)) walk) (by nlinarith)
  have pathTotal := numExtrema_eq_signed_counts walk.val (by simpa using widthValid)
  have counts := bottomExtension_signed_counts rows rowsPositive walk
  omega

theorem bottomSideWalk_extrema (rows width : Nat) (rowsPositive : 0 < rows) (widthValid : 2 ≤ width)
    (height : CanonicalBottomHeights rows width) :
    numExtrema height.val = numExtrema ((bottomSideWalkEquiv rows width rowsPositive) height).val := by
  let equivalence := bottomSideWalkEquiv rows width rowsPositive
  let walk := equivalence height
  have reconstruction := congrArg (fun value : CanonicalBottomHeights rows width => value.val)
    (equivalence.symm_apply_apply height)
  change productHeight (negateWalk (rampWalk rows)) walk = height.val at reconstruction
  have counts := bottomExtension_numExtrema rows rowsPositive widthValid walk
  rw [reconstruction] at counts
  exact counts

noncomputable def countedBottomSideEquiv (rows width degree : Nat) (rowsPositive : 0 < rows) (widthValid : 2 ≤ width) :
    {height : CanonicalBottomHeights rows width // numExtrema height.val = degree} ≃
      {walk : CanonicalWalk width // numExtrema walk.val = degree} :=
  (bottomSideWalkEquiv rows width rowsPositive).subtypeEquiv (by
    intro height
    rw [bottomSideWalk_extrema rows width rowsPositive widthValid height])

theorem bottomSide_extrema_card (rows junctions turnCount : Nat) (rowsPositive : 0 < rows) :
    Nat.card {height : CanonicalBottomHeights rows (junctions + 2) // numExtrema height.val = turnCount + 2} =
      2 * junctions.choose turnCount :=
  (Nat.card_congr (countedBottomSideEquiv rows (junctions + 2) (turnCount + 2) rowsPositive (by omega))).trans
    (SignedPathWalks.canonicalWalk_extrema_card junctions turnCount)

abbrev CanonicalSideHeights (rows width : Nat) (side : Side) :=
  {height : Cell rows width -> Int // IsCanonicalHeight height ∧ MinimaOnSide height side}

def sideTransposeEquiv (rows width : Nat) (side : Side) :
    CanonicalSideHeights rows width side ≃ CanonicalSideHeights width rows (transposeSide side) where
  toFun := fun height => ⟨swapH height.val, swapH_isCanonical height.property.1,
    (transpose_minimaOnSide_iff height.val side).mpr height.property.2⟩
  invFun := fun height => ⟨swapH height.val, swapH_isCanonical height.property.1, by
    have transported := (transpose_minimaOnSide_iff height.val (transposeSide side)).mpr height.property.2
    simpa only [transposeSide_involutive] using transported⟩
  left_inv := by
    intro height
    apply Subtype.ext
    funext point
    cases point
    rfl
  right_inv := by
    intro height
    apply Subtype.ext
    funext point
    cases point
    rfl

noncomputable def countedSideTransposeEquiv (rows width degree : Nat) (side : Side) :
    {height : CanonicalSideHeights rows width side // numExtrema height.val = degree} ≃
      {height : CanonicalSideHeights width rows (transposeSide side) // numExtrema height.val = degree} :=
  (sideTransposeEquiv rows width side).subtypeEquiv (by
    intro height
    change (numExtrema height.val = degree) ↔ (numExtrema (swapH height.val) = degree)
    rw [swapH_numExtrema])

theorem leftSide_extrema_card (width junctions turnCount : Nat) (widthPositive : 0 < width) :
    Nat.card {height : CanonicalSideHeights (junctions + 2) width Side.left // numExtrema height.val = turnCount + 2} =
      2 * junctions.choose turnCount :=
  (Nat.card_congr (countedSideTransposeEquiv (junctions + 2) width (turnCount + 2) Side.left)).trans
    (topSide_extrema_card width junctions turnCount widthPositive)

theorem rightSide_extrema_card (width junctions turnCount : Nat) (widthPositive : 0 < width) :
    Nat.card {height : CanonicalSideHeights (junctions + 2) width Side.right // numExtrema height.val = turnCount + 2} =
      2 * junctions.choose turnCount :=
  (Nat.card_congr (countedSideTransposeEquiv (junctions + 2) width (turnCount + 2) Side.right)).trans
    (bottomSide_extrema_card width junctions turnCount widthPositive)

#print axioms topSide_profile
#print axioms rampWalk_minimum_iff
#print axioms rampWalk_maximum_iff
#print axioms rampWalk_signed_counts
#print axioms topSideWalkEquiv
#print axioms topExtension_signed_counts
#print axioms topExtension_numExtrema
#print axioms topSideWalk_extrema
#print axioms countedTopSideEquiv
#print axioms topSide_extrema_card
#print axioms reflectRowsH_isHeight
#print axioms reflectRowsH_minimum_iff
#print axioms bottomSide_profile
#print axioms bottomSideWalkEquiv
#print axioms bottomExtension_signed_counts
#print axioms bottomExtension_numExtrema
#print axioms bottomSideWalk_extrema
#print axioms countedBottomSideEquiv
#print axioms bottomSide_extrema_card
#print axioms sideTransposeEquiv
#print axioms countedSideTransposeEquiv
#print axioms leftSide_extrema_card
#print axioms rightSide_extrema_card

end OrigamiCone.Sequel.SideWalkCounts
