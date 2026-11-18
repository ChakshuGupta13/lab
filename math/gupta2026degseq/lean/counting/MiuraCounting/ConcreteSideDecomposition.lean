import MiuraCounting.ConcreteSideWalkCounts
import OrigamiCone.Degree2

namespace OrigamiCone.Sequel.SideDecomposition

open OrigamiCone FrozenExpansion JointReduction JointFibers JointTypePartition SplitPolynomials SignedHeightCounts SingleSideFamilies SeparableProducts SeparableRunBound RemainderPolynomials SideWalkCounts

theorem on_distinct_sides_unique {rows width : Nat} (rowsValid : 2 ≤ rows) (widthValid : 2 ≤ width)
    (firstSide secondSide : Side) (different : firstSide ≠ secondSide) (first second : Cell rows width)
    (firstOnFirst : OnSide firstSide first) (firstOnSecond : OnSide secondSide first)
    (secondOnFirst : OnSide firstSide second) (secondOnSecond : OnSide secondSide second) : first = second := by
  apply Prod.ext <;> apply Fin.ext <;>
    cases firstSide <;> cases secondSide <;> simp_all [OnSide] <;> omega

theorem on_distinct_sides_corner {rows width : Nat} (rowsValid : 2 ≤ rows) (widthValid : 2 ≤ width)
    (firstSide secondSide : Side) (different : firstSide ≠ secondSide) (point : Cell rows width)
    (onFirst : OnSide firstSide point) (onSecond : OnSide secondSide point) :
    IsEndpoint point.1 ∧ IsEndpoint point.2 := by
  cases firstSide <;> cases secondSide <;> simp_all [OnSide, IsEndpoint] <;> omega

theorem two_sides_numExtrema_eq_two {rows width : Nat} (height : Cell rows width -> Int)
    (heightValid : IsHeight height) (rowsValid : 2 ≤ rows) (widthValid : 2 ≤ width)
    (firstSide secondSide : Side) (different : firstSide ≠ secondSide)
    (onFirst : MinimaOnSide height firstSide) (onSecond : MinimaOnSide height secondSide) :
    numExtrema height = 2 := by
  classical
  haveI : Nonempty (Cell rows width) := ⟨(⟨0, by omega⟩, ⟨0, by omega⟩)⟩
  obtain ⟨apex, minimum⟩ := exists_strictLocalMin heightValid (by omega) (by omega)
  have unique : ∀ point, IsStrictLocalMin height point -> point = apex := by
    intro point pointMinimum
    exact on_distinct_sides_unique rowsValid widthValid firstSide secondSide different point apex
      (onFirst point pointMinimum) (onSecond point pointMinimum) (onFirst apex minimum) (onSecond apex minimum)
  have corner := on_distinct_sides_corner rowsValid widthValid firstSide secondSide different apex
    (onFirst apex minimum) (onSecond apex minimum)
  let reflected := fun point => -height point
  have reflectedValid : IsHeight reflected := isHeight_neg heightValid
  have coneEquality : reflected = coneC apex (-height apex) := by
    funext point
    have value := cone_min heightValid unique point
    change -height point = -height apex - gdist apex point
    omega
  have degree := (degree_two_iff_corner rowsValid widthValid reflectedValid).mpr
    ⟨apex, -height apex, corner.1, corner.2, coneEquality⟩
  have count : numExtrema reflected = 2 := by
    change (Finset.univ.filter (IsStrictLocalExtremum reflected)).card = 2
    rw [← degree_eq_extrema reflectedValid (by nlinarith)]
    exact degree
  have sameExtrema : numExtrema reflected = numExtrema height := by
    change numExtrema (GridNegation.negH height) = numExtrema height
    unfold numExtrema
    congr 1
    ext point
    simp only [Finset.mem_filter, GridNegation.negH_extremum_iff]
  rw [sameExtrema] at count
  exact count

abbrev CountedCanonicalHeights (rows width degree : Nat) :=
  {height : Cell rows width -> Int // IsCanonicalHeight height ∧ numExtrema height = degree}

abbrev CountedSingleSideHeights (rows width degree : Nat) :=
  {height : CountedCanonicalHeights rows width degree // IsSingleSide height.val}

abbrev CountedSideHeights (rows width degree : Nat) (side : Side) :=
  {height : CanonicalSideHeights rows width side // numExtrema height.val = degree}

noncomputable def sideChoiceEquiv (rows width degree : Nat) (rowsValid : 2 ≤ rows) (widthValid : 2 ≤ width)
    (degreeLarge : 3 ≤ degree) :
    (Sigma fun side : Side => CountedSideHeights rows width degree side) ≃
      CountedSingleSideHeights rows width degree := by
  let realize : (Sigma fun side : Side => CountedSideHeights rows width degree side) ->
      CountedSingleSideHeights rows width degree :=
    fun source => ⟨⟨source.snd.val.val, source.snd.val.property.1, source.snd.property⟩,
      ⟨source.fst, source.snd.val.property.2⟩⟩
  apply Equiv.ofBijective realize
  constructor
  · rintro ⟨firstSide, first⟩ ⟨secondSide, second⟩ equality
    have sameHeight := congrArg (fun height : CountedSingleSideHeights rows width degree => height.val.val) equality
    change first.val.val = second.val.val at sameHeight
    have sameSide : firstSide = secondSide := by
      by_contra different
      have onSecond : MinimaOnSide first.val.val secondSide := by
        rw [sameHeight]
        exact second.val.property.2
      have two := two_sides_numExtrema_eq_two first.val.val first.val.property.1.1 rowsValid widthValid
        firstSide secondSide different first.val.property.2 onSecond
      have count := first.property
      omega
    subst secondSide
    have same : first = second := Subtype.ext (Subtype.ext sameHeight)
    cases same
    rfl
  · rintro ⟨⟨height, canonical, count⟩, ⟨side, property⟩⟩
    exact ⟨⟨side, ⟨⟨height, canonical, property⟩, count⟩⟩, rfl⟩

theorem countedCanonical_finite (rows width degree : Nat) (rowsValid : 2 ≤ rows) (widthValid : 2 ≤ width) :
    Finite (CountedCanonicalHeights rows width degree) :=
  Set.finite_coe_iff.mpr (Ed_fibre_finite degree (by omega) (by omega))

theorem countedSide_finite (rows width degree : Nat) (rowsValid : 2 ≤ rows) (widthValid : 2 ≤ width) (side : Side) :
    Finite (CountedSideHeights rows width degree side) := by
  letI : Finite (CountedCanonicalHeights rows width degree) := countedCanonical_finite rows width degree rowsValid widthValid
  let encode : CountedSideHeights rows width degree side -> CountedCanonicalHeights rows width degree :=
    fun source => ⟨source.val.val, source.val.property.1, source.property⟩
  apply Finite.of_injective encode
  intro first second equality
  apply Subtype.ext
  apply Subtype.ext
  exact congrArg (fun height : CountedCanonicalHeights rows width degree => height.val) equality

local instance sideFintype : Fintype Side :=
  Fintype.ofList [Side.top, Side.bottom, Side.left, Side.right] (by intro side; cases side <;> simp)

theorem countedSide_card (rows width degree : Nat) (rowsValid : 2 ≤ rows) (widthValid : 2 ≤ width)
    (degreeLarge : 3 ≤ degree) (side : Side) :
    Nat.card (CountedSideHeights rows width degree side) =
      if side = Side.top ∨ side = Side.bottom then 2 * (width - 2).choose (degree - 2)
      else 2 * (rows - 2).choose (degree - 2) := by
  obtain ⟨rowJunctions, rfl⟩ : ∃ junctions : Nat, rows = junctions + 2 := ⟨rows - 2, by omega⟩
  obtain ⟨columnJunctions, rfl⟩ : ∃ junctions : Nat, width = junctions + 2 := ⟨width - 2, by omega⟩
  obtain ⟨turnCount, rfl⟩ : ∃ turns : Nat, degree = turns + 2 := ⟨degree - 2, by omega⟩
  cases side with
  | top =>
    change Nat.card (CountedSideHeights (rowJunctions + 2) (columnJunctions + 2) (turnCount + 2) Side.top) =
      2 * columnJunctions.choose turnCount
    exact topSide_extrema_card (rowJunctions + 2) columnJunctions turnCount (by omega)
  | bottom =>
    change Nat.card (CountedSideHeights (rowJunctions + 2) (columnJunctions + 2) (turnCount + 2) Side.bottom) =
      2 * columnJunctions.choose turnCount
    exact bottomSide_extrema_card (rowJunctions + 2) columnJunctions turnCount (by omega)
  | left =>
    change Nat.card (CountedSideHeights (rowJunctions + 2) (columnJunctions + 2) (turnCount + 2) Side.left) =
      2 * rowJunctions.choose turnCount
    exact leftSide_extrema_card (columnJunctions + 2) rowJunctions turnCount (by omega)
  | right =>
    change Nat.card (CountedSideHeights (rowJunctions + 2) (columnJunctions + 2) (turnCount + 2) Side.right) =
      2 * rowJunctions.choose turnCount
    exact rightSide_extrema_card (columnJunctions + 2) rowJunctions turnCount (by omega)

theorem singleSide_card_eq_sum (rows width degree : Nat) (rowsValid : 2 ≤ rows) (widthValid : 2 ≤ width)
    (degreeLarge : 3 ≤ degree) :
    Nat.card (CountedSingleSideHeights rows width degree) =
      ∑ side : Side, Nat.card (CountedSideHeights rows width degree side) := by
  letI (side : Side) : Finite (CountedSideHeights rows width degree side) :=
    countedSide_finite rows width degree rowsValid widthValid side
  exact (Nat.card_congr (sideChoiceEquiv rows width degree rowsValid widthValid degreeLarge)).symm.trans Nat.card_sigma

theorem singleSide_card (rows width degree : Nat) (rowsValid : 2 ≤ rows) (widthValid : 2 ≤ width)
    (degreeLarge : 3 ≤ degree) :
    Nat.card (CountedSingleSideHeights rows width degree) =
      4 * (rows - 2).choose (degree - 2) + 4 * (width - 2).choose (degree - 2) := by
  rw [singleSide_card_eq_sum rows width degree rowsValid widthValid degreeLarge]
  have allSides : (Finset.univ : Finset Side) = {Side.top, Side.bottom, Side.left, Side.right} := by
    ext side
    cases side <;> simp
  rw [allSides]
  simp [countedSide_card rows width degree rowsValid widthValid degreeLarge]
  omega

#print axioms on_distinct_sides_unique
#print axioms on_distinct_sides_corner
#print axioms two_sides_numExtrema_eq_two
#print axioms sideChoiceEquiv
#print axioms countedCanonical_finite
#print axioms countedSide_finite
#print axioms countedSide_card
#print axioms singleSide_card_eq_sum
#print axioms singleSide_card

end OrigamiCone.Sequel.SideDecomposition
