import MiuraCounting.ConcreteSignedPathCounts
import MiuraCounting.ConcreteSideDecomposition
import MiuraCounting.ConcreteTotalLeading

namespace OrigamiCone.Sequel.SignedSideCounts

open OrigamiCone SeparableProducts SideWalkCounts SideDecomposition SignedPathCounts SingleSideFamilies SignedHeightCounts SplitPolynomials

abbrev PathSplitHeights (length minima maxima : Nat) :=
  {walk : CanonicalWalk length // SignedExtrema.count walk.val SignedExtrema.Kind.minimum = minima ∧
    SignedExtrema.count walk.val SignedExtrema.Kind.maximum = maxima}

abbrev SideSplitHeights (rows width minima maxima : Nat) (side : Side) :=
  {height : CanonicalSideHeights rows width side //
    SignedExtrema.count height.val SignedExtrema.Kind.minimum = minima ∧
      SignedExtrema.count height.val SignedExtrema.Kind.maximum = maxima}

noncomputable def pathSplitCount (length minima maxima : Nat) : Nat :=
  Nat.card (PathSplitHeights length minima maxima)

theorem pathSplitCount_formula (length minima maxima : Nat) (lengthValid : 2 ≤ length)
    (minimaPositive : 0 < minima) (maximaPositive : 0 < maxima) :
    pathSplitCount length minima maxima =
      if |(minima : Int) - (maxima : Int)| ≤ 1 then
        (1 + if minima = maxima then 1 else 0) * (length - 2).choose (minima + maxima - 2) else 0 :=
  canonical_path_signed_binomial length minima maxima lengthValid minimaPositive maximaPositive

theorem topSideWalk_signed_counts (rows width : Nat) (rowsPositive : 0 < rows)
    (height : CanonicalTopHeights rows width) :
    SignedExtrema.count height.val SignedExtrema.Kind.minimum =
        SignedExtrema.count ((topSideWalkEquiv rows width rowsPositive) height).val SignedExtrema.Kind.minimum ∧
      SignedExtrema.count height.val SignedExtrema.Kind.maximum =
        SignedExtrema.count ((topSideWalkEquiv rows width rowsPositive) height).val SignedExtrema.Kind.maximum := by
  let equivalence := topSideWalkEquiv rows width rowsPositive
  let walk := equivalence height
  have reconstruction := congrArg (fun value : CanonicalTopHeights rows width => value.val)
    (equivalence.symm_apply_apply height)
  change productHeight (rampWalk rows) walk = height.val at reconstruction
  have counts := topExtension_signed_counts rows rowsPositive walk
  rw [reconstruction] at counts
  exact counts

theorem bottomSideWalk_signed_counts (rows width : Nat) (rowsPositive : 0 < rows)
    (height : CanonicalBottomHeights rows width) :
    SignedExtrema.count height.val SignedExtrema.Kind.minimum =
        SignedExtrema.count ((bottomSideWalkEquiv rows width rowsPositive) height).val SignedExtrema.Kind.minimum ∧
      SignedExtrema.count height.val SignedExtrema.Kind.maximum =
        SignedExtrema.count ((bottomSideWalkEquiv rows width rowsPositive) height).val SignedExtrema.Kind.maximum := by
  let equivalence := bottomSideWalkEquiv rows width rowsPositive
  let walk := equivalence height
  have reconstruction := congrArg (fun value : CanonicalBottomHeights rows width => value.val)
    (equivalence.symm_apply_apply height)
  change productHeight (SeparableRunBound.negateWalk (rampWalk rows)) walk = height.val at reconstruction
  have counts := bottomExtension_signed_counts rows rowsPositive walk
  rw [reconstruction] at counts
  exact counts

def topSignedSideEquiv (rows width minima maxima : Nat) (rowsPositive : 0 < rows) :
    SideSplitHeights rows width minima maxima Side.top ≃ PathSplitHeights width minima maxima :=
  (topSideWalkEquiv rows width rowsPositive).subtypeEquiv (by
    intro height
    rw [(topSideWalk_signed_counts rows width rowsPositive height).1,
      (topSideWalk_signed_counts rows width rowsPositive height).2])

def bottomSignedSideEquiv (rows width minima maxima : Nat) (rowsPositive : 0 < rows) :
    SideSplitHeights rows width minima maxima Side.bottom ≃ PathSplitHeights width minima maxima :=
  (bottomSideWalkEquiv rows width rowsPositive).subtypeEquiv (by
    intro height
    rw [(bottomSideWalk_signed_counts rows width rowsPositive height).1,
      (bottomSideWalk_signed_counts rows width rowsPositive height).2])

def signedSideTransposeEquiv (rows width minima maxima : Nat) (side : Side) :
    SideSplitHeights rows width minima maxima side ≃
      SideSplitHeights width rows minima maxima (transposeSide side) :=
  (sideTransposeEquiv rows width side).subtypeEquiv (by
    intro height
    have minimumCount := Nat.card_congr (SignedExtrema.transposeSignedEquiv height.val SignedExtrema.Kind.minimum)
    have maximumCount := Nat.card_congr (SignedExtrema.transposeSignedEquiv height.val SignedExtrema.Kind.maximum)
    change SignedExtrema.count (swapH height.val) SignedExtrema.Kind.minimum =
      SignedExtrema.count height.val SignedExtrema.Kind.minimum at minimumCount
    change SignedExtrema.count (swapH height.val) SignedExtrema.Kind.maximum =
      SignedExtrema.count height.val SignedExtrema.Kind.maximum at maximumCount
    change (SignedExtrema.count height.val SignedExtrema.Kind.minimum = minima ∧
      SignedExtrema.count height.val SignedExtrema.Kind.maximum = maxima) ↔
      (SignedExtrema.count (swapH height.val) SignedExtrema.Kind.minimum = minima ∧
        SignedExtrema.count (swapH height.val) SignedExtrema.Kind.maximum = maxima)
    rw [minimumCount, maximumCount])

theorem signedSide_card (rows width minima maxima : Nat) (rowsPositive : 0 < rows)
    (widthPositive : 0 < width) (side : Side) :
    Nat.card (SideSplitHeights rows width minima maxima side) =
      if side = Side.top ∨ side = Side.bottom then pathSplitCount width minima maxima
      else pathSplitCount rows minima maxima := by
  cases side with
  | top => exact Nat.card_congr (topSignedSideEquiv rows width minima maxima rowsPositive)
  | bottom => exact Nat.card_congr (bottomSignedSideEquiv rows width minima maxima rowsPositive)
  | left =>
    exact (Nat.card_congr (signedSideTransposeEquiv rows width minima maxima Side.left)).trans
      (Nat.card_congr (topSignedSideEquiv width rows minima maxima widthPositive))
  | right =>
    exact (Nat.card_congr (signedSideTransposeEquiv rows width minima maxima Side.right)).trans
      (Nat.card_congr (bottomSignedSideEquiv width rows minima maxima widthPositive))

abbrev SingleSideSplitHeights (rows width minima maxima : Nat) :=
  {height : CanonicalSplitHeights rows width minima maxima // IsSingleSide height.val}

theorem signedSide_finite (rows width minima maxima : Nat) (rowsValid : 2 ≤ rows) (widthValid : 2 ≤ width)
    (side : Side) : Finite (SideSplitHeights rows width minima maxima side) := by
  letI : Finite (CanonicalSplitHeights rows width minima maxima) :=
    canonicalSplit_finite rows width minima maxima rowsValid widthValid
  let forget : SideSplitHeights rows width minima maxima side -> CanonicalSplitHeights rows width minima maxima :=
    fun height => ⟨height.val.val, height.val.property.1, height.property⟩
  apply Finite.of_injective forget
  intro first second equality
  exact Subtype.ext (Subtype.ext (congrArg
    (fun height : CanonicalSplitHeights rows width minima maxima => height.val) equality))

noncomputable def signedSideChoiceEquiv (rows width minima maxima : Nat) (rowsValid : 2 ≤ rows)
    (widthValid : 2 ≤ width) (degreeLarge : 3 ≤ minima + maxima) :
    (Sigma fun side : Side => SideSplitHeights rows width minima maxima side) ≃
      SingleSideSplitHeights rows width minima maxima := by
  let forget : (Sigma fun side : Side => SideSplitHeights rows width minima maxima side) ->
      SingleSideSplitHeights rows width minima maxima := fun source =>
    ⟨⟨source.snd.val.val, source.snd.val.property.1, source.snd.property⟩, source.fst, source.snd.val.property.2⟩
  apply Equiv.ofBijective forget
  constructor
  · rintro ⟨firstSide, first⟩ ⟨secondSide, second⟩ equality
    have sameHeight := congrArg (fun height : SingleSideSplitHeights rows width minima maxima => height.val.val) equality
    change first.val.val = second.val.val at sameHeight
    have sameSide : firstSide = secondSide := by
      by_contra different
      have secondSideOnFirst : MinimaOnSide first.val.val secondSide := by
        rw [sameHeight]
        exact second.val.property.2
      have two := two_sides_numExtrema_eq_two first.val.val first.val.property.1.1 rowsValid widthValid
        firstSide secondSide different first.val.property.2 secondSideOnFirst
      have total := JointTypePartition.numExtrema_eq_signed_counts first.val.val (by nlinarith)
      rw [first.property.1, first.property.2] at total
      omega
    subst secondSide
    exact congrArg (Sigma.mk firstSide) (Subtype.ext (Subtype.ext sameHeight))
  · rintro ⟨height, side, onSide⟩
    refine ⟨⟨side, ⟨⟨height.val, height.property.1, onSide⟩, height.property.2⟩⟩, ?_⟩
    rfl

local instance sideFintype : Fintype Side :=
  Fintype.ofList [Side.top, Side.bottom, Side.left, Side.right] (by intro side; cases side <;> simp)

theorem singleSideSplit_card (rows width minima maxima : Nat) (rowsValid : 2 ≤ rows) (widthValid : 2 ≤ width)
    (degreeLarge : 3 ≤ minima + maxima) :
    Nat.card (SingleSideSplitHeights rows width minima maxima) =
      2 * pathSplitCount width minima maxima + 2 * pathSplitCount rows minima maxima := by
  classical
  letI (side : Side) : Finite (SideSplitHeights rows width minima maxima side) :=
    signedSide_finite rows width minima maxima rowsValid widthValid side
  have sum : Nat.card (SingleSideSplitHeights rows width minima maxima) =
      ∑ side : Side, Nat.card (SideSplitHeights rows width minima maxima side) :=
    (Nat.card_congr (signedSideChoiceEquiv rows width minima maxima rowsValid widthValid degreeLarge)).symm.trans Nat.card_sigma
  rw [sum]
  have allSides : (Finset.univ : Finset Side) = {Side.top, Side.bottom, Side.left, Side.right} := by
    ext side
    cases side <;> simp
  rw [allSides]
  simp_rw [signedSide_card rows width minima maxima (by omega) (by omega)]
  simp
  omega

def splitRemainderComplementEquiv (rows width minima maxima : Nat) :
    {height : CanonicalSplitHeights rows width minima maxima // ¬ IsSingleSide height.val} ≃
      CanonicalRemainderSplitHeights rows width minima maxima where
  toFun height := ⟨height.val.val, height.val.property.1, height.val.property.2.1,
    height.val.property.2.2, height.property⟩
  invFun height := ⟨⟨height.val, height.property.1, height.property.2.1, height.property.2.2.1⟩,
    height.property.2.2.2⟩
  left_inv _ := Subtype.ext (Subtype.ext rfl)
  right_inv _ := Subtype.ext rfl

noncomputable def splitSideRemainderEquiv (rows width minima maxima : Nat) :
    SingleSideSplitHeights rows width minima maxima ⊕ CanonicalRemainderSplitHeights rows width minima maxima ≃
      CanonicalSplitHeights rows width minima maxima := by
  classical
  exact (Equiv.sumCongr (Equiv.refl _) (splitRemainderComplementEquiv rows width minima maxima).symm).trans
    (Equiv.sumCompl (fun height : CanonicalSplitHeights rows width minima maxima => IsSingleSide height.val))

theorem canonicalSplit_signed_side_decomposition (rows width minima maxima : Nat)
    (rowsValid : 2 ≤ rows) (widthValid : 2 ≤ width) (degreeLarge : 3 ≤ minima + maxima) :
    Nat.card (CanonicalSplitHeights rows width minima maxima) =
      2 * pathSplitCount width minima maxima + 2 * pathSplitCount rows minima maxima +
        Nat.card (CanonicalRemainderSplitHeights rows width minima maxima) := by
  classical
  letI : Finite (CanonicalSplitHeights rows width minima maxima) :=
    canonicalSplit_finite rows width minima maxima rowsValid widthValid
  letI : Finite (CanonicalRemainderSplitHeights rows width minima maxima) :=
    Finite.of_equiv _ (splitRemainderComplementEquiv rows width minima maxima)
  have partition : Nat.card (CanonicalSplitHeights rows width minima maxima) =
      Nat.card (SingleSideSplitHeights rows width minima maxima) +
        Nat.card (CanonicalRemainderSplitHeights rows width minima maxima) :=
    (Nat.card_congr (splitSideRemainderEquiv rows width minima maxima)).symm.trans Nat.card_sum
  rw [partition, singleSideSplit_card rows width minima maxima rowsValid widthValid degreeLarge]

theorem singleSideSplit_zero_of_unbalanced (rows width minima maxima : Nat)
    (rowsValid : 2 ≤ rows) (widthValid : 2 ≤ width) (degreeLarge : 3 ≤ minima + maxima)
    (minimaPositive : 0 < minima) (maximaPositive : 0 < maxima)
    (unbalanced : 1 < |(minima : Int) - (maxima : Int)|) :
    Nat.card (SingleSideSplitHeights rows width minima maxima) = 0 := by
  rw [singleSideSplit_card rows width minima maxima rowsValid widthValid degreeLarge,
    pathSplitCount_formula width minima maxima widthValid minimaPositive maximaPositive,
    pathSplitCount_formula rows minima maxima rowsValid minimaPositive maximaPositive,
    if_neg (not_le_of_gt unbalanced)]
  simp [not_le_of_gt unbalanced]

noncomputable def signedSidePolynomial (minima maxima : Nat) : MvPolynomial (Fin 2) Rat :=
  if |(minima : Int) - (maxima : Int)| ≤ 1 then
    MvPolynomial.C ((2 * (1 + if minima = maxima then 1 else 0) : Nat) : Rat) *
      ((TotalLeading.sideAxisPolynomial (minima + maxima)).toMvPolynomial (0 : Fin 2) +
        (TotalLeading.sideAxisPolynomial (minima + maxima)).toMvPolynomial (1 : Fin 2))
  else 0

theorem signedSidePolynomial_eval (rows width minima maxima : Nat) (minimaPositive : 0 < minima)
    (maximaPositive : 0 < maxima) (degreeLarge : 3 ≤ minima + maxima)
    (rowBound : minima + maxima ≤ rows) (columnBound : minima + maxima ≤ width) :
    evalAt (signedSidePolynomial minima maxima) rows width =
      (Nat.card (SingleSideSplitHeights rows width minima maxima) : Rat) := by
  rw [singleSideSplit_card rows width minima maxima (by omega) (by omega) degreeLarge,
    pathSplitCount_formula width minima maxima (by omega) minimaPositive maximaPositive,
    pathSplitCount_formula rows minima maxima (by omega) minimaPositive maximaPositive]
  by_cases supported : |(minima : Int) - (maxima : Int)| ≤ 1
  · simp only [signedSidePolynomial, if_pos supported]
    unfold evalAt
    rw [map_mul, map_add, MvPolynomial.eval_C, MvPolynomial.eval_toMvPolynomial,
      MvPolynomial.eval_toMvPolynomial]
    change ((2 * (1 + if minima = maxima then 1 else 0) : Nat) : Rat) *
      ((TotalLeading.sideAxisPolynomial (minima + maxima)).eval (rows : Rat) +
        (TotalLeading.sideAxisPolynomial (minima + maxima)).eval (width : Rat)) = _
    rw [TotalLeading.sideAxisPolynomial_eval (minima + maxima) rows (by omega) rowBound,
      TotalLeading.sideAxisPolynomial_eval (minima + maxima) width (by omega) columnBound]
    push_cast
    ring
  · simp [signedSidePolynomial, supported, evalAt]

theorem signedSidePolynomial_leading (minima maxima : Nat) :
    MvPolynomial.homogeneousComponent (minima + maxima - 2) (signedSidePolynomial minima maxima) =
      if |(minima : Int) - (maxima : Int)| ≤ 1 then
        MvPolynomial.C (((2 * (1 + if minima = maxima then 1 else 0) : Nat) : Rat) /
          ((minima + maxima - 2).factorial : Rat)) *
            (MvPolynomial.X (0 : Fin 2) ^ (minima + maxima - 2) +
              MvPolynomial.X (1 : Fin 2) ^ (minima + maxima - 2))
      else 0 := by
  by_cases supported : |(minima : Int) - (maxima : Int)| ≤ 1
  · have coefficient : (TotalLeading.sideAxisPolynomial (minima + maxima)).coeff (minima + maxima - 2) =
        ((minima + maxima - 2).factorial : Rat)⁻¹ :=
      Polynomial.coeff_preHilbertPoly_self Rat (minima + maxima - 2) (minima + maxima)
    rw [signedSidePolynomial, if_pos supported, if_pos supported,
      MvPolynomial.homogeneousComponent_C_mul, map_add,
      TotalLeading.toMvPolynomial_homogeneousComponent, TotalLeading.toMvPolynomial_homogeneousComponent,
      coefficient]
    simp only [div_eq_mul_inv, map_mul]
    ring
  · simp only [signedSidePolynomial, if_neg supported, map_zero]

#print axioms pathSplitCount
#print axioms pathSplitCount_formula
#print axioms topSideWalk_signed_counts
#print axioms bottomSideWalk_signed_counts
#print axioms topSignedSideEquiv
#print axioms bottomSignedSideEquiv
#print axioms signedSideTransposeEquiv
#print axioms signedSide_card
#print axioms signedSide_finite
#print axioms signedSideChoiceEquiv
#print axioms singleSideSplit_card
#print axioms splitRemainderComplementEquiv
#print axioms splitSideRemainderEquiv
#print axioms canonicalSplit_signed_side_decomposition
#print axioms singleSideSplit_zero_of_unbalanced
#print axioms signedSidePolynomial
#print axioms signedSidePolynomial_eval
#print axioms signedSidePolynomial_leading

end OrigamiCone.Sequel.SignedSideCounts
