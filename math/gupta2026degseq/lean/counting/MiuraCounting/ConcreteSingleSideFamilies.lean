import MiuraCounting.ConcreteSignedHeightCounts
import OrigamiCone.SequelSplitEdge

namespace OrigamiCone.Sequel.SingleSideFamilies

open OrigamiCone FrozenExpansion JointReduction JointFibers JointTypePartition SplitPolynomials SignedHeightCounts

def OnSide {rows width : Nat} (side : Side) (point : Cell rows width) : Prop :=
  match side with
  | Side.top => point.1.val = 0
  | Side.bottom => point.1.val + 1 = rows
  | Side.left => point.2.val = 0
  | Side.right => point.2.val + 1 = width

def MinimaOnSide {rows width : Nat} (height : Cell rows width -> Int) (side : Side) : Prop :=
  ∀ point, IsStrictLocalMin height point -> OnSide side point

def IsSingleSide {rows width : Nat} (height : Cell rows width -> Int) : Prop :=
  ∃ side, MinimaOnSide height side

theorem cellExpand_onSide_iff {rows width : Nat} (column : Fin width) (point : Cell rows (width - 1))
    (before : 0 < column.val) (after : column.val + 1 < width) (side : Side) :
    OnSide side (cellExpand column point before after) ↔ OnSide side point := by
  have pointBound := point.2.isLt
  cases side <;> unfold OnSide cellExpand <;> split_ifs <;> dsimp only <;> omega

theorem contract_minimaOnSide_iff {rows width : Nat} (height : Cell rows width -> Int)
    (heightValid : IsHeight height) (rowsPositive : 0 < rows)
    (column : Fin width) (frozen : frozenColumn height column) (side : Side) :
    MinimaOnSide (contractAt height rowsPositive column frozen) side ↔ MinimaOnSide height side := by
  constructor
  · intro contracted point minimum
    let equivalence := SignedExtrema.contractSignedEquiv height heightValid rowsPositive
      column frozen SignedExtrema.Kind.minimum
    let original : SignedExtrema.Sites height SignedExtrema.Kind.minimum := ⟨point, minimum⟩
    let preimage := equivalence.symm original
    have smallSide := contracted preimage.val preimage.property
    have largeSide := (cellExpand_onSide_iff column preimage.val frozen.1 frozen.2.1 side).mpr smallSide
    have roundtrip := congrArg (fun site : SignedExtrema.Sites height SignedExtrema.Kind.minimum => site.val)
      (equivalence.apply_symm_apply original)
    change cellExpand column preimage.val frozen.1 frozen.2.1 = point at roundtrip
    rw [roundtrip] at largeSide
    exact largeSide
  · intro original point minimum
    have largeMinimum := (SignedExtrema.contract_kind_iff height heightValid rowsPositive
      column frozen SignedExtrema.Kind.minimum point).mp minimum
    exact (cellExpand_onSide_iff column point frozen.1 frozen.2.1 side).mp
      (original _ largeMinimum)

theorem contract_singleSide_iff {rows width : Nat} (height : Cell rows width -> Int)
    (heightValid : IsHeight height) (rowsPositive : 0 < rows)
    (column : Fin width) (frozen : frozenColumn height column) :
    IsSingleSide (contractAt height rowsPositive column frozen) ↔ IsSingleSide height := by
  unfold IsSingleSide
  exact exists_congr (fun side => contract_minimaOnSide_iff height heightValid rowsPositive column frozen side)

def transposeSide : Side -> Side
  | Side.top => Side.left
  | Side.bottom => Side.right
  | Side.left => Side.top
  | Side.right => Side.bottom

theorem transposeSide_involutive (side : Side) : transposeSide (transposeSide side) = side := by
  cases side <;> rfl

theorem transpose_onSide_iff {rows width : Nat} (point : Cell rows width) (side : Side) :
    OnSide (transposeSide side) (point.2, point.1) ↔ OnSide side point := by
  cases side <;> rfl

theorem transpose_minimaOnSide_iff {rows width : Nat} (height : Cell rows width -> Int) (side : Side) :
    MinimaOnSide (swapH height) (transposeSide side) ↔ MinimaOnSide height side := by
  constructor
  · intro transposed point minimum
    have transposedMinimum := (swapH_strictMin_iff (h := height) (point.2, point.1)).mpr minimum
    exact (transpose_onSide_iff point side).mp (transposed _ transposedMinimum)
  · intro original point minimum
    have originalMinimum := (swapH_strictMin_iff (h := height) point).mp minimum
    exact (transpose_onSide_iff (point.2, point.1) side).mpr (original _ originalMinimum)

theorem transpose_singleSide_iff {rows width : Nat} (height : Cell rows width -> Int) :
    IsSingleSide (swapH height) ↔ IsSingleSide height := by
  constructor
  · rintro ⟨side, property⟩
    refine ⟨transposeSide side, ?_⟩
    apply (transpose_minimaOnSide_iff height (transposeSide side)).mp
    simpa only [transposeSide_involutive] using property
  · rintro ⟨side, property⟩
    exact ⟨transposeSide side, (transpose_minimaOnSide_iff height side).mpr property⟩

theorem columnStep_singleSide_iff {source target : GridState} (step : ColumnStep source target) :
    IsSingleSide target.columns.height ↔ IsSingleSide source.columns.height := by
  obtain ⟨next, ⟨column, frozen, nextFrozen, rfl⟩, rfl⟩ := step
  exact contract_singleSide_iff source.columns.height source.columns.canonical.1
    (rows_positive source) column frozen

theorem rowStep_singleSide_iff {source target : GridState} (step : RowStep source target) :
    IsSingleSide target.columns.height ↔ IsSingleSide source.columns.height := by
  have transposed := columnStep_singleSide_iff step
  change IsSingleSide (swapH target.columns.height) ↔ IsSingleSide (swapH source.columns.height) at transposed
  exact (transpose_singleSide_iff target.columns.height).symm.trans
    (transposed.trans (transpose_singleSide_iff source.columns.height))

theorem step_singleSide_iff {source target : GridState} (step : Step source target) :
    IsSingleSide target.columns.height ↔ IsSingleSide source.columns.height := by
  rcases step with column | row
  · exact columnStep_singleSide_iff column
  · exact rowStep_singleSide_iff row

theorem reduction_singleSide_iff {source target : GridState} (path : Relation.ReflTransGen Step source target) :
    IsSingleSide target.columns.height ↔ IsSingleSide source.columns.height := by
  induction path with
  | refl => exact Iff.rfl
  | tail _ step inductionHypothesis => exact (step_singleSide_iff step).trans inductionHypothesis

theorem normalFiber_singleSide_iff (core : GridState) (source : NormalFiber core) :
    IsSingleSide source.val.columns.height ↔ IsSingleSide core.columns.height := by
  have same := reduction_singleSide_iff (normalForm_path source.val)
  rw [source.property] at same
  exact same.symm

abbrev CanonicalRemainderSplitHeights (rows width minima maxima : Nat) :=
  {height : Cell rows width -> Int // IsCanonicalHeight height ∧
    SignedExtrema.count height SignedExtrema.Kind.minimum = minima ∧
    SignedExtrema.count height SignedExtrema.Kind.maximum = maxima ∧ ¬ IsSingleSide height}

def canonicalRemainderEquiv (rows width minima maxima : Nat) (rowsValid : 2 ≤ rows) (widthValid : 2 ≤ width) :
    CanonicalRemainderSplitHeights rows width minima maxima ≃
      SplitFamilyStates rows width minima maxima (fun source => ¬ IsSingleSide source.columns.height) where
  toFun := fun height =>
    ⟨⟨rows, rowsValid, ⟨width, widthValid, height.val, height.property.1⟩⟩,
      rfl, rfl, ⟨height.property.2.1, height.property.2.2.1⟩, height.property.2.2.2⟩
  invFun := by
    rintro ⟨⟨sourceRows, sourceRowsValid, ⟨sourceWidth, sourceWidthValid, height, canonical⟩⟩,
      rowEquality, widthEquality, split, remainder⟩
    cases rowEquality
    cases widthEquality
    exact ⟨height, canonical, split.1, split.2, remainder⟩
  left_inv := fun _ => Subtype.ext rfl
  right_inv := by
    rintro ⟨⟨sourceRows, sourceRowsValid, ⟨sourceWidth, sourceWidthValid, height, canonical⟩⟩,
      rowEquality, widthEquality, split, remainder⟩
    cases rowEquality
    cases widthEquality
    rfl

theorem canonicalRemainder_finite (rows width minima maxima : Nat)
    (rowsValid : 2 ≤ rows) (widthValid : 2 ≤ width) :
    Finite (CanonicalRemainderSplitHeights rows width minima maxima) := by
  letI : Finite (SplitFamilyStates rows width minima maxima
      (fun source => ¬ IsSingleSide source.columns.height)) := splitFamilyStates_finite rows width minima maxima _
  exact Finite.of_equiv _ (canonicalRemainderEquiv rows width minima maxima rowsValid widthValid).symm

theorem canonicalRemainder_eventually_polynomial (minima maxima : Nat) (degreeBound : 3 ≤ minima + maxima) :
    ∃ polynomial : MvPolynomial (Fin 2) Rat,
      (∀ coordinate, polynomial.degreeOf coordinate ≤ minima + maxima - 2) ∧
      ∀ rows width : Nat, 2 * (minima + maxima) - 1 ≤ rows ->
        2 * (minima + maxima) - 1 ≤ width ->
          evalAt polynomial rows width =
            (Nat.card (CanonicalRemainderSplitHeights rows width minima maxima) : Rat) := by
  obtain ⟨polynomial, degrees, agreement⟩ := splitFamily_eventually_polynomial minima maxima degreeBound
    (fun source => ¬ IsSingleSide source.columns.height)
    (fun core source => not_congr (normalFiber_singleSide_iff core source))
  refine ⟨polynomial, degrees, ?_⟩
  intro rows width rowBound columnBound
  have counts := Nat.card_congr (canonicalRemainderEquiv rows width minima maxima (by omega) (by omega))
  rw [counts]
  exact agreement rows width rowBound columnBound

#print axioms cellExpand_onSide_iff
#print axioms contract_minimaOnSide_iff
#print axioms contract_singleSide_iff
#print axioms transposeSide_involutive
#print axioms transpose_onSide_iff
#print axioms transpose_minimaOnSide_iff
#print axioms transpose_singleSide_iff
#print axioms columnStep_singleSide_iff
#print axioms rowStep_singleSide_iff
#print axioms step_singleSide_iff
#print axioms reduction_singleSide_iff
#print axioms normalFiber_singleSide_iff
#print axioms canonicalRemainderEquiv
#print axioms canonicalRemainder_finite
#print axioms canonicalRemainder_eventually_polynomial

end OrigamiCone.Sequel.SingleSideFamilies
