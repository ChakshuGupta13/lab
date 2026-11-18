import MiuraCounting.ConcreteJointFibers

namespace OrigamiCone.Sequel.JointTypePartition

open OrigamiCone FrozenExpansion JointReduction JointFibers

abbrev ReducedTypes (degree : Nat) :=
  {core : GridState // Reduced core ∧ numExtrema core.columns.height = degree}

abbrev FixedStates (rows width degree : Nat) :=
  {source : GridState // source.rows = rows ∧ source.columns.width = width ∧
    numExtrema source.columns.height = degree}

theorem reducedTypes_finite (degree : Nat) : Finite (ReducedTypes degree) :=
  Set.finite_coe_iff.mpr (reduced_types_finite degree)

theorem fixedStates_finite (rows width degree : Nat) : Finite (FixedStates rows width degree) := by
  apply Set.finite_coe_iff.mpr
  apply (bounded_grids_finite (max rows width)).subset
  intro source properties
  refine ⟨?_, ?_⟩
  · rw [properties.1]
    exact Nat.le_max_left rows width
  · rw [properties.2.1]
    exact Nat.le_max_right rows width

noncomputable def normalType {rows width degree : Nat} (source : FixedStates rows width degree) :
    ReducedTypes degree :=
  ⟨normalForm source.val, normalForm_reduced source.val,
    (normalForm_extrema source.val).trans source.property.2.2⟩

noncomputable def fixedStatesFiberEquiv (rows width degree : Nat) :
    FixedStates rows width degree ≃ Sigma (fun core : ReducedTypes degree =>
      DimensionFiber core.val rows width) where
  toFun := fun source => ⟨normalType source,
    ⟨⟨source.val, rfl⟩, source.property.1, source.property.2.1⟩⟩
  invFun := fun source => ⟨source.snd.val.val,
    source.snd.property.1, source.snd.property.2,
    (normalFiber_extrema source.fst.val source.snd.val).trans source.fst.property.2⟩
  left_inv := fun _ => Subtype.ext rfl
  right_inv := by
    rintro ⟨⟨core, coreReduced, coreCount⟩, ⟨⟨source, sourceNormal⟩, dimensions⟩⟩
    cases sourceNormal
    rfl

noncomputable instance reducedTypesFintype (degree : Nat) : Fintype (ReducedTypes degree) := by
  letI : Finite (ReducedTypes degree) := reducedTypes_finite degree
  exact Fintype.ofFinite _

theorem dimensionFiber_finite_all (core : GridState) (rows width : Nat) :
    Finite (DimensionFiber core rows width) := by
  letI : Finite (FixedStates rows width (numExtrema core.columns.height)) :=
    fixedStates_finite rows width _
  let encode : DimensionFiber core rows width -> FixedStates rows width (numExtrema core.columns.height) :=
    fun source => ⟨source.val.val, source.property.1, source.property.2,
      normalFiber_extrema core source.val⟩
  apply Finite.of_injective encode
  intro first second equality
  apply Subtype.ext
  apply Subtype.ext
  exact congrArg (fun source : FixedStates rows width (numExtrema core.columns.height) => source.val) equality

theorem fixedStates_card (rows width degree : Nat) :
    Nat.card (FixedStates rows width degree) =
      ∑ core : ReducedTypes degree, Nat.card (DimensionFiber core.val rows width) := by
  letI (core : ReducedTypes degree) : Finite (DimensionFiber core.val rows width) :=
    dimensionFiber_finite_all core.val rows width
  exact (Nat.card_congr (fixedStatesFiberEquiv rows width degree)).trans Nat.card_sigma

abbrev FilteredStates (rows width degree : Nat) (predicate : GridState -> Prop) :=
  {source : FixedStates rows width degree // predicate source.val}

abbrev FilteredTypes (degree : Nat) (predicate : GridState -> Prop) :=
  {core : ReducedTypes degree // predicate core.val}

noncomputable instance filteredTypesFintype (degree : Nat) (predicate : GridState -> Prop) :
    Fintype (FilteredTypes degree predicate) := Fintype.ofFinite _

noncomputable def filteredStatesFiberEquiv (rows width degree : Nat) (predicate : GridState -> Prop)
    (invariant : ∀ core : GridState, ∀ source : NormalFiber core,
      predicate source.val ↔ predicate core) :
    FilteredStates rows width degree predicate ≃ Sigma (fun core : FilteredTypes degree predicate =>
      DimensionFiber core.val.val rows width) where
  toFun := fun source =>
    ⟨⟨normalType source.val,
      (invariant (normalForm source.val.val) ⟨source.val.val, rfl⟩).mp source.property⟩,
      ⟨⟨source.val.val, rfl⟩, source.val.property.1, source.val.property.2.1⟩⟩
  invFun := fun source =>
    ⟨⟨source.snd.val.val, source.snd.property.1, source.snd.property.2,
      (normalFiber_extrema source.fst.val.val source.snd.val).trans source.fst.val.property.2⟩,
      (invariant source.fst.val.val source.snd.val).mpr source.fst.property⟩
  left_inv := fun _ => Subtype.ext (Subtype.ext rfl)
  right_inv := by
    rintro ⟨⟨⟨core, coreReduced, coreCount⟩, coreProperty⟩, ⟨⟨source, sourceNormal⟩, dimensions⟩⟩
    cases sourceNormal
    rfl

theorem filteredStates_finite (rows width degree : Nat) (predicate : GridState -> Prop) :
    Finite (FilteredStates rows width degree predicate) := by
  letI : Finite (FixedStates rows width degree) := fixedStates_finite rows width degree
  infer_instance

theorem filteredStates_card (rows width degree : Nat) (predicate : GridState -> Prop)
    (invariant : ∀ core : GridState, ∀ source : NormalFiber core,
      predicate source.val ↔ predicate core) :
    Nat.card (FilteredStates rows width degree predicate) =
      ∑ core : FilteredTypes degree predicate, Nat.card (DimensionFiber core.val.val rows width) := by
  letI (core : FilteredTypes degree predicate) : Finite (DimensionFiber core.val.val rows width) :=
    dimensionFiber_finite_all core.val.val rows width
  exact (Nat.card_congr (filteredStatesFiberEquiv rows width degree predicate invariant)).trans Nat.card_sigma

def HasSplit (minima maxima : Nat) (source : GridState) : Prop :=
  SignedExtrema.count source.columns.height SignedExtrema.Kind.minimum = minima ∧
    SignedExtrema.count source.columns.height SignedExtrema.Kind.maximum = maxima

theorem normalFiber_hasSplit_iff (core : GridState) (source : NormalFiber core) (minima maxima : Nat) :
    HasSplit minima maxima source.val ↔ HasSplit minima maxima core := by
  unfold HasSplit
  rw [normalFiber_signedCount core source SignedExtrema.Kind.minimum,
    normalFiber_signedCount core source SignedExtrema.Kind.maximum]

theorem splitStates_card (rows width degree minima maxima : Nat) :
    Nat.card (FilteredStates rows width degree (HasSplit minima maxima)) =
      ∑ core : FilteredTypes degree (HasSplit minima maxima),
        Nat.card (DimensionFiber core.val.val rows width) :=
  filteredStates_card rows width degree (HasSplit minima maxima)
    (fun core source => normalFiber_hasSplit_iff core source minima maxima)

theorem nonseparableSplitStates_card (rows width degree minima maxima : Nat) :
    Nat.card (FilteredStates rows width degree (fun source =>
      HasSplit minima maxima source ∧ ¬ AdditivelySeparable source.columns.height)) =
      ∑ core : FilteredTypes degree (fun source =>
        HasSplit minima maxima source ∧ ¬ AdditivelySeparable source.columns.height),
        Nat.card (DimensionFiber core.val.val rows width) := by
  apply filteredStates_card
  intro core source
  exact and_congr (normalFiber_hasSplit_iff core source minima maxima)
    (not_congr (normalFiber_separable_iff core source))

noncomputable def coreWeight (core : GridState) (rows width : Nat) : Nat :=
  if 0 < numFrozenRuns (swapH core.columns.height) ∧ 0 < numFrozenRuns core.columns.height then
    (rows - core.rows + numFrozenRuns (swapH core.columns.height) - 1).choose
      (numFrozenRuns (swapH core.columns.height) - 1) *
    (width - core.columns.width + numFrozenRuns core.columns.height - 1).choose
      (numFrozenRuns core.columns.height - 1)
  else 0

theorem dimensionFiber_stable_card (core : GridState) (reduced : Reduced core)
    (degree : Nat) (degreeBound : 3 ≤ degree) (counts : numExtrema core.columns.height = degree)
    (rows width : Nat) (rowBound : 2 * degree - 1 ≤ rows) (columnBound : 2 * degree - 1 ≤ width) :
    Nat.card (DimensionFiber core rows width) = coreWeight core rows width := by
  have coreBounds := reduced_bounds core reduced degree counts
  unfold coreWeight
  split_ifs with positive
  · exact dimensionFiber_card core reduced rows width (coreBounds.1.trans rowBound)
      (coreBounds.2.1.trans columnBound) positive.1 positive.2
  · have noRuns : numFrozenRuns (swapH core.columns.height) = 0 ∨
        numFrozenRuns core.columns.height = 0 := by omega
    letI : IsEmpty (DimensionFiber core rows width) :=
      dimensionFiber_zero_runs_empty_stable core reduced degree degreeBound counts
        rows width rowBound columnBound noRuns
    exact Nat.card_of_isEmpty

theorem filteredStates_stable_card (rows width degree : Nat) (degreeBound : 3 ≤ degree)
    (rowBound : 2 * degree - 1 ≤ rows) (columnBound : 2 * degree - 1 ≤ width)
    (predicate : GridState -> Prop)
    (invariant : ∀ core : GridState, ∀ source : NormalFiber core,
      predicate source.val ↔ predicate core) :
    Nat.card (FilteredStates rows width degree predicate) =
      ∑ core : FilteredTypes degree predicate, coreWeight core.val.val rows width := by
  rw [filteredStates_card rows width degree predicate invariant]
  apply Finset.sum_congr rfl
  intro core _
  exact dimensionFiber_stable_card core.val.val core.val.property.1 degree degreeBound
    core.val.property.2 rows width rowBound columnBound

theorem splitStates_stable_card (rows width degree minima maxima : Nat) (degreeBound : 3 ≤ degree)
    (rowBound : 2 * degree - 1 ≤ rows) (columnBound : 2 * degree - 1 ≤ width) :
    Nat.card (FilteredStates rows width degree (HasSplit minima maxima)) =
      ∑ core : FilteredTypes degree (HasSplit minima maxima), coreWeight core.val.val rows width :=
  filteredStates_stable_card rows width degree degreeBound rowBound columnBound
    (HasSplit minima maxima) (fun core source => normalFiber_hasSplit_iff core source minima maxima)

theorem nonseparableSplitStates_stable_card (rows width degree minima maxima : Nat)
    (degreeBound : 3 ≤ degree) (rowBound : 2 * degree - 1 ≤ rows) (columnBound : 2 * degree - 1 ≤ width) :
    Nat.card (FilteredStates rows width degree (fun source =>
      HasSplit minima maxima source ∧ ¬ AdditivelySeparable source.columns.height)) =
      ∑ core : FilteredTypes degree (fun source =>
        HasSplit minima maxima source ∧ ¬ AdditivelySeparable source.columns.height),
        coreWeight core.val.val rows width := by
  apply filteredStates_stable_card rows width degree degreeBound rowBound columnBound
  intro core source
  exact and_congr (normalFiber_hasSplit_iff core source minima maxima)
    (not_congr (normalFiber_separable_iff core source))

noncomputable def signedExtremumEquiv {rows width : Nat} (height : Cell rows width -> Int)
    (nontrivial : 2 ≤ rows * width) :
    {point : Cell rows width // IsStrictLocalExtremum height point} ≃
      (SignedExtrema.Sites height SignedExtrema.Kind.minimum ⊕
        SignedExtrema.Sites height SignedExtrema.Kind.maximum) where
  toFun := fun point => if minimum : IsStrictLocalMin height point.val then
    Sum.inl ⟨point.val, minimum⟩ else Sum.inr ⟨point.val, point.property.resolve_right minimum⟩
  invFun := fun point => match point with
    | Sum.inl minimum => ⟨minimum.val, Or.inr minimum.property⟩
    | Sum.inr maximum => ⟨maximum.val, Or.inl maximum.property⟩
  left_inv := by
    intro point
    dsimp only
    split_ifs <;> rfl
  right_inv := by
    rintro (minimum | maximum)
    · have property : IsStrictLocalMin height minimum.val := minimum.property
      simp only [dif_pos property]
    · have notMinimum : ¬ IsStrictLocalMin height maximum.val :=
        fun minimum => max_min_excl nontrivial maximum.property minimum
      simp only [dif_neg notMinimum]

theorem numExtrema_eq_signed_counts {rows width : Nat} (height : Cell rows width -> Int)
    (nontrivial : 2 ≤ rows * width) :
    numExtrema height = SignedExtrema.count height SignedExtrema.Kind.minimum +
      SignedExtrema.count height SignedExtrema.Kind.maximum := by
  classical
  have signed := Nat.card_congr (signedExtremumEquiv height nontrivial)
  rw [Nat.card_sum] at signed
  have unsigned : Nat.card {point : Cell rows width // IsStrictLocalExtremum height point} = numExtrema height := by
    let equivalence : {point : Cell rows width // IsStrictLocalExtremum height point} ≃
        (Finset.univ.filter (IsStrictLocalExtremum height)) :=
      Equiv.subtypeEquivRight (by intro point; simp)
    rw [Nat.card_congr equivalence, Nat.card_eq_fintype_card, Fintype.card_coe]
    rfl
  rw [unsigned] at signed
  exact signed

theorem hasSplit_degree (source : GridState) (minima maxima : Nat)
    (split : HasSplit minima maxima source) : numExtrema source.columns.height = minima + maxima := by
  have total := numExtrema_eq_signed_counts source.columns.height (by
    have rowBound := source.rowsValid
    have columnBound := source.columns.widthValid
    nlinarith)
  exact total.trans (congrArg₂ Nat.add split.1 split.2)

abbrev SplitFamilyStates (rows width minima maxima : Nat) (predicate : GridState -> Prop) :=
  {source : GridState // source.rows = rows ∧ source.columns.width = width ∧
    HasSplit minima maxima source ∧ predicate source}

noncomputable def splitFamilyEquiv (rows width minima maxima : Nat) (predicate : GridState -> Prop) :
    SplitFamilyStates rows width minima maxima predicate ≃
      FilteredStates rows width (minima + maxima) (fun source => HasSplit minima maxima source ∧ predicate source) where
  toFun := fun source =>
    ⟨⟨source.val, source.property.1, source.property.2.1,
      hasSplit_degree source.val minima maxima source.property.2.2.1⟩, source.property.2.2⟩
  invFun := fun source => ⟨source.val.val, source.val.property.1, source.val.property.2.1, source.property⟩
  left_inv := fun _ => Subtype.ext rfl
  right_inv := fun _ => Subtype.ext (Subtype.ext rfl)

theorem splitFamilyStates_finite (rows width minima maxima : Nat) (predicate : GridState -> Prop) :
    Finite (SplitFamilyStates rows width minima maxima predicate) := by
  letI : Finite (FilteredStates rows width (minima + maxima)
      (fun source => HasSplit minima maxima source ∧ predicate source)) := filteredStates_finite _ _ _ _
  exact Finite.of_equiv _ (splitFamilyEquiv rows width minima maxima predicate).symm

theorem splitFamilyStates_stable_card (rows width minima maxima : Nat)
    (degreeBound : 3 ≤ minima + maxima)
    (rowBound : 2 * (minima + maxima) - 1 ≤ rows)
    (columnBound : 2 * (minima + maxima) - 1 ≤ width)
    (predicate : GridState -> Prop)
    (invariant : ∀ core : GridState, ∀ source : NormalFiber core,
      predicate source.val ↔ predicate core) :
    Nat.card (SplitFamilyStates rows width minima maxima predicate) =
      ∑ core : FilteredTypes (minima + maxima) (fun source =>
        HasSplit minima maxima source ∧ predicate source), coreWeight core.val.val rows width := by
  rw [Nat.card_congr (splitFamilyEquiv rows width minima maxima predicate)]
  apply filteredStates_stable_card rows width (minima + maxima) degreeBound rowBound columnBound
  intro core source
  exact and_congr (normalFiber_hasSplit_iff core source minima maxima) (invariant core source)

#print axioms reducedTypes_finite
#print axioms fixedStates_finite
#print axioms fixedStatesFiberEquiv
#print axioms dimensionFiber_finite_all
#print axioms fixedStates_card
#print axioms filteredStatesFiberEquiv
#print axioms filteredStates_finite
#print axioms filteredStates_card
#print axioms normalFiber_hasSplit_iff
#print axioms splitStates_card
#print axioms nonseparableSplitStates_card
#print axioms dimensionFiber_stable_card
#print axioms filteredStates_stable_card
#print axioms splitStates_stable_card
#print axioms nonseparableSplitStates_stable_card
#print axioms signedExtremumEquiv
#print axioms numExtrema_eq_signed_counts
#print axioms hasSplit_degree
#print axioms splitFamilyEquiv
#print axioms splitFamilyStates_finite
#print axioms splitFamilyStates_stable_card

end OrigamiCone.Sequel.JointTypePartition
