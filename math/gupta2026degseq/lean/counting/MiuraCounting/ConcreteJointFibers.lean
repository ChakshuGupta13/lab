import MiuraCounting.ConcreteJointNormalForm
import MiuraCounting.ConcreteRunFiberEquivalence
import MiuraCounting.ConcreteSignedExtrema

namespace OrigamiCone.Sequel.JointFibers

open OrigamiCone FrozenExpansion RunLengthFibers JointReduction

open scoped BigOperators

def RankedColumnStep (rank : Nat) (source target : GridState) : Prop :=
  ∃ next : FrozenReduction.CanonicalState source.rows,
    RunFiberEquivalence.RankedStep (rows_positive source) rank source.columns next ∧
      target = {source with columns := next}

theorem rankedColumnStep_step {rank : Nat} {source target : GridState}
    (step : RankedColumnStep rank source target) : ColumnStep source target := by
  obtain ⟨next, oldStep, rfl⟩ := step
  exact ⟨next, RunFiberEquivalence.rankedStep_step (rows_positive source) rank oldStep, rfl⟩

theorem rankedColumnStep_lengths {rank : Nat} {source target : GridState}
    (step : RankedColumnStep rank source target) (index : Nat) :
    runLengthAtRank source.columns.height index = runLengthAtRank target.columns.height index +
      if rank = index then 1 else 0 := by
  obtain ⟨next, oldStep, rfl⟩ := step
  exact RunFiberEquivalence.rankedStep_lengths (rows_positive source) rank oldStep index

theorem rankedColumnStep_rowProfiles {rank : Nat} {source target : GridState}
    (step : RankedColumnStep rank source target) :
    numFrozenRuns (swapH target.columns.height) = numFrozenRuns (swapH source.columns.height) ∧
      ∀ index : Nat, runLengthAtRank (swapH target.columns.height) index =
        runLengthAtRank (swapH source.columns.height) index := by
  obtain ⟨next, oldStep, rfl⟩ := step
  exact columnStep_rowProfiles source (RunFiberEquivalence.rankedStep_step (rows_positive source) rank oldStep)

theorem rankedColumnStep_exists (rank : Nat) (source : GridState)
    (longRun : 1 < runLengthAtRank source.columns.height rank) :
    ∃ target : GridState, RankedColumnStep rank source target := by
  obtain ⟨next, oldStep⟩ := RunFiberEquivalence.rankedStep_exists (rows_positive source) rank source.columns longRun
  exact ⟨{source with columns := next}, next, oldStep, rfl⟩

theorem rankedColumnStep_sources_unique {rank : Nat} {first second target : GridState}
    (firstStep : RankedColumnStep rank first target) (secondStep : RankedColumnStep rank second target) :
    first = second := by
  have sameRows : first.rows = second.rows :=
    (columnStep_rows (rankedColumnStep_step firstStep)).symm.trans
      (columnStep_rows (rankedColumnStep_step secondStep))
  rcases first with ⟨rows, rowsValid, firstColumns⟩
  rcases second with ⟨otherRows, otherRowsValid, secondColumns⟩
  cases sameRows
  obtain ⟨firstNext, firstOld, rfl⟩ := firstStep
  obtain ⟨secondNext, secondOld, gridEquality⟩ := secondStep
  have nextEquality : firstNext = secondNext := by
    cases gridEquality
    rfl
  rw [← nextEquality] at secondOld
  have columnsEquality := RunFiberEquivalence.rankedStep_sources_unique
    (rows_positive ⟨rows, rowsValid, firstColumns⟩) rank firstOld secondOld
  refine state_eq_of_rows_columns _ _ ?_ ?_
  · rfl
  · exact heq_of_eq columnsEquality

inductive Axis where
  | row
  | column
  deriving DecidableEq

noncomputable def runProfile (source : GridState) (axis : Axis) (rank : Nat) : Nat :=
  match axis with
  | Axis.row => runLengthAtRank (swapH source.columns.height) rank
  | Axis.column => runLengthAtRank source.columns.height rank

def RankedStep (axis : Axis) (rank : Nat) (source target : GridState) : Prop :=
  match axis with
  | Axis.row => RankedColumnStep rank (transpose source) (transpose target)
  | Axis.column => RankedColumnStep rank source target

theorem rankedStep_step {axis : Axis} {rank : Nat} {source target : GridState}
    (step : RankedStep axis rank source target) : Step source target := by
  cases axis with
  | row => exact Or.inr (rankedColumnStep_step step)
  | column => exact Or.inl (rankedColumnStep_step step)

theorem rankedStep_lengths {axis : Axis} {rank : Nat} {source target : GridState}
    (step : RankedStep axis rank source target) (probe : Axis) (index : Nat) :
    runProfile source probe index = runProfile target probe index +
      if axis = probe ∧ rank = index then 1 else 0 := by
  cases axis with
  | row =>
    cases probe with
    | row => simpa [runProfile] using rankedColumnStep_lengths step index
    | column => simpa [runProfile] using ((rankedColumnStep_rowProfiles step).2 index).symm
  | column =>
    cases probe with
    | row => simpa [runProfile] using ((rankedColumnStep_rowProfiles step).2 index).symm
    | column => simpa [runProfile] using rankedColumnStep_lengths step index

theorem rankedStep_exists (axis : Axis) (rank : Nat) (source : GridState)
    (longRun : 1 < runProfile source axis rank) : ∃ target : GridState, RankedStep axis rank source target := by
  cases axis with
  | row =>
    obtain ⟨target, step⟩ := rankedColumnStep_exists rank (transpose source) longRun
    refine ⟨transpose target, ?_⟩
    change RankedColumnStep rank (transpose source) (transpose (transpose target))
    rwa [transpose_involutive]
  | column => exact rankedColumnStep_exists rank source longRun

theorem rankedStep_sources_unique {axis : Axis} {rank : Nat} {first second target : GridState}
    (firstStep : RankedStep axis rank first target) (secondStep : RankedStep axis rank second target) :
    first = second := by
  cases axis with
  | row =>
    have equality := rankedColumnStep_sources_unique firstStep secondStep
    exact (transpose_involutive first).symm.trans ((congrArg transpose equality).trans (transpose_involutive second))
  | column => exact rankedColumnStep_sources_unique firstStep secondStep

theorem reduced_iff_profiles (source : GridState) : Reduced source ↔
    ∀ axis : Axis, ∀ rank : Nat, runProfile source axis rank ≤ 1 := by
  constructor
  · intro reduced axis
    cases axis with
    | row => exact (RunFiberEquivalence.reduced_iff_runLength_le_one (rows_positive (transpose source))
        (transpose source).columns).mp reduced.2
    | column => exact (RunFiberEquivalence.reduced_iff_runLength_le_one (rows_positive source) source.columns).mp reduced.1
  · intro bounded
    exact ⟨(RunFiberEquivalence.reduced_iff_runLength_le_one (rows_positive source) source.columns).mpr (bounded Axis.column),
      (RunFiberEquivalence.reduced_iff_runLength_le_one (rows_positive (transpose source))
        (transpose source).columns).mpr (bounded Axis.row)⟩

theorem normalForm_profiles_injective (first second : GridState)
    (sameNormal : normalForm first = normalForm second)
    (sameProfiles : ∀ axis : Axis, ∀ rank : Nat, runProfile first axis rank = runProfile second axis rank) :
    first = second := by
  have general : ∀ measure : Nat, ∀ source : GridState, source.rows + source.columns.width = measure ->
      ∀ other : GridState, normalForm source = normalForm other ->
        (∀ axis : Axis, ∀ rank : Nat, runProfile source axis rank = runProfile other axis rank) ->
        source = other := by
    intro measure
    induction measure using Nat.strong_induction_on with
    | h measure inductionHypothesis =>
      intro source measureEquality other sameNormal sameProfiles
      by_cases reduced : Reduced source
      · have otherReduced : Reduced other := (reduced_iff_profiles other).mpr (by
          intro axis rank
          rw [← sameProfiles axis rank]
          exact (reduced_iff_profiles source).mp reduced axis rank)
        exact (normalForm_fixed source reduced).symm.trans (sameNormal.trans (normalForm_fixed other otherReduced))
      · have notBounded : ¬ ∀ axis : Axis, ∀ rank : Nat, runProfile source axis rank ≤ 1 :=
          fun bounded => reduced ((reduced_iff_profiles source).mpr bounded)
        obtain ⟨axis, notBoundedAxis⟩ := not_forall.mp notBounded
        obtain ⟨rank, notBoundedRank⟩ := not_forall.mp notBoundedAxis
        have longRun : 1 < runProfile source axis rank := by omega
        obtain ⟨sourceNext, sourceStep⟩ := rankedStep_exists axis rank source longRun
        have otherLong : 1 < runProfile other axis rank := by
          rw [← sameProfiles axis rank]
          exact longRun
        obtain ⟨otherNext, otherStep⟩ := rankedStep_exists axis rank other otherLong
        have sourcePath := Relation.ReflTransGen.single (rankedStep_step sourceStep)
        have otherPath := Relation.ReflTransGen.single (rankedStep_step otherStep)
        have nextNormal : normalForm sourceNext = normalForm otherNext :=
          (normalForm_eq_of_path sourcePath).symm.trans (sameNormal.trans (normalForm_eq_of_path otherPath))
        have nextProfiles : ∀ probe : Axis, ∀ index : Nat,
            runProfile sourceNext probe index = runProfile otherNext probe index := by
          intro probe index
          have sourceUpdate := rankedStep_lengths sourceStep probe index
          have otherUpdate := rankedStep_lengths otherStep probe index
          have originalEquality := sameProfiles probe index
          omega
        have shorter := step_measure (rankedStep_step sourceStep)
        have nextEqual := inductionHypothesis (sourceNext.rows + sourceNext.columns.width)
          (by omega) sourceNext rfl otherNext nextNormal nextProfiles
        rw [← nextEqual] at otherStep
        exact rankedStep_sources_unique sourceStep otherStep
  exact general (first.rows + first.columns.width) first rfl second sameNormal sameProfiles

theorem compositionProfile_cast {firstRuns secondRuns : Nat} (counts : firstRuns = secondRuns)
    (extras : Fin secondRuns -> Nat) (rank : Nat) :
    RunFiberEquivalence.compositionProfile (fun index : Fin firstRuns => extras (Fin.cast counts index)) rank =
      RunFiberEquivalence.compositionProfile extras rank := by
  cases counts
  rfl

theorem columnNormalForm_eq_of_old (core : GridState)
    (source : FrozenReduction.CanonicalState core.rows)
    (oldNormal : FrozenReduction.normalForm (rows_positive core) source = core.columns) :
    columnNormalForm {core with columns := source} = core := by
  refine state_eq_of_rows_columns _ _ ?_ ?_
  · rfl
  · exact heq_of_eq oldNormal

theorem rowNormalForm_eq_of_old (core : GridState)
    (source : FrozenReduction.CanonicalState (transpose core).rows)
    (oldNormal : FrozenReduction.normalForm (rows_positive (transpose core)) source = (transpose core).columns) :
    rowNormalForm (transpose {transpose core with columns := source}) = core := by
  have equality := congrArg transpose (columnNormalForm_eq_of_old (transpose core) source oldNormal)
  simpa only [rowNormalForm, transpose_involutive] using equality

theorem normalForm_eq_of_phases {source intermediate core : GridState}
    (columnPhase : columnNormalForm source = intermediate) (rowPhase : rowNormalForm intermediate = core) :
    normalForm source = core :=
  (column_then_row_eq_normalForm source).symm.trans ((congrArg rowNormalForm columnPhase).trans rowPhase)

theorem profiles_realized (core : GridState) (reduced : Reduced core)
    (rowExtras : Fin (numFrozenRuns (swapH core.columns.height)) -> Nat)
    (columnExtras : Fin (numFrozenRuns core.columns.height) -> Nat) :
    ∃ source : GridState, normalForm source = core ∧
      (∀ rank : Nat, runProfile source Axis.row rank = RunFiberEquivalence.compositionProfile rowExtras rank) ∧
      ∀ rank : Nat, runProfile source Axis.column rank = RunFiberEquivalence.compositionProfile columnExtras rank := by
  obtain ⟨rowSource, rowNormal, rowProfile⟩ := RunFiberEquivalence.compositionProfile_realized
    (rows_positive (transpose core)) (transpose core).columns reduced.2 rowExtras
  let intermediate := transpose {transpose core with columns := rowSource}
  have rowPhase : rowNormalForm intermediate = core := rowNormalForm_eq_of_old core rowSource rowNormal
  have columnsReduced : FrozenReduction.Reduced intermediate.columns :=
    (rowNormalForm_columns_reduced_iff intermediate).mp (by rw [rowPhase]; exact reduced.1)
  have columnPreserved := rowNormalForm_columnProfiles intermediate
  rw [rowPhase] at columnPreserved
  have counts : numFrozenRuns intermediate.columns.height = numFrozenRuns core.columns.height := columnPreserved.1.symm
  let adaptedColumns : Fin (numFrozenRuns intermediate.columns.height) -> Nat :=
    fun index => columnExtras (Fin.cast counts index)
  obtain ⟨columnSource, columnNormal, columnProfile⟩ := RunFiberEquivalence.compositionProfile_realized
    (rows_positive intermediate) intermediate.columns columnsReduced adaptedColumns
  let source : GridState := {intermediate with columns := columnSource}
  have columnPhase : columnNormalForm source = intermediate :=
    columnNormalForm_eq_of_old intermediate columnSource columnNormal
  have rowPreserved := columnNormalForm_rowProfiles source
  rw [columnPhase] at rowPreserved
  refine ⟨source, normalForm_eq_of_phases columnPhase rowPhase, ?_, ?_⟩
  · intro rank
    exact ((rowPreserved.2 rank).symm).trans (rowProfile rank)
  · intro rank
    exact (columnProfile rank).trans (compositionProfile_cast counts columnExtras rank)

theorem lengthProfile_eq_composition {rows width runs : Nat} (height : Cell rows width -> Int)
    (counts : numFrozenRuns height = runs) (rank : Nat) :
    runLengthAtRank height rank = RunFiberEquivalence.compositionProfile
      (fun index : Fin runs => runLengthAtRank height (index.val + 1) - 1) rank := by
  by_cases within : 0 < rank ∧ rank ≤ runs
  · let index : Fin (numFrozenRuns height) := ⟨rank - 1, by rw [counts]; omega⟩
    have indexRank : index.val + 1 = rank := by dsimp [index]; omega
    have positive : 0 < runLengthAtRank height rank := by
      have atIndex : 0 < runLengthAtRank height (index.val + 1) := runLengthTuple_pos height index
      rwa [indexRank] at atIndex
    rw [RunFiberEquivalence.compositionProfile, dif_pos within]
    change runLengthAtRank height rank = (runLengthAtRank height (rank - 1 + 1) - 1) + 1
    rw [show rank - 1 + 1 = rank from by omega]
    exact (Nat.sub_add_cancel (Nat.succ_le_of_lt positive)).symm
  · have outside : ¬ (0 < rank ∧ rank ≤ numFrozenRuns height) := by simpa only [counts] using within
    rw [RunFiberEquivalence.runLengthAtRank_eq_zero_of_outside height rank outside]
    simp [RunFiberEquivalence.compositionProfile, within]

abbrev NormalFiber (core : GridState) := {source : GridState // normalForm source = core}

abbrev ExtraLists (core : GridState) :=
  (Fin (numFrozenRuns (swapH core.columns.height)) -> Nat) × (Fin (numFrozenRuns core.columns.height) -> Nat)

noncomputable def fiberExtras (core : GridState) (source : NormalFiber core) : ExtraLists core :=
  ⟨fun index => runProfile source.val Axis.row (index.val + 1) - 1,
    fun index => runProfile source.val Axis.column (index.val + 1) - 1⟩

theorem fiber_rowProfile (core : GridState) (source : NormalFiber core) (rank : Nat) :
    runProfile source.val Axis.row rank =
      RunFiberEquivalence.compositionProfile (fiberExtras core source).1 rank := by
  have counts := normalForm_rowRuns source.val
  rw [source.property] at counts
  exact lengthProfile_eq_composition (swapH source.val.columns.height) counts.symm rank

theorem fiber_columnProfile (core : GridState) (source : NormalFiber core) (rank : Nat) :
    runProfile source.val Axis.column rank =
      RunFiberEquivalence.compositionProfile (fiberExtras core source).2 rank := by
  have counts := normalForm_columnRuns source.val
  rw [source.property] at counts
  exact lengthProfile_eq_composition source.val.columns.height counts.symm rank

noncomputable def normalFiberEquiv (core : GridState) (reduced : Reduced core) :
    NormalFiber core ≃ ExtraLists core :=
  Equiv.ofBijective (fiberExtras core) (by
    constructor
    · intro first second equality
      apply Subtype.ext
      apply normalForm_profiles_injective first.val second.val (first.property.trans second.property.symm)
      intro axis rank
      cases axis with
      | row => rw [fiber_rowProfile core first rank, fiber_rowProfile core second rank, congrArg Prod.fst equality]
      | column => rw [fiber_columnProfile core first rank, fiber_columnProfile core second rank, congrArg Prod.snd equality]
    · intro extras
      obtain ⟨source, sourceNormal, rowProfile, columnProfile⟩ := profiles_realized core reduced extras.1 extras.2
      refine ⟨⟨source, sourceNormal⟩, ?_⟩
      apply Prod.ext
      · funext index
        change runProfile source Axis.row (index.val + 1) - 1 = extras.1 index
        rw [rowProfile (index.val + 1), RunFiberEquivalence.compositionProfile_at_index]
        omega
      · funext index
        change runProfile source Axis.column (index.val + 1) - 1 = extras.2 index
        rw [columnProfile (index.val + 1), RunFiberEquivalence.compositionProfile_at_index]
        omega)

theorem sum_reindexed_extras {rows width runs : Nat} (height : Cell rows width -> Int)
    (counts : numFrozenRuns height = runs) :
    (∑ index : Fin runs, (runLengthAtRank height (index.val + 1) - 1)) =
      ∑ index : Fin (numFrozenRuns height), extraLengthTuple height index := by
  classical
  apply Finset.sum_bij (fun index _ => Fin.cast counts.symm index)
  · intro index _
    exact Finset.mem_univ _
  · intro first _ second _ equality
    exact Fin.ext (congrArg (fun position : Fin (numFrozenRuns height) => position.val) equality)
  · intro index _
    refine ⟨Fin.cast counts index, Finset.mem_univ _, ?_⟩
    apply Fin.ext
    rfl
  · intro index _
    rfl

theorem width_eq_normal_add_extras {rows : Nat} (rowsPositive : 0 < rows)
    (source : FrozenReduction.CanonicalState rows) :
    source.width = (FrozenReduction.normalForm rowsPositive source).width +
      ∑ index : Fin (numFrozenRuns source.height), extraLengthTuple source.height index := by
  have total := sum_extraLengthTuple_eq_removed rowsPositive source
  have bound := FrozenReduction.reduction_width_le rowsPositive (FrozenReduction.normalForm_path rowsPositive source)
  omega

theorem normalForm_rows (source : GridState) : (normalForm source).rows = (rowNormalForm source).rows := by
  rw [← row_then_column_eq_normalForm source]
  rfl

theorem normalForm_columns (source : GridState) :
    (normalForm source).columns.width = (columnNormalForm source).columns.width := by
  rw [← column_then_row_eq_normalForm source]
  rfl

theorem fiber_dimensions (core : GridState) (source : NormalFiber core) :
    source.val.rows = core.rows +
      (∑ index : Fin (numFrozenRuns (swapH core.columns.height)), (fiberExtras core source).1 index) ∧
    source.val.columns.width = core.columns.width +
      (∑ index : Fin (numFrozenRuns core.columns.height), (fiberExtras core source).2 index) := by
  have rowCounts := normalForm_rowRuns source.val
  have columnCounts := normalForm_columnRuns source.val
  have rowSize := normalForm_rows source.val
  have columnSize := normalForm_columns source.val
  rw [source.property] at rowCounts columnCounts rowSize columnSize
  have rowTotals := sum_reindexed_extras (swapH source.val.columns.height) rowCounts.symm
  have columnTotals := sum_reindexed_extras source.val.columns.height columnCounts.symm
  have rowIdentity := width_eq_normal_add_extras (rows_positive (transpose source.val)) (transpose source.val).columns
  have columnIdentity := width_eq_normal_add_extras (rows_positive source.val) source.val.columns
  change source.val.rows = (rowNormalForm source.val).rows +
    ∑ index : Fin (numFrozenRuns (swapH source.val.columns.height)),
      extraLengthTuple (swapH source.val.columns.height) index at rowIdentity
  change source.val.columns.width = (columnNormalForm source.val).columns.width +
    ∑ index : Fin (numFrozenRuns source.val.columns.height),
      extraLengthTuple source.val.columns.height index at columnIdentity
  rw [← rowTotals, ← rowSize] at rowIdentity
  rw [← columnTotals, ← columnSize] at columnIdentity
  exact ⟨rowIdentity, columnIdentity⟩

abbrev WeakComposition (runs total : Nat) := {extras : Fin runs -> Nat // (∑ index, extras index) = total}

abbrev GrowthFiber (core : GridState) (rowTotal columnTotal : Nat) :=
  {source : NormalFiber core // source.val.rows = core.rows + rowTotal ∧
    source.val.columns.width = core.columns.width + columnTotal}

noncomputable def growthFiberEquiv (core : GridState) (reduced : Reduced core) (rowTotal columnTotal : Nat) :
    GrowthFiber core rowTotal columnTotal ≃
      WeakComposition (numFrozenRuns (swapH core.columns.height)) rowTotal ×
        WeakComposition (numFrozenRuns core.columns.height) columnTotal :=
  ((normalFiberEquiv core reduced).subtypeEquiv
    (p := fun source : NormalFiber core => source.val.rows = core.rows + rowTotal ∧
      source.val.columns.width = core.columns.width + columnTotal)
    (q := fun extras : ExtraLists core => (∑ index, extras.1 index) = rowTotal ∧
      (∑ index, extras.2 index) = columnTotal)
    (by
      intro source
      change (source.val.rows = core.rows + rowTotal ∧
        source.val.columns.width = core.columns.width + columnTotal) ↔
        ((∑ index, (fiberExtras core source).1 index) = rowTotal ∧
          (∑ index, (fiberExtras core source).2 index) = columnTotal)
      have dimensions := fiber_dimensions core source
      omega)).trans (Equiv.subtypeProdEquivProd
        (p := fun extras : Fin (numFrozenRuns (swapH core.columns.height)) -> Nat => (∑ index, extras index) = rowTotal)
        (q := fun extras : Fin (numFrozenRuns core.columns.height) -> Nat => (∑ index, extras index) = columnTotal))

theorem weakComposition_card (runs total : Nat) (positive : 0 < runs) :
    Nat.card (WeakComposition runs total) = (total + runs - 1).choose (runs - 1) := by
  rw [Nat.card_congr (RunFiberEquivalence.compositionFinsetEquiv runs total),
    Nat.card_eq_fintype_card, Fintype.card_coe]
  obtain ⟨smaller, counts⟩ : ∃ smaller : Nat, runs = smaller + 1 := ⟨runs - 1, by omega⟩
  rw [counts, antidiagonalTuple_card]
  congr 1

theorem weakComposition_card_zero (total : Nat) :
    Nat.card (WeakComposition 0 total) = if total = 0 then 1 else 0 := by
  rw [Nat.card_congr (RunFiberEquivalence.compositionFinsetEquiv 0 total),
    Nat.card_eq_fintype_card, Fintype.card_coe]
  cases total <;> simp

theorem growthFiber_finite (core : GridState) (reduced : Reduced core) (rowTotal columnTotal : Nat) :
    Finite (GrowthFiber core rowTotal columnTotal) := by
  let equivalence := (growthFiberEquiv core reduced rowTotal columnTotal).trans
    (Equiv.prodCongr
      (RunFiberEquivalence.compositionFinsetEquiv (numFrozenRuns (swapH core.columns.height)) rowTotal)
      (RunFiberEquivalence.compositionFinsetEquiv (numFrozenRuns core.columns.height) columnTotal))
  exact Finite.of_equiv _ equivalence.symm

theorem growthFiber_card (core : GridState) (reduced : Reduced core) (rowTotal columnTotal : Nat)
    (rowPositive : 0 < numFrozenRuns (swapH core.columns.height))
    (columnPositive : 0 < numFrozenRuns core.columns.height) :
    Nat.card (GrowthFiber core rowTotal columnTotal) =
      (rowTotal + numFrozenRuns (swapH core.columns.height) - 1).choose
        (numFrozenRuns (swapH core.columns.height) - 1) *
      (columnTotal + numFrozenRuns core.columns.height - 1).choose (numFrozenRuns core.columns.height - 1) := by
  rw [Nat.card_congr (growthFiberEquiv core reduced rowTotal columnTotal), Nat.card_prod,
    weakComposition_card _ _ rowPositive, weakComposition_card _ _ columnPositive]

abbrev DimensionFiber (core : GridState) (rows width : Nat) :=
  {source : NormalFiber core // source.val.rows = rows ∧ source.val.columns.width = width}

noncomputable def dimensionGrowthEquiv (core : GridState) (rows width : Nat)
    (rowBound : core.rows ≤ rows) (columnBound : core.columns.width ≤ width) :
    DimensionFiber core rows width ≃ GrowthFiber core (rows - core.rows) (width - core.columns.width) :=
  Equiv.subtypeEquivRight (by intro source; omega)

theorem dimensionFiber_finite (core : GridState) (reduced : Reduced core) (rows width : Nat)
    (rowBound : core.rows ≤ rows) (columnBound : core.columns.width ≤ width) :
    Finite (DimensionFiber core rows width) := by
  letI : Finite (GrowthFiber core (rows - core.rows) (width - core.columns.width)) :=
    growthFiber_finite core reduced _ _
  exact Finite.of_equiv _ (dimensionGrowthEquiv core rows width rowBound columnBound).symm

theorem dimensionFiber_card (core : GridState) (reduced : Reduced core) (rows width : Nat)
    (rowBound : core.rows ≤ rows) (columnBound : core.columns.width ≤ width)
    (rowPositive : 0 < numFrozenRuns (swapH core.columns.height))
    (columnPositive : 0 < numFrozenRuns core.columns.height) :
    Nat.card (DimensionFiber core rows width) =
      (rows - core.rows + numFrozenRuns (swapH core.columns.height) - 1).choose
        (numFrozenRuns (swapH core.columns.height) - 1) *
      (width - core.columns.width + numFrozenRuns core.columns.height - 1).choose
        (numFrozenRuns core.columns.height - 1) :=
  (Nat.card_congr (dimensionGrowthEquiv core rows width rowBound columnBound)).trans
    (growthFiber_card core reduced _ _ rowPositive columnPositive)

theorem dimensionFiber_zero_rows_empty (core : GridState) (rows width : Nat)
    (differentRows : rows ≠ core.rows) (noRows : numFrozenRuns (swapH core.columns.height) = 0) :
    IsEmpty (DimensionFiber core rows width) := by
  refine ⟨fun source => ?_⟩
  have zeroTotal : (∑ index : Fin (numFrozenRuns (swapH core.columns.height)),
      (fiberExtras core source.val).1 index) = 0 := by
    apply Finset.sum_eq_zero
    intro index _
    have bound := index.isLt
    omega
  have rowIdentity := (fiber_dimensions core source.val).1
  rw [zeroTotal, Nat.add_zero] at rowIdentity
  exact differentRows (source.property.1.symm.trans rowIdentity)

theorem dimensionFiber_zero_columns_empty (core : GridState) (rows width : Nat)
    (differentColumns : width ≠ core.columns.width) (noColumns : numFrozenRuns core.columns.height = 0) :
    IsEmpty (DimensionFiber core rows width) := by
  refine ⟨fun source => ?_⟩
  have zeroTotal : (∑ index : Fin (numFrozenRuns core.columns.height),
      (fiberExtras core source.val).2 index) = 0 := by
    apply Finset.sum_eq_zero
    intro index _
    have bound := index.isLt
    omega
  have columnIdentity := (fiber_dimensions core source.val).2
  rw [zeroTotal, Nat.add_zero] at columnIdentity
  exact differentColumns (source.property.2.symm.trans columnIdentity)

theorem dimensionFiber_zero_runs_empty_stable (core : GridState) (reduced : Reduced core)
    (degree : Nat) (degreeBound : 3 ≤ degree) (counts : numExtrema core.columns.height = degree)
    (rows width : Nat) (rowBound : 2 * degree - 1 ≤ rows) (columnBound : 2 * degree - 1 ≤ width)
    (noRuns : numFrozenRuns (swapH core.columns.height) = 0 ∨ numFrozenRuns core.columns.height = 0) :
    IsEmpty (DimensionFiber core rows width) := by
  rcases noRuns with noRows | noColumns
  · have bound := reduced_zero_row_runs_bound core reduced degree counts noRows
    exact dimensionFiber_zero_rows_empty core rows width (by omega) noRows
  · have bound := reduced_zero_column_runs_bound core reduced.1 degree counts noColumns
    exact dimensionFiber_zero_columns_empty core rows width (by omega) noColumns

theorem normalFiber_extrema (core : GridState) (source : NormalFiber core) :
    numExtrema source.val.columns.height = numExtrema core.columns.height := by
  have counts := JointReduction.normalForm_extrema source.val
  rw [source.property] at counts
  exact counts.symm

theorem normalFiber_separable_iff (core : GridState) (source : NormalFiber core) :
    AdditivelySeparable source.val.columns.height ↔ AdditivelySeparable core.columns.height := by
  have equivalent := normalForm_separable_iff source.val
  rw [source.property] at equivalent
  exact equivalent.symm

theorem normalFiber_signedEquiv (core : GridState) (source : NormalFiber core) (kind : SignedExtrema.Kind) :
    Nonempty (SignedExtrema.Sites source.val.columns.height kind ≃ SignedExtrema.Sites core.columns.height kind) := by
  have equivalent := SignedExtrema.normalForm_equiv source.val kind
  rw [source.property] at equivalent
  obtain ⟨equivalence⟩ := equivalent
  exact ⟨equivalence.symm⟩

theorem normalFiber_signedCount (core : GridState) (source : NormalFiber core) (kind : SignedExtrema.Kind) :
    SignedExtrema.count source.val.columns.height kind = SignedExtrema.count core.columns.height kind := by
  obtain ⟨equivalence⟩ := normalFiber_signedEquiv core source kind
  exact Nat.card_congr equivalence

theorem normalFiber_split (core : GridState) (source : NormalFiber core) :
    (SignedExtrema.count source.val.columns.height SignedExtrema.Kind.minimum,
      SignedExtrema.count source.val.columns.height SignedExtrema.Kind.maximum) =
    (SignedExtrema.count core.columns.height SignedExtrema.Kind.minimum,
      SignedExtrema.count core.columns.height SignedExtrema.Kind.maximum) :=
  Prod.ext (normalFiber_signedCount core source SignedExtrema.Kind.minimum)
    (normalFiber_signedCount core source SignedExtrema.Kind.maximum)

#print axioms rankedColumnStep_step
#print axioms rankedColumnStep_lengths
#print axioms rankedColumnStep_rowProfiles
#print axioms rankedColumnStep_exists
#print axioms rankedColumnStep_sources_unique
#print axioms rankedStep_step
#print axioms rankedStep_lengths
#print axioms rankedStep_exists
#print axioms rankedStep_sources_unique
#print axioms reduced_iff_profiles
#print axioms normalForm_profiles_injective
#print axioms compositionProfile_cast
#print axioms columnNormalForm_eq_of_old
#print axioms rowNormalForm_eq_of_old
#print axioms normalForm_eq_of_phases
#print axioms profiles_realized
#print axioms lengthProfile_eq_composition
#print axioms fiber_rowProfile
#print axioms fiber_columnProfile
#print axioms normalFiberEquiv
#print axioms sum_reindexed_extras
#print axioms width_eq_normal_add_extras
#print axioms normalForm_rows
#print axioms normalForm_columns
#print axioms fiber_dimensions
#print axioms growthFiberEquiv
#print axioms weakComposition_card
#print axioms weakComposition_card_zero
#print axioms growthFiber_finite
#print axioms growthFiber_card
#print axioms dimensionGrowthEquiv
#print axioms dimensionFiber_finite
#print axioms dimensionFiber_card
#print axioms dimensionFiber_zero_rows_empty
#print axioms dimensionFiber_zero_columns_empty
#print axioms dimensionFiber_zero_runs_empty_stable
#print axioms normalFiber_extrema
#print axioms normalFiber_separable_iff
#print axioms normalFiber_signedEquiv
#print axioms normalFiber_signedCount
#print axioms normalFiber_split

end OrigamiCone.Sequel.JointFibers
