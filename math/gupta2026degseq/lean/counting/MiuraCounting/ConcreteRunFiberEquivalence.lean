import MiuraCounting.ConcreteRunLengthFibers
import OrigamiCone.SequelEdStarsBars

namespace OrigamiCone.Sequel.RunFiberEquivalence

open OrigamiCone FrozenExpansion FrozenReduction RunLengthFibers

open scoped BigOperators

noncomputable def expandState {rows : Nat} (rowsPositive : 0 < rows)
    (source : CanonicalState rows) (mark : Fin source.width)
    (frozen : frozenColumn source.height mark) : CanonicalState rows where
  width := source.width + 1
  widthValid := by have := source.widthValid; omega
  height := expandAfter source.height mark (frozenSlope source.height rowsPositive mark frozen)
  canonical := expandAfter_canonical source.height source.canonical mark _
    (frozenSlope_pm source.height source.canonical.1 rowsPositive mark frozen)

def RankedStep {rows : Nat} (rowsPositive : 0 < rows) (rank : Nat)
    (source target : CanonicalState rows) : Prop :=
  ∃ column : Fin source.width, ∃ frozen : frozenColumn source.height column,
    ∃ nextFrozen : frozenColumn source.height ⟨column.val + 1, frozen.2.1⟩,
      target = contractState rowsPositive source column frozen nextFrozen ∧
        runPrefixCount source.height column = rank

theorem rankedStep_step {rows : Nat} (rowsPositive : 0 < rows) (rank : Nat)
    {source target : CanonicalState rows} (step : RankedStep rowsPositive rank source target) :
    Step rowsPositive source target := by
  obtain ⟨column, frozen, nextFrozen, equality, _⟩ := step
  exact ⟨column, frozen, nextFrozen, equality⟩

theorem rankedStep_lengths {rows : Nat} (rowsPositive : 0 < rows) (rank : Nat)
    {source target : CanonicalState rows} (step : RankedStep rowsPositive rank source target)
    (index : Nat) :
    runLengthAtRank source.height index = runLengthAtRank target.height index +
      if rank = index then 1 else 0 := by
  obtain ⟨column, frozen, nextFrozen, rfl, atRank⟩ := step
  cases source with
  | mk width widthValid height canonical =>
    cases width with
    | zero => omega
    | succ width =>
      simpa only [atRank] using
        runLengthAtRank_contractAt height canonical.1 rowsPositive column frozen nextFrozen index

theorem rankedStep_exists {rows : Nat} (rowsPositive : 0 < rows) (rank : Nat)
    (source : CanonicalState rows) (longRun : 1 < runLengthAtRank source.height rank) :
    ∃ target : CanonicalState rows, RankedStep rowsPositive rank source target := by
  obtain ⟨column, frozen, nextFrozen, atRank⟩ := adjacent_frozen_of_runLength source.height
    source.canonical.1 rowsPositive source.widthValid rank longRun
  exact ⟨contractState rowsPositive source column frozen nextFrozen,
    column, frozen, nextFrozen, rfl, atRank⟩

theorem rankedStep_reconstructs {rows : Nat} (rowsPositive : 0 < rows) (rank : Nat)
    {source target : CanonicalState rows} (step : RankedStep rowsPositive rank source target) :
    ∃ mark : Fin target.width, ∃ frozen : frozenColumn target.height mark,
      runPrefixCount target.height mark = rank ∧ expandState rowsPositive target mark frozen = source := by
  obtain ⟨column, frozen, nextFrozen, rfl, atRank⟩ := step
  cases source with
  | mk width widthValid height canonical =>
    cases width with
    | zero => omega
    | succ width =>
      let mark : Fin width := ⟨column.val, by
        have interior := frozen.2.1
        change column.val + 1 < width + 1 at interior
        omega⟩
      have markFrozen := contracted_marker_frozen height canonical.1 rowsPositive column frozen nextFrozen
      refine ⟨mark, markFrozen,
        (runPrefixCount_contracted_marker height canonical.1 rowsPositive column frozen nextFrozen).trans atRank, ?_⟩
      refine state_eq_of_width_height _ _ ?_ ?_
      · rfl
      · exact heq_of_eq (expand_contract_at_rank height canonical.1 rowsPositive column frozen nextFrozen
          mark markFrozen (runPrefixCount_contracted_marker height canonical.1 rowsPositive column frozen nextFrozen))

theorem expandState_same_rank {rows : Nat} (rowsPositive : 0 < rows)
    (source : CanonicalState rows) (first second : Fin source.width)
    (firstFrozen : frozenColumn source.height first) (secondFrozen : frozenColumn source.height second)
    (sameRank : runPrefixCount source.height first = runPrefixCount source.height second) :
    expandState rowsPositive source first firstFrozen = expandState rowsPositive source second secondFrozen := by
  refine state_eq_of_width_height _ _ ?_ ?_
  · rfl
  · exact heq_of_eq (expandFrozen_same_rank source.height source.canonical.1 rowsPositive
      first second firstFrozen secondFrozen sameRank)

theorem rankedStep_sources_unique {rows : Nat} (rowsPositive : 0 < rows) (rank : Nat)
    {first second target : CanonicalState rows}
    (firstStep : RankedStep rowsPositive rank first target)
    (secondStep : RankedStep rowsPositive rank second target) : first = second := by
  obtain ⟨firstMark, firstFrozen, firstRank, firstRecovery⟩ := rankedStep_reconstructs rowsPositive rank firstStep
  obtain ⟨secondMark, secondFrozen, secondRank, secondRecovery⟩ := rankedStep_reconstructs rowsPositive rank secondStep
  exact firstRecovery.symm.trans ((expandState_same_rank rowsPositive target firstMark secondMark
    firstFrozen secondFrozen (firstRank.trans secondRank.symm)).trans secondRecovery)

theorem reduced_iff_runLength_le_one {rows : Nat} (rowsPositive : 0 < rows)
    (source : CanonicalState rows) : Reduced source ↔ ∀ rank : Nat, runLengthAtRank source.height rank ≤ 1 := by
  constructor
  · intro reduced rank
    let reducedHeight : ReducedHF rows := ⟨source.width, source.widthValid, source.height,
      source.canonical.1, source.canonical.2, reduced⟩
    exact reduced_runLengthAtRank_le_one reducedHeight rowsPositive rank
  · intro bounded column _ pair
    have longRun := adjacent_frozen_runLength_gt_one source.height column pair.1 pair.2
    have bound := bounded (runPrefixCount source.height column)
    omega

theorem normalForm_runLengths_injective {rows : Nat} (rowsPositive : 0 < rows)
    (first second : CanonicalState rows)
    (sameNormal : normalForm rowsPositive first = normalForm rowsPositive second)
    (sameLengths : ∀ rank : Nat, runLengthAtRank first.height rank = runLengthAtRank second.height rank) :
    first = second := by
  have general : ∀ width : Nat, ∀ source : CanonicalState rows, source.width = width ->
      ∀ other : CanonicalState rows,
        normalForm rowsPositive source = normalForm rowsPositive other ->
        (∀ rank : Nat, runLengthAtRank source.height rank = runLengthAtRank other.height rank) ->
        source = other := by
    intro width
    induction width using Nat.strong_induction_on with
    | h width inductionHypothesis =>
      intro source sourceWidth other sameNormal sameLengths
      by_cases reduced : Reduced source
      · have otherReduced : Reduced other := (reduced_iff_runLength_le_one rowsPositive other).mpr (by
          intro rank
          rw [← sameLengths rank]
          exact (reduced_iff_runLength_le_one rowsPositive source).mp reduced rank)
        exact (normalForm_fixed rowsPositive source reduced).symm.trans
          (sameNormal.trans (normalForm_fixed rowsPositive other otherReduced))
      · have notBounded : ¬ ∀ rank : Nat, runLengthAtRank source.height rank ≤ 1 :=
          fun bounded => reduced ((reduced_iff_runLength_le_one rowsPositive source).mpr bounded)
        obtain ⟨rank, notBoundedAtRank⟩ := not_forall.mp notBounded
        have longRun : 1 < runLengthAtRank source.height rank := by omega
        obtain ⟨sourceNext, sourceStep⟩ := rankedStep_exists rowsPositive rank source longRun
        have otherLong : 1 < runLengthAtRank other.height rank := by
          rw [← sameLengths rank]
          exact longRun
        obtain ⟨otherNext, otherStep⟩ := rankedStep_exists rowsPositive rank other otherLong
        have sourcePath := Relation.ReflTransGen.single (rankedStep_step rowsPositive rank sourceStep)
        have otherPath := Relation.ReflTransGen.single (rankedStep_step rowsPositive rank otherStep)
        have nextNormal : normalForm rowsPositive sourceNext = normalForm rowsPositive otherNext :=
          (normalForm_eq_of_path rowsPositive sourcePath).symm.trans
            (sameNormal.trans (normalForm_eq_of_path rowsPositive otherPath))
        have nextLengths : ∀ index : Nat,
            runLengthAtRank sourceNext.height index = runLengthAtRank otherNext.height index := by
          intro index
          have sourceUpdate := rankedStep_lengths rowsPositive rank sourceStep index
          have otherUpdate := rankedStep_lengths rowsPositive rank otherStep index
          have originalEquality := sameLengths index
          omega
        have lower := step_width_lt rowsPositive (rankedStep_step rowsPositive rank sourceStep)
        have nextEqual := inductionHypothesis sourceNext.width (by omega) sourceNext rfl otherNext
          nextNormal nextLengths
        rw [← nextEqual] at otherStep
        exact rankedStep_sources_unique rowsPositive rank sourceStep otherStep
  exact general first.width first rfl second sameNormal sameLengths

theorem rankedStep_expand {rows : Nat} (rowsPositive : 0 < rows)
    (source : CanonicalState rows) (mark : Fin source.width)
    (frozen : frozenColumn source.height mark) :
    RankedStep rowsPositive (runPrefixCount source.height mark)
      (expandState rowsPositive source mark frozen) source := by
  let grown := expandState rowsPositive source mark frozen
  let grownMark : Fin (source.width + 1) := ⟨mark.val, by have := mark.isLt; omega⟩
  have retained := retained_frozen source.height source.canonical.1 rowsPositive mark frozen
  have inserted := inserted_frozen source.height mark frozen.2.1
    (frozenSlope source.height rowsPositive mark frozen)
    (frozenSlope_eq_right source.height source.canonical.1 rowsPositive mark frozen)
  have shrinkHeight : contractAt grown.height rowsPositive grownMark retained = source.height :=
    (contract_adjacent_equal grown.height grown.canonical.1 rowsPositive grownMark retained inserted).trans
      (contract_inserted source.height rowsPositive mark _ inserted)
  have shrinkState : contractState rowsPositive grown grownMark retained inserted = source := by
    refine state_eq_of_width_height _ _ ?_ ?_
    · rfl
    · exact heq_of_eq shrinkHeight
  have atRank : runPrefixCount grown.height grownMark = runPrefixCount source.height mark := by
    have relation := runPrefixCount_contracted_marker grown.height grown.canonical.1 rowsPositive
      grownMark retained inserted
    rw [shrinkHeight] at relation
    exact relation.symm
  exact ⟨grownMark, retained, inserted, shrinkState.symm, atRank⟩

def compositionProfile {runs : Nat} (extras : Fin runs -> Nat) (rank : Nat) : Nat :=
  if within : 0 < rank ∧ rank ≤ runs then extras ⟨rank - 1, by omega⟩ + 1 else 0

theorem runLengthAtRank_eq_zero_of_outside {rows width : Nat} (height : Cell rows width -> Int)
    (rank : Nat) (outside : ¬ (0 < rank ∧ rank ≤ numFrozenRuns height)) :
    runLengthAtRank height rank = 0 := by
  classical
  unfold runLengthAtRank
  apply Finset.card_eq_zero.mpr
  ext column
  constructor
  · intro member
    obtain ⟨_, inactive, atRank⟩ := Finset.mem_filter.mp member
    have positive := runPrefixCount_pos_of_inactive height column inactive
    have bound := runPrefixCount_le_numFrozenRuns height column
    exact False.elim (outside ⟨by omega, by omega⟩)
  · intro member
    simp at member

theorem compositionProfile_at_index {runs : Nat} (extras : Fin runs -> Nat) (index : Fin runs) :
    compositionProfile extras (index.val + 1) = extras index + 1 := by
  have within : 0 < index.val + 1 ∧ index.val + 1 ≤ runs :=
    ⟨Nat.succ_pos _, Nat.succ_le_of_lt index.isLt⟩
  simp [compositionProfile, within]

theorem reduced_zero_profile {rows : Nat} (rowsPositive : 0 < rows)
    (source : CanonicalState rows) (reduced : Reduced source) (rank : Nat) :
    runLengthAtRank source.height rank =
      compositionProfile (fun _index : Fin (numFrozenRuns source.height) => 0) rank := by
  by_cases within : 0 < rank ∧ rank ≤ numFrozenRuns source.height
  · let index : Fin (numFrozenRuns source.height) := ⟨rank - 1, by omega⟩
    have indexRank : index.val + 1 = rank := by dsimp [index]; omega
    rw [← indexRank, compositionProfile_at_index]
    exact reduced_runLengthTuple_eq_one
      (⟨source.width, source.widthValid, source.height, source.canonical.1, source.canonical.2, reduced⟩ :
        ReducedHF rows) rowsPositive index
  · rw [runLengthAtRank_eq_zero_of_outside source.height rank within]
    simp [compositionProfile, within]

theorem compositionProfile_decrease {runs : Nat} (extras : Fin runs -> Nat)
    (chosen : Fin runs) (positive : 0 < extras chosen) (rank : Nat) :
    compositionProfile extras rank =
      compositionProfile (Function.update extras chosen (extras chosen - 1)) rank +
        if chosen.val + 1 = rank then 1 else 0 := by
  by_cases atChosen : chosen.val + 1 = rank
  · rw [← atChosen, compositionProfile_at_index, compositionProfile_at_index]
    simp only [Function.update_self, ite_true]
    omega
  · by_cases within : 0 < rank ∧ rank ≤ runs
    · let index : Fin runs := ⟨rank - 1, by omega⟩
      have indexRank : index.val + 1 = rank := by dsimp [index]; omega
      have notChosen : index ≠ chosen := by
        intro equality
        apply atChosen
        rw [← equality]
        exact indexRank
      have differentRank : chosen.val + 1 ≠ index.val + 1 :=
        fun equality => atChosen (equality.trans indexRank)
      rw [← indexRank, compositionProfile_at_index, compositionProfile_at_index]
      rw [Function.update_of_ne notChosen, if_neg differentRank, Nat.add_zero]
    · simp [compositionProfile, within, atChosen]

theorem sum_decrease_add_one {runs : Nat} (extras : Fin runs -> Nat)
    (chosen : Fin runs) (positive : 0 < extras chosen) :
    (∑ index : Fin runs, Function.update extras chosen (extras chosen - 1) index) + 1 =
      ∑ index : Fin runs, extras index := by
  classical
  have pointwise : ∀ index : Fin runs,
      Function.update extras chosen (extras chosen - 1) index + (if index = chosen then 1 else 0) =
        extras index := by
    intro index
    by_cases same : index = chosen
    · rw [same]
      simp only [Function.update_self, ite_true]
      omega
    · rw [Function.update_of_ne same, if_neg same, Nat.add_zero]
  calc
    _ = (∑ index : Fin runs, Function.update extras chosen (extras chosen - 1) index) +
        ∑ index : Fin runs, if index = chosen then 1 else 0 := by simp
    _ = (∑ index : Fin runs,
        (Function.update extras chosen (extras chosen - 1) index + (if index = chosen then 1 else 0))) :=
      Finset.sum_add_distrib.symm
    _ = _ := Finset.sum_congr rfl (fun index _ => pointwise index)

theorem compositionProfile_realized {rows : Nat} (rowsPositive : 0 < rows)
    (core : CanonicalState rows) (reduced : Reduced core)
    (extras : Fin (numFrozenRuns core.height) -> Nat) :
    ∃ source : CanonicalState rows, normalForm rowsPositive source = core ∧
      ∀ rank : Nat, runLengthAtRank source.height rank = compositionProfile extras rank := by
  classical
  have construct : ∀ total : Nat, ∀ extras : Fin (numFrozenRuns core.height) -> Nat,
      (∑ index : Fin (numFrozenRuns core.height), extras index) = total ->
        ∃ source : CanonicalState rows, normalForm rowsPositive source = core ∧
          ∀ rank : Nat, runLengthAtRank source.height rank = compositionProfile extras rank := by
    intro total
    induction total using Nat.strong_induction_on with
    | h total inductionHypothesis =>
      intro extras sumEquality
      by_cases totalZero : total = 0
      · have allZero : extras = fun _index => 0 := by
          funext index
          have bound : extras index ≤ ∑ position : Fin (numFrozenRuns core.height), extras position :=
            Finset.single_le_sum (fun position _ => Nat.zero_le (extras position)) (Finset.mem_univ index)
          omega
        rw [allZero]
        exact ⟨core, normalForm_fixed rowsPositive core reduced, reduced_zero_profile rowsPositive core reduced⟩
      · have existsPositive : ∃ chosen : Fin (numFrozenRuns core.height), 0 < extras chosen := by
          by_contra noPositive
          have allZero : extras = fun _index => 0 := by
            funext index
            have notPositive : ¬ 0 < extras index := fun positive => noPositive ⟨index, positive⟩
            omega
          have sumZero : (∑ index : Fin (numFrozenRuns core.height), extras index) = 0 := by
            rw [allZero]
            simp
          omega
        obtain ⟨chosen, positive⟩ := existsPositive
        let smaller := Function.update extras chosen (extras chosen - 1)
        have smallerSum : (∑ index : Fin (numFrozenRuns core.height), smaller index) + 1 = total :=
          (sum_decrease_add_one extras chosen positive).trans sumEquality
        obtain ⟨intermediate, intermediateNormal, intermediateProfile⟩ :=
          inductionHypothesis (∑ index : Fin (numFrozenRuns core.height), smaller index)
            (by omega) smaller rfl
        have selectedLength : 0 < runLengthAtRank intermediate.height (chosen.val + 1) := by
          rw [intermediateProfile (chosen.val + 1), compositionProfile_at_index]
          exact Nat.succ_pos _
        obtain ⟨mark, member⟩ := Finset.card_pos.mp selectedLength
        obtain ⟨_, inactive, atRank⟩ := Finset.mem_filter.mp member
        have frozen := (column_active_or_frozen intermediate.height intermediate.canonical.1 rowsPositive
          intermediate.widthValid mark).resolve_left inactive
        let grown := expandState rowsPositive intermediate mark frozen
        have grownStep : RankedStep rowsPositive (chosen.val + 1) grown intermediate := by
          simpa only [atRank] using rankedStep_expand rowsPositive intermediate mark frozen
        have grownNormal : normalForm rowsPositive grown = core :=
          (normalForm_eq_of_path rowsPositive
            (Relation.ReflTransGen.single (rankedStep_step rowsPositive (chosen.val + 1) grownStep))).trans
              intermediateNormal
        refine ⟨grown, grownNormal, ?_⟩
        intro rank
        rw [rankedStep_lengths rowsPositive (chosen.val + 1) grownStep rank, intermediateProfile rank]
        exact (compositionProfile_decrease extras chosen positive rank).symm
  exact construct (∑ index : Fin (numFrozenRuns core.height), extras index) extras rfl

abbrev NormalFiber {rows : Nat} (rowsPositive : 0 < rows) (core : CanonicalState rows) :=
  {source : CanonicalState rows // normalForm rowsPositive source = core}

noncomputable def fiberExtras {rows : Nat} (rowsPositive : 0 < rows) (core : CanonicalState rows)
    (source : NormalFiber rowsPositive core) : Fin (numFrozenRuns core.height) -> Nat :=
  fun index => runLengthAtRank source.val.height (index.val + 1) - 1

theorem normalForm_runCount {rows : Nat} (rowsPositive : 0 < rows)
    {source core : CanonicalState rows} (equality : normalForm rowsPositive source = core) :
    numFrozenRuns source.height = numFrozenRuns core.height := by
  have counts := reduction_run_count rowsPositive (normalForm_path rowsPositive source)
  rw [equality] at counts
  exact counts.symm

theorem fiber_profile {rows : Nat} (rowsPositive : 0 < rows) (core : CanonicalState rows)
    (source : NormalFiber rowsPositive core) (rank : Nat) :
    runLengthAtRank source.val.height rank = compositionProfile (fiberExtras rowsPositive core source) rank := by
  have counts := normalForm_runCount rowsPositive source.property
  by_cases within : 0 < rank ∧ rank ≤ numFrozenRuns core.height
  · let index : Fin (numFrozenRuns source.val.height) := ⟨rank - 1, by rw [counts]; omega⟩
    have indexRank : index.val + 1 = rank := by dsimp [index]; omega
    have positive : 0 < runLengthAtRank source.val.height rank := by
      have atIndex : 0 < runLengthAtRank source.val.height (index.val + 1) :=
        runLengthTuple_pos source.val.height index
      rwa [indexRank] at atIndex
    rw [compositionProfile, dif_pos within]
    change runLengthAtRank source.val.height rank =
      (runLengthAtRank source.val.height (rank - 1 + 1) - 1) + 1
    rw [show rank - 1 + 1 = rank from by omega]
    exact (Nat.sub_add_cancel (Nat.succ_le_of_lt positive)).symm
  · have outsideSource : ¬ (0 < rank ∧ rank ≤ numFrozenRuns source.val.height) := by
      simpa only [counts] using within
    rw [runLengthAtRank_eq_zero_of_outside source.val.height rank outsideSource]
    simp [compositionProfile, within]

noncomputable def normalFiberEquiv {rows : Nat} (rowsPositive : 0 < rows)
    (core : CanonicalState rows) (reduced : Reduced core) :
    NormalFiber rowsPositive core ≃ (Fin (numFrozenRuns core.height) -> Nat) :=
  Equiv.ofBijective (fiberExtras rowsPositive core) (by
    constructor
    · intro first second equality
      apply Subtype.ext
      apply normalForm_runLengths_injective rowsPositive first.val second.val
        (first.property.trans second.property.symm)
      intro rank
      rw [fiber_profile rowsPositive core first rank, fiber_profile rowsPositive core second rank, equality]
    · intro extras
      obtain ⟨source, sourceNormal, sourceProfile⟩ := compositionProfile_realized rowsPositive core reduced extras
      refine ⟨⟨source, sourceNormal⟩, ?_⟩
      funext index
      change runLengthAtRank source.height (index.val + 1) - 1 = extras index
      rw [sourceProfile (index.val + 1), compositionProfile_at_index]
      omega)

theorem fiberExtras_width {rows : Nat} (rowsPositive : 0 < rows) (core : CanonicalState rows)
    (source : NormalFiber rowsPositive core) :
    source.val.width = core.width +
      ∑ index : Fin (numFrozenRuns core.height), fiberExtras rowsPositive core source index := by
  classical
  have counts := normalForm_runCount rowsPositive source.property
  have totals : (∑ index : Fin (numFrozenRuns core.height), fiberExtras rowsPositive core source index) =
      ∑ index : Fin (numFrozenRuns source.val.height), extraLengthTuple source.val.height index := by
    apply Finset.sum_bij (fun index _ => Fin.cast counts.symm index)
    · intro index _
      exact Finset.mem_univ _
    · intro first _ second _ equality
      exact Fin.ext (congrArg (fun position : Fin (numFrozenRuns source.val.height) => position.val) equality)
    · intro index _
      refine ⟨Fin.cast counts index, Finset.mem_univ _, ?_⟩
      apply Fin.ext
      rfl
    · intro index _
      rfl
  have removed := sum_extraLengthTuple_eq_removed rowsPositive source.val
  have widthBound := reduction_width_le rowsPositive (normalForm_path rowsPositive source.val)
  rw [source.property] at removed widthBound
  rw [← totals] at removed
  omega

abbrev GrowthFiber {rows : Nat} (rowsPositive : 0 < rows) (core : CanonicalState rows) (total : Nat) :=
  {source : NormalFiber rowsPositive core // source.val.width = core.width + total}

noncomputable def growthFiberEquiv {rows : Nat} (rowsPositive : 0 < rows)
    (core : CanonicalState rows) (reduced : Reduced core) (total : Nat) :
    GrowthFiber rowsPositive core total ≃
      {extras : Fin (numFrozenRuns core.height) -> Nat // (∑ index, extras index) = total} :=
  (normalFiberEquiv rowsPositive core reduced).subtypeEquiv (by
    intro source
    change source.val.width = core.width + total ↔
      (∑ index : Fin (numFrozenRuns core.height), fiberExtras rowsPositive core source index) = total
    have widthIdentity := fiberExtras_width rowsPositive core source
    omega)

noncomputable def compositionFinsetEquiv (runs total : Nat) :
    {extras : Fin runs -> Nat // (∑ index, extras index) = total} ≃
      (Finset.Nat.antidiagonalTuple runs total) :=
  Equiv.subtypeEquivRight (by
    intro extras
    simp only [Finset.Nat.mem_antidiagonalTuple])

theorem growthFiber_finite {rows : Nat} (rowsPositive : 0 < rows)
    (core : CanonicalState rows) (reduced : Reduced core) (total : Nat) :
    Finite (GrowthFiber rowsPositive core total) :=
  Finite.of_equiv (Finset.Nat.antidiagonalTuple (numFrozenRuns core.height) total)
    ((growthFiberEquiv rowsPositive core reduced total).trans
      (compositionFinsetEquiv (numFrozenRuns core.height) total)).symm

theorem growthFiber_card {rows : Nat} (rowsPositive : 0 < rows)
    (core : CanonicalState rows) (reduced : Reduced core) (total : Nat)
    (positive : 0 < numFrozenRuns core.height) :
    Nat.card (GrowthFiber rowsPositive core total) =
      (total + numFrozenRuns core.height - 1).choose (numFrozenRuns core.height - 1) := by
  have equivalence := (growthFiberEquiv rowsPositive core reduced total).trans
    (compositionFinsetEquiv (numFrozenRuns core.height) total)
  rw [Nat.card_congr equivalence, Nat.card_eq_fintype_card, Fintype.card_coe]
  obtain ⟨runs, counts⟩ : ∃ runs : Nat, numFrozenRuns core.height = runs + 1 :=
    ⟨numFrozenRuns core.height - 1, by omega⟩
  rw [counts, antidiagonalTuple_card]
  congr 1

theorem growthFiber_card_zero_runs {rows : Nat} (rowsPositive : 0 < rows)
    (core : CanonicalState rows) (reduced : Reduced core) (total : Nat)
    (noRuns : numFrozenRuns core.height = 0) :
    Nat.card (GrowthFiber rowsPositive core total) = if total = 0 then 1 else 0 := by
  have equivalence := (growthFiberEquiv rowsPositive core reduced total).trans
    (compositionFinsetEquiv (numFrozenRuns core.height) total)
  rw [Nat.card_congr equivalence, Nat.card_eq_fintype_card, Fintype.card_coe, noRuns]
  cases total <;> simp

theorem normalFiber_extrema {rows : Nat} (rowsPositive : 0 < rows) (core : CanonicalState rows)
    (source : NormalFiber rowsPositive core) : numExtrema source.val.height = numExtrema core.height := by
  have counts := normalForm_extrema rowsPositive source.val
  rw [source.property] at counts
  exact counts.symm

theorem normalFiber_zero_runs_unique {rows : Nat} (rowsPositive : 0 < rows)
    (core : CanonicalState rows) (reduced : Reduced core) (noRuns : numFrozenRuns core.height = 0)
    (source : NormalFiber rowsPositive core) : source.val = core := by
  let root : NormalFiber rowsPositive core := ⟨core, normalForm_fixed rowsPositive core reduced⟩
  have extrasEquality : fiberExtras rowsPositive core source = fiberExtras rowsPositive core root := by
    funext index
    have bound := index.isLt
    omega
  have sourceEquality : source = root := (normalFiberEquiv rowsPositive core reduced).injective extrasEquality
  exact congrArg Subtype.val sourceEquality

abbrev WidthFiber {rows : Nat} (rowsPositive : 0 < rows) (core : CanonicalState rows) (width : Nat) :=
  {source : NormalFiber rowsPositive core // source.val.width = width}

noncomputable def widthGrowthEquiv {rows : Nat} (rowsPositive : 0 < rows)
    (core : CanonicalState rows) (width : Nat) (widthBound : core.width ≤ width) :
    WidthFiber rowsPositive core width ≃ GrowthFiber rowsPositive core (width - core.width) :=
  Equiv.subtypeEquivRight (by intro source; omega)

theorem widthFiber_finite {rows : Nat} (rowsPositive : 0 < rows)
    (core : CanonicalState rows) (reduced : Reduced core) (width : Nat) (widthBound : core.width ≤ width) :
    Finite (WidthFiber rowsPositive core width) := by
  letI : Finite (GrowthFiber rowsPositive core (width - core.width)) :=
    growthFiber_finite rowsPositive core reduced (width - core.width)
  exact Finite.of_equiv _ (widthGrowthEquiv rowsPositive core width widthBound).symm

theorem widthFiber_card {rows : Nat} (rowsPositive : 0 < rows)
    (core : CanonicalState rows) (reduced : Reduced core) (width : Nat)
    (widthBound : core.width ≤ width) (positive : 0 < numFrozenRuns core.height) :
    Nat.card (WidthFiber rowsPositive core width) =
      (width - core.width + numFrozenRuns core.height - 1).choose (numFrozenRuns core.height - 1) :=
  (Nat.card_congr (widthGrowthEquiv rowsPositive core width widthBound)).trans
    (growthFiber_card rowsPositive core reduced (width - core.width) positive)

theorem widthFiber_zero_runs_empty {rows : Nat} (rowsPositive : 0 < rows)
    (core : CanonicalState rows) (reduced : Reduced core) (width : Nat)
    (differentWidth : width ≠ core.width) (noRuns : numFrozenRuns core.height = 0) :
    IsEmpty (WidthFiber rowsPositive core width) := by
  refine ⟨fun source => ?_⟩
  have sourceEquality := normalFiber_zero_runs_unique rowsPositive core reduced noRuns source.val
  have widthEquality : source.val.val.width = core.width := congrArg CanonicalState.width sourceEquality
  exact differentWidth (source.property.symm.trans widthEquality)

#print axioms rankedStep_step
#print axioms rankedStep_lengths
#print axioms rankedStep_exists
#print axioms rankedStep_reconstructs
#print axioms expandState_same_rank
#print axioms rankedStep_sources_unique
#print axioms reduced_iff_runLength_le_one
#print axioms normalForm_runLengths_injective
#print axioms rankedStep_expand
#print axioms runLengthAtRank_eq_zero_of_outside
#print axioms compositionProfile_at_index
#print axioms reduced_zero_profile
#print axioms compositionProfile_decrease
#print axioms sum_decrease_add_one
#print axioms compositionProfile_realized
#print axioms normalForm_runCount
#print axioms fiber_profile
#print axioms normalFiberEquiv
#print axioms fiberExtras_width
#print axioms growthFiberEquiv
#print axioms compositionFinsetEquiv
#print axioms growthFiber_finite
#print axioms growthFiber_card
#print axioms growthFiber_card_zero_runs
#print axioms normalFiber_extrema
#print axioms normalFiber_zero_runs_unique
#print axioms widthGrowthEquiv
#print axioms widthFiber_finite
#print axioms widthFiber_card
#print axioms widthFiber_zero_runs_empty

end OrigamiCone.Sequel.RunFiberEquivalence
