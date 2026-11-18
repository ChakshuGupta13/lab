import MiuraCounting.ConcreteFrozenExpansion
import OrigamiCone.SequelEdReducedHF
import MiuraCounting.AtomicReconstruction

namespace OrigamiCone.Sequel.FrozenReduction

open OrigamiCone FrozenExpansion

theorem separated_contractions_commute {rows width : Nat} (height : Cell rows (width + 2) -> Int)
    (heightValid : IsHeight height) (rowsPositive : 0 < rows)
    (lower upper : Fin (width + 2)) (separated : lower.val + 1 < upper.val)
    (lowerFrozen : frozenColumn height lower) (upperFrozen : frozenColumn height upper) :
    ∃ upperAfter : frozenColumn (contractAt height rowsPositive lower lowerFrozen)
        ⟨upper.val - 1, by have := upper.isLt; omega⟩,
      ∃ lowerAfter : frozenColumn (contractAt height rowsPositive upper upperFrozen)
          ⟨lower.val, by have := upper.isLt; omega⟩,
        contractAt (contractAt height rowsPositive lower lowerFrozen) rowsPositive
            ⟨upper.val - 1, by have := upper.isLt; omega⟩ upperAfter =
          contractAt (contractAt height rowsPositive upper upperFrozen) rowsPositive
            ⟨lower.val, by have := upper.isLt; omega⟩ lowerAfter := by
  let shiftedUpper : Fin (width + 1) := ⟨upper.val - 1, by have := upper.isLt; omega⟩
  let shiftedLower : Fin (width + 1) := ⟨lower.val, by have := upper.isLt; omega⟩
  let lowerSlope := frozenSlope height rowsPositive lower lowerFrozen
  let upperSlope := frozenSlope height rowsPositive upper upperFrozen
  have keptUpper : keptColumn lower shiftedUpper = upper := by
    have after : ¬ upper.val - 1 < lower.val := by omega
    have index : upper.val - 1 + 1 = upper.val := by omega
    apply Fin.ext
    simp only [keptColumn, shiftedUpper, after, dite_false]
    exact index
  have keptLower : keptColumn upper shiftedLower = lower := by
    have before : lower.val < upper.val := by omega
    apply Fin.ext
    simp [keptColumn, shiftedLower, before]
  have upperAfter : frozenColumn (contractAt height rowsPositive lower lowerFrozen) shiftedUpper := by
    apply (frozenColumn_contractAt_iff height heightValid rowsPositive lower lowerFrozen shiftedUpper).mpr
    rw [keptUpper]
    exact upperFrozen
  have lowerAfter : frozenColumn (contractAt height rowsPositive upper upperFrozen) shiftedLower := by
    apply (frozenColumn_contractAt_iff height heightValid rowsPositive upper upperFrozen shiftedLower).mpr
    rw [keptLower]
    exact lowerFrozen
  refine ⟨upperAfter, lowerAfter, ?_⟩
  have upperRaw : frozenColumn (deleteColumn height lower lowerSlope) shiftedUpper := by
    rw [deleteColumn_eq_contractAt]
    exact upperAfter
  have lowerRaw : frozenColumn (deleteColumn height upper upperSlope) shiftedLower := by
    rw [deleteColumn_eq_contractAt]
    exact lowerAfter
  have upperSlopeIdentity :
      frozenSlope (contractAt height rowsPositive lower lowerFrozen) rowsPositive shiftedUpper upperAfter =
        upperSlope := by
    have identity := upperSlope_deleteLower height rowsPositive lower upper separated lowerSlope upperFrozen upperRaw
    simpa only [deleteColumn_eq_contractAt] using identity
  have lowerSlopeIdentity :
      frozenSlope (contractAt height rowsPositive upper upperFrozen) rowsPositive shiftedLower lowerAfter =
        lowerSlope := by
    have identity := lowerSlope_deleteUpper height rowsPositive lower upper (by omega) upperSlope lowerFrozen lowerRaw
    simpa only [deleteColumn_eq_contractAt] using identity
  rw [← deleteColumn_eq_contractAt (contractAt height rowsPositive lower lowerFrozen)
      rowsPositive shiftedUpper upperAfter, upperSlopeIdentity,
    ← deleteColumn_eq_contractAt (contractAt height rowsPositive upper upperFrozen)
      rowsPositive shiftedLower lowerAfter, lowerSlopeIdentity]
  calc
    _ = deleteColumn (deleteColumn height lower lowerSlope) shiftedUpper upperSlope := by
      rw [deleteColumn_eq_contractAt]
    _ = deleteColumn (deleteColumn height upper upperSlope) shiftedLower lowerSlope :=
      deleteColumns_commute height lower upper (by omega) lowerSlope upperSlope
    _ = _ := by rw [deleteColumn_eq_contractAt]

structure CanonicalState (rows : Nat) where
  width : Nat
  widthValid : 2 ≤ width
  height : Cell rows width -> Int
  canonical : IsCanonicalHeight height

noncomputable def contractState {rows : Nat} (rowsPositive : 0 < rows)
    (source : CanonicalState rows) (column : Fin source.width)
    (frozen : frozenColumn source.height column)
    (nextFrozen : frozenColumn source.height ⟨column.val + 1, frozen.2.1⟩) :
    CanonicalState rows where
  width := source.width - 1
  widthValid := by
    have positive := frozen.1
    have nextInterior := nextFrozen.2.1
    change column.val + 1 + 1 < source.width at nextInterior
    omega
  height := contractAt source.height rowsPositive column frozen
  canonical := contractAt_isCanonicalHeight source.height source.canonical.1
    rowsPositive column frozen source.canonical

def Step {rows : Nat} (rowsPositive : 0 < rows) (source target : CanonicalState rows) : Prop :=
  ∃ column : Fin source.width, ∃ frozen : frozenColumn source.height column,
    ∃ nextFrozen : frozenColumn source.height ⟨column.val + 1, frozen.2.1⟩,
      target = contractState rowsPositive source column frozen nextFrozen

theorem contractState_width {rows : Nat} (rowsPositive : 0 < rows)
    (source : CanonicalState rows) (column : Fin source.width)
    (frozen : frozenColumn source.height column)
    (nextFrozen : frozenColumn source.height ⟨column.val + 1, frozen.2.1⟩) :
    (contractState rowsPositive source column frozen nextFrozen).width = source.width - 1 := rfl

theorem contractState_extrema {rows : Nat} (rowsPositive : 0 < rows)
    (source : CanonicalState rows) (column : Fin source.width)
    (frozen : frozenColumn source.height column)
    (nextFrozen : frozenColumn source.height ⟨column.val + 1, frozen.2.1⟩) :
    numExtrema (contractState rowsPositive source column frozen nextFrozen).height =
      numExtrema source.height :=
  contractAt_numExtrema_eq source.height source.canonical.1 rowsPositive column frozen

theorem step_width_lt {rows : Nat} (rowsPositive : 0 < rows)
    {source target : CanonicalState rows} (step : Step rowsPositive source target) :
    target.width < source.width := by
  obtain ⟨column, frozen, nextFrozen, rfl⟩ := step
  rw [contractState_width]
  have sourceWidth := source.widthValid
  omega

theorem step_extrema {rows : Nat} (rowsPositive : 0 < rows)
    {source target : CanonicalState rows} (step : Step rowsPositive source target) :
    numExtrema target.height = numExtrema source.height := by
  obtain ⟨column, frozen, nextFrozen, rfl⟩ := step
  exact contractState_extrema rowsPositive source column frozen nextFrozen

def Reduced {rows : Nat} (source : CanonicalState rows) : Prop :=
  ∀ column : Fin source.width, ∀ adjacent : column.val + 1 < source.width,
    ¬ (frozenColumn source.height column ∧
      frozenColumn source.height ⟨column.val + 1, adjacent⟩)

theorem reduced_no_step {rows : Nat} (rowsPositive : 0 < rows)
    {source target : CanonicalState rows} (reduced : Reduced source)
    (step : Step rowsPositive source target) : False := by
  obtain ⟨column, frozen, nextFrozen, rfl⟩ := step
  exact reduced column frozen.2.1 ⟨frozen, nextFrozen⟩

theorem reduction_width_le {rows : Nat} (rowsPositive : 0 < rows)
    {source target : CanonicalState rows} (path : Relation.ReflTransGen (Step rowsPositive) source target) :
    target.width ≤ source.width := by
  induction path with
  | refl => exact le_refl _
  | tail _ step inductionHypothesis =>
    exact (step_width_lt rowsPositive step).le.trans inductionHypothesis

theorem reduction_extrema {rows : Nat} (rowsPositive : 0 < rows)
    {source target : CanonicalState rows} (path : Relation.ReflTransGen (Step rowsPositive) source target) :
    numExtrema target.height = numExtrema source.height := by
  induction path with
  | refl => rfl
  | tail _ step inductionHypothesis =>
    exact (step_extrema rowsPositive step).trans inductionHypothesis

theorem reduced_exists {rows : Nat} (rowsPositive : 0 < rows) (source : CanonicalState rows) :
    ∃ target : CanonicalState rows, Relation.ReflTransGen (Step rowsPositive) source target ∧ Reduced target := by
  classical
  suffices ∀ bound : Nat, ∀ source : CanonicalState rows, source.width ≤ bound ->
      ∃ target : CanonicalState rows, Relation.ReflTransGen (Step rowsPositive) source target ∧ Reduced target from
    this source.width source (le_refl _)
  intro bound
  induction bound using Nat.strong_induction_on with
  | h bound inductionHypothesis =>
    intro source widthBound
    by_cases reduced : Reduced source
    · exact ⟨source, Relation.ReflTransGen.refl, reduced⟩
    · have adjacentFrozen : ∃ column : Fin source.width, ∃ adjacent : column.val + 1 < source.width,
          frozenColumn source.height column ∧
            frozenColumn source.height ⟨column.val + 1, adjacent⟩ := by
        by_contra absent
        apply reduced
        intro column adjacent pair
        exact absent ⟨column, adjacent, pair⟩
      obtain ⟨column, adjacent, frozen, nextFrozen⟩ := adjacentFrozen
      let next := contractState rowsPositive source column frozen nextFrozen
      have step : Step rowsPositive source next := ⟨column, frozen, nextFrozen, rfl⟩
      have nextBound : next.width < bound := (step_width_lt rowsPositive step).trans_le widthBound
      obtain ⟨target, path, targetReduced⟩ :=
        inductionHypothesis next.width nextBound next (le_refl _)
      exact ⟨target, Relation.ReflTransGen.head step path, targetReduced⟩

theorem reduced_path_identity {rows : Nat} (rowsPositive : 0 < rows)
    {source target : CanonicalState rows} (reduced : Reduced source)
    (path : Relation.ReflTransGen (Step rowsPositive) source target) : target = source := by
  exact (Relation.reflTransGen_iff_eq
    (fun _ step => reduced_no_step rowsPositive reduced step)).mp path

theorem state_eq_of_width_height {rows : Nat} (first second : CanonicalState rows)
    (widthEquality : first.width = second.width) (heightEquality : HEq first.height second.height) :
    first = second := by
  cases first
  cases second
  cases widthEquality
  cases heightEquality
  rfl

theorem separated_steps_join {rows : Nat} (rowsPositive : 0 < rows)
    (source : CanonicalState rows) (lower upper : Fin source.width)
    (separated : lower.val + 1 < upper.val)
    (lowerFrozen : frozenColumn source.height lower)
    (lowerNext : frozenColumn source.height ⟨lower.val + 1, lowerFrozen.2.1⟩)
    (upperFrozen : frozenColumn source.height upper)
    (upperNext : frozenColumn source.height ⟨upper.val + 1, upperFrozen.2.1⟩) :
    ∃ common : CanonicalState rows,
      Step rowsPositive (contractState rowsPositive source lower lowerFrozen lowerNext) common ∧
        Step rowsPositive (contractState rowsPositive source upper upperFrozen upperNext) common := by
  rcases source with ⟨sourceWidth, sourceWidthValid, height, canonical⟩
  dsimp only at *
  obtain ⟨width, widthEquality⟩ : ∃ width : Nat, sourceWidth = width + 2 :=
    ⟨sourceWidth - 2, by omega⟩
  subst sourceWidth
  let source : CanonicalState rows := ⟨width + 2, sourceWidthValid, height, canonical⟩
  let shiftedUpper : Fin (width + 1) := ⟨upper.val - 1, by have := upper.isLt; omega⟩
  let shiftedLower : Fin (width + 1) := ⟨lower.val, by have := upper.isLt; omega⟩
  obtain ⟨upperAfter, lowerAfter, arraysEqual⟩ :=
    separated_contractions_commute height canonical.1 rowsPositive lower upper separated lowerFrozen upperFrozen
  have keptUpperNext :
      keptColumn lower ⟨shiftedUpper.val + 1, upperAfter.2.1⟩ =
        ⟨upper.val + 1, upperFrozen.2.1⟩ := by
    have after : ¬ upper.val - 1 + 1 < lower.val := by omega
    have index : upper.val - 1 + 1 + 1 = upper.val + 1 := by omega
    apply Fin.ext
    simp only [keptColumn, shiftedUpper, after, dite_false]
    exact index
  have keptLowerNext :
      keptColumn upper ⟨shiftedLower.val + 1, lowerAfter.2.1⟩ =
        ⟨lower.val + 1, lowerFrozen.2.1⟩ := by
    apply Fin.ext
    simp [keptColumn, shiftedLower, separated]
  have upperNextAfter : frozenColumn (contractAt height rowsPositive lower lowerFrozen)
      ⟨shiftedUpper.val + 1, upperAfter.2.1⟩ := by
    apply (frozenColumn_contractAt_iff height canonical.1 rowsPositive lower lowerFrozen _).mpr
    rw [keptUpperNext]
    exact upperNext
  have lowerNextAfter : frozenColumn (contractAt height rowsPositive upper upperFrozen)
      ⟨shiftedLower.val + 1, lowerAfter.2.1⟩ := by
    apply (frozenColumn_contractAt_iff height canonical.1 rowsPositive upper upperFrozen _).mpr
    rw [keptLowerNext]
    exact lowerNext
  let left := contractState rowsPositive source lower lowerFrozen lowerNext
  let right := contractState rowsPositive source upper upperFrozen upperNext
  let common := contractState rowsPositive left shiftedUpper upperAfter upperNextAfter
  refine ⟨common, ⟨shiftedUpper, upperAfter, upperNextAfter, rfl⟩,
    ⟨shiftedLower, lowerAfter, lowerNextAfter, ?_⟩⟩
  apply state_eq_of_width_height common
    (contractState rowsPositive right shiftedLower lowerAfter lowerNextAfter) rfl
  exact heq_of_eq arraysEqual

theorem adjacent_states_equal {rows : Nat} (rowsPositive : 0 < rows)
    (source : CanonicalState rows) (lower upper : Fin source.width)
    (adjacent : lower.val + 1 = upper.val)
    (lowerFrozen : frozenColumn source.height lower)
    (lowerNext : frozenColumn source.height ⟨lower.val + 1, lowerFrozen.2.1⟩)
    (upperFrozen : frozenColumn source.height upper)
    (upperNext : frozenColumn source.height ⟨upper.val + 1, upperFrozen.2.1⟩) :
    contractState rowsPositive source lower lowerFrozen lowerNext =
      contractState rowsPositive source upper upperFrozen upperNext := by
  apply state_eq_of_width_height
    (contractState rowsPositive source lower lowerFrozen lowerNext)
    (contractState rowsPositive source upper upperFrozen upperNext) rfl
  rcases source with ⟨width, widthValid, height, canonical⟩
  cases width with
  | zero => omega
  | succ width =>
    have sameColumn : upper = ⟨lower.val + 1, lowerFrozen.2.1⟩ := Fin.ext adjacent.symm
    subst upper
    exact heq_of_eq (contract_adjacent_equal height canonical.1 rowsPositive lower lowerFrozen lowerNext)

theorem step_diamond {rows : Nat} (rowsPositive : 0 < rows)
    {source first second : CanonicalState rows}
    (firstStep : Step rowsPositive source first) (secondStep : Step rowsPositive source second) :
    ∃ common : CanonicalState rows, Relation.ReflGen (Step rowsPositive) first common ∧
      Relation.ReflGen (Step rowsPositive) second common := by
  obtain ⟨lower, lowerFrozen, lowerNext, rfl⟩ := firstStep
  obtain ⟨upper, upperFrozen, upperNext, rfl⟩ := secondStep
  by_cases same : lower = upper
  · subst upper
    exact ⟨_, Relation.ReflGen.refl, Relation.ReflGen.refl⟩
  · have ordering : lower.val < upper.val ∨ upper.val < lower.val := by
      have different : lower.val ≠ upper.val := fun equality => same (Fin.ext equality)
      omega
    rcases ordering with ordered | reverse
    · by_cases adjacent : lower.val + 1 = upper.val
      · have equality := adjacent_states_equal rowsPositive source lower upper adjacent
          lowerFrozen lowerNext upperFrozen upperNext
        refine ⟨contractState rowsPositive source upper upperFrozen upperNext, ?_, Relation.ReflGen.refl⟩
        rw [equality]
      · obtain ⟨common, left, right⟩ := separated_steps_join rowsPositive source lower upper (by omega)
          lowerFrozen lowerNext upperFrozen upperNext
        exact ⟨common, Relation.ReflGen.single left, Relation.ReflGen.single right⟩
    · by_cases adjacent : upper.val + 1 = lower.val
      · have equality := adjacent_states_equal rowsPositive source upper lower adjacent
          upperFrozen upperNext lowerFrozen lowerNext
        refine ⟨contractState rowsPositive source lower lowerFrozen lowerNext, Relation.ReflGen.refl, ?_⟩
        rw [equality]
      · obtain ⟨common, right, left⟩ := separated_steps_join rowsPositive source upper lower (by omega)
          upperFrozen upperNext lowerFrozen lowerNext
        exact ⟨common, Relation.ReflGen.single left, Relation.ReflGen.single right⟩

theorem reduction_confluent {rows : Nat} (rowsPositive : 0 < rows)
    {source first second : CanonicalState rows}
    (firstPath : Relation.ReflTransGen (Step rowsPositive) source first)
    (secondPath : Relation.ReflTransGen (Step rowsPositive) source second) :
    Relation.Join (Relation.ReflTransGen (Step rowsPositive)) first second := by
  apply Relation.church_rosser (r := Step rowsPositive) _ firstPath secondPath
  intro start left right firstStep secondStep
  obtain ⟨common, leftPath, rightPath⟩ := step_diamond rowsPositive firstStep secondStep
  exact ⟨common, leftPath, rightPath.to_reflTransGen⟩

theorem reduced_unique {rows : Nat} (rowsPositive : 0 < rows)
    {source first second : CanonicalState rows}
    (firstPath : Relation.ReflTransGen (Step rowsPositive) source first)
    (secondPath : Relation.ReflTransGen (Step rowsPositive) source second)
    (firstReduced : Reduced first) (secondReduced : Reduced second) : first = second := by
  obtain ⟨common, firstToCommon, secondToCommon⟩ := reduction_confluent rowsPositive firstPath secondPath
  have firstEquality := reduced_path_identity rowsPositive firstReduced firstToCommon
  have secondEquality := reduced_path_identity rowsPositive secondReduced secondToCommon
  exact firstEquality.symm.trans secondEquality

noncomputable def normalForm {rows : Nat} (rowsPositive : 0 < rows)
    (source : CanonicalState rows) : CanonicalState rows :=
  Classical.choose (reduced_exists rowsPositive source)

theorem normalForm_path {rows : Nat} (rowsPositive : 0 < rows) (source : CanonicalState rows) :
    Relation.ReflTransGen (Step rowsPositive) source (normalForm rowsPositive source) :=
  (Classical.choose_spec (reduced_exists rowsPositive source)).1

theorem normalForm_reduced {rows : Nat} (rowsPositive : 0 < rows) (source : CanonicalState rows) :
    Reduced (normalForm rowsPositive source) :=
  (Classical.choose_spec (reduced_exists rowsPositive source)).2

theorem normalForm_extrema {rows : Nat} (rowsPositive : 0 < rows) (source : CanonicalState rows) :
    numExtrema (normalForm rowsPositive source).height = numExtrema source.height :=
  reduction_extrema rowsPositive (normalForm_path rowsPositive source)

theorem normalForm_eq_of_path {rows : Nat} (rowsPositive : 0 < rows)
    {source target : CanonicalState rows} (path : Relation.ReflTransGen (Step rowsPositive) source target) :
    normalForm rowsPositive source = normalForm rowsPositive target :=
  reduced_unique rowsPositive (normalForm_path rowsPositive source)
    (path.trans (normalForm_path rowsPositive target))
    (normalForm_reduced rowsPositive source) (normalForm_reduced rowsPositive target)

theorem normalForm_fixed {rows : Nat} (rowsPositive : 0 < rows)
    (source : CanonicalState rows) (reduced : Reduced source) : normalForm rowsPositive source = source :=
  reduced_path_identity rowsPositive reduced (normalForm_path rowsPositive source)

theorem normalForm_idempotent {rows : Nat} (rowsPositive : 0 < rows) (source : CanonicalState rows) :
    normalForm rowsPositive (normalForm rowsPositive source) = normalForm rowsPositive source :=
  normalForm_fixed rowsPositive _ (normalForm_reduced rowsPositive source)

noncomputable def normalReducedHeight {rows : Nat} (rowsPositive : 0 < rows)
    (source : CanonicalState rows) : ReducedHF rows where
  W := (normalForm rowsPositive source).width
  hW := (normalForm rowsPositive source).widthValid
  h := (normalForm rowsPositive source).height
  isHeight := (normalForm rowsPositive source).canonical.1
  isCanonical := (normalForm rowsPositive source).canonical.2
  reduced := normalForm_reduced rowsPositive source

theorem normalForm_width_bound {rows : Nat} (rowsPositive : 0 < rows)
    (source : CanonicalState rows) (degree : Nat) (sourceExtrema : numExtrema source.height = degree) :
    (normalForm rowsPositive source).width ≤ 2 * degree - 1 :=
  ReducedHF.width_le_two_d_sub_one (normalReducedHeight rowsPositive source) rowsPositive degree
    ((normalForm_extrema rowsPositive source).trans sourceExtrema)

theorem normalForm_run_bound {rows : Nat} (rowsPositive : 0 < rows)
    (source : CanonicalState rows) (degree : Nat) (sourceExtrema : numExtrema source.height = degree) :
    numFrozenColumns (normalForm rowsPositive source).height + 1 ≤ degree := by
  change numFrozenColumns (normalReducedHeight rowsPositive source).h + 1 ≤ degree
  rw [ReducedHF.numFrozenColumns_eq_numFrozenRuns _ rowsPositive]
  exact numFrozenRuns_lt_of_numExtrema_eq (normalReducedHeight rowsPositive source).h
    (normalReducedHeight rowsPositive source).isHeight rowsPositive
    (by have := (normalReducedHeight rowsPositive source).hW; omega) degree
    ((normalForm_extrema rowsPositive source).trans sourceExtrema)

theorem bounded_states_finite {rows : Nat} (rowsPositive : 0 < rows) (bound : Nat) :
    {source : CanonicalState rows | source.width ≤ bound}.Finite := by
  classical
  let Dimensions := {dimension : Fin (bound + 1) // 2 ≤ dimension.val}
  letI (dimension : Dimensions) :
      Fintype {height : Cell rows dimension.val.val -> Int // IsCanonicalHeight height} :=
    (CanonicalHeights_finite (m := rows) (n := dimension.val.val) (by omega)
      (by have := dimension.property; omega)).fintype
  let Shapes := Sigma fun dimension : Dimensions =>
    {height : Cell rows dimension.val.val -> Int // IsCanonicalHeight height}
  let encode : {source : CanonicalState rows // source.width ≤ bound} -> Shapes := fun source =>
    ⟨⟨⟨source.val.width, by have := source.property; omega⟩, source.val.widthValid⟩,
      ⟨source.val.height, source.val.canonical⟩⟩
  let decode : Shapes -> {source : CanonicalState rows // source.width ≤ bound} := fun shape =>
    ⟨⟨shape.fst.val.val, shape.fst.property, shape.snd.val, shape.snd.property⟩,
      show shape.fst.val.val ≤ bound from by have := shape.fst.val.isLt; omega⟩
  have inverse : Function.LeftInverse decode encode := by
    intro source
    apply Subtype.ext
    cases source with
    | mk source widthBound =>
      cases source
      rfl
  have finiteDomain : Finite {source : CanonicalState rows // source.width ≤ bound} :=
    Finite.of_injective encode inverse.injective
  exact Set.finite_coe_iff.mp finiteDomain

theorem reduced_types_finite {rows : Nat} (rowsPositive : 0 < rows) (degree : Nat) :
    {source : CanonicalState rows | Reduced source ∧ numExtrema source.height = degree}.Finite := by
  apply (bounded_states_finite rowsPositive (2 * degree - 1)).subset
  intro source properties
  have bound := normalForm_width_bound rowsPositive source degree properties.2
  rw [normalForm_fixed rowsPositive source properties.1] at bound
  exact bound

theorem reduction_plan {rows : Nat} (rowsPositive : 0 < rows)
    {source target : CanonicalState rows} (path : Relation.ReflTransGen (Step rowsPositive) source target) :
    ∃ plan : ContractionPlan source.width target.width,
      planTracks rowsPositive plan source.height target.height := by
  induction path using Relation.ReflTransGen.head_induction_on with
  | refl => exact ⟨ContractionPlan.keep _, rfl⟩
  | head firstStep _remainingPath inductionHypothesis =>
    obtain ⟨column, frozen, nextFrozen, rfl⟩ := firstStep
    obtain ⟨remainingPlan, remainingTracks⟩ := inductionHypothesis
    refine ⟨ContractionPlan.delete column frozen.1 frozen.2.1
      (frozenSlope _ rowsPositive column frozen) remainingPlan, ?_⟩
    exact ⟨frozen, nextFrozen, rfl, remainingTracks⟩

theorem normalForm_reconstructs {rows : Nat} (rowsPositive : 0 < rows) (source : CanonicalState rows) :
    ∃ plan : ContractionPlan source.width (normalForm rowsPositive source).width,
      planTracks rowsPositive plan source.height (normalForm rowsPositive source).height ∧
        reconstructPlan plan (normalForm rowsPositive source).height = source.height := by
  obtain ⟨plan, tracks⟩ := reduction_plan rowsPositive (normalForm_path rowsPositive source)
  refine ⟨plan, tracks, ?_⟩
  exact reconstructPlan_of_tracks rowsPositive plan source.height
    (normalForm rowsPositive source).height source.canonical.1 tracks

#print axioms separated_contractions_commute
#print axioms contractState_width
#print axioms contractState_extrema
#print axioms step_width_lt
#print axioms step_extrema
#print axioms reduced_no_step
#print axioms reduction_width_le
#print axioms reduction_extrema
#print axioms reduced_exists
#print axioms reduced_path_identity
#print axioms state_eq_of_width_height
#print axioms separated_steps_join
#print axioms adjacent_states_equal
#print axioms step_diamond
#print axioms reduction_confluent
#print axioms reduced_unique
#print axioms normalForm_path
#print axioms normalForm_reduced
#print axioms normalForm_extrema
#print axioms normalForm_eq_of_path
#print axioms normalForm_fixed
#print axioms normalForm_idempotent
#print axioms normalForm_width_bound
#print axioms normalForm_run_bound
#print axioms bounded_states_finite
#print axioms reduced_types_finite
#print axioms reduction_plan
#print axioms normalForm_reconstructs

end OrigamiCone.Sequel.FrozenReduction
