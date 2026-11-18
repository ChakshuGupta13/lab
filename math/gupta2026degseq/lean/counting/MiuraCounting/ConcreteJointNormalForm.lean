import MiuraCounting.ConcreteRunLengthFibers

namespace OrigamiCone.Sequel.JointReduction

open OrigamiCone FrozenExpansion RunLengthFibers

structure GridState where
  rows : Nat
  rowsValid : 2 ≤ rows
  columns : FrozenReduction.CanonicalState rows

theorem rows_positive (source : GridState) : 0 < source.rows := by
  have bound := source.rowsValid
  omega

def transpose (source : GridState) : GridState where
  rows := source.columns.width
  rowsValid := source.columns.widthValid
  columns := {
    width := source.rows
    widthValid := source.rowsValid
    height := swapH source.columns.height
    canonical := swapH_isCanonical source.columns.canonical }

theorem transpose_involutive (source : GridState) : transpose (transpose source) = source := by
  cases source with
  | mk rows rowsValid columns =>
    cases columns
    rfl

theorem transpose_extrema (source : GridState) :
    numExtrema (transpose source).columns.height = numExtrema source.columns.height :=
  swapH_numExtrema source.columns.height

def ColumnStep (source target : GridState) : Prop :=
  ∃ next : FrozenReduction.CanonicalState source.rows,
    FrozenReduction.Step (rows_positive source) source.columns next ∧
      target = {source with columns := next}

def RowStep (source target : GridState) : Prop := ColumnStep (transpose source) (transpose target)

def Step (source target : GridState) : Prop := ColumnStep source target ∨ RowStep source target

def Reduced (source : GridState) : Prop :=
  FrozenReduction.Reduced source.columns ∧ FrozenReduction.Reduced (transpose source).columns

theorem columnStep_rows {source target : GridState} (step : ColumnStep source target) :
    target.rows = source.rows := by
  obtain ⟨next, _, rfl⟩ := step
  rfl

theorem columnStep_measure {source target : GridState} (step : ColumnStep source target) :
    target.rows + target.columns.width < source.rows + source.columns.width := by
  obtain ⟨next, oldStep, rfl⟩ := step
  exact Nat.add_lt_add_left (FrozenReduction.step_width_lt (rows_positive source) oldStep) source.rows

theorem columnStep_extrema {source target : GridState} (step : ColumnStep source target) :
    numExtrema target.columns.height = numExtrema source.columns.height := by
  obtain ⟨next, oldStep, rfl⟩ := step
  exact FrozenReduction.step_extrema (rows_positive source) oldStep

theorem rowStep_measure {source target : GridState} (step : RowStep source target) :
    target.rows + target.columns.width < source.rows + source.columns.width := by
  have shorter := columnStep_measure step
  change target.columns.width + target.rows < source.columns.width + source.rows at shorter
  omega

theorem rowStep_extrema {source target : GridState} (step : RowStep source target) :
    numExtrema target.columns.height = numExtrema source.columns.height := by
  have counts := columnStep_extrema step
  simpa only [transpose_extrema] using counts

theorem step_measure {source target : GridState} (step : Step source target) :
    target.rows + target.columns.width < source.rows + source.columns.width := by
  rcases step with column | row
  · exact columnStep_measure column
  · exact rowStep_measure row

theorem step_extrema {source target : GridState} (step : Step source target) :
    numExtrema target.columns.height = numExtrema source.columns.height := by
  rcases step with column | row
  · exact columnStep_extrema column
  · exact rowStep_extrema row

theorem columnStep_exists_of_not_reduced (source : GridState)
    (notReduced : ¬ FrozenReduction.Reduced source.columns) :
    ∃ target : GridState, ColumnStep source target := by
  by_contra noNext
  apply notReduced
  intro column _ pair
  let next := FrozenReduction.contractState (rows_positive source) source.columns column pair.1 pair.2
  exact noNext ⟨{source with columns := next}, next, ⟨column, pair.1, pair.2, rfl⟩, rfl⟩

theorem step_exists_of_not_reduced (source : GridState) (notReduced : ¬ Reduced source) :
    ∃ target : GridState, Step source target := by
  by_cases columnsReduced : FrozenReduction.Reduced source.columns
  · have rowsNotReduced : ¬ FrozenReduction.Reduced (transpose source).columns :=
      fun rowsReduced => notReduced ⟨columnsReduced, rowsReduced⟩
    obtain ⟨target, columnStep⟩ := columnStep_exists_of_not_reduced (transpose source) rowsNotReduced
    refine ⟨transpose target, Or.inr ?_⟩
    change ColumnStep (transpose source) (transpose (transpose target))
    rwa [transpose_involutive]
  · obtain ⟨target, columnStep⟩ := columnStep_exists_of_not_reduced source columnsReduced
    exact ⟨target, Or.inl columnStep⟩

theorem reduced_no_step {source : GridState} (reduced : Reduced source)
    {target : GridState} (step : Step source target) : False := by
  rcases step with column | row
  · obtain ⟨next, oldStep, _⟩ := column
    exact FrozenReduction.reduced_no_step (rows_positive source) reduced.1 oldStep
  · obtain ⟨next, oldStep, _⟩ := row
    exact FrozenReduction.reduced_no_step (rows_positive (transpose source)) reduced.2 oldStep

theorem reduced_exists (source : GridState) :
    ∃ target : GridState, Relation.ReflTransGen Step source target ∧ Reduced target := by
  have general : ∀ measure : Nat, ∀ source : GridState, source.rows + source.columns.width = measure ->
      ∃ target : GridState, Relation.ReflTransGen Step source target ∧ Reduced target := by
    intro measure
    induction measure using Nat.strong_induction_on with
    | h measure inductionHypothesis =>
      intro source measureEquality
      by_cases reduced : Reduced source
      · exact ⟨source, Relation.ReflTransGen.refl, reduced⟩
      · obtain ⟨next, nextStep⟩ := step_exists_of_not_reduced source reduced
        have shorter := step_measure nextStep
        obtain ⟨target, path, targetReduced⟩ :=
          inductionHypothesis (next.rows + next.columns.width) (by omega) next rfl
        exact ⟨target, (Relation.ReflTransGen.single nextStep).trans path, targetReduced⟩
  exact general (source.rows + source.columns.width) source rfl

theorem state_eq_of_rows_columns (first second : GridState) (sameRows : first.rows = second.rows)
    (sameColumns : HEq first.columns second.columns) : first = second := by
  cases first
  cases second
  cases sameRows
  cases sameColumns
  rfl

theorem column_refl_lift (source : GridState) {first second : FrozenReduction.CanonicalState source.rows}
    (path : Relation.ReflGen (FrozenReduction.Step (rows_positive source)) first second) :
    Relation.ReflGen Step {source with columns := first} {source with columns := second} := by
  cases path with
  | refl => exact Relation.ReflGen.refl
  | single step => exact Relation.ReflGen.single (Or.inl ⟨second, step, rfl⟩)

theorem columnStep_diamond {source first second : GridState}
    (firstStep : ColumnStep source first) (secondStep : ColumnStep source second) :
    ∃ common : GridState, Relation.ReflGen Step first common ∧ Relation.ReflGen Step second common := by
  obtain ⟨firstColumns, firstOld, rfl⟩ := firstStep
  obtain ⟨secondColumns, secondOld, rfl⟩ := secondStep
  obtain ⟨common, left, right⟩ := FrozenReduction.step_diamond (rows_positive source) firstOld secondOld
  exact ⟨{source with columns := common}, column_refl_lift source left, column_refl_lift source right⟩

theorem step_transpose {source target : GridState} (step : Step source target) :
    Step (transpose source) (transpose target) := by
  rcases step with column | row
  · refine Or.inr ?_
    change ColumnStep (transpose (transpose source)) (transpose (transpose target))
    simpa only [transpose_involutive] using column
  · exact Or.inl row

theorem refl_transpose {source target : GridState} (path : Relation.ReflGen Step source target) :
    Relation.ReflGen Step (transpose source) (transpose target) := by
  cases path with
  | refl => exact Relation.ReflGen.refl
  | single step => exact Relation.ReflGen.single (step_transpose step)

theorem rowStep_diamond {source first second : GridState}
    (firstStep : RowStep source first) (secondStep : RowStep source second) :
    ∃ common : GridState, Relation.ReflGen Step first common ∧ Relation.ReflGen Step second common := by
  obtain ⟨common, left, right⟩ := columnStep_diamond firstStep secondStep
  refine ⟨transpose common, ?_, ?_⟩
  · simpa only [transpose_involutive] using refl_transpose left
  · simpa only [transpose_involutive] using refl_transpose right

theorem rowStep_iff {source target : GridState} : RowStep source target ↔
    ∃ next : FrozenReduction.CanonicalState (transpose source).rows,
      FrozenReduction.Step (rows_positive (transpose source)) (transpose source).columns next ∧
        target = transpose {transpose source with columns := next} := by
  constructor
  · intro step
    obtain ⟨next, oldStep, equality⟩ := step
    exact ⟨next, oldStep, (transpose_involutive target).symm.trans (congrArg transpose equality)⟩
  · rintro ⟨next, oldStep, rfl⟩
    change ColumnStep (transpose source) (transpose (transpose {transpose source with columns := next}))
    rw [transpose_involutive]
    exact ⟨next, oldStep, rfl⟩

theorem mixed_steps_join {source first second : GridState}
    (columnStep : ColumnStep source first) (rowStep : RowStep source second) :
    ∃ common : GridState, RowStep first common ∧ ColumnStep second common := by
  obtain ⟨firstColumns, firstOld, rfl⟩ := columnStep
  obtain ⟨secondRows, secondOld, rfl⟩ := rowStep_iff.mp rowStep
  rcases source with ⟨sourceRows, rowsValid, ⟨sourceWidth, widthValid, height, canonical⟩⟩
  cases sourceRows with
  | zero => omega
  | succ rows =>
    cases sourceWidth with
    | zero => omega
    | succ width =>
      obtain ⟨column, columnFrozen, columnNext, rfl⟩ := firstOld
      obtain ⟨row, rowFrozen, rowNext, rfl⟩ := secondOld
      have rowsPositive : 0 < rows := by omega
      have widthPositive : 0 < width := by omega
      obtain ⟨rowAfter, columnAfter, arraysEqual⟩ := contractAxes_commute height canonical.1
        rowsPositive widthPositive row column rowFrozen columnFrozen
      have rowNextAfter : frozenColumn
          (swapH (contractAt height (by omega) column columnFrozen))
          ⟨row.val + 1, rowAfter.2.1⟩ :=
        (rowFrozen_contractAt_iff height canonical.1 (by omega) column columnFrozen _).mpr rowNext
      have columnNextAfter : frozenColumn (contractRows height (by omega) row rowFrozen)
          ⟨column.val + 1, columnAfter.2.1⟩ :=
        (columnFrozen_contractRows_iff height canonical.1 (by omega) row rowFrozen _).mpr columnNext
      let original : GridState := ⟨rows + 1, rowsValid, ⟨width + 1, widthValid, height, canonical⟩⟩
      let columnState := FrozenReduction.contractState (rows_positive original) original.columns
        column columnFrozen columnNext
      let afterColumn : GridState := {original with columns := columnState}
      let rowState := FrozenReduction.contractState (rows_positive (transpose original)) (transpose original).columns
        row rowFrozen rowNext
      let afterRow : GridState := transpose {transpose original with columns := rowState}
      let leftRows := FrozenReduction.contractState (rows_positive (transpose afterColumn))
        (transpose afterColumn).columns row rowAfter rowNextAfter
      let commonLeft : GridState := transpose {transpose afterColumn with columns := leftRows}
      let rightColumns := FrozenReduction.contractState (rows_positive afterRow) afterRow.columns
        column columnAfter columnNextAfter
      let commonRight : GridState := {afterRow with columns := rightColumns}
      have commonEqual : commonLeft = commonRight := by
        refine state_eq_of_rows_columns _ _ ?_ ?_
        · rfl
        · apply heq_of_eq
          refine FrozenReduction.state_eq_of_width_height _ _ ?_ ?_
          · rfl
          · exact heq_of_eq arraysEqual
      refine ⟨commonLeft, ?_, ?_⟩
      · exact rowStep_iff.mpr ⟨leftRows, ⟨row, rowAfter, rowNextAfter, rfl⟩, rfl⟩
      · rw [commonEqual]
        exact ⟨rightColumns, ⟨column, columnAfter, columnNextAfter, rfl⟩, rfl⟩

theorem step_diamond {source first second : GridState} (firstStep : Step source first)
    (secondStep : Step source second) :
    ∃ common : GridState, Relation.ReflGen Step first common ∧ Relation.ReflGen Step second common := by
  rcases firstStep with firstColumn | firstRow
  · rcases secondStep with secondColumn | secondRow
    · exact columnStep_diamond firstColumn secondColumn
    · obtain ⟨common, afterFirst, afterSecond⟩ := mixed_steps_join firstColumn secondRow
      exact ⟨common, Relation.ReflGen.single (Or.inr afterFirst), Relation.ReflGen.single (Or.inl afterSecond)⟩
  · rcases secondStep with secondColumn | secondRow
    · obtain ⟨common, afterSecond, afterFirst⟩ := mixed_steps_join secondColumn firstRow
      exact ⟨common, Relation.ReflGen.single (Or.inl afterFirst), Relation.ReflGen.single (Or.inr afterSecond)⟩
    · exact rowStep_diamond firstRow secondRow

theorem reduction_confluent {source first second : GridState}
    (firstPath : Relation.ReflTransGen Step source first)
    (secondPath : Relation.ReflTransGen Step source second) :
    Relation.Join (Relation.ReflTransGen Step) first second := by
  apply Relation.church_rosser (r := Step) _ firstPath secondPath
  intro start left right firstStep secondStep
  obtain ⟨common, leftPath, rightPath⟩ := step_diamond firstStep secondStep
  exact ⟨common, leftPath, rightPath.to_reflTransGen⟩

theorem reduced_path_identity {source target : GridState} (reduced : Reduced source)
    (path : Relation.ReflTransGen Step source target) : target = source := by
  exact (Relation.reflTransGen_iff_eq (fun _ step => reduced_no_step reduced step)).mp path

theorem reduced_unique {source first second : GridState}
    (firstPath : Relation.ReflTransGen Step source first)
    (secondPath : Relation.ReflTransGen Step source second)
    (firstReduced : Reduced first) (secondReduced : Reduced second) : first = second := by
  obtain ⟨common, firstToCommon, secondToCommon⟩ := reduction_confluent firstPath secondPath
  exact (reduced_path_identity firstReduced firstToCommon).symm.trans
    (reduced_path_identity secondReduced secondToCommon)

noncomputable def normalForm (source : GridState) : GridState := Classical.choose (reduced_exists source)

theorem normalForm_path (source : GridState) : Relation.ReflTransGen Step source (normalForm source) :=
  (Classical.choose_spec (reduced_exists source)).1

theorem normalForm_reduced (source : GridState) : Reduced (normalForm source) :=
  (Classical.choose_spec (reduced_exists source)).2

theorem normalForm_fixed (source : GridState) (reduced : Reduced source) : normalForm source = source :=
  reduced_path_identity reduced (normalForm_path source)

theorem normalForm_eq_of_path {source target : GridState} (path : Relation.ReflTransGen Step source target) :
    normalForm source = normalForm target :=
  reduced_unique (normalForm_path source) (path.trans (normalForm_path target))
    (normalForm_reduced source) (normalForm_reduced target)

theorem reduction_extrema {source target : GridState} (path : Relation.ReflTransGen Step source target) :
    numExtrema target.columns.height = numExtrema source.columns.height := by
  induction path with
  | refl => rfl
  | tail _ step inductionHypothesis => exact (step_extrema step).trans inductionHypothesis

theorem normalForm_extrema (source : GridState) :
    numExtrema (normalForm source).columns.height = numExtrema source.columns.height :=
  reduction_extrema (normalForm_path source)

theorem columnStep_rowFrozen_iff (source : GridState)
    {target : FrozenReduction.CanonicalState source.rows}
    (step : FrozenReduction.Step (rows_positive source) source.columns target) (row : Fin source.rows) :
    frozenColumn (swapH target.height) row ↔ frozenColumn (swapH source.columns.height) row := by
  obtain ⟨column, frozen, nextFrozen, rfl⟩ := step
  rcases source with ⟨rows, rowsValid, ⟨width, widthValid, height, canonical⟩⟩
  cases width with
  | zero => omega
  | succ width => exact rowFrozen_contractAt_iff height canonical.1 (by omega) column frozen row

theorem active_iff_of_frozen_iff {firstRows secondRows width : Nat}
    (first : Cell firstRows width -> Int) (second : Cell secondRows width -> Int)
    (firstValid : IsHeight first) (secondValid : IsHeight second)
    (firstPositive : 0 < firstRows) (secondPositive : 0 < secondRows) (widthValid : 2 ≤ width)
    (column : Fin width) (frozenSame : frozenColumn first column ↔ frozenColumn second column) :
    activeColumn first column ↔ activeColumn second column := by
  constructor
  · intro firstActive
    rcases column_active_or_frozen second secondValid secondPositive widthValid column with active | frozen
    · exact active
    · exact False.elim (frozenColumn_not_active first column (frozenSame.mpr frozen) firstActive)
  · intro secondActive
    rcases column_active_or_frozen first firstValid firstPositive widthValid column with active | frozen
    · exact active
    · exact False.elim (frozenColumn_not_active second column (frozenSame.mp frozen) secondActive)

theorem runProfiles_eq_of_active_iff {firstRows secondRows width : Nat}
    (first : Cell firstRows width -> Int) (second : Cell secondRows width -> Int)
    (activeSame : ∀ column : Fin width, activeColumn first column ↔ activeColumn second column) :
    numFrozenRuns first = numFrozenRuns second ∧
      ∀ rank : Nat, runLengthAtRank first rank = runLengthAtRank second rank := by
  have predicateEquality : activeColumn first = activeColumn second := funext (fun column => propext (activeSame column))
  have startsEquality : frozenRunStart first = frozenRunStart second := by
    funext column
    simp only [frozenRunStart, predicateEquality]
  constructor
  · simp only [numFrozenRuns, startsEquality]
  · intro rank
    simp only [runLengthAtRank, runPrefixCount, startsEquality, predicateEquality]

theorem columnStep_rowProfiles (source : GridState)
    {target : FrozenReduction.CanonicalState source.rows}
    (step : FrozenReduction.Step (rows_positive source) source.columns target) :
    numFrozenRuns (swapH target.height) = numFrozenRuns (swapH source.columns.height) ∧
      ∀ rank : Nat, runLengthAtRank (swapH target.height) rank =
        runLengthAtRank (swapH source.columns.height) rank := by
  apply runProfiles_eq_of_active_iff
  intro row
  exact active_iff_of_frozen_iff (swapH target.height) (swapH source.columns.height)
    (swapH_isHeight target.canonical.1) (swapH_isHeight source.columns.canonical.1)
    (by have := target.widthValid; omega) (rows_positive (transpose source)) source.rowsValid row
    (columnStep_rowFrozen_iff source step row)

theorem columnStep_rowReduced_iff (source : GridState)
    {target : FrozenReduction.CanonicalState source.rows}
    (step : FrozenReduction.Step (rows_positive source) source.columns target) :
    FrozenReduction.Reduced (transpose {source with columns := target}).columns ↔
      FrozenReduction.Reduced (transpose source).columns := by
  constructor
  · intro reduced row nextWithin pair
    exact reduced row nextWithin ⟨(columnStep_rowFrozen_iff source step row).mpr pair.1,
      (columnStep_rowFrozen_iff source step _).mpr pair.2⟩
  · intro reduced row nextWithin pair
    exact reduced row nextWithin ⟨(columnStep_rowFrozen_iff source step row).mp pair.1,
      (columnStep_rowFrozen_iff source step _).mp pair.2⟩

theorem column_path_lift (source : GridState) {first second : FrozenReduction.CanonicalState source.rows}
    (path : Relation.ReflTransGen (FrozenReduction.Step (rows_positive source)) first second) :
    Relation.ReflTransGen Step {source with columns := first} {source with columns := second} := by
  induction path with
  | refl => exact Relation.ReflTransGen.refl
  | tail _ step inductionHypothesis => exact inductionHypothesis.tail (Or.inl ⟨_, step, rfl⟩)

theorem reduction_transpose {source target : GridState} (path : Relation.ReflTransGen Step source target) :
    Relation.ReflTransGen Step (transpose source) (transpose target) := by
  induction path with
  | refl => exact Relation.ReflTransGen.refl
  | tail _ step inductionHypothesis => exact inductionHypothesis.tail (step_transpose step)

theorem columnPath_rowProfiles (source : GridState) {first second : FrozenReduction.CanonicalState source.rows}
    (path : Relation.ReflTransGen (FrozenReduction.Step (rows_positive source)) first second) :
    numFrozenRuns (swapH second.height) = numFrozenRuns (swapH first.height) ∧
      ∀ rank : Nat, runLengthAtRank (swapH second.height) rank = runLengthAtRank (swapH first.height) rank := by
  induction path with
  | refl => exact ⟨rfl, fun _ => rfl⟩
  | tail _ step inductionHypothesis =>
    have last := columnStep_rowProfiles {source with columns := _} step
    exact ⟨last.1.trans inductionHypothesis.1, fun rank => (last.2 rank).trans (inductionHypothesis.2 rank)⟩

theorem columnPath_rowReduced_iff (source : GridState)
    {first second : FrozenReduction.CanonicalState source.rows}
    (path : Relation.ReflTransGen (FrozenReduction.Step (rows_positive source)) first second) :
    FrozenReduction.Reduced (transpose {source with columns := second}).columns ↔
      FrozenReduction.Reduced (transpose {source with columns := first}).columns := by
  induction path with
  | refl => rfl
  | tail _ step inductionHypothesis =>
    exact (columnStep_rowReduced_iff {source with columns := _} step).trans inductionHypothesis

noncomputable def columnNormalForm (source : GridState) : GridState :=
  {source with columns := FrozenReduction.normalForm (rows_positive source) source.columns}

noncomputable def rowNormalForm (source : GridState) : GridState :=
  transpose (columnNormalForm (transpose source))

theorem columnNormalForm_path (source : GridState) :
    Relation.ReflTransGen Step source (columnNormalForm source) :=
  column_path_lift source (FrozenReduction.normalForm_path (rows_positive source) source.columns)

theorem rowNormalForm_path (source : GridState) : Relation.ReflTransGen Step source (rowNormalForm source) := by
  have path := reduction_transpose (columnNormalForm_path (transpose source))
  simpa only [transpose_involutive] using path

theorem columnNormalForm_columns_reduced (source : GridState) :
    FrozenReduction.Reduced (columnNormalForm source).columns :=
  FrozenReduction.normalForm_reduced (rows_positive source) source.columns

theorem columnNormalForm_rows_reduced_iff (source : GridState) :
    FrozenReduction.Reduced (transpose (columnNormalForm source)).columns ↔
      FrozenReduction.Reduced (transpose source).columns :=
  columnPath_rowReduced_iff source (FrozenReduction.normalForm_path (rows_positive source) source.columns)

theorem rowNormalForm_rows_reduced (source : GridState) :
    FrozenReduction.Reduced (transpose (rowNormalForm source)).columns := by
  simpa only [rowNormalForm, transpose_involutive] using columnNormalForm_columns_reduced (transpose source)

theorem rowNormalForm_columns_reduced_iff (source : GridState) :
    FrozenReduction.Reduced (rowNormalForm source).columns ↔ FrozenReduction.Reduced source.columns := by
  simpa only [transpose_involutive] using columnNormalForm_rows_reduced_iff (transpose source)

theorem column_then_row_reduced (source : GridState) : Reduced (rowNormalForm (columnNormalForm source)) :=
  ⟨(rowNormalForm_columns_reduced_iff (columnNormalForm source)).mpr (columnNormalForm_columns_reduced source),
    rowNormalForm_rows_reduced (columnNormalForm source)⟩

theorem row_then_column_reduced (source : GridState) : Reduced (columnNormalForm (rowNormalForm source)) :=
  ⟨columnNormalForm_columns_reduced (rowNormalForm source),
    (columnNormalForm_rows_reduced_iff (rowNormalForm source)).mpr (rowNormalForm_rows_reduced source)⟩

theorem column_then_row_eq_normalForm (source : GridState) :
    rowNormalForm (columnNormalForm source) = normalForm source :=
  (reduced_unique (normalForm_path source)
    ((columnNormalForm_path source).trans (rowNormalForm_path (columnNormalForm source)))
    (normalForm_reduced source) (column_then_row_reduced source)).symm

theorem row_then_column_eq_normalForm (source : GridState) :
    columnNormalForm (rowNormalForm source) = normalForm source :=
  (reduced_unique (normalForm_path source)
    ((rowNormalForm_path source).trans (columnNormalForm_path (rowNormalForm source)))
    (normalForm_reduced source) (row_then_column_reduced source)).symm

theorem axis_orders_equal (source : GridState) :
    rowNormalForm (columnNormalForm source) = columnNormalForm (rowNormalForm source) :=
  (column_then_row_eq_normalForm source).trans (row_then_column_eq_normalForm source).symm

theorem columnNormalForm_rowProfiles (source : GridState) :
    numFrozenRuns (swapH (columnNormalForm source).columns.height) = numFrozenRuns (swapH source.columns.height) ∧
      ∀ rank : Nat, runLengthAtRank (swapH (columnNormalForm source).columns.height) rank =
        runLengthAtRank (swapH source.columns.height) rank :=
  columnPath_rowProfiles source (FrozenReduction.normalForm_path (rows_positive source) source.columns)

theorem rowNormalForm_columnProfiles (source : GridState) :
    numFrozenRuns (rowNormalForm source).columns.height = numFrozenRuns source.columns.height ∧
      ∀ rank : Nat, runLengthAtRank (rowNormalForm source).columns.height rank =
        runLengthAtRank source.columns.height rank :=
  columnNormalForm_rowProfiles (transpose source)

theorem columnNormalForm_runCount (source : GridState) :
    numFrozenRuns (columnNormalForm source).columns.height = numFrozenRuns source.columns.height :=
  reduction_run_count (rows_positive source)
    (FrozenReduction.normalForm_path (rows_positive source) source.columns)

theorem rowNormalForm_rowRunCount (source : GridState) :
    numFrozenRuns (swapH (rowNormalForm source).columns.height) = numFrozenRuns (swapH source.columns.height) :=
  columnNormalForm_runCount (transpose source)

theorem normalForm_columnRuns (source : GridState) :
    numFrozenRuns (normalForm source).columns.height = numFrozenRuns source.columns.height := by
  rw [← column_then_row_eq_normalForm source]
  exact (rowNormalForm_columnProfiles (columnNormalForm source)).1.trans (columnNormalForm_runCount source)

theorem normalForm_rowRuns (source : GridState) :
    numFrozenRuns (swapH (normalForm source).columns.height) = numFrozenRuns (swapH source.columns.height) := by
  rw [← column_then_row_eq_normalForm source]
  exact (rowNormalForm_rowRunCount (columnNormalForm source)).trans (columnNormalForm_rowProfiles source).1

theorem reduced_column_bounds (source : GridState) (reduced : FrozenReduction.Reduced source.columns)
    (degree : Nat) (counts : numExtrema source.columns.height = degree) :
    source.columns.width ≤ 2 * degree - 1 ∧ numFrozenRuns source.columns.height + 1 ≤ degree := by
  let reducedHeight : ReducedHF source.rows := ⟨source.columns.width, source.columns.widthValid,
    source.columns.height, source.columns.canonical.1, source.columns.canonical.2, reduced⟩
  have runBound := FrozenReduction.normalForm_run_bound (rows_positive source) source.columns degree counts
  rw [FrozenReduction.normalForm_fixed (rows_positive source) source.columns reduced,
    ReducedHF.numFrozenColumns_eq_numFrozenRuns reducedHeight (rows_positive source)] at runBound
  exact ⟨ReducedHF.width_le_two_d_sub_one reducedHeight (rows_positive source) degree counts, runBound⟩

theorem reduced_bounds (source : GridState) (reduced : Reduced source)
    (degree : Nat) (counts : numExtrema source.columns.height = degree) :
    source.rows ≤ 2 * degree - 1 ∧ source.columns.width ≤ 2 * degree - 1 ∧
      numFrozenRuns (swapH source.columns.height) + 1 ≤ degree ∧
        numFrozenRuns source.columns.height + 1 ≤ degree := by
  have columnBounds := reduced_column_bounds source reduced.1 degree counts
  have rowBounds := reduced_column_bounds (transpose source) reduced.2 degree ((transpose_extrema source).trans counts)
  exact ⟨rowBounds.1, columnBounds.1, rowBounds.2, columnBounds.2⟩

theorem normalForm_bounds (source : GridState) (degree : Nat)
    (counts : numExtrema source.columns.height = degree) :
    (normalForm source).rows ≤ 2 * degree - 1 ∧ (normalForm source).columns.width ≤ 2 * degree - 1 ∧
      numFrozenRuns (swapH (normalForm source).columns.height) + 1 ≤ degree ∧
        numFrozenRuns (normalForm source).columns.height + 1 ≤ degree :=
  reduced_bounds (normalForm source) (normalForm_reduced source) degree ((normalForm_extrema source).trans counts)

theorem reduced_zero_column_runs_bound (source : GridState)
    (reduced : FrozenReduction.Reduced source.columns) (degree : Nat)
    (counts : numExtrema source.columns.height = degree) (noRuns : numFrozenRuns source.columns.height = 0) :
    source.columns.width ≤ degree := by
  let reducedHeight : ReducedHF source.rows := ⟨source.columns.width, source.columns.widthValid,
    source.columns.height, source.columns.canonical.1, source.columns.canonical.2, reduced⟩
  have partition := numActive_add_numFrozen_eq_n source.columns.height source.columns.canonical.1
    (rows_positive source) source.columns.widthValid
  have frozenCount := ReducedHF.numFrozenColumns_eq_numFrozenRuns reducedHeight (rows_positive source)
  have activeBound := numActiveColumns_le_of_numExtrema_eq source.columns.height degree counts
  change numFrozenColumns source.columns.height = numFrozenRuns source.columns.height at frozenCount
  omega

theorem reduced_zero_row_runs_bound (source : GridState) (reduced : Reduced source) (degree : Nat)
    (counts : numExtrema source.columns.height = degree)
    (noRuns : numFrozenRuns (swapH source.columns.height) = 0) : source.rows ≤ degree :=
  reduced_zero_column_runs_bound (transpose source) reduced.2 degree ((transpose_extrema source).trans counts) noRuns

theorem bounded_grids_finite (bound : Nat) :
    {source : GridState | source.rows ≤ bound ∧ source.columns.width ≤ bound}.Finite := by
  classical
  let Dimensions := {dimension : Fin (bound + 1) // 2 ≤ dimension.val}
  letI (dimension : Dimensions) :
      Fintype {columns : FrozenReduction.CanonicalState dimension.val.val // columns.width ≤ bound} :=
    (FrozenReduction.bounded_states_finite (rows := dimension.val.val)
      (by have := dimension.property; omega) bound).fintype
  let Shapes := Sigma fun dimension : Dimensions =>
    {columns : FrozenReduction.CanonicalState dimension.val.val // columns.width ≤ bound}
  let encode : {source : GridState // source.rows ≤ bound ∧ source.columns.width ≤ bound} -> Shapes :=
    fun source => ⟨⟨⟨source.val.rows, by have := source.property.1; omega⟩, source.val.rowsValid⟩,
      ⟨source.val.columns, source.property.2⟩⟩
  let decode : Shapes -> {source : GridState // source.rows ≤ bound ∧ source.columns.width ≤ bound} :=
    fun shape => ⟨⟨shape.fst.val.val, shape.fst.property, shape.snd.val⟩,
      ⟨by
        change shape.fst.val.val ≤ bound
        have := shape.fst.val.isLt
        omega, shape.snd.property⟩⟩
  have inverse : Function.LeftInverse decode encode := by
    intro source
    apply Subtype.ext
    cases source with
    | mk source bounds =>
      cases source
      rfl
  have finiteDomain : Finite {source : GridState // source.rows ≤ bound ∧ source.columns.width ≤ bound} :=
    Finite.of_injective encode inverse.injective
  exact Set.finite_coe_iff.mp finiteDomain

theorem reduced_types_finite (degree : Nat) :
    {source : GridState | Reduced source ∧ numExtrema source.columns.height = degree}.Finite := by
  apply (bounded_grids_finite (2 * degree - 1)).subset
  intro source properties
  have bounds := reduced_bounds source properties.1 degree properties.2
  exact ⟨bounds.1, bounds.2.1⟩

theorem columnStep_separable_iff {source target : GridState} (step : ColumnStep source target) :
    AdditivelySeparable target.columns.height ↔ AdditivelySeparable source.columns.height := by
  obtain ⟨next, ⟨column, frozen, nextFrozen, rfl⟩, rfl⟩ := step
  rcases source with ⟨rows, rowsValid, ⟨width, widthValid, height, canonical⟩⟩
  cases width with
  | zero => omega
  | succ width => exact contractAt_separable_iff height canonical.1 (by omega) column frozen

theorem rowStep_separable_iff {source target : GridState} (step : RowStep source target) :
    AdditivelySeparable target.columns.height ↔ AdditivelySeparable source.columns.height := by
  have equivalent := columnStep_separable_iff step
  simpa only [transpose, separable_swap_iff] using equivalent

theorem step_separable_iff {source target : GridState} (step : Step source target) :
    AdditivelySeparable target.columns.height ↔ AdditivelySeparable source.columns.height := by
  rcases step with column | row
  · exact columnStep_separable_iff column
  · exact rowStep_separable_iff row

theorem reduction_separable_iff {source target : GridState} (path : Relation.ReflTransGen Step source target) :
    AdditivelySeparable target.columns.height ↔ AdditivelySeparable source.columns.height := by
  induction path with
  | refl => rfl
  | tail _ step inductionHypothesis => exact (step_separable_iff step).trans inductionHypothesis

theorem normalForm_separable_iff (source : GridState) :
    AdditivelySeparable (normalForm source).columns.height ↔ AdditivelySeparable source.columns.height :=
  reduction_separable_iff (normalForm_path source)

#print axioms rows_positive
#print axioms transpose_involutive
#print axioms transpose_extrema
#print axioms columnStep_rows
#print axioms columnStep_measure
#print axioms columnStep_extrema
#print axioms rowStep_measure
#print axioms rowStep_extrema
#print axioms step_measure
#print axioms step_extrema
#print axioms columnStep_exists_of_not_reduced
#print axioms step_exists_of_not_reduced
#print axioms reduced_no_step
#print axioms reduced_exists
#print axioms state_eq_of_rows_columns
#print axioms column_refl_lift
#print axioms columnStep_diamond
#print axioms step_transpose
#print axioms refl_transpose
#print axioms rowStep_diamond
#print axioms rowStep_iff
#print axioms mixed_steps_join
#print axioms step_diamond
#print axioms reduction_confluent
#print axioms reduced_path_identity
#print axioms reduced_unique
#print axioms normalForm_path
#print axioms normalForm_reduced
#print axioms normalForm_fixed
#print axioms normalForm_eq_of_path
#print axioms reduction_extrema
#print axioms normalForm_extrema
#print axioms columnStep_rowFrozen_iff
#print axioms active_iff_of_frozen_iff
#print axioms runProfiles_eq_of_active_iff
#print axioms columnStep_rowProfiles
#print axioms columnStep_rowReduced_iff
#print axioms column_path_lift
#print axioms reduction_transpose
#print axioms columnPath_rowProfiles
#print axioms columnPath_rowReduced_iff
#print axioms columnNormalForm_path
#print axioms rowNormalForm_path
#print axioms columnNormalForm_columns_reduced
#print axioms columnNormalForm_rows_reduced_iff
#print axioms rowNormalForm_rows_reduced
#print axioms rowNormalForm_columns_reduced_iff
#print axioms column_then_row_reduced
#print axioms row_then_column_reduced
#print axioms column_then_row_eq_normalForm
#print axioms row_then_column_eq_normalForm
#print axioms axis_orders_equal
#print axioms columnNormalForm_rowProfiles
#print axioms rowNormalForm_columnProfiles
#print axioms columnNormalForm_runCount
#print axioms rowNormalForm_rowRunCount
#print axioms normalForm_columnRuns
#print axioms normalForm_rowRuns
#print axioms reduced_column_bounds
#print axioms reduced_bounds
#print axioms normalForm_bounds
#print axioms reduced_zero_column_runs_bound
#print axioms reduced_zero_row_runs_bound
#print axioms bounded_grids_finite
#print axioms reduced_types_finite
#print axioms columnStep_separable_iff
#print axioms rowStep_separable_iff
#print axioms step_separable_iff
#print axioms reduction_separable_iff
#print axioms normalForm_separable_iff

end OrigamiCone.Sequel.JointReduction
