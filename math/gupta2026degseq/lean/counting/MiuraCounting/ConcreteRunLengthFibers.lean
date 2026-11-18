import MiuraCounting.ConcreteReductionNormalForm

namespace OrigamiCone.Sequel.RunLengthFibers

open OrigamiCone FrozenExpansion FrozenReduction

open scoped BigOperators

def runStartLift {width : Nat} (deleted : Fin (width + 1)) (column : Fin width) : Fin (width + 1) :=
  if before : column.val ≤ deleted.val then
    ⟨column.val, by have := column.isLt; omega⟩
  else ⟨column.val + 1, by have := column.isLt; omega⟩

theorem runStartLift_injective {width : Nat} (deleted : Fin (width + 1)) :
    Function.Injective (runStartLift deleted) := by
  intro first second equality
  have values := congrArg Fin.val equality
  by_cases firstBefore : first.val ≤ deleted.val <;>
    by_cases secondBefore : second.val ≤ deleted.val <;>
      simp [runStartLift, firstBefore, secondBefore] at values <;>
        apply Fin.ext <;> omega

theorem activity_runStartLift_iff {rows width : Nat} (height : Cell rows (width + 1) -> Int)
    (heightValid : IsHeight height) (rowsPositive : 0 < rows)
    (deleted : Fin (width + 1)) (deletedFrozen : frozenColumn height deleted)
    (nextFrozen : frozenColumn height ⟨deleted.val + 1, deletedFrozen.2.1⟩) (column : Fin width) :
    activeColumn (contractAt height rowsPositive deleted deletedFrozen) column ↔
      activeColumn height (runStartLift deleted column) := by
  by_cases seam : column.val = deleted.val
  · have lifted : runStartLift deleted column = deleted := by
      apply Fin.ext
      simp [runStartLift, seam]
    have sameColumn : (⟨deleted.val, by have := deletedFrozen.2.1; omega⟩ : Fin width) = column :=
      Fin.ext seam.symm
    have contractedInactive : ¬ activeColumn (contractAt height rowsPositive deleted deletedFrozen) column := by
      have markerFrozen := contracted_marker_frozen height heightValid rowsPositive deleted deletedFrozen nextFrozen
      rw [sameColumn] at markerFrozen
      exact frozenColumn_not_active _ _ markerFrozen
    rw [lifted]
    exact iff_of_false contractedInactive (frozenColumn_not_active height deleted deletedFrozen)
  · have sameLift : keptColumn deleted column = runStartLift deleted column := by
      by_cases before : column.val < deleted.val
      · simp [keptColumn, runStartLift, before, Nat.le_of_lt before]
      · have after : ¬ column.val ≤ deleted.val := by omega
        simp [keptColumn, runStartLift, before, after]
    rw [← sameLift]
    exact activeColumn_contractAt_iff height heightValid rowsPositive deleted deletedFrozen column

theorem runStartLift_value {width : Nat} (deleted : Fin (width + 1)) (column : Fin width) :
    (runStartLift deleted column).val = if column.val ≤ deleted.val then column.val else column.val + 1 := by
  unfold runStartLift
  split_ifs <;> rfl

theorem runStartLift_zero_iff {width : Nat} (deleted : Fin (width + 1)) (column : Fin width) :
    (runStartLift deleted column).val = 0 ↔ column.val = 0 := by
  rw [runStartLift_value]
  split_ifs with before
  · rfl
  · have nonzero : column.val ≠ 0 := by omega
    simp [nonzero]

theorem runStartLift_positive_iff {width : Nat} (deleted : Fin (width + 1)) (column : Fin width) :
    0 < (runStartLift deleted column).val ↔ 0 < column.val := by
  rw [runStartLift_value]
  split_ifs <;> omega

theorem predecessor_activity_iff {rows width : Nat} (height : Cell rows (width + 1) -> Int)
    (heightValid : IsHeight height) (rowsPositive : 0 < rows)
    (deleted : Fin (width + 1)) (deletedFrozen : frozenColumn height deleted)
    (nextFrozen : frozenColumn height ⟨deleted.val + 1, deletedFrozen.2.1⟩)
    (column : Fin width) (columnPositive : 0 < column.val) :
    activeColumn (contractAt height rowsPositive deleted deletedFrozen)
        ⟨column.val - 1, by have := column.isLt; omega⟩ ↔
      activeColumn height
        ⟨(runStartLift deleted column).val - 1,
          by have := (runStartLift deleted column).isLt; omega⟩ := by
  by_cases seam : column.val = deleted.val + 1
  · have previous : (⟨column.val - 1, by have := column.isLt; omega⟩ : Fin width) =
        ⟨deleted.val, by have := deletedFrozen.2.1; omega⟩ := by
      apply Fin.ext
      change column.val - 1 = deleted.val
      omega
    have liftedValue : (runStartLift deleted column).val = deleted.val + 2 := by
      rw [runStartLift_value, if_neg (by omega)]
      omega
    have oldPrevious :
        (⟨(runStartLift deleted column).val - 1,
          by have := (runStartLift deleted column).isLt; omega⟩ : Fin (width + 1)) =
          ⟨deleted.val + 1, deletedFrozen.2.1⟩ := by
      apply Fin.ext
      change (runStartLift deleted column).val - 1 = deleted.val + 1
      omega
    rw [previous, oldPrevious]
    exact iff_of_false
      (frozenColumn_not_active _ _ (contracted_marker_frozen height heightValid rowsPositive
        deleted deletedFrozen nextFrozen)) (frozenColumn_not_active height _ nextFrozen)
  · have previousLift :
        runStartLift deleted ⟨column.val - 1, by have := column.isLt; omega⟩ =
          ⟨(runStartLift deleted column).val - 1,
            by have := (runStartLift deleted column).isLt; omega⟩ := by
      apply Fin.ext
      by_cases before : column.val ≤ deleted.val
      · have previousBefore : column.val - 1 ≤ deleted.val := by omega
        simp [runStartLift, before, previousBefore]
      · have previousAfter : ¬ column.val - 1 ≤ deleted.val := by omega
        have preceding : column.val - 1 + 1 = column.val := by omega
        simp [runStartLift, before, previousAfter, preceding]
    have relation := activity_runStartLift_iff height heightValid rowsPositive deleted deletedFrozen nextFrozen
      ⟨column.val - 1, by have := column.isLt; omega⟩
    rw [previousLift] at relation
    exact relation

theorem frozenRunStart_contractAt_iff {rows width : Nat} (height : Cell rows (width + 1) -> Int)
    (heightValid : IsHeight height) (rowsPositive : 0 < rows)
    (deleted : Fin (width + 1)) (deletedFrozen : frozenColumn height deleted)
    (nextFrozen : frozenColumn height ⟨deleted.val + 1, deletedFrozen.2.1⟩) (column : Fin width) :
    frozenRunStart (contractAt height rowsPositive deleted deletedFrozen) column ↔
      frozenRunStart height (runStartLift deleted column) := by
  have current := activity_runStartLift_iff height heightValid rowsPositive deleted deletedFrozen nextFrozen column
  constructor
  · rintro ⟨inactive, first | ⟨positive, previous⟩⟩
    · exact ⟨(not_congr current).mp inactive, Or.inl ((runStartLift_zero_iff deleted column).mpr first)⟩
    · refine ⟨(not_congr current).mp inactive,
        Or.inr ⟨(runStartLift_positive_iff deleted column).mpr positive, ?_⟩⟩
      exact (predecessor_activity_iff height heightValid rowsPositive deleted deletedFrozen nextFrozen column positive).mp previous
  · rintro ⟨inactive, first | ⟨positive, previous⟩⟩
    · exact ⟨(not_congr current).mpr inactive, Or.inl ((runStartLift_zero_iff deleted column).mp first)⟩
    · have columnPositive := (runStartLift_positive_iff deleted column).mp positive
      refine ⟨(not_congr current).mpr inactive, Or.inr ⟨columnPositive, ?_⟩⟩
      exact (predecessor_activity_iff height heightValid rowsPositive deleted deletedFrozen nextFrozen column columnPositive).mpr previous

theorem contractAt_numFrozenRuns {rows width : Nat} (height : Cell rows (width + 1) -> Int)
    (heightValid : IsHeight height) (rowsPositive : 0 < rows)
    (deleted : Fin (width + 1)) (deletedFrozen : frozenColumn height deleted)
    (nextFrozen : frozenColumn height ⟨deleted.val + 1, deletedFrozen.2.1⟩) :
    numFrozenRuns (contractAt height rowsPositive deleted deletedFrozen) = numFrozenRuns height := by
  classical
  unfold numFrozenRuns
  apply Finset.card_bij (fun column _ => runStartLift deleted column)
  · intro column member
    refine Finset.mem_filter.mpr ⟨Finset.mem_univ _, ?_⟩
    exact (frozenRunStart_contractAt_iff height heightValid rowsPositive deleted deletedFrozen nextFrozen column).mp
      (Finset.mem_filter.mp member).2
  · intro first _ second _ equality
    exact runStartLift_injective deleted equality
  · intro column member
    have start := (Finset.mem_filter.mp member).2
    have excluded : column.val ≠ deleted.val + 1 := by
      intro atRemovedSuccessor
      rcases start.2 with first | ⟨positive, previous⟩
      · omega
      · have previousColumn :
            (⟨column.val - 1, by have := column.isLt; omega⟩ : Fin (width + 1)) = deleted := by
          apply Fin.ext
          change column.val - 1 = deleted.val
          omega
        rw [previousColumn] at previous
        exact frozenColumn_not_active height deleted deletedFrozen previous
    have preimage : ∃ original : Fin width, runStartLift deleted original = column := by
      by_cases before : column.val ≤ deleted.val
      · have within : column.val < width := by have := deletedFrozen.2.1; omega
        refine ⟨⟨column.val, within⟩, ?_⟩
        apply Fin.ext
        simp [runStartLift, before]
      · have within : column.val - 1 < width := by have := column.isLt; omega
        have after : ¬ column.val - 1 ≤ deleted.val := by omega
        have restored : column.val - 1 + 1 = column.val := by omega
        refine ⟨⟨column.val - 1, within⟩, ?_⟩
        apply Fin.ext
        simp [runStartLift, after, restored]
    obtain ⟨original, lifted⟩ := preimage
    refine ⟨original, Finset.mem_filter.mpr ⟨Finset.mem_univ _, ?_⟩, lifted⟩
    apply (frozenRunStart_contractAt_iff height heightValid rowsPositive deleted deletedFrozen nextFrozen original).mpr
    rw [lifted]
    exact start

theorem step_run_count {rows : Nat} (rowsPositive : 0 < rows)
    {source target : CanonicalState rows} (step : Step rowsPositive source target) :
    numFrozenRuns target.height = numFrozenRuns source.height := by
  obtain ⟨column, frozen, nextFrozen, rfl⟩ := step
  cases source with
  | mk width widthValid height canonical =>
    cases width with
    | zero => omega
    | succ width => exact contractAt_numFrozenRuns height canonical.1 rowsPositive column frozen nextFrozen

theorem reduction_run_count {rows : Nat} (rowsPositive : 0 < rows)
    {source target : CanonicalState rows} (path : Relation.ReflTransGen (Step rowsPositive) source target) :
    numFrozenRuns target.height = numFrozenRuns source.height := by
  induction path with
  | refl => rfl
  | tail _ step inductionHypothesis => exact (step_run_count rowsPositive step).trans inductionHypothesis

theorem normalForm_run_count {rows : Nat} (rowsPositive : 0 < rows) (source : CanonicalState rows) :
    numFrozenRuns source.height = numFrozenColumns (normalForm rowsPositive source).height := by
  calc
    _ = numFrozenRuns (normalForm rowsPositive source).height :=
      (reduction_run_count rowsPositive (normalForm_path rowsPositive source)).symm
    _ = _ := (ReducedHF.numFrozenColumns_eq_numFrozenRuns
      (normalReducedHeight rowsPositive source) rowsPositive).symm

theorem keptColumn_injective {width : Nat} (deleted : Fin (width + 1)) :
    Function.Injective (keptColumn deleted) := by
  intro first second equality
  have values := congrArg Fin.val equality
  by_cases firstBefore : first.val < deleted.val <;>
    by_cases secondBefore : second.val < deleted.val <;>
      simp [keptColumn, firstBefore, secondBefore] at values <;>
        apply Fin.ext <;> omega

theorem contractAt_numActiveColumns {rows width : Nat} (height : Cell rows (width + 1) -> Int)
    (heightValid : IsHeight height) (rowsPositive : 0 < rows)
    (deleted : Fin (width + 1)) (deletedFrozen : frozenColumn height deleted) :
    numActiveColumns (contractAt height rowsPositive deleted deletedFrozen) = numActiveColumns height := by
  classical
  unfold numActiveColumns
  apply Finset.card_bij (fun column _ => keptColumn deleted column)
  · intro column member
    refine Finset.mem_filter.mpr ⟨Finset.mem_univ _, ?_⟩
    exact (activeColumn_contractAt_iff height heightValid rowsPositive deleted deletedFrozen column).mp
      (Finset.mem_filter.mp member).2
  · intro first _ second _ equality
    exact keptColumn_injective deleted equality
  · intro column member
    have active := (Finset.mem_filter.mp member).2
    have excluded : column ≠ deleted := by
      intro same
      rw [same] at active
      exact frozenColumn_not_active height deleted deletedFrozen active
    obtain ⟨⟨row, original⟩, lifted⟩ := cellExpand_surjOn deleted deletedFrozen.1 deletedFrozen.2.1
      (⟨0, rowsPositive⟩, column) excluded
    have columnEquality : keptColumn deleted original = column := by
      have equality := congrArg Prod.snd lifted
      simpa only [cellExpand_keptColumn] using equality
    refine ⟨original, Finset.mem_filter.mpr ⟨Finset.mem_univ _, ?_⟩, columnEquality⟩
    apply (activeColumn_contractAt_iff height heightValid rowsPositive deleted deletedFrozen original).mpr
    rw [columnEquality]
    exact active

theorem step_active_count {rows : Nat} (rowsPositive : 0 < rows)
    {source target : CanonicalState rows} (step : Step rowsPositive source target) :
    numActiveColumns target.height = numActiveColumns source.height := by
  obtain ⟨column, frozen, nextFrozen, rfl⟩ := step
  cases source with
  | mk width widthValid height canonical =>
    cases width with
    | zero => omega
    | succ width => exact contractAt_numActiveColumns height canonical.1 rowsPositive column frozen

theorem reduction_active_count {rows : Nat} (rowsPositive : 0 < rows)
    {source target : CanonicalState rows} (path : Relation.ReflTransGen (Step rowsPositive) source target) :
    numActiveColumns target.height = numActiveColumns source.height := by
  induction path with
  | refl => rfl
  | tail _ step inductionHypothesis => exact (step_active_count rowsPositive step).trans inductionHypothesis

theorem normalForm_active_count {rows : Nat} (rowsPositive : 0 < rows) (source : CanonicalState rows) :
    numActiveColumns (normalForm rowsPositive source).height = numActiveColumns source.height :=
  reduction_active_count rowsPositive (normalForm_path rowsPositive source)

theorem extra_columns_identity {rows : Nat} (rowsPositive : 0 < rows) (source : CanonicalState rows) :
    source.width - (normalForm rowsPositive source).width =
      numFrozenColumns source.height - numFrozenRuns source.height := by
  have sourcePartition := numActive_add_numFrozen_eq_n source.height source.canonical.1
    rowsPositive source.widthValid
  have targetPartition := numActive_add_numFrozen_eq_n (normalForm rowsPositive source).height
    (normalForm rowsPositive source).canonical.1 rowsPositive (normalForm rowsPositive source).widthValid
  have activeCount := normalForm_active_count rowsPositive source
  have runCount := normalForm_run_count rowsPositive source
  omega

theorem runStartLift_le_kept_iff {width : Nat} (deleted : Fin (width + 1)) (start column : Fin width) :
    (runStartLift deleted start).val ≤ (keptColumn deleted column).val ↔ start.val ≤ column.val := by
  by_cases startBefore : start.val ≤ deleted.val <;>
    by_cases columnBefore : column.val < deleted.val <;>
      simp [runStartLift, keptColumn, startBefore, columnBefore] <;> omega

theorem runStartLift_surjective_on_starts {rows width : Nat} (height : Cell rows (width + 1) -> Int)
    (heightValid : IsHeight height) (rowsPositive : 0 < rows)
    (deleted : Fin (width + 1)) (deletedFrozen : frozenColumn height deleted)
    (nextFrozen : frozenColumn height ⟨deleted.val + 1, deletedFrozen.2.1⟩)
    (column : Fin (width + 1)) (start : frozenRunStart height column) :
    ∃ original : Fin width, frozenRunStart (contractAt height rowsPositive deleted deletedFrozen) original ∧
      runStartLift deleted original = column := by
  have excluded : column.val ≠ deleted.val + 1 := by
    intro atRemovedSuccessor
    rcases start.2 with first | ⟨positive, previous⟩
    · omega
    · have previousColumn :
          (⟨column.val - 1, by have := column.isLt; omega⟩ : Fin (width + 1)) = deleted := by
        apply Fin.ext
        change column.val - 1 = deleted.val
        omega
      rw [previousColumn] at previous
      exact frozenColumn_not_active height deleted deletedFrozen previous
  have preimage : ∃ original : Fin width, runStartLift deleted original = column := by
    by_cases before : column.val ≤ deleted.val
    · have within : column.val < width := by have := deletedFrozen.2.1; omega
      refine ⟨⟨column.val, within⟩, ?_⟩
      apply Fin.ext
      simp [runStartLift, before]
    · have within : column.val - 1 < width := by have := column.isLt; omega
      have after : ¬ column.val - 1 ≤ deleted.val := by omega
      have restored : column.val - 1 + 1 = column.val := by omega
      refine ⟨⟨column.val - 1, within⟩, ?_⟩
      apply Fin.ext
      simp [runStartLift, after, restored]
  obtain ⟨original, lifted⟩ := preimage
  refine ⟨original, ?_, lifted⟩
  apply (frozenRunStart_contractAt_iff height heightValid rowsPositive deleted deletedFrozen nextFrozen original).mpr
  rw [lifted]
  exact start

noncomputable def runPrefixCount {rows width : Nat} (height : Cell rows width -> Int)
    (column : Fin width) : Nat :=
  (Finset.univ.filter (fun start => frozenRunStart height start ∧ start.val ≤ column.val)).card

theorem runPrefixCount_contractAt {rows width : Nat} (height : Cell rows (width + 1) -> Int)
    (heightValid : IsHeight height) (rowsPositive : 0 < rows)
    (deleted : Fin (width + 1)) (deletedFrozen : frozenColumn height deleted)
    (nextFrozen : frozenColumn height ⟨deleted.val + 1, deletedFrozen.2.1⟩) (column : Fin width) :
    runPrefixCount (contractAt height rowsPositive deleted deletedFrozen) column =
      runPrefixCount height (keptColumn deleted column) := by
  classical
  unfold runPrefixCount
  apply Finset.card_bij (fun start _ => runStartLift deleted start)
  · intro start member
    obtain ⟨_, isStart, before⟩ := Finset.mem_filter.mp member
    refine Finset.mem_filter.mpr ⟨Finset.mem_univ _, ?_, ?_⟩
    · exact (frozenRunStart_contractAt_iff height heightValid rowsPositive deleted deletedFrozen nextFrozen start).mp isStart
    · exact (runStartLift_le_kept_iff deleted start column).mpr before
  · intro first _ second _ equality
    exact runStartLift_injective deleted equality
  · intro start member
    obtain ⟨_, isStart, before⟩ := Finset.mem_filter.mp member
    obtain ⟨original, originalStart, lifted⟩ :=
      runStartLift_surjective_on_starts height heightValid rowsPositive deleted deletedFrozen nextFrozen start isStart
    refine ⟨original, Finset.mem_filter.mpr ⟨Finset.mem_univ _, originalStart, ?_⟩, lifted⟩
    apply (runStartLift_le_kept_iff deleted original column).mp
    rw [lifted]
    exact before

noncomputable def runLengthAtRank {rows width : Nat} (height : Cell rows width -> Int) (rank : Nat) : Nat :=
  (Finset.univ.filter (fun column => ¬ activeColumn height column ∧ runPrefixCount height column = rank)).card

theorem keptColumn_ne_deleted {width : Nat} (deleted : Fin (width + 1)) (column : Fin width) :
    keptColumn deleted column ≠ deleted := by
  intro equality
  have values := congrArg Fin.val equality
  by_cases before : column.val < deleted.val <;>
    simp [keptColumn, before] at values <;> omega

theorem runLengthAtRank_contractAt {rows width : Nat} (height : Cell rows (width + 1) -> Int)
    (heightValid : IsHeight height) (rowsPositive : 0 < rows)
    (deleted : Fin (width + 1)) (deletedFrozen : frozenColumn height deleted)
    (nextFrozen : frozenColumn height ⟨deleted.val + 1, deletedFrozen.2.1⟩) (rank : Nat) :
    runLengthAtRank height rank = runLengthAtRank (contractAt height rowsPositive deleted deletedFrozen) rank +
      if runPrefixCount height deleted = rank then 1 else 0 := by
  classical
  let candidates := Finset.univ.filter (fun column =>
    ¬ activeColumn height column ∧ runPrefixCount height column = rank)
  have remaining :
      runLengthAtRank (contractAt height rowsPositive deleted deletedFrozen) rank =
        (candidates.erase deleted).card := by
    unfold runLengthAtRank
    apply Finset.card_bij (fun column _ => keptColumn deleted column)
    · intro column member
      obtain ⟨_, inactive, atRank⟩ := Finset.mem_filter.mp member
      refine Finset.mem_erase.mpr ⟨keptColumn_ne_deleted deleted column, ?_⟩
      refine Finset.mem_filter.mpr ⟨Finset.mem_univ _, ?_, ?_⟩
      · exact (not_congr (activeColumn_contractAt_iff height heightValid rowsPositive deleted deletedFrozen column)).mp inactive
      · rw [← runPrefixCount_contractAt height heightValid rowsPositive deleted deletedFrozen nextFrozen column]
        exact atRank
    · intro first _ second _ equality
      exact keptColumn_injective deleted equality
    · intro column member
      obtain ⟨notDeleted, candidate⟩ := Finset.mem_erase.mp member
      obtain ⟨_, inactive, atRank⟩ := Finset.mem_filter.mp candidate
      obtain ⟨⟨row, original⟩, lifted⟩ := cellExpand_surjOn deleted deletedFrozen.1 deletedFrozen.2.1
        (⟨0, rowsPositive⟩, column) notDeleted
      have columnEquality : keptColumn deleted original = column := by
        have equality := congrArg Prod.snd lifted
        simpa only [cellExpand_keptColumn] using equality
      refine ⟨original, Finset.mem_filter.mpr ⟨Finset.mem_univ _, ?_, ?_⟩, columnEquality⟩
      · apply (not_congr (activeColumn_contractAt_iff height heightValid rowsPositive deleted deletedFrozen original)).mpr
        rw [columnEquality]
        exact inactive
      · rw [runPrefixCount_contractAt height heightValid rowsPositive deleted deletedFrozen nextFrozen original,
          columnEquality]
        exact atRank
  by_cases deletedRank : runPrefixCount height deleted = rank
  · have member : deleted ∈ candidates := Finset.mem_filter.mpr ⟨Finset.mem_univ _,
      frozenColumn_not_active height deleted deletedFrozen, deletedRank⟩
    rw [remaining, if_pos deletedRank]
    exact (Finset.card_erase_add_one member).symm
  · have notMember : deleted ∉ candidates := by
      intro member
      exact deletedRank (Finset.mem_filter.mp member).2.2
    rw [remaining, if_neg deletedRank, Finset.erase_eq_of_notMem notMember, add_zero]
    rfl

theorem runPrefixCount_mono {rows width : Nat} (height : Cell rows width -> Int)
    {first second : Fin width} (ordered : first.val ≤ second.val) :
    runPrefixCount height first ≤ runPrefixCount height second := by
  classical
  apply Finset.card_le_card
  intro start member
  obtain ⟨_, isStart, before⟩ := Finset.mem_filter.mp member
  exact Finset.mem_filter.mpr ⟨Finset.mem_univ _, isStart, before.trans ordered⟩

theorem runPrefixCount_lt_of_start {rows width : Nat} (height : Cell rows width -> Int)
    {first second : Fin width} (ordered : first.val < second.val)
    (secondStart : frozenRunStart height second) :
    runPrefixCount height first < runPrefixCount height second := by
  classical
  apply Finset.card_lt_card
  constructor
  · intro start member
    obtain ⟨_, isStart, before⟩ := Finset.mem_filter.mp member
    exact Finset.mem_filter.mpr ⟨Finset.mem_univ _, isStart, by omega⟩
  · intro reverse
    have member := reverse (Finset.mem_filter.mpr
      ⟨Finset.mem_univ second, secondStart, Nat.le_refl second.val⟩)
    have before := (Finset.mem_filter.mp member).2.2
    omega

theorem runPrefixCount_injective_on_starts {rows width : Nat} (height : Cell rows width -> Int)
    {first second : Fin width} (firstStart : frozenRunStart height first)
    (secondStart : frozenRunStart height second)
    (sameRank : runPrefixCount height first = runPrefixCount height second) : first = second := by
  apply Fin.ext
  by_contra different
  rcases Nat.lt_or_gt_of_ne different with ordered | ordered
  · have strict := runPrefixCount_lt_of_start height ordered secondStart
    omega
  · have strict := runPrefixCount_lt_of_start height ordered firstStart
    omega

theorem runPrefixCount_pos_of_start {rows width : Nat} (height : Cell rows width -> Int)
    (column : Fin width) (isStart : frozenRunStart height column) :
    0 < runPrefixCount height column := by
  classical
  exact Finset.card_pos.mpr ⟨column,
    Finset.mem_filter.mpr ⟨Finset.mem_univ _, isStart, Nat.le_refl _⟩⟩

theorem runLengthAtRank_pos_of_inactive {rows width : Nat} (height : Cell rows width -> Int)
    (column : Fin width) (inactive : ¬ activeColumn height column) :
    0 < runLengthAtRank height (runPrefixCount height column) := by
  classical
  exact Finset.card_pos.mpr ⟨column,
    Finset.mem_filter.mpr ⟨Finset.mem_univ _, inactive, rfl⟩⟩

theorem reduced_runLengthAtRank_eq_one {rows : Nat} (reduced : ReducedHF rows)
    (rowsPositive : 0 < rows) (column : Fin reduced.W)
    (columnFrozen : frozenColumn reduced.h column) :
    runLengthAtRank reduced.h (runPrefixCount reduced.h column) = 1 := by
  classical
  have singleton : Finset.univ.filter (fun candidate =>
      ¬ activeColumn reduced.h candidate ∧
        runPrefixCount reduced.h candidate = runPrefixCount reduced.h column) = {column} := by
    ext candidate
    simp only [Finset.mem_filter, Finset.mem_univ, true_and, Finset.mem_singleton]
    constructor
    · rintro ⟨inactive, sameRank⟩
      have candidateFrozen : frozenColumn reduced.h candidate := by
        rcases column_active_or_frozen reduced.h reduced.isHeight rowsPositive reduced.hW candidate
          with active | frozen
        · exact False.elim (inactive active)
        · exact frozen
      exact runPrefixCount_injective_on_starts reduced.h
        ((reduced.frozenColumn_iff_frozenRunStart rowsPositive candidate).mp candidateFrozen)
        ((reduced.frozenColumn_iff_frozenRunStart rowsPositive column).mp columnFrozen) sameRank
    · intro equality
      rw [equality]
      exact ⟨frozenColumn_not_active reduced.h column columnFrozen, rfl⟩
  unfold runLengthAtRank
  rw [singleton, Finset.card_singleton]

theorem inactive_has_runStart {rows width : Nat} (height : Cell rows width -> Int)
    (column : Fin width) (inactive : ¬ activeColumn height column) :
    ∃ start : Fin width, frozenRunStart height start ∧ start.val ≤ column.val := by
  have existsFor : ∀ index : Nat, ∀ within : index < width,
      ¬ activeColumn height ⟨index, within⟩ ->
        ∃ start : Fin width, frozenRunStart height start ∧ start.val ≤ index := by
    intro index
    induction index with
    | zero =>
      intro within atZero
      exact ⟨⟨0, within⟩, ⟨atZero, Or.inl rfl⟩, Nat.le_refl _⟩
    | succ index inductionHypothesis =>
      intro within atSuccessor
      have previousWithin : index < width := by omega
      by_cases previousActive : activeColumn height ⟨index, previousWithin⟩
      · refine ⟨⟨index + 1, within⟩,
          ⟨atSuccessor, Or.inr ⟨Nat.succ_pos index, ?_⟩⟩, Nat.le_refl _⟩
        simpa using previousActive
      · obtain ⟨start, isStart, before⟩ := inductionHypothesis previousWithin previousActive
        exact ⟨start, isStart, by omega⟩
  exact existsFor column.val column.isLt inactive

theorem runPrefixCount_pos_of_inactive {rows width : Nat} (height : Cell rows width -> Int)
    (column : Fin width) (inactive : ¬ activeColumn height column) :
    0 < runPrefixCount height column := by
  classical
  obtain ⟨start, isStart, before⟩ := inactive_has_runStart height column inactive
  exact Finset.card_pos.mpr ⟨start,
    Finset.mem_filter.mpr ⟨Finset.mem_univ _, isStart, before⟩⟩

theorem runPrefixCount_le_numFrozenRuns {rows width : Nat} (height : Cell rows width -> Int)
    (column : Fin width) : runPrefixCount height column ≤ numFrozenRuns height := by
  classical
  apply Finset.card_le_card
  intro start member
  exact Finset.mem_filter.mpr ⟨Finset.mem_univ _, (Finset.mem_filter.mp member).2.1⟩

theorem runRank_realized {rows width : Nat} (height : Cell rows width -> Int)
    (rank : Fin (numFrozenRuns height)) :
    ∃ column : Fin width, frozenRunStart height column ∧
      runPrefixCount height column = rank.val + 1 := by
  classical
  let starts := Finset.univ.filter (frozenRunStart height)
  let labels := starts.image (fun column => runPrefixCount height column - 1)
  have contained : labels ⊆ Finset.range (numFrozenRuns height) := by
    intro label member
    obtain ⟨column, startMember, rfl⟩ := Finset.mem_image.mp member
    have isStart := (Finset.mem_filter.mp startMember).2
    have positive := runPrefixCount_pos_of_start height column isStart
    have bound := runPrefixCount_le_numFrozenRuns height column
    exact Finset.mem_range.mpr (by omega)
  have injective : Set.InjOn (fun column => runPrefixCount height column - 1) starts := by
    intro first firstMember second secondMember equality
    change first ∈ Finset.univ.filter (frozenRunStart height) at firstMember
    change second ∈ Finset.univ.filter (frozenRunStart height) at secondMember
    change runPrefixCount height first - 1 = runPrefixCount height second - 1 at equality
    have firstStart := (Finset.mem_filter.mp firstMember).2
    have secondStart := (Finset.mem_filter.mp secondMember).2
    have firstPositive := runPrefixCount_pos_of_start height first firstStart
    have secondPositive := runPrefixCount_pos_of_start height second secondStart
    exact runPrefixCount_injective_on_starts height firstStart secondStart (by omega)
  have labelsCount : labels.card = numFrozenRuns height := by
    calc
      _ = starts.card := Finset.card_image_iff.mpr injective
      _ = _ := rfl
  have allLabels : labels = Finset.range (numFrozenRuns height) :=
    Finset.eq_of_subset_of_card_le contained (by simp only [Finset.card_range, labelsCount, le_refl])
  have member : rank.val ∈ labels := by
    rw [allLabels]
    exact Finset.mem_range.mpr rank.isLt
  obtain ⟨column, startMember, atRank⟩ := Finset.mem_image.mp member
  have isStart := (Finset.mem_filter.mp startMember).2
  have positive := runPrefixCount_pos_of_start height column isStart
  exact ⟨column, isStart, by omega⟩

noncomputable def runLengthTuple {rows width : Nat} (height : Cell rows width -> Int) :
    Fin (numFrozenRuns height) -> Nat := fun rank => runLengthAtRank height (rank.val + 1)

noncomputable def extraLengthTuple {rows width : Nat} (height : Cell rows width -> Int) :
    Fin (numFrozenRuns height) -> Nat := fun rank => runLengthTuple height rank - 1

theorem runLengthTuple_pos {rows width : Nat} (height : Cell rows width -> Int)
    (rank : Fin (numFrozenRuns height)) : 0 < runLengthTuple height rank := by
  obtain ⟨column, isStart, atRank⟩ := runRank_realized height rank
  unfold runLengthTuple
  rw [← atRank]
  exact runLengthAtRank_pos_of_inactive height column isStart.1

theorem reduced_runLengthTuple_eq_one {rows : Nat} (reduced : ReducedHF rows)
    (rowsPositive : 0 < rows) (rank : Fin (numFrozenRuns reduced.h)) :
    runLengthTuple reduced.h rank = 1 := by
  obtain ⟨column, isStart, atRank⟩ := runRank_realized reduced.h rank
  unfold runLengthTuple
  rw [← atRank]
  exact reduced_runLengthAtRank_eq_one reduced rowsPositive column
    ((reduced.frozenColumn_iff_frozenRunStart rowsPositive column).mpr isStart)

theorem sum_runLengthAtRank {rows width : Nat} (height : Cell rows width -> Int) :
    (∑ rank ∈ Finset.range (numFrozenRuns height), runLengthAtRank height (rank + 1)) =
      (Finset.univ.filter (fun column => ¬ activeColumn height column)).card := by
  classical
  have partition : (Finset.univ.filter (fun column => ¬ activeColumn height column)).card =
      ∑ rank ∈ Finset.range (numFrozenRuns height),
        ((Finset.univ.filter (fun column => ¬ activeColumn height column)).filter
          (fun column => runPrefixCount height column - 1 = rank)).card := by
    apply Finset.card_eq_sum_card_fiberwise
    intro column member
    change column ∈ Finset.univ.filter (fun column => ¬ activeColumn height column) at member
    have inactive := (Finset.mem_filter.mp member).2
    have positive := runPrefixCount_pos_of_inactive height column inactive
    have bound := runPrefixCount_le_numFrozenRuns height column
    change runPrefixCount height column - 1 ∈ Finset.range (numFrozenRuns height)
    exact Finset.mem_range.mpr (by omega)
  rw [partition]
  apply Finset.sum_congr rfl
  intro rank _
  unfold runLengthAtRank
  apply congrArg Finset.card
  ext column
  simp only [Finset.mem_filter, Finset.mem_univ, true_and]
  constructor
  · rintro ⟨inactive, atRank⟩
    exact ⟨inactive, by omega⟩
  · rintro ⟨inactive, atRank⟩
    have positive := runPrefixCount_pos_of_inactive height column inactive
    exact ⟨inactive, by omega⟩

theorem sum_runLengthTuple {rows width : Nat} (height : Cell rows width -> Int)
    (heightValid : IsHeight height) (rowsPositive : 0 < rows) (widthValid : 2 ≤ width) :
    (∑ rank : Fin (numFrozenRuns height), runLengthTuple height rank) =
      numFrozenColumns height := by
  classical
  have indexed : (∑ rank : Fin (numFrozenRuns height), runLengthTuple height rank) =
      ∑ rank ∈ Finset.range (numFrozenRuns height), runLengthAtRank height (rank + 1) := by
    apply Finset.sum_bij (fun rank _ => rank.val)
    · intro rank _
      exact Finset.mem_range.mpr rank.isLt
    · intro first _ second _ equality
      exact Fin.ext equality
    · intro rank member
      exact ⟨⟨rank, Finset.mem_range.mp member⟩, Finset.mem_univ _, rfl⟩
    · intro rank _
      rfl
  rw [indexed, sum_runLengthAtRank]
  unfold numFrozenColumns
  apply congrArg Finset.card
  ext column
  simp only [Finset.mem_filter, Finset.mem_univ, true_and]
  constructor
  · intro inactive
    exact (column_active_or_frozen height heightValid rowsPositive widthValid column).resolve_left inactive
  · exact frozenColumn_not_active height column

theorem sum_extraLengthTuple_add_runs {rows width : Nat} (height : Cell rows width -> Int)
    (heightValid : IsHeight height) (rowsPositive : 0 < rows) (widthValid : 2 ≤ width) :
    (∑ rank : Fin (numFrozenRuns height), extraLengthTuple height rank) + numFrozenRuns height =
      numFrozenColumns height := by
  classical
  calc
    _ = (∑ rank : Fin (numFrozenRuns height), extraLengthTuple height rank) +
        ∑ _rank : Fin (numFrozenRuns height), 1 := by simp
    _ = (∑ rank : Fin (numFrozenRuns height), (extraLengthTuple height rank + 1)) :=
      Finset.sum_add_distrib.symm
    _ = (∑ rank : Fin (numFrozenRuns height), runLengthTuple height rank) := by
      apply Finset.sum_congr rfl
      intro rank _
      exact Nat.sub_add_cancel (Nat.succ_le_of_lt (runLengthTuple_pos height rank))
    _ = _ := sum_runLengthTuple height heightValid rowsPositive widthValid

theorem sum_extraLengthTuple_eq_removed {rows : Nat} (rowsPositive : 0 < rows)
    (source : CanonicalState rows) :
    (∑ rank : Fin (numFrozenRuns source.height), extraLengthTuple source.height rank) =
      source.width - (normalForm rowsPositive source).width := by
  have total := sum_extraLengthTuple_add_runs source.height source.canonical.1 rowsPositive source.widthValid
  have removed := extra_columns_identity rowsPositive source
  omega

theorem runPrefixCount_next_eq_of_not_start {rows width : Nat} (height : Cell rows width -> Int)
    (column : Fin width) (nextWithin : column.val + 1 < width)
    (notStart : ¬ frozenRunStart height ⟨column.val + 1, nextWithin⟩) :
    runPrefixCount height ⟨column.val + 1, nextWithin⟩ = runPrefixCount height column := by
  classical
  unfold runPrefixCount
  apply congrArg Finset.card
  ext start
  simp only [Finset.mem_filter, Finset.mem_univ, true_and]
  constructor
  · rintro ⟨isStart, before⟩
    refine ⟨isStart, ?_⟩
    by_contra notBefore
    have same : start = ⟨column.val + 1, nextWithin⟩ := Fin.ext (by
      change start.val = column.val + 1
      change start.val ≤ column.val + 1 at before
      omega)
    exact notStart (same ▸ isStart)
  · rintro ⟨isStart, before⟩
    exact ⟨isStart, before.trans (Nat.le_succ column.val)⟩

theorem adjacent_inactive_prefix_eq {rows width : Nat} (height : Cell rows width -> Int)
    (column : Fin width) (nextWithin : column.val + 1 < width)
    (inactive : ¬ activeColumn height column) :
    runPrefixCount height ⟨column.val + 1, nextWithin⟩ = runPrefixCount height column := by
  apply runPrefixCount_next_eq_of_not_start height column nextWithin
  rintro ⟨_, atZero | ⟨_, previousActive⟩⟩
  · change column.val + 1 = 0 at atZero
    omega
  · apply inactive
    simpa using previousActive

theorem inactive_has_runStart_interval {rows width : Nat} (height : Cell rows width -> Int)
    (column : Fin width) (inactive : ¬ activeColumn height column) :
    ∃ start : Fin width, frozenRunStart height start ∧ start.val ≤ column.val ∧
      ∀ candidate : Fin width, start.val ≤ candidate.val -> candidate.val ≤ column.val ->
        ¬ activeColumn height candidate := by
  have existsFor : ∀ index : Nat, ∀ within : index < width,
      ¬ activeColumn height ⟨index, within⟩ ->
        ∃ start : Fin width, frozenRunStart height start ∧ start.val ≤ index ∧
          ∀ candidate : Fin width, start.val ≤ candidate.val -> candidate.val ≤ index ->
            ¬ activeColumn height candidate := by
    intro index
    induction index with
    | zero =>
      intro within atZero
      refine ⟨⟨0, within⟩, ⟨atZero, Or.inl rfl⟩, Nat.le_refl _, ?_⟩
      intro candidate _ before
      have equality : candidate = ⟨0, within⟩ := Fin.ext (by omega)
      rw [equality]
      exact atZero
    | succ index inductionHypothesis =>
      intro within atSuccessor
      have previousWithin : index < width := by omega
      by_cases previousActive : activeColumn height ⟨index, previousWithin⟩
      · refine ⟨⟨index + 1, within⟩,
          ⟨atSuccessor, Or.inr ⟨Nat.succ_pos index, by simpa using previousActive⟩⟩,
          Nat.le_refl _, ?_⟩
        intro candidate after before
        have equality : candidate = ⟨index + 1, within⟩ := Fin.ext (by
          change candidate.val = index + 1
          change index + 1 ≤ candidate.val at after
          omega)
        rw [equality]
        exact atSuccessor
      · obtain ⟨start, isStart, before, interval⟩ := inductionHypothesis previousWithin previousActive
        refine ⟨start, isStart, by omega, ?_⟩
        intro candidate afterStart beforeSuccessor
        by_cases earlier : candidate.val ≤ index
        · exact interval candidate afterStart earlier
        · have equality : candidate = ⟨index + 1, within⟩ := Fin.ext (by
          change candidate.val = index + 1
          omega)
          rw [equality]
          exact atSuccessor
  exact existsFor column.val column.isLt inactive

theorem same_rank_interval_inactive {rows width : Nat} (height : Cell rows width -> Int)
    {first last : Fin width} (lastInactive : ¬ activeColumn height last)
    (sameRank : runPrefixCount height first = runPrefixCount height last)
    (column : Fin width) (afterFirst : first.val ≤ column.val) (beforeLast : column.val ≤ last.val) :
    ¬ activeColumn height column := by
  obtain ⟨start, isStart, before, interval⟩ := inactive_has_runStart_interval height last lastInactive
  have beforeFirst : start.val ≤ first.val := by
    by_contra notBefore
    have strict := runPrefixCount_lt_of_start height (show first.val < start.val from by omega) isStart
    have bound := runPrefixCount_mono height before
    omega
  exact interval column (beforeFirst.trans afterFirst) beforeLast

theorem expandAfter_adjacent_equal {rows width : Nat} (height : Cell rows width -> Int)
    (mark : Fin width) (nextWithin : mark.val + 1 < width) (slope : Int)
    (rightStep : ∀ row : Fin rows,
      height (row, ⟨mark.val + 1, nextWithin⟩) - height (row, mark) = slope) :
    expandAfter height mark slope = expandAfter height ⟨mark.val + 1, nextWithin⟩ slope := by
  funext point
  by_cases before : point.2.val ≤ mark.val
  · rw [expandAfter_left _ _ _ _ before,
      expandAfter_left _ _ _ _ (show point.2.val ≤ mark.val + 1 from by omega)]
  · by_cases beforeNext : point.2.val ≤ mark.val + 1
    · rw [expandAfter_right _ _ _ _ before, expandAfter_left _ _ _ _ beforeNext]
      have previous : (⟨point.2.val - 1, by have := point.2.isLt; omega⟩ : Fin width) = mark :=
        Fin.ext (by change point.2.val - 1 = mark.val; omega)
      have current : (⟨point.2.val, by have := nextWithin; omega⟩ : Fin width) =
          ⟨mark.val + 1, nextWithin⟩ :=
        Fin.ext (by change point.2.val = mark.val + 1; omega)
      rw [previous, current]
      have step := rightStep point.1
      linarith
    · rw [expandAfter_right _ _ _ _ before, expandAfter_right _ _ _ _ beforeNext]

theorem expandFrozen_adjacent_equal {rows width : Nat} (height : Cell rows width -> Int)
    (heightValid : IsHeight height) (rowsPositive : 0 < rows)
    (mark : Fin width) (markFrozen : frozenColumn height mark)
    (nextFrozen : frozenColumn height ⟨mark.val + 1, markFrozen.2.1⟩) :
    expandAfter height mark (frozenSlope height rowsPositive mark markFrozen) =
      expandAfter height ⟨mark.val + 1, markFrozen.2.1⟩
        (frozenSlope height rowsPositive ⟨mark.val + 1, markFrozen.2.1⟩ nextFrozen) := by
  have sameSlope : frozenSlope height rowsPositive ⟨mark.val + 1, markFrozen.2.1⟩ nextFrozen =
      frozenSlope height rowsPositive mark markFrozen := by
    simpa [frozenSlope] using
      frozenSlope_eq_right height heightValid rowsPositive mark markFrozen ⟨0, rowsPositive⟩
  rw [sameSlope]
  exact expandAfter_adjacent_equal height mark markFrozen.2.1 _
    (frozenSlope_eq_right height heightValid rowsPositive mark markFrozen)

theorem expandFrozen_ordered_equal {rows width : Nat} (height : Cell rows width -> Int)
    (heightValid : IsHeight height) (rowsPositive : 0 < rows)
    (first last : Fin width) (firstFrozen : frozenColumn height first)
    (lastFrozen : frozenColumn height last) (ordered : first.val ≤ last.val)
    (sameRank : runPrefixCount height first = runPrefixCount height last) :
    expandAfter height first (frozenSlope height rowsPositive first firstFrozen) =
      expandAfter height last (frozenSlope height rowsPositive last lastFrozen) := by
  have widthValid : 2 ≤ width := by have := firstFrozen.2.1; omega
  have along : ∀ index : Nat, first.val ≤ index -> index ≤ last.val ->
      ∀ within : index < width, ∀ frozen : frozenColumn height ⟨index, within⟩,
        expandAfter height first (frozenSlope height rowsPositive first firstFrozen) =
          expandAfter height ⟨index, within⟩ (frozenSlope height rowsPositive ⟨index, within⟩ frozen) := by
    intro index lower
    induction index, lower using Nat.le_induction with
    | base =>
      intro _ within frozen
      rfl
    | succ index lower inductionHypothesis =>
      intro upper within frozen
      have previousWithin : index < width := by omega
      have previousInactive : ¬ activeColumn height ⟨index, previousWithin⟩ :=
        same_rank_interval_inactive height (frozenColumn_not_active height last lastFrozen)
          sameRank ⟨index, previousWithin⟩ lower (Nat.le_of_succ_le upper)
      have previousFrozen : frozenColumn height ⟨index, previousWithin⟩ :=
        (column_active_or_frozen height heightValid rowsPositive widthValid _).resolve_left previousInactive
      exact (inductionHypothesis (by omega) previousWithin previousFrozen).trans
        (expandFrozen_adjacent_equal height heightValid rowsPositive
          ⟨index, previousWithin⟩ previousFrozen frozen)
  exact along last.val ordered (Nat.le_refl _) last.isLt lastFrozen

theorem expandFrozen_same_rank {rows width : Nat} (height : Cell rows width -> Int)
    (heightValid : IsHeight height) (rowsPositive : 0 < rows)
    (first last : Fin width) (firstFrozen : frozenColumn height first)
    (lastFrozen : frozenColumn height last)
    (sameRank : runPrefixCount height first = runPrefixCount height last) :
    expandAfter height first (frozenSlope height rowsPositive first firstFrozen) =
      expandAfter height last (frozenSlope height rowsPositive last lastFrozen) := by
  rcases Nat.le_total first.val last.val with ordered | reversed
  · exact expandFrozen_ordered_equal height heightValid rowsPositive first last firstFrozen lastFrozen
      ordered sameRank
  · exact (expandFrozen_ordered_equal height heightValid rowsPositive last first lastFrozen firstFrozen
      reversed sameRank.symm).symm

theorem contracted_marker_slope {rows width : Nat} (height : Cell rows (width + 1) -> Int)
    (heightValid : IsHeight height) (rowsPositive : 0 < rows)
    (deleted : Fin (width + 1)) (deletedFrozen : frozenColumn height deleted)
    (nextFrozen : frozenColumn height ⟨deleted.val + 1, deletedFrozen.2.1⟩) :
    frozenSlope (contractAt height rowsPositive deleted deletedFrozen) rowsPositive
        ⟨deleted.val, by have := deletedFrozen.2.1; omega⟩
        (contracted_marker_frozen height heightValid rowsPositive deleted deletedFrozen nextFrozen) =
      frozenSlope height rowsPositive deleted deletedFrozen := by
  change contractAt height rowsPositive deleted deletedFrozen
        (⟨0, rowsPositive⟩, ⟨deleted.val, by have := deletedFrozen.2.1; omega⟩) -
      contractAt height rowsPositive deleted deletedFrozen
        (⟨0, rowsPositive⟩, ⟨deleted.val - 1, by have := deletedFrozen.2.1; omega⟩) = _
  rw [contractAt_right _ _ _ _ _ (show ¬ deleted.val < deleted.val from Nat.lt_irrefl _),
    contractAt_left _ _ _ _ _ (show deleted.val - 1 < deleted.val from by
      have := deletedFrozen.1
      omega)]
  have left := frozenSlope_eq height heightValid rowsPositive deleted deletedFrozen ⟨0, rowsPositive⟩
  have right := frozenSlope_eq_right height heightValid rowsPositive deleted deletedFrozen ⟨0, rowsPositive⟩
  linarith

theorem runPrefixCount_contracted_marker {rows width : Nat} (height : Cell rows (width + 1) -> Int)
    (heightValid : IsHeight height) (rowsPositive : 0 < rows)
    (deleted : Fin (width + 1)) (deletedFrozen : frozenColumn height deleted)
    (nextFrozen : frozenColumn height ⟨deleted.val + 1, deletedFrozen.2.1⟩) :
    runPrefixCount (contractAt height rowsPositive deleted deletedFrozen)
        ⟨deleted.val, by have := deletedFrozen.2.1; omega⟩ = runPrefixCount height deleted := by
  have kept : keptColumn deleted ⟨deleted.val, by have := deletedFrozen.2.1; omega⟩ =
      ⟨deleted.val + 1, deletedFrozen.2.1⟩ := by
    apply Fin.ext
    simp [keptColumn]
  rw [runPrefixCount_contractAt height heightValid rowsPositive deleted deletedFrozen nextFrozen, kept]
  exact adjacent_inactive_prefix_eq height deleted deletedFrozen.2.1
    (frozenColumn_not_active height deleted deletedFrozen)

theorem expand_contract_at_rank {rows width : Nat} (height : Cell rows (width + 1) -> Int)
    (heightValid : IsHeight height) (rowsPositive : 0 < rows)
    (deleted : Fin (width + 1)) (deletedFrozen : frozenColumn height deleted)
    (nextFrozen : frozenColumn height ⟨deleted.val + 1, deletedFrozen.2.1⟩)
    (mark : Fin width) (markFrozen : frozenColumn (contractAt height rowsPositive deleted deletedFrozen) mark)
    (sameRank : runPrefixCount (contractAt height rowsPositive deleted deletedFrozen) mark =
      runPrefixCount height deleted) :
    expandAfter (contractAt height rowsPositive deleted deletedFrozen) mark
        (frozenSlope (contractAt height rowsPositive deleted deletedFrozen) rowsPositive mark markFrozen) =
      height := by
  have atSeam := contracted_marker_frozen height heightValid rowsPositive deleted deletedFrozen nextFrozen
  have sameExpansion := expandFrozen_same_rank (contractAt height rowsPositive deleted deletedFrozen)
    (contractAt_isHeight height heightValid rowsPositive deleted deletedFrozen) rowsPositive mark
    ⟨deleted.val, by have := deletedFrozen.2.1; omega⟩ markFrozen atSeam
    (sameRank.trans (runPrefixCount_contracted_marker height heightValid rowsPositive deleted deletedFrozen nextFrozen).symm)
  rw [contracted_marker_slope height heightValid rowsPositive deleted deletedFrozen nextFrozen] at sameExpansion
  exact sameExpansion.trans (expand_contract_at height heightValid rowsPositive deleted deletedFrozen)

theorem adjacent_frozen_of_inactive_not_start {rows width : Nat} (height : Cell rows width -> Int)
    (heightValid : IsHeight height) (rowsPositive : 0 < rows) (widthValid : 2 ≤ width)
    (column : Fin width) (inactive : ¬ activeColumn height column)
    (notStart : ¬ frozenRunStart height column) :
    ∃ previous : Fin width, ∃ previousFrozen : frozenColumn height previous,
      frozenColumn height ⟨previous.val + 1, previousFrozen.2.1⟩ ∧
        runPrefixCount height previous = runPrefixCount height column := by
  have positive : 0 < column.val := by
    by_cases atZero : column.val = 0
    · exact False.elim (notStart ⟨inactive, Or.inl atZero⟩)
    · omega
  let previous : Fin width := ⟨column.val - 1, by have := column.isLt; omega⟩
  have previousInactive : ¬ activeColumn height previous := by
    intro previousActive
    exact notStart ⟨inactive, Or.inr ⟨positive, previousActive⟩⟩
  have previousFrozen : frozenColumn height previous :=
    (column_active_or_frozen height heightValid rowsPositive widthValid previous).resolve_left previousInactive
  have currentFrozen : frozenColumn height column :=
    (column_active_or_frozen height heightValid rowsPositive widthValid column).resolve_left inactive
  have nextSame : (⟨previous.val + 1, previousFrozen.2.1⟩ : Fin width) = column := by
    apply Fin.ext
    change column.val - 1 + 1 = column.val
    omega
  refine ⟨previous, previousFrozen, ?_, ?_⟩
  · rw [nextSame]
    exact currentFrozen
  · have relation := adjacent_inactive_prefix_eq height previous previousFrozen.2.1 previousInactive
    rw [nextSame] at relation
    exact relation.symm

theorem adjacent_frozen_of_runLength {rows width : Nat} (height : Cell rows width -> Int)
    (heightValid : IsHeight height) (rowsPositive : 0 < rows) (widthValid : 2 ≤ width)
    (rank : Nat) (longRun : 1 < runLengthAtRank height rank) :
    ∃ column : Fin width, ∃ frozen : frozenColumn height column,
      frozenColumn height ⟨column.val + 1, frozen.2.1⟩ ∧ runPrefixCount height column = rank := by
  classical
  by_contra noPair
  have allStarts : ∀ column : Fin width, ¬ activeColumn height column ->
      runPrefixCount height column = rank -> frozenRunStart height column := by
    intro column inactive atRank
    by_contra notStart
    obtain ⟨previous, previousFrozen, nextFrozen, sameRank⟩ :=
      adjacent_frozen_of_inactive_not_start height heightValid rowsPositive widthValid column inactive notStart
    exact noPair ⟨previous, previousFrozen, nextFrozen, sameRank.trans atRank⟩
  have bound : runLengthAtRank height rank ≤ 1 := by
    unfold runLengthAtRank
    apply Finset.card_le_one.mpr
    intro first firstMember second secondMember
    obtain ⟨_, firstInactive, firstRank⟩ := Finset.mem_filter.mp firstMember
    obtain ⟨_, secondInactive, secondRank⟩ := Finset.mem_filter.mp secondMember
    exact runPrefixCount_injective_on_starts height
      (allStarts first firstInactive firstRank) (allStarts second secondInactive secondRank)
      (firstRank.trans secondRank.symm)
  omega

theorem reduced_runLengthAtRank_le_one {rows : Nat} (reduced : ReducedHF rows)
    (rowsPositive : 0 < rows) (rank : Nat) : runLengthAtRank reduced.h rank ≤ 1 := by
  classical
  unfold runLengthAtRank
  apply Finset.card_le_one.mpr
  intro first firstMember second secondMember
  obtain ⟨_, firstInactive, firstRank⟩ := Finset.mem_filter.mp firstMember
  obtain ⟨_, secondInactive, secondRank⟩ := Finset.mem_filter.mp secondMember
  have firstFrozen :=
    (column_active_or_frozen reduced.h reduced.isHeight rowsPositive reduced.hW first).resolve_left firstInactive
  have secondFrozen :=
    (column_active_or_frozen reduced.h reduced.isHeight rowsPositive reduced.hW second).resolve_left secondInactive
  exact runPrefixCount_injective_on_starts reduced.h
    ((reduced.frozenColumn_iff_frozenRunStart rowsPositive first).mp firstFrozen)
    ((reduced.frozenColumn_iff_frozenRunStart rowsPositive second).mp secondFrozen)
    (firstRank.trans secondRank.symm)

theorem adjacent_frozen_runLength_gt_one {rows width : Nat} (height : Cell rows width -> Int)
    (column : Fin width) (columnFrozen : frozenColumn height column)
    (nextFrozen : frozenColumn height ⟨column.val + 1, columnFrozen.2.1⟩) :
    1 < runLengthAtRank height (runPrefixCount height column) := by
  classical
  unfold runLengthAtRank
  apply Finset.one_lt_card.mpr
  refine ⟨column, Finset.mem_filter.mpr ⟨Finset.mem_univ _,
    frozenColumn_not_active height column columnFrozen, rfl⟩,
    ⟨column.val + 1, columnFrozen.2.1⟩, Finset.mem_filter.mpr ⟨Finset.mem_univ _,
      frozenColumn_not_active height _ nextFrozen,
      adjacent_inactive_prefix_eq height column columnFrozen.2.1
        (frozenColumn_not_active height column columnFrozen)⟩, ?_⟩
  intro equality
  have values := congrArg Fin.val equality
  change column.val = column.val + 1 at values
  omega

#print axioms runStartLift_injective
#print axioms activity_runStartLift_iff
#print axioms runStartLift_value
#print axioms runStartLift_zero_iff
#print axioms runStartLift_positive_iff
#print axioms predecessor_activity_iff
#print axioms frozenRunStart_contractAt_iff
#print axioms contractAt_numFrozenRuns
#print axioms step_run_count
#print axioms reduction_run_count
#print axioms normalForm_run_count
#print axioms keptColumn_injective
#print axioms contractAt_numActiveColumns
#print axioms step_active_count
#print axioms reduction_active_count
#print axioms normalForm_active_count
#print axioms extra_columns_identity
#print axioms runStartLift_le_kept_iff
#print axioms runStartLift_surjective_on_starts
#print axioms runPrefixCount_contractAt
#print axioms keptColumn_ne_deleted
#print axioms runLengthAtRank_contractAt
#print axioms runPrefixCount_mono
#print axioms runPrefixCount_lt_of_start
#print axioms runPrefixCount_injective_on_starts
#print axioms runPrefixCount_pos_of_start
#print axioms runLengthAtRank_pos_of_inactive
#print axioms reduced_runLengthAtRank_eq_one
#print axioms inactive_has_runStart
#print axioms runPrefixCount_pos_of_inactive
#print axioms runPrefixCount_le_numFrozenRuns
#print axioms runRank_realized
#print axioms runLengthTuple_pos
#print axioms reduced_runLengthTuple_eq_one
#print axioms sum_runLengthAtRank
#print axioms sum_runLengthTuple
#print axioms sum_extraLengthTuple_add_runs
#print axioms sum_extraLengthTuple_eq_removed
#print axioms runPrefixCount_next_eq_of_not_start
#print axioms adjacent_inactive_prefix_eq
#print axioms inactive_has_runStart_interval
#print axioms same_rank_interval_inactive
#print axioms expandAfter_adjacent_equal
#print axioms expandFrozen_adjacent_equal
#print axioms expandFrozen_ordered_equal
#print axioms expandFrozen_same_rank
#print axioms contracted_marker_slope
#print axioms runPrefixCount_contracted_marker
#print axioms expand_contract_at_rank
#print axioms adjacent_frozen_of_inactive_not_start
#print axioms adjacent_frozen_of_runLength
#print axioms reduced_runLengthAtRank_le_one
#print axioms adjacent_frozen_runLength_gt_one

end OrigamiCone.Sequel.RunLengthFibers
