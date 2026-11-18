import OrigamiCone.SequelEd
import OrigamiCone.SequelBinom

namespace OrigamiCone.Sequel.ConcreteWalk

open OrigamiCone

def turnSign {junctions : Nat} (turns : Fin junctions -> Bool) : Nat -> Int
  | 0 => 1
  | index + 1 =>
    if within : index < junctions then
      if turns ⟨index, within⟩ then -turnSign turns index else turnSign turns index
    else turnSign turns index

theorem turnSign_unit {junctions : Nat} (turns : Fin junctions -> Bool) (index : Nat) :
    turnSign turns index = 1 ∨ turnSign turns index = -1 := by
  induction index with
  | zero => simp [turnSign]
  | succ index inductionHypothesis =>
    rw [turnSign]
    split_ifs <;> omega

theorem turnSign_step {junctions : Nat} (turns : Fin junctions -> Bool)
    (junction : Fin junctions) :
    turnSign turns (junction.val + 1) =
      if turns junction then -turnSign turns junction.val else turnSign turns junction.val := by
  rw [turnSign, dif_pos junction.isLt]

theorem turnSign_changes {junctions : Nat} (turns : Fin junctions -> Bool)
    (junction : Fin junctions) :
    turnSign turns (junction.val + 1) ≠ turnSign turns junction.val ↔ turns junction = true := by
  rw [turnSign_step]
  have unit := turnSign_unit turns junction.val
  cases selected : turns junction <;> rcases unit with direction | direction <;>
    simp [direction]

def walkAt {junctions : Nat} (turns : Fin junctions -> Bool) : Nat -> Int
  | 0 => 0
  | index + 1 => walkAt turns index + turnSign turns index

theorem walkAt_succ {junctions : Nat} (turns : Fin junctions -> Bool) (index : Nat) :
    walkAt turns (index + 1) = walkAt turns index + turnSign turns index := rfl

def turnHeight {junctions : Nat} (rows : Nat) (turns : Fin junctions -> Bool) :
    Cell rows (junctions + 2) -> Int :=
  fun cell => (cell.1.val : Int) + walkAt turns cell.2.val

theorem adjacent_coordinate_cases {rows width : Nat} (source target : Cell rows width)
    (adjacent : adj source target) :
    (source.1.val = target.1.val ∧ source.2.val + 1 = target.2.val) ∨
    (source.1.val = target.1.val ∧ target.2.val + 1 = source.2.val) ∨
    (source.2.val = target.2.val ∧ source.1.val + 1 = target.1.val) ∨
    (source.2.val = target.2.val ∧ target.1.val + 1 = source.1.val) := by
  simp only [adj, gdist] at adjacent
  omega

theorem turnHeight_isHeight {junctions : Nat} (rows : Nat) (turns : Fin junctions -> Bool) :
    IsHeight (turnHeight rows turns) := by
  rintro ⟨sourceRow, sourceColumn⟩ ⟨targetRow, targetColumn⟩ adjacent
  have cases := adjacent_coordinate_cases (sourceRow, sourceColumn) (targetRow, targetColumn) adjacent
  dsimp only [Prod.fst, Prod.snd] at cases
  change |(sourceRow.val : Int) + walkAt turns sourceColumn.val -
    ((targetRow.val : Int) + walkAt turns targetColumn.val)| = 1
  rcases cases with ⟨sameRow, nextColumn⟩ | ⟨sameRow, previousColumn⟩ |
    ⟨sameColumn, nextRow⟩ | ⟨sameColumn, previousRow⟩
  · have step : walkAt turns targetColumn.val =
        walkAt turns sourceColumn.val + turnSign turns sourceColumn.val := by
      rw [← nextColumn, walkAt_succ]
    have difference : (sourceRow.val : Int) + walkAt turns sourceColumn.val -
        ((targetRow.val : Int) + walkAt turns targetColumn.val) = -turnSign turns sourceColumn.val := by
      rw [sameRow, step]
      ring
    rw [difference]
    rcases turnSign_unit turns sourceColumn.val with direction | direction <;> simp [direction]
  · have step : walkAt turns sourceColumn.val =
        walkAt turns targetColumn.val + turnSign turns targetColumn.val := by
      rw [← previousColumn, walkAt_succ]
    have difference : (sourceRow.val : Int) + walkAt turns sourceColumn.val -
        ((targetRow.val : Int) + walkAt turns targetColumn.val) = turnSign turns targetColumn.val := by
      rw [sameRow, step]
      ring
    rw [difference]
    rcases turnSign_unit turns targetColumn.val with direction | direction <;> simp [direction]
  · have sameWalk := congrArg (walkAt turns) sameColumn
    have difference : (sourceRow.val : Int) + walkAt turns sourceColumn.val -
        ((targetRow.val : Int) + walkAt turns targetColumn.val) = -1 := by omega
    rw [difference]
    norm_num
  · have sameWalk := congrArg (walkAt turns) sameColumn
    have difference : (sourceRow.val : Int) + walkAt turns sourceColumn.val -
        ((targetRow.val : Int) + walkAt turns targetColumn.val) = 1 := by omega
    rw [difference]
    norm_num

theorem turnHeight_canonical {junctions : Nat} (rows : Nat) (turns : Fin junctions -> Bool) :
    IsCanonicalHeight (turnHeight rows turns) := by
  refine ⟨turnHeight_isHeight rows turns, ?_⟩
  intro cell rowZero columnZero
  simp [turnHeight, rowZero, columnZero, walkAt]

theorem turnHeight_injective {junctions rows : Nat} (rowsPositive : 0 < rows) :
    Function.Injective (turnHeight rows : (Fin junctions -> Bool) -> Cell rows (junctions + 2) -> Int) := by
  intro turns other sameHeight
  have sameWalk : ∀ column : Fin (junctions + 2),
      walkAt turns column.val = walkAt other column.val := by
    intro column
    have equality := congrFun sameHeight (⟨0, rowsPositive⟩, column)
    simpa only [turnHeight, Nat.cast_zero, zero_add] using equality
  have sameSign : ∀ index : Nat, index < junctions + 1 ->
      turnSign turns index = turnSign other index := by
    intro index within
    have source : walkAt turns index = walkAt other index := sameWalk ⟨index, by omega⟩
    have target : walkAt turns (index + 1) = walkAt other (index + 1) :=
      sameWalk ⟨index + 1, by omega⟩
    simp only [walkAt_succ] at target
    omega
  funext junction
  have sameChoice : turns junction = true ↔ other junction = true := by
    rw [← turnSign_changes turns junction, ← turnSign_changes other junction,
      sameSign junction.val (by have := junction.isLt; omega),
      sameSign (junction.val + 1) (by have := junction.isLt; omega)]
  cases selected : turns junction <;> cases otherSelected : other junction <;> simp_all

def walkMinimum {junctions : Nat} (turns : Fin junctions -> Bool)
    (column : Fin (junctions + 2)) : Prop :=
  (0 < column.val -> turnSign turns (column.val - 1) = -1) ∧
    (column.val + 1 < junctions + 2 -> turnSign turns column.val = 1)

def walkMaximum {junctions : Nat} (turns : Fin junctions -> Bool)
    (column : Fin (junctions + 2)) : Prop :=
  (0 < column.val -> turnSign turns (column.val - 1) = 1) ∧
    (column.val + 1 < junctions + 2 -> turnSign turns column.val = -1)

theorem turnHeight_minimum {junctions rows : Nat} (turns : Fin junctions -> Bool)
    (row : Fin rows) (column : Fin (junctions + 2)) :
    IsStrictLocalMin (turnHeight rows turns) (row, column) ↔
      row.val = 0 ∧ walkMinimum turns column := by
  constructor
  · intro minimum
    have topRow : row.val = 0 := by
      by_contra notTop
      have neighborBound : row.val - 1 < rows := by omega
      have equality := minimum (⟨row.val - 1, neighborBound⟩, column) (by
        simp only [adj, gdist]
        omega)
      change ((row.val - 1 : Nat) : Int) + walkAt turns column.val =
        (row.val : Int) + walkAt turns column.val + 1 at equality
      omega
    refine ⟨topRow, ?_, ?_⟩
    · intro leftPresent
      have neighborBound : column.val - 1 < junctions + 2 := by omega
      have equality := minimum (row, ⟨column.val - 1, neighborBound⟩) (by
        simp only [adj, gdist]
        omega)
      change (row.val : Int) + walkAt turns (column.val - 1) =
        (row.val : Int) + walkAt turns column.val + 1 at equality
      have step := walkAt_succ turns (column.val - 1)
      have previous : column.val - 1 + 1 = column.val := by omega
      rw [previous] at step
      omega
    · intro rightPresent
      have equality := minimum (row, ⟨column.val + 1, rightPresent⟩) (by
        simp only [adj, gdist]
        omega)
      change (row.val : Int) + walkAt turns (column.val + 1) =
        (row.val : Int) + walkAt turns column.val + 1 at equality
      rw [walkAt_succ] at equality
      omega
  · rintro ⟨topRow, leftCondition, rightCondition⟩ ⟨neighborRow, neighborColumn⟩ adjacent
    change (neighborRow.val : Int) + walkAt turns neighborColumn.val =
      (row.val : Int) + walkAt turns column.val + 1
    have cases := adjacent_coordinate_cases (row, column) (neighborRow, neighborColumn) adjacent
    dsimp only [Prod.fst, Prod.snd] at cases
    rcases cases with
      ⟨sameRow, nextColumn⟩ | ⟨sameRow, previousColumn⟩ | ⟨sameColumn, nextRow⟩ | ⟨sameColumn, previousRow⟩
    · have direction := rightCondition (by have := neighborColumn.isLt; omega)
      have step := walkAt_succ turns column.val
      rw [nextColumn] at step
      omega
    · have direction := leftCondition (by omega)
      have previous : column.val - 1 = neighborColumn.val := by omega
      rw [previous] at direction
      have step := walkAt_succ turns neighborColumn.val
      rw [previousColumn] at step
      omega
    · have sameWalk := congrArg (walkAt turns) sameColumn
      omega
    · omega

theorem turnHeight_maximum {junctions rows : Nat} (turns : Fin junctions -> Bool)
    (row : Fin rows) (column : Fin (junctions + 2)) :
    IsStrictLocalMax (turnHeight rows turns) (row, column) ↔
      row.val + 1 = rows ∧ walkMaximum turns column := by
  constructor
  · intro maximum
    have bottomRow : row.val + 1 = rows := by
      by_contra notBottom
      have neighborBound : row.val + 1 < rows := by omega
      have equality := maximum (⟨row.val + 1, neighborBound⟩, column) (by
        simp only [adj, gdist]
        omega)
      change ((row.val + 1 : Nat) : Int) + walkAt turns column.val =
        (row.val : Int) + walkAt turns column.val - 1 at equality
      omega
    refine ⟨bottomRow, ?_, ?_⟩
    · intro leftPresent
      have neighborBound : column.val - 1 < junctions + 2 := by omega
      have equality := maximum (row, ⟨column.val - 1, neighborBound⟩) (by
        simp only [adj, gdist]
        omega)
      change (row.val : Int) + walkAt turns (column.val - 1) =
        (row.val : Int) + walkAt turns column.val - 1 at equality
      have step := walkAt_succ turns (column.val - 1)
      have previous : column.val - 1 + 1 = column.val := by omega
      rw [previous] at step
      omega
    · intro rightPresent
      have equality := maximum (row, ⟨column.val + 1, rightPresent⟩) (by
        simp only [adj, gdist]
        omega)
      change (row.val : Int) + walkAt turns (column.val + 1) =
        (row.val : Int) + walkAt turns column.val - 1 at equality
      rw [walkAt_succ] at equality
      omega
  · rintro ⟨bottomRow, leftCondition, rightCondition⟩ ⟨neighborRow, neighborColumn⟩ adjacent
    change (neighborRow.val : Int) + walkAt turns neighborColumn.val =
      (row.val : Int) + walkAt turns column.val - 1
    have cases := adjacent_coordinate_cases (row, column) (neighborRow, neighborColumn) adjacent
    dsimp only [Prod.fst, Prod.snd] at cases
    rcases cases with
      ⟨sameRow, nextColumn⟩ | ⟨sameRow, previousColumn⟩ | ⟨sameColumn, nextRow⟩ | ⟨sameColumn, previousRow⟩
    · have direction := rightCondition (by have := neighborColumn.isLt; omega)
      have step := walkAt_succ turns column.val
      rw [nextColumn] at step
      omega
    · have direction := leftCondition (by omega)
      have previous : column.val - 1 = neighborColumn.val := by omega
      rw [previous] at direction
      have step := walkAt_succ turns neighborColumn.val
      rw [previousColumn] at step
      omega
    · have := neighborRow.isLt
      omega
    · have sameWalk := congrArg (walkAt turns) sameColumn
      omega

theorem walk_min_max_disjoint {junctions : Nat} (turns : Fin junctions -> Bool)
    (column : Fin (junctions + 2)) (minimum : walkMinimum turns column)
    (maximum : walkMaximum turns column) : False := by
  by_cases first : column.val = 0
  · have rightPresent : column.val + 1 < junctions + 2 := by omega
    have up := minimum.2 rightPresent
    have down := maximum.2 rightPresent
    omega
  · have leftPresent : 0 < column.val := by omega
    have down := minimum.1 leftPresent
    have up := maximum.1 leftPresent
    omega

def walkExtremum {junctions : Nat} (turns : Fin junctions -> Bool)
    (column : Fin (junctions + 2)) : Prop :=
  walkMinimum turns column ∨ walkMaximum turns column

instance {junctions : Nat} (turns : Fin junctions -> Bool) (column : Fin (junctions + 2)) :
    Decidable (walkExtremum turns column) := by
  unfold walkExtremum walkMinimum walkMaximum
  infer_instance

theorem walkExtremum_first {junctions : Nat} (turns : Fin junctions -> Bool) :
    walkExtremum turns (0 : Fin (junctions + 2)) := by
  simp [walkExtremum, walkMinimum, turnSign]

theorem walkExtremum_last {junctions : Nat} (turns : Fin junctions -> Bool) :
    walkExtremum turns (Fin.last (junctions + 1)) := by
  rcases turnSign_unit turns junctions with direction | direction <;>
    simp [walkExtremum, walkMinimum, walkMaximum, direction]

theorem walkExtremum_middle {junctions : Nat} (turns : Fin junctions -> Bool)
    (junction : Fin junctions) :
    walkExtremum turns ⟨junction.val + 1, by have := junction.isLt; omega⟩ ↔
      turns junction = true := by
  rw [← turnSign_changes turns junction]
  have leftPresent : 0 < junction.val + 1 := by omega
  have rightPresent : junction.val + 1 + 1 < junctions + 2 := by have := junction.isLt; omega
  have previous : junction.val + 1 - 1 = junction.val := by omega
  simp only [walkExtremum, walkMinimum, walkMaximum, previous,
    leftPresent, rightPresent, true_implies]
  rcases turnSign_unit turns junction.val with before | before <;>
    rcases turnSign_unit turns (junction.val + 1) with after | after <;> simp [before, after]

theorem walkExtremum_count {junctions : Nat} (turns : Fin junctions -> Bool) :
    (Finset.univ.filter (walkExtremum turns)).card =
      (Finset.univ.filter (fun junction => turns junction = true)).card + 2 := by
  classical
  have middle : ∀ junction : Fin junctions,
      walkExtremum turns junction.castSucc.succ ↔ turns junction = true :=
    fun junction => walkExtremum_middle turns junction
  have last : walkExtremum turns (Fin.last junctions).succ := walkExtremum_last turns
  simp only [Finset.card_eq_sum_ones, Finset.sum_filter]
  rw [Fin.sum_univ_succ, Fin.sum_univ_castSucc]
  simp only [walkExtremum_first, last, middle, if_true]
  omega

theorem turnHeight_numExtrema {junctions rows : Nat} (rowsPositive : 0 < rows)
    (turns : Fin junctions -> Bool) :
    numExtrema (turnHeight rows turns) =
      (Finset.univ.filter (fun junction => turns junction = true)).card + 2 := by
  classical
  have gridToPath :
      (Finset.univ.filter (IsStrictLocalExtremum (turnHeight rows turns))).card =
        (Finset.univ.filter (walkExtremum turns)).card := by
    apply Finset.card_bij (fun cell _ => cell.2)
    · intro cell member
      have extremum := (Finset.mem_filter.mp member).2
      refine Finset.mem_filter.mpr ⟨Finset.mem_univ _, ?_⟩
      rcases extremum with maximum | minimum
      · exact Or.inr ((turnHeight_maximum turns cell.1 cell.2).mp maximum).2
      · exact Or.inl ((turnHeight_minimum turns cell.1 cell.2).mp minimum).2
    · intro source sourceMember target targetMember sameColumn
      have sourceExtremum := (Finset.mem_filter.mp sourceMember).2
      have targetExtremum := (Finset.mem_filter.mp targetMember).2
      rcases sourceExtremum with sourceMaximum | sourceMinimum
      · have sourceData := (turnHeight_maximum turns source.1 source.2).mp sourceMaximum
        rcases targetExtremum with targetMaximum | targetMinimum
        · have targetData := (turnHeight_maximum turns target.1 target.2).mp targetMaximum
          exact Prod.ext (Fin.ext (by omega)) sameColumn
        · have targetData := (turnHeight_minimum turns target.1 target.2).mp targetMinimum
          have minimum : walkMinimum turns source.2 := by rw [sameColumn]; exact targetData.2
          exact False.elim (walk_min_max_disjoint turns source.2 minimum sourceData.2)
      · have sourceData := (turnHeight_minimum turns source.1 source.2).mp sourceMinimum
        rcases targetExtremum with targetMaximum | targetMinimum
        · have targetData := (turnHeight_maximum turns target.1 target.2).mp targetMaximum
          have maximum : walkMaximum turns source.2 := by rw [sameColumn]; exact targetData.2
          exact False.elim (walk_min_max_disjoint turns source.2 sourceData.2 maximum)
        · have targetData := (turnHeight_minimum turns target.1 target.2).mp targetMinimum
          exact Prod.ext (Fin.ext (by omega)) sameColumn
    · intro column member
      rcases (Finset.mem_filter.mp member).2 with minimum | maximum
      · refine ⟨(⟨0, rowsPositive⟩, column), Finset.mem_filter.mpr ⟨Finset.mem_univ _, ?_⟩, rfl⟩
        exact Or.inr ((turnHeight_minimum turns _ _).mpr ⟨rfl, minimum⟩)
      · have bottomBound : rows - 1 < rows := by omega
        refine ⟨(⟨rows - 1, bottomBound⟩, column), Finset.mem_filter.mpr ⟨Finset.mem_univ _, ?_⟩, rfl⟩
        exact Or.inl ((turnHeight_maximum turns _ _).mpr ⟨by
          change rows - 1 + 1 = rows
          omega, maximum⟩)
  exact gridToPath.trans (walkExtremum_count turns)

theorem binomial_le_Ed {junctions rows : Nat} (rowsPositive : 0 < rows) (turnCount : Nat) :
    junctions.choose turnCount ≤ Ed (turnCount + 2) rows (junctions + 2) := by
  classical
  let heights := (turnPatterns junctions turnCount).image (turnHeight rows)
  have lands : (heights : Set (Cell rows (junctions + 2) -> Int)) ⊆
      {height | IsCanonicalHeight height ∧ numExtrema height = turnCount + 2} := by
    intro height member
    change height ∈ (turnPatterns junctions turnCount).image (turnHeight rows) at member
    obtain ⟨turns, patternMember, rfl⟩ := Finset.mem_image.mp member
    have turnCard : (Finset.univ.filter (fun junction => turns junction = true)).card = turnCount := by
      simpa only [turnPatterns, Finset.mem_filter, Finset.mem_univ, true_and] using patternMember
    refine ⟨turnHeight_canonical rows turns, ?_⟩
    rw [turnHeight_numExtrema rowsPositive turns, turnCard]
  calc
    junctions.choose turnCount = (turnPatterns junctions turnCount).card :=
      (card_turnPatterns junctions turnCount).symm
    _ = heights.card :=
      (Finset.card_image_of_injective _ (turnHeight_injective rowsPositive)).symm
    _ = (heights : Set (Cell rows (junctions + 2) -> Int)).ncard := by simp
    _ ≤ {height : Cell rows (junctions + 2) -> Int |
        IsCanonicalHeight height ∧ numExtrema height = turnCount + 2}.ncard :=
      Set.ncard_le_ncard lands (Ed_fibre_finite (turnCount + 2) (by omega) (by omega))
    _ = Ed (turnCount + 2) rows (junctions + 2) := rfl

theorem shifted_binomial_le_Ed {rows width degree : Nat} (rowsPositive : 0 < rows)
    (widthValid : 2 ≤ width) (degreeValid : 2 ≤ degree) :
    (width - 2).choose (degree - 2) ≤ Ed degree rows width := by
  have bound := binomial_le_Ed (junctions := width - 2) rowsPositive (degree - 2)
  simpa only [Nat.sub_add_cancel widthValid, Nat.sub_add_cancel degreeValid] using bound

#print axioms turnSign_unit
#print axioms turnSign_step
#print axioms turnSign_changes
#print axioms walkAt_succ
#print axioms turnHeight_isHeight
#print axioms turnHeight_canonical
#print axioms turnHeight_injective
#print axioms adjacent_coordinate_cases
#print axioms turnHeight_minimum
#print axioms turnHeight_maximum
#print axioms walk_min_max_disjoint
#print axioms walkExtremum_first
#print axioms walkExtremum_last
#print axioms walkExtremum_middle
#print axioms walkExtremum_count
#print axioms turnHeight_numExtrema
#print axioms binomial_le_Ed
#print axioms shifted_binomial_le_Ed

end OrigamiCone.Sequel.ConcreteWalk
