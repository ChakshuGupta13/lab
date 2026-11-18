import OrigamiCone.ColReflectTransport
import MiuraCounting.ConcreteGridRestrictions

namespace OrigamiCone.Sequel.GridReflection

open OrigamiCone

def reflectH {rows width : Nat} (height : Cell rows width -> Int) : Cell rows width -> Int :=
  fun point => height (colRefl point)

theorem reflectH_involutive {rows width : Nat} (height : Cell rows width -> Int) :
    reflectH (reflectH height) = height := by
  funext point
  exact congrArg height (colRefl_involutive point)

theorem reflectH_isHeight {rows width : Nat} (height : Cell rows width -> Int)
    (heightValid : IsHeight height) : IsHeight (reflectH height) := by
  intro first second adjacent
  exact heightValid (colRefl first) (colRefl second) ((colRefl_adj first second).mpr adjacent)

theorem reflectH_maximum_iff {rows width : Nat} (height : Cell rows width -> Int) (point : Cell rows width) :
    IsStrictLocalMax (reflectH height) point ↔ IsStrictLocalMax height (colRefl point) := by
  constructor
  · intro maximum neighbor adjacent
    have reflectedAdjacent : adj point (colRefl neighbor) := by
      apply (colRefl_adj point (colRefl neighbor)).mp
      simpa only [colRefl_involutive] using adjacent
    have value := maximum (colRefl neighbor) reflectedAdjacent
    simpa only [reflectH, colRefl_involutive] using value
  · intro maximum neighbor adjacent
    exact maximum (colRefl neighbor) ((colRefl_adj point neighbor).mpr adjacent)

theorem reflectH_minimum_iff {rows width : Nat} (height : Cell rows width -> Int) (point : Cell rows width) :
    IsStrictLocalMin (reflectH height) point ↔ IsStrictLocalMin height (colRefl point) := by
  constructor
  · intro minimum neighbor adjacent
    have reflectedAdjacent : adj point (colRefl neighbor) := by
      apply (colRefl_adj point (colRefl neighbor)).mp
      simpa only [colRefl_involutive] using adjacent
    have value := minimum (colRefl neighbor) reflectedAdjacent
    simpa only [reflectH, colRefl_involutive] using value
  · intro minimum neighbor adjacent
    exact minimum (colRefl neighbor) ((colRefl_adj point neighbor).mpr adjacent)

theorem reflectH_extremum_iff {rows width : Nat} (height : Cell rows width -> Int) (point : Cell rows width) :
    IsStrictLocalExtremum (reflectH height) point ↔ IsStrictLocalExtremum height (colRefl point) := by
  exact or_congr (reflectH_maximum_iff height point) (reflectH_minimum_iff height point)

theorem reflectH_row_profile {rows width : Nat} (height : Cell rows width -> Int)
    (row : Fin rows) (apex : Fin width) (sign : Int)
    (profile : ∀ column : Fin width, height (row, column) = height (row, apex) + sign * |(column.val : Int) - apex.val|) :
    ∀ column : Fin width, reflectH height (row, column) = reflectH height (row, Fin.rev apex) +
      sign * |(column.val : Int) - (Fin.rev apex).val| := by
  intro column
  have distance : |((Fin.rev column).val : Int) - apex.val| =
      |(column.val : Int) - (Fin.rev apex).val| := by
    have negative : ((Fin.rev column).val : Int) - apex.val =
        -((column.val : Int) - (Fin.rev apex).val) := by
      have first := Fin.val_rev column
      have second := Fin.val_rev apex
      have firstWithin := column.isLt
      have secondWithin := apex.isLt
      omega
    rw [negative, abs_neg]
  change height (row, Fin.rev column) = height (row, Fin.rev (Fin.rev apex)) +
    sign * |(column.val : Int) - (Fin.rev apex).val|
  rw [Fin.rev_rev, ← distance]
  exact profile (Fin.rev column)

theorem reflectH_transverse {rows width : Nat} (height : Cell rows width -> Int) (row : Fin rows) (sign : Int)
    (transverse : ∀ column : Fin width, ∀ other : Fin rows, ((row.val : Int) - other.val).natAbs = 1 ->
      height (other, column) = height (row, column) + sign) :
    ∀ column : Fin width, ∀ other : Fin rows, ((row.val : Int) - other.val).natAbs = 1 ->
      reflectH height (other, column) = reflectH height (row, column) + sign :=
  fun column other adjacent => transverse (Fin.rev column) other adjacent

theorem reflect_right_boundary_extremum {rows width : Nat} (height : Cell rows width -> Int)
    (widthPositive : 0 < width) (row : Fin rows)
    (extremum : IsStrictLocalExtremum (reflectH height) (row, ⟨width - 1, by omega⟩)) :
    IsStrictLocalExtremum height (row, ⟨0, widthPositive⟩) := by
  have original := (reflectH_extremum_iff height (row, ⟨width - 1, by omega⟩)).mp extremum
  have reversed : Fin.rev (⟨width - 1, by omega⟩ : Fin width) = ⟨0, widthPositive⟩ := by
    apply Fin.ext
    rw [Fin.val_rev]
    change width - ((width - 1) + 1) = 0
    omega
  change IsStrictLocalExtremum height (row, Fin.rev (⟨width - 1, by omega⟩ : Fin width)) at original
  rwa [reversed] at original

theorem left_extremum_above_cut {rows width : Nat} (height : Cell rows width -> Int)
    (heightValid : IsHeight height) (widthValid : 2 ≤ width) (cut : Fin rows) (cutPositive : 0 < cut.val)
    (apex : Fin width) (apexPositive : 0 < apex.val) (sign : Int) (unitSign : sign = 1 ∨ sign = -1)
    (profile : ∀ column : Fin width, height (cut, column) = height (cut, apex) + sign * |(column.val : Int) - apex.val|)
    (transverse : ∀ column : Fin width, ∀ other : Fin rows, ((cut.val : Int) - other.val).natAbs = 1 ->
      height (other, column) = height (cut, column) + sign) :
    ∃ row : Fin rows, row.val < cut.val ∧ IsStrictLocalExtremum height (row, ⟨0, by omega⟩) := by
  have beforeRight : (Fin.rev apex).val < width - 1 := by
    rw [Fin.val_rev]
    have := apex.isLt
    omega
  obtain ⟨row, before, extremum⟩ := GridRestrictions.right_extremum_above_cut (reflectH height)
    (reflectH_isHeight height heightValid) widthValid cut cutPositive (Fin.rev apex) beforeRight sign unitSign
    (reflectH_row_profile height cut apex sign profile) (reflectH_transverse height cut sign transverse)
  exact ⟨row, before, reflect_right_boundary_extremum height (by omega) row extremum⟩

theorem left_extremum_below_cut {rows width : Nat} (height : Cell rows width -> Int)
    (heightValid : IsHeight height) (widthValid : 2 ≤ width) (cut : Fin rows) (belowCut : cut.val + 1 < rows)
    (apex : Fin width) (apexPositive : 0 < apex.val) (sign : Int) (unitSign : sign = 1 ∨ sign = -1)
    (profile : ∀ column : Fin width, height (cut, column) = height (cut, apex) + sign * |(column.val : Int) - apex.val|)
    (transverse : ∀ column : Fin width, ∀ other : Fin rows, ((cut.val : Int) - other.val).natAbs = 1 ->
      height (other, column) = height (cut, column) + sign) :
    ∃ row : Fin rows, cut.val < row.val ∧ IsStrictLocalExtremum height (row, ⟨0, by omega⟩) := by
  have beforeRight : (Fin.rev apex).val < width - 1 := by
    rw [Fin.val_rev]
    have := apex.isLt
    omega
  obtain ⟨row, after, extremum⟩ := GridRestrictions.right_extremum_below_cut (reflectH height)
    (reflectH_isHeight height heightValid) widthValid cut belowCut (Fin.rev apex) beforeRight sign unitSign
    (reflectH_row_profile height cut apex sign profile) (reflectH_transverse height cut sign transverse)
  exact ⟨row, after, reflect_right_boundary_extremum height (by omega) row extremum⟩

#print axioms reflectH_involutive
#print axioms reflectH_isHeight
#print axioms reflectH_maximum_iff
#print axioms reflectH_minimum_iff
#print axioms reflectH_extremum_iff
#print axioms reflectH_row_profile
#print axioms reflectH_transverse
#print axioms reflect_right_boundary_extremum
#print axioms left_extremum_above_cut
#print axioms left_extremum_below_cut

end OrigamiCone.Sequel.GridReflection
