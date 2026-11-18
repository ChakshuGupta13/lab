import OrigamiCone.SequelEdCellExpand
import OrigamiCone.SequelEdContractExtremum

namespace OrigamiCone.Sequel

open OrigamiCone

def reinsertAt {m n : Nat} (contracted : Cell m (n - 1) -> Int)
    (column : Fin n) (positive : 0 < column.val) (interior : column.val + 1 < n)
    (slope : Int) : Cell m n -> Int := fun point =>
  if left : point.2.val < column.val then
    contracted (point.1, ⟨point.2.val, by omega⟩)
  else if atRemoved : point.2.val = column.val then
    contracted (point.1, ⟨column.val - 1, by omega⟩) + slope
  else
    contracted (point.1, ⟨point.2.val - 1, by have := point.2.isLt; omega⟩) + slope

theorem reinsertAt_contractAt {m n : Nat} (height : Cell m n -> Int)
    (heightValid : IsHeight height) (rowsPositive : 0 < m)
    (column : Fin n) (columnFrozen : frozenColumn height column) :
    reinsertAt (contractAt height rowsPositive column columnFrozen)
      column columnFrozen.1 columnFrozen.2.1
      (frozenSlope height rowsPositive column columnFrozen) = height := by
  funext point
  unfold reinsertAt
  split_ifs with left atRemoved
  · rw [contractAt_left height rowsPositive column columnFrozen _ left]
  · rw [contractAt_left height rowsPositive column columnFrozen _ (by
      change column.val - 1 < column.val
      have := columnFrozen.1
      omega)]
    have sameColumn : point.2 = column := Fin.ext atRemoved
    have slopeIdentity := frozenSlope_eq height heightValid rowsPositive column columnFrozen point.1
    change height (point.1, ⟨column.val - 1, by have := column.isLt; omega⟩) +
      frozenSlope height rowsPositive column columnFrozen = height point
    have samePoint : point = (point.1, column) := Prod.ext rfl sameColumn
    have pointValue := congrArg height samePoint
    linarith
  · have right : ¬ point.2.val - 1 < column.val := by omega
    rw [contractAt_right height rowsPositive column columnFrozen _ right]
    have restoredColumn : (⟨point.2.val - 1 + 1, by have := point.2.isLt; omega⟩ : Fin n) = point.2 := by
      apply Fin.ext
      have := columnFrozen.1
      change point.2.val - 1 + 1 = point.2.val
      omega
    rw [restoredColumn]
    ring

example : reinsertAt (m := 2) (n := 3)
    (fun point => (point.1.val + point.2.val : Nat))
    ⟨1, by decide⟩ (by decide) (by decide) 0
    (⟨0, by decide⟩, ⟨1, by decide⟩) ≠ (1 : Int) := by
  decide

inductive ContractionPlan : Nat -> Nat -> Type where
  | keep (width : Nat) : ContractionPlan width width
  | delete {sourceWidth finalWidth : Nat}
      (column : Fin sourceWidth) (positive : 0 < column.val)
      (interior : column.val + 1 < sourceWidth) (slope : Int)
      (remaining : ContractionPlan (sourceWidth - 1) finalWidth) :
      ContractionPlan sourceWidth finalWidth

def reconstructPlan {rows sourceWidth finalWidth : Nat}
    (plan : ContractionPlan sourceWidth finalWidth)
    (finalHeight : Cell rows finalWidth -> Int) : Cell rows sourceWidth -> Int :=
  match plan with
  | .keep _ => finalHeight
  | .delete column positive interior slope remaining =>
      reinsertAt (reconstructPlan remaining finalHeight) column positive interior slope

def planTracks {rows sourceWidth finalWidth : Nat} (rowsPositive : 0 < rows)
    (plan : ContractionPlan sourceWidth finalWidth)
    (source : Cell rows sourceWidth -> Int)
    (target : Cell rows finalWidth -> Int) : Prop :=
  match plan with
  | .keep _ => source = target
  | .delete column _ interior slope remaining =>
      ∃ frozen : frozenColumn source column,
        frozenColumn source ⟨column.val + 1, interior⟩ ∧
        slope = frozenSlope source rowsPositive column frozen ∧
        planTracks rowsPositive remaining
          (contractAt source rowsPositive column frozen) target

theorem reconstructPlan_of_tracks {rows sourceWidth finalWidth : Nat}
    (rowsPositive : 0 < rows) (plan : ContractionPlan sourceWidth finalWidth)
    (source : Cell rows sourceWidth -> Int)
    (target : Cell rows finalWidth -> Int) (sourceValid : IsHeight source)
    (tracks : planTracks rowsPositive plan source target) :
    reconstructPlan plan target = source := by
  induction plan with
  | keep => exact tracks.symm
  | delete column positive interior slope remaining inductionHypothesis =>
      obtain ⟨frozen, nextFrozen, sameSlope, remainingTracks⟩ := tracks
      have contractedValid := contractAt_isHeight source sourceValid rowsPositive column frozen
      have recovered := inductionHypothesis
        (contractAt source rowsPositive column frozen) target contractedValid remainingTracks
      change reinsertAt (reconstructPlan remaining target) column positive interior slope = source
      rw [recovered, sameSlope]
      exact reinsertAt_contractAt source sourceValid rowsPositive column frozen

theorem exists_reduced_with_plan {rows : Nat} (rowsPositive : 0 < rows) :
    ∀ (width : Nat) (_ : 2 <= width) (source : Cell rows width -> Int),
      IsCanonicalHeight source ->
      ∃ (finalWidth : Nat) (target : Cell rows finalWidth -> Int),
        IsCanonicalHeight target ∧
        (∀ column : Fin finalWidth, ∀ adjacent : column.val + 1 < finalWidth,
          ¬ (frozenColumn target column ∧
            frozenColumn target ⟨column.val + 1, adjacent⟩)) ∧
        numExtrema target = numExtrema source ∧
        finalWidth <= width ∧ 2 <= finalWidth ∧
        ∃ plan : ContractionPlan width finalWidth,
          planTracks rowsPositive plan source target := by
  intro width
  induction width using Nat.strong_induction_on with
  | _ width inductionHypothesis =>
      intro widthValid source sourceCanonical
      by_cases adjacentFrozen : ∃ column : Fin width, ∃ adjacent : column.val + 1 < width,
          frozenColumn source column ∧ frozenColumn source ⟨column.val + 1, adjacent⟩
      · obtain ⟨column, adjacent, columnFrozen, nextFrozen⟩ := adjacentFrozen
        have widthAtLeastFour : 4 <= width := by
          have columnPositive : 0 < column.val := columnFrozen.1
          have nextInterior : column.val + 1 + 1 < width := nextFrozen.2.1
          omega
        have contractedCanonical := contractAt_isCanonicalHeight source sourceCanonical.1
          rowsPositive column columnFrozen sourceCanonical
        obtain ⟨finalWidth, target, targetCanonical, targetReduced,
          sameExtrema, finalUpper, finalLower, remaining, remainingTracks⟩ :=
          inductionHypothesis (width - 1) (by omega) (by omega)
            (contractAt source rowsPositive column columnFrozen) contractedCanonical
        refine ⟨finalWidth, target, targetCanonical, targetReduced, ?_, by omega,
          finalLower, .delete column columnFrozen.1 columnFrozen.2.1
            (frozenSlope source rowsPositive column columnFrozen) remaining, ?_⟩
        · exact sameExtrema.trans (contractAt_numExtrema_eq source sourceCanonical.1
            rowsPositive column columnFrozen)
        · exact ⟨columnFrozen, nextFrozen, rfl, remainingTracks⟩
      · refine ⟨width, source, sourceCanonical, ?_, rfl, le_refl _, widthValid,
          .keep width, rfl⟩
        intro column adjacent frozen
        exact adjacentFrozen ⟨column, adjacent, frozen⟩

theorem planTracks_reduced_identity {rows sourceWidth finalWidth : Nat}
    (rowsPositive : 0 < rows) (plan : ContractionPlan sourceWidth finalWidth)
    (source : Cell rows sourceWidth -> Int)
    (target : Cell rows finalWidth -> Int)
    (sourceReduced : ∀ column : Fin sourceWidth,
      ∀ adjacent : column.val + 1 < sourceWidth,
        ¬ (frozenColumn source column ∧
          frozenColumn source ⟨column.val + 1, adjacent⟩))
    (tracks : planTracks rowsPositive plan source target) :
    (⟨finalWidth, target⟩ : Sigma fun width : Nat => Cell rows width -> Int) =
      ⟨sourceWidth, source⟩ := by
  cases plan with
  | keep =>
      exact congrArg
        (fun height =>
          (⟨sourceWidth, height⟩ : Sigma fun width : Nat => Cell rows width -> Int))
        tracks.symm
  | delete column positive interior slope remaining =>
      obtain ⟨frozen, nextFrozen, sameSlope, remainingTracks⟩ := tracks
      exact False.elim (sourceReduced column interior ⟨frozen, nextFrozen⟩)

#print axioms reinsertAt_contractAt
#print axioms reconstructPlan_of_tracks
#print axioms exists_reduced_with_plan
#print axioms planTracks_reduced_identity

end OrigamiCone.Sequel
