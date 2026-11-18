import MiuraCounting.ConcreteSideDecomposition

namespace OrigamiCone.Sequel.DegreeRemainder

open OrigamiCone FrozenExpansion JointReduction JointFibers JointTypePartition SplitPolynomials SingleSideFamilies SideDecomposition RemainderPolynomials

noncomputable section

abbrev CountedRemainderHeights (rows width degree : Nat) :=
  {height : CountedCanonicalHeights rows width degree // ¬ IsSingleSide height.val}

def remainderStatesEquiv (rows width degree : Nat) (rowsValid : 2 ≤ rows) (widthValid : 2 ≤ width) :
    CountedRemainderHeights rows width degree ≃
      FilteredStates rows width degree (fun source => ¬ IsSingleSide source.columns.height) where
  toFun height := ⟨⟨⟨rows, rowsValid, width, widthValid, height.val.val, height.val.property.1⟩,
    rfl, rfl, height.val.property.2⟩, height.property⟩
  invFun := by
    rintro ⟨⟨⟨stateRows, stateRowsValid, ⟨stateWidth, stateWidthValid, height, canonical⟩⟩,
      rfl, rfl, counts⟩, notSingleSide⟩
    exact ⟨⟨height, canonical, counts⟩, notSingleSide⟩
  left_inv := by
    rintro ⟨⟨height, canonical, counts⟩, notSingleSide⟩
    rfl
  right_inv := by
    rintro ⟨⟨⟨stateRows, stateRowsValid, ⟨stateWidth, stateWidthValid, height, canonical⟩⟩,
      rfl, rfl, counts⟩, notSingleSide⟩
    rfl

def canonicalSideRemainderEquiv (rows width degree : Nat) :
    CountedSingleSideHeights rows width degree ⊕ CountedRemainderHeights rows width degree ≃
      CountedCanonicalHeights rows width degree := by
  classical
  exact Equiv.sumCompl (fun height : CountedCanonicalHeights rows width degree => IsSingleSide height.val)

theorem Ed_partition (rows width degree : Nat) (rowsValid : 2 ≤ rows) (widthValid : 2 ≤ width) :
    Ed degree rows width = Nat.card (CountedSingleSideHeights rows width degree) +
      Nat.card (CountedRemainderHeights rows width degree) := by
  letI : Finite (CountedCanonicalHeights rows width degree) :=
    countedCanonical_finite rows width degree rowsValid widthValid
  rw [Ed_eq_natCard]
  exact (Nat.card_congr (canonicalSideRemainderEquiv rows width degree)).symm.trans Nat.card_sum

theorem Ed_singleSide_remainder (rows width degree : Nat) (rowsValid : 2 ≤ rows) (widthValid : 2 ≤ width)
    (degreeLarge : 3 ≤ degree) :
    Ed degree rows width = 4 * (rows - 2).choose (degree - 2) +
      4 * (width - 2).choose (degree - 2) + Nat.card (CountedRemainderHeights rows width degree) := by
  rw [Ed_partition rows width degree rowsValid widthValid,
    singleSide_card rows width degree rowsValid widthValid degreeLarge]

theorem remainder_card_eq_sum (rows width degree : Nat) (degreeBound : 3 ≤ degree)
    (rowBound : 2 * degree - 1 ≤ rows) (columnBound : 2 * degree - 1 ≤ width) :
    Nat.card (CountedRemainderHeights rows width degree) =
      ∑ core : FilteredTypes degree (fun source => ¬ IsSingleSide source.columns.height),
        coreWeight core.val.val rows width := by
  rw [Nat.card_congr (remainderStatesEquiv rows width degree (by omega) (by omega))]
  apply filteredStates_stable_card rows width degree degreeBound rowBound columnBound
  intro core source
  exact not_congr (normalFiber_singleSide_iff core source)

def remainderPolynomial (degree : Nat) : MvPolynomial (Fin 2) Rat :=
  ∑ core : FilteredTypes degree (fun source => ¬ IsSingleSide source.columns.height), corePolynomial core.val.val

theorem remainderPolynomial_eval (rows width degree : Nat) (degreeBound : 3 ≤ degree)
    (rowBound : 2 * degree - 1 ≤ rows) (columnBound : 2 * degree - 1 ≤ width) :
    evalAt (remainderPolynomial degree) rows width = (Nat.card (CountedRemainderHeights rows width degree) : Rat) := by
  classical
  rw [remainder_card_eq_sum rows width degree degreeBound rowBound columnBound]
  simp only [remainderPolynomial, evalAt, map_sum, Nat.cast_sum]
  apply Finset.sum_congr rfl
  intro core _
  have coreBounds := reduced_bounds core.val.val core.val.property.1 degree core.val.property.2
  exact corePolynomial_eval core.val.val rows width (coreBounds.1.trans rowBound)
    (coreBounds.2.1.trans columnBound)

theorem remainderPolynomial_totalDegree (degree : Nat) (degreeBound : 5 ≤ degree) :
    (remainderPolynomial degree).totalDegree ≤ degree - 3 := by
  unfold remainderPolynomial
  apply MvPolynomial.totalDegree_finsetSum_le
  intro core _
  exact remainder_core_totalDegree core.val.val degree degreeBound core.val.property.2 core.property

#print axioms remainderStatesEquiv
#print axioms canonicalSideRemainderEquiv
#print axioms Ed_partition
#print axioms Ed_singleSide_remainder
#print axioms remainder_card_eq_sum
#print axioms remainderPolynomial
#print axioms remainderPolynomial_eval
#print axioms remainderPolynomial_totalDegree

end

end OrigamiCone.Sequel.DegreeRemainder
