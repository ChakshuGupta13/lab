import MiuraCounting.ConcreteSeparableRunBound

namespace OrigamiCone.Sequel.RemainderPolynomials

open OrigamiCone FrozenExpansion JointReduction JointFibers JointTypePartition SplitPolynomials SignedHeightCounts SingleSideFamilies SeparableProducts SeparableRunBound

theorem remainder_core_totalDegree (core : GridState) (degree : Nat)
    (degreeBound : 5 ≤ degree) (counts : numExtrema core.columns.height = degree)
    (notSingleSide : ¬ IsSingleSide core.columns.height) :
    (corePolynomial core).totalDegree ≤ degree - 3 := by
  by_cases positive : 0 < numFrozenRuns (swapH core.columns.height) ∧ 0 < numFrozenRuns core.columns.height
  · have runs := nonsingleSide_run_bound core.columns.height core.columns.canonical
      core.rowsValid core.columns.widthValid (by omega) notSingleSide
    rw [counts] at runs
    have bound := corePolynomial_totalDegree core
    omega
  · rw [corePolynomial, if_neg positive, MvPolynomial.totalDegree_zero]
    exact Nat.zero_le _

theorem remainder_familyPolynomial_totalDegree (minima maxima : Nat) (degreeBound : 5 ≤ minima + maxima) :
    (familyPolynomial minima maxima (fun source => ¬ IsSingleSide source.columns.height)).totalDegree ≤
      minima + maxima - 3 := by
  unfold familyPolynomial
  apply MvPolynomial.totalDegree_finsetSum_le
  intro core _
  exact remainder_core_totalDegree core.val.val (minima + maxima) degreeBound core.val.property.2 core.property.2

theorem canonicalRemainder_eventually_total_polynomial (minima maxima : Nat)
    (degreeBound : 5 ≤ minima + maxima) :
    ∃ polynomial : MvPolynomial (Fin 2) Rat,
      polynomial.totalDegree ≤ minima + maxima - 3 ∧
      ∀ rows width : Nat, 2 * (minima + maxima) - 1 ≤ rows ->
        2 * (minima + maxima) - 1 ≤ width ->
          evalAt polynomial rows width =
            (Nat.card (CanonicalRemainderSplitHeights rows width minima maxima) : Rat) := by
  refine ⟨familyPolynomial minima maxima (fun source => ¬ IsSingleSide source.columns.height),
    remainder_familyPolynomial_totalDegree minima maxima degreeBound, ?_⟩
  intro rows width rowBound columnBound
  have counts := Nat.card_congr (canonicalRemainderEquiv rows width minima maxima (by omega) (by omega))
  rw [counts]
  apply familyPolynomial_eval minima maxima rows width (by omega) rowBound columnBound
  intro core source
  exact not_congr (normalFiber_singleSide_iff core source)

theorem polynomial_eq_on_nat_tail (first second : MvPolynomial (Fin 2) Rat) (threshold : Nat)
    (agreement : ∀ rows width : Nat, threshold ≤ rows -> threshold ≤ width ->
      evalAt first rows width = evalAt second rows width) : first = second := by
  let tail : Set Rat := Set.range (fun size : Nat => ((size + threshold : Nat) : Rat))
  have infinite : tail.Infinite := by
    apply Set.infinite_range_of_injective
    intro firstSize secondSize equality
    change ((firstSize + threshold : Nat) : Rat) = ((secondSize + threshold : Nat) : Rat) at equality
    have same : firstSize + threshold = secondSize + threshold := by exact_mod_cast equality
    omega
  apply MvPolynomial.funext_set (fun _ : Fin 2 => tail) (fun _ => infinite)
  intro assignment member
  obtain ⟨rows, rowValue⟩ := member (0 : Fin 2) (Set.mem_univ _)
  obtain ⟨width, columnValue⟩ := member (1 : Fin 2) (Set.mem_univ _)
  have assignmentEquality : assignment = (fun coordinate : Fin 2 =>
      if coordinate = 0 then ((rows + threshold : Nat) : Rat) else ((width + threshold : Nat) : Rat)) := by
    funext coordinate
    fin_cases coordinate
    · simpa using rowValue.symm
    · simpa using columnValue.symm
  rw [assignmentEquality]
  exact agreement (rows + threshold) (width + threshold) (by omega) (by omega)

theorem canonicalRemainder_polynomial_totalDegree (minima maxima : Nat) (degreeBound : 5 ≤ minima + maxima)
    (polynomial : MvPolynomial (Fin 2) Rat)
    (agreement : ∀ rows width : Nat, 2 * (minima + maxima) - 1 ≤ rows ->
      2 * (minima + maxima) - 1 ≤ width ->
        evalAt polynomial rows width =
          (Nat.card (CanonicalRemainderSplitHeights rows width minima maxima) : Rat)) :
    polynomial.totalDegree ≤ minima + maxima - 3 := by
  obtain ⟨witness, degree, witnessAgreement⟩ := canonicalRemainder_eventually_total_polynomial minima maxima degreeBound
  have same := polynomial_eq_on_nat_tail polynomial witness (2 * (minima + maxima) - 1) (by
    intro rows width rowBound columnBound
    exact (agreement rows width rowBound columnBound).trans (witnessAgreement rows width rowBound columnBound).symm)
  rw [same]
  exact degree

theorem canonicalSplit_and_remainder_eventually_polynomial (minima maxima : Nat) (degreeBound : 3 ≤ minima + maxima) :
    ∃ full remainder : MvPolynomial (Fin 2) Rat,
      (∀ coordinate, full.degreeOf coordinate ≤ minima + maxima - 2) ∧
      (∀ coordinate, remainder.degreeOf coordinate ≤ minima + maxima - 2) ∧
      ∀ rows width : Nat, 2 * (minima + maxima) - 1 ≤ rows ->
        2 * (minima + maxima) - 1 ≤ width ->
          evalAt full rows width = (Nat.card (CanonicalSplitHeights rows width minima maxima) : Rat) ∧
            evalAt remainder rows width = (Nat.card (CanonicalRemainderSplitHeights rows width minima maxima) : Rat) := by
  obtain ⟨full, fullDegree, fullAgreement⟩ := canonicalSplit_eventually_polynomial minima maxima degreeBound
  obtain ⟨remainder, remainderDegree, remainderAgreement⟩ := canonicalRemainder_eventually_polynomial minima maxima degreeBound
  exact ⟨full, remainder, fullDegree, remainderDegree, fun rows width rowBound columnBound =>
    ⟨fullAgreement rows width rowBound columnBound, remainderAgreement rows width rowBound columnBound⟩⟩

#print axioms remainder_core_totalDegree
#print axioms remainder_familyPolynomial_totalDegree
#print axioms canonicalRemainder_eventually_total_polynomial
#print axioms polynomial_eq_on_nat_tail
#print axioms canonicalRemainder_polynomial_totalDegree
#print axioms canonicalSplit_and_remainder_eventually_polynomial

end OrigamiCone.Sequel.RemainderPolynomials
