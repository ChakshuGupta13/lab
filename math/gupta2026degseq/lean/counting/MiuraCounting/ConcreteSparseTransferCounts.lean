import MiuraCounting.ConcreteSmallCountChecks

namespace OrigamiCone.Sequel.SparseTransferCounts

open SmallCountChecks

def extensionCount (rows : Nat) : Nat -> Nat -> PathColouring rows -> PathColouring rows -> Nat
  | 0, degree, left, middle => if boundaryCount middle left = degree then 1 else 0
  | width + 1, degree, left, middle =>
      ∑ right : ProperColumn rows,
        if PathAdjacent middle right.val then
          if columnExtremaCount left middle right.val ≤ degree then
            extensionCount rows width (degree - columnExtremaCount left middle right.val) middle right.val
          else 0
        else 0

theorem sum_columns_eq_proper {rows : Nat} (weight : PathColouring rows -> Int)
    (zeroImproper : ∀ column, ¬ IsPathProperColouring column -> weight column = 0) :
    (∑ column : PathColouring rows, weight column) = ∑ column : ProperColumn rows, weight column.val := by
  classical
  calc
    (∑ column : PathColouring rows, weight column) =
        ∑ column ∈ Finset.univ.filter IsPathProperColouring, weight column := by
      rw [Finset.sum_filter]
      apply Finset.sum_congr rfl
      intro column _
      by_cases proper : IsPathProperColouring column
      · simp only [if_pos proper]
      · rw [if_neg proper, zeroImproper column proper]
    _ = ∑ column : ProperColumn rows, weight column.val :=
      Finset.sum_subtype _ (by intro column; simp) weight

theorem phi_coeff_eq_extensionCount (rows width degree : Nat) (source : TransferState rows) :
    (phi rows width source).coeff degree = (extensionCount rows width degree source.val.1 source.val.2 : Int) := by
  classical
  induction width generalizing degree source with
  | zero =>
    simp only [phi_zero, rightBdyVec, extensionCount, boundaryCount, Polynomial.coeff_monomial,
      Nat.cast_ite, Nat.cast_one, Nat.cast_zero]
  | succ width inductionHypothesis =>
    let weight (right : PathColouring rows) : Int :=
      if columnExtremaCount source.val.1 source.val.2 right ≤ degree then
        (extensionCount rows width (degree - columnExtremaCount source.val.1 source.val.2 right)
          source.val.2 right : Int)
      else 0
    rw [phi_succ, Polynomial.finset_sum_coeff]
    calc
      (∑ target : TransferState rows,
          (transferMatrix rows source target * phi rows width target).coeff degree) =
          ∑ target : TransferState rows, if source.val.2 = target.val.1 then weight target.val.2 else 0 := by
        apply Finset.sum_congr rfl
        intro target _
        by_cases matching : source.val.2 = target.val.1
        · rw [transferMatrix, if_pos matching, Polynomial.coeff_X_pow_mul', inductionHypothesis,
            if_pos matching]
          simp only [weight, ← matching]
        · simp only [transferMatrix, if_neg matching, zero_mul, Polynomial.coeff_zero]
      _ = ∑ right : PathColouring rows, if PathAdjacent source.val.2 right then weight right else 0 :=
        sum_matching_targets source weight
      _ = ∑ right : ProperColumn rows, if PathAdjacent source.val.2 right.val then weight right.val else 0 := by
        apply sum_columns_eq_proper
        intro column improper
        have notAdjacent : ¬ PathAdjacent source.val.2 column := fun near => improper near.2.1
        simp only [if_neg notAdjacent]
      _ = (extensionCount rows (width + 1) degree source.val.1 source.val.2 : Int) := by
        simp only [extensionCount, Nat.cast_sum, Nat.cast_ite, Nat.cast_zero]
        rfl

def transferCount (rows width degree : Nat) : Nat :=
  ∑ pair : ProperPairState rows,
    let source := (transferStateEquivProper rows).symm pair
    if IsNormalizedLeft source then
      if boundaryCount source.val.1 source.val.2 ≤ degree then
        extensionCount rows width (degree - boundaryCount source.val.1 source.val.2) source.val.1 source.val.2
      else 0
    else 0

theorem Ed_eq_transferCount (rows width degree : Nat) (rowsPositive : 0 < rows) :
    (Ed degree rows (width + 2) : Int) = (transferCount rows width degree : Int) := by
  classical
  rw [Ed_eq_cnPoly_coeff, cnPoly_eq_matrix_power rows width rowsPositive]
  change (∑ source : TransferState rows, leftBdyVecNorm rows source * phi rows width source).coeff degree = _
  rw [Polynomial.finset_sum_coeff]
  let weight (source : TransferState rows) : Int :=
    if IsNormalizedLeft source then
      if boundaryCount source.val.1 source.val.2 ≤ degree then
        (extensionCount rows width (degree - boundaryCount source.val.1 source.val.2)
          source.val.1 source.val.2 : Int)
      else 0
    else 0
  calc
    (∑ source : TransferState rows, (leftBdyVecNorm rows source * phi rows width source).coeff degree) =
        ∑ source : TransferState rows, weight source := by
      apply Finset.sum_congr rfl
      intro source _
      by_cases normalized : IsNormalizedLeft source
      · rw [leftBdyVecNorm, if_pos normalized, leftBdyVec, Polynomial.monomial_one_right_eq_X_pow,
          Polynomial.coeff_X_pow_mul', phi_coeff_eq_extensionCount]
        simp only [weight, if_pos normalized, boundaryCount]
      · simp only [leftBdyVecNorm, if_neg normalized, zero_mul, Polynomial.coeff_zero, weight]
    _ = ∑ pair : ProperPairState rows, weight ((transferStateEquivProper rows).symm pair) :=
      (Equiv.sum_comp (transferStateEquivProper rows).symm weight).symm
    _ = (transferCount rows width degree : Int) := by
      simp only [transferCount, Nat.cast_sum, Nat.cast_ite, Nat.cast_zero]
      rfl

set_option maxHeartbeats 600000 in
set_option maxRecDepth 65536 in
theorem transferCount_four_two_four : transferCount 4 2 4 = 76 := by
  decide

theorem Ed_four_four_four : Ed 4 4 4 = 76 := by
  have count := Ed_eq_transferCount 4 2 4 (by decide)
  rw [transferCount_four_two_four] at count
  exact_mod_cast count

set_option maxHeartbeats 600000 in
set_option maxRecDepth 65536 in
theorem transferCount_four_three_four : transferCount 4 3 4 = 108 := by
  decide

theorem Ed_four_four_five : Ed 4 4 5 = 108 := by
  have count := Ed_eq_transferCount 4 3 4 (by decide)
  rw [transferCount_four_three_four] at count
  exact_mod_cast count

set_option maxHeartbeats 600000 in
set_option maxRecDepth 65536 in
theorem transferCount_five_three_four : transferCount 5 3 4 = 146 := by
  decide

theorem Ed_four_five_five : Ed 4 5 5 = 146 := by
  have count := Ed_eq_transferCount 5 3 4 (by decide)
  rw [transferCount_five_three_four] at count
  exact_mod_cast count

#print axioms extensionCount
#print axioms sum_columns_eq_proper
#print axioms phi_coeff_eq_extensionCount
#print axioms transferCount
#print axioms Ed_eq_transferCount
#print axioms transferCount_four_two_four
#print axioms Ed_four_four_four
#print axioms transferCount_four_three_four
#print axioms Ed_four_four_five
#print axioms transferCount_five_three_four
#print axioms Ed_four_five_five

end OrigamiCone.Sequel.SparseTransferCounts
