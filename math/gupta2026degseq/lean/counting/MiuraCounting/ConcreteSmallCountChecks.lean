import OrigamiCone.SequelEdMatrixPower

namespace OrigamiCone.Sequel.SmallCountChecks

def boundaryCount {rows : Nat} (own other : PathColouring rows) : Nat :=
  (Finset.univ.filter (fun row => IsBoundaryExtremum own other row)).card

def threeColumnCount (rows degree : Nat) : Nat :=
  ∑ source : TransferState rows,
    if IsNormalizedLeft source then
      ∑ right : PathColouring rows,
        if PathAdjacent source.val.2 right ∧
          boundaryCount source.val.1 source.val.2 + columnExtremaCount source.val.1 source.val.2 right +
            boundaryCount right source.val.2 = degree then 1 else 0
    else 0

theorem sum_matching_targets {rows : Nat} (source : TransferState rows)
    (weight : PathColouring rows -> Int) :
    (∑ target : TransferState rows, if source.val.2 = target.val.1 then weight target.val.2 else 0) =
      ∑ right : PathColouring rows, if PathAdjacent source.val.2 right then weight right else 0 := by
  classical
  rw [← Finset.sum_filter, ← Finset.sum_filter]
  apply Finset.sum_bij (fun target _ => target.val.2)
  · intro target member
    have same := (Finset.mem_filter.mp member).2
    apply Finset.mem_filter.mpr
    exact ⟨Finset.mem_univ _, by rw [same]; exact target.property⟩
  · intro first firstMember second secondMember sameRight
    apply Subtype.ext
    apply Prod.ext
    · exact (Finset.mem_filter.mp firstMember).2.symm.trans (Finset.mem_filter.mp secondMember).2
    · exact sameRight
  · intro right member
    let target : TransferState rows := ⟨(source.val.2, right), (Finset.mem_filter.mp member).2⟩
    exact ⟨target, Finset.mem_filter.mpr ⟨Finset.mem_univ _, rfl⟩, rfl⟩
  · intro target _
    rfl

theorem threeColumn_entry_coefficient {rows : Nat} (source target : TransferState rows) (degree : Nat) :
    (leftBdyVecNorm rows source * transferMatrix rows source target * rightBdyVec rows target).coeff degree =
      if IsNormalizedLeft source ∧ source.val.2 = target.val.1 ∧
        boundaryCount source.val.1 source.val.2 + columnExtremaCount source.val.1 source.val.2 target.val.2 +
          boundaryCount target.val.2 source.val.2 = degree then (1 : Int) else 0 := by
  classical
  by_cases normalized : IsNormalizedLeft source
  · by_cases matching : source.val.2 = target.val.1
    · simp [leftBdyVecNorm, normalized, transferMatrix, matching, leftBdyVec, rightBdyVec,
      ← Polynomial.monomial_one_right_eq_X_pow, Polynomial.monomial_mul_monomial,
      Polynomial.coeff_monomial, boundaryCount]
    · simp [leftBdyVecNorm, normalized, transferMatrix, matching]
  · simp [leftBdyVecNorm, normalized]

theorem Ed_threeColumnCount (rows degree : Nat) (rowsPositive : 0 < rows) :
    (Ed degree rows 3 : Int) = (threeColumnCount rows degree : Int) := by
  classical
  rw [Ed_eq_cnPoly_coeff, cnPoly_eq_matrix_power rows 1 rowsPositive]
  simp only [pow_one, dotProduct, Matrix.mulVec, Finset.mul_sum, Polynomial.finset_sum_coeff]
  simp_rw [← mul_assoc, threeColumn_entry_coefficient]
  simp only [threeColumnCount, Nat.cast_sum, Nat.cast_ite, Nat.cast_zero, Nat.cast_one]
  apply Finset.sum_congr rfl
  intro source _
  by_cases normalized : IsNormalizedLeft source
  · simp only [normalized, true_and, if_true]
    simpa only [ite_and] using sum_matching_targets source (fun right =>
      if boundaryCount source.val.1 source.val.2 + columnExtremaCount source.val.1 source.val.2 right +
        boundaryCount right source.val.2 = degree then (1 : Int) else 0)
  · simp [normalized]

def ProperColumn (rows : Nat) : Type :=
  {column : PathColouring rows // IsPathProperColouring column}

instance properColumnFintype (rows : Nat) : Fintype (ProperColumn rows) := by
  unfold ProperColumn
  infer_instance

def ProperPairState (rows : Nat) : Type :=
  {pair : ProperColumn rows × ProperColumn rows // ∀ row, pair.1.val row ≠ pair.2.val row}

instance properPairStateFintype (rows : Nat) : Fintype (ProperPairState rows) := by
  unfold ProperPairState
  infer_instance

def transferStateEquivProper (rows : Nat) : TransferState rows ≃ ProperPairState rows where
  toFun source := ⟨(⟨source.val.1, source.property.1⟩, ⟨source.val.2, source.property.2.1⟩), source.property.2.2⟩
  invFun source := ⟨(source.val.1.val, source.val.2.val), source.val.1.property,
    source.val.2.property, source.property⟩
  left_inv := fun _ => rfl
  right_inv := fun _ => rfl

def compactThreeColumnCount (rows degree : Nat) : Nat :=
  ∑ pair : ProperPairState rows,
    let source := (transferStateEquivProper rows).symm pair
    if IsNormalizedLeft source then
      ∑ right : ProperColumn rows,
        if PathAdjacent source.val.2 right.val ∧
          boundaryCount source.val.1 source.val.2 + columnExtremaCount source.val.1 source.val.2 right.val +
            boundaryCount right.val source.val.2 = degree then 1 else 0
    else 0

theorem threeColumnCount_eq_compact (rows degree : Nat) :
    threeColumnCount rows degree = compactThreeColumnCount rows degree := by
  classical
  let weight (source : TransferState rows) (right : PathColouring rows) : Nat :=
    if PathAdjacent source.val.2 right ∧
      boundaryCount source.val.1 source.val.2 + columnExtremaCount source.val.1 source.val.2 right +
        boundaryCount right source.val.2 = degree then 1 else 0
  calc
    threeColumnCount rows degree =
        ∑ pair : ProperPairState rows,
          let source := (transferStateEquivProper rows).symm pair
          if IsNormalizedLeft source then ∑ right : PathColouring rows, weight source right else 0 :=
      (Equiv.sum_comp (transferStateEquivProper rows).symm
        (fun source => if IsNormalizedLeft source then ∑ right : PathColouring rows, weight source right else 0)).symm
    _ = compactThreeColumnCount rows degree := by
      apply Finset.sum_congr rfl
      intro pair _
      let source := (transferStateEquivProper rows).symm pair
      change (if IsNormalizedLeft source then ∑ right : PathColouring rows, weight source right else 0) =
        if IsNormalizedLeft source then ∑ right : ProperColumn rows, weight source right.val else 0
      by_cases normalized : IsNormalizedLeft source
      · rw [if_pos normalized, if_pos normalized]
        calc
          (∑ right : PathColouring rows, weight source right) =
              ∑ right ∈ Finset.univ.filter IsPathProperColouring, weight source right := by
            rw [Finset.sum_filter]
            apply Finset.sum_congr rfl
            intro right _
            by_cases proper : IsPathProperColouring right
            · simp only [if_pos proper]
            · have notAdjacent : ¬ PathAdjacent source.val.2 right := fun near => proper near.2.1
              simp only [if_neg proper, weight, notAdjacent, false_and, if_false]
          _ = ∑ right : ProperColumn rows, weight source right.val :=
            Finset.sum_subtype _ (by intro right; simp) (weight source)
      · simp only [if_neg normalized]

set_option maxRecDepth 8192 in
theorem threeColumnCount_three_four : threeColumnCount 3 4 = 26 := by
  decide

set_option maxRecDepth 65536 in
theorem threeColumnCount_four_four : threeColumnCount 4 4 = 48 := by
  decide

set_option maxHeartbeats 600000 in
set_option maxRecDepth 65536 in
theorem threeColumnCount_five_four : threeColumnCount 5 4 = 74 := by
  rw [threeColumnCount_eq_compact]
  decide

theorem Ed_four_three_three : Ed 4 3 3 = 26 := by
  have count := Ed_threeColumnCount 3 4 (by decide)
  rw [threeColumnCount_three_four] at count
  exact_mod_cast count

#print axioms boundaryCount
#print axioms threeColumnCount
#print axioms sum_matching_targets
#print axioms threeColumn_entry_coefficient
#print axioms Ed_threeColumnCount
#print axioms transferStateEquivProper
#print axioms compactThreeColumnCount
#print axioms threeColumnCount_eq_compact
#print axioms threeColumnCount_three_four
#print axioms threeColumnCount_four_four
#print axioms threeColumnCount_five_four
#print axioms Ed_four_three_three

end OrigamiCone.Sequel.SmallCountChecks
