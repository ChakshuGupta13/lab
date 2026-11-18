import MiuraCounting.ConcreteSparseTransferCounts
import MiuraCounting.ConcreteUniformOnset
import MiuraCounting.ConcreteSmallDegrees
import MiuraCounting.ConcreteTotalLeading

namespace OrigamiCone.Sequel.SmallDegreePolynomials

open ConcreteTransfer SmallCountChecks SparseTransferCounts SplitPolynomials

def degreeFourFormula (rows width : Rat) : Rat :=
  2 * rows ^ 2 + 2 * width ^ 2 + 6 * rows * width - 10 * (rows + width) - 4

noncomputable def degreeFourRowPolynomial (rows : Rat) : Polynomial Rat :=
  Polynomial.C 2 * Polynomial.X ^ 2 + Polynomial.C (6 * rows - 10) * Polynomial.X +
    Polynomial.C (2 * rows ^ 2 - 10 * rows - 4)

theorem degreeFourFormula_symm (rows width : Rat) : degreeFourFormula rows width = degreeFourFormula width rows := by
  unfold degreeFourFormula
  ring

theorem degreeFourRowPolynomial_eval (rows width : Rat) :
    (degreeFourRowPolynomial rows).eval width = degreeFourFormula rows width := by
  simp only [degreeFourRowPolynomial, Polynomial.eval_add, Polynomial.eval_mul, Polynomial.eval_C,
    Polynomial.eval_pow, Polynomial.eval_X]
  unfold degreeFourFormula
  ring

theorem degreeFourRowPolynomial_degree (rows : Rat) : (degreeFourRowPolynomial rows).natDegree ≤ 2 := by
  unfold degreeFourRowPolynomial
  refine (Polynomial.natDegree_add_le _ _).trans (max_le ?_ ?_)
  · refine (Polynomial.natDegree_add_le _ _).trans (max_le ?_ ?_)
    · exact (Polynomial.natDegree_C_mul_le _ _).trans (by simp)
    · exact (Polynomial.natDegree_C_mul_le _ _).trans (by simp)
  · simp only [Polynomial.natDegree_C, Nat.zero_le]

theorem quadratic_eq_of_nodes (first second : Polynomial Rat)
    (firstBound : first.natDegree ≤ 2) (secondBound : second.natDegree ≤ 2)
    (agreement : ∀ index : Fin 3,
      first.eval ((index.val + 3 : Nat) : Rat) = second.eval ((index.val + 3 : Nat) : Rat)) : first = second := by
  refine Polynomial.eq_of_natDegree_lt_card_of_eval_eq first second
    (f := fun index : Fin 3 => ((index.val + 3 : Nat) : Rat)) ?_ agreement ?_
  · intro firstIndex secondIndex equality
    apply Fin.ext
    change ((firstIndex.val + 3 : Nat) : Rat) = ((secondIndex.val + 3 : Nat) : Rat) at equality
    have sameNat : firstIndex.val + 3 = secondIndex.val + 3 := by exact_mod_cast equality
    omega
  · change max first.natDegree second.natDegree < 3
    omega

theorem Ed_four_four_three : Ed 4 4 3 = 48 := by
  have count := Ed_threeColumnCount 4 4 (by decide)
  rw [threeColumnCount_four_four] at count
  exact_mod_cast count

theorem Ed_four_five_three : Ed 4 5 3 = 74 := by
  have count := Ed_threeColumnCount 5 4 (by decide)
  rw [threeColumnCount_five_four] at count
  exact_mod_cast count

theorem Ed_four_nodes (row column : Fin 3) :
    (Ed 4 (row.val + 3) (column.val + 3) : Rat) =
      degreeFourFormula ((row.val + 3 : Nat) : Rat) ((column.val + 3 : Nat) : Rat) := by
  have nodeThreeFour : Ed 4 3 4 = 48 := (Ed_symm 4 (by decide) (by decide)).trans Ed_four_four_three
  have nodeThreeFive : Ed 4 3 5 = 74 := (Ed_symm 4 (by decide) (by decide)).trans Ed_four_five_three
  have nodeFiveFour : Ed 4 5 4 = 108 := (Ed_symm 4 (by decide) (by decide)).trans Ed_four_four_five
  fin_cases row <;> fin_cases column <;>
    norm_num [degreeFourFormula, Ed_four_three_three, Ed_four_four_three, Ed_four_five_three,
      Ed_four_four_four, Ed_four_four_five, Ed_four_five_five, nodeThreeFour, nodeThreeFive, nodeFiveFour]

theorem Ed_four_nodeRow (row : Fin 3) (width : Nat) (widthValid : 3 ≤ width) :
    (Ed 4 (row.val + 3) width : Rat) = degreeFourFormula ((row.val + 3 : Nat) : Rat) (width : Rat) := by
  obtain ⟨polynomial, bound, agreement⟩ :=
    Ed_uniform_onset (degree := 4) (rows := row.val + 3) (by decide) (by omega)
  have identification : polynomial = degreeFourRowPolynomial ((row.val + 3 : Nat) : Rat) := by
    apply quadratic_eq_of_nodes polynomial _ bound (degreeFourRowPolynomial_degree _)
    intro index
    rw [← agreement (index.val + 3) (by omega), degreeFourRowPolynomial_eval]
    exact Ed_four_nodes row index
  rw [agreement width widthValid, identification, degreeFourRowPolynomial_eval]

theorem Ed_four (rows width : Nat) (rowsValid : 3 ≤ rows) (widthValid : 3 ≤ width) :
    (Ed 4 rows width : Rat) = degreeFourFormula (rows : Rat) (width : Rat) := by
  obtain ⟨polynomial, bound, agreement⟩ :=
    Ed_uniform_onset (degree := 4) (rows := width) (by decide) (by omega)
  have identification : polynomial = degreeFourRowPolynomial (width : Rat) := by
    apply quadratic_eq_of_nodes polynomial _ bound (degreeFourRowPolynomial_degree _)
    intro index
    rw [← agreement (index.val + 3) (by omega), degreeFourRowPolynomial_eval,
      Ed_symm 4 (by omega) (by omega)]
    exact (Ed_four_nodeRow index width widthValid).trans (degreeFourFormula_symm _ _)
  calc
    (Ed 4 rows width : Rat) = (Ed 4 width rows : Rat) := by rw [Ed_symm 4 (by omega) (by omega)]
    _ = polynomial.eval (rows : Rat) := agreement rows rowsValid
    _ = (degreeFourRowPolynomial (width : Rat)).eval (rows : Rat) := by rw [identification]
    _ = degreeFourFormula (width : Rat) (rows : Rat) := degreeFourRowPolynomial_eval _ _
    _ = degreeFourFormula (rows : Rat) (width : Rat) := degreeFourFormula_symm _ _

noncomputable def degreeThreePolynomial : MvPolynomial (Fin 2) Rat :=
  MvPolynomial.C 4 * (MvPolynomial.X 0 + MvPolynomial.X 1) - MvPolynomial.C 16

noncomputable def degreeFourPolynomial : MvPolynomial (Fin 2) Rat :=
  MvPolynomial.C 2 * (MvPolynomial.X 0 ^ 2 + MvPolynomial.X 1 ^ 2) +
    MvPolynomial.C 6 * (MvPolynomial.X 0 * MvPolynomial.X 1) -
      MvPolynomial.C 10 * (MvPolynomial.X 0 + MvPolynomial.X 1) - MvPolynomial.C 4

theorem degreeThreePolynomial_eval (rows width : Nat) (rowsValid : 2 ≤ rows) (widthValid : 2 ≤ width) :
    evalAt degreeThreePolynomial rows width = (Ed 3 rows width : Rat) := by
  rw [SmallDegrees.Ed_three rows width rowsValid widthValid]
  unfold degreeThreePolynomial evalAt
  simp only [map_sub, map_mul, map_add, MvPolynomial.eval_C, MvPolynomial.eval_X]
  change (4 : Rat) * ((rows : Rat) + (width : Rat)) - 16 = ((4 * (rows + width - 4) : Nat) : Rat)
  have enough : 4 ≤ rows + width := by omega
  rw [Nat.cast_mul, Nat.cast_sub enough, Nat.cast_add]
  norm_num
  ring

theorem degreeFourPolynomial_eval (rows width : Nat) (rowsValid : 3 ≤ rows) (widthValid : 3 ≤ width) :
    evalAt degreeFourPolynomial rows width = (Ed 4 rows width : Rat) := by
  rw [Ed_four rows width rowsValid widthValid]
  unfold degreeFourPolynomial evalAt
  simp only [map_sub, map_mul, map_add, map_pow, MvPolynomial.eval_C, MvPolynomial.eval_X]
  change (2 : Rat) * ((rows : Rat) ^ 2 + (width : Rat) ^ 2) + 6 * ((rows : Rat) * (width : Rat)) -
    10 * ((rows : Rat) + (width : Rat)) - 4 = degreeFourFormula (rows : Rat) (width : Rat)
  unfold degreeFourFormula
  ring

theorem degreeThreePolynomial_totalDegree_le : degreeThreePolynomial.totalDegree ≤ 1 := by
  have linear : (MvPolynomial.X (0 : Fin 2) + MvPolynomial.X (1 : Fin 2) : MvPolynomial (Fin 2) Rat).totalDegree ≤ 1 := by
    simpa using MvPolynomial.totalDegree_add
      (MvPolynomial.X (0 : Fin 2) : MvPolynomial (Fin 2) Rat) (MvPolynomial.X (1 : Fin 2))
  have scaled := MvPolynomial.totalDegree_mul (MvPolynomial.C (4 : Rat))
    (MvPolynomial.X (0 : Fin 2) + MvPolynomial.X (1 : Fin 2))
  simp only [MvPolynomial.totalDegree_C, Nat.zero_add] at scaled
  unfold degreeThreePolynomial
  exact (MvPolynomial.totalDegree_sub _ _).trans (max_le (scaled.trans linear) (by simp))

theorem degreeFourPolynomial_totalDegree_le : degreeFourPolynomial.totalDegree ≤ 2 := by
  have linear : (MvPolynomial.X (0 : Fin 2) + MvPolynomial.X (1 : Fin 2) : MvPolynomial (Fin 2) Rat).totalDegree ≤ 1 := by
    simpa using MvPolynomial.totalDegree_add
      (MvPolynomial.X (0 : Fin 2) : MvPolynomial (Fin 2) Rat) (MvPolynomial.X (1 : Fin 2))
  have square (coordinate : Fin 2) : (MvPolynomial.X coordinate ^ 2 : MvPolynomial (Fin 2) Rat).totalDegree ≤ 2 := by
    simp
  have squares : (MvPolynomial.X (0 : Fin 2) ^ 2 + MvPolynomial.X (1 : Fin 2) ^ 2 :
      MvPolynomial (Fin 2) Rat).totalDegree ≤ 2 :=
    (MvPolynomial.totalDegree_add _ _).trans (max_le (square 0) (square 1))
  have mixed : (MvPolynomial.X (0 : Fin 2) * MvPolynomial.X (1 : Fin 2) : MvPolynomial (Fin 2) Rat).totalDegree ≤ 2 := by
    simpa using MvPolynomial.totalDegree_mul
      (MvPolynomial.X (0 : Fin 2) : MvPolynomial (Fin 2) Rat) (MvPolynomial.X (1 : Fin 2))
  have scale (scalar : Rat) (polynomial : MvPolynomial (Fin 2) Rat) (bound : Nat)
      (bounded : polynomial.totalDegree ≤ bound) : (MvPolynomial.C scalar * polynomial).totalDegree ≤ bound := by
    have product := MvPolynomial.totalDegree_mul (MvPolynomial.C scalar) polynomial
    simp only [MvPolynomial.totalDegree_C, Nat.zero_add] at product
    exact product.trans bounded
  unfold degreeFourPolynomial
  refine (MvPolynomial.totalDegree_sub _ _).trans (max_le ?_ (by simp))
  refine (MvPolynomial.totalDegree_sub _ _).trans (max_le ?_ ?_)
  · exact (MvPolynomial.totalDegree_add _ _).trans (max_le (scale 2 _ 2 squares) (scale 6 _ 2 mixed))
  · exact (scale 10 _ 1 linear).trans (by decide)

theorem component_coordinate (coordinate : Fin 2) (degree : Nat) :
    MvPolynomial.homogeneousComponent degree (MvPolynomial.X coordinate : MvPolynomial (Fin 2) Rat) =
      if degree = 1 then MvPolynomial.X coordinate else 0 :=
  MvPolynomial.homogeneousComponent_of_mem (MvPolynomial.isHomogeneous_X (R := Rat) coordinate)

theorem component_coordinate_power (coordinate : Fin 2) (exponent degree : Nat) :
    MvPolynomial.homogeneousComponent degree (MvPolynomial.X coordinate ^ exponent : MvPolynomial (Fin 2) Rat) =
      if degree = exponent then MvPolynomial.X coordinate ^ exponent else 0 := by
  have homogeneous : (MvPolynomial.X coordinate ^ exponent : MvPolynomial (Fin 2) Rat).IsHomogeneous exponent := by
    simpa using (MvPolynomial.isHomogeneous_X (R := Rat) coordinate).pow exponent
  exact MvPolynomial.homogeneousComponent_of_mem homogeneous

theorem component_mixed (degree : Nat) :
    MvPolynomial.homogeneousComponent degree
        (MvPolynomial.X (0 : Fin 2) * MvPolynomial.X (1 : Fin 2) : MvPolynomial (Fin 2) Rat) =
      if degree = 2 then MvPolynomial.X 0 * MvPolynomial.X 1 else 0 :=
  MvPolynomial.homogeneousComponent_of_mem
    ((MvPolynomial.isHomogeneous_X (R := Rat) (0 : Fin 2)).mul
      (MvPolynomial.isHomogeneous_X (R := Rat) (1 : Fin 2)))

theorem degreeThreePolynomial_leading :
    MvPolynomial.homogeneousComponent 1 degreeThreePolynomial =
      MvPolynomial.C 4 * (MvPolynomial.X 0 + MvPolynomial.X 1) := by
  have constantGone : MvPolynomial.homogeneousComponent 1 (MvPolynomial.C (16 : Rat) :
      MvPolynomial (Fin 2) Rat) = 0 :=
    MvPolynomial.homogeneousComponent_eq_zero 1 _ (by simp)
  rw [degreeThreePolynomial, map_sub, MvPolynomial.homogeneousComponent_C_mul, map_add,
    component_coordinate, component_coordinate, constantGone]
  simp

theorem degreeFourPolynomial_leading :
    MvPolynomial.homogeneousComponent 2 degreeFourPolynomial =
      MvPolynomial.C 2 * (MvPolynomial.X 0 ^ 2 + MvPolynomial.X 1 ^ 2) +
        MvPolynomial.C 6 * (MvPolynomial.X 0 * MvPolynomial.X 1) := by
  have constantGone : MvPolynomial.homogeneousComponent 2 (MvPolynomial.C (4 : Rat) :
      MvPolynomial (Fin 2) Rat) = 0 :=
    MvPolynomial.homogeneousComponent_eq_zero 2 _ (by simp)
  rw [degreeFourPolynomial, map_sub, map_sub, map_add,
    MvPolynomial.homogeneousComponent_C_mul, map_add,
    component_coordinate_power, component_coordinate_power,
    MvPolynomial.homogeneousComponent_C_mul, component_mixed,
    MvPolynomial.homogeneousComponent_C_mul, map_add,
    component_coordinate, component_coordinate, constantGone]
  norm_num

theorem degreeThreePolynomial_totalDegree : degreeThreePolynomial.totalDegree = 1 := by
  apply Nat.le_antisymm degreeThreePolynomial_totalDegree_le
  by_contra notBound
  have strict : degreeThreePolynomial.totalDegree < 1 := by omega
  have vanishes := MvPolynomial.homogeneousComponent_eq_zero 1 degreeThreePolynomial strict
  rw [degreeThreePolynomial_leading] at vanishes
  have evaluated := congrArg (MvPolynomial.eval (fun _ : Fin 2 => (1 : Rat))) vanishes
  norm_num [map_mul, map_add, MvPolynomial.eval_C, MvPolynomial.eval_X] at evaluated

theorem degreeFourPolynomial_totalDegree : degreeFourPolynomial.totalDegree = 2 := by
  apply Nat.le_antisymm degreeFourPolynomial_totalDegree_le
  by_contra notBound
  have strict : degreeFourPolynomial.totalDegree < 2 := by omega
  have vanishes := MvPolynomial.homogeneousComponent_eq_zero 2 degreeFourPolynomial strict
  rw [degreeFourPolynomial_leading] at vanishes
  have evaluated := congrArg (MvPolynomial.eval (fun _ : Fin 2 => (1 : Rat))) vanishes
  norm_num [map_mul, map_add, map_pow, MvPolynomial.eval_C, MvPolynomial.eval_X] at evaluated

theorem Ed_full_totalDegree_and_leading (degree : Nat) (degreeBound : 3 ≤ degree) :
    ∃ polynomial : MvPolynomial (Fin 2) Rat,
      (∀ rows width : Nat, degree - 1 ≤ rows -> degree - 1 ≤ width ->
        evalAt polynomial rows width = (Ed degree rows width : Rat)) ∧
      polynomial.totalDegree = degree - 2 ∧
      (degree = 3 -> polynomial = degreeThreePolynomial) ∧
      (degree = 4 -> polynomial = degreeFourPolynomial) ∧
      (5 ≤ degree -> MvPolynomial.homogeneousComponent (degree - 2) polynomial =
        MvPolynomial.C ((4 : Rat) / ((degree - 2).factorial : Rat)) *
          (MvPolynomial.X (0 : Fin 2) ^ (degree - 2) + MvPolynomial.X (1 : Fin 2) ^ (degree - 2))) := by
  by_cases isThree : degree = 3
  · subst degree
    refine ⟨degreeThreePolynomial, ?_, degreeThreePolynomial_totalDegree, ?_, ?_, ?_⟩
    · intro rows width rowsValid widthValid
      exact degreeThreePolynomial_eval rows width rowsValid widthValid
    · intro _
      rfl
    · intro impossible
      omega
    · intro impossible
      omega
  · by_cases isFour : degree = 4
    · subst degree
      refine ⟨degreeFourPolynomial, ?_, degreeFourPolynomial_totalDegree, ?_, ?_, ?_⟩
      · intro rows width rowsValid widthValid
        exact degreeFourPolynomial_eval rows width rowsValid widthValid
      · intro impossible
        omega
      · intro _
        rfl
      · intro impossible
        omega
    · obtain ⟨polynomial, agreement, totalDegree, leading⟩ :=
        TotalLeading.Ed_totalDegree_and_leading degree (by omega)
      refine ⟨polynomial, agreement, totalDegree, ?_, ?_, ?_⟩
      · intro impossible
        omega
      · intro impossible
        omega
      · intro _
        exact leading

theorem representing_full_totalDegree_and_leading (degree : Nat) (degreeBound : 3 ≤ degree)
    (polynomial : MvPolynomial (Fin 2) Rat)
    (agreement : ∀ rows width : Nat, degree - 1 ≤ rows -> degree - 1 ≤ width ->
      evalAt polynomial rows width = (Ed degree rows width : Rat)) :
    polynomial.totalDegree = degree - 2 ∧
      (degree = 3 -> polynomial = degreeThreePolynomial) ∧
      (degree = 4 -> polynomial = degreeFourPolynomial) ∧
      (5 ≤ degree -> MvPolynomial.homogeneousComponent (degree - 2) polynomial =
        MvPolynomial.C ((4 : Rat) / ((degree - 2).factorial : Rat)) *
          (MvPolynomial.X (0 : Fin 2) ^ (degree - 2) + MvPolynomial.X (1 : Fin 2) ^ (degree - 2))) := by
  obtain ⟨witness, witnessAgreement, totalDegree, atThree, atFour, leading⟩ :=
    Ed_full_totalDegree_and_leading degree degreeBound
  have identification : polynomial = witness := by
    apply RemainderPolynomials.polynomial_eq_on_nat_tail polynomial witness (degree - 1)
    intro rows width rowsValid widthValid
    exact (agreement rows width rowsValid widthValid).trans (witnessAgreement rows width rowsValid widthValid).symm
  rw [identification]
  exact ⟨totalDegree, atThree, atFour, leading⟩

#print axioms degreeFourFormula
#print axioms degreeFourRowPolynomial
#print axioms degreeFourFormula_symm
#print axioms degreeFourRowPolynomial_eval
#print axioms degreeFourRowPolynomial_degree
#print axioms quadratic_eq_of_nodes
#print axioms Ed_four_four_three
#print axioms Ed_four_five_three
#print axioms Ed_four_nodes
#print axioms Ed_four_nodeRow
#print axioms Ed_four
#print axioms degreeThreePolynomial
#print axioms degreeFourPolynomial
#print axioms degreeThreePolynomial_eval
#print axioms degreeFourPolynomial_eval
#print axioms degreeThreePolynomial_totalDegree_le
#print axioms degreeFourPolynomial_totalDegree_le
#print axioms component_coordinate
#print axioms component_coordinate_power
#print axioms component_mixed
#print axioms degreeThreePolynomial_leading
#print axioms degreeFourPolynomial_leading
#print axioms degreeThreePolynomial_totalDegree
#print axioms degreeFourPolynomial_totalDegree
#print axioms Ed_full_totalDegree_and_leading
#print axioms representing_full_totalDegree_and_leading

end OrigamiCone.Sequel.SmallDegreePolynomials
