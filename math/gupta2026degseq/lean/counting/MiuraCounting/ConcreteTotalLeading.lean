import MiuraCounting.ConcreteDegreeRemainder
import MiuraCounting.ConcreteUniformOnset
import Mathlib.RingTheory.MvPolynomial.Homogeneous

namespace OrigamiCone.Sequel.TotalLeading

open OrigamiCone SplitPolynomials SideDecomposition DegreeRemainder RemainderPolynomials

noncomputable section

def sideAxisPolynomial (degree : Nat) : Polynomial Rat :=
  Polynomial.preHilbertPoly Rat (degree - 2) degree

def sidePolynomial (degree : Nat) : MvPolynomial (Fin 2) Rat :=
  MvPolynomial.C 4 * ((sideAxisPolynomial degree).toMvPolynomial (0 : Fin 2) +
    (sideAxisPolynomial degree).toMvPolynomial (1 : Fin 2))

def totalPolynomial (degree : Nat) : MvPolynomial (Fin 2) Rat :=
  sidePolynomial degree + remainderPolynomial degree

theorem sideAxisPolynomial_eval (degree size : Nat) (degreeBound : 2 ≤ degree) (large : degree ≤ size) :
    (sideAxisPolynomial degree).eval (size : Rat) = ((size - 2).choose (degree - 2) : Rat) := by
  rw [sideAxisPolynomial, Polynomial.preHilbertPoly_eq_choose_sub_add Rat (degree - 2) large]
  congr 2
  omega

theorem sidePolynomial_eval (rows width degree : Nat) (degreeBound : 3 ≤ degree)
    (rowBound : degree ≤ rows) (columnBound : degree ≤ width) :
    evalAt (sidePolynomial degree) rows width = (Nat.card (CountedSingleSideHeights rows width degree) : Rat) := by
  rw [singleSide_card rows width degree (by omega) (by omega) degreeBound]
  unfold sidePolynomial evalAt
  rw [map_mul, map_add, MvPolynomial.eval_C, MvPolynomial.eval_toMvPolynomial,
    MvPolynomial.eval_toMvPolynomial]
  change (4 : Rat) * ((sideAxisPolynomial degree).eval (rows : Rat) +
    (sideAxisPolynomial degree).eval (width : Rat)) = _
  rw [sideAxisPolynomial_eval degree rows (by omega) rowBound,
    sideAxisPolynomial_eval degree width (by omega) columnBound]
  push_cast
  ring

theorem sidePolynomial_totalDegree_le (degree : Nat) : (sidePolynomial degree).totalDegree ≤ degree - 2 := by
  have rowBound := toMvPolynomial_totalDegree_le (sideAxisPolynomial degree) (0 : Fin 2)
  have columnBound := toMvPolynomial_totalDegree_le (sideAxisPolynomial degree) (1 : Fin 2)
  simp only [sideAxisPolynomial, Polynomial.natDegree_preHilbertPoly] at rowBound columnBound
  unfold sidePolynomial
  refine (MvPolynomial.totalDegree_mul _ _).trans ?_
  simp only [MvPolynomial.totalDegree_C, Nat.zero_add]
  exact (MvPolynomial.totalDegree_add _ _).trans (max_le rowBound columnBound)

theorem toMvPolynomial_homogeneousComponent (polynomial : Polynomial Rat) (coordinate : Fin 2) (degree : Nat) :
    MvPolynomial.homogeneousComponent degree (polynomial.toMvPolynomial coordinate) =
      MvPolynomial.C (polynomial.coeff degree) * MvPolynomial.X coordinate ^ degree := by
  classical
  have powerHomogeneous (index : Nat) :
      (MvPolynomial.X coordinate ^ index : MvPolynomial (Fin 2) Rat).IsHomogeneous index := by
    simpa using (MvPolynomial.isHomogeneous_X (R := Rat) coordinate).pow index
  conv_lhs => rw [polynomial.as_sum_range_C_mul_X_pow]
  simp only [map_sum, map_mul, map_pow, Polynomial.toMvPolynomial_C, Polynomial.toMvPolynomial_X,
    MvPolynomial.homogeneousComponent_C_mul]
  rw [Finset.sum_eq_single degree]
  · rw [MvPolynomial.homogeneousComponent_of_mem (powerHomogeneous degree)]
    simp
  · intro index _ different
    rw [MvPolynomial.homogeneousComponent_of_mem (powerHomogeneous index)]
    simp [Ne.symm different]
  · intro notMember
    have above : polynomial.natDegree < degree := by
      simp only [Finset.mem_range] at notMember
      omega
    rw [Polynomial.coeff_eq_zero_of_natDegree_lt above]
    simp

theorem sidePolynomial_leading (degree : Nat) :
    MvPolynomial.homogeneousComponent (degree - 2) (sidePolynomial degree) =
      MvPolynomial.C ((4 : Rat) / ((degree - 2).factorial : Rat)) *
        (MvPolynomial.X (0 : Fin 2) ^ (degree - 2) + MvPolynomial.X (1 : Fin 2) ^ (degree - 2)) := by
  have coefficient : (sideAxisPolynomial degree).coeff (degree - 2) = ((degree - 2).factorial : Rat)⁻¹ :=
    Polynomial.coeff_preHilbertPoly_self Rat (degree - 2) degree
  unfold sidePolynomial
  rw [MvPolynomial.homogeneousComponent_C_mul, map_add,
    toMvPolynomial_homogeneousComponent, toMvPolynomial_homogeneousComponent, coefficient]
  simp only [div_eq_mul_inv, map_mul]
  ring

theorem totalPolynomial_eval (rows width degree : Nat) (degreeBound : 5 ≤ degree)
    (rowBound : 2 * degree - 1 ≤ rows) (columnBound : 2 * degree - 1 ≤ width) :
    evalAt (totalPolynomial degree) rows width = (Ed degree rows width : Rat) := by
  change evalAt (sidePolynomial degree + remainderPolynomial degree) rows width = _
  unfold evalAt
  rw [map_add]
  change evalAt (sidePolynomial degree) rows width + evalAt (remainderPolynomial degree) rows width = _
  rw [sidePolynomial_eval rows width degree (by omega) (by omega) (by omega),
    remainderPolynomial_eval rows width degree (by omega) rowBound columnBound,
    Ed_partition rows width degree (by omega) (by omega), Nat.cast_add]

theorem totalPolynomial_totalDegree_le (degree : Nat) (degreeBound : 5 ≤ degree) :
    (totalPolynomial degree).totalDegree ≤ degree - 2 := by
  unfold totalPolynomial
  refine (MvPolynomial.totalDegree_add _ _).trans (max_le (sidePolynomial_totalDegree_le degree) ?_)
  exact (remainderPolynomial_totalDegree degree degreeBound).trans (by omega)

theorem totalPolynomial_leading (degree : Nat) (degreeBound : 5 ≤ degree) :
    MvPolynomial.homogeneousComponent (degree - 2) (totalPolynomial degree) =
      MvPolynomial.C ((4 : Rat) / ((degree - 2).factorial : Rat)) *
        (MvPolynomial.X (0 : Fin 2) ^ (degree - 2) + MvPolynomial.X (1 : Fin 2) ^ (degree - 2)) := by
  have remainderBound := remainderPolynomial_totalDegree degree degreeBound
  have strict : (remainderPolynomial degree).totalDegree < degree - 2 := by omega
  rw [totalPolynomial, map_add, MvPolynomial.homogeneousComponent_eq_zero _ _ strict, add_zero,
    sidePolynomial_leading]

theorem leadingForm_ne_zero (degree : Nat) :
    (MvPolynomial.C ((4 : Rat) / ((degree - 2).factorial : Rat)) *
      (MvPolynomial.X (0 : Fin 2) ^ (degree - 2) + MvPolynomial.X (1 : Fin 2) ^ (degree - 2)) :
        MvPolynomial (Fin 2) Rat) ≠ 0 := by
  intro equality
  have evaluated := congrArg (MvPolynomial.eval (fun _ : Fin 2 => (1 : Rat))) equality
  simp only [map_mul, map_add, map_pow, MvPolynomial.eval_C, MvPolynomial.eval_X, one_pow, map_zero] at evaluated
  have factorialPositive : (0 : Rat) < ((degree - 2).factorial : Rat) := by
    exact_mod_cast Nat.factorial_pos (degree - 2)
  have coefficientPositive : (0 : Rat) < 4 / ((degree - 2).factorial : Rat) := by positivity
  nlinarith

theorem sidePolynomial_totalDegree (degree : Nat) : (sidePolynomial degree).totalDegree = degree - 2 := by
  apply Nat.le_antisymm (sidePolynomial_totalDegree_le degree)
  by_contra notBound
  have strict : (sidePolynomial degree).totalDegree < degree - 2 := by omega
  have vanishes := MvPolynomial.homogeneousComponent_eq_zero (degree - 2) (sidePolynomial degree) strict
  rw [sidePolynomial_leading degree] at vanishes
  exact leadingForm_ne_zero degree vanishes

theorem totalPolynomial_totalDegree (degree : Nat) (degreeBound : 5 ≤ degree) :
    (totalPolynomial degree).totalDegree = degree - 2 := by
  apply Nat.le_antisymm (totalPolynomial_totalDegree_le degree degreeBound)
  by_contra notBound
  have strict : (totalPolynomial degree).totalDegree < degree - 2 := by omega
  have vanishes := MvPolynomial.homogeneousComponent_eq_zero (degree - 2) (totalPolynomial degree) strict
  rw [totalPolynomial_leading degree degreeBound] at vanishes
  exact leadingForm_ne_zero degree vanishes

theorem representing_polynomial_totalDegree_and_leading (degree : Nat) (degreeBound : 5 ≤ degree)
    (polynomial : MvPolynomial (Fin 2) Rat)
    (agreement : ∀ rows width : Nat, degree - 1 ≤ rows -> degree - 1 ≤ width ->
      evalAt polynomial rows width = (Ed degree rows width : Rat)) :
    polynomial.totalDegree = degree - 2 ∧
      MvPolynomial.homogeneousComponent (degree - 2) polynomial =
        MvPolynomial.C ((4 : Rat) / ((degree - 2).factorial : Rat)) *
          (MvPolynomial.X (0 : Fin 2) ^ (degree - 2) + MvPolynomial.X (1 : Fin 2) ^ (degree - 2)) := by
  have identification : polynomial = totalPolynomial degree := by
    apply polynomial_eq_on_nat_tail polynomial (totalPolynomial degree) (2 * degree - 1)
    intro rows width rowBound columnBound
    exact (agreement rows width (by omega) (by omega)).trans
      (totalPolynomial_eval rows width degree degreeBound rowBound columnBound).symm
  rw [identification]
  exact ⟨totalPolynomial_totalDegree degree degreeBound, totalPolynomial_leading degree degreeBound⟩

theorem Ed_exists_stable_mvPolynomial (degree : Nat) (degreeBound : 3 ≤ degree) :
    ∃ polynomial : MvPolynomial (Fin 2) Rat,
      ∀ rows width : Nat, degree - 1 ≤ rows -> degree - 1 ≤ width ->
        evalAt polynomial rows width = (Ed degree rows width : Rat) := by
  classical
  obtain ⟨widthPieces, rowPieces, _, _, agreement, _⟩ :=
    ConcreteTransfer.Ed_symmetric_bivariate_upper degreeBound
  refine ⟨∑ index, (widthPieces index).toMvPolynomial (1 : Fin 2) *
    (rowPieces index).toMvPolynomial (0 : Fin 2), ?_⟩
  intro rows width rowBound columnBound
  rw [agreement rows width rowBound columnBound]
  unfold evalAt
  simp only [map_sum, map_mul, MvPolynomial.eval_toMvPolynomial]
  rfl

theorem Ed_totalDegree_and_leading (degree : Nat) (degreeBound : 5 ≤ degree) :
    ∃ polynomial : MvPolynomial (Fin 2) Rat,
      (∀ rows width : Nat, degree - 1 ≤ rows -> degree - 1 ≤ width ->
        evalAt polynomial rows width = (Ed degree rows width : Rat)) ∧
      polynomial.totalDegree = degree - 2 ∧
      MvPolynomial.homogeneousComponent (degree - 2) polynomial =
        MvPolynomial.C ((4 : Rat) / ((degree - 2).factorial : Rat)) *
          (MvPolynomial.X (0 : Fin 2) ^ (degree - 2) + MvPolynomial.X (1 : Fin 2) ^ (degree - 2)) := by
  obtain ⟨polynomial, agreement⟩ := Ed_exists_stable_mvPolynomial degree (by omega)
  exact ⟨polynomial, agreement,
    representing_polynomial_totalDegree_and_leading degree degreeBound polynomial agreement⟩

#print axioms sideAxisPolynomial
#print axioms sidePolynomial
#print axioms totalPolynomial
#print axioms sideAxisPolynomial_eval
#print axioms sidePolynomial_eval
#print axioms sidePolynomial_totalDegree_le
#print axioms toMvPolynomial_homogeneousComponent
#print axioms sidePolynomial_leading
#print axioms totalPolynomial_eval
#print axioms totalPolynomial_totalDegree_le
#print axioms totalPolynomial_leading
#print axioms leadingForm_ne_zero
#print axioms sidePolynomial_totalDegree
#print axioms totalPolynomial_totalDegree
#print axioms representing_polynomial_totalDegree_and_leading
#print axioms Ed_exists_stable_mvPolynomial
#print axioms Ed_totalDegree_and_leading

end

end OrigamiCone.Sequel.TotalLeading
