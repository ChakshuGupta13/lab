import MiuraCounting.ConcreteJointTypePartition
import MiuraCounting.ConcreteUniformRunBound

namespace OrigamiCone.Sequel.SplitPolynomials

open OrigamiCone FrozenExpansion JointReduction JointFibers JointTypePartition

noncomputable def axisPolynomial (dimension runs : Nat) : Polynomial Rat :=
  Polynomial.preHilbertPoly Rat (runs - 1) dimension

theorem axisPolynomial_natDegree (dimension runs : Nat) :
    (axisPolynomial dimension runs).natDegree = runs - 1 :=
  Polynomial.natDegree_preHilbertPoly Rat (runs - 1) dimension

theorem axisPolynomial_eval (dimension runs size : Nat) (positive : 0 < runs) (large : dimension ≤ size) :
    (axisPolynomial dimension runs).eval (size : Rat) =
      ((size - dimension + runs - 1).choose (runs - 1) : Rat) := by
  have top : size - dimension + (runs - 1) = size - dimension + runs - 1 := by omega
  rw [axisPolynomial, Polynomial.preHilbertPoly_eq_choose_sub_add Rat (runs - 1) large, top]

noncomputable def corePolynomial (core : GridState) : MvPolynomial (Fin 2) Rat :=
  if 0 < numFrozenRuns (swapH core.columns.height) ∧ 0 < numFrozenRuns core.columns.height then
    (axisPolynomial core.rows (numFrozenRuns (swapH core.columns.height))).toMvPolynomial (0 : Fin 2) *
      (axisPolynomial core.columns.width (numFrozenRuns core.columns.height)).toMvPolynomial (1 : Fin 2)
  else 0

noncomputable def evalAt (polynomial : MvPolynomial (Fin 2) Rat) (rows width : Nat) : Rat :=
  MvPolynomial.eval (fun coordinate => if coordinate = 0 then (rows : Rat) else (width : Rat)) polynomial

theorem corePolynomial_eval (core : GridState) (rows width : Nat)
    (rowBound : core.rows ≤ rows) (columnBound : core.columns.width ≤ width) :
    evalAt (corePolynomial core) rows width = (coreWeight core rows width : Rat) := by
  by_cases positive : 0 < numFrozenRuns (swapH core.columns.height) ∧ 0 < numFrozenRuns core.columns.height
  · unfold evalAt
    rw [corePolynomial, if_pos positive, map_mul, MvPolynomial.eval_toMvPolynomial,
      MvPolynomial.eval_toMvPolynomial]
    change (axisPolynomial core.rows (numFrozenRuns (swapH core.columns.height))).eval (rows : Rat) *
      (axisPolynomial core.columns.width (numFrozenRuns core.columns.height)).eval (width : Rat) = _
    rw [axisPolynomial_eval _ _ rows positive.1 rowBound,
      axisPolynomial_eval _ _ width positive.2 columnBound]
    simp only [coreWeight, if_pos positive, Nat.cast_mul]
  · unfold evalAt
    rw [corePolynomial, if_neg positive, map_zero]
    simp only [coreWeight, if_neg positive, Nat.cast_zero]

noncomputable def familyPolynomial (minima maxima : Nat) (predicate : GridState -> Prop) :
    MvPolynomial (Fin 2) Rat :=
  ∑ core : FilteredTypes (minima + maxima) (fun source => HasSplit minima maxima source ∧ predicate source),
    corePolynomial core.val.val

theorem familyPolynomial_eval (minima maxima rows width : Nat)
    (degreeBound : 3 ≤ minima + maxima)
    (rowBound : 2 * (minima + maxima) - 1 ≤ rows)
    (columnBound : 2 * (minima + maxima) - 1 ≤ width)
    (predicate : GridState -> Prop)
    (invariant : ∀ core : GridState, ∀ source : NormalFiber core,
      predicate source.val ↔ predicate core) :
    evalAt (familyPolynomial minima maxima predicate) rows width =
      (Nat.card (SplitFamilyStates rows width minima maxima predicate) : Rat) := by
  rw [splitFamilyStates_stable_card rows width minima maxima degreeBound rowBound columnBound predicate invariant]
  unfold familyPolynomial evalAt
  rw [map_sum, Nat.cast_sum]
  apply Finset.sum_congr rfl
  intro core _
  have bounds := reduced_bounds core.val.val core.val.property.1 (minima + maxima) core.val.property.2
  exact corePolynomial_eval core.val.val rows width (bounds.1.trans rowBound) (bounds.2.1.trans columnBound)

theorem toMvPolynomial_totalDegree_le (polynomial : Polynomial Rat) (coordinate : Fin 2) :
    (polynomial.toMvPolynomial coordinate).totalDegree ≤ polynomial.natDegree := by
  classical
  conv_lhs => rw [polynomial.as_sum_range_C_mul_X_pow]
  rw [map_sum]
  apply MvPolynomial.totalDegree_finsetSum_le
  intro index member
  simp only [map_mul, map_pow, Polynomial.toMvPolynomial_C, Polynomial.toMvPolynomial_X]
  have product := MvPolynomial.totalDegree_mul (MvPolynomial.C (polynomial.coeff index))
    ((MvPolynomial.X coordinate : MvPolynomial (Fin 2) Rat) ^ index)
  have power := MvPolynomial.totalDegree_pow (MvPolynomial.X coordinate : MvPolynomial (Fin 2) Rat) index
  simp only [MvPolynomial.totalDegree_C, MvPolynomial.totalDegree_X, Nat.zero_add, Nat.mul_one] at product power
  have indexBound := Finset.mem_range.mp member
  omega

theorem toMvPolynomial_degreeOf_le (polynomial : Polynomial Rat) (coordinate target : Fin 2) :
    (polynomial.toMvPolynomial coordinate).degreeOf target ≤
      if target = coordinate then polynomial.natDegree else 0 := by
  classical
  conv_lhs => rw [polynomial.as_sum_range_C_mul_X_pow]
  rw [map_sum]
  refine (MvPolynomial.degreeOf_sum_le target _ _).trans (Finset.sup_le ?_)
  intro index member
  simp only [map_mul, map_pow, Polynomial.toMvPolynomial_C, Polynomial.toMvPolynomial_X]
  have product := MvPolynomial.degreeOf_mul_le target (MvPolynomial.C (polynomial.coeff index))
    ((MvPolynomial.X coordinate : MvPolynomial (Fin 2) Rat) ^ index)
  have power := MvPolynomial.degreeOf_pow_le target
    (MvPolynomial.X coordinate : MvPolynomial (Fin 2) Rat) index
  have indexBound := Finset.mem_range.mp member
  by_cases same : target = coordinate
  · simp only [MvPolynomial.degreeOf_C, MvPolynomial.degreeOf_X, if_pos same, Nat.zero_add, Nat.mul_one] at product power
    rw [if_pos same]
    omega
  · simp only [MvPolynomial.degreeOf_C, MvPolynomial.degreeOf_X, if_neg same, Nat.zero_add, Nat.mul_zero] at product power
    rw [if_neg same]
    omega

theorem corePolynomial_totalDegree (core : GridState) :
    (corePolynomial core).totalDegree ≤
      (numFrozenRuns (swapH core.columns.height) - 1) + (numFrozenRuns core.columns.height - 1) := by
  unfold corePolynomial
  split_ifs with positive
  · have product := MvPolynomial.totalDegree_mul
      ((axisPolynomial core.rows (numFrozenRuns (swapH core.columns.height))).toMvPolynomial (0 : Fin 2))
      ((axisPolynomial core.columns.width (numFrozenRuns core.columns.height)).toMvPolynomial (1 : Fin 2))
    have rowDegree := toMvPolynomial_totalDegree_le
      (axisPolynomial core.rows (numFrozenRuns (swapH core.columns.height))) (0 : Fin 2)
    have columnDegree := toMvPolynomial_totalDegree_le
      (axisPolynomial core.columns.width (numFrozenRuns core.columns.height)) (1 : Fin 2)
    rw [axisPolynomial_natDegree] at rowDegree columnDegree
    omega
  · rw [MvPolynomial.totalDegree_zero]
    omega

theorem corePolynomial_degreeOf (core : GridState) (coordinate : Fin 2) :
    (corePolynomial core).degreeOf coordinate ≤
      if coordinate = 0 then numFrozenRuns (swapH core.columns.height) - 1
      else numFrozenRuns core.columns.height - 1 := by
  by_cases positive : 0 < numFrozenRuns (swapH core.columns.height) ∧ 0 < numFrozenRuns core.columns.height
  · rw [corePolynomial, if_pos positive]
    have product := MvPolynomial.degreeOf_mul_le coordinate
      ((axisPolynomial core.rows (numFrozenRuns (swapH core.columns.height))).toMvPolynomial (0 : Fin 2))
      ((axisPolynomial core.columns.width (numFrozenRuns core.columns.height)).toMvPolynomial (1 : Fin 2))
    have rowDegree := toMvPolynomial_degreeOf_le
      (axisPolynomial core.rows (numFrozenRuns (swapH core.columns.height))) (0 : Fin 2) coordinate
    have columnDegree := toMvPolynomial_degreeOf_le
      (axisPolynomial core.columns.width (numFrozenRuns core.columns.height)) (1 : Fin 2) coordinate
    rw [axisPolynomial_natDegree] at rowDegree columnDegree
    fin_cases coordinate <;> norm_num at product rowDegree columnDegree ⊢ <;> omega
  · rw [corePolynomial, if_neg positive, MvPolynomial.degreeOf_zero]
    exact Nat.zero_le _

theorem familyPolynomial_degreeOf (minima maxima : Nat) (degreeBound : 3 ≤ minima + maxima)
    (predicate : GridState -> Prop) (coordinate : Fin 2) :
    (familyPolynomial minima maxima predicate).degreeOf coordinate ≤ minima + maxima - 2 := by
  unfold familyPolynomial
  refine (MvPolynomial.degreeOf_sum_le coordinate _ _).trans (Finset.sup_le ?_)
  intro core _
  have bounds := reduced_bounds core.val.val core.val.property.1 (minima + maxima) core.val.property.2
  have polynomialBound := corePolynomial_degreeOf core.val.val coordinate
  fin_cases coordinate <;> norm_num at polynomialBound ⊢ <;> omega

theorem nonseparable_core_totalDegree (core : GridState) (degree : Nat)
    (degreeBound : 5 ≤ degree) (counts : numExtrema core.columns.height = degree)
    (nonseparable : ¬ AdditivelySeparable core.columns.height) :
    (corePolynomial core).totalDegree ≤ degree - 3 := by
  by_cases positive : 0 < numFrozenRuns (swapH core.columns.height) ∧ 0 < numFrozenRuns core.columns.height
  · have runs := UniformRunBound.nonseparable_run_bound core.columns.height core.columns.canonical.1
      core.rowsValid core.columns.widthValid (by omega) nonseparable
    rw [counts] at runs
    have bound := corePolynomial_totalDegree core
    omega
  · rw [corePolynomial, if_neg positive, MvPolynomial.totalDegree_zero]
    exact Nat.zero_le _

theorem nonseparable_familyPolynomial_totalDegree (minima maxima : Nat)
    (degreeBound : 5 ≤ minima + maxima) :
    (familyPolynomial minima maxima (fun source => ¬ AdditivelySeparable source.columns.height)).totalDegree ≤
      minima + maxima - 3 := by
  unfold familyPolynomial
  apply MvPolynomial.totalDegree_finsetSum_le
  intro core _
  exact nonseparable_core_totalDegree core.val.val (minima + maxima) degreeBound
    core.val.property.2 core.property.2

theorem splitFamily_eventually_polynomial (minima maxima : Nat) (degreeBound : 3 ≤ minima + maxima)
    (predicate : GridState -> Prop)
    (invariant : ∀ core : GridState, ∀ source : NormalFiber core,
      predicate source.val ↔ predicate core) :
    ∃ polynomial : MvPolynomial (Fin 2) Rat,
      (∀ coordinate, polynomial.degreeOf coordinate ≤ minima + maxima - 2) ∧
      ∀ rows width : Nat, 2 * (minima + maxima) - 1 ≤ rows ->
        2 * (minima + maxima) - 1 ≤ width ->
          evalAt polynomial rows width = (Nat.card (SplitFamilyStates rows width minima maxima predicate) : Rat) := by
  refine ⟨familyPolynomial minima maxima predicate, ?_, ?_⟩
  · exact familyPolynomial_degreeOf minima maxima degreeBound predicate
  · intro rows width rowBound columnBound
    exact familyPolynomial_eval minima maxima rows width degreeBound rowBound columnBound predicate invariant

theorem nonseparableSplit_eventually_polynomial (minima maxima : Nat) (degreeBound : 5 ≤ minima + maxima) :
    ∃ polynomial : MvPolynomial (Fin 2) Rat,
      polynomial.totalDegree ≤ minima + maxima - 3 ∧
      ∀ rows width : Nat, 2 * (minima + maxima) - 1 ≤ rows ->
        2 * (minima + maxima) - 1 ≤ width ->
          evalAt polynomial rows width =
            (Nat.card (SplitFamilyStates rows width minima maxima
              (fun source => ¬ AdditivelySeparable source.columns.height)) : Rat) := by
  refine ⟨familyPolynomial minima maxima (fun source => ¬ AdditivelySeparable source.columns.height),
    nonseparable_familyPolynomial_totalDegree minima maxima degreeBound, ?_⟩
  intro rows width rowBound columnBound
  apply familyPolynomial_eval minima maxima rows width (by omega) rowBound columnBound
  intro core source
  exact not_congr (normalFiber_separable_iff core source)

#print axioms axisPolynomial_natDegree
#print axioms axisPolynomial_eval
#print axioms corePolynomial_eval
#print axioms familyPolynomial_eval
#print axioms toMvPolynomial_totalDegree_le
#print axioms toMvPolynomial_degreeOf_le
#print axioms corePolynomial_totalDegree
#print axioms corePolynomial_degreeOf
#print axioms familyPolynomial_degreeOf
#print axioms nonseparable_core_totalDegree
#print axioms nonseparable_familyPolynomial_totalDegree
#print axioms splitFamily_eventually_polynomial
#print axioms nonseparableSplit_eventually_polynomial

end OrigamiCone.Sequel.SplitPolynomials
