import MiuraCounting.ConcreteCoefficientSeries
import MiuraCounting.CoefficientOnset
import OrigamiCone.SequelPolyAssembly

namespace OrigamiCone.Sequel.ConcreteTransfer

theorem Ed_uniform_onset {degree rows : Nat} (degreeValid : 3 ≤ degree) (rowsValid : 2 ≤ rows) :
    ∃ polynomial : Polynomial Rat, polynomial.natDegree ≤ degree - 2 ∧
      ∀ width : Nat, degree - 1 ≤ width ->
        (Ed degree rows width : Rat) = polynomial.eval (width : Rat) := by
  obtain ⟨numerator, numeratorBound, identity⟩ :=
    degreeCountWidthSeries_bounded_numerator rowsValid degree degreeValid
  have adjustedBound : numerator.natDegree ≤ 2 * (degree - 1) - 1 :=
    numeratorBound.trans (by omega)
  obtain ⟨polynomial, polynomialBound, coefficients⟩ :=
    CoefficientOnset.rational_coefficients_polynomial_from (degreeCountWidthSeries rows degree)
      numerator (degree - 1) (by omega) adjustedBound identity
  refine ⟨polynomial, polynomialBound.trans (by omega), ?_⟩
  intro width widthValid
  have gridWidth : 2 ≤ width := by omega
  simpa only [degreeCountWidthSeries, PowerSeries.coeff_mk, gridWidth, if_true] using
    coefficients width widthValid

theorem Ed_symmetric_bivariate_upper {degree : Nat} (degreeValid : 3 ≤ degree) :
    ∃ widthPieces rowPieces : Fin (degree - 2 + 1) -> Polynomial Rat,
      (∀ index, (widthPieces index).natDegree ≤ degree - 2) ∧
      (∀ index, (rowPieces index).natDegree ≤ degree - 2) ∧
      (∀ rows width : Nat, degree - 1 ≤ rows -> degree - 1 ≤ width ->
        (Ed degree rows width : Rat) =
          ∑ index, (widthPieces index).eval (width : Rat) * (rowPieces index).eval (rows : Rat)) ∧
      (∀ rows width : Rat,
        (∑ index, (widthPieces index).eval width * (rowPieces index).eval rows) =
          ∑ index, (widthPieces index).eval rows * (rowPieces index).eval width) := by
  apply thm_poly_polynomial_symmetry (D := degree - 2) (lo := degree - 1)
    (fun rows width => (Ed degree rows width : Rat))
  · intro rows rowsValid
    exact Ed_uniform_onset degreeValid (by omega)
  · intro width widthValid
    obtain ⟨polynomial, bound, evaluations⟩ :=
      Ed_uniform_onset degreeValid (rows := width) (by omega)
    refine ⟨polynomial, bound, ?_⟩
    intro rows rowsValid
    rw [Ed_symm degree (by omega) (by omega)]
    exact evaluations rows rowsValid
  · intro rows width rowsValid widthValid
    exact congrArg (fun count : Nat => (count : Rat)) (Ed_symm degree (by omega) (by omega))

#print axioms Ed_uniform_onset
#print axioms Ed_symmetric_bivariate_upper

end OrigamiCone.Sequel.ConcreteTransfer
