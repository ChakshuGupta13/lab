import MiuraCounting.ConcreteSplitPolynomials

namespace OrigamiCone.Sequel.SignedHeightCounts

open OrigamiCone FrozenExpansion JointReduction JointFibers JointTypePartition SplitPolynomials

abbrev CanonicalSplitHeights (rows width minima maxima : Nat) :=
  {height : Cell rows width -> Int // IsCanonicalHeight height ∧
    SignedExtrema.count height SignedExtrema.Kind.minimum = minima ∧
    SignedExtrema.count height SignedExtrema.Kind.maximum = maxima}

abbrev CanonicalNonseparableSplitHeights (rows width minima maxima : Nat) :=
  {height : Cell rows width -> Int // IsCanonicalHeight height ∧
    SignedExtrema.count height SignedExtrema.Kind.minimum = minima ∧
    SignedExtrema.count height SignedExtrema.Kind.maximum = maxima ∧ ¬ AdditivelySeparable height}

def canonicalSplitEquiv (rows width minima maxima : Nat) (rowsValid : 2 ≤ rows) (widthValid : 2 ≤ width) :
    CanonicalSplitHeights rows width minima maxima ≃
      SplitFamilyStates rows width minima maxima (fun _ => True) where
  toFun := fun height =>
    ⟨⟨rows, rowsValid, ⟨width, widthValid, height.val, height.property.1⟩⟩,
      rfl, rfl, height.property.2, True.intro⟩
  invFun := by
    rintro ⟨⟨sourceRows, sourceRowsValid, ⟨sourceWidth, sourceWidthValid, height, canonical⟩⟩,
      rowEquality, widthEquality, split, _⟩
    cases rowEquality
    cases widthEquality
    exact ⟨height, canonical, split.1, split.2⟩
  left_inv := fun _ => Subtype.ext rfl
  right_inv := by
    rintro ⟨⟨sourceRows, sourceRowsValid, ⟨sourceWidth, sourceWidthValid, height, canonical⟩⟩,
      rowEquality, widthEquality, split, _⟩
    cases rowEquality
    cases widthEquality
    rfl

def canonicalNonseparableSplitEquiv (rows width minima maxima : Nat)
    (rowsValid : 2 ≤ rows) (widthValid : 2 ≤ width) :
    CanonicalNonseparableSplitHeights rows width minima maxima ≃
      SplitFamilyStates rows width minima maxima (fun source => ¬ AdditivelySeparable source.columns.height) where
  toFun := fun height =>
    ⟨⟨rows, rowsValid, ⟨width, widthValid, height.val, height.property.1⟩⟩,
      rfl, rfl, ⟨height.property.2.1, height.property.2.2.1⟩, height.property.2.2.2⟩
  invFun := by
    rintro ⟨⟨sourceRows, sourceRowsValid, ⟨sourceWidth, sourceWidthValid, height, canonical⟩⟩,
      rowEquality, widthEquality, split, nonseparable⟩
    cases rowEquality
    cases widthEquality
    exact ⟨height, canonical, split.1, split.2, nonseparable⟩
  left_inv := fun _ => Subtype.ext rfl
  right_inv := by
    rintro ⟨⟨sourceRows, sourceRowsValid, ⟨sourceWidth, sourceWidthValid, height, canonical⟩⟩,
      rowEquality, widthEquality, split, nonseparable⟩
    cases rowEquality
    cases widthEquality
    rfl

theorem canonicalSplit_finite (rows width minima maxima : Nat) (rowsValid : 2 ≤ rows) (widthValid : 2 ≤ width) :
    Finite (CanonicalSplitHeights rows width minima maxima) := by
  letI : Finite (SplitFamilyStates rows width minima maxima (fun _ => True)) :=
    splitFamilyStates_finite rows width minima maxima _
  exact Finite.of_equiv _ (canonicalSplitEquiv rows width minima maxima rowsValid widthValid).symm

theorem canonicalNonseparableSplit_finite (rows width minima maxima : Nat)
    (rowsValid : 2 ≤ rows) (widthValid : 2 ≤ width) :
    Finite (CanonicalNonseparableSplitHeights rows width minima maxima) := by
  letI : Finite (SplitFamilyStates rows width minima maxima
      (fun source => ¬ AdditivelySeparable source.columns.height)) := splitFamilyStates_finite rows width minima maxima _
  exact Finite.of_equiv _ (canonicalNonseparableSplitEquiv rows width minima maxima rowsValid widthValid).symm

theorem canonicalSplit_eventually_polynomial (minima maxima : Nat) (degreeBound : 3 ≤ minima + maxima) :
    ∃ polynomial : MvPolynomial (Fin 2) Rat,
      (∀ coordinate, polynomial.degreeOf coordinate ≤ minima + maxima - 2) ∧
      ∀ rows width : Nat, 2 * (minima + maxima) - 1 ≤ rows ->
        2 * (minima + maxima) - 1 ≤ width ->
          evalAt polynomial rows width = (Nat.card (CanonicalSplitHeights rows width minima maxima) : Rat) := by
  obtain ⟨polynomial, degrees, agreement⟩ := splitFamily_eventually_polynomial minima maxima degreeBound
    (fun _ => True) (fun _ _ => Iff.rfl)
  refine ⟨polynomial, degrees, ?_⟩
  intro rows width rowBound columnBound
  have counts := Nat.card_congr (canonicalSplitEquiv rows width minima maxima (by omega) (by omega))
  rw [counts]
  exact agreement rows width rowBound columnBound

theorem canonicalNonseparableSplit_eventually_polynomial (minima maxima : Nat)
    (degreeBound : 5 ≤ minima + maxima) :
    ∃ polynomial : MvPolynomial (Fin 2) Rat,
      polynomial.totalDegree ≤ minima + maxima - 3 ∧
      ∀ rows width : Nat, 2 * (minima + maxima) - 1 ≤ rows ->
        2 * (minima + maxima) - 1 ≤ width ->
          evalAt polynomial rows width =
            (Nat.card (CanonicalNonseparableSplitHeights rows width minima maxima) : Rat) := by
  obtain ⟨polynomial, degree, agreement⟩ := nonseparableSplit_eventually_polynomial minima maxima degreeBound
  refine ⟨polynomial, degree, ?_⟩
  intro rows width rowBound columnBound
  have counts := Nat.card_congr (canonicalNonseparableSplitEquiv rows width minima maxima (by omega) (by omega))
  rw [counts]
  exact agreement rows width rowBound columnBound

#print axioms canonicalSplitEquiv
#print axioms canonicalNonseparableSplitEquiv
#print axioms canonicalSplit_finite
#print axioms canonicalNonseparableSplit_finite
#print axioms canonicalSplit_eventually_polynomial
#print axioms canonicalNonseparableSplit_eventually_polynomial

end OrigamiCone.Sequel.SignedHeightCounts
