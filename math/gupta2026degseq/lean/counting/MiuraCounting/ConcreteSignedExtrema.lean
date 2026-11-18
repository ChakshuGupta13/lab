import MiuraCounting.ConcreteJointNormalForm

namespace OrigamiCone.Sequel.SignedExtrema

open OrigamiCone FrozenExpansion JointReduction

inductive Kind where
  | maximum
  | minimum

def HasKind {rows width : Nat} (height : Cell rows width -> Int) (kind : Kind) (point : Cell rows width) : Prop :=
  match kind with
  | Kind.maximum => IsStrictLocalMax height point
  | Kind.minimum => IsStrictLocalMin height point

abbrev Sites {rows width : Nat} (height : Cell rows width -> Int) (kind : Kind) :=
  {point : Cell rows width // HasKind height kind point}

noncomputable def count {rows width : Nat} (height : Cell rows width -> Int) (kind : Kind) : Nat :=
  Nat.card (Sites height kind)

theorem contract_kind_iff {rows width : Nat} (height : Cell rows width -> Int)
    (heightValid : IsHeight height) (rowsPositive : 0 < rows)
    (column : Fin width) (frozen : frozenColumn height column) (kind : Kind) (point : Cell rows (width - 1)) :
    HasKind (contractAt height rowsPositive column frozen) kind point ↔
      HasKind height kind (cellExpand column point frozen.1 frozen.2.1) := by
  cases kind with
  | maximum => exact ⟨h_max_of_contractAt_isSLM height heightValid rowsPositive column frozen point,
      contractAt_isSLM_of_h_max height heightValid rowsPositive column frozen point⟩
  | minimum => exact ⟨h_min_of_contractAt_isSLm height heightValid rowsPositive column frozen point,
      contractAt_isSLm_of_h_min height heightValid rowsPositive column frozen point⟩

noncomputable def contractSignedEquiv {rows width : Nat} (height : Cell rows width -> Int)
    (heightValid : IsHeight height) (rowsPositive : 0 < rows)
    (column : Fin width) (frozen : frozenColumn height column) (kind : Kind) :
    Sites (contractAt height rowsPositive column frozen) kind ≃ Sites height kind :=
  Equiv.ofBijective
    (fun point => ⟨cellExpand column point.val frozen.1 frozen.2.1,
      (contract_kind_iff height heightValid rowsPositive column frozen kind point.val).mp point.property⟩)
    (by
      constructor
      · intro first second equality
        apply Subtype.ext
        exact cellExpand_injective column frozen.1 frozen.2.1 (congrArg Subtype.val equality)
      · intro point
        have notDeleted : point.val.2 ≠ column := by
          intro equality
          have extremum : IsStrictLocalExtremum height point.val := by
            cases kind with
            | maximum => exact Or.inl point.property
            | minimum => exact Or.inr point.property
          apply frozenColumn_not_active height column frozen
          refine ⟨point.val.1, ?_⟩
          have samePoint : (point.val.1, column) = point.val := Prod.ext rfl equality.symm
          rw [samePoint]
          exact extremum
        obtain ⟨original, equality⟩ := cellExpand_surjOn column frozen.1 frozen.2.1 point.val notDeleted
        refine ⟨⟨original, ?_⟩, ?_⟩
        · apply (contract_kind_iff height heightValid rowsPositive column frozen kind original).mpr
          rw [equality]
          exact point.property
        · exact Subtype.ext equality)

noncomputable def transposeSignedEquiv {rows width : Nat} (height : Cell rows width -> Int) (kind : Kind) :
    Sites (swapH height) kind ≃ Sites height kind :=
  (Equiv.prodComm (Fin width) (Fin rows)).subtypeEquiv (by
    intro point
    cases kind with
    | maximum => exact swapH_strictMax_iff (h := height) point
    | minimum => exact swapH_strictMin_iff (h := height) point)

theorem columnStep_equiv {source target : GridState} (step : ColumnStep source target) (kind : Kind) :
    Nonempty (Sites target.columns.height kind ≃ Sites source.columns.height kind) := by
  obtain ⟨next, ⟨column, frozen, nextFrozen, rfl⟩, rfl⟩ := step
  exact ⟨contractSignedEquiv source.columns.height source.columns.canonical.1 (rows_positive source)
    column frozen kind⟩

theorem rowStep_equiv {source target : GridState} (step : RowStep source target) (kind : Kind) :
    Nonempty (Sites target.columns.height kind ≃ Sites source.columns.height kind) := by
  obtain ⟨equivalence⟩ := columnStep_equiv step kind
  exact ⟨(transposeSignedEquiv target.columns.height kind).symm.trans
    (equivalence.trans (transposeSignedEquiv source.columns.height kind))⟩

theorem step_equiv {source target : GridState} (step : Step source target) (kind : Kind) :
    Nonempty (Sites target.columns.height kind ≃ Sites source.columns.height kind) := by
  rcases step with column | row
  · exact columnStep_equiv column kind
  · exact rowStep_equiv row kind

theorem reduction_equiv {source target : GridState} (path : Relation.ReflTransGen Step source target) (kind : Kind) :
    Nonempty (Sites target.columns.height kind ≃ Sites source.columns.height kind) := by
  induction path with
  | refl => exact ⟨Equiv.refl _⟩
  | tail _ step inductionHypothesis =>
    obtain ⟨previous⟩ := inductionHypothesis
    obtain ⟨last⟩ := step_equiv step kind
    exact ⟨last.trans previous⟩

theorem normalForm_equiv (source : GridState) (kind : Kind) :
    Nonempty (Sites (normalForm source).columns.height kind ≃ Sites source.columns.height kind) :=
  reduction_equiv (normalForm_path source) kind

theorem normalForm_count (source : GridState) (kind : Kind) :
    count (normalForm source).columns.height kind = count source.columns.height kind := by
  obtain ⟨equivalence⟩ := normalForm_equiv source kind
  exact Nat.card_congr equivalence

#print axioms contract_kind_iff
#print axioms contractSignedEquiv
#print axioms transposeSignedEquiv
#print axioms columnStep_equiv
#print axioms rowStep_equiv
#print axioms step_equiv
#print axioms reduction_equiv
#print axioms normalForm_equiv
#print axioms normalForm_count

end OrigamiCone.Sequel.SignedExtrema
