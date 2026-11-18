import MiuraCounting.ConcreteSingleSideFamilies

namespace OrigamiCone.Sequel.SeparableProducts

open OrigamiCone FrozenExpansion JointReduction JointFibers JointTypePartition SplitPolynomials SignedHeightCounts SingleSideFamilies

abbrev CanonicalWalk (length : Nat) := {walk : Cell 1 length -> Int // IsCanonicalHeight walk}

abbrev CanonicalSeparableHeights (rows width : Nat) :=
  {height : Cell rows width -> Int // IsCanonicalHeight height ∧ AdditivelySeparable height}

def productHeight {rows width : Nat} (rowWalk : CanonicalWalk rows) (columnWalk : CanonicalWalk width) :
    Cell rows width -> Int := fun point => rowWalk.val (0, point.1) + columnWalk.val (0, point.2)

theorem separable_origin_profile {rows width : Nat} (height : Cell rows width -> Int)
    (canonical : IsCanonicalHeight height) (rowsPositive : 0 < rows) (widthPositive : 0 < width)
    (separable : AdditivelySeparable height) :
    ∀ row column, height (row, column) =
      height (row, ⟨0, widthPositive⟩) + height (⟨0, rowsPositive⟩, column) := by
  obtain ⟨rowPart, columnPart, profile⟩ := separable
  have root := canonical.2 (⟨0, rowsPositive⟩, ⟨0, widthPositive⟩) rfl rfl
  have rootProfile := profile ⟨0, rowsPositive⟩ ⟨0, widthPositive⟩
  intro row column
  have rowProfile := profile row ⟨0, widthPositive⟩
  have columnProfile := profile ⟨0, rowsPositive⟩ column
  have value := profile row column
  omega

theorem productHeight_isHeight {rows width : Nat} (rowWalk : CanonicalWalk rows) (columnWalk : CanonicalWalk width) :
    IsHeight (productHeight rowWalk columnWalk) := by
  intro first second near
  by_cases sameRow : first.1 = second.1
  · have columnNear : adj (((0 : Fin 1), first.2) : Cell 1 width) (0, second.2) := by
      simpa [adj, gdist, sameRow] using near
    have step := columnWalk.property.1 _ _ columnNear
    simpa only [productHeight, sameRow, add_sub_add_left_eq_sub] using step
  · have sameColumn : first.2 = second.2 := by
      apply Fin.ext
      have differentRows : first.1.val ≠ second.1.val := fun equality => sameRow (Fin.ext equality)
      simp only [adj, gdist] at near
      omega
    have rowNear : adj (((0 : Fin 1), first.1) : Cell 1 rows) (0, second.1) := by
      simpa [adj, gdist, sameColumn] using near
    have step := rowWalk.property.1 _ _ rowNear
    simpa only [productHeight, sameColumn, add_sub_add_right_eq_sub] using step

theorem productHeight_canonical {rows width : Nat} (rowWalk : CanonicalWalk rows) (columnWalk : CanonicalWalk width) :
    IsCanonicalHeight (productHeight rowWalk columnWalk) := by
  refine ⟨productHeight_isHeight rowWalk columnWalk, ?_⟩
  intro point rowZero columnZero
  have rowRoot := rowWalk.property.2 ((0 : Fin 1), point.1) rfl rowZero
  have columnRoot := columnWalk.property.2 ((0 : Fin 1), point.2) rfl columnZero
  change rowWalk.val (0, point.1) + columnWalk.val (0, point.2) = 0
  rw [rowRoot, columnRoot]
  rfl

def rowRestriction {rows width : Nat} (height : Cell rows width -> Int)
    (canonical : IsCanonicalHeight height) (widthPositive : 0 < width) : CanonicalWalk rows :=
  ⟨fun point => height (point.2, ⟨0, widthPositive⟩),
    LeafProfiles.rowLine_isHeight (swapH height) (swapH_isHeight canonical.1) ⟨0, widthPositive⟩,
    by
      intro point _ rowZero
      exact canonical.2 (point.2, ⟨0, widthPositive⟩) rowZero rfl⟩

def columnRestriction {rows width : Nat} (height : Cell rows width -> Int)
    (canonical : IsCanonicalHeight height) (rowsPositive : 0 < rows) : CanonicalWalk width :=
  ⟨fun point => height (⟨0, rowsPositive⟩, point.2),
    LeafProfiles.rowLine_isHeight height canonical.1 ⟨0, rowsPositive⟩,
    by
      intro point _ columnZero
      exact canonical.2 (⟨0, rowsPositive⟩, point.2) rfl columnZero⟩

def canonicalSeparableEquiv (rows width : Nat) (rowsPositive : 0 < rows) (widthPositive : 0 < width) :
    CanonicalSeparableHeights rows width ≃ (CanonicalWalk rows × CanonicalWalk width) where
  toFun := fun height =>
    (rowRestriction height.val height.property.1 widthPositive,
      columnRestriction height.val height.property.1 rowsPositive)
  invFun := fun walks => ⟨productHeight walks.1 walks.2, productHeight_canonical walks.1 walks.2,
    ⟨fun row => walks.1.val (0, row), fun column => walks.2.val (0, column), fun _ _ => rfl⟩⟩
  left_inv := by
    intro height
    apply Subtype.ext
    funext point
    exact (separable_origin_profile height.val height.property.1 rowsPositive widthPositive
      height.property.2 point.1 point.2).symm
  right_inv := by
    intro walks
    apply Prod.ext
    · apply Subtype.ext
      funext point
      have columnRoot := walks.2.property.2 ((0 : Fin 1), ⟨0, widthPositive⟩) rfl rfl
      change walks.1.val (0, point.2) + walks.2.val (0, ⟨0, widthPositive⟩) = walks.1.val point
      rw [columnRoot, add_zero]
      exact congrArg walks.1.val (Prod.ext (Subsingleton.elim _ _) rfl)
    · apply Subtype.ext
      funext point
      have rowRoot := walks.1.property.2 ((0 : Fin 1), ⟨0, rowsPositive⟩) rfl rfl
      change walks.1.val (0, ⟨0, rowsPositive⟩) + walks.2.val (0, point.2) = walks.2.val point
      rw [rowRoot, zero_add]
      exact congrArg walks.2.val (Prod.ext (Subsingleton.elim _ _) rfl)

theorem pathPoint_eta {length : Nat} (point : Cell 1 length) : ((0 : Fin 1), point.2) = point :=
  Prod.ext (Subsingleton.elim _ _) rfl

theorem product_increments_iff {rows width : Nat} (rowWalk : CanonicalWalk rows) (columnWalk : CanonicalWalk width)
    (point : Cell rows width) (step : Int) :
    (∀ neighbor, adj point neighbor -> productHeight rowWalk columnWalk neighbor = productHeight rowWalk columnWalk point + step) ↔
    (∀ neighbor, adj (((0 : Fin 1), point.1) : Cell 1 rows) neighbor ->
      rowWalk.val neighbor = rowWalk.val (0, point.1) + step) ∧
    (∀ neighbor, adj (((0 : Fin 1), point.2) : Cell 1 width) neighbor ->
      columnWalk.val neighbor = columnWalk.val (0, point.2) + step) := by
  constructor
  · intro increments
    constructor
    · intro neighbor near
      have neighborZero : neighbor.1 = (0 : Fin 1) := Subsingleton.elim _ _
      have gridNear : adj point (neighbor.2, point.2) := by
        simpa [adj, gdist, neighborZero] using near
      have value := increments (neighbor.2, point.2) gridNear
      change rowWalk.val (0, neighbor.2) + columnWalk.val (0, point.2) =
        rowWalk.val (0, point.1) + columnWalk.val (0, point.2) + step at value
      rw [← pathPoint_eta neighbor]
      omega
    · intro neighbor near
      have neighborZero : neighbor.1 = (0 : Fin 1) := Subsingleton.elim _ _
      have gridNear : adj point (point.1, neighbor.2) := by
        simpa [adj, gdist, neighborZero] using near
      have value := increments (point.1, neighbor.2) gridNear
      change rowWalk.val (0, point.1) + columnWalk.val (0, neighbor.2) =
        rowWalk.val (0, point.1) + columnWalk.val (0, point.2) + step at value
      rw [← pathPoint_eta neighbor]
      omega
  · rintro ⟨rowIncrements, columnIncrements⟩ neighbor near
    change rowWalk.val (0, neighbor.1) + columnWalk.val (0, neighbor.2) =
      rowWalk.val (0, point.1) + columnWalk.val (0, point.2) + step
    by_cases sameRow : point.1 = neighbor.1
    · have columnNear : adj (((0 : Fin 1), point.2) : Cell 1 width) (0, neighbor.2) := by
        simpa [adj, gdist, sameRow] using near
      have value := columnIncrements _ columnNear
      rw [sameRow]
      omega
    · have sameColumn : point.2 = neighbor.2 := by
        apply Fin.ext
        have differentRows : point.1.val ≠ neighbor.1.val := fun equality => sameRow (Fin.ext equality)
        simp only [adj, gdist] at near
        omega
      have rowNear : adj (((0 : Fin 1), point.1) : Cell 1 rows) (0, neighbor.1) := by
        simpa [adj, gdist, sameColumn] using near
      have value := rowIncrements _ rowNear
      rw [sameColumn]
      omega

theorem product_kind_iff {rows width : Nat} (rowWalk : CanonicalWalk rows) (columnWalk : CanonicalWalk width)
    (kind : SignedExtrema.Kind) (point : Cell rows width) :
    SignedExtrema.HasKind (productHeight rowWalk columnWalk) kind point ↔
      SignedExtrema.HasKind rowWalk.val kind (0, point.1) ∧
        SignedExtrema.HasKind columnWalk.val kind (0, point.2) := by
  cases kind with
  | maximum =>
    simpa only [SignedExtrema.HasKind, IsStrictLocalMax, sub_eq_add_neg] using
      product_increments_iff rowWalk columnWalk point (-1)
  | minimum => exact product_increments_iff rowWalk columnWalk point 1

def productSignedSiteEquiv {rows width : Nat} (rowWalk : CanonicalWalk rows) (columnWalk : CanonicalWalk width)
    (kind : SignedExtrema.Kind) :
    SignedExtrema.Sites (productHeight rowWalk columnWalk) kind ≃
      (SignedExtrema.Sites rowWalk.val kind × SignedExtrema.Sites columnWalk.val kind) where
  toFun := fun point =>
    (⟨(0, point.val.1), ((product_kind_iff rowWalk columnWalk kind point.val).mp point.property).1⟩,
      ⟨(0, point.val.2), ((product_kind_iff rowWalk columnWalk kind point.val).mp point.property).2⟩)
  invFun := fun points =>
    ⟨(points.1.val.2, points.2.val.2), (product_kind_iff rowWalk columnWalk kind _).mpr
      ⟨by simpa only [pathPoint_eta] using points.1.property,
        by simpa only [pathPoint_eta] using points.2.property⟩⟩
  left_inv := fun _ => Subtype.ext rfl
  right_inv := by
    intro points
    apply Prod.ext
    · exact Subtype.ext (pathPoint_eta points.1.val)
    · exact Subtype.ext (pathPoint_eta points.2.val)

theorem product_signed_count {rows width : Nat} (rowWalk : CanonicalWalk rows) (columnWalk : CanonicalWalk width)
    (kind : SignedExtrema.Kind) :
    SignedExtrema.count (productHeight rowWalk columnWalk) kind =
      SignedExtrema.count rowWalk.val kind * SignedExtrema.count columnWalk.val kind := by
  have counts := Nat.card_congr (productSignedSiteEquiv rowWalk columnWalk kind)
  rw [Nat.card_prod] at counts
  exact counts

#print axioms separable_origin_profile
#print axioms productHeight_isHeight
#print axioms productHeight_canonical
#print axioms canonicalSeparableEquiv
#print axioms product_increments_iff
#print axioms product_kind_iff
#print axioms productSignedSiteEquiv
#print axioms product_signed_count

end OrigamiCone.Sequel.SeparableProducts
