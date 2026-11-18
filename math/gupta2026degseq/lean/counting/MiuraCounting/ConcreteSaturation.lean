import MiuraCounting.ConcreteJointNormalForm

namespace OrigamiCone.Sequel.Saturation

open OrigamiCone FrozenExpansion

theorem height_distance_bound {rows width : Nat} (height : Cell rows width -> Int)
    (heightValid : IsHeight height) (first second : Cell rows width) :
    |height first - height second| ≤ gdist first second :=
  height_lipschitz heightValid (gdist first second).toNat first second
    (le_of_eq (Int.toNat_of_nonneg (gdist_nonneg first second)).symm)

theorem between_saturated_rise {rows width : Nat} (height : Cell rows width -> Int)
    (heightValid : IsHeight height) (first last point : Cell rows width)
    (saturated : height last - height first = gdist first last)
    (between : gdist first point + gdist point last = gdist first last) :
    height point = height first + gdist first point := by
  have firstBound := height_distance_bound height heightValid first point
  have lastBound := height_distance_bound height heightValid point last
  have firstSide := (abs_le.mp firstBound).1
  have lastSide := (abs_le.mp lastBound).1
  omega

theorem between_saturated_fall {rows width : Nat} (height : Cell rows width -> Int)
    (heightValid : IsHeight height) (first last point : Cell rows width)
    (saturated : height first - height last = gdist first last)
    (between : gdist first point + gdist point last = gdist first last) :
    height point = height first - gdist first point := by
  have firstBound := height_distance_bound height heightValid first point
  have lastBound := height_distance_bound height heightValid point last
  have firstSide := (abs_le.mp firstBound).2
  have lastSide := (abs_le.mp lastBound).2
  omega

theorem rectangle_geodesic {rows width : Nat} (first last point : Cell rows width)
    (rowBetween : min first.1.val last.1.val ≤ point.1.val ∧ point.1.val ≤ max first.1.val last.1.val)
    (columnBetween : min first.2.val last.2.val ≤ point.2.val ∧ point.2.val ≤ max first.2.val last.2.val) :
    gdist first point + gdist point last = gdist first last := by
  unfold gdist
  omega

theorem opposite_corner_profile {rows width : Nat} (height : Cell rows width -> Int)
    (heightValid : IsHeight height) (rowsPositive : 0 < rows) (widthPositive : 0 < width)
    (saturated : height (⟨rows - 1, by omega⟩, ⟨width - 1, by omega⟩) -
      height (⟨0, rowsPositive⟩, ⟨0, widthPositive⟩) =
        gdist (⟨0, rowsPositive⟩, ⟨0, widthPositive⟩)
          (⟨rows - 1, by omega⟩, ⟨width - 1, by omega⟩)) :
    ∀ point : Cell rows width,
      height point = height (⟨0, rowsPositive⟩, ⟨0, widthPositive⟩) + point.1.val + point.2.val := by
  intro point
  have between : gdist (⟨0, rowsPositive⟩, ⟨0, widthPositive⟩) point +
      gdist point (⟨rows - 1, by omega⟩, ⟨width - 1, by omega⟩) =
      gdist (⟨0, rowsPositive⟩, ⟨0, widthPositive⟩)
        (⟨rows - 1, by omega⟩, ⟨width - 1, by omega⟩) := by
    unfold gdist
    dsimp only []
    have := point.1.isLt
    have := point.2.isLt
    omega
  have profile := between_saturated_rise height heightValid _ _ point saturated between
  have distance : gdist (⟨0, rowsPositive⟩, ⟨0, widthPositive⟩) point =
      (point.1.val : Int) + point.2.val := by
    unfold gdist
    dsimp only []
    omega
  rw [distance] at profile
  omega

theorem opposite_corner_separable {rows width : Nat} (height : Cell rows width -> Int)
    (heightValid : IsHeight height) (rowsPositive : 0 < rows) (widthPositive : 0 < width)
    (saturated : height (⟨rows - 1, by omega⟩, ⟨width - 1, by omega⟩) -
      height (⟨0, rowsPositive⟩, ⟨0, widthPositive⟩) =
        gdist (⟨0, rowsPositive⟩, ⟨0, widthPositive⟩)
          (⟨rows - 1, by omega⟩, ⟨width - 1, by omega⟩)) :
    AdditivelySeparable height := by
  refine ⟨fun row => height (⟨0, rowsPositive⟩, ⟨0, widthPositive⟩) + row.val,
    fun column => (column.val : Int), ?_⟩
  intro row column
  exact opposite_corner_profile height heightValid rowsPositive widthPositive saturated (row, column)

theorem between_saturated_signed {rows width : Nat} (height : Cell rows width -> Int)
    (heightValid : IsHeight height) (first last point : Cell rows width) (sign : Int)
    (unitSign : sign = 1 ∨ sign = -1)
    (saturated : height last - height first = sign * gdist first last)
    (between : gdist first point + gdist point last = gdist first last) :
    height point = height first + sign * gdist first point := by
  rcases unitSign with rfl | rfl
  · simp only [one_mul] at saturated ⊢
    exact between_saturated_rise height heightValid first last point saturated between
  · simp only [neg_one_mul] at saturated ⊢
    have falling : height first - height last = gdist first last := by omega
    simpa only [sub_eq_add_neg] using between_saturated_fall height heightValid first last point falling between

theorem distance_cone_separable {rows width : Nat} (height : Cell rows width -> Int)
    (apex : Cell rows width) (sign : Int)
    (profile : ∀ point : Cell rows width, height point = height apex + sign * gdist apex point) :
    AdditivelySeparable height := by
  refine ⟨fun row => height apex + sign * |(row.val : Int) - apex.1.val|,
    fun column => sign * |(column.val : Int) - apex.2.val|, ?_⟩
  intro row column
  have distance : gdist apex (row, column) =
      |(row.val : Int) - apex.1.val| + |(column.val : Int) - apex.2.val| := by
    simp [gdist, abs_sub_comm]
  rw [profile (row, column), distance]
  ring

theorem opposite_endpoints_separable {rows width : Nat} (height : Cell rows width -> Int)
    (heightValid : IsHeight height) (first last : Cell rows width) (sign : Int)
    (unitSign : sign = 1 ∨ sign = -1)
    (saturated : height last - height first = sign * gdist first last)
    (rowCover : min first.1.val last.1.val = 0 ∧ max first.1.val last.1.val + 1 = rows)
    (columnCover : min first.2.val last.2.val = 0 ∧ max first.2.val last.2.val + 1 = width) :
    AdditivelySeparable height := by
  apply distance_cone_separable height first sign
  intro point
  apply between_saturated_signed height heightValid first last point sign unitSign saturated
  apply rectangle_geodesic
  · have within := point.1.isLt
    omega
  · have within := point.2.isLt
    omega

#print axioms height_distance_bound
#print axioms between_saturated_rise
#print axioms between_saturated_fall
#print axioms rectangle_geodesic
#print axioms opposite_corner_profile
#print axioms opposite_corner_separable
#print axioms between_saturated_signed
#print axioms distance_cone_separable
#print axioms opposite_endpoints_separable

end OrigamiCone.Sequel.Saturation
