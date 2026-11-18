import OrigamiCone.BoundaryNonCornerCount

namespace OrigamiCone

variable {m n : ℕ}

def IsGridCorner (q : Cell m n) : Prop := IsEndpoint q.1 ∧ IsEndpoint q.2

instance {m n : ℕ} (q : Cell m n) : Decidable (IsGridCorner q) :=
  show Decidable (IsEndpoint q.1 ∧ IsEndpoint q.2) from inferInstance

theorem card_isGridCorner (hm : 2 ≤ m) (hn : 2 ≤ n) :
    (Finset.univ.filter (IsGridCorner : Cell m n → Prop)).card = 4 := by

  rw [show Finset.univ.filter (IsGridCorner : Cell m n → Prop)
        = (Finset.univ.filter (IsEndpoint : Fin m → Prop))
              ×ˢ (Finset.univ.filter (IsEndpoint : Fin n → Prop)) from by
      ext ⟨a, b⟩
      simp [Finset.mem_filter, Finset.mem_product, IsGridCorner]]

  rw [Finset.card_product, card_isEndpoint_fin m hm, card_isEndpoint_fin n hn]

end OrigamiCone
