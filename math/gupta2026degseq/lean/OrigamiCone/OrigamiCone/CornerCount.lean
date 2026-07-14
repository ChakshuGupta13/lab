import OrigamiCone.BoundaryNonCornerCount

/-!
# Corner cell count: `4`

The lattice count underlying **Theorem 3.2** (`thm:deg2`, degree-2 count = 4):

  `|\{q : Cell\,m\,n \mid \mathrm{IsGridCorner}\,q\}| = 4`,

for every `m, n ≥ 2`.  Four corners: `(0, 0)`, `(0, n − 1)`, `(m − 1, 0)`,
`(m − 1, n − 1)`.

The paper's `thm:deg2` proof:

> *"For `m, n ≥ 2`, the graph `OFG(M_{m,n})` has exactly four vertices of
>  degree 2, the corner gradients."*

This module formalises the lattice cardinality `4` that any degree-2 count
theorem needs — paralleling `card_boundaryNonCorner = 2(m + n − 4)` in
`BoundaryNonCornerCount.lean`.  The remaining bridge (each corner gives one
shift-class of degree-2 height functions) is the next unit of work toward
formalising `thm:deg2`.

Proof: rewrite `IsGridCorner` as a product filter via `Finset.product`, apply
`Finset.card_product`, and substitute `card_isEndpoint_fin = 2` twice.

Results:
* `IsGridCorner` — both coordinates are path endpoints.
* `card_isGridCorner` — `|\{q : Cell\,m\,n \mid \mathrm{IsGridCorner}\,q\}| = 4`.

No `sorry`.
-/

namespace OrigamiCone

variable {m n : ℕ}

/-- **Corner predicate**: both coordinates of `q` are path endpoints
(`IsEndpoint q.1 ∧ IsEndpoint q.2`).  The corners of `Cell m n` are
`(0, 0)`, `(0, n − 1)`, `(m − 1, 0)`, `(m − 1, n − 1)` — four cells for
`m, n ≥ 2` (degenerate when `m = 1` or `n = 1`). -/
def IsGridCorner (q : Cell m n) : Prop := IsEndpoint q.1 ∧ IsEndpoint q.2

/-- `IsGridCorner q` is decidable (conjunction of decidable `IsEndpoint`s). -/
instance {m n : ℕ} (q : Cell m n) : Decidable (IsGridCorner q) :=
  show Decidable (IsEndpoint q.1 ∧ IsEndpoint q.2) from inferInstance

/-! ## The main count -/

/-- **Lattice count of corner cells** in `Cell m n`.  For `m, n ≥ 2`, the
number of cells `q : Cell m n` satisfying `IsGridCorner q` is `4`.

This is the cardinality the paper's proof of `thm:deg2` invokes: there are
exactly four corner cells of the `m × n` grid, hence (after the cone-at-corner
bridge, deferred) exactly four degree-2 OFG vertices. -/
theorem card_isGridCorner (hm : 2 ≤ m) (hn : 2 ≤ n) :
    (Finset.univ.filter (IsGridCorner : Cell m n → Prop)).card = 4 := by
  -- Rewrite the filter as a product of two 1-D `IsEndpoint` filters.
  rw [show Finset.univ.filter (IsGridCorner : Cell m n → Prop)
        = (Finset.univ.filter (IsEndpoint : Fin m → Prop))
              ×ˢ (Finset.univ.filter (IsEndpoint : Fin n → Prop)) from by
      ext ⟨a, b⟩
      simp [Finset.mem_filter, Finset.mem_product, IsGridCorner]]
  -- Cardinality of the product = product of cardinalities.
  rw [Finset.card_product, card_isEndpoint_fin m hm, card_isEndpoint_fin n hn]

end OrigamiCone
