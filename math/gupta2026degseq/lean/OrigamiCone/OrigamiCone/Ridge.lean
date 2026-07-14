import OrigamiCone.Basic

/-!
# The Ridge Lemma, off-ridge half (Section 3, `lem:ridge` case 2)

Formalisation of the **off-ridge** direction of the Ridge Lemma (`lem:ridge`),
which underlies both the degree-4 and degree-5 counts.

The Ridge Lemma characterises the strict local maxima of a cone-pair envelope
`h = min(d(p₁,·), δ + d(p₂,·))` as (1) the doubly-admissible cells on the ridge
`{d(p₁,·) = δ + d(p₂,·)}`, and (2) certain grid corners off the ridge.  The key
structural fact behind case (2) is the paper's sentence:

> only one cone is active … `v` is a strict local maximum of the single distance
> cone `d(p₁,·)` … Every interior cell and every non-corner boundary cell has a
> strictly farther neighbour, so such a maximum is a grid corner.

This file proves exactly that single-cone fact: **a strict local maximum of a
distance cone `d(p,·)` is a grid corner** (`gdist_strictMax_imp_corner`), together
with its converse for a fixed corner (`corner_strictMax_iff` for the origin
corner): the corner `(0,0)` is a strict local maximum of `d(p,·)` iff `p` avoids
both incident sides (`p.1 ≥ 1` and `p.2 ≥ 1`).

The remaining, heavier half of the Ridge Lemma — the on-ridge characterisation of
maxima as the doubly-admissible ridge cells, and the lattice counts of the
degree-4/5 families built on it — is not formalised here.

Results:
* `IsCorner` — the four grid corners;
* `exists_farther_of_interior_row`, `_col` — an interior coordinate yields a
  strictly farther neighbour;
* `gdist_strictMax_imp_corner` — a strict local maximum of `d(p,·)` is a corner;
* `corner_origin_strictMax_iff` — the converse at the origin corner.

No `sorry`.
-/

namespace OrigamiCone

variable {m n : ℕ}

/-- A cell is a **grid corner** when both coordinates are at an extreme. -/
def IsCorner (v : Cell m n) : Prop :=
  (v.1.val = 0 ∨ v.1.val = m - 1) ∧ (v.2.val = 0 ∨ v.2.val = n - 1)

/-- **Interior row gives a farther neighbour.** If `v`'s row is strictly interior
(`0 < i < m−1`), some vertical neighbour is one step farther from `p`. -/
lemma exists_farther_of_interior_row {p v : Cell m n}
    (hi : 0 < v.1.val ∧ v.1.val < m - 1) :
    ∃ u, adj v u ∧ gdist p u = gdist p v + 1 := by
  have h1 := v.1.isLt
  rcases (by omega : p.1.val ≤ v.1.val ∨ v.1.val < p.1.val) with hpv | hpv
  · -- move down a row (increasing index, away from `p`)
    refine ⟨(⟨v.1.val + 1, by omega⟩, v.2), ?_, ?_⟩
    · unfold adj gdist; dsimp only; omega
    · unfold gdist; dsimp only; omega
  · -- move up a row
    refine ⟨(⟨v.1.val - 1, by omega⟩, v.2), ?_, ?_⟩
    · unfold adj gdist; dsimp only; omega
    · unfold gdist; dsimp only; omega

/-- **Interior column gives a farther neighbour.** Dual of
`exists_farther_of_interior_row`. -/
lemma exists_farther_of_interior_col {p v : Cell m n}
    (hj : 0 < v.2.val ∧ v.2.val < n - 1) :
    ∃ u, adj v u ∧ gdist p u = gdist p v + 1 := by
  have h2 := v.2.isLt
  rcases (by omega : p.2.val ≤ v.2.val ∨ v.2.val < p.2.val) with hpv | hpv
  · refine ⟨(v.1, ⟨v.2.val + 1, by omega⟩), ?_, ?_⟩
    · unfold adj gdist; dsimp only; omega
    · unfold gdist; dsimp only; omega
  · refine ⟨(v.1, ⟨v.2.val - 1, by omega⟩), ?_, ?_⟩
    · unfold adj gdist; dsimp only; omega
    · unfold gdist; dsimp only; omega

/-- **Off-ridge half of the Ridge Lemma.** A strict local maximum of a single
distance cone `d(p,·)` is a grid corner: every interior cell and every non-corner
boundary cell has a strictly farther neighbour, so cannot be a maximum. -/
theorem gdist_strictMax_imp_corner {p v : Cell m n}
    (hv : IsStrictLocalMax (gdist p) v) : IsCorner v := by
  by_contra hc
  by_cases hrow : v.1.val = 0 ∨ v.1.val = m - 1
  · by_cases hcol : v.2.val = 0 ∨ v.2.val = n - 1
    · exact hc ⟨hrow, hcol⟩
    · have hji : 0 < v.2.val ∧ v.2.val < n - 1 := by have := v.2.isLt; omega
      obtain ⟨u, hadj, hfar⟩ := exists_farther_of_interior_col (p := p) hji
      have := hv u hadj; omega
  · have hii : 0 < v.1.val ∧ v.1.val < m - 1 := by have := v.1.isLt; omega
    obtain ⟨u, hadj, hfar⟩ := exists_farther_of_interior_row (p := p) hii
    have := hv u hadj; omega

/-- **Converse at the origin corner.** The corner `(0,0)` is a strict local
maximum of the distance cone `d(p,·)` iff `p` avoids both sides incident to it,
i.e. `p` lies in neither row `0` nor column `0` (`1 ≤ p.1` and `1 ≤ p.2`).  This
is the qualifying-corner condition of the Ridge Lemma, case 2, at one corner. -/
lemma corner_origin_strictMax_iff (hm : 2 ≤ m) (hn : 2 ≤ n) (p : Cell m n)
    (v : Cell m n) (hv0 : v.1.val = 0 ∧ v.2.val = 0) :
    IsStrictLocalMax (gdist p) v ↔ 1 ≤ p.1.val ∧ 1 ≤ p.2.val := by
  obtain ⟨hi, hj⟩ := hv0
  constructor
  · intro hmax
    -- the two neighbours `(1,0)` and `(0,1)` must be closer to `p`
    have hu1 : adj v (⟨1, by omega⟩, v.2) := by unfold adj gdist; dsimp only; omega
    have hu2 : adj v (v.1, ⟨1, by omega⟩) := by unfold adj gdist; dsimp only; omega
    have e1 := hmax _ hu1
    have e2 := hmax _ hu2
    unfold gdist at e1 e2
    dsimp only at e1 e2
    omega
  · rintro ⟨hp1, hp2⟩ u hu
    unfold adj gdist at hu
    unfold gdist
    omega

end OrigamiCone
