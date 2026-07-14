import OrigamiCone.ConePair
import OrigamiCone.Ridge

/-!
# The Ridge Lemma (Section 3, `lem:ridge`), full maxima characterisation

Formalisation of the **Ridge Lemma** (`lem:ridge`), end-to-end: a complete
characterisation of the strict local maxima of a cone-pair envelope
`h = min(d(p₁,·), δ + d(p₂,·))`.  The paper states that, with apexes differing in
both coordinates, the strict local maxima of `h` are exactly

1. the **doubly admissible** cells on the ridge `{d(p₁,·) = δ + d(p₂,·)}` (a row
   is admissible if it is a boundary row or lies strictly between the apex rows,
   likewise for columns), and
2. the grid corners off the ridge whose **active** apex (the one with the smaller
   cone value there) avoids both sides incident to the corner.

The development proceeds in layers:

* `ridge_strictMax_iff_closer` — parity core: a ridge cell is a maximum iff every
  neighbour is strictly closer to at least one apex (no parity hypothesis; each
  cone changes by `±1` across an edge).
* `ridge_strictMax_iff_admissible` — the on-ridge half (class 1): a ridge cell is
  a maximum iff doubly admissible (apexes differing in both coordinates).
* `corner_strictMax_iff`, `gdist_strictMax_iff` — single distance cone: a strict
  local maximum of `d(p,·)` is exactly a corner whose incident sides `p` avoids.
* `cpe_strictMax_iff_cone_left` / `_right` — off the ridge `cpe` agrees locally
  with the strictly smaller cone (the parity condition forces a gap of at least
  two), so a `cpe` maximum off the ridge is a single-cone maximum.
* `cpe_strictMax_imp` — the forward direction, end-to-end (every maximum is a
  doubly admissible ridge cell or a corner).
* `cpe_strictMax_iff` — the **full Ridge Lemma**: a cell is a maximum iff it is a
  doubly admissible ridge cell or a `QualifyingCorner` off the ridge.

The on-ridge and corner characterisations both turn out to need no `−D < δ < D`
“both apexes active” hypothesis, so the formal statement is strictly stronger than
the paper's.  Only the lattice **counts** of the degree-4/5 families built on top
of the Ridge Lemma (Theorems `thm:deg4count`, `thm:deg5count`) are not formalised
here.

No `sorry`.
-/

namespace OrigamiCone

variable {m n : ℕ}

/-- A cell lies on the **ridge** of the cone pair when the two cones agree:
`d(p₁,v) = δ + d(p₂,v)`. -/
def OnRidge (p₁ p₂ : Cell m n) (δ : ℤ) (v : Cell m n) : Prop :=
  gdist p₁ v = δ + gdist p₂ v

/-- On the ridge, the envelope value is the common cone value `d(p₁,v)`. -/
lemma cpe_on_ridge {p₁ p₂ : Cell m n} {δ : ℤ} {v : Cell m n}
    (hv : OnRidge p₁ p₂ δ v) : cpe p₁ p₂ δ v = gdist p₁ v := by
  unfold cpe
  unfold OnRidge at hv
  rw [← hv, min_self]

/-- **Ridge local-maximum characterisation** (parity core of `lem:ridge`, case 1).
For a ridge cell `v`, the cone-pair envelope `cpe` has a strict local maximum at
`v` iff every grid neighbour is strictly closer to at least one apex.  No parity
hypothesis is needed: each distance cone changes by exactly one across an edge,
so the envelope value at a neighbour is the common ridge value minus one exactly
when some cone decreases there. -/
lemma ridge_strictMax_iff_closer {p₁ p₂ : Cell m n} {δ : ℤ} {v : Cell m n}
    (hv : OnRidge p₁ p₂ δ v) :
    IsStrictLocalMax (cpe p₁ p₂ δ) v ↔
      ∀ u, adj v u → gdist p₁ u < gdist p₁ v ∨ gdist p₂ u < gdist p₂ v := by
  have hcv : cpe p₁ p₂ δ v = gdist p₁ v := cpe_on_ridge hv
  unfold OnRidge at hv
  constructor
  · intro hmax u huv
    have hu := hmax u huv
    rw [hcv] at hu
    have s1 := gdist_adj_step (q := p₁) huv
    have s2 := gdist_adj_step (q := p₂) huv
    unfold cpe at hu
    rcases le_total (gdist p₁ u) (δ + gdist p₂ u) with hm | hm
    · rw [min_eq_left hm] at hu; omega
    · rw [min_eq_right hm] at hu; omega
  · intro hclose u huv
    have hc := hclose u huv
    have s1 := gdist_adj_step (q := p₁) huv
    have s2 := gdist_adj_step (q := p₂) huv
    show cpe p₁ p₂ δ u = cpe p₁ p₂ δ v - 1
    rw [hcv]
    unfold cpe
    rcases le_total (gdist p₁ u) (δ + gdist p₂ u) with hm | hm
    · rw [min_eq_left hm]; omega
    · rw [min_eq_right hm]; omega

/-- **Row admissibility.** A row is admissible if it is a boundary row or lies
strictly between the two apex rows. -/
def RowAdm (p₁ p₂ v : Cell m n) : Prop :=
  v.1.val = 0 ∨ v.1.val = m - 1 ∨
    (min p₁.1.val p₂.1.val < v.1.val ∧ v.1.val < max p₁.1.val p₂.1.val)

/-- **Column admissibility.** Dual of `RowAdm` for columns. -/
def ColAdm (p₁ p₂ v : Cell m n) : Prop :=
  v.2.val = 0 ∨ v.2.val = n - 1 ∨
    (min p₁.2.val p₂.2.val < v.2.val ∧ v.2.val < max p₁.2.val p₂.2.val)

/-- A grid neighbour differs from `v` in exactly one coordinate by one. -/
private lemma adj_split {v u : Cell m n} (h : adj v u) :
    (u.1 = v.1 ∧ (u.2.val = v.2.val + 1 ∨ u.2.val + 1 = v.2.val)) ∨
    (u.2 = v.2 ∧ (u.1.val = v.1.val + 1 ∨ u.1.val + 1 = v.1.val)) := by
  have hn : ((v.1.val : ℤ) - u.1.val).natAbs + ((v.2.val : ℤ) - u.2.val).natAbs = 1 := by
    have h' := h; unfold adj gdist at h'; exact_mod_cast h'
  by_cases hr : u.1.val = v.1.val
  · left; exact ⟨Fin.ext hr, by omega⟩
  · right; exact ⟨Fin.ext (by omega), by omega⟩

/-- **Ridge maxima are the doubly admissible cells** (`lem:ridge`, case 1).
For a ridge cell `v`, when the two apexes differ in both coordinates, the
cone-pair envelope has a strict local maximum at `v` iff `v`'s row and column are
both admissible.  This is the on-ridge half of the Ridge Lemma; the off-ridge
maxima are the grid corners of `Ridge.gdist_strictMax_imp_corner`. -/
theorem ridge_strictMax_iff_admissible {p₁ p₂ : Cell m n} {δ : ℤ}
    (hr : p₁.1.val ≠ p₂.1.val) (hs : p₁.2.val ≠ p₂.2.val)
    {v : Cell m n} (hv : OnRidge p₁ p₂ δ v) :
    IsStrictLocalMax (cpe p₁ p₂ δ) v ↔ RowAdm p₁ p₂ v ∧ ColAdm p₁ p₂ v := by
  have hp1 := p₁.1.isLt; have hp2 := p₂.1.isLt
  have hq1 := p₁.2.isLt; have hq2 := p₂.2.isLt
  have hvi := v.1.isLt; have hvj := v.2.isLt
  rw [ridge_strictMax_iff_closer hv]
  constructor
  · intro hclose
    refine ⟨?_, ?_⟩
    · -- RowAdm: use the vertical neighbours
      rcases (by omega : v.1.val = 0 ∨ v.1.val = m - 1 ∨
          (0 < v.1.val ∧ v.1.val < m - 1)) with h | h | h
      · exact Or.inl h
      · exact Or.inr (Or.inl h)
      · refine Or.inr (Or.inr ⟨?_, ?_⟩)
        · have hb : v.1.val - 1 < m := by omega
          have hadj : adj v (⟨v.1.val - 1, hb⟩, v.2) := by
            unfold adj gdist; dsimp only; omega
          have hcl := hclose (⟨v.1.val - 1, hb⟩, v.2) hadj
          unfold gdist at hcl; dsimp only at hcl; omega
        · have hb : v.1.val + 1 < m := by omega
          have hadj : adj v (⟨v.1.val + 1, hb⟩, v.2) := by
            unfold adj gdist; dsimp only; omega
          have hcl := hclose (⟨v.1.val + 1, hb⟩, v.2) hadj
          unfold gdist at hcl; dsimp only at hcl; omega
    · -- ColAdm: use the horizontal neighbours
      rcases (by omega : v.2.val = 0 ∨ v.2.val = n - 1 ∨
          (0 < v.2.val ∧ v.2.val < n - 1)) with h | h | h
      · exact Or.inl h
      · exact Or.inr (Or.inl h)
      · refine Or.inr (Or.inr ⟨?_, ?_⟩)
        · have hb : v.2.val - 1 < n := by omega
          have hadj : adj v (v.1, ⟨v.2.val - 1, hb⟩) := by
            unfold adj gdist; dsimp only; omega
          have hcl := hclose (v.1, ⟨v.2.val - 1, hb⟩) hadj
          unfold gdist at hcl; dsimp only at hcl; omega
        · have hb : v.2.val + 1 < n := by omega
          have hadj : adj v (v.1, ⟨v.2.val + 1, hb⟩) := by
            unfold adj gdist; dsimp only; omega
          have hcl := hclose (v.1, ⟨v.2.val + 1, hb⟩) hadj
          unfold gdist at hcl; dsimp only at hcl; omega
  · rintro ⟨hrow, hcol⟩ u huv
    rcases adj_split huv with ⟨he, hjj⟩ | ⟨he, hii⟩
    · -- column-changing neighbour: closeness governed by ColAdm
      have hev : u.1.val = v.1.val := by rw [he]
      have hui := u.2.isLt
      unfold RowAdm at hrow; unfold ColAdm at hcol
      unfold gdist; omega
    · -- row-changing neighbour: closeness governed by RowAdm
      have hev : u.2.val = v.2.val := by rw [he]
      have hui := u.1.isLt
      unfold RowAdm at hrow; unfold ColAdm at hcol
      unfold gdist; omega

/-- **Off-ridge bridge.** Off the ridge, under the parity condition on `δ`, the
envelope agrees locally with the strictly smaller cone (the two cones differ by an
even amount, hence by at least two, so the smaller stays the minimum across every
edge).  A strict local maximum of `cpe` off the ridge is therefore a strict local
maximum of a single distance cone, hence a grid corner by
`gdist_strictMax_imp_corner`. -/
lemma cpe_strictMax_off_ridge_imp_corner {p₁ p₂ : Cell m n} {δ : ℤ}
    (hδ : (δ - gdist p₁ p₂) % 2 = 0) {v : Cell m n}
    (hoff : ¬ OnRidge p₁ p₂ δ v)
    (hmax : IsStrictLocalMax (cpe p₁ p₂ δ) v) : IsCorner v := by
  have hpar := cone_pair_same_parity hδ v
  unfold OnRidge at hoff
  rcases (by omega : gdist p₁ v < δ + gdist p₂ v ∨ gdist p₁ v > δ + gdist p₂ v)
      with hlt | hgt
  · -- cone₁ is strictly smaller at `v` (gap ≥ 2 by parity)
    refine gdist_strictMax_imp_corner (p := p₁) ?_
    intro u huv
    have hmu := hmax u huv
    have s1 := gdist_adj_step (q := p₁) huv
    have s2 := gdist_adj_step (q := p₂) huv
    unfold cpe at hmu
    omega
  · -- cone₂ is strictly smaller at `v`
    refine gdist_strictMax_imp_corner (p := p₂) ?_
    intro u huv
    have hmu := hmax u huv
    have s1 := gdist_adj_step (q := p₁) huv
    have s2 := gdist_adj_step (q := p₂) huv
    unfold cpe at hmu
    omega

/-- **Forward direction of the Ridge Lemma**, end-to-end (parity-correct `δ`,
apexes differing in both coordinates).  Every strict local maximum of the
cone-pair envelope is either a doubly admissible cell on the ridge or a grid
corner.  This composes the on-ridge characterisation
(`ridge_strictMax_iff_admissible`) with the off-ridge bridge
(`cpe_strictMax_off_ridge_imp_corner`). -/
theorem cpe_strictMax_imp {p₁ p₂ : Cell m n} {δ : ℤ}
    (hδ : (δ - gdist p₁ p₂) % 2 = 0)
    (hr : p₁.1.val ≠ p₂.1.val) (hs : p₁.2.val ≠ p₂.2.val)
    {v : Cell m n} (hmax : IsStrictLocalMax (cpe p₁ p₂ δ) v) :
    (OnRidge p₁ p₂ δ v ∧ RowAdm p₁ p₂ v ∧ ColAdm p₁ p₂ v) ∨ IsCorner v := by
  by_cases hon : OnRidge p₁ p₂ δ v
  · exact Or.inl ⟨hon, (ridge_strictMax_iff_admissible hr hs hon).mp hmax⟩
  · exact Or.inr (cpe_strictMax_off_ridge_imp_corner hδ hon hmax)

/-- **Single-cone strict maxima at a corner.**  For `m, n ≥ 2` and a grid corner
`v`, the distance cone `d(p,·)` has a strict local maximum at `v` iff `p` avoids
both sides incident to `v` — that is, `p` differs from `v` in each coordinate
(equivalently, `p` is on the strictly interior side of each of `v`'s two extreme
coordinates).  Generalises `Ridge.corner_origin_strictMax_iff` to all four
corners. -/
lemma corner_strictMax_iff (hm : 2 ≤ m) (hn : 2 ≤ n) {p v : Cell m n}
    (hc : IsCorner v) :
    IsStrictLocalMax (gdist p) v ↔ p.1.val ≠ v.1.val ∧ p.2.val ≠ v.2.val := by
  obtain ⟨hrow, hcol⟩ := hc
  have hp1 := p.1.isLt; have hp2 := p.2.isLt
  have hv1 := v.1.isLt; have hv2 := v.2.isLt
  constructor
  · intro hmax
    refine ⟨?_, ?_⟩
    · rcases hrow with h0 | h1
      · have hadj : adj v (⟨1, hm⟩, v.2) := by unfold adj gdist; dsimp only; omega
        have hcl := hmax _ hadj
        unfold gdist at hcl; dsimp only at hcl; omega
      · have hadj : adj v (⟨m - 2, by omega⟩, v.2) := by unfold adj gdist; dsimp only; omega
        have hcl := hmax _ hadj
        unfold gdist at hcl; dsimp only at hcl; omega
    · rcases hcol with h0 | h1
      · have hadj : adj v (v.1, ⟨1, hn⟩) := by unfold adj gdist; dsimp only; omega
        have hcl := hmax _ hadj
        unfold gdist at hcl; dsimp only at hcl; omega
      · have hadj : adj v (v.1, ⟨n - 2, by omega⟩) := by unfold adj gdist; dsimp only; omega
        have hcl := hmax _ hadj
        unfold gdist at hcl; dsimp only at hcl; omega
  · rintro ⟨hpr, hpc⟩ u huv
    rcases adj_split huv with ⟨he, hjj⟩ | ⟨he, hii⟩
    · have hev : u.1.val = v.1.val := by rw [he]
      have hui := u.2.isLt
      unfold gdist; omega
    · have hev : u.2.val = v.2.val := by rw [he]
      have hui := u.1.isLt
      unfold gdist; omega

/-- **Single-cone strict maxima, full characterisation.**  A distance cone has a
strict local maximum at `v` iff `v` is a corner whose incident sides `p` avoids.
Combines `Ridge.gdist_strictMax_imp_corner` (a maximum is a corner) with
`corner_strictMax_iff`. -/
lemma gdist_strictMax_iff (hm : 2 ≤ m) (hn : 2 ≤ n) {p v : Cell m n} :
    IsStrictLocalMax (gdist p) v ↔
      IsCorner v ∧ p.1.val ≠ v.1.val ∧ p.2.val ≠ v.2.val := by
  constructor
  · intro h
    have hcor := gdist_strictMax_imp_corner h
    exact ⟨hcor, (corner_strictMax_iff hm hn hcor).mp h⟩
  · rintro ⟨hc, hav⟩
    exact (corner_strictMax_iff hm hn hc).mpr hav

/-- **Off-ridge, left cone active.**  Where `d(p₁,·)` is strictly smaller (so by
parity smaller by at least two), the envelope agrees with `d(p₁,·)` on `v` and all
its neighbours, so `cpe` and `d(p₁,·)` have a strict local maximum at `v`
simultaneously. -/
lemma cpe_strictMax_iff_cone_left {p₁ p₂ : Cell m n} {δ : ℤ}
    (hδ : (δ - gdist p₁ p₂) % 2 = 0) {v : Cell m n}
    (hlt : gdist p₁ v < δ + gdist p₂ v) :
    IsStrictLocalMax (cpe p₁ p₂ δ) v ↔ IsStrictLocalMax (gdist p₁) v := by
  have hpar := cone_pair_same_parity hδ v
  have hcv : cpe p₁ p₂ δ v = gdist p₁ v := by unfold cpe; exact min_eq_left (le_of_lt hlt)
  constructor
  · intro hmax u huv
    have hmu := hmax u huv
    have s1 := gdist_adj_step (q := p₁) huv
    have s2 := gdist_adj_step (q := p₂) huv
    have hle : gdist p₁ u ≤ δ + gdist p₂ u := by omega
    have hcu : cpe p₁ p₂ δ u = gdist p₁ u := by unfold cpe; exact min_eq_left hle
    rw [hcu, hcv] at hmu; exact hmu
  · intro hmax u huv
    have hmu := hmax u huv
    have s1 := gdist_adj_step (q := p₁) huv
    have s2 := gdist_adj_step (q := p₂) huv
    have hle : gdist p₁ u ≤ δ + gdist p₂ u := by omega
    have hcu : cpe p₁ p₂ δ u = gdist p₁ u := by unfold cpe; exact min_eq_left hle
    show cpe p₁ p₂ δ u = cpe p₁ p₂ δ v - 1
    rw [hcu, hcv]; exact hmu

/-- **Off-ridge, right cone active.**  Dual of `cpe_strictMax_iff_cone_left`. -/
lemma cpe_strictMax_iff_cone_right {p₁ p₂ : Cell m n} {δ : ℤ}
    (hδ : (δ - gdist p₁ p₂) % 2 = 0) {v : Cell m n}
    (hgt : δ + gdist p₂ v < gdist p₁ v) :
    IsStrictLocalMax (cpe p₁ p₂ δ) v ↔ IsStrictLocalMax (gdist p₂) v := by
  have hpar := cone_pair_same_parity hδ v
  have hcv : cpe p₁ p₂ δ v = δ + gdist p₂ v := by unfold cpe; exact min_eq_right (le_of_lt hgt)
  constructor
  · intro hmax u huv
    have hmu := hmax u huv
    have s1 := gdist_adj_step (q := p₁) huv
    have s2 := gdist_adj_step (q := p₂) huv
    have hle : δ + gdist p₂ u ≤ gdist p₁ u := by omega
    have hcu : cpe p₁ p₂ δ u = δ + gdist p₂ u := by unfold cpe; exact min_eq_right hle
    rw [hcu, hcv] at hmu; omega
  · intro hmax u huv
    have hmu := hmax u huv
    have s1 := gdist_adj_step (q := p₁) huv
    have s2 := gdist_adj_step (q := p₂) huv
    have hle : δ + gdist p₂ u ≤ gdist p₁ u := by omega
    have hcu : cpe p₁ p₂ δ u = δ + gdist p₂ u := by unfold cpe; exact min_eq_right hle
    show cpe p₁ p₂ δ u = cpe p₁ p₂ δ v - 1
    rw [hcu, hcv]; omega

/-- A grid corner off the ridge **qualifies** when its active apex (the one with
the strictly smaller cone value there) avoids both sides incident to it.  The two
implications cover the two activity cases; off the ridge exactly one fires. -/
def QualifyingCorner (p₁ p₂ : Cell m n) (δ : ℤ) (v : Cell m n) : Prop :=
  IsCorner v ∧
    (gdist p₁ v < δ + gdist p₂ v → p₁.1.val ≠ v.1.val ∧ p₁.2.val ≠ v.2.val) ∧
    (δ + gdist p₂ v < gdist p₁ v → p₂.1.val ≠ v.1.val ∧ p₂.2.val ≠ v.2.val)

/-- **The Ridge Lemma** (`lem:ridge`), full maxima characterisation, end-to-end.
For `m, n ≥ 2`, parity-correct `δ`, and apexes differing in both coordinates, a
cell is a strict local maximum of the cone-pair envelope iff it is either a doubly
admissible cell on the ridge, or a qualifying grid corner off the ridge. -/
theorem cpe_strictMax_iff (hm : 2 ≤ m) (hn : 2 ≤ n) {p₁ p₂ : Cell m n} {δ : ℤ}
    (hδ : (δ - gdist p₁ p₂) % 2 = 0)
    (hr : p₁.1.val ≠ p₂.1.val) (hs : p₁.2.val ≠ p₂.2.val) {v : Cell m n} :
    IsStrictLocalMax (cpe p₁ p₂ δ) v ↔
      (OnRidge p₁ p₂ δ v ∧ RowAdm p₁ p₂ v ∧ ColAdm p₁ p₂ v) ∨
      (¬ OnRidge p₁ p₂ δ v ∧ QualifyingCorner p₁ p₂ δ v) := by
  by_cases hon : OnRidge p₁ p₂ δ v
  · rw [ridge_strictMax_iff_admissible hr hs hon]
    constructor
    · intro h; exact Or.inl ⟨hon, h⟩
    · rintro (⟨_, h⟩ | ⟨hno, _⟩)
      · exact h
      · exact absurd hon hno
  · -- off the ridge: one cone is strictly smaller
    have hne : gdist p₁ v ≠ δ + gdist p₂ v := hon
    rcases (by omega : gdist p₁ v < δ + gdist p₂ v ∨ δ + gdist p₂ v < gdist p₁ v)
        with hlt | hgt
    · rw [cpe_strictMax_iff_cone_left hδ hlt, gdist_strictMax_iff hm hn]
      constructor
      · rintro ⟨hcor, ha1, ha2⟩
        exact Or.inr ⟨hon, hcor, fun _ => ⟨ha1, ha2⟩, fun h => absurd h (by omega)⟩
      · rintro (⟨hr', _⟩ | ⟨_, hcor, himpl, _⟩)
        · exact absurd hr' hon
        · obtain ⟨ha1, ha2⟩ := himpl hlt
          exact ⟨hcor, ha1, ha2⟩
    · rw [cpe_strictMax_iff_cone_right hδ hgt, gdist_strictMax_iff hm hn]
      constructor
      · rintro ⟨hcor, ha1, ha2⟩
        exact Or.inr ⟨hon, hcor, fun h => absurd h (by omega), fun _ => ⟨ha1, ha2⟩⟩
      · rintro (⟨hr', _⟩ | ⟨_, hcor, _, himpl⟩)
        · exact absurd hr' hon
        · obtain ⟨ha1, ha2⟩ := himpl hgt
          exact ⟨hcor, ha1, ha2⟩

end OrigamiCone

