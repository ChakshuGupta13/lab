import OrigamiCone.Basic

/-!
# Neighbourhood of an initial segment (Section 4, Lemma `lem:Ninit`)

Formalisation of **Lemma (Neighbourhood of an initial segment)** of Section 4 of
the paper *"The origami flip graph of the m × n Miura-ori: degree sequence and
diameter via height functions"*.

The *simplicial order* on the grid `Cell m n = Fin m × Fin n` orders cells by
antidiagonal `i + j`, ties broken by increasing row `i`.  An *initial segment*
`I(s)` is the set of the `s` smallest cells; equivalently — the order being
linear — a downward-closed set (`IsInitialSeg`).  The lemma states:

> the closed neighbourhood `N[I(s)]` of an initial segment is an initial segment.

The paper proves this in the orientation `m ≥ n`.  We prove the
**orientation-free** statement (no hypothesis relating `m` and `n`), which is
strictly stronger and still establishes the paper's claim.  The orientation only
matters for the *isoperimetric optimality* of the antidiagonal segment (a
separate claim of Section 4), not for the initial-segment closure proved here;
this was confirmed by exhaustive enumeration in both orientations.

The proof replaces the paper's antidiagonal-prefix bookkeeping with a *simplicial
predecessor* `spred`: the lowest-key grid neighbour of a cell, namely `(i−1, j)`
if `i ≥ 1` and `(0, j−1)` if `i = 0`.  Two observations drive it:

* for a downward-closed `S`, a cell lies in `N[S]` iff it lies in `S` or its
  predecessor does (`spred` is `≤` every neighbour, so it witnesses adjacency to
  `S` whenever any neighbour does); and
* `spred` is monotone enough that `d ⊏ c` forces `d ≤ spred c` or
  `spred d ≤ spred c` (`spred_mono`).

Together these make the closed neighbourhood of a down-set a down-set.

Results:
* `sle`, `slt`, `IsInitialSeg`, `closedNbhd`, `spred` — definitions;
* `spred_adj`, `spred_sle_nbhr`, `spred_mono`, `spred_mem_closedNbhd` —
  predecessor facts;
* `closedNbhd_isInitialSeg` — the lemma (orientation-free).

No `sorry`.
-/

namespace OrigamiCone

variable {m n : ℕ}

/-- **Simplicial order** (`≤`): by antidiagonal `i + j`, ties by increasing row
`i`.  A linear order; its downward-closed sets are the initial segments. -/
def sle (c d : Cell m n) : Prop :=
  c.1.val + c.2.val < d.1.val + d.2.val ∨
    (c.1.val + c.2.val = d.1.val + d.2.val ∧ c.1.val ≤ d.1.val)

/-- **Strict simplicial order** (`⊏`). -/
def slt (c d : Cell m n) : Prop :=
  c.1.val + c.2.val < d.1.val + d.2.val ∨
    (c.1.val + c.2.val = d.1.val + d.2.val ∧ c.1.val < d.1.val)

/-- **Initial segment** of the simplicial order: a downward-closed set.  This is
the order-theoretic reading of "`I(s)` = the first `s` cells", since the
simplicial order is linear. -/
def IsInitialSeg (S : Set (Cell m n)) : Prop :=
  ∀ {c d : Cell m n}, c ∈ S → sle d c → d ∈ S

/-- **Closed neighbourhood** of a set, as a predicate: a cell is in `N[S]` iff it
lies in `S` or is grid-adjacent to a cell of `S`. -/
def closedNbhd (S : Set (Cell m n)) : Set (Cell m n) :=
  {c | c ∈ S ∨ ∃ u, adj c u ∧ u ∈ S}

/-- **Simplicial predecessor**: the lowest-key grid neighbour of a cell, namely
`(i−1, j)` when `i ≥ 1` and `(0, j−1)` when `i = 0`.  At the origin it returns the
origin (harmless: it is used only at non-origin cells). -/
def spred (c : Cell m n) : Cell m n :=
  if h : 0 < c.1.val then
    (⟨c.1.val - 1, by have := c.1.isLt; omega⟩, c.2)
  else
    (c.1, ⟨c.2.val - 1, by have := c.2.isLt; omega⟩)

/-- `sle` decomposes as strict-or-equal. -/
lemma sle_iff_slt_or_eq {c d : Cell m n} : sle c d ↔ slt c d ∨ c = d := by
  constructor
  · rintro (h | ⟨he, hle⟩)
    · exact Or.inl (Or.inl h)
    · rcases lt_or_eq_of_le hle with hlt | heq
      · exact Or.inl (Or.inr ⟨he, hlt⟩)
      · refine Or.inr ?_
        have hj : c.2.val = d.2.val := by omega
        exact Prod.ext (Fin.ext heq) (Fin.ext hj)
  · rintro (h | rfl)
    · rcases h with h | ⟨he, hlt⟩
      · exact Or.inl h
      · exact Or.inr ⟨he, le_of_lt hlt⟩
    · exact Or.inr ⟨rfl, le_rfl⟩

/-- The predecessor of a non-origin cell is a grid neighbour. -/
lemma spred_adj {c : Cell m n} (hc : 0 < c.1.val + c.2.val) : adj c (spred c) := by
  have h1 := c.1.isLt
  have h2 := c.2.isLt
  unfold adj gdist spred
  split_ifs with h <;> dsimp only <;> omega

/-- The predecessor is `≤` every grid neighbour: if `u` is adjacent to `c`, then
`spred c ≤ u`.  (It is the lowest-key neighbour.) -/
lemma spred_sle_nbhr {c u : Cell m n} (hu : adj c u) : sle (spred c) u := by
  have h1 := c.1.isLt
  have h2 := c.2.isLt
  have hu1 := u.1.isLt
  have hu2 := u.2.isLt
  unfold adj gdist at hu
  unfold sle spred
  split_ifs with h <;> dsimp only <;> omega

/-- **Predecessor monotonicity** (the crux).  If `d` is strictly below `c`, then
either `d` is already `≤ spred c`, or `spred d ≤ spred c`. -/
lemma spred_mono {c d : Cell m n} (h : slt d c) :
    sle d (spred c) ∨ sle (spred d) (spred c) := by
  have h1 := c.1.isLt
  have h2 := c.2.isLt
  have hd1 := d.1.isLt
  have hd2 := d.2.isLt
  unfold slt at h
  unfold sle spred
  split_ifs with hci hdi <;> dsimp only <;> omega

/-- If the predecessor of `d` lies in `S`, then `d` lies in `N[S]`: either `d` is
a non-origin cell adjacent to `spred d ∈ S`, or `d` is the origin and equals its
own predecessor. -/
lemma spred_mem_closedNbhd {S : Set (Cell m n)} {d : Cell m n}
    (h : spred d ∈ S) : d ∈ closedNbhd S := by
  by_cases hd0 : 0 < d.1.val + d.2.val
  · exact Or.inr ⟨spred d, spred_adj hd0, h⟩
  · have hsp : spred d = d := by
      have h2 := d.2.isLt
      unfold spred
      rw [dif_neg (by omega)]
      exact Prod.ext rfl (Fin.ext (by dsimp only; omega))
    rw [hsp] at h
    exact Or.inl h

/-- **Lemma `lem:Ninit` (orientation-free).** The closed neighbourhood of an
initial segment of the simplicial order is again an initial segment.  In
particular `N[I(s)]` is an initial segment for every `s`, on every grid `m × n`
(no orientation hypothesis). -/
theorem closedNbhd_isInitialSeg {S : Set (Cell m n)} (hS : IsInitialSeg S) :
    IsInitialSeg (closedNbhd S) := by
  intro c d hc hdc
  -- `d ⊑ c` and `c ∈ N[S]`; show `d ∈ N[S]`.
  rcases sle_iff_slt_or_eq.1 hdc with hlt | rfl
  · -- strict case `d ⊏ c`
    rcases hc with hcS | ⟨u, hcu, huS⟩
    · -- `c ∈ S`: then `d ∈ S ⊆ N[S]`.
      exact Or.inl (hS hcS (sle_iff_slt_or_eq.2 (Or.inl hlt)))
    · -- `c ∉ S` but adjacent to `u ∈ S`.  Then `spred c ∈ S`.
      have hspcS : spred c ∈ S := hS huS (spred_sle_nbhr hcu)
      rcases spred_mono hlt with hle | hple
      · -- `d ≤ spred c ∈ S` ⇒ `d ∈ S`.
        exact Or.inl (hS hspcS hle)
      · -- `spred d ≤ spred c ∈ S` ⇒ `spred d ∈ S` ⇒ `d ∈ N[S]`.
        exact spred_mem_closedNbhd (hS hspcS hple)
  · -- `d = c`: membership is unchanged.
    exact hc

end OrigamiCone
