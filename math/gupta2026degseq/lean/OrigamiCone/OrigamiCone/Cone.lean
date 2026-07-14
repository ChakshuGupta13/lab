import OrigamiCone.Basic

/-!
# The distance cone and its unique maximum

Formalisation of the paragraph after Lemma 2.2 in the paper:

> For each grid vertex `q`, the cone at `q` is the height function
> `h_q(v) := d(q,(1,1)) - d(q,v)`. Every vertex other than `q` has a neighbour
> nearer `q`, so `d(q,·)` has its only local minimum at `q` and the cone `h_q`
> has `q` as its only local maximum.

We define the cone at `q` with a free basepoint `b`,
`cone q b v := gdist q b - gdist q v`; the paper fixes `b` to the corner `(1,1)`
so that `h_q(1,1) = 0`, but the structural facts below (that `cone q b` is a
height function and that `q` is its unique strict local maximum) are independent
of the basepoint, so proving them for arbitrary `b` is strictly more general and
specialises to the paper's normalised cone.

Results:
* `cone_isHeight`   — the cone is a height function;
* `cone_max_at`     — `q` is a strict local maximum of the cone;
* `cone_unique_max` — `q` is the *only* strict local maximum.

Together with `cone_max` of `OrigamiCone.Basic`, these say the cone is exactly
the height function the Cone Lemma reconstructs from its apex.  No `sorry`.
-/

namespace OrigamiCone

variable {m n : ℕ}

/-- The distance cone at `q` with basepoint `b`:
`cone q b v = gdist q b - gdist q v`.  The paper's cone `h_q` is `cone q corner`,
where `corner = (1,1)`. -/
def cone (q b : Cell m n) : Cell m n → ℤ := fun v => gdist q b - gdist q v

/-- **Distance to a fixed cell changes by exactly one across an edge.** For
adjacent `p, p'`, the grid distances `gdist q p` and `gdist q p'` differ by one.
This is the bipartite-parity fact underlying the cone being a height function:
the triangle inequality bounds the gap by one, and the opposite parities of
adjacent cells force it to be exactly one. -/
lemma gdist_adj_step {q p p' : Cell m n} (h : adj p p') :
    gdist q p = gdist q p' + 1 ∨ gdist q p = gdist q p' - 1 := by
  unfold adj gdist at *
  omega

/-- **The cone is a height function.** -/
lemma cone_isHeight (q b : Cell m n) : IsHeight (cone q b) := by
  intro p p' hpp'
  rw [abs_eq (by norm_num : (0 : ℤ) ≤ 1)]
  simp only [cone]
  rcases gdist_adj_step (q := q) hpp' with h | h
  · right; omega
  · left; omega

/-- **`q` is a strict local maximum of its cone**: every neighbour is one lower.
-/
lemma cone_max_at (q b : Cell m n) : IsStrictLocalMax (cone q b) q := by
  intro u hu
  have hu1 : gdist q u = 1 := hu
  have h0 : gdist q q = 0 := gdist_self q
  simp only [cone]
  omega

/-- **`q` is the only strict local maximum of its cone.** Any other cell has, by
`exists_step_toward`, a neighbour strictly nearer `q`, hence one on which the
cone is strictly higher — so it is not a local maximum. -/
lemma cone_unique_max (q b : Cell m n) :
    ∀ q', IsStrictLocalMax (cone q b) q' → q' = q := by
  intro q' hq'
  by_contra hne
  obtain ⟨u, hadj, hd⟩ := exists_step_toward (show q' ≠ q from hne)
  have hval := hq' u hadj
  simp only [cone] at hval
  have hc1 : gdist q u = gdist u q := gdist_comm q u
  have hc2 : gdist q q' = gdist q' q := gdist_comm q q'
  omega

/-- The cone really is recovered by the Cone Lemma from its apex: combining
`cone_isHeight`, `cone_unique_max`, and `cone_max` of `OrigamiCone.Basic` gives
`cone q b v = cone q b q - gdist q v` for every `v`, the apex form of the cone.
-/
lemma cone_eq_apex_form (q b : Cell m n) (v : Cell m n) :
    cone q b v = cone q b q - gdist q v :=
  cone_max (cone_isHeight q b) (cone_unique_max q b) v

end OrigamiCone
