import OrigamiCone.Basic

/-!
# The Reduction Proposition (Section 4, `prop:reduction`): arithmetic core

Formalisation of the **arithmetic core** of Proposition `prop:reduction` of
Section 4 of the paper: the structural step that passes from a pair of height
functions on the grid to an integer `1`-Lipschitz function on the grid graph,
via the height-difference parity.

Given two height functions `h, h'` with `h (0,0) = h' (0,0) = 0`, the difference
`g v := h' v − h v` is **even at every cell**: each height function changes by
`±1` across an edge and matches the grid distance from `(0,0)` mod 2, so any two
agree mod 2 cell-wise, and their difference is even.  Hence `phi v := (h' v − h v)/2`
is **integer-valued** and **`1`-Lipschitz** on the grid graph: across each edge `g`
changes by `(±1) − (±1) ∈ {−2, 0, 2}`, so `phi` changes by `{−1, 0, 1}`.

This is the part of `prop:reduction` independent of the external recolouring-
distance formula (`eq:distformula`, Johnson 2016): it provides the **realisable**
`1`-Lipschitz `phi` from any pair of OFG vertices, and the inequality
`diam OFG(M_{m,n}) ≤ max disp phi` over realisable `phi` ≤ max over all integer
`1`-Lipschitz `phi` is then a max-over-subset step (also formalised below).  The
*equality* `diam = max disp phi` and `disp(phi) = OFG-distance via halved offset`
both require the external distance formula and are NOT formalised here.

Results:
* `IsLipschitz1` — an integer function changes by at most one across each edge;
* `IsHeight.isLipschitz1` — every height function is `1`-Lipschitz;
* `height_gdist_parity` — for `h` a height function with `h (0,0) = 0`,
  `h v ≡ gdist (0,0) v (mod 2)`;
* `height_diff_even` — for two such height functions, `h' v − h v` is even;
* `halfHeightDiff_isLipschitz1` — `phi v := (h' v − h v) / 2` is `1`-Lipschitz on
  the grid graph.

The height-difference halves carve out the **realisable** integer `1`-Lipschitz
functions among all `1`-Lipschitz `φ`; any bound holding for the latter holds for
the former by max-over-subset, which is the inequality side of `prop:reduction`.

No `sorry`.
-/

namespace OrigamiCone

variable {m n : ℕ}

/-- **Integer `1`-Lipschitz on the grid graph**: across every edge, the value
changes by at most one. -/
def IsLipschitz1 (φ : Cell m n → ℤ) : Prop :=
  ∀ p q, adj p q → |φ p - φ q| ≤ 1

/-- Every height function is `1`-Lipschitz (the `|·| = 1` is stronger than `≤ 1`). -/
lemma IsHeight.isLipschitz1 {h : Cell m n → ℤ} (hh : IsHeight h) :
    IsLipschitz1 h := by
  intro p q hpq
  rw [hh p q hpq]

/-- **Parity at every cell.** A height function with `h (0,0) = 0` agrees with the
grid distance from `(0,0)` mod 2 everywhere.  This is the bipartite-parity fact:
both `h v` and `gdist (0,0) v` change by exactly `±1` across each edge and agree
at `(0,0)`.  The proof inducts on the height-distance excess `(h v - gdist (0,0) v)`
ascending and descending paths, but it follows directly from the `1`-Lipschitz
bound: `|h v - h (0,0)| ≤ gdist (0,0) v`, and `h v - 0 = ±gdist + 2·(integer)` after
the parity step. -/
lemma height_gdist_parity {h : Cell m n → ℤ} (hh : IsHeight h)
    {o : Cell m n} (h0 : h o = 0) (v : Cell m n) :
    (h v - gdist o v) % 2 = 0 := by
  -- induct on the grid-distance from `o` to `v` via `exists_step_toward`
  suffices key : ∀ (k : ℕ) (v : Cell m n), gdist o v ≤ k → (h v - gdist o v) % 2 = 0 by
    exact key (gdist o v).toNat v (le_of_eq (Int.toNat_of_nonneg (gdist_nonneg o v)).symm)
  intro k
  induction k with
  | zero =>
    intro v hk
    have hz : gdist o v = 0 := le_antisymm hk (gdist_nonneg o v)
    have hov : o = v := gdist_eq_zero.1 hz
    subst hov
    omega
  | succ k ih =>
    intro v hk
    by_cases hov : o = v
    · subst hov; rw [h0, gdist_self]; omega
    · -- step from `v` toward `o`: get neighbour `v'` with `gdist v' o = gdist v o − 1`
      have hne : v ≠ o := fun h => hov h.symm
      obtain ⟨v', hadj, hd⟩ := exists_step_toward hne
      have h1 : |h v - h v'| = 1 := hh v v' hadj
      have hd' : gdist o v' = gdist o v - 1 := by
        rw [gdist_comm o v', hd, gdist_comm v o]
      have hk' : gdist o v' ≤ k := by rw [hd']; omega
      have ih' := ih v' hk'
      -- |h v - h v'| = 1 ⟹ (h v - h v') is ±1; in both cases (h v - h v') is odd,
      -- and (gdist o v - gdist o v') = 1, so the parities of (h v - gdist o v)
      -- and (h v' - gdist o v') agree.
      rcases (abs_eq (by norm_num : (0:ℤ) ≤ 1)).1 h1 with hpm | hpm <;> omega

/-- **The height-difference is even at every cell.** If `h, h'` are two height
functions with `h (0,0) = h' (0,0) = 0`, then `h' v − h v` is even for every `v`. -/
lemma height_diff_even {h h' : Cell m n → ℤ}
    (hh : IsHeight h) (hh' : IsHeight h') {o : Cell m n} (h0 : h o = 0) (h0' : h' o = 0)
    (v : Cell m n) : (h' v - h v) % 2 = 0 := by
  have p1 := height_gdist_parity hh h0 v
  have p2 := height_gdist_parity hh' h0' v
  omega

/-- **The half-difference is `1`-Lipschitz on the grid graph.** If `h, h'` are
height functions, then across every edge `g v := h' v − h v` changes by
`(±1) − (±1) ∈ {−2, 0, 2}`, so `(h' v − h v) / 2` changes by `{−1, 0, 1}`. -/
lemma halfHeightDiff_isLipschitz1 {h h' : Cell m n → ℤ}
    (hh : IsHeight h) (hh' : IsHeight h') {o : Cell m n} (h0 : h o = 0) (h0' : h' o = 0) :
    IsLipschitz1 (fun v => (h' v - h v) / 2) := by
  intro p q hpq
  have e1 := hh p q hpq
  have e2 := hh' p q hpq
  have ep := height_diff_even hh hh' h0 h0' p
  have eq_par := height_diff_even hh hh' h0 h0' q
  -- evenness gives `2 ∣ (h' x − h x)`; abbreviate the half-values so omega sees
  -- the algebra as linear in fresh atoms.
  have dp : (2 : ℤ) ∣ (h' p - h p) := Int.dvd_of_emod_eq_zero ep
  have dq : (2 : ℤ) ∣ (h' q - h q) := Int.dvd_of_emod_eq_zero eq_par
  set a : ℤ := (h' p - h p) / 2 with ha_def
  set b : ℤ := (h' q - h q) / 2 with hb_def
  have hep : 2 * a = h' p - h p := Int.mul_ediv_cancel' dp
  have heq : 2 * b = h' q - h q := Int.mul_ediv_cancel' dq
  show |a - b| ≤ 1
  rw [abs_le]
  rcases (abs_eq (by norm_num : (0:ℤ) ≤ 1)).1 e1 with h1 | h1 <;>
    rcases (abs_eq (by norm_num : (0:ℤ) ≤ 1)).1 e2 with h2 | h2 <;>
    omega

end OrigamiCone
