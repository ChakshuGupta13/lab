import OrigamiCone.Cone

/-!
# The Envelope Lemma (Lemma 3.x)

Formalisation of the **Envelope Lemma**:

> Let `h` be a height function with strict-local-minimum set `P` and
> strict-local-maximum set `Q`.  Then for every vertex `v`,
> `h(v) = min_{p∈P} (h(p) + d(p,v)) = max_{q∈Q} (h(q) - d(q,v))`.

This generalises the Cone Lemma (`OrigamiCone.Basic.cone_max`, the case of a
single extremum) to arbitrary extremum sets.  We state each equality as the
conjunction of *attainment* (the value is realised by some extremum) and a
*bound* (every extremum gives an inequality):

* `envelope_min` — `h v = h p + d(p,v)` for some strict local min `p`, and
  `h v ≤ h p + d(p,v)` for every strict local min `p`;
* `envelope_max` — the dual statement over strict local maxima, via `-h`.

The attainment direction is the "descend to a strict local minimum" argument; the
bound is the `1`-Lipschitz property (`height_lipschitz`).  No `sorry`.
-/

namespace OrigamiCone

variable {m n : ℕ}

/-- **Descent to a strict local minimum.** From any cell `v` there is a strict
local minimum `p` with `h v = h p + d(p,v)` (a shortest descending path from `v`
ends at `p`).  This is the attainment half of the Envelope Lemma. -/
lemma exists_descent_to_min {h : Cell m n → ℤ} (hh : IsHeight h) (v : Cell m n) :
    ∃ p, IsStrictLocalMin h p ∧ h v = h p + gdist p v := by
  haveI : Nonempty (Cell m n) := ⟨v⟩
  obtain ⟨w0, hw0⟩ := Finite.exists_min h
  -- induct on the height excess `(h v - h w0).toNat`
  suffices key : ∀ (k : ℕ) (v : Cell m n), (h v - h w0).toNat ≤ k →
      ∃ p, IsStrictLocalMin h p ∧ h v = h p + gdist p v by
    exact key (h v - h w0).toNat v le_rfl
  intro k
  induction k with
  | zero =>
    intro v hk
    have hmin0 : h v = h w0 := by have := hw0 v; omega
    have hvmin : IsStrictLocalMin h v := by
      intro u hu
      have hle := hw0 u
      have h1 := hh v u hu
      rcases (abs_eq (by norm_num : (0:ℤ) ≤ 1)).1 h1 with h2 | h2 <;> omega
    exact ⟨v, hvmin, by simp [gdist_self]⟩
  | succ k ih =>
    intro v hk
    by_cases hvmin : IsStrictLocalMin h v
    · exact ⟨v, hvmin, by simp [gdist_self]⟩
    · -- `v` is not a strict local min, so it has a strictly lower neighbour
      simp only [IsStrictLocalMin, not_forall] at hvmin
      obtain ⟨u, hu⟩ := hvmin
      have hadj : adj v u := by tauto
      have hune : h u ≠ h v + 1 := by tauto
      have h1 := hh v u hadj
      have hdown : h u = h v - 1 := by
        rcases (abs_eq (by norm_num : (0:ℤ) ≤ 1)).1 h1 with h2 | h2
        · omega
        · exact absurd (by omega : h u = h v + 1) hune
      have hdefu : (h u - h w0).toNat ≤ k := by have := hw0 u; omega
      obtain ⟨p, hpmin, hpu⟩ := ih u hdefu
      -- combine: h v = h u + 1 = h p + d(p,u) + 1, and d(p,v) = d(p,u) + 1
      have htri := gdist_triangle p u v
      have huv : gdist u v = 1 := by rw [gdist_comm]; exact hadj
      have hlip : |h p - h v| ≤ gdist p v :=
        height_lipschitz hh (gdist p v).toNat p v
          (le_of_eq (Int.toNat_of_nonneg (gdist_nonneg p v)).symm)
      have hge0 := gdist_nonneg p u
      have hpv : gdist p v = gdist p u + 1 := by
        rw [abs_le] at hlip; omega
      exact ⟨p, hpmin, by omega⟩

/-- **Envelope lower bound.** A height function never exceeds the cone over any
cell: `h v ≤ h p + d(p,v)` for every `p` (in particular every strict local
minimum).  This is the `1`-Lipschitz bound. -/
lemma envelope_le {h : Cell m n → ℤ} (hh : IsHeight h) (p v : Cell m n) :
    h v ≤ h p + gdist p v := by
  have hlip : |h v - h p| ≤ gdist v p :=
    height_lipschitz hh (gdist v p).toNat v p
      (le_of_eq (Int.toNat_of_nonneg (gdist_nonneg v p)).symm)
  rw [gdist_comm v p, abs_le] at hlip
  omega

/-- **Envelope Lemma, minimum form.** `h v` equals `h p + d(p,v)` for some strict
local minimum `p`, and is at most `h p + d(p,v)` for every strict local minimum
`p`: that is, `h v = min_{p} (h p + d(p,v))` over the strict local minima. -/
theorem envelope_min {h : Cell m n → ℤ} (hh : IsHeight h) (v : Cell m n) :
    (∃ p, IsStrictLocalMin h p ∧ h v = h p + gdist p v)
      ∧ (∀ p, IsStrictLocalMin h p → h v ≤ h p + gdist p v) :=
  ⟨exists_descent_to_min hh v, fun p _ => envelope_le hh p v⟩

/-- **Envelope Lemma, maximum form.** Dually, `h v = h q - d(q,v)` for some strict
local maximum `q`, and `h q - d(q,v) ≤ h v` for every strict local maximum `q`:
`h v = max_{q} (h q - d(q,v))` over the strict local maxima.  Proved by applying
the minimum form to `-h`. -/
theorem envelope_max {h : Cell m n → ℤ} (hh : IsHeight h) (v : Cell m n) :
    (∃ q, IsStrictLocalMax h q ∧ h v = h q - gdist q v)
      ∧ (∀ q, IsStrictLocalMax h q → h q - gdist q v ≤ h v) := by
  have hh' : IsHeight (fun w => -h w) := by
    intro a b hab
    have := hh a b hab
    rw [show (-h a) - (-h b) = -(h a - h b) by ring, abs_neg]; exact this
  obtain ⟨⟨p, hpmin, hpeq⟩, hple⟩ := envelope_min hh' v
  constructor
  · refine ⟨p, ?_, ?_⟩
    · intro u hu
      have := hpmin u hu; simp only at this; linarith
    · have := hpeq; simp only at this; linarith
  · intro q hq
    have hq' : IsStrictLocalMin (fun w => -h w) q := by
      intro u hu; have := hq u hu; simp only; linarith
    have := hple q hq'; linarith

end OrigamiCone
