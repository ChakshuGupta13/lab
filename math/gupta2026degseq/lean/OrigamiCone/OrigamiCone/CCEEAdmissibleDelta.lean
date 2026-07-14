import Mathlib.Data.Finset.Card
import Mathlib.Data.Int.Interval
import Mathlib.Tactic.Linarith

/-!
# CC|EE admissible-δ count, opposite-corner pair (Sub-6a of `thm:deg4count`)

For an **opposite-corner** cone-pair apex configuration in `Cell m n` with
`m, n ≥ 3` — e.g. `(BL, TR) = ((0, 0), (m-1, n-1))` — the cone-pair envelope
`cpe v = min(acell v, δ + L - acell v)` (Sub-3, where `L = m + n - 2 = D` is the
Manhattan distance between the apexes) is a tent in `s = acell v ∈ {0, …, L}`.

Under the **parity discipline** (`δ ≡ D (mod 2)`, i.e. `δ + L` even, from the
Parity Lemma), the tent peak is at the integer level `s* = (δ + L) / 2`.

A degree-4 OFG vertex from this cone-pair requires the **peak antidiagonal**
`A_{s*}` to contain exactly 2 cells (Sub-5).  By Sub-5 (combined with the fact
that other antidiagonals have either 1 cell or ≥ 3 cells when `min(m, n) ≥ 3`),
this happens iff `s* ∈ {1, L - 1}`, i.e. iff `δ ∈ {2 - L, L - 2}`.

This module proves the **count**: under the closed-form admissibility predicate
`δ = 2 - L ∨ δ = L - 2`, the set has cardinality 2 in the in-range Finset
`Icc (-L) L`.  Combined with the dual TL-BR opposite pair (one more factor of 2)
and the adjacent-corner kill (Sub-4 → no deg-4 from those), this is half of the
geometric proof that `familyCCEE = 4`.

Result:
* `ccee_admissible_delta_count` — `|{δ ∈ [-L, L] : δ = 2 - L ∨ δ = L - 2}| = 2`
  for `m, n ≥ 3`, where `L = m + n - 2`.
* `ccee_admissible_delta_set` — the same set, equal to `{2 - L, L - 2}`.

No `sorry`.
-/

namespace OrigamiCone

variable {m n : ℕ}

/-- **CC|EE admissible-δ set, opposite-corner pair.**  For `m, n ≥ 3` and
`L = m + n - 2`, the in-range parity-disciplined offsets producing a 2-element
peak antidiagonal are exactly `{2 - L, L - 2}`. -/
theorem ccee_admissible_delta_set (hm : 3 ≤ m) (hn : 3 ≤ n) :
    (Finset.Icc (-((m + n - 2 : ℕ) : ℤ)) ((m + n - 2 : ℕ) : ℤ)).filter
        (fun δ : ℤ => δ = 2 - ((m + n - 2 : ℕ) : ℤ)
                  ∨ δ = ((m + n - 2 : ℕ) : ℤ) - 2)
      = {2 - ((m + n - 2 : ℕ) : ℤ), ((m + n - 2 : ℕ) : ℤ) - 2} := by
  -- L ≥ 4 from m, n ≥ 3.
  have hL_cast : ((m + n - 2 : ℕ) : ℤ) = (m : ℤ) + n - 2 := by omega
  have hL_ge : ((m + n - 2 : ℕ) : ℤ) ≥ 4 := by rw [hL_cast]; omega
  ext δ
  simp only [Finset.mem_filter, Finset.mem_Icc, Finset.mem_insert,
    Finset.mem_singleton]
  constructor
  · rintro ⟨_, h | h⟩
    · exact Or.inl h
    · exact Or.inr h
  · rintro (h | h) <;> subst h
    · refine ⟨⟨?_, ?_⟩, Or.inl rfl⟩
      · linarith
      · linarith
    · refine ⟨⟨?_, ?_⟩, Or.inr rfl⟩
      · linarith
      · linarith

/-- **CC|EE admissible-δ count, opposite-corner pair.**  For `m, n ≥ 3` and
`L = m + n - 2`, the count of in-range parity-disciplined offsets producing a
2-element peak antidiagonal is exactly 2. -/
theorem ccee_admissible_delta_count (hm : 3 ≤ m) (hn : 3 ≤ n) :
    ((Finset.Icc (-((m + n - 2 : ℕ) : ℤ)) ((m + n - 2 : ℕ) : ℤ)).filter
        (fun δ : ℤ => δ = 2 - ((m + n - 2 : ℕ) : ℤ)
                  ∨ δ = ((m + n - 2 : ℕ) : ℤ) - 2)).card = 2 := by
  rw [ccee_admissible_delta_set hm hn]
  apply Finset.card_pair
  -- 2 - L ≠ L - 2, using L ≥ 4.
  have hL_cast : ((m + n - 2 : ℕ) : ℤ) = (m : ℤ) + n - 2 := by omega
  intro h
  -- h : 2 - L = L - 2 ⟹ 2L = 4 ⟹ L = 2, contradicting L ≥ 4.
  rw [hL_cast] at h
  omega

end OrigamiCone
