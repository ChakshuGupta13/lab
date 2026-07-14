import Mathlib.RingTheory.Polynomial.Pochhammer
import Mathlib.Data.Nat.Choose.Basic
import Mathlib.Data.Nat.Factorial.Basic

/-!
# Composition-count polynomial

For each fixed `r : ℕ` the natural binomial coefficient `Nat.choose · r`
is the evaluation of a polynomial in `ℚ[X]` of natural degree at most `r`.
Shifted by a fixed offset `w : ℕ`, this becomes the composition count
`Nat.choose (n - w + r - 1) (r - 1)` — the number of ways to write a free
length of `n - w + r` as an ordered sum of `r` positive integers — which,
as a function of `n : ℕ` on the region `n ≥ w`, is a polynomial in `n` of
natural degree at most `r - 1`.

Two named results:

* `Nat.choose_as_poly r` — polynomial witness for `Nat.choose · r`.
* `composition_count_as_poly w r hr` — polynomial witness for the
  composition count.

The composition-count polynomial is used by `SequelUniformOnsetProof`
(pending) to conclude that each configuration type's row count is a
polynomial in `n`, hence `Ed d m ·` (a finite sum over types) is a
polynomial in `n` of degree at most `d` on the uniform region `n ≥ N_d` —
the paper's `Lemma 8.5`.

The core bridge is `descPochhammer_eval_nat_cast`: for `k : ℕ`, the
polynomial `descPochhammer ℚ r` evaluated at `(k : ℚ)` equals
`(k.descFactorial r : ℚ)`, by induction on `r` using
`descPochhammer_succ_eval` and `Nat.descFactorial_succ`.
-/

namespace OrigamiCone.Sequel

open Polynomial

/-- Evaluating the rational descending Pochhammer polynomial at a natural
number yields the natural descending factorial (cast to `ℚ`). -/
theorem descPochhammer_eval_nat_cast (r k : ℕ) :
    (descPochhammer ℚ r).eval (k : ℚ) = (k.descFactorial r : ℚ) := by
  induction r with
  | zero => simp
  | succ r ih =>
    rw [descPochhammer_succ_eval, ih, Nat.descFactorial_succ]
    rcases Nat.lt_or_ge k r with hlt | hge
    · -- k < r: the descending factorial itself vanishes; both sides are 0.
      have hz : k.descFactorial r = 0 :=
        Nat.descFactorial_eq_zero_iff_lt.mpr hlt
      simp [hz]
    · -- r ≤ k: natural subtraction agrees with rational subtraction.
      push_cast [Nat.cast_sub hge]
      ring

/-- **Choose-as-polynomial**: for each `r : ℕ`, there is a polynomial
`p : ℚ[X]` of natural degree at most `r` such that `p.eval (k : ℚ) =
Nat.choose k r` for every `k : ℕ`.

The witness is `C ((r!)⁻¹) * descPochhammer ℚ r`; its evaluation at `k`
equals `k.descFactorial r / r! = Nat.choose k r` via
`Nat.descFactorial_eq_factorial_mul_choose`. -/
theorem Nat.choose_as_poly (r : ℕ) :
    ∃ p : Polynomial ℚ, p.natDegree ≤ r ∧
      ∀ k : ℕ, (Nat.choose k r : ℚ) = p.eval (k : ℚ) := by
  refine ⟨C ((r.factorial : ℚ)⁻¹) * descPochhammer ℚ r, ?_, ?_⟩
  · calc (C ((r.factorial : ℚ)⁻¹) * descPochhammer ℚ r).natDegree
        ≤ (descPochhammer ℚ r).natDegree := Polynomial.natDegree_C_mul_le _ _
      _ = r := descPochhammer_natDegree (R := ℚ) r
  · intro k
    have hr : (r.factorial : ℚ) ≠ 0 :=
      Nat.cast_ne_zero.mpr r.factorial_ne_zero
    rw [eval_mul, eval_C, descPochhammer_eval_nat_cast,
        Nat.descFactorial_eq_factorial_mul_choose]
    push_cast
    rw [← mul_assoc, inv_mul_cancel₀ hr, one_mul]

/-- **Composition-count polynomial**: for each `w r : ℕ` with `1 ≤ r`, the
composition count `Nat.choose (n - w + r - 1) (r - 1)` — the number of ways
to split a free length of `n - w + r` columns into `r` runs each of length
at least `1` — is, as a function of `n ≥ w`, a polynomial in `n` of natural
degree at most `r - 1`.

The witness is `q.comp (X + C ((r - 1 : ℚ) - w))` where `q` is the
`(r - 1)`-choose polynomial from `Nat.choose_as_poly`. -/
theorem composition_count_as_poly (w r : ℕ) (hr : 1 ≤ r) :
    ∃ p : Polynomial ℚ, p.natDegree ≤ r - 1 ∧
      ∀ n : ℕ, w ≤ n →
        (Nat.choose (n - w + r - 1) (r - 1) : ℚ) = p.eval (n : ℚ) := by
  obtain ⟨q, hqdeg, hqev⟩ := Nat.choose_as_poly (r - 1)
  refine ⟨q.comp (X + C (((r - 1 : ℕ) : ℚ) - (w : ℚ))), ?_, ?_⟩
  · -- natDegree of `q ∘ (X + c)` is at most `natDegree q · natDegree (X + c) = natDegree q · 1`.
    have hshift_deg : (X + C (((r - 1 : ℕ) : ℚ) - (w : ℚ))).natDegree = 1 :=
      natDegree_X_add_C _
    calc (q.comp (X + C (((r - 1 : ℕ) : ℚ) - (w : ℚ)))).natDegree
        ≤ q.natDegree * (X + C (((r - 1 : ℕ) : ℚ) - (w : ℚ))).natDegree :=
          Polynomial.natDegree_comp_le
      _ = q.natDegree * 1 := by rw [hshift_deg]
      _ = q.natDegree := Nat.mul_one _
      _ ≤ r - 1 := hqdeg
  · intro n hn
    -- Cast `n - w + r - 1 : ℕ` to `n + (r - 1) - w : ℕ`, then to
    -- `(n : ℚ) + (r - 1) - w`.
    have hshift : (n - w + r - 1 : ℕ) = n + (r - 1) - w := by omega
    have hle : w ≤ n + (r - 1) := by omega
    rw [hshift, hqev, eval_comp, eval_add, eval_X, eval_C]
    congr 1
    push_cast [Nat.cast_sub hle, Nat.cast_sub hr]
    ring

end OrigamiCone.Sequel
