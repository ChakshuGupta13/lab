import OrigamiCone.DiameterLower

/-!
# The antidiagonal `acell` is a height function

The antidiagonal `acell v = v.1.val + v.2.val` is a canonical example of a
**height function** on the grid `Cell m n`: adjacent cells differ by exactly
one in `acell` value.  This is implicit in the paper's diameter construction
(`thm:diam` exhibits `acell` as a witness, calling it `a(i, j) = i + j`),
but no Lean lemma had it packaged as `IsHeight acell` until now.

The proof: adjacency `adj p q` means `|p₁ − q₁| + |p₂ − q₂| = 1`, so exactly
one coordinate differs by `1` and the other agrees.  Then
`|acell p − acell q| = |(p₁ − q₁) + (p₂ − q₂)| = 1`.

This lemma is one of the foundational facts the eventual `diam ≥ D(m, n)`
bridge will use: the construction in `cornerGradient_medianMin`
(`DiameterLower.lean`) takes `h₊₊ = acell` (modulo translation) as one of
the two opposite corner gradients whose OFG distance witnesses the lower
bound.  Until then, the lemma stands as a standalone foundational fact.

Result:
* `acell_isHeight` — `IsHeight (acell (m := m) (n := n))`.

No `sorry`.
-/

namespace OrigamiCone

variable {m n : ℕ}

/-- **The antidiagonal is a height function**.  Adjacent cells (Manhattan
distance one) differ by exactly `±1` in `acell` value, since one coordinate
agrees and the other differs by `1`.

This makes `acell` a vertex of `OFG(M_{m,n})` (after the mkV shift quotient).
It is one of the canonical degree-2 vertices — the gradient `h₊₊` of the
paper — and serves as the witness for the lower-bound `diam ≥ D(m, n)`
construction (`cornerGradient_medianMin` in `DiameterLower.lean`). -/
theorem acell_isHeight : IsHeight (acell (m := m) (n := n)) := by
  intro p q hpq
  -- adj p q means gdist p q = 1, i.e. |p.1 - q.1| + |p.2 - q.2| = 1 (ℤ-cast of ℕ).
  unfold adj gdist at hpq
  -- Unfold acell on both sides.
  unfold acell
  -- Goal: |((p.1.val : ℤ) + p.2.val) - ((q.1.val : ℤ) + q.2.val)| = 1.
  -- Let d1 = p.1.val - q.1.val, d2 = p.2.val - q.2.val (ℤ).
  -- Hypothesis: d1.natAbs + d2.natAbs = 1 (after cast).
  -- Convert the ℤ-cast sum to a pure-ℕ sum first.
  have hpq' : ((p.1.val : ℤ) - q.1.val).natAbs + ((p.2.val : ℤ) - q.2.val).natAbs = 1 := by
    exact_mod_cast hpq
  -- Rewrite the goal's abs as natAbs via Int.abs_eq_natAbs, then apply omega.
  rw [show |((p.1.val : ℤ) + p.2.val) - ((q.1.val : ℤ) + q.2.val)|
        = (((p.1.val : ℤ) + p.2.val) - ((q.1.val : ℤ) + q.2.val)).natAbs from
      (Int.abs_eq_natAbs _)]
  -- Refactor the inner expression as (p.1 - q.1) + (p.2 - q.2).
  rw [show ((p.1.val : ℤ) + p.2.val) - ((q.1.val : ℤ) + q.2.val)
        = ((p.1.val : ℤ) - q.1.val) + ((p.2.val : ℤ) - q.2.val) from by ring]
  -- omega can now handle: from |d1|+|d2|=1 in ℕ, deduce |d1+d2| = 1 (since one is 0, other ±1).
  omega

end OrigamiCone
