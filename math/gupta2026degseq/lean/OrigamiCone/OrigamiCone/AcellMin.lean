import OrigamiCone.AcellGradient
import OrigamiCone.ConeClassification

/-!
# `acell` has unique strict local min at `(0, 0)`

Companion to `AcellGradient.lean`'s `acell_unique_max`: the antidiagonal
`acell` attains a strict local minimum **uniquely** at the bottom-left corner
`(0, 0)`.

Mechanism: via `acell_eq_coneC_top`, `acell = coneC ⟨m-1, n-1⟩ (m+n-2)`.
Strict local minima of `coneC q C` are exactly strict local maxima of the
distance function `Dq q` (= `gdist q ·`) — by `coneC_min_iff_Dq_max`
(`Degree2.lean`).  For `q = ⟨m-1, n-1⟩` (the top-right corner), the
strict local maxima of `Dq q` are characterised by `Dq_strictMax_iff`
(`ConeClassification.lean`) as cells `v` where each coordinate of `v` is
a path endpoint on the **far side** of `q`.  Both coordinates of `q` are
at their right ends, so the far-side endpoint in each coordinate is `0`,
giving the unique strict local min at `(0, 0)`.

Requires `m, n ≥ 2` (the same regime as `Dq_strictMax_iff`).

Result:
* `acell_unique_min` — `(0, 0)` is the unique strict local min of `acell`.

No `sorry`.
-/

namespace OrigamiCone

variable {m n : ℕ}

/-- **`acell` has unique strict local min at the bottom-left corner.**

For `m, n ≥ 2`, the cell `(⟨0, _⟩, ⟨0, _⟩)` is the only strict local minimum
of `acell`.  Dual to `acell_unique_max`: via the cone identification
`acell = coneC ⟨m-1, n-1⟩ (m+n-2)`, the minima of `acell` are the
strict local maxima of `Dq ⟨m-1, n-1⟩`, which lie at the far-side
endpoints in each coordinate — both `0` for the top-right apex. -/
theorem acell_unique_min (hm : 2 ≤ m) (hn : 2 ≤ n) :
    ∀ v', IsStrictLocalMin (acell (m := m) (n := n)) v'
      → v' = (⟨0, by omega⟩, ⟨0, by omega⟩) := by
  intro v' hv'
  -- Rewrite acell as the corner cone, then use coneC_min_iff_Dq_max.
  rw [acell_eq_coneC_top (by omega) (by omega)] at hv'
  rw [coneC_min_iff_Dq_max] at hv'
  -- hv' : IsStrictLocalMax (Dq (⟨m-1, _⟩, ⟨n-1, _⟩)) v'
  -- Apply Dq_strictMax_iff to get PathEnd conditions on each coordinate.
  rw [Dq_strictMax_iff hm hn] at hv'
  obtain ⟨hr, hc⟩ := hv'
  -- hr : PathEnd ⟨m-1, _⟩ v'.1 = (v'.1.val = 0 ∧ (m-1) ≠ 0) ∨ (v'.1.val + 1 = m ∧ (m-1) + 1 ≠ m)
  -- For m ≥ 2, the second disjunct's (m-1) + 1 = m, so (m-1) + 1 ≠ m is False.
  -- Hence v'.1.val = 0.
  -- Symmetric for column.
  unfold PathEnd at hr hc
  -- Reduce the Fin.mk projections (⟨m-1, _⟩.val = m-1 etc.) so omega can see them.
  dsimp only at hr hc
  -- Extract v'.1.val = 0 and v'.2.val = 0.
  have hr0 : v'.1.val = 0 := by
    rcases hr with ⟨h0, _⟩ | ⟨_, hne⟩
    · exact h0
    · exfalso; apply hne; omega
  have hc0 : v'.2.val = 0 := by
    rcases hc with ⟨h0, _⟩ | ⟨_, hne⟩
    · exact h0
    · exfalso; apply hne; omega
  -- Conclude v' = (⟨0, _⟩, ⟨0, _⟩) by Fin extension.
  ext
  · show v'.1.val = 0; exact hr0
  · show v'.2.val = 0; exact hc0

end OrigamiCone
