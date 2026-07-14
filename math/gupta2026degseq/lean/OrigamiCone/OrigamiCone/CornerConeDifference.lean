import OrigamiCone.AcellDual
import OrigamiCone.DiameterLower

/-!
# Corner-cone difference has median-min `2 · D(m, n)`

The paper's `thm:diam` lower-bound argument exhibits two opposite corner
gradients `h₊₊` and `h₋₋ = −h₊₊` as the witnesses whose OFG distance equals
`D(m, n)`.  In the cone-classification machinery (`Degree2.lean`):

* `h₊₊ = coneC ⟨m − 1, n − 1⟩ (m + n − 2)`   (top-right corner, peak `m + n − 2`)
* `h₋₋ = coneC ⟨0, 0⟩ 0`                       (bottom-left corner, peak `0`)

This module packages the diameter lower-bound arithmetic in that cone
language by combining three pre-existing pieces:

* `acell_eq_coneC_top`        (AcellGradient.lean, identifies `h₊₊` with `acell`),
* `negAcell_eq_coneC_bottom`  (AcellDual.lean,     identifies `h₋₋` with `−acell`),
* `cornerGradient_medianMin`  (DiameterLower.lean, the `−2 · acell` arithmetic).

The packaged statement says: the difference of the two corner cones,
`h₋₋ − h₊₊`, has minimised dispersion `2 · D(m, n)`.  Combined with the
external recolouring-distance bridge (Johnson 2016, `eq:distformula`) this
gives `diam OFG(M_{m, n}) ≥ D(m, n)`.  The external bridge is **not**
formalised here.

Result:
* `coneCornerDifference_medianMin` — the packaged statement.

No `sorry`.
-/

namespace OrigamiCone

variable {m n : ℕ}

/-- **Diameter lower-bound arithmetic in cone form.**

For `m, n ≥ 1`, the pointwise difference of the bottom-left corner cone
`h₋₋ = coneC ⟨0, 0⟩ 0` and the top-right corner cone
`h₊₊ = coneC ⟨m − 1, n − 1⟩ (m + n − 2)` has minimised dispersion `2 · D`,
where `D` is the minimised dispersion of the antidiagonal `acell`.

Combined with the external recolouring-distance theorem (Johnson 2016,
`eq:distformula`) — which says `dist φ ψ = ½ · min_K Σ_v |(ψ − φ) v − K|`
— this gives `diam OFG(M_{m, n}) ≥ D(m, n)`.  The halving in the distance
formula cancels the doubling here; the entire arithmetic of the paper's
`thm:diam` lower bound is packaged in this single statement.

Proof: the cone-cone difference unfolds pointwise to `−2 · acell` by
`acell_eq_coneC_top` (top-right) and `negAcell_eq_coneC_bottom`
(bottom-left); apply `cornerGradient_medianMin`. -/
theorem coneCornerDifference_medianMin (hm : 1 ≤ m) (hn : 1 ≤ n)
    {D : ℤ} (hD : IsMedianMin (acell (m := m) (n := n)) D) :
    IsMedianMin
      (fun v : Cell m n =>
         coneC (⟨0, by omega⟩, ⟨0, by omega⟩) (0 : ℤ) v
         - coneC (⟨m - 1, by omega⟩, ⟨n - 1, by omega⟩) ((m + n - 2 : ℕ) : ℤ) v)
      (2 * D) := by
  -- Rewrite the cone-cone difference as `−2 · acell` pointwise, then apply
  -- the arithmetic lower-bound theorem from `DiameterLower.lean`.
  have h_eq :
      (fun v : Cell m n =>
         coneC (⟨0, by omega⟩, ⟨0, by omega⟩) (0 : ℤ) v
         - coneC (⟨m - 1, by omega⟩, ⟨n - 1, by omega⟩) ((m + n - 2 : ℕ) : ℤ) v)
      = (fun v : Cell m n => -2 * acell v) := by
    funext v
    -- `h_bot : -acell v = coneC ⟨0, _⟩ ⟨0, _⟩ 0 v`
    have h_bot := congr_fun (negAcell_eq_coneC_bottom hm hn) v
    -- `h_top : acell v = coneC ⟨m-1, _⟩ ⟨n-1, _⟩ (m+n-2) v`
    have h_top := congr_fun (acell_eq_coneC_top hm hn) v
    linarith
  rw [h_eq]
  exact cornerGradient_medianMin hD

end OrigamiCone
