import OrigamiCone.Diameter

/-!
# The diameter quantity `D(m,n)` and the lower-bound arithmetic (Section 4)

Formalisation of the **self-contained arithmetic** of the diameter lower bound
(Theorem `thm:diam`) of Section 4, building on the median-dispersion machinery of
`Diameter.lean`.

The paper's lower bound `diam OFG(M_{m,n}) ≥ D(m,n)` is obtained by exhibiting two
explicit vertices — the opposite corner gradients `h₊₊(i,j) = (i−1)+(j−1)` and
`h₋₋ = −h₊₊` — whose OFG distance equals `D(m,n) = min_K Σ_{i,j} |(i+j) − K|`.
By the distance formula (`eq:distformula`, an **external** input citing Johnson et
al.) that distance is `½ min_K Σ_v |(h₋₋ − h₊₊)(v) − K|`, and the difference is
`h₋₋ − h₊₊ = −2·a` where `a(v) = i + j` is the antidiagonal.  The arithmetic
content — everything except the external distance formula — is:

> `min_K Σ_v |−2·a(v) − K| = 2 · D(m,n)`, so the halved distance is `D(m,n)`.

This file proves exactly that, plus that `D(m,n)` is **well-defined** (the
median-minimum is unique) and **symmetric** (`D(m,n) = D(n,m)`).  The remaining
step to `diam ≥ D` is the external distance formula, which is **not** formalised
here (it requires the recolouring-reconfiguration distance of Cereceda–van den
Heuvel–Johnson, a separate development).

We index cells `0`-based, so the antidiagonal is `a(i,j) = i + j` and the paper's
`1`-based `(i−1)+(j−1)` is the same function; median-dispersion is
translation-invariant, so the shift is immaterial.

Results:
* `disp_neg`, `IsMedianMin.neg` — dispersion is invariant under sign flip;
* `medianMin_unique` — the minimised dispersion is unique, so `D(m,n)` is
  well-defined;
* `acell` — the antidiagonal `a(v) = i + j`;
* `cornerGradient_medianMin` — the corner-gradient difference `−2·a` has
  minimised dispersion `2·D`, the arithmetic core of `thm:diam`;
* `medianMin_swap` — `D(m,n) = D(n,m)`.

No `sorry`.
-/

namespace OrigamiCone

variable {m n : ℕ}

/-- Sign flip on the offset: `disp (−φ) K = disp φ (−K)`. -/
lemma disp_neg (φ : Cell m n → ℤ) (K : ℤ) :
    disp (fun v => -φ v) K = disp φ (-K) := by
  unfold disp
  refine Finset.sum_congr rfl fun v _ => ?_
  rw [show -φ v - K = -(φ v - (-K)) by ring, abs_neg]

/-- **Dispersion is invariant under sign flip.** If `D` is the minimised
dispersion of `φ`, it is also the minimised dispersion of `−φ`.  (`ℓ¹`
median-dispersion is symmetric: reflecting the values reflects the optimal
offset.) -/
lemma IsMedianMin.neg {φ : Cell m n → ℤ} {D : ℤ} (h : IsMedianMin φ D) :
    IsMedianMin (fun v => -φ v) D := by
  obtain ⟨hlb, K0, hK0⟩ := h
  refine ⟨fun K => ?_, -K0, ?_⟩
  · rw [disp_neg]; exact hlb (-K)
  · rw [disp_neg]
    have : - -K0 = K0 := by ring
    rw [this]; exact hK0

/-- **The minimised dispersion is unique.** Hence `D(m,n)`, the value of
`IsMedianMin` for the antidiagonal, is a well-defined integer. -/
lemma medianMin_unique {φ : Cell m n → ℤ} {D D' : ℤ}
    (h : IsMedianMin φ D) (h' : IsMedianMin φ D') : D = D' := by
  obtain ⟨hlb, K0, hK0⟩ := h
  obtain ⟨hlb', K0', hK0'⟩ := h'
  have h1 : D ≤ D' := hK0' ▸ hlb K0'
  have h2 : D' ≤ D := hK0 ▸ hlb' K0
  omega

/-- The **antidiagonal** `a(v) = i + j` (0-based), as an integer function on the
grid.  `D(m,n)` is the minimised dispersion of `acell`. -/
def acell (v : Cell m n) : ℤ := (v.1.val : ℤ) + (v.2.val : ℤ)

/-- **Arithmetic core of the diameter lower bound** (Theorem `thm:diam`).
If `D` is the minimised dispersion of the antidiagonal `a`, then the
corner-gradient difference `h₋₋ − h₊₊ = −2·a` has minimised dispersion `2·D`.
Combined with the external distance formula `dist = ½ min_K Σ|·−K|`, the OFG
distance between the two opposite corner gradients is `D(m,n)`, giving
`diam ≥ D(m,n)`.  The halving in the distance formula cancels the doubling here;
this is the entire arithmetic of the lower bound. -/
theorem cornerGradient_medianMin {D : ℤ} (hD : IsMedianMin (acell (m := m) (n := n)) D) :
    IsMedianMin (fun v : Cell m n => -2 * acell v) (2 * D) := by
  have h2 := medianMin_two_mul hD.neg
  have hfun : (fun v : Cell m n => 2 * -acell v) = (fun v : Cell m n => -2 * acell v) := by
    funext v; ring
  rwa [hfun] at h2

/-- The coordinate swap `Cell m n ≃ Cell n m` carries the antidiagonal to the
antidiagonal: `a(swap v) = a(v)`. -/
private lemma acell_prodComm (v : Cell m n) :
    acell ((Equiv.prodComm (Fin m) (Fin n)) v) = acell v := by
  simp only [acell, Equiv.prodComm_apply, Prod.fst_swap, Prod.snd_swap]
  ring

/-- **Symmetry of the diameter quantity** `D(m,n) = D(n,m)`.  The grids
`G_{m,n}` and `G_{n,m}` are isomorphic by swapping coordinates, and the
antidiagonal is symmetric, so the minimised dispersions agree. -/
lemma medianMin_swap {D : ℤ} (h : IsMedianMin (acell (m := m) (n := n)) D) :
    IsMedianMin (acell (m := n) (n := m)) D := by
  have key : ∀ K, disp (acell (m := n) (n := m)) K = disp (acell (m := m) (n := n)) K := by
    intro K
    unfold disp
    rw [← Equiv.sum_comp (Equiv.prodComm (Fin m) (Fin n))
          (fun v => |acell (m := n) (n := m) v - K|)]
    refine Finset.sum_congr rfl fun v _ => ?_
    rw [acell_prodComm]
  obtain ⟨hlb, K0, hK0⟩ := h
  exact ⟨fun K => (key K).symm ▸ hlb K, K0, (key K0).symm ▸ hK0⟩

end OrigamiCone
