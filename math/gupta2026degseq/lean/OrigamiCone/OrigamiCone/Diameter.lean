import OrigamiCone.Basic

/-!
# Median dispersion: the arithmetic backbone of the diameter (Section 4)

Formalisation of the self-contained median-dispersion machinery underlying
**Section 4 (The diameter)** of the paper
*"The origami flip graph of the m × n Miura-ori: degree sequence and diameter
via height functions"*.

The diameter section reads the OFG distance off the height representation as an
ℓ¹ median-dispersion (eq. `distformula`):
`dgrid_OFG(h, h') = ½ · min_K Σ_v |h'(v) − h(v) − K|`.
That identity rests on the shortest-recolouring-distance theorem of Johnson et
al. and Cereceda et al.; it is an **external** input and is **not** formalised
here.  What *is* formalised is the median-dispersion arithmetic the section's
own proofs use on top of that identity — in particular the even-offset scaling
step at the heart of the diameter lower bound (Theorem `thm:diam`):

> the difference `h_{--} − h_{++} = −2((i−1)+(j−1))` is even, so the convex map
> `K ↦ Σ |2 g(v) + K|` minimises at an even offset, and `½ min_K Σ |2 g(v) + K|
> = min_{K'} Σ |g(v) − K'|`.

We work with cells `Cell m n = Fin m × Fin n` and the dispersion at a fixed
offset `K`, then a predicate `IsMedianMin φ D` recording that `D` is the
minimised dispersion (a lower bound attained at some offset).  This avoids any
conditional-completeness machinery for `min over ℤ` while faithfully capturing
"`D = min_K Σ_v |φ(v) − K|`".

Results in this file:
* `disp`, `IsMedianMin` — definitions;
* `disp_nonneg` — dispersion is nonnegative;
* `medianMin_exists` — the minimised dispersion exists (the `min over ℤ` is
  attained), so `IsMedianMin φ` is non-vacuous for every `φ`;
* `disp_add_const`, `IsMedianMin.add_const` — translation invariance of the
  minimised dispersion (the "`ℓ¹` median-dispersion is unchanged by adding a
  constant" step of the lower-bound proof);
* `disp_two_mul`, `medianMin_two_mul` — the **even-offset scaling identity**:
  doubling the function doubles the minimised dispersion,
  `min_K Σ |2 g(v) − K| = 2 · min_{K'} Σ |g(v) − K'|`.  This is the arithmetic
  heart of the diameter lower bound (Theorem `thm:diam`): the difference
  `h_{--} − h_{++}` is even, so the minimising offset may be taken even and the
  factor `½` cancels the doubling.

No `sorry`.
-/

namespace OrigamiCone

variable {m n : ℕ}

/-- **Dispersion at offset `K`.** The ℓ¹ distance `Σ_v |φ(v) − K|` of an integer
function `φ` on the grid to the constant function `K`. -/
def disp (φ : Cell m n → ℤ) (K : ℤ) : ℤ := ∑ v : Cell m n, |φ v - K|

/-- Dispersion is nonnegative: a sum of absolute values. -/
lemma disp_nonneg (φ : Cell m n → ℤ) (K : ℤ) : 0 ≤ disp φ K :=
  Finset.sum_nonneg fun _ _ => abs_nonneg _

/-- **`D` is the minimised median-dispersion of `φ`**: it lower-bounds the
dispersion at every offset and is attained at some offset.  This is the formal
reading of `D = min_{K ∈ ℤ} Σ_v |φ(v) − K|`. -/
def IsMedianMin (φ : Cell m n → ℤ) (D : ℤ) : Prop :=
  (∀ K, D ≤ disp φ K) ∧ (∃ K, disp φ K = D)

/-- **The minimised dispersion exists.** The map `K ↦ disp φ K` attains its
minimum over `ℤ`: the set of achieved dispersion values is nonempty and bounded
below by `0`, hence has a least element.  So `D = min_K Σ_v |φ(v) − K|` is
well-defined for every `φ`. -/
lemma medianMin_exists (φ : Cell m n → ℤ) : ∃ D, IsMedianMin φ D := by
  classical
  obtain ⟨lb, ⟨K0, hK0⟩, hleast⟩ :=
    Int.exists_least_of_bdd (P := fun d => ∃ K, disp φ K = d)
      ⟨0, fun d hd => by obtain ⟨K, hK⟩ := hd; rw [← hK]; exact disp_nonneg φ K⟩
      ⟨disp φ 0, 0, rfl⟩
  exact ⟨lb, fun K => hleast (disp φ K) ⟨K, rfl⟩, K0, hK0⟩

/-- Shifting `φ` by a constant `c` shifts the minimising offset by `c`:
`disp (φ + c) K = disp φ (K − c)`. -/
lemma disp_add_const (φ : Cell m n → ℤ) (c K : ℤ) :
    disp (fun v => φ v + c) K = disp φ (K - c) := by
  unfold disp
  refine Finset.sum_congr rfl ?_
  intro v _
  congr 1
  ring

/-- **Translation invariance of the minimised dispersion.** If `D` is the
minimised dispersion of `φ`, it is also the minimised dispersion of `φ + c` for
any constant `c`.  (The "`ℓ¹` median-dispersion is unchanged by adding a constant
to the argument" step in the proof of Theorem `thm:diam`.) -/
lemma IsMedianMin.add_const {φ : Cell m n → ℤ} {D : ℤ}
    (h : IsMedianMin φ D) (c : ℤ) : IsMedianMin (fun v => φ v + c) D := by
  obtain ⟨hlb, K0, hK0⟩ := h
  refine ⟨fun K => ?_, K0 + c, ?_⟩
  · rw [disp_add_const]; exact hlb (K - c)
  · rw [disp_add_const]
    have hc : K0 + c - c = K0 := by ring
    rw [hc]; exact hK0

/-- **Doubling at an even offset.** `disp (2g) (2c) = 2 · disp g c`, since
`|2 g(v) − 2c| = 2 |g(v) − c|`. -/
lemma disp_two_mul (g : Cell m n → ℤ) (c : ℤ) :
    disp (fun v => 2 * g v) (2 * c) = 2 * disp g c := by
  unfold disp
  rw [Finset.mul_sum]
  refine Finset.sum_congr rfl fun v _ => ?_
  show |2 * g v - 2 * c| = 2 * |g v - c|
  rw [show 2 * g v - 2 * c = 2 * (g v - c) by ring, abs_mul]
  norm_num

/-- The per-cell identity behind the even-offset scaling: `2|2t − 1| = |2t| +
|2t − 2|`.  It holds for every integer `t`; it is stated for the doubled argument
`2t` because the values appearing in `disp (2g)` are exactly the even integers
`2 g(v)`, which is what lets the dispersion at an odd offset factor as the average
of the two adjacent even offsets. -/
private lemma two_abs_two_mul_odd (t : ℤ) :
    2 * |2 * t - 1| = |2 * t| + |2 * t - 2| := by
  rcases (by omega : t ≤ 0 ∨ 1 ≤ t) with h | h
  · rw [abs_of_nonpos (by omega : 2 * t - 1 ≤ 0),
        abs_of_nonpos (by omega : 2 * t ≤ 0),
        abs_of_nonpos (by omega : 2 * t - 2 ≤ 0)]
    ring
  · rw [abs_of_pos (by omega : (0 : ℤ) < 2 * t - 1),
        abs_of_pos (by omega : (0 : ℤ) < 2 * t),
        abs_of_nonneg (by omega : (0 : ℤ) ≤ 2 * t - 2)]
    ring

/-- **Even-offset scaling identity** (the arithmetic core of Theorem `thm:diam`).
If `D` is the minimised dispersion of `g`, then `2D` is the minimised dispersion
of `2g`.  Equivalently `min_K Σ_v |2 g(v) − K| = 2 · min_{K'} Σ_v |g(v) − K'|`: the
convex map `K ↦ Σ_v |2 g(v) − K|` minimises at an even offset, so the factor `½`
in the OFG distance formula cancels the doubling of the corner gradient
difference `h_{--} − h_{++} = −2((i−1)+(j−1))`. -/
theorem medianMin_two_mul {g : Cell m n → ℤ} {D : ℤ} (h : IsMedianMin g D) :
    IsMedianMin (fun v => 2 * g v) (2 * D) := by
  obtain ⟨hlb, K0, hK0⟩ := h
  refine ⟨fun K => ?_, 2 * K0, ?_⟩
  · -- lower bound: `2D ≤ disp (2g) K` for every offset `K`, by parity of `K`
    obtain ⟨k, hk | hk⟩ := Int.even_or_odd' K
    · -- even offset `K = 2k`
      subst hk
      rw [disp_two_mul g k]
      have := hlb k; omega
    · -- odd offset `K = 2k+1`: average of the two adjacent even offsets
      subst hk
      have key : 2 * disp (fun v => 2 * g v) (2 * k + 1)
          = disp (fun v => 2 * g v) (2 * k) + disp (fun v => 2 * g v) (2 * (k + 1)) := by
        unfold disp
        rw [Finset.mul_sum, ← Finset.sum_add_distrib]
        refine Finset.sum_congr rfl fun v _ => ?_
        show 2 * |2 * g v - (2 * k + 1)| = |2 * g v - 2 * k| + |2 * g v - 2 * (k + 1)|
        rw [show 2 * g v - (2 * k + 1) = 2 * (g v - k) - 1 by ring,
            show 2 * g v - 2 * k = 2 * (g v - k) by ring,
            show 2 * g v - 2 * (k + 1) = 2 * (g v - k) - 2 by ring]
        exact two_abs_two_mul_odd (g v - k)
      have e1 := disp_two_mul g k
      have e2 := disp_two_mul g (k + 1)
      have b1 := hlb k
      have b2 := hlb (k + 1)
      omega
  · -- attainment at the even offset `2 K0`
    rw [disp_two_mul g K0, hK0]

end OrigamiCone
