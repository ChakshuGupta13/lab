/-
B2/B3 — `pairButterfly`: a single Cooley-Tukey butterfly lifted from
`Fin 2 → K` to act on coordinates `i, j` of `Fin n → K`, identity on the
rest. The difference `pairButterfly i j z - pairButterfly i j 0` is then
a rank-1 perturbation when `i ≠ j` and `z ≠ 0`, with image spanned by
`e i - e j`.
-/

import NttFaultRank.Basic

namespace NttFaultRank

variable {K : Type*} [Field K] {n : ℕ}

/-- A single Cooley-Tukey butterfly on coordinates `i, j` of an n-vector. -/
def pairButterfly (i j : Fin n) (z : K) : (Fin n → K) →ₗ[K] (Fin n → K) where
  toFun v := fun k =>
    if k = i then v i + z * v j
    else if k = j then v i - z * v j
    else v k
  map_add' x y := by
    funext k
    simp only [Pi.add_apply]
    by_cases hi : k = i
    · rw [hi]; simp; ring
    · by_cases hj : k = j
      · have hji : j ≠ i := fun h => hi (hj.trans h)
        rw [hj]; simp [hji]; ring
      · simp [hi, hj]
  map_smul' c x := by
    funext k
    simp only [Pi.smul_apply, smul_eq_mul]
    by_cases hi : k = i
    · rw [hi]; simp; ring
    · by_cases hj : k = j
      · have hji : j ≠ i := fun h => hi (hj.trans h)
        rw [hj]; simp [hji]; ring
      · simp [hi, hj]

@[simp] lemma pairButterfly_apply_i {i j : Fin n} (z : K) (v : Fin n → K) :
    pairButterfly i j z v i = v i + z * v j := by
  simp [pairButterfly]

@[simp] lemma pairButterfly_apply_j {i j : Fin n} (hij : i ≠ j)
    (z : K) (v : Fin n → K) :
    pairButterfly i j z v j = v i - z * v j := by
  simp [pairButterfly, hij.symm]

lemma pairButterfly_apply_other {i j k : Fin n} (z : K) (v : Fin n → K)
    (hki : k ≠ i) (hkj : k ≠ j) :
    pairButterfly i j z v k = v k := by
  simp [pairButterfly, hki, hkj]

/-- The difference `pairButterfly i j z - pairButterfly i j 0`. -/
def pairButterflyDiff (i j : Fin n) (z : K) : (Fin n → K) →ₗ[K] (Fin n → K) :=
  pairButterfly i j z - pairButterfly i j 0

@[simp] lemma pairButterflyDiff_apply_i {i j : Fin n} (z : K) (v : Fin n → K) :
    pairButterflyDiff i j z v i = z * v j := by
  simp [pairButterflyDiff, LinearMap.sub_apply]

@[simp] lemma pairButterflyDiff_apply_j {i j : Fin n} (hij : i ≠ j)
    (z : K) (v : Fin n → K) :
    pairButterflyDiff i j z v j = -(z * v j) := by
  simp [pairButterflyDiff, LinearMap.sub_apply, pairButterfly_apply_j hij]

lemma pairButterflyDiff_apply_other {i j k : Fin n} (z : K) (v : Fin n → K)
    (hki : k ≠ i) (hkj : k ≠ j) :
    pairButterflyDiff i j z v k = 0 := by
  simp [pairButterflyDiff, LinearMap.sub_apply,
        pairButterfly_apply_other z v hki hkj,
        pairButterfly_apply_other (0 : K) v hki hkj]

/-- The spanning direction `e i - e j`. -/
def pairDir (i j : Fin n) : Fin n → K := fun k =>
  if k = i then 1 else if k = j then -1 else 0

@[simp] lemma pairDir_apply_i (i j : Fin n) : (pairDir i j : Fin n → K) i = 1 := by
  simp [pairDir]

@[simp] lemma pairDir_apply_j {i j : Fin n} (hij : i ≠ j) :
    (pairDir i j : Fin n → K) j = -1 := by
  simp [pairDir, hij.symm]

lemma pairDir_apply_other {i j k : Fin n} (hki : k ≠ i) (hkj : k ≠ j) :
    (pairDir i j : Fin n → K) k = 0 := by
  simp [pairDir, hki, hkj]

/-- `pairDir i j` is nonzero. -/
lemma pairDir_ne_zero (i j : Fin n) : (pairDir i j : Fin n → K) ≠ 0 := by
  intro h
  have h0 : (pairDir i j : Fin n → K) i = (0 : Fin n → K) i := by rw [h]
  simp at h0

/-- Image of `pairButterflyDiff` is contained in the span of `pairDir i j`. -/
lemma pairButterflyDiff_range_le_span (i j : Fin n) (hij : i ≠ j) (z : K) :
    LinearMap.range (pairButterflyDiff i j z) ≤ K ∙ (pairDir i j : Fin n → K) := by
  rintro w ⟨v, rfl⟩
  refine Submodule.mem_span_singleton.mpr ⟨z * v j, ?_⟩
  funext k
  by_cases hi : k = i
  · subst hi
    rw [pairButterflyDiff_apply_i]
    simp
  · by_cases hj : k = j
    · subst hj
      rw [pairButterflyDiff_apply_j hij]
      simp [pairDir_apply_j hij]
    · rw [pairButterflyDiff_apply_other z v hi hj]
      simp [pairDir_apply_other hi hj]

/-- The input that is `1` at `j` and `0` elsewhere maps to `z • pairDir i j`. -/
lemma pairButterflyDiff_apply_basis (i j : Fin n) (hij : i ≠ j) (z : K) :
    pairButterflyDiff i j z (fun k => if k = j then 1 else 0)
      = z • (pairDir i j : Fin n → K) := by
  set v : Fin n → K := fun k => if k = j then 1 else 0 with hv
  have hvi : v i = 0 := by simp [hv, hij]
  have hvj : v j = 1 := by simp [hv]
  funext k
  by_cases hi : k = i
  · subst hi
    rw [pairButterflyDiff_apply_i, hvj]
    simp
  · by_cases hj : k = j
    · subst hj
      rw [pairButterflyDiff_apply_j hij, hvj]
      simp [pairDir_apply_j hij]
    · rw [pairButterflyDiff_apply_other z v hi hj]
      simp [pairDir_apply_other hi hj]

/-- When `i ≠ j` and `z ≠ 0`, the range equals `K ∙ pairDir i j`. -/
lemma pairButterflyDiff_range_eq_span {i j : Fin n} (hij : i ≠ j)
    {z : K} (hz : z ≠ 0) :
    LinearMap.range (pairButterflyDiff i j z) = K ∙ (pairDir i j : Fin n → K) := by
  apply le_antisymm (pairButterflyDiff_range_le_span i j hij z)
  have hu : IsUnit z := isUnit_iff_ne_zero.mpr hz
  have hspan_eq : (K ∙ (z • (pairDir i j : Fin n → K))) = K ∙ (pairDir i j : Fin n → K) :=
    Submodule.span_singleton_smul_eq hu (pairDir i j : Fin n → K)
  rw [← hspan_eq, Submodule.span_singleton_le_iff_mem]
  exact ⟨_, pairButterflyDiff_apply_basis i j hij z⟩

/-- **B3 — Per-pair-butterfly rank-1.** When `i ≠ j` and `z ≠ 0`, the
    lifted difference map has rank 1, with image `K ∙ (e i - e j)`. -/
theorem pairButterflyDiff_rank_eq_one {i j : Fin n} (hij : i ≠ j)
    {z : K} (hz : z ≠ 0) :
    Module.finrank K (LinearMap.range (pairButterflyDiff i j z)) = 1 := by
  rw [pairButterflyDiff_range_eq_span hij hz]
  exact finrank_span_singleton (pairDir_ne_zero i j)

end NttFaultRank

