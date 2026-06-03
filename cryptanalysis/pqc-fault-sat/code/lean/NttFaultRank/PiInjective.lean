/-
Phase F8 — Image-independence via scalar projection.

Paper Lemma `lem:piscalar` (main.tex lines 425-490): the bit-channel
projection π_ℓ acts as the invertible scalar `(1/2)^ℓ`, hence the
seven `T_ℓ` images are linearly independent.

  Strategy (4 steps, per F8 blocker analysis):
    1. `freshIdx hk s := ⟨L_k + s.val, _⟩` (paper's `[0^k | 1 | s]`).
       Bit-pattern lemmas: `bitOf k (freshIdx hk s) = 1`,
       `bitOf k' (freshIdx hk s) = 0` for `k' < k`.
    2. Layer-`k` base case: at `bSideIdx ⟨k,_⟩ g₀ s`,
       `layerInv ⟨k,_⟩ (pert (runPrefix Z' k w)) = bSide (runPrefix Z' k w) ⟨k,_⟩ g₀ s`.
    3. `(1/2)^m` scalar accumulation through `runPrefixInv` (induction on m).
    4. SupIndep peeling from top index using Steps 1-3.
-/

import NttFaultRank.IntImageInV

namespace NttFaultRank

variable {K : Type*} [Field K]

namespace LayerSpec
variable (spec : LayerSpec)

/-! ### Step 1 — `freshIdx` and its bit-pattern -/

/-- Useful: `0 < spec.G ⟨k, hk⟩` (the group count is positive when `n > 0`). -/
lemma G_pos (k : ℕ) (hk : k < spec.N) (hn : 0 < spec.n) :
    0 < spec.G ⟨k, hk⟩ := by
  have hgl := spec.hGL ⟨k, hk⟩
  rcases Nat.eq_zero_or_pos (spec.G ⟨k, hk⟩) with hz | hp
  · exfalso; rw [hz] at hgl; simp at hgl; omega
  · exact hp

/-- Useful: `0 < spec.L ⟨k, hk⟩`. -/
lemma L_pos (k : ℕ) (hk : k < spec.N) (hn : 0 < spec.n) :
    0 < spec.L ⟨k, hk⟩ := by
  have hgl := spec.hGL ⟨k, hk⟩
  rcases Nat.eq_zero_or_pos (spec.L ⟨k, hk⟩) with hz | hp
  · exfalso; rw [hz] at hgl; simp at hgl; omega
  · exact hp

/-- `2 * L ⟨k, hk⟩ ≤ spec.n` for `KyberLike spec` (matches `two_L_le_n` from F5). -/
lemma two_L_le_n_F8 [KyberLike spec] (k : ℕ) (hk : k < spec.N) :
    2 * spec.L ⟨k, hk⟩ ≤ spec.n := by
  have h0 : 0 < spec.N := lt_of_le_of_lt (Nat.zero_le _) hk
  have hL0 : 2 * spec.L ⟨0, h0⟩ = spec.n := KyberLike.L_zero_doubled h0
  have hanti : spec.L ⟨k, hk⟩ ≤ spec.L ⟨0, h0⟩ :=
    spec.L_antitone hk h0 (Nat.zero_le _)
  omega

/-- The "fresh" flat index `[0^k | 1 | s]`: value `L_k + s.val`. -/
def freshIdx [KyberLike spec] {k : ℕ} (hk : k < spec.N) (s : Fin (spec.L ⟨k, hk⟩)) :
    Fin spec.n :=
  ⟨spec.L ⟨k, hk⟩ + s.val,
    lt_of_lt_of_le (by have := s.isLt; omega) (spec.two_L_le_n_F8 k hk)⟩

@[simp] lemma freshIdx_val [KyberLike spec] {k : ℕ} (hk : k < spec.N)
    (s : Fin (spec.L ⟨k, hk⟩)) :
    (spec.freshIdx hk s).val = spec.L ⟨k, hk⟩ + s.val := rfl

/-- `bitOf k (freshIdx hk s) = 1`. -/
lemma bitOf_freshIdx_self [KyberLike spec] {k : ℕ} (hk : k < spec.N)
    (s : Fin (spec.L ⟨k, hk⟩)) :
    spec.bitOf ⟨k, hk⟩ (spec.freshIdx hk s) = 1 := by
  unfold bitOf
  rw [spec.freshIdx_val]
  have hL_pos : 0 < spec.L ⟨k, hk⟩ := spec.L_pos k hk
    (lt_of_le_of_lt (Nat.zero_le _) (spec.freshIdx hk s).isLt)
  have hs := s.isLt
  -- (L + s) / L = 1 since L ≤ L + s < 2 L; then 1 % 2 = 1.
  have hdiv : (spec.L ⟨k, hk⟩ + s.val) / spec.L ⟨k, hk⟩ = 1 := by
    rw [show spec.L ⟨k, hk⟩ + s.val = s.val + 1 * spec.L ⟨k, hk⟩ from by ring]
    rw [Nat.add_mul_div_right _ _ hL_pos, Nat.div_eq_of_lt hs]
  rw [hdiv]

/-- `bitOf k' (freshIdx hk s) = 0` for any `k' < k`.
    Uses `two_L_dvd_L`: `L_{k'} ≥ 2 * L_k` so `L_k + s < L_{k'}`. -/
lemma bitOf_freshIdx_lower [KyberLike spec] {k k' : ℕ}
    (hk : k < spec.N) (hk' : k' < spec.N) (hk'k : k' < k)
    (s : Fin (spec.L ⟨k, hk⟩)) :
    spec.bitOf ⟨k', hk'⟩ (spec.freshIdx hk s) = 0 := by
  apply spec.bitOf_eq_zero_of_lt_L
  -- freshIdx.val = L_k + s.val < 2 * L_k ≤ L_{k'}.
  obtain ⟨m, hm⟩ := spec.two_L_dvd_L hk' hk hk'k
  have hLk_pos : 0 < spec.L ⟨k, hk⟩ := spec.L_pos k hk
    (lt_of_le_of_lt (Nat.zero_le _) (spec.freshIdx hk s).isLt)
  have hm_pos : 0 < m := by
    by_contra hzero
    push_neg at hzero
    interval_cases m
    · simp at hm
      have hk'_pos := spec.L_pos k' hk' (lt_of_le_of_lt (Nat.zero_le _)
        (spec.freshIdx hk s).isLt)
      omega
  rw [spec.freshIdx_val, hm]
  have hs := s.isLt
  nlinarith

/-! ### Step 2 — Layer-`k` base case: value formula at `bSideIdx g₀ s` -/

/-- For any vector `v`, the b-side value at group `g` of `layerInv (pert v)`
    is `(Z ⟨k,_⟩ g − Z' ⟨k,_⟩ g) / Z ⟨k,_⟩ g · bSide v ⟨k,_⟩ g j`. -/
lemma bSide_layerInv_pert
    {Z : spec.Twiddles K} (hZ : ∀ ℓ g, Z ℓ g ≠ 0) (h2 : (2 : K) ≠ 0)
    (F : spec.FaultSet) (k : ℕ) (hk : k < spec.N)
    (v : Fin spec.n → K) (g : Fin (spec.G ⟨k, hk⟩))
    (j : Fin (spec.L ⟨k, hk⟩)) :
    spec.bSide
        ((spec.layerInv Z ⟨k, hk⟩ (fun g => hZ ⟨k, hk⟩ g) h2)
          (spec.perturbation Z F ⟨k, hk⟩ v))
        ⟨k, hk⟩ g j
      = (Z ⟨k, hk⟩ g - (spec.faultedTwiddles Z F) ⟨k, hk⟩ g) / (Z ⟨k, hk⟩ g)
          * spec.bSide v ⟨k, hk⟩ g j := by
  rw [spec.bSide_layerInv Z ⟨k, hk⟩ (fun g => hZ ⟨k, hk⟩ g) h2 _ g j,
      spec.aSide_perturbation Z F ⟨k, hk⟩ v g j,
      spec.bSide_perturbation Z F ⟨k, hk⟩ v g j]
  -- (((Z g - Z' g) - (Z' g - Z g)) * bSide v) / (2 * Z g)
  -- = (2 * (Z g - Z' g) * bSide v) / (2 * Z g)
  -- = ((Z g - Z' g) / Z g) * bSide v.
  have hZg : Z ⟨k, hk⟩ g ≠ 0 := hZ ⟨k, hk⟩ g
  have h2Zg : (2 : K) * Z ⟨k, hk⟩ g ≠ 0 := mul_ne_zero h2 hZg
  -- Cleaner: clear denominators via field_simp.
  field_simp
  ring

/-- **Step 2 headline.** At the b-side of the faulted group `g₀`, the value of
    `layerInv (pert (runPrefix Z' k w))` equals `bSide (runPrefix Z' k w) ⟨k,_⟩ g₀ s`. -/
lemma bSide_layerInv_pert_g0
    {Z : spec.Twiddles K} (hZ : ∀ ℓ g, Z ℓ g ≠ 0) (h2 : (2 : K) ≠ 0)
    {F : spec.FaultSet} (hF : spec.OneFaultPerLayer F)
    (k : ℕ) (hk : k < spec.N)
    (w : Fin spec.n → K)
    (s : Fin (spec.L ⟨k, hk⟩)) :
    spec.bSide
        ((spec.layerInv Z ⟨k, hk⟩ (fun g => hZ ⟨k, hk⟩ g) h2)
          (spec.perturbation Z F ⟨k, hk⟩ w))
        ⟨k, hk⟩ (Finset.card_eq_one.mp (hF ⟨k, hk⟩)).choose s
      = spec.bSide w ⟨k, hk⟩ (Finset.card_eq_one.mp (hF ⟨k, hk⟩)).choose s := by
  set g₀ := (Finset.card_eq_one.mp (hF ⟨k, hk⟩)).choose with hg₀_def
  rw [spec.bSide_layerInv_pert hZ h2 F k hk w g₀ s]
  -- (Z g₀ - Z' g₀) / Z g₀ = 1 since Z' g₀ = 0.
  have hZ'_zero : (spec.faultedTwiddles Z F) ⟨k, hk⟩ g₀ = 0 := by
    have hg₀_spec : F ⟨k, hk⟩ = {g₀} :=
      (Finset.card_eq_one.mp (hF ⟨k, hk⟩)).choose_spec
    unfold faultedTwiddles
    simp [hg₀_spec]
  rw [hZ'_zero, sub_zero, div_self (hZ ⟨k, hk⟩ g₀), one_mul]

/-- Mirror: at b-side of any group `g ≠ g₀`, the value is zero. -/
lemma bSide_layerInv_pert_off
    {Z : spec.Twiddles K} (hZ : ∀ ℓ g, Z ℓ g ≠ 0) (h2 : (2 : K) ≠ 0)
    {F : spec.FaultSet} (hF : spec.OneFaultPerLayer F)
    (k : ℕ) (hk : k < spec.N) (w : Fin spec.n → K)
    (g : Fin (spec.G ⟨k, hk⟩))
    (hg : g ≠ (Finset.card_eq_one.mp (hF ⟨k, hk⟩)).choose)
    (s : Fin (spec.L ⟨k, hk⟩)) :
    spec.bSide
        ((spec.layerInv Z ⟨k, hk⟩ (fun g => hZ ⟨k, hk⟩ g) h2)
          (spec.perturbation Z F ⟨k, hk⟩ w))
        ⟨k, hk⟩ g s = 0 := by
  rw [spec.bSide_layerInv_pert hZ h2 F k hk w g s]
  -- (Z g - Z' g) = 0 since Z' g = Z g.
  have hZ'_eq : (spec.faultedTwiddles Z F) ⟨k, hk⟩ g = Z ⟨k, hk⟩ g := by
    have hg₀_spec : F ⟨k, hk⟩ = {(Finset.card_eq_one.mp (hF ⟨k, hk⟩)).choose} :=
      (Finset.card_eq_one.mp (hF ⟨k, hk⟩)).choose_spec
    show (if g ∈ F ⟨k, hk⟩ then 0 else Z ⟨k, hk⟩ g) = Z ⟨k, hk⟩ g
    rw [hg₀_spec, if_neg (by simpa using hg)]
  rw [hZ'_eq, sub_self, zero_div, zero_mul]

/-! ### Step 3a — Sum-of-`L_i` bound

  The invariant `(∑ i : Fin m, L_i) + 2 L_m = n` holds for every `m < N`,
  because `L_{m-1} = 2 L_m` absorbs into the trailing `2 L_m` term:
  `(∑_{i<m} L_i) + 2 L_m = (∑_{i<m-1} L_i) + L_{m-1} + 2 L_m
                         = (∑_{i<m-1} L_i) + 2 L_{m-1}`.
  At the base `m = 0`: `0 + 2 L_0 = n` (KyberLike.L_zero_doubled).
  Combined with `L_k ≤ L_m` for `m ≤ k` (antitonicity), we get the bound
  needed for `orbitIdx`. -/

lemma sum_Li_plus_2L_eq_n [KyberLike spec] :
    ∀ (m : ℕ) (hm : m < spec.N),
      (∑ i : Fin m, spec.L ⟨i.val, lt_of_lt_of_le i.isLt hm.le⟩)
        + 2 * spec.L ⟨m, hm⟩ = spec.n := by
  intro m hm
  induction m with
  | zero =>
    -- ∑ over Fin 0 = 0, then 2 L_0 = n.
    simp
    exact KyberLike.L_zero_doubled hm
  | succ m ih =>
    -- ∑_{i:Fin (m+1)} L_i + 2 L_{m+1}
    --   = (∑_{i:Fin m} L_i.castSucc) + L_⟨m,_⟩ + 2 L_{m+1}
    --   = (∑_{i:Fin m} L_i.castSucc) + L_⟨m,_⟩ + L_⟨m,_⟩  [by L_succ]
    --   = (∑_{i:Fin m} L_i.castSucc) + 2 L_⟨m,_⟩
    -- and this equals n by ih.
    have hm_prev : m < spec.N := Nat.lt_of_succ_lt hm
    have hLs := KyberLike.L_succ (spec := spec) m hm
    have ih' := ih hm_prev
    rw [Fin.sum_univ_castSucc]
    show (∑ i : Fin m, spec.L ⟨i.val, _⟩) + spec.L ⟨m, hm_prev⟩
            + 2 * spec.L ⟨m + 1, hm⟩ = spec.n
    have hL_eq : spec.L ⟨m, hm_prev⟩ + 2 * spec.L ⟨m + 1, hm⟩
                = 2 * spec.L ⟨m, hm_prev⟩ := by omega
    -- Rewrite (∑ + L_m + 2 L_{m+1}) = (∑ + (L_m + 2 L_{m+1})) = (∑ + 2 L_m).
    rw [show (∑ i : Fin m, spec.L ⟨i.val, _⟩) + spec.L ⟨m, hm_prev⟩
            + 2 * spec.L ⟨m + 1, hm⟩
          = (∑ i : Fin m, spec.L ⟨i.val, _⟩)
            + (spec.L ⟨m, hm_prev⟩ + 2 * spec.L ⟨m + 1, hm⟩) from by ring]
    rw [hL_eq]
    exact ih'

/-- The bound used in `orbitIdx`: for `m ≤ k < N`,
    `(∑_{i:Fin m} L_i) + 2 L_k ≤ n`. -/
lemma sum_Li_plus_2Lk_le_n [KyberLike spec]
    {k : ℕ} (hk : k < spec.N) {m : ℕ} (hmk : m ≤ k) :
    (∑ i : Fin m, spec.L ⟨i.val,
        lt_of_lt_of_le i.isLt (le_trans hmk hk.le)⟩)
      + 2 * spec.L ⟨k, hk⟩ ≤ spec.n := by
  have hm : m < spec.N := lt_of_le_of_lt hmk hk
  have heq := spec.sum_Li_plus_2L_eq_n m hm
  have hanti : spec.L ⟨k, hk⟩ ≤ spec.L ⟨m, hm⟩ :=
    spec.L_antitone hk hm hmk
  omega

/-! ### Step 3b — `orbitIdx`

  The orbit of `freshIdx hk s` under `m` inverse butterflies at layers
  `0, …, m-1`. Each bit `i` of `y : Fin (2^m)` toggles whether the
  position has been "flipped" at layer `i` (i.e. partner-jumped by `+ L_i`).
  All `2^m` orbit positions remain in `Fin spec.n` by the 3a bound. -/

/-- The offset added to `freshIdx hk s` by orbit index `y : Fin (2^m)`. -/
def orbitOffset [KyberLike spec] {k : ℕ} (hk : k < spec.N)
    (m : ℕ) (hmk : m ≤ k) (y : ℕ) : ℕ :=
  ∑ i : Fin m, if y.testBit i.val
    then spec.L ⟨i.val, lt_of_lt_of_le i.isLt (le_trans hmk hk.le)⟩ else 0

/-- The orbit-offset is bounded by the sum of all `L_i` for `i < m`. -/
lemma orbitOffset_le_sum [KyberLike spec] {k : ℕ} (hk : k < spec.N)
    {m : ℕ} (hmk : m ≤ k) (y : ℕ) :
    spec.orbitOffset hk m hmk y
      ≤ ∑ i : Fin m, spec.L ⟨i.val,
          lt_of_lt_of_le i.isLt (le_trans hmk hk.le)⟩ := by
  unfold orbitOffset
  apply Finset.sum_le_sum
  intro i _
  split_ifs
  · rfl
  · exact Nat.zero_le _

/-- The `y`-th orbit position of `freshIdx hk s`. -/
def orbitIdx [KyberLike spec] {k : ℕ} (hk : k < spec.N)
    (s : Fin (spec.L ⟨k, hk⟩))
    (m : ℕ) (hmk : m ≤ k) (y : Fin (2^m)) : Fin spec.n :=
  ⟨spec.L ⟨k, hk⟩ + s.val + spec.orbitOffset hk m hmk y.val,
   by
    have hbound := spec.sum_Li_plus_2Lk_le_n hk hmk
    have hoffset := spec.orbitOffset_le_sum hk hmk y.val
    have hs := s.isLt
    omega⟩

@[simp] lemma orbitIdx_val [KyberLike spec] {k : ℕ} (hk : k < spec.N)
    (s : Fin (spec.L ⟨k, hk⟩))
    (m : ℕ) (hmk : m ≤ k) (y : Fin (2^m)) :
    (spec.orbitIdx hk s m hmk y).val
      = spec.L ⟨k, hk⟩ + s.val + spec.orbitOffset hk m hmk y.val := rfl

/-- At `m = 0`, the only orbit element is `freshIdx hk s` itself. -/
lemma orbitIdx_zero [KyberLike spec] {k : ℕ} (hk : k < spec.N)
    (s : Fin (spec.L ⟨k, hk⟩)) (y : Fin 1) :
    spec.orbitIdx hk s 0 (Nat.zero_le _) y = spec.freshIdx hk s := by
  apply Fin.ext
  show spec.L ⟨k, hk⟩ + s.val + spec.orbitOffset hk 0 _ y.val
       = spec.L ⟨k, hk⟩ + s.val
  unfold orbitOffset
  simp

/-! ### Step 3c — Universal scalar formula via `runPrefixInv` induction -/

/-- For `i < m`, `L_i = 2^(m-i) * L_m` (Kyber recurrence iterated). -/
lemma L_eq_pow_L_succ [KyberLike spec] :
    ∀ {m i : ℕ} (hm : m < spec.N) (him : i ≤ m),
      spec.L ⟨i, lt_of_le_of_lt him hm⟩
        = 2^(m - i) * spec.L ⟨m, hm⟩ := by
  intro m
  induction m with
  | zero =>
    intro i hm him
    interval_cases i
    simp
  | succ m ih =>
    intro i hm him
    by_cases hieq : i = m + 1
    · subst hieq; simp
    · have him' : i ≤ m := by omega
      have hm_prev : m < spec.N := Nat.lt_of_succ_lt hm
      have ih' := ih hm_prev him'
      have hLs := KyberLike.L_succ (spec := spec) m hm
      -- L_i = 2^(m-i) * L_m = 2^(m-i) * (2 * L_{m+1}) = 2^(m+1-i) * L_{m+1}.
      rw [ih']
      have hpow : 2^(m + 1 - i) = 2^(m - i) * 2 := by
        have : m + 1 - i = (m - i) + 1 := by omega
        rw [this, pow_succ]
      rw [hpow]
      have h2L : 2 * spec.L ⟨m + 1, hm⟩ = spec.L ⟨m, hm_prev⟩ := by linarith
      rw [← h2L]; ring

/-- Bit `β_m` of any orbit position at step `m` is `0`, provided `m < k`.
    Reason: orbit value = `L_k + s + ∑_{i<m} c_i L_i`, where
    - `L_k + s < 2 L_k ≤ L_m` (since `k > m` makes `L_k ≤ L_m / 2`),
    - each `L_i = 2^{m-i} L_m` with `m > i`, hence even multiple of `L_m`.
    Thus `(orbit / L_m) % 2 = 0`. -/
lemma bitOf_orbitIdx_self_zero [KyberLike spec]
    {k : ℕ} (hk : k < spec.N) (s : Fin (spec.L ⟨k, hk⟩))
    (m : ℕ) (hmk : m < k) (y : Fin (2^m)) :
    spec.bitOf ⟨m, lt_of_lt_of_le hmk hk.le⟩
        (spec.orbitIdx hk s m (Nat.le_of_lt hmk) y) = 0 := by
  unfold bitOf
  rw [spec.orbitIdx_val]
  -- Set L_m and compute (L_k + s + offset) / L_m mod 2.
  have hmN : m < spec.N := lt_of_lt_of_le hmk hk.le
  -- L_k = 2^(k-m) * L_m / 2^(k-m)? No: L_m = 2^(k-m) * L_k.
  -- So L_k = L_m / 2^(k-m). Need 2^(k-m) > 1, i.e. k > m. ✓
  have hL_m_eq : spec.L ⟨m, hmN⟩ = 2^(k - m) * spec.L ⟨k, hk⟩ :=
    spec.L_eq_pow_L_succ hk (Nat.le_of_lt hmk)
  have hL_k_pos : 0 < spec.L ⟨k, hk⟩ := spec.L_pos k hk
    (lt_of_le_of_lt (Nat.zero_le _) (spec.freshIdx hk s).isLt)
  have hL_m_pos : 0 < spec.L ⟨m, hmN⟩ := by rw [hL_m_eq]; positivity
  -- Step 1: L_k + s.val < L_m. From 2 L_k ≤ L_m (since k > m, k ≥ m+1).
  have h2L_le : 2 * spec.L ⟨k, hk⟩ ≤ spec.L ⟨m, hmN⟩ := by
    rw [hL_m_eq]
    have hpow : 2^(k - m) ≥ 2 := by
      have hkm : k - m ≥ 1 := by omega
      calc 2^(k - m) ≥ 2^1 := Nat.pow_le_pow_right (by norm_num) hkm
        _ = 2 := by norm_num
    have : 2 * spec.L ⟨k, hk⟩ ≤ 2^(k - m) * spec.L ⟨k, hk⟩ :=
      Nat.mul_le_mul_right _ hpow
    omega
  have hs := s.isLt
  have hsum_lt : spec.L ⟨k, hk⟩ + s.val < spec.L ⟨m, hmN⟩ := by omega
  -- Step 2: orbitOffset is divisible by L_m.
  -- orbitOffset = ∑_{i:Fin m, y.testBit i} L_i, each L_i = 2^{m-i} L_m, so
  -- L_m divides each L_i (for i ≤ m), hence divides the sum.
  -- Moreover the QUOTIENT is even since 2^{m-i} ≥ 2 for i < m.
  -- (L_k + s + offset) / L_m = 0 + offset/L_m = even, mod 2 = 0.
  have hoffset_eq : spec.orbitOffset hk m (Nat.le_of_lt hmk) y.val
                    = (∑ i : Fin m, if y.val.testBit i.val
                        then 2^(m - i.val) else 0) * spec.L ⟨m, hmN⟩ := by
    unfold orbitOffset
    rw [Finset.sum_mul]
    apply Finset.sum_congr rfl
    intro i _
    have hiN : i.val < spec.N := lt_of_lt_of_le i.isLt
      (le_trans (Nat.le_of_lt hmk) hk.le)
    have hi_le_m : i.val ≤ m := Nat.le_of_lt i.isLt
    have hLi_eq : spec.L ⟨i.val, hiN⟩
                  = 2^(m - i.val) * spec.L ⟨m, hmN⟩ :=
      spec.L_eq_pow_L_succ hmN hi_le_m
    split_ifs with h
    · rw [hLi_eq]
    · simp
  rw [hoffset_eq]
  -- Now: (L_k + s + c · L_m) / L_m = c + (L_k + s) / L_m = c.
  rw [show spec.L ⟨k, hk⟩ + s.val
          + (∑ i : Fin m, if y.val.testBit i.val
              then 2^(m - i.val) else 0) * spec.L ⟨m, hmN⟩
        = (spec.L ⟨k, hk⟩ + s.val)
          + (∑ i : Fin m, if y.val.testBit i.val
              then 2^(m - i.val) else 0) * spec.L ⟨m, hmN⟩ from by ring]
  rw [Nat.add_mul_div_right _ _ hL_m_pos, Nat.div_eq_of_lt hsum_lt, zero_add]
  -- Each 2^{m-i} for i < m has m-i ≥ 1, hence is even, so the whole sum is even.
  have hsum_even : 2 ∣ ∑ i : Fin m, if y.val.testBit i.val
                          then 2^(m - i.val) else 0 := by
    apply Finset.dvd_sum
    intro i _
    have hi_lt_m : i.val < m := i.isLt
    have hmi_pos : 1 ≤ m - i.val := by omega
    have hdvd : 2 ∣ 2^(m - i.val) := by
      have hk' : m - i.val = (m - i.val - 1) + 1 := by omega
      exact hk' ▸ (dvd_pow_self 2 (by omega))
    split_ifs
    · exact hdvd
    · exact ⟨0, rfl⟩
  obtain ⟨c, hc⟩ := hsum_even
  rw [hc]
  rw [show 2 * c = c * 2 from by ring, Nat.mul_mod_left]

/-- Inserting the `(m+1)`-th coordinate to an orbit offset:
    `orbitOffset_{m+1}(y.val) = orbitOffset_m(y.val)` (bit `m` of `y.val` is 0
    when `y < 2^m`). -/
lemma orbitOffset_castSucc [KyberLike spec] {k : ℕ} (hk : k < spec.N)
    {m : ℕ} (hmk : m + 1 ≤ k) (y : Fin (2^m)) :
    spec.orbitOffset hk (m+1) hmk y.val
      = spec.orbitOffset hk m (Nat.le_of_succ_le hmk) y.val := by
  unfold orbitOffset
  rw [Fin.sum_univ_castSucc]
  -- Add the last term (i = m). Bit m of y.val is 0 since y.val < 2^m.
  have hbit_m : ¬ y.val.testBit m := by
    rw [Nat.testBit_lt_two_pow y.isLt]; trivial
  simp [hbit_m]

/-- Toggling bit `m` (adding `2^m`) extends the orbit by `L_m`. -/
lemma orbitOffset_succ_with_bit [KyberLike spec] {k : ℕ} (hk : k < spec.N)
    {m : ℕ} (hmk : m + 1 ≤ k) (y : Fin (2^m)) :
    spec.orbitOffset hk (m+1) hmk (y.val + 2^m)
      = spec.orbitOffset hk m (Nat.le_of_succ_le hmk) y.val
        + spec.L ⟨m, lt_of_le_of_lt (Nat.le_of_succ_le hmk) hk⟩ := by
  unfold orbitOffset
  rw [Fin.sum_univ_castSucc]
  -- Bit m of (y.val + 2^m) is true (y.val < 2^m has bit m false; + 2^m flips it).
  have hbit_m : (y.val + 2^m).testBit m = true := by
    rw [Nat.add_comm, Nat.testBit_two_pow_add_eq]
    rw [Nat.testBit_lt_two_pow y.isLt]
    rfl
  -- For i < m: bit i of (y.val + 2^m) = bit i of y.val.
  have hbit_lt : ∀ i : Fin m,
      (y.val + 2^m).testBit i.val = y.val.testBit i.val := by
    intro i
    rw [Nat.add_comm, Nat.testBit_two_pow_add_gt i.isLt]
  -- Now compute.
  simp [Fin.val_last, hbit_m]
  apply Finset.sum_congr rfl
  intro i _
  rw [hbit_lt i]

/-- Partner of an orbit position at layer `m` equals the orbit position at
    step `m+1` with bit `m` toggled. -/
lemma partnerIdx_orbitIdx_eq [KyberLike spec]
    {k : ℕ} (hk : k < spec.N) (s : Fin (spec.L ⟨k, hk⟩))
    {m : ℕ} (hmk : m + 1 ≤ k) (y : Fin (2^m)) :
    spec.partnerIdx ⟨m, lt_of_le_of_lt (Nat.le_of_succ_le hmk) hk⟩
        (spec.orbitIdx hk s m (Nat.le_of_succ_le hmk) y)
      = spec.orbitIdx hk s (m+1) hmk
          ⟨y.val + 2^m, by have := y.isLt; rw [pow_succ]; omega⟩ := by
  apply Fin.ext
  -- LHS: partnerIdx adds L_m (since orbit position is on a-side).
  have hmN : m < spec.N := lt_of_le_of_lt (Nat.le_of_succ_le hmk) hk
  have hmk' : m < k := hmk
  have hbit_zero := spec.bitOf_orbitIdx_self_zero hk s m hmk' y
  set hmk_le : m ≤ k := Nat.le_of_succ_le hmk with hmk_le_def
  -- bitOf m position = 0 ⇒ position % (2 L_m) < L_m (a-side condition).
  -- Convert: bitOf m i = (i.val / L_m) % 2.
  have hn_pos : 0 < spec.n :=
    lt_of_le_of_lt (Nat.zero_le _) (spec.freshIdx hk s).isLt
  have hL_m_pos : 0 < spec.L ⟨m, hmN⟩ := spec.L_pos m hmN hn_pos
  have hmod : (spec.orbitIdx hk s m hmk_le y).val % (2 * spec.L ⟨m, hmN⟩)
                < spec.L ⟨m, hmN⟩ := by
    -- i.val % (2 L) < L iff (i / L) % 2 = 0.
    by_contra hge
    push_neg at hge
    -- bitOf m = (i / L) % 2 = 1 then.
    unfold bitOf at hbit_zero
    -- Provide div/mod arithmetic to omega.
    set p := (spec.orbitIdx hk s m hmk_le y).val with hp_def
    set L := spec.L ⟨m, hmN⟩ with hL_def
    have h2L_pos : 0 < 2 * L := by omega
    have hdiv2L := Nat.div_add_mod p (2 * L)
    have hmod2L_lt : p % (2 * L) < 2 * L := Nat.mod_lt _ h2L_pos
    -- (p / L) = (p / 2L) * 2 + (p % 2L) / L
    have hdivL_eq : p / L = (p / (2 * L)) * 2 + (p % (2 * L)) / L := by
      have hkey : p = L * (2 * (p / (2 * L))) + p % (2 * L) := by
        have := Nat.div_add_mod p (2 * L)
        linarith
      conv_lhs => rw [hkey]
      rw [Nat.add_comm, Nat.add_mul_div_left _ _ hL_m_pos]
      ring
    have hp_mod_div : (p % (2 * L)) / L = 1 := by
      have hge2 : L ≤ p % (2 * L) := hge
      have hlt2 : p % (2 * L) < 2 * L := hmod2L_lt
      have : p % (2 * L) = L + (p % (2 * L) - L) := by omega
      rw [this, Nat.add_comm, Nat.add_div_right _ hL_m_pos,
          Nat.div_eq_of_lt (by omega : p % (2 * L) - L < L)]
    rw [hp_mod_div] at hdivL_eq
    -- p / L = (p / 2L) * 2 + 1, so (p / L) % 2 = 1.
    have hmod2 : (p / L) % 2 = 1 := by
      rw [hdivL_eq]
      rw [show p / (2 * L) * 2 + 1 = 1 + p / (2 * L) * 2 from by ring,
          Nat.add_mul_mod_self_right]
    rw [hmod2] at hbit_zero
    exact absurd hbit_zero (by decide)
  rw [spec.partnerIdx_val_aside hmod]
  -- Now compute orbitIdx values directly.
  show (spec.orbitIdx hk s m hmk_le y).val + spec.L ⟨m, hmN⟩
        = (spec.orbitIdx hk s (m + 1) hmk
            ⟨y.val + 2 ^ m, _⟩).val
  rw [spec.orbitIdx_val, spec.orbitIdx_val]
  rw [spec.orbitOffset_succ_with_bit hk hmk y]
  ring

/-- Orbit at step `m` equals orbit at step `m+1` with `y` embedded via
    `castAdd` (bit `m` of `y` stays 0). -/
lemma orbitIdx_castSucc [KyberLike spec] {k : ℕ} (hk : k < spec.N)
    (s : Fin (spec.L ⟨k, hk⟩))
    {m : ℕ} (hmk : m + 1 ≤ k) (y : Fin (2^m)) :
    spec.orbitIdx hk s (m+1) hmk
        ⟨y.val, by have := y.isLt; rw [pow_succ]; omega⟩
      = spec.orbitIdx hk s m (Nat.le_of_succ_le hmk) y := by
  apply Fin.ext
  rw [spec.orbitIdx_val, spec.orbitIdx_val]
  rw [spec.orbitOffset_castSucc hk hmk y]

/-! ### Step A — `runPrefixInv_zero` -/

/-- At length 0, `runPrefixInv` is the identity. -/
lemma runPrefixInv_zero {Z : spec.Twiddles K} (hZ : ∀ ℓ g, Z ℓ g ≠ 0)
    (h2 : (2 : K) ≠ 0) (hk : (0 : ℕ) ≤ spec.N) (v : Fin spec.n → K) :
    spec.runPrefixInv Z h2 0 hk (fun i g => hZ _ g) v = v := by
  apply (spec.runPrefixEquiv Z h2 0 hk (fun i g => hZ _ g)).injective
  show (spec.runPrefixEquiv Z h2 0 hk (fun i g => hZ _ g))
        ((spec.runPrefixEquiv Z h2 0 hk (fun i g => hZ _ g)).symm v) = _
  rw [LinearEquiv.apply_symm_apply]
  show v = spec.runPrefix Z 0 hk v
  unfold runPrefix
  simp

/-! ### Step B — Splitting a sum over `Fin (2^(m+1))` -/

/-- Splits a sum over `Fin (2^(m+1))` into the "low" half (y : Fin (2^m) at
    index y.val) and the "high" half (at index y.val + 2^m). -/
lemma sum_fin_two_pow_succ_split {M : Type*} [AddCommMonoid M] (m : ℕ)
    (f : Fin (2^(m+1)) → M) :
    ∑ y' : Fin (2^(m+1)), f y'
      = (∑ y : Fin (2^m), f ⟨y.val,
            by have := y.isLt; rw [pow_succ]; omega⟩)
        + (∑ y : Fin (2^m), f ⟨y.val + 2^m,
            by have := y.isLt; rw [pow_succ]; omega⟩) := by
  -- Bridge via Fin (2^m + 2^m) ≃ Fin (2^(m+1)).
  have hcast : 2^m + 2^m = 2^(m+1) := by rw [pow_succ]; ring
  rw [show (∑ y' : Fin (2^(m+1)), f y')
        = (∑ i : Fin (2^m + 2^m), f (Fin.cast hcast i)) from
        (Fin.sum_congr' f hcast).symm]
  rw [Fin.sum_univ_add]
  have h1 : (∑ y : Fin (2^m), f (Fin.cast hcast (Fin.castAdd (2^m) y)))
          = (∑ y : Fin (2^m), f ⟨y.val,
              by have := y.isLt; rw [pow_succ]; omega⟩) := by
    refine Finset.sum_congr rfl (fun y _ => ?_)
    rfl
  have h2 : (∑ y : Fin (2^m), f (Fin.cast hcast (Fin.natAdd (2^m) y)))
          = (∑ y : Fin (2^m), f ⟨y.val + 2^m,
              by have := y.isLt; rw [pow_succ]; omega⟩) := by
    refine Finset.sum_congr rfl (fun y _ => ?_)
    congr 1; apply Fin.ext
    show 2^m + y.val = y.val + 2^m
    omega
  rw [h1, h2]

/-! ### Step C — `layerInv_at_orbit`

  At an orbit position `orbit_m y`, the value of `layerInv ⟨m,_⟩ v` is the
  average of `v` at the orbit position and at its partner (= orbit position
  `(m+1, y + 2^m)`). The twiddle cancels because the orbit is on the a-side
  of layer m (the formula `(a + b) / 2` has no twiddle dependence). -/

lemma layerInv_at_orbit [KyberLike spec]
    {Z : spec.Twiddles K} (hZ : ∀ ℓ g, Z ℓ g ≠ 0) (h2 : (2 : K) ≠ 0)
    {k : ℕ} (hk : k < spec.N) (s : Fin (spec.L ⟨k, hk⟩))
    {m : ℕ} (hmk : m + 1 ≤ k) (v : Fin spec.n → K) (y : Fin (2^m)) :
    (spec.layerInv Z ⟨m, lt_of_le_of_lt (Nat.le_of_succ_le hmk) hk⟩
        (fun g => hZ _ g) h2) v
        (spec.orbitIdx hk s m (Nat.le_of_succ_le hmk) y)
      = (v (spec.orbitIdx hk s m (Nat.le_of_succ_le hmk) y)
          + v (spec.orbitIdx hk s (m+1) hmk
              ⟨y.val + 2^m, by have := y.isLt; rw [pow_succ]; omega⟩)) / 2 := by
  -- The orbit position is on a-side at layer m (bitOf m = 0).
  set hmN : m < spec.N := lt_of_le_of_lt (Nat.le_of_succ_le hmk) hk
  set hmk_le : m ≤ k := Nat.le_of_succ_le hmk
  have hbit_zero := spec.bitOf_orbitIdx_self_zero hk s m hmk y
  -- Decompose orbit_m y as aSideIdx ⟨m,_⟩ g j.
  obtain ⟨g, j, hgj_eq⟩ :=
    spec.exists_aSide_decomp ⟨m, hmN⟩ (spec.orbitIdx hk s m hmk_le y) hbit_zero
  -- LHS via aSide_eq_apply.
  have hLHS : (spec.layerInv Z ⟨m, hmN⟩ (fun g => hZ _ g) h2) v
                (spec.orbitIdx hk s m hmk_le y)
            = spec.aSide ((spec.layerInv Z ⟨m, hmN⟩ (fun g => hZ _ g) h2) v)
                ⟨m, hmN⟩ g j := by
    rw [hgj_eq]
    exact (spec.aSide_eq_apply _ _ _ _).symm
  -- Apply aSide_layerInv formula.
  rw [hLHS, spec.aSide_layerInv Z ⟨m, hmN⟩ (fun g => hZ _ g) h2 v g j]
  -- aSide v g j = v (aSideIdx g j) = v (orbit_m y).
  rw [spec.aSide_eq_apply, spec.bSide_eq_apply, ← hgj_eq]
  -- Now goal: (v (orbit_m y) + v (bSideIdx ⟨m,_⟩ g j)) / 2 = (v (orbit_m y) + v (orbit_{m+1} ⟨y+2^m, _⟩)) / 2.
  -- bSideIdx ⟨m,_⟩ g j = partnerIdx ⟨m,_⟩ (aSideIdx ⟨m,_⟩ g j) = partnerIdx ⟨m,_⟩ (orbit_m y)
  --   = orbit_{m+1} ⟨y + 2^m, _⟩ (by partnerIdx_orbitIdx_eq).
  have hpartner : spec.bSideIdx ⟨m, hmN⟩ g j
                = spec.orbitIdx hk s (m+1) hmk
                    ⟨y.val + 2^m, by have := y.isLt; rw [pow_succ]; omega⟩ := by
    rw [← spec.partnerIdx_aSideIdx ⟨m, hmN⟩ g j, ← hgj_eq]
    exact spec.partnerIdx_orbitIdx_eq hk s hmk y
  rw [hpartner]

/-! ### Step D — `runPrefixInv_at_freshIdx_aux` main induction

  The key universal lemma (verified by k=2 hand computation in session 3):

    runPrefixInv Z h2 m _ v (freshIdx hk s)
      = (1 / (2 : K))^m * ∑ y : Fin (2^m), v (orbitIdx hk s m _ y).

  Proved by induction on m. -/

lemma runPrefixInv_at_freshIdx_aux [KyberLike spec]
    {Z : spec.Twiddles K} (hZ : ∀ ℓ g, Z ℓ g ≠ 0) (h2 : (2 : K) ≠ 0)
    {k : ℕ} (hk : k < spec.N) (s : Fin (spec.L ⟨k, hk⟩)) :
    ∀ (m : ℕ) (hmk : m ≤ k) (v : Fin spec.n → K),
      (spec.runPrefixInv Z h2 m (le_trans hmk (Nat.le_of_lt hk))
          (fun i g => hZ _ g)) v (spec.freshIdx hk s)
        = (1 / (2 : K))^m
            * ∑ y : Fin (2^m), v (spec.orbitIdx hk s m hmk y)
  | 0, _, v => by
      rw [spec.runPrefixInv_zero hZ h2 _ v]
      rw [pow_zero, one_mul]
      rw [show (∑ y : Fin (2^0), v (spec.orbitIdx hk s 0 (Nat.zero_le _) y))
            = v (spec.orbitIdx hk s 0 (Nat.zero_le _) ⟨0, by decide⟩) from
              Fin.sum_univ_one _]
      exact (congrArg v (spec.orbitIdx_zero hk s ⟨0, by decide⟩)).symm
  | m+1, hmk, v => by
      -- Peel layer m via runPrefixInv_succ.
      have hmk_le : m ≤ k := Nat.le_of_succ_le hmk
      have hmN : m < spec.N := lt_of_le_of_lt hmk_le hk
      -- After peel: layerInv ⟨m,_⟩ v plays the role of v in the IH at m.
      rw [spec.runPrefixInv_succ hZ h2 m (le_trans hmk (Nat.le_of_lt hk)) v]
      -- Apply IH at m with v' = layerInv ⟨m,_⟩ v.
      set v' := (spec.layerInv Z ⟨m, hmN⟩ (fun g => hZ ⟨m, hmN⟩ g) h2) v
      have ih := runPrefixInv_at_freshIdx_aux hZ h2 hk s m hmk_le v'
      rw [ih]
      -- Goal: (1/2)^m * ∑ y : Fin (2^m), v' (orbit_m y)
      --     = (1/2)^(m+1) * ∑ y' : Fin (2^(m+1)), v (orbit_{m+1} y').
      -- Apply Step C (layerInv_at_orbit) to each term in the sum.
      have hstep : ∀ y : Fin (2^m),
          v' (spec.orbitIdx hk s m hmk_le y)
            = (v (spec.orbitIdx hk s m hmk_le y)
                + v (spec.orbitIdx hk s (m+1) hmk
                    ⟨y.val + 2^m, by have := y.isLt; rw [pow_succ]; omega⟩)) / 2 := by
        intro y
        exact spec.layerInv_at_orbit hZ h2 hk s hmk v y
      rw [Finset.sum_congr rfl (fun y _ => hstep y)]
      -- Now: (1/2)^m * ∑ y, (v(orbit_m y) + v(orbit_{m+1} ⟨y+2^m,_⟩)) / 2.
      -- Rewrite (a + b) / 2 = (1/2) * (a + b) inside the sum.
      have hav : ∀ y : Fin (2^m),
          (v (spec.orbitIdx hk s m hmk_le y)
            + v (spec.orbitIdx hk s (m+1) hmk
                ⟨y.val + 2^m, by have := y.isLt; rw [pow_succ]; omega⟩)) / 2
          = (1 / (2 : K))
            * (v (spec.orbitIdx hk s m hmk_le y)
                + v (spec.orbitIdx hk s (m+1) hmk
                    ⟨y.val + 2^m, by have := y.isLt; rw [pow_succ]; omega⟩)) := by
        intro y; ring
      rw [Finset.sum_congr rfl (fun y _ => hav y)]
      rw [← Finset.mul_sum]
      rw [Finset.sum_add_distrib]
      -- LHS: (1/2)^m * ((1/2) * (∑ v(orbit_m y) + ∑ v(orbit_{m+1} ⟨y+2^m,_⟩))).
      -- RHS: (1/2)^(m+1) * ∑ y' : Fin (2^(m+1)), v(orbit_{m+1} y').
      -- Use sum_fin_two_pow_succ_split on RHS.
      rw [show (1 / (2 : K))^(m+1) = (1 / 2)^m * (1 / 2) from by rw [pow_succ]]
      rw [sum_fin_two_pow_succ_split (m := m) (f := fun y' => v (spec.orbitIdx hk s (m+1) hmk y'))]
      -- Reindex low half via orbitIdx_castSucc.
      have hlow : (∑ y : Fin (2^m),
                    v (spec.orbitIdx hk s (m+1) hmk
                          ⟨y.val, by have := y.isLt; rw [pow_succ]; omega⟩))
                = ∑ y : Fin (2^m), v (spec.orbitIdx hk s m hmk_le y) := by
        apply Finset.sum_congr rfl
        intro y _
        rw [spec.orbitIdx_castSucc hk s hmk y]
      rw [hlow]
      ring

/-! ### Step E — Collapse to `pi_scalar_at_fresh` headline -/

/-- `spec.G ⟨k, hk⟩ = 2^k` for KyberLike (induction via `G_succ`). -/
lemma G_eq_pow2 [KyberLike spec] (hn : 0 < spec.n) :
    ∀ (k : ℕ) (hk : k < spec.N), spec.G ⟨k, hk⟩ = 2^k
  | 0, hk => by
      have hL0 := KyberLike.L_zero_doubled (spec := spec) hk
      have hgl := spec.hGL ⟨0, hk⟩
      rw [hL0] at hgl
      -- hgl : G_0 * n = n. With n > 0, G_0 = 1.
      have hG0 : spec.G ⟨0, hk⟩ = 1 := by
        rcases Nat.eq_zero_or_pos (spec.G ⟨0, hk⟩) with h0 | h1
        · exfalso; rw [h0] at hgl; simp at hgl; omega
        · -- G_0 ≥ 1; G_0 * n = n ⇒ G_0 = 1 (cancel n).
          have : spec.G ⟨0, hk⟩ * spec.n = 1 * spec.n := by rw [hgl, one_mul]
          exact Nat.eq_of_mul_eq_mul_right hn this
      rw [hG0, pow_zero]
  | k+1, hk => by
      have hk' : k < spec.N := Nat.lt_of_succ_lt hk
      rw [KyberLike.G_succ (spec := spec) k hk, G_eq_pow2 hn k hk', pow_succ]
      ring

/-- `orbitOffset` is divisible by `2 * L_k` (each term `L_i` is a multiple of `2 L_k`). -/
lemma two_L_dvd_orbitOffset [KyberLike spec] {k : ℕ} (hk : k < spec.N)
    {m : ℕ} (hmk : m ≤ k) (y : ℕ) :
    (2 * spec.L ⟨k, hk⟩) ∣ spec.orbitOffset hk m hmk y := by
  unfold orbitOffset
  apply Finset.dvd_sum
  intro i _
  split_ifs
  · -- L_i = 2 L_k * 2^(k-i-1) when i < k.
    have hi : i.val < k := lt_of_lt_of_le i.isLt hmk
    have hiN : i.val < spec.N := lt_of_lt_of_le i.isLt (le_trans hmk hk.le)
    have hLi : spec.L ⟨i.val, hiN⟩ = 2^(k - i.val) * spec.L ⟨k, hk⟩ :=
      spec.L_eq_pow_L_succ hk (Nat.le_of_lt hi)
    rw [hLi]
    have h1 : k - i.val = (k - i.val - 1) + 1 := by omega
    rw [h1, pow_succ]
    refine ⟨2^(k - i.val - 1), ?_⟩
    ring
  · exact dvd_zero _

/-- The "group at layer k" of orbit index `y : Fin (2^k)`: divide `orbitOffset` by `2 L_k`. -/
def groupAtK [KyberLike spec] {k : ℕ} (hk : k < spec.N) (hn : 0 < spec.n)
    (y : Fin (2^k)) : Fin (spec.G ⟨k, hk⟩) :=
  ⟨spec.orbitOffset hk k le_rfl y.val / (2 * spec.L ⟨k, hk⟩), by
    -- orbitOffset + 2 L_k ≤ n = G_k * (2 L_k); so orbitOffset ≤ (G_k - 1) * (2 L_k);
    -- divided by 2 L_k ≤ G_k - 1 < G_k.
    have hbound := spec.sum_Li_plus_2Lk_le_n hk (le_refl k)
    have hoffset := spec.orbitOffset_le_sum hk (le_refl k) y.val
    have hgl := spec.hGL ⟨k, hk⟩
    have h2Lk_pos : 0 < 2 * spec.L ⟨k, hk⟩ := by
      have := spec.L_pos k hk hn
      omega
    have h_off_le : spec.orbitOffset hk k le_rfl y.val ≤ spec.n - 2 * spec.L ⟨k, hk⟩ := by omega
    have h_n_eq : spec.n = spec.G ⟨k, hk⟩ * (2 * spec.L ⟨k, hk⟩) := hgl.symm
    -- div bound
    have : spec.orbitOffset hk k le_rfl y.val / (2 * spec.L ⟨k, hk⟩)
            < spec.G ⟨k, hk⟩ := by
      rw [Nat.div_lt_iff_lt_mul h2Lk_pos]
      rw [show spec.G ⟨k, hk⟩ * (2 * spec.L ⟨k, hk⟩) = spec.n from h_n_eq.symm]
      omega
    exact this⟩

/-- Defining property of `groupAtK`: `(groupAtK y).val * (2 L_k) = orbitOffset y`. -/
lemma groupAtK_val_mul [KyberLike spec] {k : ℕ} (hk : k < spec.N) (hn : 0 < spec.n)
    (y : Fin (2^k)) :
    (spec.groupAtK hk hn y).val * (2 * spec.L ⟨k, hk⟩)
      = spec.orbitOffset hk k le_rfl y.val := by
  show (spec.orbitOffset hk k le_rfl y.val / (2 * spec.L ⟨k, hk⟩)) * (2 * spec.L ⟨k, hk⟩)
       = spec.orbitOffset hk k le_rfl y.val
  exact Nat.div_mul_cancel (spec.two_L_dvd_orbitOffset hk (le_refl k) y.val)

/-- `orbitIdx hk s k _ y = bSideIdx ⟨k,_⟩ (groupAtK y) s`. -/
lemma orbit_eq_bSideIdx_groupAtK [KyberLike spec] {k : ℕ} (hk : k < spec.N)
    (hn : 0 < spec.n) (s : Fin (spec.L ⟨k, hk⟩)) (y : Fin (2^k)) :
    spec.orbitIdx hk s k le_rfl y = spec.bSideIdx ⟨k, hk⟩ (spec.groupAtK hk hn y) s := by
  apply Fin.ext
  rw [spec.bSideIdx_val, spec.orbitIdx_val]
  rw [← spec.groupAtK_val_mul hk hn y]
  ring

/-- Helper: the sum `∑_{j < k} (if b j then 2^j else 0)` has bit `i` equal to `b i`
    for `i < k`. Standard binary representation uniqueness. Quantified over `b`
    so the IH can be applied to a shifted function. -/
private lemma sum_pow_two_testBit :
    ∀ (k : ℕ) (b : ℕ → Bool) (i : ℕ), i < k →
      (∑ j ∈ Finset.range k, if b j then 2^j else 0).testBit i = b i
  | 0, _, i, hi => by omega
  | k+1, b, i, hi => by
    rw [Finset.sum_range_succ' (fun j => if b j then 2^j else 0) k]
    have h_doubled : ∀ j ∈ Finset.range k,
        (if b (j + 1) then 2^(j + 1) else 0) = 2 * (if b (j + 1) then 2^j else 0) := by
      intro j _
      split_ifs <;> simp [pow_succ, mul_comm]
    rw [Finset.sum_congr rfl h_doubled, ← Finset.mul_sum]
    set S'' := ∑ j ∈ Finset.range k, if b (j + 1) then 2^j else 0 with hS_def
    match i with
    | 0 =>
      simp only [pow_zero]
      rw [Nat.testBit_zero]
      split_ifs with hb0
      · have h1 : (2 * S'' + 1) % 2 = 1 := by omega
        rw [h1]; rw [hb0]; rfl
      · have h0 : (2 * S'' + 0) % 2 = 0 := by omega
        rw [h0]
        have hb0' : b 0 = false := by
          cases hbb : b 0 with
          | false => rfl
          | true => exact absurd hbb hb0
        rw [hb0']; rfl
    | i'+1 =>
      have hi' : i' < k := by omega
      rw [Nat.testBit_succ]
      have h_div : (2 * S'' + (if b 0 then 2^0 else 0)) / 2 = S'' := by
        split_ifs with hb0
        · simp only [pow_zero]; omega
        · simp only [add_zero]; omega
      rw [h_div]
      -- Apply IH at k with shifted function (b ∘ Nat.succ).
      exact sum_pow_two_testBit k (fun j => b (j + 1)) i' hi'

/-- For i < k, the bit at position `k-1-i` of `orbitOffset y / (2 L_k)` equals
    `y.testBit i`. -/
lemma orbitOffset_testBit_eq [KyberLike spec] {k : ℕ} (hk : k < spec.N)
    (hn : 0 < spec.n) (y : ℕ) (i : ℕ) (hi : i < k) :
    y.testBit i =
      ((spec.orbitOffset hk k le_rfl y) / (2 * spec.L ⟨k, hk⟩)).testBit (k - 1 - i) := by
  -- Express orbitOffset y / (2 L_k) as a sum of distinct powers of 2.
  have h2Lk_pos : 0 < 2 * spec.L ⟨k, hk⟩ := by
    have := spec.L_pos k hk hn; omega
  have h_form : spec.orbitOffset hk k le_rfl y
              = (2 * spec.L ⟨k, hk⟩) *
                  ∑ j : Fin k, (if y.testBit j.val then 2^(k - 1 - j.val) else 0) := by
    unfold orbitOffset
    rw [Finset.mul_sum]
    apply Finset.sum_congr rfl
    intro j _
    have hjN : j.val < spec.N := lt_of_lt_of_le j.isLt hk.le
    have hLj : spec.L ⟨j.val, hjN⟩ = 2^(k - j.val) * spec.L ⟨k, hk⟩ :=
      spec.L_eq_pow_L_succ hk (Nat.le_of_lt j.isLt)
    split_ifs with hb
    · rw [hLj]
      have hkj : k - j.val = (k - 1 - j.val) + 1 := by have := j.isLt; omega
      rw [hkj, pow_succ]; ring
    · ring
  rw [h_form, Nat.mul_div_cancel_left _ h2Lk_pos]
  -- Reindex j ↔ j' = k - 1 - j to convert ∑ (y.testBit j) * 2^(k-1-j) into ∑ b' j' * 2^j'.
  have hreidx : (∑ j : Fin k, if y.testBit j.val then 2^(k - 1 - j.val) else 0)
              = ∑ j' ∈ Finset.range k, if y.testBit (k - 1 - j') then 2^j' else 0 := by
    rw [show (∑ j : Fin k, if y.testBit j.val then 2^(k - 1 - j.val) else 0)
          = ∑ j ∈ Finset.range k, if y.testBit j then 2^(k - 1 - j) else 0 from
          (Finset.sum_range fun j => if y.testBit j then 2^(k - 1 - j) else 0).symm]
    apply Finset.sum_nbij' (fun j => k - 1 - j) (fun j' => k - 1 - j')
    · intro j hj; simp only [Finset.mem_range] at hj ⊢; omega
    · intro j' hj'; simp only [Finset.mem_range] at hj' ⊢; omega
    · intro j hj; simp only [Finset.mem_range] at hj; omega
    · intro j' hj'; simp only [Finset.mem_range] at hj'; omega
    · intro j hj; simp only [Finset.mem_range] at hj
      have hkmj : k - 1 - (k - 1 - j) = j := by omega
      rw [hkmj]
  rw [hreidx]
  symm
  -- Apply helper with b = (j ↦ y.testBit (k - 1 - j)), at position (k - 1 - i).
  have h := sum_pow_two_testBit k (fun j => y.testBit (k - 1 - j)) (k - 1 - i) (by omega)
  simp only at h
  rw [h]
  -- y.testBit (k - 1 - (k - 1 - i)) = y.testBit i.
  have hki : k - 1 - (k - 1 - i) = i := by omega
  rw [hki]
lemma groupAtK_injective [KyberLike spec] {k : ℕ} (hk : k < spec.N) (hn : 0 < spec.n) :
    Function.Injective (spec.groupAtK hk hn) := by
  intro y₁ y₂ hg
  have h_off_eq : spec.orbitOffset hk k le_rfl y₁.val
                = spec.orbitOffset hk k le_rfl y₂.val := by
    have h1 := spec.groupAtK_val_mul hk hn y₁
    have h2 := spec.groupAtK_val_mul hk hn y₂
    have hv : (spec.groupAtK hk hn y₁).val = (spec.groupAtK hk hn y₂).val := by
      rw [hg]
    -- y₁.val_off = g₁ * 2L = g₂ * 2L = y₂.val_off.
    rw [← h1, ← h2, hv]
  apply Fin.ext
  apply Nat.eq_of_testBit_eq
  intro i
  by_cases hi : i < k
  · have h1 := spec.orbitOffset_testBit_eq hk hn y₁.val i hi
    have h2 := spec.orbitOffset_testBit_eq hk hn y₂.val i hi
    rw [h_off_eq] at h1
    rw [h1, h2]
  · push_neg at hi
    have hpow : 2^k ≤ 2^i := Nat.pow_le_pow_right (by norm_num) hi
    have h1 : y₁.val.testBit i = false :=
      Nat.testBit_lt_two_pow (Nat.lt_of_lt_of_le y₁.isLt hpow)
    have h2 : y₂.val.testBit i = false :=
      Nat.testBit_lt_two_pow (Nat.lt_of_lt_of_le y₂.isLt hpow)
    rw [h1, h2]

/-- `groupAtK` is surjective (card-equal + injective). -/
lemma groupAtK_surjective [KyberLike spec] {k : ℕ} (hk : k < spec.N) (hn : 0 < spec.n) :
    Function.Surjective (spec.groupAtK hk hn) := by
  have hcard : Fintype.card (Fin (2^k)) = Fintype.card (Fin (spec.G ⟨k, hk⟩)) := by
    simp [spec.G_eq_pow2 hn k hk]
  exact ((Fintype.bijective_iff_injective_and_card _).mpr
    ⟨spec.groupAtK_injective hk hn, hcard⟩).surjective

/-! ### Step E3 — Headline `pi_scalar_at_fresh` -/

/-- **F8 main lemma (paper `lem:piscalar`).** The π-projection at `freshIdx ⟨k,_⟩ s`
    equals `(1/2)^k · bSide (runPrefix Z' k w) ⟨k,_⟩ g₀ s`. -/
theorem pi_scalar_at_fresh
    [KyberLike spec]
    {Z : spec.Twiddles K} (hZ : ∀ ℓ g, Z ℓ g ≠ 0) (h2 : (2 : K) ≠ 0)
    {F : spec.FaultSet} (hF : spec.OneFaultPerLayer F)
    (k : ℕ) (hk : k < spec.N) (w : Fin spec.n → K)
    (s : Fin (spec.L ⟨k, hk⟩)) :
    spec.nttInv hZ h2 (spec.teleTerm Z (spec.faultedTwiddles Z F) k w)
        (spec.freshIdx hk s)
      = (1 / (2 : K))^k * spec.bSide
          (spec.runPrefix (spec.faultedTwiddles Z F) k (Nat.le_of_lt hk) w)
          ⟨k, hk⟩
          (Finset.card_eq_one.mp (hF ⟨k, hk⟩)).choose s := by
  have hn : 0 < spec.n := lt_of_le_of_lt (Nat.zero_le _) (spec.freshIdx hk s).isLt
  set g₀ := (Finset.card_eq_one.mp (hF ⟨k, hk⟩)).choose with hg₀_def
  -- Step 1: rewrite via teleTerm_factor + layer_diff = perturbation + nttInv_suffix_apply.
  rw [spec.teleTerm_factor Z (spec.faultedTwiddles Z F) k hk,
      spec.layer_diff_eq_perturbation Z F k hk]
  rw [LinearMap.comp_apply, LinearMap.comp_apply]
  rw [spec.nttInv_suffix_apply hZ h2 k hk]
  -- Step 2: apply the main induction at m = k with v = layerInv ⟨k,_⟩ (pert (runPrefix Z' k w)).
  rw [spec.runPrefixInv_at_freshIdx_aux hZ h2 hk s k le_rfl _]
  congr 1
  -- Step 3: show ∑ y : Fin (2^k), v (orbit_k y) = bSide (runPrefix Z' k w) ⟨k,_⟩ g₀ s.
  -- Reindex via groupAtK to sum over Fin G_k, then collapse via _g0/_off lemmas.
  -- Get the preimage y_g₀ of g₀ under groupAtK.
  obtain ⟨y_g₀, hy_g₀⟩ := spec.groupAtK_surjective hk hn g₀
  -- Apply Finset.sum_eq_single at y_g₀.
  rw [Finset.sum_eq_single y_g₀]
  · -- Main term: orbit_k y_g₀ = bSideIdx ⟨k,_⟩ g₀ s ⇒ value = bSide (runPrefix Z' k w) ⟨k,_⟩ g₀ s.
    rw [spec.orbit_eq_bSideIdx_groupAtK hk hn s y_g₀, hy_g₀]
    -- (layerInv (pert (runPrefix Z' k w))) (bSideIdx ⟨k,_⟩ g₀ s)
    --   = bSide (layerInv (pert (runPrefix Z' k w))) ⟨k,_⟩ g₀ s     [by bSide_eq_apply]
    --   = bSide (runPrefix Z' k w) ⟨k,_⟩ g₀ s                          [by bSide_layerInv_pert_g0]
    rw [← spec.bSide_eq_apply
          ((spec.layerInv Z ⟨k, hk⟩ (fun g => hZ ⟨k, hk⟩ g) h2)
            (spec.perturbation Z F ⟨k, hk⟩
              (spec.runPrefix (spec.faultedTwiddles Z F) k (Nat.le_of_lt hk) w)))
          ⟨k, hk⟩ g₀ s]
    exact spec.bSide_layerInv_pert_g0 hZ h2 hF k hk
      (spec.runPrefix (spec.faultedTwiddles Z F) k (Nat.le_of_lt hk) w) s
  · -- Off-witness: for y ≠ y_g₀, value at orbit_k y = 0.
    intro y _ hy_ne
    rw [spec.orbit_eq_bSideIdx_groupAtK hk hn s y]
    rw [← spec.bSide_eq_apply
          ((spec.layerInv Z ⟨k, hk⟩ (fun g => hZ ⟨k, hk⟩ g) h2)
            (spec.perturbation Z F ⟨k, hk⟩
              (spec.runPrefix (spec.faultedTwiddles Z F) k (Nat.le_of_lt hk) w)))
          ⟨k, hk⟩ (spec.groupAtK hk hn y) s]
    -- groupAtK y ≠ g₀ (by injectivity + hy_ne).
    have hg_ne : spec.groupAtK hk hn y ≠ g₀ := by
      intro hg
      apply hy_ne
      exact spec.groupAtK_injective hk hn (hg.trans hy_g₀.symm)
    exact spec.bSide_layerInv_pert_off hZ h2 hF k hk
      (spec.runPrefix (spec.faultedTwiddles Z F) k (Nat.le_of_lt hk) w)
      (spec.groupAtK hk hn y) hg_ne s
  · intro hmem; exact absurd (Finset.mem_univ _) hmem

/-! ### Step F — `teleTerm_images_supIndep` headline -/

/-- If `bSide v ⟨k,_⟩ g₀ s = 0` for every `s`, then `perturbation Z F ⟨k,_⟩ v = 0`. -/
lemma pert_vanish_of_bSide_zero
    (Z : spec.Twiddles K) {F : spec.FaultSet} (hF : spec.OneFaultPerLayer F)
    (k : ℕ) (hk : k < spec.N) (v : Fin spec.n → K)
    (h_van : ∀ s : Fin (spec.L ⟨k, hk⟩),
      spec.bSide v ⟨k, hk⟩ (Finset.card_eq_one.mp (hF ⟨k, hk⟩)).choose s = 0) :
    spec.perturbation Z F ⟨k, hk⟩ v = 0 := by
  set g₀ := (Finset.card_eq_one.mp (hF ⟨k, hk⟩)).choose with hg₀_def
  -- pert acts coordinate-wise: at every i, value is 0.
  -- Show: ∀ i, (pert v) i = 0.
  funext i
  -- Two cases: bitOf k i = 0 (a-side) or = 1 (b-side).
  by_cases hbit : spec.bitOf ⟨k, hk⟩ i = 0
  · -- a-side: i = aSideIdx ⟨k,_⟩ g j.
    obtain ⟨g, j, rfl⟩ := spec.exists_aSide_decomp ⟨k, hk⟩ i hbit
    show spec.aSide (spec.perturbation Z F ⟨k, hk⟩ v) ⟨k, hk⟩ g j = 0
    rw [spec.aSide_perturbation Z F ⟨k, hk⟩ v g j]
    -- (Z g - Z' g) * bSide v g j. Need this = 0.
    by_cases hg : g = g₀
    · subst hg
      -- bSide v g₀ j = 0 by hypothesis.
      rw [h_van j]; ring
    · -- Z g = Z' g for g ≠ g₀.
      have hZ'_eq : (spec.faultedTwiddles Z F) ⟨k, hk⟩ g = Z ⟨k, hk⟩ g := by
        have hg₀_spec : F ⟨k, hk⟩ = {g₀} :=
          (Finset.card_eq_one.mp (hF ⟨k, hk⟩)).choose_spec
        show (if g ∈ F ⟨k, hk⟩ then 0 else Z ⟨k, hk⟩ g) = Z ⟨k, hk⟩ g
        rw [hg₀_spec, if_neg (by simpa using hg)]
      rw [hZ'_eq, sub_self, zero_mul]
  · -- b-side: i = bSideIdx ⟨k,_⟩ g j.
    have hbit1 : spec.bitOf ⟨k, hk⟩ i = 1 := by
      have h_lt : spec.bitOf ⟨k, hk⟩ i < 2 := Nat.mod_lt _ (by norm_num)
      unfold bitOf at hbit ⊢; omega
    obtain ⟨g, j, rfl⟩ := spec.exists_bSide_decomp ⟨k, hk⟩ i hbit1
    show spec.bSide (spec.perturbation Z F ⟨k, hk⟩ v) ⟨k, hk⟩ g j = 0
    rw [spec.bSide_perturbation Z F ⟨k, hk⟩ v g j]
    -- (Z' g - Z g) * bSide v g j. Need this = 0.
    by_cases hg : g = g₀
    · subst hg; rw [h_van j]; ring
    · have hZ'_eq : (spec.faultedTwiddles Z F) ⟨k, hk⟩ g = Z ⟨k, hk⟩ g := by
        have hg₀_spec : F ⟨k, hk⟩ = {g₀} :=
          (Finset.card_eq_one.mp (hF ⟨k, hk⟩)).choose_spec
        show (if g ∈ F ⟨k, hk⟩ then 0 else Z ⟨k, hk⟩ g) = Z ⟨k, hk⟩ g
        rw [hg₀_spec, if_neg (by simpa using hg)]
      rw [hZ'_eq, sub_self, zero_mul]

/-- If `pert (runPrefix Z' k w) = 0`, then `teleTerm Z Z' k w = 0`. -/
lemma teleTerm_eq_zero_of_pert_zero
    {Z : spec.Twiddles K} (F : spec.FaultSet)
    (k : ℕ) (hk : k < spec.N) (w : Fin spec.n → K)
    (h_pert : spec.perturbation Z F ⟨k, hk⟩
        (spec.runPrefix (spec.faultedTwiddles Z F) k (Nat.le_of_lt hk) w) = 0) :
    spec.teleTerm Z (spec.faultedTwiddles Z F) k w = 0 := by
  rw [spec.teleTerm_factor Z (spec.faultedTwiddles Z F) k hk,
      spec.layer_diff_eq_perturbation Z F k hk]
  rw [LinearMap.comp_apply, LinearMap.comp_apply]
  rw [h_pert]
  exact map_zero _

/-- **F8 headline (asymmetric form for F9).** For any layer `a` and a vector
    `v ∈ range(teleTerm Z (faultedTwiddles Z F) a.val)`, if `v` also lies in the
    supremum of `range(teleTerm Z (faultedTwiddles Z F) b.val)` over `b < a`,
    then `v = 0`. F9 will use this for the rank-additivity argument by peeling
    from the top. -/
theorem teleTerm_image_disjoint_lower
    [KyberLike spec]
    {Z : spec.Twiddles K} (hZ : ∀ ℓ g, Z ℓ g ≠ 0) (h2 : (2 : K) ≠ 0)
    {F : spec.FaultSet} (hF : spec.OneFaultPerLayer F)
    (a : Fin spec.N) :
    Disjoint
      (LinearMap.range (spec.teleTerm Z (spec.faultedTwiddles Z F) a.val))
      (⨆ b : Fin spec.N, ⨆ _ : b.val < a.val,
        LinearMap.range (spec.teleTerm Z (spec.faultedTwiddles Z F) b.val)) := by
  rw [Submodule.disjoint_def]
  intro v hv_a hv_sup
  obtain ⟨w_a, hw_a⟩ := hv_a
  set ha := a.isLt
  -- Step 1: nttInv v ∈ ⨆ b < a, V_b. Use range_le for each individual range, then iSup_le.
  have h_each : ∀ b : Fin spec.N, b.val < a.val →
      LinearMap.range (spec.teleTerm Z (spec.faultedTwiddles Z F) b.val)
        ≤ (spec.V (K := K) b).comap (spec.nttInv hZ h2) := by
    intro b hba
    intro w hw
    -- nttInv w ∈ V_b by F7 (which gives range(nttInv ∘ teleTerm) ⊆ V_b).
    obtain ⟨u, hu⟩ := hw
    have h_in : spec.nttInv hZ h2 (spec.teleTerm Z (spec.faultedTwiddles Z F) b.val u)
                  ∈ spec.V (K := K) b :=
      spec.teleTerm_intt_image_subset_V hZ h2 hF b.val b.isLt ⟨u, rfl⟩
    show spec.nttInv hZ h2 w ∈ spec.V (K := K) b
    rw [← hu]
    exact h_in
  -- Combine into ⨆_lemma form.
  have h_combined :
      (⨆ b : Fin spec.N, ⨆ _ : b.val < a.val,
        LinearMap.range (spec.teleTerm Z (spec.faultedTwiddles Z F) b.val))
        ≤ (⨆ b : Fin spec.N, ⨆ _ : b.val < a.val, spec.V (K := K) b).comap
            (spec.nttInv hZ h2) := by
    refine iSup_le (fun b => iSup_le (fun hba => ?_))
    refine le_trans (h_each b hba) ?_
    refine Submodule.comap_mono ?_
    exact le_iSup_of_le b (le_iSup_of_le hba le_rfl)
  have h_nttInv_v_in : spec.nttInv hZ h2 v ∈
      (⨆ b : Fin spec.N, ⨆ _ : b.val < a.val, spec.V (K := K) b) :=
    h_combined hv_sup
  -- Step 2: nttInv v vanishes at freshIdx ⟨a.val, ha⟩ s for every s.
  -- Use iSup_induction with the predicate "value at freshIdx ⟨a.val,_⟩ s = 0".
  have h_van : ∀ s : Fin (spec.L ⟨a.val, ha⟩),
      spec.nttInv hZ h2 v (spec.freshIdx ha s) = 0 := by
    intro s
    refine Submodule.iSup_induction
      (fun b : Fin spec.N => ⨆ _ : b.val < a.val, spec.V (K := K) b)
      (motive := fun w => w (spec.freshIdx ha s) = 0)
      h_nttInv_v_in ?_ rfl ?_
    · intro b w hw
      -- hw : w ∈ ⨆ _ : b.val < a.val, V_b.
      -- If b.val < a.val, w ∈ V_b ⇒ w vanishes at freshIdx (since bit b = 0).
      refine Submodule.iSup_induction
        (fun _ : b.val < a.val => spec.V (K := K) b)
        (motive := fun w => w (spec.freshIdx ha s) = 0)
        hw ?_ rfl ?_
      · intro hba w' hw'
        have hbit : spec.bitOf b (spec.freshIdx ha s) = 0 := by
          show spec.bitOf ⟨b.val, b.isLt⟩ (spec.freshIdx ha s) = 0
          exact spec.bitOf_freshIdx_lower ha b.isLt hba s
        exact spec.bit_zero_vanish_of_mem_V b w' hw' (spec.freshIdx ha s) hbit
      · intro u v hu hv
        show (u + v) _ = 0
        rw [Pi.add_apply, hu, hv, add_zero]
    · intro u v hu hv
      show (u + v) _ = 0
      rw [Pi.add_apply, hu, hv, add_zero]
  -- Step 3: pi_scalar_at_fresh ⇒ bSide ... = 0 for all s.
  have h_bSide_zero : ∀ s : Fin (spec.L ⟨a.val, ha⟩),
      spec.bSide
          (spec.runPrefix (spec.faultedTwiddles Z F) a.val (Nat.le_of_lt ha) w_a)
          ⟨a.val, ha⟩
          (Finset.card_eq_one.mp (hF ⟨a.val, ha⟩)).choose s = 0 := by
    intro s
    have hpi := spec.pi_scalar_at_fresh hZ h2 hF a.val ha w_a s
    rw [hw_a] at hpi
    rw [h_van s] at hpi
    have h_pow_ne : (1 / (2 : K))^a.val ≠ 0 := pow_ne_zero _ (one_div_ne_zero h2)
    exact (mul_eq_zero.mp hpi.symm).resolve_left h_pow_ne
  -- Step 4: pert = 0, hence teleTerm w_a = 0, hence v = 0.
  have h_pert : spec.perturbation Z F ⟨a.val, ha⟩
      (spec.runPrefix (spec.faultedTwiddles Z F) a.val (Nat.le_of_lt ha) w_a) = 0 :=
    spec.pert_vanish_of_bSide_zero Z hF a.val ha _ h_bSide_zero
  have h_teleTerm : spec.teleTerm Z (spec.faultedTwiddles Z F) a.val w_a = 0 :=
    spec.teleTerm_eq_zero_of_pert_zero F a.val ha w_a h_pert
  rw [← hw_a, h_teleTerm]

end LayerSpec
end NttFaultRank
