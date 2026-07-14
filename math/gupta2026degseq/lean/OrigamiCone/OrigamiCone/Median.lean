import OrigamiCone.LevelCrossing

/-!
# Median form of the dispersion (Section 4)

Specialisation of the K-parametric level-crossing identity
(`disp_eq_levelCrossing` in `LevelCrossing.lean`) to the paper's published form
(§4, lines 1011–1013):

  `min_K disp φ K = ∑_ℓ min(c_ℓ, N − c_ℓ)`,  where `c_ℓ = #{v : φ v ≤ ℓ}`.

The argument routes through a `K`-parametric decomposition
`disp φ K = ∑_{[L,K)} c_ℓ + ∑_{[K,U)} (N − c_ℓ)` and the monotonicity-prefix
fact that the "small-c_ℓ" levels form an initial segment of `[L, U)`.

Results:
* `cLeq`, `cLeq_nonneg`, `cLeq_le_card`, `cLeq_mono` — the cumulative
  sublevel count `#{v : φ v ≤ ℓ}` and basic facts;
* `crossingCount_eq` — per-level rewrite of the level-crossing summand:
  `c_ℓ` if `ℓ < K`, else `N − c_ℓ`;
* `disp_eq_cLeq_sum` — the `K`-parametric decomposition of `disp φ K`;
* `disp_ge_sum_min` — `disp φ K ≥ ∑_{[L,U)} min(c_ℓ, N − c_ℓ)` for every `K`;
* `medianK`, `medianK_ge`, `medianK_le` — the constructive optimum
  `K* := L + #{ℓ ∈ [L,U) : 2 c_ℓ ≤ N}` and its bounds;
* `Ico_prefix_eq` — monotonicity-prefix structure of the small-`c_ℓ` levels;
* `isMedianMin_sum_min` — full characterisation: `IsMedianMin φ D`
  where `D := ∑_{[L,U)} min(c_ℓ, N − c_ℓ)`.

No `sorry`.
-/

namespace OrigamiCone

variable {m n : ℕ}

/-! ## Cumulative sublevel count `cLeq` -/

/-- **Cumulative sublevel count.** The number of cells with `φ v ≤ ℓ`, cast to
ℤ for convenient algebra against `disp`. -/
def cLeq (φ : Cell m n → ℤ) (ℓ : ℤ) : ℤ :=
  ((Finset.univ : Finset (Cell m n)).filter (fun v => φ v ≤ ℓ)).card

lemma cLeq_nonneg (φ : Cell m n → ℤ) (ℓ : ℤ) : 0 ≤ cLeq φ ℓ :=
  Int.natCast_nonneg _

lemma cLeq_le_card (φ : Cell m n → ℤ) (ℓ : ℤ) :
    cLeq φ ℓ ≤ (Fintype.card (Cell m n) : ℤ) := by
  unfold cLeq
  rw [← Finset.card_univ]
  exact_mod_cast Finset.card_filter_le _ _

lemma cLeq_mono (φ : Cell m n → ℤ) {ℓ ℓ' : ℤ} (h : ℓ ≤ ℓ') :
    cLeq φ ℓ ≤ cLeq φ ℓ' := by
  unfold cLeq
  have hsub : (Finset.univ : Finset (Cell m n)).filter (fun v => φ v ≤ ℓ) ⊆
      Finset.univ.filter (fun v => φ v ≤ ℓ') := by
    intro v hv
    rw [Finset.mem_filter] at hv ⊢
    exact ⟨hv.1, hv.2.trans h⟩
  exact_mod_cast Finset.card_le_card hsub

/-! ## Per-level rewrite of the crossing count -/

/-- **Per-level crossing-count rewrite.**  The crossing count at level `ℓ`
(number of cells `v` with the integer `ℓ` strictly between `min(φv, K)` and
`max(φv, K)`) equals `cLeq φ ℓ` when `ℓ < K`, and `N − cLeq φ ℓ` when `K ≤ ℓ`. -/
lemma crossingCount_eq (φ : Cell m n → ℤ) (K ℓ : ℤ) :
    (((Finset.univ : Finset (Cell m n)).filter
        (fun v => min (φ v) K ≤ ℓ ∧ ℓ < max (φ v) K)).card : ℤ) =
      if ℓ < K then cLeq φ ℓ else (Fintype.card (Cell m n) : ℤ) - cLeq φ ℓ := by
  split_ifs with hℓK
  · -- ℓ < K: crossing filter equals "φ v ≤ ℓ" filter.
    unfold cLeq
    have heq :
        ((Finset.univ : Finset (Cell m n)).filter
          (fun v => min (φ v) K ≤ ℓ ∧ ℓ < max (φ v) K)) =
        Finset.univ.filter (fun v => φ v ≤ ℓ) := by
      apply Finset.filter_congr
      intro v _
      constructor
      · rintro ⟨hmin, _⟩
        rcases (by omega : φ v ≤ K ∨ K < φ v) with hvK | hvK
        · rwa [min_eq_left hvK] at hmin
        · rw [min_eq_right hvK.le] at hmin; omega
      · intro hφℓ
        have hvK : φ v ≤ K := hφℓ.trans hℓK.le
        rw [min_eq_left hvK, max_eq_right hvK]
        exact ⟨hφℓ, hℓK⟩
    rw [heq]
  · -- K ≤ ℓ: crossing filter equals "ℓ < φ v" filter, whose card is N − cLeq.
    push_neg at hℓK
    have heq :
        ((Finset.univ : Finset (Cell m n)).filter
          (fun v => min (φ v) K ≤ ℓ ∧ ℓ < max (φ v) K)) =
        Finset.univ.filter (fun v => ℓ < φ v) := by
      apply Finset.filter_congr
      intro v _
      constructor
      · rintro ⟨_, hmax⟩
        rcases (by omega : φ v ≤ K ∨ K < φ v) with hvK | hvK
        · rw [max_eq_right hvK] at hmax; omega
        · rwa [max_eq_left hvK.le] at hmax
      · intro hℓφ
        have hvK : K < φ v := lt_of_le_of_lt hℓK hℓφ
        rw [min_eq_right hvK.le, max_eq_left hvK.le]
        exact ⟨hℓK, hℓφ⟩
    rw [heq]
    -- Use "(filter p).card + (filter ¬p).card = card univ".
    unfold cLeq
    have heq2 : ((Finset.univ : Finset (Cell m n)).filter (fun v => ℓ < φ v)) =
        Finset.univ.filter (fun v => ¬ (φ v ≤ ℓ)) := by
      apply Finset.filter_congr
      intro v _
      exact ⟨not_le.mpr, fun h => by omega⟩
    rw [heq2]
    have hsum :=
      Finset.card_filter_add_card_filter_not
        (s := (Finset.univ : Finset (Cell m n))) (p := fun v => φ v ≤ ℓ)
    rw [Finset.card_univ] at hsum
    have : (((Finset.univ : Finset (Cell m n)).filter (fun v => ¬ φ v ≤ ℓ)).card : ℤ) =
        (Fintype.card (Cell m n) : ℤ) -
          (((Finset.univ : Finset (Cell m n)).filter (fun v => φ v ≤ ℓ)).card : ℤ) := by
      have h1 : (((Finset.univ : Finset (Cell m n)).filter (fun v => φ v ≤ ℓ)).card : ℤ) +
                (((Finset.univ : Finset (Cell m n)).filter (fun v => ¬ φ v ≤ ℓ)).card : ℤ) =
                (Fintype.card (Cell m n) : ℤ) := by exact_mod_cast hsum
      linarith
    exact this

/-! ## `K`-parametric decomposition of `disp` -/

/-- **`K`-parametric decomposition of `disp`.**  For any uniform range
`[L, U)` containing all values `φ v` and the offset `K`, the dispersion
decomposes as the sum of `c_ℓ` over levels below `K` plus the sum of `N − c_ℓ`
over levels at or above `K`. -/
theorem disp_eq_cLeq_sum (φ : Cell m n → ℤ) (K L U : ℤ)
    (hK : L ≤ K ∧ K ≤ U) (hφ : ∀ v, L ≤ φ v ∧ φ v ≤ U) :
    disp φ K =
      (∑ ℓ ∈ Finset.Ico L K, cLeq φ ℓ) +
      (∑ ℓ ∈ Finset.Ico K U,
        ((Fintype.card (Cell m n) : ℤ) - cLeq φ ℓ)) := by
  rw [disp_eq_levelCrossing φ K L U hK hφ]
  rw [show (Finset.Ico L U) = Finset.Ico L K ∪ Finset.Ico K U from ?_]
  · rw [Finset.sum_union (Finset.Ico_disjoint_Ico_consecutive L K U)]
    congr 1
    · apply Finset.sum_congr rfl
      intro ℓ hℓ
      rw [Finset.mem_Ico] at hℓ
      have := crossingCount_eq φ K ℓ
      rw [if_pos hℓ.2] at this
      exact this
    · apply Finset.sum_congr rfl
      intro ℓ hℓ
      rw [Finset.mem_Ico] at hℓ
      have := crossingCount_eq φ K ℓ
      rw [if_neg (not_lt.mpr hℓ.1)] at this
      exact this
  · exact (Finset.Ico_union_Ico_eq_Ico hK.1 hK.2).symm

/-! ## Lower bound: `disp φ K ≥ ∑_ℓ min(c_ℓ, N − c_ℓ)` -/

/-- **Lower bound of dispersion via the level-crossing identity.**  For every
offset `K` (within the uniform range `[L, U)`), the dispersion is at least the
termwise `min(c_ℓ, N − c_ℓ)` sum. -/
theorem disp_ge_sum_min (φ : Cell m n → ℤ) (K L U : ℤ)
    (hK : L ≤ K ∧ K ≤ U) (hφ : ∀ v, L ≤ φ v ∧ φ v ≤ U) :
    (∑ ℓ ∈ Finset.Ico L U,
        min (cLeq φ ℓ) ((Fintype.card (Cell m n) : ℤ) - cLeq φ ℓ))
      ≤ disp φ K := by
  rw [disp_eq_cLeq_sum φ K L U hK hφ,
      show (Finset.Ico L U) = Finset.Ico L K ∪ Finset.Ico K U from
        (Finset.Ico_union_Ico_eq_Ico hK.1 hK.2).symm,
      Finset.sum_union (Finset.Ico_disjoint_Ico_consecutive L K U)]
  gcongr ?_ + ?_
  · apply Finset.sum_le_sum
    intro ℓ _; exact min_le_left _ _
  · apply Finset.sum_le_sum
    intro ℓ _; exact min_le_right _ _

/-! ## The median offset `medianK` and the full characterisation -/

/-- **The median offset.**  Defined as `L + (# of levels in [L, U) with
`2 · c_ℓ ≤ N`)`.  By monotonicity of `c_ℓ`, this is exactly the right shift
to make the per-level minimum split at this `K`. -/
def medianK (φ : Cell m n → ℤ) (L U : ℤ) : ℤ :=
  L + ((Finset.Ico L U).filter
        (fun ℓ => 2 * cLeq φ ℓ ≤ (Fintype.card (Cell m n) : ℤ))).card

lemma medianK_ge (φ : Cell m n → ℤ) (L U : ℤ) : L ≤ medianK φ L U := by
  unfold medianK
  have : (0 : ℤ) ≤ (((Finset.Ico L U).filter
      (fun ℓ => 2 * cLeq φ ℓ ≤ (Fintype.card (Cell m n) : ℤ))).card : ℤ) :=
    Int.natCast_nonneg _
  linarith

lemma medianK_le (φ : Cell m n → ℤ) (L U : ℤ) (hLU : L ≤ U) :
    medianK φ L U ≤ U := by
  unfold medianK
  have h : ((Finset.Ico L U).filter
      (fun ℓ => 2 * cLeq φ ℓ ≤ (Fintype.card (Cell m n) : ℤ))).card ≤
      (Finset.Ico L U).card := Finset.card_filter_le _ _
  have h2 : ((Finset.Ico L U).card : ℤ) = U - L := by
    rw [Int.card_Ico, Int.toNat_of_nonneg (by linarith)]
  have h3 : (((Finset.Ico L U).filter
        (fun ℓ => 2 * cLeq φ ℓ ≤ (Fintype.card (Cell m n) : ℤ))).card : ℤ) ≤
      ((Finset.Ico L U).card : ℤ) := by exact_mod_cast h
  linarith

/-- **Prefix property:** in `Ico L U`, the levels with `2 c_ℓ ≤ N` are precisely
the prefix `Ico L (medianK φ L U)`.  By monotonicity of `cLeq`. -/
lemma Ico_prefix_eq (φ : Cell m n → ℤ) (L U : ℤ) :
    Finset.Ico L (medianK φ L U) =
    (Finset.Ico L U).filter (fun ℓ => 2 * cLeq φ ℓ ≤ (Fintype.card (Cell m n) : ℤ)) := by
  set S := (Finset.Ico L U).filter
    (fun ℓ => 2 * cLeq φ ℓ ≤ (Fintype.card (Cell m n) : ℤ)) with hSdef
  symm
  apply Finset.eq_of_subset_of_card_le
  · -- S ⊆ Ico L (medianK φ L U)
    intro ℓ hℓ
    rw [hSdef, Finset.mem_filter, Finset.mem_Ico] at hℓ
    rw [Finset.mem_Ico]
    refine ⟨hℓ.1.1, ?_⟩
    -- Show ℓ < medianK φ L U = L + S.card.  By prefix property,
    -- all ℓ' ∈ [L, ℓ] are in S, so |S| ≥ ℓ − L + 1.
    have hpref : Finset.Ico L (ℓ + 1) ⊆ S := by
      intro ℓ' hℓ'
      rw [Finset.mem_Ico] at hℓ'
      rw [hSdef, Finset.mem_filter, Finset.mem_Ico]
      refine ⟨⟨hℓ'.1, by omega⟩, ?_⟩
      have hmono := cLeq_mono φ (by omega : ℓ' ≤ ℓ)
      linarith [hℓ.2]
    have hcard : (Finset.Ico L (ℓ + 1)).card ≤ S.card := Finset.card_le_card hpref
    have hIcocard : (Finset.Ico L (ℓ + 1)).card = (ℓ + 1 - L).toNat := Int.card_Ico _ _
    have hcardZ : (ℓ + 1 - L : ℤ) ≤ (S.card : ℤ) := by
      rw [hIcocard] at hcard
      have h2 : ((ℓ + 1 - L).toNat : ℤ) = ℓ + 1 - L :=
        Int.toNat_of_nonneg (by linarith)
      have h1 : ((ℓ + 1 - L).toNat : ℤ) ≤ (S.card : ℤ) := by exact_mod_cast hcard
      linarith
    unfold medianK
    linarith
  · -- (Ico L (medianK)).card ≤ S.card
    rw [Int.card_Ico]
    unfold medianK
    rw [show L + (S.card : ℤ) - L = ((S.card : ℕ) : ℤ) from by ring,
        Int.toNat_natCast]

/-- **`disp φ` is minimised by `medianK φ L U`, with value `∑_ℓ min(c_ℓ, N − c_ℓ)`.**
This is the paper-faithful **median characterisation** of the dispersion
minimum (§4, lines 1011–1013). -/
theorem isMedianMin_sum_min (φ : Cell m n → ℤ) (L U : ℤ)
    (hLU : L ≤ U) (hφ : ∀ v, L ≤ φ v ∧ φ v ≤ U) :
    IsMedianMin φ
      (∑ ℓ ∈ Finset.Ico L U,
        min (cLeq φ ℓ) ((Fintype.card (Cell m n) : ℤ) - cLeq φ ℓ)) := by
  refine ⟨fun K => ?_, ⟨medianK φ L U, ?_⟩⟩
  · -- Lower bound for arbitrary `K`.
    -- Clip `K` to `[L, U]`; `disp` at clipped `K` equals or is ≥ that at `K`.
    -- Actually the lower bound for K ∈ [L, U] is `disp_ge_sum_min`; for K
    -- outside, `disp φ K` is even larger because the function is convex.
    -- We handle by cases: K < L, K ∈ [L, U], K > U.
    rcases (by omega : K < L ∨ L ≤ K) with hKL | hKL
    · -- K < L: every |φ v − K| = (φ v − K) ≥ (φ v − L), so disp φ K ≥ disp φ L.
      have hge : disp φ L ≤ disp φ K := by
        unfold disp
        apply Finset.sum_le_sum
        intro v _
        have hv := (hφ v).1
        rw [abs_of_nonneg (by linarith), abs_of_nonneg (by linarith)]
        linarith
      calc (∑ ℓ ∈ Finset.Ico L U, min (cLeq φ ℓ)
              ((Fintype.card (Cell m n) : ℤ) - cLeq φ ℓ))
          ≤ disp φ L := disp_ge_sum_min φ L L U ⟨le_refl L, hLU⟩ hφ
        _ ≤ disp φ K := hge
    · rcases (by omega : K ≤ U ∨ U < K) with hKU | hKU
      · -- L ≤ K ≤ U: direct.
        exact disp_ge_sum_min φ K L U ⟨hKL, hKU⟩ hφ
      · -- K > U: every |φ v − K| = (K − φ v) ≥ (U − φ v), so disp φ K ≥ disp φ U.
        have hge : disp φ U ≤ disp φ K := by
          unfold disp
          apply Finset.sum_le_sum
          intro v _
          have hv := (hφ v).2
          rw [abs_of_nonpos (by linarith), abs_of_nonpos (by linarith)]
          linarith
        calc (∑ ℓ ∈ Finset.Ico L U, min (cLeq φ ℓ)
                ((Fintype.card (Cell m n) : ℤ) - cLeq φ ℓ))
            ≤ disp φ U := disp_ge_sum_min φ U L U ⟨hLU, le_refl U⟩ hφ
          _ ≤ disp φ K := hge
  · -- Achievement: `disp φ (medianK φ L U) = ∑_ℓ min(...)`.
    have hKge := medianK_ge φ L U
    have hKle := medianK_le φ L U hLU
    rw [disp_eq_cLeq_sum φ (medianK φ L U) L U ⟨hKge, hKle⟩ hφ]
    -- Split the Σ over Ico L U using the prefix decomposition.
    have hsplit :
        (∑ ℓ ∈ Finset.Ico L U,
            min (cLeq φ ℓ) ((Fintype.card (Cell m n) : ℤ) - cLeq φ ℓ)) =
          (∑ ℓ ∈ Finset.Ico L (medianK φ L U),
              min (cLeq φ ℓ) ((Fintype.card (Cell m n) : ℤ) - cLeq φ ℓ)) +
          (∑ ℓ ∈ Finset.Ico (medianK φ L U) U,
              min (cLeq φ ℓ) ((Fintype.card (Cell m n) : ℤ) - cLeq φ ℓ)) := by
      rw [show (Finset.Ico L U) = Finset.Ico L (medianK φ L U) ∪
                 Finset.Ico (medianK φ L U) U from
            (Finset.Ico_union_Ico_eq_Ico hKge hKle).symm,
          Finset.sum_union
            (Finset.Ico_disjoint_Ico_consecutive L (medianK φ L U) U)]
    rw [hsplit]
    congr 1
    · -- Σ_{Ico L medianK} cLeq = Σ_{Ico L medianK} min(cLeq, N-cLeq).
      -- For ℓ in this range, 2*cLeq ≤ N (prefix), so cLeq ≤ N-cLeq, so min = cLeq.
      apply Finset.sum_congr rfl
      intro ℓ hℓ
      have hℓinS : ℓ ∈ (Finset.Ico L U).filter
          (fun ℓ => 2 * cLeq φ ℓ ≤ (Fintype.card (Cell m n) : ℤ)) := by
        rw [← Ico_prefix_eq φ L U]; exact hℓ
      rw [Finset.mem_filter] at hℓinS
      have h2c : 2 * cLeq φ ℓ ≤ (Fintype.card (Cell m n) : ℤ) := hℓinS.2
      rw [min_eq_left (by linarith)]
    · -- Σ_{Ico medianK U} (N - cLeq) = Σ_{Ico medianK U} min(cLeq, N - cLeq).
      -- For ℓ ∉ Ico L medianK but ∈ Ico L U: 2*cLeq > N (suffix), so min = N - cLeq.
      apply Finset.sum_congr rfl
      intro ℓ hℓ
      rw [Finset.mem_Ico] at hℓ
      obtain ⟨hge, hlt⟩ := hℓ
      have hℓInU : ℓ ∈ Finset.Ico L U := Finset.mem_Ico.mpr ⟨hKge.trans hge, hlt⟩
      -- Show ℓ ∉ Ico L (medianK), so ℓ ∉ filter (small-c), so 2*cLeq > N.
      have hℓnotPref : ℓ ∉ Finset.Ico L (medianK φ L U) := by
        rw [Finset.mem_Ico]; intro ⟨_, hlt'⟩; omega
      rw [Ico_prefix_eq φ L U] at hℓnotPref
      have hℓnotS : ℓ ∉ (Finset.Ico L U).filter
          (fun ℓ => 2 * cLeq φ ℓ ≤ (Fintype.card (Cell m n) : ℤ)) := hℓnotPref
      rw [Finset.mem_filter, not_and_or] at hℓnotS
      have h2c : (Fintype.card (Cell m n) : ℤ) < 2 * cLeq φ ℓ := by
        rcases hℓnotS with h | h
        · exact absurd hℓInU h
        · push_neg at h; exact h
      rw [min_eq_right (by linarith)]

end OrigamiCone
