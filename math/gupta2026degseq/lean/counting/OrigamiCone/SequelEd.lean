import OrigamiCone.QuotientDegree
import OrigamiCone.SequelPolyAssembly
import Mathlib.Data.Fintype.Pi

namespace OrigamiCone.Sequel

open OrigamiCone

variable {m n : ℕ}

def numExtrema (h : Cell m n → ℤ) : ℕ :=
  (Finset.univ.filter (fun v : Cell m n => IsStrictLocalExtremum h v)).card

def IsCanonicalHeight (h : Cell m n → ℤ) : Prop :=
  IsHeight h ∧ ∀ p : Cell m n, p.1.val = 0 → p.2.val = 0 → h p = 0

def CanonicalHeights (m n : ℕ) : Set (Cell m n → ℤ) :=
  { h | IsCanonicalHeight h }

noncomputable def Ed (d m n : ℕ) : ℕ :=
  { h : Cell m n → ℤ | IsCanonicalHeight h ∧ numExtrema h = d }.ncard

theorem Ed_def (d m n : ℕ) :
    Ed d m n
      = { h : Cell m n → ℤ | IsCanonicalHeight h ∧ numExtrema h = d }.ncard :=
  rfl

theorem canonical_heights_partition_by_extrema :
    CanonicalHeights m n
      = ⋃ d, { h : Cell m n → ℤ | IsCanonicalHeight h ∧ numExtrema h = d } := by
  ext h
  simp only [CanonicalHeights, Set.mem_iUnion, Set.mem_setOf_eq]
  refine ⟨fun hc => ⟨numExtrema h, hc, rfl⟩, ?_⟩
  rintro ⟨_, hc, _⟩
  exact hc

theorem Ed_fibres_disjoint (d₁ d₂ : ℕ) (hd : d₁ ≠ d₂) :
    Disjoint
      ({ h : Cell m n → ℤ | IsCanonicalHeight h ∧ numExtrema h = d₁ })
      ({ h : Cell m n → ℤ | IsCanonicalHeight h ∧ numExtrema h = d₂ }) := by
  rw [Set.disjoint_iff_forall_ne]
  rintro h ⟨_, hd₁⟩ h' ⟨_, hd₂⟩ rfl
  exact hd (hd₁.symm.trans hd₂)

theorem CanonicalHeights_finite (hm : 1 ≤ m) (hn : 1 ≤ n) :
    (CanonicalHeights m n).Finite := by
  set B : ℕ := m + n - 2 with hB_def

  have h_bounded_finite :
      {h : Cell m n → ℤ | ∀ p, h p ∈ Set.Icc (-(B : ℤ)) (B : ℤ)}.Finite :=
    Set.Finite.pi' (fun _ => Set.finite_Icc _ _)
  refine h_bounded_finite.subset ?_
  unfold CanonicalHeights IsCanonicalHeight
  rintro h ⟨hh_height, hh_zero⟩ p

  let origin : Cell m n := (⟨0, hm⟩, ⟨0, hn⟩)

  have h_origin_zero : h origin = 0 := hh_zero origin rfl rfl

  have hp1 : p.1.val ≤ m - 1 := Nat.le_sub_one_of_lt p.1.isLt
  have hp2 : p.2.val ≤ n - 1 := Nat.le_sub_one_of_lt p.2.isLt
  have h_gdist_bound : gdist origin p ≤ (B : ℤ) := by
    show (((((⟨0, hm⟩ : Fin m)).val : ℤ) - (p.1.val : ℤ)).natAbs
          + (((⟨0, hn⟩ : Fin n).val : ℤ) - (p.2.val : ℤ)).natAbs : ℕ) ≤ (B : ℤ)
    have hn1 : ((((⟨0, hm⟩ : Fin m)).val : ℤ) - (p.1.val : ℤ)).natAbs = p.1.val := by
      simp
    have hn2 : ((((⟨0, hn⟩ : Fin n)).val : ℤ) - (p.2.val : ℤ)).natAbs = p.2.val := by
      simp
    rw [hn1, hn2]
    push_cast
    omega

  have h_gd_nonneg : (0 : ℤ) ≤ gdist origin p := gdist_nonneg _ _
  have h_toNat : ((gdist origin p).toNat : ℤ) = gdist origin p :=
    Int.toNat_of_nonneg h_gd_nonneg
  have h_lip : |h origin - h p| ≤ gdist origin p :=
    height_lipschitz hh_height (gdist origin p).toNat origin p (by rw [h_toNat])
  rw [h_origin_zero, zero_sub, abs_neg] at h_lip

  rw [Set.mem_Icc]
  have h_abs : |h p| ≤ (B : ℤ) := le_trans h_lip h_gdist_bound
  exact abs_le.mp h_abs

theorem Ed_fibre_finite (d : ℕ) (hm : 1 ≤ m) (hn : 1 ≤ n) :
    { h : Cell m n → ℤ | IsCanonicalHeight h ∧ numExtrema h = d }.Finite := by
  refine (CanonicalHeights_finite hm hn).subset ?_
  rintro h ⟨hc, _⟩
  exact hc

theorem Ed_eq_natCard (d m n : ℕ) :
    Ed d m n
      = Nat.card
          { h : Cell m n → ℤ // IsCanonicalHeight h ∧ numExtrema h = d } :=
  (Nat.card_coe_set_eq _).symm

theorem Ed_eq_finset_card (d : ℕ) (hm : 1 ≤ m) (hn : 1 ≤ n) :
    Ed d m n = (Ed_fibre_finite d hm hn).toFinset.card :=
  Set.ncard_eq_toFinset_card _ (Ed_fibre_finite d hm hn)

theorem Ed_le_ncard_canonicalHeights (d : ℕ) (hm : 1 ≤ m) (hn : 1 ≤ n) :
    Ed d m n ≤ (CanonicalHeights m n).ncard :=
  Set.ncard_le_ncard (fun _ ⟨hc, _⟩ => hc) (CanonicalHeights_finite hm hn)

theorem numExtrema_le_mn (h : Cell m n → ℤ) : numExtrema h ≤ m * n := by
  have h_le : numExtrema h ≤ (Finset.univ : Finset (Cell m n)).card :=
    Finset.card_filter_le _ _
  rw [Finset.card_univ, Fintype.card_prod, Fintype.card_fin, Fintype.card_fin] at h_le
  exact h_le

theorem sum_Ed_eq_ncard_canonicalHeights (hm : 1 ≤ m) (hn : 1 ≤ n) :
    ∑ d ∈ Finset.range (m * n + 1), Ed d m n = (CanonicalHeights m n).ncard := by
  rw [Set.ncard_eq_toFinset_card _ (CanonicalHeights_finite hm hn)]
  have h_num_le :
      Set.MapsTo numExtrema
        (↑((CanonicalHeights_finite hm hn).toFinset) : Set (Cell m n → ℤ))
        (↑(Finset.range (m * n + 1)) : Set ℕ) := by
    intro h _
    rw [Finset.mem_coe, Finset.mem_range]
    exact Nat.lt_succ_of_le (numExtrema_le_mn h)
  rw [Finset.card_eq_sum_card_fiberwise h_num_le]
  refine Finset.sum_congr rfl fun d _ => ?_
  rw [Ed_eq_finset_card _ hm hn]
  congr 1
  ext h
  simp only [Set.Finite.mem_toFinset, Finset.mem_filter, Set.mem_setOf_eq]
  tauto

open QuotientModel in

theorem canonical_unique (hm : 1 ≤ m) (hn : 1 ≤ n)
    {h h' : Cell m n → ℤ} (hc : IsCanonicalHeight h)
    (hc' : IsCanonicalHeight h') (heq : ShiftEq h h') : h = h' := by
  obtain ⟨k, hk⟩ := heq

  have h_orig : h ((⟨0, hm⟩, ⟨0, hn⟩) : Cell m n) = 0 :=
    hc.2 _ rfl rfl
  have h'_orig : h' ((⟨0, hm⟩, ⟨0, hn⟩) : Cell m n) = 0 :=
    hc'.2 _ rfl rfl
  have hk_zero : k = 0 := by
    have := congrFun hk ((⟨0, hm⟩, ⟨0, hn⟩) : Cell m n)
    rw [h'_orig, h_orig] at this
    linarith
  subst hk_zero
  funext v
  have := congrFun hk v
  simpa using this.symm

def canonicalise (h : Cell m n → ℤ) (hm : 1 ≤ m) (hn : 1 ≤ n) :
    Cell m n → ℤ :=
  fun p => h p - h ((⟨0, hm⟩, ⟨0, hn⟩) : Cell m n)

theorem canonicalise_shiftEq (h : Cell m n → ℤ) (hm : 1 ≤ m) (hn : 1 ≤ n) :
    QuotientModel.ShiftEq h (canonicalise h hm hn) := by
  refine ⟨-h ((⟨0, hm⟩, ⟨0, hn⟩) : Cell m n), ?_⟩
  funext v
  unfold canonicalise
  ring

theorem canonicalise_isHeight (hh : IsHeight h) (hm : 1 ≤ m) (hn : 1 ≤ n) :
    IsHeight (canonicalise h hm hn) := by

  have : canonicalise h hm hn
      = fun p => h p + (-h ((⟨0, hm⟩, ⟨0, hn⟩) : Cell m n)) := by
    funext p
    unfold canonicalise
    ring
  rw [this]
  exact QuotientModel.isHeight_add_const hh _

theorem canonicalise_isCanonicalHeight (hh : IsHeight h)
    (hm : 1 ≤ m) (hn : 1 ≤ n) :
    IsCanonicalHeight (canonicalise h hm hn) := by
  refine ⟨canonicalise_isHeight hh hm hn, ?_⟩
  intro p hp1 hp2

  have hp_eq : p = ((⟨0, hm⟩, ⟨0, hn⟩) : Cell m n) := by
    ext <;> simp [hp1, hp2]
  rw [hp_eq]
  unfold canonicalise
  ring

theorem numExtrema_canonicalise (h : Cell m n → ℤ) (hm : 1 ≤ m) (hn : 1 ≤ n) :
    numExtrema (canonicalise h hm hn) = numExtrema h := by
  unfold numExtrema
  congr 1
  ext v
  simp only [Finset.mem_filter, Finset.mem_univ, true_and]
  have heq : canonicalise h hm hn
      = fun p => h p + (-h ((⟨0, hm⟩, ⟨0, hn⟩) : Cell m n)) := by
    funext p; unfold canonicalise; ring
  rw [heq]
  exact QuotientModel.strictExtremum_add_const

def HeightOFGVertex (m n : ℕ) : Set (QuotientModel.OFGVertex m n) :=
  { C | ∃ h : Cell m n → ℤ, IsHeight h ∧ QuotientModel.mkV h = C }

theorem mkV_injOn_canonical (hm : 1 ≤ m) (hn : 1 ≤ n) :
    Set.InjOn QuotientModel.mkV (CanonicalHeights m n) := by
  intro h hh h' hh' heq
  rw [QuotientModel.mkV_eq_iff] at heq
  exact canonical_unique hm hn hh hh' heq

theorem mkV_image_canonical (hm : 1 ≤ m) (hn : 1 ≤ n) :
    QuotientModel.mkV '' (CanonicalHeights m n) = HeightOFGVertex m n := by
  ext C
  constructor
  · rintro ⟨h, hh, rfl⟩
    exact ⟨h, hh.1, rfl⟩
  · rintro ⟨h, hh, rfl⟩
    refine ⟨canonicalise h hm hn, canonicalise_isCanonicalHeight hh hm hn, ?_⟩

    exact QuotientModel.mkV_eq_iff.mpr (canonicalise_shiftEq h hm hn).symm

theorem canonicalHeights_ncard_eq_heightOFGVertex (hm : 1 ≤ m) (hn : 1 ≤ n) :
    (CanonicalHeights m n).ncard = (HeightOFGVertex m n).ncard := by
  rw [← mkV_image_canonical hm hn]
  exact (Set.ncard_image_of_injOn (mkV_injOn_canonical hm hn)).symm

theorem mkV_image_fibre (d : ℕ) (hm : 1 ≤ m) (hn : 1 ≤ n) (hmn : 3 ≤ m * n) :
    QuotientModel.mkV ''
        { h : Cell m n → ℤ | IsCanonicalHeight h ∧ numExtrema h = d }
      = { C : QuotientModel.OFGVertex m n |
          C ∈ HeightOFGVertex m n ∧ QuotientModel.OFGDegree C = d } := by
  ext C
  constructor
  · rintro ⟨h, ⟨hc, hd⟩, rfl⟩
    refine ⟨⟨h, hc.1, rfl⟩, ?_⟩

    rw [QuotientModel.ofgDegree_eq_extrema hc.1 hmn]
    exact hd
  · rintro ⟨⟨h_orig, hh_orig, hmkv⟩, hOFGd⟩
    refine ⟨canonicalise h_orig hm hn,
      ⟨canonicalise_isCanonicalHeight hh_orig hm hn, ?_⟩, ?_⟩
    ·
      rw [numExtrema_canonicalise h_orig hm hn]
      have h1 : QuotientModel.OFGDegree (QuotientModel.mkV h_orig)
                  = (Finset.univ.filter (IsStrictLocalExtremum h_orig)).card :=
        QuotientModel.ofgDegree_eq_extrema hh_orig hmn
      show (Finset.univ.filter (IsStrictLocalExtremum h_orig)).card = d
      rw [← h1, hmkv]
      exact hOFGd
    ·
      rw [← hmkv]
      exact QuotientModel.mkV_eq_iff.mpr (canonicalise_shiftEq h_orig hm hn).symm

theorem Ed_eq_ncard_heightOFG_degree
    (d : ℕ) (hm : 1 ≤ m) (hn : 1 ≤ n) (hmn : 3 ≤ m * n) :
    Ed d m n = { C : QuotientModel.OFGVertex m n |
                 C ∈ HeightOFGVertex m n ∧ QuotientModel.OFGDegree C = d }.ncard := by
  unfold Ed
  have inj : Set.InjOn QuotientModel.mkV
      { h : Cell m n → ℤ | IsCanonicalHeight h ∧ numExtrema h = d } :=
    (mkV_injOn_canonical hm hn).mono (fun _ ⟨hc, _⟩ => hc)
  rw [← mkV_image_fibre d hm hn hmn]
  exact (Set.ncard_image_of_injOn inj).symm

def swapH (h : Cell m n → ℤ) : Cell n m → ℤ := fun q => h q.swap

theorem adj_swap {p q : Cell n m} : adj p q ↔ adj p.swap q.swap := by
  unfold adj gdist
  simp only [Prod.fst_swap, Prod.snd_swap]
  constructor
  · intro h; omega
  · intro h; omega

theorem swapH_isHeight {h : Cell m n → ℤ} (hh : IsHeight h) :
    IsHeight (swapH h) := by
  intro p q hpq
  show |h p.swap - h q.swap| = 1
  exact hh p.swap q.swap (adj_swap.mp hpq)

theorem swapH_isCanonical {h : Cell m n → ℤ} (hc : IsCanonicalHeight h) :
    IsCanonicalHeight (swapH h) := by
  refine ⟨swapH_isHeight hc.1, ?_⟩
  intro p hp1 hp2
  show h p.swap = 0
  exact hc.2 p.swap hp2 hp1

theorem swapH_strictMax_iff {h : Cell m n → ℤ} (v : Cell n m) :
    IsStrictLocalMax (swapH h) v ↔ IsStrictLocalMax h v.swap := by
  constructor
  · intro hmax w hw
    have hu : adj v w.swap := by
      rw [adj_swap]; simpa using hw
    simpa [swapH, Prod.swap_swap] using hmax w.swap hu
  · intro hmax u hu
    show h u.swap = h v.swap - 1
    exact hmax u.swap (adj_swap.mp hu)

theorem swapH_strictMin_iff {h : Cell m n → ℤ} (v : Cell n m) :
    IsStrictLocalMin (swapH h) v ↔ IsStrictLocalMin h v.swap := by
  constructor
  · intro hmin w hw
    have hu : adj v w.swap := by
      rw [adj_swap]; simpa using hw
    simpa [swapH, Prod.swap_swap] using hmin w.swap hu
  · intro hmin u hu
    show h u.swap = h v.swap + 1
    exact hmin u.swap (adj_swap.mp hu)

theorem swapH_strictExtremum_iff {h : Cell m n → ℤ} (v : Cell n m) :
    IsStrictLocalExtremum (swapH h) v ↔ IsStrictLocalExtremum h v.swap := by
  unfold IsStrictLocalExtremum
  rw [swapH_strictMax_iff, swapH_strictMin_iff]

theorem swapH_numExtrema (h : Cell m n → ℤ) :
    numExtrema (swapH h) = numExtrema h := by
  unfold numExtrema
  have h_inj : Function.Injective (Prod.swap : Cell n m → Cell m n) :=
    Prod.swap_injective
  rw [← Finset.card_image_of_injective _ h_inj]
  congr 1
  ext v
  simp only [Finset.mem_image, Finset.mem_filter, Finset.mem_univ, true_and]
  constructor
  · rintro ⟨w, hw, rfl⟩
    exact (swapH_strictExtremum_iff w).mp hw
  · intro hv
    refine ⟨v.swap, ?_, by simp⟩
    rw [swapH_strictExtremum_iff]
    simpa using hv

theorem Ed_symm (d : ℕ) (_hm : 1 ≤ m) (_hn : 1 ≤ n) :
    Ed d m n = Ed d n m := by
  unfold Ed

  have swapH_inj : Function.Injective (@swapH m n) := by
    intro h h' heq
    funext p
    have := congrFun heq p.swap
    simpa [swapH] using this
  have inj : Set.InjOn (@swapH m n)
      { h : Cell m n → ℤ | IsCanonicalHeight h ∧ numExtrema h = d } :=
    swapH_inj.injOn
  have himg : swapH ''
      { h : Cell m n → ℤ | IsCanonicalHeight h ∧ numExtrema h = d }
      = { h : Cell n m → ℤ | IsCanonicalHeight h ∧ numExtrema h = d } := by
    ext h'
    constructor
    · rintro ⟨h, ⟨hc, hd⟩, rfl⟩
      exact ⟨swapH_isCanonical hc, (swapH_numExtrema h).trans hd⟩
    · rintro ⟨hc', hd'⟩

      refine ⟨swapH h', ⟨swapH_isCanonical hc', ?_⟩, ?_⟩
      · exact (swapH_numExtrema h').trans hd'
      · funext p
        simp [swapH]
  rw [← himg, Set.ncard_image_of_injOn inj]

theorem exists_strictLocalMax {h : Cell m n → ℤ} (hh : IsHeight h)
    (hm : 1 ≤ m) (hn : 1 ≤ n) :
    ∃ v : Cell m n, IsStrictLocalMax h v := by
  have h_ne : (Finset.univ : Finset (Cell m n)).Nonempty :=
    ⟨(⟨0, hm⟩, ⟨0, hn⟩), Finset.mem_univ _⟩
  obtain ⟨v, _, hv⟩ := Finset.exists_max_image Finset.univ h h_ne
  refine ⟨v, ?_⟩
  intro u hu
  have h_le : h u ≤ h v := hv u (Finset.mem_univ u)
  have h_diff : |h v - h u| = 1 := hh v u hu
  have h_nonneg : (0 : ℤ) ≤ h v - h u := by linarith
  have : h v - h u = 1 := by rw [abs_of_nonneg h_nonneg] at h_diff; exact h_diff
  linarith

theorem exists_strictLocalMin {h : Cell m n → ℤ} (hh : IsHeight h)
    (hm : 1 ≤ m) (hn : 1 ≤ n) :
    ∃ v : Cell m n, IsStrictLocalMin h v := by
  have h_ne : (Finset.univ : Finset (Cell m n)).Nonempty :=
    ⟨(⟨0, hm⟩, ⟨0, hn⟩), Finset.mem_univ _⟩
  obtain ⟨v, _, hv⟩ := Finset.exists_min_image Finset.univ h h_ne
  refine ⟨v, ?_⟩
  intro u hu
  have h_ge : h v ≤ h u := hv u (Finset.mem_univ u)
  have h_diff : |h v - h u| = 1 := hh v u hu
  have h_nonpos : h v - h u ≤ 0 := by linarith
  have : h v - h u = -1 := by
    rw [abs_of_nonpos h_nonpos] at h_diff; linarith
  linarith

theorem numExtrema_ge_two {h : Cell m n → ℤ} (hh : IsHeight h)
    (hm : 1 ≤ m) (hn : 1 ≤ n) (hmn : 2 ≤ m * n) :
    2 ≤ numExtrema h := by
  obtain ⟨v_max, hv_max⟩ := exists_strictLocalMax hh hm hn
  obtain ⟨v_min, hv_min⟩ := exists_strictLocalMin hh hm hn
  have h_ne : v_max ≠ v_min :=
    fun heq => max_min_excl hmn hv_max (heq ▸ hv_min)
  unfold numExtrema
  have hsub : ({v_max, v_min} : Finset (Cell m n)) ⊆
      Finset.univ.filter (IsStrictLocalExtremum h) := by
    intro x hx
    rw [Finset.mem_insert, Finset.mem_singleton] at hx
    refine Finset.mem_filter.mpr ⟨Finset.mem_univ _, ?_⟩
    rcases hx with rfl | rfl
    · exact Or.inl hv_max
    · exact Or.inr hv_min
  calc 2 = ({v_max, v_min} : Finset (Cell m n)).card := by
        rw [Finset.card_insert_of_notMem (by simp [h_ne]), Finset.card_singleton]
    _ ≤ _ := Finset.card_le_card hsub

theorem Ed_lt_two_eq_zero {d : ℕ} (hd : d < 2)
    (hm : 1 ≤ m) (hn : 1 ≤ n) (hmn : 2 ≤ m * n) : Ed d m n = 0 := by
  suffices h_empty :
      { h : Cell m n → ℤ | IsCanonicalHeight h ∧ numExtrema h = d } = ∅ by
    rw [Ed_def, h_empty, Set.ncard_empty]
  ext h
  simp only [Set.mem_setOf_eq, Set.mem_empty_iff_false, iff_false]
  rintro ⟨hc, hd_eq⟩
  have := numExtrema_ge_two hc.1 hm hn hmn
  omega

theorem Ed_thm_poly_of_perAxis (d D lo : ℕ) (hlo : 1 ≤ lo)
    (hrow : ∀ a, lo ≤ a → ∃ p : Polynomial ℚ, p.natDegree ≤ D ∧
      ∀ b, lo ≤ b → (Ed d a b : ℚ) = p.eval (b : ℚ))
    (hcol : ∀ b, lo ≤ b → ∃ q : Polynomial ℚ, q.natDegree ≤ D ∧
      ∀ a, lo ≤ a → (Ed d a b : ℚ) = q.eval (a : ℚ)) :
    ∃ (g L : Fin (D + 1) → Polynomial ℚ),
      (∀ i, (g i).natDegree ≤ D) ∧ (∀ i, (L i).natDegree ≤ D) ∧
      (∀ a b, lo ≤ a → lo ≤ b →
        (Ed d a b : ℚ) = ∑ i, (g i).eval (b : ℚ) * (L i).eval (a : ℚ)) ∧
      (∀ a b : ℚ,
        (∑ i, (g i).eval b * (L i).eval a) =
          (∑ i, (g i).eval a * (L i).eval b)) := by
  apply thm_poly_polynomial_symmetry (fun m n => (Ed d m n : ℚ)) hrow hcol
  intro a b ha hb
  have h_ed_symm : Ed d a b = Ed d b a :=
    Ed_symm d (le_trans hlo ha) (le_trans hlo hb)
  exact_mod_cast h_ed_symm

theorem Ed_hrow_of_lt_two {d : ℕ} (hd : d < 2) {lo : ℕ} (hlo : 2 ≤ lo) :
    ∀ a, lo ≤ a → ∃ p : Polynomial ℚ, p.natDegree ≤ 0 ∧
      ∀ b, lo ≤ b → (Ed d a b : ℚ) = p.eval (b : ℚ) := by
  intro a ha
  refine ⟨0, by simp, ?_⟩
  intro b hb
  have h_ab_ge_2 : 2 ≤ a * b := by nlinarith
  have h_zero : Ed d a b = 0 :=
    Ed_lt_two_eq_zero hd (by omega) (by omega) h_ab_ge_2
  simp [h_zero]

theorem Ed_hcol_of_lt_two {d : ℕ} (hd : d < 2) {lo : ℕ} (hlo : 2 ≤ lo) :
    ∀ b, lo ≤ b → ∃ q : Polynomial ℚ, q.natDegree ≤ 0 ∧
      ∀ a, lo ≤ a → (Ed d a b : ℚ) = q.eval (a : ℚ) := by
  intro b hb
  refine ⟨0, by simp, ?_⟩
  intro a ha
  have h_ab_ge_2 : 2 ≤ a * b := by nlinarith
  have h_zero : Ed d a b = 0 :=
    Ed_lt_two_eq_zero hd (by omega) (by omega) h_ab_ge_2
  simp [h_zero]

theorem Ed_lt_two_thm_poly {d : ℕ} (hd : d < 2) {lo : ℕ} (hlo : 2 ≤ lo) :
    ∃ (g L : Fin 1 → Polynomial ℚ),
      (∀ i, (g i).natDegree ≤ 0) ∧ (∀ i, (L i).natDegree ≤ 0) ∧
      (∀ a b, lo ≤ a → lo ≤ b →
        (Ed d a b : ℚ) = ∑ i, (g i).eval (b : ℚ) * (L i).eval (a : ℚ)) ∧
      (∀ a b : ℚ,
        (∑ i, (g i).eval b * (L i).eval a) =
          (∑ i, (g i).eval a * (L i).eval b)) :=
  Ed_thm_poly_of_perAxis d 0 lo (by omega)
    (Ed_hrow_of_lt_two hd hlo) (Ed_hcol_of_lt_two hd hlo)

theorem Ed_gt_mn_eq_zero {d m n : ℕ} (hd : m * n < d) :
    Ed d m n = 0 := by
  suffices h_empty :
      { h : Cell m n → ℤ | IsCanonicalHeight h ∧ numExtrema h = d } = ∅ by
    rw [Ed_def, h_empty, Set.ncard_empty]
  ext h
  simp only [Set.mem_setOf_eq, Set.mem_empty_iff_false, iff_false]
  rintro ⟨_, hd_eq⟩
  have := numExtrema_le_mn h
  omega

end OrigamiCone.Sequel
