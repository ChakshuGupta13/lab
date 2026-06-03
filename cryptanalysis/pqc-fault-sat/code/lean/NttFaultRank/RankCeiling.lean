/-
Phase F9 — Rank ceiling.

Combines F1 (telescope identity), F5 (per-term rank equality), and F8
(asymmetric image disjointness) to derive:

    finrank(range(faultDiff Z F)) ≤ ∑ ℓ, spec.L ⟨ℓ.val, ℓ.isLt⟩
                                  = spec.n - spec.L ⟨spec.N - 1, _⟩

For Kyber (n = 256, L_6 = 2): bound is 254.

The matching LOWER bound `finrank ≥ ∑ L_ℓ` (required to close Theorem 2 (⊆)
via rank-nullity) is NOT proved here — it requires a top-down LI construction
(paper §"Equivalence of three statements"), deferred to a separate session or
to F10 via a different route. See `/memories/repo/pqc-fault-sat-rank-bound.md`
"F9 BLOCKER" for analysis of three rejected approaches.

What this file delivers (upper bound + arithmetic):
- `range_faultDiff_le_iSup`: range D_K ⊆ ⨆ range(teleTerm_ℓ).
- `finrank_iSup_telescope_eq_sum_L`: finrank(⨆) = ∑ L_ℓ via F8 disjointness
  + F5 per-term rank + Mathlib's `finrank_sup_add_finrank_inf_eq`.
- `faultDiff_rank_le_sum_L`: sandwich. The headline upper bound.
- `sum_L_eq_n_sub_L_last`: arithmetic from F8's `sum_Li_plus_2L_eq_n`.
- `faultDiff_rank_le`: `finrank ≤ n - L_{N-1}`.
- `faultDiff_kyber_rank_le`: `finrank ≤ 254`.
-/

import NttFaultRank.PiInjective
import Mathlib.LinearAlgebra.FiniteDimensional.Lemmas

namespace NttFaultRank

variable {K : Type*} [Field K]

namespace LayerSpec
variable (spec : LayerSpec)

/-! ### Step 1a — Upper bound by inclusion -/

/-- `range(faultDiff Z F) ⊆ ⨆ ℓ, range(teleTerm Z (faultedTwiddles Z F) ℓ.val)`. -/
lemma range_faultDiff_le_iSup
    (Z : spec.Twiddles K) (F : spec.FaultSet) :
    LinearMap.range (spec.faultDiff Z F)
      ≤ ⨆ ℓ : Fin spec.N,
          LinearMap.range (spec.teleTerm Z (spec.faultedTwiddles Z F) ℓ.val) := by
  -- faultDiff = ∑ k ∈ range N, teleTerm Z Z' k by diff_ntt_telescope.
  -- Convert the Finset.range N sum to a Fin spec.N sum (or use the range form).
  -- We use the range form directly.
  intro v hv
  obtain ⟨w, hw⟩ := hv
  -- faultDiff Z F = ntt Z - nttFault Z F, equals the telescope sum.
  have htele : spec.faultDiff Z F
              = ∑ k ∈ Finset.range spec.N,
                  spec.teleTerm Z (spec.faultedTwiddles Z F) k := by
    show spec.ntt Z - spec.nttFault Z F = _
    exact spec.diff_ntt_telescope Z F
  rw [htele] at hw
  -- hw : (∑ k ∈ range N, teleTerm Z Z' k) w = v.
  rw [← hw, LinearMap.coe_sum, Finset.sum_apply]
  refine Submodule.sum_mem _ (fun k hk_mem => ?_)
  -- For each k ∈ range N, teleTerm Z Z' k w ∈ range(teleTerm) ⊆ ⨆ ...
  have hk_lt : k < spec.N := Finset.mem_range.mp hk_mem
  exact Submodule.mem_iSup_of_mem ⟨k, hk_lt⟩ ⟨w, rfl⟩

/-! ### Step 1b — finrank of the iSup via top-down peeling -/

/-- The partial supremum of `range(teleTerm b)` over indices `b.val < n`. -/
private def partialSup (Z : spec.Twiddles K) (F : spec.FaultSet) (n : ℕ) :
    Submodule K (Fin spec.n → K) :=
  ⨆ b : Fin spec.N, ⨆ _ : b.val < n,
    LinearMap.range (spec.teleTerm Z (spec.faultedTwiddles Z F) b.val)

@[simp] lemma partialSup_zero (Z : spec.Twiddles K) (F : spec.FaultSet) :
    spec.partialSup Z F 0 = ⊥ := by
  unfold partialSup
  refine le_antisymm ?_ bot_le
  refine iSup_le (fun b => ?_)
  refine iSup_le (fun hb => ?_)
  exact absurd hb (Nat.not_lt_zero _)

/-- Splitting lemma: `partialSup (n+1) = partialSup n ⊔ range(teleTerm ⟨n, hn⟩)`
    when `n < spec.N`. -/
private lemma partialSup_succ (Z : spec.Twiddles K) (F : spec.FaultSet)
    {n : ℕ} (hn : n < spec.N) :
    spec.partialSup Z F (n + 1)
      = spec.partialSup Z F n ⊔
          LinearMap.range (spec.teleTerm Z (spec.faultedTwiddles Z F) n) := by
  unfold partialSup
  refine le_antisymm ?_ ?_
  · -- ⨆ b, ⨆ _ : b.val < n+1, ... ≤ (⨆ b, ⨆ _ : b.val < n, ...) ⊔ range(teleTerm n)
    refine iSup_le (fun b => iSup_le (fun hb_lt => ?_))
    by_cases hbn : b.val < n
    · -- Falls in partialSup n.
      refine le_sup_of_le_left ?_
      exact le_iSup_of_le b (le_iSup_of_le hbn le_rfl)
    · -- b.val = n.
      have hb_eq : b.val = n := by omega
      refine le_sup_of_le_right ?_
      -- range(teleTerm b.val) = range(teleTerm n) because b.val = n.
      rw [hb_eq]
  · -- (⨆ b, ⨆ _ : b.val < n, ...) ⊔ range(teleTerm n) ≤ ⨆ b, ⨆ _ : b.val < n+1, ...
    refine sup_le ?_ ?_
    · -- partialSup n ≤ partialSup (n+1).
      refine iSup_le (fun b => iSup_le (fun hb => ?_))
      exact le_iSup_of_le b (le_iSup_of_le (by omega) le_rfl)
    · -- range(teleTerm n) ≤ ⨆ b, ⨆ _ : b.val < n+1, ... at b = ⟨n, hn⟩.
      exact le_iSup_of_le ⟨n, hn⟩ (le_iSup_of_le (by simp) le_rfl)

/-- `finrank(partialSup n) = ∑ b : Fin n, L_b` by induction on n. Exposed for
    future use (e.g. F10 codimension arguments). -/
lemma finrank_partialSup_eq_sum [KyberLike spec]
    {Z : spec.Twiddles K} (hZ : ∀ ℓ g, Z ℓ g ≠ 0) (h2 : (2 : K) ≠ 0)
    {F : spec.FaultSet} (hF : spec.OneFaultPerLayer F) :
    ∀ (n : ℕ) (hn : n ≤ spec.N),
      Module.finrank K (spec.partialSup Z F n)
        = ∑ b : Fin n, spec.L ⟨b.val, lt_of_lt_of_le b.isLt hn⟩
  | 0, _ => by
      rw [partialSup_zero]
      simp
  | n + 1, hn => by
      have hn' : n < spec.N := hn
      have hn_le : n ≤ spec.N := Nat.le_of_succ_le hn
      have ih := finrank_partialSup_eq_sum hZ h2 hF n hn_le
      rw [spec.partialSup_succ Z F hn']
      have h_disj : Disjoint (spec.partialSup Z F n)
                      (LinearMap.range
                        (spec.teleTerm Z (spec.faultedTwiddles Z F) n)) := by
        have hF8 := spec.teleTerm_image_disjoint_lower hZ h2 hF ⟨n, hn'⟩
        exact Disjoint.symm hF8
      have h_sum :
          Module.finrank K
              (↥(spec.partialSup Z F n ⊔
                LinearMap.range
                  (spec.teleTerm Z (spec.faultedTwiddles Z F) n)))
            = Module.finrank K ↥(spec.partialSup Z F n)
              + Module.finrank K
                  ↥(LinearMap.range
                    (spec.teleTerm Z (spec.faultedTwiddles Z F) n)) := by
        have hsi := Submodule.finrank_sup_add_finrank_inf_eq
                  (spec.partialSup Z F n)
                  (LinearMap.range (spec.teleTerm Z (spec.faultedTwiddles Z F) n))
        rw [disjoint_iff.mp h_disj, finrank_bot, Nat.add_zero] at hsi
        exact hsi
      rw [h_sum, ih]
      rw [spec.teleTerm_rank_eq hZ h2 hF n hn']
      -- ∑ b : Fin n, L_b + L_n = ∑ b : Fin (n+1), L_b via Fin.sum_univ_castSucc.
      rw [Fin.sum_univ_castSucc
            (f := fun b : Fin (n+1) =>
              spec.L ⟨b.val, lt_of_lt_of_le (by have := b.isLt; omega) hn⟩)]
      -- Last term: L ⟨Fin.last n,_⟩ = L ⟨n, hn'⟩.
      rfl

/-- **Step 1b headline.** `finrank(⨆ ℓ, range(teleTerm ℓ)) = ∑ ℓ, L_ℓ`. -/
lemma finrank_iSup_telescope_eq_sum_L [KyberLike spec]
    {Z : spec.Twiddles K} (hZ : ∀ ℓ g, Z ℓ g ≠ 0) (h2 : (2 : K) ≠ 0)
    {F : spec.FaultSet} (hF : spec.OneFaultPerLayer F) :
    Module.finrank K
        ↑(⨆ ℓ : Fin spec.N,
          LinearMap.range (spec.teleTerm Z (spec.faultedTwiddles Z F) ℓ.val))
      = ∑ ℓ : Fin spec.N, spec.L ⟨ℓ.val, ℓ.isLt⟩ := by
  have heq : (⨆ ℓ : Fin spec.N,
              LinearMap.range (spec.teleTerm Z (spec.faultedTwiddles Z F) ℓ.val))
            = spec.partialSup Z F spec.N := by
    unfold partialSup
    refine le_antisymm ?_ ?_
    · refine iSup_le (fun ℓ => ?_)
      exact le_iSup_of_le ℓ (le_iSup_of_le ℓ.isLt le_rfl)
    · refine iSup_le (fun b => iSup_le (fun _ => ?_))
      exact le_iSup_of_le b le_rfl
  rw [heq, spec.finrank_partialSup_eq_sum hZ h2 hF spec.N le_rfl]

/-! ### Step 1c — Upper bound -/

/-- **Step 1 headline.** Upper bound on `finrank(range(faultDiff))`. -/
theorem faultDiff_rank_le_sum_L [KyberLike spec]
    {Z : spec.Twiddles K} (hZ : ∀ ℓ g, Z ℓ g ≠ 0) (h2 : (2 : K) ≠ 0)
    {F : spec.FaultSet} (hF : spec.OneFaultPerLayer F) :
    Module.finrank K ↑(LinearMap.range (spec.faultDiff Z F))
      ≤ ∑ ℓ : Fin spec.N, spec.L ⟨ℓ.val, ℓ.isLt⟩ := by
  rw [← spec.finrank_iSup_telescope_eq_sum_L hZ h2 hF]
  exact Submodule.finrank_mono (spec.range_faultDiff_le_iSup Z F)

/-! ### Step 3 — Arithmetic `∑ L_ℓ = n - L_{N-1}` -/

/-- The sum of all `L_ℓ` equals `n - L_{N-1}` (geometric series via `sum_Li_plus_2L_eq_n`). -/
lemma sum_L_eq_n_sub_L_last [KyberLike spec] (hN : 0 < spec.N) :
    (∑ ℓ : Fin spec.N, spec.L ⟨ℓ.val, ℓ.isLt⟩)
      = spec.n - spec.L ⟨spec.N - 1, Nat.sub_lt hN Nat.one_pos⟩ := by
  -- Split: ∑ ℓ : Fin N, L_ℓ = (∑ ℓ : Fin (N-1), L_ℓ) + L_{N-1}.
  -- Use sum_Li_plus_2L_eq_n at m = N-1: (∑ ℓ : Fin (N-1), L_ℓ) + 2 L_{N-1} = n.
  -- Hence (∑ ℓ : Fin (N-1), L_ℓ) = n - 2 L_{N-1}, so (∑ ℓ : Fin N, L_ℓ) = n - 2 L_{N-1} + L_{N-1} = n - L_{N-1}.
  obtain ⟨N', hN'⟩ : ∃ N', spec.N = N' + 1 := ⟨spec.N - 1, by omega⟩
  have hN'_lt : N' < spec.N := by omega
  have hN'_eq : spec.N - 1 = N' := by omega
  -- Rewrite the sum over Fin (N'+1) = Fin N' + 1 (the last index).
  have h_split : (∑ ℓ : Fin spec.N, spec.L ⟨ℓ.val, ℓ.isLt⟩)
              = (∑ ℓ : Fin N', spec.L ⟨ℓ.val,
                  lt_of_lt_of_le ℓ.isLt (by omega : N' ≤ spec.N)⟩)
                + spec.L ⟨N', hN'_lt⟩ := by
    -- Use Fin.sum_univ_castSucc on Fin (N' + 1).
    have h_cast := Fin.sum_univ_castSucc
      (n := N')
      (f := fun (ℓ : Fin (N' + 1)) =>
        spec.L ⟨ℓ.val, by have := ℓ.isLt; omega⟩)
    -- The sum on the LHS of h_cast is over Fin (N' + 1) = Fin spec.N.
    -- Bridge via Fin.cast.
    have h_bridge : (∑ ℓ : Fin spec.N, spec.L ⟨ℓ.val, ℓ.isLt⟩)
                  = ∑ ℓ : Fin (N' + 1), spec.L ⟨ℓ.val, by have := ℓ.isLt; omega⟩ := by
      apply Finset.sum_nbij' (fun (ℓ : Fin spec.N) => Fin.mk ℓ.val (by have := ℓ.isLt; omega))
        (fun (ℓ : Fin (N' + 1)) => Fin.mk ℓ.val (by have := ℓ.isLt; omega))
      · intros; exact Finset.mem_univ _
      · intros; exact Finset.mem_univ _
      · intros; rfl
      · intros; rfl
      · intros; rfl
    rw [h_bridge, h_cast]
    rfl
  rw [h_split]
  -- Apply sum_Li_plus_2L_eq_n at m = N'.
  have h_partial := spec.sum_Li_plus_2L_eq_n N' hN'_lt
  -- h_partial : (∑ i : Fin N', L_i) + 2 L_{N'} = n.
  -- So (∑ i : Fin N', L_i) = n - 2 L_{N'}.
  -- Sub L_{N'} forward and bridge ⟨N-1, _⟩ = ⟨N', _⟩.
  have h_idx : (⟨spec.N - 1, Nat.sub_lt hN Nat.one_pos⟩ : Fin spec.N)
             = ⟨N', hN'_lt⟩ := by
    apply Fin.ext; omega
  rw [h_idx]
  have hL_pos : 0 ≤ spec.L ⟨N', hN'_lt⟩ := Nat.zero_le _
  omega

/-! ### Step 4 — Headlines (upper bound only) -/

/-- **F9 (upper bound).** `finrank(range(faultDiff)) ≤ n - L_{N-1}`. -/
theorem faultDiff_rank_le [KyberLike spec]
    {Z : spec.Twiddles K} (hZ : ∀ ℓ g, Z ℓ g ≠ 0) (h2 : (2 : K) ≠ 0)
    {F : spec.FaultSet} (hF : spec.OneFaultPerLayer F) (hN : 0 < spec.N) :
    Module.finrank K ↑(LinearMap.range (spec.faultDiff Z F))
      ≤ spec.n - spec.L ⟨spec.N - 1, Nat.sub_lt hN Nat.one_pos⟩ := by
  calc Module.finrank K (LinearMap.range (spec.faultDiff Z F))
      ≤ ∑ ℓ : Fin spec.N, spec.L ⟨ℓ.val, ℓ.isLt⟩ :=
        spec.faultDiff_rank_le_sum_L hZ h2 hF
    _ = spec.n - spec.L ⟨spec.N - 1, _⟩ := spec.sum_L_eq_n_sub_L_last hN

/-! ### Step L1+L2 (F9 SESSION 4) — Top-layer LI building block

  Partial progress toward the lower bound `finrank ≥ ∑ L_b`. This file
  establishes:

      `LinearIndependent K (fun s : Fin (L_{N-1}) =>
          faultDiff Z F (basis (freshIdx ⟨N-1,_⟩ s)))`,

  giving `finrank(range(faultDiff)) ≥ L_{N-1}`. For Kyber `L_6 = 2`, this
  only matches the kernel codimension (the 2 from `span{e_0, e_1}`).

  The full lower bound `≥ ∑ L_b` requires extending this to every layer
  with backward-substitution corrections (see memory's "F9 LOWER BOUND"
  catalog). The top-layer case is the structural foundation.

  Mechanism: `faultDiff = ∑_b teleTerm_b`. For `b < N-1`, the `nttInv ∘
  teleTerm_b` image lies in `V_b` (F7) which vanishes at `freshIdx ⟨N-1,_⟩`
  via `bitOf_freshIdx_lower`. For `b = N-1`, `pi_scalar_at_fresh` gives
  `(1/2)^{N-1} · bSide`, and F3's `runPrefix_basis_value_indicator` shows
  that bSide is the Kronecker delta in `(s_in, s_out)`. -/

/-- Lower contribution vanishes: for `b < N-1`, the `(nttInv ∘ teleTerm b)`
    image evaluated at any top-layer `freshIdx` position is zero. -/
private lemma nttInv_teleTerm_low_at_freshTop_eq_zero [KyberLike spec]
    {Z : spec.Twiddles K} (hZ : ∀ ℓ g, Z ℓ g ≠ 0) (h2 : (2 : K) ≠ 0)
    {F : spec.FaultSet} (hF : spec.OneFaultPerLayer F) (hN : 0 < spec.N)
    (b : ℕ) (hb : b < spec.N - 1)
    (w : Fin spec.n → K)
    (s_out : Fin (spec.L ⟨spec.N - 1, Nat.sub_lt hN Nat.one_pos⟩)) :
    spec.nttInv hZ h2 (spec.teleTerm Z (spec.faultedTwiddles Z F) b w)
        (spec.freshIdx (Nat.sub_lt hN Nat.one_pos) s_out) = 0 := by
  have hbN : b < spec.N := lt_trans hb (Nat.sub_lt hN Nat.one_pos)
  -- nttInv (teleTerm b w) ∈ V_⟨b, hbN⟩ (F7).
  have hmem : spec.nttInv hZ h2 (spec.teleTerm Z (spec.faultedTwiddles Z F) b w)
                ∈ spec.V (K := K) ⟨b, hbN⟩ := by
    apply spec.teleTerm_intt_image_subset_V hZ h2 hF b hbN
    exact ⟨w, rfl⟩
  -- bitOf b (freshIdx ⟨N-1,_⟩ s_out) = 0 since b < N-1.
  have hbit : spec.bitOf ⟨b, hbN⟩
                (spec.freshIdx (Nat.sub_lt hN Nat.one_pos) s_out) = 0 :=
    spec.bitOf_freshIdx_lower (Nat.sub_lt hN Nat.one_pos) hbN hb s_out
  exact spec.bit_zero_vanish_of_mem_V ⟨b, hbN⟩ _ hmem _ hbit

/-- Top contribution diagonal: for `b = N-1`, the value equals
    `(1/2)^{N-1} * δ_{s_in, s_out}`. -/
private lemma nttInv_teleTerm_top_at_freshTop_eq [KyberLike spec]
    {Z : spec.Twiddles K} (hZ : ∀ ℓ g, Z ℓ g ≠ 0) (h2 : (2 : K) ≠ 0)
    {F : spec.FaultSet} (hF : spec.OneFaultPerLayer F) (hN : 0 < spec.N)
    (s_in s_out : Fin (spec.L ⟨spec.N - 1, Nat.sub_lt hN Nat.one_pos⟩)) :
    spec.nttInv hZ h2
        (spec.teleTerm Z (spec.faultedTwiddles Z F) (spec.N - 1)
          (spec.basis (spec.freshIdx (Nat.sub_lt hN Nat.one_pos) s_in)))
        (spec.freshIdx (Nat.sub_lt hN Nat.one_pos) s_out)
      = (1 / (2 : K))^(spec.N - 1) * (if s_in = s_out then 1 else 0) := by
  -- Apply pi_scalar_at_fresh.
  rw [spec.pi_scalar_at_fresh hZ h2 hF (spec.N - 1) (Nat.sub_lt hN Nat.one_pos)
        (spec.basis (spec.freshIdx (Nat.sub_lt hN Nat.one_pos) s_in)) s_out]
  -- Now: (1/2)^{N-1} * bSide(runPrefix Z' (N-1) (basis(freshIdx _ s_in))) ⟨N-1,_⟩ g₀ s_out
  -- Compute bSide via F3.
  congr 1
  rw [spec.bSide_eq_apply]
  have hj_lt : (spec.freshIdx (Nat.sub_lt hN Nat.one_pos) s_in).val
                  < 2 * spec.L ⟨spec.N - 1, Nat.sub_lt hN Nat.one_pos⟩ := by
    rw [spec.freshIdx_val]; have := s_in.isLt; omega
  rw [spec.runPrefix_basis_value_indicator (spec.faultedTwiddles Z F)
        ⟨spec.N - 1, Nat.sub_lt hN Nat.one_pos⟩
        (spec.freshIdx (Nat.sub_lt hN Nat.one_pos) s_in) hj_lt
        (spec.N - 1) le_rfl
        (spec.bSideIdx ⟨spec.N - 1, Nat.sub_lt hN Nat.one_pos⟩
          (Finset.card_eq_one.mp (hF ⟨spec.N - 1, Nat.sub_lt hN Nat.one_pos⟩)).choose
          s_out)]
  -- Compute the modular condition.
  have hL_pos : 0 < spec.L ⟨spec.N - 1, Nat.sub_lt hN Nat.one_pos⟩ := by
    have := s_in.isLt; omega
  have h2L_pos : 0 < 2 * spec.L ⟨spec.N - 1, Nat.sub_lt hN Nat.one_pos⟩ := by omega
  have h_bs_val := spec.bSideIdx_val ⟨spec.N - 1, Nat.sub_lt hN Nat.one_pos⟩
    (Finset.card_eq_one.mp (hF ⟨spec.N - 1, Nat.sub_lt hN Nat.one_pos⟩)).choose s_out
  have h_sum_lt : spec.L ⟨spec.N - 1, Nat.sub_lt hN Nat.one_pos⟩ + s_out.val
                    < 2 * spec.L ⟨spec.N - 1, Nat.sub_lt hN Nat.one_pos⟩ := by
    have := s_out.isLt; omega
  have h_mod : (spec.bSideIdx ⟨spec.N - 1, Nat.sub_lt hN Nat.one_pos⟩
                  (Finset.card_eq_one.mp
                    (hF ⟨spec.N - 1, Nat.sub_lt hN Nat.one_pos⟩)).choose s_out).val
                  % (2 * spec.L ⟨spec.N - 1, Nat.sub_lt hN Nat.one_pos⟩)
                = spec.L ⟨spec.N - 1, Nat.sub_lt hN Nat.one_pos⟩ + s_out.val := by
    rw [h_bs_val]
    rw [show (Finset.card_eq_one.mp
              (hF ⟨spec.N - 1, Nat.sub_lt hN Nat.one_pos⟩)).choose.val
            * (2 * spec.L ⟨spec.N - 1, Nat.sub_lt hN Nat.one_pos⟩)
            + spec.L ⟨spec.N - 1, Nat.sub_lt hN Nat.one_pos⟩ + s_out.val
          = (spec.L ⟨spec.N - 1, Nat.sub_lt hN Nat.one_pos⟩ + s_out.val)
            + (Finset.card_eq_one.mp
                (hF ⟨spec.N - 1, Nat.sub_lt hN Nat.one_pos⟩)).choose.val
              * (2 * spec.L ⟨spec.N - 1, Nat.sub_lt hN Nat.one_pos⟩) from by ring]
    rw [Nat.add_mul_mod_self_right]
    exact Nat.mod_eq_of_lt h_sum_lt
  rw [h_mod, spec.freshIdx_val]
  by_cases hss : s_in = s_out
  · subst hss
    simp
  · rw [if_neg hss, if_neg]
    intro habs
    apply hss
    apply Fin.ext
    omega

/-- The diagonal value formula at top-layer fresh positions. -/
lemma nttInv_faultDiff_freshTop_diag [KyberLike spec]
    {Z : spec.Twiddles K} (hZ : ∀ ℓ g, Z ℓ g ≠ 0) (h2 : (2 : K) ≠ 0)
    {F : spec.FaultSet} (hF : spec.OneFaultPerLayer F) (hN : 0 < spec.N)
    (s_in s_out : Fin (spec.L ⟨spec.N - 1, Nat.sub_lt hN Nat.one_pos⟩)) :
    spec.nttInv hZ h2 (spec.faultDiff Z F
        (spec.basis (spec.freshIdx (Nat.sub_lt hN Nat.one_pos) s_in)))
        (spec.freshIdx (Nat.sub_lt hN Nat.one_pos) s_out)
      = (1 / (2 : K))^(spec.N - 1) * (if s_in = s_out then 1 else 0) := by
  -- Step 1: faultDiff = ∑_b teleTerm_b (F1 diff_ntt_telescope).
  have hdecomp : spec.faultDiff Z F =
      ∑ b ∈ Finset.range spec.N,
        spec.teleTerm Z (spec.faultedTwiddles Z F) b := by
    show spec.ntt Z - spec.nttFault Z F = _
    exact spec.diff_ntt_telescope Z F
  rw [hdecomp]
  rw [LinearMap.coe_sum, Finset.sum_apply, map_sum, Finset.sum_apply]
  -- Step 2: split sum at b = N-1 via Finset.sum_range_succ on N = (N-1)+1.
  -- Use Finset.range_add_one or Finset.sum_Ico_succ etc.
  -- Cleaner: rewrite Finset.range N = Finset.range (N-1) ∪ {N-1}.
  have h_range_split : Finset.range spec.N
        = insert (spec.N - 1) (Finset.range (spec.N - 1)) := by
    have : Finset.range spec.N = Finset.range ((spec.N - 1) + 1) := by
      congr 1; omega
    rw [this, Finset.range_succ]
  rw [h_range_split, Finset.sum_insert (by simp)]
  -- LHS: top_term + ∑ b ∈ Finset.range (N-1), ...
  rw [Finset.sum_eq_zero (s := Finset.range (spec.N - 1)) (f := fun b =>
        (spec.nttInv hZ h2 (spec.teleTerm Z (spec.faultedTwiddles Z F) b
          (spec.basis (spec.freshIdx (Nat.sub_lt hN Nat.one_pos) s_in))))
          (spec.freshIdx (Nat.sub_lt hN Nat.one_pos) s_out))]
  · rw [add_zero]
    exact spec.nttInv_teleTerm_top_at_freshTop_eq hZ h2 hF hN s_in s_out
  · intro b hb
    rw [Finset.mem_range] at hb
    exact spec.nttInv_teleTerm_low_at_freshTop_eq_zero hZ h2 hF hN b hb _ s_out

/-- **L1+L2 headline: top-layer LI**. The L_{N-1} basis vectors at top-layer
    fresh indices have linearly independent images under `faultDiff`. -/
lemma topLayer_faultDiff_LI [KyberLike spec]
    {Z : spec.Twiddles K} (hZ : ∀ ℓ g, Z ℓ g ≠ 0) (h2 : (2 : K) ≠ 0)
    {F : spec.FaultSet} (hF : spec.OneFaultPerLayer F) (hN : 0 < spec.N) :
    LinearIndependent K
      (fun s : Fin (spec.L ⟨spec.N - 1, Nat.sub_lt hN Nat.one_pos⟩) =>
        spec.faultDiff Z F (spec.basis
          (spec.freshIdx (Nat.sub_lt hN Nat.one_pos) s))) := by
  rw [Fintype.linearIndependent_iff]
  intro c hsum s_out
  -- Apply nttInv to hsum and evaluate at freshIdx s_out.
  have h_at : (spec.nttInv hZ h2 (∑ s_in, c s_in •
                spec.faultDiff Z F (spec.basis
                  (spec.freshIdx (Nat.sub_lt hN Nat.one_pos) s_in))))
                (spec.freshIdx (Nat.sub_lt hN Nat.one_pos) s_out)
            = (0 : Fin spec.n → K)
                (spec.freshIdx (Nat.sub_lt hN Nat.one_pos) s_out) := by
    rw [hsum, map_zero]
  rw [map_sum, Finset.sum_apply] at h_at
  have h_each : ∀ s_in,
      (spec.nttInv hZ h2 (c s_in •
          spec.faultDiff Z F (spec.basis
            (spec.freshIdx (Nat.sub_lt hN Nat.one_pos) s_in))))
        (spec.freshIdx (Nat.sub_lt hN Nat.one_pos) s_out)
      = c s_in * ((1 / (2 : K))^(spec.N - 1) * (if s_in = s_out then 1 else 0)) := by
    intro s_in
    rw [map_smul, Pi.smul_apply, smul_eq_mul]
    rw [spec.nttInv_faultDiff_freshTop_diag hZ h2 hF hN s_in s_out]
  rw [Finset.sum_congr rfl (fun s_in _ => h_each s_in)] at h_at
  -- Collapse sum at s_in = s_out.
  rw [Finset.sum_eq_single s_out] at h_at
  · -- h_at : c s_out * ((1/2)^{N-1} * 1) = 0.
    rw [if_pos rfl, mul_one] at h_at
    have hpow_ne : (1 / (2 : K))^(spec.N - 1) ≠ 0 :=
      pow_ne_zero _ (one_div_ne_zero h2)
    have h_zero : c s_out * (1 / (2 : K))^(spec.N - 1) = 0 := by
      rw [h_at]; rfl
    exact (mul_eq_zero.mp h_zero).resolve_right hpow_ne
  · intro s_in _ hne
    rw [if_neg hne, mul_zero, mul_zero]
  · intro h; exact absurd (Finset.mem_univ _) h

/-! ### Step 6 — Kernel dimension lower bound (defender's bound)

  Combining the rank upper bound with rank–nullity gives a clean
  lower bound on the kernel dimension. For Kyber this says at
  least 2 secret coefficients remain unrecoverable from `D_K(s)`,
  matching the paper's security claim from below. -/

/-- **R1.** `dim(ker(faultDiff Z F)) ≥ spec.L ⟨spec.N - 1, _⟩`. Follows from
    `faultDiff_rank_le` via rank–nullity. -/
theorem faultDiff_ker_dim_ge [KyberLike spec]
    {Z : spec.Twiddles K} (hZ : ∀ ℓ g, Z ℓ g ≠ 0) (h2 : (2 : K) ≠ 0)
    {F : spec.FaultSet} (hF : spec.OneFaultPerLayer F) (hN : 0 < spec.N) :
    spec.L ⟨spec.N - 1, Nat.sub_lt hN Nat.one_pos⟩
      ≤ Module.finrank K ↑(LinearMap.ker (spec.faultDiff Z F)) := by
  have hrn := LinearMap.finrank_range_add_finrank_ker (spec.faultDiff Z F)
  have hub := spec.faultDiff_rank_le hZ h2 hF hN
  have hdim : Module.finrank K (Fin spec.n → K) = spec.n := Module.finrank_fin_fun K
  rw [hdim] at hrn
  -- hrn : finrank range + finrank ker = spec.n
  -- hub : finrank range ≤ spec.n - L_{N-1}
  -- ⇒ finrank ker ≥ L_{N-1}
  have hL_le : spec.L ⟨spec.N - 1, Nat.sub_lt hN Nat.one_pos⟩ ≤ spec.n := by
    have := spec.two_L_le_n_F8 (spec.N - 1) (Nat.sub_lt hN Nat.one_pos)
    omega
  omega

end LayerSpec

/-- **F9 Kyber specialisation (upper bound).** `finrank ≤ 254`. -/
theorem faultDiff_kyber_rank_le
    {Z : kyberSpec.Twiddles K} (hZ : ∀ ℓ g, Z ℓ g ≠ 0) (h2 : (2 : K) ≠ 0)
    {F : kyberSpec.FaultSet} (hF : kyberSpec.OneFaultPerLayer F) :
    Module.finrank K ↑(LinearMap.range (kyberSpec.faultDiff Z F)) ≤ 254 := by
  have h := kyberSpec.faultDiff_rank_le (K := K) hZ h2 hF (by decide : 0 < kyberSpec.N)
  -- h : finrank ≤ kyberSpec.n - kyberSpec.L ⟨kyberSpec.N - 1, _⟩
  -- kyberSpec.n = 256, kyberSpec.L ⟨6, _⟩ = 2, so 256 - 2 = 254.
  have h_val : kyberSpec.n - kyberSpec.L
                  ⟨kyberSpec.N - 1, Nat.sub_lt (by decide : 0 < kyberSpec.N) Nat.one_pos⟩
              = 254 := by decide
  omega

/-- **R1 Kyber specialisation (defender's bound).** `dim(ker(faultDiff Z F)) ≥ 2`.
    Captures the security claim: at least 2 secret coefficients remain
    unrecoverable from `D_K(s)` regardless of which 7 twiddles are zeroed. -/
theorem faultDiff_kyber_ker_dim_ge_two
    {Z : kyberSpec.Twiddles K} (hZ : ∀ ℓ g, Z ℓ g ≠ 0) (h2 : (2 : K) ≠ 0)
    {F : kyberSpec.FaultSet} (hF : kyberSpec.OneFaultPerLayer F) :
    2 ≤ Module.finrank K ↑(LinearMap.ker (kyberSpec.faultDiff Z F)) := by
  have h := kyberSpec.faultDiff_ker_dim_ge (K := K) hZ h2 hF (by decide : 0 < kyberSpec.N)
  have h_val : kyberSpec.L
                  ⟨kyberSpec.N - 1, Nat.sub_lt (by decide : 0 < kyberSpec.N) Nat.one_pos⟩
              = 2 := by decide
  omega

end NttFaultRank
