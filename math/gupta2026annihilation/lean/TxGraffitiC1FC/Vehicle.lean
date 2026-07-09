import Mathlib
import TxGraffitiC1FC.Invariants

/-!
# Item 2 — the annihilation-number Caro–Wei vehicle over the real invariants

Ports the lab's abstract algebraic core (`annihilation_caroWei`,
`domains/math/src/txgraffiti-c1/lean/CaroWeiAnnihilation.lean`) from a degree
vector `Fin n → ℕ` to an arbitrary finite vertex type `V`, then proves the
graph-level vehicle `a ≤ (Δ+1)/2 · W` (Conjecture.lean's `annih_vehicle`) over
the real `annihilationNumber`. Three lemmas, all proved and kernel-checked:

* `caroWei_abstract` (A): the ported algebraic core over `V`.
* `exists_annih_head` (B): the annihilation number is the cardinality of a vertex
  head-set whose degree sum is `≤ #edges` (attained-sup + multiset realizability).
* `head_sum_le_maxDegree` (C): such a head-set has `Σ_{v∈H} deg v ≤ Δ·(n−|H|)`.

`vehicle_bound` wires A+B+C over the real `annihilationNumber`; it is
`Conjecture.annih_vehicle` modulo the definitional unfolding of `caroWeiWeight`
(wired at cleanup, item 7). Axioms: `[propext, Classical.choice, Quot.sound]`.
-/

namespace TxGraffitiC1FC

open Finset

/-- Pointwise Caro–Wei estimate (ported verbatim from the lab core):
`2/(Δ+1) ≤ 1/(k+1) + k/(Δ(Δ+1))` for `1 ≤ Δ`, `k ≤ Δ`. -/
theorem pointwise (Δ k : ℕ) (hΔ : 1 ≤ Δ) (hk : k ≤ Δ) :
    (2 : ℚ) / ((Δ : ℚ) + 1)
      ≤ 1 / ((k : ℚ) + 1) + (k : ℚ) / ((Δ : ℚ) * ((Δ : ℚ) + 1)) := by
  have hk1_pos : (0 : ℚ) < (k : ℚ) + 1 := by positivity
  have hΔ_pos  : (0 : ℚ) < (Δ : ℚ) := by exact_mod_cast hΔ
  have hΔ1_pos : (0 : ℚ) < (Δ : ℚ) + 1 := by linarith
  have h_core : (0 : ℚ) ≤ ((Δ : ℚ) - k) * ((Δ : ℚ) - k - 1) := by
    rcases eq_or_lt_of_le hk with heq | hlt
    · subst heq
      have : ((k : ℚ) - k) * ((k : ℚ) - k - 1) = 0 := by ring
      linarith
    · have h1 : ((k : ℚ)) + 1 ≤ ((Δ : ℚ)) := by exact_mod_cast hlt
      have h2 : (0 : ℚ) ≤ (Δ : ℚ) - k - 1 := by linarith
      have h3 : (0 : ℚ) ≤ (Δ : ℚ) - k := by linarith
      exact mul_nonneg h3 h2
  have h_id :
      1 / ((k : ℚ) + 1) + (k : ℚ) / ((Δ : ℚ) * ((Δ : ℚ) + 1))
        - 2 / ((Δ : ℚ) + 1)
      = ((Δ : ℚ) - k) * ((Δ : ℚ) - k - 1)
          / ((Δ : ℚ) * ((Δ : ℚ) + 1) * ((k : ℚ) + 1)) := by
    field_simp; ring
  have h_denom :
      (0 : ℚ) < (Δ : ℚ) * ((Δ : ℚ) + 1) * ((k : ℚ) + 1) := by positivity
  have h_diff_nn :
      (0 : ℚ) ≤ 1 / ((k : ℚ) + 1) + (k : ℚ) / ((Δ : ℚ) * ((Δ : ℚ) + 1))
                - 2 / ((Δ : ℚ) + 1) := by
    rw [h_id]; exact div_nonneg h_core h_denom.le
  linarith

/-- **Annihilation Caro–Wei bound (algebraic core), over an arbitrary finite
vertex type.** Ported from the lab `annihilation_caroWei` (`Fin n → V`). Given a
degree function `d : V → ℕ` with every value `≤ Δ` (`1 ≤ Δ`) and a head set `H`
with `Σ_{i∈H} d i ≤ Δ·(|V| − |H|)`, one has `|H| ≤ (Δ+1)/2 · Σ_i 1/(d i + 1)`. -/
theorem caroWei_abstract {V : Type*} [Fintype V] [DecidableEq V]
    (Δ : ℕ) (d : V → ℕ) (hΔ : 1 ≤ Δ) (hd_le : ∀ i, d i ≤ Δ)
    (H : Finset V)
    (h_head_sum : (∑ i ∈ H, d i) ≤ Δ * (Fintype.card V - H.card)) :
    (H.card : ℚ) ≤ ((Δ : ℚ) + 1) / 2 * ∑ i : V, 1 / ((d i : ℚ) + 1) := by
  set a := H.card with ha_def
  have hΔ_pos  : (0 : ℚ) < (Δ : ℚ) := by exact_mod_cast hΔ
  have hΔ1_pos : (0 : ℚ) < (Δ : ℚ) + 1 := by linarith
  have hHsub   : H ⊆ Finset.univ := H.subset_univ
  have h_tcard : (Finset.univ \ H).card = Fintype.card V - a := by
    rw [← Finset.compl_eq_univ_sdiff, Finset.card_compl, ha_def]
  -- Tail bound: each tail term ≥ 1/(Δ+1).
  have h_tail :
      ((Fintype.card V - a : ℕ) : ℚ) / ((Δ : ℚ) + 1)
        ≤ ∑ i ∈ Finset.univ \ H, 1 / ((d i : ℚ) + 1) := by
    have h_ptw : ∀ i ∈ Finset.univ \ H,
        (1 : ℚ) / ((Δ : ℚ) + 1) ≤ 1 / ((d i : ℚ) + 1) := by
      intro i _
      have hdi_le : (d i : ℚ) ≤ (Δ : ℚ) := by exact_mod_cast hd_le i
      have hdi1_pos : (0 : ℚ) < (d i : ℚ) + 1 := by positivity
      exact one_div_le_one_div_of_le hdi1_pos (by linarith)
    have h_sum := Finset.sum_le_sum h_ptw
    have h_const :
        (∑ _ ∈ Finset.univ \ H, (1 : ℚ) / ((Δ : ℚ) + 1))
          = ((Fintype.card V - a : ℕ) : ℚ) / ((Δ : ℚ) + 1) := by
      rw [Finset.sum_const, h_tcard, nsmul_eq_mul]; ring
    linarith
  -- Head bound via the pointwise estimate.
  have h_head :
      (a : ℚ) * (2 / ((Δ : ℚ) + 1))
          - (∑ i ∈ H, (d i : ℚ)) / ((Δ : ℚ) * ((Δ : ℚ) + 1))
        ≤ ∑ i ∈ H, 1 / ((d i : ℚ) + 1) := by
    have h_ptw : ∀ i ∈ H,
        2 / ((Δ : ℚ) + 1) - (d i : ℚ) / ((Δ : ℚ) * ((Δ : ℚ) + 1))
          ≤ 1 / ((d i : ℚ) + 1) := by
      intro i _
      have := pointwise Δ (d i) hΔ (hd_le i)
      linarith
    have h_sum := Finset.sum_le_sum h_ptw
    have h_eq :
        (∑ i ∈ H, (2 / ((Δ : ℚ) + 1) - (d i : ℚ) / ((Δ : ℚ) * ((Δ : ℚ) + 1))))
          = (a : ℚ) * (2 / ((Δ : ℚ) + 1))
            - (∑ i ∈ H, (d i : ℚ)) / ((Δ : ℚ) * ((Δ : ℚ) + 1)) := by
      rw [Finset.sum_sub_distrib, Finset.sum_const, nsmul_eq_mul,
          ← Finset.sum_div, ha_def]
    linarith
  -- Cancellation: (Σ_H d)/(Δ(Δ+1)) ≤ (n−a)/(Δ+1).
  have h_sumQ :
      (∑ i ∈ H, (d i : ℚ)) ≤ (Δ : ℚ) * ((Fintype.card V - a : ℕ) : ℚ) := by
    have : ((∑ i ∈ H, d i : ℕ) : ℚ)
            ≤ ((Δ : ℕ) : ℚ) * (((Fintype.card V - a : ℕ)) : ℚ) := by
      exact_mod_cast h_head_sum
    simpa using this
  have h_mid :
      (∑ i ∈ H, (d i : ℚ)) / ((Δ : ℚ) * ((Δ : ℚ) + 1))
        ≤ ((Fintype.card V - a : ℕ) : ℚ) / ((Δ : ℚ) + 1) := by
    have h_simp :
        (Δ : ℚ) * ((Fintype.card V - a : ℕ) : ℚ) / ((Δ : ℚ) * ((Δ : ℚ) + 1))
          = ((Fintype.card V - a : ℕ) : ℚ) / ((Δ : ℚ) + 1) := by
      rw [mul_div_mul_left _ _ hΔ_pos.ne']
    calc (∑ i ∈ H, (d i : ℚ)) / ((Δ : ℚ) * ((Δ : ℚ) + 1))
        ≤ (Δ : ℚ) * ((Fintype.card V - a : ℕ) : ℚ) / ((Δ : ℚ) * ((Δ : ℚ) + 1)) := by
          gcongr
      _ = ((Fintype.card V - a : ℕ) : ℚ) / ((Δ : ℚ) + 1) := h_simp
  -- Total split: W = head + tail.
  have h_split :
      (∑ i : V, 1 / ((d i : ℚ) + 1))
        = (∑ i ∈ H, 1 / ((d i : ℚ) + 1))
          + (∑ i ∈ Finset.univ \ H, 1 / ((d i : ℚ) + 1)) := by
    have h := Finset.sum_sdiff hHsub (f := fun i => (1 : ℚ) / ((d i : ℚ) + 1))
    linarith
  have W_lb :
      2 * (a : ℚ) / ((Δ : ℚ) + 1) ≤ ∑ i : V, 1 / ((d i : ℚ) + 1) := by
    rw [h_split]
    have h_rw :
        (a : ℚ) * (2 / ((Δ : ℚ) + 1)) = 2 * (a : ℚ) / ((Δ : ℚ) + 1) := by ring
    linarith
  have h_mul :
      2 * (a : ℚ) ≤ ((Δ : ℚ) + 1) * ∑ i : V, 1 / ((d i : ℚ) + 1) := by
    have := (div_le_iff₀ hΔ1_pos).mp W_lb
    linarith
  linarith

end TxGraffitiC1FC

namespace Multiset

/-- A sub-multiset of `A.map f` is the `f`-image of some sub-multiset of `A`
(no injectivity needed). Proof by multiset induction on `A`. -/
theorem exists_le_of_le_map {α β : Type*} [DecidableEq β] {f : α → β} :
    ∀ {A : Multiset α} {S : Multiset β}, S ≤ A.map f → ∃ T ≤ A, T.map f = S := by
  intro A
  induction A using Multiset.induction with
  | empty =>
      intro S h
      rw [Multiset.map_zero, Multiset.le_zero] at h
      exact ⟨0, le_rfl, by rw [Multiset.map_zero, h]⟩
  | cons a A' ih =>
      intro S h
      rw [Multiset.map_cons] at h
      by_cases hmem : f a ∈ S
      · have h' : (S.erase (f a)) ≤ A'.map f := Multiset.erase_le_iff_le_cons.mpr h
        obtain ⟨T', hT'le, hT'map⟩ := ih h'
        exact ⟨a ::ₘ T', Multiset.cons_le_cons a hT'le, by
          rw [Multiset.map_cons, hT'map, Multiset.cons_erase hmem]⟩
      · have hSle2 : S ≤ A'.map f := by
          rw [Multiset.le_iff_count]
          intro x
          rcases eq_or_ne x (f a) with rfl | hne
          · rw [Multiset.count_eq_zero.mpr hmem]; exact Nat.zero_le _
          · have hx : Multiset.count x S ≤ Multiset.count x (f a ::ₘ A'.map f) :=
              (Multiset.le_iff_count.mp h) x
            rwa [Multiset.count_cons_of_ne hne] at hx
        obtain ⟨T', hT'le, hT'map⟩ := ih hSle2
        exact ⟨T', le_trans hT'le (Multiset.le_cons_self A' a), hT'map⟩

end Multiset

namespace SimpleGraph

variable {V : Type*} [Fintype V] [DecidableEq V] (G : SimpleGraph V)
  [DecidableRel G.Adj]

/-- **(B) Greedy/prefix (item 2, the crux).** The annihilation number is the
cardinality of some vertex head-set whose degree sum is at most the edge count.
The filtered finset of feasible sub-multisets is nonempty (contains `0`), so its
`sup` over `Multiset.card` is attained at some `S`; `S ≤ degreeMultiset =
univ.val.map degree` is realized (`Multiset.exists_le_of_le_map`) as a nodup
sub-multiset `T` of the vertices, and `H := T.toFinset` has `|H| = a` and
degree-sum `= S.sum ≤ #edges`. -/
theorem exists_annih_head :
    ∃ H : Finset V, H.card = G.annihilationNumber ∧
      (∑ v ∈ H, G.degree v) ≤ G.edgeFinset.card := by
  classical
  -- the feasible sub-multisets form a nonempty finset (0 is feasible)
  have hFne :
      ((Finset.Iic (degreeMultiset G)).filter
          (fun s => Multiset.sum s ≤ G.edgeFinset.card)).Nonempty := by
    refine ⟨0, ?_⟩
    rw [Finset.mem_filter, Finset.mem_Iic]
    exact ⟨Multiset.zero_le _, by simp⟩
  -- the annihilation number (= that sup) is attained at some feasible S
  obtain ⟨S, hSmem, hSsup⟩ := Finset.exists_mem_eq_sup _ hFne Multiset.card
  rw [Finset.mem_filter, Finset.mem_Iic] at hSmem
  obtain ⟨hSle, hSsum⟩ := hSmem
  have hann : G.annihilationNumber
      = ((Finset.Iic (degreeMultiset G)).filter
          (fun s => Multiset.sum s ≤ G.edgeFinset.card)).sup Multiset.card := rfl
  -- realize S ≤ degreeMultiset = univ.val.map degree as a nodup vertex multiset T
  have hSle' : S ≤ Finset.univ.val.map (fun v => G.degree v) := hSle
  obtain ⟨T, hTle, hTmap⟩ := Multiset.exists_le_of_le_map hSle'
  have hTnd : T.Nodup := Multiset.nodup_of_le hTle Finset.univ.nodup
  have hTval : T.toFinset.val = T := by
    rw [Multiset.toFinset_val, Multiset.dedup_eq_self.mpr hTnd]
  refine ⟨T.toFinset, ?_, ?_⟩
  · -- |T.toFinset| = |T| = |S| = annihilationNumber
    rw [Multiset.toFinset_card_of_nodup hTnd, hann, hSsup, ← hTmap, Multiset.card_map]
  · -- ∑_{v∈H} deg v = (T.map deg).sum = S.sum ≤ #edges
    have hdef : (∑ v ∈ T.toFinset, G.degree v)
        = (T.toFinset.val.map (fun v => G.degree v)).sum := rfl
    rw [hdef, hTval, hTmap]
    exact hSsum

/-- **(C) Head-sum bound (item 2, tractable).** A vertex head-set whose degree
sum is `≤ #edges` has degree sum `≤ Δ·(n − |H|)`. Chain
`Σ_H d ≤ m ≤ Σ_T d ≤ Δ(n−|H|)` (T = complement) via the handshake `Σ_v d = 2m`
and `d v ≤ Δ`. -/
theorem head_sum_le_maxDegree (H : Finset V)
    (h_le_edges : (∑ v ∈ H, G.degree v) ≤ G.edgeFinset.card) :
    (∑ v ∈ H, G.degree v) ≤ G.maxDegree * (Fintype.card V - H.card) := by
  classical
  have hsub : H ⊆ Finset.univ := H.subset_univ
  have hTcard : (Finset.univ \ H).card = Fintype.card V - H.card := by
    rw [← Finset.compl_eq_univ_sdiff, Finset.card_compl]
  -- handshake split: Σ_H d + Σ_T d = 2m
  have hsplit :
      (∑ v ∈ H, G.degree v) + (∑ v ∈ Finset.univ \ H, G.degree v)
        = 2 * G.edgeFinset.card := by
    have key := Finset.sum_sdiff hsub (f := fun v => G.degree v)
    rw [G.sum_degrees_eq_twice_card_edges] at key
    omega
  -- tail bound: Σ_T d ≤ Δ·|T| = Δ·(n − |H|)
  have htail :
      (∑ v ∈ Finset.univ \ H, G.degree v)
        ≤ G.maxDegree * (Fintype.card V - H.card) := by
    calc (∑ v ∈ Finset.univ \ H, G.degree v)
        ≤ ∑ _v ∈ Finset.univ \ H, G.maxDegree :=
          Finset.sum_le_sum (fun v _ => G.degree_le_maxDegree v)
      _ = (Finset.univ \ H).card • G.maxDegree := by rw [Finset.sum_const]
      _ = (Fintype.card V - H.card) * G.maxDegree := by rw [smul_eq_mul, hTcard]
      _ = G.maxDegree * (Fintype.card V - H.card) := Nat.mul_comm _ _
  -- Σ_H d ≤ m and m ≤ Σ_T d ≤ Δ(n−|H|), so Σ_H d ≤ Δ(n−|H|).
  omega

/-- **Vehicle bound over the real annihilation number** (item 2 assembly): A+B+C
compose to `a ≤ (Δ+1)/2 · Σ_v 1/(deg v + 1)`. This is `Conjecture.annih_vehicle`
modulo the definitional unfolding of `caroWeiWeight` (wired at cleanup, item 7). -/
theorem vehicle_bound (hΔ : 1 ≤ G.maxDegree) :
    (G.annihilationNumber : ℚ)
      ≤ ((G.maxDegree : ℚ) + 1) / 2 * ∑ v : V, 1 / ((G.degree v : ℚ) + 1) := by
  obtain ⟨H, hHcard, hHsum⟩ := G.exists_annih_head
  have hd_le : ∀ v, G.degree v ≤ G.maxDegree := fun v => G.degree_le_maxDegree v
  have hC := G.head_sum_le_maxDegree H hHsum
  have hA := TxGraffitiC1FC.caroWei_abstract
    G.maxDegree (fun v => G.degree v) hΔ hd_le H hC
  rw [← hHcard]
  exact hA

end SimpleGraph
