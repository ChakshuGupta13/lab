import Mathlib

namespace TxGraffitiC1FC

/-- `MajorizesSorted a b`: for `a b : List ℕ` regarded as already sorted in **descending**
order, `a` majorizes `b` when they have equal total and every prefix sum of `a` dominates
that of `b`. On sorted lists this is the standard majorization order. -/
def MajorizesSorted (a b : List ℕ) : Prop :=
  a.sum = b.sum ∧ ∀ k, (b.take k).sum ≤ (a.take k).sum

namespace MajorizesSorted

@[refl]
theorem refl (a : List ℕ) : MajorizesSorted a a :=
  ⟨rfl, fun _ => le_rfl⟩

theorem trans {a b c : List ℕ}
    (hab : MajorizesSorted a b) (hbc : MajorizesSorted b c) :
    MajorizesSorted a c :=
  ⟨hab.1.trans hbc.1, fun k => (hbc.2 k).trans (hab.2 k)⟩

/-- Total-sum equality is the `k = a.length` (or any `k ≥ length`) prefix, so majorization
pins the totals: recorded here as the defining first component. -/
theorem sum_eq {a b : List ℕ} (h : MajorizesSorted a b) : a.sum = b.sum := h.1

/-- The prefix-sum domination, extracted. -/
theorem take_sum_le {a b : List ℕ} (h : MajorizesSorted a b) (k : ℕ) :
    (b.take k).sum ≤ (a.take k).sum := h.2 k

end MajorizesSorted

/-- **Shared prefix preserves prefix-sum domination.** If every prefix sum of `B` is
dominated by that of `A`, then prepending a common `pre` keeps every prefix sum of
`pre ++ B` dominated by that of `pre ++ A`. -/
theorem take_sum_le_append_left (pre A B : List ℕ)
    (h : ∀ m, (B.take m).sum ≤ (A.take m).sum) (k : ℕ) :
    ((pre ++ B).take k).sum ≤ ((pre ++ A).take k).sum := by
  induction pre generalizing k with
  | nil => simpa using h k
  | cons p ps ih =>
    cases k with
    | zero => simp
    | succ k =>
      simp only [List.cons_append, List.take_succ_cons, List.sum_cons]
      exact Nat.add_le_add_left (ih k) p

/-- **A deep `+1` raises every prefix sum by at most 1.** Increasing the coordinate `y`
(sitting after a block `mid`) to `y + 1` changes each prefix sum by `0` (before `y` is
reached) or `1` (after), never more. -/
theorem take_sum_deep_succ_le (mid post : List ℕ) (y : ℕ) (m : ℕ) :
    ((mid ++ (y + 1) :: post).take m).sum ≤ ((mid ++ y :: post).take m).sum + 1 := by
  induction mid generalizing m with
  | nil =>
    cases m with
    | zero => simp
    | succ m => simp only [List.nil_append, List.take_succ_cons, List.sum_cons]; omega
  | cons p ps ih =>
    cases m with
    | zero => simp
    | succ m =>
      simp only [List.cons_append, List.take_succ_cons, List.sum_cons]
      have := ih m; omega

theorem majorizesSorted_unit_transfer (pre mid post : List ℕ) (x y : ℕ) (hx : 1 ≤ x) :
    MajorizesSorted (pre ++ (x :: mid ++ y :: post))
                    (pre ++ ((x - 1) :: mid ++ (y + 1) :: post)) := by
  refine ⟨?_, ?_⟩
  · -- equal totals: (x-1) + (y+1) = x + y since x ≥ 1
    simp only [List.sum_append, List.sum_cons]
    omega
  · -- prefix domination via the two helpers
    intro k
    apply take_sum_le_append_left
    intro m
    cases m with
    | zero => simp
    | succ m =>
      simp only [List.cons_append, List.take_succ_cons, List.sum_cons]
      have hB := take_sum_deep_succ_le mid post y m
      omega

/-- **Prefix bound by an elementwise maximum.** For any elementwise upper bound `m` of `l`,
the length-`k` prefix sum is at most `m` plus the length-`(k-1)` prefix sum of `l` with one
copy of `m` deleted. Pure list arithmetic — no sortedness, and true whether or not `m ∈ l`. -/
theorem take_sum_le_max_add_erase (m : ℕ) :
    ∀ (l : List ℕ), (∀ x ∈ l, x ≤ m) → ∀ k,
      (l.take k).sum ≤ m + ((l.erase m).take (k - 1)).sum := by
  intro l
  induction l with
  | nil => intro _ k; simp
  | cons a l0 ih =>
    intro hb k
    have hb0 : ∀ x ∈ l0, x ≤ m := fun x hx => hb x (by simp [hx])
    have ham : a ≤ m := hb a (by simp)
    cases k with
    | zero => simp
    | succ k' =>
      rw [List.take_succ_cons, List.sum_cons, Nat.add_sub_cancel]
      by_cases ha : a = m
      · subst ha
        rw [List.erase_cons_head]
      · rw [List.erase_cons_tail (by simpa using ha)]
        cases k' with
        | zero => simpa using ham
        | succ k'' =>
          rw [List.take_succ_cons, List.sum_cons]
          have h := ih hb0 (k'' + 1)
          rw [Nat.add_sub_cancel] at h
          omega

/-- **Descending-sorted lists maximize prefix sums (rearrangement).** If `l` is sorted in
descending order and `l'` is any permutation of `l`, then every prefix sum of `l'` is
dominated by that of `l`. Equivalently `MajorizesSorted l l'` once totals match — the
sorted representative is the majorization-maximum of its permutation class. -/
theorem take_sum_le_of_sorted_perm :
    ∀ (l : List ℕ), List.Pairwise (· ≥ ·) l → ∀ (l' : List ℕ), l'.Perm l → ∀ k,
      (l'.take k).sum ≤ (l.take k).sum := by
  intro l
  induction l with
  | nil => intro _ l' hp k; simp [List.perm_nil.mp hp]
  | cons m t ih =>
    intro hsort l' hp k
    rw [List.pairwise_cons] at hsort
    obtain ⟨hmax, hts⟩ := hsort
    have hb : ∀ x ∈ l', x ≤ m := by
      intro x hx
      rcases List.mem_cons.mp (hp.mem_iff.mp hx) with h | h
      · exact h.le
      · exact hmax x h
    cases k with
    | zero => simp
    | succ k' =>
      rw [List.take_succ_cons, List.sum_cons]
      have hpe : (l'.erase m).Perm t := by
        have := hp.erase m
        rwa [List.erase_cons_head] at this
      have hdag := take_sum_le_max_add_erase m l' hb (k' + 1)
      rw [Nat.add_sub_cancel] at hdag
      have hih := ih hts (l'.erase m) hpe k'
      omega

theorem majorizesSorted_of_sorted_perm (l l' : List ℕ)
    (hl : List.Pairwise (· ≥ ·) l) (hp : l'.Perm l) :
    MajorizesSorted l l' :=
  ⟨(hp.sum_eq).symm, fun k => take_sum_le_of_sorted_perm l hl l' hp k⟩

/-- **Karamata bound (easy direction).** For every `t`, the sum of any `k`-prefix is at most
`k·t` plus the truncated total `Σᵢ (lᵢ - t)` (truncated `ℕ` subtraction = positive part).
Holds for an arbitrary list — no sortedness. The dual equality (for sorted lists) makes
`(l.take k).sum` the minimum over `t`, which is how prefix sums of two multisets get
compared. -/
theorem take_sum_le_mul_add_map_sub (l : List ℕ) (k t : ℕ) :
    (l.take k).sum ≤ k * t + (l.map (· - t)).sum := by
  induction l generalizing k with
  | nil => simp
  | cons a l0 ih =>
    cases k with
    | zero => simp
    | succ k' =>
      simp only [List.take_succ_cons, List.sum_cons, List.map_cons, Nat.succ_mul]
      have := ih k'
      omega

/-- Truncating by a lower bound `t` of every element and re-adding `t` per element recovers
the sum: `Σ(xᵢ - t) + t·|u| = Σ xᵢ` when `t ≤ xᵢ` for all `i`. -/
theorem map_sub_sum_add (t : ℕ) :
    ∀ (u : List ℕ), (∀ x ∈ u, t ≤ x) → (u.map (· - t)).sum + t * u.length = u.sum := by
  intro u
  induction u with
  | nil => simp
  | cons a u0 ih =>
    intro h
    have ha : t ≤ a := h a (by simp)
    have h0 : ∀ x ∈ u0, t ≤ x := fun x hx => h x (List.mem_cons_of_mem a hx)
    simp only [List.map_cons, List.sum_cons, List.length_cons, Nat.mul_succ]
    have := ih h0
    omega

/-- Truncating by an upper bound `t` of every element kills the sum: `Σ(xᵢ - t) = 0` when
`xᵢ ≤ t` for all `i` (truncated `ℕ` subtraction). -/
theorem map_sub_sum_eq_zero (t : ℕ) :
    ∀ (w : List ℕ), (∀ x ∈ w, x ≤ t) → (w.map (· - t)).sum = 0 := by
  intro w
  induction w with
  | nil => simp
  | cons a w0 ih =>
    intro h
    have ha : a ≤ t := h a (by simp)
    have h0 : ∀ x ∈ w0, x ≤ t := fun x hx => h x (List.mem_cons_of_mem a hx)
    simp [ih h0, Nat.sub_eq_zero_of_le ha]

/-- **Karamata equality (sorted direction).** For a descending-sorted list there is a
threshold `t` (the largest untaken element, `(l.drop k).headI`) at which the `(A)` bound is
tight: `(l.take k).sum = k·t + Σ(lᵢ - t)`. Hence `(l.take k).sum = minₜ (k·t + Σ(lᵢ-t))`,
the LP-dual form of "sum of the `k` largest". -/
theorem exists_take_sum_eq_mul_add_map_sub (l : List ℕ)
    (hl : List.Pairwise (· ≥ ·) l) (k : ℕ) :
    ∃ t, (l.take k).sum = k * t + (l.map (· - t)).sum := by
  refine ⟨(l.drop k).headI, ?_⟩
  set t := (l.drop k).headI with ht
  have hsplit : l.take k ++ l.drop k = l := List.take_append_drop k l
  have hpair : List.Pairwise (· ≥ ·) (l.take k ++ l.drop k) := by rw [hsplit]; exact hl
  rw [List.pairwise_append] at hpair
  obtain ⟨_, hdk, hcross⟩ := hpair
  have hdrop_le : ∀ y ∈ l.drop k, y ≤ t := by
    intro y hy
    match hd : l.drop k with
    | [] => rw [hd] at hy; simp at hy
    | c :: ds =>
      rw [hd] at hdk
      have hymem : y ∈ c :: ds := hd ▸ hy
      rw [ht, hd]; simp only [List.headI_cons]
      rcases List.mem_cons.mp hymem with h | h
      · exact h.le
      · exact (List.pairwise_cons.mp hdk).1 y h
  have htake_ge : ∀ x ∈ l.take k, t ≤ x := by
    intro x hx
    match hd : l.drop k with
    | [] => rw [ht, hd]; simp
    | c :: ds =>
      rw [ht, hd]; simp only [List.headI_cons]
      exact hcross x hx c (hd ▸ List.mem_cons_self)
  have h2 : ((l.drop k).map (· - t)).sum = 0 := map_sub_sum_eq_zero t _ hdrop_le
  have h1 := map_sub_sum_add t (l.take k) htake_ge
  have hkt : k * t = t * (l.take k).length := by
    by_cases hk : k ≤ l.length
    · rw [List.length_take, Nat.min_eq_left hk, Nat.mul_comm]
    · have hnil : l.drop k = [] := List.drop_eq_nil_of_le (not_le.mp hk).le
      rw [ht, hnil]; simp
  have hmap : l.map (· - t) = (l.take k).map (· - t) ++ (l.drop k).map (· - t) := by
    rw [← List.map_append, hsplit]
  rw [hmap, List.sum_append, h2, Nat.add_zero, hkt]
  omega

/-- **Truncation characterization of `MajorizesSorted` (the linchpin).** If `a` is
descending-sorted, `a` and `b` have equal totals, and for every threshold `t` the truncated
total of `b` is at most that of `a` (`Σ(bᵢ-t) ≤ Σ(aᵢ-t)`), then `a` majorizes `b`. The
truncated-total condition is **permutation-invariant**, so it can be checked on the raw
multisets without sorting `b` — this is exactly what sidesteps the dead positional→sorted
route (`MajorizesSorted a b ↛ MajorizesSorted (sort a) (sort b)`). Proof: for each `k`, pick
the tight threshold `t` for `a` (from `exists_take_sum_eq_mul_add_map_sub`); then
`(b.take k).sum ≤ k·t + Σ(bᵢ-t) ≤ k·t + Σ(aᵢ-t) = (a.take k).sum`.

Note `b` need not be sorted; the conclusion's second component then bounds `b`'s *given-order*
prefixes (weaker than multiset majorization). Consume it as genuine majorization only when `b`
is also sorted — which the downstream `b = π'` (a `mergeSort` output) is. -/
theorem majorizesSorted_of_map_sub_le (a b : List ℕ)
    (ha : List.Pairwise (· ≥ ·) a) (hsum : a.sum = b.sum)
    (hconv : ∀ t, (b.map (· - t)).sum ≤ (a.map (· - t)).sum) :
    MajorizesSorted a b := by
  refine ⟨hsum, fun k => ?_⟩
  obtain ⟨t, ht⟩ := exists_take_sum_eq_mul_add_map_sub a ha k
  have hA := take_sum_le_mul_add_map_sub b k t
  have hc := hconv t
  omega

/-- **Sum-count identity.** For a `0/1` selector `e` (same length as `rest`), the truncated
total of the decremented list plus the count of decremented positions exceeding `t` equals the
truncated total of `rest`: `Σ(restᵢ - eᵢ - t) + Σᵢ [t < restᵢ]·eᵢ = Σ(restᵢ - t)`. -/
theorem zipWith_sub_map_sub_sum_add (t : ℕ) :
    ∀ (rest e : List ℕ), e.length = rest.length → (∀ x ∈ e, x ≤ 1) →
      ((List.zipWith (· - ·) rest e).map (· - t)).sum
        + (List.zipWith (fun x ei => if t < x then ei else 0) rest e).sum
      = (rest.map (· - t)).sum := by
  intro rest
  induction rest with
  | nil =>
    intro e he _
    have : e = [] := List.eq_nil_of_length_eq_zero (by simpa using he)
    subst this; simp
  | cons a rest0 ih =>
    intro e he hb
    match e with
    | [] => simp at he
    | ei :: e0 =>
      have hlen : e0.length = rest0.length := by simpa using he
      have hei : ei ≤ 1 := hb ei (by simp)
      have hb0 : ∀ x ∈ e0, x ≤ 1 := fun x hx => hb x (by simp [hx])
      simp only [List.zipWith_cons_cons, List.map_cons, List.sum_cons]
      have hpp : a - ei - t + (if t < a then ei else 0) = a - t := by split_ifs <;> omega
      have hih := ih e0 hlen hb0
      omega

/-- A non-increasing `0/1` list is a block of ones then zeros, so its `k`-prefix sum is
`min k (total ones)`. -/
theorem take_sum_eq_min_of_pairwise_le_one :
    ∀ (l : List ℕ), List.Pairwise (· ≥ ·) l → (∀ x ∈ l, x ≤ 1) → ∀ k,
      (l.take k).sum = min k l.sum := by
  intro l
  induction l with
  | nil => intro _ _ k; simp
  | cons a l0 ih =>
    intro hsort hbin k
    rw [List.pairwise_cons] at hsort
    obtain ⟨hhead, hsort0⟩ := hsort
    have hbin0 : ∀ x ∈ l0, x ≤ 1 := fun x hx => hbin x (by simp [hx])
    have ha : a ≤ 1 := hbin a (by simp)
    cases k with
    | zero => simp
    | succ k' =>
      rw [List.take_succ_cons, List.sum_cons, List.sum_cons]
      have hih := ih hsort0 hbin0 k'
      interval_cases a
      · have hz : ∀ x ∈ l0, x = 0 := fun x hx => Nat.le_zero.mp (hhead x hx)
        have hsum0 : l0.sum = 0 := List.sum_eq_zero hz
        have htake0 : (l0.take k').sum = 0 := by rw [hih, hsum0]; simp
        omega
      · omega

/-- Count bound (2a): the weighted decrement count is at most the selector total. -/
theorem zipWith_ite_sum_le_selector (t : ℕ) :
    ∀ (rest e : List ℕ),
      (List.zipWith (fun x ei => if t < x then ei else 0) rest e).sum ≤ e.sum := by
  intro rest
  induction rest with
  | nil => intro e; simp
  | cons a rest0 ih =>
    intro e
    cases e with
    | nil => simp
    | cons ei e0 =>
      simp only [List.zipWith_cons_cons, List.sum_cons]
      have := ih e0
      have hle : (if t < a then ei else 0) ≤ ei := by split_ifs <;> omega
      omega

/-- Count bound (2b): with a `0/1` selector, the weighted decrement count is at most the
number of `rest` positions exceeding `t`. -/
theorem zipWith_ite_sum_le_map (t : ℕ) :
    ∀ (rest e : List ℕ), (∀ x ∈ e, x ≤ 1) →
      (List.zipWith (fun x ei => if t < x then ei else 0) rest e).sum
        ≤ (rest.map (fun x => if t < x then 1 else 0)).sum := by
  intro rest
  induction rest with
  | nil => intro e _; simp
  | cons a rest0 ih =>
    intro e hb
    cases e with
    | nil => simp
    | cons ei e0 =>
      have hei : ei ≤ 1 := hb ei (by simp)
      have hb0 : ∀ x ∈ e0, x ≤ 1 := fun x hx => hb x (by simp [hx])
      simp only [List.zipWith_cons_cons, List.map_cons, List.sum_cons]
      have := ih e0 hb0
      have hle : (if t < a then ei else 0) ≤ (if t < a then 1 else 0) := by split_ifs <;> omega
      omega

/-- **Threshold-count bound (2c / sub-lemma 2).** For descending-sorted `rest` and a `0/1`
selector `e` (of total `Δ = e.sum`), the weighted count of decremented positions exceeding
`t` is maximized by taking the `Δ` largest: it is at most the count over `rest.take Δ`. From
(2a) `≤ Δ`, (2b) `≤ #{restᵢ > t}`, and the binary-block prefix sum `min Δ (#{restᵢ > t})`. -/
theorem wcount_le_take (t : ℕ) (rest e : List ℕ)
    (hbin : ∀ x ∈ e, x ≤ 1) (hsort : List.Pairwise (· ≥ ·) rest) :
    (List.zipWith (fun x ei => if t < x then ei else 0) rest e).sum
      ≤ ((rest.take e.sum).map (fun x => if t < x then 1 else 0)).sum := by
  have h2a := zipWith_ite_sum_le_selector t rest e
  have h2b := zipWith_ite_sum_le_map t rest e hbin
  have hgsort : List.Pairwise (· ≥ ·) (rest.map (fun x => if t < x then 1 else 0)) := by
    rw [List.pairwise_map]
    exact hsort.imp (fun {a b} hab => by split_ifs <;> omega)
  have hgbin : ∀ x ∈ rest.map (fun x => if t < x then 1 else 0), x ≤ 1 := by
    intro x hx
    rw [List.mem_map] at hx
    obtain ⟨y, _, rfl⟩ := hx
    split_ifs <;> omega
  have hbt := take_sum_eq_min_of_pairwise_le_one _ hgsort hgbin e.sum
  rw [List.map_take, hbt]
  omega

/-- Per-position map identity: decrement-then-truncate plus the exceed-`t` indicator recovers
truncation. `Σ((xᵢ-1)-t) + Σ[t<xᵢ] = Σ(xᵢ-t)`. -/
theorem map_pred_trunc_add (t : ℕ) :
    ∀ (u : List ℕ),
      ((u.map (· - 1)).map (· - t)).sum + (u.map (fun x => if t < x then 1 else 0)).sum
        = (u.map (· - t)).sum := by
  intro u
  induction u with
  | nil => simp
  | cons a u0 ih =>
    simp only [List.map_cons, List.sum_cons]
    have hpp : a - 1 - t + (if t < a then 1 else 0) = a - t := by split_ifs <;> omega
    omega

/-- **Sum-count identity for the top-`Δ` decrement (π').** Decrementing `rest`'s first `Δ`
entries and truncating, plus the count of those first `Δ` exceeding `t`, recovers `rest`'s
truncated total. This is the `dec_T` analogue of `zipWith_sub_map_sub_sum_add`, stated on the
concrete `(rest.take Δ).map (·-1) ++ rest.drop Δ` form (= `π'` before re-sorting). -/
theorem trunc_dec_top_add (t : ℕ) (rest : List ℕ) (Δ : ℕ) :
    ((((rest.take Δ).map (· - 1)) ++ rest.drop Δ).map (· - t)).sum
      + ((rest.take Δ).map (fun x => if t < x then 1 else 0)).sum
    = (rest.map (· - t)).sum := by
  rw [List.map_append, List.sum_append]
  have hpp := map_pred_trunc_add t (rest.take Δ)
  have hsplit : (rest.map (· - t)).sum
      = ((rest.take Δ).map (· - t)).sum + ((rest.drop Δ).map (· - t)).sum := by
    conv_lhs => rw [← List.take_append_drop Δ rest]
    rw [List.map_append, List.sum_append]
  omega

theorem trunc_sum_dec_top_le (t : ℕ) (rest e : List ℕ)
    (hbin : ∀ x ∈ e, x ≤ 1) (hlen : e.length = rest.length)
    (hsort : List.Pairwise (· ≥ ·) rest) :
    ((((rest.take e.sum).map (· - 1)) ++ rest.drop e.sum).map (· - t)).sum
      ≤ ((List.zipWith (· - ·) rest e).map (· - t)).sum := by
  have h1' := trunc_dec_top_add t rest e.sum
  have h1 := zipWith_sub_map_sub_sum_add t rest e hlen hbin
  have h2 := wcount_le_take t rest e hbin hsort
  omega

theorem majorizesSorted_of_dec (rest e σ π' : List ℕ)
    (hbin : ∀ x ∈ e, x ≤ 1) (hlen : e.length = rest.length)
    (hsortrest : List.Pairwise (· ≥ ·) rest)
    (hσsort : List.Pairwise (· ≥ ·) σ)
    (hσperm : σ.Perm (List.zipWith (· - ·) rest e))
    (hπperm : π'.Perm (((rest.take e.sum).map (· - 1)) ++ rest.drop e.sum))
    (hsum : σ.sum = π'.sum) :
    MajorizesSorted σ π' := by
  apply majorizesSorted_of_map_sub_le σ π' hσsort hsum
  intro t
  have hσm : (σ.map (· - t)).sum = ((List.zipWith (· - ·) rest e).map (· - t)).sum :=
    (hσperm.map _).sum_eq
  have hπm : (π'.map (· - t)).sum
      = ((((rest.take e.sum).map (· - 1)) ++ rest.drop e.sum).map (· - t)).sum :=
    (hπperm.map _).sum_eq
  rw [hσm, hπm]
  exact trunc_sum_dec_top_le t rest e hbin hlen hsortrest

/-- **zipWith of the split maps is the map of differences.** For a list of `(value, flag)` pairs,
`zipWith (·-·)` of the value-list and the flag-list equals the elementwise difference. Pure list
identity used to rewrite the capstone's `zipWith (·-·) rest e` into a single `map`. -/
theorem zipWith_sub_map_fst_snd (L : List (ℕ × ℕ)) :
    List.zipWith (· - ·) (L.map Prod.fst) (L.map Prod.snd) = L.map (fun p => p.1 - p.2) := by
  induction L with
  | nil => rfl
  | cons hd tl ih => simp [ih]

/-- If every pair's second component is at most its first, the projected value-list dominates the
flag-list pointwise. With `pairsort_align` this supplies the `e ≤ rest` fact the assemble's `hsum`
needs (a graph's adjacency flags never exceed the degrees). -/
theorem forall₂_ge_map_fst_snd : ∀ (S : List (ℕ × ℕ)), (∀ p ∈ S, p.2 ≤ p.1) →
    List.Forall₂ (· ≥ ·) (S.map Prod.fst) (S.map Prod.snd)
  | [], _ => List.Forall₂.nil
  | p :: t, h =>
    List.Forall₂.cons (h p (by simp))
      (forall₂_ge_map_fst_snd t (fun q hq => h q (List.mem_cons_of_mem p hq)))

/-- **Pair-sort alignment (graph-assemble helper).** Sorting a list of `(value, flag)` pairs by
value (descending) yields a sorted value-list `rest` with a flag-list `e` that stays aligned to
it through the sort: `e` is binary (if the input flags are), the lengths match, `e.sum` is
preserved, and the pointwise difference `value - flag` is a permutation of `zipWith (·-·) rest e`.
This is exactly the aligned `rest`/`e` package the capstone `majorizesSorted_of_dec` consumes —
the alignment a bare degree-multiset formula (`degreeMultiset_induce_compl`) cannot by itself
supply, since sorting the values alone forgets which flag belonged to which value. -/
theorem pairsort_align (l : List (ℕ × ℕ)) (hbin : ∀ p ∈ l, p.2 ≤ 1)
    (hpair : ∀ p ∈ l, p.2 ≤ p.1) :
    ∃ rest e : List ℕ,
      List.Pairwise (· ≥ ·) rest ∧
      (∀ x ∈ e, x ≤ 1) ∧
      e.length = rest.length ∧
      e.sum = (l.map Prod.snd).sum ∧
      rest.Perm (l.map Prod.fst) ∧
      (l.map (fun p => p.1 - p.2)).Perm (List.zipWith (· - ·) rest e) ∧
      List.Forall₂ (· ≥ ·) rest e := by
  classical
  let r : ℕ × ℕ → ℕ × ℕ → Prop := fun a b => a.1 ≥ b.1
  haveI : DecidableRel r := fun a b => Nat.decLe b.1 a.1
  haveI : IsTrans (ℕ × ℕ) r := ⟨fun _ _ _ hab hbc => le_trans hbc hab⟩
  haveI : Std.Total r := ⟨fun a b => le_total b.1 a.1⟩
  set S := l.mergeSort (r · ·) with hSdef
  have hSperm : S.Perm l := List.mergeSort_perm l _
  have hSsort : List.Pairwise r S := List.pairwise_mergeSort' r l
  refine ⟨S.map Prod.fst, S.map Prod.snd, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · rw [List.pairwise_map]; exact hSsort
  · intro x hx
    rw [List.mem_map] at hx
    obtain ⟨p, hpS, rfl⟩ := hx
    exact hbin p (hSperm.mem_iff.mp hpS)
  · rw [List.length_map, List.length_map]
  · exact (hSperm.map Prod.snd).sum_eq
  · exact hSperm.map Prod.fst
  · rw [zipWith_sub_map_fst_snd S]
    exact (hSperm.map (fun p => p.1 - p.2)).symm
  · exact forall₂_ge_map_fst_snd S (fun p hp => hpair p (hSperm.mem_iff.mp hp))

/-- **Sorted map is a permutation of the `toList` map.** The descending sort of a mapped Finset
multiset is permutation-equal to mapping the Finset's `toList`. Pure `Finset`/`Multiset` bridge
(no graph content) that both the σ-side and the `rest`-side of the graph assemble reduce to:
`sort(deg(G-v))` and `sort(non-v degrees)` are permutations of the corresponding `toList` maps that
`pairsort_align`'s pair list is built from. -/
theorem sort_map_perm_toList_map {V : Type} [DecidableEq V] (s : Finset V) (f : V → ℕ) :
    (((s.val.map f).sort (· ≥ ·))).Perm (s.toList.map f) := by
  rw [← Multiset.coe_eq_coe, Multiset.sort_eq, ← Multiset.map_coe, Finset.coe_toList]

/-- If every entry of `l` is at least one, its length is at most its sum. -/
theorem length_le_sum_of_one_le : ∀ (l : List ℕ), (∀ x ∈ l, 1 ≤ x) → l.length ≤ l.sum
  | [], _ => by simp
  | a :: t, h => by
    have ha : 1 ≤ a := h a (by simp)
    have ih := length_le_sum_of_one_le t (fun x hx => h x (List.mem_cons_of_mem a hx))
    simp only [List.length_cons, List.sum_cons]; omega

/-- Dual: if every entry is at most one, the sum is at most the length. -/
theorem sum_le_length_of_le_one : ∀ (l : List ℕ), (∀ x ∈ l, x ≤ 1) → l.sum ≤ l.length
  | [], _ => by simp
  | a :: t, h => by
    have ha : a ≤ 1 := h a (by simp)
    have ih := sum_le_length_of_le_one t (fun x hx => h x (List.mem_cons_of_mem a hx))
    simp only [List.length_cons, List.sum_cons]; omega

/-- Decrementing every (positive) entry of `l` by one drops the sum by `l.length`. -/
theorem sum_map_pred_of_pos : ∀ (l : List ℕ), (∀ x ∈ l, 1 ≤ x) →
    (l.map (· - 1)).sum = l.sum - l.length
  | [], _ => by simp
  | a :: t, h => by
    have ha : 1 ≤ a := h a (by simp)
    have ht : ∀ x ∈ t, 1 ≤ x := fun x hx => h x (List.mem_cons_of_mem a hx)
    have ih := sum_map_pred_of_pos t ht
    have hlen := length_le_sum_of_one_le t ht
    simp only [List.map_cons, List.sum_cons, List.length_cons, ih]; omega

/-- A pointwise-`≥` pair of lists has monotone sums. -/
theorem sum_le_of_forall₂_ge (rest e : List ℕ) (h : List.Forall₂ (· ≥ ·) rest e) :
    e.sum ≤ rest.sum := by
  induction h with
  | nil => simp
  | cons hab _ ih => simp only [List.sum_cons]; omega

/-- **Sum of the aligned decrement.** When `e` is pointwise `≤ rest`, `zipWith (·-·)` has no
truncation, so its sum is `rest.sum - e.sum`. This computes `σ.sum` for the graph assemble
(`σ ~ zipWith (·-·) rest e`, and the graph's adjacency flags never exceed the degrees). -/
theorem sum_zipWith_sub_of_forall₂ (rest e : List ℕ) (h : List.Forall₂ (· ≥ ·) rest e) :
    (List.zipWith (· - ·) rest e).sum = rest.sum - e.sum := by
  induction h with
  | nil => simp
  | cons hab htail ih =>
    have hle := sum_le_of_forall₂_ge _ _ htail
    simp only [List.zipWith_cons_cons, List.sum_cons, ih]; omega

/-- **Sum of the top-`k` decrement.** Decrementing the top `k` (all-positive) entries drops the
sum by exactly `k`. This computes `π'.sum` for the graph assemble (`π' ~ (rest.take Δ).map (·-1)
++ rest.drop Δ`, and the top-Δ degrees are positive because `v`'s Δ neighbours have degree ≥ 1). -/
theorem sum_dec_top_eq (rest : List ℕ) (k : ℕ) (hk : k ≤ rest.length)
    (hpos : ∀ x ∈ rest.take k, 1 ≤ x) :
    (((rest.take k).map (· - 1)) ++ rest.drop k).sum = rest.sum - k := by
  rw [List.sum_append, sum_map_pred_of_pos (rest.take k) hpos, List.length_take,
    Nat.min_eq_left hk]
  have hsplit : (rest.take k).sum + (rest.drop k).sum = rest.sum := by
    rw [← List.sum_append, List.take_append_drop]
  have hlen := length_le_sum_of_one_le (rest.take k) hpos
  rw [List.length_take, Nat.min_eq_left hk] at hlen
  omega

/-- **Top-Δ positivity.** If `rest` is descending-sorted and pointwise dominates a binary flag
list `e`, then the top `e.sum` entries of `rest` are all positive. (Graph: `v`'s `Δ = e.sum`
neighbours each have degree ≥ 1, and the degrees are sorted, so the top Δ are ≥ 1.) Supplies the
`hpos` hypothesis of `sum_dec_top_eq` for the assemble's `hsum`. -/
theorem take_sum_pos : ∀ (rest e : List ℕ), rest.Pairwise (· ≥ ·) →
    List.Forall₂ (· ≥ ·) rest e → (∀ x ∈ e, x ≤ 1) → ∀ x ∈ rest.take e.sum, 1 ≤ x
  | [], [], _, _, _ => by simp
  | [], _ :: _, _, hf, _ => by cases hf
  | _ :: _, [], _, hf, _ => by cases hf
  | a :: rest', b :: e', hsort, hf, hbin => by
    obtain ⟨hab, htail⟩ := List.forall₂_cons.mp hf
    rw [List.pairwise_cons] at hsort
    have hbin' : ∀ x ∈ e', x ≤ 1 := fun x hx => hbin x (List.mem_cons_of_mem b hx)
    have hb : b ≤ 1 := hbin b (List.mem_cons_self ..)
    have ih : ∀ x ∈ rest'.take e'.sum, 1 ≤ x := take_sum_pos rest' e' hsort.2 htail hbin'
    intro x hx
    rw [List.sum_cons] at hx
    cases hbe : b + e'.sum with
    | zero => rw [hbe] at hx; simp at hx
    | succ m =>
      have hm_le : m ≤ e'.sum := by omega
      rw [hbe, List.take_succ_cons, List.mem_cons] at hx
      rcases hx with rfl | hx
      · rcases Nat.eq_zero_or_pos e'.sum with he | he
        · omega
        · cases rest' with
          | nil => rw [List.forall₂_nil_left_iff] at htail; subst htail; simp at he
          | cons c rest'' =>
            have hcmem : c ∈ (c :: rest'').take e'.sum := by
              obtain ⟨k, hk⟩ := Nat.exists_eq_succ_of_ne_zero (by omega : e'.sum ≠ 0)
              rw [hk, List.take_succ_cons]; exact List.mem_cons_self ..
            have hc1 := ih c hcmem
            have hac := hsort.1 c (List.mem_cons_self ..)
            omega
      · have hxe : x ∈ rest'.take e'.sum := by
          have heq : rest'.take m = (rest'.take e'.sum).take m := by
            rw [List.take_take, Nat.min_eq_left hm_le]
          rw [heq] at hx
          exact List.take_subset _ _ hx
        exact ih x hxe

end TxGraffitiC1FC
