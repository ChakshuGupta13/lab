import Mathlib.Data.Fin.Tuple.Sort

/-!
# Rearrangement preserves pointwise order (fact (iii) of `prop:monotone`)

Paper fact (iii) of the **Monotone reduction** (Section 4,
`prop:monotone`, main.tex L1047-1051): if `x k ≤ y k` for all `k`, then
the increasing rearrangements satisfy `(x ∘ sort x) k ≤ (y ∘ sort y) k`
for all `k`.

Paper's proof: at least `k+1` indices `i` satisfy `y i ≤ y↑ k`; at each
such `i`, `x i ≤ y i ≤ y↑ k`; so at least `k+1` entries of `x` are
`≤ y↑ k`, giving `x↑ k ≤ y↑ k`.

In Lean, the counting characterisation is provided by
`Tuple.lt_card_le_iff_apply_le_of_monotone`:

  `j < #{ i | f i ≤ a } ↔ (f ∘ sort f) j ≤ a`     (for sorted `f`).

Apply with `f = y ∘ sort y, a = (y ∘ sort y) k`: get `k < #{ i | (y ∘ sort y) i ≤ (y ∘ sort y) k}`. The subset `{i | (y ∘ sort y) i ≤ (y ∘ sort y) k}` injects
into `{i | (x ∘ sort x) i ≤ ?}` via the chain `x ≤ y ≤ y↑ k`; combine via
the same lemma applied to `f = x ∘ sort x` to get the conclusion.

Convention: we work with the **sorted versions** `x ∘ sort x`,
`y ∘ sort y` rather than `x`, `y` directly, because Mathlib's lemma is
stated for an already-sorted function.  The full statement
"`x k ≤ y k ∀ k ⟹ x↑ k ≤ y↑ k ∀ k`" is `sort_pointwise_le`.

Results:
* `sort_pointwise_le` — fact (iii), the pointwise ordering theorem.

No `sorry`.
-/

namespace OrigamiCone

variable {L : ℕ}

/-- **Fact (iii) of `prop:monotone`.**  If `x k ≤ y k` for every `k`,
then the sorted rearrangements also satisfy `(x ∘ sort x) k ≤ (y ∘ sort y) k`
for every `k`.

Proof via Mathlib's `Tuple.lt_card_le_iff_apply_le_of_monotone`: the
sorted-position characterisation `j < #{ i | f i ≤ a } ↔ (f ∘ sort f) j ≤ a`.
Apply at `f = y ∘ sort y, a = (y ∘ sort y) k` to get
`k < #{ i | (y ∘ sort y) i ≤ (y ∘ sort y) k }`.  The subset injects via the
chain `x ≤ y ≤ y↑ k` into `{ i | (x ∘ sort x) i ≤ (y ∘ sort y) k }`; reverse
the iff for `f = x ∘ sort x` to conclude. -/
theorem sort_pointwise_le {x y : Fin L → ℤ} (hxy : ∀ k, x k ≤ y k) (k : Fin L) :
    (x ∘ Tuple.sort x) k ≤ (y ∘ Tuple.sort y) k := by
  -- Abbreviations for the two sorted rearrangements.
  set xs := x ∘ Tuple.sort x with hxs
  set ys := y ∘ Tuple.sort y with hys
  -- Both are monotone (Mathlib).
  have hxs_mono : Monotone xs := Tuple.monotone_sort x
  have hys_mono : Monotone ys := Tuple.monotone_sort y
  -- Set `a = ys k`.  Goal becomes `xs k ≤ a`.
  set a := ys k with ha
  -- By Mathlib's lemma applied to `ys`: `k < #{i | ys i ≤ a} ↔ ys k ≤ a`.
  -- The RHS is `ys k ≤ ys k`, true.  So `k < #{i | ys i ≤ a}`.
  have h_ys_card : ↑k < Fintype.card {i // ys i ≤ a} := by
    rw [Tuple.lt_card_le_iff_apply_le_of_monotone ys a hys_mono k]
  -- Inject `{i | ys i ≤ a}` into `{i | xs i ≤ a}` via the chain
  --   xs i ≤ ys i ≤ a
  -- (using `xs i ≤ ys i` from the rearrangement-pointwise lemma below,
  -- but here we use a more direct route: invariance under the original
  -- pointwise bound via a different application of the same lemma.)
  --
  -- Direct route: it suffices to show
  --   k < #{ i | xs i ≤ a }
  -- and then apply the iff in the other direction.  We injectively map
  -- `{i | ys i ≤ a}` to `{i | xs i ≤ a}` using the SOURCE indices
  -- (post-sort): if `ys i ≤ a`, then `y (sort y i) ≤ a`, and via `x ≤ y`,
  -- `x (sort y i) ≤ a`; the post-sort position of `sort y i` under `sort x`
  -- may differ, but the COUNT does not — what we need is "at least k+1
  -- distinct indices `j` with `x j ≤ a`".  Get them by varying `i`.
  --
  -- Cleaner: show `Fintype.card {j // x j ≤ a} ≥ Fintype.card {i // ys i ≤ a}`
  -- and rewrite `{j // x j ≤ a}` to `{j // xs j ≤ a}` via the sort permutation.
  --
  -- We do this in two steps:
  --   (1) `#{i | ys i ≤ a} ≤ #{j | x j ≤ a}` via the source-index injection
  --       `i ↦ Tuple.sort y i` plus the pointwise bound `x ≤ y`.
  --   (2) `#{j | x j ≤ a} = #{j | xs j ≤ a}` because `Tuple.sort x` is a
  --       bijection `Fin L → Fin L`.
  --
  -- Step (1).  The map `i ↦ Tuple.sort y i` is injective on `Fin L` (as
  -- `Tuple.sort y` is a permutation), and it carries
  -- `{i | ys i ≤ a} = {i | y (sort y i) ≤ a}` into `{j | y j ≤ a}` since
  -- `ys i = y (sort y i)`.  Via the pointwise bound `x ≤ y`, `{j | y j ≤ a}
  -- ⊆ {j | x j ≤ a}`.
  have h_x_card_ge : Fintype.card {i // ys i ≤ a} ≤ Fintype.card {j // x j ≤ a} := by
    -- Build the injection `{i // ys i ≤ a} ↪ {j // x j ≤ a}` via
    -- `i ↦ Tuple.sort y i`, with the post-image satisfying `x ≤ a`
    -- because `x (sort y i) ≤ y (sort y i) = ys i ≤ a`.
    have hinj : Function.Injective
        (fun (i : {i // ys i ≤ a}) =>
          (⟨Tuple.sort y i.1, by
            have h1 := hxy (Tuple.sort y i.1)
            have h2 : ys i.1 ≤ a := i.2
            -- ys i.1 = y (sort y i.1)
            show x (Tuple.sort y i.1) ≤ a
            calc x (Tuple.sort y i.1)
                _ ≤ y (Tuple.sort y i.1) := h1
                _ = ys i.1 := rfl
                _ ≤ a := h2⟩ : {j // x j ≤ a})) := by
      intro a' b' h_eq
      have : Tuple.sort y a'.1 = Tuple.sort y b'.1 := by
        simpa using h_eq
      have h_a_b : a'.1 = b'.1 := (Tuple.sort y).injective this
      exact Subtype.ext h_a_b
    exact Fintype.card_le_of_injective _ hinj
  -- Step (2).  `{j | x j ≤ a}` and `{j | xs j ≤ a}` are in bijection via
  -- `Tuple.sort x : Fin L → Fin L`.
  have h_xs_card_eq : Fintype.card {j // x j ≤ a} = Fintype.card {j // xs j ≤ a} := by
    apply Fintype.card_congr
    refine
      ⟨fun j => ⟨(Tuple.sort x).symm j.1, ?_⟩,
       fun j => ⟨Tuple.sort x j.1, ?_⟩,
       ?_, ?_⟩
    · show xs ((Tuple.sort x).symm j.1) ≤ a
      have : xs ((Tuple.sort x).symm j.1) = x j.1 := by
        show x (Tuple.sort x ((Tuple.sort x).symm j.1)) = x j.1
        rw [Equiv.apply_symm_apply]
      rw [this]; exact j.2
    · show x (Tuple.sort x j.1) ≤ a
      have : x (Tuple.sort x j.1) = xs j.1 := rfl
      rw [this]; exact j.2
    · intro j; apply Subtype.ext
      show Tuple.sort x ((Tuple.sort x).symm j.1) = j.1
      exact Equiv.apply_symm_apply _ _
    · intro j; apply Subtype.ext
      show (Tuple.sort x).symm (Tuple.sort x j.1) = j.1
      exact Equiv.symm_apply_apply _ _
  -- Chain: k < #{i | ys i ≤ a} ≤ #{j | x j ≤ a} = #{j | xs j ≤ a}.
  have h_xs_card : ↑k < Fintype.card {j // xs j ≤ a} := by
    calc (↑k : ℕ) < Fintype.card {i // ys i ≤ a} := h_ys_card
      _ ≤ Fintype.card {j // x j ≤ a} := h_x_card_ge
      _ = Fintype.card {j // xs j ≤ a} := h_xs_card_eq
  -- Reverse the iff for `xs` to extract `xs k ≤ a`.
  exact (Tuple.lt_card_le_iff_apply_le_of_monotone xs a hxs_mono k).mp h_xs_card

end OrigamiCone
