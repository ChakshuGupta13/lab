import Mathlib
import TxGraffitiC1FC.Invariants

/-!
# Favaron's residue bound `R ≤ α` (item 6)

Target: `residue_le_indepNum : (G.residue : ℚ) ≤ (G.indepNum : ℚ)`.

**Proof structure** (computationally verified, `axiom-verification/favaron_verify.py`, 0 failures
/ 40k random graphs):
strong induction on `Fintype.card V`; delete a maximum-degree vertex `v`, then
`R(G) ≤ R(G−v) ≤ α(G−v) ≤ α(G)`. The two easy links are `α(G−v) ≤ α(G)` (an
independent set of `G−v` is one of `G`) and the IH `R(G−v) ≤ α(G−v)`. The hard link
is the **residue monotonicity** `R(G) ≤ R(G−v)` for a max-degree `v`, which is the
graphical-majorization content of Favaron–Mahéo–Saclé (1991): NOT a pure-sequence
fact (`axiom-verification/favaron_seq_lemma.py`: the naive "decrement-largest is minimal" claim
has 17k counterexamples; even full graphicality of `(Δ::tail)` leaves 4.5k — the
decremented sequences themselves must be graphical). This monotonicity is assumed as a
single named, computationally-verified, cited AXIOM
(`residue_le_residue_induce_compl_of_maxDegree`); everything else is proved here, so the
whole theorem depends only on that one named classical axiom (no `sorry`/`sorryAx`).

This file builds bottom-up: sequence lemmas about `residueAux` first (no graph, no
subtypes), then the graph base case, then the induction, then the cited axiom.
-/

namespace SimpleGraph

open Finset
open scoped Classical

set_option linter.unusedSectionVars false

variable {V : Type*} [Fintype V] [DecidableEq V]

/-! ## Sequence lemmas about `residueAux` (subtype-free) -/

/-- The residue of an all-zero list is its length (every entry is an isolated vertex).
This is the base case: an edgeless graph has all-zero degree sequence. -/
theorem residueAux_replicate_zero (n : ℕ) : residueAux (List.replicate n 0) = n := by
  cases n with
  | zero => simp [residueAux]
  | succ k =>
    rw [List.replicate_succ]
    simp only [residueAux, List.length_replicate]
    omega

/-- The residue never exceeds the list length (`R ≤ n`). -/
theorem residueAux_le_length (l : List ℕ) : residueAux l ≤ l.length := by
  induction l using residueAux.induct with
  | case1 => simp [residueAux]
  | case2 s =>
    simp only [residueAux, List.length_cons]
    omega
  | case3 d rest hd ih =>
    rw [residueAux]
    · calc residueAux (havelHakimiStep (d :: rest))
          ≤ (havelHakimiStep (d :: rest)).length := ih
        _ = rest.length := havelHakimiStep_length_cons d rest
        _ ≤ (d :: rest).length := by simp
    · exact hd

/-! ## Graph base case: an edgeless graph has `R = card V = α` -/

/-- The independence number never exceeds the number of vertices. -/
theorem indepNum_le_card (G : SimpleGraph V) : G.indepNum ≤ Fintype.card V := by
  obtain ⟨s, _, hcard⟩ := G.exists_isNIndepSet_indepNum
  rw [← hcard]
  exact Finset.card_le_univ s

/-- **α-bridge**: an independent set of an induced subgraph is an independent set of
the ambient graph, so `α(G[s]) ≤ α(G)`. (Not gated on the crux; closes the
`α(G−v) ≤ α(G)` link of the induction.) -/
theorem indepNum_induce_le (G : SimpleGraph V) (s : Set V) :
    (G.induce s).indepNum ≤ G.indepNum := by
  classical
  obtain ⟨t, htind, htcard⟩ := (G.induce s).exists_isNIndepSet_indepNum
  rw [← htcard, ← Finset.card_map (Function.Embedding.subtype (· ∈ s)) (s := t)]
  apply SimpleGraph.IsIndepSet.card_le_indepNum
  intro a ha b hb hne hadj
  simp only [Finset.coe_map, Set.mem_image, Finset.mem_coe,
    Function.Embedding.coe_subtype] at ha hb
  obtain ⟨a', ha't, rfl⟩ := ha
  obtain ⟨b', hb't, rfl⟩ := hb
  exact htind ha't hb't (fun h => hne (by rw [h])) hadj

/-- Residue of an edgeless graph (all degrees zero): every vertex is isolated, so
`R = card V`. Stated over `G` with its ambient instance to avoid `⊥`-instance clashes. -/
theorem residue_eq_card_of_degree_zero (G : SimpleGraph V) [DecidableRel G.Adj]
    (h : ∀ v, G.degree v = 0) : G.residue = Fintype.card V := by
  unfold residue
  have hsort :
      (Finset.univ.val.map fun v => G.degree v).sort (· ≥ ·)
        = List.replicate (Fintype.card V) 0 := by
    rw [List.eq_replicate_iff]
    refine ⟨?_, ?_⟩
    · rw [Multiset.length_sort, Multiset.card_map, Finset.card_val, Finset.card_univ]
    · intro b hb
      rw [Multiset.mem_sort, Multiset.mem_map] at hb
      obtain ⟨v, _, rfl⟩ := hb
      exact h v
  rw [hsort, residueAux_replicate_zero]

/-- Independence number of an edgeless graph (all degrees zero): the whole vertex set
is independent, so `α = card V`. -/
theorem indepNum_eq_card_of_degree_zero (G : SimpleGraph V) [DecidableRel G.Adj]
    (h : ∀ v, G.degree v = 0) : G.indepNum = Fintype.card V := by
  refine le_antisymm G.indepNum_le_card ?_
  have huniv : G.IsIndepSet (Finset.univ : Finset V) := by
    intro a _ b _ _ hab
    have hpos : 0 < G.degree a := (G.degree_pos_iff_exists_adj a).mpr ⟨b, hab⟩
    rw [h a] at hpos; exact absurd hpos (lt_irrefl 0)
  calc Fintype.card V = (Finset.univ : Finset V).card := (Finset.card_univ).symm
    _ ≤ G.indepNum := huniv.card_le_indepNum
/-! ## The residue-monotonicity crux (assumed as a cited classical axiom) -/

/-- **AXIOM — cited classical input.** The residue monotonicity `R(G) ≤ R(G − v)` for a
maximum-degree vertex `v` **of positive degree**. This is the graphical-majorization step
underlying the classical bound `R ≤ α` first established by Favaron, Mahéo & Saclé (1991); the
paper being formalized (Gupta, arXiv:2606.29553) *uses* that classical bound rather than
re-proving it, so it is assumed here as a named, cited axiom rather than re-derived. It is NOT
`sorry`: `#print axioms` shows it by name, so the dependence is explicit and auditable.

Computationally verified: for all graphs with an edge up to 8 vertices
(`axiom-verification/favaron_verify.py`, guarded by `max degree ≥ 1`, 0 failures). The `0 < G.degree v`
hypothesis is essential — on an edgeless graph `R(G) ≤ R(G−v)` reads `n ≤ n−1` and is false;
that case is discharged separately (`residue_eq_card_of_degree_zero`).

**Precise content** (`axiom-verification/favaron_majorize.py`, exhaustive over all graphical sequences
`n ≤ 7`): *residue is Schur-convex on graphical sequences* — if `a, b` are both graphical of
equal length and `a` majorizes `b`, then `residue a ≥ residue b`. The axiom is this applied to
`σ := sort(deg(G−v))` (decrement `v`'s neighbours) which majorizes `π' := HH(sort(deg G))`
(decrement the `Δ` largest), both graphical, giving `R(G−v) = residue σ ≥ residue π' = R(G)`.

A native Lean proof needs a combinatorial majorization + graphical-sequence library Mathlib
lacks (Muirhead unit-transfer decomposition + graphicality preservation + per-transfer residue
monotonicity). Six computational probes rule out every shortcut: the pure-sequence form
(`favaron_seq_lemma.py`, 17k counterexamples), graphicality of `(Δ::tail)` alone
(`favaron_refined.py`, 4.5k), pointwise monotonicity of `residueAux` and pointwise-minimality of
decrement-largest (`favaron_ptwise.py`, both false), and the inductive-via-HH route (HH on
majorization-comparable sequences diverges: different heads ⟹ different sums ⟹ no longer
comparable). Replacing this axiom with a native proof is a self-contained multi-week project. -/
axiom residue_le_residue_induce_compl_of_maxDegree
    (G : SimpleGraph V) [DecidableRel G.Adj] {v : V}
    (hv : ∀ w, G.degree w ≤ G.degree v) (hpos : 0 < G.degree v) :
    G.residue ≤ (G.induce ({v}ᶜ : Set V)).residue

/-! ## Main theorem: `R ≤ α` (modulo the crux above) -/

/-- **Favaron's residue bound** `R(G) ≤ α(G)`, assembled by strong induction on
`|V|`: delete a maximum-degree vertex `v`; then
`R(G) ≤ R(G−v) ≤ α(G−v) ≤ α(G)` via the residue monotonicity crux, the IH, and the
α-bridge. Every link is proved here except the monotonicity crux, which is the named,
computationally-verified cited axiom `residue_le_residue_induce_compl_of_maxDegree`
(Favaron–Mahéo–Saclé 1991) — so `#print axioms` names it explicitly, no `sorry`. -/
theorem residue_le_indepNum (G : SimpleGraph V) [DecidableRel G.Adj] :
    G.residue ≤ G.indepNum := by
  suffices h : ∀ (n : ℕ) (W : Type _) [Fintype W] [DecidableEq W]
      (H : SimpleGraph W) [DecidableRel H.Adj], Fintype.card W = n → H.residue ≤ H.indepNum by
    exact h (Fintype.card V) V G rfl
  intro n
  induction n using Nat.strong_induction_on with
  | _ n IH =>
    intro W _ _ H _ hcard
    rcases isEmpty_or_nonempty W with hem | hne
    · have hz : ∀ w, H.degree w = 0 := fun w => (hem.false w).elim
      rw [residue_eq_card_of_degree_zero H hz]
      exact (indepNum_eq_card_of_degree_zero H hz).ge
    · obtain ⟨v, hv⟩ := Finite.exists_max (fun w => H.degree w)
      by_cases hdeg : H.degree v = 0
      · have hz : ∀ w, H.degree w = 0 := fun w => Nat.le_zero.mp (hdeg ▸ hv w)
        rw [residue_eq_card_of_degree_zero H hz]
        exact (indepNum_eq_card_of_degree_zero H hz).ge
      · have hcard' : Fintype.card ↥({v}ᶜ : Set W) = n - 1 := by
          rw [← hcard, ← Set.toFinset_card, Set.toFinset_compl, Set.toFinset_singleton,
            Finset.card_compl, Finset.card_singleton]
        have hlt : n - 1 < n := by
          have hpos : 0 < Fintype.card W := Fintype.card_pos
          omega
        have hIH := IH (n - 1) hlt ↥({v}ᶜ : Set W) (H.induce _) hcard'
        calc H.residue
            ≤ (H.induce ({v}ᶜ : Set W)).residue :=
              H.residue_le_residue_induce_compl_of_maxDegree hv (Nat.pos_of_ne_zero hdeg)
          _ ≤ (H.induce ({v}ᶜ : Set W)).indepNum := hIH
          _ ≤ H.indepNum := H.indepNum_induce_le _

/-- Rational-valued form, for the `Conjecture.lean` assembly. -/
theorem residue_le_indepNum_rat (G : SimpleGraph V) [DecidableRel G.Adj] :
    (G.residue : ℚ) ≤ (G.indepNum : ℚ) := by
  exact_mod_cast G.residue_le_indepNum

end SimpleGraph
