import Mathlib
import TxGraffitiC1FC.Graphical

namespace SimpleGraph

variable {V : Type} [DecidableEq V]

open Classical in
/-- **The 2-switch.** From a graph `G` and four vertices, remove the edges `{a,b}` and `{c,d}` and
add `{a,c}` and `{b,d}`. On the intended inputs (`{a,b},{c,d}` edges, `{a,c},{b,d}` non-edges, the
four vertices distinct) this preserves every degree while changing the adjacency pattern. Built
from Mathlib's `deleteEdges` and `fromEdgeSet` so symmetry/looplessness are automatic. -/
noncomputable def twoSwitch (G : SimpleGraph V) (a b c d : V) : SimpleGraph V :=
  (G.deleteEdges {s(a, b), s(c, d)}) ⊔ SimpleGraph.fromEdgeSet {s(a, c), s(b, d)}

/-- Adjacency in the 2-switch: keep `G`-edges that are neither `{a,b}` nor `{c,d}`, and add the
non-loop pairs `{a,c}`, `{b,d}`. -/
theorem twoSwitch_adj (G : SimpleGraph V) (a b c d x y : V) :
    (G.twoSwitch a b c d).Adj x y ↔
      (G.Adj x y ∧ s(x, y) ≠ s(a, b) ∧ s(x, y) ≠ s(c, d))
      ∨ ((s(x, y) = s(a, c) ∨ s(x, y) = s(b, d)) ∧ x ≠ y) := by
  simp only [twoSwitch, sup_adj, deleteEdges_adj, fromEdgeSet_adj, Set.mem_insert_iff,
    Set.mem_singleton_iff]
  tauto

open Classical in
/-- **The 2-switch preserves every degree.** Given the intended inputs — `{a,b},{c,d}` edges,
`{a,c},{b,d}` non-edges, the four vertices pairwise distinct — each of `a,b,c,d` swaps exactly one
neighbour (`a: −b+c`, `b: −a+d`, `c: −d+a`, `d: −c+b`) and every other vertex is untouched, so all
degrees are unchanged. This is the core property that makes the 2-switch a move between
realisations of the same degree sequence. -/
theorem twoSwitch_degree (G : SimpleGraph V) [Fintype V] (a b c d : V)
    (hab : G.Adj a b) (hcd : G.Adj c d) (hac : ¬ G.Adj a c) (hbd : ¬ G.Adj b d)
    (hac' : a ≠ c) (had : a ≠ d) (hbc : b ≠ c) (hbd' : b ≠ d) (x : V) :
    (G.twoSwitch a b c d).degree x = G.degree x := by
  have hab' : a ≠ b := hab.ne
  have hcd' : c ≠ d := hcd.ne
  have card_swap : ∀ (S : Finset V) (o n : V), o ∈ S → n ∉ S →
      (insert n (S.erase o)).card = S.card := fun S o n ho hn => by
    rw [Finset.card_insert_of_notMem (fun h => hn (Finset.mem_of_mem_erase h)),
      Finset.card_erase_of_mem ho]
    have : 1 ≤ S.card := Finset.card_pos.mpr ⟨o, ho⟩
    omega
  rw [← SimpleGraph.card_neighborFinset_eq_degree, ← SimpleGraph.card_neighborFinset_eq_degree]
  by_cases hxa : x = a
  · subst x
    have he : (G.twoSwitch a b c d).neighborFinset a = insert c ((G.neighborFinset a).erase b) := by
      ext y
      simp only [mem_neighborFinset, twoSwitch_adj, Finset.mem_insert, Finset.mem_erase, Sym2.eq_iff]
      aesop
    have hbm : b ∈ G.neighborFinset a := by rw [mem_neighborFinset]; exact hab
    have hcn : c ∉ G.neighborFinset a := by rw [mem_neighborFinset]; exact hac
    rw [he, card_swap _ b c hbm hcn]
  · by_cases hxb : x = b
    · subst x
      have he : (G.twoSwitch a b c d).neighborFinset b = insert d ((G.neighborFinset b).erase a) := by
        ext y
        simp only [mem_neighborFinset, twoSwitch_adj, Finset.mem_insert, Finset.mem_erase, Sym2.eq_iff]
        aesop
      have ham : a ∈ G.neighborFinset b := by rw [mem_neighborFinset]; exact hab.symm
      have hdn : d ∉ G.neighborFinset b := by rw [mem_neighborFinset]; exact hbd
      rw [he, card_swap _ a d ham hdn]
    · by_cases hxc : x = c
      · subst x
        have he : (G.twoSwitch a b c d).neighborFinset c = insert a ((G.neighborFinset c).erase d) := by
          ext y
          simp only [mem_neighborFinset, twoSwitch_adj, Finset.mem_insert, Finset.mem_erase, Sym2.eq_iff]
          aesop
        have hdm : d ∈ G.neighborFinset c := by rw [mem_neighborFinset]; exact hcd
        have han : a ∉ G.neighborFinset c := by rw [mem_neighborFinset]; exact fun h => hac h.symm
        rw [he, card_swap _ d a hdm han]
      · by_cases hxd : x = d
        · subst x
          have he : (G.twoSwitch a b c d).neighborFinset d = insert b ((G.neighborFinset d).erase c) := by
            ext y
            simp only [mem_neighborFinset, twoSwitch_adj, Finset.mem_insert, Finset.mem_erase, Sym2.eq_iff]
            aesop
          have hcm : c ∈ G.neighborFinset d := by rw [mem_neighborFinset]; exact hcd.symm
          have hbn : b ∉ G.neighborFinset d := by rw [mem_neighborFinset]; exact fun h => hbd h.symm
          rw [he, card_swap _ c b hcm hbn]
        · congr 1
          ext y
          simp only [mem_neighborFinset, twoSwitch_adj, Sym2.eq_iff]
          aesop

open Classical in
/-- The **adjacency-weight objective**: the total `wt`-weight of `v`'s neighbours. It selects an
extremal realisation of a degree sequence in the Havel–Hakimi argument — with `wt` decreasing in
degree-rank, the realisation maximising `adjWeight` at `v` has `v` adjacent to the highest-degree
vertices. -/
noncomputable def adjWeight (G : SimpleGraph V) [Fintype V] (wt : V → ℕ) (v : V) : ℕ :=
  ∑ w ∈ G.neighborFinset v, wt w

open Classical in
/-- **A West-form 2-switch strictly raises the adjacency weight of `v`.** The 2-switch replaces `v`'s
neighbour `vⱼ` (an edge, `hvj`) by the non-neighbour `vᵢ` (`hvi`); when `vᵢ` outweighs `vⱼ` the total
weight of `v`'s neighbourhood strictly increases. Only the `v`-side of the switch matters here — the
`{vᵢ,w}`/`{vⱼ,w}` conditions that make it degree-preserving (`twoSwitch_degree`) are applied
separately by the caller. This is the non-maximality engine of the Havel–Hakimi extremal argument:
at a maximiser of `adjWeight` no such switch exists, forcing `v` adjacent to the top-weight vertices. -/
theorem adjWeight_lt_twoSwitch (G : SimpleGraph V) [Fintype V] (v vj vi w : V) (wt : V → ℕ)
    (hvj : G.Adj v vj) (hvi : ¬ G.Adj v vi)
    (hvvi : v ≠ vi) (hvw : v ≠ w) (hwt : wt vj < wt vi) :
    adjWeight G wt v < adjWeight (G.twoSwitch v vj vi w) wt v := by
  have hvvj : v ≠ vj := hvj.ne
  have he : (G.twoSwitch v vj vi w).neighborFinset v
      = insert vi ((G.neighborFinset v).erase vj) := by
    ext y
    simp only [mem_neighborFinset, twoSwitch_adj, Finset.mem_insert, Finset.mem_erase, Sym2.eq_iff]
    aesop
  have hjm : vj ∈ G.neighborFinset v := by rw [mem_neighborFinset]; exact hvj
  have him : vi ∉ G.neighborFinset v := by rw [mem_neighborFinset]; exact hvi
  have hsplit : wt vj + ∑ u ∈ (G.neighborFinset v).erase vj, wt u
      = ∑ u ∈ G.neighborFinset v, wt u := Finset.add_sum_erase _ wt hjm
  unfold adjWeight
  rw [he, Finset.sum_insert (fun h => him (Finset.mem_of_mem_erase h))]
  omega

open Classical in
/-- **Existence of the swap partner.** If `v ~ vⱼ` but `v ≁ vᵢ` with `deg vᵢ ≥ deg vⱼ`, then some
vertex `w ∉ {v, vᵢ, vⱼ}` is adjacent to `vᵢ` but not to `vⱼ`. This `w` is exactly the fourth vertex of
a degree-preserving West-form 2-switch `twoSwitch v vⱼ vᵢ w` (its edges `{vᵢ,w}` and non-edges
`{vⱼ,w}`). Counting: `v ∈ N(vⱼ) \ N(vᵢ)` and `|N(vᵢ)| ≥ |N(vⱼ)|` force `N(vᵢ) \ (N(vⱼ) ∪ {vⱼ})`
nonempty — in the `vᵢ ~ vⱼ` case `vᵢ ∈ N(vⱼ) \ N(vᵢ)` too, giving the extra room to avoid `vⱼ`. -/
theorem exists_twoSwitch_partner (G : SimpleGraph V) [Fintype V] (v vi vj : V)
    (hvj : G.Adj v vj) (hvi : ¬ G.Adj v vi) (hvvi : v ≠ vi)
    (hdeg : G.degree vj ≤ G.degree vi) :
    ∃ w, w ≠ v ∧ w ≠ vi ∧ w ≠ vj ∧ G.Adj vi w ∧ ¬ G.Adj vj w := by
  have hvNj : v ∈ G.neighborFinset vj := by rw [mem_neighborFinset]; exact hvj.symm
  have hvNi : v ∉ G.neighborFinset vi := by rw [mem_neighborFinset]; exact fun h => hvi h.symm
  suffices hne : (G.neighborFinset vi \ insert vj (G.neighborFinset vj)).Nonempty by
    obtain ⟨w, hw⟩ := hne
    rw [Finset.mem_sdiff, Finset.mem_insert, not_or] at hw
    obtain ⟨hwNi, hwvj, hwNj⟩ := hw
    have hadj : G.Adj vi w := (mem_neighborFinset G vi w).mp hwNi
    refine ⟨w, ?_, hadj.ne', hwvj, hadj, ?_⟩
    · rintro rfl; exact hvNi hwNi
    · rw [← mem_neighborFinset]; exact hwNj
  rw [Finset.sdiff_nonempty]
  intro hsub
  have hkey : (G.neighborFinset vi).erase vj ⊆ (G.neighborFinset vj).erase v := by
    intro w hw
    obtain ⟨hwvj, hwNi⟩ := Finset.mem_erase.mp hw
    rcases Finset.mem_insert.mp (hsub hwNi) with h | h
    · exact absurd h hwvj
    · exact Finset.mem_erase.mpr ⟨fun hwv => hvNi (hwv ▸ hwNi), h⟩
  by_cases hij : G.Adj vi vj
  · have hvjNi : vj ∈ G.neighborFinset vi := (mem_neighborFinset G vi vj).mpr hij
    have hviNj : vi ∈ G.neighborFinset vj := (mem_neighborFinset G vj vi).mpr hij.symm
    have hkey2 : (G.neighborFinset vi).erase vj ⊆ ((G.neighborFinset vj).erase v).erase vi := by
      intro w hw
      have hwNi : w ∈ G.neighborFinset vi := (Finset.mem_erase.mp hw).2
      exact Finset.mem_erase.mpr ⟨fun hwvi => ((mem_neighborFinset G vi vi).mp (hwvi ▸ hwNi)).ne rfl, hkey hw⟩
    have hc := Finset.card_le_card hkey2
    rw [Finset.card_erase_of_mem hvjNi,
        Finset.card_erase_of_mem (Finset.mem_erase.mpr ⟨hvvi.symm, hviNj⟩),
        Finset.card_erase_of_mem hvNj,
        card_neighborFinset_eq_degree, card_neighborFinset_eq_degree] at hc
    have hpi : 1 ≤ G.degree vi := by
      rw [← card_neighborFinset_eq_degree]; exact Finset.card_pos.mpr ⟨vj, hvjNi⟩
    have hpj : 2 ≤ G.degree vj := by
      rw [← card_neighborFinset_eq_degree]
      exact Finset.one_lt_card.mpr ⟨v, hvNj, vi, hviNj, hvvi⟩
    omega
  · have hvjNi : vj ∉ G.neighborFinset vi := by rw [mem_neighborFinset]; exact hij
    rw [Finset.erase_eq_of_notMem hvjNi] at hkey
    have hc := Finset.card_le_card hkey
    rw [Finset.card_erase_of_mem hvNj, card_neighborFinset_eq_degree,
        card_neighborFinset_eq_degree] at hc
    have hpj : 1 ≤ G.degree vj := by
      rw [← card_neighborFinset_eq_degree]; exact Finset.card_pos.mpr ⟨v, hvNj⟩
    omega

open Classical in
/-- **The 2-switch preserves the whole degree multiset.** Immediate from `twoSwitch_degree` (every
vertex degree is preserved) via `Multiset.map_congr`. Consequently a 2-switch keeps a graph inside
the set of realisations of a fixed degree sequence — the invariant the Havel–Hakimi extremal argument
optimises within. -/
theorem twoSwitch_degreeMultiset (G : SimpleGraph V) [Fintype V] (a b c d : V)
    (hab : G.Adj a b) (hcd : G.Adj c d) (hac : ¬ G.Adj a c) (hbd : ¬ G.Adj b d)
    (hac' : a ≠ c) (had : a ≠ d) (hbc : b ≠ c) (hbd' : b ≠ d) :
    (G.twoSwitch a b c d).degreeMultiset = G.degreeMultiset := by
  unfold degreeMultiset
  exact Multiset.map_congr rfl
    (fun x _ => twoSwitch_degree G a b c d hab hcd hac hbd hac' had hbc hbd' x)

open Classical in
/-- **A maximiser of the adjacency weight exists** among all graphs sharing `G`'s degree function.
That set is finite (`SimpleGraph V` is a `Fintype`) and nonempty (it contains `G`), so
`adjWeight · wt v₀` attains a maximum on it. The Havel–Hakimi extremal argument runs at this
maximiser: since every degree-preserving 2-switch keeps a graph in the set (`twoSwitch_degree`) and,
in the relevant configuration, strictly raises the objective (`adjWeight_lt_twoSwitch`), no such
switch can exist at the maximiser — forcing `v₀` adjacent to the highest-weight vertices. -/
theorem exists_adjWeight_max (G : SimpleGraph V) [Fintype V] (wt : V → ℕ) (v₀ : V) :
    ∃ Gmax : SimpleGraph V, (∀ x, Gmax.degree x = G.degree x) ∧
      ∀ G' : SimpleGraph V, (∀ x, G'.degree x = G.degree x) →
        adjWeight G' wt v₀ ≤ adjWeight Gmax wt v₀ := by
  obtain ⟨Gmax, hmem, hmax⟩ :=
    Finset.exists_max_image
      (Finset.univ.filter (fun G' : SimpleGraph V => ∀ x, G'.degree x = G.degree x))
      (fun G' => adjWeight G' wt v₀)
      ⟨G, Finset.mem_filter.mpr ⟨Finset.mem_univ _, fun _ => rfl⟩⟩
  rw [Finset.mem_filter] at hmem
  exact ⟨Gmax, hmem.2, fun G' hG' => hmax G' (Finset.mem_filter.mpr ⟨Finset.mem_univ _, hG'⟩)⟩

open Classical in
/-- **At an adjacency-weight maximiser, `v₀`'s neighbours carry the top degrees.** Among all graphs
sharing `G`'s degree function, some maximiser `Gmax` has no *inversion*: every vertex not adjacent to
`v₀` (other than `v₀` itself) has degree at most that of every neighbour of `v₀`. Otherwise a
`v₀`-anchored West 2-switch — partner from `exists_twoSwitch_partner`, degree-preserving by
`twoSwitch_degree` so it stays in the set — would strictly raise `adjWeight` by
`adjWeight_lt_twoSwitch`, contradicting maximality. Taking `v₀` of maximum degree, this says its
neighbours realise the Δ largest degrees: the Havel–Hakimi adjacency. -/
theorem exists_realization_noInversion (G : SimpleGraph V) [Fintype V] (v₀ : V) :
    ∃ Gmax : SimpleGraph V, (∀ x, Gmax.degree x = G.degree x) ∧
      ∀ vi vj, vi ≠ v₀ → ¬ Gmax.Adj v₀ vi → Gmax.Adj v₀ vj →
        Gmax.degree vi ≤ Gmax.degree vj := by
  obtain ⟨Gmax, hdeg, hmax⟩ := exists_adjWeight_max G (fun u => G.degree u) v₀
  refine ⟨Gmax, hdeg, fun vi vj hvi_ne hvi_nadj hvj_adj => ?_⟩
  by_contra hle
  rw [not_le] at hle
  have hv0vi : v₀ ≠ vi := hvi_ne.symm
  obtain ⟨w, hwv, hwvi, hwvj, hadj_iw, hnadj_jw⟩ :=
    exists_twoSwitch_partner Gmax v₀ vi vj hvj_adj hvi_nadj hv0vi hle.le
  have hvjvi : vj ≠ vi := fun h => ne_of_lt hle (congrArg (fun z => Gmax.degree z) h)
  have hstep : adjWeight Gmax (fun u => G.degree u) v₀
      < adjWeight (Gmax.twoSwitch v₀ vj vi w) (fun u => G.degree u) v₀ :=
    adjWeight_lt_twoSwitch Gmax v₀ vj vi w (fun u => G.degree u)
      hvj_adj hvi_nadj hv0vi hwv.symm
      (by show G.degree vj < G.degree vi; rw [← hdeg vj, ← hdeg vi]; exact hle)
  have hin : adjWeight (Gmax.twoSwitch v₀ vj vi w) (fun u => G.degree u) v₀
      ≤ adjWeight Gmax (fun u => G.degree u) v₀ :=
    hmax (Gmax.twoSwitch v₀ vj vi w)
      (fun x => (twoSwitch_degree Gmax v₀ vj vi w hvj_adj hadj_iw hvi_nadj hnadj_jw
        hv0vi hwv.symm hvjvi hwvj.symm x).trans (hdeg x))
  exact absurd (lt_of_lt_of_le hstep hin) (lt_irrefl _)

open Classical in
/-- **Induced degree multiset splits into decremented neighbours and unchanged non-neighbours.**
Deleting `v` decrements exactly the degrees of `v`'s neighbours; every other vertex keeps its degree.
This separates `degreeMultiset (G - v)` into the two pieces the Havel–Hakimi bookkeeping compares
against `havelHakimiStep`: the top-Δ (neighbour) degrees each `−1`, and the remaining degrees. -/
theorem degreeMultiset_induce_compl_split (G : SimpleGraph V) [Fintype V] (v : V) :
    (G.induce ({v}ᶜ : Set V)).degreeMultiset
      = (G.neighborFinset v).val.map (fun w => G.degree w - 1)
        + ((Finset.univ.erase v).filter (fun w => ¬ G.Adj w v)).val.map (fun w => G.degree w) := by
  have hNfilter : (Finset.univ.erase v).filter (fun w => G.Adj w v) = G.neighborFinset v := by
    ext w
    simp only [Finset.mem_filter, Finset.mem_erase, Finset.mem_univ, and_true, mem_neighborFinset]
    exact ⟨fun ⟨_, h⟩ => G.adj_symm h, fun h => ⟨(G.ne_of_adj h).symm, G.adj_symm h⟩⟩
  rw [degreeMultiset_induce_compl,
    ← Multiset.filter_add_not (fun w => G.Adj w v) (Finset.univ.erase v).val, Multiset.map_add]
  congr 1
  · rw [← Finset.filter_val, hNfilter]
    apply Multiset.map_congr rfl
    intro w hw
    have hadj : G.Adj w v := G.adj_symm ((mem_neighborFinset G v w).mp (Finset.mem_val.mp hw))
    rw [if_pos hadj]
  · rw [← Finset.filter_val]
    apply Multiset.map_congr rfl
    intro w hw
    have hnadj : ¬ G.Adj w v := (Finset.mem_filter.mp (Finset.mem_val.mp hw)).2
    rw [if_neg hnadj, Nat.sub_zero]

/-- **Domination splits a sorted multiset at the boundary.** If every element of `A` is at least
every element of `B`, then sorting `A + B` descending places exactly `A` in the first `|A|` entries
and `B` in the remaining entries. With `A` = neighbour degrees and `B` = non-neighbour degrees this
is the "no inversion ⟹ the neighbours are the top-Δ" step of the Havel–Hakimi bookkeeping. Proof:
`A.sort ++ B.sort` is descending-sorted (both halves sorted, and every `A`-element dominates every
`B`-element) and is a permutation of `(A+B).sort`, hence equal to it; then `take`/`drop` at the
boundary `|A|` peel off the two halves. -/
theorem multiset_top_split (A B : Multiset ℕ) (hdom : ∀ a ∈ A, ∀ b ∈ B, b ≤ a) :
    A = ↑(((A + B).sort (· ≥ ·)).take A.card)
      ∧ B = ↑(((A + B).sort (· ≥ ·)).drop A.card) := by
  have hs1 : ((A + B).sort (· ≥ ·)).SortedGE := (Multiset.pairwise_sort _ _).sortedGE
  have hs2 : (A.sort (· ≥ ·) ++ B.sort (· ≥ ·)).SortedGE := by
    rw [List.sortedGE_iff_pairwise, List.pairwise_append]
    refine ⟨Multiset.pairwise_sort _ _, Multiset.pairwise_sort _ _, ?_⟩
    intro a ha b hb
    rw [Multiset.mem_sort] at ha hb
    exact hdom a ha b hb
  have hperm : ((A + B).sort (· ≥ ·)).Perm (A.sort (· ≥ ·) ++ B.sort (· ≥ ·)) := by
    rw [← Multiset.coe_eq_coe, ← Multiset.coe_add]
    simp only [Multiset.sort_eq]
  have hcat : (A + B).sort (· ≥ ·) = A.sort (· ≥ ·) ++ B.sort (· ≥ ·) :=
    List.Perm.eq_of_sortedGE hs1 hs2 hperm
  have hlen : A.card = (A.sort (· ≥ ·)).length := by rw [Multiset.length_sort]
  refine ⟨?_, ?_⟩
  · rw [hcat, hlen, List.take_left, Multiset.sort_eq]
  · rw [hcat, hlen, List.drop_left, Multiset.sort_eq]

open Classical in
/-- **The maximiser's vertex-deletion realises one Havel–Hakimi step.** If `v₀` has maximum degree
and the graph has no inversion at `v₀` (`v₀`'s neighbours carry the top degrees), then deleting `v₀`
produces exactly the degree multiset of `havelHakimiStep` applied to the sorted degree sequence:
`v₀`'s Δ neighbours are the top-Δ, each decremented, and the remaining degrees are untouched. This
is the Havel–Hakimi reduction realised on an actual graph — the heart of the graphicality theorem. -/
theorem induce_degreeMultiset_eq_havelHakimiStep (Gmax : SimpleGraph V) [Fintype V] (v₀ : V)
    (hmax : ∀ w, Gmax.degree w ≤ Gmax.degree v₀)
    (hinv : ∀ vi vj, vi ≠ v₀ → ¬ Gmax.Adj v₀ vi → Gmax.Adj v₀ vj →
      Gmax.degree vi ≤ Gmax.degree vj) :
    (Gmax.induce ({v₀}ᶜ : Set V)).degreeMultiset
      = ↑(havelHakimiStep ((Gmax.degreeMultiset).sort (· ≥ ·))) := by
  have hNfilter : (Finset.univ.erase v₀).filter (fun w => Gmax.Adj w v₀)
      = Gmax.neighborFinset v₀ := by
    ext w
    simp only [Finset.mem_filter, Finset.mem_erase, Finset.mem_univ, and_true, mem_neighborFinset]
    exact ⟨fun ⟨_, h⟩ => Gmax.adj_symm h, fun h => ⟨(Gmax.ne_of_adj h).symm, Gmax.adj_symm h⟩⟩
  set A : Multiset ℕ := (Gmax.neighborFinset v₀).val.map (fun w => Gmax.degree w) with hAdef
  set B : Multiset ℕ :=
    ((Finset.univ.erase v₀).filter (fun w => ¬ Gmax.Adj w v₀)).val.map (fun w => Gmax.degree w)
    with hBdef
  set rest : Multiset ℕ := (Finset.univ.erase v₀).val.map (fun w => Gmax.degree w) with hrestdef
  have hAB : A + B = rest := by
    rw [hAdef, hBdef, hrestdef, ← hNfilter, Finset.filter_val, Finset.filter_val,
      ← Multiset.map_add, Multiset.filter_add_not]
  have hdom : ∀ a ∈ A, ∀ b ∈ B, b ≤ a := by
    intro a ha b hb
    rw [hAdef, Multiset.mem_map] at ha
    obtain ⟨wa, hwa, rfl⟩ := ha
    rw [hBdef, Multiset.mem_map] at hb
    obtain ⟨wb, hwb, rfl⟩ := hb
    rw [Finset.mem_val, mem_neighborFinset] at hwa
    rw [Finset.mem_val, Finset.mem_filter, Finset.mem_erase] at hwb
    exact hinv wb wa hwb.1.1 (fun h => hwb.2 (Gmax.adj_symm h)) hwa
  have hcard : A.card = Gmax.degree v₀ := by
    rw [hAdef, Multiset.card_map]; exact card_neighborFinset_eq_degree Gmax v₀
  obtain ⟨hAtop, hBdrop⟩ := multiset_top_split A B hdom
  rw [hAB, hcard] at hAtop hBdrop
  have hcons : (Gmax.degreeMultiset).sort (· ≥ ·)
      = Gmax.degree v₀ :: rest.sort (· ≥ ·) := by
    rw [SimpleGraph.degreeMultiset, hrestdef]; exact sortedDeg_cons Gmax v₀ hmax
  have hinduce : (Gmax.induce ({v₀}ᶜ : Set V)).degreeMultiset = A.map (· - 1) + B := by
    rw [degreeMultiset_induce_compl_split, ← hBdef, hAdef]
    congr 1
    exact (Multiset.map_map (· - 1) (fun w => Gmax.degree w) (Gmax.neighborFinset v₀).val).symm
  rw [hinduce, hcons]
  unfold havelHakimiStep
  simp only [List.splitAt_eq]
  rw [hAtop, hBdrop, Multiset.map_coe, Multiset.coe_add, Multiset.coe_eq_coe]
  exact (List.mergeSort_perm _ _).symm

open Classical in
/-- **Havel (1955) / Hakimi (1962): one reduction step preserves graphicality — native proof.**
If a descending-sorted degree list `d` is graphical, so is `havelHakimiStep d`. Proof: realise `d`
by a graph `G`; among all realisations with `G`'s degree function, pick one (`Gmax`) maximising the
adjacency weight of a maximum-degree vertex `v₀` (`exists_realization_noInversion`) — there `v₀`'s
neighbours carry the top-Δ degrees; deleting `v₀` then realises exactly `havelHakimiStep d`
(`induce_degreeMultiset_eq_havelHakimiStep`), which is graphical by `graphical_induce_compl`. This
is the classical 2-switch extremal argument, fully formalised — it discharges the
`havelHakimiStep_preserves_graphical` axiom. -/
theorem havelHakimiStep_preserves_graphical' (d : List ℕ) (hd : d.Pairwise (· ≥ ·))
    (hg : Graphical (d : Multiset ℕ)) :
    Graphical ((havelHakimiStep d : List ℕ) : Multiset ℕ) := by
  cases d with
  | nil => simpa [havelHakimiStep] using hg
  | cons d₀ rest =>
    obtain ⟨V, hFin, hDec, G, hAdj, hGd⟩ := hg
    rw [Subsingleton.elim hAdj (Classical.decRel G.Adj)] at hGd
    clear hAdj
    have hcardV : Fintype.card V = rest.length + 1 := by
      have h1 : Fintype.card V = Multiset.card G.degreeMultiset := by
        rw [SimpleGraph.degreeMultiset, Multiset.card_map]; rfl
      rw [h1, hGd, Multiset.coe_card, List.length_cons]
    haveI : Nonempty V := Fintype.card_pos_iff.mp (by rw [hcardV]; omega)
    obtain ⟨v₀, -, hv₀⟩ :=
      Finset.exists_max_image Finset.univ (fun w => G.degree w) Finset.univ_nonempty
    obtain ⟨Gmax, hdeg, hinv⟩ := exists_realization_noInversion G v₀
    have hmaxGmax : ∀ w, Gmax.degree w ≤ Gmax.degree v₀ := fun w => by
      rw [hdeg w, hdeg v₀]; exact hv₀ w (Finset.mem_univ w)
    have hGmaxeq : Gmax.degreeMultiset = G.degreeMultiset := by
      unfold SimpleGraph.degreeMultiset
      exact Multiset.map_congr rfl (fun x _ => hdeg x)
    have hsortd : (Gmax.degreeMultiset).sort (· ≥ ·) = d₀ :: rest := by
      rw [hGmaxeq, hGd]
      apply List.Perm.eq_of_sortedGE (Multiset.pairwise_sort _ _).sortedGE hd.sortedGE
      rw [← Multiset.coe_eq_coe, Multiset.sort_eq]
    have hcore := induce_degreeMultiset_eq_havelHakimiStep Gmax v₀ hmaxGmax hinv
    rw [hsortd] at hcore
    rw [← hcore]
    exact graphical_induce_compl Gmax v₀

end SimpleGraph
