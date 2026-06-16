# Proof of the saturation-number / line-graph identity

This note proves the identity used as a computational cross-check in
`code/graph_utils.py` (`mu_star_linegraph` vs `mu_star_bruteforce`).
The identity is **not** proven in the paper itself — there it is cited to the
literature (Davila 2025) and the two code paths only confirm it empirically on
all 261,080 graphs with $n \le 9$. This is the mathematical proof that it holds
for **every** finite simple graph, which no amount of enumeration can establish.

## The theorem

For every finite simple graph $G$,
$$\mu^\ast(G) = i(L(G)),$$
where $\mu^\ast(G)$ is the minimum size of a **maximal matching** of $G$, and
$i(L(G))$ is the minimum size of a **maximal independent set** of the line graph
$L(G)$ (the independent domination number).

The whole proof rests on one structural fact: **the vertices of $L(G)$ are the
edges of $G$.** So a set $M \subseteq E(G)$ and a set $S \subseteq V(L(G))$ are
the *same kind of object* — a subset of edges. The claim is that "maximal
matching of $G$" and "maximal independent set of $L(G)$" pick out the **same
subsets**. Two lemmas establish that.

Recall the line-graph adjacency: for distinct edges $e, f$,
$$e \sim f \text{ in } L(G) \iff e \cap f \neq \emptyset \quad(\text{they share an endpoint}).$$

## Lemma A — matching $\iff$ independent set

$M \subseteq E(G)$ is a matching in $G$ $\iff$ $M$ is an independent set in $L(G)$.

**Proof.** Both sides say the same thing about every pair. $M$ is a matching
$\iff$ no two distinct $e, f \in M$ share an endpoint $\iff$ no two distinct
$e, f \in M$ satisfy $e \cap f \neq \emptyset$ $\iff$ no two distinct
$e, f \in M$ are adjacent in $L(G)$ $\iff$ $M$ is independent. $\square$

## Lemma B — maximal $\iff$ dominating

If $M$ is a matching (equivalently, independent), then $M$ is a **maximal**
matching of $G$ $\iff$ $M$ is a **dominating** set of $L(G)$.

**Proof.** Take any $f \in E(G) \setminus M$.

- ($\Rightarrow$, contrapositive) If $f$ has *no* neighbor in $M$, then $f$ is
  disjoint from every $e \in M$, so $M \cup \{f\}$ is still pairwise-disjoint —
  a matching strictly larger than $M$, contradicting maximality. Hence every
  $f \notin M$ has a neighbor in $M$: $M$ dominates $L(G)$.
- ($\Leftarrow$) If $M$ dominates, every $f \notin M$ is adjacent to some
  $e \in M$, i.e. $e \cap f \neq \emptyset$, so $M \cup \{f\}$ is not a matching.
  No edge extends $M$, so $M$ is maximal. $\square$

## Combine

A set is a **maximal independent set** of a graph iff it is independent *and*
dominating (standard: an independent set is maximal exactly when nothing outside
it is non-adjacent to all of it). Chaining the lemmas, for every
$M \subseteq E(G)$:
$$
M \text{ is a maximal matching of } G
\;\overset{\text{A}+\text{B}}{\iff}\;
M \text{ is independent and dominating in } L(G)
\iff
M \text{ is a maximal independent set of } L(G).
$$

So the family $\mathcal{M} = \{\text{maximal matchings of } G\}$ and the family
$\mathcal{S} = \{\text{maximal independent sets of } L(G)\}$ are **literally
equal as collections of subsets of $E(G)$** — the bijection is the identity map.
Taking the minimum cardinality over equal, nonempty, finite families:
$$\mu^\ast(G) = \min_{M \in \mathcal{M}} |M| = \min_{S \in \mathcal{S}} |S| = i(L(G)). \qquad\blacksquare$$

## Why this holds for *any* graph

This is the precise difference between the enumeration and a proof:

- **The enumeration** ($n \le 9$, 261,080 graphs) tested that two *programs* —
  `mu_star_linegraph` and `mu_star_bruteforce` — return the same number on each
  concrete input. That can never establish a $\forall$-statement: there are
  infinitely many finite graphs, and "it held on the ones I tried" is
  induction's enemy.
- **The proof** never instantiates $G$. Every step is a logical equivalence
  quantified over an *arbitrary* edge or pair of edges of an *arbitrary* graph.
  Lemma A quantifies over all pairs $\{e, f\}$; Lemma B over all $f \notin M$.
  Nowhere does $|V(G)|$, $|E(G)|$, connectivity, or any finite case split
  appear. Because the argument manipulates only the generic structure "a graph
  and its edges," its conclusion is valid for every graph that *has* that
  structure — i.e. all of them.

That universality is the entire reason a proof is worth more than the
cross-check: it converts "true on 261,080 instances" into "true, period."

## Edge cases a correct proof (and formalization) must not skip

- **No edges:** $E(G) = \emptyset$. The only matching is $\emptyset$, vacuously
  maximal, so $\mu^\ast = 0$; $L(G)$ has no vertices, $\emptyset$ is the only
  (vacuously dominating) maximal independent set, so $i = 0$. ✓ (the code
  special-cases this).
- **Isolated vertices:** contribute no edges, never appear in $L(G)$; correctly
  ignored by both sides.
- **Disconnected $G$:** the argument is purely local to edge incidences —
  connectivity is never used, so the identity holds for disconnected graphs too
  (more general than the paper needs).

## Lean formalization status (Mathlib `HEAD` 5450b53, 2026-06-14)

A machine-checked proof would be a natural follow-up. What
Mathlib provides and what must be added:

| Needed | In Mathlib? |
|---|---|
| `SimpleGraph.lineGraph` (on `G.edgeSet`) | ✅ yes (`Combinatorics/SimpleGraph/LineGraph.lean`) |
| `SimpleGraph.Subgraph.IsMatching` | ✅ yes (`Combinatorics/SimpleGraph/Matching.lean`, as a subgraph property) |
| `SimpleGraph.IsIndepSet` (independent set) | ✅ yes |
| **maximal** matching / saturation number | ❌ not defined |
| **dominating** set / independent domination number | ❌ not in Mathlib |

So the formalization must first **define** maximal-matching, dominating-set, and
independent-domination-number, then prove Lemmas A and B and the min-equality.
The math is exactly the two short lemmas above, but the glue between Mathlib's
*subgraph*-based matchings and the *edge-set*-based line graph is the main
engineering cost.
