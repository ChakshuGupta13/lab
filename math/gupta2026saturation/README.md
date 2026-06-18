# Sharp bounds between the saturation number and the harmonic index

Verification code for the paper *"Sharp bounds between the saturation number and
the harmonic index"* (Chakshu Gupta, Georgia Institute of Technology), which
refutes **Conjecture 4** of Davila, Brimkov, and Pepper, *"In Reverie Together:
Ten Years of Mathematical Discovery with a Machine Collaborator"*
([arXiv:2507.17780](https://arxiv.org/abs/2507.17780)) and bounds the two
invariants sharply in both directions.

**Paper**: [arXiv:2606.15761](https://arxiv.org/abs/2606.15761) (math.CO, June 2026).

## The conjecture

**Conjecture 4** (TxGraffiti, open since 2023): *if $G$ is a nontrivial
connected graph, then $\mu^\ast(G) \le H(G)$,* where $\mu^\ast(G)$ is the **saturation
number** (the minimum cardinality of a maximal matching) and $H(G) =
\sum_{uv \in E(G)} 2/(d(u)+d(v))$ is the **harmonic index**.

## The result

The conjecture is **false**, as first shown by Bıyıkoğlu (*MATCH Commun. Math.
Comput. Chem.* **96**(3):1097–1099, 2026). This paper complements that
refutation with a smallest counterexample, exact separation rates, sharp
two-sided bounds, and a conjecture for subquartic graphs.

- The friendship graph $F_4$ (four triangles sharing one hub, $n = 9$) has
  $\mu^\ast = 4 > 18/5 = H$. An exhaustive search confirms nine vertices is the
  smallest order admitting a counterexample (none exist for $n \le 8$).
- The harmonic index does not bound the saturation number up to **any**
  multiplicative constant: the family $G_{m,k} = K_{m,2k}$ plus a perfect
  matching on the $2k$-side has $\mu^\ast/H \to m+1$ as $k \to \infty$, which is
  unbounded over $m$. This family coincides with Bıyıkoğlu's construction.
- The conjecture fails even on **trees**. The subdivided star $S_k$ (a centre
  joined to $k$ paths of length two) is a counterexample for every $k \ge 5$;
  the smallest, $S_5$, has $n = 11$, is triangle-free and bipartite, and an
  exhaustive search confirms eleven is the smallest triangle-free order. Yet
  every nontrivial tree $T$ satisfies $\mu^\ast(T) < \frac{3}{2} H(T)$, with the
  constant $3/2$ best possible — so on trees the failure is bounded, in
  contrast to the unbounded windmill.
- The harmonic index does bound the saturation number **from below**: every
  graph with an edge satisfies $H(G) < 4\mu^\ast(G)$, the constant $4$ best
  possible (balanced double stars $D_k$ have $\mu^\ast = 1$ and $H \to 4$). On a
  nontrivial tree this closes a sharp two-sided band
  $\frac{1}{4} H(T) < \mu^\ast(T) < \frac{3}{2} H(T)$, both constants best
  possible ($D_k$ below, $S_k$ above).
- The conjecture **does** hold for every regular graph, where $H(G) = n/2 \ge
  \lfloor n/2 \rfloor \ge \mu^\ast(G)$.
- Among **subquartic** graphs (maximum degree at most four), an exhaustive
  search finds no counterexample on up to nine vertices; exactly six meet
  $\mu^\ast = H$ ($K_2$, $C_4$, $K_4$, $K_{3,3}$, $K_{4,4}$, and the subdivided
  star $S_4$). The paper conjectures $\mu^\ast \le H$ holds throughout this
  class.

All harmonic-index arithmetic uses `fractions.Fraction`, so every equality and
every violation is decided exactly, without floating-point rounding.

## Files

| File | What it verifies |
|------|------------------|
| [`code/graph_utils.py`](code/graph_utils.py) | Shared invariant computations (exact arithmetic) |
| [`code/verify_c4_exhaustive.py`](code/verify_c4_exhaustive.py) | Exhaustive check on all connected graphs $n \le 9$ |
| [`code/verify_friendship.py`](code/verify_friendship.py) | Friendship family $F_k$ closed forms and crossover |
| [`code/verify_unbounded.py`](code/verify_unbounded.py) | $G_{m,k}$ unbounded separation |
| [`code/verify_subdivided_star.py`](code/verify_subdivided_star.py) | Subdivided star $S_k$, and the exhaustive tree bound $\mu^\ast(T) < \frac{3}{2} H(T)$ for $n \le 16$ |
| [`code/verify_lower_bound.py`](code/verify_lower_bound.py) | Lower bound $H(G) < 4\mu^\ast(G)$ (exhaustive $n \le 8$), double-star sharpness, and the two-sided tree band for $n \le 16$ |
| [`code/verify_regular.py`](code/verify_regular.py) | Regular-graph lemma + Sumner equality |
| [`code/verify_subquartic.py`](code/verify_subquartic.py) | Subquartic ($\Delta \le 4$) enumeration $n \le 9$: no counterexample, and the six equality graphs |
| [`code/verify_n9_characterization.py`](code/verify_n9_characterization.py) | Characterization of all eight counterexamples at $n=9$ |
| [`code/verify_all.py`](code/verify_all.py) | Runner for all scripts (`--quick` skips the $n=9$ search) |

## The saturation-number identity

The code computes $\mu^\ast(G)$ two independent ways — directly (minimum maximal
matching) and via the identity $\mu^\ast(G) = i(L(G))$ (independent domination
number of the line graph) — and checks they agree on all 261,080 graphs with
$n \le 9$. That is an empirical cross-check, not a proof. A full pen-and-paper
proof of the identity for **every** finite simple graph is in
[`identity-proof.md`](identity-proof.md).

## Requirements

- Python 3.10+
- networkx ≥ 3.0
- nauty `geng` on `PATH` (Homebrew: `brew install nauty`)

## Usage

```bash
cd code

# Full verification (includes the ~80s n=9 exhaustive search)
python verify_all.py

# Quick mode (skips the ~80s n=9 search; keeps the n<=16 tree enumeration, ~12s)
python verify_all.py --quick
```
