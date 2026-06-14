# The saturation number is not bounded by the harmonic index

Verification code for the note *"The saturation number is not bounded by the
harmonic index"* (Chakshu Gupta, Georgia Institute of Technology), which refutes
**Conjecture 4** of Davila, Brimkov, and Pepper, *"In Reverie Together: Ten
Years of Mathematical Discovery with a Machine Collaborator"*
([arXiv:2507.17780](https://arxiv.org/abs/2507.17780)).

> The arXiv link for the note itself will be added here once it is announced.

## The conjecture

**Conjecture 4** (TxGraffiti, open since 2023): *if $G$ is a nontrivial
connected graph, then $\mu^*(G) \le H(G)$,* where $\mu^*(G)$ is the **saturation
number** (the minimum cardinality of a maximal matching) and $H(G) =
\sum_{uv \in E(G)} 2/(d(u)+d(v))$ is the **harmonic index**.

## The result

The conjecture is **false**.

- The friendship graph $F_4$ (four triangles sharing one hub, $n = 9$) has
  $\mu^* = 4 > 18/5 = H$. An exhaustive search confirms nine vertices is the
  smallest order admitting a counterexample (none exist for $n \le 8$).
- The harmonic index does not bound the saturation number up to **any**
  multiplicative constant: the family $G_{m,k} = K_{m,2k}$ plus a perfect
  matching on the $2k$-side has $\mu^*/H \to m+1$ as $k \to \infty$, which is
  unbounded over $m$.
- The conjecture **does** hold for every regular graph, where $H(G) = n/2 \ge
  \lfloor n/2 \rfloor \ge \mu^*(G)$.

All harmonic-index arithmetic uses `fractions.Fraction`, so every equality and
every violation is decided exactly, without floating-point rounding.

## Files

| File | What it verifies |
|------|------------------|
| [`code/graph_utils.py`](code/graph_utils.py) | Shared invariant computations (exact arithmetic) |
| [`code/verify_c4_exhaustive.py`](code/verify_c4_exhaustive.py) | Exhaustive check on all connected graphs $n \le 9$ |
| [`code/verify_friendship.py`](code/verify_friendship.py) | Friendship family $F_k$ closed forms and crossover |
| [`code/verify_unbounded.py`](code/verify_unbounded.py) | $G_{m,k}$ unbounded separation |
| [`code/verify_regular.py`](code/verify_regular.py) | Regular-graph lemma + Sumner equality |
| [`code/verify_n9_characterization.py`](code/verify_n9_characterization.py) | Characterization of all eight counterexamples at $n=9$ |
| [`code/verify_all.py`](code/verify_all.py) | Runner for all scripts (`--quick` skips the $n=9$ search) |

## The saturation-number identity

The code computes $\mu^*(G)$ two independent ways — directly (minimum maximal
matching) and via the identity $\mu^*(G) = i(L(G))$ (independent domination
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

# Quick mode (constructive proofs only, ~5s)
python verify_all.py --quick
```
