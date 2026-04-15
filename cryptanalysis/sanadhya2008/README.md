# Collision Attacks on 22 and 24-Step SHA-256

Implementation of the collision attacks from:

> Sanadhya, S.K. & Sarkar, P. (2008). *New Collision Attacks Against Up To 24-step SHA-2*.
> LNCS vol. 5365, pp. 91–103.
> [DOI: 10.1007/978-3-540-89754-5_8](https://doi.org/10.1007/978-3-540-89754-5_8)
> [ePrint: 2008/271](https://eprint.iacr.org/2008/271)

## Attack model

Both attacks target the SHA-256 compression function reduced to 22 or 24 steps, starting from the standard IV. The attacker constructs both messages — there is no fixed input message. Each run produces a different colliding pair $(M, M')$ such that:

$$H_{22}(\text{IV}, M) = H_{22}(\text{IV}, M') \quad \text{and} \quad H_{24}(\text{IV}, M) = H_{24}(\text{IV}, M')$$

where $H_n$ denotes the SHA-256 compression function reduced to $n$ steps.

## Attacks

### 22-step collision (deterministic)

Uses a 9-step local collision at steps 7–15 (Column II of the SS local collision). The attack sets register values deterministically via message word selection, producing a collision in a single shot.

- **Complexity**: instant (deterministic, no search)
- **Differential**: $\Delta W_7 = +1$, $\Delta W_{15} = -1$

### 24-step collision (probabilistic)

Uses a 9-step local collision at steps 10–18 (Column I, $u=1$). Extends the 22-step attack with additional probabilistic conditions on $\sigma_1(W_{17})$ and $\sigma_1(W_{18})$.

- **Complexity**: ${\sim}2^{28.5}$ reduced-round evaluations (~30–120 seconds on modern hardware)
- **Differential**:

$$\Delta W_{10} = +1, \quad \Delta W_{11} = -1, \quad \Delta W_{12} = \mathtt{0x00006000}, \quad \Delta W_{13} = \mathtt{0xff006001}, \quad \Delta W_{17} = +1, \quad \Delta W_{18} = -1$$

Includes the guess-then-determine algorithm from Appendix B.1 for efficiently solving the non-linear constraint $D = \sigma_0(W_1) - W_1$.

## Build and run

Requires C++17. No external dependencies.

```bash
# Build
g++ -std=c++17 -O2 -Wall -Wextra -o 22_step code/22_step.cpp
g++ -std=c++17 -O2 -Wall -Wextra -o 24_step code/24_step.cpp

# Run
./22_step       # instant, prints collision
./24_step       # ~30-120s, prints collision
```

Both programs exit 0 on success (collision confirmed), 1 on failure.

## Example output (22-step)

```
22-step collision: CONFIRMED
```

The program prints both message blocks (W and W'), their differential, and the matching internal states after 22 compression steps.

## Citation

```bibtex
@inproceedings{Sanadhya2008new,
  author    = {Somitra Kumar Sanadhya and Palash Sarkar},
  title     = {New Collision Attacks Against Up To 24-step {SHA-2}},
  booktitle = {Progress in Cryptology -- INDOCRYPT 2008},
  series    = {LNCS},
  volume    = {5365},
  pages     = {91--103},
  year      = {2008},
  doi       = {10.1007/978-3-540-89754-5_8},
}
```

## License

MIT
