# Message Modification: Constraint Structure and Experimental Analysis

Analysis of how Wang-style message modification satisfies the forward
characteristic (Table 3, steps 21--29) in the Lamberger--Mendel 2011
46-step attack. Findings from twelve scratch experiments (mm1--mm12,
in scratch/) are consolidated below.

---

## 1. Algebraic structure of the step diff

The SHA-256 step function updates state $(A, B, C, D, E, F, G, H)$ as:

$$
A' = \Sigma_0(A) + \mathrm{Maj}(A,B,C) + \Sigma_1(E) + \mathrm{Ch}(E,F,G) + H + K_i + W_i
$$

$$
E' = D + \Sigma_1(E) + \mathrm{Ch}(E,F,G) + H + K_i + W_i
$$

$$
B' = A,\quad C' = B,\quad D' = C,\quad F' = E,\quad G' = F,\quad H' = G
$$

$W_i$ enters linearly in $A'$ and $E'$ only. Differencing paths 0 and 2:

$$
\Delta A' = C_A + \Delta W \qquad \text{where } C_A = \Delta\Sigma_0 + \Delta\mathrm{Maj} + \Delta\Sigma_1 + \Delta\mathrm{Ch} + \Delta H
$$

$$
\Delta E' = C_E + \Delta W \qquad \text{where } C_E = \Delta D + \Delta\Sigma_1 + \Delta\mathrm{Ch} + \Delta H
$$

Both $C_A$ and $C_E$ are determined entirely by the current state pair.
$W$ cancels from both. The six register-shift diffs
($\Delta B' = \Delta A$, $\Delta C' = \Delta B$, etc.) carry no freedom
at all.

**Consistency condition.** Requiring a single $\Delta W$ to satisfy both
target $\Delta A'$ and target $\Delta E'$ gives:

$$
\Delta A'^\star - \Delta E'^\star = C_A - C_E = \Delta\Sigma_0(A) + \Delta\mathrm{Maj}(A,B,C) - \Delta D
$$

This is necessary and sufficient, independent of $W$, and dependent on
absolute state values through the nonlinear $\Sigma_0$ and $\mathrm{Maj}$.

**Scope of $W$'s influence.** $W_i$ does not affect the state diff at
step $i{+}1$ (that diff is $W$-independent). It affects the absolute
values of $A_{i+1}$ and $E_{i+1}$, which determine the consistency
condition at step $i{+}2$ through $\Sigma_0$ and $\mathrm{Maj}$
evaluated at the next state. The relevant control horizon is therefore
two steps, not one.

---

## 2. Experimental results

### 2.1 Diff is $W$-independent (mm1, mm2, mm3, mm4)

The state diff at step $i{+}1$ is fully determined by the state pair at
step $i$: any $W_i$ (shared by both paths) produces the same diff.
This was verified with random $W$ values on the Table 1 instance (mm2:
all pass for steps 21--30).

However, the diff at step $i{+}2$ depends on the absolute state at step
$i{+}1$, which $W_i$ controls. Random $W$ through steps 22--29 (mm3)
diverges from the characteristic by step 23. Different absolute states
with the same diff produce different output diffs (mm4), confirming that
the nonlinear functions' differential behavior is
operating-point-dependent.

### 2.2 The consistency condition in practice (mm7, mm8)

Algebraic derivation (mm7) confirmed
$\Delta A' = C_A + \Delta W$ and $\Delta E' = C_E + \Delta W$. At the
Table 1 meeting point, the consistency condition for step 22 holds
(both paths require $\Delta W = \mathtt{0xc0000000}$).

Sweeping $W_{21}$ over all $2^{32}$ values (mm8): exactly $2^{26}$
($= 2^{-6}$ of the space) produce a state at step 22 where the
consistency condition for step 23 holds. Table 1's
$W_{21} = \mathtt{0x50accb03}$ is among them.

### 2.3 Greedy selection fails (mm9)

Chaining greedily (pick first consistent $W$ at each step): the first
hit $W_{21} = \mathtt{0x107bd2e8}$ leads to a state at step 22 where
zero out of $2^{32}$ $W_{22}$ values satisfy step-24 consistency. The
chain is dead at step 22.

### 2.4 Table 1 is globally consistent (mm10)

The actual Table 1 instance satisfies the word-level consistency
condition at all 45 steps. Every $\Delta W_{\text{actual}}$ equals
$\Delta W_{\text{required}}$. The message was constructed so the
consistency chain holds throughout.

### 2.5 Table 1's $W_{21}$ enables step 22 (mm11)

Using the Table 1 $W_{21}$ (not the greedy first-hit), $2^{28.17}$
($\approx 2^{-3.83}$ of the space) $W_{22}$ values produce step-24
consistency. Table 1's $W_{22} = \mathtt{0x41247d3c}$ is among them.
The greedy first-hit $W_{21}$ produces a state where nothing works;
the Table 1 $W_{21}$ produces one where roughly 1 in 14 $W_{22}$
values work.

### 2.6 Per-step consistency cost (mm12)

At each step 21--28, using the actual Table 1 state, the fraction of
$W_i$ values satisfying the consistency condition at step $i{+}2$:

| Step | Consistency cost | Table 3 conditions |
|------|-----------------|-------------------|
| 21   | $2^{-6}$        | 23 bits           |
| 22   | $2^{-3.83}$     | 12 bits           |
| 23   | $2^{-1}$        | 10 bits           |
| 24   | $2^{-1}$        | 7 bits            |
| 25   | $2^{0}$ (free)  | 4 bits            |
| 26   | $2^{0}$ (free)  | 4 bits            |
| 27   | $2^{0}$ (free)  | 4 bits            |
| 28   | $2^{0}$ (free)  | 1 bit             |

Total consistency cost: $\sim 2^{-12}$. The remaining $\sim 54$ of the
66 Table 3 conditions are bit-level constraints on absolute register
values, satisfied by Wang-style direct construction. The state diff at
step 30 matches the target exactly (zero everywhere except
$H = \mathtt{0xc0000000}$).

---

## 3. Two-layer decomposition

The 66 signed-bit conditions in Table 3 (steps 21--29) decompose into
two structurally distinct layers:

1. **Consistency layer** ($\sim 12$ bits total). Ensures
   $\Delta A' - \Delta E'$ propagates correctly across steps.
   Constrains the nonlinear interaction of $\Sigma_0$ and
   $\mathrm{Maj}$ with the absolute state. Costs are front-loaded:
   $\sim 6$ bits at step 21, $\sim 4$ at step 22, $\sim 1$ at steps
   23--24, zero beyond.

2. **Bit-condition layer**. Ensures $\mathrm{Ch}$ and $\mathrm{Maj}$
   produce the specified signed-bit output at each step. These are
   direct constraints on individual bits of registers, satisfied
   constructively by setting those bits and computing $W$ from the
   step equation.

Wang-style message modification satisfies both layers simultaneously:
at each step, the 32 bits of $A_{i+1}$ provide enough freedom to cover
both the bit conditions and the consistency requirement. Piece 3b
(below) quantifies the exact bit budgets.

---

## 3b. Bit condition census (Piece 3b)

Derived from the Table 1 absolute states by computing modular and XOR
diffs at each register, per step (scratch_bit_conds.cpp).

### State-level constraints per step

Total XOR-flipping bits across all 8 registers, and the outgoing $A$
constraint (bits of $A_{i+1}$ fixed by the characteristic):

| Step | Total constrained bits | $\Delta A_{\text{out}}$ bits | Free $A$ bits | Ch constraints | Maj constraints |
|------|----------------------|--------------------------|---------------|----------------|----------------|
| 21   | 31                   | 0                        | 32            | 6              | 0              |
| 22   | 13                   | 0                        | 32            | 5              | 0              |
| 23   | 12                   | 1                        | 31            | 3              | 0              |
| 24   |  7                   | 0                        | 32            | 3              | 0              |
| 25   |  4                   | 0                        | 32            | 3              | 0              |
| 26   |  4                   | 0                        | 32            | 3              | 0              |
| 27   |  4                   | 0                        | 32            | 1              | 0              |
| 28   |  1                   | 0                        | 32            | 1              | 0              |
| 29   |  1                   | 0                        | 32            | 1              | 0              |
| 30   |  1                   | 0                        | 32            | 0              | 0              |

### Key observations

1. **$\Delta A_{\text{out}}$ is nearly always zero.** Only step 23
   constrains 1 bit of the outgoing $A$ ($\Delta A_{24} =
   \mathtt{0xc0000000}$, bit 30 must be 1 in path 0). All other steps
   produce $\Delta A = 0$, leaving all 32 bits free for the Wang
   modifier.

2. **Most constrained bits are in B--H registers.** These are register
   shifts from prior steps, inherited automatically. The Wang modifier
   does not set them; they are consequences of earlier $A$ and $E$
   choices.

3. **$\mathrm{Ch}$ and $\mathrm{Maj}$ produce zero modular diff at
   every step.** Both $\Delta\mathrm{Ch} = 0$ and
   $\Delta\mathrm{Maj} = 0$ throughout steps 21--30. The boolean
   functions must produce no net difference, which constrains specific
   bits of $E$ (for $\mathrm{Ch}$) but leaves most bits free.
   $\mathrm{Maj}$ imposes zero constraints because $\Delta A = 0$ at
   almost every step.

4. **Modular vs. XOR mismatch.** At step 21,
   $E$: $\Delta_{\text{mod}} = \mathtt{fffbef00}$ but
   $\Delta_{\text{xor}} = \mathtt{00041100}$ (only 3 bits actually
   flip, the rest are carries). The signed-bit representation is
   sparse despite the large modular diff.

5. **$\Sigma_0(A)$ and $\Sigma_1(E)$ diffs.** $\Delta\Sigma_0 = 0$
   whenever $\Delta A = 0$ (steps 21--22, 24--30).
   $\Delta\Sigma_1 = 0$ whenever $\Delta E = 0$ (steps 22--23,
   25--26, 28--30). Where nonzero, some have carries (step 27:
   $\Delta\Sigma_1$ mod $\neq$ xor) but the modular arithmetic
   still works because the consistency condition is checked
   algebraically.

### Revised constraint budget

The earlier estimate of "$\sim 54$ bit conditions" was speculative.
The actual picture:

- The outgoing $A$ is constrained at only 1 step (step 23, 1 bit).
- The $\mathrm{Ch}$ interaction constrains 1--6 bits per step, but
  these constrain input-state bits ($E$, $F$, $G$), not the $A$ being
  chosen. Since $\Delta E_{i+1} = \Delta D_i + \Delta W$, and
  $\Delta D$ is a register shift, the $E$ constraint is automatically
  satisfied once $W$ is chosen.
- The Wang modifier's real budget at each step is: 32 bits of $A$,
  minus the consistency cost (from §2), minus at most 1 outgoing
  $\Delta A$ bit. This leaves 25--31 free bits at every step.

The bit-condition layer is far cheaper than initially estimated because
the diagonal-absorption structure concentrates all constraints in the
register-shift chain (B--H), not in $A$.

---

## 4. Pitfall catalog

Six misconceptions encountered during experimentation:

| # | Misconception | Correction |
|---|--------------|------------|
| 1 | Any $W$ works for message modification. | Any $W$ preserves the current diff ($W$-independent). The constraint is on which $W$ values produce an absolute state where the NEXT step's consistency condition holds. |
| 2 | Same diff implies same output diff. | False for nonlinear functions. $\Sigma_0$, $\mathrm{Maj}$, $\mathrm{Ch}$, $\Sigma_1$ all depend on operating point, not just diff. |
| 3 | Search for $W$ giving the correct step-$(i{+}1)$ diff. | Wrong target: the diff is always correct. Search for $W$ satisfying the step-$(i{+}2)$ consistency condition. |
| 4 | Greedy per-step $W$ selection suffices. | Catastrophically wrong. First-hit $W_{21}$ led to zero viable $W_{22}$ at step 22. Steps are coupled through nonlinear functions. |
| 5 | An inconsistency result means the paper is wrong. | The Table 1 instance is consistent at all 45 steps. The apparent inconsistency was from computing required $\Delta W$ instead of checking whether the state already satisfies the consistency condition. |
| 6 | Message modification is per-step independent. | $W_i$ sets the absolute state at step $i{+}1$, which determines whether consistency holds at step $i{+}2$ and beyond. Joint selection across steps is required. |

---

## 5. Generic vs. instance-specific properties

**Generic** (holds for any characteristic with this step structure):

- The consistency condition
  $\Delta A' - \Delta E' = \Delta\Sigma_0 + \Delta\mathrm{Maj} - \Delta D$
  is always $W$-independent. This is an algebraic identity of the step
  function.
- Six of eight output diffs are register shifts with zero freedom.
- $W$ provides exactly one degree of freedom per step (controlling
  $\Delta A'$ and $\Delta E'$ jointly through $\Delta W$).
- Step coupling through nonlinear functions: the choice of $W_i$ at
  step $i$ constrains what is achievable at step $i{+}2$ and beyond.
  Greedy selection generically fails.
- The two-layer decomposition (consistency + bit conditions) applies to
  any diagonal-absorption characteristic.
- Later steps in the active range tend toward trivial consistency as
  the state diff simplifies (fewer active bits in $\Sigma_0$ and
  $\mathrm{Maj}$).

**Instance-specific** (depends on the particular Table 1 message/IV):

- The concrete $W$ values, absolute states, and per-step hit rates.
- Whether a candidate characteristic is jointly solvable for a given
  $\gamma$ and initial state.

---

## 6. Implications for characteristic search (Piece 2b)

A candidate forward characteristic must satisfy the following:

1. **Register-shift consistency.**
   $d_{i+1}[1{..}3, 5{..}7] = d_i[0{..}2, 4{..}6]$ at every step.
   Mandatory; violated characteristics are immediately infeasible.

2. **Consistency-chain budget.** At each step, the number of bit
   conditions plus the consistency cost must not exceed 32. The
   consistency cost depends on the specific diff pattern and drops
   rapidly for diagonal-absorption structures.

3. **Empirical feasibility test.** For a candidate characteristic:
   pick a random state at step 21 with the correct $\gamma$ diff, run
   mm12-style $2^{32}$ sweeps at each step to measure hit rates. If
   the total consistency cost is comparable to $\sim 2^{-12}$ and no
   step has zero hits, the characteristic is viable. A zero-hit step
   indicates the consistency condition is unsatisfiable for that diff
   structure.

4. **Diagonal absorption helps.** The Table 3 pattern ($B$-chain
   shifts through $C$, $D$; $E$-chain shifts through $F$, $G$, $H$;
   all absorbed by step 30) ensures the register-shift conditions are
   trivially satisfied. New characteristics should preserve this
   structure.

5. **Sign-change freedom.** The paper gains a factor of $2^{10}$ from
   allowing sign variants ($H_0$, $H_3$, step-1 inputs). Candidate
   characteristics should preserve this multiplicity.

---

## 7. Implementation: Piece 3 (msg_modify.cpp)

Given state pair $(s_0, s_2)$ at step 21 with
$s_2 - s_0 = \gamma$, construct $W_{21}, \ldots, W_{29}$ satisfying the
characteristic:

> **For** $i = 21$ **to** $29$:
> 1. Check consistency: $C_A - C_E = \Delta A'^\star - \Delta E'^\star$.
>    If not, return failure (retry with different random choices).
> 2. Compute $\Delta W = \Delta A'^\star_{i+1} - C_A$.
> 3. Choose random $A_{i+1}$ satisfying next-step consistency (one-step
>    lookahead). If $i = 29$, any $A$ works.
> 4. Compute $W^{(0)}_i = \texttt{W\_to\_set\_A}(s_0, i, A_{i+1})$.
>    Set $W^{(2)}_i = W^{(0)}_i + \Delta W$.
> 5. Advance both states.

### Implementation findings

1. **Greedy one-step-ahead lookahead works.** The mm9 result ("greedy
   fails") referred to greedy selection *without* consistency checking.
   Greedy *with* next-step consistency checking succeeds at
   $\sim 2^{-11}$ per attempt on random states.

2. **Retry loop required.** Lookahead does not guarantee success;
   the Table 1 instance required 4 attempts. The outer loop simply
   retries with fresh random $A$ choices.

3. **$\Delta W$ is zero at steps 22--29.** For the Table 3
   characteristic, only step 21 has nonzero $\Delta W$
   ($\mathtt{0xc0000000}$). This is a consequence of the
   diagonal-absorption structure: the state diff entering steps
   22--29 only involves register shifts, not $A$ or $E$.

4. **Absolute $W$ values are not unique.** Different random $A$
   choices yield different $W$ values, but the state diffs at every
   step match the characteristic exactly. The Table 1 $W$ values are
   one valid solution among many.

5. **Measured success rate: $\sim 2^{-10.9}$** (34/65536 on random
   states). Consistent with the $\sim 2^{-12}$ total consistency cost
   from §2, since the random $A$ search at each step partially
   compensates for the per-step cost.

---

## 8. Scratch experiment index

All files in scratch/:

| File | Label | Result |
|------|-------|--------|
| scratch_msg_mod.cpp   | mm1  | $W$ controls only $A'$ and $E'$; $\Delta A' - \Delta E'$ is $W$-independent |
| scratch_msg_mod2.cpp  | mm2  | State diff is fully $W$-independent (random-$W$ verification) |
| scratch_msg_mod3.cpp  | mm3  | Random $W$ preserves step-22 diff but diverges at step 23+ |
| scratch_msg_mod4.cpp  | mm4  | Output diff depends on absolute state, not just input diff |
| scratch_msg_mod5.cpp  | mm5  | $0/2^{20}$ $W$ match at step 22 (wrong question: diff was already correct) |
| scratch_msg_mod6.cpp  | mm6  | Greedy $2^{24}$ sweep: 100% at step 21, 0% at step 22 (wrong question) |
| scratch_msg_mod7.cpp  | mm7  | Analytical: $\Delta A' = C_A + \Delta W$; consistency condition derived |
| scratch_msg_mod8.cpp  | mm8  | $2^{-6}$ of $W_{21}$ satisfy step-23 consistency; Table 1 $W$ included |
| scratch_msg_mod9.cpp  | mm9  | Greedy chain: first-hit $W_{21}$ dead-ends at step 22 ($0/2^{32}$) |
| scratch_msg_mod10.cpp | mm10 | Table 1 satisfies consistency at all 45 steps |
| scratch_msg_mod11.cpp | mm11 | With Table 1 $W_{21}$: $2^{-3.83}$ of $W_{22}$ work |
| scratch_msg_mod12.cpp | mm12 | Per-step hit rates: $-6, -3.83, -1, -1, 0, 0, 0, 0$ |
| scratch_bit_conds.cpp | bc1  | Bit condition census: per-step XOR flips, $\Delta A_{\text{out}}$, Ch/Maj constraints, free-bit counts |
| scratch_wang_mod.cpp  | wm1  | Wang-style modifier prototype: reproduces exact Table 1 $W$ values when given Table 1 $A$; verified consistency/dW at every step |
| scratch_fwd_arch.cpp | fa1 | Forward architecture tests A--J: Table 1 verification, Eqs 21--22, expansion linearity |
| scratch_exp_linearity.cpp | el1 | Cascade analysis: random M, per-step $\Delta^2 W$ survival (0 at step 25) |
| scratch_cascade_map.cpp | cm1 | Algebraic cascade mapping: identifies which $\Delta^2\sigma$ terms are CTRL vs FIXED |
| scratch_density.cpp | dn1 | Per-step $\Delta^2\sigma$ density against Table 1 diffs; confirms only 3 critical inputs |
| scratch_joint_density.cpp | jd1 | Exhaustive 2^{32} joint $\Delta^2\sigma_1 \wedge \Delta^2\sigma_0$ density at W[21] and W[30] |

---

## 9. Expansion linearity: the resolved picture

### 9.1 Background

The second-order collision requires the message expansion to be
"second-order linear": $\Delta^2 W[i] = W_{12}[i] - W_1[i] - W_2[i] + W_0[i] = 0$
for $i = 0, \ldots, 45$.

For $i \leq 15$ this is automatic (message words are additive).
For $i \geq 16$ the recurrence gives:

$$
\Delta^2 W[i] = \Delta^2\sigma_1(W[i{-}2]) + \Delta^2 W[i{-}7] + \Delta^2\sigma_0(W[i{-}15]) + \Delta^2 W[i{-}16]
$$

A $\Delta^2\sigma$ term is zero whenever EITHER input diff is zero:
$dW_{a_1}[j] = 0$ or $dW_{a_2}[j] = 0$ forces
$\Delta^2\sigma(W_0[j], dW_{a_1}[j], dW_{a_2}[j]) = 0$.

### 9.2 Why the cascade appeared intractable

Earlier experiments (el1, Parts 1--3) measured expansion linearity for
RANDOM $M$ and found 0 survivors out of $2^{20}$ already at step 25 (even
conditioned on Eqs 21--22). This was because random $M$ does not satisfy
the first-order $a_2$ expansion characteristic: $dW_{a_2}[22\ldots29]$ and
$dW_{a_2}[31\ldots44]$ are generically nonzero, creating dozens of non-trivial
$\Delta^2\sigma$ terms. The cascade of conditions was overwhelming.

### 9.3 The key observation: $dW_{a_2}$ sparsity

For the Table 1 instance (dn1), the first-order $a_2$ expansion diffs are:

| Steps | $dW_{a_2}$ | Reason |
|-------|------------|--------|
| 0--15 | dense (all nonzero) | input difference $dM_{a_2}$ |
| 16--20 | nonzero | schedule expansion of dense input |
| 21 | 0xc0000000 ($= -2^{30}$) | characteristic design |
| **22--29** | **0** | characteristic design (linearized model) |
| 30 | 0x40000000 ($= +2^{30}$) | characteristic design |
| **31--44** | **0** | characteristic design (linearized model) |
| 45 | 0x087ff000 | characteristic design |

Since $dW_{a_1}[i]$ is nonzero only for $i \in \{6\} \cup [21, 45]$,
the condition "$dW_{a_1} \neq 0$ AND $dW_{a_2} \neq 0$" restricts
the non-trivial $\Delta^2\sigma$ terms to inputs at steps
$\{6, 21, 30, 45\}$ only.

### 9.4 The three critical inputs

Tracing the recurrence (cm1), every non-trivial $\Delta^2\sigma$ arrives from
one of three independent inputs:

| Input | Feeds into | Type | Exact joint density ($2^{32}$ scan) |
|-------|-----------|------|-------------------------------------|
| $W_0[6] = M_0[6]$ | $\sigma_0$ at step 21 | FIXED (Eq 22) | $2^{-3.12}$ (σ0 only) |
| $W_0[21]$ | $\sigma_1$ at step 23, $\sigma_0$ at step 36 | CTRL (modifiable) | $2^{-2.19}$ (joint $\sigma_1 \wedge \sigma_0$) |
| $W_0[30]$ | $\sigma_1$ at step 32, $\sigma_0$ at step 45 | FIXED (probabilistic) | $2^{-11.62}$ (joint $\sigma_1 \wedge \sigma_0$) |

"CTRL" means $W_0[21]$ is one of the 9 modified words ($W_0[21\ldots29]$);
message modification can search over $A_0[22]$ values until the joint
condition is met. Cost: $\approx 2^{2.19}$ trials per step, absorbed into
the modification loop.

"FIXED" means $W_0[30]$ is determined by the schedule from $M_0[0\ldots15]$.
No message modification can change it. The density $2^{-11.62}$ must be paid
probabilistically.

### 9.5 Revised complexity accounting

$$
\underbrace{2^{40}}_{\text{backward}} \;\times\;
\underbrace{2^{6}}_{\text{Eqs 21--22}} \;\times\;
\underbrace{2^{11.62}}_{\text{W[30] expansion}} \;\approx\; 2^{57.6}
$$

The paper quotes $2^{46}$, accounting only for the backward characteristic
($2^{40}$) plus Eqs 21--22 ($2^6$). The $W[30]$ joint condition
($2^{11.62}$) is not explicitly mentioned but offsets partially against
the "additional characteristics" factor ($2^{10}$) and the observation that
"the complexity will be lower in practice."

The net complexity $\approx 2^{47.6}$ (using the $2^{10}$ additional-chars
credit) is still well below the generic $2^{85}$.

### 9.6 Corrected cascade summary (all 46 steps)

| Step range | $\Delta^2 W$ | Notes |
|------------|-------------|-------|
| 0--15 | 0 (trivial) | message words additive |
| 16--20 | 0 (trivial) | all $dW_{a_1} = 0$ in these $\sigma$ inputs |
| 21 | $= \Delta^2\sigma_0(M[6])$ | Eq 22; conditional on Eq 21 |
| 22 | 0 (trivial) | $\sigma_1$ input W[20]: $dW_{a_1}[20]=0$; $\sigma_0$ input W[7]: $dW_{a_1}[7]=0$ |
| 23 | $= \Delta^2\sigma_1(W[21])$ | CTRL; choose $W_0[21]$ |
| 24--31 | 0 (trivial) | $\sigma_1$ inputs W[22--29]: all have $dW_{a_2}=0$ |
| 32 | $= \Delta^2\sigma_1(W[30])$ | FIXED; $dW_{a_1}[30] \neq 0$, $dW_{a_2}[30] = 2^{30}$ |
| 33--35 | 0 | $\sigma_1$ inputs W[31--33]: $dW_{a_2}=0$; $\sigma_0$ inputs below 21 |
| 36 | $\Delta^2\sigma_0(W[21])$ | CTRL (same input as step 23) |
| 37--44 | 0 | $\sigma_0$ inputs W[22--29]: $dW_{a_2}=0$; $\sigma_1$ inputs: $dW_{a_2}=0$ |
| 45 | $\Delta^2\sigma_0(W[30])$ | FIXED (same input as step 32) |

### 9.7 Implication for implementation

1. The expansion linearity is NOT a separate problem. It reduces to 3 point
   conditions, two of which are already accounted for (Eqs 21--22),
   one of which is CTRL (absorbed into message modification), and one of
   which is FIXED (adds $\approx 2^{11.6}$ to the search, offset by
   additional characteristics).

2. The msg_modify.cpp approach of choosing $W[21\ldots29]$ via Wang-style
   modification remains correct in principle. The ADDITIONAL requirement
   is: when choosing $W_0[21]$, ensure $\Delta^2\sigma_1(W_0[21]) = 0$ AND
   $\Delta^2\sigma_0(W_0[21]) = 0$ (density $2^{-2.19}$, roughly 1 in 4.6).

3. The schedule-inconsistency problem from Piece 4 is separate: once the
   correct $W[21\ldots29]$ values are known, we must still find
   $M[0\ldots15]$ whose schedule produces them. This is a nonlinear system
   of 9 equations in 16 unknowns with 7 surplus degrees of freedom.

4. The $W_0[30]$ condition requires filtering: iterate the outer loop
   (random $M$ satisfying Eqs 21--22) until $W_0[30]$ also lands in the
   $\sim 2^{-11.6}$ solution set. This is the dominant expansion cost.
