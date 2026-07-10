# Self-Improving Instructions Pipeline

A reference implementation of a pipeline that lets an editor-embedded AI assistant
improve its own configuration — instruction files, prompts, custom agents, skills —
from a single user's implicit corrections, with no benchmark, reward, or ground-truth
oracle.

It observes live **GitHub Copilot** sessions in **VS Code**, detects recurring
friction, asks a frontier model to draft an instruction edit, routes that edit through
the user for approval, applies it as a single revertable commit, and then attempts to
measure whether the friction dropped.

Developed alongside a write-up of a two-month, single-user deployment. The design
question it explores: without a benchmark or oracle, can such a loop tell whether its
own edits helped? In this deployment the **measuring half of the loop, not the
proposing half, was the binding constraint** — detecting friction and drafting repairs
proved reliable; confirming that a repair helped, at single-user volume with no
counterfactual, did not.

## How it works

A six-stage closed loop over live sessions:

1. **Observe** — two editor hooks (`hook_post_tool_use.py`, `hook_user_prompt_submit.py`)
   capture tool calls, their errors, and a snapshot of each turn; the assistant's own
   in-transcript self-signals are read from the snapshots.
2. **Detect** — six structural detectors (`detectors.py`) turn the event stream into
   per-pattern time series:

   | Detector | Signal it captures |
   |---|---|
   | Intra-session self-correction | the assistant reverses its own action within a session |
   | Cross-session user correction | a user correction recurs across sessions |
   | Tool-error recurrence | the same tool error recurs |
   | Rule-on-disk correction | corrections persist despite a matching rule on disk |
   | Self-signal recurrence | the assistant's own friction tag recurs |
   | User steering | the user interrupts the assistant mid-response |

3. **Mine (L2)** — a semantic miner (`pattern_miner.py`, `agents/Learning-Miner.agent.md`)
   fires once three new self-signals accumulate and classifies latent friction the
   structural rules miss.
4. **Author** — once a pattern matures (a day has passed, or enough new observations
   have accumulated), a frontier model (`lm_authorer.py`,
   `agents/Learning-Authorer.agent.md`) drafts a repair in one of three tiers: an
   ephemeral nudge, an instruction-file edit, or a new hook or sub-agent.
5. **Gate and apply** — every proposal is surfaced to the user (apply / dismiss /
   defer); on approval `apply_proposal.py` lands the edit as a single revertable
   commit. It never touches the user's mandatory files or unrelated work, and never
   applies or reverts a change without approval.
6. **Measure** — `watch_effectiveness.py` re-reads the pattern's rate over the first
   `K` matching post-fix sessions in the same workspace and compares it to the pre-fix
   rate, returning `VALIDATED`, `SUGGEST_REVERT`, or `INSUFFICIENT_DATA`.

## Models

The Python layer is model-agnostic: it passes no model flag to the Copilot CLI and never
inspects the deployed model. The model choice lives in the two agent files' frontmatter,
invoked headlessly through the **GitHub Copilot CLI** (version 1.0.59). The reported
deployment ran **Claude Sonnet 4.5**; the agents as shipped here pin **Claude Opus 4.8**.
Repointing to another Copilot model is a one-line frontmatter edit.

## Layout

```
scripts/   the pipeline: detectors, the two hooks, the L2 miner, the authorer,
           apply/dismiss, effectiveness tracking, schema, template resolver
agents/    the two Copilot agents: Learning-Miner (L2), Learning-Authorer
```

## Caveats

This is a **reference snapshot**, not a turnkey package. It assumes the author's
research-monorepo layout (a `domains/` tree and a `~/.copilot/` state directory) and a
specific Copilot CLI version. Paths and guardrails reflect that environment. The live
event store and the deployment data are **not** included — they are single-user private
data.

## License

CC-BY-4.0 (repository root `LICENSE`).
