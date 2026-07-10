#!/usr/bin/env python3
"""
template_resolver.py — expand a validated `template` block into concrete
`proposal.operations` for apply_proposal.py.

Called by lm_authorer.py AFTER validate_output() confirms the JSON shape but
BEFORE the proposal is written to staging. See
.github/learning/integration-design.md Decision 5.

Design contract:
  - resolve(template) is DETERMINISTIC for given params. Same params → same ops.
  - Output ops conform to apply_proposal.py's op shape: {"op","path","content"}.
  - All paths land under .github/ (apply_proposal.py enforces this too).
  - Per-instantiation directory naming: .github/scripts/templates/<id>__<pid8>/
    for T1/T3 (per-friction); .github/scripts/templates/T2/ for T2 (workspace
    singleton, Decision 11). pid8 is supplied by the caller via template params
    is NOT — the caller passes pid separately to resolve().

This module imports nothing from the live pipeline and makes no LM call and no
DB access — it is pure params → ops expansion, unit-testable in isolation.

NOTE: T2 is workspace-wide infrastructure (Decision 11) and is deployed
out-of-band (user-triggered migration), NOT via the Authorer proposal flow.
resolve_T2 is provided for completeness/symmetry and for the rare case where a
T2 corpus registration is proposed as an operation, but the primary T2
deployment path does not go through this resolver.
"""

from __future__ import annotations

import json
from typing import Any


# --- runtime logging snippet injected into every generated instantiation -----
# The measurement plan (integration-design.md §3) requires each instantiation
# to append structured runtime logs the watcher reads. Each resolver embeds a
# tiny fail-open logger so the writer exists from day one (Adversary WRONG #4).
_LOGGER_SNIPPET = '''
def _log(event: dict) -> None:
    """Fail-open append of one JSON line to the instantiation's log.jsonl.

    Never raises: measurement logging must not break the mechanism. O(1) single
    line write so the T1 hot-path (UserPromptSubmit) stays within deadline.
    """
    import json as _json, time as _time
    from pathlib import Path as _Path
    try:
        rec = {"ts": int(_time.time()), **event}
        p = _Path(__file__).resolve().parent / "log.jsonl"
        with p.open("a", encoding="utf-8") as fh:
            fh.write(_json.dumps(rec, sort_keys=True) + "\\n")
    except Exception:
        pass
'''


def _pid8(pid: str) -> str:
    """First 8 chars of a pattern pid for directory naming (Decision 3)."""
    return (pid or "nopid000")[:8]


def _t1_instance_dir(pid: str) -> str:
    return f".github/scripts/templates/T1__{_pid8(pid)}"


def _t3_instance_dir(pid: str) -> str:
    return f".github/scripts/templates/T3__{_pid8(pid)}"


def resolve_T1(params: dict[str, Any], pid: str) -> list[dict]:
    """Expand T1 (retrieve-then-gate) params into operations.

    Emits a per-friction instantiation directory with a config.json capturing
    the slot values, plus a companion instruction file describing the gate.
    The actual retrieve/gate logic is referenced from the reference impl at
    .github/mechanism-templates/T1/ (executable documentation); the per-friction
    copy carries only the baked-in parameters + the runtime logger.
    """
    d = _t1_instance_dir(pid)
    config = {
        "template": "T1",
        "pid": pid,
        "trigger": params["trigger"],
        "corpus_globs": params["corpus_globs"],
        "sufficiency_check": params["sufficiency_check"],
        "max_iterations": params["max_iterations"],
        "fallback_action": params["fallback_action"],
        "scope": params["scope"],
    }
    ops = [
        {
            "op": "create",
            "path": f"{d}/config.json",
            "content": json.dumps(config, indent=2, sort_keys=True) + "\n",
        },
        {
            "op": "create",
            "path": f"{d}/logger.py",
            "content": '"""Runtime logger for this T1 instantiation."""\n' + _LOGGER_SNIPPET,
        },
        {
            "op": "create",
            "path": f".github/instructions/T1__{_pid8(pid)}.instructions.md",
            "content": _t1_instruction_md(config),
        },
    ]
    return ops


def resolve_T2(params: dict[str, Any], pid: str) -> list[dict]:
    """Expand T2 (dependency-tracked memoization) params into operations.

    T2 is a workspace singleton (Decision 11): rather than a per-friction dir,
    a T2 proposal registers a corpus into the shared
    .github/scripts/templates/T2/corpora.json. This resolver emits a single
    operation that the apply step interprets as a corpora-registration. Because
    apply_proposal.py only does file create/edit/delete, the registration is
    expressed as an edit to corpora.json — but since deterministic merge of JSON
    is fragile through a text-diff apply, the primary T2 deployment path is the
    user-triggered migration script, NOT this resolver. This function exists for
    symmetry and returns a config fragment the migration script can consume.
    """
    fragment = {
        "id": params.get("scope", "unnamed-corpus"),
        "glob": params["corpus_glob"],
        "required_fields": params["required_fields"],
        "status_enum": params["status_enum"],
        "supersession_field": params["supersession_field"],
        "lint_command": params["lint_command"],
        "enforce": False,
        "note": "Forward-only: lint reports violations but does not block.",
    }
    return [
        {
            "op": "create",
            "path": f".github/scripts/templates/T2/pending-registration-{_pid8(pid)}.json",
            "content": json.dumps(fragment, indent=2, sort_keys=True) + "\n",
        },
    ]


def resolve_T3(params: dict[str, Any], pid: str) -> list[dict]:
    """Expand T3 (reviewer-veto-before-commit) params into operations.

    Emits a per-friction instantiation directory with a config.json capturing
    the gate configuration plus the runtime logger. The gate_runner logic is
    referenced from the reference impl at .github/mechanism-templates/T3/.

    Deliberately emits NO companion .instructions.md (unlike resolve_T1): per
    Decision 10, T3 gates run pipeline-internally inside the L3 authorer
    slow-path. The chat agent never invokes T3 directly, so there is no
    agent-facing behavior to document in an instruction file. The catalog
    skeleton lists an instruction file as optional; for the pipeline-internal
    deployment it is omitted.
    """
    d = _t3_instance_dir(pid)
    config = {
        "template": "T3",
        "pid": pid,
        "proposal_kind": params["proposal_kind"],
        "syntactic_gate": params["syntactic_gate"],
        "semantic_gate": params["semantic_gate"],
        "rejected_buffer_path": params["rejected_buffer_path"],
        "max_attempts": params["max_attempts"],
        "scope": params["scope"],
    }
    return [
        {
            "op": "create",
            "path": f"{d}/config.json",
            "content": json.dumps(config, indent=2, sort_keys=True) + "\n",
        },
        {
            "op": "create",
            "path": f"{d}/logger.py",
            "content": '"""Runtime logger for this T3 instantiation."""\n' + _LOGGER_SNIPPET,
        },
    ]


_RESOLVERS = {
    "T1": resolve_T1,
    "T2": resolve_T2,
    "T3": resolve_T3,
}


def resolve(template: dict, pid: str) -> list[dict]:
    """Dispatch on template.id. Returns the list of operations.

    Raises KeyError on unknown id — callers must validate the template block
    (lm_authorer.validate_output) before calling resolve().
    """
    tid = template["id"]
    resolver = _RESOLVERS[tid]
    return resolver(template["params"], pid)


def _t1_instruction_md(config: dict) -> str:
    """Companion instruction file describing how the agent reacts to the gate."""
    return f"""---
description: "T1 retrieve-then-gate instantiation for pattern {config['pid']}. Auto-generated by template_resolver.py; do not hand-edit."
applyTo: "**"
---

# T1 retrieve-then-gate — {config['pid']}

Auto-generated from a mechanism-template instantiation. Source pattern:
`{config['pid']}`. Scope: `{config['scope']}`.

Before producing an answer matching the trigger `{config['trigger']}`, the
agent should consult the retrieval gate over:

{chr(10).join(f"- `{g}`" for g in config['corpus_globs'])}

**Sufficiency criterion**: {config['sufficiency_check']}

If retrieval is insufficient after {config['max_iterations']} iteration(s),
fall back to: `{config['fallback_action']}`.

This file is regenerated whenever the instantiation's parameters change. See
`.github/mechanism-templates/T1/README.md` for the mechanism contract.
"""
