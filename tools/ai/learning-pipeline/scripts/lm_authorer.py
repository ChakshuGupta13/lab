#!/usr/bin/env python3
"""
LM authorer wrapper — invokes Learning-Authorer agent via `copilot -p`.

CRITICAL fixes from Adversary review #3:
  - Per-pid mkdir-lock (`.github/learning/staging/<pid>/.lock/`)
  - Atomic write (tmp + rename)
  - Time-based rate limit on lm_eval_state.last_eval_at (≥24h default)
  - Observation-count gate via next_check_after_n_observations
  - CLI version check at startup (Adversary concern B)
  - Strict JSON schema validation of authorer output
  - JSON-parsable-or-reject contract

This script is invoked ASYNC by the slow-path pipeline runner.
It is NEVER called from the hot-path (UserPromptSubmit) hook.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

from .schema import open_db, state_root
from .template_resolver import resolve as resolve_template


# -------- CLI version check (Adversary B) ----------------------------------

REQUIRED_CLI_FLAGS = {"--prompt", "--output-format", "--agent", "--add-dir", "--allow-all-tools"}


def find_copilot_cli() -> str | None:
    """Return path to the `copilot` binary, or None if missing."""
    return shutil.which("copilot")


def copilot_cli_supports_required_flags() -> tuple[bool, str]:
    """Verify the installed CLI has the flags we need.
    Returns (ok, message)."""
    binary = find_copilot_cli()
    if not binary:
        return False, "copilot CLI not found on PATH"
    try:
        help_text = subprocess.run(
            [binary, "--help"], capture_output=True, text=True, timeout=10
        ).stdout
    except (subprocess.SubprocessError, OSError) as e:
        return False, f"copilot --help failed: {e}"
    missing = [f for f in REQUIRED_CLI_FLAGS if f not in help_text]
    if missing:
        return False, f"copilot CLI missing flags: {missing}"
    return True, "ok"


# -------- Per-pid lock + atomic write -------------------------------------

def staging_dir_for(pid: str) -> Path:
    """Per-pid staging directory under .github/learning/staging/<pid>/."""
    workspace_root = _workspace_root()
    return workspace_root / ".github" / "learning" / "staging" / pid


def _workspace_root() -> Path:
    """Find the workspace root by walking up from this script's location."""
    here = Path(__file__).resolve()
    # .github/scripts/learning/v2/lm_authorer.py → walk up 4
    return here.parent.parent.parent.parent.parent


def acquire_lock(pid: str) -> bool:
    """Atomic mkdir-lock per pid. Returns True if acquired."""
    lock_dir = staging_dir_for(pid) / ".lock"
    lock_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        lock_dir.mkdir()
        (lock_dir / "pid").write_text(str(os.getpid()))
        return True
    except FileExistsError:
        # Stale-PID detection: if owner is dead, reclaim.
        try:
            owner_pid = int((lock_dir / "pid").read_text().strip())
            os.kill(owner_pid, 0)  # raises if dead
            return False
        except (FileNotFoundError, ValueError, ProcessLookupError, PermissionError):
            try:
                shutil.rmtree(lock_dir)
                lock_dir.mkdir()
                (lock_dir / "pid").write_text(str(os.getpid()))
                return True
            except FileExistsError:
                return False


def release_lock(pid: str) -> None:
    lock_dir = staging_dir_for(pid) / ".lock"
    try:
        shutil.rmtree(lock_dir)
    except FileNotFoundError:
        pass


def atomic_write_json(path: Path, payload: dict) -> None:
    """Write payload to path atomically via tmp+rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)


# -------- Rate limit + observation gate -----------------------------------

def should_evaluate(conn: sqlite3.Connection, pid: str, min_seconds_between: int = 86400) -> tuple[bool, str]:
    """Returns (should_run, reason).

    Gate logic (AND not OR — fixes earlier bug where a fresh burst of
    observations after a recent eval was blocked for 24h):
      - SKIP if pattern is in the '<unknown>' workspace bucket
        (cannot be matched for effectiveness — wastes a CLI call).
      - RUN if no prior eval row exists.
      - RUN if EITHER the time floor has elapsed (>= min_seconds_between)
        OR enough new observations have accumulated (delta_obs >= next_after).
      - Equivalently SKIP only if BOTH gates fail (time floor not yet
        elapsed AND observation delta below threshold). This means a fast
        burst of new evidence overrides the 24h time floor.
    Rate-limit knobs are per-pid; the LM sets next_check_after_n_observations
    in each eval based on pattern volatility.
    """
    row = conn.execute(
        "SELECT n_observations, workspace FROM patterns WHERE pid = ?", (pid,)
    ).fetchone()
    if row is None:
        return False, "pattern not found"
    n_obs, workspace = row
    if workspace in (None, "<unknown>"):
        return False, "unknown-workspace pattern: cannot match for effectiveness"
    row = conn.execute(
        "SELECT last_eval_at, last_eval_n_observations, next_check_after_n_observations, min_seconds_between_evals "
        "FROM lm_eval_state WHERE pid = ?",
        (pid,),
    ).fetchone()
    if row is None:
        # First eval — no rate limit applies.
        return True, "first eval"
    last_eval_at, last_n_obs, next_after, min_sec = row
    now = int(time.time())
    elapsed = now - (last_eval_at or 0)
    delta = n_obs - (last_n_obs or 0)
    min_s = min_sec or min_seconds_between
    need_obs = next_after or 1
    time_floor_passed = elapsed >= min_s
    burst_threshold_passed = delta >= need_obs
    if not time_floor_passed and not burst_threshold_passed:
        # Both gates fail — genuinely too soon and too few new observations.
        return False, (
            f"rate-limit: only {elapsed}s elapsed (need {min_s}) and "
            f"only {delta} new obs since last eval (need {need_obs})"
        )
    why = []
    if burst_threshold_passed:
        why.append(f"burst: {delta} new obs >= {need_obs}")
    if time_floor_passed:
        why.append(f"timer: {elapsed}s >= {min_s}")
    return True, "ok (" + ", ".join(why) + ")"


def record_eval_attempt(
    conn: sqlite3.Connection, pid: str, n_obs_now: int, verdict: str | None,
    next_check_after: int = 5, min_seconds_between: int = 86400,
) -> None:
    """Upsert lm_eval_state for this pid."""
    conn.execute(
        "INSERT INTO lm_eval_state "
        "(pid, last_eval_at, last_eval_n_observations, last_verdict, "
        " next_check_after_n_observations, min_seconds_between_evals) "
        "VALUES (?, ?, ?, ?, ?, ?) "
        "ON CONFLICT(pid) DO UPDATE SET "
        "  last_eval_at = excluded.last_eval_at, "
        "  last_eval_n_observations = excluded.last_eval_n_observations, "
        "  last_verdict = excluded.last_verdict, "
        "  next_check_after_n_observations = excluded.next_check_after_n_observations, "
        "  min_seconds_between_evals = excluded.min_seconds_between_evals",
        (pid, int(time.time()), n_obs_now, verdict, next_check_after, min_seconds_between),
    )


# -------- Authorer invocation ---------------------------------------------

PROMPT_TEMPLATE = """\
You are Learning-Authorer. Read the JSON input below, follow your agent spec, and emit STRICT JSON.

INPUT:
{input_json}

Now output the result JSON. No prose, no fences, just the JSON object.
"""


def build_input(conn: sqlite3.Connection, pid: str, workspace_root: Path) -> dict:
    """Gather pattern + prior attempts + customization inventory for the authorer."""
    pat = conn.execute(
        "SELECT pid, detector, key, workspace, first_seen_ts, last_seen_ts, "
        "       n_observations, n_sessions, status "
        "FROM patterns WHERE pid = ?",
        (pid,),
    ).fetchone()
    if not pat:
        raise ValueError(f"unknown pid {pid}")
    pattern = dict(zip(
        ["pid", "detector", "key", "workspace", "first_seen_ts", "last_seen_ts",
         "n_observations", "n_sessions", "status"],
        pat,
    ))
    pattern["recent_observations"] = [
        {"ts": ts, "session_id": sid, "evidence": ev[:200]}
        for ts, sid, ev in conn.execute(
            "SELECT ts, session_id, evidence FROM pattern_observations "
            "WHERE pid = ? ORDER BY ts DESC LIMIT 10",
            (pid,),
        )
    ]
    prior = [
        dict(zip(
            ["tier", "applied_at", "target_paths", "outcome", "post_rate", "pre_rate"],
            row,
        ))
        for row in conn.execute(
            "SELECT tier, applied_at, target_paths, outcome, post_rate, pre_rate "
            "FROM pattern_fix_history WHERE pid = ? ORDER BY applied_at",
            (pid,),
        )
    ]
    inventory = _customization_inventory(workspace_root)
    # Novelty check (principle #3): read content of instruction files whose
    # names share keywords with the pattern key. This gives the authorer the
    # ACTUAL rule text to check for overlap, not just the filename.
    pattern_key = pattern.get("key", "")
    inventory["relevant_instruction_content"] = _read_relevant_instructions(
        workspace_root, pattern_key, inventory.get("instructions", [])
    )
    return {
        "pattern": pattern,
        "prior_attempts": prior,
        "customization_inventory": inventory,
        "workspace_root": str(workspace_root),
    }


def _customization_inventory(workspace_root: Path) -> dict:
    """List files under the customization tree + read user-memory rules.

    Reads mandatory.md and user-level prompts so the LM authorer knows which
    rules already exist and does not propose duplicates (design-gap fix
    2026-06-06). The content is truncated to 2000 chars to stay within
    reasonable prompt budget.

    Also reads the CONTENT of instruction files whose names share keywords
    with the pattern being evaluated (novelty-check, principle #3 from
    autonomous-systems survey). This prevents the authorer from creating
    a new file when an existing one already addresses the friction."""
    out = {}
    for sub in ["instructions", "prompts", "agents", "hooks"]:
        d = workspace_root / ".github" / sub
        if d.is_dir():
            out[sub] = sorted(str(p.relative_to(workspace_root)) for p in d.glob("*.*"))
        else:
            out[sub] = []
    # User-memory mandatory.md — read content so the authorer sees existing rules.
    # Path: the VS Code memory system stores user-level memory at /memories/mandatory.md
    # which maps to ~/.copilot/memory/mandatory.md on disk. But user-memory is
    # also loaded into every session context automatically (first 200 lines).
    # We read from the known user-prompts location as well.
    user_rules_content = ""
    for candidate in [
        Path.home() / "Library" / "Application Support" / "Code" / "User"
        / "globalStorage" / "github.copilot-chat" / "memory-tool" / "memories" / "mandatory.md",
        Path.home() / "Library" / "Application Support" / "Code" / "User" / "prompts" / "mandatory.md",
        Path.home() / ".copilot" / "memory" / "mandatory.md",
    ]:
        if candidate.is_file():
            try:
                user_rules_content = candidate.read_text(encoding="utf-8")[:2000]
                break
            except OSError:
                pass
    out["user_memory_rules"] = user_rules_content or "(not found)"
    # Also read user-level instruction files (~/Library/.../Code/User/prompts/*.instructions.md)
    user_prompts_dir = Path.home() / "Library" / "Application Support" / "Code" / "User" / "prompts"
    if user_prompts_dir.is_dir():
        out["user_instructions"] = sorted(
            p.name for p in user_prompts_dir.glob("*.instructions.md")
        )
    else:
        out["user_instructions"] = []
    return out


def _read_relevant_instructions(
    workspace_root: Path, pattern_key: str, instruction_paths: list[str],
) -> dict[str, str]:
    """Read content of instruction files whose names share keywords with the
    pattern key. Returns {relative_path: content_truncated}. Max 3 files,
    each truncated to 1500 chars. Keywords extracted from the pattern key
    by splitting on | and non-alphanumeric characters."""
    import re as _re
    # Extract keywords from pattern key (e.g., "create_file|File already exists" → {"create", "file", "already", "exists"})
    words = set(_re.split(r"[^a-zA-Z0-9]+", pattern_key.lower()))
    # Aggressive stopword list: remove short words, common English, and workspace names
    stopwords = {
        "", "the", "a", "an", "to", "in", "of", "for", "and", "or", "is", "was",
        "not", "it", "you", "must", "use", "can", "do", "has", "had", "be", "are",
        "this", "that", "with", "from", "will", "at", "by", "on", "no", "yes",
        "research", "local", "collision", "differential", "attack", "reduced",
    }
    words = {w for w in words if len(w) > 3 and w not in stopwords}
    if not words:
        return {}
    result: dict[str, str] = {}
    scored: list[tuple[int, str]] = []
    for rel_path in instruction_paths:
        name_lower = rel_path.lower()
        # Score by how many pattern keywords appear in the filename
        score = sum(1 for w in words if w in name_lower)
        if score >= 2:  # require at least 2 keyword matches
            scored.append((score, rel_path))
    # Sort by score descending, take top 3
    scored.sort(key=lambda x: -x[0])
    for _, rel_path in scored[:3]:
        abs_path = workspace_root / rel_path
        if abs_path.is_file():
            try:
                result[rel_path] = abs_path.read_text(encoding="utf-8")[:1500]
            except OSError:
                pass
    return result


REQUIRED_OUTPUT_FIELDS = {
    "pid", "mature", "tier", "justification", "surfacing_blurb",
    "target_rate_drop", "watch_window_K", "next_check_after_n_observations",
    "min_observations_required",
}

# Per-template required parameter slots. MUST stay in sync with the slot tables
# in .github/mechanism-templates/README.md (catalog is the source of truth).
# See .github/learning/integration-design.md Decision 4. A future T4 needs both
# the catalog and this map updated; drift surfaces as "unknown template id" or
# "wrong slots" rejections at validate_output().
TEMPLATE_REQUIRED_SLOTS: dict[str, frozenset[str]] = {
    "T1": frozenset({
        "trigger", "corpus_globs", "sufficiency_check",
        "max_iterations", "fallback_action", "scope",
    }),
    "T2": frozenset({
        "corpus_glob", "required_fields", "status_enum",
        "supersession_field", "lint_command", "scope",
    }),
    "T3": frozenset({
        "proposal_kind", "syntactic_gate", "semantic_gate",
        "rejected_buffer_path", "max_attempts", "scope",
    }),
}


def _validate_template_block(template: object) -> tuple[bool, str]:
    """Validate an optional `template` block. Pure schema check, no I/O.

    Returns (True, 'ok') when the block is well-formed. Called only when the
    payload carries a `template` key (absence is valid — the proposal falls
    through to the free-form path).
    """
    if not isinstance(template, dict):
        return False, "template must be an object"
    tid = template.get("id")
    if tid not in TEMPLATE_REQUIRED_SLOTS:
        return False, (
            f"unknown template id {tid!r}; expected one of "
            f"{sorted(TEMPLATE_REQUIRED_SLOTS)} (see catalog README)"
        )
    params = template.get("params")
    if not isinstance(params, dict):
        return False, f"template.params must be an object for {tid}"
    required = TEMPLATE_REQUIRED_SLOTS[tid]
    got = set(params.keys())
    missing = required - got
    extra = got - required
    if missing:
        return False, f"{tid} params missing slots: {sorted(missing)}"
    if extra:
        return False, f"{tid} params has unknown slots: {sorted(extra)}"
    reduction = template.get("expected_friction_reduction")
    if not isinstance(reduction, str) or not reduction.strip():
        return False, "template.expected_friction_reduction must be a non-empty string"
    return True, "ok"


def validate_output(payload: dict, expected_pid: str) -> tuple[bool, str]:
    missing = REQUIRED_OUTPUT_FIELDS - set(payload.keys())
    if missing:
        return False, f"missing fields: {sorted(missing)}"
    if payload.get("pid") != expected_pid:
        return False, f"pid mismatch: got {payload.get('pid')!r}, expected {expected_pid!r}"
    if not isinstance(payload.get("mature"), bool):
        return False, "mature must be bool"
    tier = payload.get("tier")
    if tier not in (0, 1, 2, 3):
        return False, f"tier must be 0/1/2/3, got {tier!r}"
    blurb = payload.get("surfacing_blurb") or ""
    if len(blurb) > 280:
        return False, f"surfacing_blurb >280 chars ({len(blurb)})"
    proposal = payload.get("proposal")
    if payload.get("mature") and tier in (2, 3):
        if not isinstance(proposal, dict):
            return False, "mature TIER 2/3 needs proposal object"
        for op in proposal.get("operations", []):
            path = op.get("path", "")
            if not path.startswith(".github/"):
                return False, f"target path outside .github/: {path!r}"
            if "mandatory.md" in path:
                return False, f"never propose touching mandatory.md: {path!r}"
            if "/domains/" in path or path.startswith("domains/"):
                return False, f"never propose touching domains/: {path!r}"
    # Optional template block (integration with .github/mechanism-templates/).
    # Absent OR null -> unchanged behavior (S1: a null value is treated as
    # "no template", not a malformed one). Present (non-null) -> schema-checked,
    # and permitted ONLY on mature TIER 2/3 proposals: a template always expands
    # into file operations, so attaching it to TIER 0/1 would create an
    # inconsistent state (apply_proposal.py writes nothing for TIER 0/1, so the
    # resolver's operations would be silently discarded). See Adversary W1.
    tmpl = payload.get("template")
    if tmpl is not None:
        if not (payload.get("mature") and tier in (2, 3)):
            return False, "template block only allowed on mature TIER 2/3 proposals"
        ok, err = _validate_template_block(tmpl)
        if not ok:
            return False, err
    return True, "ok"



# -------- Session cleanup (no sidebar pollution) ---------------------------

COPILOT_SESSION_DB = Path.home() / ".copilot" / "session-store.db"
COPILOT_SESSION_STATE_DIR = Path.home() / ".copilot" / "session-state"

# Tables that have a FK to sessions(id). Order: children first, parent last.
_SESSION_CHILD_TABLES = (
    "turns",
    "checkpoints",
    "session_files",
    "session_refs",
    "forge_trajectory_events",
    "dynamic_context_items",
)


def cleanup_copilot_session(session_uuid: str) -> tuple[bool, str]:
    """Delete one copilot session by UUID from ~/.copilot/session-store.db.

    Both the DB rows (sessions + children) and the on-disk state directory
    are removed. Idempotent: missing rows / dirs are not errors.

    This makes the sidebar pollution from `copilot -p` disappear after each
    authorer invocation. There may be a brief visual flash while the
    invocation is in flight; this cleanup prevents persistent buildup.
    """
    if not COPILOT_SESSION_DB.is_file():
        return True, "no session-store.db; nothing to clean"
    try:
        conn = sqlite3.connect(str(COPILOT_SESSION_DB), timeout=5.0)
        conn.execute("PRAGMA busy_timeout=5000")
        # Delete child rows first to satisfy FK constraints (no CASCADE in schema).
        for tbl in _SESSION_CHILD_TABLES:
            try:
                conn.execute(f"DELETE FROM {tbl} WHERE session_id = ?", (session_uuid,))
            except sqlite3.OperationalError:
                # Table may not exist in older copilot CLI versions.
                pass
        # Search index is FTS, may not have a session_id column.
        try:
            conn.execute(
                "DELETE FROM search_index WHERE session_id = ?", (session_uuid,)
            )
        except sqlite3.OperationalError:
            pass
        conn.execute("DELETE FROM sessions WHERE id = ?", (session_uuid,))
        conn.commit()
        # Passive WAL checkpoint — flush completed transactions only.
        # DO NOT use TRUNCATE: it blocks concurrent writers (VS Code) and
        # can discard their in-flight session data.
        try:
            conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
        except sqlite3.OperationalError:
            pass
        conn.close()
    except sqlite3.Error as e:
        return False, f"DB cleanup failed: {e}"
    # Remove the on-disk state directory if present.
    state_dir = COPILOT_SESSION_STATE_DIR / session_uuid
    if state_dir.is_dir():
        try:
            shutil.rmtree(state_dir)
        except OSError as e:
            return False, f"state-dir cleanup failed: {e}"
    return True, "cleaned"


def _session_id_for(pid: str) -> str:
    """Deterministic UUIDv5 from a fixed namespace + pid.
    Reusing the same UUID across runs means copilot CLI resumes the same
    session row in ~/.copilot/session-store.db instead of creating a new
    one each invocation. Downstream cleanup_sessions.py wipes that row
    after the authorer returns, so sidebar pollution is limited to a
    sub-second flash.
    """
    import uuid
    namespace = uuid.UUID("18b3c8d0-1111-4111-8111-1eaa11111111")  # arbitrary fixed
    return str(uuid.uuid5(namespace, f"learning-authorer:{pid}"))


def invoke_copilot_cli(pid: str, input_payload: dict, workspace_root: Path,
                       timeout_seconds: int = 120) -> tuple[bool, dict | str]:
    """Run `copilot -p ... --agent Learning-Authorer` and return parsed JSON or error."""
    staging = staging_dir_for(pid)
    staging.mkdir(parents=True, exist_ok=True)
    # Write the input alongside so the authorer can see exactly what it received.
    atomic_write_json(staging / "input.json", input_payload)
    binary = find_copilot_cli()
    if not binary:
        return False, "copilot CLI missing"
    prompt_text = PROMPT_TEMPLATE.format(input_json=json.dumps(input_payload, indent=2))
    session_id = _session_id_for(pid)
    cmd = [
        binary, "-p", prompt_text,
        "--agent", "Learning-Authorer",
        "--output-format", "json",
        "--allow-all-tools",
        "--add-dir", str(staging),
        "--no-color",
        "-s",  # silent: response-only
        "--session-id", session_id,
        "--name", f"learn-{pid[:8]}",  # human-readable sidebar label
    ]
    try:
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True,
                timeout=timeout_seconds, cwd=str(workspace_root),
            )
        except subprocess.TimeoutExpired:
            return False, f"copilot CLI timed out after {timeout_seconds}s"
        except (subprocess.SubprocessError, OSError) as e:
            return False, f"copilot CLI exec failed: {e}"
        if proc.returncode != 0:
            return False, f"copilot CLI exit {proc.returncode}: stderr={proc.stderr[:400]}"
        candidate = _extract_authorer_payload(proc.stdout, pid)
        if candidate is None:
            return False, f"could not find authorer JSON in CLI output (head): {proc.stdout[:500]}"
        return True, candidate
    finally:
        # ALWAYS clean up the CLI session — prevents sidebar pollution.
        cleanup_copilot_session(session_id)


def _extract_authorer_payload(stdout: str, pid: str) -> dict | None:
    """Scan JSONL output from `copilot -p --output-format json` for the
    authorer's payload.

    Real CLI shape (verified 2026-06-05 with copilot 1.0.59):
      {"type":"session.mcp_server_status_changed", ...}
      {"type":"session.mcp_servers_loaded", ...}
      {"type":"user.message", ...}
      {"type":"assistant.turn_start", ...}
      {"type":"assistant.message_start", ...}
      {"type":"assistant.message_delta", "data":{"deltaContent":"..."}}
      {"type":"assistant.message", "data":{"content":"<final text>", ...}}
      {"type":"assistant.turn_end", ...}
      {"type":"result", "exitCode":0, ...}

    The LLM's final text lives in the LAST `assistant.message` event's
    `data.content` field. The text may be raw JSON, JSON wrapped in
    ```json fences, or JSON preceded by prose (which the agent spec
    forbids but we tolerate by hunting the first {...} block).
    """
    # First: scan for assistant.message events, keep the last one.
    last_content = None
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        if obj.get("type") == "assistant.message":
            data = obj.get("data") or {}
            content = data.get("content")
            if isinstance(content, str) and content.strip():
                last_content = content

    if last_content is None:
        # Fallback: maybe the entire stdout is a single JSON document with our pid.
        try:
            full = json.loads(stdout.strip())
            if isinstance(full, dict) and full.get("pid") == pid:
                return full
        except json.JSONDecodeError:
            pass
        return None

    # Strip code fences.
    s = last_content.strip()
    if s.startswith("```"):
        # ```json or ```\n
        s = s.split("\n", 1)[1] if "\n" in s else s[3:]
        if s.endswith("```"):
            s = s[: -3].strip()
        else:
            # Trailing fence might be on its own line earlier in the string.
            s = s.rsplit("```", 1)[0].strip()

    # Try direct JSON parse.
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    # Last resort: hunt the first balanced {...} block in the content.
    start = s.find("{")
    if start < 0:
        return None
    depth = 0
    for i, ch in enumerate(s[start:], start=start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = s[start : i + 1]
                try:
                    obj = json.loads(candidate)
                    if isinstance(obj, dict):
                        return obj
                except json.JSONDecodeError:
                    return None
                break
    return None


def run_for_pid(pid: str) -> dict:
    """End-to-end: lock, evaluate, invoke CLI, validate, stash, record eval."""
    ok, msg = copilot_cli_supports_required_flags()
    if not ok:
        return {"ok": False, "reason": msg}
    workspace_root = _workspace_root()
    conn = open_db()
    try:
        ok, reason = should_evaluate(conn, pid)
        if not ok:
            return {"ok": False, "reason": reason, "skipped": True}
        if not acquire_lock(pid):
            return {"ok": False, "reason": "lock held"}
        try:
            input_payload = build_input(conn, pid, workspace_root)
            n_obs_now = input_payload["pattern"]["n_observations"]
            ok, output = invoke_copilot_cli(pid, input_payload, workspace_root)
            if not ok:
                # Still record the attempt so we don't retry immediately.
                record_eval_attempt(conn, pid, n_obs_now, verdict=None)
                return {"ok": False, "reason": output, "n_obs": n_obs_now}
            ok, err = validate_output(output, pid)
            if not ok:
                record_eval_attempt(conn, pid, n_obs_now, verdict=None)
                return {"ok": False, "reason": f"output validation: {err}",
                        "raw_output": output}
            # Resolve an optional template block into concrete operations
            # (integration-design.md Decision 5). The resolver is deterministic
            # and OVERWRITES any hand-authored operations, so the Authorer
            # cannot smuggle free-form changes under a template label.
            template = output.get("template")
            if template:
                try:
                    resolved_ops = resolve_template(template, pid)
                except Exception as e:  # noqa: BLE001
                    record_eval_attempt(conn, pid, n_obs_now, verdict=None)
                    return {"ok": False,
                            "reason": f"template resolution failed: {e}",
                            "raw_output": output}
                if not isinstance(output.get("proposal"), dict):
                    output["proposal"] = {}
                output["proposal"]["operations"] = resolved_ops
                output["proposal"]["target_paths"] = sorted(
                    {op["path"] for op in resolved_ops}
                )
                # Re-validate so resolved operations pass the same path
                # whitelist as hand-authored ones (apply_proposal.py also
                # enforces this at apply time; this is defense in depth).
                ok, err = validate_output(output, pid)
                if not ok:
                    record_eval_attempt(conn, pid, n_obs_now, verdict=None)
                    return {"ok": False,
                            "reason": f"post-resolution validation: {err}",
                            "raw_output": output}
            # Stash the validated proposal atomically.
            atomic_write_json(staging_dir_for(pid) / "proposal.json", output)
            verdict_label = (
                f"TIER{output['tier']}" if output["mature"] else "OBSERVE"
            )
            record_eval_attempt(
                conn, pid, n_obs_now, verdict=verdict_label,
                next_check_after=int(output.get("next_check_after_n_observations", 5)),
            )
            # Mark pattern as ACTIVE if mature, else leave OBSERVE.
            if output["mature"]:
                conn.execute(
                    "UPDATE patterns SET status = 'ACTIVE' WHERE pid = ? AND status = 'OBSERVE'",
                    (pid,),
                )
            return {"ok": True, "verdict": verdict_label, "output": output}
        finally:
            release_lock(pid)
    finally:
        conn.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pid", required=True, help="Pattern id to evaluate")
    args = ap.parse_args()
    result = run_for_pid(args.pid)
    print(json.dumps(result, indent=2, default=str))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
