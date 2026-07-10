#!/usr/bin/env python3
"""
UserPromptSubmit hook hot path.

Contract:
  stdin:  JSON with at least {prompt, sessionId, cwd, hookEventName}
  stdout: JSON, MAY include hookSpecificOutput.additionalContext OR be `{"continue": true}` on no-op
  exit:   0 always (fail-open)

Discipline:
  - <100 ms typical, hard internal deadline 80 ms before bailing
  - Read-only on DB except for cheap surfacing_state inserts
  - Never blocks the user's prompt
  - Async-spawns the slow-path runner if needed (extractor + detector + lm_authorer)
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

DEADLINE_MS = 80


def _now_ms() -> int:
    return int(time.time() * 1000)


def _bail(reason: str = "") -> int:
    """Emit a continue-only response. Reason is silently ignored (no logging on hot path)."""
    print(json.dumps({"continue": True}))
    return 0


def _emit(additional_context: str) -> int:
    print(json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "UserPromptSubmit",
            "additionalContext": additional_context,
        }
    }))
    return 0


def _workspace_root() -> Path:
    here = Path(__file__).resolve()
    # .github/scripts/learning/v2/hook_user_prompt_submit.py → walk up 4
    return here.parent.parent.parent.parent.parent


def _resolve_learning_enabled() -> str:
    """Resolve LEARNING_PIPELINE_ENABLED with file-based fallback.

    VS Code launched from Dock/Spotlight inherits env from launchd, not
    ~/.zshenv. The env var may be missing even though the user set it.
    Fallback: check for .github/learning/ENABLED marker file.
    """
    val = os.environ.get("LEARNING_PIPELINE_ENABLED", "")
    if val == "1":
        return "1"
    # File-based fallback: existence of marker = enabled
    marker = _workspace_root() / ".github" / "learning" / "ENABLED"
    if marker.is_file():
        return "1"
    return val


def _resolve_path() -> str:
    """Build a PATH that includes locations GUI-launched VS Code may miss.

    macOS GUI apps inherit a minimal PATH from launchd (typically just
    /usr/bin:/bin:/usr/sbin:/sbin). Tools installed via Homebrew, fnm/nvm,
    pip --user, etc. are invisible. Augment with known locations.
    """
    base = os.environ.get("PATH", "/usr/bin:/bin:/usr/sbin:/sbin")
    home = os.environ.get("HOME", "")
    extras = [
        "/opt/homebrew/bin",
        "/usr/local/bin",
    ]
    if home:
        # fnm (stable path, not per-shell multishell symlinks)
        import glob as _glob
        fnm_bins = sorted(_glob.glob(f"{home}/.local/share/fnm/node-versions/*/installation/bin"))
        if fnm_bins:
            extras.append(fnm_bins[-1])  # latest version
        # nvm fallback
        nvm_bins = sorted(_glob.glob(f"{home}/.nvm/versions/node/*/bin"))
        if nvm_bins:
            extras.append(nvm_bins[-1])
        extras.append(f"{home}/.local/bin")
    # Prepend extras that aren't already present
    parts = base.split(":")
    for p in reversed(extras):
        if p and p not in parts:
            parts.insert(0, p)
    return ":".join(parts)


def _is_primary_workspace() -> bool:
    """Is this the primary workspace (sees ALL cross-workspace proposals)?

    Marker file at .github/learning/PRIMARY_WORKSPACE makes this repo the
    primary surfacing destination. Without it, the hook filters proposals
    by workspace name (default behavior — each workspace sees only its own).
    """
    return (_workspace_root() / ".github" / "learning" / "PRIMARY_WORKSPACE").is_file()


def _sweep_pipeline_sessions() -> None:
    """Delete leftover pipeline CLI sessions from ~/.copilot/session-store.db.

    The L2 miner and L3 authorer create `copilot -p` sessions with `summary`
    starting with 'l2-' or 'learn-'. The cleanup in their finally blocks
    races with the CLI's async session finalization — the row sometimes
    reappears after the DELETE. This sweep runs on every UserPromptSubmit
    (<5ms) and catches any that slipped through, keeping the sidebar clean.
    """
    import sqlite3 as _sql
    session_db = Path.home() / ".copilot" / "session-store.db"
    if not session_db.is_file():
        return
    try:
        sconn = _sql.connect(str(session_db), timeout=2.0)
        sconn.execute("PRAGMA busy_timeout=2000")
        # Find pipeline sessions by summary prefix
        stale = sconn.execute(
            "SELECT id FROM sessions WHERE "
            "summary LIKE 'l2-%' OR summary LIKE 'learn-%' "
            "OR summary LIKE 'You are Learning-Miner%' "
            "OR summary LIKE 'You are Learning-Authorer%'"
        ).fetchall()
        if not stale:
            sconn.close()
            return
        stale_ids = [r[0] for r in stale]
        for sid in stale_ids:
            for tbl in ("turns", "checkpoints", "session_files", "session_refs",
                        "forge_trajectory_events", "dynamic_context_items"):
                try:
                    sconn.execute(f"DELETE FROM {tbl} WHERE session_id = ?", (sid,))
                except _sql.OperationalError:
                    pass
            try:
                sconn.execute("DELETE FROM search_index WHERE session_id = ?", (sid,))
            except _sql.OperationalError:
                pass
            sconn.execute("DELETE FROM sessions WHERE id = ?", (sid,))
        sconn.commit()
        # Passive WAL checkpoint — flush completed transactions only.
        # DO NOT use TRUNCATE: it blocks concurrent writers (VS Code's own
        # session management) and can discard their in-flight data.
        try:
            sconn.execute("PRAGMA wal_checkpoint(PASSIVE)")
        except _sql.OperationalError:
            pass
        sconn.close()
        # Also remove state directories
        state_dir = Path.home() / ".copilot" / "session-state"
        for sid in stale_ids:
            d = state_dir / sid
            if d.is_dir():
                try:
                    shutil.rmtree(d)
                except OSError:
                    pass
    except (_sql.Error, OSError):
        pass


# ---- Transcript snapshot for L2 miner ----
# Instead of regex-based introspection (fundamentally incomplete — new
# expressions always missing), the hook saves the last agent response to a
# known location. The L2 miner reads it and does LLM-based classification.
# This replaces INTROSPECTION_PATTERNS which were removed because:
#   - 0% agent compliance on self-signal tags in other sessions
#   - regex patterns always miss new expressions of friction
#   - LLM classification of the same text is strictly more capable

LAST_RESPONSE_DIR = ".github/learning/staging"


def _last_response_path(session_id: str) -> Path:
    """Per-session snapshot path. Avoids cross-session overwrites."""
    return _workspace_root() / LAST_RESPONSE_DIR / f".last_response-{session_id[:16]}.json"


def _snapshot_last_response(transcript_path: str, session_id: str, workspace: str) -> None:
    """Read the last agent response from the transcript and write it to a
    per-session file for the L2 miner to consume. Cheap I/O only — no
    classification. The L2 miner does the semantic analysis async."""
    tp = Path(transcript_path)
    if not tp.is_file():
        return
    try:
        size = tp.stat().st_size
        read_from = max(0, size - 50_000)
        with tp.open("r", encoding="utf-8", errors="replace") as f:
            if read_from > 0:
                f.seek(read_from)
                f.readline()  # skip partial line
            tail_lines = f.readlines()
    except OSError:
        return

    # Find the last assistant.message
    last_content = None
    last_ts = None
    for line in reversed(tail_lines):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        etype = obj.get("type", "")
        if etype == "assistant.message":
            data = obj.get("data") or {}
            content = data.get("content")
            if isinstance(content, str) and len(content) > 20:
                last_content = content
                last_ts = obj.get("timestamp")
                break

    if not last_content:
        return

    # Write snapshot for L2 miner (atomic tmp+rename)
    snapshot = {
        "session_id": session_id,
        "workspace": workspace,
        "timestamp": last_ts,
        "content": last_content[:5000],  # cap at 5K chars for LLM budget
    }
    out_path = _last_response_path(session_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.name + f".{os.getpid()}.tmp")
    try:
        tmp.write_text(json.dumps(snapshot), encoding="utf-8")
        os.replace(tmp, out_path)
    except OSError:
        pass


def _workspace_label(cwd: str) -> str:
    return cwd.rstrip("/").split("/")[-1] or cwd


def main() -> int:
    started_ms = _now_ms()
    deadline_ms = started_ms + DEADLINE_MS

    # Parse stdin (Adversary A: be defensive)
    try:
        raw = sys.stdin.read()
        stdin_payload = json.loads(raw) if raw else {}
    except (json.JSONDecodeError, OSError):
        return _bail("stdin parse")

    session_id = stdin_payload.get("sessionId") or stdin_payload.get("session_id") or ""
    cwd = stdin_payload.get("cwd", "")
    prompt = stdin_payload.get("prompt", "") or ""
    transcript_path = stdin_payload.get("transcript_path") or ""
    workspace = _workspace_label(cwd)

    # Open DB (cheap with WAL + low timeout)
    try:
        from .schema import open_db, state_root
    except ImportError:
        # When invoked directly by the hook shell wrapper, our module path
        # might not be set up; do a sys.path adjustment.
        here = Path(__file__).resolve().parent
        sys.path.insert(0, str(here.parent))
        from v2.schema import open_db, state_root  # type: ignore

    try:
        conn = open_db()
    except Exception:
        return _bail("db open")

    try:
        # ---- Step 0: sweep stale pipeline CLI sessions ----
        # The L2 miner and L3 authorer spawn `copilot -p` which creates rows
        # in ~/.copilot/session-store.db. The cleanup in their finally blocks
        # races with the CLI's async session-finalization writes — the row
        # often reappears AFTER our DELETE. Sweeping here (every prompt,
        # <5ms) catches any that slipped through. Keyed on the `summary`
        # field which always starts with 'l2-' or 'learn-' for pipeline
        # sessions.
        try:
            _sweep_pipeline_sessions()
        except Exception:
            pass

        # ---- Step 1: bump session turn counter ----
        now_s = int(time.time())
        try:
            conn.execute(
                "INSERT INTO session_turn_counter (session_id, last_turn, updated_at) "
                "VALUES (?, 1, ?) "
                "ON CONFLICT(session_id) DO UPDATE SET "
                "  last_turn = last_turn + 1, updated_at = excluded.updated_at",
                (session_id, now_s),
            )
            turn = conn.execute(
                "SELECT last_turn FROM session_turn_counter WHERE session_id = ?",
                (session_id,),
            ).fetchone()[0]
        except Exception:
            return _bail("turn counter")
        if _now_ms() > deadline_ms:
            return _bail("deadline after turn-counter")

        # ---- Step 1b: snapshot last agent response for L2 miner ----
        # Saves the last response text to a known file. The L2 miner reads it
        # and does LLM-based semantic classification (replaces regex-based
        # introspection which was fundamentally incomplete).
        try:
            if transcript_path and session_id:
                _snapshot_last_response(transcript_path, session_id, workspace)
        except Exception:
            pass  # fail-open

        # ---- Step 2: expire stale surfacings ----
        try:
            conn.execute(
                "UPDATE surfacing_state SET status = 'EXPIRED' "
                "WHERE session_id = ? AND status = 'OPEN' "
                "  AND surfaced_at_turn + ttl_turns <= ?",
                (session_id, turn),
            )
        except Exception:
            pass

        # ---- Step 3: find a ready surfacing to emit ----
        # Rules: pid in cached_surfacing for this workspace (or wildcard '*'),
        # no OPEN surfacing for this pid in this session, at most ONE per turn.
        # Health alerts use pid='_health_alert' with workspace='*'.
        # If this is the PRIMARY workspace (marker file present), see ALL
        # proposals regardless of origin workspace.
        #
        # AUTOPILOT SKIP: if the prompt contains autopilot indicators, don't
        # surface anything — the user isn't watching. The suggestion stays in
        # cached_surfacing and will be picked up on the next interactive session.
        autopilot_phrases = (
            "work autonomously", "user is not available",
            "user not available", "autonomous mode",
        )
        is_autopilot = any(p in prompt.lower() for p in autopilot_phrases)
        row = None
        if not is_autopilot:
            try:
                if _is_primary_workspace():
                    row = conn.execute(
                        "SELECT c.pid, c.tier, c.surfacing_blurb "
                        "FROM cached_surfacing c "
                        "LEFT JOIN patterns p ON p.pid = c.pid "
                        "WHERE (p.status = 'ACTIVE' OR c.pid LIKE '_%') "
                        "  AND NOT EXISTS ("
                        "    SELECT 1 FROM surfacing_state s "
                        "    WHERE s.pid = c.pid AND s.session_id = ? "
                        "  ) "
                        "ORDER BY p.n_observations DESC "
                        "LIMIT 1",
                        (session_id,),
                    ).fetchone()
                else:
                    row = conn.execute(
                        "SELECT c.pid, c.tier, c.surfacing_blurb "
                        "FROM cached_surfacing c "
                        "LEFT JOIN patterns p ON p.pid = c.pid "
                        "WHERE (c.workspace = ? OR c.workspace = '*') "
                        "  AND (p.status = 'ACTIVE' OR c.pid LIKE '_%') "
                        "  AND NOT EXISTS ("
                        "    SELECT 1 FROM surfacing_state s "
                        "    WHERE s.pid = c.pid AND s.session_id = ? "
                        "  ) "
                        "ORDER BY p.n_observations DESC "
                        "LIMIT 1",
                        (workspace, session_id),
                    ).fetchone()
            except Exception:
                row = None

        # ---- Step 4: maybe spawn slow path (extractor + detector + authorer) ----
        # Cheap heuristic: if last_run.txt is older than 1h OR doesn't exist,
        # spawn the slow path in the background. Don't block on it.
        # Bug fix per Adversary post-build SUSPICIOUS #4: spawn BEFORE deadline
        # check so a slow surfacing query can never permanently halt the pipeline.
        #
        # SAFETY GUARD (2026-06-05 post-deploy): only spawn when the user has
        # explicitly opted in via the LEARNING_PIPELINE_ENABLED=1 env var.
        # Reason: every copilot -p invocation creates a persistent session in
        # ~/.copilot/session-store.db (visible in the VS Code sidebar). The user
        # must opt in to that consequence per-session. Without the env var the
        # hook still does its cheap surfacing work but never spawns the LM.
        if os.environ.get("LEARNING_PIPELINE_ENABLED") == "1":
            try:
                last_run_file = state_root() / "state" / "last-run.txt"
                need_run = True
                if last_run_file.is_file():
                    try:
                        last_run = int(last_run_file.read_text().strip())
                        need_run = (time.time() - last_run) > 3600  # 1 hour
                    except (ValueError, OSError):
                        pass
                if need_run:
                    _spawn_slow_path()
            except Exception:
                pass

            # ---- Step 4b: L2 pattern miner trigger (async, fire-and-forget) ----
            # Spawns the Copilot CLI to run Learning-Miner. Creates a sidebar
            # session that the sweep (Step 0) cleans on the next prompt.
            try:
                l2_last_row = conn.execute(
                    "SELECT MAX(run_at) FROM l2_runs WHERE session_id = ?",
                    (session_id,),
                ).fetchone()
                last_l2_s = (l2_last_row[0] if l2_last_row else None) or 0
                last_l2_ms = last_l2_s * 1000
                n_new_signals = conn.execute(
                    "SELECT COUNT(*) FROM self_signals "
                    "WHERE session_id = ? AND ts > ?",
                    (session_id, last_l2_ms),
                ).fetchone()[0]
                if n_new_signals >= 3 and not _l2_in_flight(session_id):
                    _spawn_l2_miner(session_id)
            except Exception:
                pass

        if _now_ms() > deadline_ms:
            # No work, but bail with continue.
            return _bail("deadline after surfacing query")

        # ---- Step 5: emit ----
        if row:
            pid, tier, blurb = row
            try:
                conn.execute(
                    "INSERT INTO surfacing_state "
                    "(pid, session_id, surfaced_at, surfaced_at_turn, ttl_turns, status) "
                    "VALUES (?, ?, ?, ?, 3, 'OPEN')",
                    (pid, session_id, now_s, turn),
                )
            except Exception:
                pass

            # Enrich with justification + pattern stats + target path so the
            # agent can present Problem / Proposal / Pattern clearly.
            justification = ""
            target_path = ""
            n_obs = 0
            n_sess = 0
            detector = ""
            try:
                pat = conn.execute(
                    "SELECT detector, n_observations, n_sessions FROM patterns WHERE pid=?",
                    (pid,),
                ).fetchone()
                if pat:
                    detector, n_obs, n_sess = pat
            except Exception:
                pass
            # Read justification + target from proposal JSON if available
            try:
                proposal_path = conn.execute(
                    "SELECT proposal_path FROM cached_surfacing WHERE pid=?", (pid,),
                ).fetchone()
                if proposal_path and proposal_path[0]:
                    import pathlib
                    pp = pathlib.Path(proposal_path[0])
                    if pp.is_file():
                        pdata = json.loads(pp.read_text())
                        justification = pdata.get("justification", "")
                        ops = (pdata.get("proposal") or {}).get("operations", [])
                        if ops:
                            target_path = ops[0].get("path", "")
            except Exception:
                pass

            ctx = (
                f"<learning-suggestion id={pid} tier={tier}>\n"
                f"PROBLEM: {justification or blurb}\n"
                f"PROPOSAL: {blurb}\n"
                f"PATTERN: {detector} — {n_obs} observations across {n_sess} sessions\n"
            )
            if target_path:
                ctx += f"TARGET: {target_path}\n"
            ctx += (
                f"reply: apply / dismiss / <free feedback>\n"
                f"</learning-suggestion>"
            )
            return _emit(ctx)

        return _bail("nothing to surface")
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _l2_lockfile_for(session_id: str) -> Path:
    """In-flight indicator: presence = miner is currently spawned for this session.
    Stored under the same staging tree so it gets cleaned up if user `rm -rf`s
    staging. Stale lockfiles (>10 min old) are treated as expired."""
    return _workspace_root() / ".github" / "learning" / "staging" / f".l2-inflight-{session_id[:16]}"


def _l2_in_flight(session_id: str) -> bool:
    """True if another miner is currently spawned for this session.
    Lockfile older than 10 minutes is considered stale (the miner timed out
    or was killed) and ignored."""
    lock = _l2_lockfile_for(session_id)
    if not lock.exists():
        return False
    try:
        age = time.time() - lock.stat().st_mtime
        if age > 600:  # 10 min stale window — pattern_miner timeout is 90s
            try:
                lock.unlink()
            except OSError:
                pass
            return False
        return True
    except OSError:
        return False


def _spawn_slow_path() -> None:
    """Async-spawn the slow-path runner. Never blocks. Never raises."""
    here = Path(__file__).resolve().parent
    runner = here / "run_slow_path.sh"
    if not runner.is_file():
        return
    try:
        subprocess.Popen(
            ["bash", str(runner)],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            env={
                "HOME": os.environ.get("HOME", ""),
                "PATH": _resolve_path(),
                "WORKSPACE_ROOT": str(_workspace_root()),
                "LEARNING_PIPELINE_ENABLED": _resolve_learning_enabled(),
            },
        )
    except Exception:
        pass


def _spawn_l2_miner(session_id: str) -> None:
    """Async-spawn pattern_miner.py for this session. Fire-and-forget.
    Creates a lockfile so concurrent UserPromptSubmit events don't double-spawn
    (see _l2_in_flight). The miner removes the lockfile on its own exit; if it
    dies hard, the 10-min staleness window in _l2_in_flight reclaims it."""
    here = Path(__file__).resolve().parent
    script = here / "pattern_miner.py"
    if not script.is_file():
        return
    workspace_root = _workspace_root()
    lock = _l2_lockfile_for(session_id)
    try:
        lock.parent.mkdir(parents=True, exist_ok=True)
        lock.touch()
    except OSError:
        # Couldn't create lock — spawn anyway, accept the (small) race.
        pass
    try:
        env = {
            "HOME": os.environ.get("HOME", ""),
            "PATH": _resolve_path(),
            "LEARNING_PIPELINE_ENABLED": _resolve_learning_enabled(),
            "PYTHONPATH": str(workspace_root / ".github" / "scripts" / "learning"),
            "_LEARNING_L2_LOCKFILE": str(lock),
        }
        subprocess.Popen(
            ["python3", "-m", "v2.pattern_miner", "--session-id", session_id],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            env=env,
            cwd=str(workspace_root / ".github" / "scripts" / "learning"),
        )
    except Exception:
        # Spawn failed — release the lock immediately so future prompts can retry.
        try:
            lock.unlink()
        except OSError:
            pass


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        # Absolute fail-open. The hook MUST NOT crash the user's prompt.
        try:
            print(json.dumps({"continue": True}))
        except Exception:
            pass
        sys.exit(0)
