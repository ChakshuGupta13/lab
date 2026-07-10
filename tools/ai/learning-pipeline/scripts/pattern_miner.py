#!/usr/bin/env python3
"""
pattern_miner.py — L2 layer of the rolling-learning pipeline.

Invokes Learning-Miner agent via `copilot -p`. Reads recent session activity
(turns + self-signals + active patterns) and emits semantic features into
l2_feature_queue. Slow path drains the queue into pattern_observations.

Trigger: ≥3 new self-signals in current session since last l2_runs entry.

Shares hygiene with lm_authorer.py:
  - LEARNING_PIPELINE_ENABLED=1 gate
  - --session-id deterministic per (session_id, fire_count)
  - cleanup_copilot_session after every invocation
  - atomic tmp+rename for staging artefacts
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import subprocess
import sys
import time
import uuid
from pathlib import Path

from .schema import open_db
from .lm_authorer import (
    cleanup_copilot_session,
    copilot_cli_supports_required_flags,
    find_copilot_cli,
    atomic_write_json,
)


L2_TRIGGER_MIN_SIGNALS = 3       # ≥3 new self-signals since last l2_runs row
L2_TURN_WINDOW = 10              # last N user turns to include in input
L2_SIGNAL_WINDOW = 10            # last N self-signals to include
L2_TIMEOUT_S = 90                # per CLI invocation
L2_CONFIDENCE_FLOOR = 0.6        # below this, semantic feature dropped
L2_FRICTION_CONFIDENCE_FLOOR = 0.7  # friction features need higher confidence
L2_MAX_FEATURES = 3              # max semantic features per invocation (friction unlimited)
L2_SESSION_NS = uuid.UUID("18b3c8d0-2222-4222-8222-2eaa22222222")
FEATURE_NAME_RX = re.compile(r"^(?:friction:[a-z][a-z0-9-]{0,30}|[a-z][a-z0-9-]{1,39})$")


# -------- Trigger -----------------------------------------------------------

def signals_since_last_l2(conn: sqlite3.Connection, session_id: str) -> int:
    """Count self_signals in this session since last l2_runs row.
    self_signals.ts is in ms; l2_runs.run_at is in seconds — convert."""
    row = conn.execute(
        "SELECT MAX(run_at) FROM l2_runs WHERE session_id = ?", (session_id,)
    ).fetchone()
    last_run_at_s = (row[0] if row else None) or 0
    last_run_ms = last_run_at_s * 1000
    cur = conn.execute(
        "SELECT COUNT(*) FROM self_signals WHERE session_id = ? AND ts > ?",
        (session_id, last_run_ms),
    )
    return cur.fetchone()[0]


def should_run_l2(conn: sqlite3.Connection, session_id: str) -> tuple[bool, str]:
    n = signals_since_last_l2(conn, session_id)
    if n < L2_TRIGGER_MIN_SIGNALS:
        return False, f"only {n} new self-signals (need {L2_TRIGGER_MIN_SIGNALS})"
    return True, f"{n} new self-signals since last l2_runs row"


# -------- Input gathering --------------------------------------------------

def build_input(conn: sqlite3.Connection, session_id: str) -> dict:
    sess = conn.execute(
        "SELECT workspace_path FROM sessions WHERE session_id = ?", (session_id,)
    ).fetchone()
    if not sess:
        return {"error": f"session {session_id} not found"}
    workspace_path = sess[0]
    workspace = (workspace_path or "").rstrip("/").split("/")[-1] or "<unknown>"

    turns = []
    for ts, content in conn.execute(
        "SELECT ts, content FROM user_messages WHERE session_id = ? "
        "ORDER BY ts DESC LIMIT ?",
        (session_id, L2_TURN_WINDOW),
    ).fetchall():
        turns.append({"role": "user", "ts": ts, "content": (content or "")[:1500]})
    turns.sort(key=lambda t: t["ts"])

    signals = []
    for ts, sig_type, evidence in conn.execute(
        "SELECT ts, type, evidence FROM self_signals WHERE session_id = ? "
        "ORDER BY ts DESC LIMIT ?",
        (session_id, L2_SIGNAL_WINDOW),
    ).fetchall():
        signals.append({"type": sig_type, "ts": ts, "evidence": evidence or ""})
    signals.sort(key=lambda s: s["ts"])

    active_patterns = []
    for pid, detector, key, n_obs in conn.execute(
        "SELECT pid, detector, key, n_observations FROM patterns "
        "WHERE workspace = ? AND status IN ('ACTIVE', 'OBSERVE') "
        "ORDER BY n_observations DESC LIMIT 20",
        (workspace,),
    ).fetchall():
        active_patterns.append({
            "pid": pid, "detector": detector, "key": key, "n_observations": n_obs,
        })

    # Read the last agent response snapshot (written by UserPromptSubmit hook).
    # This replaces regex-based introspection — the LLM does semantic
    # classification instead of regex pattern matching.
    last_response = None
    snapshot_path = (
        _workspace_root() / ".github" / "learning" / "staging"
        / f".last_response-{session_id[:16]}.json"
    )
    if snapshot_path.is_file():
        try:
            snap = json.loads(snapshot_path.read_text(encoding="utf-8"))
            if snap.get("session_id") == session_id:
                last_response = snap.get("content", "")[:5000]
            # Clean up after reading (one-shot consumption)
            snapshot_path.unlink(missing_ok=True)
        except (json.JSONDecodeError, OSError):
            pass

    result = {
        "session_id": session_id,
        "workspace": workspace,
        "recent_turns": turns,
        "recent_self_signals": signals,
        "active_patterns": active_patterns,
    }
    if last_response:
        result["last_agent_response"] = last_response
    return result


# -------- CLI invocation --------------------------------------------------

PROMPT_TEMPLATE = """\
You are Learning-Miner. Read the JSON input below, follow your agent spec, and emit STRICT JSON.

INPUT:
{input_json}

Now output the result JSON. No prose, no fences, just the JSON object.
"""


def _session_uuid_for(session_id: str, fire_count: int) -> str:
    """Deterministic per (session_id, fire_count) so concurrent fires don't
    collide on copilot session-store rows."""
    return str(uuid.uuid5(L2_SESSION_NS, f"learning-miner:{session_id}:{fire_count}"))


def _extract_miner_payload(stdout: str) -> dict | None:
    """Walk assistant.message events for the last one's data.content (JSON)."""
    last = None
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and obj.get("type") == "assistant.message":
            data = obj.get("data") or {}
            c = data.get("content")
            if isinstance(c, str) and c.strip():
                last = c
    if last is None:
        return None
    s = last.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[1] if "\n" in s else s[3:]
        if s.endswith("```"):
            s = s[:-3].strip()
        else:
            s = s.rsplit("```", 1)[0].strip()
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
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
                try:
                    obj = json.loads(s[start : i + 1])
                    return obj if isinstance(obj, dict) else None
                except json.JSONDecodeError:
                    return None
    return None


def _validate_features(payload: dict, session_id: str) -> list[dict]:
    if not isinstance(payload, dict):
        return []
    if payload.get("session_id") != session_id:
        return []
    feats_in = payload.get("features") or []
    if not isinstance(feats_in, list):
        return []
    semantic = []
    friction = []
    for f in feats_in:
        if not isinstance(f, dict):
            continue
        name = (f.get("name") or "").strip()
        if not FEATURE_NAME_RX.match(name):
            continue
        try:
            conf = float(f.get("confidence", 0.0))
        except (TypeError, ValueError):
            continue
        is_friction = name.startswith("friction:")
        floor = L2_FRICTION_CONFIDENCE_FLOOR if is_friction else L2_CONFIDENCE_FLOOR
        if conf < floor:
            continue
        evidence = (f.get("evidence") or "")[:200]
        entry = {"name": name, "evidence": evidence, "confidence": conf}
        if is_friction:
            friction.append(entry)
        else:
            semantic.append(entry)
    # Cap semantic features; friction unlimited
    return friction + semantic[:L2_MAX_FEATURES]


def _record_l2_run(
    conn: sqlite3.Connection, session_id: str, workspace: str,
    run_at: int, signals_count: int, features_emitted: int,
    ok: bool, reason: str,
) -> None:
    conn.execute(
        "INSERT INTO l2_runs "
        "(session_id, workspace, run_at, signals_count, features_emitted, ok, reason) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (session_id, workspace, run_at, signals_count, features_emitted,
         1 if ok else 0, reason[:400]),
    )


def _workspace_root() -> Path:
    here = Path(__file__).resolve()
    return here.parent.parent.parent.parent.parent


def run_l2_for(session_id: str) -> dict:
    """Single L2 invocation for one session. Returns summary dict."""
    if os.environ.get("LEARNING_PIPELINE_ENABLED") != "1":
        return {"ok": False, "reason": "LEARNING_PIPELINE_ENABLED!=1"}
    ok, msg = copilot_cli_supports_required_flags()
    if not ok:
        return {"ok": False, "reason": msg}
    workspace_root = _workspace_root()
    conn = open_db()
    try:
        ok, reason = should_run_l2(conn, session_id)
        if not ok:
            return {"ok": False, "reason": reason, "skipped": True}
        input_payload = build_input(conn, session_id)
        if input_payload.get("error"):
            return {"ok": False, "reason": input_payload["error"]}
        fire_count = conn.execute(
            "SELECT COUNT(*) FROM l2_runs WHERE session_id = ?", (session_id,)
        ).fetchone()[0]
        cli_session_id = _session_uuid_for(session_id, fire_count)
        workspace = input_payload["workspace"]
        signals_count = signals_since_last_l2(conn, session_id)
        run_at = int(time.time())

        # Everything after cli_session_id creation MUST clean up the CLI
        # session, even on crash. Otherwise the sidebar gets polluted.
        try:
            staging = (
                workspace_root / ".github" / "learning" / "staging"
                / f"l2-{cli_session_id[:8]}"
            )
            staging.mkdir(parents=True, exist_ok=True)
            atomic_write_json(staging / "input.json", input_payload)

            binary = find_copilot_cli()
            if not binary:
                _record_l2_run(conn, session_id, workspace, run_at,
                               signals_count, 0, False, "copilot CLI not found")
                return {"ok": False, "reason": "copilot CLI not found on PATH"}

            prompt_text = PROMPT_TEMPLATE.format(
                input_json=json.dumps(input_payload, indent=2)
            )
            cmd = [
                binary, "-p", prompt_text,
                "--agent", "Learning-Miner",
                "--output-format", "json",
                "--allow-all-tools",
                "--add-dir", str(staging),
                "--no-color", "-s",
                "--session-id", cli_session_id,
                "--name", f"l2-{session_id[:8]}-{fire_count}",
            ]
            try:
                proc = subprocess.run(
                    cmd, capture_output=True, text=True,
                    timeout=L2_TIMEOUT_S, cwd=str(workspace_root),
                )
            except subprocess.TimeoutExpired:
                _record_l2_run(conn, session_id, workspace, run_at,
                               signals_count, 0, False, "CLI timeout")
                return {"ok": False, "reason": f"CLI timeout after {L2_TIMEOUT_S}s"}
            except (subprocess.SubprocessError, OSError) as e:
                _record_l2_run(conn, session_id, workspace, run_at,
                               signals_count, 0, False, f"exec: {e}")
                return {"ok": False, "reason": f"exec failed: {e}"}

            if proc.returncode != 0:
                _record_l2_run(conn, session_id, workspace, run_at,
                               signals_count, 0, False,
                               f"exit {proc.returncode}: {proc.stderr[:200]}")
                return {"ok": False, "reason": f"CLI exit {proc.returncode}"}
            payload = _extract_miner_payload(proc.stdout)
            if payload is None:
                _record_l2_run(conn, session_id, workspace, run_at,
                               signals_count, 0, False, "no JSON in output")
                return {"ok": False, "reason": "no JSON in output",
                        "raw_head": proc.stdout[:400]}
            features = _validate_features(payload, session_id)
            atomic_write_json(staging / "features.json",
                              {"raw": payload, "validated": features})

            # Write friction features back to self_signals so they feed
            # detectors + the rest of the pipeline the same way structural
            # tool-error signals do.
            FRICTION_ALLOWLIST = {
                "friction:admitted-mistake", "friction:retry-redo",
                "friction:missed-info", "friction:conceded-to-user",
                "friction:backtracked", "friction:over-engineered",
                "friction:polling-fixation", "friction:lost-context",
                "friction:verification-skipped",
            }
            friction_ts = int(time.time() * 1000)
            for f in features:
                if not f["name"].startswith("friction:"):
                    continue
                if f["name"] not in FRICTION_ALLOWLIST:
                    continue
                sig_type = f["name"].replace("friction:", "agent-", 1)
                conn.execute(
                    "INSERT INTO self_signals "
                    "(session_id, ts, type, evidence, workspace) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (session_id, friction_ts, sig_type, f["evidence"], workspace),
                )

            for f in features:
                conn.execute(
                    "INSERT INTO l2_feature_queue "
                    "(session_id, workspace, feature_name, evidence, confidence, enqueued_at) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (session_id, workspace, f["name"], f["evidence"],
                     f["confidence"], int(time.time())),
                )
            _record_l2_run(conn, session_id, workspace, run_at,
                           signals_count, len(features), True,
                           f"ok ({len(features)} features)")
            return {
                "ok": True,
                "session_id": session_id,
                "signals_processed": signals_count,
                "features_emitted": len(features),
                "features": features,
            }
        finally:
            # ALWAYS clean up the CLI session — prevents sidebar pollution.
            cleanup_copilot_session(cli_session_id)
    finally:
        conn.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session-id", required=True)
    args = ap.parse_args()
    # Release the hook's in-flight lockfile on exit (success or failure).
    # Path passed via env by hook_user_prompt_submit._spawn_l2_miner.
    lockfile = os.environ.get("_LEARNING_L2_LOCKFILE")
    try:
        result = run_l2_for(args.session_id)
        print(json.dumps(result, indent=2))
        return 0 if result.get("ok") else 1
    finally:
        if lockfile:
            try:
                Path(lockfile).unlink()
            except OSError:
                pass


if __name__ == "__main__":
    sys.exit(main())
