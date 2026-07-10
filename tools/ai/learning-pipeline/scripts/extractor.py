#!/usr/bin/env python3
"""
Extractor — incremental, append-only ingestion of VS Code Copilot debug logs
into v2 events.db.

Reuses v1 extract_signals.py parsing logic (process_session, find_phrase_matches,
iter_jsonl, etc.) but writes into the v2 schema (which is a superset of v1 raw
event tables).

Hot path discipline: this script CAN take seconds on cold start. The
UserPromptSubmit hook must NEVER call this synchronously — it spawns it
async via the slow-path runner.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
import time
from pathlib import Path
from typing import Iterator

from .schema import open_db, state_dir


WORKSPACE_STORAGE_GLOBS = [
    Path.home() / "Library" / "Application Support" / "Code" / "User" / "workspaceStorage",
    Path.home() / ".config" / "Code" / "User" / "workspaceStorage",
]


# Reuse v1 phrase lists. We import from v1's extract_signals to avoid drift.
CORRECTION_PHRASES = [
    r"\bstop polling\b",
    r"\bstop checking\b",
    r"\bdon'?t fixate\b",
    r"\bdon'?t poll\b",
    r"\bi already (?:said|told)\b",
    r"\balready said\b",
    r"\balready told\b",
    r"\bi said\b",
    r"\bi told you\b",
    r"\bnot what i (?:asked|wanted|said)\b",
    r"\byou (?:are|'re) (?:fixating|repeating)\b",
    r"\bpush(?:ing)? back\b",
    r"\bi disagree\b",
    r"\bthat'?s wrong\b",
    r"\bthis is wrong\b",
    r"\bwrong (?:formula|approach|answer|number|claim)\b",
]

SELF_CORRECTION_PHRASES = [
    r"\b(?:let me |i'?ll |i will |should |need to )stop polling\b",
    r"\b(?:let me |i'?ll |i will |should |need to )stop checking\b",
    r"\bi'?ll wait for the (?:terminal )?notification\b",
    r"\bi'?ll check (?:back )?(?:in|later|when)\b",
]


def _compile(patterns: list[str]) -> list[tuple[str, re.Pattern]]:
    out = []
    for p in patterns:
        try:
            out.append((p, re.compile(p, re.IGNORECASE)))
        except re.error:
            continue
    return out


USER_RX = _compile(CORRECTION_PHRASES)
AGENT_RX = _compile(SELF_CORRECTION_PHRASES)


def _find_matches(
    text: str, compiled: list[tuple[str, re.Pattern]]
) -> list[tuple[str, str]]:
    if not text:
        return []
    out = []
    for src, rx in compiled:
        m = rx.search(text)
        if m:
            start = max(0, m.start() - 60)
            end = min(len(text), m.end() + 60)
            snippet = text[start:end].replace("\n", " ").strip()
            out.append((src, snippet))
    return out


def _iter_jsonl(path: Path) -> Iterator[dict]:
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue
    except OSError as e:
        sys.stderr.write(f"warn: cannot read {path}: {e}\n")


def _workspace_storage_root() -> Path | None:
    for c in WORKSPACE_STORAGE_GLOBS:
        if c.is_dir():
            return c
    return None


def _find_sessions(root: Path) -> list[tuple[str, str, Path]]:
    out = []
    for ws_dir in sorted(root.iterdir()):
        if not ws_dir.is_dir():
            continue
        dbg = ws_dir / "GitHub.copilot-chat" / "debug-logs"
        if not dbg.is_dir():
            continue
        for sess_dir in sorted(dbg.iterdir()):
            if not sess_dir.is_dir():
                continue
            main = sess_dir / "main.jsonl"
            if main.is_file() and main.stat().st_size > 0:
                out.append((ws_dir.name, sess_dir.name, main))
    return out


def _read_workspace_path(ws_dir: Path) -> str | None:
    wj = ws_dir / "workspace.json"
    if not wj.is_file():
        return None
    try:
        d = json.loads(wj.read_text())
        return d.get("folder") or d.get("configuration")
    except (json.JSONDecodeError, OSError):
        return None


def _workspace_label(workspace_path: str | None) -> str:
    """Mirror detectors._workspace_label so self_signals rows carry the same key."""
    if not workspace_path:
        return "<unknown>"
    return workspace_path.rstrip("/").split("/")[-1] or workspace_path


# Self-signal tag parser (strict whitelist).
# Matches `<self-signal type=X/>` or `<self-signal type='X' evidence='...'/>` etc.
# Tolerates single quotes, double quotes, or unquoted single-token type values.
SELF_SIGNAL_RX = re.compile(
    r"""<self-signal\s+
        type\s*=\s*['"]?(?P<type>[a-z-]+)['"]?
        (?:\s+evidence\s*=\s*(?:'([^']{0,200})'|"([^"]{0,200})"|(\S+)))?
        \s*/?\s*>""",
    re.VERBOSE | re.IGNORECASE,
)
SELF_SIGNAL_WHITELIST = frozenset({
    # Agent-emitted (via dispatch instruction — unreliable, bonus data)
    "tool-failed",
    "corrected-mistake",
    "user-pushback",
    "repeated-attempt",
    "gap-noticed",
    "convention-violated",
    "time-stuck",
    # Structurally injected by Detector F
    "user-steering",
    # Structurally injected by PostToolUse hook
    "tool-file-already-exists",
    "tool-file-not-found",
    "tool-permission-denied",
    "tool-timeout",
    "tool-generic-error",
    # Structurally injected by forced transcript introspection (UserPromptSubmit)
    "agent-admitted-mistake",
    "agent-retry",
    "agent-missed-info",
    "agent-conceded",
    "agent-backtrack",
    "agent-polling-announce",
    "agent-wait-announce",
})


def _find_self_signals(text: str) -> list[tuple[str, str]]:
    """Return [(type, evidence), ...] for whitelisted self-signal tags in text."""
    if not text:
        return []
    out = []
    seen_types = set()  # dedup per-reply per spec
    for m in SELF_SIGNAL_RX.finditer(text):
        sig_type = (m.group("type") or "").strip().lower()
        if sig_type not in SELF_SIGNAL_WHITELIST:
            continue
        if sig_type in seen_types:
            continue
        seen_types.add(sig_type)
        evidence = next(
            (g for g in (m.group(2), m.group(3), m.group(4)) if g),
            "",
        )[:200]
        out.append((sig_type, evidence))
    return out


def process_session(
    conn: sqlite3.Connection,
    ws_id: str,
    workspace_path: str | None,
    session_id: str,
    log_path: Path,
) -> tuple[int, int]:
    """Parse one session's main.jsonl into raw event tables.
    Returns (n_events, n_signal_rows). Idempotent via INSERT OR REPLACE on sessions
    and dedup via already_processed() at caller."""
    n_events = 0
    n_signals = 0
    first_ts = None
    last_ts = None
    user_msg_rows = []
    tool_call_rows = []
    agent_signal_rows = []
    user_correction_rows = []
    discovered_rows = []
    self_signal_rows = []
    workspace_label = _workspace_label(workspace_path)

    for ev in _iter_jsonl(log_path):
        n_events += 1
        ts = ev.get("ts")
        if isinstance(ts, int):
            first_ts = ts if first_ts is None or ts < first_ts else first_ts
            last_ts = ts if last_ts is None or ts > last_ts else last_ts
        etype = ev.get("type")
        attrs = ev.get("attrs") or {}

        if etype == "user_message":
            content = attrs.get("content", "") or ""
            user_msg_rows.append((session_id, ts or 0, content))
            for phrase, snippet in _find_matches(content, USER_RX):
                user_correction_rows.append((session_id, ts or 0, phrase, snippet))
                n_signals += 1
        elif etype == "tool_call":
            status = ev.get("status", "ok")
            name = ev.get("name", "")
            err = ""
            if status == "error":
                err = (attrs.get("error") or attrs.get("result") or "")[:200]
            tool_call_rows.append((session_id, ts or 0, name, status, err))
        elif etype == "agent_response":
            response_raw = attrs.get("response", "") or ""
            reasoning_raw = attrs.get("reasoning", "") or ""
            combined = ""
            if response_raw:
                try:
                    parts = json.loads(response_raw)
                    if isinstance(parts, list):
                        for p in parts:
                            if isinstance(p, dict):
                                c = p.get("content")
                                if isinstance(c, str):
                                    combined += " " + c
                                elif isinstance(c, list):
                                    for inner in c:
                                        if isinstance(inner, dict) and isinstance(
                                            inner.get("text"), str
                                        ):
                                            combined += " " + inner["text"]
                except json.JSONDecodeError:
                    combined += " " + response_raw
            combined += " " + reasoning_raw
            for phrase, snippet in _find_matches(combined, AGENT_RX):
                agent_signal_rows.append((session_id, ts or 0, phrase, snippet))
                n_signals += 1
            # Self-signal tags: structured friction reports from the agent.
            for sig_type, evidence in _find_self_signals(combined):
                self_signal_rows.append(
                    (session_id, ts or 0, sig_type, evidence, workspace_label)
                )
                n_signals += 1
        elif etype == "discovery":
            name = ev.get("name", "")
            details = (attrs.get("details") or "")[:1000]
            m = re.search(r"loaded:\s*\[([^\]]+)\]", details)
            if m:
                names = [n.strip() for n in m.group(1).split(",") if n.strip()]
                discovered_rows.append((session_id, name, json.dumps(names)))

    conn.execute("BEGIN")
    try:
        conn.execute(
            "INSERT OR REPLACE INTO sessions"
            "(session_id, workspace_storage_id, workspace_path, log_path,"
            " first_event_ts, last_event_ts, total_events)"
            " VALUES (?,?,?,?,?,?,?)",
            (
                session_id, ws_id, workspace_path, str(log_path),
                first_ts, last_ts, n_events,
            ),
        )
        if user_msg_rows:
            conn.executemany(
                "INSERT INTO user_messages(session_id, ts, content) VALUES (?,?,?)",
                user_msg_rows,
            )
        if tool_call_rows:
            conn.executemany(
                "INSERT INTO tool_calls(session_id, ts, name, status, error_excerpt)"
                " VALUES (?,?,?,?,?)",
                tool_call_rows,
            )
        if agent_signal_rows:
            conn.executemany(
                "INSERT INTO agent_self_signals(session_id, ts, phrase, snippet)"
                " VALUES (?,?,?,?)",
                agent_signal_rows,
            )
        if user_correction_rows:
            conn.executemany(
                "INSERT INTO user_correction_signals(session_id, ts, phrase, snippet)"
                " VALUES (?,?,?,?)",
                user_correction_rows,
            )
        if discovered_rows:
            conn.executemany(
                "INSERT INTO discovered_rules(session_id, discovery_type, rule_names_json)"
                " VALUES (?,?,?)",
                discovered_rows,
            )
        if self_signal_rows:
            conn.executemany(
                "INSERT INTO self_signals(session_id, ts, type, evidence, workspace)"
                " VALUES (?,?,?,?,?)",
                self_signal_rows,
            )
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
    return n_events, n_signals


def already_processed(conn: sqlite3.Connection, session_id: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM sessions WHERE session_id = ? LIMIT 1", (session_id,)
    ).fetchone() is not None


def extract_incremental(verbose: bool = False) -> dict:
    """Append-only ingest. Returns summary stats."""
    started = time.time()
    conn = open_db()
    n_new = 0
    n_events = 0
    n_signals = 0
    try:
        root = _workspace_storage_root()
        if root is None:
            return {"error": "no workspaceStorage dir found"}
        sessions = _find_sessions(root)
        for ws_id, sess_id, log_path in sessions:
            if already_processed(conn, sess_id):
                continue
            wp = _read_workspace_path(root / ws_id)
            try:
                ev, sg = process_session(conn, ws_id, wp, sess_id, log_path)
            except Exception as e:
                sys.stderr.write(f"warn: failed {sess_id}: {e}\n")
                continue
            n_new += 1
            n_events += ev
            n_signals += sg
            if verbose:
                sys.stderr.write(f"  + {sess_id[:8]} ({ws_id[:8]}) {ev} ev, {sg} sig\n")
        return {
            "new_sessions": n_new,
            "new_events": n_events,
            "new_signals": n_signals,
            "duration_s": round(time.time() - started, 3),
        }
    finally:
        conn.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    result = extract_incremental(verbose=args.verbose)
    print(json.dumps(result, indent=2))
    return 0 if "error" not in result else 1


if __name__ == "__main__":
    sys.exit(main())
