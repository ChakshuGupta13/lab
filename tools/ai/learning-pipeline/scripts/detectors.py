#!/usr/bin/env python3
"""
Detectors — produce pattern observations from the raw event tables.

Each detector defines:
  - a stable name (used in pid hash)
  - a key schema (documented inline) — see KEY_SCHEMA constants
  - a function that scans raw events and inserts into pattern_observations,
    updating patterns + maintaining denormalized counts.

CRITICAL: detectors are idempotent. Re-running a detector must not double-count.
We use (pid, ts, session_id) as the dedup tuple.

CRITICAL fix from Adversary concern #2: every detector documents its key schema
explicitly so the LM evaluator sees consistent identity across reruns.

CRITICAL fix from Adversary concern #9: every pattern has a workspace, used as
the "matching context" for effectiveness checks.

Adapted from v1 extract_signals.py emit_signal_{a,b,c,d} but writes to v2
pattern tables instead of emitting JSONL.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import sys
import time
from typing import Iterable


# Re-use v1 phrase lists. These define the "key" component for some detectors.
USER_CORRECTION_PHRASES = [
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

AGENT_SELF_CORRECTION_PHRASES = [
    r"\b(?:let me |i'?ll |i will |should |need to )stop polling\b",
    r"\b(?:let me |i'?ll |i will |should |need to )stop checking\b",
    r"\bi'?ll wait for the (?:terminal )?notification\b",
    r"\bi'?ll check (?:back )?(?:in|later|when)\b",
]


def _pid(detector: str, key: str) -> str:
    """Stable per-pattern id (10 hex chars)."""
    h = hashlib.sha256(f"{detector}|{key}".encode("utf-8")).hexdigest()
    return h[:10]


def _workspace_label(workspace_path: str | None) -> str:
    """Map workspace URI → basename (used as 'matching context')."""
    if not workspace_path:
        return "<unknown>"
    return workspace_path.rstrip("/").split("/")[-1] or workspace_path


def _ensure_pattern(
    conn: sqlite3.Connection,
    pid: str,
    detector: str,
    key: str,
    workspace: str,
    ts: int,
) -> None:
    """Insert pattern row if missing. Workspace is set at first observation;
    if the same pattern fires in a different workspace, we create a new pid
    via the detector adding workspace to its key — by design (per Adversary
    #9). Detectors that aggregate cross-workspace use workspace='*' as key."""
    conn.execute(
        "INSERT OR IGNORE INTO patterns "
        "(pid, detector, key, workspace, first_seen_ts, last_seen_ts, "
        " n_observations, n_sessions, status, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, 0, 0, 'OBSERVE', ?)",
        (pid, detector, key, workspace, ts, ts, int(time.time())),
    )


def _record_observation(
    conn: sqlite3.Connection,
    pid: str,
    ts: int,
    session_id: str,
    workspace: str,
    evidence: str,
) -> bool:
    """Insert one observation. Returns True if new (deduped on pid+ts+session).
    Updates denormalized counts on success."""
    # Dedup. Composite check is fine — pattern_observations has idx_obs_pid_ts +
    # idx_obs_pid_session, so the lookup is fast.
    existing = conn.execute(
        "SELECT 1 FROM pattern_observations WHERE pid=? AND ts=? AND session_id=? LIMIT 1",
        (pid, ts, session_id),
    ).fetchone()
    if existing:
        return False
    conn.execute(
        "INSERT INTO pattern_observations (pid, ts, session_id, workspace, evidence) "
        "VALUES (?, ?, ?, ?, ?)",
        (pid, ts, session_id, workspace, evidence[:400]),
    )
    # Update denormalized counts. n_sessions is a distinct-count which we
    # recompute lazily here only when a new session_id is observed (cheap when
    # observations are dominated by repeats within a session).
    conn.execute(
        "UPDATE patterns SET "
        "  n_observations = n_observations + 1, "
        "  last_seen_ts = MAX(last_seen_ts, ?), "
        "  n_sessions = (SELECT COUNT(DISTINCT session_id) FROM pattern_observations WHERE pid=?) "
        "WHERE pid=?",
        (ts, pid, pid),
    )
    return True


# ---------------------------------------------------------------------------
# Detector A: intra-session agent self-correction (X-window)
# Key schema:  phrase  (the regex source that matched)
# Workspace:   the session's workspace label (so same phrase in different
#              workspace = different pid; matches Adversary #9 design)
# Observations: per agent_self_signals row.
# ---------------------------------------------------------------------------
DETECTOR_A = "intra_session_self_correction"

def run_detector_a(conn: sqlite3.Connection) -> int:
    """Walk agent_self_signals, materialise pattern + observations."""
    n = 0
    rows = conn.execute(
        "SELECT a.session_id, a.ts, a.phrase, a.snippet, s.workspace_path "
        "FROM agent_self_signals a "
        "LEFT JOIN sessions s ON s.session_id = a.session_id "
        "WHERE a.ts > 0"
    ).fetchall()
    for session_id, ts, phrase, snippet, wp in rows:
        workspace = _workspace_label(wp)
        key = f"{phrase}|{workspace}"
        pid = _pid(DETECTOR_A, key)
        _ensure_pattern(conn, pid, DETECTOR_A, key, workspace, ts)
        if _record_observation(conn, pid, ts, session_id, workspace, snippet or ""):
            n += 1
    return n


# ---------------------------------------------------------------------------
# Detector B: cross-session user-correction phrase recurrence (Y-window)
# Key schema:  phrase  (regex source)
# Workspace:   the session's workspace label
# Observations: per user_correction_signals row.
# ---------------------------------------------------------------------------
DETECTOR_B = "cross_session_user_correction"

def run_detector_b(conn: sqlite3.Connection) -> int:
    n = 0
    rows = conn.execute(
        "SELECT u.session_id, u.ts, u.phrase, u.snippet, s.workspace_path "
        "FROM user_correction_signals u "
        "LEFT JOIN sessions s ON s.session_id = u.session_id "
        "WHERE u.ts > 0"
    ).fetchall()
    for session_id, ts, phrase, snippet, wp in rows:
        workspace = _workspace_label(wp)
        key = f"{phrase}|{workspace}"
        pid = _pid(DETECTOR_B, key)
        _ensure_pattern(conn, pid, DETECTOR_B, key, workspace, ts)
        if _record_observation(conn, pid, ts, session_id, workspace, snippet or ""):
            n += 1
    return n


# ---------------------------------------------------------------------------
# Detector C: recurring tool error (Z-window — slow accumulator)
# Key schema:  tool_name|error_excerpt[:80]|workspace
# Filters out user-cancellation noise.
# Observations: one per tool_calls row with status='error' (excluding noise).
# ---------------------------------------------------------------------------
DETECTOR_C = "tool_error_recurrence"

TOOL_ERROR_NOISE_SUBSTRINGS = ("canceled", "cancelled", "user canceled")

def run_detector_c(conn: sqlite3.Connection) -> int:
    n = 0
    rows = conn.execute(
        "SELECT t.session_id, t.ts, t.name, t.error_excerpt, s.workspace_path "
        "FROM tool_calls t "
        "LEFT JOIN sessions s ON s.session_id = t.session_id "
        "WHERE t.status = 'error' AND t.error_excerpt <> '' AND t.ts > 0"
    ).fetchall()
    for session_id, ts, name, err, wp in rows:
        if err is None:
            continue
        err_lower = err.lower()
        if any(noise in err_lower for noise in TOOL_ERROR_NOISE_SUBSTRINGS):
            continue
        excerpt = err[:80]
        workspace = _workspace_label(wp)
        key = f"{name}|{excerpt}|{workspace}"
        pid = _pid(DETECTOR_C, key)
        _ensure_pattern(conn, pid, DETECTOR_C, key, workspace, ts)
        if _record_observation(conn, pid, ts, session_id, workspace, err[:400]):
            n += 1
    return n


# ---------------------------------------------------------------------------
# Detector D: rule-on-disk-with-corrections (the v1 "Signal A" — advisory).
# Each session with >=3 user corrections AND a rule-discovery event materialises
# one observation. Key = "session_with_corrections_despite_rules" — we surface
# this as a single meta-pattern per workspace (LM looks at it to spot rule-prompt
# mismatches).
# Key schema:  literal "rules_on_disk"  (per-workspace via workspace dimension)
# ---------------------------------------------------------------------------
DETECTOR_D = "rules_on_disk_with_corrections"

def run_detector_d(conn: sqlite3.Connection) -> int:
    """Materialise one observation per session that satisfies the condition."""
    n = 0
    # Sessions with >=3 distinct correction phrases, and at least one discovery row.
    rows = conn.execute(
        """
        SELECT s.session_id, s.last_event_ts, s.workspace_path,
               COUNT(u.phrase) AS n_corr
        FROM sessions s
        JOIN user_correction_signals u ON u.session_id = s.session_id
        WHERE EXISTS (
            SELECT 1 FROM discovered_rules d
            WHERE d.session_id = s.session_id
              AND d.discovery_type = 'Instructions Discovery'
        )
        GROUP BY s.session_id
        HAVING n_corr >= 3
        """
    ).fetchall()
    for session_id, ts, wp, n_corr in rows:
        workspace = _workspace_label(wp)
        key = f"rules_on_disk|{workspace}"
        pid = _pid(DETECTOR_D, key)
        _ensure_pattern(conn, pid, DETECTOR_D, key, workspace, ts)
        evidence = json.dumps({"n_corrections": n_corr, "session_id": session_id})
        if _record_observation(conn, pid, ts, session_id, workspace, evidence):
            n += 1
    return n


DETECTORS = [
    (DETECTOR_A, run_detector_a),
    (DETECTOR_B, run_detector_b),
    (DETECTOR_C, run_detector_c),
    (DETECTOR_D, run_detector_d),
    # Detector E added below.
]


# ---------------------------------------------------------------------------
# Detector E: self-signal recurrence (agent-emitted friction tags).
# Key schema:  type|workspace
# Observations: per self_signals row.
# This is the structural counterpart to Detector A/B but driven by the chat
# agent's own admitted friction (whitelisted tag types), not regex on phrasing.
# ---------------------------------------------------------------------------
DETECTOR_E = "self_signal_recurrence"


def run_detector_e(conn: sqlite3.Connection) -> int:
    n = 0
    rows = conn.execute(
        "SELECT session_id, ts, type, evidence, workspace FROM self_signals "
        "WHERE ts > 0"
    ).fetchall()
    for session_id, ts, sig_type, evidence, workspace in rows:
        workspace = workspace or "<unknown>"
        key = f"{sig_type}|{workspace}"
        pid = _pid(DETECTOR_E, key)
        _ensure_pattern(conn, pid, DETECTOR_E, key, workspace, ts)
        if _record_observation(conn, pid, ts, session_id, workspace, evidence or ""):
            n += 1
    return n


DETECTORS.append((DETECTOR_E, run_detector_e))


# ---------------------------------------------------------------------------
# Detector F: user steering (consecutive user messages without agent turn).
# When the user sends a new prompt before the agent finishes responding, the
# prior response is truncated. This is the strongest implicit friction signal:
# the user SAW the agent going wrong and interrupted. Self-signals are lost
# in this scenario because they're appended at the end of the (truncated) reply.
# Key schema:  "user_steering"|workspace
# Detection: for each session, sort user_messages by ts. When two consecutive
# messages have no agent_self_signals row (our proxy for completed agent turn)
# between them, count it as one steering event. Evidence = the 2nd message
# (the corrective prompt).
# ---------------------------------------------------------------------------
DETECTOR_F = "user_steering"


def run_detector_f(conn: sqlite3.Connection) -> int:
    """Detect user-interruption steering events from consecutive user_messages."""
    n = 0
    # Get all sessions that have user_messages.
    sessions = conn.execute(
        "SELECT DISTINCT session_id FROM user_messages"
    ).fetchall()
    for (session_id,) in sessions:
        ws_row = conn.execute(
            "SELECT workspace_path FROM sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        workspace = _workspace_label(ws_row[0] if ws_row else None)

        # Get all user_message timestamps for this session, ordered.
        msgs = conn.execute(
            "SELECT ts, content FROM user_messages "
            "WHERE session_id = ? ORDER BY ts",
            (session_id,),
        ).fetchall()
        if len(msgs) < 2:
            continue

        for i in range(1, len(msgs)):
            prev_ts = msgs[i - 1][0]
            curr_ts = msgs[i][0]
            curr_content = msgs[i][1] or ""
            # Check: was there any agent activity between prev and curr?
            # Use agent_self_signals as proxy for "agent responded" — if zero
            # agent signals exist between these two user msgs AND the gap is
            # < 120 seconds (rapid correction, not a natural pause), count it.
            gap_ms = curr_ts - prev_ts
            if gap_ms > 120_000:
                # >2 min gap = probably a natural turn, not a steer
                continue
            agent_between = conn.execute(
                "SELECT COUNT(*) FROM agent_self_signals "
                "WHERE session_id = ? AND ts > ? AND ts < ?",
                (session_id, prev_ts, curr_ts),
            ).fetchone()[0]
            if agent_between > 0:
                continue
            # Also check tool_calls as a more reliable proxy for "agent was active"
            tools_between = conn.execute(
                "SELECT COUNT(*) FROM tool_calls "
                "WHERE session_id = ? AND ts > ? AND ts < ?",
                (session_id, prev_ts, curr_ts),
            ).fetchone()[0]
            if tools_between > 2:
                # Agent did substantial work between msgs — not a steer
                continue
            # This looks like a steering event: 2 user msgs in <2 min with
            # minimal agent activity between them.
            key = f"user_steering|{workspace}"
            pid = _pid(DETECTOR_F, key)
            _ensure_pattern(conn, pid, DETECTOR_F, key, workspace, curr_ts)
            evidence = curr_content[:200]
            if _record_observation(conn, pid, curr_ts, session_id, workspace, evidence):
                n += 1
                # Also write to self_signals so the L2 trigger (which counts
                # self_signals rows) sees steering events. This is the implicit
                # version of self-signal: the USER indicated friction by
                # interrupting, not the agent self-reporting.
                existing = conn.execute(
                    "SELECT 1 FROM self_signals WHERE session_id=? AND ts=? AND type='user-steering' LIMIT 1",
                    (session_id, curr_ts),
                ).fetchone()
                if not existing:
                    conn.execute(
                        "INSERT INTO self_signals (session_id, ts, type, evidence, workspace) "
                        "VALUES (?, ?, 'user-steering', ?, ?)",
                        (session_id, curr_ts, evidence[:200], workspace),
                    )
    return n


DETECTORS.append((DETECTOR_F, run_detector_f))


def run_all(conn: sqlite3.Connection, verbose: bool = False) -> dict:
    """Run every detector, return summary."""
    out = {}
    for name, fn in DETECTORS:
        try:
            n = fn(conn)
        except sqlite3.Error as e:
            out[name] = {"error": str(e)}
            if verbose:
                sys.stderr.write(f"detector {name}: ERROR {e}\n")
            continue
        out[name] = {"new_observations": n}
        if verbose:
            sys.stderr.write(f"detector {name}: +{n} obs\n")
    return out


def main() -> int:
    from .schema import open_db
    conn = open_db()
    try:
        result = run_all(conn, verbose=True)
        sys.stderr.write("\nSummary:\n")
        for name, info in result.items():
            sys.stderr.write(f"  {name:40s}  {info}\n")
        # Print a few patterns sorted by observation count.
        sys.stderr.write("\nTop 5 patterns by n_observations:\n")
        for row in conn.execute(
            "SELECT pid, detector, key, workspace, n_observations, n_sessions, status "
            "FROM patterns ORDER BY n_observations DESC LIMIT 5"
        ):
            sys.stderr.write(f"  {row}\n")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
