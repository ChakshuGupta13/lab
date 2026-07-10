#!/usr/bin/env python3
"""
v2 schema for the rolling-learning pipeline.

State lives at $LEARNING_STATE_ROOT (defaults to ~/.copilot/learning-pipeline/v2).
The v1 directory (~/.copilot/learning-pipeline/v1) is left intact — v2 starts fresh.

Tables (v2 additions in CAPS):
  Inherited raw-event tables (sessions, user_messages, tool_calls, agent_self_signals,
    user_correction_signals, discovered_rules) — unchanged from v1.
  PATTERNS                 -- one row per detected pattern (stable pid)
  PATTERN_OBSERVATIONS     -- append-only time series per pattern
  PATTERN_FIX_HISTORY      -- one row per apply, with outcome
  SURFACING_STATE          -- open surfacings per (pid, session)
  LM_EVAL_STATE            -- per-pid LM evaluation metadata + rate limit
  CACHED_SURFACING         -- pre-computed surfacings the hook can emit without work

Hot-path discipline: the UserPromptSubmit hook MUST NOT do expensive aggregation.
All counts are denormalized in `patterns.n_observations` updated by the detector.
"""

from __future__ import annotations

import os
import sqlite3
import sys
from pathlib import Path


SCHEMA_VERSION = "2"


def state_root() -> Path:
    """v2 state root, overridable via LEARNING_STATE_ROOT for tests."""
    env = os.environ.get("LEARNING_STATE_ROOT")
    if env:
        return Path(env).expanduser()
    return Path.home() / ".copilot" / "learning-pipeline" / "v2"


def state_dir() -> Path:
    return state_root() / "state"


def db_path() -> Path:
    return state_dir() / "events.db"


# Reuse v1 raw-event schema verbatim. Detector code reads from these.
V1_RAW_EVENT_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    workspace_storage_id TEXT,
    workspace_path TEXT,
    log_path TEXT,
    first_event_ts INTEGER,
    last_event_ts INTEGER,
    total_events INTEGER,
    processed_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);

CREATE TABLE IF NOT EXISTS user_messages (
    session_id TEXT NOT NULL,
    ts INTEGER NOT NULL,
    content TEXT,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);
CREATE INDEX IF NOT EXISTS idx_user_messages_session ON user_messages(session_id);
CREATE INDEX IF NOT EXISTS idx_user_messages_ts ON user_messages(ts);

CREATE TABLE IF NOT EXISTS tool_calls (
    session_id TEXT NOT NULL,
    ts INTEGER NOT NULL,
    name TEXT,
    status TEXT,
    error_excerpt TEXT,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);
CREATE INDEX IF NOT EXISTS idx_tool_calls_session ON tool_calls(session_id);
CREATE INDEX IF NOT EXISTS idx_tool_calls_error ON tool_calls(status, name);

CREATE TABLE IF NOT EXISTS agent_self_signals (
    session_id TEXT NOT NULL,
    ts INTEGER NOT NULL,
    phrase TEXT,
    snippet TEXT,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);
CREATE INDEX IF NOT EXISTS idx_self_signals_session ON agent_self_signals(session_id);

CREATE TABLE IF NOT EXISTS discovered_rules (
    session_id TEXT NOT NULL,
    discovery_type TEXT,
    rule_names_json TEXT,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);

CREATE TABLE IF NOT EXISTS user_correction_signals (
    session_id TEXT NOT NULL,
    ts INTEGER NOT NULL,
    phrase TEXT,
    snippet TEXT,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);
CREATE INDEX IF NOT EXISTS idx_correction_session ON user_correction_signals(session_id);
CREATE INDEX IF NOT EXISTS idx_correction_phrase ON user_correction_signals(phrase);

CREATE TABLE IF NOT EXISTS pipeline_meta (
    key TEXT PRIMARY KEY,
    value TEXT
);
"""

V2_SCHEMA = """
-- One row per detected pattern.
-- pid = stable hash of (detector, normalized_key) — see detector key-schema docs.
-- status transitions: OBSERVE -> ACTIVE (LM surfaced) -> {VALIDATED_T1, VALIDATED_T2,
--   STRUCTURAL_GAP, DISMISSED}.  Patterns NEVER deleted.
CREATE TABLE IF NOT EXISTS patterns (
    pid TEXT PRIMARY KEY,
    detector TEXT NOT NULL,
    key TEXT NOT NULL,
    workspace TEXT,                -- "same workspace" matching context (CRITICAL #9 fix)
    first_seen_ts INTEGER NOT NULL,
    last_seen_ts INTEGER NOT NULL,
    n_observations INTEGER NOT NULL DEFAULT 0,   -- denormalized count (hot-path)
    n_sessions INTEGER NOT NULL DEFAULT 0,        -- denormalized distinct sessions
    status TEXT NOT NULL DEFAULT 'OBSERVE',
    created_at INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_patterns_status ON patterns(status);
CREATE INDEX IF NOT EXISTS idx_patterns_workspace ON patterns(workspace);

-- Append-only time series of pattern observations.
CREATE TABLE IF NOT EXISTS pattern_observations (
    pid TEXT NOT NULL,
    ts INTEGER NOT NULL,
    session_id TEXT NOT NULL,
    workspace TEXT,
    evidence TEXT,
    FOREIGN KEY (pid) REFERENCES patterns(pid)
);
CREATE INDEX IF NOT EXISTS idx_obs_pid_ts ON pattern_observations(pid, ts);
CREATE INDEX IF NOT EXISTS idx_obs_pid_session ON pattern_observations(pid, session_id);

-- One row per applied fix (TIER1, TIER2, TIER3).
-- TIER1 fixes commit_sha is NULL (ephemeral additionalContext nudge — no commit).
CREATE TABLE IF NOT EXISTS pattern_fix_history (
    fix_id INTEGER PRIMARY KEY AUTOINCREMENT,
    pid TEXT NOT NULL,
    applied_at INTEGER NOT NULL,
    tier INTEGER NOT NULL,            -- 1, 2, or 3
    commit_sha TEXT,                  -- NULL for TIER1
    target_paths TEXT,                -- JSON array of paths touched
    watch_window_K INTEGER NOT NULL,
    target_rate_drop REAL NOT NULL,
    pre_rate REAL NOT NULL,
    -- Outcome filled in by watch_effectiveness.py.
    outcome TEXT,
    -- 'VALIDATED' | 'SUGGEST_REVERT' | 'INSUFFICIENT_DATA' | 'ESCALATE' | NULL
    outcome_at INTEGER,
    post_rate REAL,
    n_post_sessions INTEGER,
    FOREIGN KEY (pid) REFERENCES patterns(pid)
);
CREATE INDEX IF NOT EXISTS idx_fix_pid ON pattern_fix_history(pid);
CREATE INDEX IF NOT EXISTS idx_fix_outcome ON pattern_fix_history(outcome);

-- One row per surfacing emitted into a session.
CREATE TABLE IF NOT EXISTS surfacing_state (
    surfacing_id INTEGER PRIMARY KEY AUTOINCREMENT,
    pid TEXT NOT NULL,
    session_id TEXT NOT NULL,
    surfaced_at INTEGER NOT NULL,
    surfaced_at_turn INTEGER NOT NULL,
    ttl_turns INTEGER NOT NULL DEFAULT 3,
    status TEXT NOT NULL DEFAULT 'OPEN',
    -- 'OPEN' | 'DISPATCHED' | 'EXPIRED'
    dispatched_intent TEXT,
    -- 'apply' | 'dismiss' | 'refine' | NULL
    dispatched_at INTEGER,
    FOREIGN KEY (pid) REFERENCES patterns(pid)
);
CREATE INDEX IF NOT EXISTS idx_surf_session_status ON surfacing_state(session_id, status);
CREATE INDEX IF NOT EXISTS idx_surf_pid ON surfacing_state(pid);

-- LM evaluation rate limiting (CRITICAL #3 fix).
CREATE TABLE IF NOT EXISTS lm_eval_state (
    pid TEXT PRIMARY KEY,
    last_eval_at INTEGER,              -- unix epoch seconds
    last_eval_n_observations INTEGER NOT NULL DEFAULT 0,
    last_verdict TEXT,                 -- 'OBSERVE' | 'TIER1' | 'TIER2' | 'TIER3' | NULL
    next_check_after_n_observations INTEGER NOT NULL DEFAULT 1,
    min_seconds_between_evals INTEGER NOT NULL DEFAULT 86400,  -- 24h floor
    FOREIGN KEY (pid) REFERENCES patterns(pid)
);

-- Pre-computed surfacings the hook can emit without doing real work.
-- Updated only by the slow-path pipeline. Hot path is read-only.
CREATE TABLE IF NOT EXISTS cached_surfacing (
    pid TEXT PRIMARY KEY,
    tier INTEGER NOT NULL,
    surfacing_blurb TEXT NOT NULL,     -- short text injected as additionalContext
    proposal_path TEXT,                -- path to .github/learning/staging/<pid>/proposal.json
    workspace TEXT,                    -- only emit in matching workspace
    ready_at INTEGER NOT NULL,
    expires_at INTEGER,                -- absolute epoch when the cache row is stale
    FOREIGN KEY (pid) REFERENCES patterns(pid)
);
CREATE INDEX IF NOT EXISTS idx_cached_workspace ON cached_surfacing(workspace);

-- Per-session turn counter — increments on every UserPromptSubmit, used by TTL math.
CREATE TABLE IF NOT EXISTS session_turn_counter (
    session_id TEXT PRIMARY KEY,
    last_turn INTEGER NOT NULL DEFAULT 0,
    updated_at INTEGER NOT NULL
);

-- Self-signals emitted by chat agents via <self-signal type=X evidence='...'/> tags.
-- Strict whitelist of types (see learning-pipeline-dispatch.instructions.md);
-- extractor discards rows with unknown type. The L2 pattern miner reads from
-- this table; threshold to trigger L2 = N new self_signals since last l2_runs row.
CREATE TABLE IF NOT EXISTS self_signals (
    session_id TEXT NOT NULL,
    ts INTEGER NOT NULL,                -- milliseconds (v1 extractor convention)
    type TEXT NOT NULL,                 -- one of: tool-failed, corrected-mistake,
                                        --  user-pushback, repeated-attempt,
                                        --  gap-noticed, convention-violated, time-stuck
    evidence TEXT,                      -- <=60 chars excerpt, may be empty
    workspace TEXT,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);
CREATE INDEX IF NOT EXISTS idx_self_session_ts ON self_signals(session_id, ts);
CREATE INDEX IF NOT EXISTS idx_self_type ON self_signals(type);

-- L2 (pattern miner) run history. One row per L2 invocation per session.
-- Used to decide when to fire next L2: trigger iff
-- (SELECT COUNT(*) FROM self_signals WHERE session_id=? AND ts > last_run_ts) >= 3.
CREATE TABLE IF NOT EXISTS l2_runs (
    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    workspace TEXT,
    run_at INTEGER NOT NULL,             -- unix seconds
    signals_count INTEGER NOT NULL,      -- how many signals this run processed
    features_emitted INTEGER NOT NULL DEFAULT 0,
    ok INTEGER NOT NULL DEFAULT 0,       -- 0 = error, 1 = success
    reason TEXT,                         -- error msg or 'ok'
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);
CREATE INDEX IF NOT EXISTS idx_l2_session ON l2_runs(session_id, run_at);

-- L2 output queue: feature observations emitted by pattern miner, waiting to
-- be drained by slow path into pattern_observations. Append-only; slow path
-- moves rows to pattern_observations (via feature_name → detector mapping) and
-- DELETEs them.
CREATE TABLE IF NOT EXISTS l2_feature_queue (
    queue_id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    workspace TEXT,
    feature_name TEXT NOT NULL,
    evidence TEXT,
    confidence REAL,
    enqueued_at INTEGER NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);
CREATE INDEX IF NOT EXISTS idx_l2_queue_session ON l2_feature_queue(session_id);
CREATE INDEX IF NOT EXISTS idx_l2_queue_feature ON l2_feature_queue(feature_name);
"""


# Template-instantiation columns on pattern_fix_history (integration with
# .github/mechanism-templates/). Added as a backward-compatible in-place
# migration rather than a SCHEMA_VERSION bump: the columns are nullable and
# never read by non-template code paths (watcher and apply only consult them
# when template_id IS NOT NULL). See .github/learning/integration-design.md
# Decision 6. CREATE TABLE IF NOT EXISTS in V2_SCHEMA does NOT add columns to
# an existing table, so existing v2 DBs need this explicit ALTER pass.
_TEMPLATE_COLUMNS = (
    ("template_id", "TEXT"),            # catalog template id, e.g. 'T1' (nullable)
    ("template_params_json", "TEXT"),   # serialized template.params (nullable)
    ("template_metric_json", "TEXT"),   # per-template watcher metric (nullable)
)


def _migrate_v2_add_template_cols(conn: sqlite3.Connection) -> None:
    """Idempotently add the template_* columns to pattern_fix_history.

    Safe to call on every open: it checks PRAGMA table_info first and only
    ALTERs columns that are absent. Does NOT bump SCHEMA_VERSION — the columns
    are nullable additions that old code never reads.
    """
    existing = {
        row[1]  # row = (cid, name, type, notnull, dflt_value, pk)
        for row in conn.execute("PRAGMA table_info(pattern_fix_history)")
    }
    for name, coltype in _TEMPLATE_COLUMNS:
        if name not in existing:
            conn.execute(
                f"ALTER TABLE pattern_fix_history ADD COLUMN {name} {coltype}"
            )


def open_db(path: Path | None = None) -> sqlite3.Connection:
    """Open v2 DB, applying schema idempotently. Caller closes."""
    p = path or db_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(p), isolation_level=None, timeout=5.0)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.executescript(V1_RAW_EVENT_SCHEMA)
    conn.executescript(V2_SCHEMA)
    # In-place column migration for template integration. Runs unconditionally
    # (idempotent via PRAGMA check) so an existing v2 DB gains the columns the
    # first time new code opens it. Must run AFTER the table exists (above) and
    # BEFORE the version check (below), so it applies regardless of whether the
    # schema_version meta row is present yet.
    _migrate_v2_add_template_cols(conn)
    # Schema-version stamping, with mismatch detection.
    cur = conn.execute("SELECT value FROM pipeline_meta WHERE key='schema_version'")
    row = cur.fetchone()
    if row is None:
        conn.execute(
            "INSERT INTO pipeline_meta(key, value) VALUES ('schema_version', ?)",
            (SCHEMA_VERSION,),
        )
    elif row[0] != SCHEMA_VERSION:
        sys.stderr.write(
            f"events.db schema mismatch: file is v{row[0]}, code is v{SCHEMA_VERSION}. "
            "Move ~/.copilot/learning-pipeline/v2/state/events.db aside to rebuild.\n"
        )
        sys.exit(2)
    return conn


def smoke_test() -> int:
    """Self-test: open the DB, list tables, print row counts."""
    conn = open_db()
    try:
        tables = [
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            )
        ]
        print(f"state_root: {state_root()}")
        print(f"db_path:    {db_path()}")
        print(f"tables ({len(tables)}):")
        for t in tables:
            n = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            print(f"  {t:30s}  {n} rows")
        meta = dict(conn.execute("SELECT key, value FROM pipeline_meta").fetchall())
        print(f"meta: {meta}")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(smoke_test())
