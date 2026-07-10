#!/usr/bin/env bash
#
# Slow-path runner for the rolling-learning pipeline v2.
#
# Spawned async by hook_user_prompt_submit.py when the cache is stale (>1h).
# Pipeline: extractor -> detectors -> drain L2 queue -> per-pattern LM authorer
# (rate-gated) -> updates cached_surfacing for the hot path.
#
# Atomic mkdir-lock per Adversary v1 #5. Per-pid locks are handled inside
# lm_authorer.py.
#
# SAFETY GATE (2026-06-05): refuses to run unless LEARNING_PIPELINE_ENABLED=1
# is in the environment. This matches the gate in lm_authorer.py and
# apply_proposal.py so a manual `bash run_slow_path.sh` invocation does not
# autonomously consume premium requests via the LM authorer step. Steps that
# do not call the LM (extractor, detectors, L2 queue drain, cache rebuild,
# watcher) are also gated together — simpler than partial gating, and the
# extractor's regex/SQL detectors do not need to run if no LM step will
# follow. To run without the gate (e.g., when reading-only state), use the
# Python modules directly: `python3 -m v2.extractor` etc.

set -u

if [ "${LEARNING_PIPELINE_ENABLED:-0}" != "1" ]; then
    echo "[run_slow_path] LEARNING_PIPELINE_ENABLED!=1; refusing to run (would otherwise spend premium requests)" >&2
    exit 0
fi

STATE_ROOT="$HOME/.copilot/learning-pipeline/v2"
STATE_DIR="$STATE_ROOT/state"
LOCK_DIR="$STATE_DIR/.pipeline.lock.dir"
LAST_RUN="$STATE_DIR/last-run.txt"
LOG_DIR="$STATE_ROOT/logs"

WORKSPACE_ROOT="${WORKSPACE_ROOT:-$PWD}"
SCRIPT_DIR="$WORKSPACE_ROOT/.github/scripts/learning"

mkdir -p "$STATE_DIR" "$LOG_DIR" || exit 1
LOG="$LOG_DIR/pipeline-$(date +%Y-%m-%d).log"

log() {
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >> "$LOG"
}

acquire_lock() {
    if mkdir "$LOCK_DIR" 2>/dev/null; then
        echo $$ > "$LOCK_DIR/pid"
        return 0
    fi
    if [ -f "$LOCK_DIR/pid" ]; then
        local owner_pid
        owner_pid=$(cat "$LOCK_DIR/pid" 2>/dev/null || echo "")
        if [ -n "$owner_pid" ] && ! kill -0 "$owner_pid" 2>/dev/null; then
            log "stale lock from $owner_pid; reclaim"
            rm -rf "$LOCK_DIR"
            mkdir "$LOCK_DIR" 2>/dev/null && { echo $$ > "$LOCK_DIR/pid"; return 0; }
        fi
    fi
    return 1
}

release_lock() { rm -rf "$LOCK_DIR"; }

if ! acquire_lock; then
    log "lock held; exit"
    exit 0
fi
trap release_lock EXIT

log "v2 slow-path start (pid=$$, root=$WORKSPACE_ROOT)"

PY="$(command -v python3)"
if [ -z "$PY" ]; then
    log "ERROR: python3 missing"
    exit 1
fi

cd "$SCRIPT_DIR" || { log "ERROR: cannot cd $SCRIPT_DIR"; exit 1; }

# Step 1: incremental extraction
log "extractor..."
"$PY" -m v2.extractor --verbose >> "$LOG" 2>&1 || log "extractor errored"

# Step 2: detectors
log "detectors..."
"$PY" -m v2.detectors >> "$LOG" 2>&1 || log "detectors errored"

# Step 2b: drain L2 feature queue into pattern_observations.
# Each queued feature maps to a pattern under detector='semantic_feature',
# key='<feature_name>|<workspace>'. Existing patterns get a new observation;
# new features create a new pattern row (status='OBSERVE') and start counting.
log "draining L2 feature queue..."
"$PY" <<PY_DRAIN >> "$LOG" 2>&1
import sys, time, hashlib
sys.path.insert(0, ".")
from v2.schema import open_db

def _pid(detector, key):
    return hashlib.sha256(f"{detector}|{key}".encode()).hexdigest()[:10]

conn = open_db()
try:
    rows = conn.execute(
        "SELECT queue_id, session_id, workspace, feature_name, evidence, "
        "       confidence, enqueued_at "
        "FROM l2_feature_queue ORDER BY enqueued_at"
    ).fetchall()
    drained = 0
    for queue_id, session_id, workspace, feature_name, evidence, confidence, enqueued_at in rows:
        detector = "semantic_feature"
        key = f"{feature_name}|{workspace}"
        pid = _pid(detector, key)
        ts_ms = enqueued_at * 1000
        # Ensure pattern row exists.
        conn.execute(
            "INSERT OR IGNORE INTO patterns "
            "(pid, detector, key, workspace, first_seen_ts, last_seen_ts, "
            " n_observations, n_sessions, status, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, 0, 0, 'OBSERVE', ?)",
            (pid, detector, key, workspace, ts_ms, ts_ms, enqueued_at),
        )
        # Dedup observation (pid, ts, session_id).
        if conn.execute(
            "SELECT 1 FROM pattern_observations WHERE pid=? AND ts=? AND session_id=? LIMIT 1",
            (pid, ts_ms, session_id),
        ).fetchone() is None:
            conn.execute(
                "INSERT INTO pattern_observations (pid, ts, session_id, workspace, evidence) "
                "VALUES (?, ?, ?, ?, ?)",
                (pid, ts_ms, session_id, workspace, (evidence or "")[:400]),
            )
            conn.execute(
                "UPDATE patterns SET "
                "  n_observations = n_observations + 1, "
                "  last_seen_ts = MAX(last_seen_ts, ?), "
                "  n_sessions = (SELECT COUNT(DISTINCT session_id) FROM pattern_observations WHERE pid=?) "
                "WHERE pid=?",
                (ts_ms, pid, pid),
            )
        # Remove from queue (idempotent — already in pattern_observations).
        conn.execute("DELETE FROM l2_feature_queue WHERE queue_id=?", (queue_id,))
        drained += 1
    print(f"drained {drained} L2 features")
finally:
    conn.close()
PY_DRAIN

# Step 3: LM authorer for OBSERVE patterns above threshold, rate-gated
# Pick patterns: status='OBSERVE' AND n_observations >= 3 AND not recently evaluated.
# Up to N_AUTHORED_PER_RUN per run.
N_AUTHORED_PER_RUN=3
log "authorer (up to $N_AUTHORED_PER_RUN patterns)..."

# Get candidate pids via SQL (cheap, no LM call yet).
CANDIDATES=$("$PY" <<PY_EOF 2>>"$LOG"
import sys
sys.path.insert(0, ".")
from v2.schema import open_db
from v2.lm_authorer import should_evaluate
conn = open_db()
try:
    rows = conn.execute(
        "SELECT p.pid FROM patterns p "
        "LEFT JOIN lm_eval_state e ON e.pid = p.pid "
        "WHERE p.status = 'OBSERVE' AND p.n_observations >= 3 "
        "ORDER BY p.n_observations DESC, p.last_seen_ts DESC LIMIT 20"
    ).fetchall()
    out = []
    for (pid,) in rows:
        ok, _ = should_evaluate(conn, pid)
        if ok:
            out.append(pid)
    for pid in out[:$N_AUTHORED_PER_RUN]:
        print(pid)
finally:
    conn.close()
PY_EOF
)

if [ -z "$CANDIDATES" ]; then
    log "no candidates eligible for authoring this run"
else
    for pid in $CANDIDATES; do
        log "authoring pid=$pid"
        "$PY" -m v2.lm_authorer --pid "$pid" >> "$LOG" 2>&1 || log "authorer for $pid errored"
    done
fi

# Step 4: rebuild cached_surfacing from validated proposals in staging.
log "rebuilding cached_surfacing..."
# Heredoc is UNQUOTED so $WORKSPACE_ROOT is shell-expanded into the python source.
# (Bug fix per Adversary post-build CRITICAL #1: previously quoted, so python
# saw the literal string '$WORKSPACE_ROOT' and the cache rebuild silently no-op'd.)
"$PY" <<PY_EOF >> "$LOG" 2>&1
import json, time, sys
from pathlib import Path
sys.path.insert(0, ".")
from v2.schema import open_db

WORKSPACE_ROOT = Path("$WORKSPACE_ROOT")
STAGING = WORKSPACE_ROOT / ".github" / "learning" / "staging"
conn = open_db()
try:
    # Clear cache for patterns no longer ACTIVE, OR dismissed, OR already applied.
    conn.execute("DELETE FROM cached_surfacing WHERE pid NOT IN (SELECT pid FROM patterns WHERE status='ACTIVE')")
    conn.execute("DELETE FROM cached_surfacing WHERE pid IN (SELECT pid FROM pattern_fix_history)")
    conn.execute("DELETE FROM cached_surfacing WHERE pid IN (SELECT pid FROM patterns WHERE status='DISMISSED')")
    if not STAGING.is_dir():
        print("no staging dir; nothing to cache")
    else:
        for pid_dir in STAGING.iterdir():
            if not pid_dir.is_dir():
                continue
            prop_path = pid_dir / "proposal.json"
            if not prop_path.is_file():
                continue
            try:
                prop = json.loads(prop_path.read_text())
            except Exception as e:
                print(f"  skip {pid_dir.name}: {e}")
                continue
            pid = prop.get("pid")
            if not prop.get("mature"):
                continue
            tier = prop.get("tier", 0)
            if tier == 0:
                continue
            blurb = prop.get("surfacing_blurb", "")[:280]
            pat = conn.execute("SELECT workspace, status FROM patterns WHERE pid=?", (pid,)).fetchone()
            if not pat:
                continue
            # Skip DISMISSED patterns — user already said no.
            if pat[1] == 'DISMISSED':
                continue
            # Skip if this pid already has an applied fix — re-surfacing would
            # cause "target file already exists" on second apply attempt.
            has_fix = conn.execute("SELECT 1 FROM pattern_fix_history WHERE pid=?", (pid,)).fetchone()
            if has_fix:
                continue
            workspace = pat[0]
            conn.execute(
                "INSERT INTO cached_surfacing (pid, tier, surfacing_blurb, proposal_path, workspace, ready_at, expires_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(pid) DO UPDATE SET "
                "  tier=excluded.tier, surfacing_blurb=excluded.surfacing_blurb, "
                "  proposal_path=excluded.proposal_path, workspace=excluded.workspace, "
                "  ready_at=excluded.ready_at, expires_at=excluded.expires_at",
                (pid, tier, blurb, str(prop_path), workspace,
                 int(time.time()), int(time.time()) + 7*86400),
            )
            print(f"  cached pid={pid} tier={tier} workspace={workspace}")
finally:
    conn.close()
PY_EOF

# Step 5: effectiveness watcher
log "watch_effectiveness..."
"$PY" -m v2.watch_effectiveness >> "$LOG" 2>&1 || log "watch errored"

# Step 6: TTL cleanup — delete raw-event rows older than 30 days.
# Pattern-layer tables (patterns, pattern_observations, pattern_fix_history,
# lm_eval_state) are NEVER cleaned — the time-series IS the value.
# Raw-event tables only serve extraction; after detectors run, observations
# are promoted to pattern_observations and the raw rows are disposable.
log "TTL cleanup..."
"$PY" <<PY_TTL >> "$LOG" 2>&1
import sys, time
sys.path.insert(0, ".")
from v2.schema import open_db

TTL_DAYS = 30
cutoff_s = int(time.time()) - TTL_DAYS * 86400
cutoff_ms = cutoff_s * 1000  # raw-event ts columns are in milliseconds

conn = open_db()
try:
    totals = {}
    # Raw event tables (ts in milliseconds) — 30-day TTL
    for tbl, ts_col in [
        ("user_messages", "ts"),
        ("tool_calls", "ts"),
        ("agent_self_signals", "ts"),
        ("user_correction_signals", "ts"),
        ("discovered_rules", None),  # no ts column; delete by session age
        ("self_signals", "ts"),
    ]:
        if ts_col:
            cur = conn.execute(f"DELETE FROM {tbl} WHERE {ts_col} < ?", (cutoff_ms,))
        else:
            # discovered_rules has no own timestamp; delete for old sessions
            cur = conn.execute(
                f"DELETE FROM {tbl} WHERE session_id IN "
                "(SELECT session_id FROM sessions WHERE last_event_ts < ?)",
                (cutoff_ms,),
            )
        totals[tbl] = cur.rowcount

    # Session metadata (last_event_ts in milliseconds)
    cur = conn.execute("DELETE FROM sessions WHERE last_event_ts < ?", (cutoff_ms,))
    totals["sessions"] = cur.rowcount

    # Operational tables (ts in seconds)
    cur = conn.execute("DELETE FROM session_turn_counter WHERE updated_at < ?", (cutoff_s,))
    totals["session_turn_counter"] = cur.rowcount
    cur = conn.execute("DELETE FROM surfacing_state WHERE surfaced_at < ?", (cutoff_s,))
    totals["surfacing_state"] = cur.rowcount
    cur = conn.execute("DELETE FROM l2_runs WHERE run_at < ?", (cutoff_s,))
    totals["l2_runs"] = cur.rowcount

    # Stale L2 queue (anything older than 7 days = missed by slow path; discard)
    stale_q = int(time.time()) - 7 * 86400
    cur = conn.execute("DELETE FROM l2_feature_queue WHERE enqueued_at < ?", (stale_q,))
    totals["l2_feature_queue_stale"] = cur.rowcount

    # Pattern-layer tables — 1-year TTL. These are the pipeline's value but
    # unbounded growth is still unhealthy. Observations older than 1 year are
    # unlikely to influence current patterns. Patterns themselves stay (they're
    # just metadata rows); only their observations and fix-history get pruned.
    PATTERN_TTL_DAYS = 365
    p_cutoff_ms = (int(time.time()) - PATTERN_TTL_DAYS * 86400) * 1000
    p_cutoff_s = int(time.time()) - PATTERN_TTL_DAYS * 86400
    cur = conn.execute("DELETE FROM pattern_observations WHERE ts < ?", (p_cutoff_ms,))
    totals["pattern_observations_1y"] = cur.rowcount
    cur = conn.execute("DELETE FROM pattern_fix_history WHERE applied_at < ?", (p_cutoff_s,))
    totals["pattern_fix_history_1y"] = cur.rowcount
    # Recompute n_observations for any pattern that had rows deleted
    if totals["pattern_observations_1y"] > 0:
        conn.execute(
            "UPDATE patterns SET "
            "  n_observations = (SELECT COUNT(*) FROM pattern_observations WHERE pid = patterns.pid), "
            "  n_sessions = (SELECT COUNT(DISTINCT session_id) FROM pattern_observations WHERE pid = patterns.pid)"
        )

    cleaned = sum(totals.values())
    if cleaned > 0:
        print(f"TTL cleanup: deleted {cleaned} rows total")
        for tbl, n in sorted(totals.items()):
            if n > 0:
                print(f"  {tbl}: {n}")
        # Reclaim disk space periodically (SQLite doesn't auto-shrink)
        conn.execute("PRAGMA incremental_vacuum(100)")
    else:
        print("TTL cleanup: nothing to delete (all data within TTL windows)")
finally:
    conn.close()
PY_TTL

# Step 6b: staging directory cleanup — remove T0 (immature) staging dirs
# older than 30 days. These are proposals the authorer judged not ready;
# keeping them forever wastes disk and clutters staging listings.
log "staging TTL cleanup..."
"$PY" <<PY_STAGING_TTL >> "$LOG" 2>&1
import json, time, shutil
from pathlib import Path

STAGING = Path("$WORKSPACE_ROOT") / ".github" / "learning" / "staging"
TTL_SECONDS = 30 * 86400
now = time.time()
cleaned = 0

if STAGING.is_dir():
    for pid_dir in STAGING.iterdir():
        if not pid_dir.is_dir():
            continue
        prop = pid_dir / "proposal.json"
        if not prop.is_file():
            continue
        try:
            d = json.loads(prop.read_text())
        except Exception:
            continue
        if d.get("tier", 0) != 0:
            continue
        if (now - prop.stat().st_mtime) > TTL_SECONDS:
            try:
                shutil.rmtree(pid_dir)
                cleaned += 1
            except OSError:
                pass

if cleaned:
    print(f"staging TTL: removed {cleaned} stale T0 dirs")
else:
    print("staging TTL: nothing to clean")
PY_STAGING_TTL

# Step 7: health monitor — check DB size + growth rate, surface alert if abnormal.
log "health check..."
"$PY" <<PY_HEALTH >> "$LOG" 2>&1
import sys, os, time, json
from pathlib import Path
sys.path.insert(0, ".")
from v2.schema import open_db, db_path, state_root

DB = db_path()
STATE = state_root() / "state"
HEALTH_FILE = STATE / "health.json"

# Thresholds
MAX_DB_SIZE_MB = 20          # alert if DB exceeds this
MAX_GROWTH_MB_PER_DAY = 1.0  # alert if growing faster than this
MAX_PATTERN_COUNT = 200      # alert if pattern table is unusually large

alerts = []

# 1. DB file size
db_size_bytes = DB.stat().st_size if DB.is_file() else 0
db_size_mb = db_size_bytes / (1024 * 1024)
if db_size_mb > MAX_DB_SIZE_MB:
    alerts.append(f"DB size {db_size_mb:.1f} MB exceeds {MAX_DB_SIZE_MB} MB threshold")

# 2. Growth rate (compare to last health check)
prev = {}
if HEALTH_FILE.is_file():
    try:
        prev = json.loads(HEALTH_FILE.read_text())
    except Exception:
        pass
prev_size = prev.get("db_size_bytes", db_size_bytes)
prev_ts = prev.get("checked_at", int(time.time()))
elapsed_days = max((int(time.time()) - prev_ts) / 86400, 0.01)
growth_mb = (db_size_bytes - prev_size) / (1024 * 1024)
growth_rate = growth_mb / elapsed_days
if growth_rate > MAX_GROWTH_MB_PER_DAY and elapsed_days > 0.5:
    alerts.append(f"DB growing at {growth_rate:.2f} MB/day (threshold: {MAX_GROWTH_MB_PER_DAY})")

# 3. Pattern count
conn = open_db()
try:
    n_patterns = conn.execute("SELECT COUNT(*) FROM patterns").fetchone()[0]
    n_obs = conn.execute("SELECT COUNT(*) FROM pattern_observations").fetchone()[0]
    n_raw = sum(
        conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        for t in ["user_messages", "tool_calls", "sessions"]
    )
finally:
    conn.close()
if n_patterns > MAX_PATTERN_COUNT:
    alerts.append(f"Pattern count {n_patterns} exceeds {MAX_PATTERN_COUNT} (possible detector noise)")

# 4. Write health snapshot
snapshot = {
    "checked_at": int(time.time()),
    "db_size_bytes": db_size_bytes,
    "db_size_mb": round(db_size_mb, 2),
    "growth_rate_mb_per_day": round(growth_rate, 3),
    "n_patterns": n_patterns,
    "n_observations": n_obs,
    "n_raw_events": n_raw,
    "alerts": alerts,
}
HEALTH_FILE.write_text(json.dumps(snapshot, indent=2))

if alerts:
    print(f"HEALTH ALERTS ({len(alerts)}):")
    for a in alerts:
        print(f"  ⚠️  {a}")
    # Write alerts to cached_surfacing so the hook surfaces them to the user
    # as a <learning-suggestion> on the next prompt.
    conn = open_db()
    try:
        blurb = "Pipeline health alert: " + "; ".join(alerts)[:250]
        conn.execute(
            "INSERT INTO cached_surfacing (pid, tier, surfacing_blurb, proposal_path, workspace, ready_at, expires_at) "
            "VALUES ('_health_alert', 0, ?, NULL, '*', ?, ?) "
            "ON CONFLICT(pid) DO UPDATE SET surfacing_blurb=excluded.surfacing_blurb, ready_at=excluded.ready_at",
            (blurb, int(time.time()), int(time.time()) + 7*86400),
        )
    finally:
        conn.close()
else:
    print(f"Health OK: {db_size_mb:.1f} MB, {n_patterns} patterns, {n_obs} obs, growth {growth_rate:.3f} MB/day")
    # Clear any prior health alert
    conn = open_db()
    try:
        conn.execute("DELETE FROM cached_surfacing WHERE pid='_health_alert'")
    finally:
        conn.close()
PY_HEALTH

date +%s > "$LAST_RUN"
log "v2 slow-path complete"
