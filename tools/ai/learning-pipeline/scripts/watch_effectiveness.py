#!/usr/bin/env python3
"""
watch_effectiveness.py — monitor applied fixes, decide outcome, surface
revert suggestion if rate didn't drop. NEVER auto-reverts (Adversary fix #1).

Run from the slow-path pipeline. Idempotent: only acts on open fixes whose
watch window has completed.

Outcomes written to pattern_fix_history.outcome:
  - 'VALIDATED'           : rate dropped enough; fix is good
  - 'SUGGEST_REVERT'      : rate did NOT drop; surface a revert suggestion
                            (a new cached_surfacing entry for the user to see)
  - 'INSUFFICIENT_DATA'   : not enough new sessions yet OR pre-rate too low
  - (NULL until decided)

Matching context: "same workspace" for all tiers (Adversary fix #9).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .schema import open_db


MIN_PRE_OBS_FOR_AUTO = 5  # Adversary #6 fix
SUGGEST_REVERT_BLURB_TEMPLATE = (
    "Earlier fix for pattern {pid} (T{tier}, commit {sha}) did not reduce the "
    "rate as expected (pre={pre:.2f}/sess, post={post:.2f}/sess in {k} sessions). "
    "Revert? Reply: revert / keep / refine"
)
# Appended to a SUGGEST_REVERT blurb when the reverted fix was a T1 template
# (integration-design.md Decision 7): a T1 hook may have injected context the
# agent acted on in prior commits, which a plain revert cannot undo.
T1_REVERT_WARNING = " Note: T1 may have left injected context in prior commits."


def _workspace_root() -> Path:
    """Repo root, derived from this file's location (.github/scripts/learning/v2/)."""
    return Path(__file__).resolve().parents[4]


def _instance_log_path(template_id: str, pid: str, workspace_root: Path) -> Path:
    """Path to a template instantiation's runtime log.jsonl.

    T1/T3 are per-friction (`<id>__<pid8>/`); T2 is a workspace singleton
    (`T2/`). Mirrors the directory layout produced by template_resolver.py.
    """
    base = workspace_root / ".github" / "scripts" / "templates"
    if template_id == "T2":
        return base / "T2" / "log.jsonl"
    return base / f"{template_id}__{pid[:8]}" / "log.jsonl"


def _read_log(path: Path, since_ts: int = 0) -> list[dict]:
    """Read JSONL log entries with ts >= since_ts. Fail-open: returns [] on any
    error (a measurement read must never crash the watcher)."""
    if not path.is_file():
        return []
    out: list[dict] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(rec, dict) and rec.get("ts", 0) >= since_ts:
                out.append(rec)
    except OSError:
        return []
    return out


def _template_specific_metric(fix_row: dict, workspace_root: Path) -> dict | None:
    """Per-template runtime metric from the instantiation's log.jsonl.

    Returns None for non-template fixes. For template fixes, returns a dict with
    `kind`, `n_log_entries`, and a template-specific rate:
      - T1: trigger_rate (fraction of hook fires) — 0/None ⇒ the hook is a noop
      - T2: violation_rate (violations / files scanned) — contract conformance
      - T3: rejection_rate (fraction rejected) — 0/None ⇒ the gate is decorative
    Fail-open: a missing log yields a metric with n_log_entries=0 and rate None.
    See integration-design.md §3.
    """
    template_id = fix_row.get("template_id")
    if not template_id:
        return None
    log_path = _instance_log_path(template_id, fix_row["pid"], workspace_root)
    entries = _read_log(log_path, since_ts=fix_row.get("applied_at", 0) or 0)
    n = len(entries)
    if template_id == "T1":
        fired = sum(1 for e in entries if e.get("fired"))
        return {"kind": "T1", "n_log_entries": n,
                "trigger_rate": (fired / n) if n else None}
    if template_id == "T2":
        scanned = sum(e.get("n_files_scanned", 0) for e in entries)
        viol = sum(e.get("n_violations", 0) for e in entries)
        return {"kind": "T2", "n_log_entries": n,
                "violation_rate": (viol / scanned) if scanned else None}
    if template_id == "T3":
        # Only count entries that actually carry the gate-outcome fields, so a
        # heartbeat / malformed entry can neither inflate nor deflate the
        # rejection rate (Adversary S1). An entry is a rejection iff it ran a
        # gate and at least one gate did not pass.
        graded = [
            e for e in entries
            if ("syntactic_pass" in e or "semantic_pass" in e)
        ]
        rejected = sum(
            1 for e in graded
            if not (e.get("syntactic_pass") and e.get("semantic_pass"))
        )
        return {"kind": "T3", "n_log_entries": n,
                "rejection_rate": (rejected / len(graded)) if graded else None}
    return {"kind": template_id, "n_log_entries": n}



def _matching_sessions_since(conn, workspace: str, since_ts_seconds: int) -> list[str]:
    """Sessions in same workspace whose last_event_ts >= since_ts.

    NOTE: sessions.last_event_ts is in MILLISECONDS (v1 extractor convention).
    fix_history.applied_at is in SECONDS. Convert.
    """
    since_ts_ms = since_ts_seconds * 1000
    rows = conn.execute(
        "SELECT session_id FROM sessions "
        "WHERE substr(workspace_path, -length(?), length(?)) = ? "
        "  AND last_event_ts >= ?",
        (workspace, workspace, workspace, since_ts_ms),
    ).fetchall()
    return [r[0] for r in rows]


def _post_rate(conn, pid: str, sessions: list[str]) -> tuple[float, int]:
    """Observations of this pid across given sessions / |sessions|."""
    if not sessions:
        return (0.0, 0)
    placeholders = ",".join("?" * len(sessions))
    n_obs = conn.execute(
        f"SELECT COUNT(*) FROM pattern_observations "
        f"WHERE pid = ? AND session_id IN ({placeholders})",
        (pid, *sessions),
    ).fetchone()[0]
    return (n_obs / len(sessions), len(sessions))


def evaluate_one(conn, fix_row: dict) -> dict:
    """Evaluate one open fix. Returns {decision, ...}."""
    pid = fix_row["pid"]
    pat = conn.execute("SELECT workspace FROM patterns WHERE pid=?", (pid,)).fetchone()
    if not pat:
        return {"decision": "skip", "reason": "pattern missing"}
    workspace = pat[0]
    matching = _matching_sessions_since(conn, workspace, fix_row["applied_at"])
    n_post_sess = len(matching)
    if n_post_sess < fix_row["watch_window_K"]:
        return {"decision": "wait", "n_post_sess": n_post_sess,
                "need": fix_row["watch_window_K"]}
    post_rate, _ = _post_rate(conn, pid, matching[: fix_row["watch_window_K"]])
    pre_rate = fix_row["pre_rate"]
    target_drop = fix_row["target_rate_drop"]
    # Insufficient-data floor: low pre-rate → don't auto-judge.
    if pre_rate * fix_row["watch_window_K"] < MIN_PRE_OBS_FOR_AUTO:
        return {"decision": "insufficient_data", "pre_rate": pre_rate,
                "post_rate": post_rate, "n_post_sess": n_post_sess}
    # Decision rule: if post_rate <= pre_rate * (1 - target_drop), VALIDATED.
    threshold = pre_rate * (1.0 - target_drop)
    decision = "VALIDATED" if post_rate <= threshold else "SUGGEST_REVERT"
    return {
        "decision": decision, "pre_rate": pre_rate, "post_rate": post_rate,
        "threshold": threshold, "n_post_sess": n_post_sess,
    }


def watch_all(conn) -> dict:
    """Process every open fix."""
    open_fixes = conn.execute(
        "SELECT fix_id, pid, applied_at, tier, commit_sha, watch_window_K, "
        "       target_rate_drop, pre_rate, template_id "
        "FROM pattern_fix_history WHERE outcome IS NULL "
        "ORDER BY applied_at"
    ).fetchall()
    workspace_root = _workspace_root()
    summary = {"n_open": len(open_fixes), "decisions": []}
    for row in open_fixes:
        fix_id, pid, applied_at, tier, sha, K, target_drop, pre_rate, template_id = row
        fix = {
            "fix_id": fix_id, "pid": pid, "applied_at": applied_at,
            "tier": tier, "commit_sha": sha,
            "watch_window_K": K, "target_rate_drop": target_drop, "pre_rate": pre_rate,
            "template_id": template_id,
        }
        result = evaluate_one(conn, fix)
        # Per-template runtime metric (None for free-form fixes).
        metric = _template_specific_metric(fix, workspace_root)
        metric_json = json.dumps(metric, sort_keys=True) if metric is not None else None
        summary["decisions"].append(
            {"fix_id": fix_id, **result,
             **({"template_metric": metric} if metric is not None else {})}
        )
        # Persist outcome if final.
        if result["decision"] in ("VALIDATED", "SUGGEST_REVERT", "insufficient_data"):
            outcome = result["decision"]
            if outcome == "insufficient_data":
                outcome = "INSUFFICIENT_DATA"
            conn.execute(
                "UPDATE pattern_fix_history SET "
                "  outcome=?, outcome_at=?, post_rate=?, n_post_sessions=?, "
                "  template_metric_json=? "
                "WHERE fix_id=?",
                (
                    outcome, int(time.time()),
                    result.get("post_rate"), result.get("n_post_sess"),
                    metric_json,
                    fix_id,
                ),
            )
            if outcome == "VALIDATED":
                conn.execute(
                    "UPDATE patterns SET status=? WHERE pid=?",
                    (f"VALIDATED-T{tier}", pid),
                )
                # Cached surfacing for this pid no longer needed.
                conn.execute("DELETE FROM cached_surfacing WHERE pid=?", (pid,))
            elif outcome == "SUGGEST_REVERT":
                # Surface revert suggestion. NEVER auto-revert.
                workspace = conn.execute(
                    "SELECT workspace FROM patterns WHERE pid=?", (pid,)
                ).fetchone()[0]
                blurb = SUGGEST_REVERT_BLURB_TEMPLATE.format(
                    pid=pid, tier=tier, sha=(sha[:8] if sha else "ephemeral"),
                    pre=pre_rate, post=result["post_rate"], k=result["n_post_sess"],
                )
                # Decision 7: warn that a T1 hook may have left injected context.
                if template_id == "T1":
                    blurb = (blurb + T1_REVERT_WARNING)
                blurb = blurb[:280]
                conn.execute(
                    "INSERT INTO cached_surfacing "
                    "(pid, tier, surfacing_blurb, proposal_path, workspace, ready_at, expires_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?) "
                    "ON CONFLICT(pid) DO UPDATE SET "
                    "  surfacing_blurb=excluded.surfacing_blurb, "
                    "  ready_at=excluded.ready_at, expires_at=excluded.expires_at",
                    (pid, tier, blurb, None, workspace,
                     int(time.time()), int(time.time()) + 7 * 86400),
                )
                # Re-enable surfacing.
                conn.execute("UPDATE patterns SET status='ACTIVE' WHERE pid=?", (pid,))
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    args = ap.parse_args()
    conn = open_db()
    try:
        print(json.dumps(watch_all(conn), indent=2))
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
