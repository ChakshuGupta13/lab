#!/usr/bin/env python3
"""
apply_proposal.py — apply a validated learning proposal as one git commit.

CRITICAL hygiene (Adversary D fix):
  - Refuses to write any path outside `.github/` (no domain artifacts, no mandatory.md)
  - Refuses if working tree has uncommitted changes to any target path
  - Refuses if proposal hasn't been validated
  - One git commit per apply; commit message includes pid for traceability + revert
  - Dry-run mode shows the diff that would be applied

Invoked by the foreground agent after the user explicitly approves a surfaced
suggestion (per .github/instructions/learning-pipeline-dispatch.instructions.md).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

try:
    from .schema import open_db
except ImportError:
    # Direct invocation (python3 apply_proposal.py --pid ...)
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from v2.schema import open_db  # type: ignore

CAMPAIGN_LOG = Path(__file__).resolve().parent.parent.parent.parent.parent / ".github" / "learning" / "campaign_log.jsonl"


ALLOWED_PATH_PREFIX = ".github/"
FORBIDDEN_PATH_SUBSTRINGS = ("mandatory.md", "/domains/", "domains/")


def _workspace_root() -> Path:
    here = Path(__file__).resolve()
    return here.parent.parent.parent.parent.parent


def _staging_proposal_path(workspace_root: Path, pid: str) -> Path:
    return workspace_root / ".github" / "learning" / "staging" / pid / "proposal.json"


def _validate_path(path: str) -> tuple[bool, str]:
    """Strict path whitelist check."""
    if not path.startswith(ALLOWED_PATH_PREFIX):
        return False, f"path must start with {ALLOWED_PATH_PREFIX}: {path!r}"
    for forbidden in FORBIDDEN_PATH_SUBSTRINGS:
        if forbidden in path:
            return False, f"path contains forbidden substring {forbidden!r}: {path!r}"
    if ".." in Path(path).parts:
        return False, f"path contains .. : {path!r}"
    return True, "ok"


def _git(workspace_root: Path, *args: str, capture: bool = True) -> tuple[int, str, str]:
    """Run git, return (rc, stdout, stderr)."""
    proc = subprocess.run(
        ["git", *args], cwd=str(workspace_root),
        capture_output=capture, text=True, timeout=30,
    )
    return proc.returncode, (proc.stdout or "").strip(), (proc.stderr or "").strip()


def _working_tree_clean_for(workspace_root: Path, paths: list[str]) -> tuple[bool, str]:
    """Confirm none of the target paths have uncommitted changes."""
    rc, out, err = _git(workspace_root, "status", "--porcelain", "--", *paths)
    if rc != 0:
        return False, f"git status failed: {err}"
    if out:
        return False, f"target paths have uncommitted changes:\n{out}"
    return True, "ok"


def _apply_operation(workspace_root: Path, op: dict, dry_run: bool = False) -> tuple[bool, str]:
    """Apply one create/edit/delete op."""
    kind = op.get("op")
    path = op.get("path", "")
    content = op.get("content", "")
    ok, err = _validate_path(path)
    if not ok:
        return False, err
    target = workspace_root / path
    if kind == "create":
        if target.exists():
            return False, f"create: target already exists: {path}"
        if dry_run:
            return True, f"DRY: would create {path} ({len(content)} bytes)"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return True, f"created {path}"
    elif kind == "edit":
        if not target.is_file():
            return False, f"edit: target missing: {path}"
        # Apply as unified diff using `git apply` for safety.
        # Bug fix per Adversary post-build WRONG #3: removed dead `_git(... "-")`
        # call that would hang reading stdin for up to 30s.
        if dry_run:
            return _check_or_apply_diff(workspace_root, content, apply=False)
        return _check_or_apply_diff(workspace_root, content, apply=True)
    elif kind == "delete":
        if not target.exists():
            return False, f"delete: target missing: {path}"
        if dry_run:
            return True, f"DRY: would delete {path}"
        target.unlink()
        return True, f"deleted {path}"
    else:
        return False, f"unknown op kind: {kind!r}"


def _check_or_apply_diff(workspace_root: Path, diff_text: str, apply: bool) -> tuple[bool, str]:
    """Run `git apply [--check]` with diff on stdin."""
    args = ["git", "apply"]
    if not apply:
        args.append("--check")
    proc = subprocess.run(
        args, input=diff_text, cwd=str(workspace_root),
        capture_output=True, text=True, timeout=30,
    )
    if proc.returncode != 0:
        return False, f"git apply failed: {proc.stderr.strip()}"
    return True, ("applied diff" if apply else "diff valid")


def _record_apply(
    conn, pid: str, tier: int, commit_sha: str | None, target_paths: list[str],
    pre_rate: float, target_rate_drop: float, watch_window_K: int,
    template_id: str | None = None, template_params_json: str | None = None,
) -> int:
    """Append row to pattern_fix_history. Returns fix_id.

    template_id / template_params_json are populated only for template-backed
    proposals (integration-design.md Decision 6); they are NULL for free-form
    and ephemeral fixes. The composition-drift linter reads template_params_json
    from prior rows, so persisting it here is what makes Decision 9 enforceable
    across sessions (staging dirs are ephemeral and overwritten).
    """
    cur = conn.execute(
        "INSERT INTO pattern_fix_history "
        "(pid, applied_at, tier, commit_sha, target_paths, watch_window_K, "
        " target_rate_drop, pre_rate, template_id, template_params_json) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            pid, int(time.time()), tier, commit_sha,
            json.dumps(target_paths), watch_window_K, target_rate_drop, pre_rate,
            template_id, template_params_json,
        ),
    )
    return cur.lastrowid


# --- Composition-drift linter (integration-design.md Decision 9) -------------
# Prevents silently assembling T1+T2+T3 on the SAME corpus across separate
# user approvals. The check is at corpus level (not target-path level): the
# three templates write to different instantiation dirs, so their target_paths
# never overlap, but they all operate on the same runtime corpus.

def _extract_corpus_globs(template_params: dict) -> set[str]:
    """Pull the corpus glob(s) from a template's params.

    T1 uses `corpus_globs` (list); T2 uses `corpus_glob` (string); T3 gates
    proposals, not corpora, so it has neither → empty set → never composes.
    """
    out: set[str] = set()
    globs = template_params.get("corpus_globs")
    if isinstance(globs, list):
        out.update(_norm_glob(g) for g in globs if isinstance(g, str) and g)
    single = template_params.get("corpus_glob")
    if isinstance(single, str) and single:
        out.add(_norm_glob(single))
    return out


def _norm_glob(glob: str) -> str:
    """Normalize a glob for overlap comparison: strip a leading './'.

    Path operations frequently produce a './' prefix; without stripping it,
    './a/*.md' would be treated as disjoint from 'a/*.md' (a false-negative that
    lets a composition slip through). See Adversary S1.
    """
    while glob.startswith("./"):
        glob = glob[2:]
    return glob


def _glob_dir(glob: str) -> str:
    """Directory portion of a literal glob (everything before the last '/')."""
    return glob.rsplit("/", 1)[0] if "/" in glob else ""


def _globs_overlap(a: str, b: str) -> bool:
    """Conservative overlap test for the closed set of literal globs (no '**').

    Two globs overlap if identical, or if one's directory equals or is a parent
    of the other's directory. Per Decision 9's known limit, '**' wildcards are
    out of scope and treated as non-overlapping unless string-equal.
    """
    if a == b:
        return True
    if "**" in a or "**" in b:
        return a == b
    da, db = _glob_dir(a), _glob_dir(b)
    if da == db:
        return True
    # parent/child directory containment
    da_s, db_s = da.rstrip("/") + "/", db.rstrip("/") + "/"
    return da_s.startswith(db_s) or db_s.startswith(da_s)


def _check_composition_drift(
    conn, pid: str, template: dict, composition_ack: object,
) -> tuple[bool, str]:
    """Refuse a template apply that composes with a prior template instantiation
    on an overlapping corpus unless every prior overlapping fix_id is listed in
    `composition_ack`.

    Returns (ok, message). ok=True means safe to proceed.
    """
    params = template.get("params") or {}
    incoming = _extract_corpus_globs(params)
    if not incoming:
        return True, "no corpus (e.g. T3); composition not applicable"
    # Prior template instantiations with a recorded corpus. Exclude the
    # proposal's OWN pid: re-applying a corrected template to the same corpus
    # after a revert is not a cross-template composition event (Adversary W1).
    rows = conn.execute(
        "SELECT fix_id, template_id, template_params_json FROM pattern_fix_history "
        "WHERE template_id IS NOT NULL AND template_params_json IS NOT NULL "
        "AND pid != ?",
        (pid,),
    ).fetchall()
    overlapping: list[int] = []
    for fix_id, _tid, tparams_json in rows:
        try:
            prior_params = json.loads(tparams_json)
        except (json.JSONDecodeError, TypeError):
            continue
        # A row storing the JSON literal 'null' (or any non-object) would crash
        # _extract_corpus_globs; skip it so one bad row cannot break the linter
        # for every future proposal (Adversary S3).
        if not isinstance(prior_params, dict):
            continue
        prior_globs = _extract_corpus_globs(prior_params)
        if any(_globs_overlap(i, p) for i in incoming for p in prior_globs):
            overlapping.append(fix_id)
    if not overlapping:
        return True, "no overlapping prior template instantiation"
    ack = composition_ack if isinstance(composition_ack, list) else []
    missing = [fid for fid in overlapping if fid not in ack]
    if missing:
        return False, (
            "composition drift: this template overlaps prior instantiation(s) "
            f"{sorted(overlapping)} on corpus {sorted(incoming)}. The proposal must "
            f"carry composition_ack listing every overlapping fix_id; missing {sorted(missing)}. "
            "Composing templates on the same corpus requires explicit user approval "
            "(see .github/mechanism-templates/README.md Composition rules)."
        )
    return True, f"composition acknowledged for prior fixes {sorted(overlapping)}"



def _compute_pre_rate(conn, pid: str, k_sessions: int) -> float:
    """Pattern observations per session over the last k_sessions for matching workspace.

    NOTE on units: pattern_observations.ts and sessions.last_event_ts are in
    MILLISECONDS (v1 extractor convention). All v2 wall-clock fields (applied_at,
    last_eval_at, etc.) are in SECONDS. This function operates on counts only,
    so it is unit-safe.
    """
    row = conn.execute(
        "SELECT workspace FROM patterns WHERE pid = ?", (pid,)
    ).fetchone()
    if not row:
        return 0.0
    workspace = row[0]
    # Number of sessions in this workspace, ordered by recency.
    n_sessions = conn.execute(
        "SELECT COUNT(*) FROM ("
        "  SELECT s.session_id FROM sessions s "
        "  WHERE substr(s.workspace_path, -length(?), length(?)) = ? "
        "    AND s.last_event_ts IS NOT NULL "
        "  ORDER BY s.last_event_ts DESC LIMIT ?"
        ")",
        (workspace, workspace, workspace, k_sessions),
    ).fetchone()[0]
    if n_sessions == 0:
        return 0.0
    # Observations of this pid in those K most-recent sessions.
    n_obs = conn.execute(
        "SELECT COUNT(*) FROM pattern_observations o "
        "WHERE o.pid = ? AND o.session_id IN ("
        "  SELECT s.session_id FROM sessions s "
        "  WHERE substr(s.workspace_path, -length(?), length(?)) = ? "
        "    AND s.last_event_ts IS NOT NULL "
        "  ORDER BY s.last_event_ts DESC LIMIT ?"
        ")",
        (pid, workspace, workspace, workspace, k_sessions),
    ).fetchone()[0]
    return n_obs / max(n_sessions, 1)


def apply_proposal(pid: str, dry_run: bool = False) -> dict:
    """Main entry. Returns {ok, summary, commit_sha?}.

    Safety gate (2026-06-05 post-deploy): requires LEARNING_PIPELINE_ENABLED=1
    in the environment. The dispatch instruction file (loaded into every
    Copilot session) is too easy a vector for accidental application — even a
    perfectly-classified user "apply" reply shouldn't write to the repo unless
    the user has explicitly opted into the pipeline for this terminal session.
    Dry-run bypasses the gate so the user can preview without enabling.
    """
    if not dry_run and os.environ.get("LEARNING_PIPELINE_ENABLED") != "1":
        return {
            "ok": False,
            "reason": "LEARNING_PIPELINE_ENABLED=1 not set in environment; refusing to write.",
            "hint": "Set the env var in the terminal where the agent runs apply_proposal.py.",
        }
    workspace_root = _workspace_root()
    proposal_path = _staging_proposal_path(workspace_root, pid)
    if not proposal_path.is_file():
        return {"ok": False, "reason": f"no proposal at {proposal_path}"}
    try:
        proposal = json.loads(proposal_path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        return {"ok": False, "reason": f"proposal unreadable: {e}"}
    if not proposal.get("mature"):
        return {"ok": False, "reason": "proposal not mature"}
    tier = proposal.get("tier", 0)
    if tier == 0:
        return {"ok": False, "reason": "tier 0 (advice-only); no apply"}
    # Composition-drift gate (Decision 9). Runs for any template-backed proposal
    # BEFORE the tier branches, so it covers dry-run and both TIER 2/3. No-ops
    # for free-form proposals (no `template`). A template is only valid on a
    # mature TIER 2/3 proposal (enforced upstream by validate_output), so this
    # never fires for TIER 1.
    template = proposal.get("template")
    if template:
        conn = open_db()
        try:
            ok, msg = _check_composition_drift(
                conn, pid, template, proposal.get("composition_ack"),
            )
        finally:
            conn.close()
        if not ok:
            return {"ok": False, "reason": msg}
    if tier == 1:
        # Ephemeral nudge; nothing to write. Just record the apply.
        conn = open_db()
        try:
            pre_rate = _compute_pre_rate(conn, pid, k_sessions=proposal.get("watch_window_K", 5))
            fix_id = _record_apply(
                conn, pid, tier=1, commit_sha=None,
                target_paths=[],
                pre_rate=pre_rate,
                target_rate_drop=proposal.get("target_rate_drop", 0.5),
                watch_window_K=proposal.get("watch_window_K", 5),
            )
            # Mark pattern; TIER1 keeps it ACTIVE so the nudge keeps surfacing
            # (the nudge IS the fix). The watcher will downgrade it later.
            conn.execute("UPDATE patterns SET status='ACTIVE' WHERE pid=?", (pid,))
            return {
                "ok": True, "summary": "TIER1 applied (ephemeral)",
                "fix_id": fix_id, "commit_sha": None, "dry_run": dry_run,
            }
        finally:
            conn.close()
    # TIER 2 or 3: write files + commit.
    proposal_obj = proposal.get("proposal", {})
    ops = proposal_obj.get("operations", [])
    if not ops:
        return {"ok": False, "reason": f"tier={tier} but no operations"}
    target_paths = sorted({op.get("path", "") for op in ops if op.get("path")})
    # Validate every path BEFORE writing anything.
    for op in ops:
        ok, err = _validate_path(op.get("path", ""))
        if not ok:
            return {"ok": False, "reason": f"path validation: {err}"}
    # Working tree must be clean for these paths.
    ok, err = _working_tree_clean_for(workspace_root, target_paths)
    if not ok:
        return {"ok": False, "reason": err}
    # Dry-run: validate all ops without writing.
    if dry_run:
        results = []
        for op in ops:
            ok, msg = _apply_operation(workspace_root, op, dry_run=True)
            results.append({"op": op.get("op"), "path": op.get("path"), "ok": ok, "msg": msg})
            if not ok:
                return {"ok": False, "reason": f"dry-run failed: {msg}", "results": results}
        return {"ok": True, "summary": "dry-run pass", "results": results, "dry_run": True}
    # Apply. Track which ops actually wrote (for rollback of untracked creates).
    # Bug fix per Adversary post-build CRITICAL #2: git checkout -- cannot
    # remove untracked files created by failed-after-partial-success apply.
    applied = []
    created_paths: list[str] = []
    for op in ops:
        ok, msg = _apply_operation(workspace_root, op, dry_run=False)
        if not ok:
            # Roll back: tracked edits via `git checkout`, then unlink untracked creates.
            _git(workspace_root, "checkout", "--", *target_paths)
            for created in created_paths:
                try:
                    (workspace_root / created).unlink()
                except FileNotFoundError:
                    pass
                except OSError:
                    pass
            return {"ok": False, "reason": f"apply failed: {msg}", "applied_before_fail": applied}
        applied.append({"op": op.get("op"), "path": op.get("path"), "msg": msg})
        if op.get("op") == "create":
            created_paths.append(op.get("path"))
    # Stage + commit.
    rc, _, err = _git(workspace_root, "add", "--", *target_paths)
    if rc != 0:
        _git(workspace_root, "checkout", "--", *target_paths)
        return {"ok": False, "reason": f"git add failed: {err}"}
    # Commit message format: standard convention + pid for revert lookup.
    blurb = (proposal.get("surfacing_blurb") or "")[:80].replace("\n", " ")
    msg = (
        f"feat(learning): apply T{tier} pid={pid}\n\n"
        f"{blurb}\n\n"
        f"why\n- pattern {pid} surfaced via rolling-learning pipeline; user approved.\n\n"
        f"what\n- {chr(10).join('- ' + p for p in target_paths)}\n\n"
        f"testing\n- automatic: watch_window_K={proposal.get('watch_window_K')} sessions "
        f"target_rate_drop={proposal.get('target_rate_drop')}\n"
    )
    rc, sha_out, err = _git(workspace_root, "commit", "-m", msg)
    if rc != 0:
        return {"ok": False, "reason": f"git commit failed: {err}"}
    rc, sha, _ = _git(workspace_root, "rev-parse", "HEAD")
    if rc != 0:
        return {"ok": False, "reason": "could not read commit sha"}
    # Record in DB.
    conn = open_db()
    try:
        pre_rate = _compute_pre_rate(conn, pid, k_sessions=proposal.get("watch_window_K", 5))
        fix_id = _record_apply(
            conn, pid, tier=tier, commit_sha=sha,
            target_paths=target_paths,
            pre_rate=pre_rate,
            target_rate_drop=proposal.get("target_rate_drop", 0.5),
            watch_window_K=proposal.get("watch_window_K", 5),
            template_id=(template or {}).get("id") if template else None,
            template_params_json=(
                json.dumps((template or {}).get("params"), sort_keys=True)
                if template else None
            ),
        )
        # Mark pattern ACTIVE so the watcher sees it.
        conn.execute("UPDATE patterns SET status='ACTIVE' WHERE pid=?", (pid,))
        # Mark surfacing dispatched.
        conn.execute(
            "UPDATE surfacing_state SET status='DISPATCHED', "
            "  dispatched_intent='apply', dispatched_at=? "
            "WHERE pid=? AND status='OPEN'",
            (int(time.time()), pid),
        )
        # Remove cached_surfacing — fix is in flight.
        conn.execute("DELETE FROM cached_surfacing WHERE pid=?", (pid,))
    finally:
        conn.close()
    return {
        "ok": True, "summary": f"T{tier} applied + committed",
        "fix_id": fix_id, "commit_sha": sha,
        "tier": tier,
        "target_paths": target_paths,
        "dry_run": False,
    }


def _append_campaign_log(entry: dict) -> None:
    """Append one JSON line to the campaign log. Best-effort."""
    try:
        CAMPAIGN_LOG.parent.mkdir(parents=True, exist_ok=True)
        with CAMPAIGN_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, separators=(",", ":")) + "\n")
    except OSError:
        pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pid", required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    result = apply_proposal(args.pid, dry_run=args.dry_run)
    if result.get("ok") and not args.dry_run:
        _append_campaign_log({
            "event": "applied",
            "pid": args.pid,
            "tier": result.get("tier"),
            "commit_sha": result.get("commit_sha"),
            "target_paths": result.get("target_paths", []),
            "ts": int(time.time()),
        })
    print(json.dumps(result, indent=2))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
