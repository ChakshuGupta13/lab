#!/usr/bin/env python3
"""
PostToolUse hook — structural signal extraction from tool outcomes.

Fires after EVERY tool call. Detects:
  - Tool errors (any non-success response)
  - Specific recurring failures (create_file on existing file, etc.)
  - Bib entries written without R1 provenance (% verified: comment)
  - Cross-scope file edits (different domain/paper than session's first edit)

Writes directly to self_signals table with type='tool-error-structural'.
Injects additionalContext warnings for bib provenance and scope violations.
No agent cooperation needed. 100% reliable.

Contract:
  stdin:  JSON with tool_name, tool_input, tool_response, tool_use_id,
          session_id (or snake_case variants), cwd
  stdout: JSON (continue:true on no-op, or additionalContext on signal)
  exit:   0 always (fail-open)
  budget: <20ms typical
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path


DEADLINE_MS = 15  # tighter than UserPromptSubmit — this fires per tool call


def _now_ms() -> int:
    return int(time.time() * 1000)


def _bail() -> int:
    print(json.dumps({"continue": True}))
    return 0


# Error patterns worth recording. Each tuple: (compiled_regex, signal_subtype).
# These are checked against tool_response text.
ERROR_PATTERNS = [
    (re.compile(r"file already exists", re.I), "file-already-exists"),
    (re.compile(r"no such file or directory", re.I), "file-not-found"),
    (re.compile(r"permission denied", re.I), "permission-denied"),
    (re.compile(r"timed? ?out", re.I), "timeout"),
    (re.compile(r"error|failed|cannot|unable", re.I), "generic-error"),
]

# File-edit tool names that carry a filePath in tool_input.
FILE_EDIT_TOOLS = {
    "replace_string_in_file",
    "multi_replace_string_in_file",
    "create_file",
}

# Regex to find bib entry keys: @article{Key, or @inproceedings{Key, etc.
_BIB_ENTRY_RE = re.compile(r"@\w+\{([^,\s]+)\s*,")


def _check_bib_provenance(payload: dict) -> str | None:
    """If a .bib file was just edited, check entries for % verified: comments.

    Returns an additionalContext warning string, or None.
    Fast path: returns None immediately if the edit is not on a .bib file.
    """
    tool_name = payload.get("tool_name", "")
    tool_input = payload.get("tool_input", {})
    if isinstance(tool_input, str):
        try:
            tool_input = json.loads(tool_input)
        except (json.JSONDecodeError, TypeError):
            return None

    # Determine file path(s) from the tool input
    paths: list[str] = []
    if tool_name in FILE_EDIT_TOOLS:
        fp = tool_input.get("filePath", "")
        if fp:
            paths.append(fp)
    if tool_name == "multi_replace_string_in_file":
        for r in tool_input.get("replacements", []):
            fp = r.get("filePath", "")
            if fp:
                paths.append(fp)

    # Filter to .bib files only
    bib_paths = [p for p in paths if p.endswith(".bib")]
    if not bib_paths:
        return None

    # For each .bib file, read it and check entries
    unverified_keys: list[str] = []
    for bib_path in set(bib_paths):
        try:
            content = Path(bib_path).read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue

        lines = content.splitlines()
        for i, line in enumerate(lines):
            m = _BIB_ENTRY_RE.match(line.strip())
            if not m:
                continue
            key = m.group(1)
            # Search the next 30 lines (typical entry length) for % verified:
            window = "\n".join(lines[i : i + 30])
            if "% verified:" not in window:
                unverified_keys.append(key)

    if not unverified_keys:
        return None

    keys_str = ", ".join(unverified_keys[:5])
    more = f" (+{len(unverified_keys) - 5} more)" if len(unverified_keys) > 5 else ""
    return (
        f"⚠️ R1 PROVENANCE MISSING: bib entries [{keys_str}{more}] lack "
        f"'% verified:' comment. Source-grounding rule R1 requires checkable "
        f"provenance for every bib entry before commit. Add '% verified: "
        f"getting_started/<file>.txt pp.X (YYYY-MM-DD)' after each entry."
    )


# --- Scope enforcement ---
# Matches: domains/<domain>/src/<paper>/ or domains/<domain>/docs/<paper>/
# or domains/<domain>/getting_started/<anything>
_SCOPE_RE = re.compile(
    r"domains/([^/]+)/(?:src|docs)/([^/]+)/"
)
# getting_started paths encode domain but not paper — treated as domain-scoped
_GETTING_STARTED_RE = re.compile(
    r"domains/([^/]+)/getting_started/"
)
# Paths that are always allowed regardless of scope (REPO-ROOT infra files only).
# Checked ONLY after confirming the path is NOT inside domains/<domain>/.
_INFRA_SEGMENTS = {".github/", "crosslinks/"}


def _extract_scope(file_path: str) -> str | None:
    """Extract '<domain>/<paper>' from a file path, or None if not paper-scoped."""
    m = _SCOPE_RE.search(file_path)
    if m:
        return f"{m.group(1)}/{m.group(2)}"
    return None


def _extract_domain(file_path: str) -> str | None:
    """Extract '<domain>' from a getting_started path."""
    m = _GETTING_STARTED_RE.search(file_path)
    if m:
        return m.group(1)
    return None


def _is_infra_path(file_path: str) -> bool:
    """True if the path is repo-root infrastructure (not inside any domain)."""
    # If path is inside domains/, it's not infra — even if named README.md
    if "/domains/" in file_path or file_path.startswith("domains/"):
        return False
    for seg in _INFRA_SEGMENTS:
        if seg in file_path:
            return True
    return False


def _check_scope_violation(payload: dict, session_id: str) -> str | None:
    """Track per-session scope and warn on cross-scope edits.

    Uses a tiny file in /tmp keyed by session_id to remember the first
    domain/paper scope this session edited. If a subsequent edit targets a
    different scope, returns a warning string.
    """
    tool_name = payload.get("tool_name", "")
    if tool_name not in FILE_EDIT_TOOLS:
        return None

    tool_input = payload.get("tool_input", {})
    if isinstance(tool_input, str):
        try:
            tool_input = json.loads(tool_input)
        except (json.JSONDecodeError, TypeError):
            return None

    # Collect file paths from this edit
    paths: list[str] = []
    fp = tool_input.get("filePath", "")
    if fp:
        paths.append(fp)
    if tool_name == "multi_replace_string_in_file":
        for r in tool_input.get("replacements", []):
            rfp = r.get("filePath", "")
            if rfp:
                paths.append(rfp)

    if not paths:
        return None

    # Extract scopes from edited paths
    edit_scopes: set[str] = set()
    domain_only_edits: set[str] = set()
    for p in paths:
        if _is_infra_path(p):
            continue
        scope = _extract_scope(p)
        if scope:
            edit_scopes.add(scope)
            continue
        domain = _extract_domain(p)
        if domain:
            domain_only_edits.add(domain)

    if not edit_scopes and not domain_only_edits:
        return None  # editing infra files only — always OK

    # Session scope tracking via /tmp file
    if not session_id:
        return None
    scope_file = Path(f"/tmp/copilot-scope-{session_id[:16]}.txt")

    try:
        if scope_file.exists():
            established_scope = scope_file.read_text().strip()
        else:
            if edit_scopes:
                # First paper-scoped edit in this session — establish scope
                first_scope = sorted(edit_scopes)[0]
                scope_file.write_text(first_scope)
            return None

        # Check paper-level scope violations
        violations = edit_scopes - {established_scope}
        # Check domain-level violations for getting_started edits
        established_domain = established_scope.split("/")[0] if "/" in established_scope else established_scope
        domain_violations = domain_only_edits - {established_domain}

        all_violations = violations | {f"{d}/getting_started" for d in domain_violations}
        if not all_violations:
            return None

        viol_str = ", ".join(sorted(all_violations))
        return (
            f"⚠️ SCOPE VIOLATION: this session's scope is '{established_scope}' "
            f"(first paper-scoped edit), but you just edited file(s) in [{viol_str}]. "
            f"Per git-scoping rules, do NOT stage these files — they belong to a "
            f"different session. See .github/instructions/git-scoping.instructions.md."
        )
    except OSError:
        return None  # fail-open


def main() -> int:
    started = _now_ms()

    try:
        raw = sys.stdin.read()
        payload = json.loads(raw) if raw else {}
    except (json.JSONDecodeError, OSError):
        return _bail()

    tool_name = payload.get("tool_name", "")
    tool_response = payload.get("tool_response", "")
    session_id = payload.get("session_id") or payload.get("sessionId") or ""
    cwd = payload.get("cwd", "")
    workspace = cwd.rstrip("/").split("/")[-1] if cwd else "<unknown>"

    # --- Scope enforcement (highest priority — fires before bib check) ---
    try:
        if (_now_ms() - started) < DEADLINE_MS:
            scope_warning = _check_scope_violation(payload, session_id)
            if scope_warning:
                print(json.dumps({"additionalContext": scope_warning}))
                return 0
    except Exception:
        pass  # fail-open

    # --- Bib provenance check (fires on successful edits, not just errors) ---
    # Must run AFTER confirming the edit succeeded — a failed edit should still
    # go through the error-pattern check below, not be masked by a stale bib warning.
    bib_warning = None
    try:
        # Only check bib provenance if the response does NOT look like an error
        resp_looks_ok = True
        if tool_response and isinstance(tool_response, str):
            for pat, _ in ERROR_PATTERNS:
                if pat.search(tool_response):
                    resp_looks_ok = False
                    break
        if resp_looks_ok and (_now_ms() - started) < DEADLINE_MS:
            bib_warning = _check_bib_provenance(payload)
    except Exception:
        pass  # fail-open

    if bib_warning:
        print(json.dumps({"additionalContext": bib_warning}))
        return 0

    # Quick exit if no error signal in the response
    if not tool_response or not isinstance(tool_response, str):
        return _bail()

    # Check for error patterns
    matched_subtype = None
    for pattern, subtype in ERROR_PATTERNS:
        if pattern.search(tool_response):
            matched_subtype = subtype
            break  # first match wins (ordered from specific to generic)

    if not matched_subtype:
        return _bail()

    if _now_ms() - started > DEADLINE_MS:
        return _bail()

    # Write to self_signals
    try:
        here = Path(__file__).resolve().parent
        sys.path.insert(0, str(here.parent))
        from v2.schema import open_db
        conn = open_db()
        try:
            ts_ms = _now_ms()
            evidence = f"{tool_name}: {tool_response[:150]}"
            # Dedup: don't write the same (session, tool, subtype) within 5 seconds
            recent = conn.execute(
                "SELECT 1 FROM self_signals "
                "WHERE session_id=? AND type=? AND evidence LIKE ? AND ts > ? LIMIT 1",
                (session_id, f"tool-{matched_subtype}", f"{tool_name}:%", ts_ms - 5000),
            ).fetchone()
            if not recent:
                conn.execute(
                    "INSERT INTO self_signals (session_id, ts, type, evidence, workspace) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (session_id, ts_ms, f"tool-{matched_subtype}", evidence[:200], workspace),
                )
        finally:
            conn.close()
    except Exception:
        pass  # fail-open

    return _bail()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        try:
            print(json.dumps({"continue": True}))
        except Exception:
            pass
        sys.exit(0)
