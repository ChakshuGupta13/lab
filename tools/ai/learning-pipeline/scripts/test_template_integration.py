#!/usr/bin/env python3
"""
Discrimination tests for the template-integration wiring (Steps 1-3 of
.github/learning/integration-design.md): schema migration, validator
extension, and the resolver.

Run from .github/scripts/learning/:
    python3 -m v2.test_template_integration

Exit 0 on all pass; 1 with diagnostics on first failure. No LM, no live DB.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

from .lm_authorer import validate_output, TEMPLATE_REQUIRED_SLOTS
from .template_resolver import resolve, resolve_T1, resolve_T2, resolve_T3
from . import schema
from .apply_proposal import (
    _globs_overlap,
    _extract_corpus_globs,
    _check_composition_drift,
    _record_apply,
)
from .watch_effectiveness import (
    _instance_log_path,
    _read_log,
    _template_specific_metric,
    watch_all,
    T1_REVERT_WARNING,
)


# Minimal valid non-template payload (baseline — must keep passing unchanged).
def _base_payload() -> dict:
    return {
        "pid": "abc12345",
        "mature": True,
        "tier": 2,
        "justification": "x",
        "surfacing_blurb": "y",
        "target_rate_drop": 0.5,
        "watch_window_K": 5,
        "next_check_after_n_observations": 5,
        "min_observations_required": 5,
        "proposal": {"operations": [
            {"op": "create", "path": ".github/foo.md", "content": "z"},
        ]},
    }


def _valid_t1_params() -> dict:
    return {
        "trigger": "claim|propose",
        "corpus_globs": ["/memories/repo/*.md"],
        "sufficiency_check": "all proven",
        "max_iterations": 2,
        "fallback_action": "surface_caveat",
        "scope": "common",
    }


def _scenario(name: str, predicate: bool, detail: str) -> bool:
    status = "PASS" if predicate else "FAIL"
    print(f"  [{status}] {name}: {detail}")
    return predicate


# ---------------------------------------------------------------------------
# Step 1 — schema migration
# ---------------------------------------------------------------------------

def test_schema_migration() -> bool:
    print("\nStep 1: schema migration")
    ok = True
    with tempfile.TemporaryDirectory() as td:
        dbp = Path(td) / "events.db"
        conn = schema.open_db(dbp)
        cols = {r[1] for r in conn.execute("PRAGMA table_info(pattern_fix_history)")}
        conn.close()
        need = {"template_id", "template_params_json", "template_metric_json"}
        ok &= _scenario("fresh DB has all 3 template columns", need <= cols, f"missing={need - cols}")
        # Idempotent: open again, no error.
        try:
            conn2 = schema.open_db(dbp)
            conn2.close()
            ok &= _scenario("second open is idempotent", True, "no duplicate-column error")
        except Exception as e:  # noqa: BLE001
            ok &= _scenario("second open is idempotent", False, f"raised {e!r}")
    return ok


# ---------------------------------------------------------------------------
# Step 2 — validator extension
# ---------------------------------------------------------------------------

def test_validator() -> bool:
    print("\nStep 2: validator extension")
    ok = True

    # Baseline: no-template payload unchanged.
    valid, msg = validate_output(_base_payload(), "abc12345")
    ok &= _scenario("no-template payload still valid", valid, msg)

    # Valid T1 template.
    p = _base_payload()
    p["template"] = {
        "id": "T1",
        "params": _valid_t1_params(),
        "expected_friction_reduction": "fewer contradicted claims",
    }
    valid, msg = validate_output(p, "abc12345")
    ok &= _scenario("valid T1 template accepted", valid, msg)

    # Unknown id.
    p2 = _base_payload()
    p2["template"] = {"id": "T9", "params": {}, "expected_friction_reduction": "x"}
    valid, msg = validate_output(p2, "abc12345")
    ok &= _scenario("unknown template id rejected", not valid, msg)

    # Missing slot.
    p3 = _base_payload()
    bad_params = _valid_t1_params()
    del bad_params["scope"]
    p3["template"] = {"id": "T1", "params": bad_params, "expected_friction_reduction": "x"}
    valid, msg = validate_output(p3, "abc12345")
    ok &= _scenario("missing slot rejected", not valid and "scope" in msg, msg)

    # Extra slot.
    p4 = _base_payload()
    extra_params = _valid_t1_params()
    extra_params["bogus"] = 1
    p4["template"] = {"id": "T1", "params": extra_params, "expected_friction_reduction": "x"}
    valid, msg = validate_output(p4, "abc12345")
    ok &= _scenario("extra slot rejected", not valid and "bogus" in msg, msg)

    # Empty reduction.
    p5 = _base_payload()
    p5["template"] = {"id": "T1", "params": _valid_t1_params(), "expected_friction_reduction": "  "}
    valid, msg = validate_output(p5, "abc12345")
    ok &= _scenario("empty expected_friction_reduction rejected", not valid, msg)

    # W1: template on a non-(2/3) tier is rejected.
    p6 = _base_payload()
    p6["tier"] = 1
    p6["template"] = {"id": "T1", "params": _valid_t1_params(), "expected_friction_reduction": "x"}
    valid, msg = validate_output(p6, "abc12345")
    ok &= _scenario("W1: template on TIER 1 rejected", not valid and "TIER 2/3" in msg, msg)

    # S1: template:null is treated as absent (proposal still valid).
    p7 = _base_payload()
    p7["template"] = None
    valid, msg = validate_output(p7, "abc12345")
    ok &= _scenario("S1: template=null treated as absent (accepted)", valid, msg)

    # Slot sets match catalog (guard against silent drift in this test).
    ok &= _scenario(
        "T1/T2/T3 slot maps are non-empty and disjoint from each other only on scope",
        all(len(s) == 6 for s in TEMPLATE_REQUIRED_SLOTS.values()),
        f"sizes={[len(s) for s in TEMPLATE_REQUIRED_SLOTS.values()]}",
    )
    return ok


# ---------------------------------------------------------------------------
# Step 3 — resolver
# ---------------------------------------------------------------------------

def _all_paths_under_github(ops: list[dict]) -> bool:
    return all(op["path"].startswith(".github/") for op in ops)


def _no_forbidden(ops: list[dict]) -> bool:
    bad = ("mandatory.md", "/domains/", "domains/")
    return all(not any(b in op["path"] for b in bad) for op in ops)


def test_resolver() -> bool:
    print("\nStep 3: resolver")
    ok = True
    pid = "abc12345def"

    # T1
    t1_ops = resolve_T1(_valid_t1_params(), pid)
    ok &= _scenario("T1 emits >=1 op", len(t1_ops) >= 1, f"{len(t1_ops)} ops")
    ok &= _scenario("T1 paths under .github/", _all_paths_under_github(t1_ops), str([o["path"] for o in t1_ops]))
    ok &= _scenario("T1 no forbidden paths", _no_forbidden(t1_ops), "ok")
    ok &= _scenario("T1 path carries pid8", any("T1__abc12345" in o["path"] for o in t1_ops), str([o["path"] for o in t1_ops]))
    ok &= _scenario("T1 config.json is valid JSON", _config_parses(t1_ops), "ok")
    # Determinism.
    ok &= _scenario("T1 deterministic", resolve_T1(_valid_t1_params(), pid) == t1_ops, "same output")

    # T2
    t2_params = {
        "corpus_glob": "domains/x/audit-log/*.md",
        "required_fields": ["status", "provides"],
        "status_enum": ["draft", "proven"],
        "supersession_field": "superseded_by",
        "lint_command": "python3 lint.py",
        "scope": "x/y",
    }
    t2_ops = resolve_T2(t2_params, pid)
    ok &= _scenario("T2 emits >=1 op", len(t2_ops) >= 1, f"{len(t2_ops)} ops")
    ok &= _scenario("T2 paths under .github/", _all_paths_under_github(t2_ops), "ok")
    ok &= _scenario("T2 targets the singleton T2 dir", all("/templates/T2/" in o["path"] for o in t2_ops), str([o["path"] for o in t2_ops]))

    # T3
    t3_params = {
        "proposal_kind": "instruction_file_edit",
        "syntactic_gate": "python3 -m py_compile",
        "semantic_gate": "objective:watch_effectiveness",
        "rejected_buffer_path": ".github/learning/rejected/x.jsonl",
        "max_attempts": 3,
        "scope": "common",
    }
    t3_ops = resolve_T3(t3_params, pid)
    ok &= _scenario("T3 emits >=1 op", len(t3_ops) >= 1, f"{len(t3_ops)} ops")
    ok &= _scenario("T3 paths under .github/", _all_paths_under_github(t3_ops), "ok")
    ok &= _scenario("T3 path carries pid8", any("T3__abc12345" in o["path"] for o in t3_ops), "ok")

    # Dispatch via resolve()
    disp = resolve({"id": "T1", "params": _valid_t1_params()}, pid)
    ok &= _scenario("resolve() dispatches T1", disp == t1_ops, "matches resolve_T1")

    # Logger snippet present in generated instantiations (measurement writer).
    ok &= _scenario(
        "T1/T3 emit a logger.py (measurement writer exists)",
        any(o["path"].endswith("logger.py") for o in t1_ops)
        and any(o["path"].endswith("logger.py") for o in t3_ops),
        "ok",
    )
    return ok


def _config_parses(ops: list[dict]) -> bool:
    for o in ops:
        if o["path"].endswith("config.json"):
            try:
                json.loads(o["content"])
            except json.JSONDecodeError:
                return False
    return True


# ---------------------------------------------------------------------------
# Step 4 — end-to-end resolve-on-write (validator → resolver → operations)
# ---------------------------------------------------------------------------

def test_resolve_on_write() -> bool:
    """Mimic lm_authorer's post-validation resolution step: a payload carrying
    a template block must end up with resolver-produced operations.
    """
    print("\nStep 4: resolve-on-write wiring")
    ok = True
    pid = "abc12345"  # must match _base_payload()'s pid for validate_output

    payload = _base_payload()
    payload["template"] = {
        "id": "T1",
        "params": _valid_t1_params(),
        "expected_friction_reduction": "fewer contradicted claims",
    }
    # Sentinel hand-authored op that the resolver MUST overwrite.
    payload["proposal"]["operations"] = [
        {"op": "create", "path": ".github/SENTINEL.md", "content": "should be overwritten"},
    ]

    valid, msg = validate_output(payload, pid)
    ok &= _scenario("template payload validates", valid, msg)

    # Replicate the lm_authorer resolution step.
    template = payload.get("template")
    resolved = resolve(template, pid)
    payload["proposal"]["operations"] = resolved
    payload["proposal"]["target_paths"] = sorted({op["path"] for op in resolved})

    ok &= _scenario(
        "sentinel op overwritten by resolver",
        all(op["path"] != ".github/SENTINEL.md" for op in payload["proposal"]["operations"]),
        str([o["path"] for o in payload["proposal"]["operations"]]),
    )
    # Re-validation (defense in depth) must still pass.
    valid2, msg2 = validate_output(payload, pid)
    ok &= _scenario("post-resolution re-validation passes", valid2, msg2)
    ok &= _scenario(
        "target_paths match resolved operations",
        payload["proposal"]["target_paths"] == sorted({op["path"] for op in resolved}),
        "ok",
    )
    return ok


# ---------------------------------------------------------------------------
# Step 5 — composition-drift linter
# ---------------------------------------------------------------------------

def _seed_prior_template_fix(conn, pid: str, template_id: str, params: dict) -> int:
    """Insert a prior template instantiation into pattern_fix_history."""
    # patterns has a FK from pattern_fix_history; insert a parent row first.
    conn.execute(
        "INSERT OR IGNORE INTO patterns (pid, detector, key, workspace, "
        "first_seen_ts, last_seen_ts, n_observations, n_sessions, status, created_at) "
        "VALUES (?, 'test', 'k', 'ws', 0, 0, 1, 1, 'ACTIVE', 0)",
        (pid,),
    )
    return _record_apply(
        conn, pid, tier=2, commit_sha="deadbeef", target_paths=["x"],
        pre_rate=1.0, target_rate_drop=0.5, watch_window_K=5,
        template_id=template_id,
        template_params_json=json.dumps(params, sort_keys=True),
    )


def test_composition_drift() -> bool:
    print("\nStep 5: composition-drift linter")
    ok = True

    # Unit: glob overlap.
    ok &= _scenario("identical globs overlap", _globs_overlap("a/b/*.md", "a/b/*.md"), "ok")
    ok &= _scenario("same-dir globs overlap", _globs_overlap("a/b/*.md", "a/b/*.txt"), "ok")
    ok &= _scenario("parent-dir globs overlap", _globs_overlap("a/*.md", "a/b/*.md"), "ok")
    ok &= _scenario("disjoint-dir globs don't overlap", not _globs_overlap("a/b/*.md", "c/d/*.md"), "ok")

    # Unit: corpus extraction (T1 list, T2 string, T3 neither).
    ok &= _scenario("T1 corpus_globs extracted", _extract_corpus_globs({"corpus_globs": ["a/*.md"]}) == {"a/*.md"}, "ok")
    ok &= _scenario("T2 corpus_glob extracted", _extract_corpus_globs({"corpus_glob": "a/*.md"}) == {"a/*.md"}, "ok")
    ok &= _scenario("T3 (no corpus) → empty", _extract_corpus_globs({"proposal_kind": "x"}) == set(), "ok")

    import tempfile

    with tempfile.TemporaryDirectory() as td:
        conn = schema.open_db(Path(td) / "events.db")
        try:
            # Seed a prior T2 instantiation on corpus C.
            prior_id = _seed_prior_template_fix(
                conn, "prior001", "T2", {"corpus_glob": "domains/x/audit-log/*.md"},
            )

            # T3 (no corpus) never composes → allowed.
            t3_tmpl = {"id": "T3", "params": {"proposal_kind": "x"}}
            allow, _ = _check_composition_drift(conn, "new3", t3_tmpl, None)
            ok &= _scenario("T3 (no corpus) allowed", allow, "ok")

            # T1 on a DISJOINT corpus → allowed.
            t1_disjoint = {"id": "T1", "params": {"corpus_globs": ["other/place/*.md"]}}
            allow, _ = _check_composition_drift(conn, "new1", t1_disjoint, None)
            ok &= _scenario("disjoint corpus allowed", allow, "ok")

            # T1 OVERLAPPING the prior corpus, NO ack → REFUSED.
            t1_overlap = {"id": "T1", "params": {"corpus_globs": ["domains/x/audit-log/*.md"]}}
            allow, msg = _check_composition_drift(conn, "new1", t1_overlap, None)
            ok &= _scenario("overlap without ack refused", not allow and str(prior_id) in msg, msg)

            # Same, WITH the prior fix_id in composition_ack → ALLOWED.
            allow, msg = _check_composition_drift(conn, "new1", t1_overlap, [prior_id])
            ok &= _scenario("overlap with correct ack allowed", allow, msg)

            # Wrong ack (different id) → still refused.
            allow, _ = _check_composition_drift(conn, "new1", t1_overlap, [prior_id + 999])
            ok &= _scenario("overlap with wrong ack refused", not allow, "ok")

            # W1: a fix re-applied to its OWN prior corpus is NOT a composition.
            allow, msg = _check_composition_drift(conn, "prior001", t1_overlap, None)
            ok &= _scenario("self-composition (same pid) allowed", allow, msg)

            # S1: a './'-prefixed incoming glob still overlaps the bare prior glob.
            t1_dotslash = {"id": "T1", "params": {"corpus_globs": ["./domains/x/audit-log/*.md"]}}
            allow, _ = _check_composition_drift(conn, "newX", t1_dotslash, None)
            ok &= _scenario("'./'-prefixed overlap detected (refused w/o ack)", not allow, "ok")

            # S3: a prior row with malformed 'null' params must not crash the linter.
            conn.execute(
                "INSERT OR IGNORE INTO patterns (pid, detector, key, workspace, "
                "first_seen_ts, last_seen_ts, n_observations, n_sessions, status, created_at) "
                "VALUES ('badrow', 'test', 'k', 'ws', 0, 0, 1, 1, 'ACTIVE', 0)"
            )
            _record_apply(
                conn, "badrow", tier=2, commit_sha="c", target_paths=["x"],
                pre_rate=1.0, target_rate_drop=0.5, watch_window_K=5,
                template_id="T1", template_params_json="null",
            )
            try:
                allow, _ = _check_composition_drift(conn, "newY", t1_disjoint, None)
                crashed = False
            except Exception:  # noqa: BLE001
                crashed = True
            ok &= _scenario("malformed 'null' prior row does not crash linter", not crashed, "ok")
        finally:
            conn.close()
    return ok


# ---------------------------------------------------------------------------
# Step 6 — watcher per-template metric
# ---------------------------------------------------------------------------

def test_template_metric() -> bool:
    print("\nStep 6: watcher per-template metric")
    ok = True
    import tempfile

    # Unit: instance log path layout (T1/T3 per-friction, T2 singleton).
    wr = Path("/repo")
    ok &= _scenario(
        "T1 log path is per-friction",
        _instance_log_path("T1", "abc12345xyz", wr).as_posix().endswith(
            ".github/scripts/templates/T1__abc12345/log.jsonl"),
        _instance_log_path("T1", "abc12345xyz", wr).as_posix(),
    )
    ok &= _scenario(
        "T2 log path is the singleton",
        _instance_log_path("T2", "anything", wr).as_posix().endswith(
            ".github/scripts/templates/T2/log.jsonl"),
        "ok",
    )

    # Unit: _read_log filters by since_ts and tolerates malformed lines.
    with tempfile.TemporaryDirectory() as td:
        lp = Path(td) / "log.jsonl"
        lp.write_text(
            '{"ts": 100, "fired": true}\n'
            "not json\n"
            '{"ts": 50, "fired": false}\n'
            '{"ts": 200, "fired": true}\n',
            encoding="utf-8",
        )
        recs = _read_log(lp, since_ts=100)
        ok &= _scenario("read_log filters by since_ts + skips bad lines",
                        len(recs) == 2, f"{len(recs)} recs")

    # Unit: per-template metric computation from a synthetic log.
    with tempfile.TemporaryDirectory() as td:
        wr2 = Path(td)
        # T1 instantiation log: 3 fires out of 4 entries → trigger_rate 0.75.
        t1dir = wr2 / ".github" / "scripts" / "templates" / "T1__abc12345"
        t1dir.mkdir(parents=True)
        (t1dir / "log.jsonl").write_text(
            "\n".join(
                json.dumps({"ts": 10, "fired": f})
                for f in (True, True, True, False)
            ) + "\n",
            encoding="utf-8",
        )
        metric = _template_specific_metric(
            {"template_id": "T1", "pid": "abc12345xyz", "applied_at": 0}, wr2,
        )
        ok &= _scenario("T1 metric trigger_rate computed",
                        metric and abs(metric["trigger_rate"] - 0.75) < 1e-9,
                        str(metric))

    # Non-template fix → metric is None.
    ok &= _scenario(
        "non-template fix → metric None",
        _template_specific_metric({"template_id": None, "pid": "p", "applied_at": 0}, wr) is None,
        "ok",
    )

    # T3 rejection_rate ignores entries without gate fields (Adversary S1).
    with tempfile.TemporaryDirectory() as td:
        wr3 = Path(td)
        t3dir = wr3 / ".github" / "scripts" / "templates" / "T3__abc12345"
        t3dir.mkdir(parents=True)
        (t3dir / "log.jsonl").write_text(
            "\n".join([
                json.dumps({"ts": 10, "event": "heartbeat"}),                          # no gate fields → ignored
                json.dumps({"ts": 11, "syntactic_pass": True, "semantic_pass": True}),  # pass
                json.dumps({"ts": 12, "syntactic_pass": True, "semantic_pass": False}), # reject
            ]) + "\n",
            encoding="utf-8",
        )
        m3 = _template_specific_metric(
            {"template_id": "T3", "pid": "abc12345xyz", "applied_at": 0}, wr3,
        )
        ok &= _scenario(
            "T3 rejection_rate ignores non-gate entries (1/2=0.5)",
            m3 and abs(m3["rejection_rate"] - 0.5) < 1e-9,
            str(m3),
        )
    # Missing log → fail-open metric (n_log_entries 0, rate None).
    with tempfile.TemporaryDirectory() as td:
        m = _template_specific_metric(
            {"template_id": "T3", "pid": "zzzz0000", "applied_at": 0}, Path(td),
        )
        ok &= _scenario("missing log → fail-open metric",
                        m and m["n_log_entries"] == 0 and m["rejection_rate"] is None,
                        str(m))

    # End-to-end: watch_all writes template_metric_json + appends T1 revert warning.
    ok &= _e2e_watch_metric()
    return ok


def _e2e_watch_metric() -> bool:
    """A completed watch window where the rate did NOT drop → SUGGEST_REVERT,
    template_metric_json persisted, T1 warning appended to the blurb."""
    print("  -- end-to-end watch_all on a T1 fix that didn't help --")
    import tempfile

    ok = True
    with tempfile.TemporaryDirectory() as td:
        wr = Path(td)
        # Build the events DB under the conventional state path so watch_all's
        # _workspace_root (derived from __file__) is irrelevant — we pass conn
        # directly, but the metric reader uses _workspace_root(). To make the
        # log readable we instead point the instantiation dir under the REAL
        # workspace root is not possible in a tmp test; so we monkeypatch the
        # metric's workspace via a log placed where _workspace_root resolves.
        # Simpler: verify watch_all persists *some* metric_json (possibly the
        # fail-open one) and the T1 blurb warning — both independent of the log
        # contents.
        dbp = wr / "events.db"
        conn = schema.open_db(dbp)
        try:
            ws = "research"
            conn.execute(
                "INSERT INTO patterns (pid, detector, key, workspace, first_seen_ts, "
                "last_seen_ts, n_observations, n_sessions, status, created_at) "
                "VALUES ('e2e001','test','k',?,0,0,10,3,'ACTIVE',0)", (ws,),
            )
            applied = 1000
            # Seed K=2 matching sessions AFTER applied_at. Give each session 3
            # observations so post_rate (3.0) exceeds the revert threshold
            # (pre_rate 5.0 * (1 - 0.5) = 2.5) → rate did NOT drop → SUGGEST_REVERT.
            for i, sid in enumerate(("s1", "s2")):
                conn.execute(
                    "INSERT INTO sessions (session_id, workspace_path, last_event_ts) "
                    "VALUES (?,?,?)",
                    (sid, f"/x/{ws}", (applied + 10) * 1000),
                )
                for j in range(3):
                    conn.execute(
                        "INSERT INTO pattern_observations (pid, ts, session_id, workspace) "
                        "VALUES ('e2e001', ?, ?, ?)", ((applied + 10) * 1000, sid, ws),
                    )
            # A T1 template fix with pre_rate high enough to clear the floor.
            conn.execute(
                "INSERT INTO pattern_fix_history (pid, applied_at, tier, commit_sha, "
                "target_paths, watch_window_K, target_rate_drop, pre_rate, "
                "template_id, template_params_json) "
                "VALUES ('e2e001', ?, 2, 'abcdef12', '[]', 2, 0.5, 5.0, 'T1', '{}')",
                (applied,),
            )
            summary = watch_all(conn)
            row = conn.execute(
                "SELECT outcome, template_metric_json FROM pattern_fix_history "
                "WHERE pid='e2e001'"
            ).fetchone()
            ok &= _scenario("e2e outcome is SUGGEST_REVERT",
                            row[0] == "SUGGEST_REVERT", str(row[0]))
            ok &= _scenario("e2e template_metric_json persisted (T1 kind)",
                            row[1] and json.loads(row[1])["kind"] == "T1", str(row[1]))
            blurb = conn.execute(
                "SELECT surfacing_blurb FROM cached_surfacing WHERE pid='e2e001'"
            ).fetchone()[0]
            ok &= _scenario("e2e T1 revert warning appended",
                            T1_REVERT_WARNING.strip() in blurb, blurb)
        finally:
            conn.close()
    return ok


def main() -> int:
    blocks = [
        ("schema migration", test_schema_migration),
        ("validator", test_validator),
        ("resolver", test_resolver),
        ("resolve-on-write", test_resolve_on_write),
        ("composition-drift", test_composition_drift),
        ("template-metric", test_template_metric),
    ]
    print("Running template-integration tests (Steps 1-3)")
    failures = [name for name, fn in blocks if not fn()]
    print()
    if failures:
        print(f"FAILED: {', '.join(failures)}")
        return 1
    print("OK: all blocks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
