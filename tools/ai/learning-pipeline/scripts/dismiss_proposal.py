#!/usr/bin/env python3
"""dismiss_proposal.py — mark a pattern DISMISSED (user said no).
Idempotent. Removes cached surfacing so the pattern stops being suggested."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

try:
    from .schema import open_db
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from v2.schema import open_db  # type: ignore

CAMPAIGN_LOG = Path(__file__).resolve().parent.parent.parent.parent.parent / ".github" / "learning" / "campaign_log.jsonl"


def dismiss(pid: str) -> dict:
    conn = open_db()
    try:
        row = conn.execute("SELECT status FROM patterns WHERE pid=?", (pid,)).fetchone()
        if not row:
            return {"ok": False, "reason": f"unknown pid {pid}"}
        conn.execute("UPDATE patterns SET status='DISMISSED' WHERE pid=?", (pid,))
        conn.execute("DELETE FROM cached_surfacing WHERE pid=?", (pid,))
        conn.execute(
            "UPDATE surfacing_state SET status='DISPATCHED', "
            "  dispatched_intent='dismiss', dispatched_at=? "
            "WHERE pid=? AND status='OPEN'",
            (int(time.time()), pid),
        )
        return {"ok": True, "pid": pid, "prior_status": row[0]}
    finally:
        conn.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pid", required=True)
    args = ap.parse_args()
    result = dismiss(args.pid)
    if result.get("ok"):
        try:
            CAMPAIGN_LOG.parent.mkdir(parents=True, exist_ok=True)
            with CAMPAIGN_LOG.open("a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "event": "dismissed",
                    "pid": args.pid,
                    "prior_status": result.get("prior_status"),
                    "ts": int(time.time()),
                }, separators=(",", ":")) + "\n")
        except OSError:
            pass
    print(json.dumps(result, indent=2))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
