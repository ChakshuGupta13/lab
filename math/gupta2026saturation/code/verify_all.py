#!/usr/bin/env python3
"""Run all verification scripts for the TxGraffiti C4 refutation.

Usage: python verify_all.py [--quick]
  --quick: skip n=9 exhaustive search (~82s) and run only constructive proofs.
"""
import sys
import subprocess


SCRIPTS = [
    ("Friendship family F_k", "verify_friendship.py"),
    ("Unbounded separation G_{m,k}", "verify_unbounded.py"),
    ("Subdivided star S_k + tree bound", "verify_subdivided_star.py"),
    ("Regular-graph lemma", "verify_regular.py"),
    ("n=9 characterization", "verify_n9_characterization.py"),
    ("Exhaustive C4 check n=2..9", "verify_c4_exhaustive.py"),
]


def main():
    quick = "--quick" in sys.argv
    failed = []

    for name, script in SCRIPTS:
        if quick and script == "verify_c4_exhaustive.py":
            print(f"\n{'='*60}")
            print(f"SKIP (--quick): {name}")
            continue

        print(f"\n{'='*60}")
        print(f"Running: {name} ({script})")
        print("=" * 60)
        rc = subprocess.call([sys.executable, script])
        if rc != 0:
            failed.append(name)
            print(f"*** FAILED (rc={rc}) ***")

    print(f"\n{'='*60}")
    if failed:
        print(f"FAILURES: {failed}")
        sys.exit(1)
    else:
        print("ALL VERIFICATIONS PASSED")


if __name__ == "__main__":
    main()
