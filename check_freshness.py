"""Daily staleness guard: exit 0 when the local data is fresh, 2 when it is STALE.

Same check as ``python -m laliga check-freshness``. Kept at the repository root
so ``auto_update.ps1`` can call it by path. ``LALIGA_FRESHNESS_NOW`` overrides
the clock.
"""

import sys

from laliga.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["check-freshness", *sys.argv[1:]]))
