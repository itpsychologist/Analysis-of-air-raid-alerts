"""CLI: build data/processed/ (raion-level alert intervals) from the latest raw snapshots.

Usage:
    .venv/bin/python scripts/build_dataset.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.build import PROCESSED_DIR, build, save  # noqa: E402


def main() -> None:
    intervals, coverage = build()
    save(intervals, coverage)
    print(f"{len(intervals):,} intervals for {len(coverage['units'])} units -> {PROCESSED_DIR}")
    for w in coverage["windows"]:
        print(f"  {w['source']:13s} {w['start']} .. {w['end']}")


if __name__ == "__main__":
    main()
