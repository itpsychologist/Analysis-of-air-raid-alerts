"""CLI: download public posts of the Air Force channel (@kpszsu) from the t.me web preview.

Writes a new immutable snapshot data/raw/telegram/<channel>/<fetched_at>.jsonl.gz;
an interrupted run resumes from its .part file. ~20 posts per request with a 2-3 s
pause, so a year of posts takes 1-2 hours.

Usage:
    .venv/bin/python scripts/fetch_telegram.py [--since 2025-11-01] [--channel kpszsu]
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.telegram import fetch_channel  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--channel", default="kpszsu")
    ap.add_argument("--since", default="2025-11-01", help="oldest publication date to fetch (UTC)")
    ap.add_argument("--delay", type=float, default=2.0, help="seconds between requests")
    args = ap.parse_args()
    fetch_channel(args.channel, pd.Timestamp(args.since, tz="UTC"), delay=args.delay,
                  log=lambda s: print(s, flush=True))


if __name__ == "__main__":
    main()
