"""CLI: download fresh immutable snapshots of the alert data.

Usage:
    .venv/bin/python scripts/fetch_data.py          # GitHub long-history dataset (~seconds)
    .venv/bin/python scripts/fetch_data.py --api    # + alerts.in.ua 30-day history (~13 min)
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.fetch_api import snapshot_history  # noqa: E402
from src.data.load_dataset import download_all, snapshot_info  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api", action="store_true", help="also snapshot alerts.in.ua 30-day history")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    for name, path in download_all().items():
        size_mb = path.stat().st_size / 1e6
        print(f"{name}: {path} ({size_mb:.1f} MB gz) {snapshot_info(path)}")
    if args.api:
        print(f"alerts.in.ua: {snapshot_history()}")


if __name__ == "__main__":
    main()
