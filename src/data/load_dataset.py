"""
load_dataset.py — Download immutable snapshots of the long-history alerts dataset.
==================================================================================
Source: https://github.com/Vadimkin/ukrainian-air-raid-sirens-dataset (MIT, updated daily)

  - official_data_uk.csv   from 2022-03-15, levels oblast / raion / hromada
  - volunteer_data_uk.csv  from 2022-02-25, oblast level only (eTryvoga)

Each download is stored as data/raw/github/<dataset>/<fetched_at>_<commit>.csv.gz
and never overwritten, so an evaluation can always name the exact data snapshot
it was run on. Ukrainian-language files are used because their oblast names match
the keys of etl.OBLAST_UID_MAP and the alerts.in.ua API.
"""

from __future__ import annotations

import gzip
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

REPO = "Vadimkin/ukrainian-air-raid-sirens-dataset"
DATASETS = ("official_data_uk", "volunteer_data_uk")
RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "github"
REQUEST_TIMEOUT = 60  # seconds


def latest_commit_sha(branch: str = "main") -> str:
    """Return the short SHA of the dataset repo's latest commit (snapshot provenance)."""
    response = requests.get(
        f"https://api.github.com/repos/{REPO}/commits/{branch}",
        headers={"Accept": "application/vnd.github+json"},
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    return response.json()["sha"][:12]


def download_snapshot(dataset: str, sha: str, fetched_at: datetime) -> Path:
    """
    Download one dataset CSV at a pinned commit and store it gzip-compressed.

    Pinning to the commit SHA (not the branch) guarantees the stored bytes match
    the recorded provenance even if the repo is updated mid-download.
    """
    target_dir = RAW_DIR / dataset
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{fetched_at:%Y%m%dT%H%M%SZ}_{sha}.csv.gz"
    if target.exists():
        logger.info("Snapshot already exists: %s", target)
        return target

    url = f"https://raw.githubusercontent.com/{REPO}/{sha}/datasets/{dataset}.csv"
    tmp = target.with_suffix(".part")
    with requests.get(url, stream=True, timeout=REQUEST_TIMEOUT) as response:
        response.raise_for_status()
        # GitHub serves Content-Encoding: gzip; decode it so we don't double-compress
        response.raw.decode_content = True
        with gzip.open(tmp, "wb") as out:
            shutil.copyfileobj(response.raw, out)
    tmp.rename(target)
    logger.info("Saved %s", target)
    return target


def download_all() -> dict[str, Path]:
    """Download every dataset at the current upstream commit. Returns {dataset: path}."""
    fetched_at = datetime.now(timezone.utc)
    sha = latest_commit_sha()
    return {name: download_snapshot(name, sha, fetched_at) for name in DATASETS}


def latest_snapshot(dataset: str) -> Path:
    """Path of the most recent stored snapshot for a dataset."""
    snapshots = sorted((RAW_DIR / dataset).glob("*.csv.gz"))
    if not snapshots:
        raise FileNotFoundError(f"No snapshots for {dataset}; run scripts/fetch_data.py first.")
    return snapshots[-1]


def snapshot_info(path: Path) -> dict[str, str]:
    """Parse fetched_at and upstream commit from a snapshot filename."""
    stamp, sha = path.name.removesuffix(".csv.gz").split("_", 1)
    fetched_at = datetime.strptime(stamp, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
    return {"fetched_at": fetched_at.isoformat(), "commit": sha}
