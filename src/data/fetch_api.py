"""
fetch_api.py — Immutable raw snapshots of the alerts.in.ua 30-day regional history.
===================================================================================
Stores the untouched JSON per oblast under
data/raw/alerts_in_ua/<fetched_at>/<oblast_uid>.json, so the "tail" of recent data
used for the final evaluation can always be traced back to an exact download.

Rate limit: the history endpoint allows 2 req/min, hence the 31 s pause
(same as etl.fetch_and_process_data); a full run takes ~13 minutes.
"""

from __future__ import annotations

import json
import logging
import os
import time
import tomllib
from datetime import datetime, timezone
from pathlib import Path

from etl import OBLAST_UID_MAP, fetch_oblast_history

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "alerts_in_ua"
RATE_LIMIT_PAUSE = 31  # seconds


def get_token() -> str:
    """Token from ALERTS_API_TOKEN env var or .streamlit/secrets.toml (same as app.py)."""
    token = os.getenv("ALERTS_API_TOKEN", "").strip()
    if token:
        return token
    secrets = PROJECT_ROOT / ".streamlit" / "secrets.toml"
    with secrets.open("rb") as fh:
        return tomllib.load(fh)["ALERTS_API_TOKEN"]


def snapshot_history(token: str | None = None) -> Path:
    """Download month_ago history for every oblast. Returns the snapshot directory."""
    token = token or get_token()
    fetched_at = datetime.now(timezone.utc)
    out_dir = RAW_DIR / f"{fetched_at:%Y%m%dT%H%M%SZ}"
    out_dir.mkdir(parents=True, exist_ok=False)

    failures: dict[str, str] = {}
    oblasts = list(OBLAST_UID_MAP.items())
    for idx, (name, uid) in enumerate(oblasts):
        try:
            records = fetch_oblast_history(token, uid, "month_ago")
            (out_dir / f"{uid}.json").write_text(
                json.dumps(records, ensure_ascii=False), encoding="utf-8"
            )
            logger.info("[%d/%d] %s: %d records", idx + 1, len(oblasts), name, len(records))
        except RuntimeError as exc:
            failures[name] = str(exc)
            logger.warning("[%d/%d] %s failed: %s", idx + 1, len(oblasts), name, exc)
        if idx < len(oblasts) - 1:
            time.sleep(RATE_LIMIT_PAUSE)

    manifest = {
        "fetched_at": fetched_at.isoformat(),
        "endpoint": "/v1/regions/{uid}/alerts/month_ago.json",
        "failures": failures,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_dir


def latest_api_snapshot() -> Path:
    """Most recent complete snapshot directory (one that has a manifest)."""
    snapshots = sorted(p.parent for p in RAW_DIR.glob("*/manifest.json"))
    if not snapshots:
        raise FileNotFoundError("No alerts.in.ua snapshots; run scripts/fetch_data.py --api first.")
    return snapshots[-1]
