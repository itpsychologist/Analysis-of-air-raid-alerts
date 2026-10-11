"""
telegram.py — Public Telegram channel posts from the web preview (t.me/s/<channel>).
====================================================================================
No account or API keys: the public preview serves ~20 posts per page and pages
backwards with ?before=<message_id>. Requests are sequential with a delay and back
off on errors / rate limiting.

Snapshot (immutable, PLAN.md 2.2): data/raw/telegram/<channel>/<fetched_at>.jsonl.gz,
one record per post:

    channel | message_id | date_utc (publication time, ISO) | edited (bool) | text

Leakage rules (PLAN.md 2.3, 3.1.7): only the publication time counts as "known at";
dates mentioned in the text are ignored. The preview shows the current text of an
edited post, not the original, so edited posts are flagged: features keep the post
but ignore its content (src/features/text.py). Dropping them instead would make a
post's presence at t depend on an edit made later. Posts deleted before the download are missing (survivorship;
documented, not fixable from the preview).
"""

from __future__ import annotations

import gzip
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "telegram"
PREVIEW_URL = "https://t.me/s/{channel}"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; air-raid-alerts-research/1.0)"}


def parse_page(html: str, channel: str) -> list[dict]:
    """Posts on one preview page, oldest first."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    posts = []
    for m in soup.select("div.tgme_widget_message[data-post]"):
        name, _, mid = m["data-post"].partition("/")
        time_tag = m.select_one(".tgme_widget_message_meta time[datetime]") or m.select_one("time[datetime]")
        if name.lower() != channel.lower() or not mid.isdigit() or time_tag is None:
            continue
        meta = m.select_one(".tgme_widget_message_meta")
        text_div = m.select_one("div.tgme_widget_message_text")
        if text_div is not None:
            for br in text_div.find_all("br"):
                br.replace_with("\n")
        posts.append({
            "channel": channel,
            "message_id": int(mid),
            "date_utc": pd.Timestamp(time_tag["datetime"]).tz_convert("UTC").isoformat(),
            "edited": bool(meta and "edited" in meta.get_text(" ", strip=True).lower()),
            "text": text_div.get_text("", strip=False).strip() if text_div is not None else "",
        })
    return sorted(posts, key=lambda p: p["message_id"])


def _get(session: requests.Session, url: str, params: dict, retries: int = 6) -> str:
    for attempt in range(retries):
        try:
            r = session.get(url, params=params, headers=HEADERS, timeout=30)
            if r.status_code == 200:
                return r.text
            wait = 60 * (attempt + 1) if r.status_code == 429 else 10 * (attempt + 1)
        except requests.RequestException:
            wait = 10 * (attempt + 1)
        time.sleep(wait)
    raise RuntimeError(f"giving up on {url} {params}")


def fetch_channel(channel: str, since: pd.Timestamp, delay: float = 2.0, log=print) -> Path:
    """
    Download posts newer than `since` (UTC) into a new snapshot file, newest page first.
    Progress goes to a `.part` file next to it; rerunning resumes from the oldest post
    fetched so far.
    """
    out_dir = RAW_DIR / channel
    out_dir.mkdir(parents=True, exist_ok=True)
    parts = sorted(out_dir.glob("*.jsonl.part"))
    if parts:
        part = parts[-1]
        seen = [json.loads(line) for line in part.read_text(encoding="utf-8").splitlines() if line]
        before = min(p["message_id"] for p in seen) if seen else None
        log(f"resuming {part.name} from message {before}")
    else:
        part = out_dir / f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.jsonl.part"
        before = None

    session = requests.Session()
    url = PREVIEW_URL.format(channel=channel)
    pages = 0
    with part.open("a", encoding="utf-8") as f:
        while True:
            posts = parse_page(_get(session, url, {"before": before} if before else {}), channel)
            if not posts:
                break
            for p in posts:
                if before is None or p["message_id"] < before:
                    f.write(json.dumps(p, ensure_ascii=False) + "\n")
            f.flush()
            oldest = posts[0]
            pages += 1
            if pages % 50 == 0:
                log(f"{pages} pages, at {oldest['date_utc']} (id {oldest['message_id']})")
            if pd.Timestamp(oldest["date_utc"]) < since or oldest["message_id"] <= 1:
                break
            before = oldest["message_id"]
            time.sleep(delay + random.uniform(0, delay / 2))

    final = part.with_suffix("").with_suffix(".jsonl.gz")
    lines = sorted(set(part.read_text(encoding="utf-8").splitlines()))
    with gzip.open(final, "wt", encoding="utf-8") as g:
        g.write("\n".join(lines) + "\n")
    part.unlink()
    log(f"saved {final} ({len(lines)} posts)")
    return final


def latest_snapshot(channel: str = "kpszsu") -> Path:
    files = sorted((RAW_DIR / channel).glob("*.jsonl.gz"))
    if not files:
        raise FileNotFoundError(f"no snapshot in {RAW_DIR / channel}; run scripts/fetch_telegram.py")
    return files[-1]


def load_messages(path: Path, drop_edited: bool = True) -> pd.DataFrame:
    """Posts as a DataFrame sorted by publication time (date_utc tz-aware UTC)."""
    with gzip.open(path, "rt", encoding="utf-8") as g:
        df = pd.DataFrame([json.loads(line) for line in g if line.strip()])
    df = df.drop_duplicates("message_id")
    df["date_utc"] = pd.to_datetime(df["date_utc"], utc=True)
    if drop_edited:
        df = df[~df["edited"]]
    return df.sort_values(["date_utc", "message_id"]).reset_index(drop=True)
