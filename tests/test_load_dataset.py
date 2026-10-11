from datetime import datetime, timezone

import pytest

from src.data import load_dataset


def test_latest_snapshot_picks_newest_and_info_parses_name(tmp_path, monkeypatch):
    monkeypatch.setattr(load_dataset, "RAW_DIR", tmp_path)
    d = tmp_path / "official_data_uk"
    d.mkdir()
    for name in ["20261010T050000Z_aaaa.csv.gz", "20261011T100347Z_bf0c323c78e3.csv.gz"]:
        (d / name).write_bytes(b"")
    latest = load_dataset.latest_snapshot("official_data_uk")
    assert latest.name.startswith("20261011")
    assert load_dataset.snapshot_info(latest) == {
        "fetched_at": datetime(2026, 10, 11, 10, 3, 47, tzinfo=timezone.utc).isoformat(),
        "commit": "bf0c323c78e3",
    }


def test_latest_snapshot_without_downloads_explains_what_to_run(tmp_path, monkeypatch):
    monkeypatch.setattr(load_dataset, "RAW_DIR", tmp_path)
    with pytest.raises(FileNotFoundError, match="fetch_data.py"):
        load_dataset.latest_snapshot("official_data_uk")
