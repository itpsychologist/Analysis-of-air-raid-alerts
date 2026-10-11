import pandas as pd
import pytest

from src.data.merge import EXCLUDED_OBLASTS, hourly_activity, normalize_official, to_oblast_intervals

T = lambda s: pd.Timestamp(s, tz="UTC")  # noqa: E731


def events(rows):
    return pd.DataFrame(rows, columns=["oblast", "level", "started_at", "finished_at"]).assign(
        started_at=lambda d: pd.to_datetime(d["started_at"], utc=True),
        finished_at=lambda d: pd.to_datetime(d["finished_at"], utc=True),
    )


def test_overlapping_and_nested_alerts_merge_into_one_interval():
    ev = events([
        ("Сумська область", "raion", "2026-01-01 10:00", "2026-01-01 11:00"),
        ("Сумська область", "raion", "2026-01-01 10:30", "2026-01-01 12:00"),  # overlaps
        ("Сумська область", "raion", "2026-01-01 10:40", "2026-01-01 10:50"),  # nested
        ("Сумська область", "raion", "2026-01-01 13:00", "2026-01-01 13:30"),  # separate
    ])
    iv = to_oblast_intervals(ev)
    assert list(zip(iv.started_at, iv.finished_at, iv.n_events)) == [
        (T("2026-01-01 10:00"), T("2026-01-01 12:00"), 3),
        (T("2026-01-01 13:00"), T("2026-01-01 13:30"), 1),
    ]


def test_oblasts_are_merged_independently():
    ev = events([
        ("Сумська область", "oblast", "2026-01-01 10:00", "2026-01-01 11:00"),
        ("Одеська область", "oblast", "2026-01-01 10:30", "2026-01-01 12:00"),
    ])
    assert len(to_oblast_intervals(ev)) == 2


def test_level_filter_and_excluded_oblasts():
    ev = events([
        ("Сумська область", "hromada", "2026-01-01 10:00", "2026-01-01 11:00"),
        (next(iter(EXCLUDED_OBLASTS)), "oblast", "2026-01-01 10:00", "2026-01-01 11:00"),
    ])
    assert to_oblast_intervals(ev).empty
    assert len(to_oblast_intervals(ev, levels=("hromada",))) == 1


def test_open_alert_is_closed_at_cutoff_not_later():
    ev = events([("Сумська область", "raion", "2026-01-01 10:00", None)])
    assert to_oblast_intervals(ev).empty  # unknown end and no cutoff: dropped
    iv = to_oblast_intervals(ev, open_until=T("2026-01-01 10:20"))
    assert iv.finished_at.iloc[0] == T("2026-01-01 10:20")


def test_hourly_activity_splits_minutes_across_hours():
    iv = pd.DataFrame({
        "oblast": ["Сумська область"],
        "started_at": [T("2026-01-01 10:45")],
        "finished_at": [T("2026-01-01 12:15")],
    })
    act = hourly_activity(iv, T("2026-01-01 10:00"), T("2026-01-01 13:00"))["Сумська область"]
    assert act.tolist() == pytest.approx([15.0, 60.0, 15.0])


def test_official_exact_duplicates_are_dropped(tmp_path):
    csv = tmp_path / "official.csv"
    row = "Сумська область,,,oblast,2026-01-01 10:00:00+00:00,2026-01-01 11:00:00+00:00,official\n"
    csv.write_text("oblast,raion,hromada,level,started_at,finished_at,source\n" + row * 2, encoding="utf-8")
    assert len(normalize_official(csv)) == 1


def test_unknown_oblast_name_fails_loudly(tmp_path):
    csv = tmp_path / "official.csv"
    csv.write_text(
        "oblast,raion,hromada,level,started_at,finished_at,source\n"
        "Атлантида,,,oblast,2026-01-01 10:00:00+00:00,2026-01-01 11:00:00+00:00,official\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unknown oblast"):
        normalize_official(csv)


def test_raion_names_are_harmonized_across_sources(tmp_path):
    csv = tmp_path / "official.csv"
    csv.write_text(
        "oblast,raion,hromada,level,started_at,finished_at,source\n"
        "Харківська область,Куп’янський район,,raion,2026-01-01 10:00:00+00:00,2026-01-01 11:00:00+00:00,official\n"
        "Львівська область,Червоноградський район,,raion,2026-01-01 10:00:00+00:00,2026-01-01 11:00:00+00:00,official\n",
        encoding="utf-8",
    )
    assert sorted(normalize_official(csv)["raion"]) == ["Куп'янський район", "Шептицький район"]


def test_oblast_alert_is_broadcast_to_all_raions_and_kyiv_stays_one_unit():
    from src.data.merge import expand_to_raions

    ev = events([
        ("Сумська область", "oblast", "2025-01-01 10:00", "2025-01-01 11:00"),
        ("Сумська область", "raion", "2026-01-01 10:00", "2026-01-01 11:00"),
        ("м. Київ", "oblast", "2026-01-01 10:00", "2026-01-01 11:00"),
        ("Сумська область", "hromada", "2026-01-01 10:00", "2026-01-01 11:00"),
    ]).assign(raion=[None, "Сумський район", None, None])
    raion_map = {"Сумська область": ["Конотопський район", "Сумський район"]}
    out = expand_to_raions(ev, raion_map)
    assert sorted(zip(out.unit, out.started_at.dt.year)) == [
        ("Конотопський район", 2025),
        ("Сумський район", 2025),
        ("Сумський район", 2026),
        ("м. Київ", 2026),
    ]


def _reference_hourly(intervals, start, end, key):
    """Slow, obviously-correct oracle for hourly_activity."""
    hours = pd.date_range(start.floor("h"), end.ceil("h"), freq="h", tz="UTC", inclusive="left")
    out = pd.DataFrame(0.0, index=hours, columns=sorted(intervals[key].unique()))
    inside = intervals[(intervals.finished_at > start) & (intervals.started_at < end)]
    for r in inside.itertuples(index=False):
        s, e = max(r.started_at, start), min(r.finished_at, end)
        h = s.floor("h")
        while h < e:
            nxt = h + pd.Timedelta(hours=1)
            out.at[h, getattr(r, key)] += (min(e, nxt) - max(s, h)).total_seconds() / 60
            h = nxt
    return out


def test_hourly_activity_matches_reference_on_random_intervals():
    import numpy as np

    rng = np.random.default_rng(0)
    rows = []
    for unit in ["A", "B", "C"]:
        t = T("2026-01-01")
        for _ in range(40):
            t += pd.Timedelta(minutes=int(rng.integers(0, 300)))
            dur = pd.Timedelta(minutes=int(rng.integers(0, 400)))
            rows.append((unit, t, t + dur))
            t += dur
    iv = pd.DataFrame(rows, columns=["unit", "started_at", "finished_at"])
    start, end = T("2026-01-02 03:17"), T("2026-01-06 20:41")
    got = hourly_activity(iv, start, end, key="unit")
    want = _reference_hourly(iv, start, end, key="unit")
    pd.testing.assert_frame_equal(got, want, check_exact=False, atol=1e-9)


def test_hourly_activity_edges():
    iv = pd.DataFrame({
        "unit": ["A", "A", "B"],
        "started_at": [T("2026-01-01 10:00"), T("2026-01-01 10:40"), T("2026-01-01 09:00")],
        "finished_at": [T("2026-01-01 10:10"), T("2026-01-01 11:00"), T("2026-01-01 09:00")],
    })
    act = hourly_activity(iv, T("2026-01-01 10:00"), T("2026-01-01 12:00"), key="unit")
    assert act["A"].tolist() == pytest.approx([30.0, 0.0])  # two pieces in one hour; end on boundary
    assert act["B"].tolist() == [0.0, 0.0]  # outside window and zero-length: column kept, all zero


def test_clip_intervals_trims_to_window():
    from src.data.merge import clip_intervals

    iv = pd.DataFrame({
        "started_at": [T("2026-01-01 08:00"), T("2026-01-01 11:00"), T("2026-01-01 14:00")],
        "finished_at": [T("2026-01-01 10:30"), T("2026-01-01 12:00"), T("2026-01-01 15:00")],
    })
    out = clip_intervals(iv, T("2026-01-01 10:00"), T("2026-01-01 11:30"))
    assert list(zip(out.started_at, out.finished_at)) == [
        (T("2026-01-01 10:00"), T("2026-01-01 10:30")),
        (T("2026-01-01 11:00"), T("2026-01-01 11:30")),
    ]


def test_normalize_api_snapshot_reads_titles_and_skips_manifest(tmp_path):
    import json

    from src.data.merge import normalize_api_snapshot

    def rec(i, title, typ, start, end):
        return {"id": i, "location_title": title, "location_type": typ, "location_oblast": "Харківська область",
                "alert_type": "air_raid", "started_at": start, "finished_at": end}

    (tmp_path / "22.json").write_text(json.dumps([
        rec(1, "Куп’янський район", "raion", "2026-10-01T10:00:00Z", "2026-10-01T11:00:00Z"),
        rec(1, "Куп’янський район", "raion", "2026-10-01T10:00:00Z", "2026-10-01T11:00:00Z"),  # dup id
        rec(2, "м. Харків", "city", "2026-10-01T10:00:00Z", None),
    ]), encoding="utf-8")
    (tmp_path / "manifest.json").write_text("{}", encoding="utf-8")
    ev = normalize_api_snapshot(tmp_path)
    assert len(ev) == 2
    raion = ev[ev.level == "raion"].iloc[0]
    assert raion.raion == "Куп'янський район" and raion.source == "alerts_in_ua"
    city = ev[ev.level == "hromada"].iloc[0]
    assert city.hromada == "м. Харків" and pd.isna(city.finished_at)
