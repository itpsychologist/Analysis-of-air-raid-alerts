import pandas as pd

from src.data.build import find_outages, api_coverage_start, stitch

T = lambda s: pd.Timestamp(s, tz="UTC")  # noqa: E731


def ev(rows):
    df = pd.DataFrame(rows, columns=["oblast", "raion", "level", "started_at", "finished_at"])
    df["started_at"] = pd.to_datetime(df["started_at"], utc=True)
    df["finished_at"] = pd.to_datetime(df["finished_at"], utc=True)
    return df


def test_api_coverage_starts_at_first_full_day_of_the_month_window():
    assert api_coverage_start(T("2026-10-11 10:06")) == T("2026-09-12")
    assert api_coverage_start(T("2026-10-11 00:00")) == T("2026-09-11")


def test_stitch_keeps_sources_in_their_windows_and_leaves_the_gap_empty():
    official = ev([
        ("Сумська область", None, "oblast", "2026-09-01 10:00", "2026-09-01 11:00"),
        # Starts at least every 48 h: no source outage inside the official window
        ("Сумська область", "Конотопський район", "raion", "2026-09-03 00:00", "2026-09-03 00:30"),
        ("Сумська область", "Конотопський район", "raion", "2026-09-04 20:00", "2026-09-04 20:30"),
        ("Сумська область", "Сумський район", "raion", "2026-09-06 04:00", "2026-09-08 12:00"),  # runs into gap
        # The last start minus OFFICIAL_TAIL_TRIM (24 h) defines the end of official coverage
        ("Сумська область", "Конотопський район", "raion", "2026-09-07 05:00", "2026-09-07 05:20"),
    ])
    api = ev([
        ("Сумська область", "Сумський район", "raion", "2026-09-10 10:00", "2026-09-10 11:00"),  # in gap
        ("Сумська область", "Конотопський район", "raion", "2026-09-20 10:00", "2026-09-20 11:00"),
        ("Сумська область", "Сумський район", "raion", "2026-10-11 09:00", None),  # still open
    ])
    cutoff = T("2026-10-11 10:00")
    iv, windows, unit_oblast = stitch(official, api, cutoff)

    official_end = T("2026-09-06 05:00")
    api_start = T("2026-09-12")
    assert [w["source"] for w in windows] == ["official", "gap", "alerts_in_ua"]
    assert windows[1] == {"source": "gap", "reason": "between sources", "start": official_end.isoformat(),
                          "end": api_start.isoformat()}

    # The oblast-level alert reached both raions known from either source
    first = iv[iv.started_at == T("2026-09-01 10:00")]
    assert sorted(first.unit) == ["Конотопський район", "Сумський район"]
    # Nothing inside the gap, and the alert crossing into it is cut at official_end
    in_gap = (iv.finished_at > official_end) & (iv.started_at < api_start)
    assert not in_gap.any()
    assert iv[iv.source == "official"].finished_at.max() == official_end
    # The open API alert is closed at the cutoff, not left open or extended
    assert iv.finished_at.max() == cutoff
    assert unit_oblast["Сумський район"] == "Сумська область" and unit_oblast["м. Київ"] == "м. Київ"


def test_a_source_outage_becomes_a_gap_and_alerts_crossing_it_are_cut():
    official = ev([
        ("Сумська область", "Сумський район", "raion", "2026-08-28 10:00", "2026-08-28 11:00"),
        # The feed stops: the last start before the outage, and an alert "lasting" across it
        ("Сумська область", "Конотопський район", "raion", "2026-08-29 05:00", "2026-09-02 16:00"),
        ("Сумська область", "Сумський район", "raion", "2026-09-02 10:00", "2026-09-02 11:00"),  # feed is back
        ("Сумська область", "Сумський район", "raion", "2026-09-03 10:00", "2026-09-03 11:00"),
        ("Сумська область", "Сумський район", "raion", "2026-09-05 08:00", "2026-09-05 09:00"),
        ("Сумська область", "Сумський район", "raion", "2026-09-07 05:00", "2026-09-07 05:20"),
    ])
    api = ev([("Сумська область", "Сумський район", "raion", "2026-09-20 10:00", "2026-09-20 11:00")])
    iv, windows, _ = stitch(official, api, T("2026-10-11 10:00"))

    outage = (T("2026-08-29 05:00"), T("2026-09-02 10:00"))
    assert [w["source"] for w in windows] == ["official", "gap", "official", "gap", "alerts_in_ua"]
    assert windows[1] == {"source": "gap", "reason": "source outage", "start": outage[0].isoformat(),
                          "end": outage[1].isoformat()}
    assert not ((iv.finished_at > outage[0]) & (iv.started_at < outage[1])).any()
    crossing = iv[iv.unit == "Конотопський район"]
    # Cut at the outage: only the part after the feed is back remains (the part before is empty)
    assert crossing.started_at.tolist() == [outage[1]] and crossing.finished_at.tolist() == [T("2026-09-02 16:00")]
    assert find_outages(official["started_at"]) == [outage]
    assert find_outages(official["started_at"], pd.Timedelta(hours=200)) == []
