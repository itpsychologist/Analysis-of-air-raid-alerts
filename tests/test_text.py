"""
Text features (stage 4): tagging of Air Force posts and leakage of text features (PLAN 3.2).
"""

import numpy as np
import pandas as pd
import pytest

from src.data.telegram import parse_page
from src.features.panel import build_panel
from src.features.text import OBLAST_PATTERNS, tag_messages, text_features
from tests.test_features import COVERAGE, UNIT_OBLAST, random_intervals

T = lambda s: pd.Timestamp(s, tz="UTC")  # noqa: E731


def posts(rows) -> pd.DataFrame:
    return pd.DataFrame([{"date_utc": T(d), "text": t} for d, t in rows])


@pytest.mark.parametrize("text, cats, oblasts", [
    ("🛵 Групи БпЛА із Черкаської у Київську обл., в напрямку Білої Церкви",
     {"drone"}, {"Черкаська область", "Київська область"}),
    ("💣 КАБи на Харківщину", {"kab"}, {"Харківська область"}),
    ("🏍 Кривий Ріг: реактивний БпЛА на сході від міста, курс на південь.",
     {"drone", "jet_drone"}, {"Дніпропетровська область"}),
    ("Загроза застосування балістики! Київ, Київщина", {"ballistic", "threat"},
     {"м. Київ", "Київська область"}),
    ("🚀 Швидкісна ціль на сході Харківщини, курс на південь.", {"missile"}, {"Харківська область"}),
    ("Групи ударних БпЛА на півдні Одещини, курсом на Ізмаїл", {"drone"}, {"Одеська область"}),
    ("Зліт МіГ-31К", {"aviation"}, set()),
    ("🏍 Реактивний БпЛА на півдні  Вінничини", {"drone", "jet_drone"}, {"Вінницька область"}),
    ("🏍 Реактивний БпЛА  на Ірпінь", {"drone", "jet_drone"}, {"Київська область"}),
    ("Чоколівка.", set(), {"м. Київ"}),
    ("🚀Швидкісна ціль у напрямку Харкова", {"missile"}, {"Харківська область"}),
    ("🛵 БпЛА над Черніговом!", {"drone"}, {"Чернігівська область"}),
    ("🛵🛸 Ворожий БпЛА в напрямку Сум з півночі.", {"drone"}, {"Сумська область"}),
    ("⚠ Увага! 🛫 Активність ворожої тактичної авіації на південному напрямку", {"aviation"}, set()),
])
def test_posts_are_tagged_with_categories_and_oblasts(text, cats, oblasts):
    tags = tag_messages(posts([("2026-01-01 00:00", text)])).iloc[0]
    found_cats = {c[4:] for c in tags.index if c.startswith("cat_") and c != "cat_posts" and tags[c] == 1}
    assert found_cats == cats
    assert {o for o in OBLAST_PATTERNS if tags[o] == 1} == oblasts


def test_launched_counts_come_from_morning_reports_only():
    report = (
        "⚡️ ЗБИТО/ПОДАВЛЕНО 94 ВОРОЖІ БПЛА\n\n➖➖➖\nУ ніч на 11 жовтня (з 18:00 10 жовтня) противник атакував "
        "116 ударними БпЛА типу Shahed (95 із них - реактивні), Гербера, 3 балістичними ракетами Іскандер-М та "
        "2 крилатими ракетами Х-101 із напрямків: Орел, Курськ.\n\nЗа попередніми даними збито 94 ворожі БпЛА. "
        "Зафіксовано влучання 10 БпЛА противника на 7 локаціях."
    )
    tags = tag_messages(posts([
        ("2026-10-11 05:39", report),
        ("2026-10-11 07:00", "Група з 12 БпЛА курсом на Полтаву"),  # not a report
    ]))
    # only the "противник атакував ..." sentence counts: not the 94 shot down or the 10 hits
    assert tags["launched_drones"].iloc[0] == 116 and tags["launched_missiles"].iloc[0] == 5
    assert np.isnan(tags["launched_drones"].iloc[1])


def random_posts(seed, start, end, n=400) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    pool = ["🛵 БпЛА на Київщині", "💣 КАБи на Харківщину", "Загроза балістики для Києва",
            "У ніч на ... противник атакував 80 ударними БпЛА та 3 ракетами", "🚀 Ракета на Полтаву",
            "Реактивний БпЛА курсом на Кременчук"]
    secs = rng.integers(0, int((end - start).total_seconds()), n)
    return pd.DataFrame({"date_utc": [start + pd.Timedelta(seconds=int(s)) for s in secs],
                         "text": rng.choice(pool, n)}).sort_values("date_utc").reset_index(drop=True)


@pytest.mark.parametrize("t", [T("2025-10-10 09:00"), T("2025-10-26 01:00"), T("2025-11-02 18:00")])
def test_text_features_at_t_ignore_posts_published_at_or_after_t(t):
    panel = build_panel(random_intervals(), COVERAGE)
    covered_from = T("2025-10-03 12:34")
    msgs = random_posts(0, covered_from, T("2025-11-05"))
    F = text_features(tag_messages(msgs), panel, UNIT_OBLAST, covered_from)
    future = random_posts(1, t, T("2025-11-05"))
    msgs2 = pd.concat([msgs[msgs["date_utc"] < t], future], ignore_index=True)
    F2 = text_features(tag_messages(msgs2), panel, UNIT_OBLAST, covered_from)
    for name in F:
        pd.testing.assert_series_equal(F[name].loc[t], F2[name].loc[t], obj=f"{name} at {t}")
    assert all(F[n].loc[t].notna().all() for n in F), "every feature must be checked on real values"


def test_post_exactly_at_t_is_not_visible_and_uncovered_hours_are_nan():
    panel = build_panel(random_intervals(), COVERAGE)
    covered_from = T("2025-10-05 00:00")
    msgs = posts([("2025-10-08 09:59:59", "💣 КАБи на Харківщину"), ("2025-10-08 11:00:00", "💣 КАБи на Харківщину")])
    F = text_features(tag_messages(msgs), panel, UNIT_OBLAST, covered_from)
    assert F["txt_kab_1h"].loc[T("2025-10-08 10:00"), "H1"] == 1
    assert F["txt_kab_1h"].loc[T("2025-10-08 11:00"), "H1"] == 0  # published at 11:00:00, not before
    assert F["txt_own_1h"].loc[T("2025-10-08 10:00"), "H1"] == 1
    assert F["txt_own_1h"].loc[T("2025-10-08 10:00"), "A1"] == 0
    assert F["txt_nbr_1h"].loc[T("2025-10-08 10:00"), "P1"] == 1  # Kharkiv borders Poltava
    assert np.isnan(F["txt_kab_1h"].loc[T("2025-10-04 10:00"), "H1"])  # before the oldest post


def test_edited_posts_count_as_posts_but_their_content_is_ignored():
    """An edit after t must not change features at t: the post stays, its (edited) text is not read."""
    panel = build_panel(random_intervals(), COVERAGE)
    report = "У ніч на 8 жовтня противник атакував 120 ударними БпЛА. КАБи на Харківщину"
    msgs = posts([("2025-10-08 05:30", report)]).assign(edited=False)
    F = text_features(tag_messages(msgs), panel, UNIT_OBLAST, T("2025-10-05"))
    F_edited = text_features(tag_messages(msgs.assign(edited=True)), panel, UNIT_OBLAST, T("2025-10-05"))
    t = T("2025-10-08 07:00")
    assert F["txt_posts_3h"].loc[t, "H1"] == F_edited["txt_posts_3h"].loc[t, "H1"] == 1
    assert F["txt_since_report"].loc[t, "H1"] == F_edited["txt_since_report"].loc[t, "H1"] == pytest.approx(np.log1p(1))
    assert F["txt_kab_3h"].loc[t, "H1"] == 1 and F_edited["txt_kab_3h"].loc[t, "H1"] == 0
    assert F["txt_launched_drones"].loc[t, "H1"] == pytest.approx(np.log1p(120))
    assert np.isnan(F_edited["txt_launched_drones"].loc[t, "H1"])


def test_preview_parser_reads_ids_times_edits_and_line_breaks():
    html = """
    <div class="tgme_widget_message" data-post="kpszsu/101"><div class="tgme_widget_message_text">БпЛА<br/>на Сумщину</div>
      <div class="tgme_widget_message_meta"><span>edited</span><time datetime="2026-01-01T03:00:00+02:00"></time></div></div>
    <div class="tgme_widget_message" data-post="kpszsu/100"><div class="tgme_widget_message_text">КАБи</div>
      <div class="tgme_widget_message_meta"><time datetime="2026-01-01T00:30:00+00:00"></time></div></div>
    <div class="tgme_widget_message" data-post="other/7"><time datetime="2026-01-01T00:00:00+00:00"></time></div>
    """
    p = parse_page(html, "kpszsu")
    assert [x["message_id"] for x in p] == [100, 101]
    assert p[1]["date_utc"] == "2026-01-01T01:00:00+00:00" and p[1]["edited"] and not p[0]["edited"]
    assert p[1]["text"] == "БпЛА\nна Сумщину"
