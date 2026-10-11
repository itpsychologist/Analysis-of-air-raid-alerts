"""
text.py — Text features from Air Force channel posts (PLAN.md 2.3, 5.1, stage 4).
=================================================================================
Each post (src/data/telegram.py) is tagged with

  - threat categories (keyword regexes): drones, jet drones, ballistic, missiles, KABs, aviation, threat;
  - mentioned oblasts: oblast names, "-щина" region names and major cities;
  - launched counts from the morning reports, read only from the "противник атакував ..."
    sentence (the same post also lists shot-down and hit counts).

Hourly matrices count posts by the UTC hour of *publication*. As for the alert
history, the feature row at origin t reads only hours < t, so a post is visible at t
iff date_utc < t (PLAN 3.1.7). Dates mentioned in the text are ignored. Hours before
the oldest downloaded post are not covered and give NaN.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from src.features.geo import neighbors
from src.features.panel import Panel

CATEGORIES = {
    "drone": r"бпла|шахед|дрон|🛵|мопед",
    "jet_drone": r"реактивн",
    "ballistic": r"балістик|балістичн",
    "missile": r"ракет|крилат|калібр|х-101|х-22|х-59|х-69|🚀|швидкісн",
    "kab": r"\bкаб|💣",
    "aviation": r"ту-95|ту-160|ту-22|міг-31|зліт|борт|авіаці",
    "threat": r"загроз",
}

# Lower-case stems; apostrophes are normalised to ' first. Stems cover the і/о and ї/є
# alternation of oblique cases (Харків / Харкова, Миколаїв / Миколаєва).
# Rules frozen before the final test: their hash is stored in model_config.json.
OBLAST_PATTERNS = {
    "Вінницька область": r"вінниц|вінни[ч]+ин",
    "Волинська область": r"волин|луцьк|ковел",
    "Дніпропетровська область": r"дніпропетров|дніпровщин|\bдніпр[оау]\b|\bдніпрі\b|крив\w* р[іо]г|нікопол|павлоград|кам'янськ",
    "Донецька область": r"донеч|донецьк|краматорськ|слов'янськ|покровськ",
    "Житомирська область": r"житомир|чуднів|бердич[іе]в|коростен|новоград-волин|звягел|малин|овруч",
    "Закарпатська область": r"закарпат|ужгород|мукачев",
    "Запорізька область": r"запоріж|запорізьк",
    "Івано-Франківська область": r"івано-франків|франківщ|прикарпат",
    "Київська область": (r"київщин|київськ\w* обл|біл\w* церкв|бровар|борисп|фастів|фастов|обухів|обухов|"
                         r"вишгород|ірпін|\bбуч[аіу]\b|бородянк|глевах|українк|славутич|чорноб|макарів|"
                         r"васильків|кагарлик|переяслав|яготин|березан|ржищів|богуслав|тетіїв|сквир|баришів|"
                         r"згурів|козин|гостомел|ворзел|димер|бровари"),
    "м. Київ": (r"\bки[їє]в(?!щин|ськ)|столиц|троєщин|оболон|позняк|дарниц|святошин|борщагів|теремк|"
                r"виноградар|чоколівк|солом'ян|голосі|осокорк|печерськ"),
    "Кіровоградська область": r"кіровоград|кропивниц|світловодськ",
    "Львівська область": r"льв[іо]в|дрогоб|\bстри[йя]|шептицьк|червоноград",
    "Миколаївська область": r"микола[їє]в|вознесенськ",
    "Одеська область": r"одещин|\bодес|ізмаїл|заток|чорноморськ|білгород-дністров|татарбун",
    "Полтавська область": r"полтав|кременчу|миргород|лубн|санжар|горішн\w* плавн|карлівк|решетилів|гадяч",
    "Рівненська область": r"\bрівн(?:е|ого|ому|енщин|енськ)",
    "Сумська область": r"сумщин|сумськ|\bсум(?:и|ах|ам)?\b|шостк|охтирк|конотоп",
    "Тернопільська область": r"тернопіл|тернопол",
    "Харківська область": r"харк[іо]в|куп'янськ|ізюм|чугу[їє]в|лозов",
    "Херсонська область": r"херсон",
    "Хмельницька область": r"хмельниц|хмельниччин|старокостянтин|кам'янець",
    "Черкаська область": r"черкас|черкащин|\bуман|сміл",
    "Чернівецька область": r"чернівец|чернівеччин|буковин",
    "Чернігівська область": r"черніг[іо]в|ніжин|прилук",
}

REPORT_ATTACK = re.compile(r"противник атакува\w*(.*?)(?:\bіз напрямк|\bз напрямк|\n\n|$)", re.S)
REPORT_DRONES = re.compile(r"(\d+)\s+(?:[\w-]+\s+){0,2}(?:бпла|безпілотн\w*|шахед\w*|дрон\w*)")
REPORT_MISSILES = re.compile(r"(\d+)\s+(?:[\w-]+\s+){0,3}ракет")

WINDOWS_OWN = (1, 3, 6)
WINDOWS_COUNTRY = (1, 3, 6, 24)


def _normalise(text: str) -> str:
    return text.lower().replace("’", "'").replace("ʼ", "'").replace("`", "'")


def tag_messages(messages: pd.DataFrame) -> pd.DataFrame:
    """
    Per post: category flags, oblast mention flags, morning-report flag and launched counts.

    Edited posts (column `edited`) are kept, since dropping them would make a post's
    presence at t depend on an edit made after t. Their text shows the edited
    version, so everything derived from the text content is NaN for them; only the
    post itself (cat_posts) and whether it is a morning report are used.
    """
    text = messages["text"].fillna("").map(_normalise)
    out = pd.DataFrame({"date_utc": messages["date_utc"].to_numpy(), "cat_posts": 1.0}, index=messages.index)
    for name, pat in CATEGORIES.items():
        out[f"cat_{name}"] = text.str.contains(pat, regex=True).astype(float)
    for oblast, pat in OBLAST_PATTERNS.items():
        out[oblast] = text.str.contains(pat, regex=True).astype(float)
    attack = text.map(lambda s: m.group(1) if (m := REPORT_ATTACK.search(s)) else None)

    def total(rx):
        return attack.map(lambda seg: np.nan if seg is None else float(sum(int(n) for n in rx.findall(seg) if int(n) < 2000)))

    out["is_report"] = attack.notna().astype(float)
    out["launched_drones"] = total(REPORT_DRONES)
    out["launched_missiles"] = total(REPORT_MISSILES)
    if "edited" in messages:
        content = [c for c in out.columns if c not in ("date_utc", "cat_posts", "is_report")]
        out.loc[messages["edited"].astype(bool).to_numpy(), content] = np.nan
    return out


def text_features(tags: pd.DataFrame, panel: Panel, unit_oblast: dict[str, str],
                  covered_from: pd.Timestamp) -> dict[str, pd.DataFrame]:
    """
    Text features at each origin t of the panel (rows = panel.hours, columns = units).

    Args:
        tags:         tag_messages() output.
        covered_from: time of the oldest downloaded post; hours before it are unknown.
    """
    hours, units = panel.hours, panel.units
    hour = tags["date_utc"].dt.floor("h")
    tags = tags[(hour >= hours[0]) & (hour <= hours[-1])]
    per_hour = tags.drop(columns="date_utc").groupby(hour[tags.index]).agg(
        {**{c: "sum" for c in tags.columns if c not in ("date_utc", "launched_drones", "launched_missiles")},
         "launched_drones": "max", "launched_missiles": "max"}
    ).reindex(hours)
    counts = per_hour.drop(columns=["launched_drones", "launched_missiles", "is_report"]).fillna(0.0)

    covered = hours >= covered_from.ceil("h")
    cov_before = np.concatenate([[False], covered[:-1]])  # row t: hour t-1 covered

    def before(x: np.ndarray) -> np.ndarray:
        shifted = np.vstack([np.full((1, x.shape[1]), np.nan), x[:-1]])
        return np.where(cov_before[:, None], shifted, np.nan)

    def rolling_sum(x: np.ndarray, w: int) -> np.ndarray:
        return pd.DataFrame(x).rolling(w, min_periods=w).sum().to_numpy() if w > 1 else x

    def broadcast(col: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame(np.repeat(col.reshape(-1, 1), len(units), axis=1), index=hours, columns=units)

    f: dict[str, pd.DataFrame] = {}
    cat_cols = [c for c in counts.columns if c.startswith("cat_")]  # incl. cat_posts = all posts
    cats = before(counts[cat_cols].to_numpy())
    for w in WINDOWS_COUNTRY:
        for j, c in enumerate(cat_cols):
            f[f"txt_{c[4:]}_{w}h"] = broadcast(rolling_sum(cats[:, [j]], w))

    oblasts = list(OBLAST_PATTERNS)
    mentions = before(counts[oblasts].to_numpy())
    adj = neighbors()
    nbr = np.array([[o2 in adj.get(o1, set()) for o2 in oblasts] for o1 in oblasts], dtype=float)
    nbr_mentions = mentions @ nbr.T  # sum over neighbouring oblasts
    col = np.array([oblasts.index(unit_oblast[u]) for u in units])
    for w in WINDOWS_OWN:
        f[f"txt_own_{w}h"] = pd.DataFrame(rolling_sum(mentions, w)[:, col], index=hours, columns=units)
        if w <= 3:
            f[f"txt_nbr_{w}h"] = pd.DataFrame(rolling_sum(nbr_mentions, w)[:, col], index=hours, columns=units)

    # Latest morning report published before t (forward-filled), and hours since it
    for name in ("launched_drones", "launched_missiles"):
        last = pd.Series(before(per_hour[[name]].to_numpy())[:, 0]).ffill().to_numpy()
        f[f"txt_{name}"] = broadcast(np.where(cov_before, np.log1p(last), np.nan))
    rows = np.arange(len(hours), dtype=float)
    has_report = np.nan_to_num(before(per_hour[["is_report"]].to_numpy())[:, 0]) > 0
    last_row = pd.Series(np.where(has_report, rows, np.nan)).ffill().to_numpy()
    f["txt_since_report"] = broadcast(np.where(cov_before, np.log1p(rows - last_row), np.nan))
    return f
