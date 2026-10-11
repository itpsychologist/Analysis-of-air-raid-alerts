"""
geo.py — Oblast adjacency for spatial features.
================================================
Land borders between oblasts (Ukrainian names as in etl.OBLAST_UID_MAP). Attacks
move across the country, so alerts in neighbouring oblasts are a leading signal.
Луганська область is excluded from the units (permanent alert), so its borders
are not listed; Crimea and foreign borders are irrelevant here.
"""

from __future__ import annotations

import numpy as np

_BORDERS = {
    "Вінницька область": ["Житомирська область", "Київська область", "Черкаська область",
                          "Кіровоградська область", "Одеська область", "Хмельницька область",
                          "Чернівецька область"],
    "Волинська область": ["Рівненська область", "Львівська область"],
    "Дніпропетровська область": ["Полтавська область", "Харківська область", "Донецька область",
                                 "Запорізька область", "Херсонська область", "Миколаївська область",
                                 "Кіровоградська область"],
    "Донецька область": ["Харківська область", "Запорізька область"],
    "Житомирська область": ["Київська область", "Хмельницька область", "Рівненська область"],
    "Закарпатська область": ["Львівська область", "Івано-Франківська область"],
    "Запорізька область": ["Херсонська область"],
    "Івано-Франківська область": ["Львівська область", "Тернопільська область", "Чернівецька область"],
    "Київська область": ["м. Київ", "Черкаська область", "Полтавська область", "Чернігівська область"],
    "Кіровоградська область": ["Черкаська область", "Полтавська область", "Миколаївська область",
                               "Одеська область"],
    "Львівська область": ["Рівненська область", "Тернопільська область"],
    "Миколаївська область": ["Одеська область", "Херсонська область"],
    "Полтавська область": ["Чернігівська область", "Сумська область", "Харківська область",
                           "Черкаська область"],
    "Рівненська область": ["Тернопільська область", "Хмельницька область"],
    "Сумська область": ["Чернігівська область", "Харківська область"],
    "Тернопільська область": ["Хмельницька область", "Чернівецька область"],
    "Хмельницька область": ["Чернівецька область"],
}


def neighbors() -> dict[str, set[str]]:
    """Symmetric adjacency: oblast -> set of bordering oblasts."""
    out: dict[str, set[str]] = {}
    for a, bs in _BORDERS.items():
        for b in bs:
            out.setdefault(a, set()).add(b)
            out.setdefault(b, set()).add(a)
    return out


def group_matrices(units: list[str], unit_oblast: dict[str, str]) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (same_oblast, neighbor_oblasts), both units × units 0/1 matrices with rows
    normalised to sum to 1, so `X @ M.T` averages X over the unit's oblast (itself
    included) or over all units of the bordering oblasts.
    """
    adj = neighbors()
    ob = np.array([unit_oblast[u] for u in units])
    same = (ob[:, None] == ob[None, :]).astype(float)
    nbr = np.array([[o2 in adj.get(o1, set()) for o2 in ob] for o1 in ob], dtype=float)
    return same / same.sum(1, keepdims=True), nbr / np.maximum(nbr.sum(1, keepdims=True), 1)
