"""Pure market track: bookmaker prices turned into probabilities.

Per bookmaker and market, decimal prices ``o_i`` give implied
probabilities ``π_i = 1 / o_i`` whose sum ``B`` exceeds 1 (the overround).
The overround is removed with Shin's method (Shin 1993; Štrumbelj 2014):

    p_i = (sqrt(z² + 4 (1 − z) π_i² / B) − z) / (2 (1 − z))

with ``z`` chosen so the ``p_i`` sum to 1. Shin assigns more of the margin
to longshots than plain normalisation does, which matches the
favourite–longshot bias seen in football 1X2 prices. When no ``z`` in
[0, 1) works (for example a book with no margin), plain normalisation
``π_i / B`` is used. Bookmakers are then averaged with equal weight.

Markets, best source first:

* full-time 1X2: market 1 (Fulltime Result). Required; without it the
  fixture has no market forecast.
* half-time 1X2: market 31 (Half Time Result), else the half-time margin of
  market 29 (Half Time/Full Time), else derived from the full-time odds.
* exact scores: market 57 (Correct Score), else derived from the full-time
  odds.

"Derived from the full-time odds" means: find the Poisson rates (λ, ν)
whose Dixon–Coles score grid reproduces the de-margined home and away win
probabilities, then read the grid. Half-time uses the same rates multiplied
by the league's share of goals scored before the break. Every forecast
records which source it used.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import brentq, least_squares

from laliga.config import (
    DEFAULT_HT_GOAL_RATIO,
    MARKET_CORRECT_SCORE,
    MARKET_FULLTIME_RESULT,
    MARKET_HALF_TIME_FULL_TIME,
    MARKET_HALF_TIME_RESULT,
)
from laliga.model.dixon_coles import outcome_probabilities, score_matrix, top_scorelines

SOURCE_FT_ODDS = "odds_fulltime_result"
SOURCE_HT_ODDS = "odds_half_time_result"
SOURCE_HTFT_ODDS = "odds_half_time_full_time"
SOURCE_CS_ODDS = "odds_correct_score"
SOURCE_DERIVED = "derived_from_fulltime_odds"

SOURCE_LABELS = {
    SOURCE_FT_ODDS: "全场胜平负赔率",
    SOURCE_HT_ODDS: "半场胜平负赔率",
    SOURCE_HTFT_ODDS: "半全场赔率的半场边际",
    SOURCE_CS_ODDS: "正确比分赔率",
    SOURCE_DERIVED: "由全场胜平负赔率反推",
}

_HOME_TOKENS = {"1", "home", "h", "主", "主胜"}
_DRAW_TOKENS = {"x", "draw", "d", "tie", "平", "平局"}
_AWAY_TOKENS = {"2", "away", "a", "客", "客胜"}
_SCORE = re.compile(r"(\d{1,2})\s*[-:]\s*(\d{1,2})")
_OTHER = re.compile(r"other|any\s*other|others|其他|其它", re.IGNORECASE)


@dataclass
class MarketForecast:
    ft: tuple[float, float, float]
    ht: tuple[float, float, float]
    matrix: np.ndarray
    top_scores: list[tuple[int, int, float]]
    expected_goals_ft: tuple[float, float]
    sources: dict[str, str]
    bookmakers: dict[str, int]
    overround_ft: float
    method: str = "shin"
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "ft": _outcome_dict(self.ft),
            "ht": _outcome_dict(self.ht),
            "top_scores": _scores_dict(self.top_scores),
            "expected_goals_ft": {"home": self.expected_goals_ft[0], "away": self.expected_goals_ft[1]},
            "sources": dict(self.sources),
            "bookmakers": dict(self.bookmakers),
            "overround_ft": self.overround_ft,
            "method": self.method,
            "notes": list(self.notes),
        }


# --- removing the overround -------------------------------------------------


def implied_probabilities(prices) -> np.ndarray:
    values = np.asarray(prices, dtype=float)
    if values.ndim != 1 or values.size < 2 or np.any(~np.isfinite(values)) or np.any(values <= 1.0):
        raise ValueError("赔率必须是大于 1 的小数赔率，且至少两项。")
    return 1.0 / values


def devig_proportional(prices) -> np.ndarray:
    implied = implied_probabilities(prices)
    return implied / implied.sum()


def devig_shin(prices) -> tuple[np.ndarray, float]:
    """Shin probabilities and the fitted insider share ``z`` (0 when it fell back)."""

    implied = implied_probabilities(prices)
    booksum = float(implied.sum())
    if booksum <= 1.0 + 1e-12:
        return implied / booksum, 0.0

    def probabilities(z: float) -> np.ndarray:
        return (np.sqrt(z * z + 4.0 * (1.0 - z) * implied**2 / booksum) - z) / (2.0 * (1.0 - z))

    def gap(z: float) -> float:
        return float(probabilities(z).sum() - 1.0)

    upper = 0.999
    if gap(0.0) <= 0 or gap(upper) >= 0:
        return implied / booksum, 0.0
    z = float(brentq(gap, 0.0, upper, xtol=1e-12))
    result = probabilities(z)
    return result / result.sum(), z


def devig(prices, method: str = "shin") -> np.ndarray:
    if method == "proportional":
        return devig_proportional(prices)
    if method == "shin":
        return devig_shin(prices)[0]
    raise ValueError(f"未知的去水方法：{method}")


# --- reading selections -----------------------------------------------------


def classify_outcome(text: str, home_name: str = "", away_name: str = "") -> int | None:
    """Map a 1X2 selection label to 0 (home), 1 (draw), 2 (away)."""

    token = _norm(text)
    if not token:
        return None
    if token in _HOME_TOKENS:
        return 0
    if token in _DRAW_TOKENS:
        return 1
    if token in _AWAY_TOKENS:
        return 2
    if home_name and token == _norm(home_name):
        return 0
    if away_name and token == _norm(away_name):
        return 2
    return None


def outcome_of(label: str, name: str, home_name: str = "", away_name: str = "") -> int | None:
    found = classify_outcome(label, home_name, away_name)
    if found is None:
        found = classify_outcome(name, home_name, away_name)
    return found


def classify_htft(label: str, name: str, home_name: str = "", away_name: str = "") -> tuple[int, int] | None:
    """Half-time/full-time selection such as ``1/X`` or ``Home/Draw``."""

    for text in (label, name):
        parts = [part.strip() for part in str(text or "").split("/")]
        if len(parts) != 2:
            continue
        first = classify_outcome(parts[0], home_name, away_name)
        second = classify_outcome(parts[1], home_name, away_name)
        if first is not None and second is not None:
            return first, second
    return None


def parse_correct_score(label: str, name: str, home_name: str = "", away_name: str = "") -> tuple[int, int] | str | None:
    """Return ``(home goals, away goals)``, ``"other"``, or None when unreadable.

    The docs do not fix the label format. Both ``2:1`` / ``2-1`` read as
    home–away, and the winner-first style (``name`` = away side,
    ``label`` = ``2-1``) is turned into ``1-2``.
    """

    for score_text, side_text in ((label, name), (name, label)):
        found = _SCORE.search(str(score_text or ""))
        if not found:
            continue
        first, second = int(found.group(1)), int(found.group(2))
        side = classify_outcome(side_text, home_name, away_name) if side_text != score_text else None
        if side == 2 and first > second:
            return second, first
        if side == 0 and first < second:
            return second, first
        return first, second
    if _OTHER.search(str(label or "")) or _OTHER.search(str(name or "")):
        return "other"
    return None


# --- assembling a forecast --------------------------------------------------


def implied_rates(ft: tuple[float, float, float], rho: float, max_goals: int) -> tuple[float, float]:
    """Poisson rates whose Dixon–Coles grid reproduces home and away win probabilities."""

    target = np.array([ft[0], ft[2]], dtype=float)

    def residual(x: np.ndarray) -> np.ndarray:
        matrix = score_matrix(math.exp(x[0]), math.exp(x[1]), rho, max_goals)
        home, _draw, away = outcome_probabilities(matrix)
        return np.array([home, away]) - target

    start = np.log([1.45, 1.10])
    bounds = (np.log([0.05, 0.05]), np.log([6.0, 6.0]))
    result = least_squares(residual, start, bounds=bounds, xtol=1e-10, ftol=1e-10)
    return float(math.exp(result.x[0])), float(math.exp(result.x[1]))


def market_forecast(
    odds: pd.DataFrame,
    home_name: str = "",
    away_name: str = "",
    *,
    ht_ratio: float = DEFAULT_HT_GOAL_RATIO,
    rho: float = -0.05,
    max_goals: int = 10,
    method: str = "shin",
) -> MarketForecast | None:
    """Market forecast for one fixture, or None when no complete 1X2 book exists."""

    if odds is None or odds.empty:
        return None
    ft_books = _three_way_books(odds[odds["market_id"] == MARKET_FULLTIME_RESULT], home_name, away_name)
    if not ft_books:
        return None
    ft_probs = [devig(prices, method) for prices in ft_books.values()]
    ft = _mean_outcome(ft_probs)
    overround = float(np.mean([implied_probabilities(prices).sum() - 1.0 for prices in ft_books.values()]))
    lam_home, lam_away = implied_rates(ft, rho, max_goals)
    derived = score_matrix(lam_home, lam_away, rho, max_goals)
    sources = {"ft": SOURCE_FT_ODDS}
    bookmakers = {"ft": len(ft_books)}
    notes: list[str] = []

    ht_books = _three_way_books(odds[odds["market_id"] == MARKET_HALF_TIME_RESULT], home_name, away_name)
    if ht_books:
        ht = _mean_outcome([devig(prices, method) for prices in ht_books.values()])
        sources["ht"] = SOURCE_HT_ODDS
        bookmakers["ht"] = len(ht_books)
    else:
        htft = _htft_margins(odds[odds["market_id"] == MARKET_HALF_TIME_FULL_TIME], home_name, away_name, method)
        if htft:
            ht = _mean_outcome(htft)
            sources["ht"] = SOURCE_HTFT_ODDS
            bookmakers["ht"] = len(htft)
        else:
            ratio = float(np.clip(ht_ratio, 0.2, 0.8))
            ht = outcome_probabilities(score_matrix(lam_home * ratio, lam_away * ratio, rho, max_goals))
            sources["ht"] = SOURCE_DERIVED
            bookmakers["ht"] = 0
            notes.append("没有半场赔率，半场概率由全场赔率反推。")

    score_books = _correct_score_matrices(
        odds[odds["market_id"] == MARKET_CORRECT_SCORE], home_name, away_name, derived, max_goals
    )
    if score_books:
        matrix = np.mean(score_books, axis=0)
        matrix = matrix / matrix.sum()
        sources["scores"] = SOURCE_CS_ODDS
        bookmakers["scores"] = len(score_books)
    else:
        matrix = derived
        sources["scores"] = SOURCE_DERIVED
        bookmakers["scores"] = 0
        notes.append("没有正确比分赔率，比分由全场赔率反推。")

    return MarketForecast(
        ft=ft,
        ht=tuple(float(value) for value in ht),
        matrix=matrix,
        top_scores=top_scorelines(matrix, 3),
        expected_goals_ft=(lam_home, lam_away),
        sources=sources,
        bookmakers=bookmakers,
        overround_ft=overround,
        method=method,
        notes=notes,
    )


class MarketBook:
    """Odds grouped by fixture so repeated lookups stay cheap."""

    def __init__(self, odds: pd.DataFrame | None) -> None:
        self._by_fixture: dict[int, pd.DataFrame] = {}
        if odds is not None and not odds.empty:
            for fixture_id, part in odds.groupby("fixture_id", sort=False):
                self._by_fixture[int(fixture_id)] = part

    def __len__(self) -> int:
        return len(self._by_fixture)

    def has(self, fixture_id: int) -> bool:
        return int(fixture_id) in self._by_fixture

    def rows(self, fixture_id: int) -> pd.DataFrame | None:
        return self._by_fixture.get(int(fixture_id))

    def forecast(self, row: pd.Series, *, ht_ratio: float, rho: float = -0.05, max_goals: int = 10) -> MarketForecast | None:
        odds = self.rows(int(row["fixture_id"]))
        if odds is None:
            return None
        return market_forecast(
            odds,
            str(row.get("home_team_name") or ""),
            str(row.get("away_team_name") or ""),
            ht_ratio=ht_ratio,
            rho=rho,
            max_goals=max_goals,
        )


def _three_way_books(rows: pd.DataFrame, home_name: str, away_name: str) -> dict[int, list[float]]:
    books: dict[int, list[float]] = {}
    for bookmaker, part in rows.groupby("bookmaker_id", sort=True):
        prices: dict[int, float] = {}
        for _, odd in _latest_first(part).iterrows():
            outcome = outcome_of(odd["label"], odd["name"], home_name, away_name)
            if outcome is not None and outcome not in prices:
                prices[outcome] = float(odd["value"])
        if len(prices) == 3:
            books[int(bookmaker)] = [prices[0], prices[1], prices[2]]
    return books


def _htft_margins(rows: pd.DataFrame, home_name: str, away_name: str, method: str) -> list[np.ndarray]:
    margins = []
    for _bookmaker, part in rows.groupby("bookmaker_id", sort=True):
        prices: dict[tuple[int, int], float] = {}
        for _, odd in _latest_first(part).iterrows():
            pair = classify_htft(odd["label"], odd["name"], home_name, away_name)
            if pair is not None and pair not in prices:
                prices[pair] = float(odd["value"])
        if len(prices) != 9:
            continue
        keys = sorted(prices)
        probabilities = devig([prices[key] for key in keys], method)
        ht = np.zeros(3)
        for (half, _full), probability in zip(keys, probabilities, strict=True):
            ht[half] += probability
        margins.append(ht)
    return margins


def _correct_score_matrices(
    rows: pd.DataFrame, home_name: str, away_name: str, derived: np.ndarray, max_goals: int
) -> list[np.ndarray]:
    """One full score grid per bookmaker.

    Listed scores get their normalised prices. The remaining mass ("any
    other score" when offered, otherwise what the derived grid puts outside
    the listed cells) is spread over unlisted cells in proportion to the
    derived grid.
    """

    matrices = []
    for _bookmaker, part in rows.groupby("bookmaker_id", sort=True):
        listed: dict[tuple[int, int], float] = {}
        other_price: float | None = None
        for _, odd in _latest_first(part).iterrows():
            parsed = parse_correct_score(odd["label"], odd["name"], home_name, away_name)
            if parsed == "other":
                other_price = other_price or float(odd["value"])
            elif isinstance(parsed, tuple) and parsed[0] <= max_goals and parsed[1] <= max_goals:
                listed.setdefault(parsed, float(odd["value"]))
        if len(listed) < 3:
            continue
        cells = list(listed)
        prices = [listed[cell] for cell in cells] + ([other_price] if other_price else [])
        normalised = devig_proportional(prices)
        mask = np.zeros_like(derived, dtype=bool)
        for cell in cells:
            mask[cell] = True
        if other_price:
            other_mass = float(normalised[-1])
        else:
            other_mass = float(derived[~mask].sum())
        listed_mass = 1.0 - other_mass
        matrix = np.zeros_like(derived)
        for cell, probability in zip(cells, normalised[: len(cells)], strict=True):
            matrix[cell] = probability
        listed_total = matrix[mask].sum()
        if listed_total <= 0:
            continue
        matrix[mask] *= listed_mass / listed_total
        rest = derived[~mask]
        if rest.sum() > 0:
            matrix[~mask] = rest / rest.sum() * other_mass
        matrices.append(matrix / matrix.sum())
    return matrices


def _latest_first(part: pd.DataFrame) -> pd.DataFrame:
    if "latest_bookmaker_update" not in part.columns:
        return part
    return part.sort_values("latest_bookmaker_update", ascending=False, kind="stable")


def _mean_outcome(probabilities: list[np.ndarray]) -> tuple[float, float, float]:
    mean = np.mean(np.vstack(probabilities), axis=0)
    mean = mean / mean.sum()
    return float(mean[0]), float(mean[1]), float(mean[2])


def _norm(text) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().casefold())


def _outcome_dict(values) -> dict[str, float]:
    return {"home": float(values[0]), "draw": float(values[1]), "away": float(values[2])}


def _scores_dict(scores) -> list[dict]:
    return [{"home_goals": home, "away_goals": away, "probability": probability} for home, away, probability in scores]
