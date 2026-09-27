"""Turn SportMonks fixture payloads into a flat match table.

Home and away come from ``participants[].meta.location``. Scores come from
the ``scores`` include, keyed by ``description``:

* ``1ST_HALF`` — score at half-time
* ``2ND_HALF`` — cumulative score after 90 minutes (full-time result)
* ``CURRENT`` — live or final score, including extra time

League matches use ``2ND_HALF`` for the 1X2 result. ``CURRENT`` is only a
fallback when the match ended in regular time (state FT) and ``2ND_HALF``
is absent.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import pandas as pd

from laliga.config import (
    FINISHED_STATE_IDS,
    REGULATION_STATE_NAMES,
    SCHEDULED_STATE_IDS,
    SCHEDULED_STATE_NAMES,
    SCORE_CURRENT,
    SCORE_FT,
    SCORE_HT,
    SKIP_STATE_NAMES,
)

logger = logging.getLogger(__name__)

MATCH_COLUMNS = [
    "fixture_id",
    "league_id",
    "season_id",
    "season_name",
    "round_id",
    "round_name",
    "starting_at",
    "state_id",
    "state_name",
    "home_team_id",
    "home_team_name",
    "away_team_id",
    "away_team_name",
    "home_goals_ht",
    "away_goals_ht",
    "home_goals_ft",
    "away_goals_ft",
    "status",
]

# Optional pre-match / historical columns. Absent on older caches; preserved when present.
# home_xg/away_xg are post-match expected goals for that fixture (type 5304). Models may
# use them only as history for later matches. Odds are pre-match decimal 1X2 prices.
OPTIONAL_FLOAT_COLUMNS = [
    "home_xg",
    "away_xg",
    "home_xga",
    "away_xga",
    "odds_home",
    "odds_draw",
    "odds_away",
    "implied_home",
    "implied_draw",
    "implied_away",
]

_XG_TYPE_ID = 5304
_FULLTIME_MARKET_ID = 1
_FULLTIME_MARKET_NAMES = {
    "fulltime result",
    "full time result",
    "match winner",
    "1x2",
}
_OUTCOME_SIDES = {
    "home": "home",
    "1": "home",
    "draw": "draw",
    "x": "draw",
    "away": "away",
    "2": "away",
}


def empty_matches() -> pd.DataFrame:
    frame = pd.DataFrame(columns=MATCH_COLUMNS)
    frame["starting_at"] = pd.to_datetime(frame["starting_at"], utc=True)
    return frame


def fixtures_to_frame(fixtures: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for fixture in fixtures:
        row = parse_fixture(fixture)
        if row is not None:
            rows.append(row)
    if not rows:
        return empty_matches()
    frame = pd.DataFrame(rows)
    frame["starting_at"] = pd.to_datetime(frame["starting_at"], utc=True, errors="coerce")
    for column in (
        "fixture_id",
        "league_id",
        "season_id",
        "round_id",
        "state_id",
        "home_team_id",
        "away_team_id",
        "home_goals_ht",
        "away_goals_ht",
        "home_goals_ft",
        "away_goals_ft",
    ):
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("Int64")
    frame = frame.dropna(subset=["fixture_id", "starting_at", "home_team_id", "away_team_id"])
    frame = frame[frame["home_team_id"] != frame["away_team_id"]]
    frame = frame.sort_values(["starting_at", "fixture_id"]).reset_index(drop=True)
    for column in OPTIONAL_FLOAT_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.NA
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame[MATCH_COLUMNS + OPTIONAL_FLOAT_COLUMNS]


def parse_fixture(fixture: dict[str, Any]) -> dict[str, Any] | None:
    """Parse one fixture. Return None when the two sides cannot be identified."""

    if not isinstance(fixture, dict) or fixture.get("placeholder"):
        return None
    participants = fixture.get("participants") or []
    if any(isinstance(item, dict) and item.get("placeholder") for item in participants):
        return None
    home, away = _sides(participants, fixture.get("scores") or [])
    if home is None or away is None:
        logger.warning("跳过比赛 %s：无法识别主客队", fixture.get("id"))
        return None
    if home.get("id") is None or away.get("id") is None:
        return None

    scores = fixture.get("scores") or []
    home_id = int(home["id"])
    away_id = int(away["id"])
    ht = _goals(scores, SCORE_HT, home_id, away_id)
    ft = _goals(scores, SCORE_FT, home_id, away_id)
    state_name = _state_name(fixture)
    state_id = fixture.get("state_id")
    if ft is None and _is_regular_full_time(state_name, state_id):
        ft = _goals(scores, SCORE_CURRENT, home_id, away_id)

    season = fixture.get("season") or {}
    round_obj = fixture.get("round") or {}
    status = _status(state_name, state_id, ft)
    home_xg, away_xg, home_xga, away_xga = team_expected_goals(fixture, home_id, away_id)
    prices = aggregate_prematch_1x2(_prematch_odd_rows(fixture), kickoff=fixture.get("starting_at"))
    return {
        "fixture_id": fixture.get("id"),
        "league_id": fixture.get("league_id"),
        "season_id": fixture.get("season_id") or season.get("id"),
        "season_name": season.get("name") or "",
        "round_id": fixture.get("round_id") or round_obj.get("id"),
        "round_name": "" if round_obj.get("name") is None else str(round_obj.get("name")),
        "starting_at": fixture.get("starting_at"),
        "state_id": state_id,
        "state_name": state_name or "",
        "home_team_id": home_id,
        "home_team_name": home.get("name") or "",
        "away_team_id": away_id,
        "away_team_name": away.get("name") or "",
        "home_goals_ht": None if ht is None else ht[0],
        "away_goals_ht": None if ht is None else ht[1],
        "home_goals_ft": None if ft is None else ft[0],
        "away_goals_ft": None if ft is None else ft[1],
        "status": status,
        "home_xg": home_xg,
        "away_xg": away_xg,
        "home_xga": home_xga,
        "away_xga": away_xga,
        "odds_home": prices["odds_home"],
        "odds_draw": prices["odds_draw"],
        "odds_away": prices["odds_away"],
        "implied_home": prices["implied_home"],
        "implied_draw": prices["implied_draw"],
        "implied_away": prices["implied_away"],
    }


def _sides(
    participants: list[dict[str, Any]], scores: list[dict[str, Any]]
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    located: dict[str, dict[str, Any]] = {}
    for participant in participants:
        if not isinstance(participant, dict):
            continue
        location = (participant.get("meta") or {}).get("location")
        if location in ("home", "away"):
            located[location] = participant
    if "home" in located and "away" in located:
        return located["home"], located["away"]

    by_id = {participant.get("id"): participant for participant in participants if isinstance(participant, dict)}
    for row in scores:
        if not isinstance(row, dict):
            continue
        side = (row.get("score") or {}).get("participant")
        participant = by_id.get(row.get("participant_id"))
        if side in ("home", "away") and participant is not None:
            located[side] = participant
    return located.get("home"), located.get("away")


def _goals(
    scores: list[dict[str, Any]], description: str, home_id: int, away_id: int
) -> tuple[int, int] | None:
    home_goals: int | None = None
    away_goals: int | None = None
    for row in scores:
        if not isinstance(row, dict) or row.get("description") != description:
            continue
        score = row.get("score") or {}
        goals = score.get("goals")
        if goals is None:
            continue
        try:
            value = int(goals)
        except (TypeError, ValueError):
            continue
        participant_id = row.get("participant_id")
        side = score.get("participant")
        if participant_id == home_id or side == "home":
            home_goals = value
        elif participant_id == away_id or side == "away":
            away_goals = value
    if home_goals is None or away_goals is None:
        return None
    return home_goals, away_goals


def _state_name(fixture: dict[str, Any]) -> str | None:
    state = fixture.get("state")
    if isinstance(state, dict):
        for key in ("developer_name", "short_name", "state"):
            value = state.get(key)
            if value:
                return str(value).upper()
    return None


def _is_regular_full_time(state_name: str | None, state_id: Any) -> bool:
    if state_name == "FT":
        return True
    try:
        return int(state_id) == 5
    except (TypeError, ValueError):
        return False


def _status(state_name: str | None, state_id: Any, ft: tuple[int, int] | None) -> str:
    if state_name in SKIP_STATE_NAMES:
        return "skipped"
    try:
        numeric_state = int(state_id) if state_id is not None else None
    except (TypeError, ValueError):
        numeric_state = None

    finished_name = state_name in REGULATION_STATE_NAMES
    finished_id = numeric_state in FINISHED_STATE_IDS
    if ft is not None and (finished_name or finished_id or state_name is None):
        return "finished"
    if state_name in SCHEDULED_STATE_NAMES or numeric_state in SCHEDULED_STATE_IDS:
        return "scheduled"
    if ft is not None:
        return "finished"
    return "skipped"


def team_expected_goals(
    fixture: dict[str, Any], home_id: int, away_id: int
) -> tuple[float | None, float | None, float | None, float | None]:
    """Per-team xG (type 5304) and xGA.

    xGA is the opponent's xG from the same match. Both numbers are post-match.
    ``xGFixture`` may arrive as ``xgfixture`` or, in the expected-include
    examples, as ``expected``. Expected goals on target (type 5305) are ignored.
    """

    home_xg, away_xg = _team_xg(fixture, home_id, away_id)
    return home_xg, away_xg, away_xg, home_xg


def aggregate_prematch_1x2(rows: list[dict[str, Any]], *, kickoff: Any = None) -> dict[str, float | None]:
    """Market-average pre-kickoff 1X2, with the overround removed.

    Each bookmaker contributes at most one home, draw, and away price: the
    latest quote whose ``latest_bookmaker_update`` (else ``updated_at``, else
    ``created_at``) is strictly before kickoff. A quote with no timestamp is
    kept, because the standard pre-match feed does not send in-play rows.
    Bookmakers missing any side are dropped. The three outcomes are the mean
    of ``1/decimal`` across the remaining books, divided by their sum.
    Stored decimals are the reciprocal of those probabilities, so they match
    the de-vigged prices rather than any single raw book.
    """

    empty = {
        "odds_home": None,
        "odds_draw": None,
        "odds_away": None,
        "implied_home": None,
        "implied_draw": None,
        "implied_away": None,
    }
    kickoff_at = _as_utc(kickoff)
    chosen: dict[tuple[int, str], tuple[pd.Timestamp | None, float]] = {}
    for row in rows:
        if not isinstance(row, dict) or not _is_fulltime_result_market(row):
            continue
        if row.get("inplay") is True or row.get("is_live") is True:
            continue
        if not _is_before_kickoff(row, kickoff_at):
            continue
        side = _outcome_side(row.get("label")) or _outcome_side(row.get("name")) or _outcome_side(row.get("original_label"))
        price = _decimal_price(row.get("value"))
        book_id = _as_int(row.get("bookmaker_id"))
        if side is None or price is None or book_id is None:
            continue
        stamp = _price_time(row)
        current = chosen.get((book_id, side))
        if current is None or _is_newer_quote(stamp, current[0]):
            chosen[(book_id, side)] = (stamp, price)
    by_book: dict[int, dict[str, float]] = {}
    for (book_id, side), (_, price) in chosen.items():
        by_book.setdefault(book_id, {})[side] = price
    complete = [prices for prices in by_book.values() if {"home", "draw", "away"} <= set(prices)]
    if not complete:
        return empty
    raw = []
    for side in ("home", "draw", "away"):
        raw.append(sum(1.0 / prices[side] for prices in complete) / len(complete))
    total = sum(raw)
    if total <= 0.0 or not math.isfinite(total):
        return empty
    implied = [value / total for value in raw]
    odds = [1.0 / value for value in implied]
    return {
        "odds_home": odds[0],
        "odds_draw": odds[1],
        "odds_away": odds[2],
        "implied_home": implied[0],
        "implied_draw": implied[1],
        "implied_away": implied[2],
    }


def _team_xg(fixture: dict[str, Any], home_id: int, away_id: int) -> tuple[float | None, float | None]:
    """Full-time expected goals (type 5304) for the two sides. Other xG types are ignored."""

    chunk = None
    for key in ("xgfixture", "xGFixture", "expected"):
        candidate = fixture.get(key)
        if isinstance(candidate, list):
            chunk = candidate
            break
    if not isinstance(chunk, list):
        return None, None
    home: float | None = None
    away: float | None = None
    for row in chunk:
        if not isinstance(row, dict) or not _is_expected_goals(row):
            continue
        value = _xg_value(row)
        if value is None:
            continue
        location = str(row.get("location") or "").lower()
        participant = _as_int(row.get("participant_id"))
        if location == "home" or participant == home_id:
            home = value
        elif location == "away" or participant == away_id:
            away = value
    return home, away


def _is_expected_goals(row: dict[str, Any]) -> bool:
    if _as_int(row.get("type_id")) == _XG_TYPE_ID:
        return True
    type_obj = row.get("type") if isinstance(row.get("type"), dict) else {}
    code = str(type_obj.get("code") or row.get("code") or "").lower()
    developer = str(type_obj.get("developer_name") or "").upper()
    return code == "expected-goals" or developer == "EXPECTED_GOALS"


def _xg_value(row: dict[str, Any]) -> float | None:
    data = row.get("data") if isinstance(row.get("data"), dict) else {}
    raw = data.get("value", row.get("value"))
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    return value


def _prematch_odd_rows(fixture: dict[str, Any]) -> list[dict[str, Any]]:
    """Pre-match quotes embedded on a fixture. ``inplayOdds`` is never read."""

    rows: list[dict[str, Any]] = []
    for key in ("odds", "premiumOdds", "premiumodds"):
        chunk = fixture.get(key)
        if isinstance(chunk, list):
            rows.extend(item for item in chunk if isinstance(item, dict))
    return rows


def _as_utc(value: Any) -> pd.Timestamp | None:
    if value is None or value == "":
        return None
    stamp = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(stamp):
        return None
    return pd.Timestamp(stamp)


def _price_time(row: dict[str, Any]) -> pd.Timestamp | None:
    for key in ("latest_bookmaker_update", "updated_at", "created_at"):
        stamp = _as_utc(row.get(key))
        if stamp is not None:
            return stamp
    return None


def _is_before_kickoff(row: dict[str, Any], kickoff: pd.Timestamp | None) -> bool:
    if kickoff is None:
        return True
    stamp = _price_time(row)
    if stamp is None:
        return True
    return stamp < kickoff


def _is_newer_quote(stamp: pd.Timestamp | None, previous: pd.Timestamp | None) -> bool:
    if stamp is None:
        return previous is None
    if previous is None:
        return True
    return stamp >= previous


def _is_fulltime_result_market(row: dict[str, Any]) -> bool:
    if _as_int(row.get("market_id")) == _FULLTIME_MARKET_ID:
        return True
    market = row.get("market") if isinstance(row.get("market"), dict) else {}
    texts = (
        row.get("market_description"),
        row.get("market_name"),
        market.get("name"),
        market.get("developer_name"),
    )
    for text in texts:
        if not text:
            continue
        normalized = str(text).strip().lower().replace("_", " ")
        if normalized in _FULLTIME_MARKET_NAMES:
            return True
    return False


def _outcome_side(value: Any) -> str | None:
    if value is None:
        return None
    return _OUTCOME_SIDES.get(str(value).strip().lower())


def _decimal_price(value: Any) -> float | None:
    try:
        price = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(price) or price <= 1.0:
        return None
    return price


def _as_int(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except (TypeError, ValueError):
        return None
