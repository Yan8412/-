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
    return frame[MATCH_COLUMNS]


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
