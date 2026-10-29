"""Synthetic La Liga-shaped seasons for offline tests and ``laliga demo``.

Matches are SportMonks-shaped payloads: participants carry home/away meta,
and scores use 1ST_HALF, 2ND_HALF, 2ND_HALF_ONLY, and CURRENT. Goals are
drawn from the same attack/defence process the model is built to recover.
One promoted team replaces a relegated team in the second season. The last
round is left unplayed so ``predict --next`` has a target.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Any

import numpy as np

from laliga.config import LA_LIGA_LEAGUE_ID

TEAM_NAMES = {
    1: "北岸联",
    2: "河谷竞技",
    3: "港城",
    4: "高原联合",
    5: "海岸体育",
    6: "旧城",
    7: "山城升班马",
}

ATTACK = {1: 0.45, 2: 0.20, 3: 0.05, 4: -0.05, 5: -0.20, 6: -0.45, 7: -0.28}
DEFENCE = {1: 0.30, 2: 0.12, 3: 0.02, 4: -0.04, 5: -0.12, 6: -0.28, 7: -0.18}
MU = 0.12
HOME = 0.30
HT_SHARE = 0.45


def generate_synthetic_fixtures(seed: int = 7, today: date | None = None) -> list[dict[str, Any]]:
    """Build two seasons of raw fixtures relative to ``today`` (UTC date)."""

    today = today or datetime.now(timezone.utc).date()
    rng = np.random.default_rng(seed)
    next_round_day = today + timedelta(days=3)
    # Eight finished rounds in the current season, weekly, ending a week ago.
    season2_round_days = [next_round_day - timedelta(days=7 * (8 - index)) for index in range(8)]
    season2_start = season2_round_days[0]
    season1_rounds = double_round_robin([1, 2, 3, 4, 5, 6])
    season1_rounds += double_round_robin([1, 2, 3, 4, 5, 6])
    season1_end = season2_start - timedelta(days=21)
    season1_days = [
        season1_end - timedelta(days=7 * (len(season1_rounds) - 1 - index))
        for index in range(len(season1_rounds))
    ]

    season2_teams = [1, 2, 3, 4, 5, 7]
    season2_rounds = double_round_robin(season2_teams)
    # Play the first 8 rounds; hold the 9th as the upcoming matchday.
    played = season2_rounds[:8]
    upcoming = season2_rounds[8]

    fixtures: list[dict[str, Any]] = []
    next_id = 1000
    next_id = _append_season(
        fixtures,
        rng,
        next_id,
        season_id=9001,
        season_name=f"{season2_start.year - 1}/{season2_start.year}",
        rounds=season1_rounds,
        days=season1_days,
        scheduled=None,
    )
    next_id = _append_season(
        fixtures,
        rng,
        next_id,
        season_id=9002,
        season_name=f"{season2_start.year}/{season2_start.year + 1}",
        rounds=played,
        days=season2_round_days,
        scheduled=(upcoming, next_round_day),
    )
    return fixtures


def generate_synthetic_odds(fixtures: list[dict[str, Any]], seed: int = 11) -> list[dict[str, Any]]:
    """SportMonks-shaped pre-match odd rows for synthetic fixtures (demo and tests only).

    Prices come from the same attack/defence process as the goals, with a
    shared "market view" error, a small per-bookmaker error, and a margin.
    Bookmaker 2 uses ``1/X/2`` labels, a Half Time Result market and
    home–away correct scores with an "Any Other Score" line. Bookmaker 34
    uses ``Home/Draw/Away``, a Half Time/Full Time market instead of Half
    Time Result, and winner-first correct scores (``name`` = side).
    ``winning`` is filled for finished matches the way the live API does;
    the program must never read it.
    """

    from laliga.model.dixon_coles import outcome_probabilities, score_matrix

    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    next_id = 1
    for fixture in fixtures:
        home_id, away_id = _side_ids(fixture)
        lam_home = float(np.exp(MU + HOME + ATTACK[home_id] - DEFENCE[away_id]))
        lam_away = float(np.exp(MU + ATTACK[away_id] - DEFENCE[home_id]))
        shift = rng.normal(0.0, 0.06, size=2)
        view_home = lam_home * float(np.exp(shift[0]))
        view_away = lam_away * float(np.exp(shift[1]))
        result = _synthetic_result(fixture)
        kickoff = str(fixture["starting_at"])
        for bookmaker, margin in ((2, 0.06), (34, 0.045)):
            own = rng.normal(0.0, 0.02, size=2)
            book_home = view_home * float(np.exp(own[0]))
            book_away = view_away * float(np.exp(own[1]))
            ft_grid = score_matrix(book_home, book_away, 0.0, 10)
            ht_grid = score_matrix(book_home * HT_SHARE, book_away * HT_SHARE, 0.0, 10)
            ft = outcome_probabilities(ft_grid)
            ht = outcome_probabilities(ht_grid)
            labels = ("1", "X", "2") if bookmaker == 2 else ("Home", "Draw", "Away")
            for index, probability in enumerate(ft):
                rows.append(_odd(next_id, fixture, 1, bookmaker, labels[index], labels[index], probability, margin, kickoff, result and result["ft"] == index))
                next_id += 1
            if bookmaker == 2:
                for index, probability in enumerate(ht):
                    rows.append(_odd(next_id, fixture, 31, bookmaker, labels[index], labels[index], probability, margin, kickoff, result and result["ht"] == index))
                    next_id += 1
            else:
                for half in range(3):
                    for full in range(3):
                        probability = _htft_probability(ht_grid, book_home, book_away, half, full)
                        label = f"{labels[half]}/{labels[full]}"
                        hit = bool(result) and result["ht"] == half and result["ft"] == full
                        rows.append(_odd(next_id, fixture, 29, bookmaker, label, label, probability, margin * 2, kickoff, hit))
                        next_id += 1
            listed = 0.0
            for home_goals in range(5):
                for away_goals in range(5):
                    probability = float(ft_grid[home_goals, away_goals])
                    listed += probability
                    hit = bool(result) and result["score"] == (home_goals, away_goals)
                    if bookmaker == 2:
                        label = name = f"{home_goals}:{away_goals}"
                    else:
                        side = "Home" if home_goals > away_goals else "Away" if home_goals < away_goals else "Draw"
                        high, low = max(home_goals, away_goals), min(home_goals, away_goals)
                        label, name = f"{high}-{low}", side
                    rows.append(_odd(next_id, fixture, 57, bookmaker, label, name, probability, margin * 3, kickoff, hit))
                    next_id += 1
            if bookmaker == 2:
                hit = bool(result) and max(result["score"]) > 4
                rows.append(_odd(next_id, fixture, 57, bookmaker, "Any Other Score", "Any Other Score", max(1.0 - listed, 1e-4), margin * 3, kickoff, hit))
                next_id += 1
    return rows


def _side_ids(fixture: dict[str, Any]) -> tuple[int, int]:
    located = {item["meta"]["location"]: int(item["id"]) for item in fixture["participants"]}
    return located["home"], located["away"]


def _synthetic_result(fixture: dict[str, Any]) -> dict[str, Any] | None:
    goals: dict[str, dict[str, int]] = {}
    for row in fixture.get("scores") or []:
        goals.setdefault(row["description"], {})[row["score"]["participant"]] = int(row["score"]["goals"])
    ft = goals.get("2ND_HALF")
    ht = goals.get("1ST_HALF")
    if not ft or not ht:
        return None

    def outcome(pair: dict[str, int]) -> int:
        return 0 if pair["home"] > pair["away"] else 1 if pair["home"] == pair["away"] else 2

    return {"ft": outcome(ft), "ht": outcome(ht), "score": (ft["home"], ft["away"])}


def _htft_probability(ht_grid, lam_home: float, lam_away: float, half: int, full: int) -> float:
    """P(half-time outcome, full-time outcome) when second-half goals are independent Poisson."""

    from laliga.model.dixon_coles import score_matrix

    second = score_matrix(lam_home * (1 - HT_SHARE), lam_away * (1 - HT_SHARE), 0.0, 10)
    ht_diff = _goal_difference(ht_grid)
    span = ht_grid.shape[0] - 1
    differences = np.arange(-span, span + 1)
    keep = {0: differences > 0, 1: differences == 0, 2: differences < 0}[half]
    ft_diff = np.convolve(np.where(keep, ht_diff, 0.0), _goal_difference(second))
    ft_differences = np.arange(-2 * span, 2 * span + 1)
    wanted = {0: ft_differences > 0, 1: ft_differences == 0, 2: ft_differences < 0}[full]
    return max(float(ft_diff[wanted].sum()), 1e-4)


def _goal_difference(grid) -> np.ndarray:
    size = grid.shape[0]
    distribution = np.zeros(2 * size - 1)
    for offset in range(-(size - 1), size):
        distribution[offset + size - 1] = float(np.trace(grid, offset=-offset))
    return distribution


def _odd(
    odd_id: int,
    fixture: dict[str, Any],
    market_id: int,
    bookmaker_id: int,
    label: str,
    name: str,
    probability: float,
    margin: float,
    kickoff: str,
    winning: bool | None,
) -> dict[str, Any]:
    price = max(1.01, round(1.0 / (max(probability, 1e-4) * (1.0 + margin)), 2))
    descriptions = {1: "Fulltime Result", 29: "Half Time/Full Time", 31: "Half Time Result", 57: "Correct Score"}
    updated = (datetime.fromisoformat(kickoff) - timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S")
    return {
        "id": odd_id,
        "fixture_id": fixture["id"],
        "market_id": market_id,
        "bookmaker_id": bookmaker_id,
        "label": label,
        "value": f"{price:.2f}",
        "name": name,
        "sort_order": None,
        "market_description": descriptions[market_id],
        "probability": f"{100.0 / price:.2f}%",
        "dp3": f"{price:.3f}",
        "winning": bool(winning),
        "stopped": False,
        "total": None,
        "handicap": None,
        "participants": None,
        "latest_bookmaker_update": updated,
    }


def double_round_robin(team_ids: list[int]) -> list[list[tuple[int, int]]]:
    """Home-and-away round robin. Each pair meets twice, once at each ground."""

    first_half = single_round_robin(team_ids)
    return first_half + [[(away, home) for home, away in pairs] for pairs in first_half]


def single_round_robin(team_ids: list[int]) -> list[list[tuple[int, int]]]:
    """Circle-method rounds. Even team counts only; each round pairs everyone once."""

    teams: list[int | None] = list(team_ids)
    if len(teams) % 2 == 1:
        teams.append(None)
    count = len(teams)
    rounds: list[list[tuple[int, int]]] = []
    rotation = list(teams)
    for round_index in range(count - 1):
        pairs: list[tuple[int, int]] = []
        for index in range(count // 2):
            left = rotation[index]
            right = rotation[count - 1 - index]
            if left is None or right is None:
                continue
            if round_index % 2 == 0:
                pairs.append((left, right))
            else:
                pairs.append((right, left))
        rounds.append(pairs)
        rotation = [rotation[0], rotation[-1], *rotation[1:-1]]
    return rounds


def _append_season(
    fixtures: list[dict[str, Any]],
    rng: np.random.Generator,
    next_id: int,
    *,
    season_id: int,
    season_name: str,
    rounds: list[list[tuple[int, int]]],
    days: list[date],
    scheduled: tuple[list[tuple[int, int]], date] | None,
) -> int:
    kickoffs = ("13:00:00", "15:15:00", "20:00:00", "18:30:00")
    for round_number, (pairs, day) in enumerate(zip(rounds, days, strict=True), start=1):
        for index, (home_id, away_id) in enumerate(pairs):
            fixtures.append(
                _finished_fixture(
                    rng,
                    next_id,
                    season_id,
                    season_name,
                    round_number,
                    day,
                    kickoffs[index % len(kickoffs)],
                    home_id,
                    away_id,
                )
            )
            next_id += 1
    if scheduled is not None:
        pairs, day = scheduled
        round_number = len(rounds) + 1
        for index, (home_id, away_id) in enumerate(pairs):
            fixtures.append(
                _scheduled_fixture(
                    next_id,
                    season_id,
                    season_name,
                    round_number,
                    day,
                    kickoffs[index % len(kickoffs)],
                    home_id,
                    away_id,
                )
            )
            next_id += 1
    return next_id


def _finished_fixture(
    rng: np.random.Generator,
    fixture_id: int,
    season_id: int,
    season_name: str,
    round_number: int,
    day: date,
    clock: str,
    home_id: int,
    away_id: int,
) -> dict[str, Any]:
    lam_home = np.exp(MU + HOME + ATTACK[home_id] - DEFENCE[away_id])
    lam_away = np.exp(MU + ATTACK[away_id] - DEFENCE[home_id])
    home_goals = int(rng.poisson(lam_home))
    away_goals = int(rng.poisson(lam_away))
    ht_home = int(rng.binomial(home_goals, HT_SHARE))
    ht_away = int(rng.binomial(away_goals, HT_SHARE))
    return _shell(
        fixture_id,
        season_id,
        season_name,
        round_number,
        day,
        clock,
        home_id,
        away_id,
        state_id=5,
        state_name="FT",
        scores=_score_rows(fixture_id, home_id, away_id, ht_home, ht_away, home_goals, away_goals),
        result_info="finished",
    )


def _scheduled_fixture(
    fixture_id: int,
    season_id: int,
    season_name: str,
    round_number: int,
    day: date,
    clock: str,
    home_id: int,
    away_id: int,
) -> dict[str, Any]:
    return _shell(
        fixture_id,
        season_id,
        season_name,
        round_number,
        day,
        clock,
        home_id,
        away_id,
        state_id=1,
        state_name="NS",
        scores=[],
        result_info=None,
    )


def _shell(
    fixture_id: int,
    season_id: int,
    season_name: str,
    round_number: int,
    day: date,
    clock: str,
    home_id: int,
    away_id: int,
    *,
    state_id: int,
    state_name: str,
    scores: list[dict[str, Any]],
    result_info: str | None,
) -> dict[str, Any]:
    round_id = season_id * 100 + round_number
    return {
        "id": fixture_id,
        "sport_id": 1,
        "league_id": LA_LIGA_LEAGUE_ID,
        "season_id": season_id,
        "stage_id": season_id,
        "round_id": round_id,
        "state_id": state_id,
        "name": f"{TEAM_NAMES[home_id]} vs {TEAM_NAMES[away_id]}",
        "starting_at": f"{day.isoformat()} {clock}",
        "result_info": result_info,
        "length": 90,
        "placeholder": False,
        "season": {"id": season_id, "name": season_name, "league_id": LA_LIGA_LEAGUE_ID},
        "state": {
            "id": state_id,
            "state": state_name,
            "name": "Full Time" if state_name == "FT" else "Not Started",
            "short_name": state_name,
            "developer_name": state_name,
        },
        "round": {"id": round_id, "name": str(round_number), "finished": state_name == "FT"},
        "participants": [
            {
                "id": away_id,
                "name": TEAM_NAMES[away_id],
                "meta": {"location": "away", "winner": False, "position": 2},
            },
            {
                "id": home_id,
                "name": TEAM_NAMES[home_id],
                "meta": {"location": "home", "winner": False, "position": 1},
            },
        ],
        "scores": scores,
    }


def _score_rows(
    fixture_id: int,
    home_id: int,
    away_id: int,
    ht_home: int,
    ht_away: int,
    ft_home: int,
    ft_away: int,
) -> list[dict[str, Any]]:
    rows = [
        ("1ST_HALF", 1, home_id, "home", ht_home),
        ("1ST_HALF", 1, away_id, "away", ht_away),
        ("2ND_HALF_ONLY", 48996, home_id, "home", ft_home - ht_home),
        ("2ND_HALF_ONLY", 48996, away_id, "away", ft_away - ht_away),
        ("2ND_HALF", 2, home_id, "home", ft_home),
        ("2ND_HALF", 2, away_id, "away", ft_away),
        ("CURRENT", 1525, home_id, "home", ft_home),
        ("CURRENT", 1525, away_id, "away", ft_away),
    ]
    return [
        {
            "id": fixture_id * 10 + index,
            "fixture_id": fixture_id,
            "type_id": type_id,
            "participant_id": participant_id,
            "score": {"goals": goals, "participant": side},
            "description": description,
        }
        for index, (description, type_id, participant_id, side, goals) in enumerate(rows)
    ]
