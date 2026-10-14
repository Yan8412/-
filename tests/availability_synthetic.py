"""Synthetic schedule with squads, starting XIs and sidelined records, for tests only.

Missing regular starters lower a team's expected goals by ``attack_effect``
each and raise the opponent's by ``defence_effect``. With both at 0,
availability carries no signal.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from laliga.synthetic import single_round_robin

TEAMS = list(range(1, 11))
REGULARS = 11
SQUAD = 18


def build(
    *,
    seed: int = 3,
    attack_effect: float = 0.15,
    defence_effect: float = 0.10,
    cycles: int = 3,
    today: date | None = None,
    upcoming: bool = False,
):
    rng = np.random.default_rng(seed)
    attack = {team: float(rng.normal(0, 0.25)) for team in TEAMS}
    defence = {team: float(rng.normal(0, 0.2)) for team in TEAMS}
    first = single_round_robin(TEAMS)
    rounds = []
    for _ in range(cycles):
        rounds += first + [[(away, home) for home, away in pairs] for pairs in first]
    end = today or date(2026, 5, 1)
    if upcoming:
        start = end + timedelta(days=3) - timedelta(days=7 * len(rounds))
    else:
        start = end - timedelta(days=7 * (len(rounds) - 1))
    per_season = len(first) * 2
    matches, lineups, sidelined, fetched = [], [], [], []
    fixture_id = 5000
    for round_index, pairs in enumerate(rounds + ([rounds[0]] if upcoming else [])):
        day = start + timedelta(days=7 * round_index)
        scheduled = upcoming and round_index == len(rounds)
        season_id = 100 + round_index // per_season
        for slot, (home, away) in enumerate(pairs):
            kickoff = pd.Timestamp(f"{day.isoformat()} {13 + 2 * slot}:00:00", tz="UTC")
            missing = {}
            for team in (home, away):
                count = int(rng.choice([0, 1, 2, 3], p=[0.4, 0.3, 0.2, 0.1]))
                out = sorted(rng.choice(REGULARS, size=count, replace=False).tolist())
                missing[team] = out
                for player in out:
                    sidelined.append(
                        {
                            "fixture_id": fixture_id,
                            "team_id": team,
                            "player_id": team * 100 + player,
                            "sideline_id": fixture_id * 100 + team * 10 + player,
                            "type_id": 535,
                            "category": "injury" if player % 2 else "suspension",
                            "start_date": (day - timedelta(days=3)).isoformat(),
                            "end_date": (day + timedelta(days=4)).isoformat(),
                            "games_missed": 1,
                            "completed": True,
                        }
                    )
                if not scheduled:
                    available = [player for player in range(REGULARS) if player not in out]
                    bench = list(range(REGULARS, SQUAD))
                    xi = available + bench[: REGULARS - len(available)]
                    for player in range(SQUAD):
                        lineups.append(
                            {
                                "fixture_id": fixture_id,
                                "team_id": team,
                                "player_id": team * 100 + player,
                                "player_name": f"P{team}-{player}",
                                "type_id": 11 if player in xi else 12,
                                "position_id": 24 if player == 0 else 25 + (player % 3),
                                "formation_position": None,
                            }
                        )
            m_home, m_away = len(missing[home]), len(missing[away])
            lam_home = np.exp(0.15 + 0.25 + attack[home] - defence[away] - attack_effect * m_home + defence_effect * m_away)
            lam_away = np.exp(0.15 + attack[away] - defence[home] - attack_effect * m_away + defence_effect * m_home)
            row = {
                "fixture_id": fixture_id,
                "league_id": 564,
                "season_id": season_id,
                "season_name": f"S{season_id}",
                "round_id": round_index + 1,
                "round_name": str(round_index + 1),
                "starting_at": kickoff,
                "state_id": 1 if scheduled else 5,
                "state_name": "NS" if scheduled else "FT",
                "home_team_id": home,
                "home_team_name": f"T{home}",
                "away_team_id": away,
                "away_team_name": f"T{away}",
                "status": "scheduled" if scheduled else "finished",
                "home_goals_ft": None,
                "away_goals_ft": None,
                "home_goals_ht": None,
                "away_goals_ht": None,
            }
            if not scheduled:
                goals_home = int(rng.poisson(lam_home))
                goals_away = int(rng.poisson(lam_away))
                row.update(
                    {
                        "home_goals_ft": goals_home,
                        "away_goals_ft": goals_away,
                        "home_goals_ht": int(rng.binomial(goals_home, 0.45)),
                        "away_goals_ht": int(rng.binomial(goals_away, 0.45)),
                    }
                )
            matches.append(row)
            fetched.append(
                {
                    "fixture_id": fixture_id,
                    "fetched_at": "2026-05-02T00:00:00+00:00",
                    "kickoff_utc": kickoff.isoformat(),
                    "status_at_fetch": row["status"],
                    "lineup_confirmed": None if scheduled else True,
                    "n_lineup": 0 if scheduled else 2 * SQUAD,
                    "n_starters": 0 if scheduled else 22,
                    "n_sidelined": m_home + m_away,
                }
            )
            fixture_id += 1
    frame = pd.DataFrame(matches)
    for column in ("home_goals_ft", "away_goals_ft", "home_goals_ht", "away_goals_ht"):
        frame[column] = frame[column].astype("Int64")
    return frame, pd.DataFrame(lineups), pd.DataFrame(sidelined), pd.DataFrame(fetched)
