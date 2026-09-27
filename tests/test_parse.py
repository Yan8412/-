import pandas as pd

from laliga.data.parse import fixtures_to_frame, parse_fixture


def _participant(team_id, name, location):
    return {"id": team_id, "name": name, "meta": {"location": location, "winner": False, "position": 1}}


def _score(description, participant_id, side, goals, type_id):
    return {
        "description": description,
        "type_id": type_id,
        "participant_id": participant_id,
        "score": {"goals": goals, "participant": side},
    }


def test_documented_napoli_score_split():
    # Shape follows the SportMonks scores tutorial: 1ST_HALF is the break,
    # 2ND_HALF is the cumulative 90-minute score, 2ND_HALF_ONLY is the second half.
    fixture = {
        "id": 19424997,
        "league_id": 564,
        "season_id": 25533,
        "state_id": 5,
        "starting_at": "2025-11-22 19:45:00",
        "state": {"id": 5, "developer_name": "FT", "short_name": "FT"},
        "season": {"id": 25533, "name": "2025/2026"},
        "round": {"id": 371860, "name": "13"},
        "participants": [
            _participant(708, "Atalanta", "away"),
            _participant(597, "Napoli", "home"),
        ],
        "scores": [
            _score("2ND_HALF_ONLY", 708, "away", 1, 48996),
            _score("2ND_HALF_ONLY", 597, "home", 0, 48996),
            _score("CURRENT", 597, "home", 3, 1525),
            _score("CURRENT", 708, "away", 1, 1525),
            _score("1ST_HALF", 597, "home", 3, 1),
            _score("2ND_HALF", 708, "away", 1, 2),
            _score("2ND_HALF", 597, "home", 3, 2),
            _score("1ST_HALF", 708, "away", 0, 1),
        ],
    }
    row = parse_fixture(fixture)
    assert row["home_team_name"] == "Napoli"
    assert row["away_team_name"] == "Atalanta"
    assert (row["home_goals_ht"], row["away_goals_ht"]) == (3, 0)
    assert (row["home_goals_ft"], row["away_goals_ft"]) == (3, 1)
    assert row["status"] == "finished"
    assert row["season_name"] == "2025/2026"
    assert row["round_name"] == "13"


def test_extra_time_uses_regulation_score_not_current():
    fixture = {
        "id": 10,
        "league_id": 564,
        "season_id": 1,
        "state_id": 7,
        "starting_at": "2024-01-02 20:00:00",
        "state": {"developer_name": "AET"},
        "participants": [
            _participant(1, "Home", "home"),
            _participant(2, "Away", "away"),
        ],
        "scores": [
            _score("1ST_HALF", 1, "home", 0, 1),
            _score("1ST_HALF", 2, "away", 0, 1),
            _score("2ND_HALF", 1, "home", 1, 2),
            _score("2ND_HALF", 2, "away", 1, 2),
            _score("CURRENT", 1, "home", 2, 1525),
            _score("CURRENT", 2, "away", 1, 1525),
        ],
    }
    row = parse_fixture(fixture)
    assert (row["home_goals_ft"], row["away_goals_ft"]) == (1, 1)
    assert row["status"] == "finished"


def test_full_time_without_second_half_falls_back_to_current():
    fixture = {
        "id": 11,
        "league_id": 564,
        "season_id": 1,
        "state_id": 5,
        "starting_at": "2024-01-03 20:00:00",
        "state": {"developer_name": "FT"},
        "participants": [
            _participant(1, "Home", "home"),
            _participant(2, "Away", "away"),
        ],
        "scores": [
            _score("CURRENT", 1, "home", 2, 1525),
            _score("CURRENT", 2, "away", 0, 1525),
        ],
    }
    row = parse_fixture(fixture)
    assert (row["home_goals_ft"], row["away_goals_ft"]) == (2, 0)
    assert row["home_goals_ht"] is None


def test_postponed_and_placeholder_are_not_training_rows():
    postponed = {
        "id": 12,
        "league_id": 564,
        "state_id": 10,
        "starting_at": "2024-01-04 20:00:00",
        "state": {"developer_name": "POSTPONED"},
        "participants": [_participant(1, "Home", "home"), _participant(2, "Away", "away")],
        "scores": [],
    }
    placeholder = {"id": 13, "placeholder": True, "participants": []}
    frame = fixtures_to_frame([postponed, placeholder])
    assert list(frame["status"]) == ["skipped"]
    assert len(frame) == 1


def test_sides_can_be_inferred_from_score_participant_ids():
    fixture = {
        "id": 14,
        "league_id": 564,
        "state_id": 1,
        "starting_at": "2024-05-01 18:00:00",
        "state": {"developer_name": "NS"},
        "participants": [
            {"id": 5, "name": "Host"},
            {"id": 6, "name": "Visitor"},
        ],
        "scores": [
            _score("CURRENT", 5, "home", 0, 1525),
            _score("CURRENT", 6, "away", 0, 1525),
        ],
    }
    row = parse_fixture(fixture)
    assert row["home_team_id"] == 5
    assert row["away_team_name"] == "Visitor"
    assert row["status"] == "scheduled"


def test_frame_timestamps_are_utc():
    fixture = {
        "id": 15,
        "league_id": 564,
        "state_id": 5,
        "starting_at": "2024-02-02 16:15:00",
        "state": {"developer_name": "FT"},
        "participants": [_participant(1, "Home", "home"), _participant(2, "Away", "away")],
        "scores": [
            _score("2ND_HALF", 1, "home", 1, 2),
            _score("2ND_HALF", 2, "away", 1, 2),
        ],
    }
    frame = fixtures_to_frame([fixture])
    assert frame.loc[0, "starting_at"] == pd.Timestamp("2024-02-02 16:15:00", tz="UTC")
