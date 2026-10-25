"""Staleness guard used by the daily run before it retrains."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from laliga.data.store import MatchStore
from laliga.freshness import EXIT_OK, EXIT_STALE, NOW_ENV, check_freshness, resolve_now

REPO = Path(__file__).resolve().parents[1]
NOW = pd.Timestamp("2026-10-14T17:00:00Z")


def _row(fixture_id, kickoff, status, *, season_id=27965, state=None):
    finished = status == "finished"
    return {
        "fixture_id": fixture_id,
        "league_id": 564,
        "season_id": season_id,
        "season_name": "2026/2027",
        "round_id": 1,
        "round_name": "1",
        "starting_at": pd.Timestamp(kickoff),
        "state_id": 5 if finished else 1,
        "state_name": state or ("FT" if finished else "NS"),
        "home_team_id": 10 + fixture_id % 7,
        "home_team_name": "Home",
        "away_team_id": 30 + fixture_id % 5,
        "away_team_name": "Away",
        "home_goals_ht": 0 if finished else None,
        "away_goals_ht": 0 if finished else None,
        "home_goals_ft": 1 if finished else None,
        "away_goals_ft": 0 if finished else None,
        "status": status,
    }


def _frame(rows):
    return pd.DataFrame(rows)


def test_recent_result_and_listed_fixture_is_ok():
    frame = _frame([_row(1, NOW - pd.Timedelta(days=3), "finished"), _row(2, NOW + pd.Timedelta(days=4), "scheduled")])
    result = check_freshness(frame, NOW)
    assert result.status == "OK" and result.exit_code == EXIT_OK


def test_fixture_unplayed_more_than_48_hours_after_kickoff_is_stale():
    frame = _frame(
        [
            _row(1, NOW - pd.Timedelta(days=3), "finished"),
            _row(2, NOW - pd.Timedelta(hours=49), "scheduled"),
            _row(3, NOW + pd.Timedelta(days=4), "scheduled"),
        ]
    )
    result = check_freshness(frame, NOW)
    assert result.status == "STALE" and result.exit_code == EXIT_STALE
    assert result.overdue[0]["fixture_id"] == 2
    assert "48" in result.format()


def test_fixture_47_hours_after_kickoff_is_still_within_grace():
    frame = _frame([_row(1, NOW - pd.Timedelta(days=2), "finished"), _row(2, NOW - pd.Timedelta(hours=47), "scheduled")])
    assert check_freshness(frame, NOW).status == "OK"


def test_match_stuck_in_play_for_days_is_stale_but_postponed_is_not():
    stuck = _frame([_row(1, NOW - pd.Timedelta(days=1), "finished"), _row(2, NOW - pd.Timedelta(days=3), "skipped", state="INPLAY_2ND_HALF")])
    assert check_freshness(stuck, NOW).stale
    postponed = _frame([_row(1, NOW - pd.Timedelta(days=1), "finished"), _row(2, NOW - pd.Timedelta(days=3), "skipped", state="POSTPONED")])
    assert not check_freshness(postponed, NOW).stale


def test_old_result_with_nothing_listed_mid_season_is_stale():
    frame = _frame([_row(index, NOW - pd.Timedelta(days=12 + index), "finished") for index in range(1, 40)])
    result = check_freshness(frame, NOW)
    assert result.stale
    assert "10" in result.reasons[0]


def test_international_break_is_ok_when_the_next_round_is_listed():
    frame = _frame([_row(1, NOW - pd.Timedelta(days=14), "finished"), _row(2, NOW + pd.Timedelta(days=3), "scheduled")])
    result = check_freshness(frame, NOW)
    assert result.status == "OK"
    assert "国家队" in result.format()


def test_off_season_after_a_complete_season_is_ok():
    end = pd.Timestamp("2026-05-24T19:00:00Z")
    frame = _frame([_row(index, end - pd.Timedelta(hours=6 * index), "finished", season_id=25659) for index in range(380)])
    result = check_freshness(frame, pd.Timestamp("2026-07-20T12:00:00Z"))
    assert result.status == "OK"
    assert "休赛期" in result.format()


def test_season_list_marking_the_season_finished_also_counts_as_off_season():
    end = pd.Timestamp("2026-05-24T19:00:00Z")
    frame = _frame([_row(index, end - pd.Timedelta(days=index), "finished", season_id=25659) for index in range(40)])
    later = pd.Timestamp("2026-07-01T00:00:00Z")
    assert check_freshness(frame, later).stale
    seasons = {"seasons": [{"id": 25659, "finished": True}]}
    assert not check_freshness(frame, later, seasons_doc=seasons).stale


def test_empty_table_is_stale():
    from laliga.data.parse import empty_matches

    assert check_freshness(empty_matches(), NOW).stale


def test_now_override_from_environment(monkeypatch):
    monkeypatch.setenv(NOW_ENV, "2026-10-01T00:00:00Z")
    assert resolve_now() == pd.Timestamp("2026-10-01T00:00:00Z")
    monkeypatch.setenv(NOW_ENV, "not a date")
    with pytest.raises(ValueError):
        resolve_now()


def _seed(tmp_path: Path) -> None:
    frame = _frame(
        [
            _row(1, pd.Timestamp("2026-10-05T19:00:00Z"), "finished"),
            _row(2, pd.Timestamp("2026-10-18T19:00:00Z"), "scheduled"),
        ]
    )
    MatchStore(tmp_path).replace(frame)


def _run(args, tmp_path, now):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO)
    env["LALIGA_DATA_DIR"] = str(tmp_path)
    env[NOW_ENV] = now
    env.pop("SPORTMONKS_API_TOKEN", None)
    return subprocess.run(args, cwd=REPO, env=env, text=True, encoding="utf-8", capture_output=True, check=False)


@pytest.mark.parametrize(
    ("now", "code", "word"),
    [
        ("2026-10-14T12:00:00Z", 0, "OK"),
        ("2026-10-20T20:00:00Z", 2, "STALE"),
    ],
)
def test_cli_and_root_script_exit_codes_follow_the_clock_override(tmp_path, now, code, word):
    _seed(tmp_path)
    for args in (
        [sys.executable, "-m", "laliga", "check-freshness"],
        [sys.executable, str(REPO / "check_freshness.py")],
    ):
        result = _run(args, tmp_path, now)
        assert result.returncode == code, result.stderr
        assert result.stdout.startswith(word)
    as_json = _run([sys.executable, "-m", "laliga", "check-freshness", "--json"], tmp_path, now)
    assert json.loads(as_json.stdout)["status"] == word


def test_cli_rejects_an_unparseable_clock_override(tmp_path):
    _seed(tmp_path)
    result = _run([sys.executable, "-m", "laliga", "check-freshness"], tmp_path, "yesterday-ish")
    assert result.returncode == 1
    assert NOW_ENV in result.stderr
