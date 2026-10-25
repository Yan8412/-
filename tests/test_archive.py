"""Prediction archive: append-only, nothing after kickoff, hits joined from results."""

import json

import pandas as pd
import pytest

from laliga.archive import (
    LIVE,
    RECONSTRUCTION,
    PredictionArchive,
    build_history,
    history_report,
    reconstruct_season,
    select_shown,
    summarize_history,
)
from laliga.config import ModelConfig
from laliga.data.parse import fixtures_to_frame
from laliga.data.store import MatchStore
from laliga.synthetic import generate_synthetic_fixtures

KICKOFF = "2026-10-18T19:00:00+00:00"


def _prediction(fixture_id=1, kickoff=KICKOFF, home=0.5, draw=0.3, away=0.2, tops=((1, 0), (1, 1), (2, 1)), ht=(0.4, 0.45, 0.15)):
    return {
        "fixture_id": fixture_id,
        "starting_at": kickoff,
        "season_id": 27965,
        "season_name": "2026/2027",
        "round_name": "9",
        "home_team_id": 10,
        "home_team": "Home",
        "away_team_id": 20,
        "away_team": "Away",
        "ft": {"home": home, "draw": draw, "away": away},
        "ht": {"home": ht[0], "draw": ht[1], "away": ht[2]},
        "top_scores": [{"home_goals": h, "away_goals": a, "probability": 0.1} for h, a in tops],
        "expected_goals_ft": {"home": 1.5, "away": 1.0},
        "expected_goals_ht": {"home": 0.7, "away": 0.4},
    }


def _match(fixture_id=1, kickoff=KICKOFF, status="finished", ft=(2, 1), ht=(1, 1), state="FT"):
    return {
        "fixture_id": fixture_id,
        "league_id": 564,
        "season_id": 27965,
        "season_name": "2026/2027",
        "round_id": 9,
        "round_name": "9",
        "starting_at": kickoff,
        "state_id": 5 if status == "finished" else 1,
        "state_name": state,
        "home_team_id": 10,
        "home_team_name": "Home",
        "away_team_id": 20,
        "away_team_name": "Away",
        "home_goals_ht": ht[0] if status == "finished" else None,
        "away_goals_ht": ht[1] if status == "finished" else None,
        "home_goals_ft": ft[0] if status == "finished" else None,
        "away_goals_ft": ft[1] if status == "finished" else None,
        "status": status,
    }


def _history_matches(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    frame["starting_at"] = pd.to_datetime(frame["starting_at"], utc=True)
    return frame


def _earlier_results(n=30) -> list[dict]:
    rows = []
    for index in range(n):
        kickoff = pd.Timestamp("2026-08-15T19:00:00+00:00") + pd.Timedelta(days=index)
        score = (1, 0) if index % 3 else (1, 1)
        rows.append(_match(fixture_id=100 + index, kickoff=kickoff.isoformat(), ft=score, ht=(0, 0)))
    return rows


def test_records_after_kickoff_are_refused_and_earlier_lines_never_change(tmp_path):
    archive = PredictionArchive(tmp_path)
    first = archive.append([_prediction()], model_version="v", origin="test", generated_at="2026-10-17T08:00:00Z")
    assert first.accepted == 1
    original = archive.live_path.read_bytes()

    second = archive.append([_prediction(home=0.6, draw=0.2)], model_version="v", origin="test", generated_at="2026-10-18T09:00:00Z")
    assert second.accepted == 1
    assert archive.live_path.read_bytes().startswith(original)

    at_kickoff = archive.append([_prediction(home=0.9, draw=0.05, away=0.05)], model_version="v", origin="test", generated_at=KICKOFF)
    after = archive.append([_prediction(home=0.9, draw=0.05, away=0.05)], model_version="v", origin="test", generated_at="2026-10-18T21:00:00Z")
    assert at_kickoff.accepted == 0 and after.accepted == 0
    assert at_kickoff.refused and after.refused
    records = archive.live_records()
    assert len(records) == 2
    assert all(record["source"] == LIVE for record in records)
    assert archive.live_path.read_bytes().startswith(original)

    shown = select_shown(records)
    assert shown[1]["ft"]["home"] == pytest.approx(0.6)
    assert shown[1]["generated_at"].startswith("2026-10-18T09:00")
    assert shown[1]["n_generations"] == 2


def test_record_carries_every_field_the_history_needs(tmp_path):
    archive = PredictionArchive(tmp_path)
    archive.append([_prediction()], model_version="laliga-x/dixon_coles-v1", origin="cli", generated_at="2026-10-17T08:00:00Z")
    record = archive.live_records()[0]
    for key in (
        "fixture_id",
        "kickoff_utc",
        "home_team",
        "away_team",
        "ft",
        "ht",
        "top_scores",
        "expected_goals_ft",
        "expected_goals_ht",
        "model_version",
        "generated_at",
    ):
        assert key in record
    assert record["kickoff_utc"] == KICKOFF
    assert len(record["top_scores"]) == 3
    assert record["model_version"] == "laliga-x/dixon_coles-v1"


def test_a_line_written_after_kickoff_by_hand_is_still_ignored(tmp_path):
    archive = PredictionArchive(tmp_path)
    archive.append([_prediction()], model_version="v", origin="test", generated_at="2026-10-17T08:00:00Z")
    forged = archive.live_records()[0] | {"generated_at": "2026-10-18T22:00:00+00:00", "ft": {"home": 0.0, "draw": 0.0, "away": 1.0}}
    with open(archive.live_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(forged) + "\n")
        handle.write('{"fixture_id": 1, "truncated')
    shown = select_shown(archive.live_records())
    assert shown[1]["ft"]["home"] == pytest.approx(0.5)


def test_kickoff_moved_earlier_drops_lines_written_after_the_real_kickoff(tmp_path):
    archive = PredictionArchive(tmp_path)
    archive.append([_prediction(home=0.4)], model_version="v", origin="t", generated_at="2026-10-16T08:00:00Z")
    archive.append([_prediction(home=0.7, draw=0.2, away=0.1)], model_version="v", origin="t", generated_at="2026-10-18T12:00:00Z")
    moved = _history_matches([_match(kickoff="2026-10-18T10:00:00+00:00")])
    shown = select_shown(archive.live_records(), moved)
    assert shown[1]["ft"]["home"] == pytest.approx(0.4)


def test_finished_or_already_started_fixture_in_the_match_table_is_refused(tmp_path):
    archive = PredictionArchive(tmp_path)
    finished = _history_matches([_match()])
    result = archive.append([_prediction()], model_version="v", origin="t", generated_at="2026-10-17T08:00:00Z", matches=finished)
    assert result.accepted == 0
    assert "finished" in result.refused[0][1]


def test_hits_are_joined_from_results_with_baselines_from_earlier_matches_only(tmp_path):
    archive = PredictionArchive(tmp_path)
    archive.append([_prediction(tops=((1, 0), (2, 1), (1, 1)))], model_version="v", origin="t", generated_at="2026-10-17T08:00:00Z")
    later = _match(fixture_id=999, kickoff="2026-10-17T12:00:00+00:00", ft=(5, 5), ht=(3, 3))
    matches = _history_matches(_earlier_results() + [later, _match(ft=(2, 1), ht=(1, 1))])
    rows = build_history(archive.live_records(), matches, now="2026-10-20T00:00:00Z")
    assert len(rows) == 1
    row = rows[0]
    assert row["status"] == "finished"
    assert row["actual_ft"] == "2-1" and row["actual_ht"] == "1-1"
    assert row["ft_pick"] == "home" and row["ft_hit"] is True
    assert row["ht_pick"] == "draw" and row["ht_hit"] is True
    assert row["top3_hit"] is True
    assert row["always_home_ft_hit"] is True and row["always_home_ht_hit"] is False
    baseline = row["baseline"]
    assert baseline["n_matches"] == 30
    assert "5-5" not in baseline["top_scores"]
    assert baseline["top_scores"][0] == "1-0"
    assert baseline["top3_hit"] is False


def test_misses_awaiting_and_postponed_are_reported_separately(tmp_path):
    archive = PredictionArchive(tmp_path)
    archive.append(
        [
            _prediction(fixture_id=1, tops=((1, 0), (1, 1), (0, 0))),
            _prediction(fixture_id=2, kickoff="2026-10-18T21:00:00+00:00"),
            _prediction(fixture_id=3, kickoff="2026-10-19T19:00:00+00:00"),
            _prediction(fixture_id=4, kickoff="2026-10-25T19:00:00+00:00"),
        ],
        model_version="v",
        origin="t",
        generated_at="2026-10-17T08:00:00Z",
    )
    matches = _history_matches(
        _earlier_results()
        + [
            _match(fixture_id=1, ft=(0, 2), ht=(0, 1)),
            _match(fixture_id=2, kickoff="2026-10-18T21:00:00+00:00", status="scheduled", state="NS"),
            _match(fixture_id=3, kickoff="2026-10-19T19:00:00+00:00", status="skipped", state="POSTPONED"),
            _match(fixture_id=4, kickoff="2026-10-25T19:00:00+00:00", status="scheduled", state="NS"),
        ]
    )
    rows = build_history(archive.live_records(), matches, now="2026-10-20T00:00:00Z")
    by_id = {row["fixture_id"]: row for row in rows}
    assert set(by_id) == {1, 2, 3}
    assert by_id[1]["ft_hit"] is False and by_id[1]["ht_hit"] is False and by_id[1]["top3_hit"] is False
    assert by_id[2]["status"] == "awaiting"
    assert by_id[3]["status"] == "not_played"
    summary = summarize_history(rows)
    assert summary["n_finished"] == 1
    assert summary["n_awaiting"] == 1
    assert summary["n_not_played"] == 1
    assert summary["model"]["ft_hit_rate"] == 0.0
    assert summary["always_home"]["ft_hit_rate"] == 0.0


def test_running_hit_rates_accumulate_in_kickoff_order(tmp_path):
    archive = PredictionArchive(tmp_path)
    kickoffs = ["2026-10-18T13:00:00+00:00", "2026-10-18T15:00:00+00:00", "2026-10-18T17:00:00+00:00"]
    archive.append(
        [_prediction(fixture_id=index + 1, kickoff=kickoff) for index, kickoff in enumerate(kickoffs)],
        model_version="v",
        origin="t",
        generated_at="2026-10-17T08:00:00Z",
    )
    results = [(1, 0), (0, 1), (2, 0)]
    matches = _history_matches(
        [_match(fixture_id=index + 1, kickoff=kickoff, ft=score, ht=(0, 0)) for index, (kickoff, score) in enumerate(zip(kickoffs, results))]
    )
    rows = build_history(archive.live_records(), matches, now="2026-10-19T00:00:00Z")
    assert [row["running"]["n"] for row in rows] == [1, 2, 3]
    assert [round(row["running"]["ft_hit_rate"], 4) for row in rows] == [1.0, 0.5, round(2 / 3, 4)]
    assert summarize_history(rows)["model"]["top3_hit_rate"] == pytest.approx(1 / 3)


def test_reconstruction_is_time_ordered_labelled_and_kept_out_of_live_rates(tmp_path):
    today = pd.Timestamp("2026-10-14").date()
    matches = fixtures_to_frame(generate_synthetic_fixtures(seed=3, today=today))
    store = MatchStore(tmp_path)
    store.replace(matches)
    built = reconstruct_season(store.load(), ModelConfig(max_iter=40, min_train_matches=18, min_ht_matches=12), generated_at="2026-10-14T00:00:00Z")
    season = int(matches.sort_values("starting_at").iloc[-1]["season_id"])
    played = matches[(matches["season_id"] == season) & (matches["status"] == "finished")]
    assert built.season_id == season
    assert len(built.records) == len(played)
    for record in built.records:
        assert record["source"] == RECONSTRUCTION
        cutoff = pd.Timestamp(record["information_cutoff"])
        kickoff = pd.Timestamp(record["kickoff_utc"])
        assert cutoff == kickoff.floor("D")

    archive = PredictionArchive(tmp_path)
    archive.replace_reconstruction(built.records)
    with pytest.raises(ValueError):
        archive.replace_reconstruction([built.records[0] | {"source": LIVE}])
    report = history_report(tmp_path, store.load(), now="2026-10-14T00:00:00Z")
    assert report["live"]["rows"] == []
    assert report["live"]["summary"]["n_finished"] == 0
    assert report["backtest_reconstruction"]["summary"]["n_finished"] == len(played)
    assert not archive.live_path.exists()


def test_reconstruction_of_a_day_does_not_depend_on_that_day_or_later(tmp_path):
    today = pd.Timestamp("2026-10-14").date()
    matches = fixtures_to_frame(generate_synthetic_fixtures(seed=5, today=today))
    config = ModelConfig(max_iter=40, min_train_matches=18, min_ht_matches=12)
    base = reconstruct_season(matches, config, generated_at="2026-10-14T00:00:00Z")
    season = base.season_id
    finished = matches[(matches["season_id"] == season) & (matches["status"] == "finished")]
    last_day = finished["starting_at"].dt.floor("D").max()
    changed = matches.copy()
    mask = changed["starting_at"].dt.floor("D") >= last_day
    changed.loc[mask & (changed["status"] == "finished"), ["home_goals_ft", "away_goals_ft"]] = [9, 0]
    again = reconstruct_season(changed, config, generated_at="2026-10-14T00:00:00Z")
    before_last = [record for record in base.records if pd.Timestamp(record["kickoff_utc"]).floor("D") < last_day]
    same = {record["fixture_id"]: record["ft"] for record in again.records}
    for record in before_last:
        assert same[record["fixture_id"]] == record["ft"]
