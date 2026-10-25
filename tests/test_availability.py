"""Lineups / sidelined: parsing, fetching, pre-match features, and the walk-forward adjustment."""

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from laliga.config import ModelConfig
from laliga.data.availability import (
    AVAILABILITY_DENIED,
    AVAILABILITY_INCLUDES,
    AvailabilityStore,
    fetch_availability,
    parse_availability,
)
from laliga.data.client import SportMonksClient
from laliga.data.store import MatchStore
from laliga.model.availability import (
    AvailabilityFeatures,
    Coefficients,
    compare_availability,
    fit_coefficients,
    make_adjuster,
)
from tests import availability_synthetic

REPO = Path(__file__).resolve().parents[1]
CONFIG = ModelConfig(max_iter=40, min_train_matches=90, min_ht_matches=20)

DOC_PAYLOAD = {
    "id": 19134562,
    "league_id": 564,
    "starting_at": "2025-01-16 20:00:00",
    "lineups": [
        {"id": 1, "fixture_id": 19134562, "player_id": 162536, "team_id": 9, "position_id": 25, "type_id": 11, "formation_position": 3, "player_name": "Rúben Dias", "jersey_number": 3},
        {"id": 2, "fixture_id": 19134562, "player_id": 982, "team_id": 9, "position_id": 26, "type_id": 12, "formation_position": None, "player_name": "Bench Guy", "jersey_number": 20},
    ],
    "sidelined": [
        {
            "id": 468798,
            "fixture_id": 19134562,
            "sideline_id": 112256,
            "participant_id": 65,
            "player_id": 3596,
            "type_id": 1619,
            "sideline": {
                "id": 112256,
                "player_id": 3596,
                "type_id": 1619,
                "category": "doubtful",
                "team_id": 65,
                "season_id": None,
                "start_date": "2023-05-28",
                "end_date": None,
                "games_missed": None,
                "completed": False,
            },
        }
    ],
    "metadata": [
        {"id": 7621296, "metadatable_id": 19134562, "type_id": 159, "value_type": "object", "values": {"home": "4-2-3-1", "away": "4-3-3"}},
        {"id": 7621297, "metadatable_id": 19134562, "type_id": 572, "value_type": "object", "values": {"confirmed": False}},
    ],
}


def test_documented_payload_shape_is_parsed():
    lineups, sidelined, meta = parse_availability(DOC_PAYLOAD, fetched_at="2025-01-16T18:00:00+00:00", status="scheduled")
    assert [(row["player_id"], row["type_id"]) for row in lineups] == [(162536, 11), (982, 12)]
    assert sidelined == [
        {
            "fixture_id": 19134562,
            "team_id": 65,
            "player_id": 3596,
            "sideline_id": 112256,
            "type_id": 1619,
            "category": "doubtful",
            "start_date": "2023-05-28",
            "end_date": "",
            "games_missed": None,
            "completed": False,
        }
    ]
    assert meta["lineup_confirmed"] is False
    assert meta["n_starters"] == 1 and meta["n_sidelined"] == 1


def _store_rows():
    def row(fixture_id, kickoff, status):
        finished = status == "finished"
        return {
            "fixture_id": fixture_id,
            "league_id": 564,
            "season_id": 1,
            "season_name": "2026/2027",
            "round_id": 1,
            "round_name": "1",
            "starting_at": kickoff,
            "state_id": 5 if finished else 1,
            "state_name": "FT" if finished else "NS",
            "home_team_id": 9,
            "home_team_name": "A",
            "away_team_id": 65,
            "away_team_name": "B",
            "home_goals_ht": 0 if finished else None,
            "away_goals_ht": 0 if finished else None,
            "home_goals_ft": 1 if finished else None,
            "away_goals_ft": 0 if finished else None,
            "status": status,
        }

    return pd.DataFrame(
        [
            row(1, "2026-10-04 19:00:00+00:00", "finished"),
            row(2, "2026-10-11 19:00:00+00:00", "finished"),
            row(3, "2026-10-18 19:00:00+00:00", "scheduled"),
        ]
    )


def test_fetch_caches_finished_rerequests_upcoming_and_stops_on_403(tmp_path):
    hits: list[tuple[str, dict]] = []
    mode = {"status": 200}

    def transport(url, params):
        hits.append((url, dict(params)))
        if mode["status"] == 403:
            return 403, {}, {"message": "You don't have access to this endpoint on your subscription."}
        ids = [int(item) for item in url.rsplit("/", 1)[1].split(",")]
        data = [{**DOC_PAYLOAD, "id": fixture_id} for fixture_id in ids]
        return 200, {}, {"data": data, "pagination": {"has_more": False}}

    store = MatchStore(tmp_path)
    store.replace(_store_rows())
    client = SportMonksClient("availability-token", tmp_path / "cache", transport=transport, min_interval=0, sleep=lambda _s: None)
    availability = AvailabilityStore(tmp_path)
    now = pd.Timestamp("2026-10-14T12:00:00Z")

    first = fetch_availability(client, store, availability, now=now)
    assert first.finished_requested == 2 and first.upcoming_requested == 1
    assert all(params["include"] == AVAILABILITY_INCLUDES for _, params in hits)
    assert any(url.endswith("/fixtures/multi/1,2") for url, _ in hits)
    assert any(url.endswith("/fixtures/multi/3") for url, _ in hits)
    lineups, sidelined, fetched = availability.load()
    assert sorted(fetched["fixture_id"].tolist()) == [1, 2, 3]
    assert len(lineups) == 6 and len(sidelined) == 3

    before = len(hits)
    second = fetch_availability(client, store, availability, now=now)
    assert second.finished_requested == 0 and second.upcoming_requested == 1
    assert [url for url, _ in hits[before:]] == [hits[-1][0]]
    assert hits[-1][0].endswith("/fixtures/multi/3")
    assert len(availability.load()[1]) == 3

    mode["status"] = 403
    denied = fetch_availability(client, store, availability, now=now)
    assert denied.denied == AVAILABILITY_DENIED
    assert len(availability.load()[0]) == 6


def _features(frame, lineups, sidelined, fetched):
    return AvailabilityFeatures(frame, lineups, sidelined, fetched)


def test_features_use_earlier_lineups_and_respect_record_dates():
    frame, lineups, sidelined, fetched = availability_synthetic.build(seed=1, cycles=1)
    features = _features(frame, lineups, sidelined, fetched)
    row = frame.iloc[-1]
    team = int(row["home_team_id"])
    day = pd.Timestamp(row["starting_at"]).floor("D")
    shares, n = features.starter_shares(team, day)
    assert n == 10
    assert all(0 < share <= 1 for share in shares.values())
    later = frame[frame["starting_at"] >= day]
    changed_lineups = lineups.copy()
    future = changed_lineups["fixture_id"].isin(later["fixture_id"])
    changed_lineups.loc[future, "type_id"] = 12
    assert _features(frame, changed_lineups, sidelined, fetched).starter_shares(team, day) == (shares, n)

    fixture_id = int(row["fixture_id"])
    base = sidelined[sidelined["fixture_id"] != fixture_id]
    regular = max(shares, key=shares.get)
    fringe = 9999

    def missing_with(records):
        extra = pd.DataFrame(records)
        table = pd.concat([base, extra], ignore_index=True) if records else base
        return _features(frame, lineups, table, fetched).team(fixture_id, team, row["starting_at"]).missing

    common = {"fixture_id": fixture_id, "team_id": team, "sideline_id": 1, "type_id": 1, "games_missed": 1, "completed": False}
    assert missing_with([]) == 0.0
    injured = missing_with([{**common, "player_id": regular, "category": "injury", "start_date": "2000-01-01", "end_date": ""}])
    assert injured == pytest.approx(shares[regular])
    doubtful = missing_with([{**common, "player_id": regular, "category": "doubtful", "start_date": "2000-01-01", "end_date": ""}])
    assert doubtful == pytest.approx(shares[regular] / 2)
    starts_later = (day + pd.Timedelta(days=2)).date().isoformat()
    assert missing_with([{**common, "player_id": regular, "category": "injury", "start_date": starts_later, "end_date": ""}]) == 0.0
    ended = (day - pd.Timedelta(days=2)).date().isoformat()
    assert missing_with([{**common, "player_id": regular, "category": "injury", "start_date": "2000-01-01", "end_date": ended}]) == 0.0
    assert missing_with([{**common, "player_id": fringe, "category": "injury", "start_date": "2000-01-01", "end_date": ""}]) == 0.0


def test_no_fetch_means_no_feature_rather_than_zero():
    frame, lineups, sidelined, fetched = availability_synthetic.build(seed=1, cycles=1)
    row = frame.iloc[-1]
    features = _features(frame, lineups, sidelined, fetched[fetched["fixture_id"] != int(row["fixture_id"])])
    assert features.team(int(row["fixture_id"]), int(row["home_team_id"]), row["starting_at"]).missing is None


def test_coefficients_recover_a_real_effect_and_stay_near_zero_without_one():
    rng = np.random.default_rng(0)
    rows = []
    for _ in range(2000):
        lam_h, lam_a = float(rng.uniform(0.8, 2.0)), float(rng.uniform(0.6, 1.6))
        m_h, m_a = float(rng.integers(0, 4)), float(rng.integers(0, 4))
        rows.append(
            {
                "lam_home": lam_h,
                "lam_away": lam_a,
                "home_missing": m_h,
                "away_missing": m_a,
                "score": (int(rng.poisson(lam_h * np.exp(-0.15 * m_h + 0.1 * m_a))), int(rng.poisson(lam_a * np.exp(-0.15 * m_a + 0.1 * m_h)))),
            }
        )
    fitted = fit_coefficients(rows, "missing")
    assert fitted.attack == pytest.approx(0.15, abs=0.05)
    assert fitted.defence == pytest.approx(0.10, abs=0.05)
    null_rows = [{**row, "score": (int(rng.poisson(row["lam_home"])), int(rng.poisson(row["lam_away"])))} for row in rows]
    null = fit_coefficients(null_rows, "missing")
    assert abs(null.attack) < 0.05 and abs(null.defence) < 0.05
    assert not fit_coefficients(rows[:10], "missing").active


def test_walk_forward_gains_when_absences_matter_and_not_when_they_do_not():
    frame, lineups, sidelined, fetched = availability_synthetic.build(seed=3, attack_effect=0.2, defence_effect=0.12)
    report = compare_availability(frame, _features(frame, lineups, sidelined, fetched), CONFIG, min_fit_rows=60)
    active = report.active["missing"]
    assert report.coverage["missing_active"] > 50
    assert active["missing"].ft_log_loss < active["dixon_coles"].ft_log_loss
    assert report.coefficients["missing"]["attack_per_missing_starter"] > 0.05
    assert report.models["missing"].n == report.models["dixon_coles"].n == report.models["baseline"].n
    labels = [item["label"] for item in report.paired]
    assert "Dixon–Coles − 历史频率基准" in labels
    assert json.loads(json.dumps(report.to_dict()))["schema_version"] == 1

    flat, flat_lineups, flat_sidelined, flat_fetched = availability_synthetic.build(seed=3, attack_effect=0.0, defence_effect=0.0)
    null = compare_availability(flat, _features(flat, flat_lineups, flat_sidelined, flat_fetched), CONFIG, min_fit_rows=60)
    assert null.recommend_default is False
    assert abs(null.coefficients["missing"]["attack_per_missing_starter"]) < 0.1


def test_scores_for_a_day_ignore_that_day_and_later_results():
    frame, lineups, sidelined, fetched = availability_synthetic.build(seed=4, cycles=2)
    features = _features(frame, lineups, sidelined, fetched)
    base = compare_availability(frame, features, CONFIG, min_fit_rows=40)
    last_day = frame["starting_at"].dt.floor("D").max()
    changed = frame.copy()
    on_last = changed["starting_at"].dt.floor("D") == last_day
    changed.loc[on_last, "home_goals_ft"] = 7
    again = compare_availability(changed, _features(changed, lineups, sidelined, fetched), CONFIG, min_fit_rows=40)
    by_id = {row["fixture_id"]: row for row in again.rows}
    last_ids = set(frame.loc[on_last, "fixture_id"].astype(int))
    for row in base.rows:
        for prefix in ("model", "missing", "lineup_gap"):
            assert by_id[row["fixture_id"]][f"{prefix}_ft_probs"] == row[f"{prefix}_ft_probs"]
        if row["fixture_id"] in last_ids:
            assert by_id[row["fixture_id"]]["score"][0] == 7


def test_adjuster_uses_fresh_sidelined_or_a_confirmed_lineup_only():
    today = datetime.now(timezone.utc).date()
    frame, lineups, sidelined, fetched = availability_synthetic.build(seed=5, cycles=1, today=today, upcoming=True)
    upcoming = frame[frame["status"] == "scheduled"].iloc[0]
    fixture_id = int(upcoming["fixture_id"])
    now = pd.Timestamp(datetime.now(timezone.utc))
    fresh = fetched.copy()
    fresh.loc[fresh["fixture_id"] == fixture_id, "fetched_at"] = now.isoformat()
    coefficients = {"missing": Coefficients(0.1, 0.05, 500), "lineup_gap": Coefficients(0.2, 0.1, 500)}

    stale = make_adjuster(_features(frame, lineups, sidelined, fetched), coefficients, now=now)
    multipliers, info = stale(upcoming)
    assert multipliers.neutral and info["used"] is None and "36" in info["reason"]

    adjust = make_adjuster(_features(frame, lineups, sidelined, fresh), coefficients, now=now)
    multipliers, info = adjust(upcoming)
    assert info["used"] == "missing"
    expected = np.exp(-0.1 * info["home_missing"] + 0.05 * info["away_missing"])
    assert multipliers.home == pytest.approx(expected)

    xi = []
    for team in (int(upcoming["home_team_id"]), int(upcoming["away_team_id"])):
        for player in range(11, 18):
            xi.append({"fixture_id": fixture_id, "team_id": team, "player_id": team * 100 + player, "player_name": "", "type_id": 11, "position_id": 26, "formation_position": None})
        for player in range(4):
            xi.append({"fixture_id": fixture_id, "team_id": team, "player_id": team * 100 + player, "player_name": "", "type_id": 11, "position_id": 26, "formation_position": None})
    confirmed = fresh.copy()
    confirmed.loc[confirmed["fixture_id"] == fixture_id, "lineup_confirmed"] = True
    lineup_adjust = make_adjuster(_features(frame, pd.concat([lineups, pd.DataFrame(xi)]), sidelined, confirmed), coefficients, now=now)
    multipliers, info = lineup_adjust(upcoming)
    assert info["used"] == "lineup_gap"
    assert info["home_lineup_gap"] > 3
    assert multipliers.home < 1


def _run(args, data_dir):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO)
    env["LALIGA_DATA_DIR"] = str(data_dir)
    env.pop("SPORTMONKS_API_TOKEN", None)
    return subprocess.run(args, cwd=REPO, env=env, text=True, encoding="utf-8", capture_output=True, check=False)


def test_cli_compare_then_predict_with_the_flag(tmp_path):
    today = datetime.now(timezone.utc).date()
    frame, lineups, sidelined, fetched = availability_synthetic.build(seed=6, today=today, upcoming=True)
    upcoming_ids = frame.loc[frame["status"] == "scheduled", "fixture_id"]
    fetched.loc[fetched["fixture_id"].isin(upcoming_ids), "fetched_at"] = datetime.now(timezone.utc).isoformat()
    MatchStore(tmp_path).replace(frame)
    store = AvailabilityStore(tmp_path)
    store.upsert(lineups.to_dict("records"), sidelined.to_dict("records"), fetched.to_dict("records"))

    compared = _run([sys.executable, "-m", "laliga", "compare-availability", "--data-dir", str(tmp_path), "--min-train", "90", "--max-iter", "40"], tmp_path)
    assert compared.returncode == 0, compared.stderr
    assert "Dixon–Coles + 伤停" in compared.stdout
    assert "历史频率基准" in compared.stdout
    report = json.loads((tmp_path / "predictions" / "availability_comparison.json").read_text(encoding="utf-8"))
    assert report["computed_from_stored_matches"] is True
    assert (tmp_path / "models" / "availability.json").exists()

    latest = tmp_path / "predictions" / "latest_next.json"
    predicted = _run([sys.executable, "-m", "laliga", "predict", "--data-dir", str(tmp_path), "--next", "--max-iter", "40", "--availability", "--json", str(latest)], tmp_path)
    assert predicted.returncode == 0, predicted.stderr
    assert "伤停/首发调整" in predicted.stdout
    from laliga.archive import PredictionArchive
    from laliga.report import latest_next_problem

    assert latest_next_problem(json.loads(latest.read_text(encoding="utf-8"))) == ""
    records = PredictionArchive(tmp_path).live_records()
    assert records and all("+availability" in record["model_version"] for record in records)
    assert all(record["extra"]["availability"]["used"] == "missing" for record in records)
