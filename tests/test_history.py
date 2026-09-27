"""Football-data history import, team names, and a coarser compare cadence."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from laliga.backtest import compare_models
from laliga.config import ModelConfig
from laliga.data.football_data import (
    HistoryError,
    build_history,
    download_season,
    parse_seasons,
)
from laliga.data.teams import (
    FOOTBALL_DATA_TEAM_NAMES,
    OVERLAP_FOOTBALL_DATA_NAMES,
    canonical_team,
    team_id_for,
)
from laliga.model.features import XG_FEATURES, compute_rolling_features

_CSV = """Div,Date,Time,HomeTeam,AwayTeam,FTHG,FTAG,HTHG,HTAG,PSH,PSD,PSA,PSCH,PSCD,PSCA,B365H,B365D,B365A
SP1,01/09/2024,18:00,Ath Madrid,Barcelona,1,0,0,0,2.10,3.40,3.60,1.95,3.50,3.90,2.15,3.30,3.50
SP1,01/09/2024,21:00,Betis,Celta,2,1,1,0,1.80,3.60,4.50,1.70,3.70,4.80,1.85,3.50,4.40
SP1,01/09/2024,21:00,Betis,Celta,9,9,1,0,1.80,3.60,4.50,1.70,3.70,4.80,1.85,3.50,4.40
SP1,08/09/2024,16:00,Sociedad,Valencia,0,0,0,0,,,,,,,2.40,3.20,3.10
SP1,02/09/2024,18:00,La Coruna,Sevilla,1,1,0,0,2.20,3.30,3.40,2.10,3.40,3.50,2.25,3.20,3.30
"""


def test_every_football_data_name_maps_and_shares_a_canonical_with_sportmonks():
    for name in FOOTBALL_DATA_TEAM_NAMES:
        assert canonical_team(name)
    for name in OVERLAP_FOOTBALL_DATA_NAMES:
        assert canonical_team(name)
    assert canonical_team("FC Barcelona") == canonical_team("Barcelona")
    assert canonical_team("Club Atlético de Madrid") == canonical_team("Ath Madrid")
    assert canonical_team("Atlético de Madrid") == canonical_team("Ath Madrid")
    assert canonical_team("Deportivo A Coruña") == canonical_team("La Coruna")
    assert canonical_team("Deportivo Alavés") == "alaves"
    assert canonical_team("Deportivo") == "deportivo la coruna"
    assert canonical_team("Real Betis Balompié") == canonical_team("Betis")
    with pytest.raises(KeyError, match="没有球队对照"):
        canonical_team("未知俱乐部")


def test_merge_keeps_sportmonks_scores_and_maps_the_rest():
    football = parse_seasons([_CSV], [2024])
    sport = pd.DataFrame(
        [
            {
                "fixture_id": 501,
                "league_id": 564,
                "season_id": 24001,
                "season_name": "2024/2025",
                "round_id": 1,
                "round_name": "1",
                "starting_at": "2024-09-01 16:00:00+00:00",
                "state_id": 5,
                "state_name": "FT",
                "home_team_id": 42,
                "home_team_name": "Atlético de Madrid",
                "away_team_id": 43,
                "away_team_name": "FC Barcelona",
                "home_goals_ht": 1,
                "away_goals_ht": 0,
                "home_goals_ft": 2,
                "away_goals_ft": 0,
                "status": "finished",
            },
            {
                "fixture_id": 777,
                "league_id": 564,
                "season_id": 24001,
                "season_name": "2018/2019",
                "round_id": 1,
                "round_name": "1",
                "starting_at": "2018-09-02 20:00:00+00:00",
                "state_id": 5,
                "state_name": "FT",
                "home_team_id": 90,
                "home_team_name": "未知俱乐部",
                "away_team_id": 91,
                "away_team_name": "另一未知",
                "home_goals_ht": 0,
                "away_goals_ht": 0,
                "home_goals_ft": 1,
                "away_goals_ft": 1,
                "status": "finished",
            },
            {
                "fixture_id": 802,
                "league_id": 564,
                "season_id": 24001,
                "season_name": "2024/2025",
                "round_id": 1,
                "round_name": "1",
                "starting_at": "2024-09-02 17:00:00+00:00",
                "state_id": 5,
                "state_name": "FT",
                "home_team_id": 55,
                "home_team_name": "Deportivo A Coruña",
                "away_team_id": 56,
                "away_team_name": "Sevilla",
                "home_goals_ht": 0,
                "away_goals_ht": 0,
                "home_goals_ft": 1,
                "away_goals_ft": 1,
                "status": "finished",
            },
        ]
    )
    built = build_history(football, sport)
    frame = built.frame.set_index("fixture_id")
    assert frame.loc[501, "home_goals_ft"] == 2
    assert frame.loc[501, "home_team_id"] == 42
    assert int(frame.loc[501, "ft_score_mismatch"]) == 1
    assert built.mismatches[0]["football_data"] == [1, 0]
    assert built.mismatches[0]["sportmonks"] == [2, 0]
    assert built.duplicate_football_data == 1
    betis = frame[frame["home_team_name"] == "Betis"].iloc[0]
    assert int(betis.name) >= 9_000_000_000
    assert int(betis["home_team_id"]) == team_id_for("Betis", {})
    assert betis["fair_book"] == "pinnacle_pre"
    assert float(betis["raw_implied_home"]) == pytest.approx(1 / 1.80)
    sociedad = frame[frame["home_team_name"] == "Sociedad"].iloc[0]
    assert sociedad["fair_book"] == "bet365_pre"
    assert built.fallback_bet365 == 1
    assert 777 in frame.index
    assert int(frame.loc[777, "home_team_id"]) == 90
    assert "未知俱乐部" in built.unmapped_sportmonks
    assert int(frame.loc[802, "home_team_id"]) == 55
    assert frame["home_team_id"].tolist().count(42) == 1
    assert frame["home_team_id"].tolist().count(55) == 1
    assert frame["home_xg"].isna().all()


def test_unknown_team_raises_and_download_uses_the_cache(tmp_path):
    with pytest.raises(HistoryError, match="没有球队对照"):
        parse_seasons(
            ["Div,Date,HomeTeam,AwayTeam,FTHG,FTAG\nSP1,01/09/2024,No Such Club,Barcelona,1,0\n"],
            [2024],
        )
    with pytest.raises(HistoryError, match="赛季起始年"):
        parse_seasons(["Div,Date,HomeTeam,AwayTeam,FTHG,FTAG\nSP1,01/09/2024,Barcelona,Betis,1,0\n"])
    calls = []

    def fetch(url):
        calls.append(url)
        return _CSV

    text = download_season(2024, tmp_path, fetch=fetch)
    assert text.startswith("Div")
    assert (tmp_path / "cache" / "football-data" / "SP1_2425.csv").exists()
    download_season(2024, tmp_path, fetch=fetch)
    assert calls == ["https://www.football-data.co.uk/mmz4281/2425/SP1.csv"]


def test_missing_xg_does_not_break_rolling_features():
    kickoffs = pd.date_range("2024-08-01", periods=3, freq="7D", tz="UTC")
    frame = pd.DataFrame(
        {
            "fixture_id": [1, 2, 3],
            "season_id": [1, 1, 1],
            "starting_at": kickoffs,
            "home_team_id": [1, 2, 1],
            "away_team_id": [2, 1, 2],
            "home_goals_ft": [1, 0, 2],
            "away_goals_ft": [0, 0, 1],
            "home_goals_ht": [0, 0, 1],
            "away_goals_ht": [0, 0, 0],
            "home_xg": [np.nan, np.nan, np.nan],
            "away_xg": [np.nan, np.nan, np.nan],
            "status": ["finished", "finished", "finished"],
        }
    )
    features = compute_rolling_features(frame, use_xg=True)
    assert list(XG_FEATURES) == [column for column in XG_FEATURES if column in features.columns]
    assert not np.isfinite(features.loc[0, XG_FEATURES].to_numpy(dtype=float)).any()


def test_weekly_refit_scores_the_same_matches_with_fewer_fits():
    rows = []
    for index, kickoff in enumerate(pd.date_range("2023-08-12", periods=18, freq="D", tz="UTC")):
        home = 1 + (index % 4)
        away = 1 + ((index + 1) % 4)
        rows.append(
            {
                "fixture_id": 100 + index,
                "league_id": 564,
                "season_id": 1,
                "season_name": "2023/2024",
                "round_id": 1,
                "round_name": "1",
                "starting_at": kickoff,
                "state_id": 5,
                "state_name": "FT",
                "home_team_id": home,
                "home_team_name": f"队{home}",
                "away_team_id": away,
                "away_team_name": f"队{away}",
                "home_goals_ht": index % 2,
                "away_goals_ht": 0,
                "home_goals_ft": 1 + (index % 3),
                "away_goals_ft": index % 2,
                "status": "finished",
            }
        )
    frame = pd.DataFrame(rows)
    config = ModelConfig(xi=0.001, max_iter=12, min_train_matches=6, min_matches=2, min_ht_matches=4, l2=1.0)
    daily = compare_models(frame, config, refit_every_days=1, with_xgboost=False)
    weekly = compare_models(frame, config, refit_every_days=7, with_xgboost=False)
    assert "xgboost" not in daily.models
    assert daily.models["dixon_coles"].n == weekly.models["dixon_coles"].n
    assert daily.n_folds == weekly.n_folds
    assert weekly.n_refits < daily.n_refits
    assert weekly.refit_every_days == 7


def test_unmapped_team_in_an_overlap_season_aborts_the_import():
    football = parse_seasons([_CSV], [2024])
    sport = pd.DataFrame(
        [
            {
                "fixture_id": 900,
                "league_id": 564,
                "season_id": 24001,
                "season_name": "2024/2025",
                "starting_at": "2024-09-03 18:00:00+00:00",
                "home_team_id": 90,
                "home_team_name": "未知俱乐部",
                "away_team_id": 43,
                "away_team_name": "FC Barcelona",
                "home_goals_ft": 0,
                "away_goals_ft": 1,
                "status": "finished",
            }
        ]
    )
    with pytest.raises(HistoryError, match="未知俱乐部"):
        build_history(football, sport)


def test_july_restart_keeps_the_file_season_and_unique_fixture_ids():
    restart = (
        "Div,Date,Time,HomeTeam,AwayTeam,FTHG,FTAG\n"
        "SP1,01/07/2020,18:00,Alaves,Granada,1,0\n"
    )
    following = (
        "Div,Date,Time,HomeTeam,AwayTeam,FTHG,FTAG\n"
        "SP1,25/04/2021,18:00,Villarreal,Barcelona,2,1\n"
    )
    frame = parse_seasons([restart, following], [2019, 2020])
    july = frame[frame["fd_home"] == "Alaves"].iloc[0]
    april = frame[frame["fd_home"] == "Villarreal"].iloc[0]
    assert int(july["season_start"]) == 2019
    assert july["season_name"] == "2019/2020"
    assert int(april["season_start"]) == 2020
    assert int(july["fd_fixture_id"]) == 9_000_000_000 + 2019 * 100_000 + 1
    assert int(april["fd_fixture_id"]) == 9_000_000_000 + 2020 * 100_000 + 1
    assert int(july["fd_fixture_id"]) != 9_020_200_324
    built = build_history(frame, pd.DataFrame())
    assert not built.frame["fixture_id"].duplicated().any()
    same = "Div,Date,HomeTeam,AwayTeam,FTHG,FTAG\nSP1,01/09/2024,Barcelona,Betis,1,0\n"
    with pytest.raises(HistoryError, match="fixture_id"):
        parse_seasons([same, same], [2024, 2024])


def test_football_data_kickoff_is_london_time_and_merges_on_that_date():
    text = (
        "Div,Date,Time,HomeTeam,AwayTeam,FTHG,FTAG,HTHG,HTAG\n"
        "SP1,17/08/2025,20:30,Espanol,Ath Madrid,1,0,0,0\n"
        "SP1,15/01/2025,20:00,Alaves,Granada,0,0,0,0\n"
    )
    parsed = parse_seasons([text], [2024])
    summer = parsed[parsed["fd_home"] == "Espanol"].iloc[0]
    winter = parsed[parsed["fd_home"] == "Alaves"].iloc[0]
    assert summer["starting_at"] == pd.Timestamp("2025-08-17 19:30:00+00:00")
    assert summer["match_date"] == "2025-08-17"
    assert int(summer["season_start"]) == 2024
    assert winter["starting_at"] == pd.Timestamp("2025-01-15 20:00:00+00:00")
    sport = pd.DataFrame(
        [
            {
                "fixture_id": 7980,
                "league_id": 564,
                "season_id": 25001,
                "season_name": "2025/2026",
                "starting_at": "2025-08-17 19:30:00+00:00",
                "home_team_id": 19,
                "home_team_name": "Espanyol",
                "away_team_id": 7980,
                "away_team_name": "Atlético de Madrid",
                "home_goals_ht": 0,
                "away_goals_ht": 0,
                "home_goals_ft": 2,
                "away_goals_ft": 1,
                "status": "finished",
            }
        ]
    )
    built = build_history(parsed, sport)
    assert len(built.frame) == 2
    merged = built.frame.set_index("fixture_id").loc[7980]
    assert int(merged["away_team_id"]) == 7980
    assert merged["starting_at"] == pd.Timestamp("2025-08-17 19:30:00+00:00")
    assert list(built.frame["fixture_id"]).count(7980) == 1


def test_concat_without_pinnacle_columns_does_not_warn():
    priced = _CSV
    bare = (
        "Div,Date,Time,HomeTeam,AwayTeam,FTHG,FTAG,HTHG,HTAG,PSH,PSD,PSA\n"
        "SP1,01/08/2013,16:00,Barcelona,Betis,1,0,0,0,,,\n"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        frame = parse_seasons([priced, bare], [2024, 2013])
    assert len(frame) == 6
    assert "pin_pre_home" in frame.columns


def test_report_counts_price_sources_on_the_test_rows_only():
    rows = []
    for index, kickoff in enumerate(pd.date_range("2023-08-12", periods=18, freq="D", tz="UTC")):
        home = 1 + (index % 4)
        away = 1 + ((index + 1) % 4)
        if index < 6:
            season_name, fair_book = "2022/2023", "bet365_pre"
        elif index < 12:
            season_name, fair_book = "2023/2024", "pinnacle_pre"
        elif index < 16:
            season_name, fair_book = "2024/2025", "bet365_pre"
        else:
            season_name, fair_book = "2024/2025", "sportmonks"
        rows.append(
            {
                "fixture_id": 100 + index,
                "league_id": 564,
                "season_id": 1,
                "season_name": season_name,
                "season_start": int(season_name[:4]),
                "round_id": 1,
                "round_name": "1",
                "starting_at": kickoff,
                "state_id": 5,
                "state_name": "FT",
                "home_team_id": home,
                "home_team_name": f"队{home}",
                "away_team_id": away,
                "away_team_name": f"队{away}",
                "home_goals_ht": index % 2,
                "away_goals_ht": 0,
                "home_goals_ft": 1 + (index % 3),
                "away_goals_ft": index % 2,
                "status": "finished",
                "fair_book": fair_book,
            }
        )
    config = ModelConfig(xi=0.001, max_iter=12, min_train_matches=6, min_matches=2, min_ht_matches=4, l2=1.0)
    report = compare_models(pd.DataFrame(rows), config, refit_every_days=7, with_xgboost=False, history_source="football-data")
    sources = report.price_sources
    assert sources["test_rows"] == report.models["dixon_coles"].n == 12
    assert sources["counts"] == {
        "pinnacle_pre": 6,
        "bet365_pre": 4,
        "sportmonks": 2,
        "missing": 0,
    }
    assert "2022/2023" not in sources["by_season"]
    assert sources["by_season"]["2023/2024"]["pinnacle_pre"] == 6
    assert sources["by_season"]["2024/2025"]["bet365_pre"] == 4
    assert sources["by_season"]["2024/2025"]["sportmonks"] == 2
    rendered = "\n".join(report.notes)
    assert "评测期每一场公平赛前盘的来源：Pinnacle 赛前盘（PSH） 6 场，Bet365 赛前盘 4 场，SportMonks 已存价格 2 场，没有可用赛前盘 0 场。" in rendered
    assert report.to_dict()["price_sources"]["counts"]["pinnacle_pre"] == 6
