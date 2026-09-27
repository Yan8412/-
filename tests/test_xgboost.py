"""XGBoost challenger: pre-match features, leakage, and the shared walk-forward."""

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from laliga.backtest import _comparison_notes, compare_models, walk_forward
from laliga.config import ModelConfig
from laliga.data.parse import fixtures_to_frame
from laliga.data.store import MatchStore
from laliga.model.dixon_coles import fit_dixon_coles
from laliga.model.features import (
    LEAKAGE_COLUMNS,
    compute_rolling_features,
    dixon_coles_features,
    model_feature_names,
    odds_column_status,
    odds_features,
    xg_column_status,
)
from laliga.model.metrics import calibration_summary, paired_logloss_difference, per_match_log_loss, multiclass_log_loss
from laliga.model.service import ScorelineModel, fit_models
from laliga.model.xgboost_model import XGBoostConfig, fit_outcome_model, validation_size
from laliga.synthetic import generate_synthetic_fixtures
from laliga.webapp import create_app

REPO = Path(__file__).resolve().parents[1]


def _frame(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    frame["starting_at"] = pd.to_datetime(frame["starting_at"], utc=True)
    return frame


def _match(fixture_id, kickoff, home, away, home_goals, away_goals, **extra):
    row = {
        "fixture_id": fixture_id,
        "season_id": 1,
        "season_name": "2024/2025",
        "starting_at": kickoff,
        "home_team_id": home,
        "away_team_id": away,
        "home_team_name": f"队{home}",
        "away_team_name": f"队{away}",
        "home_goals_ft": home_goals,
        "away_goals_ft": away_goals,
        "home_goals_ht": 0,
        "away_goals_ht": 0,
        "status": "finished",
    }
    row.update(extra)
    return row


def test_feature_names_do_not_include_post_match_fields():
    names = set(model_feature_names(use_xg=True, use_odds=True))
    assert LEAKAGE_COLUMNS.isdisjoint(names)
    assert "odds_implied_home" in names
    assert "home_xg_for_5" in names
    without = set(model_feature_names(use_xg=False, use_odds=False))
    assert "odds_implied_home" not in without
    assert "home_xg_for_5" not in without


def test_rolling_features_ignore_the_match_itself_same_day_results_and_the_future():
    rows = [
        _match(1, "2024-08-01 18:00:00", 1, 2, 1, 0, home_xg=1.1, away_xg=0.4),
        _match(2, "2024-08-01 20:00:00", 1, 3, 5, 0, home_xg=4.0, away_xg=0.1),
        _match(3, "2024-08-08 20:00:00", 1, 2, 0, 2, home_xg=0.2, away_xg=1.8),
        _match(4, "2024-08-15 20:00:00", 2, 3, 9, 0, home_xg=3.0, away_xg=0.2),
    ]
    features = compute_rolling_features(_frame(rows), use_xg=True).set_index("fixture_id")

    # First match has no history. Its own score and xG must not show up.
    assert math.isnan(features.loc[1, "home_gf_5"])
    assert math.isnan(features.loc[1, "home_xg_for_5"])
    assert features.loc[1, "home_matches"] == 0

    # Later kickoff on the same UTC date still cannot see the earlier result.
    assert math.isnan(features.loc[2, "home_gf_5"])
    assert math.isnan(features.loc[2, "home_xg_for_5"])
    assert features.loc[2, "home_matches"] == 0

    # Next week sees only the previous date: team 1 scored 1 and 5, so the mean is 3.
    assert features.loc[3, "home_gf_5"] == pytest.approx(3.0)
    assert features.loc[3, "home_xg_for_5"] == pytest.approx((1.1 + 4.0) / 2)
    assert features.loc[3, "home_matches"] == 2
    assert 6 < features.loc[3, "home_rest_days"] < 8

    # A future 9-0 does not change any earlier row.
    polluted = [dict(row) for row in rows]
    polluted[-1]["home_goals_ft"] = 0
    polluted[-1]["away_goals_ft"] = 8
    polluted[-1]["home_xg"] = 0.01
    again = compute_rolling_features(_frame(polluted), use_xg=True).set_index("fixture_id")
    for fixture_id in (1, 2, 3):
        for column in ("home_gf_5", "home_ga_5", "home_xg_for_5", "away_gf_5", "home_matches"):
            left = features.loc[fixture_id, column]
            right = again.loc[fixture_id, column]
            assert (math.isnan(left) and math.isnan(right)) or left == pytest.approx(right)

    # The match being featurized can change its own score without changing its features.
    own = [dict(row) for row in rows]
    own[2]["home_goals_ft"] = 7
    own[2]["home_xg"] = 6.0
    own_features = compute_rolling_features(_frame(own), use_xg=True).set_index("fixture_id")
    assert own_features.loc[3, "home_gf_5"] == pytest.approx(features.loc[3, "home_gf_5"])
    assert own_features.loc[3, "home_xg_for_5"] == pytest.approx(features.loc[3, "home_xg_for_5"])


def test_odds_features_use_only_this_rows_prices():
    row = pd.Series({"odds_home": 2.0, "odds_draw": 4.0, "odds_away": 4.0, "home_goals_ft": 5})
    implied = odds_features(row)
    assert implied["odds_implied_home"] == pytest.approx(0.5)
    assert implied["odds_implied_draw"] == pytest.approx(0.25)
    assert implied["odds_implied_away"] == pytest.approx(0.25)
    assert abs(sum(implied.values()) - 1) < 1e-12
    other = pd.Series({"odds_home": 9.0, "odds_draw": 9.0, "odds_away": 9.0})
    assert odds_features(other)["odds_implied_home"] != pytest.approx(implied["odds_implied_home"])
    missing = pd.Series({"home_goals_ft": 1})
    assert math.isnan(odds_features(missing)["odds_implied_home"])
    assert odds_column_status(pd.DataFrame([{"odds_home": 2.0, "odds_draw": 3.0, "odds_away": 3.5}])) == "present"
    assert odds_column_status(pd.DataFrame([{"home_goals_ft": 1}])) == "absent"
    assert xg_column_status(pd.DataFrame([{"home_xg": None, "away_xg": None}])) == "empty"


def test_dixon_coles_features_ignore_matches_after_the_cutoff():
    frame = fixtures_to_frame(generate_synthetic_fixtures(seed=11, today=pd.Timestamp("2026-09-25").date()))
    finished = frame[frame["status"] == "finished"].reset_index(drop=True)
    cutoff = finished.loc[40, "starting_at"]
    history = finished[finished["starting_at"] < cutoff]
    future = finished[finished["starting_at"] >= cutoff].iloc[0].copy()
    future["home_goals_ft"] = 12
    future["away_goals_ft"] = 0
    future["home_xg"] = 8.0
    polluted = pd.concat([history, pd.DataFrame([future])], ignore_index=True)
    config = dict(xi=0.001, max_goals=8, l2=1e-3, max_iter=30, min_matches=8, prior_strength=12, kind="ft")
    clean = fit_dixon_coles(history, as_of=cutoff, home_col="home_goals_ft", away_col="away_goals_ft", **config)
    dirty = fit_dixon_coles(polluted, as_of=cutoff, home_col="home_goals_ft", away_col="away_goals_ft", **config)
    home_id = int(future["home_team_id"])
    away_id = int(future["away_team_id"])
    # Wrap the single-market fit so the feature helper can read both markets.
    # Half-time is a scaled copy: still determined only by the pre-cutoff fit.
    clean_model = ScorelineModel(ft=clean, ht=clean.scaled_copy(0.45, kind="ht"), ht_source="scaled_full_time")
    dirty_model = ScorelineModel(ft=dirty, ht=dirty.scaled_copy(0.45, kind="ht"), ht_source="scaled_full_time")
    clean_features = dixon_coles_features(clean_model, home_id, away_id)
    dirty_features = dixon_coles_features(dirty_model, home_id, away_id)
    for name in clean_features:
        assert clean_features[name] == pytest.approx(dirty_features[name])
    assert clean_features["dc_p_home"] + clean_features["dc_p_draw"] + clean_features["dc_p_away"] == pytest.approx(1)


def test_early_stopping_validation_is_the_tail_of_the_training_window_only():
    rng = np.random.default_rng(1)
    rows = []
    for index in range(100):
        rows.append(
            {
                "fixture_id": 1000 + index,
                "kickoff": pd.Timestamp("2024-01-01", tz="UTC") + pd.Timedelta(days=index),
                "y_ft": int(index % 3),
                "signal": float(index % 3),
                "noise": float(rng.normal()),
            }
        )
    frame = pd.DataFrame(rows)
    config = XGBoostConfig(n_estimators=30, early_stopping_rounds=5, min_val_rows=24, min_train_rows=40)
    assert validation_size(len(frame), config) == 24
    first = fit_outcome_model(frame, ["signal", "noise"], "y_ft", config)
    second = fit_outcome_model(frame, ["signal", "noise"], "y_ft", config)
    assert first.early_stopping
    assert first.n_val == 24
    assert first.val_fixture_ids == list(range(1000 + 76, 1100))
    assert first.n_train + first.n_val == 100
    left = first.predict(frame.tail(5))
    right = second.predict(frame.tail(5))
    assert np.allclose(left, right)
    assert np.allclose(left.sum(axis=1), 1.0)
    # A match that was never passed in cannot be part of the validation slice.
    assert 999 not in first.val_fixture_ids


def test_paired_interval_contains_the_resampled_mean():
    y = np.array([0, 1, 2, 0])
    challenger = np.array([[0.6, 0.2, 0.2], [0.2, 0.5, 0.3], [0.2, 0.2, 0.6], [0.5, 0.3, 0.2]])
    reference = np.tile([1 / 3, 1 / 3, 1 / 3], (4, 1))
    stats = paired_logloss_difference(y, challenger, reference, n_bootstrap=200, seed=7)
    assert stats["ci_low"] <= stats["mean_logloss_difference"] <= stats["ci_high"]
    assert stats["mean_logloss_difference"] == pytest.approx(
        float(np.mean(per_match_log_loss(y, challenger) - per_match_log_loss(y, reference)))
    )
    assert multiclass_log_loss(y, challenger) < multiclass_log_loss(y, reference)
    summary = calibration_summary(y, challenger, n_bins=5)
    assert {item["outcome"] for item in summary["per_class"]} == {"home", "draw", "away"}
    assert sum(item["n"] for item in summary["home_probability_bins"]) == len(y)


def test_compare_uses_the_same_matches_as_dixon_coles_backtest():
    frame = fixtures_to_frame(generate_synthetic_fixtures(seed=7, today=pd.Timestamp("2026-09-25").date()))
    # Stored xG and pre-match odds. The xG values are post-match on purpose:
    # features may use earlier matches only. Odds are known before kickoff.
    finished = frame["status"] == "finished"
    frame.loc[finished, "home_xg"] = 1.2
    frame.loc[finished, "away_xg"] = 0.9
    frame.loc[finished, "odds_home"] = 2.1
    frame.loc[finished, "odds_draw"] = 3.3
    frame.loc[finished, "odds_away"] = 3.4
    config = ModelConfig(xi=0.001, max_iter=25, min_train_matches=36, min_matches=8, min_ht_matches=20, l2=4.0)
    xgb_config = XGBoostConfig(n_estimators=40, early_stopping_rounds=5, fallback_estimators=15)
    baseline = walk_forward(frame, config)
    report = compare_models(frame, config, xgb_config)
    assert report.models["dixon_coles"].n == baseline.model.n == report.models["baseline"].n
    assert report.models["xgboost"].n == baseline.model.n
    assert report.models["xgboost_with_odds"].n == baseline.model.n
    assert report.models["dixon_coles"].ft_log_loss == pytest.approx(baseline.model.ft_log_loss)
    assert report.models["baseline"].ft_log_loss == pytest.approx(baseline.baseline.ft_log_loss)
    assert report.models["dixon_coles"].ft_brier == pytest.approx(baseline.model.ft_brier)
    assert report.xg_features_used
    assert report.odds_used
    assert report.odds_coverage == pytest.approx(1.0)
    for key in ("dixon_coles", "baseline", "xgboost", "xgboost_with_odds"):
        block = report.models[key]
        assert 0 < block.ft_log_loss < 2
        assert 0 <= block.ft_accuracy <= 1
        probs = report.calibration[key]["per_class"]
        assert len(probs) == 3
    paired = next(item for item in report.paired if item["challenger"] == "xgboost" and item["reference"] == "dixon_coles")
    assert paired["ci_low"] <= paired["mean_logloss_difference"] <= paired["ci_high"]
    assert paired["n"] == baseline.model.n
    payload = report.to_dict()
    assert payload["computed_from_stored_matches"] is True
    assert payload["production_model"] == "dixon_coles"
    json.dumps(payload)


def test_odds_that_start_after_the_first_window_do_not_change_the_match_set():
    frame = fixtures_to_frame(generate_synthetic_fixtures(seed=5, today=pd.Timestamp("2026-09-25").date()))
    finished_index = frame.index[frame["status"] == "finished"]
    late = finished_index[-8:]
    frame.loc[late, "odds_home"] = 2.0
    frame.loc[late, "odds_draw"] = 3.2
    frame.loc[late, "odds_away"] = 3.6
    report = compare_models(
        frame,
        ModelConfig(xi=0.001, max_iter=20, min_train_matches=36, min_matches=8, min_ht_matches=20),
        XGBoostConfig(n_estimators=20, early_stopping_rounds=5, fallback_estimators=10),
    )
    assert "xgboost_with_odds" not in report.models
    assert report.odds_used is False
    assert any("最早一个评测日" in note for note in report.notes)
    assert report.models["xgboost"].n == report.models["dixon_coles"].n


def test_absent_odds_are_reported_instead_of_a_second_model():
    notes = _comparison_notes(
        xg_status="absent",
        odds_status="absent",
        use_odds=False,
        odds_present_in_first_window=False,
        xg_features_used=False,
    )
    text = "\n".join(notes)
    assert "含赔率的 XGBoost 没有运行" in text
    assert "没有 home_xg" in text


def test_store_preserves_optional_columns_and_drops_unknown_ones(tmp_path: Path):
    store = MatchStore(tmp_path)
    frame = fixtures_to_frame(generate_synthetic_fixtures(seed=1, today=pd.Timestamp("2026-09-25").date())).head(3).copy()
    frame["home_xg"] = [1.0, 0.5, 0.2]
    frame["away_xg"] = [0.4, 1.1, 0.3]
    frame["odds_home"] = [1.9, 2.4, 2.2]
    frame["odds_draw"] = [3.4, 3.1, 3.2]
    frame["odds_away"] = [4.2, 3.0, 3.3]
    frame["not_a_real_column"] = 1
    store.replace(frame)
    loaded = store.load()
    assert "not_a_real_column" not in loaded.columns
    assert loaded["home_xg"].tolist() == pytest.approx([1.0, 0.5, 0.2])
    assert loaded["odds_away"].tolist() == pytest.approx([4.2, 3.0, 3.3])


def test_compare_command_writes_the_file_offline(tmp_path: Path):
    frame = fixtures_to_frame(generate_synthetic_fixtures(seed=7, today=pd.Timestamp("2026-09-25").date()))
    store = MatchStore(tmp_path)
    store.replace(frame)
    env = os.environ.copy()
    env["LALIGA_DATA_DIR"] = str(tmp_path)
    env["PYTHONPATH"] = str(REPO)
    env.pop("SPORTMONKS_API_TOKEN", None)
    output = tmp_path / "predictions" / "model_comparison.json"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "laliga",
            "compare",
            "--data-dir",
            str(tmp_path),
            "--min-train",
            "36",
            "--max-iter",
            "20",
            "--output",
            str(output),
        ],
        cwd=REPO,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "XGBoost（无赔率）" in result.stdout
    assert "Dixon–Coles" in result.stdout
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["computed_from_stored_matches"] is True
    assert payload["models"]["dixon_coles"]["n"] == payload["models"]["xgboost"]["n"]
    assert "xgboost_with_odds" not in payload["models"]
    assert "含赔率的 XGBoost 没有运行" in result.stdout


def test_dashboard_shows_only_a_computed_comparison_file(tmp_path: Path):
    app = create_app(tmp_path, demo=False)
    client = app.test_client()
    empty = client.get("/backtest")
    assert "还没有回测" in empty.get_data(as_text=True)
    assert "1.2345" not in empty.get_data(as_text=True)
    assert "python -m laliga compare --min-train 320 --output data/predictions/model_comparison.json" in empty.get_data(as_text=True)

    path = tmp_path / "predictions" / "model_comparison.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "computed_from_stored_matches": True,
                "production_model": "dixon_coles",
                "n_folds": 3,
                "first_test_day": "2025-01-01",
                "last_test_day": "2025-02-01",
                "models": {
                    "dixon_coles": {"n": 12, "ft_log_loss": 1.2345, "ft_brier": 0.6, "ft_accuracy": 0.4, "ht_log_loss": 1.1},
                    "baseline": {"n": 12, "ft_log_loss": 1.1, "ft_brier": 0.66, "ft_accuracy": 0.33, "ht_log_loss": 1.09},
                    "xgboost": {"n": 12, "ft_log_loss": 1.05, "ft_brier": 0.62, "ft_accuracy": 0.42, "ht_log_loss": None},
                },
                "calibration": {
                    "xgboost": {
                        "per_class": [
                            {"outcome": "home", "mean_predicted": 0.45, "observed_frequency": 0.42, "n": 12}
                        ]
                    }
                },
                "paired": [
                    {
                        "label": "XGBoost（无赔率）− Dixon–Coles",
                        "mean_logloss_difference": -0.1845,
                        "ci_low": -0.3,
                        "ci_high": -0.05,
                        "n": 12,
                    }
                ],
                "notes": ["生产预测仍使用 Dixon–Coles。"],
                "odds_coverage": None,
            }
        ),
        encoding="utf-8",
    )
    page = client.get("/backtest").get_data(as_text=True)
    assert 'id="model-comparison"' in page
    assert "1.2345" in page
    assert "XGBoost（无赔率）" in page
    assert "-0.1845" in page

    demo = create_app(tmp_path, demo=True)
    demo_page = demo.test_client().get("/backtest").get_data(as_text=True)
    assert "合成示例数据" in demo_page
    assert "1.2345" not in demo_page

    path.write_text(json.dumps({"schema_version": 1, "models": {}}), encoding="utf-8")
    refused = client.get("/backtest").get_data(as_text=True)
    assert "1.2345" not in refused
    assert 'id="model-comparison"' not in refused


def test_fit_models_still_rejects_a_future_kickoff_as_training():
    """The production fit used by predict and backtest stays the pre-kickoff cut."""

    frame = fixtures_to_frame(generate_synthetic_fixtures(seed=3, today=pd.Timestamp("2026-09-25").date()))
    finished = frame[frame["status"] == "finished"]
    cutoff = finished["starting_at"].iloc[len(finished) // 2]
    model = fit_models(finished, cutoff, ModelConfig(max_iter=20, xi=0.001, min_ht_matches=8))
    assert pd.Timestamp(model.ft.trained_through) < pd.Timestamp(cutoff)
