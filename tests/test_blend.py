"""Walk-forward blends of Dixon–Coles and the de-vigged market.

The weight for a day is fit on earlier days only. Same-day and later results
must not move that day's probabilities.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from laliga.backtest import ComparisonReport, _attach_blends
from laliga.model.blend import blend_notes, fit_pool_weight, fit_stack, linear_pool, log_pool, stack_pool, walk_forward_blends
from laliga.model.metrics import multiclass_log_loss, summarize_rows
from laliga.report import format_comparison

DC_HOME = (0.8, 0.1, 0.1)
MARKET_HOME = (0.4, 0.3, 0.3)
DC_AWAY = (0.8, 0.1, 0.1)
MARKET_AWAY = (0.1, 0.1, 0.8)


def _add(rows: list[dict], day: str, count: int, y: int, dixon_coles: tuple, market: tuple, start_id: int) -> int:
    for offset in range(count):
        rows.append(
            {
                "fixture_id": start_id + offset,
                "match_day": day,
                "y_ft": y,
                "y_ht": None,
                "score": (1, 0),
                "dixon_coles_ft_probs": dixon_coles,
                "market_ft_probs": market,
            }
        )
    return start_id + count


def _leakage_rows(*, day2_y: int, day3_y: int) -> list[dict]:
    rows: list[dict] = []
    next_id = _add(rows, "2024-01-01", 40, 0, DC_HOME, MARKET_HOME, 1)
    next_id = _add(rows, "2024-01-02", 20, day2_y, DC_AWAY, MARKET_AWAY, next_id)
    _add(rows, "2024-01-03", 5, day3_y, (0.7, 0.2, 0.1), (0.4, 0.3, 0.3), next_id)
    return rows


def _day_probs(rows: list[dict], day: str) -> np.ndarray:
    chosen = [row for row in rows if row["match_day"] == day]
    return np.array([row["blend_linear_ft_probs"] for row in chosen], dtype=float)


def test_pool_endpoints_recover_each_input():
    dixon_coles = np.array([[2.0, 1.0, 1.0], [0.2, 0.3, 0.5]])
    market = np.array([[1.0, 1.0, 2.0], [0.5, 0.4, 0.1]])
    renorm_dc = [[0.5, 0.25, 0.25], [0.2, 0.3, 0.5]]
    renorm_market = [[0.25, 0.25, 0.5], [0.5, 0.4, 0.1]]
    assert np.allclose(linear_pool(dixon_coles, market, 1.0), renorm_dc)
    assert np.allclose(linear_pool(dixon_coles, market, 0.0), renorm_market)
    assert np.allclose(log_pool(dixon_coles, market, 1.0), renorm_dc)
    assert np.allclose(log_pool(dixon_coles, market, 0.0), renorm_market)
    assert np.allclose(stack_pool(dixon_coles, market, 1.0, 0.0), renorm_dc)
    assert np.allclose(stack_pool(dixon_coles, market, 0.0, 1.0), renorm_market)
    assert np.allclose(stack_pool(dixon_coles, market, 0.0, 0.0), np.full((2, 3), 1 / 3))


def test_tied_log_loss_keeps_the_market_weight():
    y = np.array([0, 1, 2, 0])
    probs = np.array([[0.5, 0.3, 0.2], [0.2, 0.5, 0.3], [0.2, 0.3, 0.5], [0.6, 0.2, 0.2]])
    assert fit_pool_weight(y, probs, probs, linear_pool) == 0.0
    assert fit_pool_weight(y, probs, probs, log_pool) == 0.0
    coef_a, coef_b = fit_stack(y, probs, probs)
    assert (coef_a, coef_b) != (0.0, 0.0)
    assert multiclass_log_loss(y, stack_pool(probs, probs, coef_a, coef_b)) <= multiclass_log_loss(y, probs) + 1e-12


def test_weight_uses_only_earlier_days():
    description = walk_forward_blends(_leakage_rows(day2_y=2, day3_y=0), min_history=40)
    assert description["available"] is True
    assert description["warmup_n"] == 40
    assert description["scored_n"] == 25
    assert description["linear"]["path"][0] == {"day": "2024-01-02", "n_history": 40, "weight": 1.0}
    assert description["linear"]["path"][1]["n_history"] == 60
    assert description["linear"]["path"][1]["weight"] < 1.0
    assert description["weight_fit"] == "walk_forward_previous_evaluation_days_only"

    base_rows = _leakage_rows(day2_y=2, day3_y=0)
    today_rows = _leakage_rows(day2_y=0, day3_y=0)
    future_rows = _leakage_rows(day2_y=2, day3_y=2)
    reverse_rows = list(reversed(_leakage_rows(day2_y=2, day3_y=0)))
    walk_forward_blends(base_rows, min_history=40)
    walk_forward_blends(today_rows, min_history=40)
    walk_forward_blends(future_rows, min_history=40)
    walk_forward_blends(reverse_rows, min_history=40)

    assert all("blend_linear_ft_probs" not in row for row in base_rows if row["match_day"] == "2024-01-01")
    assert np.allclose(_day_probs(base_rows, "2024-01-02"), np.array([DC_AWAY] * 20))
    assert {row["blend_linear_weight"] for row in base_rows if row["match_day"] == "2024-01-02"} == {1.0}
    # Changing this day's results, or a later day's, leaves this day's blend alone.
    assert np.allclose(_day_probs(base_rows, "2024-01-02"), _day_probs(today_rows, "2024-01-02"))
    assert np.allclose(_day_probs(base_rows, "2024-01-02"), _day_probs(future_rows, "2024-01-02"))
    assert np.allclose(_day_probs(base_rows, "2024-01-03"), _day_probs(future_rows, "2024-01-03"))
    # The next day does see the changed results.
    assert not np.allclose(_day_probs(base_rows, "2024-01-03"), _day_probs(today_rows, "2024-01-03"))
    assert np.allclose(_day_probs(base_rows, "2024-01-02"), _day_probs(reverse_rows, "2024-01-02"))
    assert np.allclose(_day_probs(base_rows, "2024-01-03"), _day_probs(reverse_rows, "2024-01-03"))


def test_oracle_weight_is_fit_on_scored_matches_only_and_labelled():
    rows: list[dict] = []
    next_id = _add(rows, "2024-02-01", 40, 0, DC_HOME, MARKET_HOME, 1)
    _add(rows, "2024-02-02", 10, 2, (0.8, 0.15, 0.05), (0.05, 0.05, 0.9), next_id)
    description = walk_forward_blends(rows, min_history=40)
    linear = description["linear"]
    assert linear["weight_on_dixon_coles"]["start"] == 1.0
    assert linear["weight_on_dixon_coles"]["median"] == 1.0
    assert linear["weight_on_dixon_coles"]["end"] == 1.0
    assert linear["oracle"]["in_sample"] is True
    assert linear["oracle"]["role"] == "hindsight_upper_bound"
    assert linear["oracle"]["weight_on_dixon_coles"] == 0.0
    assert linear["oracle"]["n"] == 10
    assert "样本内" in linear["oracle"]["label"]
    assert description["log"]["oracle"]["weight_on_dixon_coles"] == 0.0
    assert description["log"]["oracle"]["in_sample"] is True
    assert description["stack"]["oracle"]["in_sample"] is True
    assert description["stack"]["oracle"]["a"] == 0.0
    assert description["stack"]["oracle"]["b"] > 0
    text = "\n".join(blend_notes(description))
    assert "样本内上界" in text
    assert "不是走步结果" in text
    assert "hindsight_upper_bound" in json.dumps(description)


def test_devig_method_ignores_same_day_and_future_results():
    proportional = (0.55, 0.25, 0.20)
    power = (0.70, 0.18, 0.12)
    rows: list[dict] = []
    next_id = 1
    for offset in range(40):
        rows.append(
            {
                "fixture_id": next_id + offset,
                "match_day": "2024-05-01",
                "y_ft": 0,
                "dixon_coles_ft_probs": proportional,
                "market_ft_probs": proportional,
                "market_power_ft_probs": power,
            }
        )
    next_id = 41
    for offset in range(40):
        rows.append(
            {
                "fixture_id": next_id + offset,
                "match_day": "2024-05-02",
                "y_ft": 2,
                "dixon_coles_ft_probs": proportional,
                "market_ft_probs": proportional,
                "market_power_ft_probs": power,
            }
        )
    description = walk_forward_blends(rows, min_history=40)
    today = [row for row in rows if row["match_day"] == "2024-05-02"]
    assert {row["blend_devig_method"] for row in today} == {"power"}
    assert description["devig"]["path"][0]["method"] == "power"
    assert description["devig"]["oracle_repeats_walk_forward_method"] is True
    assert description["linear"]["oracle"]["weight_on_dixon_coles"] == 1.0
    assert all(row["market_ft_probs"] == proportional for row in today)
    assert np.allclose(_matrix_key(today, "blend_market_ft_probs"), np.array([power] * 40))

    leaked = [dict(row) for row in rows]
    for row in leaked:
        if row["match_day"] == "2024-05-02":
            row["y_ft"] = 0
    future = [dict(row) for row in rows]
    future.append(
        {
            "fixture_id": 900,
            "match_day": "2024-05-03",
            "y_ft": 0,
            "dixon_coles_ft_probs": proportional,
            "market_ft_probs": proportional,
            "market_power_ft_probs": power,
        }
    )
    walk_forward_blends(leaked, min_history=40)
    walk_forward_blends(future, min_history=40)
    assert {row["blend_devig_method"] for row in leaked if row["match_day"] == "2024-05-02"} == {"power"}
    assert np.allclose(
        _matrix_key([row for row in rows if row["match_day"] == "2024-05-02"], "blend_linear_ft_probs"),
        _matrix_key([row for row in leaked if row["match_day"] == "2024-05-02"], "blend_linear_ft_probs"),
    )
    assert np.allclose(
        _matrix_key([row for row in rows if row["match_day"] == "2024-05-02"], "blend_linear_ft_probs"),
        _matrix_key([row for row in future if row["match_day"] == "2024-05-02"], "blend_linear_ft_probs"),
    )


def _matrix_key(rows: list[dict], key: str) -> np.ndarray:
    return np.array([row[key] for row in rows], dtype=float)


def test_short_history_scores_nothing():
    rows: list[dict] = []
    _add(rows, "2024-03-01", 30, 0, DC_HOME, MARKET_HOME, 1)
    description = walk_forward_blends(rows, min_history=60)
    assert description["available"] is False
    assert description["eligible_n"] == 30
    assert description["scored_n"] == 0
    assert "60" in description["reason"]
    assert all("blend_linear_ft_probs" not in row for row in rows)

    empty = walk_forward_blends(
        [{"fixture_id": 1, "match_day": "2024-03-01", "y_ft": 0, "dixon_coles_ft_probs": DC_HOME}]
    )
    assert empty["available"] is False
    assert empty["eligible_n"] == 0
    with pytest.raises(ValueError):
        walk_forward_blends(rows, min_history=0)


def test_comparison_rows_share_the_post_warmup_sample():
    rows: list[dict] = []
    next_id = _add(rows, "2024-04-01", 60, 0, DC_HOME, MARKET_HOME, 1)
    _add(rows, "2024-04-02", 8, 2, DC_AWAY, MARKET_AWAY, next_id)
    rows.append(
        {
            "fixture_id": 900,
            "match_day": "2024-04-02",
            "y_ft": 1,
            "y_ht": None,
            "score": (1, 1),
            "dixon_coles_ft_probs": DC_HOME,
        }
    )
    models: dict = {}
    calibration: dict = {}
    paired: list = []
    notes: list = []
    description = _attach_blends(rows, models, calibration, paired, notes, seed=7)
    assert description["scored_n"] == 8
    assert models["blend_linear"].n == models["blend_log"].n == models["blend_stack"].n == 8
    assert models["dixon_coles_on_blend"].n == models["market_on_blend"].n == 8
    market_gap = next(item for item in paired if item["challenger"] == "blend_linear" and item["reference"] == "market")
    dixon_gap = next(item for item in paired if item["challenger"] == "blend_linear" and item["reference"] == "dixon_coles")
    assert market_gap["n"] == dixon_gap["n"] == 8
    assert market_gap["ci_low"] <= market_gap["mean_logloss_difference"] <= market_gap["ci_high"]
    assert market_gap["label"] == "线性混合 − 赛前赔率（去水位）"
    assert sum(item["n"] for item in calibration["blend_linear"]["home_probability_bins"]) == 8
    assert any("样本内上界" in note for note in notes)
    assert rows[-1].get("blend_linear_ft_probs") is None

    block = summarize_rows(rows, "dixon_coles")
    report = ComparisonReport(
        n_folds=2,
        first_test_day="2024-04-01",
        last_test_day="2024-04-02",
        min_train_matches=60,
        xg_status="absent",
        odds_status="present",
        odds_used=True,
        odds_coverage=1.0,
        xg_features_used=False,
        feature_names=[],
        odds_feature_names=[],
        notes=notes,
        models={"dixon_coles": block, "baseline": block, "xgboost": block, **models},
        calibration=calibration,
        paired=paired,
        xgboost={},
        blend=description,
    )
    rendered = format_comparison(report)
    assert "线性混合" in rendered
    assert "对数混合" in rendered
    assert "对数线性叠加" in rendered
    assert "Dixon–Coles（混合同一批）" in rendered
    assert "线性混合 − 赛前赔率（去水位）" in rendered
    json.dumps(report.to_dict())


def test_power_blend_pairs_are_stored_once():
    proportional = (0.55, 0.25, 0.20)
    power = (0.70, 0.18, 0.12)
    rows = []
    for offset in range(60):
        rows.append(
            {
                "fixture_id": offset + 1,
                "match_day": "2024-05-01",
                "y_ft": 0,
                "dixon_coles_ft_probs": proportional,
                "market_ft_probs": proportional,
                "market_power_ft_probs": power,
            }
        )
    for offset in range(8):
        rows.append(
            {
                "fixture_id": 100 + offset,
                "match_day": "2024-05-02",
                "y_ft": 2,
                "dixon_coles_ft_probs": proportional,
                "market_ft_probs": proportional,
                "market_power_ft_probs": power,
            }
        )
    paired: list[dict] = []
    description = _attach_blends(rows, {}, {}, paired, [], seed=7)
    assert description["market_prefix"] == "blend_market"
    labels = [item["label"] for item in paired]
    assert len(labels) == len(set(labels)) == 9
    assert labels.count("线性混合 − 走步所选去水位") == 1
    assert labels.count("对数混合 − 走步所选去水位") == 1
    assert labels.count("对数线性叠加 − 走步所选去水位") == 1
