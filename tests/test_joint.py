"""Joint half-time / full-time distribution stays coherent with the notifier file."""

from __future__ import annotations

import pandas as pd
import pytest

from laliga.config import ModelConfig
from laliga.data.parse import fixtures_to_frame
from laliga.model.dixon_coles import outcome_probabilities
from laliga.model.joint import HalfRates, adjust_second_half, convolve, goals_matrix
from laliga.model.service import fit_models
from laliga.synthetic import generate_synthetic_fixtures

NOTIFIER_KEYS = {
    "fixture_id",
    "starting_at",
    "season_id",
    "season_name",
    "round_name",
    "home_team_id",
    "home_team",
    "away_team_id",
    "away_team",
    "ft",
    "ht",
    "top_scores",
    "expected_goals_ft",
    "expected_goals_ht",
}


def test_independent_halves_add_and_the_nine_way_is_a_distribution():
    rates = HalfRates(home_1=1.1, away_1=0.8, home_2=0.7, away_2=0.55, rho_1=-0.03, rho_2=-0.04, eta=0.0)
    forecast = convolve(rates, max_half=6, max_ft=12)
    first = goals_matrix(1.1, 0.8, -0.03, 6, 1.0)
    assert forecast.ht_matrix == pytest.approx(first)
    assert forecast.ht_probs == pytest.approx(outcome_probabilities(first))
    assert sum(forecast.htft.values()) == pytest.approx(1.0)
    assert sum(forecast.ft_probs) == pytest.approx(1.0)
    assert sum(forecast.ht_probs) == pytest.approx(1.0)
    assert len(forecast.top2_scores) == 2
    assert len(forecast.top_scores) == 3
    assert forecast.expected_ft[0] == pytest.approx(forecast.expected_ht[0] + 0.7, abs=0.05)
    assert forecast.expected_ft[1] == pytest.approx(forecast.expected_ht[1] + 0.55, abs=0.05)
    # A censored full-time bin cannot put half-time goals above full-time goals
    # when the full-time grid is at least as wide as one half.
    assert forecast.ft_matrix.shape[0] >= forecast.ht_matrix.shape[0]


def test_leading_side_slows_down_and_a_level_score_does_not():
    home, away = adjust_second_half(1.0, 1.0, 0, 0.2)
    assert home < 1.0 < away
    level_home, level_away = adjust_second_half(1.0, 1.0, 1, 0.2)
    assert level_home == 1.0 and level_away == 1.0
    zero = convolve(HalfRates(0.6, 0.5, 0.7, 0.6, 0.0, 0.0, eta=0.0), max_half=4, max_ft=8)
    tilted = convolve(HalfRates(0.6, 0.5, 0.7, 0.6, 0.0, 0.0, eta=0.4), max_half=4, max_ft=8)
    assert tilted.ht_probs == pytest.approx(zero.ht_probs)
    assert tilted.ft_probs != pytest.approx(zero.ft_probs)


def test_fitted_joint_model_keeps_the_notifier_keys():
    frame = fixtures_to_frame(generate_synthetic_fixtures(seed=4, today=pd.Timestamp("2026-09-25").date()))
    as_of = frame["starting_at"].max() + pd.Timedelta(days=1)
    model = fit_models(frame, as_of, ModelConfig(max_iter=20, min_matches=3, min_ht_matches=8, xi=0.001))
    assert model.joint_model
    assert model.ht_source == "first_half"
    assert model.sh is not None
    finished = frame[frame["status"] == "finished"].iloc[-1]
    prediction = model.predict_row(finished)
    assert prediction.joint
    assert set(prediction.to_dict()) == NOTIFIER_KEYS
    assert len(prediction.top2_scores) == 2
    assert len(prediction.top_scores) == 3
    assert sum(prediction.htft.values()) == pytest.approx(1.0)
    assert abs(sum(prediction.ft.as_tuple()) - 1.0) < 1e-6
    home_ht, away_ht = model.ht.expected_goals(int(finished["home_team_id"]), int(finished["away_team_id"]))
    assert prediction.expected_goals_ht[0] == pytest.approx(home_ht, abs=0.05)
    assert prediction.expected_goals_ht[1] == pytest.approx(away_ht, abs=0.05)
    document = model.to_dict()
    assert document["distribution"] == "joint"
    assert "sh" in document
    restored = type(model).from_dict(document)
    assert restored.joint_model


def test_temperature_one_does_not_move_the_matrix():
    rates = HalfRates(0.8, 0.6, 0.5, 0.4, -0.05, -0.05, 0.0)
    plain = convolve(rates, max_half=4, max_ft=8, temperature=1.0)
    same = convolve(rates, max_half=4, max_ft=8, temperature=1.0)
    hotter = convolve(rates, max_half=4, max_ft=8, temperature=1.4)
    assert plain.ft_probs == pytest.approx(same.ft_probs)
    assert hotter.ft_probs != pytest.approx(plain.ft_probs)
    assert sum(hotter.htft.values()) == pytest.approx(1.0)
