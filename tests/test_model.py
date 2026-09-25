import numpy as np
import pandas as pd
import pytest

from laliga.config import ModelConfig
from laliga.data.parse import fixtures_to_frame
from laliga.model.dixon_coles import (
    FittedModel,
    TeamStrength,
    dixon_coles_tau,
    fit_dixon_coles,
    outcome_probabilities,
    score_matrix,
    shrink_strength,
    top_scorelines,
)
from laliga.model.dixon_coles import _objective
from laliga.model.metrics import iter_evaluation_folds
from laliga.model.service import fit_models
from laliga.synthetic import generate_synthetic_fixtures, single_round_robin


def _manual_model(**overrides) -> FittedModel:
    teams = {
        1: TeamStrength(1, "强队", attack=0.4, defence=0.3, matches=30),
        2: TeamStrength(2, "弱队", attack=-0.4, defence=-0.3, matches=30),
    }
    payload = dict(
        kind="ft",
        mu=0.1,
        home_adv=0.25,
        rho=-0.1,
        xi=0.0025,
        max_goals=8,
        trained_through="2024-01-01T00:00:00+00:00",
        n_matches=30,
        teams=teams,
        promoted_attack_prior=-0.2,
        promoted_defence_prior=-0.15,
        min_matches=10,
        prior_strength=12,
    )
    payload.update(overrides)
    return FittedModel(**payload)


def test_tau_matches_dixon_coles_cases():
    tau = dixon_coles_tau(
        np.array([0, 0, 1, 1, 2]),
        np.array([0, 1, 0, 1, 2]),
        np.full(5, 1.5),
        np.full(5, 1.2),
        -0.1,
    )
    assert tau[0] == pytest.approx(1 - 1.5 * 1.2 * -0.1)
    assert tau[1] == pytest.approx(1 + 1.5 * -0.1)
    assert tau[2] == pytest.approx(1 + 1.2 * -0.1)
    assert tau[3] == pytest.approx(1 - -0.1)
    assert tau[4] == pytest.approx(1.0)


def test_score_matrix_is_a_distribution_consistent_with_top_scores():
    model = _manual_model(rho=0.0)
    matrix = model.score_matrix(1, 2)
    assert matrix.shape == (9, 9)
    assert matrix.sum() == pytest.approx(1.0)
    home, draw, away = outcome_probabilities(matrix)
    assert home + draw + away == pytest.approx(1.0)
    assert home > away
    top = top_scorelines(matrix, 3)
    assert top[0][2] >= top[1][2] >= top[2][2]
    flat = matrix.ravel()
    assert top[0][2] == pytest.approx(flat.max())
    lam_home, lam_away = model.expected_goals(1, 2)
    assert lam_home == pytest.approx(np.exp(0.1 + 0.25 + 0.4 - (-0.3)))
    assert lam_away == pytest.approx(np.exp(0.1 + -0.4 - 0.3))


def test_negative_rho_increases_low_score_mass():
    base = _manual_model(rho=0.0).score_matrix(1, 2)
    corrected = _manual_model(rho=-0.1).score_matrix(1, 2)
    assert corrected[0, 0] > base[0, 0]
    assert corrected[1, 1] > base[1, 1]


def test_unknown_team_uses_promoted_prior():
    model = _manual_model()
    promoted_home, average_away = model.expected_goals(99, 1)
    # Prior attack -0.2 at home against team 1 defence 0.3.
    assert promoted_home == pytest.approx(np.exp(model.mu + model.home_adv - 0.2 - 0.3))
    # Team 1 attack 0.4 against promoted defence -0.15.
    assert average_away == pytest.approx(np.exp(model.mu + 0.4 - (-0.15)))
    matrix = score_matrix(promoted_home, average_away, model.rho, model.max_goals)
    assert matrix.sum() == pytest.approx(1.0)


def test_shrink_strength_keeps_established_teams():
    assert shrink_strength(0.8, 2, prior=-0.2, prior_strength=12, min_matches=10) == pytest.approx((2 * 0.8 + 12 * -0.2) / 14)
    assert shrink_strength(0.8, 0, prior=-0.2, prior_strength=12, min_matches=10) == pytest.approx(-0.2)
    assert shrink_strength(0.8, 10, prior=-0.2, prior_strength=12, min_matches=10) == pytest.approx(0.8)


def test_gradient_matches_finite_differences():
    rng = np.random.default_rng(0)
    home_goals = rng.integers(0, 5, size=16)
    away_goals = rng.integers(0, 5, size=16)
    home_index = rng.integers(0, 3, size=16)
    away_index = (home_index + rng.integers(1, 3, size=16)) % 3
    weights = rng.uniform(0.4, 1.0, size=16)
    objective = _objective(home_goals, away_goals, home_index, away_index, weights, n_teams=3, l2=1e-3)
    theta = np.array([0.1, 0.2, -0.04, 0.15, -0.05, 0.1, -0.02], dtype=float)
    _, gradient = objective(theta)
    numerical = np.zeros_like(theta)
    step = 1e-6
    for index in range(len(theta)):
        forward = theta.copy()
        backward = theta.copy()
        forward[index] += step
        backward[index] -= step
        numerical[index] = (objective(forward)[0] - objective(backward)[0]) / (2 * step)
    assert np.allclose(gradient, numerical, rtol=1e-4, atol=1e-5)


def test_future_match_does_not_change_the_fit():
    frame = fixtures_to_frame(generate_synthetic_fixtures(seed=11, today=pd.Timestamp("2026-09-25").date()))
    finished = frame[frame["status"] == "finished"].reset_index(drop=True)
    cutoff = finished.loc[len(finished) // 2, "starting_at"]
    history = finished[finished["starting_at"] < cutoff]
    future = finished[finished["starting_at"] >= cutoff].iloc[0].copy()
    future["home_goals_ft"] = 12
    future["away_goals_ft"] = 0
    future["home_goals_ht"] = 6
    future["away_goals_ht"] = 0
    polluted = pd.concat([history, pd.DataFrame([future])], ignore_index=True)
    config = dict(xi=0.001, max_goals=8, l2=1e-3, max_iter=40, min_matches=8, prior_strength=12, kind="ft")
    clean = fit_dixon_coles(history, as_of=cutoff, home_col="home_goals_ft", away_col="away_goals_ft", **config)
    dirty = fit_dixon_coles(polluted, as_of=cutoff, home_col="home_goals_ft", away_col="away_goals_ft", **config)
    assert clean.mu == pytest.approx(dirty.mu)
    assert clean.home_adv == pytest.approx(dirty.home_adv)
    assert clean.teams[1].attack == pytest.approx(dirty.teams[1].attack)


def test_fit_recovers_home_advantage_and_ranks_attack():
    frame = fixtures_to_frame(generate_synthetic_fixtures(seed=7, today=pd.Timestamp("2026-09-25").date()))
    finished = frame[frame["status"] == "finished"]
    as_of = finished["starting_at"].max() + pd.Timedelta(hours=1)
    model = fit_dixon_coles(
        finished,
        as_of=as_of,
        home_col="home_goals_ft",
        away_col="away_goals_ft",
        kind="ft",
        xi=0.001,
        max_goals=8,
        l2=1e-4,
        max_iter=80,
        min_matches=6,
        prior_strength=8,
    )
    assert model.home_adv > 0
    assert model.teams[1].attack > model.teams[5].attack
    assert model.n_matches == len(finished)


def test_missing_half_time_scores_scale_the_full_time_model():
    frame = fixtures_to_frame(generate_synthetic_fixtures(seed=3, today=pd.Timestamp("2026-09-25").date()))
    frame = frame.copy()
    frame["home_goals_ht"] = pd.NA
    frame["away_goals_ht"] = pd.NA
    as_of = frame["starting_at"].max() + pd.Timedelta(days=1)
    fitted = fit_models(frame, as_of, ModelConfig(max_iter=30, min_ht_matches=30, xi=0.001))
    assert fitted.ht_source == "scaled_full_time"
    assert fitted.ht.derived
    home_ft, away_ft = fitted.ft.expected_goals(1, 2)
    home_ht, away_ht = fitted.ht.expected_goals(1, 2)
    assert home_ht == pytest.approx(home_ft * 0.45)
    assert away_ht == pytest.approx(away_ft * 0.45)


def test_round_robin_pairs_everyone_once():
    rounds = single_round_robin([1, 2, 3, 4, 5, 6])
    assert len(rounds) == 5
    seen = set()
    for pairs in rounds:
        used = []
        for home, away in pairs:
            used.extend([home, away])
            seen.add(tuple(sorted((home, away))))
        assert sorted(used) == [1, 2, 3, 4, 5, 6]
    assert len(seen) == 15


def test_folds_never_include_the_test_day():
    frame = fixtures_to_frame(generate_synthetic_fixtures(seed=1, today=pd.Timestamp("2026-09-25").date()))
    finished = frame[frame["status"] == "finished"]
    folds = list(iter_evaluation_folds(finished, min_train_matches=9))
    assert folds
    for _day, train, test in folds:
        assert train["starting_at"].max() < test["starting_at"].min()
        assert set(train["fixture_id"]).isdisjoint(set(test["fixture_id"]))
