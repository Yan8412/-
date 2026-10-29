import numpy as np
import pandas as pd
import pytest

from laliga.backtest import fit_ensemble, rows_from_document, walk_forward
from laliga.config import ModelConfig, PromotionConfig
from laliga.data.odds import parse_odds
from laliga.data.parse import fixtures_to_frame
from laliga.data.store import MatchStore
from laliga.model import promotion
from laliga.model.ensemble import EnsembleWeights, PoolWeights, _loss, fit_pool, log_pool, pool_grid
from laliga.synthetic import generate_synthetic_fixtures, generate_synthetic_odds

TODAY = pd.Timestamp("2026-09-25").date()
CONFIG = ModelConfig(xi=0.001, max_iter=60, min_train_matches=48, min_matches=8, min_ht_matches=20)


# --- fusion ------------------------------------------------------------------


def test_log_pool_endpoints_and_normalisation():
    football = np.array([0.5, 0.3, 0.2])
    market = np.array([0.6, 0.25, 0.15])
    assert np.allclose(log_pool(football, market, PoolWeights(1.0, 0.0)), football)
    assert np.allclose(log_pool(football, market, PoolWeights(0.0, 1.0)), market)
    pooled = log_pool(football, market, PoolWeights(0.5, 0.5))
    geometric = np.sqrt(football * market)
    assert np.allclose(pooled, geometric / geometric.sum())
    batch = log_pool(np.vstack([football, market]), np.vstack([market, football]), PoolWeights(0.7, 0.6))
    assert np.allclose(batch.sum(axis=1), 1.0)


def test_pool_grid_is_a_distribution():
    rng = np.random.default_rng(0)
    first = rng.random((11, 11))
    second = rng.random((11, 11))
    pooled = pool_grid(first / first.sum(), second / second.sum(), PoolWeights(0.4, 0.8))
    assert pooled.sum() == pytest.approx(1.0)
    assert (pooled > 0).all()


def test_pool_loss_gradient_matches_finite_differences():
    rng = np.random.default_rng(1)
    f = rng.dirichlet([3, 2, 2], size=40)
    m = rng.dirichlet([3, 2, 2], size=40)
    y = rng.integers(0, 3, size=40)
    theta = np.array([0.6, 0.9])
    _value, gradient = _loss(theta, np.log(f), np.log(m), y)
    step = 1e-6
    numeric = []
    for index in range(2):
        shift = np.zeros(2)
        shift[index] = step
        numeric.append((_loss(theta + shift, np.log(f), np.log(m), y)[0] - _loss(theta - shift, np.log(f), np.log(m), y)[0]) / (2 * step))
    assert np.allclose(gradient, numeric, atol=1e-6)


def test_fit_pool_leans_on_the_informative_source():
    rng = np.random.default_rng(2)
    truth = rng.dirichlet([4, 2, 3], size=600)
    y = np.array([rng.choice(3, p=row) for row in truth])
    noise = rng.dirichlet([1, 1, 1], size=600)
    weights = fit_pool(noise, truth, y, min_rows=40)
    assert weights.fitted
    assert weights.market > 0.8
    assert weights.football < 0.2
    pooled = log_pool(noise, truth, weights)
    chosen = pooled[np.arange(len(y)), y]
    assert weights.log_loss == pytest.approx(float(-np.log(chosen).mean()), rel=1e-6)


def test_fit_pool_needs_enough_rows():
    weights = fit_pool(np.full((5, 3), 1 / 3), np.full((5, 3), 1 / 3), [0, 1, 2, 0, 1], min_rows=40)
    assert (weights.football, weights.market, weights.fitted) == (0.5, 0.5, False)
    assert EnsembleWeights.from_dict(None).ft == PoolWeights()


# --- promotion ---------------------------------------------------------------


def _rows(n, *, market_ft, combined_ft, football_ft=(0.4, 0.3, 0.3), with_market=True, hit_market=False):
    rows = []
    for index in range(n):
        y = 0 if index % 4 else 1
        row = {
            "y_ft": y,
            "y_ht": y,
            "score": (1, 0) if y == 0 else (1, 1),
            "model_ft_probs": football_ft,
            "model_ht_probs": football_ft,
            "model_top3": [(2, 2), (3, 3), (4, 4)],
            "market_ft_probs": market_ft if with_market else None,
            "market_ht_probs": market_ft if with_market else None,
            "market_top3": [(1, 0), (1, 1), (0, 0)] if hit_market else [(2, 2), (3, 3), (4, 4)],
            "combined_ft_probs": combined_ft if with_market else None,
            "combined_ht_probs": combined_ft if with_market else None,
            "combined_top3": [(2, 2), (3, 3), (4, 4)],
        }
        rows.append(row)
    return rows


def test_clearly_better_market_is_promoted_and_persisted(tmp_path):
    rows = _rows(80, market_ft=(0.7, 0.2, 0.1), combined_ft=(0.55, 0.25, 0.2), hit_market=True)
    decision = promotion.decide(rows, None, PromotionConfig(min_matches=60, min_z=1.0))
    assert decision["n_common"] == 80
    ft = decision["markets"]["ft"]
    assert ft["previous"] == "football" and ft["track"] == "market" and ft["changed"]
    assert ft["metrics"]["market"]["log_loss"] < ft["metrics"]["football"]["log_loss"]
    assert "升级为纯市场" in ft["reason"]
    assert decision["markets"]["scores"]["track"] == "market"

    store = MatchStore(tmp_path)
    assert promotion.load_active(store) == {"ft": "football", "ht": "football", "scores": "football"}
    promotion.apply(store, decision)
    assert promotion.load_active(store)["ft"] == "market"
    again = promotion.decide(rows, promotion.load_active(store), PromotionConfig(min_matches=60))
    assert again["markets"]["ft"]["track"] == "market" and not again["markets"]["ft"]["changed"]
    promotion.apply(store, again)
    history = store.load_json(store.active_path)["history"]
    assert [entry["changed"] for entry in history] == [["ft", "ht", "scores"], []]


def test_small_or_noisy_edges_keep_the_incumbent():
    rows = _rows(30, market_ft=(0.7, 0.2, 0.1), combined_ft=(0.5, 0.3, 0.2))
    decision = promotion.decide(rows, None, PromotionConfig(min_matches=60))
    assert all(body["track"] == "football" and not body["changed"] for body in decision["markets"].values())
    assert "少于 60 场" in decision["markets"]["ft"]["reason"]

    rng = np.random.default_rng(3)
    noisy = []
    for _ in range(200):
        y = int(rng.integers(0, 3))
        base = rng.dirichlet([2, 2, 2])
        jitter = np.clip(base + rng.normal(0, 0.01, 3), 1e-3, None)
        jitter /= jitter.sum()
        noisy.append(
            {
                "y_ft": y,
                "y_ht": None,
                "score": (0, 0),
                "model_ft_probs": tuple(base),
                "model_ht_probs": tuple(base),
                "model_top3": [(0, 0)],
                "market_ft_probs": tuple(jitter),
                "market_ht_probs": tuple(jitter),
                "market_top3": [(0, 0)],
                "combined_ft_probs": tuple(base),
                "combined_ht_probs": tuple(base),
                "combined_top3": [(0, 0)],
            }
        )
    decision = promotion.decide(noisy, None, PromotionConfig(min_matches=60, min_z=3.0))
    assert decision["markets"]["ft"]["track"] == "football"
    assert decision["markets"]["scores"]["track"] == "football"


def test_matches_without_odds_are_not_compared():
    rows = _rows(100, market_ft=(0.7, 0.2, 0.1), combined_ft=(0.6, 0.2, 0.2), with_market=False)
    decision = promotion.decide(rows, {"ft": "market"}, PromotionConfig(min_matches=60))
    assert decision["n_common"] == 0
    assert decision["markets"]["ft"]["track"] == "market"
    assert decision["markets"]["ht"]["track"] == "football"


# --- three-track backtest ----------------------------------------------------


def _synthetic():
    raw = generate_synthetic_fixtures(seed=7, today=TODAY)
    return fixtures_to_frame(raw), parse_odds(generate_synthetic_odds(raw, seed=3))


def test_backtest_without_odds_scores_football_only():
    frame, _odds = _synthetic()
    report = walk_forward(frame, CONFIG)
    assert report.tracks == {}
    assert report.n_with_odds == 0
    assert all(body["track"] == "football" for body in report.promotion["markets"].values())
    assert all(row["market_ft_probs"] is None for row in report.rows)


def test_backtest_compares_three_tracks_on_the_same_matches():
    frame, odds = _synthetic()
    report = walk_forward(frame, CONFIG, odds, promotion=PromotionConfig(min_matches=20, min_ensemble_rows=10))
    assert report.n_with_odds == report.model.n
    assert set(report.tracks) == {"football", "market", "combined", "baseline"}
    assert {block.n for block in report.tracks.values()} == {report.n_with_odds}
    assert report.tracks["market"].ft_log_loss < report.tracks["baseline"].ft_log_loss
    assert report.tracks["combined"].ft_log_loss < report.tracks["baseline"].ft_log_loss
    for row in report.rows:
        assert sum(row["combined_ft_probs"]) == pytest.approx(1.0)
        assert row["market_sources"]["ft"] == "odds_fulltime_result"
    assert report.ensemble.ft.fitted

    document = report.to_dict()
    assert document["n_common"] == report.n_with_odds
    restored = rows_from_document(document)
    assert promotion.decide(restored, None, PromotionConfig(min_matches=20))["markets"] == {
        key: {**value} for key, value in promotion.decide(report.rows, None, PromotionConfig(min_matches=20))["markets"].items()
    }


def test_combined_weights_only_see_earlier_days():
    frame, odds = _synthetic()
    config = PromotionConfig(min_matches=20, min_ensemble_rows=10)
    report = walk_forward(frame, CONFIG, odds, promotion=config)
    days = sorted({pd.Timestamp(row["starting_at"]).floor("D") for row in report.rows})
    for day in days[:4]:
        earlier = [row for row in report.rows if pd.Timestamp(row["starting_at"]).floor("D") < day]
        today = [row for row in report.rows if pd.Timestamp(row["starting_at"]).floor("D") == day]
        weights = fit_ensemble(earlier, config.min_ensemble_rows)
        for row in today:
            expected = log_pool(row["model_ft_probs"], row["market_ft_probs"], weights.ft)
            assert np.allclose(row["combined_ft_probs"], expected)
    first_day = [row for row in report.rows if pd.Timestamp(row["starting_at"]).floor("D") == days[0]]
    for row in first_day:
        assert np.allclose(row["combined_ft_probs"], log_pool(row["model_ft_probs"], row["market_ft_probs"], PoolWeights()))
