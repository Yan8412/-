"""De-vig identities and ranked probability score."""

from __future__ import annotations

import numpy as np
import pytest

from laliga.model.devig import devig, raw_implied_from_decimal
from laliga.model.metrics import multiclass_rps, paired_logloss_difference


def test_proportional_is_the_normalised_raw_price_and_shin_matches_the_published_example():
    raw = np.array([0.40, 0.35, 0.30])
    priced = devig(raw)
    assert priced["proportional"][0] == pytest.approx(raw / raw.sum())
    assert priced["proportional"][0].sum() == pytest.approx(1.0)

    odds = np.array([2.6, 2.4, 4.3])
    shin = devig(raw_implied_from_decimal(odds))["shin"][0]
    assert shin == pytest.approx([0.3729941, 0.4047794, 0.2222265], abs=1e-6)
    assert shin.sum() == pytest.approx(1.0)


def test_power_sharpens_the_favourite_when_the_book_has_an_overround():
    odds = np.array([1.5, 4.0, 7.0])
    priced = devig(raw_implied_from_decimal(odds))
    assert priced["proportional"][0].sum() == pytest.approx(1.0)
    assert priced["power"][0, 0] > priced["proportional"][0, 0]


def test_additive_is_nan_when_a_longshot_would_go_non_positive():
    raw = np.array([[0.90, 0.05, 0.50]])
    additive = devig(raw)["additive"][0]
    assert np.isnan(additive).all()
    assert np.isfinite(devig(raw)["proportional"]).all()


def test_rps_is_zero_for_a_perfect_forecast_and_known_for_a_uniform_forecast():
    perfect = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    assert multiclass_rps(np.array([0, 1]), perfect) == pytest.approx(0.0)
    uniform = np.array([[1 / 3, 1 / 3, 1 / 3]])
    assert multiclass_rps(np.array([0]), uniform) == pytest.approx(5 / 18)
    assert multiclass_rps(np.array([1]), uniform) == pytest.approx(1 / 9)


def test_paired_bootstrap_rps_interval_contains_the_mean():
    y = np.array([0, 1, 2, 0, 2])
    left = np.array(
        [
            [0.6, 0.2, 0.2],
            [0.2, 0.5, 0.3],
            [0.2, 0.2, 0.6],
            [0.5, 0.3, 0.2],
            [0.3, 0.2, 0.5],
        ]
    )
    right = np.full((5, 3), 1 / 3)
    stats = paired_logloss_difference(y, left, right, n_bootstrap=400, seed=7)
    assert stats["rps_ci_low"] <= stats["mean_rps_difference"] <= stats["rps_ci_high"]
    assert stats["ci_low"] <= stats["mean_logloss_difference"] <= stats["ci_high"]
