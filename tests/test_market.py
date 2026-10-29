import json

import numpy as np
import pandas as pd
import pytest

from laliga.data.client import SportMonksClient, SportMonksError
from laliga.data.odds import ODDS_COLUMNS, OddsUnavailable, fetch_fixture_odds, fetch_odds, parse_odds
from laliga.data.store import MatchStore
from laliga.model.dixon_coles import outcome_probabilities, score_matrix
from laliga.model.market import (
    SOURCE_CS_ODDS,
    SOURCE_DERIVED,
    SOURCE_FT_ODDS,
    SOURCE_HT_ODDS,
    SOURCE_HTFT_ODDS,
    classify_htft,
    classify_outcome,
    devig_proportional,
    devig_shin,
    implied_rates,
    market_forecast,
    parse_correct_score,
)


def _odd(market, bookmaker, label, value, name=None, fixture=10):
    return {
        "id": hash((market, bookmaker, label, name)) % 10_000,
        "fixture_id": fixture,
        "market_id": market,
        "bookmaker_id": bookmaker,
        "label": label,
        "name": label if name is None else name,
        "value": f"{value:.2f}",
        "winning": False,
        "latest_bookmaker_update": "2026-09-01 10:00:00",
    }


def _book(probabilities, margin=0.06):
    return [round(1.0 / (p * (1 + margin)), 2) for p in probabilities]


# --- overround removal -------------------------------------------------------


def test_proportional_devig_removes_overround():
    prices = [1.80, 3.60, 4.50]
    implied = 1 / np.array(prices)
    assert implied.sum() > 1.0
    probabilities = devig_proportional(prices)
    assert probabilities.sum() == pytest.approx(1.0)
    assert np.allclose(probabilities, implied / implied.sum())


def test_shin_sums_to_one_and_moves_margin_onto_longshots():
    prices = [1.25, 6.00, 12.00]
    shin, z = devig_shin(prices)
    proportional = devig_proportional(prices)
    assert shin.sum() == pytest.approx(1.0)
    assert 0 < z < 0.2
    assert shin[0] > proportional[0]
    assert shin[2] < proportional[2]


def test_shin_on_a_fair_book_changes_nothing():
    fair = [2.0, 4.0, 4.0]
    shin, z = devig_shin(fair)
    assert z == 0.0
    assert np.allclose(shin, [0.5, 0.25, 0.25])


@pytest.mark.parametrize("prices", [[1.0, 3.0, 4.0], [0.5, 3.0, 4.0], [2.0], [float("nan"), 3.0, 3.0]])
def test_devig_rejects_unusable_prices(prices):
    with pytest.raises(ValueError):
        devig_proportional(prices)


# --- selection labels --------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [("1", 0), ("Home", 0), ("X", 1), ("draw", 1), ("2", 2), ("Away", 2), ("Real Madrid", 0), ("FC Barcelona", 2), ("Over", None)],
)
def test_classify_outcome(text, expected):
    assert classify_outcome(text, "Real Madrid", "FC Barcelona") == expected


def test_classify_half_time_full_time():
    assert classify_htft("1/X", "") == (0, 1)
    assert classify_htft("", "Draw/Away") == (1, 2)
    assert classify_htft("Over 2.5", "") is None


@pytest.mark.parametrize(
    "label, name, expected",
    [
        ("2:1", "2:1", (2, 1)),
        ("0-3", "0-3", (0, 3)),
        ("2-1", "Away", (1, 2)),
        ("2-1", "Home", (2, 1)),
        ("1-1", "Draw", (1, 1)),
        ("2-0", "FC Barcelona", (0, 2)),
        ("Home", "3-1", (3, 1)),
        ("Any Other Score", "Any Other Score", "other"),
        ("Over 2.5", "Over", None),
    ],
)
def test_parse_correct_score(label, name, expected):
    assert parse_correct_score(label, name, "Real Madrid", "FC Barcelona") == expected


# --- raw odds parsing --------------------------------------------------------


def test_parse_odds_keeps_known_markets_and_never_stores_winning():
    rows = [
        _odd(1, 2, "1", 1.9),
        {**_odd(1, 2, "X", 3.4), "value": "n/a", "dp3": "3.400"},
        _odd(1, 2, "2", 1.0),
        _odd(80, 2, "Over", 1.8),
        {"market_id": 1},
    ]
    frame = parse_odds(rows)
    assert list(frame.columns) == ODDS_COLUMNS
    assert "winning" not in frame.columns
    assert frame["market_id"].tolist() == [1, 1]
    assert frame["value"].tolist() == [1.9, 3.4]


# --- assembling forecasts ----------------------------------------------------

TRUE_FT = score_matrix(1.6, 0.9, -0.05, 10)
TRUE_HT = score_matrix(0.72, 0.45, -0.05, 10)


def _full_book(bookmaker=2, *, ht=True, htft=False, correct_score=True, winner_first=False, fixture=10):
    rows = _book_rows(bookmaker, ht=ht, htft=htft, correct_score=correct_score, winner_first=winner_first)
    return [{**row, "fixture_id": fixture} for row in rows]


def _book_rows(bookmaker, *, ht, htft, correct_score, winner_first):
    ft = outcome_probabilities(TRUE_FT)
    rows = [_odd(1, bookmaker, label, price) for label, price in zip(("1", "X", "2"), _book(ft), strict=True)]
    if ht:
        half = outcome_probabilities(TRUE_HT)
        rows += [_odd(31, bookmaker, label, price) for label, price in zip(("Home", "Draw", "Away"), _book(half), strict=True)]
    if htft:
        for first, a in zip(("1", "X", "2"), outcome_probabilities(TRUE_HT), strict=True):
            for second, b in zip(("1", "X", "2"), ft, strict=True):
                rows.append(_odd(29, bookmaker, f"{first}/{second}", 1 / (a * b * 1.1)))
    if correct_score:
        listed = 0.0
        for home in range(5):
            for away in range(5):
                probability = float(TRUE_FT[home, away])
                listed += probability
                if winner_first:
                    side = "Home" if home > away else "Away" if home < away else "Draw"
                    rows.append(_odd(57, bookmaker, f"{max(home, away)}-{min(home, away)}", 1 / (probability * 1.15), name=side))
                else:
                    rows.append(_odd(57, bookmaker, f"{home}:{away}", 1 / (probability * 1.15)))
        if not winner_first:
            rows.append(_odd(57, bookmaker, "Any Other Score", 1 / ((1 - listed) * 1.15)))
    return rows


def test_market_forecast_uses_every_market_when_present():
    forecast = market_forecast(parse_odds(_full_book()), ht_ratio=0.45)
    assert forecast.sources == {"ft": SOURCE_FT_ODDS, "ht": SOURCE_HT_ODDS, "scores": SOURCE_CS_ODDS}
    assert sum(forecast.ft) == pytest.approx(1.0)
    assert sum(forecast.ht) == pytest.approx(1.0)
    assert forecast.matrix.sum() == pytest.approx(1.0)
    assert np.allclose(forecast.ft, outcome_probabilities(TRUE_FT), atol=0.02)
    assert np.allclose(forecast.ht, outcome_probabilities(TRUE_HT), atol=0.02)
    assert np.abs(forecast.matrix - TRUE_FT).max() < 0.01
    assert forecast.top_scores[0][:2] == np.unravel_index(np.argmax(TRUE_FT), TRUE_FT.shape)
    assert forecast.overround_ft == pytest.approx(0.06, abs=0.01)


def test_winner_first_correct_scores_match_home_away_labels():
    home_away = market_forecast(parse_odds(_full_book(2)), ht_ratio=0.45)
    winner_first = market_forecast(parse_odds(_full_book(34, winner_first=True)), ht_ratio=0.45)
    assert np.abs(home_away.matrix - winner_first.matrix).max() < 0.01
    assert home_away.top_scores[0][:2] == winner_first.top_scores[0][:2]


def test_half_time_falls_back_to_htft_then_to_derived():
    htft = market_forecast(parse_odds(_full_book(ht=False, htft=True)), ht_ratio=0.45)
    assert htft.sources["ht"] == SOURCE_HTFT_ODDS
    assert np.allclose(htft.ht, outcome_probabilities(TRUE_HT), atol=0.02)

    derived = market_forecast(parse_odds(_full_book(ht=False, correct_score=False)), ht_ratio=0.45)
    assert derived.sources == {"ft": SOURCE_FT_ODDS, "ht": SOURCE_DERIVED, "scores": SOURCE_DERIVED}
    assert sum(derived.ht) == pytest.approx(1.0)
    assert derived.ht[1] > derived.ft[1]
    assert len(derived.notes) == 2
    assert derived.bookmakers == {"ft": 1, "ht": 0, "scores": 0}


def test_no_fulltime_book_means_no_market_forecast():
    rows = [row for row in _full_book() if row["market_id"] != 1]
    assert market_forecast(parse_odds(rows)) is None
    incomplete = [row for row in _full_book() if not (row["market_id"] == 1 and row["label"] == "X")]
    assert market_forecast(parse_odds(incomplete)) is None
    assert market_forecast(parse_odds([])) is None


def test_bookmakers_are_averaged_after_their_own_devig():
    first = [_odd(1, 2, label, price) for label, price in zip(("1", "X", "2"), [2.0, 3.5, 4.0], strict=True)]
    second = [_odd(1, 34, label, price) for label, price in zip(("Home", "Draw", "Away"), [2.2, 3.3, 3.6], strict=True)]
    forecast = market_forecast(parse_odds(first + second), ht_ratio=0.45)
    expected = (devig_shin([2.0, 3.5, 4.0])[0] + devig_shin([2.2, 3.3, 3.6])[0]) / 2
    assert np.allclose(forecast.ft, expected / expected.sum())
    assert forecast.bookmakers["ft"] == 2


def test_implied_rates_reproduce_the_1x2_they_were_fitted_to():
    target = outcome_probabilities(score_matrix(1.9, 0.8, -0.05, 10))
    lam_home, lam_away = implied_rates(target, -0.05, 10)
    assert lam_home == pytest.approx(1.9, rel=0.02)
    assert lam_away == pytest.approx(0.8, rel=0.02)


# --- fetching ----------------------------------------------------------------


def _client(tmp_path, transport):
    return SportMonksClient("odds-token", tmp_path, transport=transport, sleep=lambda _s: None, min_interval=0, clock=lambda: 0.0)


def test_odds_request_shape_and_cache_without_token(tmp_path):
    calls = []

    def transport(url, params):
        calls.append((url, dict(params)))
        return 200, {}, {"data": _full_book()}

    client = _client(tmp_path / "cache", transport)
    rows = fetch_fixture_odds(client, 10)
    assert rows
    url, params = calls[0]
    assert url.endswith("/odds/pre-match/fixtures/10")
    assert params["filters"] == "markets:1,31,29,57"
    for path in (tmp_path / "cache").glob("*.json"):
        assert "odds-token" not in path.read_text(encoding="utf-8")


def test_missing_odds_subscription_is_a_clear_error(tmp_path):
    def transport(url, params):
        return 403, {}, {"message": "You don't have access to this endpoint on your subscription."}

    with pytest.raises(OddsUnavailable) as caught:
        fetch_fixture_odds(_client(tmp_path, transport), 10)
    assert "Odds add-on" in str(caught.value)
    assert "odds-token" not in str(caught.value)


def test_fixture_without_odds_is_empty_not_fake(tmp_path):
    def transport(url, params):
        if url.endswith("/11"):
            return 404, {}, {"message": "No result(s) found matching your request."}
        if url.endswith("/12"):
            return 200, {}, {"data": []}
        return 500, {}, {"message": "boom"}

    client = _client(tmp_path, transport)
    assert fetch_fixture_odds(client, 11) == []
    assert fetch_fixture_odds(client, 12) == []
    client.max_retries = 1
    with pytest.raises(SportMonksError):
        fetch_fixture_odds(client, 13)


def test_fetch_odds_refreshes_scheduled_and_reuses_cache_for_finished(tmp_path):
    calls = []

    def transport(url, params):
        fixture = int(url.rsplit("/", 1)[1])
        calls.append(fixture)
        return 200, {}, {"data": _full_book(fixture=fixture) if fixture != 3 else []}

    store = MatchStore(tmp_path)
    fixtures = pd.DataFrame(
        {
            "fixture_id": [1, 2, 3],
            "status": ["finished", "scheduled", "finished"],
            "starting_at": pd.to_datetime(["2026-09-01", "2026-09-30", "2026-09-02"], utc=True),
        }
    )
    client = _client(store.cache_dir, transport)
    summary = fetch_odds(client, store, fixtures)
    assert (summary.requested, summary.with_odds) == (3, 2)
    assert sorted(calls) == [1, 2, 3]
    stored = store.load_odds()
    assert set(stored["fixture_id"]) == {1, 2}

    calls.clear()
    fetch_odds(client, store, fixtures)
    assert calls == [2]
    assert len(store.load_odds()) == len(stored)


def test_cached_json_for_odds_is_plain_sportmonks_shape(tmp_path):
    def transport(url, params):
        return 200, {}, {"data": _full_book(), "rate_limit": {"remaining": 100}}

    fetch_fixture_odds(_client(tmp_path, transport), 10)
    cached = json.loads(next(tmp_path.glob("*.json")).read_text(encoding="utf-8"))
    assert cached["path"] == "odds/pre-match/fixtures/10"
    assert "api_token" not in cached["query"]
