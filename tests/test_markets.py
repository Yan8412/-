"""Backfill of pre-match 1X2 odds and team xG. Responses follow SportMonks v3."""

import pandas as pd
import pytest

from laliga.data.client import SportMonksClient
from laliga.data.fetch import FIXTURE_INCLUDES
from laliga.data.markets import fetch_stored_markets, format_market_fetch
from laliga.data.parse import aggregate_prematch_1x2
from laliga.data.store import MatchStore
from laliga.model.features import odds_features


def _client(tmp_path, transport):
    return SportMonksClient(
        "client-token",
        tmp_path / "cache",
        transport=transport,
        min_interval=0,
        sleep=lambda _seconds: None,
    )


def _row(fixture_id, status, kickoff, **extra):
    row = {
        "fixture_id": fixture_id,
        "league_id": 564,
        "season_id": 1,
        "season_name": "2024/2025",
        "round_id": 1,
        "round_name": "1",
        "starting_at": kickoff,
        "state_id": 5 if status == "finished" else 1,
        "state_name": "FT" if status == "finished" else "NS",
        "home_team_id": 10,
        "home_team_name": "Home",
        "away_team_id": 20,
        "away_team_name": "Away",
        "home_goals_ht": 0 if status == "finished" else None,
        "away_goals_ht": 0 if status == "finished" else None,
        "home_goals_ft": 1 if status == "finished" else None,
        "away_goals_ft": 0 if status == "finished" else None,
        "status": status,
    }
    row.update(extra)
    return row


def _quote(book, label, value, when):
    return {
        "id": book,
        "fixture_id": 1,
        "market_id": 1,
        "bookmaker_id": book,
        "label": label,
        "name": label,
        "value": value,
        "market_description": "Match Winner",
        "latest_bookmaker_update": when,
        "updated_at": when.replace(" ", "T") + "Z" if "T" not in when else when,
    }


def test_default_fetch_include_does_not_ask_for_odds_or_xg():
    assert FIXTURE_INCLUDES == "participants;scores;state;round;season"


def test_quote_at_or_after_kickoff_is_dropped_and_inplay_flag_is_ignored():
    kickoff = "2024-09-01 20:00:00"
    early = "2024-09-01 18:00:00"
    late = "2024-09-01 20:00:00"
    rows = [
        _quote(2, "Home", "2.00", early),
        _quote(2, "Draw", "3.40", early),
        _quote(2, "Away", "3.80", early),
        _quote(2, "Home", "1.20", late),
        _quote(2, "Draw", "1.20", late),
        _quote(2, "Away", "1.20", late),
        {**_quote(3, "Home", "9.00", early), "inplay": True},
        {**_quote(3, "Draw", "9.00", early), "inplay": True},
        {**_quote(3, "Away", "9.00", early), "inplay": True},
        {"market_id": 12, "bookmaker_id": 2, "label": "Home", "value": "1.01", "latest_bookmaker_update": early},
    ]
    prices = aggregate_prematch_1x2(rows, kickoff=kickoff)
    implied = odds_features(pd.Series(prices))
    assert implied["odds_implied_home"] == pytest.approx(prices["implied_home"])
    assert prices["odds_home"] == pytest.approx(1 / prices["implied_home"])
    raw = [1 / 2.0, 1 / 3.4, 1 / 3.8]
    total = sum(raw)
    assert prices["implied_home"] == pytest.approx(raw[0] / total)
    assert abs(sum(prices[name] for name in ("implied_home", "implied_draw", "implied_away")) - 1) < 1e-12


def test_expected_include_alias_is_team_xg_not_expected_goals_on_target(tmp_path):
    store = MatchStore(tmp_path)
    store.replace(pd.DataFrame([_row(7, "finished", "2024-09-01 20:00:00+00:00")]))
    calls = []

    def transport(url, params):
        calls.append(params.get("include"))
        if params.get("include") == "xGFixture":
            return 200, {}, {
                "data": [
                    {
                        "id": 7,
                        "expected": [
                            {"type_id": 5304, "location": "home", "participant_id": 10, "data": {"value": 1.25}},
                            {"type_id": 5304, "location": "away", "participant_id": 20, "data": {"value": 0.4}},
                            {"type_id": 5305, "location": "home", "participant_id": 10, "data": {"value": 8.0}},
                        ],
                    }
                ],
                "pagination": {"has_more": False},
            }
        return 200, {}, {"data": [{"id": 7, "odds": []}], "pagination": {"has_more": False}}

    result = fetch_stored_markets(_client(tmp_path, transport), store)
    loaded = store.load()
    assert loaded.loc[0, "home_xg"] == pytest.approx(1.25)
    assert loaded.loc[0, "away_xg"] == pytest.approx(0.4)
    assert loaded.loc[0, "home_xga"] == pytest.approx(0.4)
    assert loaded.loc[0, "away_xga"] == pytest.approx(1.25)
    assert result.xg_complete == 1
    assert "xGFixture" in calls


def test_backfill_paginates_prematch_odds_caches_without_the_token_and_skips_filled_rows(tmp_path):
    store = MatchStore(tmp_path)
    store.replace(
        pd.DataFrame(
            [
                _row(1, "finished", "2024-09-01 20:00:00+00:00"),
                _row(2, "scheduled", "2024-09-08 20:00:00+00:00"),
            ]
        )
    )
    calls = []

    def odds_page(fixture_id, include):
        if include != "odds":
            return {
                "id": fixture_id,
                "xgfixture": [
                    {"type_id": 5304, "location": "home", "participant_id": 10, "data": {"value": 1.1}},
                    {"type_id": 5304, "location": "away", "participant_id": 20, "data": {"value": 0.8}},
                ],
            }
        return {
            "id": fixture_id,
            "odds": [
                _quote(4, "Home", "1.90", "2024-09-01 12:00:00"),
                _quote(4, "Draw", "3.40", "2024-09-01 12:00:00"),
                _quote(4, "Away", "4.20", "2024-09-01 12:00:00"),
            ],
            "inplayOdds": [
                {"market_id": 1, "bookmaker_id": 4, "label": "Home", "value": "9.9"},
                {"market_id": 1, "bookmaker_id": 4, "label": "Draw", "value": "9.9"},
                {"market_id": 1, "bookmaker_id": 4, "label": "Away", "value": "9.9"},
            ],
        }

    def transport(url, params):
        calls.append((url, dict(params)))
        include = params.get("include")
        if include == "xGFixture":
            return 200, {}, {"data": [odds_page(1, include)], "pagination": {"has_more": False}}
        if "cursor" not in params:
            return 200, {}, {
                "data": [odds_page(1, include)],
                "pagination": {
                    "has_more": True,
                    "next_cursor": "https://api.sportmonks.com/v3/football/fixtures/multi/1,2?cursor=page-2&api_token=client-token",
                },
                "rate_limit": {"remaining": 100, "resets_in_seconds": 3600},
            }
        assert params["cursor"] == "page-2"
        assert "per_page" not in params
        return 200, {}, {
            "data": [odds_page(2, include)],
            "pagination": {"has_more": False},
        }

    result = fetch_stored_markets(_client(tmp_path, transport), store, chunk_size=2)
    loaded = store.load().set_index("fixture_id")
    assert loaded.loc[1, "home_xg"] == pytest.approx(1.1)
    assert loaded.loc[1, "home_xga"] == pytest.approx(0.8)
    assert pd.isna(loaded.loc[2, "home_xg"])
    assert loaded.loc[1, "implied_home"] == pytest.approx(loaded.loc[2, "implied_home"])
    assert abs(loaded.loc[1, ["implied_home", "implied_draw", "implied_away"]].sum() - 1) < 1e-9
    assert loaded.loc[1, "odds_home"] != pytest.approx(9.9)
    assert result.odds_complete == 2
    assert result.xg_complete == 1
    assert result.xg_requested == 1
    text = format_market_fetch(result)
    assert "完整赛前 1X2：2 场" in text
    assert "inplayOdds" in text
    blob = "\n".join(path.read_text(encoding="utf-8") for path in (tmp_path / "cache").glob("*.json"))
    assert "client-token" not in blob
    assert "page-2" in blob or "cursor" in blob

    calls.clear()
    again = fetch_stored_markets(_client(tmp_path, transport), store, chunk_size=2)
    assert again.odds_requested == 1
    assert again.xg_requested == 0
    assert calls
    assert all(call[1].get("include") == "odds" for call in calls)

    calls.clear()
    third = fetch_stored_markets(_client(tmp_path, transport), store, chunk_size=2)
    assert third.odds_requested == 1
    assert calls == []

    store.upsert(pd.DataFrame([_row(9, "finished", "2024-09-03 20:00:00+00:00")]))
    frame = store.load()
    frame.loc[frame["fixture_id"] == 2, "status"] = "finished"
    frame.loc[frame["fixture_id"] == 2, "home_xg"] = 0.2
    frame.loc[frame["fixture_id"] == 2, "away_xg"] = 0.3
    store.replace(frame)
    misses = []

    def quiet(url, params):
        misses.append(params.get("include"))
        return 200, {}, {"data": [{"id": 9}], "pagination": {"has_more": False}}

    fetch_stored_markets(_client(tmp_path, quiet), store, chunk_size=2)
    assert "odds" in misses
    assert "xGFixture" in misses
    misses.clear()
    fetch_stored_markets(_client(tmp_path, quiet), store, chunk_size=2)
    assert misses == []


def test_finished_row_with_odds_but_no_raw_implied_is_requested_again(tmp_path):
    store = MatchStore(tmp_path)
    store.replace(
        pd.DataFrame(
            [
                _row(
                    1,
                    "finished",
                    "2024-09-01 20:00:00+00:00",
                    odds_home=2.0,
                    odds_draw=3.4,
                    odds_away=4.0,
                    implied_home=0.48,
                    implied_draw=0.28,
                    implied_away=0.24,
                )
            ]
        )
    )
    calls = []

    def transport(url, params):
        calls.append(params.get("include"))
        return 200, {}, {
            "data": [
                {
                    "id": 1,
                    "odds": [
                        _quote(4, "Home", "1.90", "2024-09-01 12:00:00"),
                        _quote(4, "Draw", "3.40", "2024-09-01 12:00:00"),
                        _quote(4, "Away", "4.20", "2024-09-01 12:00:00"),
                    ],
                }
            ],
            "pagination": {"has_more": False},
        }

    first = fetch_stored_markets(_client(tmp_path, transport), store)
    assert first.odds_requested == 1
    assert "odds" in calls
    loaded = store.load().set_index("fixture_id")
    assert loaded.loc[1, "raw_implied_home"] > 0
    assert loaded.loc[1, ["raw_implied_home", "raw_implied_draw", "raw_implied_away"]].sum() > 1
    calls.clear()
    second = fetch_stored_markets(_client(tmp_path, transport), store)
    assert second.odds_requested == 0
    assert calls == []


def test_permission_errors_are_messages_and_do_not_wipe_the_other_feed(tmp_path):
    store = MatchStore(tmp_path)
    store.replace(
        pd.DataFrame(
            [
                _row(1, "finished", "2024-09-01 20:00:00+00:00"),
                _row(2, "finished", "2024-09-02 20:00:00+00:00"),
            ]
        )
    )
    odds_calls = []

    def transport(url, params):
        include = params.get("include")
        if include == "odds":
            odds_calls.append(url)
            return 403, {}, {"message": "You do not have access to the Odds add-on."}
        if "1,2" in url or url.endswith("1"):
            return 200, {}, {
                "data": [
                    {
                        "id": 1,
                        "xgfixture": [
                            {"type_id": 5304, "location": "home", "participant_id": 10, "data": {"value": 0.7}},
                            {"type_id": 5304, "location": "away", "participant_id": 20, "data": {"value": 1.4}},
                        ],
                    }
                ],
                "pagination": {"has_more": False},
            }
        return 404, {}, {"message": "No result(s) found matching your request."}

    result = fetch_stored_markets(_client(tmp_path, transport), store, chunk_size=1)
    loaded = store.load().set_index("fixture_id")
    assert loaded.loc[1, "home_xg"] == pytest.approx(0.7)
    assert loaded.loc[1, "away_xga"] == pytest.approx(0.7)
    assert pd.isna(loaded.loc[2, "home_xg"])
    assert result.odds_denied is not None
    assert "HTTP 403" in result.odds_denied
    assert result.xg_denied is None
    assert len(odds_calls) == 1
    text = format_market_fetch(result)
    assert "Odds & Predictions" in text


def test_daily_upsert_keeps_backfilled_odds_when_the_new_row_has_none(tmp_path):
    store = MatchStore(tmp_path)
    original = pd.DataFrame([_row(5, "finished", "2024-09-01 20:00:00+00:00", home_xg=1.0, away_xg=0.5, odds_home=2.1, odds_draw=3.3, odds_away=3.5, implied_home=0.46, implied_draw=0.29, implied_away=0.25)])
    store.replace(original)
    update = pd.DataFrame([_row(5, "finished", "2024-09-01 20:00:00+00:00")])
    update["home_goals_ft"] = 3
    store.upsert(update)
    loaded = store.load()
    assert int(loaded.loc[0, "home_goals_ft"]) == 3
    assert loaded.loc[0, "odds_home"] == pytest.approx(2.1)
    assert loaded.loc[0, "home_xg"] == pytest.approx(1.0)
    assert loaded.loc[0, "implied_away"] == pytest.approx(0.25)


def test_fetch_markets_command_prints_coverage(tmp_path, monkeypatch):
    import io
    import logging
    import sys

    from laliga.cli import main

    store = MatchStore(tmp_path)
    store.replace(pd.DataFrame([_row(3, "scheduled", "2024-09-08 20:00:00+00:00")]))

    def transport(url, params):
        return 200, {}, {
            "data": [
                {
                    "id": 3,
                    "odds": [
                        _quote(1, "Home", "2.10", "2024-09-08 12:00:00"),
                        _quote(1, "Draw", "3.30", "2024-09-08 12:00:00"),
                        _quote(1, "Away", "3.60", "2024-09-08 12:00:00"),
                    ],
                }
            ],
            "pagination": {"has_more": False},
        }

    monkeypatch.setattr("laliga.cli.open_client", lambda *args, **kwargs: _client(tmp_path, transport))
    stdout = io.StringIO()
    monkeypatch.setattr(sys, "stdout", stdout)
    monkeypatch.setattr(sys, "stderr", io.StringIO())
    root = logging.getLogger()
    handlers = root.handlers[:]
    level = root.level
    try:
        code = main(["fetch-markets", "--data-dir", str(tmp_path)])
    finally:
        root.handlers[:] = handlers
        root.setLevel(level)
    assert code == 0
    assert "完整赛前 1X2：1 场" in stdout.getvalue()
    loaded = store.load()
    assert loaded.loc[0, "odds_home"] > 1
