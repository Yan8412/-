"""Parsers against the SportMonks shapes seen on a live La Liga pull."""

from __future__ import annotations

import pandas as pd
import pytest

from laliga.data.markets import fetch_stored_markets
from laliga.data.parse import aggregate_prematch_1x2
from laliga.data.quotes import load_quotes, parse_odds_rows, save_quotes, targets_from_bundle
from laliga.data.store import MatchStore
from laliga.model.devig import devig_vector
from laliga.model.joint import HalfRates
from laliga.model.service import MarketProb, Prediction, ScoreProb
from laliga.model.tracks import _weight_summary
from laliga.pipeline import ScoredFixture, describe_alternative_tracks


def _row(market_id, label, value, stamp, **extra):
    payload = {
        "market_id": market_id,
        "label": label,
        "value": value,
        "bookmaker_id": extra.pop("bookmaker_id", 2),
        "latest_bookmaker_update": stamp,
    }
    payload.update(extra)
    return payload


def test_htft_reads_name_not_the_team_label():
    stamp = "2026-10-18T12:00:00Z"
    bundle = parse_odds_rows(
        [
            _row(
                29,
                "Sevilla - Osasuna",
                "41.00",
                stamp,
                name="Home/Away",
                market_description="Half Time/Full Time",
            ),
            _row(29, "Sevilla - Osasuna", "5.00", stamp, name="Home/Draw", market_description="Half Time/Full Time"),
        ],
        kickoff="2026-10-18T19:00:00Z",
        home_id=1,
        away_id=2,
    )
    probs = bundle["htft"]["closing"]["probs"]
    assert set(probs) == {"1/2", "1/X"}
    assert "Sevilla" not in "".join(probs)


def test_correct_score_name_is_from_the_winners_perspective():
    late = "2026-10-18T16:00:00Z"
    rows = [
        _row(57, "2", "8.50", late, name="2-1", bookmaker_id=2),
        _row(57, "1", "6.00", late, name="2-1", bookmaker_id=2),
        _row(57, "Draw", "7.00", late, name="1-1", bookmaker_id=2),
        _row(57, "1:2", "8.50", late, name="1:2", bookmaker_id=9),
        _row(57, "3:3", "25", late, name="3:3", bookmaker_id=9),
        _row(30, "2", "9.00", late, name="2-1", bookmaker_id=2),
        _row(30, "Draw", "6.50", late, name="0-0", bookmaker_id=2),
    ]
    bundle = parse_odds_rows(rows, kickoff="2026-10-18T19:00:00Z", home_id=1, away_id=2)
    scores = bundle["correct_score"]["closing"]["probs"]
    assert "1-2" in scores
    assert "2-1" in scores
    assert "1-1" in scores
    assert "3-3" in scores
    assert set(bundle["ht_correct_score"]["closing"]["probs"]) == {"1-2", "0-0"}


def test_asian_handicap_detects_home_line_and_own_line_books():
    late = "2026-10-18T16:00:00Z"
    rows = []
    # Book 20 prints the home line on both sides.
    for handicap, home, away in (("-1.0", "2.40", "1.60"), ("-0.5", "1.90", "1.90"), ("0", "1.55", "2.40")):
        rows.append(_row(6, "Home", home, late, bookmaker_id=20, handicap=handicap, name=handicap))
        rows.append(_row(6, "Away", away, late, bookmaker_id=20, handicap=handicap, name=handicap))
    # Book 2 prints each side's own line. Away +2.5 must come back to home -2.5.
    rows.append(_row(6, "Home", "3.40", late, bookmaker_id=2, handicap="-2.5", name="Home -2.5"))
    rows.append(_row(6, "Away", "1.30", late, bookmaker_id=2, handicap="+2.5", name="Away +2.5"))
    # Book 14 quarter line, both selections carry the home split.
    rows.append(_row(6, "Home", "2.60", late, bookmaker_id=14, handicap="-1,-1.5"))
    rows.append(_row(6, "Away", "1.50", late, bookmaker_id=14, handicap="-1,-1.5"))
    bundle = parse_odds_rows(rows, kickoff="2026-10-18T19:00:00Z", home_id=1, away_id=2)
    lines = {round(row["handicap"], 4): row for row in bundle["ah"]["closing"]}
    assert -0.5 in lines
    assert 0.5 not in lines
    assert -2.5 in lines
    assert 2.5 not in lines
    assert -1.25 in lines
    ordered = sorted(lines)
    probs = [lines[line]["home_prob"] for line in ordered]
    assert probs == sorted(probs)
    assert all(row["monotonic"] for row in lines.values())
    assert bundle["ah"]["dropped_lines"] == 0
    # The paired -0.5 prices are 1.58-style home/away, not a negated +0.5.
    fair = devig_vector([1 / 1.90, 1 / 1.90], ("proportional",))
    assert lines[-0.5]["home_prob"] == pytest.approx(float(fair["proportional"][0]))


def test_over_line_is_read_from_name_when_total_is_null():
    rows = [
        _row(1, "Home", "2.00", "2026-10-18T12:00:00Z"),
        _row(1, "Draw", "3.40", "2026-10-18T12:00:00Z"),
        _row(1, "Away", "3.80", "2026-10-18T12:00:00Z"),
        _row(28, "Over", "1.80", "2026-10-18T12:00:00Z", name="3.5", total=None),
        _row(28, "Under", "2.00", "2026-10-18T12:00:00Z", name="3.5", total=None),
    ]
    bundle = parse_odds_rows(rows, kickoff="2026-10-18T19:00:00Z", home_id=1, away_id=2)
    assert bundle["ht_ou"]["closing"][0]["line"] == pytest.approx(3.5)
    targets = targets_from_bundle(bundle)
    assert targets is not None
    assert targets.ht_ou[0][0] == pytest.approx(3.5)


def test_one_timestamp_per_selection_is_not_opening_and_labels_1_and_2_are_not_lines():
    rows = [
        _row(1, "1", "2.10", "2026-10-18T12:00:00Z", name="1", total=None, handicap=None),
        _row(1, "X", "3.40", "2026-10-18T12:01:00Z", name="X", total=None, handicap=None),
        _row(1, "2", "3.60", "2026-10-18T12:02:00Z", name="2", total=None, handicap=None),
        _row(6, "1", "1.90", "2026-10-18T12:03:00Z", name="1", total=None, handicap=None),
        _row(6, "2", "1.90", "2026-10-18T12:04:00Z", name="2", total=None, handicap=None),
    ]
    bundle = parse_odds_rows(rows, kickoff="2026-10-18T19:00:00Z", home_id=1, away_id=2)
    assert bundle["opening_and_closing"] is False
    assert bundle["ft_1x2"]["history"] is False
    assert bundle["ft_1x2"]["opening"] is None
    assert bundle["ft_1x2"]["closing"]["raw"]["home"] == pytest.approx(1 / 2.10)
    assert "ah" not in bundle


def test_same_selection_with_two_timestamps_still_has_opening():
    early = "2026-10-18T10:00:00Z"
    late = "2026-10-18T16:00:00Z"
    rows = []
    for stamp, price in ((early, "2.20"), (late, "2.05")):
        rows.append(_row(1, "Home", price, stamp))
    rows.append(_row(1, "Draw", "3.40", early))
    rows.append(_row(1, "Draw", "3.50", late))
    rows.append(_row(1, "Away", "3.60", early))
    rows.append(_row(1, "Away", "3.70", late))
    bundle = parse_odds_rows(rows, kickoff="2026-10-18T19:00:00Z", home_id=1, away_id=2)
    assert bundle["opening_and_closing"] is True
    assert bundle["ft_1x2"]["opening"]["raw"]["home"] == pytest.approx(1 / 2.20)
    assert bundle["ft_1x2"]["closing"]["raw"]["home"] == pytest.approx(1 / 2.05)


def test_post_kickoff_settlement_touch_is_kept_and_live_prices_are_not():
    kickoff = "2026-10-18T19:00:00Z"
    early = "2026-10-18T12:00:00Z"
    late = "2026-10-18T21:00:00Z"
    rows = []
    for label, price in (("Home", "2.40"), ("Draw", "3.20"), ("Away", "3.10")):
        rows.append(_row(1, label, price, early, bookmaker_id=9))
    for label, price in (("Home", "2.10"), ("Draw", "3.30"), ("Away", "3.50")):
        rows.append(_row(1, label, price, late, bookmaker_id=4, stopped=True))
    for label in ("Home", "Draw", "Away"):
        rows.append(_row(1, label, "9.00", late, bookmaker_id=5))
        rows.append(_row(1, label, "8.00", late, bookmaker_id=6, stopped=True, inplay=True))
    rows.append(_row(1, "Home", "1.95", late, bookmaker_id=7, winning=False))
    rows.append(_row(1, "Draw", "3.60", late, bookmaker_id=7, winning=False))
    rows.append(_row(1, "Away", "3.90", late, bookmaker_id=7, winning=False))
    bundle = parse_odds_rows(rows, kickoff=kickoff, home_id=1, away_id=2)
    # Book 5 is the naked post-kickoff price and must not move the average.
    # Books 4 (stopped) and 7 (winning present) stay, with the pre-kickoff book 9.
    assert bundle["ft_1x2"]["closing"]["books"] == 3
    prices = aggregate_prematch_1x2(rows, kickoff=kickoff)
    assert prices["odds_home"] != pytest.approx(9.0)
    raw_home = (1 / 2.40 + 1 / 2.10 + 1 / 1.95) / 3
    assert prices["raw_implied_home"] == pytest.approx(raw_home)


def test_correct_score_is_devigged_per_book_before_averaging():
    late = "2026-10-18T16:00:00Z"
    rows = [
        _row(57, "1-0", "2.0", late, name="1-0", bookmaker_id=9),
        _row(57, "0-0", "2.0", late, name="0-0", bookmaker_id=9),
        _row(57, "1-0", "3.0", late, name="1-0", bookmaker_id=16),
        _row(57, "0-1", "1.5", late, name="0-1", bookmaker_id=16),
    ]
    bundle = parse_odds_rows(rows, kickoff="2026-10-18T19:00:00Z", home_id=1, away_id=2)
    probs = bundle["correct_score"]["closing"]["probs"]
    assert sum(probs.values()) == pytest.approx(1.0)
    book_a = {"1-0": 0.5, "0-0": 0.5}
    book_b = {"1-0": (1 / 3) / 1.0, "0-1": (1 / 1.5) / 1.0}
    averaged = {
        "1-0": (book_a["1-0"] + book_b["1-0"]) / 2,
        "0-0": book_a["0-0"],
        "0-1": book_b["0-1"],
    }
    scale = sum(averaged.values())
    expected = {label: value / scale for label, value in averaged.items()}
    assert probs["1-0"] == pytest.approx(expected["1-0"])
    assert probs["0-0"] == pytest.approx(expected["0-0"])
    union_raw = {"1-0": 1 / 2.5, "0-0": 1 / 2.0, "0-1": 1 / 1.5}
    union_scale = sum(union_raw.values())
    union = {label: value / union_scale for label, value in union_raw.items()}
    assert probs["1-0"] != pytest.approx(union["1-0"], abs=1e-4)


def test_weight_summary_counts_match_days_not_distinct_weights():
    rows = [
        {"match_day": "2026-09-01", "blend_rate_weight": 0.2},
        {"match_day": "2026-09-01", "blend_rate_weight": 0.2},
        {"match_day": "2026-09-08", "blend_rate_weight": 0.4},
        {"match_day": "2026-09-15", "blend_rate_weight": 0.4},
    ]
    summary = _weight_summary(rows)
    assert summary["n_days"] == 3
    assert summary["n_matches"] == 4
    assert summary["n_distinct_weights"] == 2


def test_predict_text_shows_market_and_blend_at_the_champion_weight(tmp_path):
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "registry.json").write_text(
        '{"champion": {"params": {"blend_market_weight": 0.65}}}',
        encoding="utf-8",
    )
    prediction = Prediction(
        fixture_id=7,
        starting_at="2026-10-20T19:00:00Z",
        season_id=1,
        season_name="2026/2027",
        round_name="10",
        home_team_id=1,
        home_team="Barcelona",
        away_team_id=2,
        away_team="Real Madrid",
        ft=MarketProb(0.55, 0.25, 0.20),
        ht=MarketProb(0.45, 0.30, 0.25),
        top_scores=[ScoreProb(1, 0, 0.12), ScoreProb(1, 1, 0.10), ScoreProb(2, 1, 0.09)],
        expected_goals_ft=(1.4, 1.1),
        expected_goals_ht=(0.6, 0.5),
        top_htft=[("1/1", 0.22), ("1/X", 0.11)],
        rates=HalfRates(0.62, 0.48, 0.78, 0.55, -0.05, -0.05, 0.0),
    )
    history = pd.DataFrame(
        [{"fixture_id": 7, "odds_home": 1.70, "odds_draw": 3.80, "odds_away": 4.60}]
    )
    scored = [ScoredFixture(prediction, ft_matrix=None, ht_matrix=None)]
    text = describe_alternative_tracks(scored, history, tmp_path)
    assert "纯足球" in text
    assert "市场" in text
    assert "市场权重 0.65" in text
    assert "blend" not in prediction.to_dict()
    bare = describe_alternative_tracks(scored, history, tmp_path / "empty")
    assert "市场权重为 0，与纯足球相同" in bare
    missing = describe_alternative_tracks(scored, pd.DataFrame([{"fixture_id": 7}]), tmp_path)
    assert "这场没有赛前 1X2" in missing


def test_fetch_markets_writes_the_quotes_file_once_per_pass(tmp_path, monkeypatch):
    from laliga.data.client import SportMonksClient

    store = MatchStore(tmp_path)
    kickoff = "2024-09-01 20:00:00+00:00"
    store.replace(
        pd.DataFrame(
            [
                {
                    "fixture_id": 1,
                    "league_id": 564,
                    "season_id": 1,
                    "season_name": "2024/2025",
                    "round_id": 1,
                    "round_name": "1",
                    "starting_at": kickoff,
                    "state_id": 5,
                    "state_name": "FT",
                    "home_team_id": 10,
                    "home_team_name": "Home",
                    "away_team_id": 20,
                    "away_team_name": "Away",
                    "home_goals_ht": 0,
                    "away_goals_ht": 0,
                    "home_goals_ft": 1,
                    "away_goals_ft": 0,
                    "status": "finished",
                },
                {
                    "fixture_id": 2,
                    "league_id": 564,
                    "season_id": 1,
                    "season_name": "2024/2025",
                    "round_id": 1,
                    "round_name": "1",
                    "starting_at": "2024-09-08 20:00:00+00:00",
                    "state_id": 1,
                    "state_name": "NS",
                    "home_team_id": 10,
                    "home_team_name": "Home",
                    "away_team_id": 20,
                    "away_team_name": "Away",
                    "home_goals_ht": None,
                    "away_goals_ht": None,
                    "home_goals_ft": None,
                    "away_goals_ft": None,
                    "status": "scheduled",
                },
            ]
        )
    )
    writes = {"n": 0}
    real_save = save_quotes

    def spy(root, payload):
        writes["n"] += 1
        real_save(root, payload)

    monkeypatch.setattr("laliga.data.quotes.save_quotes", spy)
    monkeypatch.setattr("laliga.data.markets.save_quotes", spy)

    def transport(url, params):
        include = params.get("include")
        data = []
        for fixture_id in (1, 2):
            if include == "xGFixture":
                data.append({"id": fixture_id})
                continue
            data.append(
                {
                    "id": fixture_id,
                    "odds": [
                        _row(1, "Home", "1.90", "2024-09-01 12:00:00", bookmaker_id=4),
                        _row(1, "Draw", "3.40", "2024-09-01 12:00:00", bookmaker_id=4),
                        _row(1, "Away", "4.20", "2024-09-01 12:00:00", bookmaker_id=4),
                    ],
                }
            )
        return 200, {}, {"data": data, "pagination": {"has_more": False}}

    client = SportMonksClient("client-token", tmp_path / "cache", transport=transport, min_interval=0, sleep=lambda _seconds: None)
    fetch_stored_markets(client, store, chunk_size=2)
    quotes = load_quotes(tmp_path)
    assert set(quotes) == {"1", "2"}
    # One write after the 1X2 pass and one after the extended pass, not one per match.
    assert writes["n"] == 2
