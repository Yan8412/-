"""Parsers against the SportMonks shapes seen on a live La Liga pull."""

from __future__ import annotations

import pandas as pd
import pytest

from laliga.data.markets import fetch_stored_markets
from laliga.data.parse import aggregate_prematch_1x2
from laliga.data.quotes import (
    PARSER_VERSION,
    absorb_fixture_odds,
    load_quotes,
    parse_odds_rows,
    save_quotes,
    targets_from_bundle,
)
from laliga.data.store import MatchStore
from laliga.model.devig import devig_vector
from laliga.model.joint import HalfRates
from laliga.model.service import MarketProb, Prediction, ScoreProb
from laliga.model.tracks import _weight_summary
from laliga.pipeline import ScoredFixture, describe_alternative_tracks


_NINE = ("1/1", "1/X", "1/2", "X/1", "X/X", "X/2", "2/1", "2/X", "2/2")


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


def _with_other_htft(rows, stamp, covered, *, bookmaker_id=2):
    """Pad a book to all nine HT/FT codes. ``covered`` must come from the rows under test."""

    extra = list(rows)
    for code in _NINE:
        if code in covered:
            continue
        extra.append(_row(29, code, "40", stamp, name=code, bookmaker_id=bookmaker_id))
    return extra


def _score_rows(market_id, stamp, *, bookmaker_id=2, price="20", stopped=False, high=5):
    rows = []
    for home in range(high):
        for away in range(high):
            label = f"{home}-{away}"
            rows.append(
                _row(
                    market_id,
                    label,
                    price,
                    stamp,
                    name=label,
                    bookmaker_id=bookmaker_id,
                    stopped=stopped,
                )
            )
    return rows


def test_htft_reads_name_not_the_team_label():
    stamp = "2026-10-18T12:00:00Z"
    named = [
        _row(
            29,
            "Sevilla - Osasuna",
            "41.00",
            stamp,
            name="Home/Away",
            market_description="Half Time/Full Time",
        ),
        _row(29, "Sevilla - Osasuna", "5.00", stamp, name="Home/Draw", market_description="Half Time/Full Time"),
    ]
    kickoff = "2026-10-18T19:00:00Z"
    assert "htft" not in parse_odds_rows(named, kickoff=kickoff, home_id=1, away_id=2)
    bundle = parse_odds_rows(
        _with_other_htft(named, stamp, {"1/2", "1/X"}),
        kickoff=kickoff,
        home_id=1,
        away_id=2,
    )
    probs = bundle["htft"]["closing"]["probs"]
    assert {"1/2", "1/X"} <= set(probs)
    assert len(probs) == 9
    assert bundle["htft"]["closing"]["complete"] is True
    assert "Sevilla" not in "".join(probs)


def test_correct_score_name_is_from_the_winners_perspective():
    from laliga.data.quotes import _score_label

    late = "2026-10-18T16:00:00Z"
    assert _score_label({"label": "2", "name": "2-1", "value": "8.50"}) == {"1-2": 8.50}
    assert _score_label({"label": "1", "name": "2-1", "value": "6.00"}) == {"2-1": 6.00}
    assert _score_label({"label": "Draw", "name": "1-1", "value": "7.00"}) == {"1-1": 7.00}
    assert _score_label({"label": "1:2", "name": "1:2", "value": "8.50"}) == {"1-2": 8.50}
    assert _score_label({"label": "Tie", "name": "2-1", "value": "12"}) == {}
    rows = _score_rows(57, late, bookmaker_id=2, price="12")
    rows.extend(
        [
            _row(57, "2", "8.50", late, name="2-1", bookmaker_id=2),
            _row(57, "1", "6.00", late, name="2-1", bookmaker_id=2),
            _row(57, "Draw", "7.00", late, name="1-1", bookmaker_id=2),
            _row(57, "3:3", "25", late, name="3:3", bookmaker_id=2),
            _row(30, "2", "9.00", late, name="2-1", bookmaker_id=2),
            _row(30, "Draw", "6.50", late, name="0-0", bookmaker_id=2),
        ]
    )
    bundle = parse_odds_rows(rows, kickoff="2026-10-18T19:00:00Z", home_id=1, away_id=2)
    scores = bundle["correct_score"]["closing"]["probs"]
    assert "1-2" in scores
    assert "2-1" in scores
    assert "1-1" in scores
    assert "3-3" in scores
    assert "ht_correct_score" not in bundle


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


def test_post_kickoff_rows_are_dropped_even_when_winning_is_false():
    """Fixture 19732660 shape: live rows carry winning false and stay stopped false."""

    from laliga.data.quotes import is_prematch_quote

    kickoff = "2026-05-24T16:15:00Z"
    early = "2026-05-24T12:00:00Z"
    late = "2026-05-24T18:31:00Z"
    rows = []
    for label, price in (("Home", "2.10"), ("Draw", "3.40"), ("Away", "3.40")):
        rows.append(_row(1, label, price, early, bookmaker_id=2))
    for book, draw in ((26, "1.01"), (28, "1.02"), (38, "1.03"), (57, "1.04"), (58, "1.05"), (64, "1.05")):
        rows.append(_row(1, "Home", "8.00", late, bookmaker_id=book, stopped=False, winning=False))
        rows.append(_row(1, "Draw", draw, late, bookmaker_id=book, stopped=False, winning=False))
        rows.append(_row(1, "Away", "15.00", late, bookmaker_id=book, stopped=False, winning=False))
    for label, price in (("Home", "1.40"), ("Draw", "4.20"), ("Away", "8.00")):
        rows.append(_row(1, label, price, late, bookmaker_id=4, stopped=True, winning=True))
    bundle = parse_odds_rows(
        rows,
        kickoff=kickoff,
        home_id=1,
        away_id=2,
        home_name="Deportivo Alavés",
        away_name="Levante",
    )
    assert bundle["ft_1x2"]["closing"]["books"] == 1
    assert bundle["ft_1x2"]["closing"]["raw"]["draw"] == pytest.approx(1 / 3.40)
    assert bundle["ft_1x2"]["closing"]["raw"]["home"] == pytest.approx(1 / 2.10)
    prices = aggregate_prematch_1x2(rows, kickoff=kickoff)
    assert prices["raw_implied_draw"] == pytest.approx(1 / 3.40)
    assert prices["raw_implied_home"] == pytest.approx(1 / 2.10)
    assert is_prematch_quote({"value": "2.0"}, kickoff) is True
    assert is_prematch_quote(
        {"value": "1.01", "latest_bookmaker_update": late, "winning": False, "stopped": False}, kickoff
    ) is False
    assert is_prematch_quote(
        {"value": "1.40", "latest_bookmaker_update": late, "stopped": True, "winning": True}, kickoff
    ) is False


def test_correct_score_is_devigged_per_book_before_averaging():
    from laliga.data.quotes import _average_labels

    late = "2026-10-18T16:00:00Z"
    rows = [
        _row(57, "1-0", "2.0", late, name="1-0", bookmaker_id=9),
        _row(57, "0-0", "2.0", late, name="0-0", bookmaker_id=9),
        _row(57, "1-0", "3.0", late, name="1-0", bookmaker_id=16),
        _row(57, "0-1", "1.5", late, name="0-1", bookmaker_id=16),
    ]
    bundle = parse_odds_rows(rows, kickoff="2026-10-18T19:00:00Z", home_id=1, away_id=2)
    assert "correct_score" not in bundle
    probs = _average_labels([{"1-0": 2.0, "0-0": 2.0}, {"1-0": 3.0, "0-1": 1.5}])["probs"]
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
    from laliga.report import format_tracks

    text = format_tracks({"tracks": {}, "blend_rate_weight": summary, "notes": []})
    assert "参数混合的市场权重" in text
    assert "起点 0.20" in text
    assert "终点 0.40" in text


def test_predict_text_shows_market_and_blend_at_the_champion_weight(tmp_path, monkeypatch):
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
    from laliga.model import implied_goals, joint

    grids = []
    real_convolve = joint.convolve

    def _record(*args, **kwargs):
        grids.append((kwargs.get("max_half"), kwargs.get("max_ft"), kwargs.get("temperature")))
        return real_convolve(*args, **kwargs)

    monkeypatch.setattr(joint, "convolve", _record)
    monkeypatch.setattr(implied_goals, "convolve", _record)
    text = describe_alternative_tracks(scored, history, tmp_path)
    assert "纯足球" in text
    assert "市场" in text
    assert "前两比分" in text
    assert "半全场前两档" in text
    assert "市场权重 0.65" in text
    assert grids
    assert set(grids) == {(5, 8, 1.0)}
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
    assert quotes["1"]["parser_version"] == PARSER_VERSION


def test_htft_team_labels_map_when_name_is_null():
    from laliga.data.teams import same_club

    stamp = "2024-08-18T12:00:00Z"
    kickoff = "2024-08-18T19:00:00Z"
    valencia_rows = [
        _row(
            29,
            "Draw - Valencia",
            "6.50",
            stamp,
            name=None,
            original_label=None,
            id=131425640745,
            bookmaker_id=2,
        ),
        _row(29, "Valencia - Draw", "7.00", stamp, name=None, original_label=None, bookmaker_id=2),
    ]
    assert "htft" not in parse_odds_rows(
        valencia_rows,
        kickoff=kickoff,
        home_id=1,
        away_id=2,
        home_name="Real Sociedad",
        away_name="Valencia",
    )
    valencia = parse_odds_rows(
        _with_other_htft(valencia_rows, stamp, {"X/2", "2/X"}),
        kickoff=kickoff,
        home_id=1,
        away_id=2,
        home_name="Real Sociedad",
        away_name="Valencia",
    )
    assert {"X/2", "2/X"} <= set(valencia["htft"]["closing"]["probs"])
    assert len(valencia["htft"]["closing"]["probs"]) == 9
    athletic_rows = [
        _row(29, "Draw - Athletic Bilbao", "5.50", stamp, name=None, original_label=None, bookmaker_id=2),
        _row(29, "Athletic Club - Draw", "6.00", stamp, name=None, original_label=None, bookmaker_id=2),
    ]
    athletic = parse_odds_rows(
        _with_other_htft(athletic_rows, stamp, {"X/1", "1/X"}),
        kickoff=kickoff,
        home_id=3,
        away_id=4,
        home_name="Athletic Club",
        away_name="Real Madrid",
    )
    assert {"X/1", "1/X"} <= set(athletic["htft"]["closing"]["probs"])
    assert len(athletic["htft"]["closing"]["probs"]) == 9
    named_rows = [
        _row(29, "Draw - Valencia", "6.00", stamp, name="Draw/Home", bookmaker_id=2),
        _row(29, "Valencia - Draw", "7.00", stamp, name="Home/Away", bookmaker_id=2),
    ]
    named = parse_odds_rows(
        _with_other_htft(named_rows, stamp, {"X/1", "1/2"}),
        kickoff=kickoff,
        home_id=1,
        away_id=2,
        home_name="Real Sociedad",
        away_name="Valencia",
    )
    assert {"X/1", "1/2"} <= set(named["htft"]["closing"]["probs"])
    assert len(named["htft"]["closing"]["probs"]) == 9
    assert same_club("Athletic Club", "Athletic Bilbao")
    assert same_club("Athletic", "Athletic Bilbao")
    assert not same_club("Deportivo", "Deportivo Alavés")


def test_stopped_duplicate_with_an_alias_is_not_opening_history():
    early = "2026-10-18T16:46:00Z"
    late = "2026-10-18T21:49:00Z"
    kickoff = "2026-10-19T19:00:00Z"
    from laliga.data.quotes import _score_label

    assert _score_label({"label": "Tie", "name": "2-1", "value": "12"}) == {}
    score_rows = _score_rows(57, late, price="9.00")
    score_rows.append(_row(57, "Tie", "8.00", early, name="0-0", stopped=True, bookmaker_id=2))
    score_rows.append(_row(57, "Tie", "12.00", late, name="2-1", stopped=False, bookmaker_id=2))
    scores = parse_odds_rows(score_rows, kickoff=kickoff, home_id=1, away_id=2)
    assert scores["correct_score"]["history"] is False
    assert scores["correct_score"]["opening"] is None
    assert scores["opening_and_closing"] is False
    # The stopped 8.00 is dropped because a live 9.00 exists, so every label stays 9.00.
    assert len(set(round(value, 6) for value in scores["correct_score"]["closing"]["probs"].values())) == 1
    result = parse_odds_rows(
        [
            _row(1, "1", "2.05", late, bookmaker_id=2),
            _row(1, "Athletic Club", "2.40", early, stopped=True, bookmaker_id=2),
            _row(1, "Athletic Bilbao", "3.00", "2026-10-18T15:00:00Z", stopped=True, bookmaker_id=2),
            _row(1, "Draw", "3.40", late, bookmaker_id=2),
            _row(1, "Real Madrid", "3.80", late, bookmaker_id=2),
        ],
        kickoff=kickoff,
        home_id=10,
        away_id=20,
        home_name="Athletic Club",
        away_name="Real Madrid",
    )
    assert result["opening_and_closing"] is False
    assert result["ft_1x2"]["opening"] is None
    assert result["ft_1x2"]["closing"]["raw"]["home"] == pytest.approx(1 / 2.05)
    alias = parse_odds_rows(
        [
            _row(1, "Athletic Bilbao", "2.05", late, stopped=False, bookmaker_id=2),
            _row(1, "Tie", "3.40", late, bookmaker_id=2),
            _row(1, "Real Madrid", "3.80", late, bookmaker_id=2),
        ],
        kickoff=kickoff,
        home_id=10,
        away_id=20,
        home_name="Athletic Club",
        away_name="Real Madrid",
    )
    assert alias["ft_1x2"]["closing"]["raw"]["home"] == pytest.approx(1 / 2.05)
    assert alias["ft_1x2"]["closing"]["raw"]["draw"] == pytest.approx(1 / 3.40)
    assert alias["ft_1x2"]["closing"]["raw"]["away"] == pytest.approx(1 / 3.80)
    only_stopped = parse_odds_rows(
        [
            _row(1, "Home", "2.20", early, stopped=True, bookmaker_id=2),
            _row(1, "Home", "2.05", late, stopped=True, bookmaker_id=2),
            _row(1, "Draw", "3.40", late, stopped=True, bookmaker_id=2),
            _row(1, "Away", "3.60", late, stopped=True, bookmaker_id=2),
        ],
        kickoff=kickoff,
        home_id=1,
        away_id=2,
    )
    assert only_stopped["opening_and_closing"] is False
    assert only_stopped["ft_1x2"]["history"] is False
    assert only_stopped["ft_1x2"]["opening"] is None
    assert only_stopped["ft_1x2"]["closing"]["raw"]["home"] == pytest.approx(1 / 2.05)


def test_absorb_replace_drops_markets_the_new_payload_does_not_contain(tmp_path):
    kickoff = "2026-10-18T19:00:00Z"
    early = "2026-10-18T12:00:00Z"
    both = {
        "id": 19135350,
        "odds": [
            _row(1, "Home", "2.10", early),
            _row(1, "Draw", "3.30", early),
            _row(1, "Away", "3.50", early),
            _row(80, "Over", "1.90", early, total="2.5"),
            _row(80, "Under", "1.90", early, total="2.5"),
        ],
    }
    absorb_fixture_odds(tmp_path, both, kickoff=kickoff, home_id=1, away_id=2, home_name="Valencia", away_name="Getafe")
    only_result = {"id": 19135350, "odds": both["odds"][:3]}
    absorb_fixture_odds(tmp_path, only_result, kickoff=kickoff, home_id=1, away_id=2, replace=False)
    assert "ou" in load_quotes(tmp_path)["19135350"]
    absorb_fixture_odds(tmp_path, only_result, kickoff=kickoff, home_id=1, away_id=2, replace=True)
    replaced = load_quotes(tmp_path)["19135350"]
    assert "ou" not in replaced
    assert "ft_1x2" in replaced
    late = "2026-10-18T21:16:00Z"
    inplay = {
        "id": 19135350,
        "odds": [
            _row(1, "Home", "1.05", late, winning=False, stopped=False),
            _row(1, "Draw", "1.02", late, winning=False, stopped=False),
            _row(1, "Away", "12.00", late, winning=False, stopped=False),
        ],
    }
    assert absorb_fixture_odds(tmp_path, inplay, kickoff=kickoff, home_id=1, away_id=2, replace=True) is False
    assert "19135350" not in load_quotes(tmp_path)


def test_stale_parser_version_is_reparsed_without_refresh(tmp_path, caplog):
    import logging

    from laliga.data.client import SportMonksClient

    caplog.set_level(logging.INFO, logger="laliga.data.markets")
    store = MatchStore(tmp_path)
    kickoff = "2024-08-18 19:00:00+00:00"
    store.replace(
        pd.DataFrame(
            [
                {
                    "fixture_id": 19135350,
                    "league_id": 564,
                    "season_id": 1,
                    "season_name": "2024/2025",
                    "round_id": 1,
                    "round_name": "1",
                    "starting_at": kickoff,
                    "state_id": 5,
                    "state_name": "FT",
                    "home_team_id": 10,
                    "home_team_name": "Real Sociedad",
                    "away_team_id": 20,
                    "away_team_name": "Valencia",
                    "home_goals_ht": 0,
                    "away_goals_ht": 0,
                    "home_goals_ft": 1,
                    "away_goals_ft": 1,
                    "status": "finished",
                    "odds_home": 2.1,
                    "odds_draw": 3.3,
                    "odds_away": 3.5,
                    "implied_home": 0.46,
                    "implied_draw": 0.29,
                    "implied_away": 0.25,
                    "raw_implied_home": 0.50,
                    "raw_implied_draw": 0.30,
                    "raw_implied_away": 0.28,
                    "home_xg": 1.1,
                    "away_xg": 0.8,
                }
            ]
        )
    )
    save_quotes(
        tmp_path,
        {
            "19135350": {
                "fixture_id": 19135350,
                "markets_seen": [29],
                "htft": {"closing": {"probs": {"9/9": 1.0}, "labels": {"9/9": 1.0}}},
            }
        },
    )

    def transport(url, params):
        if params.get("include") == "xGFixture":
            return 200, {}, {"data": [{"id": 19135350}], "pagination": {"has_more": False}}
        return 200, {}, {
            "data": [
                {
                    "id": 19135350,
                    "odds": [
                        _row(1, "Home", "2.10", "2024-08-18T12:00:00Z", bookmaker_id=2),
                        _row(1, "Draw", "3.30", "2024-08-18T12:00:00Z", bookmaker_id=2),
                        _row(1, "Away", "3.50", "2024-08-18T12:00:00Z", bookmaker_id=2),
                        _row(29, "Draw - Valencia", "6.50", "2024-08-18T12:00:00Z", name=None, original_label=None, bookmaker_id=2, id=131425640745),
                        _row(29, "Valencia - Draw", "7.00", "2024-08-18T12:00:00Z", name=None, original_label=None, bookmaker_id=2),
                    ],
                }
            ],
            "pagination": {"has_more": False},
        }

    client = SportMonksClient("client-token", tmp_path / "cache", transport=transport, min_interval=0, sleep=lambda _seconds: None)
    first = fetch_stored_markets(client, store)
    assert first.odds_requested == 1
    record = load_quotes(tmp_path)["19135350"]
    assert record["parser_version"] == PARSER_VERSION
    assert "9/9" not in str(record)
    assert "htft" not in record
    assert "回填扩展盘" in caplog.text
    second = fetch_stored_markets(client, store)
    assert second.odds_requested == 0


def test_empate_is_draw_and_same_prices_are_not_an_opening():
    stamp = "2026-01-26T12:00:00Z"
    kickoff = "2026-01-26T20:00:00Z"
    barcelona_rows = [
        _row(29, "Empate - Barcelona", "6.50", stamp, name=None, original_label=None, bookmaker_id=2),
        _row(29, "Barcelona - Empate", "7.00", stamp, name=None, original_label=None, bookmaker_id=2),
    ]
    barcelona = parse_odds_rows(
        _with_other_htft(barcelona_rows, stamp, {"X/1", "1/X"}),
        kickoff=kickoff,
        home_id=1,
        away_id=2,
        home_name="Barcelona",
        away_name="Levante",
    )
    assert {"X/1", "1/X"} <= set(barcelona["htft"]["closing"]["probs"])
    assert len(barcelona["htft"]["closing"]["probs"]) == 9
    osasuna_rows = [
        _row(29, "Osasuna - Empate", "5.50", stamp, name=None, original_label=None, bookmaker_id=2),
        _row(29, "Empate - Osasuna", "6.00", stamp, name=None, original_label=None, bookmaker_id=2),
    ]
    osasuna = parse_odds_rows(
        _with_other_htft(osasuna_rows, stamp, {"1/X", "X/1"}),
        kickoff=kickoff,
        home_id=3,
        away_id=4,
        home_name="Osasuna",
        away_name="Real Madrid",
    )
    assert {"1/X", "X/1"} <= set(osasuna["htft"]["closing"]["probs"])
    assert len(osasuna["htft"]["closing"]["probs"]) == 9
    # Fixture 19439499: Draw live, Tie and Empate stopped, all 3.00 on 0-0.
    scores = parse_odds_rows(
        [
            _row(30, "Draw", "3.00", "2026-01-26T19:55:00Z", name="0-0", stopped=False, bookmaker_id=2),
            _row(30, "Tie", "3.00", "2026-01-26T17:12:00Z", name="0-0", stopped=True, bookmaker_id=2),
            _row(30, "Empate", "3.00", "2026-01-26T02:19:00Z", name="0-0", stopped=True, bookmaker_id=2),
            _row(30, "1", "6.00", "2026-01-26T19:55:00Z", name="1-0", bookmaker_id=2),
        ],
        kickoff="2026-01-26T21:00:00Z",
        home_id=1,
        away_id=2,
    )
    assert scores["opening_and_closing"] is False
    assert "ht_correct_score" not in scores
    # Fixture 19439470: both labels stopped, same 1-0 price, Athletic aliases.
    athletic = parse_odds_rows(
        [
            _row(30, "Athletic Bilbao", "3.50", "2026-01-26T19:11:49Z", name="1-0", stopped=True, bookmaker_id=2),
            _row(30, "Athletic Club", "3.50", "2026-01-26T17:21:50Z", name="1-0", stopped=True, bookmaker_id=2),
            _row(30, "Draw", "4.00", "2026-01-26T19:11:49Z", name="0-0", stopped=True, bookmaker_id=2),
        ],
        kickoff="2026-01-26T21:00:00Z",
        home_id=10,
        away_id=20,
        home_name="Athletic Club",
        away_name="Real Madrid",
    )
    assert athletic["opening_and_closing"] is False
    assert "ht_correct_score" not in athletic
    moved = parse_odds_rows(
        [
            _row(30, "Athletic Club", "3.50", "2026-01-26T17:21:50Z", name="1-0", stopped=True, bookmaker_id=2),
            _row(30, "Athletic Bilbao", "4.20", "2026-01-26T19:11:49Z", name="1-0", stopped=True, bookmaker_id=2),
            _row(30, "Draw", "4.00", "2026-01-26T19:11:49Z", name="0-0", stopped=True, bookmaker_id=2),
        ],
        kickoff="2026-01-26T21:00:00Z",
        home_id=10,
        away_id=20,
        home_name="Athletic Club",
        away_name="Real Madrid",
    )
    assert "ht_correct_score" not in moved
    assert moved["opening_and_closing"] is False


def test_prematch_timestamps_parse_without_a_per_row_pandas_call():
    import time
    from datetime import datetime, timezone

    kickoff = datetime(2026, 1, 26, 21, 0, tzinfo=timezone.utc)
    rows = [
        {
            "value": "2.10",
            "latest_bookmaker_update": "2026-01-26T19:11:49Z" if index % 2 == 0 else "2026-01-26 22:00:00",
        }
        for index in range(20000)
    ]
    from laliga.data.quotes import is_prematch_quote

    started = time.perf_counter()
    kept = sum(1 for row in rows if is_prematch_quote(row, kickoff))
    elapsed = time.perf_counter() - started
    assert kept == 10000
    assert elapsed < 1.0


def test_partial_books_are_absent_and_all_stopped_rows_are_not_an_opening():
    """Fixtures 19439490–19439497 and 19439492/19439496/19439507."""

    from laliga.data.quotes import direct_probabilities
    from laliga.model.tracks import _direct_block

    late = "2026-02-22T18:00:00Z"
    early = "2026-02-19T15:06:00Z"
    mid = "2026-02-22T11:10:00Z"
    kickoff = "2026-02-22T20:00:00Z"
    partial = [
        _row(29, code, "8.00", late, name=code, bookmaker_id=2, stopped=True)
        for code in ("X/1", "1/X", "X/X", "X/2", "2/X")
    ]
    assert "htft" not in parse_odds_rows(partial, kickoff=kickoff, home_id=1, away_id=2)
    both = list(partial)
    for code in _NINE:
        both.append(_row(29, code, "9.00", late, name=code, bookmaker_id=9))
    kept = parse_odds_rows(both, kickoff=kickoff, home_id=1, away_id=2)
    probs = kept["htft"]["closing"]["probs"]
    assert set(probs) == set(_NINE)
    assert kept["htft"]["closing"]["complete"] is True
    assert kept["htft"]["history"] is False
    assert kept["htft"]["opening"] is None
    assert sum(probs.values()) == pytest.approx(1.0)
    # Book 2's 8.00 on five codes must not move the average away from book 9.
    assert len({round(value, 6) for value in probs.values()}) == 1

    level = [_row(30, score, "6.00", late, name=score, bookmaker_id=2) for score in ("0-0", "1-1", "2-2", "3-3")]
    assert "ht_correct_score" not in parse_odds_rows(level, kickoff=kickoff, home_id=1, away_id=2)

    six = ("0-0", "1-1", "2-2", "1-0", "0-1", "2-1")
    thin = []
    for score in six:
        thin.append(_row(57, score, "4.00", early, name=score, bookmaker_id=2))
        thin.append(_row(57, score, "5.00", late, name=score, bookmaker_id=2))
    assert "correct_score" not in parse_odds_rows(thin, kickoff=kickoff, home_id=1, away_id=2)
    full = list(thin)
    for home in range(11):
        for away in range(11):
            label = f"{home}-{away}"
            full.append(_row(57, label, "15", late, name=label, bookmaker_id=9))
    opened = parse_odds_rows(full, kickoff=kickoff, home_id=1, away_id=2)
    assert opened["correct_score"]["history"] is False
    assert opened["correct_score"]["opening"] is None
    assert opened["correct_score"]["closing"]["complete"] is True
    assert len(opened["correct_score"]["closing"]["probs"]) == 121
    assert opened["opening_and_closing"] is False

    stopped_scores = []
    for home in range(5):
        for away in range(5):
            label = f"{home}-{away}"
            if label == "0-0":
                stopped_scores.append(_row(30, "Empate", "5.50", early, name="0-0", stopped=True, bookmaker_id=2))
                stopped_scores.append(_row(30, "Tie", "6.00", mid, name="0-0", stopped=True, bookmaker_id=2))
            else:
                stopped_scores.append(_row(30, label, "20", mid, name=label, stopped=True, bookmaker_id=2))
    suspended = parse_odds_rows(stopped_scores, kickoff=kickoff, home_id=1, away_id=2)
    assert suspended["ht_correct_score"]["history"] is False
    assert suspended["ht_correct_score"]["opening"] is None
    assert suspended["opening_and_closing"] is False
    raw = [1 / 6.0] + [1 / 20.0] * 24
    fair = devig_vector(raw, ("proportional",))
    assert suspended["ht_correct_score"]["closing"]["probs"]["0-0"] == pytest.approx(float(fair["proportional"][0]))
    early_fair = devig_vector([1 / 5.50] + [1 / 20.0] * 24, ("proportional",))
    assert suspended["ht_correct_score"]["closing"]["probs"]["0-0"] != pytest.approx(float(early_fair["proportional"][0]), abs=1e-4)

    stopped_htft = []
    for code in _NINE:
        stopped_htft.append(_row(29, code, "8", early, name=code, stopped=True, bookmaker_id=2))
        late_price = "4.0" if code == "1/1" else "12"
        stopped_htft.append(_row(29, code, late_price, mid, name=code, stopped=True, bookmaker_id=2))
    batches = parse_odds_rows(stopped_htft, kickoff=kickoff, home_id=1, away_id=2)
    assert set(batches["htft"]["closing"]["probs"]) == set(_NINE)
    assert batches["htft"]["closing"]["complete"] is True
    assert batches["htft"]["history"] is False
    assert batches["htft"]["opening"] is None
    assert batches["htft"]["closing"]["probs"]["1/1"] > batches["htft"]["closing"]["probs"]["X/X"]

    stored = {
        "htft": {
            "closing": {
                "probs": {"X/1": 0.2, "1/X": 0.2, "X/X": 0.2, "X/2": 0.2, "2/X": 0.2},
                "complete": True,
            }
        },
        "correct_score": {"closing": {"probs": {"0-0": 0.4, "1-1": 0.3, "2-2": 0.2, "3-3": 0.1}, "complete": True}},
        "ht_correct_score": {"closing": {"probs": {}, "complete": True}},
    }
    direct = direct_probabilities(stored)
    assert direct["htft"]["complete"] is False
    assert direct["correct_score"]["complete"] is False
    assert "ht_correct_score" not in direct
    scored = []
    for index in range(6):
        scored.append(
            {
                "direct": direct,
                "htft_code": "X/X",
                "score": (1, 1),
                "y_ht": 1,
            }
        )
    block = _direct_block(scored)
    assert block["htft"]["n"] == 0
    assert block["htft"]["log_loss"] is None
    assert block["correct_score"]["n"] == 0


def test_match_nul_maps_to_draw():
    stamp = "2026-02-22T12:00:00Z"
    kickoff = "2026-02-22T20:00:00Z"
    from laliga.data.quotes import _htft_from_teams

    assert _htft_from_teams("Match nul - Real Sociedad", "Real Sociedad", "Real Oviedo") == "X/1"
    assert _htft_from_teams("Unentschieden - Valencia", "Real Sociedad", "Valencia") == "X/2"
    rows = [_row(29, "Match nul - Real Sociedad", "6.50", stamp, name=None, original_label=None, bookmaker_id=2)]
    bundle = parse_odds_rows(
        _with_other_htft(rows, stamp, {"X/1"}),
        kickoff=kickoff,
        home_id=1,
        away_id=2,
        home_name="Real Sociedad",
        away_name="Real Oviedo",
    )
    assert "X/1" in bundle["htft"]["closing"]["probs"]
    assert bundle["htft"]["closing"]["complete"] is True
    assert len(bundle["htft"]["closing"]["probs"]) == 9
    prices = aggregate_prematch_1x2(
        [
            _row(1, "Home", "2.10", stamp),
            _row(1, "Match nul", "3.40", stamp),
            _row(1, "Away", "3.80", stamp),
        ],
        kickoff=kickoff,
    )
    assert prices["raw_implied_draw"] == pytest.approx(1 / 3.40)
