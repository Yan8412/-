"""Market quotes, n-way de-vig, implied goals, and football-data totals."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from laliga.data.football_data import build_history, parse_seasons
from laliga.data.probe import format_probe, probe_recent
from laliga.data.quotes import direct_probabilities, parse_odds_rows, targets_from_bundle, targets_from_row
from laliga.model.devig import devig, devig_vector, raw_implied_from_decimal
from laliga.model.dixon_coles import outcome_probabilities, score_matrix
from laliga.model.implied_goals import MarketTargets, _over_prob, infer_rates

_CSV = """Div,Date,Time,HomeTeam,AwayTeam,FTHG,FTAG,HTHG,HTAG,B365H,B365D,B365A,B365>2.5,B365<2.5,AHh,B365AHH,B365AHA,B365C>2.5,B365C<2.5,AHCh,B365CAHH,B365CAHA
SP1,01/09/2024,18:00,Ath Madrid,Barcelona,1,0,0,0,2.10,3.40,3.60,1.90,1.90,-0.25,1.95,1.85,1.80,2.00,-0.5,2.05,1.80
"""

_BARE = """Div,Date,Time,HomeTeam,AwayTeam,FTHG,FTAG,HTHG,HTAG,B365H,B365D,B365A
SP1,08/09/2024,16:00,Sociedad,Valencia,0,0,0,0,2.40,3.20,3.10
"""


def _odd(market_id, label, value, stamp, **extra):
    row = {
        "market_id": market_id,
        "label": label,
        "value": value,
        "bookmaker_id": 2,
        "latest_bookmaker_update": stamp,
    }
    row.update(extra)
    return row


def _rows():
    early = "2026-10-17T12:00:00Z"
    late = "2026-10-18T17:00:00Z"
    rows = []
    for stamp, prices in ((early, ("2.20", "3.40", "3.30")), (late, ("2.05", "3.50", "3.60"))):
        for label, price in zip(("home", "draw", "away"), prices, strict=True):
            rows.append(_odd(1, label, price, stamp))
    rows.append(_odd(1, "home", "1.10", "2026-10-18T19:30:00Z"))
    for stamp, over, under in ((early, "1.95", "1.85"), (late, "1.80", "2.00")):
        rows.append(_odd(80, "over", over, stamp, total=2.5))
        rows.append(_odd(80, "under", under, stamp, total=2.5))
    rows.append(_odd(6, "home", "1.90", late, handicap=-0.25))
    rows.append(_odd(6, "away", "1.90", late, handicap=0.25))
    for code, price in (
        ("1/1", "4.5"),
        ("1/X", "15"),
        ("1/2", "30"),
        ("X/1", "8"),
        ("X/X", "6"),
        ("X/2", "9"),
        ("2/1", "25"),
        ("2/X", "14"),
        ("2/2", "5"),
    ):
        rows.append(_odd(29, code, price, late))
    rows.append(_odd(57, "1-0", "7.0", late))
    rows.append(_odd(57, "1-1", "7.5", late))
    return rows


def test_opening_is_the_earlier_price_and_kickoff_quotes_are_dropped():
    bundle = parse_odds_rows(_rows(), kickoff="2026-10-18T19:00:00Z", home_id=10, away_id=20)
    assert bundle["opening_and_closing"] is True
    assert 1 in bundle["markets_seen"]
    assert 80 in bundle["markets_seen"]
    assert 6 in bundle["markets_seen"]
    assert 29 in bundle["markets_seen"]
    opening = bundle["ft_1x2"]["opening"]["raw"]
    closing = bundle["ft_1x2"]["closing"]["raw"]
    assert opening["home"] == pytest.approx(1 / 2.20)
    assert closing["home"] == pytest.approx(1 / 2.05)
    assert bundle["bookmakers"] == [2]
    targets = targets_from_bundle(bundle)
    assert targets is not None
    assert targets.ft_probs is not None
    assert sum(targets.ft_probs) == pytest.approx(1.0)
    assert targets.ou and targets.ou[0][0] == pytest.approx(2.5)
    assert targets.ah and targets.ah[0][0] == pytest.approx(-0.25)
    direct = direct_probabilities(bundle)
    assert direct["htft"]["complete"] is True
    assert "1-0" in direct["correct_score"]["probs"]
    text = str(bundle)
    assert "api_token" not in text


def test_shin_vector_matches_the_three_way_and_skips_two_outcomes():
    odds = np.array([2.6, 2.4, 4.3])
    raw = raw_implied_from_decimal(odds)[0]
    assert devig_vector(raw, ("shin",))["shin"] == pytest.approx(devig(raw)["shin"][0])
    four = devig_vector(np.array([0.40, 0.30, 0.25, 0.20]), ("shin", "proportional", "power"))
    assert four["shin"].sum() == pytest.approx(1.0)
    assert four["proportional"].sum() == pytest.approx(1.0)
    assert "power" in four
    two = devig_vector(np.array([0.55, 0.52]), ("shin", "proportional"))
    assert "shin" not in two
    assert two["proportional"].sum() == pytest.approx(1.0)


def test_implied_goals_recover_known_rates_from_1x2_and_totals():
    lam_home, lam_away = 1.55, 1.05
    matrix = score_matrix(lam_home, lam_away, -0.05, 8)
    share = 0.42
    first = score_matrix(lam_home * share, lam_away * share, -0.05, 8)
    rates = infer_rates(
        MarketTargets(
            ft_probs=outcome_probabilities(matrix),
            ou=[(2.5, _over_prob(matrix, 2.5))],
            ht_ou=[(0.5, _over_prob(first, 0.5)), (1.5, _over_prob(first, 1.5))],
        )
    )
    assert rates is not None
    assert rates.lam_home == pytest.approx(lam_home, abs=0.25)
    assert rates.lam_away == pytest.approx(lam_away, abs=0.25)
    assert rates.ht_share == pytest.approx(share, abs=0.12)
    assert "ou" in rates.used and "ht_ou" in rates.used
    forecast = rates.forecast(max_half=5, max_ft=10)
    assert sum(forecast.htft.values()) == pytest.approx(1.0)
    assert len(forecast.top2_scores) == 2


def test_row_without_totals_still_prices_1x2_and_optional_columns_stay_missing():
    bare = parse_seasons([_BARE], [2024]).iloc[0]
    assert pd.isna(bare["ou25_pre_over"])
    assert pd.isna(bare["ah_pre_line"])
    targets = targets_from_row({"odds_home": 2.10, "odds_draw": 3.40, "odds_away": 3.60})
    assert targets is not None
    assert targets.ou == []
    assert targets.ah == []
    assert sum(targets.ft_probs) == pytest.approx(1.0)


def test_football_data_reads_over_under_and_asian_handicap():
    parsed = parse_seasons([_CSV], [2024])
    row = parsed.iloc[0]
    built = build_history(parsed, pd.DataFrame()).frame.iloc[0]
    assert row["ou25_pre_over"] == pytest.approx(1.90)
    assert row["ou25_pre_under"] == pytest.approx(1.90)
    assert row["ou25_close_over"] == pytest.approx(1.80)
    assert row["ah_pre_line"] == pytest.approx(-0.25)
    assert row["ah_pre_home"] == pytest.approx(1.95)
    assert row["ah_close_line"] == pytest.approx(-0.5)
    targets = targets_from_row(built)
    assert targets is not None
    assert built["fair_book"] == "bet365_pre"
    assert targets.ou[0][0] == pytest.approx(2.5)
    assert targets.ah[0][0] == pytest.approx(-0.25)
    assert 0.0 < targets.ou[0][1] < 1.0


class _ProbeClient:
    def __init__(self):
        self.calls = []

    def paginate(self, path, params=None, **_kwargs):
        self.calls.append((path, dict(params or {})))
        if path.startswith("fixtures/between"):
            yield {
                "id": 9001,
                "starting_at": "2026-10-20T19:00:00Z",
                "participants": [
                    {"id": 10, "name": "Home", "meta": {"location": "home"}},
                    {"id": 20, "name": "Away", "meta": {"location": "away"}},
                ],
            }
            return
        include = (params or {}).get("include")
        if include == "odds":
            yield {
                "id": 9001,
                "odds": [
                    {
                        "market_id": 1,
                        "label": "home",
                        "value": "2.10",
                        "bookmaker_id": 2,
                        "latest_bookmaker_update": "2026-10-20T12:00:00Z",
                        "api_token": "super-secret-token",
                    },
                    {
                        "market_id": 1,
                        "label": "draw",
                        "value": "3.30",
                        "bookmaker_id": 2,
                        "latest_bookmaker_update": "2026-10-20T12:00:00Z",
                    },
                    {
                        "market_id": 1,
                        "label": "away",
                        "value": "3.40",
                        "bookmaker_id": 2,
                        "latest_bookmaker_update": "2026-10-20T12:00:00Z",
                    },
                    {
                        "market_id": 80,
                        "label": "over",
                        "value": "1.90",
                        "total": 2.5,
                        "bookmaker_id": 2,
                        "latest_bookmaker_update": "2026-10-20T12:00:00Z",
                    },
                    {
                        "market_id": 80,
                        "label": "under",
                        "value": "1.90",
                        "total": 2.5,
                        "bookmaker_id": 2,
                        "latest_bookmaker_update": "2026-10-20T12:00:00Z",
                    },
                ],
            }
            return
        if include == "xGFixture":
            yield {
                "id": 9001,
                "xGFixture": [
                    {"type_id": 5304, "location": "home", "participant_id": 10, "data": {"value": 1.4}},
                    {"type_id": 5304, "location": "away", "participant_id": 20, "data": {"value": 0.9}},
                ],
            }


def test_probe_reports_markets_and_does_not_print_a_token(monkeypatch):
    monkeypatch.delenv("SPORTMONKS_API_TOKEN", raising=False)
    client = _ProbeClient()
    report = probe_recent(client, days=7, limit=3)
    assert report["fixtures"]
    seen = report["fixtures"][0]["markets_seen"]
    assert 1 in seen and 80 in seen
    assert report["fixtures"][0]["xg"]["type_5304"] is True
    assert report["fixtures"][0]["xg"]["home"] == pytest.approx(1.4)
    text = format_probe(report)
    assert "super-secret-token" not in text
    assert "全场胜平负" in text
    assert "SPORTMONKS_API_TOKEN" not in text
    filters = [params.get("filters") for _path, params in client.calls if params.get("include") == "odds"]
    assert filters and "markets:1" in filters[0] and "80" in filters[0]


def test_probe_cli_without_a_token_exits(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("SPORTMONKS_API_TOKEN", raising=False)
    from laliga.cli import main

    code = main(["probe", "--data-dir", str(tmp_path)])
    assert code == 2
    assert "SPORTMONKS_API_TOKEN" in capsys.readouterr().err
