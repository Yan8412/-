"""Point-in-time limit-up features, tercile cuts, and the Alpha158-style factors."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pytest

from ashare.backtest.walkforward import Fold
from ashare.cli import build_parser
from ashare.factor_screen import CLASS_CONFIRMED, CLASS_NOISE, CLASS_REVERSED, CLASS_TRAIN_ONLY, classify_ic, newey_west_t
from ashare.factors import FACTOR_NAMES, compute_one, ref_array, rolling_slope
from ashare.market import RawBar, build_symbol
from ashare.ranker import FEATURE_NAMES, feature_names, feature_vector
from ashare.rules.limits import limit_prices
from ashare.sentiment import MarketDay, MarketPanel, build_market_panel
from ashare.terciles import (
    TaggedTrade,
    WalkForwardTerciles,
    assign_tercile,
    iter_folds_with_tail,
    select_gate,
)
from tests.helpers import flat_symbol, loose_settings, trading_days


def _limit_close(previous: float, code: str = "600000") -> float:
    up, _down = limit_prices(previous, code)
    return float(up)


def _path(closes: list[float], code: str = "600000"):
    days = trading_days(len(closes))
    bars = []
    for day, close in zip(days, closes, strict=True):
        bars.append(RawBar(day, close, close, close, close, 1_000_000, close * 1_000_000))
    return build_symbol(code, "测试", bars)


def _market_day(**overrides) -> MarketDay:
    payload = dict(
        breadth=0.5,
        breadth_count=300,
        limit_up_count=10,
        broken_count=2,
        broken_rate=0.2,
        max_height=2,
        prev_limit_return=0.01,
        index_level=100.0,
        index_vs_ma=0.01,
        risk_on=True,
    )
    payload.update(overrides)
    return MarketDay(**payload)


def test_promotion_premium_and_ladder_use_only_that_day():
    # Two names leave a 1-board yesterday. One promotes, one does not.
    base = [10.0] * 8
    a_day = _limit_close(10.0)
    a_next = _limit_close(a_day)
    failed = round(a_day * 1.02, 2)
    left = _path(base + [a_day, a_next], "600001")
    right = _path(base + [a_day, failed], "600002")
    fresh = _path(base + [10.0, _limit_close(10.0)], "600003")
    panel = build_market_panel([left, right, fresh], loose_settings())
    today = left.dates[-1]
    info = panel.days[today]
    assert info.promo_1_2 == pytest.approx(0.5)
    assert math_nan(info.promo_2_3)
    assert math_nan(info.board_premium)
    assert info.ladder_gap == 0
    assert info.limit_up_count == 2

    # Yesterday's 2-board name seals again: 2→3 is 1, and it is the whole premium.
    third = _limit_close(a_next)
    taller = _path(base + [a_day, a_next, third], "600001")
    panel = build_market_panel([taller], loose_settings())
    sealed = panel.days[taller.dates[-1]]
    assert sealed.promo_2_3 == pytest.approx(1.0)
    assert sealed.board_premium == pytest.approx(third / a_next - 1.0)

    # Heights 1 and 4 on the same session, nothing on 2 or 3.
    price = 10.0
    tall = []
    for _ in range(4):
        price = _limit_close(price)
        tall.append(price)
    ladder_short = _path([10.0] * 6 + [_limit_close(10.0)], "600011")
    ladder_tall = _path([10.0] * 3 + tall, "600012")
    # Align the last bars by giving both the same number of sessions.
    assert ladder_short.dates[-1] == ladder_tall.dates[-1]
    ladder = build_market_panel([ladder_short, ladder_tall], loose_settings())
    last = ladder.days[ladder_short.dates[-1]]
    assert last.max_height == 4
    assert last.ladder_gap == 2


def test_a_skipped_session_is_not_a_promotion():
    days = trading_days(6)
    kept = [days[0], days[1], days[2], days[3], days[5]]
    closes = [10.0, 10.0, 10.0, _limit_close(10.0), _limit_close(_limit_close(10.0))]
    bars = [
        RawBar(day, close, close, close, close, 1_000_000, close * 1_000_000)
        for day, close in zip(kept, closes, strict=True)
    ]
    series = build_symbol("600000", "跳日", bars)
    other = _path([10.0] * 6, "600099")
    panel = build_market_panel([series, other], loose_settings())
    assert math_nan(panel.days[days[5]].promo_1_2)


def test_new_sentiment_fields_ignore_later_bars():
    closes = [10.0] * 12 + [_limit_close(10.0), _limit_close(_limit_close(10.0))]
    series = _path(closes, "600000")
    day = series.dates[-1]
    before = build_market_panel([series], loose_settings()).days[day]
    extended = _path(closes + [_limit_close(closes[-1])], "600000")
    after = build_market_panel([extended], loose_settings()).days[day]
    assert after == before
    features = feature_vector(series, len(series.close) - 1, "ma_pullback", 3.0, build_market_panel([series], loose_settings()))
    assert features.shape == (len(FEATURE_NAMES),)
    assert "promo_1_2" in FEATURE_NAMES
    assert "ladder_gap" in FEATURE_NAMES
    again = feature_vector(extended, len(series.close) - 1, "ma_pullback", 3.0, build_market_panel([extended], loose_settings()))
    assert np.allclose(features, again)


def test_extra_factor_columns_follow_the_name_list():
    series = flat_symbol("600000", [10.0] * 20)
    panel = build_market_panel([series], loose_settings())
    base = feature_vector(series, 19, "ma_pullback", 1.0, panel)
    extra = feature_vector(
        series,
        19,
        "ma_pullback",
        1.0,
        panel,
        extra_factors=("KMID",),
        extra_values={"KMID": float("nan")},
    )
    assert extra.shape == (len(feature_names(("KMID",))),)
    assert len(extra) == len(base) + 2
    assert extra[-2] == 0.0
    assert extra[-1] == 1.0


def test_factor_formulas_match_qlib_and_do_not_read_the_future():
    # Close rises 10 per day. qlib's own comment: the slope is 10.
    closes = [10.0 + 10.0 * step for step in range(12)]
    series = _path(closes)
    beta = compute_one(series, "BETA10")
    assert beta[9] == pytest.approx(10.0 / closes[9])
    assert rolling_slope(np.asarray(closes[:10]), 10)[-1] == pytest.approx(10.0)
    rank = compute_one(series, "RANK10")
    assert rank[9] == pytest.approx(1.0)
    roc = compute_one(series, "ROC5")
    assert roc[10] == pytest.approx(closes[5] / closes[10])
    # A negative shift would have used closes[15], which does not exist; the past lag does.
    assert np.isnan(roc[4])

    days = trading_days(3)
    bars = [
        RawBar(days[0], 10.0, 10.0, 10.0, 10.0, 100.0, 1_000.0),
        RawBar(days[1], 10.0, 12.0, 9.0, 11.0, 100.0, 1_100.0),
        RawBar(days[2], 11.0, 11.0, 11.0, 11.0, 100.0, 1_100.0),
    ]
    candle = build_symbol("600000", "K线", bars)
    assert compute_one(candle, "KMID")[1] == pytest.approx(0.1)
    assert compute_one(candle, "KLEN")[1] == pytest.approx(0.3)
    assert compute_one(candle, "KUP")[1] == pytest.approx(0.1)
    assert compute_one(candle, "KSFT")[1] == pytest.approx((22.0 - 12.0 - 9.0) / 10.0)
    rising = _path([10.0 + step for step in range(8)])
    assert compute_one(rising, "SUMP5")[5] == pytest.approx(1.0)
    assert compute_one(rising, "CNTD5")[5] == pytest.approx(1.0)
    flat = compute_one(_path([10.0] * 8), "STD5")
    assert flat[4] == pytest.approx(0.0)

    prefix = compute_one(series, "BETA10")
    bumped = _path(closes + [closes[-1] + 80.0])
    assert np.allclose(prefix, compute_one(bumped, "BETA10")[: len(prefix)], equal_nan=True)
    with pytest.raises(ValueError):
        ref_array(np.arange(5.0), -1)
    assert len(FACTOR_NAMES) == 15


def test_newey_west_shrinks_a_persistent_mean_and_labels_are_fixed():
    rng = np.random.default_rng(0)
    shock = rng.normal(size=80)
    series = np.zeros(80)
    series[0] = shock[0]
    for index in range(1, 80):
        series[index] = 0.85 * series[index - 1] + shock[index]
    series = series + 0.4
    _mean, persistent_t, count = newey_west_t(series, 4)
    _mean0, iid_t, _count = newey_west_t(series, 0)
    assert count == 80
    assert persistent_t > 0
    assert abs(persistent_t) < abs(iid_t)
    assert classify_ic(0.02, 3.0, 0.01, 2.5) == CLASS_CONFIRMED
    assert classify_ic(0.02, 3.0, 0.01, 0.4) == CLASS_TRAIN_ONLY
    assert classify_ic(0.02, 2.5, -0.02, -3.0) == CLASS_REVERSED
    assert classify_ic(0.02, 0.5, 0.03, 4.0) == CLASS_NOISE


def test_tercile_cuts_come_from_the_training_window_only():
    days = trading_days(30, date(2024, 1, 2))
    fold = Fold(days[0], days[14], days[15], days[24])
    records = {}
    for index, day in enumerate(days):
        count = index + 1
        if day == days[20]:
            count = 10_000
        records[day] = _market_day(limit_up_count=count, risk_on=True)
    panel = MarketPanel(days=records, calendar=days, limit_up={}, height={})
    walker = WalkForwardTerciles(panel, days, [fold], features=("limit_up_count",))
    assert walker.bucket(days[15], "limit_up_count") == 2
    records[days[20]] = _market_day(limit_up_count=0, risk_on=True)
    untouched = WalkForwardTerciles(panel, days, [fold], features=("limit_up_count",))
    # days[15] is inside the test window but its own value is still the train-relative one.
    # Replacing a later test day must not move the cut.
    assert untouched.bucket(days[15], "limit_up_count") == walker.bucket(days[15], "limit_up_count")
    sample = np.arange(1, 31, dtype=float)
    assert assign_tercile(1.0, sample) == 0
    assert assign_tercile(15.0, sample) == 1
    assert assign_tercile(100.0, sample) == 2
    assert assign_tercile(1.0, np.arange(3.0)) is None


def test_gate_is_chosen_from_the_trades_it_is_given():
    days = trading_days(40, date(2024, 1, 2))
    fold = Fold(days[0], days[19], days[20], days[39])
    records = {}
    for index, day in enumerate(days[:20]):
        records[day] = _market_day(limit_up_count=index + 1, broken_rate=0.2, promo_1_2=0.4)
    for offset, day in enumerate(days[20:30]):
        records[day] = _market_day(limit_up_count=1, broken_rate=0.2, promo_1_2=0.4)
    for day in days[30:]:
        records[day] = _market_day(limit_up_count=500, broken_rate=0.2, promo_1_2=0.4)
    panel = MarketPanel(days=records, calendar=days, limit_up={}, height={})
    walker = WalkForwardTerciles(panel, days, [fold])
    trades = []
    for day in days[20:30]:
        trades.append(TaggedTrade(day, day + timedelta(days=1), pnl=-20.0, buy_cost=5000.0))
    for day in days[30:]:
        trades.append(TaggedTrade(day, day + timedelta(days=1), pnl=5.0, buy_cost=5000.0))
    # The high side has only 10 names, under the 15-trade floor, so no rule.
    blocked, _rows = select_gate(trades, walker, min_side=15)
    assert blocked is None
    gate, _rows = select_gate(trades, walker, min_side=10)
    assert gate is not None
    assert gate.feature == "limit_up_count"
    assert gate.drop_tercile == 0


def test_tail_fold_does_not_swallow_its_own_test_days():
    days = trading_days(25, date(2024, 1, 2))
    folds = iter_folds_with_tail(days, train_days=10, test_days=10)
    assert folds[-1].test_start == days[20]
    assert folds[-1].train_end < folds[-1].test_start
    assert folds[-1].test_end == days[-1]


def test_study_command_is_available():
    args = build_parser().parse_args(["study"])
    assert args.command == "study"
    assert args.report_dir is None


def math_nan(value: float) -> bool:
    return isinstance(value, float) and value != value
