"""Full-market filters, vectorized signals, and catalog merge. No network."""

from datetime import date, datetime, timedelta, timezone

import numpy as np

from ashare.backtest.engine import _collect_signals, _signal_book
from ashare.data.cache import cache_is_fresh, save_bars
from ashare.data.universe import merge_names
from ashare.filters import eligible_mask, rejection_reason
from ashare.market import RawBar, build_symbol
from ashare.rules.limits import classify_board
from ashare.rules.money import to_cents
from ashare.strategies.library import builtin_strategies
from tests.helpers import flat_symbol, loose_settings, trading_days


def test_chinext_302_uses_the_growth_board_limit():
    assert classify_board("302132") == "chinext"


def test_low_swing_is_rejected_without_a_name_blacklist():
    settings = loose_settings(min_history_bars=20, min_listed_bars=5, min_amount=1, min_swing_20d=0.10)
    quiet = flat_symbol("601398", [8.0 + i * 0.01 for i in range(30)], name="工商银行")
    assert rejection_reason(quiet, 24, settings) == "近20日振幅不足"
    days = trading_days(30)
    bars = []
    for index, day in enumerate(days):
        base = 10.0 + (index % 5) * 0.8
        bars.append(RawBar(day, base, base + 0.6, base - 0.6, base + 0.2, 2_000_000, 80_000_000))
    active = build_symbol("000001", "测试股份", bars)
    assert rejection_reason(active, 24, settings) is None


def test_eligible_mask_matches_rejection_reason():
    settings = loose_settings(min_history_bars=20, min_listed_bars=5, min_amount=1_000_000, min_swing_20d=0.08)
    series = _noisy("000001", 80, seed=3)
    series.left_censored = True
    mask = eligible_mask(series, settings)
    for index in range(len(series.dates)):
        allowed = rejection_reason(series, index, settings) is None
        assert bool(mask[index]) is allowed


def test_half_up_cents_match_decimal_helper():
    from ashare.filters import _cents

    values = np.concatenate(
        [
            np.linspace(0.5, 80.0, 400),
            np.array([1.005, 1.015, 10.005, 10.125, 13.375, 8.125]),
        ]
    )
    got = _cents(values)
    for price, cents in zip(values, got, strict=True):
        assert int(cents) == to_cents(float(price))


def test_vectorized_scores_match_bar_by_bar_signals():
    series = _noisy("300001", 160, seed=7)
    for strategy in builtin_strategies():
        for params in strategy.param_grid():
            bulk = strategy.scores(series, params)
            for index in range(len(series.dates)):
                signal = strategy.signal(series, index, params)
                if signal is None:
                    assert np.isnan(bulk[index])
                else:
                    assert abs(bulk[index] - signal.score) < 1e-8


def test_fast_book_matches_the_daily_collector():
    settings = loose_settings(
        min_history_bars=20,
        min_listed_bars=5,
        min_amount=1,
        min_swing_20d=0.05,
        top_n=5,
    )
    symbols = [_noisy("000001", 120, seed=1), _noisy("300750", 120, seed=2, name="宁德时代")]
    bundled = [(item, item.default_params()) for item in builtin_strategies()]
    book = _signal_book(symbols, bundled, settings, None)
    by_code = {item.code: item for item in symbols}
    for day in symbols[0].dates:
        expected = _collect_signals(by_code, day, bundled, settings, None)
        got = book.get(day, [])
        assert [(item.code, item.strategy_id, round(item.score, 6)) for item in got] == [
            (item.code, item.strategy_id, round(item.score, 6)) for item in expected
        ]


def test_merge_keeps_delisted_names_and_current_sina_names():
    merged = merge_names(
        [("600000", "浦发银行"), ("200001", "B股")],
        [
            {
                "code": "600000",
                "name": "旧名",
                "ipo_date": "1999-11-10",
                "out_date": None,
                "listed": True,
            },
            {
                "code": "600001",
                "name": "邯郸钢铁",
                "ipo_date": "1998-01-22",
                "out_date": "2009-12-15",
                "listed": False,
            },
        ],
    )
    by_code = {item.code: item for item in merged}
    assert "200001" not in by_code
    assert by_code["600000"].name == "浦发银行"
    assert by_code["600000"].ipo_date == "1999-11-10"
    assert by_code["600001"].listed is False
    assert by_code["600001"].out_date == "2009-12-15"


def test_delisted_cache_is_fresh_when_the_file_is_new(tmp_path):
    bars = []
    day = date(2024, 1, 2)
    while len(bars) < 60:
        if day.weekday() < 5:
            bars.append(RawBar(day, 10, 10.2, 9.8, 10, 1000, 10000))
        day += timedelta(days=1)
    save_bars(tmp_path, "600001", bars, "tencent")
    from ashare.data.cache import load_bars

    loaded, fetched_at = load_bars(tmp_path, "600001")
    assert cache_is_fresh(loaded, fetched_at, date(2026, 9, 27), 60)
    stale = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat(timespec="seconds")
    assert cache_is_fresh(loaded, stale, date(2026, 9, 27), 60) is False


def _noisy(code: str, count: int, seed: int, name: str = "测试股份"):
    rng = np.random.default_rng(seed)
    days = trading_days(count)
    price = 12.0
    bars = []
    for index, day in enumerate(days):
        shock = float(rng.normal(0, 0.03))
        close = max(2.0, price * (1.0 + shock))
        high = max(price, close) * (1.0 + abs(float(rng.normal(0, 0.01))))
        low = min(price, close) * (1.0 - abs(float(rng.normal(0, 0.01))))
        volume = float(rng.integers(200_000, 2_000_000))
        bars.append(RawBar(day, price, high, low, close, volume, close * volume))
        price = close
        if index == 40:
            # A sealed limit-up so the first-board rule has something to see.
            up = close * 1.1
            bars[-1] = RawBar(day, close, up, close, up, volume * 3, up * volume * 3)
            price = up
    return build_symbol(code, name, bars)
