"""Walk-forward windows stay causal, and signals ignore future bars."""

from datetime import timedelta

from ashare.backtest.metrics import max_drawdown
from ashare.backtest.walkforward import iter_folds, select_params
from ashare.market import RawBar, build_symbol
from ashare.rules.limits import is_limit_up
from ashare.strategies.library import FirstBoardFollow
from tests.helpers import flat_symbol, loose_settings, trading_days
from tests.test_execution import Scripted


def test_folds_do_not_overlap_and_train_ends_before_test():
    days = trading_days(300)
    folds = iter_folds(days, train_days=100, test_days=50)
    assert len(folds) == 4
    covered = []
    for fold in folds:
        assert fold.train_end < fold.test_start
        assert (fold.test_end - fold.test_start).days >= 0
        covered.append((fold.test_start, fold.test_end))
    for left, right in zip(covered, covered[1:]):
        assert left[1] < right[0]


def test_param_selection_does_not_read_the_test_window():
    series = flat_symbol("000001", [10.0] * 80)
    seen: list = []

    class Spy(Scripted):
        def signal(self, series, index, params):
            seen.append(series.dates[index])
            return None

    settings = loose_settings(min_trades_for_selection=5)
    fold = iter_folds(series.dates, train_days=40, test_days=20)[0]
    select_params([series], Spy(set()), settings, fold)
    assert seen
    assert max(seen) <= fold.train_end
    assert all(day < fold.test_start for day in seen)


def test_later_dividend_does_not_rewrite_an_earlier_signal():
    days = trading_days(8)
    bars = [
        RawBar(day, 10, 10.2, 9.8, 10.0, 2_000_000, 80_000_000) for day in days[:5]
    ]
    # Day 1 limit-up, day 2 bullish follow-through.
    bars[1] = RawBar(days[1], 10.2, 11.0, 10.2, 11.0, 2_000_000, 80_000_000)
    bars[2] = RawBar(days[2], 11.05, 11.4, 11.0, 11.2, 1_500_000, 80_000_000)
    before = build_symbol("000001", "测试股份", bars)
    strategy = FirstBoardFollow()
    first = strategy.signal(before, 2, strategy.default_params())
    assert first is not None
    assert is_limit_up(before.close[1], before.limit_up[1])
    extra = RawBar(
        days[4] + timedelta(days=3),
        8,
        8,
        8,
        8,
        2_000_000,
        80_000_000,
        cash_dividend=1.0,
        bonus_ratio=0.5,
    )
    # days[4] is the 5th bar; pick a later calendar day already in `days`.
    later = bars + [
        RawBar(days[6], 8, 8.2, 7.8, 8.0, 2_000_000, 80_000_000, cash_dividend=1.0, bonus_ratio=0.5)
    ]
    after = build_symbol("000001", "测试股份", later)
    assert after.qfq_close[2] == before.qfq_close[2]
    second = strategy.signal(after, 2, strategy.default_params())
    assert second is not None
    assert second.reason == first.reason
    del extra


def test_drawdown_is_peak_to_trough():
    assert abs(max_drawdown([100, 80, 90, 70]) + 0.3) < 1e-9
    assert max_drawdown([10, 11, 12]) == 0
