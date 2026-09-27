"""T+1, limit-up buys, limit-down sells, entry bands, and slippage."""

from ashare.backtest.engine import run_backtest
from ashare.backtest.metrics import summarize
from ashare.filters import rejection_reason
from ashare.market import RawBar, SymbolSeries, build_symbol, mark_listing_censorship
from ashare.rules.execution import evaluate_buy, evaluate_sell
from ashare.rules.money import D
from ashare.strategies.base import Signal, Strategy
from tests.helpers import flat_symbol, loose_settings, trading_days


class Scripted(Strategy):
    id = "scripted"
    name = "脚本策略"
    summary = "测试用"
    warmup = 0

    def __init__(self, on_dates: set, **signal_kwargs) -> None:
        self.on_dates = on_dates
        self.signal_kwargs = signal_kwargs

    def default_params(self) -> dict:
        return {}

    def param_grid(self) -> list[dict]:
        return [{}]

    def signal(self, series, index: int, params: dict):
        if series.dates[index] not in self.on_dates:
            return None
        defaults = dict(
            score=10,
            reason="测试信号",
            take_profit_pct=0.5,
            stop_loss_pct=0.05,
            max_hold_days=10,
            entry_low_pct=-0.05,
            entry_high_pct=0.05,
        )
        defaults.update(self.signal_kwargs)
        return Signal(**defaults)


def _ohlc(series_prices, spec: dict[int, tuple[float, float, float, float]]) -> SymbolSeries:
    days = trading_days(len(series_prices))
    bars = []
    for index, (day, price) in enumerate(zip(days, series_prices, strict=True)):
        open_, high, low, close = spec.get(index, (price, price, price, price))
        bars.append(
            RawBar(
                date=day,
                open=open_,
                high=high,
                low=low,
                close=close,
                volume=2_000_000,
                amount=80_000_000,
            )
        )
    return build_symbol("000001", "测试股份", bars)


def test_buy_rejected_at_limit_up_and_outside_the_band():
    blocked = evaluate_buy(
        open_=11, low=11, limit_up=11, entry_low=9, entry_high=12, slippage=0
    )
    assert not blocked.ok
    assert blocked.reason == "涨停无法买入"
    high_open = evaluate_buy(
        open_=10.8, low=10.5, limit_up=11, entry_low=9.5, entry_high=10.2, slippage=0
    )
    assert high_open.reason == "高开超出买入区间"
    slipped = evaluate_buy(
        open_=10, low=9.8, limit_up=11, entry_low=9, entry_high=10.5, slippage="0.001"
    )
    assert slipped.ok
    assert slipped.price == D("10.01")


def test_sell_stop_gap_intraday_and_limit_down():
    gap = evaluate_sell(
        open_=9.2,
        high=9.4,
        low=9.1,
        close=9.3,
        limit_up=11,
        limit_down=9,
        stop=9.5,
        take_profit=11,
        days_held=1,
        max_hold=5,
        slippage=0,
        phase="open",
    )
    assert gap.ok and gap.price == D("9.20") and gap.reason == "stop"
    intraday = evaluate_sell(
        open_=10,
        high=10.2,
        low=9.4,
        close=9.8,
        limit_up=11,
        limit_down=9,
        stop=9.5,
        take_profit=12,
        days_held=1,
        max_hold=5,
        slippage=0,
        phase="late",
    )
    assert intraday.price == D("9.50")
    sealed = evaluate_sell(
        open_=9,
        high=9,
        low=9,
        close=9,
        limit_up=11,
        limit_down=9,
        stop=9.5,
        take_profit=11,
        days_held=2,
        max_hold=1,
        slippage=0,
        phase="late",
    )
    assert not sealed.ok
    assert sealed.reason == "跌停无法卖出"
    # Stop and target on the same bar: the stop is filled, not the target.
    both = evaluate_sell(
        open_=10,
        high=11,
        low=9.2,
        close=10.5,
        limit_up=11,
        limit_down=9,
        stop=9.5,
        take_profit=10.8,
        days_held=1,
        max_hold=5,
        slippage=0,
        phase="late",
    )
    assert both.reason == "stop"


def test_t_plus_one_blocks_same_day_stop_and_limit_rules_in_the_engine():
    prices = [10.0] * 16
    # Index 11 is the buy day. A crash on that low must not sell (T+1).
    # Index 12 trades through the 5% stop.
    series = _ohlc(
        prices,
        {
            11: (10.0, 10.0, 8.0, 10.0),
            12: (10.0, 10.0, 8.0, 9.0),
        },
    )
    signal_day = series.dates[10]
    settings = loose_settings()
    result = run_backtest(
        [series],
        [(Scripted({signal_day}), {})],
        settings,
        series.dates[0],
        series.dates[-1],
    )
    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.buy_date == series.dates[11]
    assert trade.sell_date == series.dates[12]
    assert trade.sell_date != trade.buy_date
    # A 5,000 slot cannot hold 500 shares after the 5 CNY commission floor
    # (5,005.05), so the sizer steps down to 400 shares.
    assert trade.shares == 400
    assert trade.shares % 100 == 0
    assert trade.buy_price == 10
    assert trade.sell_price == 9.5
    assert trade.reason == "stop"
    assert trade.buy_cost == 4005.04
    assert trade.sell_proceeds == 3793.06
    metrics = summarize(result)
    assert metrics.trade_count == 1
    assert metrics.win_rate == 0


def test_limit_up_open_does_not_open_a_position():
    prices = [10.0] * 16
    series = _ohlc(prices, {11: (11.0, 11.0, 11.0, 11.0)})
    settings = loose_settings()
    result = run_backtest(
        [series],
        [(Scripted({series.dates[10]}), {})],
        settings,
        series.dates[0],
        series.dates[-1],
    )
    assert result.trades == []
    assert result.open_positions == 0
    assert result.rejects.get("涨停无法买入", 0) == 1
    assert result.final_equity == settings.initial_capital


def test_sealed_limit_down_defers_the_sell():
    prices = [10.0] * 16
    series = _ohlc(
        prices,
        {
            12: (9.0, 9.0, 9.0, 9.0),
        },
    )
    settings = loose_settings()
    result = run_backtest(
        [series],
        [(Scripted({series.dates[10]}, max_hold_days=1, stop_loss_pct=0.5), {})],
        settings,
        series.dates[0],
        series.dates[-1],
    )
    assert len(result.trades) == 1
    assert result.trades[0].buy_date == series.dates[11]
    assert result.trades[0].sell_date == series.dates[13]
    assert result.rejects.get("跌停无法卖出", 0) >= 1


def test_filters_drop_st_new_suspended_and_limit_up():
    settings = loose_settings(min_history_bars=3, min_listed_bars=10, min_amount=1)
    series = flat_symbol("000001", [10.0] * 12, name="ST测试")
    assert rejection_reason(series, 5, settings) == "ST或风险警示"

    normal = flat_symbol("000001", [10.0] * 12, name="普通股份")
    normal.left_censored = False
    assert rejection_reason(normal, 5, settings) == "次新股"

    halted = flat_symbol("000001", [10.0] * 12)
    halted.volume[8] = 0
    halted.amount[8] = 0
    assert rejection_reason(halted, 8, settings) == "停牌"

    prices = [10.0] * 12
    days = trading_days(12)
    bars = []
    for index, (day, price) in enumerate(zip(days, prices, strict=True)):
        close = 11.0 if index == 8 else price
        bars.append(
            RawBar(day, price, max(price, close), price, close, 2_000_000, 80_000_000)
        )
    limit_series = build_symbol("000001", "普通股份", bars)
    assert rejection_reason(limit_series, 8, settings) == "收盘涨停无法买入"
    mark_listing_censorship([normal])
