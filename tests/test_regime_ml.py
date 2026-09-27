"""Regime gate, limit-up sentiment, ranking labels, and project-root paths."""

from __future__ import annotations

from datetime import date

import numpy as np

from ashare.backtest.engine import PendingBuy, run_backtest
from ashare.cli import build_parser, dashboard_command
from ashare.config import load_settings
from ashare.filters import _cents
from ashare.market import RawBar, build_symbol
from ashare.paths import cache_dir, default_config_path, ledger_path, model_path, project_root, report_dir
from ashare.ranker import (
    FEATURE_NAMES,
    HOLDOUT_START,
    LabeledRow,
    feature_vector,
    paper_verdict,
    pnl_tstat,
    rows_for_training,
    select_hyperparams,
)
from ashare.report import write_recommendations_csv
from ashare.rules.money import D, money
from ashare.sentiment import attach_inferred_st, build_market_panel, infer_st_mask
from ashare.strategies.base import Signal, Strategy
from ashare.trade_outcome import trade_net_return
from tests.helpers import flat_symbol, loose_settings, trading_days


class _Scored(Strategy):
    id = "scripted"
    name = "脚本策略"
    summary = "测试用"
    warmup = 0

    def __init__(self, on_dates: set[date]) -> None:
        self.on_dates = on_dates

    def default_params(self) -> dict:
        return {}

    def param_grid(self) -> list[dict]:
        return [{}]

    def signal(self, series, index: int, params: dict):
        if series.dates[index] not in self.on_dates:
            return None
        score = 100.0 if series.code.endswith("1") else 1.0
        return Signal(
            score=score,
            reason="测试信号",
            take_profit_pct=0.5,
            stop_loss_pct=0.05,
            max_hold_days=3,
            entry_low_pct=-0.05,
            entry_high_pct=0.05,
        )


def _bars(code: str, spec: list[tuple[float, float, float, float]], name: str = "测试股份"):
    days = trading_days(len(spec))
    bars = []
    for day, (open_, high, low, close) in zip(days, spec, strict=True):
        bars.append(RawBar(day, open_, high, low, close, 1_000_000, max(close, 1.0) * 1_000_000))
    return build_symbol(code, name, bars)


def _flat(n: int, price: float = 10.0) -> list[tuple[float, float, float, float]]:
    return [(price, price, price, price)] * n


def _order(series, index: int, **kwargs) -> PendingBuy:
    close = money(series.close[index])
    defaults = dict(
        code=series.code,
        name=series.name,
        strategy_id="first_board_follow",
        strategy_name="首板次日承接",
        signal_date=series.dates[index],
        score=70.0,
        entry_low=money(close * D("0.95")),
        entry_high=money(close * D("1.05")),
        take_profit_pct=0.06,
        stop_loss_pct=0.04,
        max_hold_days=3,
        reason="测试",
    )
    defaults.update(kwargs)
    return PendingBuy(**defaults)


def test_default_paths_ignore_the_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for key in ("ASHARE_REPORT_DIR", "ASHARE_CACHE_DIR", "ASHARE_LEDGER", "ASHARE_CONFIG", "ASHARE_MODEL"):
        monkeypatch.delenv(key, raising=False)
    root = project_root()
    assert report_dir() == root / "reports"
    assert cache_dir() == root / "data" / "cache"
    assert ledger_path() == root / "data" / "paper" / "ledger.json"
    assert model_path() == root / "data" / "models" / "ranker.pkl"
    assert default_config_path() == root / "config.json"
    (tmp_path / "config.json").write_text('{"initial_capital": 1}', encoding="utf-8")
    assert load_settings().initial_capital == 20_000.0
    monkeypatch.setenv("ASHARE_REPORT_DIR", str(tmp_path / "elsewhere"))
    assert report_dir() == (tmp_path / "elsewhere").resolve()
    assert report_dir(tmp_path / "flag") == (tmp_path / "flag").resolve()


def test_cli_defaults_are_empty_and_dashboard_is_headless(tmp_path):
    args = build_parser().parse_args(["daily", "--no-paper"])
    assert args.report_dir is None
    assert args.cache_dir is None
    assert args.paper is None
    command, env = dashboard_command(8501, tmp_path / "reports", tmp_path / "cache", tmp_path / "ledger.json", tmp_path / "config.json")
    assert command[command.index("--server.headless") + 1] == "true"
    assert env["STREAMLIT_SERVER_HEADLESS"] == "true"
    assert "127.0.0.1" in command


def test_st_inference_uses_only_the_five_percent_band():
    flagged = infer_st_mask(_bars("600000", _seal_spec({5, 12})))
    assert flagged[19]
    assert not flagged[10]
    assert not infer_st_mask(_bars("600000", _seal_spec({5})))[19]
    cleared = infer_st_mask(_bars("600000", _seal_spec({5, 12}, breach=8)))
    assert not cleared[19]
    assert not infer_st_mask(_bars("300001", _seal_spec({5, 12}))).any()
    base = _bars("600000", _seal_spec({5, 12}))
    longer = _bars("600000", _seal_spec({5, 12}) + _flat(5, float(base.close[-1])))
    assert np.array_equal(infer_st_mask(longer)[: len(base.close)], infer_st_mask(base))


def test_sentiment_and_features_ignore_future_bars():
    spec = _flat(40)
    # Day 15 touches the 10% limit and closes back inside: a broken limit.
    spec[15] = (10.0, 11.0, 10.0, 10.4)
    # Days 20 and 21 seal the 10% limit so the height reaches 2.
    spec[20] = (10.0, 11.0, 10.0, 11.0)
    spec[21] = (11.0, 12.1, 11.0, 12.1)
    series = _bars("600000", spec)
    panel = build_market_panel([series], loose_settings())
    day = series.dates[21]
    before = panel.days[day]
    assert panel.days[series.dates[15]].broken_count == 1
    assert panel.days[series.dates[15]].limit_up_count == 0
    assert int(panel.height[series.code][21]) == 2
    features = feature_vector(series, 21, "ma_pullback", 12.0, panel)
    assert features.shape == (len(FEATURE_NAMES),)
    assert np.isfinite(features).all()
    extended = _bars("600000", spec + _flat(8, float(series.close[-1])))
    panel_after = build_market_panel([extended], loose_settings())
    after = panel_after.days[day]
    assert after == before
    again = feature_vector(extended, 21, "ma_pullback", 12.0, panel_after)
    assert np.allclose(again, features)


def test_label_follows_the_future_path_and_drops_unbuyable_opens():
    prefix = _flat(32)
    stopped = prefix + [(10.0, 10.0, 9.0, 9.2)] + _flat(4, 9.2)
    rallied = prefix + [(10.0, 11.2, 10.0, 11.0)] + _flat(4, 11.0)
    down = _bars("600000", stopped)
    up = _bars("600000", rallied)
    settings = loose_settings()
    panel_down = build_market_panel([down], settings)
    panel_up = build_market_panel([up], settings)
    left = feature_vector(down, 30, "ma_pullback", 8.0, panel_down)
    right = feature_vector(up, 30, "ma_pullback", 8.0, panel_up)
    assert np.allclose(left, right)
    calendar = down.dates
    index = {day: pos for pos, day in enumerate(calendar)}
    loss, loss_day = trade_net_return(down, _order(down, 30), settings, calendar, index)
    gain, gain_day = trade_net_return(up, _order(up, 30), settings, up.dates, {day: pos for pos, day in enumerate(up.dates)})
    assert loss is not None and gain is not None
    assert loss < 0 < gain
    assert loss_day == down.dates[32]
    assert gain_day == up.dates[32]
    blocked = prefix[:31] + [(11.0, 11.0, 11.0, 11.0)] + _flat(5, 11.0)
    sealed = _bars("600000", blocked)
    label, sold = trade_net_return(
        sealed,
        _order(sealed, 30),
        settings,
        sealed.dates,
        {day: pos for pos, day in enumerate(sealed.dates)},
    )
    assert label is None and sold is None


def test_training_rows_stop_before_the_holdout():
    zeros = np.zeros(len(FEATURE_NAMES))
    rows = [
        LabeledRow(date(2025, 4, 1), date(2025, 4, 20), "000001", "first_board_follow", zeros, 0.01),
        LabeledRow(date(2025, 4, 20), date(2025, 4, 25), "000002", "first_board_follow", zeros, 0.02),
        LabeledRow(date(2025, 4, 24), date(2025, 4, 28), "000003", "ma_pullback", zeros, -0.01),
        LabeledRow(date(2025, 4, 26), date(2025, 4, 28), "000004", "ma_pullback", zeros, 0.03),
        LabeledRow(date(2025, 4, 1), None, "000005", "ma_pullback", zeros, None),
    ]
    kept = rows_for_training(rows, HOLDOUT_START)
    assert [row.code for row in kept] == ["000001"]
    params, _records, fallback = select_hyperparams(rows, [date(2024, 1, 2)])
    assert fallback
    assert params["max_depth"] == 3
    assert params["min_samples_leaf"] == 40


def test_regime_gate_blocks_entries_and_needs_enough_names():
    rising = []
    price = 10.0
    prices = []
    for _ in range(30):
        prices.append(price)
        price *= 1.01
    for offset in range(8):
        rising.append(flat_symbol(f"{600000 + offset:06d}", prices))
    settings = loose_settings(regime_min_names=5)
    panel = build_market_panel(rising, settings)
    assert panel.days[rising[0].dates[-1]].risk_on
    strict = loose_settings()
    assert strict.regime_min_names == 200
    assert not build_market_panel(rising, strict).days[rising[0].dates[-1]].risk_on
    falling = []
    price = 10.0
    down_prices = []
    for _ in range(30):
        down_prices.append(price)
        price *= 0.99
    for offset in range(8):
        falling.append(flat_symbol(f"{600000 + offset:06d}", down_prices))
    assert not build_market_panel(falling, settings).days[falling[0].dates[-1]].risk_on

    series = flat_symbol("000001", [10.0] * 16)
    signal_day = series.dates[10]
    blocked = run_backtest(
        [series],
        [(_Scored({signal_day}), {})],
        loose_settings(),
        series.dates[0],
        series.dates[-1],
        entry_allowed={day: False for day in series.dates},
    )
    assert blocked.trades == []
    assert blocked.rejects.get("行情过滤不开新仓", 0) >= 1


def test_rescore_keeps_only_the_top_model_names():
    left = flat_symbol("000001", [10.0] * 16)
    right = flat_symbol("000002", [10.0] * 16)
    signal_day = left.dates[10]
    settings = loose_settings(top_n=1)

    def rescore(order: PendingBuy):
        return 1.0 if order.code.endswith("1") else 50.0

    result = run_backtest(
        [left, right],
        [(_Scored({signal_day}), {})],
        settings,
        left.dates[0],
        left.dates[-1],
        rescore=rescore,
        pool_size=2,
    )
    assert [trade.code for trade in result.trades] == ["000002"]


def test_inferred_st_is_what_the_filters_see_and_csv_has_the_model_column(tmp_path):
    series = _bars("600000", _seal_spec({5, 12}))
    attach_inferred_st([series])
    assert series.st_flags is not None and bool(series.st_flags[19])
    write_recommendations_csv(
        tmp_path / "daily.csv",
        [
            {
                "code": "600000",
                "name": "测试",
                "strategy": "均线回踩",
                "score": 1.0,
                "model_score": 0.0123,
                "reason": "测试",
                "entry_low": 9.0,
                "entry_high": 10.0,
                "take_profit": 11.0,
                "stop_loss": 8.0,
                "max_hold_days": 3,
                "shares": 100,
                "budget": 1000.0,
                "signal_date": "2024-01-02",
                "close": 10.0,
            }
        ],
    )
    text = (tmp_path / "daily.csv").read_text(encoding="utf-8-sig")
    assert "模型分数" in text
    assert "0.0123" in text


def test_verdict_rules_and_tstat_are_fixed():
    assert "不足 30" in paper_verdict(0.1, 12, 3.0)
    assert "不为正" in paper_verdict(-0.01, 40, 3.0)
    assert "小于 2" in paper_verdict(0.05, 40, 1.2)
    assert "很小的资金" in paper_verdict(0.05, 40, 2.4)
    values = [1.0, 2.0, 3.0]
    assert abs(pnl_tstat(values) - (2.0 * np.sqrt(3.0))) < 1e-9


def _seal_spec(seals: set[int], breach: int | None = None, n: int = 30):
    price = 10.0
    spec = []
    for index in range(n):
        if index == breach:
            up = float(_cents(np.array([price * 1.10]))[0]) / 100.0
            spec.append((up, up, price, up))
            price = up
            continue
        if index in seals:
            up = float(_cents(np.array([price * 1.05]))[0]) / 100.0
            spec.append((up, up, up, up))
            price = up
            continue
        spec.append((price, price, price, price))
    return spec
