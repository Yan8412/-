"""Dashboard loaders and the sell hint. No Streamlit process."""

from datetime import date

from ashare.backtest.engine import BacktestResult
from ashare.dashboard_data import (
    latest_shortlist,
    load_backtest_bundle,
    load_ledger,
    save_backtest_bundle,
    sell_hint,
)


def test_sell_hint_explains_t_plus_one_and_prices():
    fresh = sell_hint(
        {
            "stop": "9.50",
            "take_profit": "11.00",
            "sessions_held": 0,
            "max_hold_days": 3,
            "buy_date": "2026-09-24",
        }
    )
    assert "9.50" in fresh
    assert "11.00" in fresh
    assert "T+1" in fresh
    assert "还剩 3 天" in fresh
    due = sell_hint(
        {
            "stop": "9.50",
            "take_profit": "11.00",
            "sessions_held": 3,
            "max_hold_days": 3,
            "buy_date": "2026-09-21",
        }
    )
    assert "已到最长持有" in due


def test_bundle_roundtrip_and_missing_files(tmp_path):
    result = BacktestResult(
        strategy_id="demo",
        strategy_name="示例",
        params={},
        start=date(2024, 1, 2),
        end=date(2024, 1, 4),
        initial_capital=20000,
        final_equity=19900,
        equity_dates=[date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)],
        equity=[20000, 19950, 19900],
    )
    path = tmp_path / "backtest_latest.json"
    save_backtest_bundle(path, [result], [(result, [])], ["腾讯"])
    loaded = load_backtest_bundle(path)
    assert loaded is not None
    assert loaded["full_sample"][0]["strategy_name"] == "示例"
    assert loaded["full_sample"][0]["equity"][-1]["equity"] == 19900.0
    assert load_backtest_bundle(tmp_path / "missing.json") is None
    assert load_ledger(tmp_path / "missing-ledger.json") is None
    assert latest_shortlist(tmp_path) == (None, [])
