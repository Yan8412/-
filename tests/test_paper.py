"""Paper ledger: no live orders, T+1, and 100-share fills."""

from datetime import timedelta

from ashare.broker.base import OrderRequest
from ashare.broker.miniqmt import MiniQMTBroker
from ashare.broker.paper import PaperBroker
from tests.helpers import flat_symbol, loose_settings


def test_miniqmt_is_not_wired_up():
    try:
        MiniQMTBroker()
    except NotImplementedError as exc:
        assert "不会" in str(exc) or "不下" in str(exc) or "实盘" in str(exc)
    else:
        raise AssertionError("miniQMT 适配在 v1 必须拒绝构造")


def test_paper_respects_t_plus_one_and_lots(tmp_path):
    settings = loose_settings(slippage_rate=0)
    series = flat_symbol("000001", [10.0] * 6)
    # Crash the session after the buy so a same-day stop would be tempting.
    series.low[2] = 8
    series.low[3] = 8
    broker = PaperBroker(tmp_path / "ledger.json", settings)
    signal_day = series.dates[1]
    ack = broker.place_orders(
        [
            OrderRequest(
                code="000001",
                name="测试股份",
                side="buy",
                shares=500,
                signal_date=signal_day,
                strategy_id="scripted",
                strategy_name="脚本策略",
                reason="测试",
                entry_low=9.5,
                entry_high=10.5,
                take_profit_pct=0.5,
                stop_loss_pct=0.05,
                max_hold_days=3,
                score=1,
            )
        ]
    )
    assert ack[0].status == "pending"
    buy_day = series.dates[2]
    broker.settle(buy_day, {"000001": series})
    snap = broker.snapshot()
    assert len(snap.positions) == 1
    assert snap.positions[0]["shares"] == 400
    assert snap.positions[0]["shares"] % 100 == 0
    assert snap.positions[0]["buy_date"] == buy_day.isoformat()
    sell_fills = [item for item in snap.fills if item["side"] == "sell"]
    assert sell_fills == []
    # Next session hits the stop that was armed at 9.50.
    broker.settle(series.dates[3], {"000001": series})
    snap = broker.snapshot()
    assert snap.positions == []
    sells = [item for item in snap.fills if item["side"] == "sell"]
    assert len(sells) == 1
    assert sells[0]["date"] == series.dates[3].isoformat()
    assert sells[0]["price"] == "9.50"
    assert snap.cash < settings.initial_capital
    # Settling the same day twice does not duplicate fills.
    before = len(broker.snapshot().fills)
    broker.settle(series.dates[3], {"000001": series})
    assert len(broker.snapshot().fills) == before
    assert signal_day < buy_day
    assert buy_day + timedelta(days=1) <= series.dates[3] or series.dates[3] > buy_day
