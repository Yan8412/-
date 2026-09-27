"""Standalone trade result for one signal, using the same fills as the backtest.

The label is the net return on the cash deployed. It is known only after the
exit, so a model trained at a cutoff may use the row only when the sell date
is strictly before that cutoff.
"""

from __future__ import annotations

from datetime import date

from ashare.backtest.engine import PendingBuy
from ashare.config import Settings
from ashare.market import SymbolSeries
from ashare.rules.costs import buy_cash_out, sell_cash_in
from ashare.rules.execution import evaluate_buy, evaluate_sell
from ashare.rules.lots import suggest_shares
from ashare.rules.money import D, money


def trade_net_return(
    series: SymbolSeries,
    order: PendingBuy,
    settings: Settings,
    calendar: list[date],
    calendar_index: dict[date, int],
) -> tuple[float | None, date | None]:
    """Return ``(net return, sell date)``.

    ``(None, None)`` means the next open was unbuyable, the name was halted,
    or the position was still open when the calendar ended. An unfinished
    trade is not a label.
    """
    signal_slot = calendar_index.get(order.signal_date)
    if signal_slot is None or signal_slot + 1 >= len(calendar):
        return None, None
    buy_day = calendar[signal_slot + 1]
    buy_index = series.date_index.get(buy_day)
    if buy_index is None or series.volume[buy_index] <= 0:
        return None, None
    decision = evaluate_buy(
        open_=series.open[buy_index],
        low=series.low[buy_index],
        limit_up=series.limit_up[buy_index],
        entry_low=order.entry_low,
        entry_high=order.entry_high,
        slippage=settings.slippage_rate,
    )
    if not decision.ok or decision.price is None:
        return None, None
    budget = settings.slot_budget(settings.initial_capital, settings.initial_capital)
    shares = suggest_shares(order.code, float(decision.price), budget, settings, buy_day)
    if shares <= 0:
        return None, None
    cost = buy_cash_out(shares, decision.price, buy_day, settings)
    stop = money(decision.price * (D(1) - D(order.stop_loss_pct)))
    take = money(decision.price * (D(1) + D(order.take_profit_pct)))
    buy_slot = signal_slot + 1
    for slot in range(buy_slot + 1, len(calendar)):
        day = calendar[slot]
        index = series.date_index.get(day)
        if index is None or series.volume[index] <= 0:
            continue
        days_held = slot - buy_slot
        sold = None
        for phase in ("open", "late"):
            outcome = evaluate_sell(
                open_=series.open[index],
                high=series.high[index],
                low=series.low[index],
                close=series.close[index],
                limit_up=series.limit_up[index],
                limit_down=series.limit_down[index],
                stop=stop,
                take_profit=take,
                days_held=days_held,
                max_hold=order.max_hold_days,
                slippage=settings.slippage_rate,
                phase=phase,
            )
            if outcome.ok and outcome.price is not None:
                sold = outcome.price
                break
        if sold is None:
            continue
        proceeds = sell_cash_in(shares, sold, day, settings)
        if cost <= 0:
            return None, None
        return float((proceeds - cost) / cost), day
    return None, None
