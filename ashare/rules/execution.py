"""Open/close fill decisions shared by the backtester and the paper broker.

Buys are rejected at the limit-up price. Sells are rejected when the bar
never trades above the limit-down price. A position bought today is not
passed to this function until the next session (T+1 is enforced by the caller).
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from ashare.rules.money import D, money, to_cents


@dataclass(frozen=True)
class BuyEval:
    ok: bool
    price: Decimal | None
    reason: str


@dataclass(frozen=True)
class SellEval:
    ok: bool
    price: Decimal | None
    reason: str


def slip_buy(open_: Decimal, slippage: Decimal, limit_up: Decimal) -> Decimal:
    """Adverse buy price, kept strictly below the limit-up print."""
    raw = money(open_ * (Decimal("1") + slippage))
    if to_cents(raw) >= to_cents(limit_up):
        raw = money(limit_up - Decimal("0.01"))
    if to_cents(raw) < to_cents(open_):
        raw = money(open_)
    return raw


def slip_sell(price: Decimal, slippage: Decimal, limit_down: Decimal) -> Decimal:
    """Adverse sell price, not below the limit-down print."""
    raw = money(price * (Decimal("1") - slippage))
    if to_cents(raw) < to_cents(limit_down):
        raw = money(limit_down)
    return raw


def evaluate_buy(
    *,
    open_: object,
    low: object,
    limit_up: object,
    entry_low: object,
    entry_high: object,
    slippage: object,
) -> BuyEval:
    """Decide whether the next-session open is a valid buy."""
    px_open = money(open_)
    px_low = money(low)
    px_up = money(limit_up)
    band_low = money(entry_low)
    band_high = money(entry_high)
    slip = D(slippage)
    if to_cents(px_open) >= to_cents(px_up) or to_cents(px_low) >= to_cents(px_up):
        return BuyEval(False, None, "涨停无法买入")
    if to_cents(px_open) > to_cents(band_high):
        return BuyEval(False, None, "高开超出买入区间")
    if to_cents(px_open) < to_cents(band_low):
        return BuyEval(False, None, "低开超出买入区间")
    return BuyEval(True, slip_buy(px_open, slip, px_up), "开盘成交")


def evaluate_sell(
    *,
    open_: object,
    high: object,
    low: object,
    close: object,
    limit_up: object,
    limit_down: object,
    stop: object,
    take_profit: object,
    days_held: int,
    max_hold: int,
    slippage: object,
    phase: str,
) -> SellEval:
    """Exit plan for one session.

    ``phase="open"`` only fires gap stops and gap take-profits, so the cash
    can fund other buys at the same open. ``phase="late"`` handles intraday
    stops, take-profits, and the max-holding-day close. If both stop and
    target are touched, the stop is assumed to happen first.
    """
    if phase not in {"open", "late"}:
        raise ValueError(f"未知撮合阶段: {phase}")
    px_open = money(open_)
    px_high = money(high)
    px_low = money(low)
    px_close = money(close)
    px_up = money(limit_up)
    px_down = money(limit_down)
    px_stop = money(stop)
    px_tp = money(take_profit)
    slip = D(slippage)
    sealed = to_cents(px_high) <= to_cents(px_down)
    if sealed:
        if phase == "late":
            return SellEval(False, None, "跌停无法卖出")
        return SellEval(False, None, "hold")

    if phase == "open":
        if to_cents(px_open) <= to_cents(px_stop) and to_cents(px_open) > to_cents(px_down):
            return SellEval(True, slip_sell(px_open, slip, px_down), "stop")
        if to_cents(px_open) >= to_cents(px_tp):
            raw = px_open if to_cents(px_open) <= to_cents(px_up) else px_up
            return SellEval(True, slip_sell(raw, slip, px_down), "take_profit")
        return SellEval(False, None, "hold")

    if to_cents(px_low) <= to_cents(px_stop):
        if to_cents(px_open) <= to_cents(px_down):
            # Could not sell the gap at the open. If the close is back at
            # limit-down, assume the queue never cleared.
            if to_cents(px_close) <= to_cents(px_down):
                return SellEval(False, None, "跌停无法卖出")
            return SellEval(True, slip_sell(px_down + Decimal("0.01"), slip, px_down), "stop")
        raw = px_stop
        if to_cents(raw) <= to_cents(px_down):
            raw = px_down + Decimal("0.01")
        return SellEval(True, slip_sell(raw, slip, px_down), "stop")

    if to_cents(px_high) >= to_cents(px_tp):
        raw = px_tp if to_cents(px_tp) <= to_cents(px_up) else px_up
        return SellEval(True, slip_sell(raw, slip, px_down), "take_profit")

    if days_held >= max_hold:
        if to_cents(px_close) <= to_cents(px_down):
            return SellEval(False, None, "跌停无法卖出")
        return SellEval(True, slip_sell(px_close, slip, px_down), "time")
    return SellEval(False, None, "hold")
