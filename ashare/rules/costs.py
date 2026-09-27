"""Commission, stamp duty, and transfer fee."""

from __future__ import annotations

from datetime import date
from decimal import Decimal

from ashare.config import Settings
from ashare.rules.money import D, money


def stamp_rate(trade_date: date, settings: Settings) -> Decimal:
    """Sell-side stamp duty. The rate was halved on 2023-08-28."""
    if trade_date < settings.stamp_duty_change:
        return D(settings.stamp_duty_rate_legacy)
    return D(settings.stamp_duty_rate)


def commission(notional: Decimal, settings: Settings) -> Decimal:
    """Both sides. Never below the broker minimum."""
    raw = notional * D(settings.commission_rate)
    return money(max(D(settings.min_commission), raw))


def transfer_fee(notional: Decimal, settings: Settings) -> Decimal:
    """ChinaClear transfer fee, both sides, rounded to cents."""
    return money(notional * D(settings.transfer_fee_rate))


def buy_cash_out(shares: int, price: object, trade_date: date, settings: Settings) -> Decimal:
    """Cash deducted on a buy. Stamp duty is not included."""
    notional = money(D(shares) * D(price))
    return money(notional + commission(notional, settings) + transfer_fee(notional, settings))


def sell_cash_in(shares: int, price: object, trade_date: date, settings: Settings) -> Decimal:
    """Cash received on a sell after commission, transfer fee, and stamp duty."""
    notional = money(D(shares) * D(price))
    stamp = money(notional * stamp_rate(trade_date, settings))
    fees = commission(notional, settings) + transfer_fee(notional, settings) + stamp
    return money(notional - fees)
