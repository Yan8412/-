"""Board-lot sizing for a small cash account."""

from __future__ import annotations

from datetime import date

from ashare.config import Settings
from ashare.rules.costs import buy_cash_out
from ashare.rules.limits import classify_board


def board_lot(code: str, settings: Settings) -> int:
    """Main board and ChiNext trade in 100-share lots.

    STAR Market (688/689) first buys are 200 shares. v1 buys STAR names in
    200-share increments so every order satisfies that minimum.
    """
    if classify_board(code) == "star":
        return settings.star_lot_size
    return settings.lot_size


def suggest_shares(
    code: str,
    price: float,
    budget: float,
    settings: Settings,
    trade_date: date,
) -> int:
    """Largest lot that fits in ``budget`` after buy-side fees. Zero if none."""
    if price <= 0 or budget <= 0:
        return 0
    lot = board_lot(code, settings)
    shares = int(budget / price / lot) * lot
    while shares >= lot:
        cost = buy_cash_out(shares, price, trade_date, settings)
        if cost <= D_budget(budget):
            return shares
        shares -= lot
    return 0


def D_budget(budget: float):
    from decimal import Decimal

    return Decimal(str(budget))
