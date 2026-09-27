"""Synthetic daily bars for rule tests. No network."""

from __future__ import annotations

from datetime import date, timedelta

from ashare.config import Settings
from ashare.market import RawBar, SymbolSeries, build_symbol


def trading_days(count: int, start: date | None = None) -> list[date]:
    day = start or date(2024, 1, 2)
    days: list[date] = []
    while len(days) < count:
        if day.weekday() < 5:
            days.append(day)
        day += timedelta(days=1)
    return days


def flat_symbol(
    code: str,
    prices: list[float],
    *,
    name: str = "测试股份",
    volume: float = 2_000_000,
    amount: float = 80_000_000,
    start: date | None = None,
) -> SymbolSeries:
    days = trading_days(len(prices), start)
    bars = [
        RawBar(
            date=day,
            open=price,
            high=price,
            low=price,
            close=price,
            volume=volume,
            amount=amount,
        )
        for day, price in zip(days, prices, strict=True)
    ]
    return build_symbol(code, name, bars)


def loose_settings(**overrides) -> Settings:
    data = dict(
        initial_capital=20_000.0,
        max_positions=4,
        max_position_pct=0.35,
        commission_rate=0.00025,
        min_commission=5.0,
        stamp_duty_rate=0.0005,
        slippage_rate=0.0,
        price_min=1.0,
        price_max=500.0,
        min_amount=0.0,
        min_history_bars=5,
        min_listed_bars=5,
        min_swing_20d=0.0,
        top_n=10,
        min_trades_for_selection=5,
        train_days=20,
        test_days=10,
    )
    data.update(overrides)
    return Settings(**data)
