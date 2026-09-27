"""Universe filters applied before a strategy is allowed to emit a signal."""

from __future__ import annotations

import math

import numpy as np

from ashare.config import Settings
from ashare.market import SymbolSeries
from ashare.rules.limits import is_limit_down, is_limit_up, is_risk_name


def rejection_reason(series: SymbolSeries, index: int, settings: Settings) -> str | None:
    """Return a Chinese reason to drop the bar, or None when it may be bought."""
    if _st_on_day(series, index):
        return "ST或风险警示"
    if index < settings.min_history_bars:
        return "历史K线不足"
    if (not series.left_censored) and index < settings.min_listed_bars:
        return "次新股"
    if series.volume[index] <= 0 or series.amount[index] <= 0:
        return "停牌"
    price = float(series.close[index])
    if price < settings.price_min or price > settings.price_max:
        return "价格不在可交易区间"
    if series.amount[index] < settings.min_amount:
        return "成交额过低"
    if settings.min_swing_20d > 0 and index >= 19:
        swing = _swing_at(series, index)
        if not math.isfinite(swing) or swing < settings.min_swing_20d:
            return "近20日振幅不足"
    if is_limit_up(series.close[index], series.limit_up[index]):
        return "收盘涨停无法买入"
    if is_limit_down(series.close[index], series.limit_down[index]):
        return "收盘跌停"
    return None


def eligible_mask(series: SymbolSeries, settings: Settings) -> np.ndarray:
    """Vector form of ``rejection_reason``. True where a buy signal is allowed."""
    n = len(series.close)
    ok = np.ones(n, dtype=bool)
    if series.st_flags is not None and len(series.st_flags) == n:
        ok &= ~np.asarray(series.st_flags, dtype=bool)
    elif is_risk_name(series.name):
        ok[:] = False
        return ok
    index = np.arange(n)
    ok &= index >= settings.min_history_bars
    if not series.left_censored:
        ok &= index >= settings.min_listed_bars
    ok &= (series.volume > 0) & (series.amount > 0)
    ok &= (series.close >= settings.price_min) & (series.close <= settings.price_max)
    ok &= series.amount >= settings.min_amount
    if settings.min_swing_20d > 0 and len(series.swing20) == n:
        swing_ok = np.isfinite(series.swing20) & (series.swing20 >= settings.min_swing_20d)
        if n > 19:
            swing_ok[:19] = False
        else:
            swing_ok[:] = False
        ok &= swing_ok
    ok &= ~limit_up_mask(series)
    ok &= ~limit_down_mask(series)
    return ok


def limit_up_mask(series: SymbolSeries) -> np.ndarray:
    return _cents(series.close) >= _cents(series.limit_up)


def limit_down_mask(series: SymbolSeries) -> np.ndarray:
    return _cents(series.close) <= _cents(series.limit_down)


def _cents(values: np.ndarray) -> np.ndarray:
    """Half-up cents for a price array. Matches ``to_cents`` on normal quotes."""
    scaled = np.asarray(values, dtype=np.float64) * 100.0
    return np.floor(scaled + 0.5 + 1e-8).astype(np.int64)


def _st_on_day(series: SymbolSeries, index: int) -> bool:
    if series.st_flags is not None and len(series.st_flags) == len(series.close):
        return bool(series.st_flags[index])
    return is_risk_name(series.name)


def _swing_at(series: SymbolSeries, index: int) -> float:
    if len(series.swing20) != len(series.close):
        return float("nan")
    return float(series.swing20[index])
