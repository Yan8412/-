"""Universe filters applied before a strategy is allowed to emit a signal."""

from __future__ import annotations

from ashare.config import Settings
from ashare.market import SymbolSeries
from ashare.rules.limits import is_limit_down, is_limit_up, is_risk_name


def rejection_reason(series: SymbolSeries, index: int, settings: Settings) -> str | None:
    """Return a Chinese reason to drop the bar, or None when it may be bought."""
    if is_risk_name(series.name):
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
    if is_limit_up(series.close[index], series.limit_up[index]):
        return "收盘涨停无法买入"
    if is_limit_down(series.close[index], series.limit_down[index]):
        return "收盘跌停"
    return None
