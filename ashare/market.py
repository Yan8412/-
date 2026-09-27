"""Per-symbol daily history used by strategies and the simulator."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np

from ashare.rules.adjust import apply_causal_adjust, ex_rights_reference
from ashare.rules.limits import limit_prices, normalize_code


@dataclass
class RawBar:
    date: date
    open: float
    high: float
    low: float
    close: float
    volume: float
    amount: float
    cash_dividend: float = 0.0
    bonus_ratio: float = 0.0


@dataclass
class SymbolSeries:
    code: str
    name: str
    dates: list[date]
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    amount: np.ndarray
    preclose: np.ndarray
    limit_up: np.ndarray
    limit_down: np.ndarray
    qfq_open: np.ndarray
    qfq_high: np.ndarray
    qfq_low: np.ndarray
    qfq_close: np.ndarray
    left_censored: bool = True
    ma5: np.ndarray = field(default_factory=lambda: np.array([]))
    ma20: np.ndarray = field(default_factory=lambda: np.array([]))
    ma60: np.ndarray = field(default_factory=lambda: np.array([]))
    dif: np.ndarray = field(default_factory=lambda: np.array([]))
    dea: np.ndarray = field(default_factory=lambda: np.array([]))
    vol_ma5: np.ndarray = field(default_factory=lambda: np.array([]))
    swing20: np.ndarray = field(default_factory=lambda: np.array([]))
    ipo_date: date | None = None
    out_date: date | None = None
    st_flags: np.ndarray | None = None
    halted: np.ndarray | None = None
    date_index: dict[date, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.date_index = {day: index for index, day in enumerate(self.dates)}


def build_symbol(code: str, name: str, bars: list[RawBar]) -> SymbolSeries:
    """Sort, drop duplicate dates, and attach limit prices plus a qfq view."""
    symbol = normalize_code(code)
    ordered: dict[date, RawBar] = {}
    for bar in bars:
        ordered[bar.date] = bar
    rows = [ordered[key] for key in sorted(ordered)]
    n = len(rows)
    open_ = [row.open for row in rows]
    high = [row.high for row in rows]
    low = [row.low for row in rows]
    close = [row.close for row in rows]
    volume = [row.volume for row in rows]
    amount = [row.amount for row in rows]
    cash = [row.cash_dividend for row in rows]
    bonus = [row.bonus_ratio for row in rows]
    preclose = [0.0] * n
    up = [0.0] * n
    down = [0.0] * n
    for i, row in enumerate(rows):
        if i == 0:
            preclose[i] = row.open
        else:
            reference = ex_rights_reference(close[i - 1], cash[i], bonus[i])
            preclose[i] = float(reference)
        limit_up, limit_down = limit_prices(preclose[i], symbol)
        up[i] = float(limit_up)
        down[i] = float(limit_down)
    q_open, q_high, q_low, q_close = apply_causal_adjust(open_, high, low, close, preclose)
    series = SymbolSeries(
        code=symbol,
        name=name,
        dates=[row.date for row in rows],
        open=np.asarray(open_, dtype=float),
        high=np.asarray(high, dtype=float),
        low=np.asarray(low, dtype=float),
        close=np.asarray(close, dtype=float),
        volume=np.asarray(volume, dtype=float),
        amount=np.asarray(amount, dtype=float),
        preclose=np.asarray(preclose, dtype=float),
        limit_up=np.asarray(up, dtype=float),
        limit_down=np.asarray(down, dtype=float),
        qfq_open=np.asarray(q_open, dtype=float),
        qfq_high=np.asarray(q_high, dtype=float),
        qfq_low=np.asarray(q_low, dtype=float),
        qfq_close=np.asarray(q_close, dtype=float),
    )
    prepare_indicators(series)
    return series


def sma(values: np.ndarray, window: int) -> np.ndarray:
    out = np.full(len(values), np.nan, dtype=float)
    if window <= 0 or len(values) < window:
        return out
    cumulative = np.cumsum(values, dtype=float)
    out[window - 1] = cumulative[window - 1] / window
    if len(values) > window:
        out[window:] = (cumulative[window:] - cumulative[:-window]) / window
    return out


def ema(values: np.ndarray, window: int) -> np.ndarray:
    out = np.full(len(values), np.nan, dtype=float)
    if len(values) < window or window <= 0:
        return out
    alpha = 2.0 / (window + 1.0)
    out[window - 1] = float(np.mean(values[:window]))
    for index in range(window, len(values)):
        out[index] = alpha * values[index] + (1.0 - alpha) * out[index - 1]
    return out


def macd_lines(close: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[np.ndarray, np.ndarray]:
    fast_line = ema(close, fast)
    slow_line = ema(close, slow)
    dif = fast_line - slow_line
    dea = np.full(len(close), np.nan, dtype=float)
    start = slow - 1
    if start < len(close):
        tail = dif[start:].copy()
        tail_ema = ema(np.nan_to_num(tail, nan=0.0), signal) if np.all(np.isnan(tail)) else _ema_skip_nan(dif, start, signal)
        dea[start:] = tail_ema
    return dif, dea


def _ema_skip_nan(dif: np.ndarray, start: int, window: int) -> np.ndarray:
    tail = dif[start:]
    return ema(tail, window)


def rolling_swing(high: np.ndarray, low: np.ndarray, close: np.ndarray, window: int = 20) -> np.ndarray:
    """(rolling high - rolling low) / close, using only bars up to each index."""
    out = np.full(len(close), np.nan, dtype=float)
    if window <= 0 or len(close) < window:
        return out
    highs = np.lib.stride_tricks.sliding_window_view(high, window).max(axis=1)
    lows = np.lib.stride_tricks.sliding_window_view(low, window).min(axis=1)
    denom = close[window - 1 :]
    with np.errstate(divide="ignore", invalid="ignore"):
        out[window - 1 :] = (highs - lows) / denom
    return out


def prepare_indicators(series: SymbolSeries) -> None:
    close = series.qfq_close
    series.ma5 = sma(close, 5)
    series.ma20 = sma(close, 20)
    series.ma60 = sma(close, 60)
    series.dif, series.dea = macd_lines(close)
    series.vol_ma5 = sma(series.volume, 5)
    series.swing20 = rolling_swing(series.high, series.low, series.close, 20)


def mark_listing_censorship(symbols: list[SymbolSeries]) -> None:
    """Stocks whose first bar is later than the sample start are treated as new listings.

    A fixed-length download left-censors old listings: their first bar sits on
    the download boundary, so the IPO date is unknown and the new-listing
    filter does not apply. Names that appear well after that boundary are
    blocked until ``min_listed_bars`` real sessions have printed.

    When Baostock supplies an IPO date and the first bar is that listing, the
    new-listing rule applies even if some other name in the universe starts earlier.
    """
    if not symbols:
        return
    dated = [item.dates[0] for item in symbols if item.dates]
    earliest = min(dated) if dated else None
    for item in symbols:
        if not item.dates or earliest is None:
            item.left_censored = True
            continue
        if item.ipo_date is not None and item.dates[0] <= item.ipo_date + timedelta(days=15):
            item.left_censored = False
            continue
        item.left_censored = item.dates[0] <= earliest + timedelta(days=20)


def master_calendar(symbols: list[SymbolSeries], coverage: float = 0.3) -> list[date]:
    """Trading days present in a meaningful fraction of the universe."""
    from collections import Counter

    counts: Counter[date] = Counter()
    for item in symbols:
        counts.update(item.dates)
    if not symbols:
        return []
    threshold = max(1, int(len(symbols) * coverage))
    return sorted(day for day, count in counts.items() if count >= threshold)
