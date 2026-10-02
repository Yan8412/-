"""Point-in-time limit-up sentiment and an explainable market-regime gate.

Every figure for a session uses that session's close and earlier bars only.
Limit width follows the board: 10% main, 20% ChiNext and STAR, 30% Beijing.
A main-board bar that is sealed inside the 5% band is treated as an ST
limit, which is the width ST names actually trade at.

Promotion rates, the consecutive-board premium, and the ladder gap follow
the emotion-cycle ideas in simonlin1212/vibe-astock ``duanxian/emotion_metrics.py``
(Apache-2.0). The arithmetic here is written against this project's own
limit-up flags. It does not call that repository's Eastmoney data layer.
"""

from __future__ import annotations

import math
from dataclasses import astuple, dataclass
from datetime import date

import numpy as np

from ashare.config import Settings
from ashare.filters import _cents
from ashare.market import SymbolSeries, master_calendar
from ashare.rules.limits import classify_board

# Consecutive-board heights above this are counted in the top bin.
_HEIGHT_CAP = 64


@dataclass(frozen=True, eq=False)
class MarketDay:
    breadth: float
    breadth_count: int
    limit_up_count: int
    broken_count: int
    broken_rate: float
    max_height: int
    prev_limit_return: float
    index_level: float
    index_vs_ma: float
    risk_on: bool
    # Share of yesterday's 1-board names that sealed again today. NaN if none.
    promo_1_2: float = float("nan")
    # Same for yesterday's 2-board names, and for names already at 3 or higher.
    promo_2_3: float = float("nan")
    promo_3p: float = float("nan")
    # Mean today-return of names whose consecutive height yesterday was >= 2.
    board_premium: float = float("nan")
    # Count of missing heights between 1 and today's maximum height.
    ladder_gap: int = 0

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MarketDay):
            return NotImplemented
        for left, right in zip(astuple(self), astuple(other), strict=True):
            if isinstance(left, float) and isinstance(right, float):
                if math.isnan(left) and math.isnan(right):
                    continue
            if left != right:
                return False
        return True


@dataclass
class MarketPanel:
    days: dict[date, MarketDay]
    calendar: list[date]
    limit_up: dict[str, np.ndarray]
    height: dict[str, np.ndarray]


def infer_st_mask(series: SymbolSeries, window: int = 20) -> np.ndarray:
    """Main-board days that look like an ST regime, using only past and current bars.

    A day is flagged when the last ``window`` sessions contain at least two
    seals on the 5% band and none of those sessions traded outside that band.
    A print wider than 5% clears the flag, because an ST name cannot do that.
    ChiNext, STAR, and Beijing keep a zero mask; their limit is already 20% or 30%.
    """
    n = len(series.close)
    out = np.zeros(n, dtype=bool)
    if n == 0 or classify_board(series.code) != "main":
        return out
    up5, down5 = _band_cents(series.preclose, 0.05)
    close_c = _cents(series.close)
    high_c = _cents(series.high)
    low_c = _cents(series.low)
    sealed = ((close_c == up5) | (close_c == down5)) & (high_c <= up5) & (low_c >= down5)
    breach = (high_c > up5) | (low_c < down5)
    seal_sum = _rolling_sum(sealed.astype(np.int32), window)
    breach_sum = _rolling_sum(breach.astype(np.int32), window)
    ready = np.arange(n) >= window - 1
    out = ready & (seal_sum >= 2) & (breach_sum == 0)
    return out


def attach_inferred_st(symbols: list[SymbolSeries]) -> int:
    """Fill ``st_flags`` where the caller has not already supplied a daily flag."""
    flagged = 0
    for series in symbols:
        if series.st_flags is not None and len(series.st_flags) == len(series.close):
            continue
        mask = infer_st_mask(series)
        series.st_flags = mask
        if bool(mask.any()):
            flagged += 1
    return flagged


def build_market_panel(symbols: list[SymbolSeries], settings: Settings) -> MarketPanel:
    """Breadth, limit-up sentiment, and the regime bit for each calendar day."""
    calendar = master_calendar(symbols)
    cal_index = {day: index for index, day in enumerate(calendar)}
    limit_up: dict[str, np.ndarray] = {}
    height: dict[str, np.ndarray] = {}
    breadth_hit = np.zeros(len(calendar), dtype=np.int32)
    breadth_tot = np.zeros(len(calendar), dtype=np.int32)
    up_count = np.zeros(len(calendar), dtype=np.int32)
    broken_count = np.zeros(len(calendar), dtype=np.int32)
    max_height = np.zeros(len(calendar), dtype=np.int32)
    ret_sum = np.zeros(len(calendar), dtype=float)
    ret_n = np.zeros(len(calendar), dtype=np.int32)
    prev_sum = np.zeros(len(calendar), dtype=float)
    prev_n = np.zeros(len(calendar), dtype=np.int32)
    promo_num = np.zeros((len(calendar), 3), dtype=np.int32)
    promo_den = np.zeros((len(calendar), 3), dtype=np.int32)
    premium_sum = np.zeros(len(calendar), dtype=float)
    premium_n = np.zeros(len(calendar), dtype=np.int32)
    height_counts = np.zeros((len(calendar), _HEIGHT_CAP), dtype=np.int16)

    for series in symbols:
        is_up, is_broken, bars_height = _limit_events(series)
        limit_up[series.code] = is_up
        height[series.code] = bars_height
        above = np.isfinite(series.ma20) & (series.qfq_close > series.ma20)
        returns = _returns(series)
        for index, day in enumerate(series.dates):
            slot = cal_index.get(day)
            if slot is None or series.volume[index] <= 0:
                continue
            if np.isfinite(series.ma20[index]):
                breadth_tot[slot] += 1
                if above[index]:
                    breadth_hit[slot] += 1
            if is_up[index]:
                up_count[slot] += 1
            if is_broken[index]:
                broken_count[slot] += 1
            if bars_height[index] > max_height[slot]:
                max_height[slot] = int(bars_height[index])
            if np.isfinite(returns[index]):
                ret_sum[slot] += returns[index]
                ret_n[slot] += 1
            bars_today = int(bars_height[index])
            if bars_today > 0:
                height_counts[slot, min(bars_today, _HEIGHT_CAP - 1)] += 1
            if index > 0 and series.volume[index - 1] > 0:
                prev_day = series.dates[index - 1]
                prev_slot = cal_index.get(prev_day)
                if prev_slot is not None and slot == prev_slot + 1:
                    if is_up[index - 1] and np.isfinite(returns[index]):
                        prev_sum[slot] += returns[index]
                        prev_n[slot] += 1
                    yesterday = int(bars_height[index - 1])
                    tier = _promo_tier(yesterday)
                    if tier >= 0:
                        promo_den[slot, tier] += 1
                        if is_up[index]:
                            promo_num[slot, tier] += 1
                    if yesterday >= 2 and np.isfinite(returns[index]):
                        premium_sum[slot] += returns[index]
                        premium_n[slot] += 1

    level = 100.0
    levels = np.empty(len(calendar), dtype=float)
    for slot in range(len(calendar)):
        if ret_n[slot] > 0:
            level *= 1.0 + ret_sum[slot] / ret_n[slot]
        levels[slot] = level
    index_ma = _trailing_mean(levels, settings.regime_ma_window)

    days: dict[date, MarketDay] = {}
    for slot, day in enumerate(calendar):
        total = int(breadth_tot[slot])
        breadth = float(breadth_hit[slot] / total) if total else float("nan")
        touched = int(up_count[slot] + broken_count[slot])
        broken_rate = float(broken_count[slot] / touched) if touched else 0.0
        prev_ret = float(prev_sum[slot] / prev_n[slot]) if prev_n[slot] else float("nan")
        ma = index_ma[slot]
        versus = float(levels[slot] / ma - 1.0) if np.isfinite(ma) and ma > 0 else float("nan")
        risk_on = bool(
            total >= settings.regime_min_names
            and np.isfinite(breadth)
            and breadth >= settings.regime_breadth_min
            and np.isfinite(versus)
            and versus >= 0.0
        )
        peak = int(max_height[slot])
        days[day] = MarketDay(
            breadth=breadth,
            breadth_count=total,
            limit_up_count=int(up_count[slot]),
            broken_count=int(broken_count[slot]),
            broken_rate=broken_rate,
            max_height=peak,
            prev_limit_return=prev_ret,
            index_level=float(levels[slot]),
            index_vs_ma=versus,
            risk_on=risk_on,
            promo_1_2=_rate(int(promo_num[slot, 0]), int(promo_den[slot, 0])),
            promo_2_3=_rate(int(promo_num[slot, 1]), int(promo_den[slot, 1])),
            promo_3p=_rate(int(promo_num[slot, 2]), int(promo_den[slot, 2])),
            board_premium=float(premium_sum[slot] / premium_n[slot]) if premium_n[slot] else float("nan"),
            ladder_gap=_ladder_gap(height_counts[slot], peak),
        )
    return MarketPanel(days=days, calendar=calendar, limit_up=limit_up, height=height)


def entry_mask(panel: MarketPanel) -> dict[date, bool]:
    return {day: info.risk_on for day, info in panel.days.items()}


def _json_number(value: float) -> float | None:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def market_day_payload(day: date, info: MarketDay) -> dict:
    """JSON-safe snapshot of one session. NaN becomes null."""
    return {
        "date": day.isoformat(),
        "risk_on": bool(info.risk_on),
        "breadth": _json_number(info.breadth),
        "breadth_count": info.breadth_count,
        "limit_up_count": info.limit_up_count,
        "broken_count": info.broken_count,
        "broken_rate": _json_number(info.broken_rate),
        "max_height": info.max_height,
        "prev_limit_return": _json_number(info.prev_limit_return),
        "index_level": _json_number(info.index_level),
        "index_vs_ma": _json_number(info.index_vs_ma),
        "promo_1_2": _json_number(info.promo_1_2),
        "promo_2_3": _json_number(info.promo_2_3),
        "promo_3p": _json_number(info.promo_3p),
        "board_premium": _json_number(info.board_premium),
        "ladder_gap": info.ladder_gap,
    }


def _promo_tier(yesterday_height: int) -> int:
    """0 = 1→2, 1 = 2→3, 2 = 3+→next. -1 when yesterday was not a board."""
    if yesterday_height == 1:
        return 0
    if yesterday_height == 2:
        return 1
    if yesterday_height >= 3:
        return 2
    return -1


def _rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return float("nan")
    return float(numerator / denominator)


def _ladder_gap(counts: np.ndarray, max_height: int) -> int:
    """How many rungs from 1 through ``max_height`` have no stock today."""
    if max_height <= 1:
        return 0
    cap = len(counts) - 1
    limit = min(max_height, cap)
    present = int(np.count_nonzero(counts[1 : limit + 1]))
    missing = limit - present
    if max_height > cap:
        missing += max_height - cap
    return int(missing)


def _limit_events(series: SymbolSeries) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Close-at-limit, broken limit (high touched, close did not), consecutive height."""
    n = len(series.close)
    close_c = _cents(series.close)
    high_c = _cents(series.high)
    board_up = _cents(series.limit_up)
    is_up = close_c >= board_up
    is_broken = (high_c >= board_up) & (close_c < board_up)
    if classify_board(series.code) == "main":
        up5, _down5 = _band_cents(series.preclose, 0.05)
        # Sealed inside the 5% band: an ST-width limit-up. A wider high is the 10% regime.
        st_seal = (close_c == up5) & (high_c <= up5) & ~is_up
        st_broken = (high_c >= up5) & (high_c < board_up) & (close_c < up5)
        is_up = is_up | st_seal
        is_broken = is_broken | st_broken
    height = np.zeros(n, dtype=np.int16)
    running = 0
    for index in range(n):
        if is_up[index]:
            running += 1
        else:
            running = 0
        height[index] = running
    return is_up, is_broken, height


def _band_cents(preclose: np.ndarray, ratio: float) -> tuple[np.ndarray, np.ndarray]:
    up = _cents(np.asarray(preclose, dtype=float) * (1.0 + ratio))
    down = _cents(np.asarray(preclose, dtype=float) * (1.0 - ratio))
    return up, down


def _returns(series: SymbolSeries) -> np.ndarray:
    out = np.full(len(series.close), np.nan, dtype=float)
    ok = series.preclose > 0
    out[ok] = series.close[ok] / series.preclose[ok] - 1.0
    return out


def _rolling_sum(values: np.ndarray, window: int) -> np.ndarray:
    cumulative = np.cumsum(values, dtype=np.int32)
    out = cumulative.copy()
    if len(values) > window:
        out[window:] = cumulative[window:] - cumulative[:-window]
    return out


def _trailing_mean(values: np.ndarray, window: int) -> np.ndarray:
    out = np.full(len(values), np.nan, dtype=float)
    if window <= 0 or len(values) < window:
        return out
    cumulative = np.cumsum(values, dtype=float)
    out[window - 1] = cumulative[window - 1] / window
    if len(values) > window:
        out[window:] = (cumulative[window:] - cumulative[:-window]) / window
    return out
