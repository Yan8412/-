"""Cross-sectional rank IC for the OHLCV factor list.

The training window ends early enough that a 5-day forward return does not
touch the confirmation window. The confirmation window ends early enough that
its labels do not touch the final holdout (2025-04-25). The holdout is not
an input to the confirmed / train-only / reversed / noise label.

Newey-West uses a Bartlett kernel with lag ``horizon - 1`` because a 5-day
forward return overlaps the next four daily IC observations.
"""

from __future__ import annotations

import gc
import logging
import math
from datetime import date

import numpy as np

logger = logging.getLogger(__name__)

from ashare.config import Settings
from ashare.factors import FACTOR_NAMES, compute_one
from ashare.market import SymbolSeries, master_calendar
from ashare.ranker import HOLDOUT_START

HORIZON = 5
NW_LAGS = HORIZON - 1
MIN_NAMES = 100
MIN_DAYS = 30
T_BAR = 2.0
# Confirmation starts at the published sample. Labels must finish before it.
CONFIRM_START = date(2024, 9, 24)

CLASS_CONFIRMED = "confirmed"
CLASS_TRAIN_ONLY = "train-only"
CLASS_REVERSED = "reversed"
CLASS_NOISE = "noise"


def classify_ic(train_mean: float, train_t: float, test_mean: float, test_t: float) -> str:
    """Precommitted labels. A test-only result stays noise so the holdout's neighbour cannot recruit a factor."""
    train_sig = _significant(train_mean, train_t)
    test_sig = _significant(test_mean, test_t)
    if train_sig and test_sig:
        if train_mean * test_mean > 0:
            return CLASS_CONFIRMED
        return CLASS_REVERSED
    if train_sig and not test_sig:
        return CLASS_TRAIN_ONLY
    return CLASS_NOISE


def newey_west_t(values: np.ndarray, lags: int) -> tuple[float, float, int]:
    """Mean, Newey-West t of that mean, and the number of finite observations.

    A zero mean with no variance returns t = 0. A non-zero constant returns
    an infinite t with the sign of the mean.
    """
    series = np.asarray(values, dtype=float)
    series = series[np.isfinite(series)]
    count = int(series.size)
    if count < 2:
        return float("nan"), float("nan"), count
    mean = float(series.mean())
    centered = series - mean
    gamma0 = float(np.dot(centered, centered) / count)
    variance = gamma0
    lag_count = max(0, int(lags))
    for lag in range(1, lag_count + 1):
        gamma = float(np.dot(centered[lag:], centered[:-lag]) / count)
        weight = 1.0 - lag / (lag_count + 1.0)
        variance += 2.0 * weight * gamma
    if variance <= 1e-18:
        if mean == 0.0:
            return mean, 0.0, count
        return mean, math.copysign(math.inf, mean), count
    scale = math.sqrt(variance / count)
    return mean, mean / scale, count


def screen_factors(
    symbols: list[SymbolSeries],
    settings: Settings,
    names: tuple[str, ...] = FACTOR_NAMES,
    batch_size: int = 250,
) -> list[dict]:
    """Rank IC of each factor. One factor matrix is allocated at a time."""
    calendar = master_calendar(symbols)
    if len(calendar) <= HORIZON:
        return [_empty(name, "日历短于 5 个交易日") for name in names]
    cal_index = {day: index for index, day in enumerate(calendar)}
    n_days = len(calendar)
    n_symbols = len(symbols)
    slots = [
        np.fromiter((cal_index.get(day, -1) for day in series.dates), dtype=np.int32, count=len(series.dates))
        for series in symbols
    ]
    forward, tradable = _forward_and_mask(symbols, settings, calendar, slots)
    rows: list[dict] = []
    for name in names:
        logger.info("因子 %s", name)
        matrix = np.full((n_symbols, n_days), np.nan, dtype=np.float32)
        _fill_factor(symbols, name, matrix, slots, tradable, batch_size)
        train_ics, test_ics, train_days, test_days = _daily_ics(matrix, forward, calendar)
        train_mean, train_t, train_n = newey_west_t(np.asarray(train_ics, dtype=float), NW_LAGS)
        test_mean, test_t, test_n = newey_west_t(np.asarray(test_ics, dtype=float), NW_LAGS)
        if train_n < MIN_DAYS:
            train_t = float("nan")
        if test_n < MIN_DAYS:
            test_t = float("nan")
        label = classify_ic(train_mean, train_t, test_mean, test_t)
        rows.append(
            {
                "name": name,
                "train_mean_ic": train_mean,
                "train_t": train_t,
                "train_days": train_n,
                "test_mean_ic": test_mean,
                "test_t": test_t,
                "test_days": test_n,
                "train_span": _span(train_days),
                "test_span": _span(test_days),
                "klass": label,
            }
        )
        del matrix
        gc.collect()
    return rows


def confirmed_names(rows: list[dict]) -> tuple[str, ...]:
    return tuple(row["name"] for row in rows if row.get("klass") == CLASS_CONFIRMED)


def _significant(mean: float, tstat: float) -> bool:
    if not math.isfinite(mean) or mean == 0.0 or not math.isfinite(tstat):
        return False
    return abs(tstat) >= T_BAR


def _empty(name: str, note: str) -> dict:
    return {
        "name": name,
        "train_mean_ic": float("nan"),
        "train_t": float("nan"),
        "train_days": 0,
        "test_mean_ic": float("nan"),
        "test_t": float("nan"),
        "test_days": 0,
        "train_span": "",
        "test_span": "",
        "klass": CLASS_NOISE,
        "note": note,
    }


def _forward_and_mask(symbols, settings: Settings, calendar: list[date], slots: list[np.ndarray]):
    n_days = len(calendar)
    n_symbols = len(symbols)
    forward = np.full((n_symbols, n_days), np.nan, dtype=np.float32)
    tradable = np.zeros((n_symbols, n_days), dtype=bool)
    for row, series in enumerate(symbols):
        slot = slots[row]
        present = slot >= 0
        if not present.any():
            continue
        indexes = np.flatnonzero(present)
        for index in indexes:
            if _in_universe(series, int(index), settings):
                tradable[row, int(slot[index])] = True
        close = series.qfq_close
        # 5 sessions on the shared calendar, not 5 bars of this stock.
        for index in indexes:
            current = int(slot[index])
            future_slot = current + HORIZON
            if future_slot >= n_days or not tradable[row, current]:
                continue
            future_index = series.date_index.get(calendar[future_slot])
            if future_index is None:
                continue
            base = float(close[index])
            nxt = float(close[future_index])
            if base > 0 and math.isfinite(base) and math.isfinite(nxt):
                forward[row, current] = np.float32(nxt / base - 1.0)
    return forward, tradable


def _in_universe(series: SymbolSeries, index: int, settings: Settings) -> bool:
    if series.volume[index] <= 0 or series.amount[index] < settings.min_amount:
        return False
    price = float(series.close[index])
    if price < settings.price_min or price > settings.price_max:
        return False
    flags = series.st_flags
    if flags is not None and index < len(flags) and bool(flags[index]):
        return False
    halted = series.halted
    if halted is not None and index < len(halted) and bool(halted[index]):
        return False
    return True


def _fill_factor(symbols, name: str, matrix: np.ndarray, slots: list[np.ndarray], tradable: np.ndarray, batch_size: int) -> None:
    count = len(symbols)
    step = max(1, batch_size)
    for start in range(0, count, step):
        stop = min(count, start + step)
        for row in range(start, stop):
            series = symbols[row]
            slot = slots[row]
            present = np.flatnonzero(slot >= 0)
            if present.size == 0:
                continue
            usable = present[tradable[row, slot[present]]]
            if usable.size == 0:
                continue
            values = compute_one(series, name)
            chosen = values[usable]
            finite = np.isfinite(chosen)
            if finite.any():
                matrix[row, slot[usable[finite]]] = chosen[finite].astype(np.float32, copy=False)


def _daily_ics(matrix: np.ndarray, forward: np.ndarray, calendar: list[date]):
    train_ics: list[float] = []
    test_ics: list[float] = []
    train_days: list[date] = []
    test_days: list[date] = []
    last = len(calendar) - HORIZON
    for slot, day in enumerate(calendar[:last]):
        future = calendar[slot + HORIZON]
        if future < CONFIRM_START:
            bucket = "train"
        elif day >= CONFIRM_START and future < HOLDOUT_START:
            bucket = "test"
        else:
            continue
        score = _rank_ic(matrix[:, slot], forward[:, slot])
        if not math.isfinite(score):
            continue
        if bucket == "train":
            train_ics.append(score)
            train_days.append(day)
        else:
            test_ics.append(score)
            test_days.append(day)
    return train_ics, test_ics, train_days, test_days


def _rank_ic(factor: np.ndarray, forward: np.ndarray) -> float:
    mask = np.isfinite(factor) & np.isfinite(forward)
    count = int(mask.sum())
    if count < MIN_NAMES:
        return float("nan")
    left = _average_rank(np.asarray(factor[mask], dtype=float))
    right = _average_rank(np.asarray(forward[mask], dtype=float))
    left = left - left.mean()
    right = right - right.mean()
    denom = math.sqrt(float(np.dot(left, left)) * float(np.dot(right, right)))
    if denom == 0.0:
        return float("nan")
    return float(np.dot(left, right) / denom)


def _average_rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ordered = values[order]
    ranks = np.empty(len(values), dtype=float)
    cursor = 0
    size = len(values)
    while cursor < size:
        nxt = cursor + 1
        while nxt < size and ordered[nxt] == ordered[cursor]:
            nxt += 1
        average = 0.5 * ((cursor + 1) + nxt)
        ranks[order[cursor:nxt]] = average
        cursor = nxt
    return ranks


def _span(days: list[date]) -> str:
    if not days:
        return ""
    return f"{days[0].isoformat()} ~ {days[-1].isoformat()}"
