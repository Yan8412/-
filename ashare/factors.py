"""Fifteen OHLCV factors, rewritten from the qlib Alpha158 definitions.

Source of the formulas: Microsoft Qlib ``qlib/contrib/data/loader.py``
(``Alpha158DL.get_feature_config``), Apache-2.0. HKUDS/Vibe-Trading's
``agent/src/factors/zoo/qlib158/`` is the same family. This module does not
copy either code base. Price inputs are this project's causal qfq series.
Volume is the raw share count; it is not split-adjusted.

Windows look backward only. ``ref(close, k)`` is ``close[t-k]``. A negative
shift is rejected.

BETA is qlib's ``Slope($close, d) / $close``. In ``qlib/data/_libs/rolling.pyx``
the time index of the newest point is the window length, so a full window is
regressed on ``1 .. d``. The qlib comment says a close that rises 10 per day
has slope 10. That is not a CAPM beta of the stock on the market. Vibe-Trading's
beta10 comment and a market-beta reading disagree with that code; the slope
above is what both the qlib expression and the cython updater compute.

STD uses the sample standard deviation (pandas ``rolling.std``, ddof=1), which
is what qlib's ``Rolling`` operator calls. Rank is the average-tie percentile
of the current close inside the window, matching ``Series.rolling(d).rank(pct=True)``
once the window is full. Bars before the window is full are NaN. Qlib also
emits a shorter window (``min_periods=1``); we do not, so an early bar cannot
pretend to be a full-window value. That is not a look into the future.
"""

from __future__ import annotations

import math

import numpy as np

from ashare.market import SymbolSeries

_EPS = 1e-12

# Stable order. The screen and the ranker both walk this tuple.
FACTOR_NAMES: tuple[str, ...] = (
    "KMID",
    "KLEN",
    "KUP",
    "KSFT",
    "RSV5",
    "ROC5",
    "MA5",
    "STD5",
    "BETA10",
    "RANK10",
    "CORR10",
    "CNTD5",
    "SUMP5",
    "MAX5",
    "VMA5",
)


def compute_factors(series: SymbolSeries, names: tuple[str, ...] | None = None) -> dict[str, np.ndarray]:
    """One array per name, aligned to ``series.dates``. Later bars are not read."""
    chosen = FACTOR_NAMES if names is None else names
    return {name: compute_one(series, name) for name in chosen}


def compute_one(series: SymbolSeries, name: str) -> np.ndarray:
    if name not in _FUNCS:
        raise KeyError(name)
    return _FUNCS[name](series)


def ref_array(values: np.ndarray, lag: int) -> np.ndarray:
    """``values[t-lag]``. ``lag`` must be positive so the result cannot see t+1."""
    if lag <= 0:
        raise ValueError("ref 只能向过去取，lag 必须为正")
    out = np.full(len(values), np.nan, dtype=float)
    if len(values) > lag:
        out[lag:] = values[:-lag]
    return out


def _prices(series: SymbolSeries) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return series.qfq_open, series.qfq_high, series.qfq_low, series.qfq_close


def _safe_div(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    out = np.full(len(num), np.nan, dtype=float)
    ok = np.isfinite(num) & np.isfinite(den) & (den != 0)
    out[ok] = num[ok] / den[ok]
    return out


def _kmid(series: SymbolSeries) -> np.ndarray:
    open_, _high, _low, close = _prices(series)
    return _safe_div(close - open_, open_)


def _klen(series: SymbolSeries) -> np.ndarray:
    open_, high, low, _close = _prices(series)
    return _safe_div(high - low, open_)


def _kup(series: SymbolSeries) -> np.ndarray:
    open_, high, _low, close = _prices(series)
    return _safe_div(high - np.maximum(open_, close), open_)


def _ksft(series: SymbolSeries) -> np.ndarray:
    open_, high, low, close = _prices(series)
    return _safe_div(2.0 * close - high - low, open_)


def _rsv(series: SymbolSeries, window: int = 5) -> np.ndarray:
    _open, high, low, close = _prices(series)
    out = np.full(len(close), np.nan, dtype=float)
    if len(close) < window:
        return out
    highs = np.lib.stride_tricks.sliding_window_view(high, window)
    lows = np.lib.stride_tricks.sliding_window_view(low, window)
    valid = np.isfinite(highs).all(axis=1) & np.isfinite(lows).all(axis=1) & np.isfinite(close[window - 1 :])
    span = highs.max(axis=1) - lows.min(axis=1)
    num = close[window - 1 :] - lows.min(axis=1)
    out[window - 1 :] = np.where(valid, num / (span + _EPS), np.nan)
    return out


def _roc(series: SymbolSeries, window: int = 5) -> np.ndarray:
    close = series.qfq_close
    return _safe_div(ref_array(close, window), close)


def _ma_ratio(series: SymbolSeries, window: int = 5) -> np.ndarray:
    close = series.qfq_close
    return _safe_div(_rolling_stat(close, window, "mean"), close)


def _std_ratio(series: SymbolSeries, window: int = 5) -> np.ndarray:
    close = series.qfq_close
    return _safe_div(_rolling_stat(close, window, "std"), close)


def _beta(series: SymbolSeries, window: int = 10) -> np.ndarray:
    close = series.qfq_close
    return _safe_div(rolling_slope(close, window), close)


def _rank(series: SymbolSeries, window: int = 10) -> np.ndarray:
    return rolling_rank_pct(series.qfq_close, window)


def _corr(series: SymbolSeries, window: int = 10) -> np.ndarray:
    close = series.qfq_close
    volume = np.log(series.volume + 1.0)
    return rolling_corr(close, volume, window)


def _cntd(series: SymbolSeries, window: int = 5) -> np.ndarray:
    close = series.qfq_close
    up = np.full(len(close), np.nan, dtype=float)
    down = np.full(len(close), np.nan, dtype=float)
    if len(close) < 2:
        return up
    prev = close[:-1]
    cur = close[1:]
    ok = np.isfinite(prev) & np.isfinite(cur)
    up[1:] = np.where(ok, (cur > prev).astype(float), np.nan)
    down[1:] = np.where(ok, (cur < prev).astype(float), np.nan)
    # The first comparison sits at index 1, so a full window ends at index ``window``.
    return _rolling_stat(up, window, "mean") - _rolling_stat(down, window, "mean")


def _sump(series: SymbolSeries, window: int = 5) -> np.ndarray:
    close = series.qfq_close
    delta = np.full(len(close), np.nan, dtype=float)
    if len(close) >= 2:
        prev = close[:-1]
        cur = close[1:]
        ok = np.isfinite(prev) & np.isfinite(cur)
        change = cur - prev
        delta[1:] = np.where(ok, change, np.nan)
    gain = np.where(np.isfinite(delta), np.maximum(delta, 0.0), np.nan)
    absolute = np.where(np.isfinite(delta), np.abs(delta), np.nan)
    total_gain = _rolling_stat(gain, window, "sum")
    total_abs = _rolling_stat(absolute, window, "sum")
    out = np.full(len(close), np.nan, dtype=float)
    ok = np.isfinite(total_gain) & np.isfinite(total_abs)
    out[ok] = total_gain[ok] / (total_abs[ok] + _EPS)
    return out


def _max_ratio(series: SymbolSeries, window: int = 5) -> np.ndarray:
    high = series.qfq_high
    close = series.qfq_close
    return _safe_div(_rolling_stat(high, window, "max"), close)


def _vma(series: SymbolSeries, window: int = 5) -> np.ndarray:
    volume = series.volume
    mean = _rolling_stat(volume, window, "mean")
    out = np.full(len(volume), np.nan, dtype=float)
    ok = np.isfinite(mean) & np.isfinite(volume)
    out[ok] = mean[ok] / (volume[ok] + _EPS)
    return out


def rolling_slope(values: np.ndarray, window: int) -> np.ndarray:
    """OLS slope of ``values`` on time index ``1 .. window``. NaN until the window is full."""
    out = np.full(len(values), np.nan, dtype=float)
    if window <= 1 or len(values) < window:
        return out
    view = np.lib.stride_tricks.sliding_window_view(np.asarray(values, dtype=float), window)
    valid = np.isfinite(view).all(axis=1)
    x = np.arange(1, window + 1, dtype=float)
    sum_x = float(x.sum())
    sum_x2 = float(np.dot(x, x))
    denom = window * sum_x2 - sum_x * sum_x
    if denom == 0:
        return out
    sum_y = view.sum(axis=1)
    sum_xy = view @ x
    slope = (window * sum_xy - sum_x * sum_y) / denom
    out[window - 1 :] = np.where(valid, slope, np.nan)
    return out


def rolling_rank_pct(values: np.ndarray, window: int) -> np.ndarray:
    """Average-tie percentile of the last point. 1 means it is strictly the maximum."""
    out = np.full(len(values), np.nan, dtype=float)
    if window <= 0 or len(values) < window:
        return out
    view = np.lib.stride_tricks.sliding_window_view(np.asarray(values, dtype=float), window)
    valid = np.isfinite(view).all(axis=1)
    current = view[:, -1:]
    less = (view < current).sum(axis=1)
    equal = (view == current).sum(axis=1)
    average = less + (equal + 1.0) / 2.0
    pct = average / float(window)
    out[window - 1 :] = np.where(valid, pct, np.nan)
    return out


def rolling_corr(left: np.ndarray, right: np.ndarray, window: int) -> np.ndarray:
    out = np.full(len(left), np.nan, dtype=float)
    if window <= 1 or len(left) < window or len(left) != len(right):
        return out
    a = np.lib.stride_tricks.sliding_window_view(np.asarray(left, dtype=float), window)
    b = np.lib.stride_tricks.sliding_window_view(np.asarray(right, dtype=float), window)
    valid = np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1)
    a_c = a - a.mean(axis=1, keepdims=True)
    b_c = b - b.mean(axis=1, keepdims=True)
    num = (a_c * b_c).sum(axis=1)
    den = np.sqrt((a_c * a_c).sum(axis=1) * (b_c * b_c).sum(axis=1))
    corr = np.full(len(num), np.nan, dtype=float)
    ok = valid & (den > 0)
    corr[ok] = num[ok] / den[ok]
    out[window - 1 :] = corr
    return out


def _rolling_stat(values: np.ndarray, window: int, kind: str) -> np.ndarray:
    out = np.full(len(values), np.nan, dtype=float)
    if window <= 0 or len(values) < window:
        return out
    view = np.lib.stride_tricks.sliding_window_view(np.asarray(values, dtype=float), window)
    valid = np.isfinite(view).all(axis=1)
    if kind == "mean":
        stat = view.mean(axis=1)
    elif kind == "std":
        stat = view.std(axis=1, ddof=1)
    elif kind == "sum":
        stat = view.sum(axis=1)
    elif kind == "max":
        stat = view.max(axis=1)
    else:
        raise ValueError(kind)
    out[window - 1 :] = np.where(valid, stat, np.nan)
    return out


_FUNCS = {
    "KMID": _kmid,
    "KLEN": _klen,
    "KUP": _kup,
    "KSFT": _ksft,
    "RSV5": _rsv,
    "ROC5": _roc,
    "MA5": _ma_ratio,
    "STD5": _std_ratio,
    "BETA10": _beta,
    "RANK10": _rank,
    "CORR10": _corr,
    "CNTD5": _cntd,
    "SUMP5": _sump,
    "MAX5": _max_ratio,
    "VMA5": _vma,
}


def finite_at(series: SymbolSeries, index: int, name: str) -> float:
    values = compute_one(series, name)
    if index < 0 or index >= len(values):
        return float("nan")
    value = float(values[index])
    if not math.isfinite(value):
        return float("nan")
    return value
