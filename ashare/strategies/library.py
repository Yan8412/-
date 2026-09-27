"""Five short-term daily strategies.

Each one refuses a close that is already at the limit-up price. The order,
if any, is for the next session's open and is cancelled when that open is
outside the entry band or at the limit-up.
"""

from __future__ import annotations

import math

import numpy as np

from ashare.filters import limit_down_mask, limit_up_mask
from ashare.market import SymbolSeries
from ashare.rules.limits import is_limit_down, is_limit_up
from ashare.strategies.base import Signal, Strategy


def _finite(value: float) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)


def _return(series: SymbolSeries, index: int) -> float | None:
    preclose = series.preclose[index]
    if preclose <= 0:
        return None
    return float(series.close[index] / preclose - 1.0)


class FirstBoardFollow(Strategy):
    """Yesterday was a first limit-up; today is a bullish candle that did not seal again."""

    vectorized = True
    id = "first_board_follow"
    name = "首板次日承接"
    summary = "昨日首次涨停，今日收阳且未再次封板，次日开盘才考虑买入。"
    warmup = 60

    def default_params(self) -> dict:
        return {
            "min_ret": -0.01,
            "max_ret": 0.06,
            "max_vol_ratio": 2.2,
            "take_profit_pct": 0.06,
            "stop_loss_pct": 0.04,
            "max_hold_days": 3,
            "entry_low_pct": -0.03,
            "entry_high_pct": 0.02,
        }

    def param_grid(self) -> list[dict]:
        grid = []
        for hold in (2, 3):
            for take in (0.05, 0.08):
                params = self.default_params()
                params["max_hold_days"] = hold
                params["take_profit_pct"] = take
                grid.append(params)
        return grid

    def signal(self, series: SymbolSeries, index: int, params: dict) -> Signal | None:
        if index < 2:
            return None
        if not is_limit_up(series.close[index - 1], series.limit_up[index - 1]):
            return None
        if is_limit_up(series.close[index - 2], series.limit_up[index - 2]):
            return None
        if is_limit_up(series.close[index], series.limit_up[index]):
            return None
        if is_limit_down(series.close[index], series.limit_down[index]):
            return None
        if series.close[index] <= series.open[index]:
            return None
        change = _return(series, index)
        if change is None or change < params["min_ret"] or change > params["max_ret"]:
            return None
        previous_volume = series.volume[index - 1]
        if previous_volume <= 0 or series.volume[index] / previous_volume > params["max_vol_ratio"]:
            return None
        score = 60.0 + change * 80.0
        reason = (
            f"昨日首板、前日未涨停；今日收阳且涨幅 {change * 100:.1f}%，"
            "未封涨停，量能未极端放大。次日开盘未涨停且落在买入区间才下单。"
        )
        return _pack(score, reason, params)

    def scores(self, series: SymbolSeries, params: dict) -> np.ndarray:
        n = len(series.close)
        out = np.full(n, np.nan, dtype=float)
        if n < 3:
            return out
        up = limit_up_mask(series)
        down = limit_down_mask(series)
        up_prev = np.zeros(n, dtype=bool)
        up_prev2 = np.zeros(n, dtype=bool)
        up_prev[1:] = up[:-1]
        up_prev2[2:] = up[:-2]
        change = _change_array(series)
        previous_volume = np.zeros(n, dtype=float)
        previous_volume[1:] = series.volume[:-1]
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = series.volume / previous_volume
        index = np.arange(n)
        keep = (
            (index >= 2)
            & up_prev
            & ~up_prev2
            & ~up
            & ~down
            & (series.close > series.open)
            & np.isfinite(change)
            & (change >= params["min_ret"])
            & (change <= params["max_ret"])
            & (previous_volume > 0)
            & (ratio <= params["max_vol_ratio"])
        )
        out[keep] = 60.0 + change[keep] * 80.0
        return out


class Breakout60(Strategy):
    """Close breaks the prior 60-session high on qfq prices, with volume confirmation."""

    vectorized = True
    id = "breakout_60"
    name = "放量突破60日高"
    summary = "除权连续价收盘突破过去60个交易日最高价，且放量、收盘靠近最高价，当日未涨停。"
    warmup = 70

    def default_params(self) -> dict:
        return {
            "min_ret": 0.02,
            "max_ret": 0.18,
            "vol_ratio": 1.5,
            "close_pos": 0.65,
            "take_profit_pct": 0.10,
            "stop_loss_pct": 0.06,
            "max_hold_days": 8,
            "entry_low_pct": -0.025,
            "entry_high_pct": 0.015,
        }

    def param_grid(self) -> list[dict]:
        grid = []
        for vol_ratio in (1.3, 1.8):
            for hold in (5, 8):
                params = self.default_params()
                params["vol_ratio"] = vol_ratio
                params["max_hold_days"] = hold
                grid.append(params)
        return grid

    def signal(self, series: SymbolSeries, index: int, params: dict) -> Signal | None:
        if index < 60:
            return None
        prior = float(series.qfq_high[index - 60 : index].max())
        if not (series.qfq_close[index] > prior):
            return None
        change = _return(series, index)
        if change is None or change < params["min_ret"] or change > params["max_ret"]:
            return None
        base = series.vol_ma5[index - 1]
        if not _finite(base) or base <= 0 or series.volume[index] / base < params["vol_ratio"]:
            return None
        span = series.high[index] - series.low[index]
        if span > 0 and (series.close[index] - series.low[index]) / span < params["close_pos"]:
            return None
        extension = series.qfq_close[index] / prior - 1.0
        score = 55.0 + min(change, 0.1) * 100.0 + min(extension, 0.05) * 80.0
        reason = (
            f"收盘突破前60日高点，当日涨幅 {change * 100:.1f}%，"
            f"量能达到近5日均量的 {series.volume[index] / base:.1f} 倍，收盘未涨停。"
        )
        return _pack(score, reason, params)

    def scores(self, series: SymbolSeries, params: dict) -> np.ndarray:
        n = len(series.close)
        out = np.full(n, np.nan, dtype=float)
        if n < 61:
            return out
        windows = np.lib.stride_tricks.sliding_window_view(series.qfq_high, 60).max(axis=1)
        prior = np.full(n, np.nan, dtype=float)
        prior[60:] = windows[: n - 60]
        change = _change_array(series)
        base = np.full(n, np.nan, dtype=float)
        base[1:] = series.vol_ma5[:-1]
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = series.volume / base
        span = series.high - series.low
        position = np.ones(n, dtype=float)
        nonzero = span > 0
        position[nonzero] = (series.close[nonzero] - series.low[nonzero]) / span[nonzero]
        index = np.arange(n)
        keep = (
            (index >= 60)
            & np.isfinite(prior)
            & (series.qfq_close > prior)
            & np.isfinite(change)
            & (change >= params["min_ret"])
            & (change <= params["max_ret"])
            & np.isfinite(base)
            & (base > 0)
            & (ratio >= params["vol_ratio"])
            & (position >= params["close_pos"])
        )
        extension = np.zeros(n, dtype=float)
        good_prior = np.isfinite(prior) & (prior != 0)
        extension[good_prior] = series.qfq_close[good_prior] / prior[good_prior] - 1.0
        out[keep] = (
            55.0
            + np.minimum(change[keep], 0.1) * 100.0
            + np.minimum(extension[keep], 0.05) * 80.0
        )
        return out


class MaPullback(Strategy):
    """Uptrend pullback that tags MA20 and closes back above it."""

    vectorized = True
    id = "ma_pullback"
    name = "均线回踩"
    summary = "MA5>MA20>MA60 且均线向上，当日回踩 MA20 后收阳站回，未涨停。"
    warmup = 65

    def default_params(self) -> dict:
        return {
            "proximity": 0.02,
            "max_ret": 0.09,
            "take_profit_pct": 0.08,
            "stop_loss_pct": 0.04,
            "max_hold_days": 8,
            "entry_low_pct": -0.02,
            "entry_high_pct": 0.012,
        }

    def param_grid(self) -> list[dict]:
        grid = []
        for proximity in (0.015, 0.03):
            for hold in (5, 8):
                params = self.default_params()
                params["proximity"] = proximity
                params["max_hold_days"] = hold
                grid.append(params)
        return grid

    def signal(self, series: SymbolSeries, index: int, params: dict) -> Signal | None:
        ma5 = series.ma5[index]
        ma20 = series.ma20[index]
        ma60 = series.ma60[index]
        ma20_prev = series.ma20[index - 5] if index >= 5 else math.nan
        if not all(_finite(item) for item in (ma5, ma20, ma60, ma20_prev)):
            return None
        if not (ma5 > ma20 > ma60 and ma20 > ma20_prev):
            return None
        if series.qfq_low[index] > ma20 * (1.0 + params["proximity"]):
            return None
        if series.qfq_close[index] < ma20:
            return None
        if series.close[index] <= series.open[index]:
            return None
        change = _return(series, index)
        if change is None or change <= 0 or change > params["max_ret"]:
            return None
        distance = abs(series.qfq_close[index] / ma20 - 1.0)
        score = 50.0 + (params["proximity"] - distance) * 200.0 + change * 40.0
        reason = (
            f"多头排列，最低价触及 MA20 附近后收阳站回，当日涨幅 {change * 100:.1f}%，未涨停。"
        )
        return _pack(score, reason, params)

    def scores(self, series: SymbolSeries, params: dict) -> np.ndarray:
        n = len(series.close)
        out = np.full(n, np.nan, dtype=float)
        if n < 2:
            return out
        ma20_prev = _shift(series.ma20, 5)
        change = _change_array(series)
        finite = (
            np.isfinite(series.ma5)
            & np.isfinite(series.ma20)
            & np.isfinite(series.ma60)
            & np.isfinite(ma20_prev)
        )
        keep = (
            finite
            & (series.ma5 > series.ma20)
            & (series.ma20 > series.ma60)
            & (series.ma20 > ma20_prev)
            & (series.qfq_low <= series.ma20 * (1.0 + params["proximity"]))
            & (series.qfq_close >= series.ma20)
            & (series.close > series.open)
            & np.isfinite(change)
            & (change > 0)
            & (change <= params["max_ret"])
        )
        distance = np.abs(series.qfq_close / series.ma20 - 1.0)
        out[keep] = 50.0 + (params["proximity"] - distance[keep]) * 200.0 + change[keep] * 40.0
        return out


class MacdGolden(Strategy):
    """MACD golden cross below zero while price holds MA20."""

    vectorized = True
    id = "macd_golden"
    name = "MACD零下金叉"
    summary = "DIF 在零轴下方上穿 DEA，收盘仍在 MA20 之上且放量，当日未涨停。"
    warmup = 60

    def default_params(self) -> dict:
        return {
            "below_zero": True,
            "min_vol_ratio": 1.0,
            "take_profit_pct": 0.08,
            "stop_loss_pct": 0.05,
            "max_hold_days": 8,
            "entry_low_pct": -0.025,
            "entry_high_pct": 0.015,
        }

    def param_grid(self) -> list[dict]:
        grid = []
        for below in (True, False):
            for hold in (6, 10):
                params = self.default_params()
                params["below_zero"] = below
                params["max_hold_days"] = hold
                grid.append(params)
        return grid

    def signal(self, series: SymbolSeries, index: int, params: dict) -> Signal | None:
        if index < 1:
            return None
        dif_now = series.dif[index]
        dea_now = series.dea[index]
        dif_prev = series.dif[index - 1]
        dea_prev = series.dea[index - 1]
        ma20 = series.ma20[index]
        if not all(_finite(item) for item in (dif_now, dea_now, dif_prev, dea_prev, ma20)):
            return None
        if not (dif_prev <= dea_prev and dif_now > dea_now):
            return None
        if params.get("below_zero", True) and dif_now >= 0:
            return None
        if series.qfq_close[index] < ma20:
            return None
        base = series.vol_ma5[index - 1]
        if not _finite(base) or base <= 0 or series.volume[index] / base < params["min_vol_ratio"]:
            return None
        change = _return(series, index)
        if change is None or change <= 0:
            return None
        score = 48.0 + min(dif_now - dif_prev, 0.5) * 20.0 + change * 50.0
        where = "零轴下方" if dif_now < 0 else "零轴附近或上方"
        reason = f"MACD 在{where}金叉，收盘站上 MA20，当日涨幅 {change * 100:.1f}%，未涨停。"
        return _pack(score, reason, params)

    def scores(self, series: SymbolSeries, params: dict) -> np.ndarray:
        n = len(series.close)
        out = np.full(n, np.nan, dtype=float)
        if n < 2:
            return out
        dif_prev = _shift(series.dif, 1)
        dea_prev = _shift(series.dea, 1)
        base = _shift(series.vol_ma5, 1)
        change = _change_array(series)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = series.volume / base
        finite = (
            np.isfinite(series.dif)
            & np.isfinite(series.dea)
            & np.isfinite(dif_prev)
            & np.isfinite(dea_prev)
            & np.isfinite(series.ma20)
        )
        keep = (
            finite
            & (dif_prev <= dea_prev)
            & (series.dif > series.dea)
            & (series.qfq_close >= series.ma20)
            & np.isfinite(base)
            & (base > 0)
            & (ratio >= params["min_vol_ratio"])
            & np.isfinite(change)
            & (change > 0)
        )
        if params.get("below_zero", True):
            keep &= series.dif < 0
        delta = series.dif - dif_prev
        out[keep] = 48.0 + np.minimum(delta[keep], 0.5) * 20.0 + change[keep] * 50.0
        return out


class ShrinkReversal(Strategy):
    """Three shrinking down days inside an uptrend, then an expanding bullish day."""

    vectorized = True
    id = "shrink_reversal"
    name = "缩量回调再放量"
    summary = "MA20 向上，连续三日缩量回调后，今日放量收阳且未涨停。"
    warmup = 30

    def default_params(self) -> dict:
        return {
            "expand": 1.4,
            "max_ret": 0.09,
            "take_profit_pct": 0.06,
            "stop_loss_pct": 0.04,
            "max_hold_days": 5,
            "entry_low_pct": -0.025,
            "entry_high_pct": 0.015,
        }

    def param_grid(self) -> list[dict]:
        grid = []
        for expand in (1.3, 1.8):
            for hold in (3, 5):
                params = self.default_params()
                params["expand"] = expand
                params["max_hold_days"] = hold
                grid.append(params)
        return grid

    def signal(self, series: SymbolSeries, index: int, params: dict) -> Signal | None:
        if index < 25:
            return None
        ma20 = series.ma20[index]
        ma20_prev = series.ma20[index - 5]
        if not (_finite(ma20) and _finite(ma20_prev) and ma20 > ma20_prev):
            return None
        closes = series.qfq_close
        volumes = series.volume
        pullback = closes[index - 1] < closes[index - 2] < closes[index - 3]
        shrink = volumes[index - 1] < volumes[index - 2] < volumes[index - 3]
        if not (pullback and shrink):
            return None
        if series.close[index] <= series.open[index]:
            return None
        if volumes[index - 1] <= 0 or volumes[index] < volumes[index - 1] * params["expand"]:
            return None
        if series.qfq_close[index] < ma20:
            return None
        change = _return(series, index)
        if change is None or change <= 0 or change > params["max_ret"]:
            return None
        score = 52.0 + (volumes[index] / volumes[index - 1]) * 4.0 + change * 40.0
        reason = (
            f"上升趋势中连续三日缩量回落，今日放量收阳，涨幅 {change * 100:.1f}%，未涨停。"
        )
        return _pack(score, reason, params)

    def scores(self, series: SymbolSeries, params: dict) -> np.ndarray:
        n = len(series.close)
        out = np.full(n, np.nan, dtype=float)
        if n < 26:
            return out
        ma20_prev = _shift(series.ma20, 5)
        close_1 = _shift(series.qfq_close, 1)
        close_2 = _shift(series.qfq_close, 2)
        close_3 = _shift(series.qfq_close, 3)
        vol_1 = _shift(series.volume, 1)
        vol_2 = _shift(series.volume, 2)
        vol_3 = _shift(series.volume, 3)
        change = _change_array(series)
        with np.errstate(divide="ignore", invalid="ignore"):
            expand = series.volume / vol_1
        index = np.arange(n)
        keep = (
            (index >= 25)
            & np.isfinite(series.ma20)
            & np.isfinite(ma20_prev)
            & (series.ma20 > ma20_prev)
            & (close_1 < close_2)
            & (close_2 < close_3)
            & (vol_1 < vol_2)
            & (vol_2 < vol_3)
            & (series.close > series.open)
            & (vol_1 > 0)
            & (expand >= params["expand"])
            & (series.qfq_close >= series.ma20)
            & np.isfinite(change)
            & (change > 0)
            & (change <= params["max_ret"])
        )
        out[keep] = 52.0 + expand[keep] * 4.0 + change[keep] * 40.0
        return out


def _shift(values: np.ndarray, bars: int) -> np.ndarray:
    out = np.full(len(values), np.nan, dtype=float)
    if bars <= 0 or bars >= len(values):
        return out
    out[bars:] = values[:-bars]
    return out


def _change_array(series: SymbolSeries) -> np.ndarray:
    out = np.full(len(series.close), np.nan, dtype=float)
    ok = series.preclose > 0
    out[ok] = series.close[ok] / series.preclose[ok] - 1.0
    return out


def _pack(score: float, reason: str, params: dict) -> Signal:
    return Signal(
        score=float(score),
        reason=reason,
        take_profit_pct=float(params["take_profit_pct"]),
        stop_loss_pct=float(params["stop_loss_pct"]),
        max_hold_days=int(params["max_hold_days"]),
        entry_low_pct=float(params["entry_low_pct"]),
        entry_high_pct=float(params["entry_high_pct"]),
    )


def builtin_strategies() -> list[Strategy]:
    return [
        FirstBoardFollow(),
        Breakout60(),
        MaPullback(),
        MacdGolden(),
        ShrinkReversal(),
    ]
