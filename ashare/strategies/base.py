"""Strategy contract. Signals may only read bars at index ``i`` and earlier."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ashare.market import SymbolSeries


@dataclass(frozen=True)
class Signal:
    score: float
    reason: str
    take_profit_pct: float
    stop_loss_pct: float
    max_hold_days: int
    entry_low_pct: float
    entry_high_pct: float


class Strategy:
    id: str
    name: str
    summary: str
    warmup: int = 60
    vectorized: bool = False

    def default_params(self) -> dict:
        raise NotImplementedError

    def param_grid(self) -> list[dict]:
        """Small grid used only inside walk-forward training windows."""
        return [self.default_params()]

    def signal(self, series: SymbolSeries, index: int, params: dict) -> Signal | None:
        raise NotImplementedError

    def scores(self, series: SymbolSeries, params: dict) -> np.ndarray:
        """Score at every bar. NaN means no signal. Override for full-market runs."""
        out = np.full(len(series.close), np.nan, dtype=float)
        for index in range(len(series.close)):
            signal = self.signal(series, index, params)
            if signal is not None:
                out[index] = signal.score
        return out
