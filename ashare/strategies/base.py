"""Strategy contract. Signals may only read bars at index ``i`` and earlier."""

from __future__ import annotations

from dataclasses import dataclass

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

    def default_params(self) -> dict:
        raise NotImplementedError

    def param_grid(self) -> list[dict]:
        """Small grid used only inside walk-forward training windows."""
        return [self.default_params()]

    def signal(self, series: SymbolSeries, index: int, params: dict) -> Signal | None:
        raise NotImplementedError
