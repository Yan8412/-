"""Broker boundary. v1 ships only the paper implementation.

A later miniQMT adapter should implement this class and translate
``OrderRequest`` into ``xtquant`` calls. Nothing in the daily pipeline
imports xtquant.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date


@dataclass(frozen=True)
class OrderRequest:
    code: str
    name: str
    side: str
    shares: int
    signal_date: date
    strategy_id: str
    strategy_name: str
    reason: str
    entry_low: float
    entry_high: float
    take_profit_pct: float
    stop_loss_pct: float
    max_hold_days: int
    score: float = 0.0


@dataclass
class OrderAck:
    order_id: str
    status: str
    message: str


@dataclass
class AccountSnapshot:
    cash: float
    equity: float
    positions: list[dict] = field(default_factory=list)
    pending: list[dict] = field(default_factory=list)
    fills: list[dict] = field(default_factory=list)


class Broker(ABC):
    """Order port. Implementations either simulate fills or talk to a broker."""

    name: str

    @abstractmethod
    def place_orders(self, orders: list[OrderRequest]) -> list[OrderAck]:
        """Accept next-open orders. v1 does not fill them in this call."""

    @abstractmethod
    def cancel_order(self, order_id: str) -> OrderAck:
        """Cancel a still-pending order."""

    @abstractmethod
    def snapshot(self) -> AccountSnapshot:
        """Cash, positions, and recent fills."""
