"""Paper account. Fills are simulated from daily bars; no broker is contacted."""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path

from ashare.broker.base import AccountSnapshot, Broker, OrderAck, OrderRequest
from ashare.config import Settings
from ashare.market import SymbolSeries
from ashare.rules.costs import buy_cash_out, sell_cash_in
from ashare.rules.execution import evaluate_buy, evaluate_sell
from ashare.rules.lots import suggest_shares
from ashare.rules.money import D, money


@dataclass
class _Position:
    code: str
    name: str
    strategy_id: str
    strategy_name: str
    shares: int
    signal_date: str
    buy_date: str
    buy_price: str
    cost_cash: str
    stop: str
    take_profit: str
    max_hold_days: int
    sessions_held: int = 0


@dataclass
class _Pending:
    order_id: str
    code: str
    name: str
    strategy_id: str
    strategy_name: str
    shares_hint: int
    signal_date: str
    score: float
    entry_low: str
    entry_high: str
    take_profit_pct: float
    stop_loss_pct: float
    max_hold_days: int
    reason: str
    status: str = "pending"


@dataclass
class _Ledger:
    initial_cash: str
    cash: str
    created_at: str
    positions: list[_Position] = field(default_factory=list)
    pending: list[_Pending] = field(default_factory=list)
    fills: list[dict] = field(default_factory=list)
    equity: list[dict] = field(default_factory=list)
    settled_dates: list[str] = field(default_factory=list)


class PaperBroker(Broker):
    """JSON ledger under ``data/paper``. Replaying history uses a separate path."""

    name = "paper"

    def __init__(self, path: Path, settings: Settings) -> None:
        self.path = Path(path)
        self.settings = settings
        self.ledger = _load(self.path, settings)

    def place_orders(self, orders: list[OrderRequest]) -> list[OrderAck]:
        acks: list[OrderAck] = []
        existing = {item.code for item in self.ledger.pending if item.status == "pending"}
        held = {item.code for item in self.ledger.positions}
        for order in orders:
            if order.side != "buy":
                acks.append(OrderAck("", "rejected", "模拟盘委托目前只接受次日开盘买入"))
                continue
            if order.shares <= 0:
                acks.append(OrderAck("", "rejected", "股数为 0"))
                continue
            if order.code in existing or order.code in held:
                acks.append(OrderAck("", "rejected", "已有持仓或未成交委托"))
                continue
            pending = _Pending(
                order_id=uuid.uuid4().hex[:12],
                code=order.code,
                name=order.name,
                strategy_id=order.strategy_id,
                strategy_name=order.strategy_name,
                shares_hint=order.shares,
                signal_date=order.signal_date.isoformat(),
                score=order.score,
                entry_low=str(money(order.entry_low)),
                entry_high=str(money(order.entry_high)),
                take_profit_pct=order.take_profit_pct,
                stop_loss_pct=order.stop_loss_pct,
                max_hold_days=order.max_hold_days,
                reason=order.reason,
            )
            self.ledger.pending.append(pending)
            existing.add(order.code)
            acks.append(OrderAck(pending.order_id, "pending", "等待下一交易日开盘撮合"))
        self._save()
        return acks

    def cancel_order(self, order_id: str) -> OrderAck:
        for item in self.ledger.pending:
            if item.order_id == order_id and item.status == "pending":
                item.status = "cancelled"
                self._save()
                return OrderAck(order_id, "cancelled", "已撤销")
        return OrderAck(order_id, "rejected", "找不到未成交委托")

    def snapshot(self) -> AccountSnapshot:
        cash = float(D(self.ledger.cash))
        equity = float(self.ledger.equity[-1]["equity"]) if self.ledger.equity else cash
        return AccountSnapshot(
            cash=cash,
            equity=equity,
            positions=[asdict(item) for item in self.ledger.positions],
            pending=[asdict(item) for item in self.ledger.pending if item.status == "pending"],
            fills=list(self.ledger.fills),
        )

    def settle(self, day: date, series_by_code: dict[str, SymbolSeries]) -> list[str]:
        """Apply one session: open exits, open buys, late exits, then mark to market."""
        key = day.isoformat()
        if key in self.ledger.settled_dates:
            return [f"{key} 已结算，跳过"]
        notes: list[str] = []
        cash = D(self.ledger.cash)
        positions = self.ledger.positions
        notes.extend(self._exit_positions(positions, series_by_code, day, "open"))
        notes.extend(self._fill_buys(series_by_code, day))
        notes.extend(self._exit_positions(self.ledger.positions, series_by_code, day, "late"))
        for pos in self.ledger.positions:
            if pos.buy_date != key:
                pos.sessions_held += 1
        equity = self._mark(day, series_by_code)
        self.ledger.equity.append({"date": key, "equity": str(equity), "cash": self.ledger.cash})
        self.ledger.settled_dates.append(key)
        self._save()
        notes.append(f"{key} 结算完成，权益 {equity} 元，现金 {self.ledger.cash} 元")
        return notes

    def _fill_buys(self, series_by_code: dict[str, SymbolSeries], day: date) -> list[str]:
        notes: list[str] = []
        held = {item.code for item in self.ledger.positions}
        cash = D(self.ledger.cash)
        equity = D(self.ledger.equity[-1]["equity"]) if self.ledger.equity else D(self.ledger.initial_cash)
        pending = [item for item in self.ledger.pending if item.status == "pending"]
        pending.sort(key=lambda item: item.score, reverse=True)
        for order in pending:
            if date.fromisoformat(order.signal_date) >= day:
                continue
            if order.code in held or len(self.ledger.positions) >= self.settings.max_positions:
                order.status = "rejected"
                notes.append(f"{order.code} 未买入：持仓已满或已持有")
                continue
            series = series_by_code.get(order.code)
            index = None if series is None else series.date_index.get(day)
            if series is None or index is None or series.volume[index] <= 0:
                notes.append(f"{order.code} 停牌或无K线，委托继续等待")
                continue
            decision = evaluate_buy(
                open_=series.open[index],
                low=series.low[index],
                limit_up=series.limit_up[index],
                entry_low=order.entry_low,
                entry_high=order.entry_high,
                slippage=self.settings.slippage_rate,
            )
            if not decision.ok or decision.price is None:
                order.status = "rejected"
                notes.append(f"{order.code} 未买入：{decision.reason}")
                continue
            budget = self.settings.slot_budget(float(equity), float(cash))
            shares = suggest_shares(order.code, float(decision.price), budget, self.settings, day)
            if shares <= 0:
                order.status = "rejected"
                notes.append(f"{order.code} 未买入：不足一手")
                continue
            cost = buy_cash_out(shares, decision.price, day, self.settings)
            if cost > cash:
                order.status = "rejected"
                notes.append(f"{order.code} 未买入：现金不足")
                continue
            cash = money(cash - cost)
            self.ledger.cash = str(cash)
            self.ledger.positions.append(
                _Position(
                    code=order.code,
                    name=order.name,
                    strategy_id=order.strategy_id,
                    strategy_name=order.strategy_name,
                    shares=shares,
                    signal_date=order.signal_date,
                    buy_date=day.isoformat(),
                    buy_price=str(decision.price),
                    cost_cash=str(cost),
                    stop=str(money(decision.price * (Decimal("1") - D(order.stop_loss_pct)))),
                    take_profit=str(money(decision.price * (Decimal("1") + D(order.take_profit_pct)))),
                    max_hold_days=order.max_hold_days,
                    sessions_held=0,
                )
            )
            held.add(order.code)
            order.status = "filled"
            self.ledger.fills.append(
                {
                    "date": day.isoformat(),
                    "code": order.code,
                    "side": "buy",
                    "shares": shares,
                    "price": str(decision.price),
                    "cash_flow": str(-cost),
                    "reason": "开盘成交",
                }
            )
            notes.append(f"{order.code} 买入 {shares} 股 @ {decision.price}")
        return notes

    def _exit_positions(
        self,
        positions: list[_Position],
        series_by_code: dict[str, SymbolSeries],
        day: date,
        phase: str,
    ) -> list[str]:
        notes: list[str] = []
        kept: list[_Position] = []
        cash = D(self.ledger.cash)
        for pos in positions:
            if pos.buy_date == day.isoformat():
                kept.append(pos)
                continue
            # sessions_held counts completed sessions after the buy. The open
            # of the next session is day 1 and is already reflected before the
            # late phase increments the counter. Use sessions_held+1 as the
            # holding day count during this session.
            days_held = pos.sessions_held + 1
            series = series_by_code.get(pos.code)
            index = None if series is None else series.date_index.get(day)
            if series is None or index is None or series.volume[index] <= 0:
                kept.append(pos)
                if phase == "late":
                    notes.append(f"{pos.code} 停牌，卖出顺延")
                continue
            decision = evaluate_sell(
                open_=series.open[index],
                high=series.high[index],
                low=series.low[index],
                close=series.close[index],
                limit_up=series.limit_up[index],
                limit_down=series.limit_down[index],
                stop=pos.stop,
                take_profit=pos.take_profit,
                days_held=days_held,
                max_hold=pos.max_hold_days,
                slippage=self.settings.slippage_rate,
                phase=phase,
            )
            if not decision.ok or decision.price is None:
                kept.append(pos)
                if phase == "late" and decision.reason not in {"hold", ""}:
                    notes.append(f"{pos.code} 未能卖出：{decision.reason}")
                continue
            proceeds = sell_cash_in(pos.shares, decision.price, day, self.settings)
            cash = money(cash + proceeds)
            self.ledger.cash = str(cash)
            self.ledger.fills.append(
                {
                    "date": day.isoformat(),
                    "code": pos.code,
                    "side": "sell",
                    "shares": pos.shares,
                    "price": str(decision.price),
                    "cash_flow": str(proceeds),
                    "reason": decision.reason,
                }
            )
            notes.append(f"{pos.code} 卖出 {pos.shares} 股 @ {decision.price}（{decision.reason}）")
        if phase == "open":
            # Buys happen before the late pass; keep the surviving list in place.
            self.ledger.positions = kept
        else:
            self.ledger.positions = kept
        return notes

    def _mark(self, day: date, series_by_code: dict[str, SymbolSeries]) -> Decimal:
        total = D(self.ledger.cash)
        for pos in self.ledger.positions:
            series = series_by_code.get(pos.code)
            index = None if series is None else series.date_index.get(day)
            price = D(pos.buy_price) if index is None else money(
                D(series.close[index]) * (Decimal("1") - D(self.settings.slippage_rate))
            )
            total += sell_cash_in(pos.shares, price, day, self.settings)
        return money(total)

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(self.ledger)
        self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _load(path: Path, settings: Settings) -> _Ledger:
    if not path.exists():
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        cash = str(money(settings.initial_capital))
        return _Ledger(initial_cash=cash, cash=cash, created_at=now)
    payload = json.loads(path.read_text(encoding="utf-8"))
    return _Ledger(
        initial_cash=payload["initial_cash"],
        cash=payload["cash"],
        created_at=payload["created_at"],
        positions=[_Position(**item) for item in payload.get("positions", [])],
        pending=[_Pending(**item) for item in payload.get("pending", [])],
        fills=list(payload.get("fills", [])),
        equity=list(payload.get("equity", [])),
        settled_dates=list(payload.get("settled_dates", [])),
    )
