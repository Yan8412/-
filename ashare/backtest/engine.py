"""Daily bar simulator with T+1, limit-up/down rejects, fees, and 100-share lots."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import date
from decimal import Decimal

import numpy as np

from ashare.config import Settings
from ashare.filters import bar_halted, eligible_mask, rejection_reason
from ashare.market import SymbolSeries, master_calendar
from ashare.rules.costs import buy_cash_out, sell_cash_in
from ashare.rules.execution import evaluate_buy, evaluate_sell
from ashare.rules.lots import suggest_shares
from ashare.rules.money import D, money
from ashare.strategies.base import Strategy


@dataclass
class ClosedTrade:
    code: str
    name: str
    strategy_id: str
    strategy_name: str
    shares: int
    signal_date: date
    buy_date: date
    sell_date: date
    buy_price: float
    sell_price: float
    buy_cost: float
    sell_proceeds: float
    pnl: float
    reason: str


@dataclass
class Position:
    code: str
    name: str
    strategy_id: str
    strategy_name: str
    shares: int
    signal_date: date
    buy_date: date
    buy_price: Decimal
    cost_cash: Decimal
    stop: Decimal
    take_profit: Decimal
    max_hold_days: int


@dataclass
class PendingBuy:
    code: str
    name: str
    strategy_id: str
    strategy_name: str
    signal_date: date
    score: float
    entry_low: Decimal
    entry_high: Decimal
    take_profit_pct: float
    stop_loss_pct: float
    max_hold_days: int
    reason: str


@dataclass
class BacktestResult:
    strategy_id: str
    strategy_name: str
    params: dict
    start: date
    end: date
    initial_capital: float
    final_equity: float
    equity_dates: list[date] = field(default_factory=list)
    equity: list[float] = field(default_factory=list)
    trades: list[ClosedTrade] = field(default_factory=list)
    rejects: dict[str, int] = field(default_factory=dict)
    open_positions: int = 0


def run_backtest(
    symbols: list[SymbolSeries],
    strategies: list[tuple[Strategy, dict]],
    settings: Settings,
    start: date,
    end: date,
    *,
    label: str | None = None,
    param_schedule: dict[str, list[tuple[date, date, dict]]] | None = None,
    entry_allowed: dict[date, bool] | None = None,
    rescore: Callable[[PendingBuy], float | None] | None = None,
    pool_size: int | None = None,
    prepared_book: dict[date, list[PendingBuy]] | None = None,
) -> BacktestResult:
    """Simulate one account.

    ``param_schedule`` maps a strategy id to inclusive ``(test_start, test_end, params)``
    windows. On dates outside every window the strategy does not open new trades.
    Training selection must pass windows that were chosen without the test prices.

    ``entry_allowed`` blocks new entries on sessions mapped to false. Open positions
    still exit. ``rescore`` replaces the rule score and keeps the top ``top_n``;
    ``pool_size`` is how many rule-ranked names are offered to that rescore.
    ``prepared_book`` skips rebuilding the rule book when the caller already has it.
    """
    calendar = [day for day in master_calendar(symbols) if start <= day <= end]
    by_code = {item.code: item for item in symbols}
    if len(strategies) == 1 and label is None:
        strategy_id = strategies[0][0].id
        strategy_name = strategies[0][0].name
        params = strategies[0][1]
    else:
        strategy_id = "combined"
        strategy_name = label or "多策略组合"
        params = {item[0].id: item[1] for item in strategies}

    cash = money(settings.initial_capital)
    positions: list[Position] = []
    pending: list[PendingBuy] = []
    fast = all(getattr(strategy, "vectorized", False) for strategy, _params in strategies)
    cap = settings.top_n if pool_size is None else pool_size
    if prepared_book is not None:
        book = prepared_book
    elif fast:
        book = _signal_book(list(by_code.values()), strategies, settings, param_schedule, cap=cap)
    else:
        book = None
    trades: list[ClosedTrade] = []
    rejects: dict[str, int] = {}
    equity_dates: list[date] = []
    equity: list[float] = []
    previous_equity = D(settings.initial_capital)
    index_of = {day: index for index, day in enumerate(calendar)}

    for offset, day in enumerate(calendar):
        next_day = calendar[offset + 1] if offset + 1 < len(calendar) else None
        positions, cash = _sell_phase(
            positions, cash, by_code, day, index_of, settings, trades, rejects, phase="open"
        )
        positions, cash = _buy_phase(
            pending, positions, cash, previous_equity, by_code, day, settings, rejects
        )
        pending = []
        positions, cash = _sell_phase(
            positions, cash, by_code, day, index_of, settings, trades, rejects, phase="late"
        )
        marked = _mark(cash, positions, by_code, day, settings)
        equity_dates.append(day)
        equity.append(float(marked))
        previous_equity = marked
        if next_day is None:
            continue
        if book is None:
            pending = _collect_signals(by_code, day, strategies, settings, param_schedule, cap=cap)
        else:
            pending = list(book.get(day, []))
        if entry_allowed is not None and not entry_allowed.get(day, True):
            for _order in pending:
                _bump(rejects, "行情过滤不开新仓")
            pending = []
        elif rescore is not None:
            scored: list[PendingBuy] = []
            for order in pending:
                value = rescore(order)
                if value is None:
                    continue
                scored.append(replace(order, score=float(value)))
            scored.sort(key=lambda item: item.score, reverse=True)
            pending = scored[: settings.top_n]

    return BacktestResult(
        strategy_id=strategy_id,
        strategy_name=strategy_name,
        params=params,
        start=start,
        end=end,
        initial_capital=settings.initial_capital,
        final_equity=equity[-1] if equity else settings.initial_capital,
        equity_dates=equity_dates,
        equity=equity,
        trades=trades,
        rejects=rejects,
        open_positions=len(positions),
    )


def _bump(rejects: dict[str, int], reason: str) -> None:
    rejects[reason] = rejects.get(reason, 0) + 1


def _sell_phase(
    positions: list[Position],
    cash: Decimal,
    by_code: dict[str, SymbolSeries],
    day: date,
    index_of: dict[date, int],
    settings: Settings,
    trades: list[ClosedTrade],
    rejects: dict[str, int],
    phase: str,
) -> tuple[list[Position], Decimal]:
    kept: list[Position] = []
    for pos in positions:
        # T+1: shares bought today are not sellable, and the open phase of the
        # buy day has already passed by the time the position exists.
        if pos.buy_date == day:
            kept.append(pos)
            continue
        series = by_code[pos.code]
        index = series.date_index.get(day)
        if index is None or bar_halted(series, index):
            if phase == "late":
                _bump(rejects, "停牌顺延")
            kept.append(pos)
            continue
        days_held = index_of[day] - index_of[pos.buy_date]
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
            slippage=settings.slippage_rate,
            phase=phase,
        )
        if not decision.ok or decision.price is None:
            if decision.reason not in {"hold", ""}:
                _bump(rejects, decision.reason)
            kept.append(pos)
            continue
        proceeds = sell_cash_in(pos.shares, decision.price, day, settings)
        cash = money(cash + proceeds)
        trades.append(
            ClosedTrade(
                code=pos.code,
                name=pos.name,
                strategy_id=pos.strategy_id,
                strategy_name=pos.strategy_name,
                shares=pos.shares,
                signal_date=pos.signal_date,
                buy_date=pos.buy_date,
                sell_date=day,
                buy_price=float(pos.buy_price),
                sell_price=float(decision.price),
                buy_cost=float(pos.cost_cash),
                sell_proceeds=float(proceeds),
                pnl=float(money(proceeds - pos.cost_cash)),
                reason=decision.reason,
            )
        )
    return kept, cash


def _buy_phase(
    pending: list[PendingBuy],
    positions: list[Position],
    cash: Decimal,
    previous_equity: Decimal,
    by_code: dict[str, SymbolSeries],
    day: date,
    settings: Settings,
    rejects: dict[str, int],
) -> tuple[list[Position], Decimal]:
    held = {pos.code for pos in positions}
    for order in sorted(pending, key=lambda item: item.score, reverse=True):
        if order.code in held:
            continue
        if len(positions) >= settings.max_positions:
            _bump(rejects, "持仓数量已满")
            continue
        series = by_code.get(order.code)
        if series is None:
            continue
        index = series.date_index.get(day)
        if index is None or bar_halted(series, index):
            _bump(rejects, "停牌无法买入")
            continue
        decision = evaluate_buy(
            open_=series.open[index],
            low=series.low[index],
            limit_up=series.limit_up[index],
            entry_low=order.entry_low,
            entry_high=order.entry_high,
            slippage=settings.slippage_rate,
        )
        if not decision.ok or decision.price is None:
            _bump(rejects, decision.reason)
            continue
        budget = settings.slot_budget(float(previous_equity), float(cash))
        shares = suggest_shares(order.code, float(decision.price), budget, settings, day)
        if shares <= 0:
            _bump(rejects, "不足一手")
            continue
        cost = buy_cash_out(shares, decision.price, day, settings)
        if cost > cash:
            _bump(rejects, "现金不足")
            continue
        cash = money(cash - cost)
        positions.append(
            Position(
                code=order.code,
                name=order.name,
                strategy_id=order.strategy_id,
                strategy_name=order.strategy_name,
                shares=shares,
                signal_date=order.signal_date,
                buy_date=day,
                buy_price=decision.price,
                cost_cash=cost,
                stop=money(decision.price * (Decimal("1") - D(order.stop_loss_pct))),
                take_profit=money(decision.price * (Decimal("1") + D(order.take_profit_pct))),
                max_hold_days=order.max_hold_days,
            )
        )
        held.add(order.code)
    return positions, cash


def _mark(
    cash: Decimal,
    positions: list[Position],
    by_code: dict[str, SymbolSeries],
    day: date,
    settings: Settings,
) -> Decimal:
    """Equity after estimated exit costs, so a fresh buy is not marked as a gain."""
    total = cash
    for pos in positions:
        series = by_code[pos.code]
        index = series.date_index.get(day)
        if index is None:
            price = pos.buy_price
        else:
            price = money(D(str(series.close[index])) * (Decimal("1") - D(settings.slippage_rate)))
        total += sell_cash_in(pos.shares, price, day, settings)
    return money(total)


def _signal_book(
    symbols: list[SymbolSeries],
    strategies: list[tuple[Strategy, dict]],
    settings: Settings,
    param_schedule: dict[str, list[tuple[date, date, dict]]] | None,
    cap: int | None = None,
) -> dict[date, list[PendingBuy]]:
    """Precompute the same top-N orders ``_collect_signals`` would emit.

    Used when every strategy publishes a vectorized score array. The daily
    shortlist still calls ``_collect_signals`` so the written reason stays
    the Chinese sentence from ``signal``.
    """
    eligible = {series.code: eligible_mask(series, settings) for series in symbols}
    variants: list[tuple[Strategy, dict, tuple]] = []
    seen: set[tuple] = set()
    for strategy, default_params in strategies:
        for params in _param_variants(strategy.id, default_params, param_schedule):
            key = (strategy.id, _param_token(params))
            if key in seen:
                continue
            seen.add(key)
            variants.append((strategy, params, key))

    hits: dict[tuple, dict[date, list[tuple[float, SymbolSeries]]]] = {
        key: defaultdict(list) for _strategy, _params, key in variants
    }
    for strategy, params, key in variants:
        grouped = hits[key]
        for series in symbols:
            scores = strategy.scores(series, params)
            mask = eligible[series.code] & np.isfinite(scores)
            if strategy.warmup > 0:
                warmup = np.arange(len(scores)) >= strategy.warmup
                mask = mask & warmup
            for index in np.flatnonzero(mask).tolist():
                grouped[series.dates[index]].append((float(scores[index]), series))

    book: dict[date, list[PendingBuy]] = {}
    signal_days: set[date] = set()
    for grouped in hits.values():
        signal_days.update(grouped)
    for day in signal_days:
        best: dict[str, tuple[float, SymbolSeries, Strategy, dict]] = {}
        for strategy, default_params in strategies:
            params = _params_for(strategy.id, day, default_params, param_schedule)
            if params is None:
                continue
            key = (strategy.id, _param_token(params))
            for score, series in hits[key].get(day, ()):
                current = best.get(series.code)
                if current is None or score > current[0]:
                    best[series.code] = (score, series, strategy, params)
        orders: list[PendingBuy] = []
        for score, series, strategy, params in best.values():
            index = series.date_index[day]
            close = money(series.close[index])
            orders.append(
                PendingBuy(
                    code=series.code,
                    name=series.name,
                    strategy_id=strategy.id,
                    strategy_name=strategy.name,
                    signal_date=day,
                    score=score,
                    entry_low=money(close * (Decimal("1") + D(params["entry_low_pct"]))),
                    entry_high=money(close * (Decimal("1") + D(params["entry_high_pct"]))),
                    take_profit_pct=float(params["take_profit_pct"]),
                    stop_loss_pct=float(params["stop_loss_pct"]),
                    max_hold_days=int(params["max_hold_days"]),
                    reason="",
                )
            )
        orders.sort(key=lambda item: item.score, reverse=True)
        limit = settings.top_n if cap is None else cap
        book[day] = orders[:limit]
    return book


def _param_variants(
    strategy_id: str,
    default_params: dict,
    schedule: dict[str, list[tuple[date, date, dict]]] | None,
) -> list[dict]:
    if schedule is None:
        return [default_params]
    found: list[dict] = []
    seen: set[str] = set()
    for _start, _end, params in schedule.get(strategy_id, []):
        token = _param_token(params)
        if token in seen:
            continue
        seen.add(token)
        found.append(params)
    return found


def _param_token(params: dict) -> str:
    return json.dumps(params, sort_keys=True, default=str)


def _collect_signals(
    by_code: dict[str, SymbolSeries],
    day: date,
    strategies: list[tuple[Strategy, dict]],
    settings: Settings,
    param_schedule: dict[str, list[tuple[date, date, dict]]] | None,
    cap: int | None = None,
) -> list[PendingBuy]:
    orders: list[PendingBuy] = []
    for series in by_code.values():
        index = series.date_index.get(day)
        if index is None:
            continue
        if rejection_reason(series, index, settings):
            continue
        best: PendingBuy | None = None
        for strategy, default_params in strategies:
            params = _params_for(strategy.id, day, default_params, param_schedule)
            if params is None:
                continue
            if index < strategy.warmup:
                continue
            signal = strategy.signal(series, index, params)
            if signal is None:
                continue
            close = money(series.close[index])
            order = PendingBuy(
                code=series.code,
                name=series.name,
                strategy_id=strategy.id,
                strategy_name=strategy.name,
                signal_date=day,
                score=signal.score,
                entry_low=money(close * (Decimal("1") + D(signal.entry_low_pct))),
                entry_high=money(close * (Decimal("1") + D(signal.entry_high_pct))),
                take_profit_pct=signal.take_profit_pct,
                stop_loss_pct=signal.stop_loss_pct,
                max_hold_days=signal.max_hold_days,
                reason=signal.reason,
            )
            if best is None or order.score > best.score:
                best = order
        if best is not None:
            orders.append(best)
    orders.sort(key=lambda item: item.score, reverse=True)
    limit = settings.top_n if cap is None else cap
    return orders[:limit]


def _params_for(
    strategy_id: str,
    day: date,
    default_params: dict,
    schedule: dict[str, list[tuple[date, date, dict]]] | None,
) -> dict | None:
    if schedule is None:
        return default_params
    windows = schedule.get(strategy_id, [])
    for start, end, params in windows:
        if start <= day <= end:
            return params
    return None
