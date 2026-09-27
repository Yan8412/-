"""Rolling walk-forward. Parameter choice on a fold never sees that fold's test dates."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date

from ashare.backtest.engine import BacktestResult, run_backtest
from ashare.backtest.metrics import Metrics, selection_score, summarize
from ashare.config import Settings
from ashare.market import SymbolSeries
from ashare.strategies.base import Strategy


@dataclass(frozen=True)
class Fold:
    train_start: date
    train_end: date
    test_start: date
    test_end: date


@dataclass
class FoldOutcome:
    fold: Fold
    strategy_id: str
    chosen_params: dict
    train_score: float
    train_metrics: Metrics
    strategy_name: str = ""


logger = logging.getLogger(__name__)


def iter_folds(days: list[date], train_days: int, test_days: int) -> list[Fold]:
    """Contiguous test blocks. Each train window ends the session before its test."""
    if train_days <= 0 or test_days <= 0:
        raise ValueError("训练窗口和测试窗口必须为正")
    folds: list[Fold] = []
    test_start_index = train_days
    while test_start_index + test_days <= len(days):
        train = days[test_start_index - train_days : test_start_index]
        test = days[test_start_index : test_start_index + test_days]
        folds.append(
            Fold(
                train_start=train[0],
                train_end=train[-1],
                test_start=test[0],
                test_end=test[-1],
            )
        )
        test_start_index += test_days
    return folds


def select_params(
    symbols: list[SymbolSeries],
    strategy: Strategy,
    settings: Settings,
    fold: Fold,
) -> tuple[dict, float, Metrics]:
    """Pick the grid point with the best in-sample score. Ties keep the earlier point."""
    best_params = strategy.default_params()
    best_score = float("-inf")
    best_metrics = summarize(
        run_backtest(symbols, [(strategy, best_params)], settings, fold.train_start, fold.train_end)
    )
    for params in strategy.param_grid():
        result = run_backtest(symbols, [(strategy, params)], settings, fold.train_start, fold.train_end)
        metrics = summarize(result)
        score = selection_score(metrics, settings.min_trades_for_selection)
        if score > best_score:
            best_score = score
            best_params = params
            best_metrics = metrics
    return best_params, best_score, best_metrics


def build_walk_forward_schedules(
    symbols: list[SymbolSeries],
    strategies: list[Strategy],
    settings: Settings,
    start: date,
    end: date,
    calendar: list[date],
) -> tuple[list[Fold], dict[str, list[tuple[date, date, dict]]], dict[str, list[FoldOutcome]]]:
    """Pick each strategy's params on the training block only.

    The combined book reuses these frozen params. It does not search the
    cartesian product of every strategy's grid.
    """
    window = [day for day in calendar if start <= day <= end]
    folds = iter_folds(window, settings.train_days, settings.test_days)
    schedules: dict[str, list[tuple[date, date, dict]]] = {item.id: [] for item in strategies}
    outcomes: dict[str, list[FoldOutcome]] = {item.id: [] for item in strategies}
    for fold in folds:
        for strategy in strategies:
            logger.info(
                "走步选参 %s 训练 %s ~ %s",
                strategy.name,
                fold.train_start.isoformat(),
                fold.train_end.isoformat(),
            )
            params, score, metrics = select_params(symbols, strategy, settings, fold)
            # The signal that creates a test-window trade is generated on the session
            # before the fill. Include train_end so the first test open can be used.
            schedules[strategy.id].append((fold.train_end, fold.test_end, params))
            outcomes[strategy.id].append(
                FoldOutcome(
                    fold=fold,
                    strategy_id=strategy.id,
                    strategy_name=strategy.name,
                    chosen_params=params,
                    train_score=score,
                    train_metrics=metrics,
                )
            )
    return folds, schedules, outcomes


def run_oos_backtest(
    symbols: list[SymbolSeries],
    strategies: list[Strategy],
    settings: Settings,
    folds: list[Fold],
    schedules: dict[str, list[tuple[date, date, dict]]],
    start: date,
    end: date,
    label: str | None = None,
) -> BacktestResult:
    """Trade the test blocks with params already chosen on earlier data."""
    bundled = [(item, item.default_params()) for item in strategies]
    if not folds:
        return run_backtest(
            symbols,
            bundled,
            settings,
            start,
            end,
            label=label,
            param_schedule={item.id: [] for item in strategies},
        )
    subset = {item.id: schedules.get(item.id, []) for item in strategies}
    result = run_backtest(
        symbols,
        bundled,
        settings,
        folds[0].train_end,
        folds[-1].test_end,
        label=label,
        param_schedule=subset,
    )
    _trim_to_test(result, folds[0].test_start)
    return result


def run_walk_forward(
    symbols: list[SymbolSeries],
    strategy: Strategy,
    settings: Settings,
    start: date,
    end: date,
    calendar: list[date],
) -> tuple[BacktestResult, list[FoldOutcome]]:
    """Select params per training block, then trade only the following test block."""
    folds, schedules, outcomes = build_walk_forward_schedules(
        symbols, [strategy], settings, start, end, calendar
    )
    result = run_oos_backtest(symbols, [strategy], settings, folds, schedules, start, end)
    return result, outcomes[strategy.id]


def _trim_to_test(result: BacktestResult, test_start: date) -> None:
    kept = [
        (day, value)
        for day, value in zip(result.equity_dates, result.equity, strict=True)
        if day >= test_start
    ]
    result.equity_dates = [day for day, _value in kept]
    result.equity = [value for _day, value in kept]
    result.trades = [trade for trade in result.trades if trade.buy_date >= test_start]
    result.start = test_start
    if result.equity:
        result.final_equity = result.equity[-1]
