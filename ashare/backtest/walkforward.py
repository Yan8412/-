"""Rolling walk-forward. Parameter choice on a fold never sees that fold's test dates."""

from __future__ import annotations

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


def run_walk_forward(
    symbols: list[SymbolSeries],
    strategy: Strategy,
    settings: Settings,
    start: date,
    end: date,
    calendar: list[date],
) -> tuple[BacktestResult, list[FoldOutcome]]:
    """Select params per training block, then trade only the following test block."""
    window = [day for day in calendar if start <= day <= end]
    folds = iter_folds(window, settings.train_days, settings.test_days)
    outcomes: list[FoldOutcome] = []
    schedule: dict[str, list[tuple[date, date, dict]]] = {strategy.id: []}
    for fold in folds:
        params, score, metrics = select_params(symbols, strategy, settings, fold)
        # The signal that creates a test-window trade is generated on the session
        # before the fill. Include train_end so the first test open can be used,
        # without letting the strategy read prices inside the test block: the
        # engine generates that signal from train_end's close only.
        schedule[strategy.id].append((fold.train_end, fold.test_end, params))
        outcomes.append(
            FoldOutcome(
                fold=fold,
                strategy_id=strategy.id,
                chosen_params=params,
                train_score=score,
                train_metrics=metrics,
            )
        )
    if not folds:
        empty = run_backtest(
            symbols,
            [(strategy, strategy.default_params())],
            settings,
            start,
            end,
            param_schedule={strategy.id: []},
        )
        return empty, []
    # Start one session before the first test day so the signal generated on
    # that close can fill at the first test open. The extra session is flat
    # and is removed from the reported equity curve.
    oos = run_backtest(
        symbols,
        [(strategy, strategy.default_params())],
        settings,
        folds[0].train_end,
        folds[-1].test_end,
        param_schedule=schedule,
    )
    _trim_to_test(oos, folds[0].test_start)
    return oos, outcomes


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
