"""Performance statistics from a finished simulation."""

from __future__ import annotations

from dataclasses import dataclass

from ashare.backtest.engine import BacktestResult


@dataclass(frozen=True)
class Metrics:
    total_return: float
    annualized_return: float
    win_rate: float
    avg_win: float
    avg_loss: float
    max_drawdown: float
    trade_count: int
    final_equity: float


def max_drawdown(equity: list[float]) -> float:
    """Peak-to-trough decline as a negative fraction. Zero when equity never falls."""
    if not equity:
        return 0.0
    peak = equity[0]
    worst = 0.0
    for value in equity:
        if value > peak:
            peak = value
        if peak > 0:
            drawdown = value / peak - 1.0
            if drawdown < worst:
                worst = drawdown
    return worst


def summarize(result: BacktestResult) -> Metrics:
    equity = result.equity
    start_value = result.initial_capital
    end_value = equity[-1] if equity else start_value
    total_return = end_value / start_value - 1.0 if start_value else 0.0
    trading_days = max(len(equity) - 1, 0)
    years = trading_days / 252.0
    if years > 0 and end_value > 0 and start_value > 0:
        annualized = (end_value / start_value) ** (1.0 / years) - 1.0
    elif end_value <= 0:
        annualized = -1.0
    else:
        annualized = 0.0
    wins = [trade.pnl for trade in result.trades if trade.pnl > 0]
    losses = [trade.pnl for trade in result.trades if trade.pnl < 0]
    trade_count = len(result.trades)
    win_rate = (len(wins) / trade_count) if trade_count else 0.0
    avg_win = sum(wins) / len(wins) if wins else 0.0
    avg_loss = sum(losses) / len(losses) if losses else 0.0
    return Metrics(
        total_return=total_return,
        annualized_return=annualized,
        win_rate=win_rate,
        avg_win=avg_win,
        avg_loss=avg_loss,
        max_drawdown=max_drawdown(equity),
        trade_count=trade_count,
        final_equity=end_value,
    )


def selection_score(metrics: Metrics, min_trades: int) -> float:
    """Walk-forward objective. Sparse samples lose to any adequately traded curve."""
    if metrics.trade_count < min_trades:
        return -1_000_000_000 + metrics.trade_count
    drawdown = abs(metrics.max_drawdown)
    floor = 0.01
    return metrics.annualized_return / max(drawdown, floor)
