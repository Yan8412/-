"""Walk-forward terciles of entry-eve sentiment.

Cut points for a test day come from risk-on sessions in that fold's training
window only. The final holdout is not used to choose which tercile, if any,
a tightened rule would drop. The out-of-sample tercile table is descriptive.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

import numpy as np

from ashare.backtest.engine import ClosedTrade
from ashare.backtest.walkforward import Fold, iter_folds
from ashare.ranker import pnl_tstat
from ashare.sentiment import MarketPanel

# Fixed before any out-of-sample tercile is read.
TERCILE_FEATURES: tuple[tuple[str, str], ...] = (
    ("limit_up_count", "涨停家数"),
    ("broken_rate", "炸板率"),
    ("promo_1_2", "1进2晋级率"),
)
MIN_TRAIN_DAYS = 9
MIN_SIDE_TRADES = 15


@dataclass(frozen=True)
class SentimentGate:
    """Drop one extreme tercile of one feature. Chosen without the holdout."""

    feature: str
    drop_tercile: int
    label: str
    train_spread: float
    train_worst_mean: float


@dataclass(frozen=True)
class TaggedTrade:
    signal_date: date
    buy_date: date
    pnl: float
    buy_cost: float


def iter_folds_with_tail(days: list[date], train_days: int, test_days: int) -> list[Fold]:
    """``iter_folds`` plus one last test block when the calendar does not divide evenly."""
    folds = iter_folds(days, train_days, test_days)
    if not days or train_days <= 0:
        return folds
    if not folds:
        if len(days) <= train_days:
            return folds
        train = days[:train_days]
        test = days[train_days:]
        return [Fold(train[0], train[-1], test[0], test[-1])]
    try:
        last_index = days.index(folds[-1].test_end)
    except ValueError:
        return folds
    if last_index >= len(days) - 1:
        return folds
    train_start_index = last_index - train_days + 1
    if train_start_index < 0:
        return folds
    train = days[train_start_index : last_index + 1]
    test = days[last_index + 1 :]
    folds.append(Fold(train[0], train[-1], test[0], test[-1]))
    return folds


def fold_covering(folds: list[Fold], day: date) -> Fold | None:
    for fold in folds:
        if fold.test_start <= day <= fold.test_end:
            return fold
    return None


def assign_tercile(value: float, train: np.ndarray) -> int | None:
    """0, 1, or 2 from the training sample's average-rank percentile.

    ``value`` itself is not added to the sample. A value above every training
    point is the top tercile. Fewer than ``MIN_TRAIN_DAYS`` finite points
    returns None.
    """
    if not math.isfinite(value):
        return None
    sample = np.asarray(train, dtype=float)
    sample = sample[np.isfinite(sample)]
    if sample.size < MIN_TRAIN_DAYS:
        return None
    less = float(np.count_nonzero(sample < value))
    equal = float(np.count_nonzero(sample == value))
    percentile = (less + 0.5 * equal) / float(sample.size)
    if percentile < 1.0 / 3.0:
        return 0
    if percentile < 2.0 / 3.0:
        return 1
    return 2


def train_sample(panel: MarketPanel, calendar: list[date], fold: Fold, feature: str) -> np.ndarray:
    """Risk-on sessions inside the training window, inclusive of both ends."""
    values: list[float] = []
    for day in calendar:
        if day < fold.train_start:
            continue
        if day > fold.train_end:
            break
        info = panel.days.get(day)
        if info is None or not info.risk_on:
            continue
        value = float(getattr(info, feature))
        if math.isfinite(value):
            values.append(value)
    return np.asarray(values, dtype=float)


class WalkForwardTerciles:
    def __init__(
        self,
        panel: MarketPanel,
        calendar: list[date],
        folds: list[Fold],
        features: tuple[str, ...] = tuple(name for name, _label in TERCILE_FEATURES),
    ) -> None:
        self.panel = panel
        self.calendar = calendar
        self.folds = folds
        self.features = features
        self._samples: dict[tuple[date, date, str], np.ndarray] = {}

    def sample(self, fold: Fold, feature: str) -> np.ndarray:
        key = (fold.train_start, fold.train_end, feature)
        cached = self._samples.get(key)
        if cached is None:
            cached = train_sample(self.panel, self.calendar, fold, feature)
            self._samples[key] = cached
        return cached

    def bucket(self, day: date, feature: str) -> int | None:
        fold = fold_covering(self.folds, day)
        if fold is None:
            return None
        info = self.panel.days.get(day)
        if info is None:
            return None
        return assign_tercile(float(getattr(info, feature)), self.sample(fold, feature))


def tag_trades(trades: list[ClosedTrade]) -> list[TaggedTrade]:
    return [
        TaggedTrade(
            signal_date=trade.signal_date,
            buy_date=trade.buy_date,
            pnl=float(trade.pnl),
            buy_cost=float(trade.buy_cost),
        )
        for trade in trades
    ]


def tercile_table(
    trades: list[TaggedTrade],
    walker: WalkForwardTerciles,
    feature: str,
    initial_capital: float,
) -> list[dict]:
    """One row per tercile. Return is the sum of after-cost pnl divided by starting capital.

    That is the contribution of these trades to a 20,000 yuan account, not a
    separate account that only traded the tercile. The t statistic is the same
    mean-pnl t used by the ranker.
    """
    groups: dict[int | None, list[TaggedTrade]] = {0: [], 1: [], 2: [], None: []}
    for trade in trades:
        groups[walker.bucket(trade.signal_date, feature)].append(trade)
    rows = []
    for bucket in (0, 1, 2, None):
        items = groups[bucket]
        pnls = [item.pnl for item in items]
        costs = [item.buy_cost for item in items if item.buy_cost > 0]
        pnl_sum = sum(pnls)
        returns = [item.pnl / item.buy_cost for item in items if item.buy_cost > 0]
        rows.append(
            {
                "tercile": bucket,
                "trade_count": len(items),
                "pnl": pnl_sum,
                "contribution": pnl_sum / initial_capital if initial_capital else 0.0,
                "mean_return": (sum(returns) / len(returns)) if returns else 0.0,
                "tstat": pnl_tstat(pnls),
                "mean_cost": (sum(costs) / len(costs)) if costs else 0.0,
            }
        )
    return rows


def select_gate(
    trades: list[TaggedTrade],
    walker: WalkForwardTerciles,
    min_side: int = MIN_SIDE_TRADES,
) -> tuple[SentimentGate | None, list[dict]]:
    """Pick at most one drop rule from pre-holdout trades.

    The worst extreme tercile is eligible only when both extremes have enough
    trades and the worse one lost money. The largest mean-pnl gap wins. Ties
    keep the earlier feature in ``TERCILE_FEATURES``.
    """
    records: list[dict] = []
    chosen: SentimentGate | None = None
    best_spread = 0.0
    for feature, label in TERCILE_FEATURES:
        groups = {0: [], 1: [], 2: []}
        for trade in trades:
            bucket = walker.bucket(trade.signal_date, feature)
            if bucket is None:
                continue
            groups[bucket].append(trade.pnl)
        means = {bucket: (sum(values) / len(values) if values else float("nan")) for bucket, values in groups.items()}
        counts = {bucket: len(values) for bucket, values in groups.items()}
        records.append(
            {
                "feature": feature,
                "label": label,
                "counts": counts,
                "means": means,
                "tstats": {bucket: pnl_tstat(values) for bucket, values in groups.items()},
            }
        )
        if counts[0] < min_side or counts[2] < min_side:
            continue
        if not math.isfinite(means[0]) or not math.isfinite(means[2]):
            continue
        if means[0] == means[2]:
            continue
        drop = 0 if means[0] < means[2] else 2
        worst = means[drop]
        if worst >= 0:
            continue
        spread = abs(means[2] - means[0])
        if chosen is None or spread > best_spread:
            best_spread = spread
            chosen = SentimentGate(
                feature=feature,
                drop_tercile=drop,
                label=label,
                train_spread=spread,
                train_worst_mean=worst,
            )
    return chosen, records


def gated_entry_mask(
    panel: MarketPanel,
    calendar: list[date],
    walker: WalkForwardTerciles,
    gate: SentimentGate,
    base: dict[date, bool],
) -> dict[date, bool]:
    """Regime mask with one frozen tercile removed. Missing cuts keep the regime bit."""
    allowed: dict[date, bool] = {}
    for day in calendar:
        if not base.get(day, False):
            allowed[day] = False
            continue
        bucket = walker.bucket(day, gate.feature)
        if bucket is None:
            allowed[day] = True
            continue
        allowed[day] = bucket != gate.drop_tercile
    return allowed
