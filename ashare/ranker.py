"""Gradient-boosting ranker for the two rule strategies.

Hyperparameters are chosen only on trades whose signal and exit are both
strictly before 2025-04-25. The holdout window is not an input to that choice.
A model that scores a session was fit only on trades that had already exited
before the refit date covering that session.
"""

from __future__ import annotations

import logging
import math
import pickle
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

from ashare.backtest.engine import PendingBuy
from ashare.config import Settings
from ashare.market import SymbolSeries
from ashare.paths import model_path
from ashare.rules.limits import classify_board
from ashare.sentiment import MarketPanel
from ashare.trade_outcome import trade_net_return

logger = logging.getLogger(__name__)

# The holdout is touched once, at the end, by the backtest. Nothing below
# may be edited in response to that result.
HOLDOUT_START = date(2025, 4, 25)
HOLDOUT_END = date(2026, 6, 23)
FULL_START = date(2024, 9, 24)
FULL_END = date(2026, 9, 24)

MIN_TRAIN_CALENDAR = 120
VALID_DAYS = 60
REFIT_EVERY = 60
MIN_TRAIN_ROWS = 200
MIN_VALID_DAYS = 8
MIN_VALID_TRADES = 20
TOP_K = 4
MIN_OOS_TRADES = 30
TSTAT_BAR = 2.0

FEATURE_NAMES: tuple[str, ...] = (
    "rule_score",
    "day_return",
    "swing20",
    "log_amount",
    "vol_ratio",
    "dist_ma20",
    "close_pos",
    "stock_height",
    "strategy_first_board",
    "strategy_pullback",
    "breadth",
    "log_limit_up_count",
    "broken_rate",
    "market_max_height",
    "prev_limit_return",
    "prev_limit_missing",
    "index_vs_ma",
    "index_ma_missing",
    "promo_1_2",
    "promo_1_2_missing",
    "promo_2_3",
    "promo_2_3_missing",
    "promo_3p",
    "promo_3p_missing",
    "board_premium",
    "board_premium_missing",
    "ladder_gap",
    "board_main",
    "board_chinext",
    "board_star",
    "board_bj",
)

# Frozen grid. The middle point is the fallback when no validation fold
# has enough completed trades. Ties keep the earlier point.
GRID: tuple[dict, ...] = (
    {"max_depth": 2, "min_samples_leaf": 40, "learning_rate": 0.05, "n_estimators": 80},
    {"max_depth": 2, "min_samples_leaf": 80, "learning_rate": 0.05, "n_estimators": 100},
    {"max_depth": 3, "min_samples_leaf": 40, "learning_rate": 0.08, "n_estimators": 80},
    {"max_depth": 3, "min_samples_leaf": 80, "learning_rate": 0.05, "n_estimators": 80},
)
FALLBACK_PARAMS: dict = dict(GRID[len(GRID) // 2])


def feature_names(extra: tuple[str, ...] = ()) -> tuple[str, ...]:
    """Base columns plus any confirmed stock factors and their missing flags."""
    names = list(FEATURE_NAMES)
    for item in extra:
        names.append(item)
        names.append(f"{item}_missing")
    return tuple(names)


@dataclass
class LabeledRow:
    signal_date: date
    exit_date: date | None
    code: str
    strategy_id: str
    features: np.ndarray
    label: float | None


@dataclass
class RankSchedule:
    """Models keyed by the first signal date they are allowed to score."""

    segments: list[tuple[date, date, GradientBoostingRegressor]]

    def model_for(self, day: date) -> GradientBoostingRegressor | None:
        for start, end, model in self.segments:
            if start <= day < end:
                return model
        return None


def feature_vector(
    series: SymbolSeries,
    index: int,
    strategy_id: str,
    rule_score: float,
    panel: MarketPanel,
    extra_factors: tuple[str, ...] = (),
    extra_values: dict[str, float] | None = None,
) -> np.ndarray:
    """Features known at the signal close. Later bars are not read."""
    close = float(series.close[index])
    preclose = float(series.preclose[index])
    day_return = close / preclose - 1.0 if preclose > 0 else 0.0
    high = float(series.high[index])
    low = float(series.low[index])
    close_pos = (close - low) / (high - low) if high > low else 0.5
    amount = float(series.amount[index])
    log_amount = math.log10(amount) if amount > 0 else 0.0
    vol_ma = float(series.vol_ma5[index]) if len(series.vol_ma5) == len(series.close) else float("nan")
    vol_ratio = float(series.volume[index]) / vol_ma if math.isfinite(vol_ma) and vol_ma > 0 else 0.0
    ma20 = float(series.ma20[index]) if len(series.ma20) == len(series.close) else float("nan")
    qfq = float(series.qfq_close[index])
    dist_ma20 = qfq / ma20 - 1.0 if math.isfinite(ma20) and ma20 > 0 else 0.0
    swing = float(series.swing20[index]) if len(series.swing20) == len(series.close) else float("nan")
    if not math.isfinite(swing):
        swing = 0.0
    height_arr = panel.height.get(series.code)
    stock_height = float(height_arr[index]) if height_arr is not None and index < len(height_arr) else 0.0
    info = panel.days.get(series.dates[index])
    if info is None:
        breadth = 0.0
        log_up = 0.0
        broken = 0.0
        max_height = 0.0
        prev = 0.0
        prev_missing = 1.0
        versus = 0.0
        versus_missing = 1.0
        promo_1_2, promo_1_2_missing = 0.0, 1.0
        promo_2_3, promo_2_3_missing = 0.0, 1.0
        promo_3p, promo_3p_missing = 0.0, 1.0
        premium, premium_missing = 0.0, 1.0
        ladder_gap = 0.0
    else:
        breadth = info.breadth if math.isfinite(info.breadth) else 0.0
        log_up = math.log1p(info.limit_up_count)
        broken = info.broken_rate if math.isfinite(info.broken_rate) else 0.0
        max_height = float(info.max_height)
        prev, prev_missing = _filled(info.prev_limit_return)
        versus, versus_missing = _filled(info.index_vs_ma)
        promo_1_2, promo_1_2_missing = _filled(info.promo_1_2)
        promo_2_3, promo_2_3_missing = _filled(info.promo_2_3)
        promo_3p, promo_3p_missing = _filled(info.promo_3p)
        premium, premium_missing = _filled(info.board_premium)
        ladder_gap = float(info.ladder_gap)
    board = classify_board(series.code)
    values = [
        float(rule_score),
        day_return,
        swing,
        log_amount,
        vol_ratio,
        dist_ma20,
        close_pos,
        stock_height,
        1.0 if strategy_id == "first_board_follow" else 0.0,
        1.0 if strategy_id == "ma_pullback" else 0.0,
        breadth,
        log_up,
        broken,
        max_height,
        prev,
        prev_missing,
        versus,
        versus_missing,
        promo_1_2,
        promo_1_2_missing,
        promo_2_3,
        promo_2_3_missing,
        promo_3p,
        promo_3p_missing,
        premium,
        premium_missing,
        ladder_gap,
        1.0 if board == "main" else 0.0,
        1.0 if board == "chinext" else 0.0,
        1.0 if board == "star" else 0.0,
        1.0 if board == "bj" else 0.0,
    ]
    looked_up = extra_values or {}
    for name in extra_factors:
        raw = looked_up.get(name, float("nan"))
        filled, missing = _filled(raw)
        values.append(filled)
        values.append(missing)
    return np.asarray(values, dtype=float)


def _filled(value: float) -> tuple[float, float]:
    if isinstance(value, (int, float)) and math.isfinite(value):
        return float(value), 0.0
    return 0.0, 1.0


def build_labeled_rows(
    symbols: list[SymbolSeries],
    book: dict[date, list[PendingBuy]],
    panel: MarketPanel,
    settings: Settings,
    extra_factors: tuple[str, ...] = (),
) -> list[LabeledRow]:
    """One row per rule candidate. Unfilled and unfinished trades have no label."""
    by_code = {item.code: item for item in symbols}
    calendar = panel.calendar
    calendar_index = {day: index for index, day in enumerate(calendar)}
    factor_cache: dict[str, dict[str, np.ndarray]] = {}
    if extra_factors:
        from ashare.factors import compute_factors

        for series in symbols:
            factor_cache[series.code] = compute_factors(series, extra_factors)
    rows: list[LabeledRow] = []
    days = sorted(book)
    for nth, day in enumerate(days):
        if nth % 100 == 0:
            logger.info("标注候选 %s / %s 日", nth, len(days))
        for order in book[day]:
            series = by_code.get(order.code)
            if series is None or day not in series.date_index:
                continue
            index = series.date_index[day]
            extra_values = None
            if extra_factors:
                arrays = factor_cache.get(series.code, {})
                extra_values = {
                    name: float(arrays[name][index]) if name in arrays and index < len(arrays[name]) else float("nan")
                    for name in extra_factors
                }
            features = feature_vector(
                series,
                index,
                order.strategy_id,
                order.score,
                panel,
                extra_factors=extra_factors,
                extra_values=extra_values,
            )
            label, exit_date = trade_net_return(series, order, settings, calendar, calendar_index)
            rows.append(
                LabeledRow(
                    signal_date=day,
                    exit_date=exit_date,
                    code=order.code,
                    strategy_id=order.strategy_id,
                    features=features,
                    label=label,
                )
            )
    logger.info("候选 %s 行，其中有标签 %s 行", len(rows), sum(row.label is not None for row in rows))
    return rows


def rows_for_training(rows: list[LabeledRow], cutoff: date) -> list[LabeledRow]:
    """Labels whose signal and exit are both strictly before ``cutoff``."""
    kept: list[LabeledRow] = []
    for row in rows:
        if row.label is None or row.exit_date is None:
            continue
        if row.signal_date < cutoff and row.exit_date < cutoff:
            kept.append(row)
    return kept


def fit_model(rows: list[LabeledRow], params: dict) -> GradientBoostingRegressor:
    model = GradientBoostingRegressor(random_state=0, **params)
    features = np.vstack([row.features for row in rows])
    labels = np.asarray([row.label for row in rows], dtype=float)
    model.fit(features, labels)
    return model


def _validation_folds(calendar: list[date]) -> list[tuple[date, date]]:
    pre = [day for day in calendar if day < HOLDOUT_START]
    folds: list[tuple[date, date]] = []
    index = MIN_TRAIN_CALENDAR
    while index + VALID_DAYS <= len(pre):
        train_cut = pre[index]
        if index + VALID_DAYS < len(pre):
            valid_cut = pre[index + VALID_DAYS]
        else:
            valid_cut = HOLDOUT_START
        folds.append((train_cut, valid_cut))
        index += REFIT_EVERY
    return folds


def _fold_score(model: GradientBoostingRegressor, valid_rows: list[LabeledRow]) -> tuple[float, int, int] | None:
    if not valid_rows:
        return None
    features = np.vstack([row.features for row in valid_rows])
    predicted = model.predict(features)
    by_day: dict[date, list[tuple[float, float]]] = defaultdict(list)
    for row, value in zip(valid_rows, predicted, strict=True):
        by_day[row.signal_date].append((float(value), float(row.label)))
    daily: list[float] = []
    trades = 0
    for items in by_day.values():
        items.sort(key=lambda pair: pair[0], reverse=True)
        chosen = items[:TOP_K]
        daily.append(sum(label for _pred, label in chosen) / len(chosen))
        trades += len(chosen)
    if len(daily) < MIN_VALID_DAYS or trades < MIN_VALID_TRADES:
        return None
    return sum(daily) / len(daily), trades, len(daily)


def select_hyperparams(rows: list[LabeledRow], calendar: list[date]) -> tuple[dict, list[dict], bool]:
    """Pick a grid point using only pre-holdout completed trades.

    Returns ``(params, records, used_fallback)``.
    """
    folds = _validation_folds(calendar)
    if not folds:
        logger.info("验证折不足，使用预先写好的中间参数 %s", FALLBACK_PARAMS)
        return dict(FALLBACK_PARAMS), [], True
    records: list[dict] = []
    best_params: dict | None = None
    best_score = float("-inf")
    any_used = False
    for params in GRID:
        fold_scores: list[float] = []
        fold_info: list[dict] = []
        for train_cut, valid_cut in folds:
            train = rows_for_training(rows, train_cut)
            if len(train) < MIN_TRAIN_ROWS:
                fold_info.append({"train_cut": train_cut.isoformat(), "skipped": "train", "train_rows": len(train)})
                continue
            valid = [
                row
                for row in rows
                if row.label is not None
                and row.exit_date is not None
                and train_cut <= row.signal_date < valid_cut
                and row.exit_date < valid_cut
                and row.signal_date < HOLDOUT_START
                and row.exit_date < HOLDOUT_START
            ]
            logger.info("训练验证折 %s → %s，训练 %s 行", train_cut, valid_cut, len(train))
            model = fit_model(train, params)
            scored = _fold_score(model, valid)
            if scored is None:
                fold_info.append(
                    {
                        "train_cut": train_cut.isoformat(),
                        "valid_cut": valid_cut.isoformat(),
                        "skipped": "valid",
                        "valid_rows": len(valid),
                    }
                )
                continue
            mean, trades, days = scored
            fold_scores.append(mean)
            fold_info.append(
                {
                    "train_cut": train_cut.isoformat(),
                    "valid_cut": valid_cut.isoformat(),
                    "score": mean,
                    "trades": trades,
                    "days": days,
                }
            )
        if not fold_scores:
            records.append({"params": dict(params), "mean_score": None, "folds": fold_info})
            continue
        any_used = True
        mean_score = sum(fold_scores) / len(fold_scores)
        logger.info("参数 %s 验证平均 %s", params, f"{mean_score:.6f}")
        records.append({"params": dict(params), "mean_score": mean_score, "folds": fold_info})
        if mean_score > best_score:
            best_score = mean_score
            best_params = dict(params)
    if not any_used or best_params is None:
        logger.info("没有可用的验证折，使用预先写好的中间参数 %s", FALLBACK_PARAMS)
        return dict(FALLBACK_PARAMS), records, True
    return best_params, records, False


def refit_points(calendar: list[date], rows: list[LabeledRow]) -> list[date]:
    """Expanding refits strictly before the holdout, every 60 sessions."""
    pre = [day for day in calendar if day < HOLDOUT_START]
    points: list[date] = []
    index = MIN_TRAIN_CALENDAR
    while index < len(pre):
        day = pre[index]
        if len(rows_for_training(rows, day)) >= MIN_TRAIN_ROWS:
            points.append(day)
        index += REFIT_EVERY
    return points


def fit_schedule(
    rows: list[LabeledRow],
    calendar: list[date],
    params: dict,
    names: tuple[str, ...] | None = None,
) -> tuple[RankSchedule, dict[str, float], GradientBoostingRegressor | None]:
    """Causal models before the holdout, then one model frozen at the holdout."""
    points = refit_points(calendar, rows)
    fitted: list[tuple[date, GradientBoostingRegressor]] = []
    for day in points:
        trained = rows_for_training(rows, day)
        logger.info("扩窗重训 %s，样本 %s", day.isoformat(), len(trained))
        fitted.append((day, fit_model(trained, params)))
    segments: list[tuple[date, date, GradientBoostingRegressor]] = []
    for index, (day, model) in enumerate(fitted):
        end = fitted[index + 1][0] if index + 1 < len(fitted) else HOLDOUT_START
        segments.append((day, end, model))
    frozen_rows = rows_for_training(rows, HOLDOUT_START)
    importances: dict[str, float] = {}
    frozen: GradientBoostingRegressor | None = None
    active = names if names is not None else FEATURE_NAMES
    if len(frozen_rows) >= MIN_TRAIN_ROWS:
        logger.info("冻结样本外模型，训练样本 %s，截止 %s", len(frozen_rows), HOLDOUT_START.isoformat())
        frozen = fit_model(frozen_rows, params)
        segments.append((HOLDOUT_START, date.max, frozen))
        importances = {
            name: float(value)
            for name, value in zip(active, frozen.feature_importances_, strict=True)
        }
    else:
        logger.info("样本外之前的完整交易只有 %s 笔，不训练模型", len(frozen_rows))
    return RankSchedule(segments), importances, frozen


def precompute_scores(rows: list[LabeledRow], schedule: RankSchedule) -> dict[tuple[date, str, str], float]:
    scores: dict[tuple[date, str, str], float] = {}
    for start, end, model in schedule.segments:
        batch = [row for row in rows if start <= row.signal_date < end]
        if not batch:
            continue
        features = np.vstack([row.features for row in batch])
        predicted = model.predict(features)
        for row, value in zip(batch, predicted, strict=True):
            scores[(row.signal_date, row.code, row.strategy_id)] = float(value)
    return scores


def save_ranker(
    path: Path,
    model: GradientBoostingRegressor,
    params: dict,
    importances: dict[str, float],
    trained_through: date,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model,
        "feature_names": list(FEATURE_NAMES),
        "params": params,
        "trained_through": trained_through.isoformat(),
        "importances": importances,
    }
    with path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_ranker(path: Path | None = None) -> dict | None:
    candidate = path if path is not None else model_path()
    if not candidate.exists():
        return None
    with candidate.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict) or "model" not in payload:
        return None
    names = payload.get("feature_names")
    if list(names or []) != list(FEATURE_NAMES):
        logger.info("模型特征和当前代码不一致，忽略 %s", candidate)
        return None
    return payload


def predict_features(payload: dict, features: np.ndarray) -> float:
    value = payload["model"].predict(np.asarray(features, dtype=float).reshape(1, -1))
    return float(value[0])


def score_candidate(
    payload: dict,
    series: SymbolSeries,
    index: int,
    strategy_id: str,
    rule_score: float,
    panel: MarketPanel,
) -> float:
    return predict_features(payload, feature_vector(series, index, strategy_id, rule_score, panel))


def pnl_tstat(pnls: list[float]) -> float:
    """t statistic of the mean trade PnL. Zero when the sample is too small or flat."""
    count = len(pnls)
    if count < 2:
        return 0.0
    mean = sum(pnls) / count
    variance = sum((item - mean) ** 2 for item in pnls) / (count - 1)
    if variance <= 0:
        return 0.0
    return mean / (math.sqrt(variance) / math.sqrt(count))


def paper_verdict(total_return: float, trade_count: int, tstat: float) -> str:
    """Precommitted reading of one out-of-sample account. Not a tuning rule."""
    if trade_count < MIN_OOS_TRADES:
        return "样本外成交不足 30 笔，不足以判断，不能据此做模拟盘。"
    if total_return <= 0:
        return "样本外总收益不为正，不适合模拟盘。"
    if abs(tstat) < TSTAT_BAR:
        return "样本外收益为正，但单笔盈亏的 t 统计量绝对值小于 2，统计上还不能说有优势，不适合当成交易依据。"
    return (
        "样本外收益为正，且单笔盈亏的 t 统计量绝对值达到 2。"
        "只适合用很小的资金做模拟盘观察；这只是一段历史窗口，不是稳定优势。"
    )
