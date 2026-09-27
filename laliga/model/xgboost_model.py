"""Modest multiclass XGBoost for full-time and half-time 1X2.

Trees are shallow, the seed is fixed, and early stopping uses only the last
slice of the training window passed in. That window must already exclude the
matches being scored. There is no search over hyperparameters.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from laliga.model.dixon_coles import FitError

XGB_SEED = 7


@dataclass(frozen=True)
class XGBoostConfig:
    max_depth: int = 3
    learning_rate: float = 0.05
    n_estimators: int = 200
    early_stopping_rounds: int = 20
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_weight: float = 5.0
    reg_lambda: float = 1.0
    seed: int = XGB_SEED
    val_fraction: float = 0.15
    min_val_rows: int = 24
    min_train_rows: int = 40
    fallback_estimators: int = 40


@dataclass
class BoostedOutcomeModel:
    booster: object
    feature_names: list[str]
    best_iteration: int
    n_rounds: int
    early_stopping: bool
    n_train: int
    n_val: int
    val_fixture_ids: list[int]

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return predict_proba(self, frame)


def validation_size(n_rows: int, config: XGBoostConfig) -> int:
    """Size of the time-ordered tail used for early stopping, or 0 when the window is short."""

    if n_rows < config.min_train_rows + config.min_val_rows:
        return 0
    val = max(config.min_val_rows, int(round(n_rows * config.val_fraction)))
    if n_rows - val < config.min_train_rows:
        return 0
    if val >= n_rows:
        return 0
    return val


def fit_outcome_model(
    frame: pd.DataFrame,
    feature_names: list[str],
    target: str,
    config: XGBoostConfig,
) -> BoostedOutcomeModel:
    """Fit ``multi:softprob`` on rows sorted by kickoff. The tail is validation only."""

    xgb = _xgboost()
    ordered = frame.sort_values(["kickoff", "fixture_id"]).reset_index(drop=True)
    usable = ordered[ordered[target].notna()].reset_index(drop=True)
    if len(usable) < 8:
        raise FitError(f"XGBoost 训练样本不足（{target} 只有 {len(usable)} 场）。")
    overlap = set(feature_names).intersection({"home_goals_ft", "away_goals_ft", "home_goals_ht", "away_goals_ht", "home_xg", "away_xg", "y_ft", "y_ht"})
    if overlap:
        raise FitError(f"XGBoost 特征包含赛后字段：{sorted(overlap)}")
    matrix = _matrix(usable, feature_names)
    labels = usable[target].to_numpy(dtype=int)
    if not set(np.unique(labels)).issubset({0, 1, 2}):
        raise FitError("XGBoost 标签必须是主胜 0、平 1、客胜 2。")
    val = validation_size(len(usable), config)
    params = _params(config)
    if val == 0:
        rounds = min(config.fallback_estimators, config.n_estimators)
        booster = xgb.train(params, xgb.DMatrix(matrix, label=labels), num_boost_round=rounds, verbose_eval=False)
        return BoostedOutcomeModel(
            booster=booster,
            feature_names=list(feature_names),
            best_iteration=rounds - 1,
            n_rounds=rounds,
            early_stopping=False,
            n_train=int(len(usable)),
            n_val=0,
            val_fixture_ids=[],
        )
    train_end = len(usable) - val
    train = usable.iloc[:train_end]
    held = usable.iloc[train_end:]
    if train["kickoff"].max() > held["kickoff"].min():
        raise FitError("XGBoost 验证集里有比训练行更早的比赛。")
    booster = xgb.train(
        params,
        xgb.DMatrix(matrix[:train_end], label=labels[:train_end]),
        num_boost_round=config.n_estimators,
        evals=[(xgb.DMatrix(matrix[train_end:], label=labels[train_end:]), "val")],
        early_stopping_rounds=config.early_stopping_rounds,
        verbose_eval=False,
    )
    best = int(booster.best_iteration)
    return BoostedOutcomeModel(
        booster=booster,
        feature_names=list(feature_names),
        best_iteration=best,
        n_rounds=best + 1,
        early_stopping=True,
        n_train=int(train_end),
        n_val=int(val),
        val_fixture_ids=[int(value) for value in held["fixture_id"].tolist()],
    )


def predict_proba(model: BoostedOutcomeModel, frame: pd.DataFrame) -> np.ndarray:
    xgb = _xgboost()
    matrix = _matrix(frame, model.feature_names)
    probabilities = model.booster.predict(xgb.DMatrix(matrix), iteration_range=(0, model.n_rounds))
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim != 2 or probabilities.shape[1] != 3:
        raise FitError("XGBoost 没有返回三项概率。")
    probabilities = np.clip(probabilities, 1e-15, None)
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def _matrix(frame: pd.DataFrame, names: list[str]) -> np.ndarray:
    values = frame[names].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    values[~np.isfinite(values)] = np.nan
    return values


def _params(config: XGBoostConfig) -> dict:
    return {
        "objective": "multi:softprob",
        "num_class": 3,
        "max_depth": config.max_depth,
        "eta": config.learning_rate,
        "subsample": config.subsample,
        "colsample_bytree": config.colsample_bytree,
        "min_child_weight": config.min_child_weight,
        "lambda": config.reg_lambda,
        "seed": config.seed,
        "nthread": 1,
        "tree_method": "hist",
        "eval_metric": "mlogloss",
        "verbosity": 0,
        "device": "cpu",
    }


def _xgboost():
    try:
        import xgboost as xgb
    except ImportError as exc:
        raise FitError("需要安装 xgboost。请运行 pip install -r requirements.txt。") from exc
    return xgb
