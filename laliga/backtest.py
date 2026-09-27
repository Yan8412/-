"""Walk-forward backtest. Training rows are always strictly earlier than the test day.

``walk_forward`` scores Dixon–Coles against the frequency baseline. That is
the production backtest. ``compare_models`` scores Dixon–Coles, the same
baseline, and an XGBoost challenger on those exact matches and cut-offs.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from laliga.config import ModelConfig
from laliga.model.baseline import fit_baseline
from laliga.model.blend import blend_notes, walk_forward_blends
from laliga.model.devig import DEVIG_METHODS, devig, raw_implied_from_decimal
from laliga.model.dixon_coles import FitError
from laliga.model.features import (
    ODDS_FEATURES,
    XG_FEATURES,
    compute_rolling_features,
    dixon_coles_features,
    model_feature_names,
    observed_feature_names,
    odds_column_status,
    odds_features,
    training_window_has_odds,
    xg_column_status,
)
from laliga.model.metrics import (
    MetricBlock,
    by_season,
    calibration_summary,
    iter_calendar_days,
    iter_evaluation_folds,
    outcome_index,
    paired_logloss_difference,
    summarize_rows,
)
from laliga.model.service import ScorelineModel, fit_models

logger = logging.getLogger(__name__)

COMPARISON_SCHEMA_VERSION = 1


@dataclass
class BacktestReport:
    n_folds: int
    first_test_day: str | None
    last_test_day: str | None
    min_train_matches: int
    model: MetricBlock
    baseline: MetricBlock
    model_by_season: dict[str, MetricBlock]
    baseline_by_season: dict[str, MetricBlock]

    def to_dict(self) -> dict:
        return {
            "n_folds": self.n_folds,
            "first_test_day": self.first_test_day,
            "last_test_day": self.last_test_day,
            "min_train_matches": self.min_train_matches,
            "model": self.model.to_dict(),
            "baseline": self.baseline.to_dict(),
            "model_by_season": {key: value.to_dict() for key, value in self.model_by_season.items()},
            "baseline_by_season": {key: value.to_dict() for key, value in self.baseline_by_season.items()},
        }


def walk_forward(matches: pd.DataFrame, config: ModelConfig) -> BacktestReport:
    finished = matches[matches["status"] == "finished"].dropna(subset=["home_goals_ft", "away_goals_ft"])
    if finished.empty:
        raise FitError("没有完场比赛，无法回测。")
    rows: list[dict] = []
    folds = 0
    first_day: pd.Timestamp | None = None
    last_day: pd.Timestamp | None = None
    previous: ScorelineModel | None = None
    for day, train, test in iter_evaluation_folds(finished, config.min_train_matches):
        as_of = pd.Timestamp(day)
        try:
            model = fit_models(
                train,
                as_of,
                config,
                initial_ft=None if previous is None else previous.ft,
                initial_ht=None if previous is None else previous.ht,
            )
        except FitError:
            continue
        baseline = fit_baseline(train)
        previous = model
        folds += 1
        first_day = day if first_day is None else first_day
        last_day = day
        for _, match in test.iterrows():
            rows.append(_row(match, model, baseline))
    if not rows:
        raise FitError(
            f"回测没有评测任何比赛。完场比赛有 {len(finished)} 场，"
            f"最少训练样本是 {config.min_train_matches}。可以降低 --min-train。"
        )
    return BacktestReport(
        n_folds=folds,
        first_test_day=None if first_day is None else pd.Timestamp(first_day).date().isoformat(),
        last_test_day=None if last_day is None else pd.Timestamp(last_day).date().isoformat(),
        min_train_matches=config.min_train_matches,
        model=summarize_rows(rows, "model"),
        baseline=summarize_rows(rows, "baseline"),
        model_by_season=by_season(rows, "model"),
        baseline_by_season=by_season(rows, "baseline"),
    )


def _row(match: pd.Series, model: ScorelineModel, baseline) -> dict:
    prediction = model.predict_row(match)
    home_ft = int(match["home_goals_ft"])
    away_ft = int(match["away_goals_ft"])
    y_ht = None
    if pd.notna(match["home_goals_ht"]) and pd.notna(match["away_goals_ht"]):
        y_ht = outcome_index(int(match["home_goals_ht"]), int(match["away_goals_ht"]))
    season_id = match["season_id"]
    return {
        "season_id": None if pd.isna(season_id) else int(season_id),
        "season_name": str(match.get("season_name") or ""),
        "y_ft": outcome_index(home_ft, away_ft),
        "y_ht": y_ht,
        "score": (home_ft, away_ft),
        "model_ft_probs": prediction.ft.as_tuple(),
        "model_ht_probs": prediction.ht.as_tuple(),
        "model_top3": [(item.home_goals, item.away_goals) for item in prediction.top_scores],
        "baseline_ft_probs": baseline.ft_probs,
        "baseline_ht_probs": baseline.ht_probs,
        "baseline_top3": [(home, away) for home, away, _ in baseline.top_scores],
    }


MODEL_LABELS = {
    "dixon_coles": "Dixon–Coles",
    "baseline": "历史频率基准",
    "xgboost": "XGBoost（无赔率）",
    "xgboost_with_odds": "XGBoost（含赔率）",
    "market": "赛前赔率（去水位）",
    "dixon_coles_on_blend": "Dixon–Coles（混合同一批）",
    "market_on_blend": "赛前赔率（去水位，混合同一批）",
    "blend_linear": "线性混合",
    "blend_log": "对数混合",
    "blend_stack": "对数线性叠加",
}

_BOOK_PREFIXES = ("market_pinnacle", "market_close", "market_b365", "market")
_METHOD_LABELS = {
    "proportional": "比例去水位",
    "power": "幂去水位",
    "shin": "Shin 去水位",
    "additive": "加法去水位",
    "odds_ratio": "赔率比去水位",
}
_HISTORY_BOOK_LABELS = {
    "market": "赛前盘（Pinnacle PSH，非收盘；缺则 Bet365）",
    "market_pinnacle": "Pinnacle 赛前盘（PSH，非收盘）",
    "market_close": "Pinnacle 收盘（PSCH，预测时不可用）",
    "market_b365": "Bet365 赛前盘（非收盘）",
}
_CORE_BEFORE_MARKET = ("dixon_coles", "baseline")
_CORE_AFTER_MARKET = (
    "xgboost",
    "xgboost_with_odds",
    "dixon_coles_on_blend",
    "market_on_blend",
    "blend_linear",
    "blend_log",
    "blend_stack",
)

COMPARISON_TABLE_KEYS = (
    "dixon_coles",
    "baseline",
    "market",
    "xgboost",
    "xgboost_with_odds",
    "dixon_coles_on_blend",
    "market_on_blend",
    "blend_linear",
    "blend_log",
    "blend_stack",
)

BLEND_PAIRS = (
    ("blend_linear", "market", "线性混合 − 赛前赔率（去水位）"),
    ("blend_linear", "dixon_coles", "线性混合 − Dixon–Coles（混合同一批）"),
    ("blend_log", "market", "对数混合 − 赛前赔率（去水位）"),
    ("blend_log", "dixon_coles", "对数混合 − Dixon–Coles（混合同一批）"),
    ("blend_stack", "market", "对数线性叠加 − 赛前赔率（去水位）"),
    ("blend_stack", "dixon_coles", "对数线性叠加 − Dixon–Coles（混合同一批）"),
)


@dataclass
class ComparisonReport:
    """Dixon–Coles, the frequency baseline, and XGBoost on one walk-forward."""

    n_folds: int
    first_test_day: str | None
    last_test_day: str | None
    min_train_matches: int
    xg_status: str
    odds_status: str
    odds_used: bool
    odds_coverage: float | None
    xg_features_used: bool
    feature_names: list[str]
    odds_feature_names: list[str]
    notes: list[str]
    models: dict[str, MetricBlock]
    calibration: dict[str, dict]
    paired: list[dict]
    xgboost: dict
    blend: dict | None = None
    history_source: str = "sportmonks"
    refit_every_days: int = 1
    n_refits: int = 0
    with_xgboost: bool = True
    by_season: dict | None = None
    price_sources: dict | None = None

    def to_dict(self) -> dict:
        payload = {
            "schema_version": COMPARISON_SCHEMA_VERSION,
            "computed_from_stored_matches": True,
            "production_model": "dixon_coles",
            "n_folds": self.n_folds,
            "first_test_day": self.first_test_day,
            "last_test_day": self.last_test_day,
            "min_train_matches": self.min_train_matches,
            "history_source": self.history_source,
            "refit_every_days": self.refit_every_days,
            "n_refits": self.n_refits,
            "with_xgboost": self.with_xgboost,
            "xg_status": self.xg_status,
            "odds_status": self.odds_status,
            "odds_used": self.odds_used,
            "odds_coverage": self.odds_coverage,
            "xg_features_used": self.xg_features_used,
            "feature_names": self.feature_names,
            "odds_feature_names": self.odds_feature_names,
            "notes": self.notes,
            "models": {key: value.to_dict() for key, value in self.models.items()},
            "calibration": self.calibration,
            "paired": self.paired,
            "xgboost": self.xgboost,
            "blend": self.blend,
            "by_season": _season_payload(self.by_season),
            "price_sources": self.price_sources,
        }
        return _json_ready(payload)


def compare_models(
    matches: pd.DataFrame,
    config: ModelConfig,
    xgb_config=None,
    *,
    refit_every_days: int = 1,
    with_xgboost: bool = True,
    history_source: str = "sportmonks",
) -> ComparisonReport:
    """Score Dixon–Coles, the baseline, and optionally XGBoost on one walk-forward.

    Dixon–Coles uses the same warm-start schedule as ``walk_forward`` when
    ``refit_every_days`` is 1: the first evaluation day starts cold, and later
    days continue from the previous evaluation fit. A larger gap reuses the
    last fit. That fit was trained on matches strictly before its own day, so
    it is also before the later day that reuses it. A separate chain fits
    Dixon–Coles on warmup days only so XGBoost features for those matches are
    still as-of that date. That chain never becomes the model being scored,
    and it is skipped when XGBoost is off.
    """

    from laliga.model.xgboost_model import XGBoostConfig, fit_outcome_model

    if refit_every_days < 1:
        raise FitError("重拟合间隔至少是 1 天。")
    if xgb_config is None:
        xgb_config = XGBoostConfig()
    finished = matches[matches["status"] == "finished"].dropna(subset=["home_goals_ft", "away_goals_ft"])
    if finished.empty:
        raise FitError("没有完场比赛，无法比较模型。")
    finished = finished.copy()
    if finished["fixture_id"].duplicated().any():
        raise FitError("比赛表里有重复的 fixture_id，无法对齐赛前特征。")

    xg_status = xg_column_status(finished)
    odds_status = odds_column_status(finished)
    use_xg = with_xgboost and xg_status == "present"
    first_train = _first_evaluation_train(finished, config.min_train_matches)
    odds_present_in_first_window = first_train is not None and training_window_has_odds(first_train)
    use_odds = with_xgboost and odds_status == "present" and odds_present_in_first_window
    base_names = model_feature_names(use_xg=use_xg, use_odds=False)
    odds_names = model_feature_names(use_xg=use_xg, use_odds=True) if use_odds else []
    rolling = compute_rolling_features(finished, use_xg=use_xg).set_index("fixture_id") if with_xgboost else None

    archived: list[dict] = []
    eval_rows: list[dict] = []
    feature_prev: ScorelineModel | None = None
    score_prev: ScorelineModel | None = None
    last_refit_day: pd.Timestamp | None = None
    cached_ft = None
    cached_ht = None
    cached_odds = None
    cached_odds_ht = None
    folds = 0
    n_refits = 0
    first_day: pd.Timestamp | None = None
    last_day: pd.Timestamp | None = None
    xg_features_used = False
    ft_stopping: list[bool] = []
    ft_rounds: list[int] = []
    ht_stopping: list[bool] = []
    odds_stopping: list[bool] = []

    for day, train, test in iter_calendar_days(finished):
        as_of = pd.Timestamp(day)
        evaluate = len(train) >= config.min_train_matches
        scoring_model: ScorelineModel | None = None
        feature_model = feature_prev
        refit_this_day = False
        if with_xgboost and len(train) >= 4 and not evaluate:
            try:
                feature_prev = _fit_scoreline(train, as_of, config, feature_prev)
                feature_model = feature_prev
            except FitError:
                feature_model = feature_prev
        elif evaluate:
            gap = 10**9 if last_refit_day is None else int((as_of - pd.Timestamp(last_refit_day)).days)
            refit_this_day = score_prev is None or gap >= refit_every_days
            if refit_this_day:
                try:
                    scoring_model = _fit_scoreline(train, as_of, config, score_prev)
                except FitError:
                    scoring_model = None
                if scoring_model is not None:
                    score_prev = scoring_model
                    last_refit_day = as_of
                    n_refits += 1
            else:
                scoring_model = score_prev
            feature_model = scoring_model if scoring_model is not None else feature_prev

        day_rows: list[dict] = []
        if with_xgboost:
            day_rows = [_feature_row(match, rolling, feature_model, use_odds=use_odds) for _, match in test.iterrows()]
        if scoring_model is not None:
            baseline = fit_baseline(train)
            ft_probs = None
            ht_probs = None
            odds_ft_probs = None
            odds_ht_probs = None
            if with_xgboost:
                if len(archived) != len(train):
                    raise FitError("赛前特征行数和训练比赛数不一致。")
                train_frame = pd.DataFrame(archived)
                test_frame = pd.DataFrame(day_rows)
                _assert_time_split(train_frame, test_frame)
                if refit_this_day or cached_ft is None:
                    ft_names = observed_feature_names(train_frame, base_names)
                    if not ft_names:
                        raise FitError("训练窗口里没有可用特征。")
                    if any(name in XG_FEATURES for name in ft_names):
                        xg_features_used = True
                    cached_ft = fit_outcome_model(train_frame, ft_names, "y_ft", xgb_config)
                    cached_ht, _ = _fit_optional_market(
                        train_frame, test_frame, ft_names, "y_ht", xgb_config, fit_outcome_model
                    )
                    ft_stopping.append(cached_ft.early_stopping)
                    ft_rounds.append(cached_ft.n_rounds)
                    if cached_ht is not None:
                        ht_stopping.append(cached_ht.early_stopping)
                    cached_odds = None
                    cached_odds_ht = None
                    if use_odds:
                        kept_odds = observed_feature_names(train_frame, odds_names)
                        if not any(name in ODDS_FEATURES for name in kept_odds):
                            raise FitError("含赔率模型的训练窗口里没有赔率特征。")
                        cached_odds = fit_outcome_model(train_frame, kept_odds, "y_ft", xgb_config)
                        cached_odds_ht, _ = _fit_optional_market(
                            train_frame, test_frame, kept_odds, "y_ht", xgb_config, fit_outcome_model
                        )
                        odds_stopping.append(cached_odds.early_stopping)
                ft_probs = cached_ft.predict(test_frame)
                ht_probs = None if cached_ht is None else cached_ht.predict(test_frame)
                if cached_odds is not None:
                    odds_ft_probs = cached_odds.predict(test_frame)
                    odds_ht_probs = None if cached_odds_ht is None else cached_odds_ht.predict(test_frame)
            folds += 1
            first_day = day if first_day is None else first_day
            last_day = day
            logger.info("对照评测日 %s，训练 %s 场，测试 %s 场。", pd.Timestamp(day).date().isoformat(), len(train), len(test))
            for index, (_, match) in enumerate(test.iterrows()):
                eval_rows.append(
                    _comparison_row(
                        match,
                        scoring_model,
                        baseline,
                        None if ft_probs is None else ft_probs[index],
                        None if ht_probs is None else ht_probs[index],
                        None if odds_ft_probs is None else odds_ft_probs[index],
                        None if odds_ht_probs is None else odds_ht_probs[index],
                    )
                )
        if with_xgboost:
            archived.extend(day_rows)

    if not eval_rows:
        raise FitError(
            f"对照没有评测任何比赛。完场比赛有 {len(finished)} 场，"
            f"最少训练样本是 {config.min_train_matches}。可以降低 --min-train。"
        )

    y_ft = np.array([row["y_ft"] for row in eval_rows], dtype=int)
    model_keys = ["dixon_coles", "baseline"]
    if with_xgboost:
        model_keys.append("xgboost")
    if use_odds:
        model_keys.append("xgboost_with_odds")
    models = {key: summarize_rows(eval_rows, key) for key in model_keys}
    calibration = {
        key: calibration_summary(y_ft, np.array([row[f"{key}_ft_probs"] for row in eval_rows], dtype=float))
        for key in model_keys
    }
    redundant_notes = _drop_redundant_markets(eval_rows)
    _summarize_markets(eval_rows, models, calibration)
    paired = _paired_block(eval_rows, y_ft, xgb_config.seed, with_xgboost=with_xgboost, history_source=history_source)
    _attach_market_pairs(eval_rows, paired, xgb_config.seed, history_source)
    odds_coverage = (
        _odds_coverage(finished, [row["fixture_id"] for row in eval_rows]) if odds_status == "present" else None
    )
    notes = _comparison_notes(
        xg_status=xg_status,
        odds_status=odds_status,
        use_odds=use_odds,
        odds_present_in_first_window=odds_present_in_first_window,
        xg_features_used=xg_features_used,
        with_xgboost=with_xgboost,
        refit_every_days=refit_every_days,
        n_refits=n_refits,
        history_source=history_source,
    )
    notes.extend(redundant_notes)
    market_rows = [row for row in eval_rows if row.get("market_ft_probs") is not None]
    if market_rows and len(market_rows) != len(eval_rows):
        notes.append(
            f"赛前赔率基准只统计有完整赛前 1X2 的 {len(market_rows)} 场评测比赛，"
            f"少于其他模型的 {len(eval_rows)} 场。配对区间也只用有该价格的场次，n 写在每一行上。"
        )
    elif market_rows and history_source == "football-data":
        notes.append(
            "公平赛前盘优先用 Pinnacle 赛前报价（PSH/PSD/PSA）。"
            "这是站点在周末前的周五下午、或中场周的周二下午采集的价格，不是开盘第一口，也不是收盘。"
            "没有 Pinnacle 时用同一时点的 Bet365。收盘（PSCH/PSCD/PSCA）单独成行，预测时不可用，不进入混合。"
        )
    elif market_rows:
        notes.append("赛前赔率基准是各家开赛前 1X2 隐含概率的平均，再去掉水位。它不用模型，也不用本场 xG。")
    price_sources = _price_sources(eval_rows) if history_source == "football-data" else None
    if price_sources is not None:
        notes.append(_format_price_sources(price_sources))
    blend = _attach_blends(eval_rows, models, calibration, paired, notes, seed=xgb_config.seed, history_source=history_source)
    season_blocks = _season_blocks(eval_rows, models, blend)
    return ComparisonReport(
        n_folds=folds,
        first_test_day=None if first_day is None else pd.Timestamp(first_day).date().isoformat(),
        last_test_day=None if last_day is None else pd.Timestamp(last_day).date().isoformat(),
        min_train_matches=config.min_train_matches,
        xg_status=xg_status,
        odds_status=odds_status,
        odds_used=use_odds,
        odds_coverage=odds_coverage,
        xg_features_used=xg_features_used,
        feature_names=base_names,
        odds_feature_names=odds_names,
        notes=notes,
        models=models,
        calibration=calibration,
        paired=paired,
        xgboost={
            "max_depth": xgb_config.max_depth,
            "learning_rate": xgb_config.learning_rate,
            "n_estimators": xgb_config.n_estimators,
            "early_stopping_rounds": xgb_config.early_stopping_rounds,
            "subsample": xgb_config.subsample,
            "colsample_bytree": xgb_config.colsample_bytree,
            "min_child_weight": xgb_config.min_child_weight,
            "reg_lambda": xgb_config.reg_lambda,
            "seed": xgb_config.seed,
            "val_fraction": xgb_config.val_fraction,
            "min_val_rows": xgb_config.min_val_rows,
            "min_train_rows": xgb_config.min_train_rows,
            "fallback_estimators": xgb_config.fallback_estimators,
            "ft_folds": len(ft_stopping),
            "ft_folds_with_early_stopping": int(sum(ft_stopping)),
            "mean_rounds_ft": float(np.mean(ft_rounds)) if ft_rounds else None,
            "ht_folds_with_early_stopping": int(sum(ht_stopping)),
            "odds_folds_with_early_stopping": int(sum(odds_stopping)),
            "enabled": with_xgboost,
        },
        blend=blend,
        history_source=history_source,
        refit_every_days=refit_every_days,
        n_refits=n_refits,
        with_xgboost=with_xgboost,
        by_season=season_blocks,
        price_sources=price_sources,
    )


def _fit_scoreline(train: pd.DataFrame, as_of: pd.Timestamp, config: ModelConfig, previous: ScorelineModel | None) -> ScorelineModel:
    return fit_models(
        train,
        as_of,
        config,
        initial_ft=None if previous is None else previous.ft,
        initial_ht=None if previous is None else previous.ht,
    )


def _first_evaluation_train(finished: pd.DataFrame, min_train_matches: int) -> pd.DataFrame | None:
    for _, train, _ in iter_calendar_days(finished):
        if len(train) >= min_train_matches:
            return train
    return None


def _feature_row(match: pd.Series, rolling: pd.DataFrame, model: ScorelineModel | None, *, use_odds: bool) -> dict:
    fixture_id = int(match["fixture_id"])
    history = rolling.loc[fixture_id]
    if isinstance(history, pd.DataFrame):
        raise FitError(f"fixture_id {fixture_id} 在特征表里出现了多次。")
    features = {name: _finite(history[name]) for name in history.index if name != "fixture_id"}
    features.update(dixon_coles_features(model, int(match["home_team_id"]), int(match["away_team_id"])))
    if use_odds:
        features.update(odds_features(match))
    leaked = LEAKAGE_NAMES.intersection(features)
    if leaked:
        raise FitError(f"赛前特征包含赛后字段：{sorted(leaked)}")
    y_ht = None
    if pd.notna(match["home_goals_ht"]) and pd.notna(match["away_goals_ht"]):
        y_ht = outcome_index(int(match["home_goals_ht"]), int(match["away_goals_ht"]))
    season_id = match["season_id"]
    row = dict(features)
    row.update(
        {
            "fixture_id": fixture_id,
            "kickoff": pd.Timestamp(match["starting_at"]),
            "season_id": None if pd.isna(season_id) else int(season_id),
            "season_name": str(match.get("season_name") or ""),
            "y_ft": outcome_index(int(match["home_goals_ft"]), int(match["away_goals_ft"])),
            "y_ht": y_ht,
            "score_home": int(match["home_goals_ft"]),
            "score_away": int(match["away_goals_ft"]),
        }
    )
    return row


LEAKAGE_NAMES = frozenset(
    {
        "home_goals_ft",
        "away_goals_ft",
        "home_goals_ht",
        "away_goals_ht",
        "home_xg",
        "away_xg",
        "home_xga",
        "away_xga",
        "y_ft",
        "y_ht",
    }
)


def _comparison_row(match, model: ScorelineModel, baseline, ft_probs, ht_probs, odds_ft, odds_ht) -> dict:
    prediction = model.predict_row(match)
    home_ft = int(match["home_goals_ft"])
    away_ft = int(match["away_goals_ft"])
    y_ht = None
    if pd.notna(match["home_goals_ht"]) and pd.notna(match["away_goals_ht"]):
        y_ht = outcome_index(int(match["home_goals_ht"]), int(match["away_goals_ht"]))
    season_id = match["season_id"]
    kickoff = pd.Timestamp(match["starting_at"])
    if kickoff.tzinfo is None:
        kickoff = kickoff.tz_localize("UTC")
    else:
        kickoff = kickoff.tz_convert("UTC")
    row = {
        "fixture_id": int(match["fixture_id"]),
        "season_id": None if pd.isna(season_id) else int(season_id),
        "season_name": str(match.get("season_name") or ""),
        "y_ft": outcome_index(home_ft, away_ft),
        "y_ht": y_ht,
        "score": (home_ft, away_ft),
        "dixon_coles_ft_probs": prediction.ft.as_tuple(),
        "dixon_coles_ht_probs": prediction.ht.as_tuple(),
        "dixon_coles_top3": [(item.home_goals, item.away_goals) for item in prediction.top_scores],
        "baseline_ft_probs": baseline.ft_probs,
        "baseline_ht_probs": baseline.ht_probs,
        "baseline_top3": [(home, away) for home, away, _ in baseline.top_scores],
        "kickoff": kickoff.isoformat(),
        "match_day": kickoff.floor("D").date().isoformat(),
    }
    if "fair_book" in match.index:
        book = match["fair_book"]
        if pd.notna(book) and str(book).strip():
            row["fair_book"] = str(book).strip()
    if ft_probs is not None:
        row["xgboost_ft_probs"] = _prob_tuple(ft_probs)
        row["xgboost_ht_probs"] = None if ht_probs is None else _prob_tuple(ht_probs)
    if odds_ft is not None:
        row["xgboost_with_odds_ft_probs"] = _prob_tuple(odds_ft)
        row["xgboost_with_odds_ht_probs"] = None if odds_ht is None else _prob_tuple(odds_ht)
    _attach_market_prices(row, match)
    return row


def _attach_blends(
    rows: list[dict],
    models: dict,
    calibration: dict,
    paired: list[dict],
    notes: list[str],
    *,
    seed: int,
    history_source: str = "sportmonks",
) -> dict:
    """Score walk-forward blends on the rows that have both Dixon–Coles and a price.

    Metric blocks for Dixon–Coles and the market are repeated on that same
    scored subset so their n matches the blend. The full-sample rows are left
    as they are.
    """

    description = walk_forward_blends(rows)
    notes.extend(blend_notes(description))
    if not description["available"]:
        return description
    scored = [row for row in rows if row.get("blend_linear_ft_probs") is not None]
    if len(scored) != description["scored_n"]:
        raise FitError("混合计分场次和写入的概率行数不一致。")
    y_ft = np.array([row["y_ft"] for row in scored], dtype=int)
    market_prefix = "market" if _blend_uses_only_proportional(scored) else "blend_market"
    for key, prefix in (
        ("dixon_coles_on_blend", "dixon_coles"),
        ("market_on_blend", market_prefix),
        ("blend_linear", "blend_linear"),
        ("blend_log", "blend_log"),
        ("blend_stack", "blend_stack"),
    ):
        models[key] = summarize_rows(scored, prefix)
        calibration[key] = calibration_summary(y_ft, np.array([row[f"{prefix}_ft_probs"] for row in scored], dtype=float))
    description["market_prefix"] = market_prefix
    if market_prefix == "blend_market":
        seen_challengers: list[str] = []
        for challenger, _reference, _label in BLEND_PAIRS:
            if not challenger.startswith("blend_") or challenger in seen_challengers:
                continue
            seen_challengers.append(challenger)
            paired.append(
                _pair_stats(
                    scored,
                    y_ft,
                    challenger,
                    "blend_market",
                    f"{MODEL_LABELS[challenger]} − 走步所选去水位",
                    seed,
                )
            )
            paired.append(
                _pair_stats(
                    scored,
                    y_ft,
                    challenger,
                    "market",
                    f"{MODEL_LABELS[challenger]} − {model_label('market', history_source)}",
                    seed,
                )
            )
            paired.append(
                _pair_stats(
                    scored,
                    y_ft,
                    challenger,
                    "dixon_coles",
                    f"{MODEL_LABELS[challenger]} − Dixon–Coles（混合同一批）",
                    seed,
                )
            )
        return description
    for challenger, reference, label in BLEND_PAIRS:
        paired.append(_pair_stats(scored, y_ft, challenger, reference, label, seed))
    return description


def _fit_optional_market(train_frame, test_frame, names, target, xgb_config, fit_outcome_model):
    labeled = train_frame[train_frame[target].notna()]
    if len(labeled) < 8:
        return None, None
    kept = observed_feature_names(labeled, names)
    if not kept:
        return None, None
    model = fit_outcome_model(labeled, kept, target, xgb_config)
    return model, model.predict(test_frame)


def _assert_time_split(train_frame: pd.DataFrame, test_frame: pd.DataFrame) -> None:
    if train_frame.empty or test_frame.empty:
        raise FitError("对照划分是空的。")
    if train_frame["kickoff"].max() >= test_frame["kickoff"].min():
        raise FitError("XGBoost 训练样本包含了测试日或更晚的比赛。")
    if set(train_frame["fixture_id"]).intersection(set(test_frame["fixture_id"])):
        raise FitError("XGBoost 训练样本包含了正在评测的比赛。")


def _paired_block(
    rows: list[dict], y_ft: np.ndarray, seed: int, *, with_xgboost: bool = True, history_source: str = "sportmonks"
) -> list[dict]:
    pairs = [("dixon_coles", "baseline", "Dixon–Coles − 历史频率基准")]
    if with_xgboost and rows and rows[0].get("xgboost_ft_probs") is not None:
        pairs = [
            ("xgboost", "dixon_coles", "XGBoost（无赔率）− Dixon–Coles"),
            ("xgboost", "baseline", "XGBoost（无赔率）− 历史频率基准"),
            *pairs,
        ]
    if rows and rows[0].get("xgboost_with_odds_ft_probs") is not None:
        pairs.extend(
            [
                ("xgboost_with_odds", "dixon_coles", "XGBoost（含赔率）− Dixon–Coles"),
                ("xgboost_with_odds", "xgboost", "XGBoost（含赔率）− XGBoost（无赔率）"),
            ]
        )
    compared = []
    for challenger, reference, label in pairs:
        compared.append(_pair_stats(rows, y_ft, challenger, reference, label, seed))
    market_rows = [row for row in rows if row.get("market_ft_probs") is not None]
    if market_rows:
        y_market = np.array([row["y_ft"] for row in market_rows], dtype=int)
        market_name = model_label("market", history_source)
        market_pairs = [("dixon_coles", "market", f"Dixon–Coles − {market_name}")]
        if with_xgboost and market_rows[0].get("xgboost_ft_probs") is not None:
            market_pairs.insert(0, ("xgboost", "market", f"XGBoost（无赔率）− {market_name}"))
        if market_rows[0].get("xgboost_with_odds_ft_probs") is not None:
            market_pairs.append(("xgboost_with_odds", "market", f"XGBoost（含赔率）− {market_name}"))
        for challenger, reference, label in market_pairs:
            compared.append(_pair_stats(market_rows, y_market, challenger, reference, label, seed))
    return compared


def _pair_stats(rows: list[dict], y_ft: np.ndarray, challenger: str, reference: str, label: str, seed: int) -> dict:
    stats = paired_logloss_difference(
        y_ft,
        np.array([row[f"{challenger}_ft_probs"] for row in rows], dtype=float),
        np.array([row[f"{reference}_ft_probs"] for row in rows], dtype=float),
        seed=seed,
    )
    return {"challenger": challenger, "reference": reference, "label": label, **stats}


def _odds_coverage(finished: pd.DataFrame, fixture_ids: list[int]) -> float:
    chosen = finished[finished["fixture_id"].astype(int).isin(set(fixture_ids))]
    if chosen.empty:
        return 0.0
    hits = 0
    for _, row in chosen.iterrows():
        implied = odds_features(row)
        if all(math.isfinite(implied[name]) for name in ODDS_FEATURES):
            hits += 1
    return hits / len(chosen)


def _comparison_notes(
    *,
    xg_status: str,
    odds_status: str,
    use_odds: bool,
    odds_present_in_first_window: bool,
    xg_features_used: bool,
    with_xgboost: bool = True,
    refit_every_days: int = 1,
    n_refits: int = 0,
    history_source: str = "sportmonks",
) -> list[str]:
    notes = [
        "生产预测、操作台按钮和每日更新仍使用 Dixon–Coles。这个对照只在运行 compare 时计算。",
        "Dixon–Coles、历史频率基准和 XGBoost 使用同一批评测比赛、同一个 UTC 日切。训练行只包含该日 00:00 UTC 之前的完场比赛。",
    ]
    if with_xgboost:
        notes.append("XGBoost 是浅树（深度 3），随机种子固定。早停只用每个训练窗口内部按时间排在最后的一段，不用评测日。")
    else:
        notes.append("这次没有训练 XGBoost。长历史默认关闭。加上 --xgboost 才会训练；没有 xG 的比赛在特征里保持缺失。")
    if refit_every_days > 1:
        notes.append(
            f"Dixon–Coles 每 {refit_every_days} 个 UTC 日重新拟合一次，这次实际拟合 {n_refits} 次。"
            "中间的评测日沿用上一次拟合，那次拟合只用了它自己那天 00:00 UTC 之前的完场比赛。"
            "历史频率基准仍然每个评测日重算。n_folds 是计分的评测日。"
        )
    if xg_features_used:
        notes.append("xG 特征只用更早比赛的赛后 xG，不用本场 xG。")
    elif xg_status == "empty":
        notes.append("比赛表有 home_xg 和 away_xg 列，但没有数值，所以没有使用 xG。")
    elif xg_status == "absent":
        notes.append(
            "比赛表没有 home_xg 和 away_xg。默认 fetch 不请求 xGFixture。"
            "回填命令是 python -m laliga fetch-markets。"
        )
    else:
        notes.append("存储的 xG 没有进入任何训练窗口的可用特征。")
    if with_xgboost and use_odds:
        notes.append("含赔率的模型额外使用本场赛前 1X2 隐含概率。它衡量的是盘口之外还剩多少信息，和不用赔率的结果不是同一个问题。")
    elif with_xgboost and odds_status == "present" and not odds_present_in_first_window:
        notes.append("本地有赛前赔率，但最早一个评测日的训练窗口里没有完整赔率，因此没有做含赔率对照，以免两行模型的比赛场次不同。")
    elif with_xgboost and odds_status == "empty":
        notes.append("赔率列是空的。含赔率的 XGBoost 没有运行。")
    elif with_xgboost:
        notes.append(
            "比赛表没有赛前赔率列（十进制赔率 odds_home、odds_draw、odds_away，"
            "或已经是概率的 implied_home、implied_draw、implied_away）。"
            "含赔率的 XGBoost 没有运行。回填命令是 python -m laliga fetch-markets。"
        )
    if history_source == "football-data":
        notes.append("这次读的是 processed/history.csv，不是每日更新用的 matches.csv。")
    return notes


_PRICE_SOURCE_ORDER = ("pinnacle_pre", "bet365_pre", "sportmonks", "missing")
_PRICE_SOURCE_LABELS = {
    "pinnacle_pre": "Pinnacle 赛前盘（PSH）",
    "bet365_pre": "Bet365 赛前盘",
    "sportmonks": "SportMonks 已存价格",
    "missing": "没有可用赛前盘",
}


def _price_sources(rows: list[dict]) -> dict:
    """Count fair_book on the evaluation rows only, not the whole history file."""

    counts = {key: 0 for key in _PRICE_SOURCE_ORDER}
    by_season: dict[str, dict[str, int]] = {}
    for row in rows:
        book = str(row.get("fair_book") or "").strip()
        if book not in counts:
            book = "missing"
        counts[book] += 1
        season = str(row.get("season_name") or "") or "未知赛季"
        bucket = by_season.setdefault(season, {key: 0 for key in _PRICE_SOURCE_ORDER})
        bucket[book] += 1
    payload = {
        "test_rows": len(rows),
        "counts": counts,
        "labels": dict(_PRICE_SOURCE_LABELS),
    }
    if len(by_season) > 1:
        payload["by_season"] = by_season
    return payload


def _format_price_sources(sources: dict) -> str:
    labels = sources["labels"]
    counts = sources["counts"]
    parts = [f"{labels[key]} {counts[key]} 场" for key in _PRICE_SOURCE_ORDER]
    lines = ["评测期每一场公平赛前盘的来源：" + "，".join(parts) + "。"]
    by_season = sources.get("by_season") or {}
    if len(by_season) > 1:
        for season in sorted(by_season):
            bucket = by_season[season]
            detail = "，".join(f"{labels[key]} {bucket[key]} 场" for key in _PRICE_SOURCE_ORDER if bucket[key])
            lines.append(f"{season}：{detail}。")
    return "\n".join(lines)


def model_label(key: str, history_source: str = "sportmonks", blend: dict | None = None) -> str:
    """Chinese row name. Closing prices say they are not available at prediction time."""

    if key == "market_on_blend" and blend and blend.get("market_prefix") == "blend_market":
        return "走步所选去水位（混合同一批）"
    if key in MODEL_LABELS and not (history_source == "football-data" and key == "market"):
        return MODEL_LABELS[key]
    book, method = _split_market_key(key)
    if book is None:
        return MODEL_LABELS.get(key, key)
    if history_source == "football-data":
        book_label = _HISTORY_BOOK_LABELS.get(book, book)
    elif book == "market" and method == "proportional":
        return "赛前赔率（去水位）"
    else:
        book_label = "赛前赔率"
    method_label = _METHOD_LABELS.get(method, method)
    if history_source != "football-data" and book == "market":
        return f"{book_label}（{method_label}）"
    return f"{book_label}，{method_label}"


def iter_model_keys(models: dict) -> list[str]:
    """Stable row order: core models, then each book's de-vig methods, then blends."""

    keys = [key for key in _CORE_BEFORE_MARKET if key in models]
    market_keys = [key for key in models if _split_market_key(key)[0] is not None and key != "market_on_blend"]
    keys.extend(sorted(market_keys, key=_market_sort_key))
    keys.extend(key for key in _CORE_AFTER_MARKET if key in models)
    return keys


def _market_sort_key(key: str) -> tuple:
    book, method = _split_market_key(key)
    book_order = {"market": 0, "market_pinnacle": 1, "market_b365": 2, "market_close": 3}
    method_order = {name: index for index, name in enumerate(DEVIG_METHODS)}
    return (book_order.get(book or "", 9), method_order.get(method or "", 9), key)


def _split_market_key(key: str) -> tuple[str | None, str | None]:
    if key == "market_on_blend":
        return None, None
    for book in _BOOK_PREFIXES:
        if key == book:
            return book, "proportional"
        prefix = book + "_"
        if key.startswith(prefix):
            method = key[len(prefix):]
            if method in DEVIG_METHODS:
                return book, method
    return None, None


def _prob_stem(book: str, method: str) -> str:
    if book == "market" and method == "proportional":
        return "market"
    if book == "market":
        return f"market_{method}"
    if method == "proportional":
        return book
    return f"{book}_{method}"


def _attach_market_prices(row: dict, match) -> None:
    """Write one probability vector per book and de-vig method.

    ``raw_implied_*`` is preferred. Decimal ``odds_*`` are used only when that
    raw price is absent, which is how a hand-entered book is read. Stored
    ``implied_*`` are already normalised, so they only fill the proportional row.
    """

    raw = _series_triple(match, "raw_implied_")
    if raw is None:
        decimal = _series_triple(match, "odds_")
        if decimal is not None and float(np.min(decimal)) > 1.0:
            raw = raw_implied_from_decimal(decimal)[0]
    if raw is not None:
        _write_book(row, "market", devig(raw.reshape(1, 3)))
    else:
        implied = odds_features(match)
        if all(math.isfinite(implied[name]) for name in ODDS_FEATURES):
            row["market_ft_probs"] = (
                implied["odds_implied_home"],
                implied["odds_implied_draw"],
                implied["odds_implied_away"],
            )
    for book, prefix in (
        ("market_pinnacle", "pin_pre_"),
        ("market_close", "pin_close_"),
        ("market_b365", "b365_pre_"),
    ):
        decimal = _series_triple(match, prefix)
        if decimal is None or float(np.min(decimal)) <= 1.0:
            continue
        _write_book(row, book, devig(raw_implied_from_decimal(decimal)))


def _write_book(row: dict, book: str, priced: dict[str, np.ndarray]) -> None:
    for method, matrix in priced.items():
        vector = matrix[0]
        if not np.isfinite(vector).all():
            continue
        row[f"{_prob_stem(book, method)}_ft_probs"] = (float(vector[0]), float(vector[1]), float(vector[2]))


def _series_triple(match, prefix: str) -> np.ndarray | None:
    values = []
    index = match.index if isinstance(match, pd.Series) else ()
    for side in ("home", "draw", "away"):
        column = f"{prefix}{side}"
        if column not in index:
            return None
        value = match[column]
        if pd.isna(value):
            return None
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(number) or number <= 0.0:
            return None
        values.append(number)
    return np.asarray(values, dtype=float)


def _drop_redundant_markets(rows: list[dict]) -> list[str]:
    """Remove a duplicate of the fair price, then methods that match proportional."""

    notes = []
    if _book_matches(rows, "market_pinnacle", "market"):
        _clear_book(rows, "market_pinnacle")
        notes.append("公平价每一场都等于 Pinnacle 赛前盘，因此不再单列 Pinnacle 赛前盘。")
    if _book_matches(rows, "market_b365", "market"):
        _clear_book(rows, "market_b365")
        notes.append("公平价每一场都等于 Bet365 赛前盘，因此不再单列 Bet365。")
    fair_methods = [method for method in DEVIG_METHODS if method != "proportional"]
    for book in ("market", "market_pinnacle", "market_close", "market_b365"):
        for method in fair_methods:
            stem = _prob_stem(book, method)
            base = f"{_prob_stem(book, 'proportional')}_ft_probs"
            if any(row.get(f"{stem}_ft_probs") is not None for row in rows) and _vectors_match(
                rows, f"{stem}_ft_probs", base
            ):
                for row in rows:
                    row.pop(f"{stem}_ft_probs", None)
    still = [
        method
        for method in fair_methods
        if any(row.get(f"{_prob_stem('market', method)}_ft_probs") is not None for row in rows)
    ]
    had_market = any(row.get("market_ft_probs") is not None for row in rows)
    if had_market and not still:
        notes.append(
            "Shin、幂、加法、赔率比和比例去水位相同，因此不单列。"
            "SportMonks 的比赛表如果只有去水位后的 odds_* / implied_*、没有 raw_implied_*，就会这样；"
            "请重新运行 python -m laliga fetch-markets。"
            "原始隐含概率已经在、但水位大约低于 0.5% 时，这几种方法也会重合。"
        )
    return notes


def _vectors_match(rows: list[dict], left_key: str, right_key: str, *, absent_ok: bool = False) -> bool:
    compared = 0
    for row in rows:
        left = row.get(left_key)
        right = row.get(right_key)
        if left is None and right is None:
            continue
        if left is None or right is None:
            return False
        if float(np.max(np.abs(np.asarray(left, dtype=float) - np.asarray(right, dtype=float)))) > 1e-6:
            return False
        compared += 1
    return compared > 0 or absent_ok


def _book_matches(rows: list[dict], book: str, other: str) -> bool:
    proportional = _prob_stem(book, "proportional")
    other_proportional = _prob_stem(other, "proportional")
    if not _vectors_match(rows, f"{proportional}_ft_probs", f"{other_proportional}_ft_probs"):
        return False
    for method in DEVIG_METHODS:
        if method == "proportional":
            continue
        if not _vectors_match(
            rows,
            f"{_prob_stem(book, method)}_ft_probs",
            f"{_prob_stem(other, method)}_ft_probs",
            absent_ok=True,
        ):
            return False
    return True


def _clear_book(rows: list[dict], book: str) -> None:
    for method in DEVIG_METHODS:
        stem = _prob_stem(book, method)
        for row in rows:
            row.pop(f"{stem}_ft_probs", None)


def _summarize_markets(rows: list[dict], models: dict, calibration: dict) -> None:
    for key in sorted({stem for row in rows for stem in _market_stems(row)}, key=_market_sort_key):
        present = [row for row in rows if row.get(f"{key}_ft_probs") is not None]
        if not present:
            continue
        models[key] = summarize_rows(present, key)
        calibration[key] = calibration_summary(
            np.array([row["y_ft"] for row in present], dtype=int),
            np.array([row[f"{key}_ft_probs"] for row in present], dtype=float),
        )


def _market_stems(row: dict) -> list[str]:
    stems = []
    for key in row:
        if not key.endswith("_ft_probs"):
            continue
        stem = key[: -len("_ft_probs")]
        if _split_market_key(stem)[0] is not None:
            stems.append(stem)
    return stems


def _attach_market_pairs(rows: list[dict], paired: list[dict], seed: int, history_source: str) -> None:
    stems = sorted({stem for row in rows for stem in _market_stems(row)}, key=_market_sort_key)
    for stem in stems:
        book, method = _split_market_key(stem)
        if book is None or (book == "market" and method == "proportional"):
            continue
        present = [row for row in rows if row.get(f"{stem}_ft_probs") is not None]
        if not present:
            continue
        y_present = np.array([row["y_ft"] for row in present], dtype=int)
        proportional = _prob_stem(book, "proportional")
        same_book = [
            row for row in present if row.get(f"{proportional}_ft_probs") is not None
        ]
        if method != "proportional" and same_book:
            paired.append(
                _pair_stats(
                    same_book,
                    np.array([row["y_ft"] for row in same_book], dtype=int),
                    stem,
                    proportional,
                    f"{model_label(stem, history_source)} − {model_label(proportional, history_source)}",
                    seed,
                )
            )
        paired.append(
            _pair_stats(
                present,
                y_present,
                stem,
                "dixon_coles",
                f"{model_label(stem, history_source)} − Dixon–Coles",
                seed,
            )
        )
    close = [row for row in rows if row.get("market_close_ft_probs") is not None and row.get("market_ft_probs") is not None]
    if close:
        paired.append(
            _pair_stats(
                close,
                np.array([row["y_ft"] for row in close], dtype=int),
                "market_close",
                "market",
                f"{model_label('market_close', history_source)} − {model_label('market', history_source)}",
                seed,
            )
        )


def _blend_uses_only_proportional(rows: list[dict]) -> bool:
    methods = {str(row.get("blend_devig_method") or "proportional") for row in rows}
    return methods <= {"proportional"}


def _season_blocks(rows: list[dict], models: dict, blend: dict | None) -> dict:
    blocks: dict[str, dict] = {}
    market_prefix = "market"
    if blend and blend.get("market_prefix") == "blend_market":
        market_prefix = "blend_market"
    for key in iter_model_keys(models):
        if key == "dixon_coles_on_blend":
            source = [row for row in rows if row.get("blend_linear_ft_probs") is not None]
            prefix = "dixon_coles"
        elif key == "market_on_blend":
            source = [row for row in rows if row.get("blend_linear_ft_probs") is not None]
            prefix = market_prefix
        elif key.startswith("blend_"):
            source = [row for row in rows if row.get(f"{key}_ft_probs") is not None]
            prefix = key
        elif key in {"dixon_coles", "baseline", "xgboost", "xgboost_with_odds"}:
            source = [row for row in rows if row.get(f"{key}_ft_probs") is not None]
            prefix = key
        else:
            source = [row for row in rows if row.get(f"{key}_ft_probs") is not None]
            prefix = key
        if not source:
            continue
        grouped = by_season(source, prefix)
        if grouped:
            blocks[key] = grouped
    return blocks


def _season_payload(blocks: dict | None) -> dict | None:
    if not blocks:
        return None
    return {
        model: {season: block.to_dict() for season, block in seasons.items()}
        for model, seasons in blocks.items()
    }


def _finite(value) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    if not math.isfinite(number):
        return math.nan
    return number


def _prob_tuple(values) -> tuple[float, float, float]:
    home, draw, away = (float(value) for value in values)
    return home, draw, away


def _json_ready(value):
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if not math.isfinite(number):
            return None
        return number
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value
