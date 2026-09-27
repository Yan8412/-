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
}


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

    def to_dict(self) -> dict:
        payload = {
            "schema_version": COMPARISON_SCHEMA_VERSION,
            "computed_from_stored_matches": True,
            "production_model": "dixon_coles",
            "n_folds": self.n_folds,
            "first_test_day": self.first_test_day,
            "last_test_day": self.last_test_day,
            "min_train_matches": self.min_train_matches,
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
        }
        return _json_ready(payload)


def compare_models(matches: pd.DataFrame, config: ModelConfig, xgb_config=None) -> ComparisonReport:
    """Score Dixon–Coles, the baseline, and XGBoost on the walk-forward matches.

    Dixon–Coles uses the same warm-start schedule as ``walk_forward``: the
    first evaluation day starts cold, and later days continue from the
    previous evaluation fit. A separate chain fits Dixon–Coles on warmup
    days only so XGBoost features for those matches are still as-of that
    date. That chain never becomes the model being scored.
    """

    from laliga.model.xgboost_model import XGBoostConfig, fit_outcome_model

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
    use_xg = xg_status == "present"
    first_train = _first_evaluation_train(finished, config.min_train_matches)
    odds_present_in_first_window = first_train is not None and training_window_has_odds(first_train)
    use_odds = odds_status == "present" and odds_present_in_first_window
    base_names = model_feature_names(use_xg=use_xg, use_odds=False)
    odds_names = model_feature_names(use_xg=use_xg, use_odds=True) if use_odds else []
    rolling = compute_rolling_features(finished, use_xg=use_xg).set_index("fixture_id")

    archived: list[dict] = []
    eval_rows: list[dict] = []
    feature_prev: ScorelineModel | None = None
    score_prev: ScorelineModel | None = None
    folds = 0
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
        if len(train) >= 4 and not evaluate:
            try:
                feature_prev = _fit_scoreline(train, as_of, config, feature_prev)
                feature_model = feature_prev
            except FitError:
                feature_model = feature_prev
        elif evaluate:
            try:
                scoring_model = _fit_scoreline(train, as_of, config, score_prev)
            except FitError:
                scoring_model = None
            feature_model = scoring_model if scoring_model is not None else feature_prev
            if scoring_model is not None:
                score_prev = scoring_model

        day_rows = [_feature_row(match, rolling, feature_model, use_odds=use_odds) for _, match in test.iterrows()]
        if scoring_model is not None:
            if len(archived) != len(train):
                raise FitError("赛前特征行数和训练比赛数不一致。")
            train_frame = pd.DataFrame(archived)
            test_frame = pd.DataFrame(day_rows)
            _assert_time_split(train_frame, test_frame)
            ft_names = observed_feature_names(train_frame, base_names)
            if not ft_names:
                raise FitError("训练窗口里没有可用特征。")
            if any(name in XG_FEATURES for name in ft_names):
                xg_features_used = True
            ft_model = fit_outcome_model(train_frame, ft_names, "y_ft", xgb_config)
            ft_probs = ft_model.predict(test_frame)
            ht_model, ht_probs = _fit_optional_market(train_frame, test_frame, ft_names, "y_ht", xgb_config, fit_outcome_model)
            odds_ft_probs = None
            odds_ht_probs = None
            if use_odds:
                kept_odds = observed_feature_names(train_frame, odds_names)
                if not any(name in ODDS_FEATURES for name in kept_odds):
                    raise FitError("含赔率模型的训练窗口里没有赔率特征。")
                odds_model = fit_outcome_model(train_frame, kept_odds, "y_ft", xgb_config)
                odds_ft_probs = odds_model.predict(test_frame)
                _, odds_ht_probs = _fit_optional_market(
                    train_frame, test_frame, kept_odds, "y_ht", xgb_config, fit_outcome_model
                )
                odds_stopping.append(odds_model.early_stopping)
            baseline = fit_baseline(train)
            ft_stopping.append(ft_model.early_stopping)
            ft_rounds.append(ft_model.n_rounds)
            if ht_model is not None:
                ht_stopping.append(ht_model.early_stopping)
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
                        ft_probs[index],
                        None if ht_probs is None else ht_probs[index],
                        None if odds_ft_probs is None else odds_ft_probs[index],
                        None if odds_ht_probs is None else odds_ht_probs[index],
                    )
                )
        archived.extend(day_rows)

    if not eval_rows:
        raise FitError(
            f"对照没有评测任何比赛。完场比赛有 {len(finished)} 场，"
            f"最少训练样本是 {config.min_train_matches}。可以降低 --min-train。"
        )

    y_ft = np.array([row["y_ft"] for row in eval_rows], dtype=int)
    model_keys = ["dixon_coles", "baseline", "xgboost"]
    if use_odds:
        model_keys.append("xgboost_with_odds")
    models = {key: summarize_rows(eval_rows, key) for key in model_keys}
    calibration = {
        key: calibration_summary(y_ft, np.array([row[f"{key}_ft_probs"] for row in eval_rows], dtype=float))
        for key in model_keys
    }
    paired = _paired_block(eval_rows, y_ft, xgb_config.seed)
    odds_coverage = _odds_coverage(finished, [row["fixture_id"] for row in eval_rows]) if use_odds else None
    notes = _comparison_notes(
        xg_status=xg_status,
        odds_status=odds_status,
        use_odds=use_odds,
        odds_present_in_first_window=odds_present_in_first_window,
        xg_features_used=xg_features_used,
    )
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
        },
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
    {"home_goals_ft", "away_goals_ft", "home_goals_ht", "away_goals_ht", "home_xg", "away_xg", "y_ft", "y_ht"}
)


def _comparison_row(match, model: ScorelineModel, baseline, ft_probs, ht_probs, odds_ft, odds_ht) -> dict:
    prediction = model.predict_row(match)
    home_ft = int(match["home_goals_ft"])
    away_ft = int(match["away_goals_ft"])
    y_ht = None
    if pd.notna(match["home_goals_ht"]) and pd.notna(match["away_goals_ht"]):
        y_ht = outcome_index(int(match["home_goals_ht"]), int(match["away_goals_ht"]))
    season_id = match["season_id"]
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
        "xgboost_ft_probs": _prob_tuple(ft_probs),
        "xgboost_ht_probs": None if ht_probs is None else _prob_tuple(ht_probs),
    }
    if odds_ft is not None:
        row["xgboost_with_odds_ft_probs"] = _prob_tuple(odds_ft)
        row["xgboost_with_odds_ht_probs"] = None if odds_ht is None else _prob_tuple(odds_ht)
    return row


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


def _paired_block(rows: list[dict], y_ft: np.ndarray, seed: int) -> list[dict]:
    pairs = [
        ("xgboost", "dixon_coles", "XGBoost（无赔率）− Dixon–Coles"),
        ("xgboost", "baseline", "XGBoost（无赔率）− 历史频率基准"),
        ("dixon_coles", "baseline", "Dixon–Coles − 历史频率基准"),
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
        stats = paired_logloss_difference(
            y_ft,
            np.array([row[f"{challenger}_ft_probs"] for row in rows], dtype=float),
            np.array([row[f"{reference}_ft_probs"] for row in rows], dtype=float),
            seed=seed,
        )
        compared.append({"challenger": challenger, "reference": reference, "label": label, **stats})
    return compared


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
) -> list[str]:
    notes = [
        "生产预测、操作台按钮和每日更新仍使用 Dixon–Coles。这个对照只在运行 compare 时计算。",
        "Dixon–Coles、历史频率基准和 XGBoost 使用同一批评测比赛、同一个 UTC 日切。训练行只包含该日 00:00 UTC 之前的完场比赛。",
        "XGBoost 是浅树（深度 3），随机种子固定。早停只用每个训练窗口内部按时间排在最后的一段，不用评测日。",
    ]
    if xg_features_used:
        notes.append("xG 特征只用更早比赛的赛后 xG，不用本场 xG。")
    elif xg_status == "empty":
        notes.append("比赛表有 home_xg 和 away_xg 列，但没有数值，所以没有使用 xG。")
    elif xg_status == "absent":
        notes.append("比赛表没有 home_xg 和 away_xg。当前默认 fetch 不请求 xGFixture，所以这次没有 xG 特征。")
    else:
        notes.append("存储的 xG 没有进入任何训练窗口的可用特征。")
    if use_odds:
        notes.append("含赔率的模型额外使用本场赛前 1X2 隐含概率。它衡量的是盘口之外还剩多少信息，和不用赔率的结果不是同一个问题。")
    elif odds_status == "present" and not odds_present_in_first_window:
        notes.append("本地有赛前赔率，但最早一个评测日的训练窗口里没有完整赔率，因此没有做含赔率对照，以免两行模型的比赛场次不同。")
    elif odds_status == "empty":
        notes.append("赔率列是空的。含赔率的 XGBoost 没有运行。")
    else:
        notes.append(
            "比赛表没有赛前赔率列（十进制赔率 odds_home、odds_draw、odds_away，"
            "或已经是概率的 implied_home、implied_draw、implied_away）。"
            "含赔率的 XGBoost 没有运行。当前默认 fetch 不请求 odds。"
        )
    return notes


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
