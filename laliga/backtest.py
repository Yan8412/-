"""Walk-forward backtest. Training rows are always strictly earlier than the test day."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from laliga.config import ModelConfig
from laliga.model.baseline import fit_baseline
from laliga.model.dixon_coles import FitError
from laliga.model.metrics import MetricBlock, by_season, iter_evaluation_folds, outcome_index, summarize_rows
from laliga.model.service import ScorelineModel, fit_models


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
