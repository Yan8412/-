"""Walk-forward backtest. Training rows are always strictly earlier than the test day.

Every evaluated match gets the football (Dixon–Coles) forecast and the
historical-frequency baseline. When local odds exist for the match it also
gets the market forecast and the combined forecast. The combined weights
used on a test day are fitted only on matches from earlier test days, so
the combined track is scored out of sample like the others.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from laliga.config import ModelConfig, PromotionConfig
from laliga.model.baseline import fit_baseline
from laliga.model.dixon_coles import FitError
from laliga.model.ensemble import EnsembleWeights, fit_pool
from laliga.model.market import MarketBook
from laliga.model.metrics import MetricBlock, by_season, iter_evaluation_folds, outcome_index, summarize_rows
from laliga.model.promotion import common_rows, decide
from laliga.model.service import ScorelineModel, fit_models, league_ht_ratio
from laliga.model.tracks import combined_track, football_track, market_track


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
    rows: list[dict] = field(default_factory=list, repr=False)
    n_with_odds: int = 0
    tracks: dict[str, MetricBlock] = field(default_factory=dict)
    ensemble: EnsembleWeights = field(default_factory=EnsembleWeights.default)
    promotion: dict | None = None

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
            "n_with_odds": self.n_with_odds,
            "n_common": len(common_rows(self.rows)),
            "tracks": {key: value.to_dict() for key, value in self.tracks.items()},
            "ensemble": self.ensemble.to_dict(),
            "promotion": self.promotion,
            "matches": [_public_row(row) for row in self.rows],
        }


def walk_forward(
    matches: pd.DataFrame,
    config: ModelConfig,
    odds: pd.DataFrame | None = None,
    *,
    promotion: PromotionConfig | None = None,
    incumbent: dict[str, str] | None = None,
) -> BacktestReport:
    promotion = promotion or PromotionConfig()
    finished = matches[matches["status"] == "finished"].dropna(subset=["home_goals_ft", "away_goals_ft"])
    if finished.empty:
        raise FitError("没有完场比赛，无法回测。")
    book = MarketBook(odds)
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
        weights = fit_ensemble(rows, promotion.min_ensemble_rows)
        ht_ratio = league_ht_ratio(train)
        previous = model
        folds += 1
        first_day = day if first_day is None else first_day
        last_day = day
        for _, match in test.iterrows():
            rows.append(_row(match, model, baseline, book, weights, ht_ratio, config.rho_prior))
    if not rows:
        raise FitError(
            f"回测没有评测任何比赛。完场比赛有 {len(finished)} 场，"
            f"最少训练样本是 {config.min_train_matches}。可以降低 --min-train。"
        )
    shared = common_rows(rows)
    tracks: dict[str, MetricBlock] = {}
    if shared:
        for name, prefix in (("football", "model"), ("market", "market"), ("combined", "combined"), ("baseline", "baseline")):
            tracks[name] = summarize_rows(shared, prefix)
    return BacktestReport(
        n_folds=folds,
        first_test_day=None if first_day is None else pd.Timestamp(first_day).date().isoformat(),
        last_test_day=None if last_day is None else pd.Timestamp(last_day).date().isoformat(),
        min_train_matches=config.min_train_matches,
        model=summarize_rows(rows, "model"),
        baseline=summarize_rows(rows, "baseline"),
        model_by_season=by_season(rows, "model"),
        baseline_by_season=by_season(rows, "baseline"),
        rows=rows,
        n_with_odds=len(shared),
        tracks=tracks,
        ensemble=fit_ensemble(rows, promotion.min_ensemble_rows),
        promotion=decide(rows, incumbent, promotion),
    )


def fit_ensemble(rows: list[dict], min_rows: int) -> EnsembleWeights:
    """Pool weights from already-scored rows that have both football and market forecasts."""

    with_market = [row for row in rows if row.get("market_ft_probs") is not None]
    ft = fit_pool(
        np.array([row["model_ft_probs"] for row in with_market], dtype=float).reshape(-1, 3),
        np.array([row["market_ft_probs"] for row in with_market], dtype=float).reshape(-1, 3),
        [row["y_ft"] for row in with_market],
        min_rows=min_rows,
    )
    with_ht = [row for row in with_market if row.get("y_ht") is not None]
    ht = fit_pool(
        np.array([row["model_ht_probs"] for row in with_ht], dtype=float).reshape(-1, 3),
        np.array([row["market_ht_probs"] for row in with_ht], dtype=float).reshape(-1, 3),
        [row["y_ht"] for row in with_ht],
        min_rows=min_rows,
    )
    return EnsembleWeights(ft=ft, ht=ht)


def _row(match: pd.Series, model: ScorelineModel, baseline, book: MarketBook, weights: EnsembleWeights, ht_ratio: float, rho: float) -> dict:
    prediction, football, _ft_matrix, _ht_matrix = football_track(model, match)
    market = market_track(book.forecast(match, ht_ratio=ht_ratio, rho=rho, max_goals=model.ft.max_goals))
    combined = combined_track(football, market, weights)
    home_ft = int(match["home_goals_ft"])
    away_ft = int(match["away_goals_ft"])
    y_ht = None
    if pd.notna(match["home_goals_ht"]) and pd.notna(match["away_goals_ht"]):
        y_ht = outcome_index(int(match["home_goals_ht"]), int(match["away_goals_ht"]))
    season_id = match["season_id"]
    row = {
        "fixture_id": int(match["fixture_id"]),
        "starting_at": pd.Timestamp(match["starting_at"]).isoformat(),
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
        "market_ft_probs": None,
        "market_ht_probs": None,
        "market_top3": None,
        "combined_ft_probs": None,
        "combined_ht_probs": None,
        "combined_top3": None,
        "market_sources": None,
    }
    if market.available:
        row.update(
            {
                "market_ft_probs": market.ft,
                "market_ht_probs": market.ht,
                "market_top3": [(home, away) for home, away, _ in market.top_scores],
                "combined_ft_probs": combined.ft,
                "combined_ht_probs": combined.ht,
                "combined_top3": [(home, away) for home, away, _ in combined.top_scores],
                "market_sources": dict(market.sources),
            }
        )
    return row


def _public_row(row: dict) -> dict:
    document = {}
    for key, value in row.items():
        if isinstance(value, tuple):
            value = list(value)
        elif isinstance(value, list):
            value = [list(item) if isinstance(item, tuple) else item for item in value]
        document[key] = value
    return document


def rows_from_document(document: dict) -> list[dict]:
    """Rebuild in-memory rows from a saved backtest JSON (lists back to tuples)."""

    rows = []
    for item in document.get("matches") or []:
        row = dict(item)
        row["score"] = tuple(row["score"])
        for key in ("model_top3", "baseline_top3", "market_top3", "combined_top3"):
            if row.get(key) is not None:
                row[key] = [tuple(score) for score in row[key]]
        rows.append(row)
    return rows
