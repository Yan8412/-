"""Fit the full-time and half-time models and score fixtures."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from laliga.config import DEFAULT_HT_GOAL_RATIO, LA_LIGA_LEAGUE_ID, ModelConfig
from laliga.model.dixon_coles import (
    MODEL_VERSION,
    FitError,
    FittedModel,
    fit_dixon_coles,
    outcome_probabilities,
    top_scorelines,
)


@dataclass
class MarketProb:
    home: float
    draw: float
    away: float

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.home, self.draw, self.away)

    def to_dict(self) -> dict[str, float]:
        return {"home": self.home, "draw": self.draw, "away": self.away}


@dataclass
class ScoreProb:
    home_goals: int
    away_goals: int
    probability: float

    def to_dict(self) -> dict:
        return {
            "home_goals": self.home_goals,
            "away_goals": self.away_goals,
            "probability": self.probability,
        }


@dataclass
class Prediction:
    fixture_id: int
    starting_at: str
    season_id: int | None
    season_name: str
    round_name: str
    home_team_id: int
    home_team: str
    away_team_id: int
    away_team: str
    ft: MarketProb
    ht: MarketProb
    top_scores: list[ScoreProb]
    expected_goals_ft: tuple[float, float]
    expected_goals_ht: tuple[float, float]

    def to_dict(self) -> dict:
        return {
            "fixture_id": self.fixture_id,
            "starting_at": self.starting_at,
            "season_id": self.season_id,
            "season_name": self.season_name,
            "round_name": self.round_name,
            "home_team_id": self.home_team_id,
            "home_team": self.home_team,
            "away_team_id": self.away_team_id,
            "away_team": self.away_team,
            "ft": self.ft.to_dict(),
            "ht": self.ht.to_dict(),
            "top_scores": [score.to_dict() for score in self.top_scores],
            "expected_goals_ft": {
                "home": self.expected_goals_ft[0],
                "away": self.expected_goals_ft[1],
            },
            "expected_goals_ht": {
                "home": self.expected_goals_ht[0],
                "away": self.expected_goals_ht[1],
            },
        }


@dataclass
class ScorelineModel:
    ft: FittedModel
    ht: FittedModel
    ht_source: str

    def predict_row(self, row: pd.Series) -> Prediction:
        home_id = int(row["home_team_id"])
        away_id = int(row["away_team_id"])
        ft_matrix = self.ft.score_matrix(home_id, away_id)
        ht_matrix = self.ht.score_matrix(home_id, away_id)
        ft_probs = outcome_probabilities(ft_matrix)
        ht_probs = outcome_probabilities(ht_matrix)
        kickoff = pd.Timestamp(row["starting_at"])
        season_id = row["season_id"]
        return Prediction(
            fixture_id=int(row["fixture_id"]),
            starting_at=kickoff.isoformat(),
            season_id=None if pd.isna(season_id) else int(season_id),
            season_name=str(row.get("season_name") or ""),
            round_name=str(row.get("round_name") or ""),
            home_team_id=home_id,
            home_team=str(row["home_team_name"]),
            away_team_id=away_id,
            away_team=str(row["away_team_name"]),
            ft=MarketProb(*ft_probs),
            ht=MarketProb(*ht_probs),
            top_scores=[ScoreProb(home, away, probability) for home, away, probability in top_scorelines(ft_matrix, 3)],
            expected_goals_ft=self.ft.expected_goals(home_id, away_id),
            expected_goals_ht=self.ht.expected_goals(home_id, away_id),
        )

    def to_dict(self) -> dict:
        return {
            "version": MODEL_VERSION,
            "league_id": LA_LIGA_LEAGUE_ID,
            "ht_source": self.ht_source,
            "ft": self.ft.to_dict(),
            "ht": self.ht.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "ScorelineModel":
        version = int(payload.get("version") or 0)
        if version != MODEL_VERSION:
            raise FitError(f"模型文件版本是 {version}，当前程序只读取版本 {MODEL_VERSION}。请重新 train。")
        return cls(
            ft=FittedModel.from_dict(payload["ft"]),
            ht=FittedModel.from_dict(payload["ht"]),
            ht_source=str(payload.get("ht_source") or "first_half"),
        )


def fit_models(
    matches: pd.DataFrame,
    as_of: pd.Timestamp,
    config: ModelConfig,
    *,
    initial_ft: FittedModel | None = None,
    initial_ht: FittedModel | None = None,
) -> ScorelineModel:
    """Fit both markets on finished matches kicked off before ``as_of``."""

    finished = _finished_before(matches, as_of)
    if len(finished) < 4:
        raise FitError("完场比赛太少，无法训练。请先 fetch 几个西甲赛季。")
    ft = fit_dixon_coles(
        finished,
        as_of=as_of,
        home_col="home_goals_ft",
        away_col="away_goals_ft",
        kind="ft",
        xi=config.xi,
        max_goals=config.max_goals,
        l2=config.l2,
        max_iter=config.max_iter,
        min_matches=config.min_matches,
        prior_strength=config.prior_strength,
        initial_theta=None if initial_ft is None else initial_ft._theta,
        rho_prior=config.rho_prior,
        rho_precision=config.rho_precision,
    )
    ht_rows = finished.dropna(subset=["home_goals_ht", "away_goals_ht"])
    if len(ht_rows) >= config.min_ht_matches:
        ht = fit_dixon_coles(
            ht_rows,
            as_of=as_of,
            home_col="home_goals_ht",
            away_col="away_goals_ht",
            kind="ht",
            xi=config.xi,
            max_goals=config.max_goals,
            l2=config.l2,
            max_iter=config.max_iter,
            min_matches=config.min_matches,
            prior_strength=config.prior_strength,
            initial_theta=None if initial_ht is None else initial_ht._theta,
            rho_prior=config.rho_prior,
            rho_precision=config.rho_precision,
        )
        source = "first_half"
    else:
        ratio = _ht_ratio(ht_rows, finished)
        ht = ft.scaled_copy(ratio, kind="ht")
        source = "scaled_full_time"
    return ScorelineModel(ft=ft, ht=ht, ht_source=source)


def model_is_fresh(model: ScorelineModel, history: pd.DataFrame, fixtures: pd.DataFrame) -> bool:
    """True when the saved fit already includes every finished result before these fixtures."""

    finished = history[history["status"] == "finished"]
    if finished.empty or fixtures.empty:
        return False
    trained = pd.Timestamp(model.ft.trained_through)
    if trained.tzinfo is None:
        trained = trained.tz_localize("UTC")
    latest_result = pd.Timestamp(finished["starting_at"].max())
    earliest_fixture = pd.Timestamp(fixtures["starting_at"].min())
    return trained >= latest_result and trained < earliest_fixture


def _finished_before(matches: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    as_of = pd.Timestamp(as_of)
    if as_of.tzinfo is None:
        as_of = as_of.tz_localize("UTC")
    frame = matches
    if "status" in frame.columns:
        frame = frame[frame["status"] == "finished"]
    return frame[frame["starting_at"] < as_of]


def _ht_ratio(ht_rows: pd.DataFrame, finished: pd.DataFrame) -> float:
    if ht_rows.empty:
        return DEFAULT_HT_GOAL_RATIO
    ht_goals = float(ht_rows["home_goals_ht"].sum() + ht_rows["away_goals_ht"].sum())
    ft_goals = float(finished["home_goals_ft"].sum() + finished["away_goals_ft"].sum())
    if ft_goals <= 0:
        return DEFAULT_HT_GOAL_RATIO
    return float(ht_goals / ft_goals)
