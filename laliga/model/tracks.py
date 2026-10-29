"""Three forecasts per fixture: football (Dixon–Coles), market (odds), combined."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from laliga.config import PROMOTION_MARKETS, TRACK_LABELS, TRACKS
from laliga.model.dixon_coles import top_scorelines
from laliga.model.ensemble import EnsembleWeights, log_pool, pool_grid
from laliga.model.market import MarketBook, MarketForecast
from laliga.model.service import Prediction, ScorelineModel

NO_ODDS_NOTE = "这场比赛没有可用的全场胜平负赔率。"


@dataclass
class TrackForecast:
    track: str
    available: bool
    ft: tuple[float, float, float] | None = None
    ht: tuple[float, float, float] | None = None
    top_scores: list[tuple[int, int, float]] = field(default_factory=list)
    matrix: np.ndarray | None = None
    sources: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    detail: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        payload = {
            "track": self.track,
            "label": TRACK_LABELS[self.track],
            "available": self.available,
            "ft": _outcome(self.ft),
            "ht": _outcome(self.ht),
            "top_scores": [
                {"home_goals": home, "away_goals": away, "probability": probability}
                for home, away, probability in self.top_scores
            ],
            "sources": dict(self.sources),
            "notes": list(self.notes),
        }
        payload.update(self.detail)
        return payload


@dataclass
class FixtureForecast:
    prediction: Prediction
    tracks: dict[str, TrackForecast]
    active: dict[str, dict]
    ft_matrix: np.ndarray
    ht_matrix: np.ndarray

    def view(self, track: str) -> TrackForecast:
        """The forecast to show for ``track``; ``active`` mixes the promoted track per market."""

        if track != "active":
            return self.tracks[track]
        chosen = {market: self.active[market]["track"] for market in PROMOTION_MARKETS}
        return TrackForecast(
            track="football",
            available=True,
            ft=self.tracks[chosen["ft"]].ft,
            ht=self.tracks[chosen["ht"]].ht,
            top_scores=self.tracks[chosen["scores"]].top_scores,
        )

    def to_dict(self) -> dict:
        document = self.prediction.to_dict()
        document["ft_matrix"] = self.ft_matrix.round(6).tolist()
        document["ht_matrix"] = self.ht_matrix.round(6).tolist()
        document["tracks"] = {name: forecast.to_dict() for name, forecast in self.tracks.items()}
        document["active"] = {market: dict(choice) for market, choice in self.active.items()}
        active_view = self.view("active")
        document["active_view"] = {
            "ft": _outcome(active_view.ft),
            "ht": _outcome(active_view.ht),
            "top_scores": [
                {"home_goals": home, "away_goals": away, "probability": probability}
                for home, away, probability in active_view.top_scores
            ],
        }
        return document


def football_track(model: ScorelineModel, row: pd.Series) -> tuple[Prediction, TrackForecast, np.ndarray, np.ndarray]:
    prediction = model.predict_row(row)
    home_id = int(row["home_team_id"])
    away_id = int(row["away_team_id"])
    ft_matrix = model.ft.score_matrix(home_id, away_id)
    ht_matrix = model.ht.score_matrix(home_id, away_id)
    forecast = TrackForecast(
        track="football",
        available=True,
        ft=prediction.ft.as_tuple(),
        ht=prediction.ht.as_tuple(),
        top_scores=[(score.home_goals, score.away_goals, score.probability) for score in prediction.top_scores],
        matrix=ft_matrix,
        sources={"ft": "dixon_coles", "ht": "dixon_coles_" + model.ht_source, "scores": "dixon_coles"},
    )
    return prediction, forecast, ft_matrix, ht_matrix


def market_track(market: MarketForecast | None) -> TrackForecast:
    if market is None:
        return TrackForecast(track="market", available=False, notes=[NO_ODDS_NOTE])
    return TrackForecast(
        track="market",
        available=True,
        ft=market.ft,
        ht=market.ht,
        top_scores=market.top_scores,
        matrix=market.matrix,
        sources=dict(market.sources),
        notes=list(market.notes),
        detail={
            "bookmakers": dict(market.bookmakers),
            "overround_ft": market.overround_ft,
            "method": market.method,
            "expected_goals_ft": {"home": market.expected_goals_ft[0], "away": market.expected_goals_ft[1]},
        },
    )


def combined_track(football: TrackForecast, market: TrackForecast, weights: EnsembleWeights) -> TrackForecast:
    if not market.available:
        return TrackForecast(track="combined", available=False, notes=["缺少赔率，无法融合。" + NO_ODDS_NOTE])
    ft = tuple(float(value) for value in log_pool(football.ft, market.ft, weights.ft))
    ht = tuple(float(value) for value in log_pool(football.ht, market.ht, weights.ht))
    matrix = pool_grid(football.matrix, market.matrix, weights.ft)
    notes = []
    if not weights.ft.fitted:
        notes.append("权重尚未用回测拟合，暂用等权（0.5, 0.5）。")
    return TrackForecast(
        track="combined",
        available=True,
        ft=ft,
        ht=ht,
        top_scores=top_scorelines(matrix, 3),
        matrix=matrix,
        sources={"ft": "log_pool", "ht": "log_pool", "scores": "log_pool"},
        notes=notes,
        detail={"weights": weights.to_dict()},
    )


def forecast_fixture(
    model: ScorelineModel,
    row: pd.Series,
    book: MarketBook,
    weights: EnsembleWeights,
    active: dict[str, str],
    *,
    ht_ratio: float,
    rho: float = -0.05,
) -> FixtureForecast:
    prediction, football, ft_matrix, ht_matrix = football_track(model, row)
    market = market_track(book.forecast(row, ht_ratio=ht_ratio, rho=rho, max_goals=model.ft.max_goals))
    combined = combined_track(football, market, weights)
    tracks = {"football": football, "market": market, "combined": combined}
    return FixtureForecast(
        prediction=prediction,
        tracks=tracks,
        active=resolve_active(tracks, active),
        ft_matrix=ft_matrix,
        ht_matrix=ht_matrix,
    )


def resolve_active(tracks: dict[str, TrackForecast], active: dict[str, str]) -> dict[str, dict]:
    """Per market, the promoted track, or football when that track has nothing for this fixture."""

    resolved = {}
    for market in PROMOTION_MARKETS:
        wanted = active.get(market, "football")
        if wanted not in TRACKS:
            wanted = "football"
        if tracks[wanted].available:
            resolved[market] = {"track": wanted, "promoted": wanted, "fallback": False}
        else:
            resolved[market] = {"track": "football", "promoted": wanted, "fallback": True}
    return resolved


def _outcome(values) -> dict[str, float] | None:
    if values is None:
        return None
    return {"home": float(values[0]), "draw": float(values[1]), "away": float(values[2])}
