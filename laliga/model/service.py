"""Fit the full-time and half-time models and score fixtures."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from laliga.config import DEFAULT_HT_GOAL_RATIO, LA_LIGA_LEAGUE_ID, ModelConfig
from laliga.model.dixon_coles import (
    LEGACY_MODEL_VERSIONS,
    MODEL_VERSION,
    FitError,
    FittedModel,
    fit_dixon_coles,
    outcome_probabilities,
    score_matrix,
    top_scorelines,
)
from laliga.model.joint import (
    HalfRates,
    JointForecast,
    apply_xg,
    convolve,
    estimate_eta,
    estimate_xg_factors,
    second_half_frame,
    state_index,
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
    # Extra outputs stay off to_dict(). latest_next.json is the notifier contract.
    top2_scores: list[ScoreProb] = field(default_factory=list)
    top_htft: list[tuple[str, float]] = field(default_factory=list)
    htft: dict[str, float] = field(default_factory=dict)
    joint: bool = False
    track: str = "football"
    state_eta: float | None = None
    params: dict | None = None
    rates: object | None = None

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


@dataclass(frozen=True)
class GoalMultipliers:
    """Factors on the Dixon–Coles expected goals of each side. 1.0 leaves the model unchanged."""

    home: float = 1.0
    away: float = 1.0

    @property
    def neutral(self) -> bool:
        return self.home == 1.0 and self.away == 1.0


@dataclass
class ScorelineModel:
    ft: FittedModel
    ht: FittedModel
    ht_source: str
    sh: FittedModel | None = None
    state_eta: float = 0.0
    xg_attack: dict[int, float] = field(default_factory=dict)
    xg_defence: dict[int, float] = field(default_factory=dict)
    xg_weight: float = 0.0
    eta_scale: float = 1.0
    nb_dispersion: float = 1.0
    sh_rate_scale: float = 1.0
    temperature: float = 1.0
    distribution: str = "independent"

    @property
    def joint_model(self) -> bool:
        return self.sh is not None and self.distribution == "joint"

    def params_dict(self) -> dict:
        return {
            "xi": self.ft.xi,
            "rho_ft": self.ft.rho,
            "rho_ht": self.ht.rho,
            "rho_sh": None if self.sh is None else self.sh.rho,
            "state_eta": self.state_eta,
            "eta_scale": self.eta_scale,
            "xg_weight": self.xg_weight,
            "nb_dispersion": self.nb_dispersion,
            "sh_rate_scale": self.sh_rate_scale,
            "temperature": self.temperature,
            "distribution": self.distribution,
            "ht_source": self.ht_source,
            "l2_matches": self.ft.n_matches,
        }

    def expected_goals(
        self, home_id: int, away_id: int, multipliers: GoalMultipliers | None = None
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        scale = multipliers or GoalMultipliers()
        ft_home, ft_away = self.ft.expected_goals(home_id, away_id)
        ht_home, ht_away = self.ht.expected_goals(home_id, away_id)
        return (ft_home * scale.home, ft_away * scale.away), (ht_home * scale.home, ht_away * scale.away)

    def half_rates(self, home_id: int, away_id: int, multipliers: GoalMultipliers | None = None) -> HalfRates | None:
        if not self.joint_model or self.sh is None:
            return None
        scale = multipliers or GoalMultipliers()
        ht_home, ht_away = self.ht.expected_goals(home_id, away_id)
        sh_home, sh_away = self.sh.expected_goals(home_id, away_id)
        attack_home = self.xg_attack.get(int(home_id), 0.0)
        attack_away = self.xg_attack.get(int(away_id), 0.0)
        defence_home = self.xg_defence.get(int(home_id), 0.0)
        defence_away = self.xg_defence.get(int(away_id), 0.0)
        weight = self.xg_weight
        ht_home = apply_xg(ht_home, attack_home, defence_away, weight)
        ht_away = apply_xg(ht_away, attack_away, defence_home, weight)
        sh_home = apply_xg(sh_home, attack_home, defence_away, weight)
        sh_away = apply_xg(sh_away, attack_away, defence_home, weight)
        return HalfRates(
            home_1=ht_home * scale.home,
            away_1=ht_away * scale.away,
            home_2=sh_home * scale.home * self.sh_rate_scale,
            away_2=sh_away * scale.away * self.sh_rate_scale,
            rho_1=self.ht.rho,
            rho_2=self.sh.rho,
            eta=self.state_eta,
        )

    def forecast(self, home_id: int, away_id: int, multipliers: GoalMultipliers | None = None) -> JointForecast | None:
        rates = self.half_rates(home_id, away_id, multipliers)
        if rates is None:
            return None
        max_half = min(6, self.ht.max_goals)
        return convolve(
            rates,
            max_half=max_half,
            max_ft=max(self.ft.max_goals, max_half * 2),
            dispersion=self.nb_dispersion,
            temperature=self.temperature,
        )

    def matrices(self, home_id: int, away_id: int, multipliers: GoalMultipliers | None = None):
        forecast = self.forecast(home_id, away_id, multipliers)
        if forecast is not None:
            return forecast.ft_matrix, forecast.ht_matrix
        if multipliers is None or multipliers.neutral:
            return self.ft.score_matrix(home_id, away_id), self.ht.score_matrix(home_id, away_id)
        (ft_home, ft_away), (ht_home, ht_away) = self.expected_goals(home_id, away_id, multipliers)
        return (
            score_matrix(ft_home, ft_away, self.ft.rho, self.ft.max_goals),
            score_matrix(ht_home, ht_away, self.ht.rho, self.ht.max_goals),
        )

    def predict_row(self, row: pd.Series, multipliers: GoalMultipliers | None = None) -> Prediction:
        home_id = int(row["home_team_id"])
        away_id = int(row["away_team_id"])
        forecast = self.forecast(home_id, away_id, multipliers)
        if forecast is None:
            ft_matrix, ht_matrix = self.matrices(home_id, away_id, multipliers)
            expected_ft, expected_ht = self.expected_goals(home_id, away_id, multipliers)
            ft_probs = outcome_probabilities(ft_matrix)
            ht_probs = outcome_probabilities(ht_matrix)
            top = top_scorelines(ft_matrix, 3)
            top_scores = [ScoreProb(home, away, probability) for home, away, probability in top]
            extra_top2 = [ScoreProb(home, away, probability) for home, away, probability in top[:2]]
            htft: dict[str, float] = {}
            top_htft: list[tuple[str, float]] = []
            joint = False
            eta = None
        else:
            ft_probs = forecast.ft_probs
            ht_probs = forecast.ht_probs
            expected_ft = forecast.expected_ft
            expected_ht = forecast.expected_ht
            top_scores = [ScoreProb(home, away, probability) for home, away, probability in forecast.top_scores]
            extra_top2 = [ScoreProb(home, away, probability) for home, away, probability in forecast.top2_scores]
            htft = dict(forecast.htft)
            top_htft = list(forecast.top_htft)
            joint = True
            eta = self.state_eta
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
            top_scores=top_scores,
            expected_goals_ft=expected_ft,
            expected_goals_ht=expected_ht,
            top2_scores=extra_top2,
            top_htft=top_htft,
            htft=htft,
            joint=joint,
            state_eta=eta,
            params=self.params_dict(),
            rates=None if forecast is None else forecast.rates,
        )

    def to_dict(self) -> dict:
        payload = {
            "version": MODEL_VERSION,
            "league_id": LA_LIGA_LEAGUE_ID,
            "ht_source": self.ht_source,
            "distribution": self.distribution,
            "state_eta": self.state_eta,
            "xg_weight": self.xg_weight,
            "eta_scale": self.eta_scale,
            "nb_dispersion": self.nb_dispersion,
            "sh_rate_scale": self.sh_rate_scale,
            "temperature": self.temperature,
            "xg_attack": {str(team_id): value for team_id, value in sorted(self.xg_attack.items())},
            "xg_defence": {str(team_id): value for team_id, value in sorted(self.xg_defence.items())},
            "ft": self.ft.to_dict(),
            "ht": self.ht.to_dict(),
        }
        if self.sh is not None:
            payload["sh"] = self.sh.to_dict()
        return payload

    @classmethod
    def from_dict(cls, payload: dict) -> "ScorelineModel":
        version = int(payload.get("version") or 0)
        if version not in LEGACY_MODEL_VERSIONS:
            raise FitError(f"模型文件版本是 {version}，当前程序只读取版本 {sorted(LEGACY_MODEL_VERSIONS)}。请重新 train。")
        sh_payload = payload.get("sh")
        return cls(
            ft=FittedModel.from_dict(payload["ft"]),
            ht=FittedModel.from_dict(payload["ht"]),
            ht_source=str(payload.get("ht_source") or "first_half"),
            sh=None if not sh_payload else FittedModel.from_dict(sh_payload),
            state_eta=float(payload.get("state_eta") or 0.0),
            xg_attack={int(key): float(value) for key, value in (payload.get("xg_attack") or {}).items()},
            xg_defence={int(key): float(value) for key, value in (payload.get("xg_defence") or {}).items()},
            xg_weight=float(payload.get("xg_weight") or 0.0),
            eta_scale=float(payload.get("eta_scale") or 1.0),
            nb_dispersion=float(payload.get("nb_dispersion") or 1.0),
            sh_rate_scale=float(payload.get("sh_rate_scale") or 1.0),
            temperature=float(payload.get("temperature") or 1.0),
            distribution=str(payload.get("distribution") or ("joint" if sh_payload else "independent")),
        )


def fit_models(
    matches: pd.DataFrame,
    as_of: pd.Timestamp,
    config: ModelConfig,
    *,
    initial_ft: FittedModel | None = None,
    initial_ht: FittedModel | None = None,
    initial_sh: FittedModel | None = None,
) -> ScorelineModel:
    """Fit full time, first half, and (when possible) second half before ``as_of``.

    When both halves can be fit, probabilities come from their convolution.
    Half-time scores are still required for that. Without them the half-time
    block stays a scaled copy of the full-time fit, and the two markets stay
    separate.
    """

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
            max_goals=min(config.max_goals, 6),
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
    sh, eta, distribution = _fit_second_half(
        finished,
        as_of,
        config,
        ft,
        ht,
        source,
        initial_sh=initial_sh,
    )
    attack, defence = estimate_xg_factors(finished, as_of, config.xi)
    return ScorelineModel(
        ft=ft,
        ht=ht,
        ht_source=source,
        sh=sh,
        state_eta=eta,
        xg_attack=attack,
        xg_defence=defence,
        xg_weight=float(config.xg_weight),
        eta_scale=float(config.eta_scale),
        nb_dispersion=float(config.nb_dispersion),
        sh_rate_scale=float(config.sh_rate_scale),
        temperature=float(config.temperature),
        distribution=distribution,
    )


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


def _fit_second_half(
    finished: pd.DataFrame,
    as_of: pd.Timestamp,
    config: ModelConfig,
    ft: FittedModel,
    ht: FittedModel,
    source: str,
    *,
    initial_sh: FittedModel | None,
) -> tuple[FittedModel | None, float, str]:
    if source != "first_half":
        return None, 0.0, "independent"
    halves = second_half_frame(finished)
    if len(halves) < config.min_ht_matches:
        ratio = max(0.2, 1.0 - _ht_ratio(finished.dropna(subset=["home_goals_ht", "away_goals_ht"]), finished))
        return ft.scaled_copy(ratio, kind="sh"), 0.0, "joint"
    try:
        sh = fit_dixon_coles(
            halves,
            as_of=as_of,
            home_col="home_goals_sh",
            away_col="away_goals_sh",
            kind="sh",
            xi=config.xi,
            max_goals=min(config.max_goals, 6),
            l2=config.l2,
            max_iter=config.max_iter,
            min_matches=config.min_matches,
            prior_strength=config.prior_strength,
            initial_theta=None if initial_sh is None else initial_sh._theta,
            rho_prior=config.rho_prior,
            rho_precision=config.rho_precision,
        )
    except FitError:
        ratio = max(0.2, 1.0 - _ht_ratio(finished.dropna(subset=["home_goals_ht", "away_goals_ht"]), finished))
        return ft.scaled_copy(ratio, kind="sh"), 0.0, "joint"
    lam_home = []
    lam_away = []
    states = []
    goals_h = []
    goals_a = []
    for row in halves.itertuples(index=False):
        home_id = int(row.home_team_id)
        away_id = int(row.away_team_id)
        sh_home, sh_away = sh.expected_goals(home_id, away_id)
        lam_home.append(sh_home)
        lam_away.append(sh_away)
        states.append(state_index(int(row.home_goals_ht), int(row.away_goals_ht)))
        goals_h.append(int(row.home_goals_sh))
        goals_a.append(int(row.away_goals_sh))
    eta = estimate_eta(
        np.asarray(goals_h),
        np.asarray(goals_a),
        np.asarray(lam_home, dtype=float),
        np.asarray(lam_away, dtype=float),
        np.asarray(states),
        sh.rho,
        scale=config.eta_scale,
        pinned=config.pinned_eta,
    )
    # ht is part of the signature so a caller can see both halves were fit.
    del ht
    return sh, eta, "joint"


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
