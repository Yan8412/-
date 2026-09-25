"""Time-decayed Dixon–Coles model for a single goals market.

Each match has independent Poisson marginals for home and away goals, with
the Dixon–Coles dependence correction on 0-0, 1-0, 0-1, and 1-1:

    λ = exp(μ + home_advantage + attack_home − defence_away)
    ν = exp(μ + attack_away − defence_home)

    τ(0, 0) = 1 − λ ν ρ
    τ(0, 1) = 1 + λ ρ
    τ(1, 0) = 1 + ν ρ
    τ(1, 1) = 1 − ρ
    τ = 1 otherwise

Attack and defence each sum to zero across teams (the highest team id is
the residual), which identifies the model together with μ. Match weights are
``exp(−ξ · days before as_of)``. Teams with few matches are shrunk toward a
promoted-team prior when they first appear after the earliest season in the
sample, and toward the league average otherwise. A team with no history uses
that promoted prior in full.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln

MODEL_VERSION = 1


class FitError(RuntimeError):
    pass


@dataclass
class TeamStrength:
    team_id: int
    name: str
    attack: float
    defence: float
    matches: int

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "attack": self.attack,
            "defence": self.defence,
            "matches": self.matches,
        }

    @classmethod
    def from_dict(cls, team_id: int, payload: dict) -> "TeamStrength":
        return cls(
            team_id=team_id,
            name=str(payload.get("name") or team_id),
            attack=float(payload["attack"]),
            defence=float(payload["defence"]),
            matches=int(payload.get("matches") or 0),
        )


@dataclass
class FittedModel:
    kind: str
    mu: float
    home_adv: float
    rho: float
    xi: float
    max_goals: int
    trained_through: str
    n_matches: int
    teams: dict[int, TeamStrength]
    promoted_attack_prior: float
    promoted_defence_prior: float
    min_matches: int
    prior_strength: float
    goal_scale: float | None = None
    derived: bool = False
    _theta: np.ndarray | None = field(default=None, repr=False, compare=False)

    def strength(self, team_id: int) -> tuple[float, float]:
        team = self.teams.get(int(team_id))
        if team is None:
            return self.promoted_attack_prior, self.promoted_defence_prior
        return team.attack, team.defence

    def team_name(self, team_id: int) -> str:
        team = self.teams.get(int(team_id))
        if team is None:
            return str(team_id)
        return team.name

    def expected_goals(self, home_id: int, away_id: int) -> tuple[float, float]:
        attack_home, defence_home = self.strength(home_id)
        attack_away, defence_away = self.strength(away_id)
        lam_home = math.exp(self.mu + self.home_adv + attack_home - defence_away)
        lam_away = math.exp(self.mu + attack_away - defence_home)
        return lam_home, lam_away

    def score_matrix(self, home_id: int, away_id: int) -> np.ndarray:
        lam_home, lam_away = self.expected_goals(home_id, away_id)
        return score_matrix(lam_home, lam_away, self.rho, self.max_goals)

    def scaled_copy(self, ratio: float, *, kind: str) -> "FittedModel":
        """Copy parameters with both expected-goal rates multiplied by ``ratio``."""

        clipped = float(np.clip(ratio, 0.2, 0.8))
        return FittedModel(
            kind=kind,
            mu=self.mu + math.log(clipped),
            home_adv=self.home_adv,
            rho=self.rho,
            xi=self.xi,
            max_goals=self.max_goals,
            trained_through=self.trained_through,
            n_matches=self.n_matches,
            teams={tid: TeamStrength(**vars(team)) for tid, team in self.teams.items()},
            promoted_attack_prior=self.promoted_attack_prior,
            promoted_defence_prior=self.promoted_defence_prior,
            min_matches=self.min_matches,
            prior_strength=self.prior_strength,
            goal_scale=clipped,
            derived=True,
        )

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "mu": self.mu,
            "home_adv": self.home_adv,
            "rho": self.rho,
            "xi": self.xi,
            "max_goals": self.max_goals,
            "trained_through": self.trained_through,
            "n_matches": self.n_matches,
            "promoted_attack_prior": self.promoted_attack_prior,
            "promoted_defence_prior": self.promoted_defence_prior,
            "min_matches": self.min_matches,
            "prior_strength": self.prior_strength,
            "goal_scale": self.goal_scale,
            "derived": self.derived,
            "teams": {str(team_id): team.to_dict() for team_id, team in sorted(self.teams.items())},
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "FittedModel":
        teams = {
            int(team_id): TeamStrength.from_dict(int(team_id), body)
            for team_id, body in (payload.get("teams") or {}).items()
        }
        return cls(
            kind=str(payload.get("kind") or "ft"),
            mu=float(payload["mu"]),
            home_adv=float(payload["home_adv"]),
            rho=float(payload["rho"]),
            xi=float(payload.get("xi") or 0.0),
            max_goals=int(payload.get("max_goals") or 10),
            trained_through=str(payload.get("trained_through") or ""),
            n_matches=int(payload.get("n_matches") or 0),
            teams=teams,
            promoted_attack_prior=float(payload.get("promoted_attack_prior") or 0.0),
            promoted_defence_prior=float(payload.get("promoted_defence_prior") or 0.0),
            min_matches=int(payload.get("min_matches") or 10),
            prior_strength=float(payload.get("prior_strength") or 12.0),
            goal_scale=None if payload.get("goal_scale") is None else float(payload["goal_scale"]),
            derived=bool(payload.get("derived")),
        )


def shrink_strength(
    value: float,
    n_matches: int,
    prior: float,
    prior_strength: float,
    min_matches: int,
) -> float:
    """Pull a low-sample estimate toward ``prior``. Estimates with enough matches stay put."""

    if n_matches >= min_matches:
        return float(value)
    weight = n_matches + prior_strength
    if weight <= 0:
        return float(prior)
    return float((n_matches * value + prior_strength * prior) / weight)


def dixon_coles_tau(
    home_goals: np.ndarray | int,
    away_goals: np.ndarray | int,
    lam_home: np.ndarray | float,
    lam_away: np.ndarray | float,
    rho: float,
) -> np.ndarray:
    """Dixon–Coles low-score correction. See the module docstring for the cases."""

    home_goals = np.asarray(home_goals)
    away_goals = np.asarray(away_goals)
    lam_home = np.asarray(lam_home, dtype=float)
    lam_away = np.asarray(lam_away, dtype=float)
    tau = np.ones(np.broadcast(home_goals, away_goals).shape, dtype=float)
    zero_zero = (home_goals == 0) & (away_goals == 0)
    zero_one = (home_goals == 0) & (away_goals == 1)
    one_zero = (home_goals == 1) & (away_goals == 0)
    one_one = (home_goals == 1) & (away_goals == 1)
    tau = np.array(tau, dtype=float, copy=True)
    tau[zero_zero] = 1.0 - lam_home[zero_zero] * lam_away[zero_zero] * rho
    tau[zero_one] = 1.0 + lam_home[zero_one] * rho
    tau[one_zero] = 1.0 + lam_away[one_zero] * rho
    tau[one_one] = 1.0 - rho
    return tau


def score_matrix(lam_home: float, lam_away: float, rho: float, max_goals: int) -> np.ndarray:
    """Renormalised probability matrix indexed by (home goals, away goals)."""

    goals = np.arange(max_goals + 1)
    home, away = np.meshgrid(goals, goals, indexing="ij")
    lam_h = np.full(home.shape, lam_home, dtype=float)
    lam_a = np.full(away.shape, lam_away, dtype=float)
    tau = dixon_coles_tau(home, away, lam_h, lam_a, rho)
    positive = tau > 0
    safe_tau = np.where(positive, tau, 1.0)
    log_prob = np.log(safe_tau) + _poisson_logpmf(home, lam_h) + _poisson_logpmf(away, lam_a)
    # Cells with a non-positive tau are impossible under the correction.
    log_prob = np.where(positive, log_prob, -np.inf)
    shifted = log_prob - np.max(log_prob)
    matrix = np.exp(shifted)
    total = matrix.sum()
    if not np.isfinite(total) or total <= 0:
        raise FitError("比分概率矩阵无效。")
    return matrix / total


def outcome_probabilities(matrix: np.ndarray) -> tuple[float, float, float]:
    """Return (home win, draw, away win) from a home-by-away goals matrix."""

    size = matrix.shape[0]
    index = np.arange(size)
    home = float(matrix[index[:, None] > index[None, :]].sum())
    draw = float(np.trace(matrix))
    away = float(matrix[index[:, None] < index[None, :]].sum())
    total = home + draw + away
    if total <= 0:
        raise FitError("胜平负概率无效。")
    return home / total, draw / total, away / total


def top_scorelines(matrix: np.ndarray, k: int = 3) -> list[tuple[int, int, float]]:
    """Most probable exact scores. Ties break toward fewer total goals."""

    size = matrix.shape[0]
    home, away = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    flat = matrix.ravel()
    order = np.lexsort((away.ravel(), home.ravel(), (home + away).ravel(), -np.round(flat, decimals=12)))
    chosen = []
    for position in order[:k]:
        chosen.append((int(home.ravel()[position]), int(away.ravel()[position]), float(flat[position])))
    return chosen


def fit_dixon_coles(
    matches: pd.DataFrame,
    *,
    as_of: pd.Timestamp,
    home_col: str,
    away_col: str,
    kind: str,
    xi: float,
    max_goals: int,
    l2: float,
    max_iter: int,
    min_matches: int,
    prior_strength: float,
    promoted_min_matches: int = 15,
    initial_theta: np.ndarray | None = None,
    rho_prior: float = -0.05,
    rho_precision: float = 80.0,
) -> FittedModel:
    """Fit one goals market on matches with ``starting_at`` strictly before ``as_of``."""

    as_of = pd.Timestamp(as_of)
    if as_of.tzinfo is None:
        as_of = as_of.tz_localize("UTC")
    else:
        as_of = as_of.tz_convert("UTC")

    kickoff = pd.to_datetime(matches["starting_at"], utc=True)
    data = matches.loc[kickoff < as_of].copy()
    data["starting_at"] = kickoff.loc[data.index]
    data = data.dropna(subset=[home_col, away_col, "home_team_id", "away_team_id", "starting_at"])
    if "fixture_id" not in data.columns:
        data["fixture_id"] = np.arange(len(data))
    for column in ("home_team_name", "away_team_name"):
        if column not in data.columns:
            data[column] = ""
    if "season_id" not in data.columns:
        data["season_id"] = 0
    data = data.sort_values(["starting_at", "fixture_id"]).reset_index(drop=True)
    if len(data) < 4:
        raise FitError(f"{kind} 训练样本不足（{len(data)} 场）。")

    home_ids = data["home_team_id"].astype(int).to_numpy()
    away_ids = data["away_team_id"].astype(int).to_numpy()
    team_ids = np.array(sorted(set(home_ids).union(set(away_ids))), dtype=int)
    if len(team_ids) < 2:
        raise FitError("至少需要两支球队才能拟合。")
    index = {int(team_id): position for position, team_id in enumerate(team_ids)}
    home_index = np.array([index[int(team_id)] for team_id in home_ids], dtype=int)
    away_index = np.array([index[int(team_id)] for team_id in away_ids], dtype=int)
    home_goals = data[home_col].to_numpy(dtype=int)
    away_goals = data[away_col].to_numpy(dtype=int)
    days = (as_of - data["starting_at"]).dt.total_seconds().to_numpy(dtype=float) / 86400.0
    days = np.clip(days, 0.0, None)
    weights = np.exp(-xi * days)

    n_teams = len(team_ids)
    theta0 = _initial_theta(home_goals, away_goals, weights, n_teams)
    if initial_theta is not None and len(initial_theta) == len(theta0):
        theta0 = np.clip(np.asarray(initial_theta, dtype=float), [lo for lo, _ in _bounds(n_teams)], [hi for _, hi in _bounds(n_teams)])
    objective = _objective(
        home_goals,
        away_goals,
        home_index,
        away_index,
        weights,
        n_teams,
        l2,
        rho_prior=rho_prior,
        rho_precision=rho_precision,
    )
    theta = _optimise(objective, theta0, n_teams, max_iter)
    mu, home_adv, rho, attack, defence = _unpack(theta, n_teams)

    counts = _match_counts(home_ids, away_ids)
    names = _latest_names(data)
    promoted_attack, promoted_defence, new_teams = estimate_promoted_prior(
        data, team_ids, attack, defence, index, min_season_matches=promoted_min_matches
    )
    teams: dict[int, TeamStrength] = {}
    for position, team_id in enumerate(team_ids):
        team_key = int(team_id)
        prior_attack = promoted_attack if team_key in new_teams else 0.0
        prior_defence = promoted_defence if team_key in new_teams else 0.0
        played = int(counts.get(team_key, 0))
        teams[team_key] = TeamStrength(
            team_id=team_key,
            name=names.get(team_key, str(team_key)),
            attack=shrink_strength(float(attack[position]), played, prior_attack, prior_strength, min_matches),
            defence=shrink_strength(float(defence[position]), played, prior_defence, prior_strength, min_matches),
            matches=played,
        )

    trained_through = pd.Timestamp(data["starting_at"].max()).isoformat()
    model = FittedModel(
        kind=kind,
        mu=float(mu),
        home_adv=float(home_adv),
        rho=float(rho),
        xi=float(xi),
        max_goals=int(max_goals),
        trained_through=trained_through,
        n_matches=int(len(data)),
        teams=teams,
        promoted_attack_prior=float(promoted_attack),
        promoted_defence_prior=float(promoted_defence),
        min_matches=int(min_matches),
        prior_strength=float(prior_strength),
    )
    model._theta = theta
    return model


def estimate_promoted_prior(
    matches: pd.DataFrame,
    team_ids: np.ndarray,
    attack: np.ndarray,
    defence: np.ndarray,
    index: dict[int, int],
    min_season_matches: int,
) -> tuple[float, float, set[int]]:
    """Mean strength of teams that entered after the first season in the sample.

    Returns ``(attack prior, defence prior, set of team ids that are new)``.
    The prior is 0 when fewer than two such teams have enough matches to be
    informative. New teams are still identified so shrinkage can target the
    prior (or zero, when the prior itself is zero).
    """

    if "season_id" not in matches.columns or matches["season_id"].nunique(dropna=True) < 2:
        return 0.0, 0.0, set()
    season_start = matches.groupby("season_id")["starting_at"].min().sort_values()
    first_season = int(season_start.index[0])
    appearances = pd.concat(
        [
            matches[["home_team_id", "season_id", "starting_at"]].rename(columns={"home_team_id": "team_id"}),
            matches[["away_team_id", "season_id", "starting_at"]].rename(columns={"away_team_id": "team_id"}),
        ],
        ignore_index=True,
    )
    appearances = appearances.dropna(subset=["team_id", "season_id"])
    appearances["team_id"] = appearances["team_id"].astype(int)
    appearances["season_id"] = appearances["season_id"].astype(int)
    first = appearances.sort_values("starting_at").groupby("team_id", as_index=False).first()
    counts = appearances.groupby("team_id").size()
    new_teams = set(first.loc[first["season_id"] != first_season, "team_id"].astype(int))
    eligible = [team_id for team_id in new_teams if int(counts.get(team_id, 0)) >= min_season_matches]
    if len(eligible) < 2:
        return 0.0, 0.0, new_teams
    attack_mean = float(np.mean([attack[index[team_id]] for team_id in eligible]))
    defence_mean = float(np.mean([defence[index[team_id]] for team_id in eligible]))
    return attack_mean, defence_mean, new_teams


def _initial_theta(home_goals: np.ndarray, away_goals: np.ndarray, weights: np.ndarray, n_teams: int) -> np.ndarray:
    weight = float(weights.sum())
    mean_home = float(np.sum(weights * home_goals) / weight)
    mean_away = float(np.sum(weights * away_goals) / weight)
    mean_home = max(mean_home, 0.05)
    mean_away = max(mean_away, 0.05)
    mu = float(np.clip(math.log(mean_away), -1.2, 1.2))
    home = float(np.clip(math.log(mean_home) - math.log(mean_away), -0.2, 0.8))
    theta = np.zeros(3 + 2 * (n_teams - 1), dtype=float)
    theta[0] = mu
    theta[1] = home
    theta[2] = -0.03
    return theta


def _optimise(objective, theta0: np.ndarray, n_teams: int, max_iter: int) -> np.ndarray:
    bounds = _bounds(n_teams)
    result = minimize(
        objective,
        theta0,
        method="L-BFGS-B",
        jac=True,
        bounds=bounds,
        options={"maxiter": max_iter, "ftol": 1e-10, "maxls": 25},
    )
    if not np.isfinite(result.fun):
        fallback = np.zeros_like(theta0)
        fallback[0] = theta0[0]
        result = minimize(
            objective,
            fallback,
            method="L-BFGS-B",
            jac=True,
            bounds=bounds,
            options={"maxiter": max_iter, "ftol": 1e-10, "maxls": 25},
        )
    if not np.isfinite(result.fun):
        raise FitError("Dixon–Coles 拟合没有得到有限的似然。")
    return np.asarray(result.x, dtype=float)


def _bounds(n_teams: int) -> list[tuple[float, float]]:
    bounds: list[tuple[float, float]] = [(-1.5, 1.5), (-0.4, 1.2), (-0.2, 0.2)]
    bounds.extend([(-2.0, 2.0)] * (2 * (n_teams - 1)))
    return bounds


def _unpack(theta: np.ndarray, n_teams: int) -> tuple[float, float, float, np.ndarray, np.ndarray]:
    mu = float(theta[0])
    home = float(theta[1])
    rho = float(theta[2])
    free = n_teams - 1
    attack = np.zeros(n_teams, dtype=float)
    defence = np.zeros(n_teams, dtype=float)
    attack[:-1] = theta[3 : 3 + free]
    defence[:-1] = theta[3 + free : 3 + 2 * free]
    attack[-1] = -attack[:-1].sum()
    defence[-1] = -defence[:-1].sum()
    return mu, home, rho, attack, defence


def _objective(
    home_goals: np.ndarray,
    away_goals: np.ndarray,
    home_index: np.ndarray,
    away_index: np.ndarray,
    weights: np.ndarray,
    n_teams: int,
    l2: float,
    rho_prior: float = -0.05,
    rho_precision: float = 80.0,
):
    """Weighted mean negative log-likelihood plus an L2 penalty, with gradient."""

    def evaluate(theta: np.ndarray) -> tuple[float, np.ndarray]:
        mu, home, rho, attack, defence = _unpack(theta, n_teams)
        lam_home = np.exp(mu + home + attack[home_index] - defence[away_index])
        lam_away = np.exp(mu + attack[away_index] - defence[home_index])
        safe = np.zeros_like(theta)
        safe[0] = mu
        if (
            not np.isfinite(lam_home).all()
            or not np.isfinite(lam_away).all()
            or np.any(lam_home > 40)
            or np.any(lam_away > 40)
        ):
            gap = theta - safe
            return 1e3 + 1e2 * float(np.dot(gap, gap)), 2e2 * gap

        tau = dixon_coles_tau(home_goals, away_goals, lam_home, lam_away, rho)
        if not np.isfinite(tau).all() or np.any(tau <= 1e-8):
            gap = theta - safe
            gap[2] += rho
            return 1e3 + 1e2 * float(np.dot(gap, gap)), 2e2 * gap

        log_prob = np.log(tau) + _poisson_logpmf(home_goals, lam_home) + _poisson_logpmf(away_goals, lam_away)
        weight_sum = float(weights.sum())
        nll = -float(np.sum(weights * log_prob) / weight_sum)
        # Priors are on the total log-likelihood, then averaged so they fade as n grows.
        penalty = l2 * float(np.sum(attack**2) + np.sum(defence**2)) / weight_sum
        penalty += 0.5 * rho_precision * (rho - rho_prior) ** 2 / weight_sum

        zero_zero = (home_goals == 0) & (away_goals == 0)
        zero_one = (home_goals == 0) & (away_goals == 1)
        one_zero = (home_goals == 1) & (away_goals == 0)
        one_one = (home_goals == 1) & (away_goals == 1)
        dtau_d_home = np.zeros(len(home_goals))
        dtau_d_away = np.zeros(len(home_goals))
        dtau_d_rho = np.zeros(len(home_goals))
        dtau_d_home[zero_zero] = -lam_away[zero_zero] * rho
        dtau_d_away[zero_zero] = -lam_home[zero_zero] * rho
        dtau_d_rho[zero_zero] = -lam_home[zero_zero] * lam_away[zero_zero]
        dtau_d_home[zero_one] = rho
        dtau_d_rho[zero_one] = lam_home[zero_one]
        dtau_d_away[one_zero] = rho
        dtau_d_rho[one_zero] = lam_away[one_zero]
        dtau_d_rho[one_one] = -1.0
        inv_tau = 1.0 / tau
        dlog_d_home = inv_tau * dtau_d_home + home_goals / lam_home - 1.0
        dlog_d_away = inv_tau * dtau_d_away + away_goals / lam_away - 1.0
        dlog_d_rho = inv_tau * dtau_d_rho

        grad_mu = float(np.sum(weights * (dlog_d_home * lam_home + dlog_d_away * lam_away)))
        grad_home = float(np.sum(weights * (dlog_d_home * lam_home)))
        grad_rho = float(np.sum(weights * dlog_d_rho))

        attack_home = dlog_d_home * lam_home
        attack_away = dlog_d_away * lam_away
        grad_attack = np.bincount(home_index, weights=weights * attack_home, minlength=n_teams)
        grad_attack += np.bincount(away_index, weights=weights * attack_away, minlength=n_teams)
        grad_defence = np.bincount(away_index, weights=weights * (-attack_home), minlength=n_teams)
        grad_defence += np.bincount(home_index, weights=weights * (-attack_away), minlength=n_teams)

        free_attack = -(grad_attack[:-1] - grad_attack[-1]) / weight_sum
        free_defence = -(grad_defence[:-1] - grad_defence[-1]) / weight_sum
        free_attack = free_attack + 2.0 * l2 * (attack[:-1] - attack[-1]) / weight_sum
        free_defence = free_defence + 2.0 * l2 * (defence[:-1] - defence[-1]) / weight_sum
        rho_gradient = -grad_rho / weight_sum + rho_precision * (rho - rho_prior) / weight_sum
        gradient = np.concatenate(
            [
                np.array([-grad_mu / weight_sum, -grad_home / weight_sum, rho_gradient]),
                free_attack,
                free_defence,
            ]
        )
        return nll + penalty, gradient

    return evaluate


def _poisson_logpmf(goals: np.ndarray, lam: np.ndarray) -> np.ndarray:
    return goals * np.log(lam) - lam - gammaln(goals + 1.0)


def _match_counts(home_ids: np.ndarray, away_ids: np.ndarray) -> dict[int, int]:
    counts: dict[int, int] = {}
    for team_id in list(home_ids) + list(away_ids):
        key = int(team_id)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _latest_names(matches: pd.DataFrame) -> dict[int, str]:
    ordered = matches.sort_values("starting_at")
    names: dict[int, str] = {}
    for team_id, name in zip(ordered["home_team_id"].astype(int), ordered["home_team_name"], strict=True):
        if isinstance(name, str) and name:
            names[int(team_id)] = name
    for team_id, name in zip(ordered["away_team_id"].astype(int), ordered["away_team_name"], strict=True):
        if isinstance(name, str) and name:
            names[int(team_id)] = name
    return names
