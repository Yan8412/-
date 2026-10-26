"""Back out per-team expected goals from pre-match 1X2, totals, and handicaps.

The market track does not fit team strengths. It solves for two full-time
rates and a first-half share so that a Dixon–Coles matrix matches the
de-vigged prices. Full-time 1X2, goals over/under, and Asian handicap
identify the rates. First-half totals or first-half Asian handicap, when
present, identify the share. Direct HT/FT, correct-score, and half-time
result prices are not used in the fit; they are kept for comparison.

Without a totals or handicap price the total-goals level is only weakly
identified by 1X2, so the solver pulls the sum of the two rates toward the
league average (default 2.55).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

from laliga.model.dixon_coles import score_matrix
from laliga.model.joint import HalfRates, JointForecast, convolve

_DEFAULT_TOTAL = 2.55
_DEFAULT_SHARE = 0.45
_RHO = -0.05
_RHO_LIMIT = 0.2


@dataclass
class MarketTargets:
    ft_probs: tuple[float, float, float] | None = None
    ou: list[tuple[float, float]] = field(default_factory=list)  # (line, P(over))
    ah: list[tuple[float, float]] = field(default_factory=list)  # (home handicap, P(home side wins the bet))
    ht_ou: list[tuple[float, float]] = field(default_factory=list)
    ht_ah: list[tuple[float, float]] = field(default_factory=list)
    total_prior: float = _DEFAULT_TOTAL
    share_prior: float = _DEFAULT_SHARE
    rho: float = _RHO


@dataclass
class ImpliedRates:
    lam_home: float
    lam_away: float
    ht_share: float
    rho: float
    loss: float
    used: list[str]
    draw_target: float | None = None

    def half_rates(self, eta: float = 0.0) -> HalfRates:
        share = float(np.clip(self.ht_share, 0.3, 0.7))
        return HalfRates(
            home_1=self.lam_home * share,
            away_1=self.lam_away * share,
            home_2=self.lam_home * (1.0 - share),
            away_2=self.lam_away * (1.0 - share),
            rho_1=self.rho,
            rho_2=self.rho,
            eta=eta,
        )

    def forecast(self, eta: float = 0.0, **kwargs) -> JointForecast:
        forecast = convolve(self.half_rates(eta), **kwargs)
        if self.draw_target is None:
            return forecast
        return _match_closing_draw(forecast, self.draw_target)


def infer_rates(targets: MarketTargets, *, max_goals: int = 8) -> ImpliedRates | None:
    """Least squares on the log rates and the first-half share. None if 1X2 is missing."""

    if targets.ft_probs is None:
        return None
    probs = np.asarray(targets.ft_probs, dtype=float)
    if probs.shape != (3,) or not np.isfinite(probs).all() or (probs <= 0).any():
        return None
    probs = probs / probs.sum()
    share_free = bool(targets.ht_ou or targets.ht_ah)
    # theta: log λh, log λa, logit share, arctanh rho. Rho is free so the
    # low-score correction can hit the closing draw; the forecast then
    # rescales draw scorelines because two convolved halves are not one matrix.
    rho_centre = float(np.clip(targets.rho, -_RHO_LIMIT + 1e-3, _RHO_LIMIT - 1e-3))
    start = np.array(
        [
            math.log(1.4),
            math.log(1.1),
            _logit((targets.share_prior - 0.3) / 0.4),
            math.atanh(rho_centre / _RHO_LIMIT),
        ],
        dtype=float,
    )

    def objective(theta: np.ndarray) -> float:
        lam_h = float(math.exp(theta[0]))
        lam_a = float(math.exp(theta[1]))
        share = 0.3 + 0.4 / (1.0 + math.exp(-theta[2])) if share_free else float(targets.share_prior)
        rho = _RHO_LIMIT * math.tanh(float(theta[3]))
        matrix = score_matrix(lam_h, lam_a, rho, max_goals)
        predicted = np.array(_outcome_probs(matrix))
        loss = 8.0 * _sq(probs, predicted)
        loss += 6.0 * (predicted[1] - probs[1]) ** 2
        loss += 0.05 * (rho - targets.rho) ** 2
        for line, over in targets.ou:
            loss += _sq_scalar(over, _over_prob(matrix, line))
        for handicap, cover in targets.ah:
            loss += _sq_scalar(cover, _ah_home_prob(matrix, handicap))
        if share_free:
            ht_matrix = score_matrix(lam_h * share, lam_a * share, rho, max_goals)
            for line, over in targets.ht_ou:
                loss += _sq_scalar(over, _over_prob(ht_matrix, line))
            for handicap, cover in targets.ht_ah:
                loss += _sq_scalar(cover, _ah_home_prob(ht_matrix, handicap))
        if not targets.ou and not targets.ah:
            loss += 0.35 * (lam_h + lam_a - targets.total_prior) ** 2
        if share_free:
            loss += 0.02 * (share - targets.share_prior) ** 2
        return float(loss)

    result = minimize(objective, start, method="Nelder-Mead", options={"maxiter": 160, "xatol": 1e-3, "fatol": 1e-6})
    theta = result.x if np.isfinite(result.fun) else start
    lam_h = float(math.exp(theta[0]))
    lam_a = float(math.exp(theta[1]))
    share = 0.3 + 0.4 / (1.0 + math.exp(-theta[2])) if share_free else float(targets.share_prior)
    fitted_rho = _RHO_LIMIT * math.tanh(float(theta[3]))
    used = ["1x2"]
    if targets.ou:
        used.append("ou")
    if targets.ah:
        used.append("ah")
    if targets.ht_ou:
        used.append("ht_ou")
    if targets.ht_ah:
        used.append("ht_ah")
    if not targets.ou and not targets.ah:
        used.append("total_prior")
    return ImpliedRates(
        lam_h,
        lam_a,
        float(share),
        float(fitted_rho),
        float(objective(theta)),
        used,
        draw_target=float(probs[1]),
    )


def _match_closing_draw(forecast: JointForecast, target_draw: float) -> JointForecast:
    """Rescale full-time draw scorelines so the convolved 1X2 matches the closing draw.

    Two half matrices convolved together are not the full-time Dixon–Coles
    matrix the solver matched, and the draw is the margin that moves. Home
    and away keep their relative weight in whatever mass is left.
    """

    matrix = np.array(forecast.ft_matrix, dtype=float, copy=True)
    draw = float(np.trace(matrix))
    target = float(target_draw)
    if draw <= 1e-8 or not math.isfinite(target) or not 0.0 < target < 1.0 or abs(draw - target) < 1e-4:
        return forecast
    alpha = target * (1.0 - draw) / (draw * (1.0 - target))
    if not math.isfinite(alpha) or alpha <= 0:
        return forecast
    alpha = float(np.clip(alpha, 0.05, 20.0))
    index = np.arange(matrix.shape[0])
    matrix[index, index] *= alpha
    total = float(matrix.sum())
    if total <= 0:
        return forecast
    matrix /= total
    htft = dict(forecast.htft)
    for code, value in list(htft.items()):
        if str(code).endswith("/X"):
            htft[code] = float(value) * alpha
    scale = sum(htft.values())
    if scale > 0:
        htft = {code: value / scale for code, value in htft.items()}
    from laliga.model.dixon_coles import outcome_probabilities, top_scorelines

    top = top_scorelines(matrix, max(len(forecast.top_scores), 2))
    ordered = sorted(htft.items(), key=lambda item: (-item[1], item[0]))
    goals_home = np.arange(matrix.shape[0], dtype=float)
    goals_away = np.arange(matrix.shape[1], dtype=float)
    expected_ft = (
        float((matrix.sum(axis=1) * goals_home).sum()),
        float((matrix.sum(axis=0) * goals_away).sum()),
    )
    return JointForecast(
        ft_matrix=matrix,
        ht_matrix=forecast.ht_matrix,
        ft_probs=outcome_probabilities(matrix),
        ht_probs=forecast.ht_probs,
        top_scores=top,
        top2_scores=top[:2],
        htft=htft,
        top_htft=ordered[:2],
        expected_ft=expected_ft,
        expected_ht=forecast.expected_ht,
        rates=forecast.rates,
    )


def _outcome_probs(matrix: np.ndarray) -> tuple[float, float, float]:
    size = matrix.shape[0]
    index = np.arange(size)
    home = float(matrix[index[:, None] > index[None, :]].sum())
    draw = float(np.trace(matrix))
    away = float(matrix[index[:, None] < index[None, :]].sum())
    total = home + draw + away
    return home / total, draw / total, away / total


def _over_prob(matrix: np.ndarray, line: float) -> float:
    """De-vigged over probability, pushes removed on integer lines."""

    goals = np.arange(matrix.shape[0])
    home, away = np.meshgrid(goals, goals, indexing="ij")
    total = home + away
    # A quarter line is the average of the two neighbouring half-points.
    if abs(line * 4 - round(line * 4)) < 1e-8 and abs(line * 2 - round(line * 2)) > 1e-8:
        lower = math.floor(line * 2) / 2
        upper = lower + 0.5
        return 0.5 * _over_prob(matrix, lower) + 0.5 * _over_prob(matrix, upper)
    over = float(matrix[total > line + 1e-9].sum())
    under = float(matrix[total < line - 1e-9].sum())
    denom = over + under
    if denom <= 0:
        return 0.5
    return over / denom


def _ah_home_prob(matrix: np.ndarray, handicap: float) -> float:
    """P(home side wins the bet), pushes removed. ``handicap`` is added to the home score.

    Quarter handicaps split the stake across the two neighbouring lines.
    """

    if abs(handicap * 4 - round(handicap * 4)) < 1e-8 and abs(handicap * 2 - round(handicap * 2)) > 1e-8:
        lower = math.floor(handicap * 2) / 2
        upper = lower + 0.5
        return 0.5 * _ah_home_prob(matrix, lower) + 0.5 * _ah_home_prob(matrix, upper)
    goals = np.arange(matrix.shape[0])
    home, away = np.meshgrid(goals, goals, indexing="ij")
    margin = home + handicap - away
    home_win = float(matrix[margin > 1e-9].sum())
    away_win = float(matrix[margin < -1e-9].sum())
    denom = home_win + away_win
    if denom <= 0:
        return 0.5
    return home_win / denom


def _sq(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.sum((left - right) ** 2))


def _sq_scalar(left: float, right: float) -> float:
    if not math.isfinite(left) or not math.isfinite(right):
        return 0.0
    return (left - right) ** 2


def _logit(value: float) -> float:
    clipped = min(max(value, 1e-4), 1 - 1e-4)
    return math.log(clipped / (1.0 - clipped))
