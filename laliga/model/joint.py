"""Joint half-time / full-time distribution from two Dixon–Coles halves.

Full-time goals are the sum of first-half and second-half goals. Each half
has its own Dixon–Coles score matrix (the existing fitter). The second half
can depend on the half-time state: a positive ``eta`` lowers the leading
side's rate and raises the trailing side's rate. Convolving the two halves
produces one distribution over (HT score, FT score). Full-time 1X2,
half-time 1X2, exact scores, and the nine HT/FT outcomes are all margins of
that distribution, so they cannot contradict each other.

``eta = 0`` is independent halves. The half-time margin is exactly the
first-half matrix. A full-time Dixon–Coles fit is still stored for team
strengths and for the availability adjustment; match probabilities come from
the convolution, not from that full-time matrix.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import gammaln

from laliga.model.dixon_coles import dixon_coles_tau, score_matrix

HTFT_CODES = ("1/1", "1/X", "1/2", "X/1", "X/X", "X/2", "2/1", "2/X", "2/2")
# Leading side slows down. Grid is wide enough that a flat sample stays at 0.
_ETA_GRID = np.round(np.linspace(-0.25, 0.45, 29), 4)
_ETA_PRIOR_SD = 0.2


@dataclass(frozen=True)
class HalfRates:
    home_1: float
    away_1: float
    home_2: float
    away_2: float
    rho_1: float
    rho_2: float
    eta: float = 0.0

    def scaled(self, home: float, away: float, sh_scale: float = 1.0) -> "HalfRates":
        return HalfRates(
            home_1=self.home_1 * home,
            away_1=self.away_1 * away,
            home_2=self.home_2 * home * sh_scale,
            away_2=self.away_2 * away * sh_scale,
            rho_1=self.rho_1,
            rho_2=self.rho_2,
            eta=self.eta,
        )


@dataclass
class JointForecast:
    ft_matrix: np.ndarray
    ht_matrix: np.ndarray
    ft_probs: tuple[float, float, float]
    ht_probs: tuple[float, float, float]
    top_scores: list[tuple[int, int, float]]
    top2_scores: list[tuple[int, int, float]]
    htft: dict[str, float]
    top_htft: list[tuple[str, float]]
    expected_ft: tuple[float, float]
    expected_ht: tuple[float, float]
    rates: HalfRates

    @property
    def top2_score_pairs(self) -> list[tuple[int, int]]:
        return [(home, away) for home, away, _ in self.top2_scores]


def result_code(home_goals: int, away_goals: int) -> str:
    if home_goals > away_goals:
        return "1"
    if home_goals == away_goals:
        return "X"
    return "2"


def htft_code(ht_home: int, ht_away: int, ft_home: int, ft_away: int) -> str:
    return f"{result_code(ht_home, ht_away)}/{result_code(ft_home, ft_away)}"


def state_index(home_goals: int, away_goals: int) -> int:
    """0 home lead, 1 level, 2 away lead."""

    if home_goals > away_goals:
        return 0
    if home_goals == away_goals:
        return 1
    return 2


def adjust_second_half(lam_home: float, lam_away: float, state: int, eta: float) -> tuple[float, float]:
    """Scale second-half rates by the half-time state.

    ``eta > 0`` means the side that is leading scores less in the second
    half and the side that is trailing scores more. A level score is unchanged.
    """

    if state == 1 or eta == 0.0:
        return lam_home, lam_away
    lead = math.exp(-eta)
    trail = math.exp(eta)
    if state == 0:
        return lam_home * lead, lam_away * trail
    return lam_home * trail, lam_away * lead


def temper_matrix(matrix: np.ndarray, temperature: float) -> np.ndarray:
    """Raise a distribution to ``1/T`` and renormalise. ``T = 1`` is a no-op."""

    if temperature == 1.0 or not np.isfinite(temperature) or temperature <= 0:
        return matrix
    logged = np.log(np.clip(matrix, 1e-300, None)) / float(temperature)
    logged -= np.max(logged)
    out = np.exp(logged)
    total = out.sum()
    if not np.isfinite(total) or total <= 0:
        return matrix
    return out / total


def goals_matrix(
    lam_home: float,
    lam_away: float,
    rho: float,
    max_goals: int,
    dispersion: float = 1.0,
) -> np.ndarray:
    """Dixon–Coles matrix. ``dispersion > 1`` uses a negative-binomial margin.

    The negative binomial keeps the Dixon–Coles mean and the low-score
    correction, and inflates variance to ``mean * dispersion``. That is the
    predictive idea in the R package goalmodel, without refitting the attack
    and defence parameters. ``dispersion == 1`` is the Poisson matrix.
    """

    if dispersion <= 1.001:
        return score_matrix(lam_home, lam_away, rho, max_goals)
    goals = np.arange(max_goals + 1)
    home, away = np.meshgrid(goals, goals, indexing="ij")
    lam_h = np.full(home.shape, max(lam_home, 1e-6), dtype=float)
    lam_a = np.full(away.shape, max(lam_away, 1e-6), dtype=float)
    tau = dixon_coles_tau(home, away, lam_h, lam_a, rho)
    positive = tau > 0
    log_tau = np.where(positive, np.log(np.clip(tau, 1e-300, None)), -np.inf)
    log_prob = log_tau + _nb_logpmf(home, lam_h, dispersion) + _nb_logpmf(away, lam_a, dispersion)
    log_prob = np.where(np.isfinite(log_prob), log_prob, -np.inf)
    if not np.isfinite(log_prob).any():
        return score_matrix(lam_home, lam_away, rho, max_goals)
    shifted = log_prob - np.max(log_prob)
    matrix = np.exp(shifted)
    total = matrix.sum()
    if not np.isfinite(total) or total <= 0:
        return score_matrix(lam_home, lam_away, rho, max_goals)
    return matrix / total


def convolve(
    rates: HalfRates,
    *,
    max_half: int = 6,
    max_ft: int = 10,
    dispersion: float = 1.0,
    temperature: float = 1.0,
    top_k: int = 3,
) -> JointForecast:
    """Convolve the two halves into one joint distribution and its margins."""

    max_half = int(max_half)
    max_ft = int(max(max_ft, max_half))
    first = temper_matrix(
        goals_matrix(rates.home_1, rates.away_1, rates.rho_1, max_half, dispersion),
        temperature,
    )
    second: dict[int, np.ndarray] = {}
    for state in (0, 1, 2):
        lam_h, lam_a = adjust_second_half(rates.home_2, rates.away_2, state, rates.eta)
        second[state] = temper_matrix(
            goals_matrix(lam_h, lam_a, rates.rho_2, max_half, dispersion),
            temperature,
        )
    ft = np.zeros((max_ft + 1, max_ft + 1), dtype=float)
    ht = np.zeros((max_half + 1, max_half + 1), dtype=float)
    htft = {code: 0.0 for code in HTFT_CODES}
    for ht_home in range(max_half + 1):
        for ht_away in range(max_half + 1):
            mass_ht = float(first[ht_home, ht_away])
            if mass_ht <= 0.0:
                continue
            ht[ht_home, ht_away] = mass_ht
            sh = second[state_index(ht_home, ht_away)]
            ht_code = result_code(ht_home, ht_away)
            for sh_home in range(max_half + 1):
                ft_home = min(max_ft, ht_home + sh_home)
                for sh_away in range(max_half + 1):
                    mass = mass_ht * float(sh[sh_home, sh_away])
                    if mass <= 0.0:
                        continue
                    ft_away = min(max_ft, ht_away + sh_away)
                    ft[ft_home, ft_away] += mass
                    code = f"{ht_code}/{result_code(ft_home, ft_away)}"
                    # Censoring onto the last bin can merge two HT/FT cells.
                    # The code is still a valid 1/X/2 pair, so the nine-way stays a distribution.
                    if code in htft:
                        htft[code] += mass
    ft_total = float(ft.sum())
    ht_total = float(ht.sum())
    if ft_total <= 0 or ht_total <= 0:
        raise ValueError("联合分布没有概率质量。")
    ft /= ft_total
    ht /= ht_total
    htft_total = sum(htft.values())
    if htft_total <= 0:
        raise ValueError("半全场分布没有概率质量。")
    htft = {code: value / htft_total for code, value in htft.items()}
    from laliga.model.dixon_coles import outcome_probabilities, top_scorelines

    top = top_scorelines(ft, top_k)
    ordered_htft = sorted(htft.items(), key=lambda item: (-item[1], item[0]))
    return JointForecast(
        ft_matrix=ft,
        ht_matrix=ht,
        ft_probs=outcome_probabilities(ft),
        ht_probs=outcome_probabilities(ht),
        top_scores=top,
        top2_scores=top[:2],
        htft=htft,
        top_htft=ordered_htft[:2],
        expected_ft=_mean_goals(ft),
        expected_ht=_mean_goals(ht),
        rates=rates,
    )


def estimate_eta(
    home_goals: np.ndarray,
    away_goals: np.ndarray,
    lam_home: np.ndarray,
    lam_away: np.ndarray,
    states: np.ndarray,
    rho: float,
    *,
    scale: float = 1.0,
    pinned: float | None = None,
) -> float:
    """One league-level state effect, fit only on the rows passed in.

    A Gaussian prior centred at 0 keeps a short sample at independence.
    ``scale`` multiplies the estimate (0 removes state dependence).
    ``pinned`` skips the grid and returns that value times ``scale``.
    """

    if pinned is not None:
        return float(np.clip(pinned * scale, -0.6, 0.8))
    if len(home_goals) < 8:
        return 0.0
    best_eta = 0.0
    best_score = -np.inf
    precision = 1.0 / (_ETA_PRIOR_SD**2)
    for eta in _ETA_GRID:
        score = _eta_loglik(home_goals, away_goals, lam_home, lam_away, states, rho, float(eta))
        score -= 0.5 * precision * float(eta) ** 2
        if score > best_score:
            best_score = score
            best_eta = float(eta)
    return float(np.clip(best_eta * scale, -0.6, 0.8))


def estimate_xg_factors(matches: pd.DataFrame, as_of: pd.Timestamp, xi: float) -> tuple[dict[int, float], dict[int, float]]:
    """Time-decayed log xG attack and defence, centred on the league mean.

    The current match's xG is not included: only rows with ``starting_at``
    strictly before ``as_of``. Teams with no xG get 0, which leaves rates unchanged.
    """

    if matches.empty or "home_xg" not in matches.columns or "away_xg" not in matches.columns:
        return {}, {}
    as_of = pd.Timestamp(as_of)
    if as_of.tzinfo is None:
        as_of = as_of.tz_localize("UTC")
    kickoff = pd.to_datetime(matches["starting_at"], utc=True)
    frame = matches.loc[kickoff < as_of].copy()
    frame["starting_at"] = kickoff.loc[frame.index]
    frame = frame.dropna(subset=["home_xg", "away_xg", "home_team_id", "away_team_id"])
    if frame.empty:
        return {}, {}
    days = (as_of - frame["starting_at"]).dt.total_seconds().to_numpy(dtype=float) / 86400.0
    weights = np.exp(-float(xi) * np.clip(days, 0.0, None))
    for_sum: dict[int, float] = {}
    for_w: dict[int, float] = {}
    against_sum: dict[int, float] = {}
    against_w: dict[int, float] = {}
    for weight, row in zip(weights, frame.itertuples(index=False), strict=True):
        home_id = int(row.home_team_id)
        away_id = int(row.away_team_id)
        home_xg = float(row.home_xg)
        away_xg = float(row.away_xg)
        if home_xg < 0 or away_xg < 0 or not math.isfinite(home_xg) or not math.isfinite(away_xg):
            continue
        _add(for_sum, for_w, home_id, home_xg, weight)
        _add(for_sum, for_w, away_id, away_xg, weight)
        _add(against_sum, against_w, home_id, away_xg, weight)
        _add(against_sum, against_w, away_id, home_xg, weight)
    if not for_w:
        return {}, {}
    league_for = sum(for_sum.values()) / sum(for_w.values())
    league_against = sum(against_sum.values()) / max(sum(against_w.values()), 1e-9)
    attack: dict[int, float] = {}
    defence: dict[int, float] = {}
    for team_id, weight in for_w.items():
        if weight <= 0 or team_id not in against_w:
            continue
        attack[team_id] = math.log((for_sum[team_id] / weight) / max(league_for, 1e-6))
        defence[team_id] = math.log((against_sum[team_id] / against_w[team_id]) / max(league_against, 1e-6))
    return attack, defence


def apply_xg(lam: float, attack: float, defence: float, weight: float) -> float:
    """Geometric blend of a Dixon–Coles rate with an xG shift. Weight 0 keeps ``lam``."""

    if weight <= 0.0 or not math.isfinite(weight):
        return lam
    return float(lam * math.exp(float(weight) * (attack - defence)))


def second_half_frame(matches: pd.DataFrame) -> pd.DataFrame:
    """Finished matches whose half-time goals do not exceed the full-time score."""

    needed = ["home_goals_ft", "away_goals_ft", "home_goals_ht", "away_goals_ht"]
    frame = matches.dropna(subset=needed).copy()
    if frame.empty:
        return frame
    sh_home = frame["home_goals_ft"].astype(int) - frame["home_goals_ht"].astype(int)
    sh_away = frame["away_goals_ft"].astype(int) - frame["away_goals_ht"].astype(int)
    keep = (sh_home >= 0) & (sh_away >= 0)
    frame = frame.loc[keep].copy()
    frame["home_goals_sh"] = sh_home.loc[keep].astype(int)
    frame["away_goals_sh"] = sh_away.loc[keep].astype(int)
    return frame


def _mean_goals(matrix: np.ndarray) -> tuple[float, float]:
    home_goals = np.arange(matrix.shape[0], dtype=float)
    away_goals = np.arange(matrix.shape[1], dtype=float)
    return float((matrix.sum(axis=1) * home_goals).sum()), float((matrix.sum(axis=0) * away_goals).sum())


def _eta_loglik(
    home_goals: np.ndarray,
    away_goals: np.ndarray,
    lam_home: np.ndarray,
    lam_away: np.ndarray,
    states: np.ndarray,
    rho: float,
    eta: float,
) -> float:
    total = 0.0
    for goals_h, goals_a, lam_h, lam_a, state in zip(home_goals, away_goals, lam_home, lam_away, states, strict=True):
        adj_h, adj_a = adjust_second_half(float(lam_h), float(lam_a), int(state), eta)
        tau = float(dixon_coles_tau(int(goals_h), int(goals_a), adj_h, adj_a, rho))
        if tau <= 0 or adj_h <= 0 or adj_a <= 0:
            total -= 20.0
            continue
        total += math.log(tau) + _poisson_logpmf_scalar(int(goals_h), adj_h) + _poisson_logpmf_scalar(int(goals_a), adj_a)
    return total


def _poisson_logpmf_scalar(goals: int, lam: float) -> float:
    if lam <= 0:
        return 0.0 if goals == 0 else -1e6
    return -lam + goals * math.log(lam) - math.lgamma(goals + 1)


def _nb_logpmf(goals: np.ndarray, mean: np.ndarray, dispersion: float) -> np.ndarray:
    """Negative binomial with mean ``mean`` and variance ``mean * dispersion``."""

    phi = max(float(dispersion), 1.001)
    # p = 1/phi, n = mean / (phi - 1), mean = n (1-p) / p.
    prob = 1.0 / phi
    size = mean / (phi - 1.0)
    # scipy nbinom pmf is for the number of failures before ``size`` successes.
    return (
        gammaln(goals + size)
        - gammaln(size)
        - gammaln(goals + 1)
        + size * np.log(prob)
        + goals * np.log(1.0 - prob)
    )


def _add(total: dict[int, float], weight_total: dict[int, float], team_id: int, value: float, weight: float) -> None:
    total[team_id] = total.get(team_id, 0.0) + weight * value
    weight_total[team_id] = weight_total.get(team_id, 0.0) + weight
