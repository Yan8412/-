"""Combined track: a logarithmic opinion pool of the football and market tracks.

For each outcome k,

    p_combined(k) ∝ p_football(k) ** a × p_market(k) ** b,    a, b ≥ 0

The weights are fitted by minimising the mean log loss of the pooled
probabilities on matches that both tracks forecast *before* they were played
(walk-forward rows from the backtest), so the pool is scored with the same
proper scoring rule it was trained on. ``a + b`` is free: a sum below 1 widens
the pooled distribution, above 1 sharpens it, which also corrects joint
over- or under-confidence. Full-time and half-time 1X2 get their own weights.
The exact-score grid is pooled cell by cell with the full-time weights.

With fewer fitting rows than the configured minimum, equal weights
(0.5, 0.5) are used and marked as not fitted.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

_FLOOR = 1e-12
_BOUNDS = [(0.0, 4.0), (0.0, 4.0)]


@dataclass
class PoolWeights:
    football: float = 0.5
    market: float = 0.5
    n: int = 0
    fitted: bool = False
    log_loss: float | None = None

    def to_dict(self) -> dict:
        return {
            "football": self.football,
            "market": self.market,
            "n": self.n,
            "fitted": self.fitted,
            "log_loss": self.log_loss,
        }

    @classmethod
    def from_dict(cls, payload: dict | None) -> "PoolWeights":
        if not payload:
            return cls()
        return cls(
            football=float(payload.get("football", 0.5)),
            market=float(payload.get("market", 0.5)),
            n=int(payload.get("n") or 0),
            fitted=bool(payload.get("fitted")),
            log_loss=None if payload.get("log_loss") is None else float(payload["log_loss"]),
        )


@dataclass
class EnsembleWeights:
    ft: PoolWeights
    ht: PoolWeights
    method: str = "log_pool"

    @classmethod
    def default(cls) -> "EnsembleWeights":
        return cls(ft=PoolWeights(), ht=PoolWeights())

    def to_dict(self) -> dict:
        return {"method": self.method, "ft": self.ft.to_dict(), "ht": self.ht.to_dict()}

    @classmethod
    def from_dict(cls, payload: dict | None) -> "EnsembleWeights":
        if not payload:
            return cls.default()
        return cls(ft=PoolWeights.from_dict(payload.get("ft")), ht=PoolWeights.from_dict(payload.get("ht")))


def log_pool(football, market, weights: PoolWeights) -> np.ndarray:
    """Pool along the last axis: one 1X2 vector or a batch of them."""

    f = np.clip(np.asarray(football, dtype=float), _FLOOR, None)
    m = np.clip(np.asarray(market, dtype=float), _FLOOR, None)
    if f.shape != m.shape:
        raise ValueError("两套概率的形状不一致。")
    logits = weights.football * np.log(f) + weights.market * np.log(m)
    logits = logits - logits.max(axis=-1, keepdims=True)
    pooled = np.exp(logits)
    return pooled / pooled.sum(axis=-1, keepdims=True)


def pool_grid(football: np.ndarray, market: np.ndarray, weights: PoolWeights) -> np.ndarray:
    """Pool two exact-score grids cell by cell and renormalise the whole grid."""

    f = np.clip(np.asarray(football, dtype=float), _FLOOR, None)
    m = np.clip(np.asarray(market, dtype=float), _FLOOR, None)
    logits = weights.football * np.log(f) + weights.market * np.log(m)
    pooled = np.exp(logits - logits.max())
    return pooled / pooled.sum()


def fit_pool(football, market, outcomes, *, min_rows: int) -> PoolWeights:
    """Fit (a, b) by minimum mean log loss. Falls back to equal weights on thin data."""

    f = np.clip(np.asarray(football, dtype=float), _FLOOR, None)
    m = np.clip(np.asarray(market, dtype=float), _FLOOR, None)
    y = np.asarray(outcomes, dtype=int)
    n = int(len(y))
    if n < max(min_rows, 1):
        return PoolWeights(n=n, fitted=False)
    log_f = np.log(f)
    log_m = np.log(m)
    result = minimize(
        lambda theta: _loss(theta, log_f, log_m, y),
        np.array([0.5, 0.5]),
        jac=True,
        method="L-BFGS-B",
        bounds=_BOUNDS,
    )
    theta = np.asarray(result.x, dtype=float)
    return PoolWeights(football=float(theta[0]), market=float(theta[1]), n=n, fitted=True, log_loss=float(result.fun))


def _loss(theta: np.ndarray, log_f: np.ndarray, log_m: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
    logits = theta[0] * log_f + theta[1] * log_m
    logits = logits - logits.max(axis=1, keepdims=True)
    log_norm = np.log(np.exp(logits).sum(axis=1))
    rows = np.arange(len(y))
    log_p = logits[rows, y] - log_norm
    pooled = np.exp(logits - log_norm[:, None])
    grad_a = -(log_f[rows, y] - (pooled * log_f).sum(axis=1))
    grad_b = -(log_m[rows, y] - (pooled * log_m).sum(axis=1))
    return float(-log_p.mean()), np.array([grad_a.mean(), grad_b.mean()])
