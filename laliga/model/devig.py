"""Remove the overround from a 1X2 price.

The proportional (multiplicative) method divides the raw implied probabilities
by their sum. Power, Shin, additive, and odds-ratio use the same raw implied
probabilities, so they need the overround. A price that was already normalised
makes every method return that same vector.

Shin follows the fixed-point iteration in Hyun Song Shin's model, as
implemented in the MIT-licensed ``shin`` package (the pure-Python optimiser,
not the Rust extension), so it runs on Windows CPython 3.13 without a
compiled wheel. Power and odds-ratio follow the formulas in the
``implied`` R package (Buchdahl / Cheung): power solves ``sum(q ** k) = 1``
with ``k > 1`` when the book is overround, which raises the favourite
relative to the proportional method.
"""

from __future__ import annotations

import math

import numpy as np

DEVIG_METHODS = ("proportional", "power", "shin", "additive", "odds_ratio")


def raw_implied_from_decimal(odds: np.ndarray) -> np.ndarray:
    """Elementwise ``1 / decimal``. Odds must be strictly above 1."""

    prices = np.asarray(odds, dtype=float)
    if prices.ndim == 1:
        prices = prices.reshape(1, -1)
    return 1.0 / prices


def devig(raw_implied: np.ndarray) -> dict[str, np.ndarray]:
    """Return one ``(n, 3)`` probability matrix per method.

    A row that a method cannot price is NaN. ``raw_implied`` is the
    bookmaker's ``1/odds`` (or the average of those), including the overround.
    """

    raw = np.asarray(raw_implied, dtype=float)
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)
    if raw.shape[1] != 3:
        raise ValueError("去水位只用于三项胜平负。")
    valid = np.isfinite(raw).all(axis=1) & (raw > 0).all(axis=1) & (raw < 1).all(axis=1)
    proportional = np.full(raw.shape, np.nan, dtype=float)
    if valid.any():
        totals = raw[valid].sum(axis=1, keepdims=True)
        proportional[valid] = raw[valid] / totals
    return {
        "proportional": proportional,
        "power": _power(raw, valid),
        "shin": _shin(raw, valid),
        "additive": _additive(raw, valid),
        "odds_ratio": _odds_ratio(raw, valid),
    }


def _power(raw: np.ndarray, valid: np.ndarray) -> np.ndarray:
    out = np.full(raw.shape, np.nan, dtype=float)
    if not valid.any():
        return out
    chosen = raw[valid]
    low = np.full(len(chosen), 0.05)
    high = np.full(len(chosen), 12.0)
    for _ in range(50):
        mid = (low + high) / 2.0
        total = np.exp(mid[:, None] * np.log(chosen)).sum(axis=1)
        bigger = total > 1.0
        low = np.where(bigger, mid, low)
        high = np.where(bigger, high, mid)
    exponent = (low + high) / 2.0
    probs = np.exp(exponent[:, None] * np.log(chosen))
    out[valid] = probs / probs.sum(axis=1, keepdims=True)
    return out


def _shin(raw: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Shin fixed point for three outcomes. ``z`` is the insider proportion.

    A row that goes outside the iteration is left as NaN. Other rows in the
    same batch still get a price.
    """

    out = np.full(raw.shape, np.nan, dtype=float)
    if not valid.any():
        return out
    chosen = raw[valid]
    total = chosen.sum(axis=1, keepdims=True)
    z = np.zeros(len(chosen), dtype=float)
    alive = np.ones(len(chosen), dtype=bool)
    for _ in range(1000):
        previous = z.copy()
        inside = z[:, None] ** 2 + 4.0 * (1.0 - z[:, None]) * chosen**2 / total
        alive &= np.isfinite(inside).all(axis=1) & (inside >= 0).all(axis=1)
        if not alive.any():
            return out
        updated = np.sqrt(np.clip(inside, 0.0, None)).sum(axis=1) - 2.0
        z = np.where(alive, updated, z)
        if float(np.max(np.abs(z[alive] - previous[alive]))) < 1e-12:
            break
    alive &= np.isfinite(z) & (z > -0.5) & (z < 0.99)
    inside = z[:, None] ** 2 + 4.0 * (1.0 - z[:, None]) * chosen**2 / total
    denom = 2.0 * (1.0 - z[:, None])
    probs = (np.sqrt(np.clip(inside, 0.0, None)) - z[:, None]) / denom
    good = alive & np.isfinite(probs).all(axis=1) & (probs > 0).all(axis=1)
    if good.any():
        probs[good] = probs[good] / probs[good].sum(axis=1, keepdims=True)
    written = np.full(chosen.shape, np.nan, dtype=float)
    written[good] = probs[good]
    out[valid] = written
    return out


def _additive(raw: np.ndarray, valid: np.ndarray) -> np.ndarray:
    out = np.full(raw.shape, np.nan, dtype=float)
    if not valid.any():
        return out
    chosen = raw[valid]
    adjusted = chosen - (chosen.sum(axis=1, keepdims=True) - 1.0) / chosen.shape[1]
    good = (adjusted > 0).all(axis=1) & np.isfinite(adjusted).all(axis=1)
    written = np.full(chosen.shape, np.nan, dtype=float)
    written[good] = adjusted[good]
    out[valid] = written
    return out


def _odds_ratio(raw: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """``p = q / (c + q - c q)``, with ``c`` chosen so the three probabilities sum to 1."""

    out = np.full(raw.shape, np.nan, dtype=float)
    if not valid.any():
        return out
    chosen = raw[valid]
    low = np.full(len(chosen), 0.05)
    high = np.full(len(chosen), 8.0)
    for _ in range(50):
        mid = (low + high) / 2.0
        total = _odds_ratio_probs(chosen, mid).sum(axis=1)
        bigger = total > 1.0
        low = np.where(bigger, mid, low)
        high = np.where(bigger, high, mid)
    probs = _odds_ratio_probs(chosen, (low + high) / 2.0)
    good = np.isfinite(probs).all(axis=1) & (probs > 0).all(axis=1)
    probs = probs / probs.sum(axis=1, keepdims=True)
    written = np.full(chosen.shape, np.nan, dtype=float)
    written[good] = probs[good]
    out[valid] = written
    return out


def _odds_ratio_probs(raw: np.ndarray, coefficient: np.ndarray) -> np.ndarray:
    scale = coefficient[:, None]
    return raw / (scale + raw - scale * raw)


def overround(raw_implied: np.ndarray) -> float:
    raw = np.asarray(raw_implied, dtype=float)
    if not np.isfinite(raw).all() or raw.ndim != 1:
        return math.nan
    return float(raw.sum())
