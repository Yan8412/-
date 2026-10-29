"""Pick the active track per market from walk-forward backtest rows.

Only matches where all three tracks produced a forecast are compared, so
every track is scored on the same games. Scores per market:

* ``ft`` / ``ht``: mean log loss of the 1X2 probabilities (lower is better).
  Brier and hit rate are reported next to it but do not decide.
* ``scores``: top-3 scoreline hit rate (higher is better).

A challenger replaces the incumbent only when (1) there are at least
``min_matches`` common matches, (2) it has the best mean score, and (3) the
paired per-match improvement over the incumbent is at least ``min_z``
standard errors above zero. Otherwise the incumbent stays. With no previous
decision the incumbent is football, the original Dixon–Coles predictor.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

import numpy as np

from laliga.config import PROMOTION_MARKET_LABELS, PROMOTION_MARKETS, TRACK_LABELS, TRACKS, PromotionConfig

TRACK_PREFIX = {"football": "model", "market": "market", "combined": "combined"}
HISTORY_LIMIT = 30


def common_rows(rows: list[dict]) -> list[dict]:
    return [row for row in rows if all(row.get(f"{TRACK_PREFIX[track]}_ft_probs") is not None for track in TRACKS)]


def per_match_losses(rows: list[dict], market: str) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
    """Loss arrays per track (lower is better) plus reported metrics."""

    losses: dict[str, np.ndarray] = {}
    metrics: dict[str, dict] = {}
    if market == "scores":
        for track in TRACKS:
            hits = np.array([row["score"] in row[f"{TRACK_PREFIX[track]}_top3"] for row in rows], dtype=float)
            losses[track] = 1.0 - hits
            metrics[track] = {"top3_hit_rate": float(hits.mean()) if len(hits) else None}
        return losses, metrics
    key = "y_ft" if market == "ft" else "y_ht"
    usable = [row for row in rows if row.get(key) is not None]
    for track in TRACKS:
        probabilities = np.array([row[f"{TRACK_PREFIX[track]}_{market}_probs"] for row in usable], dtype=float).reshape(-1, 3)
        y = np.array([row[key] for row in usable], dtype=int)
        if len(y) == 0:
            losses[track] = np.zeros(0)
            metrics[track] = {"log_loss": None, "brier": None, "accuracy": None}
            continue
        chosen = np.clip(probabilities[np.arange(len(y)), y], 1e-15, 1.0)
        one_hot = np.zeros_like(probabilities)
        one_hot[np.arange(len(y)), y] = 1.0
        losses[track] = -np.log(chosen)
        metrics[track] = {
            "log_loss": float(losses[track].mean()),
            "brier": float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))),
            "accuracy": float(np.mean(np.argmax(probabilities, axis=1) == y)),
        }
    return losses, metrics


def decide(rows: list[dict], incumbent: dict[str, str] | None, config: PromotionConfig) -> dict:
    incumbent = {market: (incumbent or {}).get(market, "football") for market in PROMOTION_MARKETS}
    shared = common_rows(rows)
    decisions = {}
    for market in PROMOTION_MARKETS:
        losses, metrics = per_match_losses(shared, market)
        decisions[market] = _decide_market(market, losses, metrics, incumbent[market], config)
    return {
        "decided_at": datetime.now(timezone.utc).isoformat(),
        "n_evaluated": len(rows),
        "n_common": len(shared),
        "rule": {
            "min_matches": config.min_matches,
            "min_z": config.min_z,
            "primary": {"ft": "ft_log_loss", "ht": "ht_log_loss", "scores": "top3_hit_rate"},
        },
        "markets": decisions,
    }


def _decide_market(market: str, losses: dict[str, np.ndarray], metrics: dict[str, dict], current: str, config: PromotionConfig) -> dict:
    n = int(len(losses[current]))
    label = PROMOTION_MARKET_LABELS[market]
    result = {
        "track": current,
        "previous": current,
        "changed": False,
        "n": n,
        "metrics": metrics,
        "best": None,
        "z": None,
        "reason": "",
    }
    if n < config.min_matches:
        result["reason"] = (
            f"{label}：三套模型都有预测的回测比赛只有 {n} 场，少于 {config.min_matches} 场，"
            f"继续使用{TRACK_LABELS[current]}。"
        )
        return result
    means = {track: float(values.mean()) for track, values in losses.items()}
    best = min(TRACKS, key=lambda track: (means[track], track != current))
    result["best"] = best
    metric_text = _metric_text(market, metrics)
    if best == current:
        result["reason"] = f"{label}：{TRACK_LABELS[current]}在 {n} 场共同比赛上仍然最好（{metric_text}），不变。"
        return result
    difference = losses[current] - losses[best]
    spread = float(difference.std(ddof=1)) if n > 1 else 0.0
    mean_gain = float(difference.mean())
    z = math.inf if spread == 0 and mean_gain > 0 else (mean_gain / (spread / math.sqrt(n)) if spread > 0 else 0.0)
    result["z"] = None if math.isinf(z) else z
    if mean_gain > 0 and z >= config.min_z:
        result["track"] = best
        result["changed"] = True
        result["reason"] = (
            f"{label}：{TRACK_LABELS[best]}在 {n} 场共同比赛上优于{TRACK_LABELS[current]}"
            f"（{metric_text}；逐场配对改进 {_z_text(z)} 个标准误，门槛 {config.min_z}），升级为{TRACK_LABELS[best]}。"
        )
    else:
        result["reason"] = (
            f"{label}：{TRACK_LABELS[best]}平均更好（{metric_text}），但逐场配对改进只有 {_z_text(z)} 个标准误，"
            f"未达门槛 {config.min_z}，继续使用{TRACK_LABELS[current]}。"
        )
    return result


def apply(store, decision: dict) -> dict:
    """Persist the decision as the active configuration and append it to the history."""

    previous = store.load_json(store.active_path) or {}
    history = list(previous.get("history") or [])
    history.append(
        {
            "decided_at": decision["decided_at"],
            "n_common": decision["n_common"],
            "tracks": {market: body["track"] for market, body in decision["markets"].items()},
            "changed": [market for market, body in decision["markets"].items() if body["changed"]],
        }
    )
    document = {
        "version": 1,
        "active": {market: body["track"] for market, body in decision["markets"].items()},
        "decision": decision,
        "history": history[-HISTORY_LIMIT:],
    }
    store.save_json(store.active_path, document)
    return document


def load_active(store) -> dict[str, str]:
    document = store.load_json(store.active_path) or {}
    active = document.get("active") or {}
    return {market: active.get(market, "football") if active.get(market) in TRACKS else "football" for market in PROMOTION_MARKETS}


def _metric_text(market: str, metrics: dict[str, dict]) -> str:
    parts = []
    for track in TRACKS:
        body = metrics[track]
        if market == "scores":
            value = body.get("top3_hit_rate")
            parts.append(f"{TRACK_LABELS[track]} 前三命中 {value * 100:.1f}%" if value is not None else f"{TRACK_LABELS[track]} —")
        else:
            value = body.get("log_loss")
            parts.append(f"{TRACK_LABELS[track]} 对数损失 {value:.4f}" if value is not None else f"{TRACK_LABELS[track]} —")
    return "，".join(parts)


def _z_text(z: float) -> str:
    return "∞" if math.isinf(z) else f"{z:.2f}"
