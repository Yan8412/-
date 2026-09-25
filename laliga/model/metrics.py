"""Proper scoring rules for 1X2 markets and a top-3 scoreline hit rate.

Log loss is the mean negative natural log of the probability on the actual
outcome. The multiclass Brier score is the mean, over matches, of the sum of
squared errors across the three outcomes (range 0 to 2). Accuracy is how
often the highest-probability outcome matches the result. The top-3 hit rate
is how often the actual full-time score is one of the three predicted
scorelines.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class MetricBlock:
    n: int
    n_ht: int
    ft_log_loss: float
    ft_brier: float
    ft_accuracy: float
    ht_log_loss: float | None
    ht_brier: float | None
    ht_accuracy: float | None
    top3_hit_rate: float

    def to_dict(self) -> dict:
        return {
            "n": self.n,
            "n_ht": self.n_ht,
            "ft_log_loss": self.ft_log_loss,
            "ft_brier": self.ft_brier,
            "ft_accuracy": self.ft_accuracy,
            "ht_log_loss": self.ht_log_loss,
            "ht_brier": self.ht_brier,
            "ht_accuracy": self.ht_accuracy,
            "top3_hit_rate": self.top3_hit_rate,
        }


def outcome_index(home_goals: int, away_goals: int) -> int:
    if home_goals > away_goals:
        return 0
    if home_goals == away_goals:
        return 1
    return 2


def multiclass_log_loss(y: np.ndarray, probabilities: np.ndarray) -> float:
    clipped = np.clip(probabilities, 1e-15, 1.0)
    clipped = clipped / clipped.sum(axis=1, keepdims=True)
    chosen = clipped[np.arange(len(y)), y]
    return float(-np.mean(np.log(chosen)))


def multiclass_brier(y: np.ndarray, probabilities: np.ndarray) -> float:
    one_hot = np.zeros_like(probabilities, dtype=float)
    one_hot[np.arange(len(y)), y] = 1.0
    return float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))


def accuracy(y: np.ndarray, probabilities: np.ndarray) -> float:
    return float(np.mean(np.argmax(probabilities, axis=1) == y))


def top3_hit_rate(actual: list[tuple[int, int]], predicted: list[list[tuple[int, int]]]) -> float:
    if not actual:
        return float("nan")
    hits = sum(score in guesses for score, guesses in zip(actual, predicted, strict=True))
    return hits / len(actual)


def summarize_rows(rows: list[dict], prefix: str) -> MetricBlock:
    """Aggregate row dicts that store ``{prefix}_ft_probs`` and related keys."""

    if not rows:
        raise ValueError("没有可汇总的比赛。")
    y_ft = np.array([row["y_ft"] for row in rows], dtype=int)
    p_ft = np.array([row[f"{prefix}_ft_probs"] for row in rows], dtype=float)
    ht_rows = [row for row in rows if row.get("y_ht") is not None]
    if ht_rows:
        y_ht = np.array([row["y_ht"] for row in ht_rows], dtype=int)
        p_ht = np.array([row[f"{prefix}_ht_probs"] for row in ht_rows], dtype=float)
        ht_log_loss = multiclass_log_loss(y_ht, p_ht)
        ht_brier = multiclass_brier(y_ht, p_ht)
        ht_accuracy = accuracy(y_ht, p_ht)
    else:
        ht_log_loss = ht_brier = ht_accuracy = None
    actual_scores = [row["score"] for row in rows]
    predicted_scores = [row[f"{prefix}_top3"] for row in rows]
    return MetricBlock(
        n=len(rows),
        n_ht=len(ht_rows),
        ft_log_loss=multiclass_log_loss(y_ft, p_ft),
        ft_brier=multiclass_brier(y_ft, p_ft),
        ft_accuracy=accuracy(y_ft, p_ft),
        ht_log_loss=ht_log_loss,
        ht_brier=ht_brier,
        ht_accuracy=ht_accuracy,
        top3_hit_rate=top3_hit_rate(actual_scores, predicted_scores),
    )


def by_season(rows: list[dict], prefix: str) -> dict[str, MetricBlock]:
    groups: dict[str, list[dict]] = {}
    for row in rows:
        key = str(row.get("season_name") or row.get("season_id") or "未知赛季")
        groups.setdefault(key, []).append(row)
    return {key: summarize_rows(group, prefix) for key, group in sorted(groups.items()) if group}


def iter_evaluation_folds(finished: pd.DataFrame, min_train_matches: int):
    """Yield ``(day, train, test)`` with every training kickoff before the test day.

    All matches on the same UTC date are predicted together. None of them, and
    nothing later, is in the training set.
    """

    if finished.empty:
        return
    ordered = finished.sort_values(["starting_at", "fixture_id"]).copy()
    ordered["match_day"] = ordered["starting_at"].dt.floor("D")
    for day in ordered["match_day"].drop_duplicates().tolist():
        train = ordered[ordered["starting_at"] < day]
        test = ordered[ordered["match_day"] == day]
        if len(train) < min_train_matches or test.empty:
            continue
        if train["starting_at"].max() >= test["starting_at"].min():
            raise AssertionError("训练样本包含了测试日或更晚的比赛。")
        yield day, train, test
