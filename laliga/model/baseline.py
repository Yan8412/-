"""Historical-frequency baseline.

Every fixture receives the same probabilities: the training-set frequencies
of home/draw/away (full-time and half-time), with one pseudo-count per
outcome. The three most common exact full-time scores are the baseline
scorelines. There is no team information and no use of future matches,
because the caller passes only the training window.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from laliga.model.dixon_coles import FitError


@dataclass
class FrequencyBaseline:
    ft_probs: tuple[float, float, float]
    ht_probs: tuple[float, float, float]
    top_scores: list[tuple[int, int, float]]
    n_matches: int
    n_ht_matches: int

    def to_dict(self) -> dict:
        return {
            "ft_probs": list(self.ft_probs),
            "ht_probs": list(self.ht_probs),
            "top_scores": [
                {"home_goals": home, "away_goals": away, "probability": probability}
                for home, away, probability in self.top_scores
            ],
            "n_matches": self.n_matches,
            "n_ht_matches": self.n_ht_matches,
        }


def fit_baseline(matches: pd.DataFrame) -> FrequencyBaseline:
    finished = matches.dropna(subset=["home_goals_ft", "away_goals_ft"])
    if finished.empty:
        raise FitError("基准模型没有完场比赛。")
    ft = _laplace_outcomes(finished["home_goals_ft"].to_numpy(), finished["away_goals_ft"].to_numpy())
    ht_rows = matches.dropna(subset=["home_goals_ht", "away_goals_ht"])
    if ht_rows.empty:
        ht = ft
    else:
        ht = _laplace_outcomes(ht_rows["home_goals_ht"].to_numpy(), ht_rows["away_goals_ht"].to_numpy())
    return FrequencyBaseline(
        ft_probs=ft,
        ht_probs=ht,
        top_scores=_top_scores(finished),
        n_matches=int(len(finished)),
        n_ht_matches=int(len(ht_rows)),
    )


def _laplace_outcomes(home_goals, away_goals) -> tuple[float, float, float]:
    home = int((home_goals > away_goals).sum())
    draw = int((home_goals == away_goals).sum())
    away = int((home_goals < away_goals).sum())
    total = home + draw + away + 3
    return ((home + 1) / total, (draw + 1) / total, (away + 1) / total)


def _top_scores(matches: pd.DataFrame, k: int = 3) -> list[tuple[int, int, float]]:
    counts: dict[tuple[int, int], int] = {}
    for home, away in zip(matches["home_goals_ft"].astype(int), matches["away_goals_ft"].astype(int), strict=True):
        key = (int(home), int(away))
        counts[key] = counts.get(key, 0) + 1
    total = sum(counts.values())
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0][0] + item[0][1], item[0][0], item[0][1]))
    return [(home, away, count / total) for (home, away), count in ordered[:k]]
