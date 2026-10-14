"""Score fixtures, publish the prediction files, and archive what was published.

The CLI and the dashboard both go through here so the daily script, the
"预测下一轮" button, and the archive see the same numbers.

``latest_next.json`` (read by the separate notifier) is a JSON list of
``Prediction.to_dict()``. Its keys must not change. The dashboard's
``latest.json`` is ``{created_at, label, note, items}`` where each item is the
same dict plus the two score matrices.
"""

from __future__ import annotations

import csv
import io
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from laliga.archive import AppendResult, PredictionArchive, model_version
from laliga.config import ModelConfig
from laliga.data.store import MatchStore
from laliga.model.dixon_coles import FitError
from laliga.model.service import GoalMultipliers, Prediction, ScorelineModel, fit_models, model_is_fresh

logger = logging.getLogger(__name__)

Adjuster = Callable[[pd.Series], tuple[GoalMultipliers, dict] | None]

LATEST_NEXT_KEYS = (
    "fixture_id",
    "starting_at",
    "season_id",
    "season_name",
    "round_name",
    "home_team_id",
    "home_team",
    "away_team_id",
    "away_team",
    "ft",
    "ht",
    "top_scores",
    "expected_goals_ft",
    "expected_goals_ht",
)


@dataclass
class ScoredFixture:
    prediction: Prediction
    ft_matrix: np.ndarray
    ht_matrix: np.ndarray
    availability: dict | None = None


def load_saved_model(store: MatchStore) -> ScorelineModel | None:
    document = store.load_model_document()
    if not document:
        return None
    try:
        return ScorelineModel.from_dict(document)
    except (KeyError, TypeError, ValueError, FitError) as exc:
        logger.warning("已保存的模型无法读取，将重新拟合：%s", exc)
        return None


def score_fixtures(
    history: pd.DataFrame,
    fixtures: pd.DataFrame,
    config: ModelConfig,
    store: MatchStore,
    *,
    refit: bool = False,
    save_refit: bool = False,
    adjuster: Adjuster | None = None,
    notify: Callable[[str], None] | None = None,
) -> list[ScoredFixture]:
    """Use the saved fit when it is current, otherwise refit per kickoff day on earlier results."""

    say = notify or logger.info
    ordered = fixtures.sort_values(["starting_at", "fixture_id"])
    eligible = history[~history["fixture_id"].isin({int(value) for value in ordered["fixture_id"]})]
    saved = None if refit else load_saved_model(store)
    if saved is not None and model_is_fresh(saved, eligible, ordered):
        say("使用已保存的模型（训练截止日期不早于最近完场、且早于这些比赛）。")
        return [_score(saved, row, adjuster) for _, row in ordered.iterrows()]

    say("按每场比赛开球日前的完场数据重新拟合。")
    scored: list[ScoredFixture] = []
    previous: ScorelineModel | None = None
    for day, group in ordered.groupby(ordered["starting_at"].dt.floor("D"), sort=True):
        model = fit_models(
            eligible,
            pd.Timestamp(day),
            config,
            initial_ft=None if previous is None else previous.ft,
            initial_ht=None if previous is None else previous.ht,
        )
        previous = model
        scored.extend(_score(model, row, adjuster) for _, row in group.iterrows())
    if save_refit and previous is not None:
        store.save_model_document(previous.to_dict())
    return scored


def _score(model: ScorelineModel, row: pd.Series, adjuster: Adjuster | None) -> ScoredFixture:
    multipliers = None
    info = None
    if adjuster is not None:
        adjusted = adjuster(row)
        if adjusted is not None:
            multipliers, info = adjusted
    home_id = int(row["home_team_id"])
    away_id = int(row["away_team_id"])
    ft_matrix, ht_matrix = model.matrices(home_id, away_id, multipliers)
    return ScoredFixture(
        prediction=model.predict_row(row, multipliers),
        ft_matrix=ft_matrix,
        ht_matrix=ht_matrix,
        availability=info,
    )


def latest_next_payload(predictions: list[Prediction]) -> list[dict]:
    return [item.to_dict() for item in predictions]


def dashboard_items(scored: list[ScoredFixture]) -> list[dict]:
    items = []
    for item in scored:
        document = item.prediction.to_dict()
        document["ft_matrix"] = item.ft_matrix.round(6).tolist()
        document["ht_matrix"] = item.ht_matrix.round(6).tolist()
        if item.availability is not None:
            document["availability"] = item.availability
        items.append(document)
    return items


def atomic_write_text(path: Path, text: str) -> None:
    """Write next to the target, then rename, so a reader never sees half a file."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def write_latest_next(path: Path, predictions: list[Prediction]) -> None:
    atomic_write_text(path, json.dumps(latest_next_payload(predictions), ensure_ascii=False, indent=2))


def write_dashboard_latest(store: MatchStore, items: list[dict], *, label: str, note: str, created_at: str | None = None) -> None:
    payload = {
        "created_at": created_at or datetime.now(timezone.utc).isoformat(),
        "label": label,
        "note": note,
        "items": items,
    }
    atomic_write_text(store.predictions_dir / "latest.json", json.dumps(payload, ensure_ascii=False, indent=2))
    atomic_write_text(store.predictions_dir / "latest.csv", dashboard_csv(items))


def archive_scored(
    store: MatchStore,
    scored: list[ScoredFixture],
    history: pd.DataFrame,
    *,
    origin: str,
    generated_at=None,
) -> AppendResult:
    used = any(item.availability is not None and item.availability.get("used") for item in scored)
    extras = {
        item.prediction.fixture_id: {"availability": item.availability}
        for item in scored
        if item.availability is not None
    }
    return PredictionArchive(store.root).append(
        [item.prediction for item in scored],
        model_version=model_version(availability=used),
        origin=origin,
        generated_at=generated_at,
        matches=history,
        extras=extras,
    )


def split_kickoff(value: str) -> tuple[str, str]:
    """Return (UTC, Asia/Singapore) clock times as YYYY-MM-DD HH:MM."""

    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")
    else:
        stamp = stamp.tz_convert("UTC")
    singapore = stamp.tz_convert("Asia/Singapore")
    return stamp.strftime("%Y-%m-%d %H:%M"), singapore.strftime("%Y-%m-%d %H:%M")


def dashboard_csv(items: list[dict]) -> str:
    buffer = io.StringIO()
    fieldnames = [
        "fixture_id",
        "kickoff_utc",
        "kickoff_singapore",
        "home_team",
        "away_team",
        "ft_home",
        "ft_draw",
        "ft_away",
        "ht_home",
        "ht_draw",
        "ht_away",
        "score_1",
        "score_1_prob",
        "score_2",
        "score_2_prob",
        "score_3",
        "score_3_prob",
    ]
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    for item in items:
        utc, singapore = split_kickoff(item["starting_at"])
        row = {
            "fixture_id": item["fixture_id"],
            "kickoff_utc": utc,
            "kickoff_singapore": singapore,
            "home_team": item["home_team"],
            "away_team": item["away_team"],
            "ft_home": item["ft"]["home"],
            "ft_draw": item["ft"]["draw"],
            "ft_away": item["ft"]["away"],
            "ht_home": item["ht"]["home"],
            "ht_draw": item["ht"]["draw"],
            "ht_away": item["ht"]["away"],
        }
        for index, score in enumerate(item["top_scores"], start=1):
            row[f"score_{index}"] = f"{score['home_goals']}-{score['away_goals']}"
            row[f"score_{index}_prob"] = score["probability"]
        writer.writerow(row)
    return buffer.getvalue()
