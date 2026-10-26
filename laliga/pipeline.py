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
            initial_sh=None if previous is None else previous.sh,
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
        document.update(prediction_extension(item.prediction))
        if item.availability is not None:
            document["availability"] = item.availability
        items.append(document)
    return items


def prediction_extension(prediction: Prediction) -> dict:
    """Fields the notifier file does not carry: top-2 scores and HT/FT."""

    payload = {
        "joint": bool(prediction.joint),
        "track": prediction.track,
        "top2_scores": [score.to_dict() for score in prediction.top2_scores],
        "top_htft": [{"code": code, "probability": float(probability)} for code, probability in prediction.top_htft],
    }
    if prediction.htft:
        payload["htft"] = {code: float(probability) for code, probability in prediction.htft.items()}
    if prediction.state_eta is not None:
        payload["state_eta"] = prediction.state_eta
    if prediction.params:
        payload["params"] = prediction.params
    return payload


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


def describe_alternative_tracks(scored: list[ScoredFixture], history: pd.DataFrame, root: Path) -> str:
    """Terminal lines for football, market, and blend. Nothing is written."""

    from laliga.data.quotes import load_quotes
    from laliga.model.tracks import TRACK_MAX_FT, TRACK_MAX_HALF, TRACK_TEMPERATURE, blend_forecast, market_rates_for_row
    from laliga.upgrade import load_registry

    weight = _blend_weight(root, load_registry)
    quotes = load_quotes(root)
    lookup = _history_lookup(history)
    lines = ["三条预测（终端显示；--no-archive 也会打印。不写入 latest_next.json）："]
    if not scored:
        lines.append("没有比赛。")
        return "\n".join(lines)
    for item in scored:
        prediction = item.prediction
        lines.append(f"{prediction.home_team} vs {prediction.away_team}")
        lines.append("  纯足球：" + _prob_phrase(prediction.ft, prediction.top_htft, prediction.top2_scores))
        row = lookup.get(int(prediction.fixture_id))
        implied = None if row is None else market_rates_for_row(row, quotes)
        if implied is None:
            lines.append("  这场没有赛前 1X2，所以没有市场预测和混合。")
            continue
        forecast = implied.forecast(eta=0.0, max_half=TRACK_MAX_HALF, max_ft=TRACK_MAX_FT, temperature=TRACK_TEMPERATURE)
        lines.append("  市场：" + _forecast_phrase(forecast))
        if prediction.rates is None:
            lines.append("  混合：这场没有半场速率，不能和市场价格混合。")
            continue
        blended = blend_forecast(prediction.rates, implied, weight)
        if weight <= 0:
            lines.append(f"  混合：市场权重为 0，与纯足球相同。{_forecast_phrase(blended)}")
        else:
            lines.append(f"  混合（市场权重 {weight:.2f}）：{_forecast_phrase(blended)}")
    return "\n".join(lines)


def _blend_weight(root: Path, load_registry) -> float:
    registry = load_registry(root) or {}
    try:
        return float(((registry.get("champion") or {}).get("params") or {}).get("blend_market_weight") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _history_lookup(history: pd.DataFrame) -> dict:
    lookup = {}
    if history is None or history.empty or "fixture_id" not in history.columns:
        return lookup
    for _, row in history.drop_duplicates("fixture_id", keep="last").iterrows():
        lookup[int(row["fixture_id"])] = row
    return lookup


def _prob_phrase(market, top_htft, top_scores=None) -> str:
    pairs = "、".join(f"{code} {probability * 100:.1f}%" for code, probability in (top_htft or [])[:2]) or "无"
    return (
        f"全场 {market.home * 100:.1f}% / {market.draw * 100:.1f}% / {market.away * 100:.1f}%"
        f"；前两比分 {_top_scores_phrase(top_scores)}；半全场前两档 {pairs}"
    )


def _forecast_phrase(forecast) -> str:
    home, draw, away = forecast.ft_probs
    pairs = "、".join(f"{code} {probability * 100:.1f}%" for code, probability in forecast.top_htft[:2]) or "无"
    return (
        f"全场 {home * 100:.1f}% / {draw * 100:.1f}% / {away * 100:.1f}%"
        f"；前两比分 {_top_scores_phrase(forecast.top2_scores)}；半全场前两档 {pairs}"
    )


def _top_scores_phrase(top_scores) -> str:
    parts = []
    for item in list(top_scores or [])[:2]:
        if hasattr(item, "home_goals"):
            parts.append(f"{item.home_goals}-{item.away_goals} {item.probability * 100:.1f}%")
            continue
        home, away, probability = item[0], item[1], item[2]
        parts.append(f"{int(home)}-{int(away)} {float(probability) * 100:.1f}%")
    return "、".join(parts) or "无"


def annotate_market_items(items: list[dict], scored: list[ScoredFixture], history: pd.DataFrame, root: Path) -> None:
    """Add market and blend summaries onto dashboard items when a price exists.

    ``latest_next.json`` is written from ``Prediction.to_dict()`` and is not
    passed through here.
    """

    from laliga.data.quotes import load_quotes
    from laliga.model.tracks import TRACK_MAX_FT, TRACK_MAX_HALF, TRACK_TEMPERATURE, blend_forecast, market_rates_for_row
    from laliga.upgrade import load_registry

    quotes = load_quotes(root)
    if history is None or history.empty:
        return
    by_id = {item.prediction.fixture_id: item for item in scored}
    lookup = history.drop_duplicates("fixture_id", keep="last").set_index("fixture_id")
    weight = 0.0
    registry = load_registry(root) or {}
    try:
        weight = float(((registry.get("champion") or {}).get("params") or {}).get("blend_market_weight") or 0.0)
    except (TypeError, ValueError):
        weight = 0.0
    for item in items:
        try:
            row = lookup.loc[int(item["fixture_id"])]
        except KeyError:
            continue
        if isinstance(row, pd.DataFrame):
            row = row.iloc[-1]
        implied = market_rates_for_row(row, quotes)
        if implied is None:
            continue
        forecast = implied.forecast(eta=0.0, max_half=TRACK_MAX_HALF, max_ft=TRACK_MAX_FT, temperature=TRACK_TEMPERATURE)
        item["market"] = _forecast_brief(forecast)
        football = by_id.get(int(item["fixture_id"]))
        if weight > 0 and football is not None and football.prediction.rates is not None:
            blended = blend_forecast(football.prediction.rates, implied, weight)
            item["blend"] = _forecast_brief(blended)
            item["blend_market_weight"] = weight


def archive_alternative_tracks(store: MatchStore, scored: list[ScoredFixture], history: pd.DataFrame, *, origin: str, generated_at=None) -> int:
    """Append market and blend lines. They do not replace the football line in the hit-rate table."""

    from laliga.archive import PredictionArchive, utc_now
    from laliga.data.quotes import load_quotes
    from laliga.model.service import MarketProb, ScoreProb
    from laliga.model.tracks import TRACK_MAX_FT, TRACK_MAX_HALF, TRACK_TEMPERATURE, blend_forecast, market_rates_for_row
    from laliga.upgrade import load_registry

    quotes = load_quotes(store.root)
    lookup = {}
    if history is not None and not history.empty:
        for _, row in history.drop_duplicates("fixture_id", keep="last").iterrows():
            lookup[int(row["fixture_id"])] = row
    weight = 0.0
    registry = load_registry(store.root) or {}
    try:
        weight = float(((registry.get("champion") or {}).get("params") or {}).get("blend_market_weight") or 0.0)
    except (TypeError, ValueError):
        weight = 0.0
    generated = generated_at or utc_now()
    accepted = 0
    archive = PredictionArchive(store.root)
    for item in scored:
        row = lookup.get(item.prediction.fixture_id)
        if row is None:
            continue
        implied = market_rates_for_row(row, quotes)
        if implied is None:
            continue
        forecast = implied.forecast(eta=0.0, max_half=TRACK_MAX_HALF, max_ft=TRACK_MAX_FT, temperature=TRACK_TEMPERATURE)
        market = _retarget(item.prediction, forecast, MarketProb, ScoreProb, track="market")
        result = archive.append(
            [market],
            model_version=model_version() + "+market",
            origin=origin + ":market",
            generated_at=generated,
            matches=history,
            extras={market.fixture_id: {"track": "market", **prediction_extension(market)}},
        )
        accepted += result.accepted
        football_rates = item.prediction.rates
        if weight > 0 and football_rates is not None:
            blended = blend_forecast(football_rates, implied, weight)
            blend = _retarget(item.prediction, blended, MarketProb, ScoreProb, track="blend")
            blend_result = archive.append(
                [blend],
                model_version=model_version() + "+blend",
                origin=origin + ":blend",
                generated_at=generated,
                matches=history,
                extras={blend.fixture_id: {"track": "blend", "blend_market_weight": weight, **prediction_extension(blend)}},
            )
            accepted += blend_result.accepted
    return accepted


def _forecast_brief(forecast) -> dict:
    return {
        "ft": {"home": forecast.ft_probs[0], "draw": forecast.ft_probs[1], "away": forecast.ft_probs[2]},
        "ht": {"home": forecast.ht_probs[0], "draw": forecast.ht_probs[1], "away": forecast.ht_probs[2]},
        "top2_scores": [
            {"home_goals": home, "away_goals": away, "probability": probability}
            for home, away, probability in forecast.top2_scores
        ],
        "top_htft": [{"code": code, "probability": probability} for code, probability in forecast.top_htft],
        "expected_goals_ft": {"home": forecast.expected_ft[0], "away": forecast.expected_ft[1]},
    }


def _retarget(prediction: Prediction, forecast, market_prob, score_prob, *, track: str) -> Prediction:
    from dataclasses import replace

    return replace(
        prediction,
        ft=market_prob(*forecast.ft_probs),
        ht=market_prob(*forecast.ht_probs),
        top_scores=[score_prob(home, away, probability) for home, away, probability in forecast.top_scores],
        top2_scores=[score_prob(home, away, probability) for home, away, probability in forecast.top2_scores],
        top_htft=list(forecast.top_htft),
        htft=dict(forecast.htft),
        expected_goals_ft=forecast.expected_ft,
        expected_goals_ht=forecast.expected_ht,
        joint=True,
        track=track,
    )


def archive_scored(
    store: MatchStore,
    scored: list[ScoredFixture],
    history: pd.DataFrame,
    *,
    origin: str,
    generated_at=None,
) -> AppendResult:
    used = any(item.availability is not None and item.availability.get("used") for item in scored)
    extras = {}
    for item in scored:
        extra = {"track": "football", **prediction_extension(item.prediction)}
        if item.availability is not None:
            extra["availability"] = item.availability
        extras[item.prediction.fixture_id] = extra
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
        "top2_1",
        "top2_1_prob",
        "top2_2",
        "top2_2_prob",
        "htft_1",
        "htft_1_prob",
        "htft_2",
        "htft_2_prob",
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
        for index, score in enumerate((item.get("top2_scores") or [])[:2], start=1):
            row[f"top2_{index}"] = f"{score['home_goals']}-{score['away_goals']}"
            row[f"top2_{index}_prob"] = score["probability"]
        for index, pair in enumerate((item.get("top_htft") or [])[:2], start=1):
            row[f"htft_{index}"] = pair.get("code") or ""
            row[f"htft_{index}_prob"] = pair.get("probability")
        writer.writerow(row)
    return buffer.getvalue()
