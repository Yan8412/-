"""Steps shared by the command line and the dashboard."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

import pandas as pd

from laliga.config import ModelConfig, api_token
from laliga.data.client import SportMonksError, open_client
from laliga.data.odds import OddsUnavailable, fetch_odds
from laliga.data.store import MatchStore
from laliga.model import promotion
from laliga.model.dixon_coles import FitError
from laliga.model.ensemble import EnsembleWeights
from laliga.model.market import MarketBook
from laliga.model.service import ScorelineModel, fit_models, league_ht_ratio, model_is_fresh
from laliga.model.tracks import FixtureForecast, forecast_fixture

logger = logging.getLogger(__name__)


def load_saved_model(store: MatchStore) -> ScorelineModel | None:
    document = store.load_model_document()
    if not document:
        return None
    try:
        return ScorelineModel.from_dict(document)
    except (KeyError, TypeError, ValueError, FitError) as exc:
        logger.warning("已保存的模型无法读取，将重新拟合：%s", exc)
        return None


def load_ensemble(store: MatchStore) -> EnsembleWeights:
    return EnsembleWeights.from_dict(store.load_json(store.ensemble_path))


def refresh_odds(store: MatchStore, fixtures: pd.DataFrame, *, demo: bool = False) -> None:
    """Fetch odds for upcoming fixtures. Failure leaves the football track untouched."""

    if demo or fixtures.empty:
        return
    if not api_token():
        logger.info("未设置 SPORTMONKS_API_TOKEN，跳过赔率。纯市场和综合模型会标注为缺赔率。")
        return
    try:
        client = open_client(store.cache_dir, refresh=False)
        summary = fetch_odds(client, store, fixtures)
    except OddsUnavailable as exc:
        logger.warning("赔率不可用：%s", exc)
        return
    except SportMonksError as exc:
        logger.warning("赔率请求失败，本次只用已缓存的赔率：%s", exc)
        return
    logger.info(summary.message())


def forecast_fixtures(
    history: pd.DataFrame,
    fixtures: pd.DataFrame,
    config: ModelConfig,
    store: MatchStore,
    *,
    refit: bool = False,
    save_model: bool = False,
) -> list[FixtureForecast]:
    """Three-track forecasts for scheduled fixtures, using only results before each kickoff day."""

    ordered = fixtures.sort_values(["starting_at", "fixture_id"])
    eligible = history[~history["fixture_id"].isin({int(value) for value in ordered["fixture_id"]})]
    book = MarketBook(store.load_odds())
    weights = load_ensemble(store)
    active = promotion.load_active(store)
    ht_ratio = league_ht_ratio(eligible)
    missing = [int(fixture_id) for fixture_id in ordered["fixture_id"] if not book.has(int(fixture_id))]
    if missing:
        logger.info("%s 场比赛没有本地赔率，这些比赛只有纯足球预测。", len(missing))

    def pack(model: ScorelineModel, row: pd.Series) -> FixtureForecast:
        return forecast_fixture(model, row, book, weights, active, ht_ratio=ht_ratio, rho=config.rho_prior)

    saved = None if refit else load_saved_model(store)
    if saved is not None and model_is_fresh(saved, eligible, ordered):
        logger.info("使用已保存的模型（训练截止日期不早于最近完场、且早于这些比赛）。")
        return [pack(saved, row) for _, row in ordered.iterrows()]

    logger.info("按每场比赛开球日前的完场数据重新拟合。")
    forecasts: list[FixtureForecast] = []
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
        forecasts.extend(pack(model, row) for _, row in group.iterrows())
    if save_model and previous is not None:
        store.save_model_document(previous.to_dict())
    return forecasts


def write_latest(store: MatchStore, forecasts: list[FixtureForecast], label: str, note: str) -> dict:
    from laliga.report import predictions_csv_text

    items = [forecast.to_dict() for forecast in forecasts]
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "label": label,
        "note": note,
        "active": promotion.load_active(store),
        "items": items,
    }
    store.predictions_dir.mkdir(parents=True, exist_ok=True)
    (store.predictions_dir / "latest.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (store.predictions_dir / "latest.csv").write_text(predictions_csv_text(items), encoding="utf-8")
    logger.info("已预测 %s 场（%s）。", len(items), label)
    return payload


def save_backtest(store: MatchStore, report, *, promote: bool) -> dict | None:
    """Write backtest files and combined weights; apply the promotion decision when asked."""

    from laliga.report import backtest_csv_text

    payload = report.to_dict()
    store.predictions_dir.mkdir(parents=True, exist_ok=True)
    (store.predictions_dir / "backtest.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (store.predictions_dir / "backtest.csv").write_text(backtest_csv_text(payload), encoding="utf-8")
    if report.n_with_odds:
        store.save_json(store.ensemble_path, {**report.ensemble.to_dict(), "fitted_on": "backtest", "created_at": payload["promotion"]["decided_at"]})
    if not promote or report.promotion is None:
        return None
    document = promotion.apply(store, report.promotion)
    for body in report.promotion["markets"].values():
        logger.info(body["reason"])
    return document
