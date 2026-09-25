"""Download La Liga seasons and fixtures from SportMonks."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator
from datetime import date, timedelta
from typing import Any

import pandas as pd

from laliga.config import LA_LIGA_LEAGUE_ID, MAX_BETWEEN_DAYS
from laliga.data.client import SportMonksClient, SportMonksError
from laliga.data.parse import fixtures_to_frame
from laliga.data.store import MatchStore

logger = logging.getLogger(__name__)

FIXTURE_INCLUDES = "participants;scores;state;round;season"


def fetch_historical(client: SportMonksClient, store: MatchStore, *, seasons: int, replace: bool) -> pd.DataFrame:
    """Fetch the most recent ``seasons`` La Liga seasons and store fixtures."""

    league = _league_payload(client)
    chosen = select_seasons(league.get("seasons") or [], seasons)
    if not chosen:
        raise SportMonksError(
            "联赛接口没有返回赛季。请确认订阅包含西甲（La Liga，联赛 ID 564）。免费计划不含西甲。"
        )
    names = ", ".join(str(item.get("name") or item.get("id")) for item in chosen)
    logger.info("将抓取 %s 个赛季：%s", len(chosen), names)

    raw: list[dict[str, Any]] = []
    for season in chosen:
        season_id = season.get("id")
        logger.info("抓取赛季 %s（id=%s）", season.get("name") or "", season_id)
        try:
            fixtures = list(fetch_season_fixtures(client, int(season_id)))
        except SportMonksError as exc:
            if _is_empty_result(exc):
                logger.warning("赛季 %s 没有比赛或当前订阅不可见：%s", season_id, exc)
                continue
            raise
        _reject_foreign_leagues(fixtures)
        raw.extend(fixtures)
        logger.info("赛季 %s 得到 %s 场", season_id, len(fixtures))

    if not raw:
        raise SportMonksError(
            "没有抓到任何西甲比赛。请确认订阅包含 La Liga（联赛 ID 564）以及这些历史赛季。免费计划不含西甲。"
        )
    frame = fixtures_to_frame(raw)
    frame = frame[frame["league_id"] == LA_LIGA_LEAGUE_ID]
    store.save_seasons(
        {
            "league_id": LA_LIGA_LEAGUE_ID,
            "league_name": league.get("name") or "La Liga",
            "seasons": [
                {
                    "id": item.get("id"),
                    "name": item.get("name"),
                    "starting_at": item.get("starting_at"),
                    "ending_at": item.get("ending_at"),
                    "is_current": item.get("is_current"),
                    "finished": item.get("finished"),
                }
                for item in chosen
            ],
        }
    )
    if replace:
        store.replace(frame)
        return frame
    return store.upsert(frame)


def fetch_window(client: SportMonksClient, store: MatchStore, start: date, end: date) -> pd.DataFrame:
    """Fetch La Liga fixtures between two dates (inclusive) and upsert them."""

    if end < start:
        raise SportMonksError("结束日期早于开始日期。")
    raw: list[dict[str, Any]] = []
    for window_start, window_end in iter_date_windows(start, end, MAX_BETWEEN_DAYS):
        logger.info("抓取赛程 %s 至 %s", window_start.isoformat(), window_end.isoformat())
        path = f"fixtures/between/{window_start.isoformat()}/{window_end.isoformat()}"
        params = {
            "include": FIXTURE_INCLUDES,
            "filters": f"fixtureLeagues:{LA_LIGA_LEAGUE_ID}",
            "per_page": 50,
        }
        try:
            fixtures = list(client.paginate(path, params))
        except SportMonksError as exc:
            if _is_empty_result(exc):
                logger.info("该日期窗口没有西甲比赛。")
                continue
            raise
        _reject_foreign_leagues(fixtures)
        raw.extend(fixtures)
    frame = fixtures_to_frame(raw)
    if not frame.empty:
        frame = frame[frame["league_id"] == LA_LIGA_LEAGUE_ID]
        return store.upsert(frame)
    return store.load()


def fetch_season_fixtures(client: SportMonksClient, season_id: int) -> Iterator[dict[str, Any]]:
    """Paginate ``GET /fixtures?filters=fixtureSeasons:{id}``."""

    params = {
        "include": FIXTURE_INCLUDES,
        "filters": f"fixtureSeasons:{season_id}",
        "per_page": 50,
    }
    return client.paginate("fixtures", params)


def select_seasons(seasons: Iterable[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """Keep the ``limit`` most recent seasons by ``starting_at``."""

    if limit < 1:
        raise SportMonksError("赛季数量至少为 1。")
    rows = [item for item in seasons if isinstance(item, dict) and item.get("id") is not None]
    rows.sort(key=lambda item: str(item.get("starting_at") or ""), reverse=True)
    return rows[:limit]


def iter_date_windows(start: date, end: date, max_inclusive_days: int = MAX_BETWEEN_DAYS) -> Iterator[tuple[date, date]]:
    """Split an inclusive date range into windows of at most ``max_inclusive_days``."""

    if max_inclusive_days < 1:
        raise ValueError("max_inclusive_days 必须为正。")
    if end < start:
        raise ValueError("结束日期早于开始日期。")
    cursor = start
    while cursor <= end:
        stop = min(end, cursor + timedelta(days=max_inclusive_days - 1))
        yield cursor, stop
        cursor = stop + timedelta(days=1)


def _league_payload(client: SportMonksClient) -> dict[str, Any]:
    try:
        body = client.get(
            f"leagues/{LA_LIGA_LEAGUE_ID}",
            {"include": "seasons;currentSeason"},
        )
    except SportMonksError as exc:
        if exc.status in (401, 403, 404) or "subscription" in str(exc).lower():
            raise SportMonksError(
                "无法读取西甲联赛（ID 564）。免费计划不含 La Liga，请换用包含西甲的订阅，并检查 token。"
            ) from exc
        raise
    data = body.get("data")
    if not isinstance(data, dict):
        raise SportMonksError("联赛接口返回的 data 不是对象。")
    return data


def _reject_foreign_leagues(fixtures: list[dict[str, Any]]) -> None:
    leagues = {item.get("league_id") for item in fixtures if isinstance(item, dict)}
    foreign = leagues - {LA_LIGA_LEAGUE_ID, None}
    if foreign:
        raise SportMonksError(
            f"接口返回了非西甲联赛 {sorted(foreign)}。已中止，避免把其他联赛写进训练集。"
            "请检查 filters=fixtureLeagues / fixtureSeasons。"
        )


def _is_empty_result(exc: SportMonksError) -> bool:
    text = str(exc).lower()
    if "subscription" in text or "don't have access" in text or "do not have access" in text:
        return False
    return "no result" in text or exc.status == 404
