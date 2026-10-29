"""Download and flatten SportMonks Standard pre-match odds.

Endpoint (Standard Odds Feed, requires the Odds add-on):

    GET /v3/football/odds/pre-match/fixtures/{fixture_id}
        ?filters=markets:1,31,29,57

The endpoint is not paginated. Each row is one selection from one bookmaker
(``market_id``, ``bookmaker_id``, ``label``, ``name``, decimal ``value``).
Coverage for older fixtures depends on the league's odds coverage; the docs
state no age cut-off and coverage back to roughly 2018.

The ``winning`` field is never stored: it is settled after the match and
would leak the result into the market track.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import pandas as pd

from laliga.config import ODDS_MARKETS
from laliga.data.client import SportMonksClient, SportMonksError

logger = logging.getLogger(__name__)

ODDS_COLUMNS = [
    "fixture_id",
    "market_id",
    "bookmaker_id",
    "label",
    "name",
    "value",
    "total",
    "handicap",
    "latest_bookmaker_update",
]

ODDS_UNAVAILABLE_HINT = (
    "当前订阅读不到 SportMonks 赔率。纯市场和综合模型需要 Odds add-on（Standard pre-match odds），"
    "并且要覆盖西甲（联赛 ID 564）。纯足球模型不受影响。"
)


class OddsUnavailable(SportMonksError):
    """The subscription cannot read the odds endpoints at all."""


@dataclass
class OddsFetchSummary:
    requested: int
    with_odds: int
    rows: int

    def message(self) -> str:
        return f"请求了 {self.requested} 场比赛的赔率，其中 {self.with_odds} 场有赔率，共 {self.rows} 条报价。"


def empty_odds() -> pd.DataFrame:
    return pd.DataFrame(columns=ODDS_COLUMNS)


def odds_path(fixture_id: int) -> str:
    return f"odds/pre-match/fixtures/{int(fixture_id)}"


def odds_params() -> dict[str, str]:
    return {"filters": "markets:" + ",".join(str(market) for market in ODDS_MARKETS)}


def parse_odds(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Flatten SportMonks odd objects. Rows without a usable decimal price are dropped."""

    parsed = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        market = _int(row.get("market_id"))
        fixture = _int(row.get("fixture_id"))
        bookmaker = _int(row.get("bookmaker_id"))
        value = _decimal(row.get("value"), row.get("dp3"))
        if market is None or fixture is None or bookmaker is None or value is None:
            continue
        if market not in ODDS_MARKETS:
            continue
        parsed.append(
            {
                "fixture_id": fixture,
                "market_id": market,
                "bookmaker_id": bookmaker,
                "label": _text(row.get("label")),
                "name": _text(row.get("name")),
                "value": value,
                "total": _text(row.get("total")),
                "handicap": _text(row.get("handicap")),
                "latest_bookmaker_update": _text(row.get("latest_bookmaker_update") or row.get("updated_at")),
            }
        )
    if not parsed:
        return empty_odds()
    return pd.DataFrame(parsed, columns=ODDS_COLUMNS)


def fetch_fixture_odds(client: SportMonksClient, fixture_id: int, *, refresh: bool | None = None) -> list[dict[str, Any]]:
    """Return raw odd rows for one fixture; an empty list when the fixture has no odds."""

    try:
        body = client.get(odds_path(fixture_id), odds_params(), refresh=refresh)
    except SportMonksError as exc:
        text = str(exc).lower()
        if exc.status in (401, 403) or "subscription" in text or "access" in text:
            raise OddsUnavailable(f"{ODDS_UNAVAILABLE_HINT} 接口返回：{exc}", status=exc.status) from exc
        if exc.status == 404 or "no result" in text:
            return []
        raise
    data = body.get("data")
    if data is None:
        return []
    if not isinstance(data, list):
        raise SportMonksError(f"赔率接口返回的 data 不是列表（比赛 {fixture_id}）。")
    return data


def fetch_odds(client: SportMonksClient, store, fixtures: pd.DataFrame) -> OddsFetchSummary:
    """Fetch odds for ``fixtures`` and replace their rows in the local odds table.

    Finished fixtures reuse the raw response cache; scheduled fixtures are
    always requested again because their prices still move.
    """

    if fixtures.empty:
        return OddsFetchSummary(0, 0, 0)
    frames: list[pd.DataFrame] = []
    with_odds = 0
    ordered = fixtures.sort_values(["starting_at", "fixture_id"])
    total = len(ordered)
    for position, (_, fixture) in enumerate(ordered.iterrows(), start=1):
        fixture_id = int(fixture["fixture_id"])
        scheduled = str(fixture.get("status") or "") == "scheduled"
        rows = fetch_fixture_odds(client, fixture_id, refresh=True if scheduled else None)
        frame = parse_odds([{**row, "fixture_id": row.get("fixture_id", fixture_id)} for row in rows])
        frame = frame[frame["fixture_id"] == fixture_id]
        if not frame.empty:
            with_odds += 1
            frames.append(frame)
        if position % 50 == 0 or position == total:
            logger.info("赔率进度 %s/%s，有赔率 %s 场", position, total, with_odds)
    combined = pd.concat(frames, ignore_index=True) if frames else empty_odds()
    store.upsert_odds(combined, {int(value) for value in ordered["fixture_id"]})
    return OddsFetchSummary(total, with_odds, int(len(combined)))


def _int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _decimal(value: Any, fallback: Any = None) -> float | None:
    for candidate in (value, fallback):
        try:
            number = float(str(candidate).strip())
        except (TypeError, ValueError):
            continue
        if number > 1.0:
            return number
    return None


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()
