"""Backfill pre-match 1X2 odds and team xG for fixtures already on disk.

The daily ``fetch`` stays on ``participants;scores;state;round;season``.
This module is the one-off (and later incremental) pass:

* xG: ``GET /fixtures/multi/{ids}?include=xGFixture``, up to 50 ids per
  request. Type 5304 only. xGA is the opponent's xG.
* Odds: ``GET /fixtures/multi/{ids}?include=odds&filters=markets:1``.
  Market 1 is Fulltime Result. ``inplayOdds`` is never requested. Quotes
  at or after kickoff are dropped, then the remaining bookmakers are
  averaged and de-vigged.

A second run asks again for rows that are still missing odds, for finished
rows that have a de-vigged price but no ``raw_implied_*`` (Shin and power
need the overround), or for finished rows still missing xG. HTTP 403 stops
that feed and leaves a message
instead of a traceback. Raw responses go through the same cache as
``fetch``.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Iterator
from dataclasses import dataclass

import pandas as pd

from laliga.data.client import SportMonksClient, SportMonksError
from laliga.data.parse import aggregate_prematch_1x2, team_expected_goals
from laliga.data.store import MatchStore
from laliga.model.features import ODDS_FEATURES, odds_features

logger = logging.getLogger(__name__)

MULTI_CHUNK = 50
ODDS_DENIED = (
    "订阅不能读取赛前赔率（GET /fixtures/multi/{ids}?include=odds&filters=markets:1，HTTP 403）。"
    "需要 Odds & Predictions 附加包，并且覆盖西甲（联赛 564，2024/2025 起）。"
    "已停止其余赔率请求，已有的赔率列保持不变。"
)
XG_DENIED = (
    "订阅不能读取 xG（GET /fixtures/multi/{ids}?include=xGFixture，HTTP 403）。"
    "需要 Pressure Index & xG（或 xG Basic）附加包。"
    "已停止其余 xG 请求，已有的 xG 列保持不变。"
)


@dataclass
class MarketFetchResult:
    fixtures: int
    odds_complete: int
    xg_complete: int
    odds_requested: int
    xg_requested: int
    odds_denied: str | None
    xg_denied: str | None


def fetch_stored_markets(
    client: SportMonksClient,
    store: MatchStore,
    *,
    force: bool = False,
    chunk_size: int = MULTI_CHUNK,
) -> MarketFetchResult:
    """Fill odds and xG on ``matches.csv``. Returns coverage, including denials."""

    frame = store.load()
    if frame.empty:
        raise SportMonksError("本地没有比赛，无法回填赔率和 xG。请先运行 fetch。")
    if chunk_size < 1:
        raise SportMonksError("chunk_size 必须为正。")

    attempts = _load_attempts(store)
    if force:
        attempts = {"odds": set(), "xg": set()}
    odds_ids = [int(row.fixture_id) for row in frame.itertuples(index=False) if _wants_odds(row, force, attempts["odds"])]
    xg_ids = [int(row.fixture_id) for row in frame.itertuples(index=False) if _wants_xg(row, force, attempts["xg"])]
    odds_denied = _pull_chunks(
        client,
        frame,
        odds_ids,
        include="odds",
        filters="markets:1",
        kind="odds",
        denial=ODDS_DENIED,
        chunk_size=chunk_size,
        attempts=attempts,
    )
    xg_denied = _pull_chunks(
        client,
        frame,
        xg_ids,
        include="xGFixture",
        filters=None,
        kind="xg",
        denial=XG_DENIED,
        chunk_size=chunk_size,
        attempts=attempts,
    )
    store.replace(frame)
    _save_attempts(store, attempts)
    return MarketFetchResult(
        fixtures=len(frame),
        odds_complete=sum(_row_has_odds(row) for _, row in frame.iterrows()),
        xg_complete=sum(_row_has_xg(row) for _, row in frame.iterrows() if row["status"] == "finished"),
        odds_requested=len(odds_ids),
        xg_requested=len(xg_ids),
        odds_denied=odds_denied,
        xg_denied=xg_denied,
    )


def format_market_fetch(result: MarketFetchResult) -> str:
    lines = [
        f"已检查 {result.fixtures} 场已存储比赛。",
        f"完整赛前 1X2：{result.odds_complete} 场（本次计划请求 {result.odds_requested} 场）。",
        f"完场且主客 xG 都有：{result.xg_complete} 场（本次计划请求 {result.xg_requested} 场）。",
        "赔率来自赛前 include=odds、filters=markets:1（全场胜平负）。",
        "每家博彩只用开赛前最后一条报价；多家取隐含概率平均后再去掉水位。不请求 inplayOdds。",
        "raw_implied_* 是去水位之前的平均 1/小数赔率。Shin 和幂去水位要用它。"
        "只有去水位概率、没有 raw_implied_* 的完场比赛会再请求一次。",
        "xG 是 xGFixture 的 type 5304。home_xga 是客队 xG，away_xga 是主队 xG。",
        "本场 xG 只留给以后的比赛做滚动均值，不会当成这场的特征。",
        "每日 fetch 和操作台「更新数据」不会跑这一步。补新比赛后再执行本命令即可，已有数值的行会跳过。",
    ]
    if result.odds_denied:
        lines.append(result.odds_denied)
    if result.xg_denied:
        lines.append(result.xg_denied)
    return "\n".join(lines)


def _pull_chunks(
    client: SportMonksClient,
    frame: pd.DataFrame,
    fixture_ids: list[int],
    *,
    include: str,
    filters: str | None,
    kind: str,
    denial: str,
    chunk_size: int,
    attempts: dict[str, set[int]],
) -> str | None:
    if not fixture_ids:
        return None
    lookup = _fixture_lookup(frame)
    total = len(fixture_ids)
    seen = 0
    for chunk in _chunks(fixture_ids, chunk_size):
        seen += len(chunk)
        logger.info("回填%s %s/%s 场", "赔率" if kind == "odds" else "xG", seen, total)
        params: dict[str, str] = {"include": include}
        if filters:
            params["filters"] = filters
        path = "fixtures/multi/" + ",".join(str(fixture_id) for fixture_id in chunk)
        returned: set[int] = set()
        try:
            for payload in client.paginate(path, params):
                if not isinstance(payload, dict):
                    continue
                try:
                    fixture_key = int(payload.get("id"))
                except (TypeError, ValueError):
                    continue
                returned.add(fixture_key)
                if _apply_payload(frame, lookup, payload, kind):
                    attempts[kind].discard(fixture_key)
                else:
                    attempts[kind].add(fixture_key)
        except SportMonksError as exc:
            if _permission_denied(exc):
                logger.warning("%s", denial)
                return denial
            if _missing_result(exc):
                logger.info("这批比赛没有%s：%s", "赔率" if kind == "odds" else "xG", exc)
                attempts[kind].update(fixture_id for fixture_id in chunk if fixture_id not in returned)
                continue
            raise
        for fixture_id in chunk:
            if fixture_id not in returned:
                attempts[kind].add(fixture_id)
    return None


def _apply_payload(frame: pd.DataFrame, lookup: dict[int, tuple[int, int, int, pd.Timestamp | None]], payload: dict, kind: str) -> bool:
    fixture_id = payload.get("id")
    try:
        fixture_key = int(fixture_id)
    except (TypeError, ValueError):
        return False
    meta = lookup.get(fixture_key)
    if meta is None:
        return False
    index, home_id, away_id, kickoff = meta
    if kind == "xg":
        home_xg, away_xg, home_xga, away_xga = team_expected_goals(payload, home_id, away_id)
        _assign(frame, index, "home_xg", home_xg)
        _assign(frame, index, "away_xg", away_xg)
        _assign(frame, index, "home_xga", home_xga)
        _assign(frame, index, "away_xga", away_xga)
        return home_xg is not None and away_xg is not None
    prices = aggregate_prematch_1x2(_odd_rows(payload), kickoff=kickoff)
    for column, value in prices.items():
        _assign(frame, index, column, value)
    return prices["odds_home"] is not None


def _odd_rows(payload: dict) -> list[dict]:
    rows: list[dict] = []
    for key in ("odds", "premiumOdds", "premiumodds"):
        chunk = payload.get(key)
        if isinstance(chunk, list):
            rows.extend(item for item in chunk if isinstance(item, dict))
    return rows


def _assign(frame: pd.DataFrame, index: int, column: str, value: float | None) -> None:
    if column not in frame.columns:
        frame[column] = pd.NA
    if value is None:
        return
    frame.at[index, column] = value


def _fixture_lookup(frame: pd.DataFrame) -> dict[int, tuple[int, int, int, pd.Timestamp | None]]:
    lookup: dict[int, tuple[int, int, int, pd.Timestamp | None]] = {}
    for index, row in frame.iterrows():
        try:
            fixture_id = int(row["fixture_id"])
            home_id = int(row["home_team_id"])
            away_id = int(row["away_team_id"])
        except (TypeError, ValueError):
            continue
        kickoff = row["starting_at"]
        stamp = None if pd.isna(kickoff) else pd.Timestamp(kickoff)
        lookup[fixture_id] = (int(index), home_id, away_id, stamp)
    return lookup


def _field(row, name: str):
    if isinstance(row, pd.Series):
        return row[name] if name in row.index else None
    return getattr(row, name, None)


def _wants_odds(row, force: bool, misses: set[int]) -> bool:
    if _field(row, "status") not in ("finished", "scheduled"):
        return False
    if force or _field(row, "status") == "scheduled":
        return True
    if _row_has_odds(row) and _row_has_raw_implied(row):
        return False
    return int(_field(row, "fixture_id")) not in misses


def _wants_xg(row, force: bool, misses: set[int]) -> bool:
    if _field(row, "status") != "finished":
        return False
    if force or not _row_has_xg(row):
        return int(_field(row, "fixture_id")) not in misses or force
    return False


def _row_has_odds(row) -> bool:
    series = row if isinstance(row, pd.Series) else pd.Series(row._asdict())
    implied = odds_features(series)
    return all(math.isfinite(implied[name]) for name in ODDS_FEATURES)


def _row_has_raw_implied(row) -> bool:
    """True when the pre-normalisation 1X2 average is stored on all three sides."""

    for side in ("home", "draw", "away"):
        value = _field(row, f"raw_implied_{side}")
        try:
            if value is None or pd.isna(value):
                return False
            number = float(value)
        except (TypeError, ValueError):
            return False
        if not math.isfinite(number) or number <= 0.0 or number >= 1.0:
            return False
    return True


def _row_has_xg(row) -> bool:
    for column in ("home_xg", "away_xg"):
        value = _field(row, column)
        try:
            if value is None or pd.isna(value):
                return False
            if not math.isfinite(float(value)):
                return False
        except (TypeError, ValueError):
            return False
    return True


def _attempts_path(store: MatchStore):
    return store.root / "processed" / "market_attempts.json"


def _load_attempts(store: MatchStore) -> dict[str, set[int]]:
    path = _attempts_path(store)
    if not path.exists():
        return {"odds": set(), "xg": set()}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"odds": set(), "xg": set()}
    return {
        "odds": {int(item) for item in payload.get("odds") or []},
        "xg": {int(item) for item in payload.get("xg") or []},
    }


def _save_attempts(store: MatchStore, attempts: dict[str, set[int]]) -> None:
    path = _attempts_path(store)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"odds": sorted(attempts["odds"]), "xg": sorted(attempts["xg"])}, indent=2),
        encoding="utf-8",
    )


def _chunks(values: list[int], size: int) -> Iterator[list[int]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _permission_denied(exc: SportMonksError) -> bool:
    if exc.status == 403:
        return True
    text = str(exc).lower()
    return "don't have access" in text or "do not have access" in text


def _missing_result(exc: SportMonksError) -> bool:
    if _permission_denied(exc):
        return False
    text = str(exc).lower()
    return exc.status == 404 or "no result" in text
