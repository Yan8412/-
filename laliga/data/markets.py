"""Backfill pre-match 1X2 odds and team xG for fixtures already on disk.

The daily ``fetch`` stays on ``participants;scores;state;round;season``.
This module is the one-off (and later incremental) pass:

* xG: ``GET /fixtures/multi/{ids}?include=xGFixture``, up to 50 ids per
  request. Type 5304 only. xGA is the opponent's xG.
* Odds: ``GET /fixtures/multi/{ids}?include=odds&filters=markets:1``.
  Market 1 is Fulltime Result. ``inplayOdds`` is never requested. A quote
  at or after kickoff is always dropped, including rows with ``stopped``
  or ``winning`` set. A missing timestamp stays. The remaining bookmakers
  are averaged and de-vigged. ``--refresh`` replaces that fixture's quote
  record. A stored record whose ``parser_version`` is not current is
  parsed again from the cached payload.

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
from laliga.data.quotes import (
    EXTENDED_FILTER,
    MARKET_NAMES,
    PARSER_VERSION,
    absorb_fixture_odds,
    load_quotes,
    save_quotes,
)
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
    quotes_saved: int = 0
    extended_note: str | None = None


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
    stored_quotes = load_quotes(store.root)
    stale_ids = _stale_quote_ids(frame, stored_quotes)
    odds_ids = []
    seen_odds: set[int] = set()
    for row in frame.itertuples(index=False):
        try:
            fixture_id = int(row.fixture_id)
        except (TypeError, ValueError):
            continue
        if fixture_id in seen_odds:
            continue
        if _wants_odds(row, force, attempts["odds"]) or fixture_id in stale_ids:
            odds_ids.append(fixture_id)
            seen_odds.add(fixture_id)
    replace_ids = set(odds_ids) if force else stale_ids
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
        quotes_root=store.root,
        replace_ids=replace_ids,
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
    extended_note = None
    quotes_saved = 0
    if odds_denied is None and odds_ids:
        extended_note, quotes_saved = _pull_extended(client, store.root, frame, odds_ids, chunk_size=chunk_size)
    elif odds_denied is None:
        quotes_saved = 0
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
        quotes_saved=quotes_saved,
        extended_note=extended_note,
    )


def format_market_fetch(result: MarketFetchResult) -> str:
    lines = [
        f"已检查 {result.fixtures} 场已存储比赛。",
        f"完整赛前 1X2：{result.odds_complete} 场（本次计划请求 {result.odds_requested} 场）。",
        f"完场且主客 xG 都有：{result.xg_complete} 场（本次计划请求 {result.xg_requested} 场）。",
        "赔率来自赛前 include=odds、filters=markets:1（全场胜平负）。",
        "每家博彩只用赛前最后一条报价；多家取隐含概率平均后再去掉水位。不请求 inplayOdds。",
        "latest_bookmaker_update 早于开球的保留。不早于开球的一律丢掉，包括 stopped 为真和 winning 为 false 的行。现场盘每一行都带 winning，不能靠它把结算和滚球分开。没有时间戳的保留。显式滚球标记一律丢掉。某个盘筛完没有报价，就留空。",
        "同一选项已经有未停盘的报价时，更早的 stopped 行不算第二口。Tie 与 Draw、队名与 1/2 先归一成同一选项。",
        "带 --refresh 时，这场的记录整份替换，这次响应里没有的盘不会留下。不带 --refresh 时，parser_version 不是当前解析器的记录会用缓存里的原始赔率再解析。",
        "raw_implied_* 是去水位之前的平均 1/小数赔率。Shin 和幂去水位要用它。"
        "只有去水位概率、没有 raw_implied_* 的完场比赛会再请求一次。",
        "同一批比赛还会再请求一次其他赛前盘："
        + "、".join(f"{market_id} {name}" for market_id, name in MARKET_NAMES.items() if market_id != 1)
        + f"。过滤条件 {EXTENDED_FILTER}。",
        "这些盘写到 processed/odds_quotes.json，每一轮请求结束时写一次，不按场重写。缺某个盘就留空，不影响 1X2 和 xG。",
        "开盘只在同一选项真有两个时间戳时才有。一行一个选项、只有 latest_bookmaker_update 时，开盘不可用，收盘就是这一口。不同选项时间不同不算历史。",
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
    quotes_root=None,
    replace_ids: set[int] | None = None,
) -> str | None:
    if not fixture_ids:
        return None
    lookup = _fixture_lookup(frame)
    total = len(fixture_ids)
    seen = 0
    quote_store = load_quotes(quotes_root) if quotes_root is not None and kind == "odds" else None
    try:
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
                    if quote_store is not None:
                        meta = lookup.get(fixture_key)
                        if meta is not None:
                            absorb_fixture_odds(
                                quotes_root,
                                payload,
                                kickoff=meta.kickoff,
                                home_id=meta.home_id,
                                away_id=meta.away_id,
                                home_name=meta.home_name,
                                away_name=meta.away_name,
                                store=quote_store,
                                flush=False,
                                replace=fixture_key in (replace_ids or set()),
                            )
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
    finally:
        if quote_store is not None:
            save_quotes(quotes_root, quote_store)


def _pull_extended(
    client: SportMonksClient,
    root,
    frame: pd.DataFrame,
    fixture_ids: list[int],
    *,
    chunk_size: int,
) -> tuple[str | None, int]:
    """Second odds pass for the markets other than 1X2. A 400 or empty book is not fatal."""

    if not fixture_ids:
        return None, 0
    lookup = _fixture_lookup(frame)
    quote_store = load_quotes(root)
    try:
        note, saved = _pull_extended_chunks(
            client, root, fixture_ids, lookup=lookup, quote_store=quote_store, chunk_size=chunk_size
        )
        if note is None:
            _stamp_parser_version(quote_store, fixture_ids)
        return note, saved
    finally:
        save_quotes(root, quote_store)


def _pull_extended_chunks(client, root, fixture_ids, *, lookup, quote_store, chunk_size: int):
    saved = 0
    note = None
    total = len(fixture_ids)
    seen = 0
    for chunk in _chunks(fixture_ids, chunk_size):
        seen += len(chunk)
        logger.info("回填扩展盘 %s/%s 场", seen, total)
        try:
            payloads = list(
                client.paginate(
                    "fixtures/multi/" + ",".join(str(item) for item in chunk),
                    {"include": "odds", "filters": EXTENDED_FILTER},
                )
            )
        except SportMonksError as exc:
            if _permission_denied(exc):
                note = "扩展盘口请求被拒绝（HTTP 403）。1X2 和 xG 已经写回，扩展盘留空。"
                break
            note = f"扩展盘口这次没有拿到（{exc}）。1X2 不受影响。"
            continue
        for payload in payloads:
            if not isinstance(payload, dict):
                continue
            try:
                fixture_key = int(payload.get("id"))
            except (TypeError, ValueError):
                continue
            meta = lookup.get(fixture_key)
            if meta is None:
                continue
            if absorb_fixture_odds(
                root,
                payload,
                kickoff=meta.kickoff,
                home_id=meta.home_id,
                away_id=meta.away_id,
                home_name=meta.home_name,
                away_name=meta.away_name,
                store=quote_store,
                flush=False,
                replace=False,
            ):
                saved += 1
    return note, saved


def _apply_payload(frame: pd.DataFrame, lookup: dict[int, "_FixtureMeta"], payload: dict, kind: str) -> bool:
    fixture_id = payload.get("id")
    try:
        fixture_key = int(fixture_id)
    except (TypeError, ValueError):
        return False
    meta = lookup.get(fixture_key)
    if meta is None:
        return False
    index, home_id, away_id, kickoff = meta.index, meta.home_id, meta.away_id, meta.kickoff
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


@dataclass(frozen=True)
class _FixtureMeta:
    index: int
    home_id: int
    away_id: int
    kickoff: pd.Timestamp | None
    home_name: str
    away_name: str


def _fixture_lookup(frame: pd.DataFrame) -> dict[int, _FixtureMeta]:
    lookup: dict[int, _FixtureMeta] = {}
    for index, row in frame.iterrows():
        try:
            fixture_id = int(row["fixture_id"])
            home_id = int(row["home_team_id"])
            away_id = int(row["away_team_id"])
        except (TypeError, ValueError):
            continue
        kickoff = row["starting_at"]
        stamp = None if pd.isna(kickoff) else pd.Timestamp(kickoff)
        lookup[fixture_id] = _FixtureMeta(
            int(index),
            home_id,
            away_id,
            stamp,
            _team_text(row, "home_team_name"),
            _team_text(row, "away_team_name"),
        )
    return lookup


def _team_text(row, name: str) -> str:
    if name not in getattr(row, "index", ()):
        return ""
    value = row[name]
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return ""
    return str(value)


def _stale_quote_ids(frame: pd.DataFrame, quotes: dict) -> set[int]:
    """Records parsed by an older parser, including records with no version."""

    stale: set[int] = set()
    if "fixture_id" not in frame.columns:
        return stale
    for fixture_id in frame["fixture_id"].tolist():
        try:
            key = int(fixture_id)
        except (TypeError, ValueError):
            continue
        record = quotes.get(str(key))
        if not isinstance(record, dict):
            continue
        try:
            version = int(record.get("parser_version") or 0)
        except (TypeError, ValueError):
            version = 0
        if version != PARSER_VERSION:
            stale.add(key)
    return stale


def _stamp_parser_version(store: dict, fixture_ids: list[int]) -> None:
    for fixture_id in fixture_ids:
        record = store.get(str(fixture_id))
        if isinstance(record, dict):
            record["parser_version"] = PARSER_VERSION


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
