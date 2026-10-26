"""Ask SportMonks what a live token actually returns for recent La Liga fixtures.

This does not guess. It requests the odds markets used by the market track
and ``include=xGFixture``, then reports which market ids, bookmakers, quote
timestamps, and xG fields came back. A missing add-on becomes a sentence,
not a traceback. The token is never printed.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta, timezone

from laliga.config import LA_LIGA_LEAGUE_ID
from laliga.data.client import SportMonksClient, SportMonksError
from laliga.data.parse import team_expected_goals
from laliga.data.quotes import MARKET_NAMES, ODDS_FILTER, _odd_rows, parse_odds_rows

XG_TYPE_ID = 5304


def probe_recent(client: SportMonksClient, *, days: int = 21, limit: int = 6) -> dict:
    if days < 1 or limit < 1:
        raise SportMonksError("probe 的天数和场数都要是正数。")
    end = datetime.now(timezone.utc).date()
    start = end - timedelta(days=days)
    fixtures = []
    try:
        for payload in client.paginate(
            f"fixtures/between/{start.isoformat()}/{end.isoformat()}",
            {"filters": f"fixtureLeagues:{LA_LIGA_LEAGUE_ID}", "include": "participants;scores;state"},
        ):
            if isinstance(payload, dict) and payload.get("id") is not None:
                fixtures.append(payload)
    except SportMonksError as exc:
        return {"ok": False, "error": str(exc), "fixtures": []}
    fixtures.sort(key=lambda item: str(item.get("starting_at") or ""))
    chosen = fixtures[-limit:]
    if not chosen:
        return {
            "ok": True,
            "error": None,
            "fixtures": [],
            "note": f"{start.isoformat()} 至 {end.isoformat()} 没有返回西甲比赛。",
        }
    ids = [int(item["id"]) for item in chosen]
    odds_payloads, odds_error = _multi(client, ids, {"include": "odds", "filters": ODDS_FILTER})
    xg_payloads, xg_error = _multi(client, ids, {"include": "xGFixture"})
    by_id = {int(item["id"]): item for item in chosen}
    reports = []
    for fixture_id in ids:
        base = by_id[fixture_id]
        home, away = _names(base)
        odds = odds_payloads.get(fixture_id) or {}
        xg = xg_payloads.get(fixture_id) or {}
        kickoff = base.get("starting_at")
        home_id, away_id = _ids(base)
        bundle = parse_odds_rows(_odd_rows(odds), kickoff=kickoff, home_id=home_id or 0, away_id=away_id or 0)
        xg_fields = _xg_summary(xg, home_id, away_id)
        reports.append(
            {
                "fixture_id": fixture_id,
                "starting_at": kickoff,
                "home": home,
                "away": away,
                "markets_seen": bundle.get("markets_seen") or [],
                "market_names": bundle.get("market_names") or {},
                "bookmakers": bundle.get("bookmakers") or [],
                "opening_and_closing": bool(bundle.get("opening_and_closing")),
                "quote_rows": len(_odd_rows(odds)),
                "xg": xg_fields,
            }
        )
    return {
        "ok": odds_error is None or xg_error is None or bool(reports),
        "error": None,
        "odds_error": odds_error,
        "xg_error": xg_error,
        "window": {"start": start.isoformat(), "end": end.isoformat()},
        "fixtures": reports,
        "catalog": {str(key): value for key, value in MARKET_NAMES.items()},
    }


def format_probe(report: dict) -> str:
    if report.get("error") and not report.get("fixtures"):
        return report["error"]
    if report.get("note") and not report.get("fixtures"):
        return report["note"]
    lines = [
        f"探测窗口 {report.get('window', {}).get('start')} 至 {report.get('window', {}).get('end')}，"
        f"{len(report.get('fixtures') or [])} 场最近的西甲比赛。",
        "市场编号对照：" + "，".join(f"{key} {name}" for key, name in (report.get("catalog") or MARKET_NAMES).items()) + "。",
    ]
    if report.get("odds_error"):
        lines.append(f"赔率：{report['odds_error']}")
    if report.get("xg_error"):
        lines.append(f"xG：{report['xg_error']}")
    market_counter: Counter[int] = Counter()
    books: set[int] = set()
    history = 0
    xg_ok = 0
    for fixture in report.get("fixtures") or []:
        for market_id in fixture.get("markets_seen") or []:
            market_counter[int(market_id)] += 1
        books.update(fixture.get("bookmakers") or [])
        history += int(bool(fixture.get("opening_and_closing")))
        if (fixture.get("xg") or {}).get("home") is not None and (fixture.get("xg") or {}).get("away") is not None:
            xg_ok += 1
        names = "、".join(
            f"{market_id} {fixture.get('market_names', {}).get(str(market_id), MARKET_NAMES.get(market_id, ''))}"
            for market_id in fixture.get("markets_seen") or []
        ) or "没有赛前盘"
        xg = fixture.get("xg") or {}
        lines.append(
            f"{fixture.get('starting_at')} {fixture.get('home')} vs {fixture.get('away')}："
            f"盘口 {names}；博彩 {len(fixture.get('bookmakers') or [])} 家；"
            f"报价 {fixture.get('quote_rows')} 条；"
            f"{'有开盘和收盘两个时间' if fixture.get('opening_and_closing') else '没有多时间戳，不能分出开盘和收盘'}；"
            f"xG 主 {xg.get('home')} 客 {xg.get('away')}（字段 {xg.get('fields') or '无'}）。"
        )
    lines.append(
        "汇总：返回过的市场 "
        + ("、".join(f"{market_id}×{count}" for market_id, count in sorted(market_counter.items())) or "无")
        + f"。博彩公司 id：{sorted(books) or '无'}。能分出开盘/收盘的比赛 {history} 场。主客 xG 都有的比赛 {xg_ok} 场。"
    )
    missing = [market_id for market_id in MARKET_NAMES if market_id not in market_counter]
    if missing:
        lines.append("这几场没有返回的市场：" + "、".join(f"{market_id} {MARKET_NAMES[market_id]}" for market_id in missing) + "。")
    lines.append("没有 token 的字段不会被编造。上面只统计这次响应里真正出现的内容。")
    return "\n".join(lines)


def _multi(client: SportMonksClient, fixture_ids: list[int], params: dict) -> tuple[dict[int, dict], str | None]:
    found: dict[int, dict] = {}
    try:
        for payload in client.paginate("fixtures/multi/" + ",".join(str(item) for item in fixture_ids), params):
            if isinstance(payload, dict) and payload.get("id") is not None:
                found[int(payload["id"])] = payload
    except SportMonksError as exc:
        return found, str(exc)
    return found, None


def _names(fixture: dict) -> tuple[str, str]:
    home = away = "?"
    for participant in fixture.get("participants") or []:
        if not isinstance(participant, dict):
            continue
        location = (participant.get("meta") or {}).get("location")
        if location == "home":
            home = participant.get("name") or home
        elif location == "away":
            away = participant.get("name") or away
    return str(home), str(away)


def _ids(fixture: dict) -> tuple[int | None, int | None]:
    home = away = None
    for participant in fixture.get("participants") or []:
        if not isinstance(participant, dict):
            continue
        location = (participant.get("meta") or {}).get("location")
        try:
            team_id = int(participant.get("id"))
        except (TypeError, ValueError):
            continue
        if location == "home":
            home = team_id
        elif location == "away":
            away = team_id
    return home, away


def _xg_summary(payload: dict, home_id: int | None, away_id: int | None) -> dict:
    rows = []
    for key in ("xgfixture", "xGFixture", "expected"):
        chunk = payload.get(key)
        if isinstance(chunk, list):
            rows.extend(item for item in chunk if isinstance(item, dict))
    fields = sorted({str(row.get("type_id")) for row in rows if row.get("type_id") is not None})
    home = away = None
    if home_id is not None and away_id is not None and rows:
        home, away, _home_against, _away_against = team_expected_goals({"xgfixture": rows}, home_id, away_id)
    type_5304 = any(str(row.get("type_id")) == str(XG_TYPE_ID) for row in rows)
    return {"home": home, "away": away, "fields": fields, "type_5304": type_5304, "rows": len(rows)}
