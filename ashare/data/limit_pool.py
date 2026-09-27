"""Daily limit-up, broken-limit, and limit-down pools.

Tonghuashun's Financial-API is the primary source. akshare's Eastmoney pools
are the fallback. Either one can be missing; the daily run still finishes.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import date, datetime
from pathlib import Path

import requests

from ashare.config import Settings
from ashare.data.tables import write_records
from ashare.market_calendar import SHANGHAI

logger = logging.getLogger(__name__)

THS_BASE = "https://fuyao.aicubes.cn"
THS_PATHS = {
    "limit_up": "/api/a-share/special-data/limit-up-pool",
    "limit_down": "/api/a-share/special-data/limit-down-pool",
    "broken": "/api/a-share/special-data/limit-break-pool",
}
PAGE_SIZE = 200
MAX_PAGES = 20


def snapshot_limit_pools(
    cache_dir: Path,
    report_dir: Path,
    day: date,
    settings: Settings,
) -> dict:
    """Write one day's pool and a small JSON summary. Failures become a warning row."""
    env_name = settings.ths_api_key_env or "THS_API_KEY"
    key = os.environ.get(env_name, "").strip()
    if not key and env_name != "HITHINK_FINANCE_API_KEY":
        key = os.environ.get("HITHINK_FINANCE_API_KEY", "").strip()
        if key:
            env_name = "HITHINK_FINANCE_API_KEY"
    source = "none"
    warning = ""
    rows: list[dict] = []
    if not key:
        warning = f"环境变量 {settings.ths_api_key_env or 'THS_API_KEY'} 未设置，跳过同花顺。"
        logger.warning(warning)
    else:
        try:
            rows = _ths_rows(day, key)
            if not rows:
                raise RuntimeError("同花顺三个池都是空的")
            source = "ths"
        except Exception as exc:  # noqa: BLE001
            warning = f"同花顺涨停池失败：{exc}"
            logger.warning("%s 改用 akshare。", warning)
    if source != "ths":
        try:
            rows = _akshare_rows(day)
            source = "akshare"
            if not rows:
                raise RuntimeError("东财涨停池返回空表")
        except Exception as exc:  # noqa: BLE001
            warning = (warning + " " if warning else "") + f"akshare 涨停池失败：{exc}"
            logger.warning(warning)
            source = "none"
            rows = []
    summary = _summarize(day, source, rows, warning.strip())
    if rows:
        stamped = []
        for row in rows:
            item = dict(row)
            item["date"] = day.isoformat()
            item["source"] = source
            stamped.append(item)
        write_records(Path(cache_dir) / "pools" / f"{day.isoformat()}.parquet", stamped)
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / "pool_latest.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def _ths_rows(day: date, key: str) -> list[dict]:
    rows: list[dict] = []
    for kind, path in THS_PATHS.items():
        page = 1
        while page <= MAX_PAGES:
            body = _ths_get(path, key, {"date_ms": _date_ms(day), "page": page, "size": PAGE_SIZE})
            items = _ths_items(body)
            for item in items:
                if not isinstance(item, dict):
                    continue
                reason = item.get("limit_up_reason") or item.get("reason") or ""
                if not reason and item.get("open_times") not in (None, ""):
                    reason = f"开板{item.get('open_times')}次"
                seal = (
                    item.get("limit_up_time")
                    or item.get("first_limit_time")
                    or item.get("last_limit_time")
                    or item.get("limit_down_time")
                    or ""
                )
                rows.append(
                    {
                        "kind": kind,
                        "code": str(item.get("ticker") or item.get("code") or ""),
                        "name": str(item.get("name") or ""),
                        "seal_time": str(seal),
                        "reason": str(reason),
                        "streak": "" if item.get("continue_day_cnt") is None else str(item.get("continue_day_cnt")),
                    }
                )
            if len(items) < PAGE_SIZE:
                break
            page += 1
    return rows


def _ths_get(path: str, key: str, params: dict) -> dict:
    url = THS_BASE + path
    last = "同花顺无响应"
    for attempt in range(4):
        try:
            response = requests.get(
                url,
                params=params,
                headers={"X-api-key": key},
                timeout=20,
            )
        except requests.RequestException as exc:
            last = str(exc)
            time.sleep(2**attempt)
            continue
        if response.status_code == 429:
            last = "HTTP 429"
            logger.warning("同花顺 HTTP 429，%s 秒后重试。", 2**attempt)
            time.sleep(2**attempt)
            continue
        if response.status_code >= 400:
            raise RuntimeError(f"HTTP {response.status_code}")
        try:
            body = response.json()
        except ValueError as exc:
            raise RuntimeError("同花顺返回的不是 JSON") from exc
        code = body.get("code") if isinstance(body, dict) else None
        # The vendor documents rate limits as HTTP 200 with code 4001.
        if code in (4001, "4001"):
            last = "限流 code 4001"
            logger.warning("同花顺限流 code 4001，%s 秒后重试。", 2**attempt)
            time.sleep(2**attempt)
            continue
        if code not in (0, "0"):
            raise RuntimeError(str(body.get("message") if isinstance(body, dict) else code))
        return body if isinstance(body, dict) else {}
    raise RuntimeError(last)


def _ths_items(body: dict) -> list:
    data = body.get("data") if isinstance(body.get("data"), dict) else body
    if not isinstance(data, dict):
        return []
    items = data.get("item")
    if items is None:
        items = data.get("items") or data.get("list") or []
    return items if isinstance(items, list) else []


def _akshare_rows(day: date) -> list[dict]:
    try:
        import akshare as ak
    except ImportError as exc:
        raise RuntimeError("未安装 akshare") from exc
    text = day.strftime("%Y%m%d")
    rows: list[dict] = []
    rows.extend(_frame_rows(ak, "stock_zt_pool_em", {"date": text}, "limit_up"))
    rows.extend(_frame_rows(ak, "stock_zt_pool_zbgc_em", {"date": text}, "broken"))
    rows.extend(_frame_rows(ak, "stock_zt_pool_dtgc_em", {"date": text}, "limit_down"))
    try:
        rows.extend(_frame_rows(ak, "stock_lhb_detail_em", {"start_date": text, "end_date": text}, "lhb"))
    except Exception as exc:  # noqa: BLE001
        logger.warning("龙虎榜没有取到：%s", exc)
    return rows


def _frame_rows(ak, name: str, kwargs: dict, kind: str) -> list[dict]:
    func = getattr(ak, name, None)
    if func is None:
        raise RuntimeError(f"akshare 没有 {name}")
    frame = func(**kwargs)
    if frame is None:
        return []
    records = frame.to_dict(orient="records") if hasattr(frame, "to_dict") else list(frame)
    out: list[dict] = []
    for row in records:
        if not isinstance(row, dict):
            continue
        out.append(
            {
                "kind": kind,
                "code": str(row.get("代码") or row.get("code") or ""),
                "name": str(row.get("名称") or row.get("name") or ""),
                "seal_time": str(
                    row.get("首次封板时间") or row.get("涨停时间") or row.get("最后封板时间") or ""
                ),
                "reason": str(row.get("涨停统计") or row.get("涨停原因") or row.get("上榜原因") or ""),
                "streak": "" if row.get("连板数") is None and row.get("连板") is None else str(row.get("连板数") if row.get("连板数") is not None else row.get("连板")),
            }
        )
    return out


def _summarize(day: date, source: str, rows: list[dict], warning: str) -> dict:
    up = [row for row in rows if row.get("kind") == "limit_up"]
    broken = [row for row in rows if row.get("kind") == "broken"]
    down = [row for row in rows if row.get("kind") == "limit_down"]
    streaks = []
    for row in up:
        try:
            streaks.append(int(float(row.get("streak"))))
        except (TypeError, ValueError):
            continue
    denom = len(up) + len(broken)
    examples = []
    ranked = sorted(up, key=lambda row: _streak(row), reverse=True)
    for row in ranked[:5]:
        examples.append(
            {
                "code": row.get("code") or "",
                "name": row.get("name") or "",
                "seal_time": row.get("seal_time") or "",
                "reason": row.get("reason") or "",
                "streak": row.get("streak") if row.get("streak") != "" else "",
            }
        )
    return {
        "date": day.isoformat(),
        "source": source,
        "limit_up_count": len(up),
        "limit_down_count": len(down),
        "broken_count": len(broken),
        "broken_rate": (len(broken) / denom) if denom else None,
        "max_streak": max(streaks) if streaks else 0,
        "examples": examples,
        "warning": warning,
        "fetched_at": datetime.now(tz=SHANGHAI).isoformat(timespec="seconds"),
    }


def _streak(row: dict) -> int:
    try:
        return int(float(row.get("streak")))
    except (TypeError, ValueError):
        return 0


def _date_ms(day: date) -> int:
    stamp = datetime(day.year, day.month, day.day, tzinfo=SHANGHAI)
    return int(stamp.timestamp() * 1000)
