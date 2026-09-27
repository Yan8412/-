"""Point-in-time ST and suspension flags from Baostock.

A daily update asks ``query_all_stock`` once per missing session. The name
(``ST`` / ``*ST`` / ``S*ST``) is that day's ST flag, and ``tradeStatus`` is
the suspension flag. Those rows live in one table and are merged on lookup.

``query_history_k_data_plus`` (``isST``, ``tradestatus``) is only the backfill
for a code with no file or a gap longer than a few weeks, and each run caps
how many of those it requests. A missing package, a failed login, or one bad
code logs a warning and does not stop the rest of the run. Days without a row
are not marked ST: today's name is not used as a substitute.
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from ashare.data.tables import read_records, write_records
from ashare.market import SymbolSeries
from ashare.market_calendar import is_trading_day, latest_completed_session
from ashare.rules.limits import classify_board, limit_prices, normalize_code

logger = logging.getLogger(__name__)

# Warmup before the 2024-09 sample. One range query per name covers it.
HISTORY_START = date(2023, 1, 1)
FIELDS = "date,code,isST,tradestatus"
# A shorter hole is the next session or a holiday gap. Bulk snapshot fills it.
BIG_GAP_DAYS = 20
# Daily runs must not walk the whole market one code at a time.
BACKFILL_CAP = 40
# One call per session. A long outage continues on the next run.
MAX_BULK_DAYS = 15

# cache_dir -> (mtime, code -> day -> (is_st, trading))
_BULK_CACHE: dict[str, tuple[int, dict[str, dict[date, tuple[bool, bool]]]]] = {}


def status_path(cache_dir: Path, code: str) -> Path:
    return Path(cache_dir) / "st" / f"{normalize_code(code)}.parquet"


def baostock_symbol(code: str) -> str:
    symbol = normalize_code(code)
    if classify_board(symbol) == "bj":
        return "bj." + symbol
    if symbol.startswith(("6", "9")):
        return "sh." + symbol
    return "sz." + symbol


def name_is_st(name: str) -> bool:
    """True when the Baostock name is an ST-style label (ST, *ST, S*ST, SST)."""
    folded = (name or "").upper().replace(" ", "").replace("*", "").replace("＊", "")
    return "ST" in folded


def load_status(cache_dir: Path, code: str) -> dict[date, tuple[bool, bool]] | None:
    """Map each cached day to ``(is_st, trading)``. None when nothing is stored.

    Per-code history and the later all-market snapshots are merged. A snapshot
    row for the same day replaces the history row.
    """
    try:
        symbol = normalize_code(code)
    except ValueError:
        return None
    path = status_path(cache_dir, symbol)
    meta = _read_meta(path)
    records = read_records(path)
    bulk = _bulk_index(cache_dir).get(symbol, {})
    if not records and meta is None and not bulk:
        return None
    out: dict[date, tuple[bool, bool]] = {}
    for row in records:
        try:
            day = date.fromisoformat(str(row.get("date"))[:10])
        except ValueError:
            continue
        out[day] = (bool(row.get("is_st")), bool(row.get("trading")))
    out.update(bulk)
    return out


def apply_trading_status(series: SymbolSeries, status: dict[date, tuple[bool, bool]] | None) -> None:
    """Attach daily ST and halt flags, and use a 5% band on main-board ST days.

    ``status`` None leaves an all-false ST mask so a later step does not fill
    the gap from today's name. Halted days stay tradable only via the bar
    itself in that case.
    """
    n = len(series.dates)
    st = np.zeros(n, dtype=bool)
    halted = np.zeros(n, dtype=bool)
    if status:
        for index, day in enumerate(series.dates):
            row = status.get(day)
            if row is None:
                continue
            is_st, trading = row
            st[index] = is_st
            halted[index] = not trading
    series.st_flags = st
    series.halted = halted
    if not st.any():
        return
    for index in np.flatnonzero(st):
        up, down = limit_prices(series.preclose[index], series.code, st=True)
        series.limit_up[index] = float(up)
        series.limit_down[index] = float(down)


def sync_st_history(
    cache_dir: Path,
    codes: list[str],
    today: date,
    now: datetime | None = None,
    backfill_cap: int = BACKFILL_CAP,
) -> dict[str, int]:
    """Fill missing sessions with one market snapshot each, then a capped backfill.

    Returns counts. A data-source failure is a warning, not an exception.
    ``backfill_cap`` limits per-code history queries. ``0`` skips that path.
    """
    del today
    expected = latest_completed_session(now)
    fresh = 0
    tail = 0
    backfill: list[tuple[str, date | None]] = []
    for code in codes:
        try:
            symbol = normalize_code(code)
        except ValueError:
            logger.warning("跳过无法识别的代码 %s", code)
            continue
        last = _last_date(cache_dir, symbol, expected)
        if last is not None and last >= expected:
            fresh += 1
            continue
        if _recently_confirmed_stale(cache_dir, symbol, expected):
            fresh += 1
            continue
        if last is None or (expected - last).days > BIG_GAP_DAYS:
            backfill.append((symbol, last))
        else:
            tail += 1
    watermark = _bulk_watermark(cache_dir)
    if watermark is None:
        watermark = _max_cached_day(cache_dir)
        if watermark is not None:
            _write_bulk_meta(cache_dir, watermark)
    missing_days: list[date] = []
    if watermark is not None and watermark < expected:
        missing_days = _trading_days_after(watermark, expected)
        if len(missing_days) > MAX_BULK_DAYS:
            logger.warning(
                "逐日 ST 缺 %s 个交易日，这次只拉前 %s 天。",
                len(missing_days),
                MAX_BULK_DAYS,
            )
            missing_days = missing_days[:MAX_BULK_DAYS]
    cap = max(0, int(backfill_cap))
    backfill_work = backfill[:cap]
    summary = {
        "fresh": fresh,
        "tail": tail,
        "updated": 0,
        "failed": 0,
        "pending": len(backfill),
        "bulk_days": 0,
        "backfill": 0,
    }
    if not missing_days and not backfill_work:
        logger.info(
            "逐日 ST 已覆盖 %s：最新 %s 只，短缺口 %s 只，待回补 %s 只。",
            expected.isoformat(),
            fresh,
            tail,
            len(backfill),
        )
        return summary
    try:
        import baostock as bs
    except ImportError:
        logger.warning("未安装 baostock，跳过逐日 ST。已有缓存仍会使用。")
        summary["failed"] = len(backfill_work) + len(missing_days)
        return summary
    login = bs.login()
    if getattr(login, "error_code", "1") != "0":
        logger.warning("Baostock 登录失败，跳过逐日 ST：%s", getattr(login, "error_msg", login))
        summary["failed"] = len(backfill_work) + len(missing_days)
        return summary
    wanted = set()
    for code in codes:
        try:
            wanted.add(normalize_code(code))
        except ValueError:
            continue
    try:
        for day in missing_days:
            rows = _query_all(bs, day)
            if rows is None:
                summary["failed"] += 1
                logger.warning("全市场 ST 快照 %s 失败，后面的交易日这次不补。", day.isoformat())
                break
            _store_bulk(cache_dir, day, rows)
            summary["bulk_days"] += 1
            summary["updated"] += sum(1 for row in rows if row["code"] in wanted)
            logger.info("全市场 ST 快照 %s，%s 行。", day.isoformat(), len(rows))
        for index, (symbol, last) in enumerate(backfill_work, start=1):
            start = HISTORY_START if last is None else last
            rows = _query(bs, symbol, start, expected)
            if rows is None:
                bs.logout()
                login = bs.login()
                if getattr(login, "error_code", "1") != "0":
                    logger.warning("Baostock 重新登录失败，剩余逐只 ST 查询停止。")
                    summary["failed"] += len(backfill_work) - index + 1
                    break
                rows = _query(bs, symbol, start, expected)
            if rows is None:
                summary["failed"] += 1
                logger.warning("Baostock 逐日 ST %s 失败，继续下一只。", symbol)
                continue
            _store(cache_dir, symbol, rows, empty_last=start)
            summary["updated"] += 1
            summary["backfill"] += 1
            if index % 200 == 0 or index == len(backfill_work):
                logger.info("逐只 ST 回补 %s/%s", index, len(backfill_work))
    finally:
        try:
            bs.logout()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Baostock 登出失败：%s", exc)
    left = max(0, len(backfill) - summary["backfill"])
    logger.info(
        "逐日 ST 完成：已是最新 %s，短缺口 %s，快照 %s 天，逐只回补 %s，还剩 %s，失败 %s。",
        summary["fresh"],
        tail,
        summary["bulk_days"],
        summary["backfill"],
        left,
        summary["failed"],
    )
    return summary


def _trading_days_after(start: date, end: date) -> list[date]:
    days: list[date] = []
    cursor = start + timedelta(days=1)
    while cursor <= end:
        if is_trading_day(cursor):
            days.append(cursor)
        cursor += timedelta(days=1)
    return days


def _stock_code(raw: str) -> str | None:
    """6-digit A-share code. Shanghai indices (sh.000001) are not stocks."""
    text = (raw or "").strip().lower()
    if "." not in text:
        return None
    market, digits = text.split(".", 1)
    if market not in {"sh", "sz", "bj"} or not digits.isdigit() or len(digits) != 6:
        return None
    if market == "sh" and not digits.startswith(("6", "9")):
        return None
    if market == "sz" and not digits.startswith(("0", "3")):
        return None
    return digits


def _query_all(bs, day: date) -> list[dict] | None:
    try:
        cursor = bs.query_all_stock(day=day.isoformat())
    except Exception as exc:  # noqa: BLE001
        logger.warning("Baostock 全市场快照 %s 异常：%s", day.isoformat(), exc)
        return None
    if getattr(cursor, "error_code", "1") != "0":
        logger.warning(
            "Baostock 全市场快照 %s 失败：%s",
            day.isoformat(),
            getattr(cursor, "error_msg", cursor),
        )
        return None
    fields = list(getattr(cursor, "fields", None) or [])
    rows: list[dict] = []
    while cursor.error_code == "0" and cursor.next():
        raw = cursor.get_row_data()
        if fields:
            data = {name: raw[i] if i < len(raw) else "" for i, name in enumerate(fields)}
        else:
            data = {
                "code": raw[0] if raw else "",
                "tradeStatus": raw[1] if len(raw) > 1 else "",
                "code_name": raw[2] if len(raw) > 2 else "",
            }
        symbol = _stock_code(str(data.get("code") or ""))
        if symbol is None:
            continue
        status = str(data.get("tradeStatus") or data.get("tradestatus") or "")
        rows.append(
            {
                "date": day.isoformat(),
                "code": symbol,
                "is_st": name_is_st(str(data.get("code_name") or "")),
                "trading": status == "1",
            }
        )
    return rows


def _bulk_path(cache_dir: Path) -> Path:
    return Path(cache_dir) / "st" / "bulk.parquet"


def _store_bulk(cache_dir: Path, day: date, rows: list[dict]) -> None:
    path = _bulk_path(cache_dir)
    merged: dict[tuple[str, str], dict] = {}
    for row in read_records(path):
        key = (str(row.get("date") or ""), str(row.get("code") or ""))
        if key[0] and key[1]:
            merged[key] = {
                "date": key[0],
                "code": key[1],
                "is_st": bool(row.get("is_st")),
                "trading": bool(row.get("trading")),
            }
    for row in rows:
        merged[(row["date"], row["code"])] = row
    ordered = [merged[key] for key in sorted(merged)]
    if ordered:
        write_records(path, ordered)
    _write_bulk_meta(cache_dir, day)
    _BULK_CACHE.pop(str(Path(cache_dir).resolve()), None)


def _bulk_index(cache_dir: Path) -> dict[str, dict[date, tuple[bool, bool]]]:
    path = _bulk_path(cache_dir)
    key = str(Path(cache_dir).resolve())
    if not path.exists():
        return {}
    stamp = path.stat().st_mtime_ns
    cached = _BULK_CACHE.get(key)
    if cached is not None and cached[0] == stamp:
        return cached[1]
    index: dict[str, dict[date, tuple[bool, bool]]] = {}
    for row in read_records(path):
        symbol = str(row.get("code") or "")
        try:
            day = date.fromisoformat(str(row.get("date") or "")[:10])
        except ValueError:
            continue
        if not symbol:
            continue
        index.setdefault(symbol, {})[day] = (bool(row.get("is_st")), bool(row.get("trading")))
    _BULK_CACHE[key] = (stamp, index)
    return index


def _bulk_watermark(cache_dir: Path) -> date | None:
    payload = _read_meta(_bulk_path(cache_dir))
    if not payload or not payload.get("last_day"):
        return None
    try:
        return date.fromisoformat(str(payload["last_day"]))
    except ValueError:
        return None


def _write_bulk_meta(cache_dir: Path, day: date) -> None:
    payload = {
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "last_day": day.isoformat(),
    }
    meta = _meta_path(_bulk_path(cache_dir))
    meta.parent.mkdir(parents=True, exist_ok=True)
    meta.write_text(json.dumps(payload), encoding="utf-8")


def _max_cached_day(cache_dir: Path) -> date | None:
    folder = Path(cache_dir) / "st"
    if not folder.exists():
        return None
    latest: date | None = None
    for meta_path in folder.glob("*.meta.json"):
        if meta_path.name == "bulk.meta.json":
            continue
        try:
            payload = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict) or not payload.get("last_date"):
            continue
        try:
            day = date.fromisoformat(str(payload["last_date"]))
        except ValueError:
            continue
        if latest is None or day > latest:
            latest = day
    return latest


def _last_date(cache_dir: Path, code: str, expected: date) -> date | None:
    del expected
    meta = _read_meta(status_path(cache_dir, code))
    if meta and meta.get("last_date"):
        try:
            return date.fromisoformat(str(meta["last_date"]))
        except ValueError:
            pass
    records = read_records(status_path(cache_dir, code))
    last: date | None = None
    for row in records:
        try:
            day = date.fromisoformat(str(row.get("date"))[:10])
        except ValueError:
            continue
        if last is None or day > last:
            last = day
    return last


def _recently_confirmed_stale(cache_dir: Path, code: str, expected: date) -> bool:
    """A long-halted or delisted file fetched this week is not requested again."""
    meta = _read_meta(status_path(cache_dir, code))
    if not meta or not meta.get("last_date") or not meta.get("fetched_at"):
        return False
    try:
        last = date.fromisoformat(str(meta["last_date"]))
        fetched = datetime.fromisoformat(str(meta["fetched_at"]))
    except ValueError:
        return False
    if (expected - last).days <= 20:
        return False
    if fetched.tzinfo is None:
        fetched = fetched.replace(tzinfo=timezone.utc)
    age_days = (datetime.now(timezone.utc) - fetched).total_seconds() / 86400.0
    return age_days <= 7


def _query(bs, code: str, start: date, end: date) -> list[dict] | None:
    try:
        cursor = bs.query_history_k_data_plus(
            baostock_symbol(code),
            FIELDS,
            start_date=start.isoformat(),
            end_date=end.isoformat(),
            frequency="d",
            adjustflag="3",
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Baostock 查询 %s 异常：%s", code, exc)
        return None
    if getattr(cursor, "error_code", "1") != "0":
        message = str(getattr(cursor, "error_msg", cursor))
        logger.warning("Baostock 查询 %s 失败：%s", code, message)
        # A rejected code is not a dead session. Re-login would drop the rest of the batch.
        if "未标识" in message:
            return []
        return None
    fields = list(getattr(cursor, "fields", None) or [])
    rows: list[dict] = []
    while cursor.error_code == "0" and cursor.next():
        raw = cursor.get_row_data()
        data = {name: raw[i] if i < len(raw) else "" for i, name in enumerate(fields)}
        text = str(data.get("date") or "")[:10]
        try:
            day = date.fromisoformat(text).isoformat()
        except ValueError:
            continue
        rows.append(
            {
                "date": day,
                "is_st": str(data.get("isST") or "0") == "1",
                "trading": str(data.get("tradestatus") or "1") == "1",
            }
        )
    return rows


def _store(cache_dir: Path, code: str, rows: list[dict], empty_last: date) -> None:
    path = status_path(cache_dir, code)
    merged: dict[str, dict] = {}
    for row in read_records(path):
        key = str(row.get("date") or "")
        if key:
            merged[key] = {"date": key, "is_st": bool(row.get("is_st")), "trading": bool(row.get("trading"))}
    for row in rows:
        merged[row["date"]] = row
    ordered = [merged[key] for key in sorted(merged)]
    if ordered:
        write_records(path, ordered)
    last = ordered[-1]["date"] if ordered else empty_last.isoformat()
    _write_meta(path, last)


def _meta_path(path: Path) -> Path:
    return path.with_suffix(".meta.json")


def _read_meta(path: Path) -> dict | None:
    meta = _meta_path(path)
    if not meta.exists():
        return None
    try:
        payload = json.loads(meta.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _write_meta(path: Path, last: str | None) -> None:
    payload = {
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "last_date": last,
    }
    meta = _meta_path(path)
    meta.parent.mkdir(parents=True, exist_ok=True)
    meta.write_text(json.dumps(payload), encoding="utf-8")
