"""Point-in-time ST and suspension flags from Baostock.

``query_history_k_data_plus`` fields ``isST`` and ``tradestatus`` are cached
per code and extended from the last stored day, the same way daily bars are.
A missing package, a failed login, or one bad code logs a warning and does
not stop the rest of the run. Days without a row are not marked ST: the
current name is not used as a substitute.
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from ashare.data.tables import read_records, write_records
from ashare.market import SymbolSeries
from ashare.market_calendar import latest_completed_session
from ashare.rules.limits import classify_board, limit_prices, normalize_code

logger = logging.getLogger(__name__)

# Warmup before the 2024-09 sample. One range query per name covers it.
HISTORY_START = date(2023, 1, 1)
FIELDS = "date,code,isST,tradestatus"


def status_path(cache_dir: Path, code: str) -> Path:
    return Path(cache_dir) / "st" / f"{normalize_code(code)}.parquet"


def baostock_symbol(code: str) -> str:
    symbol = normalize_code(code)
    if classify_board(symbol) == "bj":
        return "bj." + symbol
    if symbol.startswith(("6", "9")):
        return "sh." + symbol
    return "sz." + symbol


def load_status(cache_dir: Path, code: str) -> dict[date, tuple[bool, bool]] | None:
    """Map each cached day to ``(is_st, trading)``. None when the file is absent."""
    path = status_path(cache_dir, code)
    meta = _read_meta(path)
    records = read_records(path)
    if not records and meta is None:
        return None
    out: dict[date, tuple[bool, bool]] = {}
    for row in records:
        try:
            day = date.fromisoformat(str(row.get("date"))[:10])
        except ValueError:
            continue
        out[day] = (bool(row.get("is_st")), bool(row.get("trading")))
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
) -> dict[str, int]:
    """Download missing tails. Returns counts; never raises for a data-source failure."""
    expected = latest_completed_session(now)
    fresh = 0
    pending: list[tuple[str, date | None]] = []
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
        pending.append((symbol, last))
    summary = {"fresh": fresh, "updated": 0, "failed": 0, "pending": len(pending)}
    if not pending:
        logger.info("逐日 ST 缓存已覆盖 %s，命中 %s 只。", expected.isoformat(), fresh)
        return summary
    try:
        import baostock as bs
    except ImportError:
        logger.warning("未安装 baostock，跳过逐日 ST。已有缓存仍会使用。")
        summary["failed"] = len(pending)
        return summary
    login = bs.login()
    if getattr(login, "error_code", "1") != "0":
        logger.warning("Baostock 登录失败，跳过逐日 ST：%s", getattr(login, "error_msg", login))
        summary["failed"] = len(pending)
        return summary
    try:
        for index, (symbol, last) in enumerate(pending, start=1):
            start = HISTORY_START if last is None else last
            rows = _query(bs, symbol, start, expected)
            if rows is None:
                bs.logout()
                login = bs.login()
                if getattr(login, "error_code", "1") != "0":
                    logger.warning("Baostock 重新登录失败，剩余 ST 查询停止。")
                    summary["failed"] += len(pending) - index + 1
                    break
                rows = _query(bs, symbol, start, expected)
            if rows is None:
                summary["failed"] += 1
                logger.warning("Baostock 逐日 ST %s 失败，继续下一只。", symbol)
                continue
            _store(cache_dir, symbol, rows, empty_last=start)
            summary["updated"] += 1
            if index % 200 == 0 or index == len(pending):
                logger.info("逐日 ST 进度 %s/%s", index, len(pending))
    finally:
        try:
            bs.logout()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Baostock 登出失败：%s", exc)
    logger.info(
        "逐日 ST 完成：已是最新 %s，更新 %s，失败 %s。",
        summary["fresh"],
        summary["updated"],
        summary["failed"],
    )
    return summary


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
        logger.warning(
            "Baostock 查询 %s 失败：%s",
            code,
            getattr(cursor, "error_msg", cursor),
        )
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
    _meta_path(path).write_text(json.dumps(payload), encoding="utf-8")
