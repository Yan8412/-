"""Optional 1-minute seal stats from eltdx. Missing package or a dead host is a warning.

Configured hosts are tried first, with probing off. If they all fail and eltdx
exposes its packaged server list, the rest of that list is tried the same way
until the overall budget runs out. Auto-selection is only the fallback when
that list cannot be imported.

History depth depends on whatever the TDX host still stores. Measure it on the
machine that will run the book; a cloud probe is not that measurement.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import date, datetime, timedelta
from pathlib import Path

from ashare.config import DEFAULT_TDX_HOSTS
from ashare.data.tables import read_records, write_records
from ashare.market import SymbolSeries
from ashare.rules.limits import classify_board, normalize_code

logger = logging.getLogger(__name__)

# One session of 1-minute bars, plus a little slack.
MINUTE_COUNT = 240
NEAR_LIMIT = 0.98
MAX_NAMES = 200
# The whole optional step, including connects. Daily must not wait longer.
INTRADAY_BUDGET_SECONDS = 180
CONNECT_TIMEOUT = 5.0


def intraday_path(cache_dir: Path, day: date) -> Path:
    return Path(cache_dir) / "intraday" / f"{day.isoformat()}.parquet"


def eltdx_symbol(code: str) -> str:
    symbol = normalize_code(code)
    if classify_board(symbol) == "bj":
        return "bj" + symbol
    if symbol.startswith(("6", "9")):
        return "sh" + symbol
    return "sz" + symbol


def derive_seal(bars: list[dict], limit_up: float) -> dict:
    """First seal, how many times the seal broke, and whether it closed sealed again.

    ``bars`` are chronological dicts with ``time``, ``high``, ``low``, and ``close``.
    A minute that trades at the limit and also prints below it counts as one break.
    """
    first: str | None = None
    broken = 0
    sealed = False
    last_close: float | None = None
    for bar in bars:
        high = float(bar["high"])
        low = float(bar["low"])
        close = float(bar["close"])
        last_close = close
        touched = _at_limit(high, limit_up)
        held = _at_limit(low, limit_up)
        if not touched:
            if sealed and not _at_limit(low, limit_up):
                broken += 1
                sealed = False
            continue
        if first is None:
            first = str(bar.get("time") or "")
        if held and _at_limit(close, limit_up):
            sealed = True
            continue
        broken += 1
        sealed = _at_limit(close, limit_up)
    resealed = broken > 0 and last_close is not None and _at_limit(last_close, limit_up)
    return {
        "first_seal": first or "",
        "broken_seals": broken,
        "resealed": resealed,
    }


def near_limit_codes(symbols: list[SymbolSeries], day: date) -> list[tuple[str, float]]:
    """Codes whose high reached the limit band on ``day``, capped for the request budget."""
    found: list[tuple[str, float]] = []
    for series in symbols:
        index = series.date_index.get(day)
        if index is None:
            continue
        if series.halted is not None and index < len(series.halted) and bool(series.halted[index]):
            continue
        limit = float(series.limit_up[index])
        if limit <= 0:
            continue
        if float(series.high[index]) >= limit * NEAR_LIMIT or float(series.close[index]) >= limit * NEAR_LIMIT:
            found.append((series.code, limit))
    found.sort(key=lambda item: item[0])
    if len(found) > MAX_NAMES:
        logger.warning("涨停附近的股票有 %s 只，分时只请求前 %s 只。", len(found), MAX_NAMES)
        return found[:MAX_NAMES]
    return found


def fetch_intraday(
    cache_dir: Path,
    symbols: list[SymbolSeries],
    day: date,
    hosts: list[str] | None = None,
    budget_seconds: float = INTRADAY_BUDGET_SECONDS,
) -> list[dict]:
    """Store seal and 09:25 auction fields for the limit-up neighborhood. Optional."""
    targets = near_limit_codes(symbols, day)
    if not targets:
        return []
    try:
        from eltdx import TdxClient
    except ImportError:
        logger.warning("未安装 eltdx，跳过首封时间和炸板次数。")
        return load_intraday(cache_dir, day)
    deadline = time.monotonic() + budget_seconds
    records: list[dict] = []
    try:
        client = connect_tdx(hosts if hosts is not None else list(DEFAULT_TDX_HOSTS), deadline, TdxClient)
        try:
            depth = measure_history_depth(client, day, deadline=deadline)
            _write_depth(cache_dir, depth)
            for code, limit in targets:
                if time.monotonic() >= deadline:
                    logger.warning("分时请求超过 %.0f 秒，已停止。", budget_seconds)
                    break
                try:
                    records.append(_one(client, code, day, limit))
                except Exception as exc:  # noqa: BLE001
                    logger.warning("eltdx %s %s 失败：%s", code, day.isoformat(), exc)
        finally:
            _close_client(client)
    except Exception as exc:  # noqa: BLE001
        logger.warning("eltdx 连接失败，跳过分时：%s", exc)
        return load_intraday(cache_dir, day)
    if records:
        write_records(intraday_path(cache_dir, day), records)
    return records


def known_tdx_hosts() -> list[str]:
    """eltdx packaged 7709 list, then its built-in fallback. Empty if eltdx is absent."""
    try:
        from eltdx import hosts as host_module
    except ImportError:
        return []
    packaged: list[str] = []
    loader = getattr(host_module, "load_server_hosts", None)
    if callable(loader):
        try:
            packaged = [str(item) for item in (loader() or [])]
        except Exception as exc:  # noqa: BLE001
            logger.warning("eltdx 服务器列表读取失败：%s", exc)
    fallback = getattr(host_module, "FALLBACK_HOSTS", None) or getattr(host_module, "DEFAULT_HOSTS", ())
    return _unique_hosts([*packaged, *fallback])


def connect_tdx(hosts: list[str], deadline: float, factory):
    """Try configured hosts, then eltdx's known list, all with probing off.

    ``factory`` is ``TdxClient`` or a test double. Each direct connect uses
    ``probe_hosts=False`` and ``server_count=1``. The attempt stops when
    ``deadline`` (the intraday budget) is reached. Auto-selection runs only
    when eltdx did not expose a server list.
    """
    configured = _unique_hosts(hosts)
    extras = [host for host in known_tdx_hosts() if host not in configured]
    candidates = configured + extras
    configured_set = set(configured)
    announced = False
    for host in candidates:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("分时连接超过时间上限")
        if host not in configured_set and not announced:
            logger.info(
                "配置的通达信服务器都失败，改试已知主站 %s 台（不测速），剩余 %.0f 秒",
                len(extras),
                remaining,
            )
            announced = True
        client = None
        try:
            client = factory(
                host=host,
                probe_hosts=False,
                server_count=1,
                timeout=min(CONNECT_TIMEOUT, remaining),
            )
            _enter(client)
            logger.info("eltdx 已连接 %s", host)
            return client
        except Exception as exc:  # noqa: BLE001
            logger.warning("eltdx %s 连接失败：%s", host, exc)
            if client is not None:
                _close_client(client)
    if extras:
        raise ConnectionError("eltdx 已知服务器都连接失败")
    if time.monotonic() >= deadline:
        raise TimeoutError("分时连接超过时间上限")
    logger.warning("固定通达信服务器都失败，改用自动选站。")
    client = factory(timeout=min(CONNECT_TIMEOUT, deadline - time.monotonic()))
    _enter(client)
    return client


def _unique_hosts(values: list[str]) -> list[str]:
    hosts: list[str] = []
    for value in values:
        host = str(value).strip()
        if host and host not in hosts:
            hosts.append(host)
    return hosts


def _enter(client) -> None:
    enter = getattr(client, "__enter__", None)
    if callable(enter):
        enter()


def _close_client(client) -> None:
    try:
        close = getattr(client, "close", None)
        if callable(close):
            close()
            return
        exit_ = getattr(client, "__exit__", None)
        if callable(exit_):
            exit_(None, None, None)
    except Exception as exc:  # noqa: BLE001
        logger.warning("eltdx 关闭连接失败：%s", exc)


def load_intraday(cache_dir: Path, day: date) -> list[dict]:
    return read_records(intraday_path(cache_dir, day))


def annotate_rows(rows: list[dict], records: list[dict]) -> None:
    """Add display fields. Scores stay on the daily-bar rule so the backtest is unchanged."""
    by_code = {str(item.get("code")): item for item in records}
    for row in rows:
        extra = by_code.get(str(row.get("code")))
        if not extra:
            row.setdefault("first_seal", "")
            row.setdefault("broken_seals", "")
            row.setdefault("resealed", "")
            row.setdefault("auction_price", "")
            row.setdefault("auction_volume", "")
            continue
        row["first_seal"] = extra.get("first_seal") or ""
        row["broken_seals"] = extra.get("broken_seals", "")
        row["resealed"] = "是" if extra.get("resealed") else "否"
        price = extra.get("auction_price")
        volume = extra.get("auction_volume")
        row["auction_price"] = "" if price in (None, "") else price
        row["auction_volume"] = "" if volume in (None, "") else volume
        if row.get("strategy_id") == "first_board_follow" or extra.get("first_seal"):
            row["reason"] = f"{row.get('reason', '')} {_fact_text(row)}".strip()


def measure_history_depth(client, day: date, code: str = "sh600000", deadline: float | None = None) -> dict:
    """Walk backward until a history call comes back empty. Best effort."""
    oldest: str | None = None
    checked = 0
    cursor = day
    for _ in range(30):
        if deadline is not None and time.monotonic() >= deadline:
            break
        if cursor.weekday() >= 5:
            cursor -= timedelta(days=1)
            continue
        checked += 1
        try:
            series = client.minutes.history(code, cursor.isoformat())
        except Exception as exc:  # noqa: BLE001
            logger.warning("eltdx 历史分时探测 %s 失败：%s", cursor.isoformat(), exc)
            break
        points = getattr(series, "points", None) or []
        if not points:
            break
        oldest = cursor.isoformat()
        cursor -= timedelta(days=1)
    return {
        "probed_code": code,
        "probed_from": day.isoformat(),
        "oldest_hit": oldest,
        "sessions_hit": checked if oldest else 0,
        "probed_at": datetime.now().isoformat(timespec="seconds"),
    }


def _one(client, code: str, day: date, limit: float) -> dict:
    symbol = eltdx_symbol(code)
    bars = _minute_bars(client, symbol, day)
    facts = derive_seal(bars, limit) if bars else {"first_seal": "", "broken_seals": "", "resealed": False}
    auction_price, auction_volume = _auction(client, symbol, day)
    return {
        "date": day.isoformat(),
        "code": normalize_code(code),
        "first_seal": facts["first_seal"],
        "broken_seals": facts["broken_seals"],
        "resealed": bool(facts["resealed"]),
        "auction_price": auction_price,
        "auction_volume": auction_volume,
        "minute_bars": len(bars),
    }


def _minute_bars(client, symbol: str, day: date) -> list[dict]:
    try:
        series = client.bars.get(symbol, period="1m", count=MINUTE_COUNT)
    except Exception as exc:  # noqa: BLE001
        logger.warning("eltdx 1 分钟 K 线 %s 失败：%s", symbol, exc)
        series = None
    bars = _bars_from_kline(series, day)
    if bars:
        return bars
    try:
        minute = client.minutes.history(symbol, day.isoformat())
    except Exception as exc:  # noqa: BLE001
        logger.warning("eltdx 历史分时 %s 失败：%s", symbol, exc)
        return []
    points = []
    for point in getattr(minute, "points", None) or []:
        price = _num(getattr(point, "price", None))
        if price is None:
            continue
        label = str(getattr(point, "time_label", "") or getattr(point, "time", "") or "")
        points.append({"time": label, "high": price, "low": price, "close": price})
    return points


def _bars_from_kline(series, day: date) -> list[dict]:
    if series is None:
        return []
    out: list[dict] = []
    for bar in getattr(series, "bars", None) or []:
        stamp = getattr(bar, "time", None)
        label = str(stamp or "")
        if label and day.isoformat() not in label and label[:10] != day.isoformat():
            # Some hosts return a bare clock time for the latest session only.
            if len(label) > 8 and "-" in label:
                continue
        high = _num(getattr(bar, "high", None))
        low = _num(getattr(bar, "low", None))
        close = _num(getattr(bar, "close", None))
        if high is None or low is None or close is None:
            continue
        clock = label[11:16] if len(label) >= 16 else label
        out.append({"time": clock, "high": high, "low": low, "close": close})
    return out


def _auction(client, symbol: str, day: date) -> tuple[float | None, float | None]:
    tick = None
    try:
        tick = client.trades.opening_match_history(symbol, day.isoformat())
    except Exception as exc:  # noqa: BLE001
        logger.warning("eltdx 09:25 成交 %s 失败：%s", symbol, exc)
    if tick is not None:
        price = _num(getattr(tick, "price", None))
        volume = _num(getattr(tick, "volume", None))
        return price, volume
    try:
        series = client.auctions.series(symbol, day.isoformat())
    except Exception as exc:  # noqa: BLE001
        logger.warning("eltdx 集合竞价 %s 失败：%s", symbol, exc)
        return None, None
    points = list(getattr(series, "points", None) or [])
    if not points:
        return None, None
    last = points[-1]
    return _num(getattr(last, "price", None)), _num(getattr(last, "matched_volume", None))


def _fact_text(row: dict) -> str:
    reseal = "已回封" if row.get("resealed") == "是" else "未回封"
    auction = row.get("auction_price")
    auction_text = f"，竞价 {auction} 元" if auction not in ("", None) else ""
    volume = row.get("auction_volume")
    if volume not in ("", None):
        auction_text += f" / {volume} 手"
    seal = row.get("first_seal") or "无"
    broken = row.get("broken_seals")
    return f"分时：首封 {seal}，炸板 {broken} 次，{reseal}{auction_text}。"


def _write_depth(cache_dir: Path, depth: dict) -> None:
    path = Path(cache_dir) / "intraday" / "depth.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(depth, ensure_ascii=False, indent=2), encoding="utf-8")


def _at_limit(price: float, limit: float) -> bool:
    return round(price + 1e-8, 2) >= round(limit + 1e-8, 2)


def _num(value) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
