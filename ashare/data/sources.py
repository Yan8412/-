"""Free daily-bar and snapshot sources.

Primary path is Tencent (does not use Eastmoney). Sina is the fallback for
both the universe list and daily bars. Eastmoney push2his returned an empty
reply when this project was built (2026-09), so it is not called.

When the cache is exactly one completed session behind and that session is
today (15:30 Asia/Shanghai), ``refresh_many`` appends the close from Sina and
Tencent batch quotes instead of one kline request per name.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime

from ashare.data.cache import append_bar, cache_covers, peek_cache
from ashare.data.http_client import RateLimiter, fetch_json
from ashare.data.spot import (
    APPEND,
    CORPORATE,
    SUSPEND,
    SpotFetch,
    bar_from_spot,
    bulk_close_available,
    classify_spot,
    fetch_spot_quotes,
    is_one_session_behind,
)
from ashare.market import RawBar
from ashare.market_calendar import previous_trading_day
from ashare.rules.adjust import parse_corporate_action
from ashare.rules.limits import classify_board, is_risk_name, normalize_code

logger = logging.getLogger(__name__)

# Recent gap filled with one short request instead of the full history.
TAIL_BARS = 40
TAIL_GAP_DAYS = 40

TENCENT_URLS = (
    "https://proxy.finance.qq.com/ifzqgtimg/appstock/app/newfqkline/get?param={symbol},day,,,{count},",
    "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={symbol},day,,,{count},",
)
SINA_KLINE = (
    "https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/"
    "CN_MarketData.getKLineData?symbol={symbol}&scale=240&ma=no&datalen={count}"
)
SINA_UNIVERSE = (
    "https://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/"
    "Market_Center.getHQNodeData?page={page}&num={num}&sort=amount&asc=0&node=hs_a"
)
SINA_NODE = (
    "https://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/"
    "Market_Center.getHQNodeData?page={page}&num={num}&sort=symbol&asc=1&node={node}"
)


def tencent_symbol(code: str) -> str:
    symbol = normalize_code(code)
    board = classify_board(symbol)
    if board == "bj":
        return "bj" + symbol
    if symbol.startswith(("6", "9")):
        return "sh" + symbol
    return "sz" + symbol


class MarketData:
    def __init__(self, cache_dir, tencent_interval: float = 0.2, sina_interval: float = 0.15) -> None:
        from pathlib import Path

        self.cache_dir = Path(cache_dir)
        self.tencent_limiter = RateLimiter(tencent_interval)
        self.sina_limiter = RateLimiter(sina_interval)
        self.notes: list[str] = []
        self.refresh_note = ""

    def get_bars(self, code: str, count: int, today: date, now: datetime | None = None) -> list[RawBar]:
        from ashare.data.cache import cache_is_fresh, load_bars, merge_bars, save_bars
        from ashare.market_calendar import latest_completed_session

        symbol = normalize_code(code)
        cached, fetched_at = load_bars(self.cache_dir, symbol)
        minimum = min(count, 60)
        if cache_is_fresh(cached, fetched_at, today, minimum, now=now):
            return cached[-count:]
        if cached:
            expected = latest_completed_session(now)
            gap = (expected - cached[-1].date).days
            if len(cached) >= minimum and 1 <= gap <= TAIL_GAP_DAYS:
                downloaded = self._download(symbol, TAIL_BARS, minimum=1)
                if downloaded is not None:
                    tail, source = downloaded
                    tail = sorted(tail, key=lambda bar: bar.date)
                    if tail and tail[0].date <= cached[-1].date:
                        merged = merge_bars(cached, tail)[-count:]
                        save_bars(self.cache_dir, symbol, merged, source)
                        return merged
        downloaded = self._download(symbol, count, minimum=30)
        if downloaded is None:
            raise RuntimeError(f"{symbol} 日线获取失败 (无可用数据源)")
        bars, source = downloaded
        kept = bars[-count:]
        save_bars(self.cache_dir, symbol, kept, source)
        return kept

    def list_liquid_names(
        self,
        limit: int,
        price_min: float,
        price_max: float,
        pages: int = 25,
        page_size: int = 80,
    ) -> list[tuple[str, str]]:
        """Amount-ranked A-shares inside the price band a 20,000 CNY account can lot."""
        found: list[tuple[str, str, float]] = []
        seen: set[str] = set()
        for page in range(1, pages + 1):
            url = SINA_UNIVERSE.format(page=page, num=page_size)
            try:
                payload = fetch_json(url, self.sina_limiter)
            except Exception as exc:  # noqa: BLE001
                self.notes.append(f"新浪股票列表第 {page} 页失败: {exc}")
                break
            if not payload:
                break
            if not isinstance(payload, list):
                self.notes.append("新浪股票列表返回了非列表数据")
                break
            for row in payload:
                code = str(row.get("code") or "")
                name = str(row.get("name") or "")
                try:
                    symbol = normalize_code(code)
                    price = float(row.get("trade") or 0)
                    amount = float(row.get("amount") or 0)
                    volume = float(row.get("volume") or 0)
                except (TypeError, ValueError):
                    continue
                if symbol in seen or is_risk_name(name):
                    continue
                board = classify_board(symbol)
                if board in {"b_share", "other"}:
                    continue
                if volume <= 0 or amount <= 0:
                    continue
                if price < price_min or price > price_max:
                    continue
                seen.add(symbol)
                found.append((symbol, name, amount))
            if len(found) >= limit * 2:
                break
        found.sort(key=lambda item: item[2], reverse=True)
        if not found:
            raise RuntimeError("股票池为空。新浪行情列表没有返回可用标的。")
        return [(code, name) for code, name, _amount in found[: limit * 2]]

    def list_node(self, node: str, page_size: int = 100, page_cap: int = 80) -> list[tuple[str, str]]:
        """Every name Sina returns for one market-center node. Empty on failure."""
        found: list[tuple[str, str]] = []
        seen: set[str] = set()
        for page in range(1, page_cap + 1):
            url = SINA_NODE.format(page=page, num=page_size, node=node)
            try:
                payload = fetch_json(url, self.sina_limiter)
            except Exception as exc:  # noqa: BLE001
                self.notes.append(f"新浪节点 {node} 第 {page} 页失败: {exc}")
                break
            if not isinstance(payload, list) or not payload:
                break
            fresh = 0
            for row in payload:
                try:
                    symbol = normalize_code(str(row.get("code") or ""))
                except ValueError:
                    continue
                if symbol in seen:
                    continue
                board = classify_board(symbol)
                if board in {"b_share", "other"}:
                    continue
                seen.add(symbol)
                found.append((symbol, str(row.get("name") or symbol)))
                fresh += 1
            if fresh == 0 or len(payload) < page_size:
                break
        return found

    def refresh_many(
        self,
        codes: list[str],
        count: int,
        today: date,
        workers: int = 4,
        now: datetime | None = None,
    ) -> dict[str, str]:
        """Download stale names and drop the bars. Returns code to error message.

        A fresh file is only peeked (length, last date, fetch time, last close).
        When the latest completed session is today and a file ends on the
        previous session, that one bar comes from a batch quote. Every other
        gap still uses the per-stock kline. Downloaded bars are written and
        discarded, so a full-market pass does not keep every history in memory.
        """
        errors: dict[str, str] = {}
        pending: list[str] = []
        one_behind: list[tuple[str, float]] = []
        fresh = 0
        minimum = min(count, 60)
        allowed, session, shanghai_day = bulk_close_available(now)
        previous = previous_trading_day(session) if allowed else None
        for code in codes:
            try:
                symbol = normalize_code(code)
            except ValueError as exc:
                errors[code] = str(exc)
                continue
            length, last, fetched_at, last_close = peek_cache(self.cache_dir, symbol)
            if cache_covers(length, last, fetched_at, minimum, now=now):
                fresh += 1
                continue
            if (
                allowed
                and previous is not None
                and length >= minimum
                and is_one_session_behind(last, session)
                and last_close is not None
                and last_close > 0
            ):
                one_behind.append((symbol, last_close))
            else:
                pending.append(symbol)

        written = {"sina": 0, "tencent": 0, "akshare": 0}
        suspended = 0
        corporate = 0
        missed = 0
        if one_behind:
            try:
                fallback, suspended, corporate, missed = self._append_spot(one_behind, session, written)
            except Exception as exc:  # noqa: BLE001
                logger.warning("批量快照失败，这批改为逐只: %s", exc)
                fallback = [code for code, _close in one_behind]
                suspended = 0
                corporate = 0
                missed = len(fallback)
                written.clear()
                written.update({"sina": 0, "tencent": 0, "akshare": 0})
            pending.extend(fallback)
        else:
            fallback = []
        gap_count = len(pending) - len(fallback)
        self.refresh_note = _refresh_note(
            fresh,
            written,
            suspended,
            corporate,
            missed,
            gap_count,
            len(pending),
            allowed,
            session,
            shanghai_day,
        )
        logger.info("%s", self.refresh_note)
        if not pending:
            return errors

        def job(symbol: str) -> tuple[str, str | None]:
            try:
                self.get_bars(symbol, count, today, now=now)
                return symbol, None
            except Exception as exc:  # noqa: BLE001
                return symbol, str(exc)

        done = 0
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            for symbol, outcome in pool.map(job, pending):
                if outcome is not None:
                    errors[symbol] = outcome
                done += 1
                if done % 200 == 0 or done == len(pending):
                    logger.info("日线下载进度 %s/%s", done, len(pending))
        return errors

    def _append_spot(
        self,
        one_behind: list[tuple[str, float]],
        session: date,
        written: dict[str, int],
    ) -> tuple[list[str], int, int, int]:
        """Append closes. Returns codes that still need a per-stock download."""
        closes = {code: close for code, close in one_behind}
        fetched: SpotFetch = fetch_spot_quotes(
            list(closes),
            self.sina_limiter,
            self.tencent_limiter,
            session,
        )
        fallback: list[str] = []
        suspended = 0
        corporate = 0
        missed = 0
        previous = previous_trading_day(session)
        for code, cached_close in one_behind:
            spot = fetched.quotes.get(code)
            decision = classify_spot(spot, cached_close, session)
            if decision == SUSPEND:
                suspended += 1
                continue
            if decision == CORPORATE:
                corporate += 1
                fallback.append(code)
                continue
            if decision != APPEND or spot is None:
                missed += 1
                fallback.append(code)
                continue
            try:
                saved = append_bar(
                    self.cache_dir,
                    code,
                    bar_from_spot(spot, session),
                    f"{spot.source}-spot",
                    previous,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("批量快照写入 %s 失败: %s", code, exc)
                saved = False
            if saved:
                written[spot.source] = written.get(spot.source, 0) + 1
            else:
                missed += 1
                fallback.append(code)
        return fallback, suspended, corporate, missed

    def fetch_many(
        self,
        codes: list[str],
        count: int,
        today: date,
        workers: int = 4,
        now: datetime | None = None,
    ) -> dict[str, list[RawBar] | BaseException]:
        """Download daily bars and return them. Prefer :meth:`refresh_many` for a full market."""
        from ashare.data.cache import cache_is_fresh, load_bars

        results: dict[str, list[RawBar] | BaseException] = {}
        pending: list[str] = []
        min_bars = min(count, 60)
        for code in codes:
            try:
                symbol = normalize_code(code)
            except ValueError as exc:
                results[code] = exc
                continue
            cached, fetched_at = load_bars(self.cache_dir, symbol)
            if cache_is_fresh(cached, fetched_at, today, min_bars, now=now):
                results[symbol] = cached[-count:]
            else:
                pending.append(symbol)
        logger.info("日线缓存命中 %s 只，待下载 %s 只", len(results), len(pending))
        if not pending:
            return results

        def job(symbol: str) -> tuple[str, list[RawBar] | BaseException]:
            try:
                return symbol, self.get_bars(symbol, count, today, now=now)
            except Exception as exc:  # noqa: BLE001
                return symbol, exc

        done = 0
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            for symbol, outcome in pool.map(job, pending):
                results[symbol] = outcome
                done += 1
                if done % 200 == 0 or done == len(pending):
                    logger.info("日线下载进度 %s/%s", done, len(pending))
        return results

    def _download(self, code: str, count: int, minimum: int) -> tuple[list[RawBar], str] | None:
        errors: list[str] = []
        for loader, source in (
            (self._tencent_bars, "tencent"),
            (self._sina_bars, "sina"),
        ):
            try:
                bars = loader(code, count)
            except Exception as exc:  # noqa: BLE001 - fallback chain
                errors.append(f"{source}: {exc}")
                logger.warning("行情源 %s 拉取 %s 失败: %s", source, code, exc)
                continue
            if len(bars) < minimum:
                errors.append(f"{source}: 仅返回 {len(bars)} 根K线")
                continue
            return bars, source
        if minimum >= 30:
            detail = "; ".join(errors) or "无可用数据源"
            raise RuntimeError(f"{code} 日线获取失败 ({detail})")
        return None

    def _tencent_bars(self, code: str, count: int) -> list[RawBar]:
        symbol = tencent_symbol(code)
        last_error: Exception | None = None
        for template in TENCENT_URLS:
            url = template.format(symbol=symbol, count=count)
            try:
                payload = fetch_json(url, self.tencent_limiter)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                continue
            bars = _parse_tencent(payload, symbol)
            if bars:
                return bars
            last_error = RuntimeError("腾讯返回空K线")
        raise RuntimeError(str(last_error) if last_error else "腾讯K线不可用")

    def _sina_bars(self, code: str, count: int) -> list[RawBar]:
        symbol = tencent_symbol(code)
        url = SINA_KLINE.format(symbol=symbol, count=count)
        payload = fetch_json(url, self.sina_limiter)
        if not isinstance(payload, list):
            raise RuntimeError("新浪K线返回异常")
        bars: list[RawBar] = []
        for row in payload:
            day = date.fromisoformat(str(row["day"])[:10])
            close = float(row["close"])
            volume = float(row["volume"])
            bars.append(
                RawBar(
                    date=day,
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=close,
                    volume=volume,
                    amount=close * volume,
                )
            )
        bars.sort(key=lambda bar: bar.date)
        return bars


def _parse_tencent(payload: object, symbol: str) -> list[RawBar]:
    if not isinstance(payload, dict):
        return []
    data = payload.get("data") or {}
    node = data.get(symbol) or {}
    rows = node.get("day") or node.get("qfqday") or []
    bars: list[RawBar] = []
    for row in rows:
        if not isinstance(row, list) or len(row) < 6:
            continue
        cash, bonus = 0.0, 0.0
        if len(row) > 6 and isinstance(row[6], dict):
            cash, bonus = parse_corporate_action(row[6])
        close = float(row[2])
        volume_shares = float(row[5]) * 100.0
        amount = close * volume_shares
        if len(row) > 8 and not isinstance(row[8], dict) and row[8] not in ("", None):
            try:
                amount = float(row[8]) * 10_000.0
            except (TypeError, ValueError):
                amount = close * volume_shares
        bars.append(
            RawBar(
                date=date.fromisoformat(str(row[0])[:10]),
                open=float(row[1]),
                high=float(row[3]),
                low=float(row[4]),
                close=close,
                volume=volume_shares,
                amount=amount,
                cash_dividend=cash,
                bonus_ratio=bonus,
            )
        )
    bars.sort(key=lambda bar: bar.date)
    return bars


def _refresh_note(
    fresh: int,
    written: dict[str, int],
    suspended: int,
    corporate: int,
    missed: int,
    gap_count: int,
    per_stock: int,
    allowed: bool,
    session: date,
    shanghai_day: date,
) -> str:
    """One line naming which path served how many names."""
    if not allowed:
        return (
            f"日线路径：最近完成的交易日 {session.isoformat()} 不是上海今天 "
            f"{shanghai_day.isoformat()}，收盘快照不能当作当天收盘，改为逐只。"
            f"缓存已覆盖 {fresh} 只，逐只下载 {per_stock} 只。"
        )
    return (
        f"日线路径：最近交易日 {session.isoformat()} 就是上海今天。"
        f"缓存已覆盖 {fresh} 只，"
        f"新浪快照写入 {written.get('sina', 0)} 只，"
        f"腾讯快照写入 {written.get('tencent', 0)} 只，"
        f"akshare 快照写入 {written.get('akshare', 0)} 只，"
        f"停牌跳过 {suspended} 只，"
        f"逐只下载 {per_stock} 只"
        f"（缺口超过 1 个交易日 {gap_count} 只，"
        f"除权或复权价不一致 {corporate} 只，"
        f"快照没有可用收盘 {missed} 只）。"
    )
