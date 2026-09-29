"""One-session OHLCV from Sina and Tencent batch quotes.

After 15:30 Asia/Shanghai the live quote is that session's close. A few
requests of several hundred codes replace one kline request per name when the
cache is only that session behind. Eastmoney is not called.

``akshare.stock_zh_a_spot`` (Sina market center, not the Eastmoney function)
is used only when both batch endpoints return nothing.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import date, datetime

from ashare.data.http_client import RateLimiter, fetch_text
from ashare.market import RawBar
from ashare.market_calendar import SHANGHAI, latest_completed_session, previous_trading_day
from ashare.rules.limits import classify_board, normalize_code
from ashare.rules.money import to_cents

logger = logging.getLogger(__name__)

# Sina's list endpoint accepts about 800 codes. Stay under that and under
# typical proxy URL limits. Tencent often truncates or rejects very long
# lists, so a failed or short reply is retried in smaller batches.
SINA_BATCH = 500
SINA_FLOOR = 100
TENCENT_BATCH = 200
TENCENT_FLOOR = 50

# A print outside this band versus the quote's own previous close is treated
# as a bad snapshot and refilled from the per-stock kline. New listings can
# move far more than a board limit; a 5x bound still catches scrambled fields.
CLOSE_RATIO_MIN = 0.2
CLOSE_RATIO_MAX = 5.0

APPEND = "append"
SUSPEND = "suspend"
CORPORATE = "corporate"
REJECT = "reject"

SINA_SPOT = "https://hq.sinajs.cn/rn={stamp}&list={symbols}"
TENCENT_SPOT = "https://qt.gtimg.cn/q={symbols}"
SINA_HEADERS = {
    "Referer": "https://finance.sina.com.cn",
    "User-Agent": "Mozilla/5.0",
}
TENCENT_HEADERS = {"User-Agent": "Mozilla/5.0"}


@dataclass(frozen=True)
class SpotQuote:
    code: str
    day: date
    open: float
    high: float
    low: float
    close: float
    volume: float
    amount: float
    prev_close: float
    source: str


@dataclass
class SpotFetch:
    quotes: dict[str, SpotQuote] = field(default_factory=dict)
    sina: int = 0
    tencent: int = 0
    akshare: int = 0


def shanghai_today(now: datetime | None) -> date:
    if now is None:
        return datetime.now(SHANGHAI).date()
    if now.tzinfo is None:
        return now.replace(tzinfo=SHANGHAI).date()
    return now.astimezone(SHANGHAI).date()


def bulk_close_available(now: datetime | None) -> tuple[bool, date, date]:
    """True only when the latest completed session is today's Shanghai date.

    A weekend, a holiday, or a clock before 15:30 still has a completed
    session, but the live quote is not that session's close.
    """
    session = latest_completed_session(now)
    today = shanghai_today(now)
    return session == today, session, today


def is_one_session_behind(last: date | None, session: date) -> bool:
    return last is not None and last == previous_trading_day(session)


def classify_spot(spot: SpotQuote | None, cached_close: float, session: date) -> str:
    """Decide whether a snapshot may be appended.

    Volume of zero on the session date is a suspension: skip it, do not fall
    back. A previous close that is not the cached last close (to the cent) is
    an ex-dividend day or an adjusted cache, and must be refetched per stock
    so the corporate-action fields stay on the bar. Anything else the bulk
    path cannot trust goes back to the per-stock kline.
    """
    if spot is None or spot.day != session:
        return REJECT
    if spot.volume <= 0:
        return SUSPEND
    if cached_close <= 0 or spot.prev_close <= 0:
        return REJECT
    if not _ohlc_ok(spot):
        return REJECT
    ratio = spot.close / spot.prev_close
    if ratio < CLOSE_RATIO_MIN or ratio > CLOSE_RATIO_MAX:
        return REJECT
    if to_cents(spot.prev_close) != to_cents(cached_close):
        return CORPORATE
    return APPEND


def bar_from_spot(spot: SpotQuote, session: date) -> RawBar:
    amount = spot.amount if spot.amount > 0 else spot.close * spot.volume
    return RawBar(
        date=session,
        open=spot.open,
        high=spot.high,
        low=spot.low,
        close=spot.close,
        volume=spot.volume,
        amount=amount,
    )


def fetch_spot_quotes(
    codes: list[str],
    sina_limiter: RateLimiter,
    tencent_limiter: RateLimiter,
    session: date,
) -> SpotFetch:
    """Batch quotes for ``codes``. Missing names are simply absent."""
    wanted: list[str] = []
    seen: set[str] = set()
    for code in codes:
        try:
            symbol = normalize_code(code)
        except ValueError:
            continue
        if symbol not in seen:
            seen.add(symbol)
            wanted.append(symbol)
    result = SpotFetch()
    if not wanted:
        return result

    sina_quotes = _fetch_sina(wanted, sina_limiter)
    result.quotes.update(sina_quotes)
    result.sina = len(sina_quotes)
    missing = [code for code in wanted if code not in result.quotes]
    if missing:
        tencent_quotes = _fetch_tencent(missing, tencent_limiter)
        result.quotes.update(tencent_quotes)
        result.tencent = len(tencent_quotes)
        missing = [code for code in missing if code not in result.quotes]
    # A thin reply is not a usable bulk path. akshare's Sina snapshot is the
    # alternative; a handful of gaps stay on the per-stock kline instead.
    if missing and len(result.quotes) * 2 < len(wanted):
        for code, quote in _akshare_quotes(session).items():
            if code in missing and code not in result.quotes:
                result.quotes[code] = quote
                result.akshare += 1
    logger.info(
        "收盘快照：新浪 %s 只，腾讯 %s 只，akshare %s 只，请求 %s 只",
        result.sina,
        result.tencent,
        result.akshare,
        len(wanted),
    )
    return result


def parse_sina_spot(text: str) -> dict[str, SpotQuote]:
    """Parse ``hq.sinajs.cn`` list payload. Volume is shares, amount is yuan."""
    found: dict[str, SpotQuote] = {}
    for line in text.replace(";", "\n").splitlines():
        matched = _split_assignment(line)
        if matched is None:
            continue
        symbol, body = matched
        if not body or "hq_str_" not in symbol:
            continue
        raw_code = symbol.split("hq_str_", 1)[-1]
        fields = body.split(",")
        if len(fields) < 32:
            continue
        quote = _sina_fields(raw_code, fields)
        if quote is not None:
            found[quote.code] = quote
    return found


def parse_tencent_spot(text: str) -> dict[str, SpotQuote]:
    """Parse ``qt.gtimg.cn`` payload. Volume hands become shares; amount is wan."""
    found: dict[str, SpotQuote] = {}
    for chunk in text.split(";"):
        if "~" not in chunk:
            continue
        parts = [item.strip().strip('"') for item in chunk.split("~")]
        if len(parts) < 38:
            continue
        quote = _tencent_fields(parts)
        if quote is not None:
            found[quote.code] = quote
    return found


def _fetch_sina(codes: list[str], limiter: RateLimiter) -> dict[str, SpotQuote]:
    found: dict[str, SpotQuote] = {}
    retry: list[str] = []
    for batch in _chunks(codes, SINA_BATCH):
        parsed, ok = _one_sina(batch, limiter)
        if not ok:
            retry.extend(batch)
            continue
        found.update(parsed)
        retry.extend(code for code in batch if code not in parsed)
    if retry and SINA_BATCH > SINA_FLOOR:
        for batch in _chunks(retry, SINA_FLOOR):
            parsed, ok = _one_sina(batch, limiter)
            if ok:
                found.update(parsed)
    return found


def _fetch_tencent(codes: list[str], limiter: RateLimiter) -> dict[str, SpotQuote]:
    found: dict[str, SpotQuote] = {}
    retry: list[str] = []
    for batch in _chunks(codes, TENCENT_BATCH):
        parsed, ok = _one_tencent(batch, limiter)
        if not ok:
            retry.extend(batch)
            continue
        found.update(parsed)
        retry.extend(code for code in batch if code not in parsed)
    if retry and TENCENT_BATCH > TENCENT_FLOOR:
        for batch in _chunks(retry, TENCENT_FLOOR):
            parsed, ok = _one_tencent(batch, limiter)
            if ok:
                found.update(parsed)
    return found


def _one_sina(batch: list[str], limiter: RateLimiter) -> tuple[dict[str, SpotQuote], bool]:
    url = SINA_SPOT.format(stamp=int(time.time() * 1000), symbols=_joined(batch))
    try:
        text = fetch_text(
            url, limiter, timeout=12.0, headers=SINA_HEADERS, encoding="gbk", attempts=2
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("新浪批量快照 %s 只失败: %s", len(batch), exc)
        return {}, False
    return parse_sina_spot(text), True


def _one_tencent(batch: list[str], limiter: RateLimiter) -> tuple[dict[str, SpotQuote], bool]:
    url = TENCENT_SPOT.format(symbols=_joined(batch))
    try:
        text = fetch_text(
            url, limiter, timeout=12.0, headers=TENCENT_HEADERS, encoding="gbk", attempts=2
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("腾讯批量快照 %s 只失败: %s", len(batch), exc)
        return {}, False
    return parse_tencent_spot(text), True


def _joined(batch: list[str]) -> str:
    return ",".join(quote_symbol_board(code) for code in batch)


def _sina_fields(raw_code: str, fields: list[str]) -> SpotQuote | None:
    try:
        code = normalize_code(raw_code)
    except ValueError:
        return None
    try:
        day = date.fromisoformat(fields[30][:10])
        open_ = float(fields[1])
        prev = float(fields[2])
        close = float(fields[3])
        high = float(fields[4])
        low = float(fields[5])
        volume = float(fields[8])
        amount = float(fields[9])
    except (TypeError, ValueError):
        return None
    return SpotQuote(code, day, open_, high, low, close, volume, amount, prev, "sina")


def _tencent_fields(parts: list[str]) -> SpotQuote | None:
    raw_code = parts[2]
    if not raw_code:
        head = parts[0]
        marker = head.rfind("_")
        raw_code = head[marker + 1 :] if marker >= 0 else ""
    try:
        code = normalize_code(raw_code)
    except ValueError:
        return None
    stamp = parts[30]
    if len(stamp) < 8 or not stamp[:8].isdigit():
        return None
    try:
        day = date(int(stamp[0:4]), int(stamp[4:6]), int(stamp[6:8]))
        close = float(parts[3])
        prev = float(parts[4])
        open_ = float(parts[5])
        volume = float(parts[6]) * 100.0
        high = float(parts[33])
        low = float(parts[34])
        amount = float(parts[37]) * 10_000.0
    except (TypeError, ValueError):
        return None
    return SpotQuote(code, day, open_, high, low, close, volume, amount, prev, "tencent")


def _akshare_quotes(session: date) -> dict[str, SpotQuote]:
    """Whole-market Sina snapshot. Empty when akshare is missing or fails."""
    try:
        import akshare as ak
    except ImportError:
        logger.info("未安装 akshare，批量快照没有全市场备用。")
        return {}
    try:
        frame = ak.stock_zh_a_spot()
    except Exception as exc:  # noqa: BLE001
        logger.warning("akshare stock_zh_a_spot 失败: %s", exc)
        return {}
    rows = frame.to_dict("records") if hasattr(frame, "to_dict") else []
    found: dict[str, SpotQuote] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        quote = _akshare_row(row, session)
        if quote is not None:
            found[quote.code] = quote
    logger.info("akshare stock_zh_a_spot 返回 %s 只", len(found))
    return found


def _akshare_row(row: dict, session: date) -> SpotQuote | None:
    try:
        code = normalize_code(str(row.get("代码") or ""))
    except ValueError:
        return None
    close = _float(row.get("最新价"))
    prev = _float(row.get("昨收"))
    open_ = _float(row.get("今开"))
    high = _float(row.get("最高"))
    low = _float(row.get("最低"))
    volume = _float(row.get("成交量"))
    amount = _float(row.get("成交额"))
    if close is None or prev is None or volume is None:
        return None
    day = _row_day(row.get("时间戳")) or session
    return SpotQuote(
        code,
        day,
        open_ if open_ is not None else close,
        high if high is not None else close,
        low if low is not None else close,
        close,
        volume,
        amount if amount is not None else 0.0,
        prev,
        "akshare",
    )


def _row_day(value: object) -> date | None:
    text = str(value or "")
    if len(text) >= 10 and text[4] == "-" and text[7] == "-":
        try:
            return date.fromisoformat(text[:10])
        except ValueError:
            return None
    return None


def _float(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:
        return None
    return number


def _ohlc_ok(spot: SpotQuote) -> bool:
    try:
        prices = [to_cents(spot.open), to_cents(spot.high), to_cents(spot.low), to_cents(spot.close)]
    except (ArithmeticError, ValueError):
        return False
    if min(prices) <= 0:
        return False
    open_, high, low, close = prices
    return high >= low and low <= open_ <= high and low <= close <= high


def _chunks(items: list[str], size: int) -> list[list[str]]:
    step = max(1, size)
    return [items[index : index + step] for index in range(0, len(items), step)]


def _split_assignment(line: str) -> tuple[str, str] | None:
    text = line.strip().rstrip(";")
    if "=" not in text:
        return None
    name, raw = text.split("=", 1)
    body = raw.strip()
    if body.startswith('"') and body.endswith('"'):
        body = body[1:-1]
    return name.strip(), body


def quote_symbol_board(code: str) -> str:
    """Prefix used by the batch endpoints. Same rule as Tencent kline symbols."""
    symbol = normalize_code(code)
    if classify_board(symbol) == "bj":
        return "bj" + symbol
    if symbol.startswith(("6", "9")):
        return "sh" + symbol
    return "sz" + symbol
