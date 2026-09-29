"""Bulk close snapshots for a one-session cache gap. No network."""

from datetime import date, datetime, timedelta

from ashare.data.cache import load_bars, save_bars
from ashare.data.http_client import RateLimiter
from ashare.data.sources import MarketData
from ashare.data.spot import (
    APPEND,
    CORPORATE,
    REJECT,
    SUSPEND,
    SpotFetch,
    SpotQuote,
    bar_from_spot,
    classify_spot,
    fetch_spot_quotes,
    parse_sina_spot,
    parse_tencent_spot,
)
from ashare.market import RawBar
from ashare.market_calendar import SHANGHAI


def _bar(day: date, close: float = 10.0) -> RawBar:
    return RawBar(day, close, close + 0.2, close - 0.2, close, 1000, close * 1000)


def _bars_ending(last: date, count: int, close: float = 10.0) -> list[RawBar]:
    bars: list[RawBar] = []
    day = last
    while len(bars) < count:
        if day.weekday() < 5:
            bars.append(_bar(day, close))
        day -= timedelta(days=1)
    bars.reverse()
    return bars


def _clock(moment: str) -> datetime:
    return datetime.fromisoformat(moment).replace(tzinfo=SHANGHAI)


def _quote(
    code: str,
    *,
    day: date = date(2026, 9, 28),
    close: float = 10.3,
    prev: float = 10.0,
    volume: float = 1_000_000,
    amount: float = 10_300_000,
    open_: float = 10.2,
    high: float = 10.4,
    low: float = 10.1,
    source: str = "sina",
) -> SpotQuote:
    return SpotQuote(code, day, open_, high, low, close, volume, amount, prev, source)


def _sina_text(symbol: str = "sh600000", volume: str = "1000000", day: str = "2026-09-28") -> str:
    fields = ["0"] * 32
    fields[0] = "浦发银行"
    fields[1] = "10.20"
    fields[2] = "10.00"
    fields[3] = "10.30"
    fields[4] = "10.40"
    fields[5] = "10.10"
    fields[8] = volume
    fields[9] = "10300000"
    fields[30] = day
    fields[31] = "15:00:00"
    return f'var hq_str_{symbol}="{",".join(fields)}";'


def _tencent_text(code: str = "600000", prefix: str = "sh", volume_hands: str = "10000") -> str:
    fields = ["0"] * 50
    fields[0] = f'v_{prefix}{code}="1'
    fields[1] = "浦发银行"
    fields[2] = code
    fields[3] = "10.30"
    fields[4] = "10.00"
    fields[5] = "10.20"
    fields[6] = volume_hands
    fields[30] = "20260928150003"
    fields[33] = "10.40"
    fields[34] = "10.10"
    fields[37] = "1030"
    fields[-1] = '0"'
    return "~".join(fields)


def test_parsers_read_ohlcv_prev_close_and_units():
    sina = parse_sina_spot(_sina_text() + "\n" + 'var hq_str_sz000001="";')
    assert list(sina) == ["600000"]
    assert sina["600000"].day == date(2026, 9, 28)
    assert sina["600000"].volume == 1_000_000
    assert sina["600000"].amount == 10_300_000
    assert sina["600000"].prev_close == 10.0
    assert sina["600000"].close == 10.3

    tencent = parse_tencent_spot(_tencent_text())
    assert tencent["600000"].volume == 1_000_000
    assert tencent["600000"].amount == 10_300_000
    assert tencent["600000"].source == "tencent"
    assert tencent["600000"].day == date(2026, 9, 28)


def test_classify_spot_suspends_rejects_and_flags_corporate_actions():
    session = date(2026, 9, 28)
    assert classify_spot(_quote("600000"), 10.0, session) == APPEND
    assert classify_spot(_quote("600000"), 10.004, session) == APPEND
    assert classify_spot(_quote("600000", volume=0), 10.0, session) == SUSPEND
    assert classify_spot(_quote("600000", volume=0, day=date(2026, 9, 24)), 10.0, session) == REJECT
    assert classify_spot(_quote("600000", prev=9.5), 10.0, session) == CORPORATE
    assert classify_spot(_quote("600000", prev=10.01), 10.0, session) == CORPORATE
    assert classify_spot(_quote("600000", close=80), 10.0, session) == REJECT
    assert classify_spot(_quote("600000", high=10.0, low=10.2), 10.0, session) == REJECT
    assert classify_spot(_quote("600000", day=date(2026, 9, 27)), 10.0, session) == REJECT
    assert classify_spot(None, 10.0, session) == REJECT
    synthesized = bar_from_spot(_quote("600000", amount=0), session)
    assert abs(synthesized.amount - 10.3 * 1_000_000) < 1e-6


def test_fetch_batches_sina_then_fills_gaps_from_tencent(monkeypatch):
    monkeypatch.setattr("ashare.data.spot.SINA_BATCH", 2)
    monkeypatch.setattr("ashare.data.spot.SINA_FLOOR", 2)
    calls: list[str] = []

    def fake_fetch(url, limiter, timeout=20.0, headers=None, encoding=None, attempts=3):
        calls.append(url)
        if "sinajs" in url:
            symbols = url.split("list=", 1)[1].split(",")
            # The second code of the whole request is absent from Sina.
            lines = [_sina_text(symbol) for symbol in symbols if not symbol.endswith("000001")]
            return "\n".join(lines)
        symbol = url.split("q=", 1)[1]
        assert symbol == "sz000001"
        return _tencent_text("000001", "sz")

    monkeypatch.setattr("ashare.data.spot.fetch_text", fake_fetch)
    result = fetch_spot_quotes(
        ["600000", "000001", "300001"],
        RateLimiter(0),
        RateLimiter(0),
        date(2026, 9, 28),
    )
    assert result.sina == 2
    assert result.tencent == 1
    assert result.akshare == 0
    assert result.quotes["000001"].source == "tencent"
    assert sum("sinajs" in url for url in calls) == 2
    assert sum("gtimg" in url for url in calls) == 1


def test_fetch_uses_akshare_when_both_batch_endpoints_fail(monkeypatch):
    def fake_fetch(url, limiter, timeout=20.0, headers=None, encoding=None, attempts=3):
        raise RuntimeError("blocked")

    def fake_ak(session):
        assert session == date(2026, 9, 28)
        return {"600000": _quote("600000", source="akshare")}

    monkeypatch.setattr("ashare.data.spot.fetch_text", fake_fetch)
    monkeypatch.setattr("ashare.data.spot._akshare_quotes", fake_ak)
    result = fetch_spot_quotes(
        ["600000", "000001"],
        RateLimiter(0),
        RateLimiter(0),
        date(2026, 9, 28),
    )
    assert result.akshare == 1
    assert result.quotes["600000"].source == "akshare"
    assert "000001" not in result.quotes


def test_refresh_appends_the_session_from_bulk_quotes(tmp_path, monkeypatch):
    save_bars(tmp_path, "600000", _bars_ending(date(2026, 9, 24), 70), "tencent")
    save_bars(tmp_path, "000001", _bars_ending(date(2026, 9, 24), 70), "tencent")

    def fake_fetch(codes, sina_limiter, tencent_limiter, session):
        assert session == date(2026, 9, 28)
        assert set(codes) == {"600000", "000001"}
        return SpotFetch(
            quotes={
                "600000": _quote("600000"),
                "000001": _quote("000001", source="tencent"),
            },
            sina=1,
            tencent=1,
        )

    def boom(self, code, count, today, now=None):
        raise AssertionError(code)

    monkeypatch.setattr("ashare.data.sources.fetch_spot_quotes", fake_fetch)
    monkeypatch.setattr(MarketData, "get_bars", boom)
    source = MarketData(tmp_path)
    errors = source.refresh_many(
        ["600000", "000001"],
        700,
        date(2026, 9, 28),
        now=_clock("2026-09-28T16:00:00"),
    )
    assert errors == {}
    loaded, _fetched = load_bars(tmp_path, "600000")
    assert len(loaded) == 71
    assert loaded[-1].date == date(2026, 9, 28)
    assert loaded[-2].date == date(2026, 9, 24)
    assert loaded[-1].close == 10.3
    assert loaded[-1].volume == 1_000_000
    assert "新浪快照写入 1 只" in source.refresh_note
    assert "腾讯快照写入 1 只" in source.refresh_note
    assert "逐只下载 0 只" in source.refresh_note
    assert "就是上海今天" in source.refresh_note


def test_refresh_keeps_history_when_appending_one_bar(tmp_path, monkeypatch):
    cached = _bars_ending(date(2026, 9, 30), 70)
    save_bars(tmp_path, "600000", cached, "tencent")

    def fake_fetch(codes, sina_limiter, tencent_limiter, session):
        assert session == date(2026, 10, 8)
        return SpotFetch(quotes={"600000": _quote("600000", day=date(2026, 10, 8))}, sina=1)

    monkeypatch.setattr("ashare.data.sources.fetch_spot_quotes", fake_fetch)
    monkeypatch.setattr(MarketData, "get_bars", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("kline")))
    source = MarketData(tmp_path)
    source.refresh_many(["600000"], 700, date(2026, 10, 8), now=_clock("2026-10-08T16:00:00"))
    loaded, _fetched = load_bars(tmp_path, "600000")
    assert loaded[0].date == cached[0].date
    assert loaded[-2].date == date(2026, 9, 30)
    assert loaded[-1].date == date(2026, 10, 8)


def test_refresh_skips_bulk_when_the_completed_session_is_not_today(tmp_path, monkeypatch):
    save_bars(tmp_path, "600000", _bars_ending(date(2026, 9, 24), 70), "tencent")
    called: list[str] = []

    def fake_fetch(*args, **kwargs):
        raise AssertionError("bulk snapshot")

    def fake_get(self, code, count, today, now=None):
        called.append(code)
        return _bars_ending(date(2026, 9, 28), 70)

    monkeypatch.setattr("ashare.data.sources.fetch_spot_quotes", fake_fetch)
    monkeypatch.setattr(MarketData, "get_bars", fake_get)
    source = MarketData(tmp_path)
    source.refresh_many(["600000"], 700, date(2026, 9, 29), now=_clock("2026-09-29T10:00:00"))
    assert called == ["600000"]
    assert "不是上海今天" in source.refresh_note
    assert "2026-09-28" in source.refresh_note
    assert "2026-09-29" in source.refresh_note


def test_refresh_uses_per_stock_for_a_wider_gap(tmp_path, monkeypatch):
    save_bars(tmp_path, "600000", _bars_ending(date(2026, 9, 23), 70), "tencent")
    called: list[str] = []

    def fake_fetch(*args, **kwargs):
        raise AssertionError("bulk snapshot")

    def fake_get(self, code, count, today, now=None):
        called.append(code)
        return _bars_ending(date(2026, 9, 28), 70)

    monkeypatch.setattr("ashare.data.sources.fetch_spot_quotes", fake_fetch)
    monkeypatch.setattr(MarketData, "get_bars", fake_get)
    source = MarketData(tmp_path)
    source.refresh_many(["600000"], 700, date(2026, 9, 28), now=_clock("2026-09-28T16:00:00"))
    assert called == ["600000"]
    assert "缺口超过 1 个交易日 1 只" in source.refresh_note


def test_refresh_skips_suspended_and_refetches_corporate_actions(tmp_path, monkeypatch):
    save_bars(tmp_path, "600000", _bars_ending(date(2026, 9, 24), 70), "tencent")
    save_bars(tmp_path, "000001", _bars_ending(date(2026, 9, 24), 70, close=10.0), "tencent")
    save_bars(tmp_path, "300001", _bars_ending(date(2026, 9, 24), 70), "tencent")
    called: list[str] = []

    def fake_fetch(codes, sina_limiter, tencent_limiter, session):
        return SpotFetch(
            quotes={
                "600000": _quote("600000", volume=0),
                "000001": _quote("000001", prev=9.5),
                "300001": _quote("300001", close=80, prev=10.0),
            },
            sina=3,
        )

    def fake_get(self, code, count, today, now=None):
        called.append(code)
        bars = _bars_ending(date(2026, 9, 28), 70)
        save_bars(self.cache_dir, code, bars, "tencent")
        return bars

    monkeypatch.setattr("ashare.data.sources.fetch_spot_quotes", fake_fetch)
    monkeypatch.setattr(MarketData, "get_bars", fake_get)
    source = MarketData(tmp_path)
    errors = source.refresh_many(
        ["600000", "000001", "300001"],
        700,
        date(2026, 9, 28),
        now=_clock("2026-09-28T16:00:00"),
    )
    assert errors == {}
    assert called == ["000001", "300001"]
    suspended, _fetched = load_bars(tmp_path, "600000")
    assert suspended[-1].date == date(2026, 9, 24)
    assert "停牌跳过 1 只" in source.refresh_note
    assert "除权或复权价不一致 1 只" in source.refresh_note
    assert "快照没有可用收盘 1 只" in source.refresh_note
    assert "逐只下载 2 只" in source.refresh_note
