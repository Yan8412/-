"""Shanghai session clock and SSE weekday closures for 2024–2026.

China has had no daylight-saving time since 1991, so Asia/Shanghai is a fixed
UTC+8 offset. ``zoneinfo`` is not used: Windows CPython does not ship the IANA
database unless ``tzdata`` is installed.

Weekends never trade. Office makeup days that fall on Saturday or Sunday do
not open the exchange. The closure table covers 2024–2026 only. Extend it when
the exchange publishes the next year; until then, later weekdays are treated
as sessions.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone

SHANGHAI = timezone(timedelta(hours=8))
SESSION_CLOSE = time(15, 30)

# Inclusive closure ranges. Weekends inside a range are ignored.
# 2024 is the published schedule used while this table was written.
# 2025: 上证公告〔2024〕38号.
# 2026: 上证公告〔2025〕45号, plus the 2026 mid-autumn / national-day notice.
_CLOSURE_RANGES: tuple[tuple[date, date], ...] = (
    (date(2024, 1, 1), date(2024, 1, 1)),
    (date(2024, 2, 9), date(2024, 2, 16)),
    (date(2024, 4, 4), date(2024, 4, 5)),
    (date(2024, 5, 1), date(2024, 5, 5)),
    (date(2024, 6, 10), date(2024, 6, 10)),
    (date(2024, 9, 15), date(2024, 9, 17)),
    (date(2024, 10, 1), date(2024, 10, 7)),
    (date(2025, 1, 1), date(2025, 1, 1)),
    (date(2025, 1, 28), date(2025, 2, 4)),
    (date(2025, 4, 4), date(2025, 4, 6)),
    (date(2025, 5, 1), date(2025, 5, 5)),
    (date(2025, 5, 31), date(2025, 6, 2)),
    (date(2025, 10, 1), date(2025, 10, 8)),
    (date(2026, 1, 1), date(2026, 1, 3)),
    (date(2026, 2, 15), date(2026, 2, 23)),
    (date(2026, 4, 4), date(2026, 4, 6)),
    (date(2026, 5, 1), date(2026, 5, 5)),
    (date(2026, 6, 19), date(2026, 6, 21)),
    (date(2026, 9, 25), date(2026, 9, 27)),
    (date(2026, 10, 1), date(2026, 10, 7)),
)


def _weekday_closures() -> frozenset[date]:
    closed: set[date] = set()
    for start, end in _CLOSURE_RANGES:
        cursor = start
        while cursor <= end:
            if cursor.weekday() < 5:
                closed.add(cursor)
            cursor += timedelta(days=1)
    return frozenset(closed)


CLOSED_WEEKDAYS = _weekday_closures()


def is_trading_day(day: date) -> bool:
    return day.weekday() < 5 and day not in CLOSED_WEEKDAYS


def previous_trading_day(day: date) -> date:
    cursor = day - timedelta(days=1)
    while not is_trading_day(cursor):
        cursor -= timedelta(days=1)
    return cursor


def latest_completed_session(moment: datetime | None = None) -> date:
    """Last session whose close (15:30 Asia/Shanghai) is already in the past.

    On a trading day at or after 15:30, that day is the session. Before the
    close, and on weekends and holidays, it is the previous trading day.
    """
    if moment is None:
        local = datetime.now(SHANGHAI)
    elif moment.tzinfo is None:
        local = moment.replace(tzinfo=SHANGHAI)
    else:
        local = moment.astimezone(SHANGHAI)
    session = local.date()
    if is_trading_day(session) and local.time() >= SESSION_CLOSE:
        return session
    return previous_trading_day(session)
