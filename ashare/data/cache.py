"""JSON cache for daily bars. One file per symbol."""

from __future__ import annotations

import json
import os
from datetime import date, datetime, timezone
from pathlib import Path

from ashare.market import RawBar
from ashare.market_calendar import SHANGHAI, latest_completed_session


def cache_path(cache_dir: Path, code: str) -> Path:
    return cache_dir / "kline" / f"{code}.json"


def load_bars(cache_dir: Path, code: str) -> tuple[list[RawBar], str | None]:
    path = cache_path(cache_dir, code)
    if not path.exists():
        return [], None
    payload = json.loads(path.read_text(encoding="utf-8"))
    bars = [_bar_from_dict(item) for item in payload.get("bars", [])]
    return bars, payload.get("fetched_at")


def save_bars(cache_dir: Path, code: str, bars: list[RawBar], source: str) -> None:
    path = cache_path(cache_dir, code)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "code": code,
        "source": source,
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "bars": [_bar_to_dict(bar) for bar in bars],
    }
    text = json.dumps(payload, ensure_ascii=False)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def peek_cache(cache_dir: Path, code: str) -> tuple[int, date | None, str | None]:
    """Length, last bar date, and fetch time without building ``RawBar`` objects."""
    path = cache_path(cache_dir, code)
    if not path.exists():
        return 0, None, None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return 0, None, None
    bars = payload.get("bars") or []
    fetched_at = payload.get("fetched_at")
    if not bars or not isinstance(bars[-1], dict):
        return 0, None, fetched_at
    try:
        last = date.fromisoformat(str(bars[-1].get("date"))[:10])
    except (TypeError, ValueError):
        return len(bars), None, fetched_at
    return len(bars), last, fetched_at


def cache_covers(
    count: int,
    last: date | None,
    fetched_at: str | None,
    min_bars: int,
    now: datetime | None = None,
) -> bool:
    """True when the file already contains the latest completed Shanghai session.

    A delisted or long-halted name can sit more than 20 calendar days behind
    that session. If the file was written in the last 7 days it is already the
    full answer. A smaller gap is not fresh: a vendor that is still a few
    minutes late at 15:31 must be retried on the next run.
    """
    if count < min_bars or last is None or fetched_at is None:
        return False
    expected = latest_completed_session(now)
    if last >= expected:
        return True
    if (expected - last).days <= 20:
        return False
    age_days = _fetched_age_days(fetched_at, now)
    return age_days is not None and age_days <= 7


def cache_is_fresh(
    bars: list[RawBar],
    fetched_at: str | None,
    today: date,
    min_bars: int,
    now: datetime | None = None,
) -> bool:
    """Same rule as :func:`cache_covers`, for callers that already loaded bars.

    ``today`` stays in the signature for existing callers. The session clock
    is ``now`` (or the current Asia/Shanghai time), not that calendar date.
    """
    del today
    last = bars[-1].date if bars else None
    return cache_covers(len(bars), last, fetched_at, min_bars, now=now)


def merge_bars(existing: list[RawBar], fresh: list[RawBar]) -> list[RawBar]:
    """Combine two histories. A bar in ``fresh`` replaces the same date."""
    by_date = {bar.date: bar for bar in existing}
    for bar in fresh:
        by_date[bar.date] = bar
    return [by_date[day] for day in sorted(by_date)]


def _fetched_age_days(fetched_at: str, now: datetime | None) -> float | None:
    try:
        fetched = datetime.fromisoformat(fetched_at)
    except ValueError:
        return None
    if fetched.tzinfo is None:
        fetched = fetched.replace(tzinfo=timezone.utc)
    if now is None:
        reference = datetime.now(timezone.utc)
    elif now.tzinfo is None:
        reference = now.replace(tzinfo=SHANGHAI)
    else:
        reference = now
    return (reference.astimezone(timezone.utc) - fetched.astimezone(timezone.utc)).total_seconds() / 86400.0


def _bar_to_dict(bar: RawBar) -> dict:
    return {
        "date": bar.date.isoformat(),
        "open": bar.open,
        "high": bar.high,
        "low": bar.low,
        "close": bar.close,
        "volume": bar.volume,
        "amount": bar.amount,
        "cash_dividend": bar.cash_dividend,
        "bonus_ratio": bar.bonus_ratio,
    }


def _bar_from_dict(item: dict) -> RawBar:
    return RawBar(
        date=date.fromisoformat(item["date"]),
        open=float(item["open"]),
        high=float(item["high"]),
        low=float(item["low"]),
        close=float(item["close"]),
        volume=float(item["volume"]),
        amount=float(item["amount"]),
        cash_dividend=float(item.get("cash_dividend") or 0.0),
        bonus_ratio=float(item.get("bonus_ratio") or 0.0),
    )
