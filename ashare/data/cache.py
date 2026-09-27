"""JSON cache for daily bars. One file per symbol."""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path

from ashare.market import RawBar


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
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def cache_is_fresh(bars: list[RawBar], fetched_at: str | None, today: date, min_bars: int) -> bool:
    """Reuse a download that already covers the latest session we can expect.

    Weekends and the Mid-Autumn / National Day gaps can leave the last bar
    several calendar days behind ``today``. Six days is enough for a normal
    long weekend; older files are refreshed.
    """
    if len(bars) < min_bars or fetched_at is None:
        return False
    last = bars[-1].date
    return (today - last).days <= 6


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
