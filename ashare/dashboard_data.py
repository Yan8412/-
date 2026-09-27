"""Loaders for the local dashboard. No web framework imports."""

from __future__ import annotations

import csv
import json
from datetime import date
from pathlib import Path

from ashare.backtest.engine import BacktestResult
from ashare.backtest.metrics import summarize


def save_backtest_bundle(
    path: Path,
    full_results: list[BacktestResult],
    oos_results: list[tuple[BacktestResult, list]],
    notes: list[str],
) -> None:
    """Write metrics and equity series so the dashboard can chart them later."""
    payload = {
        "full_sample": [_series_payload(item, "full") for item in full_results],
        "out_of_sample": [_series_payload(item, "oos") for item, _folds in oos_results],
        "notes": notes,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def load_json_object(path: Path) -> dict | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return None
    return payload


def load_backtest_bundle(path: Path) -> dict | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return None
    payload.setdefault("full_sample", [])
    payload.setdefault("out_of_sample", [])
    payload.setdefault("notes", [])
    return payload


def latest_shortlist(report_dir: Path) -> tuple[Path | None, list[dict]]:
    """Return the newest ``daily_*.csv`` and its rows. Empty when none exist."""
    files = sorted(report_dir.glob("daily_*.csv"))
    if not files:
        return None, []
    path = files[-1]
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return path, rows


def load_ledger(path: Path) -> dict | None:
    """Read the paper ledger without creating one."""
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return None
    payload.setdefault("positions", [])
    payload.setdefault("pending", [])
    payload.setdefault("fills", [])
    payload.setdefault("equity", [])
    return payload


def sell_hint(position: dict) -> str:
    """Plain-language exit plan for one open paper position."""
    stop = position.get("stop", "—")
    take = position.get("take_profit", "—")
    held = int(position.get("sessions_held") or 0)
    max_hold = int(position.get("max_hold_days") or 0)
    buy_date = position.get("buy_date") or "未知日期"
    parts = [
        f"止损价 {stop} 元：最低价碰到或低开到这个价就卖。",
        f"止盈价 {take} 元：最高价碰到或高开到这个价就卖。",
    ]
    if held <= 0:
        parts.append(f"买入日是 {buy_date}，当天不能卖（T+1），下一个交易日才能卖。")
    else:
        parts.append(f"买入日是 {buy_date}，已经过了 {held} 个可卖交易日。")
    if max_hold <= 0:
        parts.append("没有设置最长持有天数。")
    elif held >= max_hold:
        parts.append(f"已到最长持有 {max_hold} 天，下一个能成交的收盘价卖出；跌停就顺延。")
    else:
        left = max_hold - held
        parts.append(f"最长持有 {max_hold} 个交易日，还剩 {left} 天，到期当天收盘卖出；跌停就顺延。")
    parts.append("同一天既碰到止损又碰到止盈时，按止损处理。")
    return "".join(parts)


def _series_payload(result: BacktestResult, sample: str) -> dict:
    metrics = summarize(result)
    return {
        "sample": sample,
        "strategy_id": result.strategy_id,
        "strategy_name": result.strategy_name,
        "start": _iso(result.start),
        "end": _iso(result.end),
        "initial_capital": result.initial_capital,
        "metrics": {
            "total_return": metrics.total_return,
            "annualized_return": metrics.annualized_return,
            "win_rate": metrics.win_rate,
            "avg_win": metrics.avg_win,
            "avg_loss": metrics.avg_loss,
            "max_drawdown": metrics.max_drawdown,
            "trade_count": metrics.trade_count,
            "final_equity": metrics.final_equity,
        },
        "equity": [
            {"date": _iso(day), "equity": round(float(value), 2)}
            for day, value in zip(result.equity_dates, result.equity, strict=True)
        ],
    }


def _iso(value: date) -> str:
    return value.isoformat()
