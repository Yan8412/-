"""Daily recommendation list and the research backtest entry points."""

from __future__ import annotations

import logging
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path

from ashare.backtest.engine import BacktestResult, _collect_signals, run_backtest
from ashare.dashboard_data import save_backtest_bundle
from ashare.backtest.walkforward import run_walk_forward
from ashare.broker.base import OrderRequest
from ashare.broker.paper import PaperBroker
from ashare.config import Settings
from ashare.data.sources import MarketData
from ashare.market import SymbolSeries, build_symbol, mark_listing_censorship, master_calendar
from ashare.report import (
    render_backtest_report,
    write_equity_chart,
    write_recommendations_csv,
    write_recommendations_markdown,
)
from ashare.rules.lots import suggest_shares
from ashare.rules.money import money
from ashare.strategies.base import Strategy
from ashare.strategies.library import builtin_strategies

logger = logging.getLogger(__name__)


def load_universe(settings: Settings, today: date, cache_dir: Path) -> tuple[list[SymbolSeries], list[str]]:
    source = MarketData(cache_dir)
    candidates = source.list_liquid_names(
        settings.universe_size, settings.price_min, settings.price_max
    )
    symbols: list[SymbolSeries] = []
    notes = list(source.notes)
    notes.append(
        "主行情源为腾讯日线（不复权，含除权信息）；单只失败时改用新浪日线。"
        "2026-09-27 探测东财 push2his 得到空响应，本次运行没有访问东财。"
        "通达信公开行情端口按 2026-09 的公开记录已不可用，未接入。"
    )
    for code, name in candidates:
        if len(symbols) >= settings.universe_size:
            break
        try:
            bars = source.get_bars(code, settings.history_bars, today)
        except Exception as exc:  # noqa: BLE001
            notes.append(f"{code} {name} 日线失败，已跳过：{exc}")
            continue
        series = build_symbol(code, name, bars)
        if len(series.dates) < settings.min_history_bars + 5:
            notes.append(f"{code} {name} 可用K线不足，已跳过")
            continue
        symbols.append(series)
        logger.info("已加载 %s %s，%s 根K线", code, name, len(series.dates))
    mark_listing_censorship(symbols)
    if not symbols:
        raise RuntimeError("没有下载到任何可用日线，无法生成推荐或回测。")
    return symbols, notes


def latest_common_day(symbols: list[SymbolSeries]) -> date:
    calendar = master_calendar(symbols)
    if not calendar:
        raise RuntimeError("交易日历为空")
    return calendar[-1]


def build_recommendations(
    symbols: list[SymbolSeries],
    settings: Settings,
    day: date,
    strategies: list[Strategy] | None = None,
) -> list[dict]:
    strategies = strategies or builtin_strategies()
    bundled = [(item, item.default_params()) for item in strategies]
    orders = _collect_signals({item.code: item for item in symbols}, day, bundled, settings, None)
    rows: list[dict] = []
    for order in orders:
        series = next(item for item in symbols if item.code == order.code)
        index = series.date_index[day]
        price = float(series.close[index])
        budget = settings.slot_budget(settings.initial_capital, settings.initial_capital)
        shares = suggest_shares(order.code, price, budget, settings, day)
        if shares <= 0:
            continue
        midpoint = (float(order.entry_low) + float(order.entry_high)) / 2.0
        rows.append(
            {
                "code": order.code,
                "name": order.name,
                "strategy": order.strategy_name,
                "strategy_id": order.strategy_id,
                "score": order.score,
                "reason": order.reason,
                "signal_date": day.isoformat(),
                "close": price,
                "entry_low": float(order.entry_low),
                "entry_high": float(order.entry_high),
                "take_profit": float(money(Decimal(str(midpoint)) * (Decimal("1") + Decimal(str(order.take_profit_pct))))),
                "stop_loss": float(money(Decimal(str(midpoint)) * (Decimal("1") - Decimal(str(order.stop_loss_pct))))),
                "take_profit_pct": order.take_profit_pct,
                "stop_loss_pct": order.stop_loss_pct,
                "max_hold_days": order.max_hold_days,
                "shares": shares,
                "budget": float(money(shares * price)),
            }
        )
    return rows


def run_daily(
    settings: Settings,
    today: date,
    cache_dir: Path,
    report_dir: Path,
    paper_path: Path | None,
) -> Path:
    symbols, notes = load_universe(settings, today, cache_dir)
    day = latest_common_day(symbols)
    rows = build_recommendations(symbols, settings, day)
    preface = [
        f"信号日：{day.isoformat()}（使用该日收盘数据，委托目标是之后的第一个交易日开盘）。",
        f"资金按 {settings.initial_capital:.0f} 元、最多 {settings.max_positions} 只、单票不超过净值的 {settings.max_position_pct:.0%} 估算。",
        "买入区间是相对信号日收盘价的容许开盘价。开盘涨停、高开超出上限、低开超出下限都不会成交。",
        "止盈价和止损价按买入区间中位数估算；模拟盘成交后会按真实成交价重算。",
        "以下股票已排除：ST、次新、停牌、收盘涨停、收盘跌停、价格或成交额不适合该资金量的标的。",
    ]
    if notes:
        preface.append("数据备注：" + "；".join(notes[:6]))
    report_dir.mkdir(parents=True, exist_ok=True)
    markdown = report_dir / f"daily_{day.isoformat()}.md"
    csv_path = report_dir / f"daily_{day.isoformat()}.csv"
    write_recommendations_markdown(markdown, rows, preface)
    write_recommendations_csv(csv_path, rows)
    if paper_path is not None and rows:
        broker = PaperBroker(paper_path, settings)
        requests = [
            OrderRequest(
                code=row["code"],
                name=row["name"],
                side="buy",
                shares=row["shares"],
                signal_date=day,
                strategy_id=row["strategy_id"],
                strategy_name=row["strategy"],
                reason=row["reason"],
                entry_low=row["entry_low"],
                entry_high=row["entry_high"],
                take_profit_pct=row["take_profit_pct"],
                stop_loss_pct=row["stop_loss_pct"],
                max_hold_days=row["max_hold_days"],
                score=row["score"],
            )
            for row in rows
        ]
        broker.place_orders(requests)
    return markdown


def run_research(
    settings: Settings,
    today: date,
    cache_dir: Path,
    report_dir: Path,
    start: date | None = None,
) -> Path:
    symbols, notes = load_universe(settings, today, cache_dir)
    calendar = master_calendar(symbols)
    end = calendar[-1]
    start = start or date(end.year - 2, end.month, min(end.day, 28))
    # Keep a warmup buffer inside the downloaded history when the requested
    # start is early enough.
    if start < calendar[0]:
        start = calendar[0]
    strategies = builtin_strategies()
    full: list[BacktestResult] = []
    oos_pairs = []
    for strategy in strategies:
        logger.info("全样本回测 %s", strategy.name)
        result = run_backtest(
            symbols,
            [(strategy, strategy.default_params())],
            settings,
            start,
            end,
        )
        full.append(result)
        logger.info("走步样本外 %s", strategy.name)
        oos, folds = run_walk_forward(symbols, strategy, settings, start, end, calendar)
        oos_pairs.append((oos, folds))
    logger.info("组合回测")
    combined = run_backtest(
        symbols,
        [(item, item.default_params()) for item in strategies],
        settings,
        start,
        end,
        label="五策略组合",
    )
    report_dir.mkdir(parents=True, exist_ok=True)
    chart_names: dict[str, str] = {}
    for result in full + [combined]:
        filename = f"equity_{result.strategy_id}.png"
        if write_equity_chart(result, report_dir / filename, f"{result.strategy_name} 全样本权益"):
            chart_names[f"{result.strategy_name} 全样本"] = filename
    for result, _folds in oos_pairs:
        filename = f"equity_{result.strategy_id}_oos.png"
        if write_equity_chart(result, report_dir / filename, f"{result.strategy_name} 样本外权益"):
            chart_names[f"{result.strategy_name} 样本外"] = filename
    assumptions = [
        f"初始资金 {settings.initial_capital:.0f} 元，最多 {settings.max_positions} 只持仓，单票预算不超过净值的 {settings.max_position_pct:.0%}，且不超过净值的 1/{settings.max_positions}。",
        f"佣金万 {settings.commission_rate * 10000:.2f}，单笔最低 {settings.min_commission:.0f} 元；卖出印花税 {settings.stamp_duty_rate:.3%}（{settings.stamp_duty_change.isoformat()} 之前为 {settings.stamp_duty_rate_legacy:.3%}）；过户费 {settings.transfer_fee_rate:.4%}。",
        f"滑点 {settings.slippage_rate:.3%}，买入更贵、卖出更便宜。",
        "主板/创业板/北交所以 100 股为一手；科创板以 200 股为最小买入单位。",
        "T+1：买入当日不能卖。开盘价达到涨停价的买单作废。最高价仍不超过跌停价的卖单作废并顺延。",
        "止损和止盈同一天都碰到时，按先止损计算。",
        f"样本区间 {start.isoformat()} 至 {end.isoformat()}。价格带 {settings.price_min:.0f}–{settings.price_max:.0f} 元，信号日成交额不低于 {settings.min_amount / 1e8:.2f} 亿元。",
    ]
    body = render_backtest_report(
        full_results=full,
        oos_results=oos_pairs,
        combined=combined,
        notes=notes,
        universe=[(item.code, item.name) for item in symbols],
        assumptions=assumptions,
        chart_names=chart_names,
    )
    path = report_dir / f"backtest_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}.md"
    path.write_text(body, encoding="utf-8")
    save_backtest_bundle(report_dir / "backtest_latest.json", full + [combined], oos_pairs, notes)
    return path


def settle_paper(settings: Settings, cache_dir: Path, paper_path: Path, today: date) -> list[str]:
    """Settle pending paper orders for every session after the signal date."""
    broker = PaperBroker(paper_path, settings)
    snapshot = broker.snapshot()
    codes = {item["code"] for item in snapshot.pending} | {item["code"] for item in snapshot.positions}
    if not codes:
        return ["模拟盘没有持仓或待成交委托。"]
    source = MarketData(cache_dir)
    series_by_code = {}
    notes: list[str] = []
    for code in sorted(codes):
        name = next((item["name"] for item in snapshot.positions + snapshot.pending if item["code"] == code), code)
        bars = source.get_bars(code, settings.history_bars, today)
        series_by_code[code] = build_symbol(code, name, bars)
    pending_dates = [date.fromisoformat(item["signal_date"]) for item in snapshot.pending]
    position_dates = [date.fromisoformat(item["buy_date"]) for item in snapshot.positions]
    origin_dates = pending_dates + position_dates
    if not origin_dates:
        return ["没有需要结算的日期。"]
    start = min(origin_dates) + timedelta(days=1)
    days: set[date] = set()
    for series in series_by_code.values():
        for day in series.dates:
            if start <= day <= today:
                days.add(day)
    if not days:
        return ["信号日之后还没有新的日线，委托保持待成交。请在下一个交易日收盘后再运行 settle。"]
    for day in sorted(days):
        notes.extend(broker.settle(day, series_by_code))
    return notes
