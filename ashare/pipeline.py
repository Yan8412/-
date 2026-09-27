"""Daily recommendation list and the research backtest entry points."""

from __future__ import annotations

import json
import logging
import math
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path

from ashare.backtest.engine import BacktestResult, _collect_signals, run_backtest
from ashare.dashboard_data import save_backtest_bundle
from ashare.backtest.walkforward import (
    FoldOutcome,
    build_walk_forward_schedules,
    run_oos_backtest,
)
from ashare.broker.base import OrderRequest
from ashare.broker.paper import PaperBroker
from ashare.config import Settings
from ashare.data.cache import load_bars
from ashare.data.sources import MarketData
from ashare.data.st_history import apply_trading_status, load_status, sync_st_history
from ashare.data.universe import coverage_notes, load_catalog, select_for_download
from ashare.market import SymbolSeries, build_symbol, mark_listing_censorship, master_calendar
from ashare.ranker import load_ranker, score_candidate
from ashare.sentiment import MarketPanel, attach_inferred_st, build_market_panel, market_day_payload
from ashare.report import (
    render_backtest_report,
    write_equity_chart,
    write_recommendations_csv,
    write_recommendations_markdown,
)
from ashare.rules.lots import suggest_shares
from ashare.rules.money import money
from ashare.strategies.base import Strategy
from ashare.strategies.library import builtin_strategies, strategies_from_ids

logger = logging.getLogger(__name__)


def load_universe(
    settings: Settings, today: date, cache_dir: Path
) -> tuple[list[SymbolSeries], list[str], list, dict]:
    if settings.full_market:
        return _load_full_market(settings, today, cache_dir)
    symbols, notes = _load_liquid_sample(settings, today, cache_dir)
    return symbols, notes, [], {}


def _load_liquid_sample(settings: Settings, today: date, cache_dir: Path) -> tuple[list[SymbolSeries], list[str]]:
    source = MarketData(cache_dir)
    candidates = source.list_liquid_names(
        settings.universe_size, settings.price_min, settings.price_max
    )
    symbols: list[SymbolSeries] = []
    notes = list(source.notes)
    notes.append(
        "主行情源为腾讯日线（不复权，含除权信息）；单只失败时改用新浪日线。"
        "2026-09-27 探测东财 push2his 得到空响应，本次运行没有访问东财。"
        "日线不走通达信。可选的 eltdx 只在收盘更新里补当天涨停附近的分时。"
    )
    sync_st_history(cache_dir, [code for code, _name in candidates], today)
    for code, name in candidates:
        if len(symbols) >= settings.universe_size:
            break
        try:
            bars = source.get_bars(code, settings.history_bars, today)
        except Exception as exc:  # noqa: BLE001
            notes.append(f"{code} {name} 日线失败，已跳过：{exc}")
            continue
        series = build_symbol(code, name, bars)
        apply_trading_status(series, load_status(cache_dir, code) or {})
        if len(series.dates) < settings.min_history_bars + 5:
            notes.append(f"{code} {name} 可用K线不足，已跳过")
            continue
        symbols.append(series)
        logger.info("已加载 %s %s，%s 根K线", code, name, len(series.dates))
    mark_listing_censorship(symbols)
    if not symbols:
        raise RuntimeError("没有下载到任何可用日线，无法生成推荐或回测。")
    return symbols, notes


def _load_full_market(
    settings: Settings, today: date, cache_dir: Path
) -> tuple[list[SymbolSeries], list[str], list, dict]:
    source = MarketData(cache_dir, tencent_interval=0.12, sina_interval=0.12)
    catalog, notes = load_catalog(cache_dir, today, source)
    notes.extend(source.notes)
    chosen, stats = select_for_download(catalog, today)
    errors = source.refresh_many(
        [item.code for item in chosen],
        settings.history_bars,
        today,
        workers=4,
    )
    codes = [item.code for item in chosen]
    st_summary = sync_st_history(cache_dir, codes, today)
    notes.append(
        "逐日 ST 与停牌：已有文件的短缺口用 Baostock query_all_stock 按天补（名称里的 ST 和 tradeStatus），"
        "没有文件或缺口很大才逐只查 isST。按当天判断，不用今天的名称。"
        f"这次已是最新 {st_summary['fresh']} 只，短缺口 {st_summary.get('tail', 0)} 只，"
        f"全市场快照 {st_summary.get('bulk_days', 0)} 天，逐只回补 {st_summary.get('backfill', 0)} 只，"
        f"失败 {st_summary['failed']} 只。"
        "没有逐日记录的股票不按当前名称整段排除。主板 ST 当日涨跌停按 5%，创业板和科创板仍是 20%，北交所仍是 30%。"
    )
    symbols: list[SymbolSeries] = []
    too_short = 0
    st_missing = 0
    st_names = 0
    for item in chosen:
        failure = errors.get(item.code)
        if failure:
            item.error = failure
            continue
        bars, _fetched_at = load_bars(cache_dir, item.code)
        if not bars:
            item.error = "无数据"
            continue
        item.bars = len(bars)
        item.last_bar = bars[-1].date.isoformat()
        if len(bars) < settings.min_history_bars + 5:
            too_short += 1
            del bars
            continue
        series = build_symbol(item.code, item.name, bars)
        del bars
        status = load_status(cache_dir, item.code)
        if status is None:
            st_missing += 1
            apply_trading_status(series, {})
        else:
            apply_trading_status(series, status)
            if series.st_flags is not None and bool(series.st_flags.any()):
                st_names += 1
        series.ipo_date = item.ipo()
        series.out_date = item.out()
        symbols.append(series)
        if len(symbols) % 500 == 0:
            logger.info("已整理 %s 只日线", len(symbols))
    stats["too_short"] = too_short
    stats["st_skipped"] = 0
    stats["st_missing"] = st_missing
    stats["st_names"] = st_names
    mark_listing_censorship(symbols)
    if not symbols:
        raise RuntimeError("没有下载到任何可用日线，无法生成推荐或回测。")
    notes.append(
        "主行情源为腾讯日线（不复权，含除权信息）；单只失败时改用新浪日线。"
        "退市代码来自 Baostock 的上市/退市日期，日线仍向腾讯请求。"
        "2026-09-27 探测东财 push2his 得到空响应，本次运行没有访问东财。"
    )
    return symbols, notes, chosen, stats


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
    cap: int | None = None,
) -> list[dict]:
    strategies = strategies or builtin_strategies()
    bundled = [(item, item.default_params()) for item in strategies]
    orders = _collect_signals({item.code: item for item in symbols}, day, bundled, settings, None, cap=cap)
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
                "model_score": None,
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
    """Refresh bars, settle an existing paper book, then write the next shortlist.

    ``paper_path`` None (the ``--no-paper`` flag) skips both settlement and new orders.
    New orders are still withheld when the regime filter is off.
    """
    symbols, notes, _chosen, _stats = load_universe(settings, today, cache_dir)
    if paper_path is not None:
        for line in settle_paper(settings, cache_dir, paper_path, today):
            logger.info("%s", line)
    attach_inferred_st(symbols)
    panel = build_market_panel(symbols, settings)
    day = latest_common_day(symbols)
    _fetch_optional_intraday(cache_dir, symbols, day, settings)
    path = publish_daily(symbols, panel, settings, day, report_dir, paper_path, notes, cache_dir)
    _snapshot_optional_pools(cache_dir, report_dir, day, settings)
    return path


def publish_daily(
    symbols: list[SymbolSeries],
    panel: MarketPanel,
    settings: Settings,
    day: date,
    report_dir: Path,
    paper_path: Path | None,
    notes: list[str],
    cache_dir: Path | None = None,
) -> Path:
    """Write the shortlist for one close. Paper orders are skipped when the regime is off."""
    strategies = strategies_from_ids(settings.strategies)
    model = load_ranker() if settings.use_ranker else None
    pool = max(settings.ml_pool, settings.top_n) if settings.use_ranker else settings.top_n
    rows = build_recommendations(symbols, settings, day, strategies, cap=pool)
    rows = _rank_rows(rows, symbols, panel, model, settings)
    if cache_dir is not None:
        from ashare.data.intraday import annotate_rows, load_intraday

        annotate_rows(rows, load_intraday(cache_dir, day))
    info = panel.days.get(day)
    risk_on = bool(info.risk_on) if info is not None else False
    preface = _daily_preface(settings, day, info, model, risk_on, notes, strategies)
    report_dir.mkdir(parents=True, exist_ok=True)
    _write_market_snapshot(report_dir, day, info, model is not None, settings.use_ranker)
    markdown = report_dir / f"daily_{day.isoformat()}.md"
    csv_path = report_dir / f"daily_{day.isoformat()}.csv"
    write_recommendations_markdown(markdown, rows, preface)
    write_recommendations_csv(csv_path, rows)
    if paper_path is not None and rows and risk_on:
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
                score=row["model_score"] if row.get("model_score") is not None else row["score"],
            )
            for row in rows
        ]
        broker.place_orders(requests)
    return markdown


def _rank_rows(
    rows: list[dict],
    symbols: list[SymbolSeries],
    panel: MarketPanel,
    model: dict | None,
    settings: Settings,
) -> list[dict]:
    if model is None:
        for row in rows:
            row["model_score"] = None
        return rows[: settings.top_n]
    by_code = {item.code: item for item in symbols}
    for row in rows:
        series = by_code[row["code"]]
        index = series.date_index[date.fromisoformat(row["signal_date"])]
        row["model_score"] = score_candidate(
            model, series, index, row["strategy_id"], float(row["score"]), panel
        )
    rows.sort(key=lambda item: float(item["model_score"]), reverse=True)
    return rows[: settings.top_n]


def _daily_preface(
    settings: Settings,
    day: date,
    info,
    model: dict | None,
    risk_on: bool,
    notes: list[str],
    strategies: list[Strategy],
) -> list[str]:
    if risk_on:
        regime = "行情过滤：允许开新仓（站上 MA20 的股票不少于设定比例，且等权指数在均线之上）。"
    else:
        regime = "今日不开新仓。候选和分数仍列在下面，供核对，这次不会写入模拟盘。"
    if not settings.use_ranker:
        model_line = "排序模型已关闭，名单按规则分。"
    elif model is None:
        model_line = "模型未训练，排序用的是规则分。训练命令：python -m ashare train。"
    else:
        model_line = f"模型分数来自梯度提升树，训练截止 {model.get('trained_through', '未知')}（该日之前已经平仓的交易）。"
    names = "、".join(item.name for item in strategies)
    preface = [
        f"信号日：{day.isoformat()}（使用该日收盘数据，委托目标是之后的第一个交易日开盘）。",
        f"候选来自{names}。",
        regime,
        _sentiment_line(info),
        model_line,
        f"资金按 {settings.initial_capital:.0f} 元、最多 {settings.max_positions} 只、单票不超过净值的 {settings.max_position_pct:.0%} 估算。",
        "买入区间是相对信号日收盘价的容许开盘价。开盘涨停、高开超出上限、低开超出下限都不会成交。",
        "止盈价和止损价按买入区间中位数估算；模拟盘成交后会按真实成交价重算。",
        "以下股票已排除：信号日当天 Baostock 标记为 ST 的股票、停牌、次新、收盘涨停、收盘跌停、价格或成交额不适合该资金量的标的、近20日振幅不足 10% 的低波动股票。没有逐日文件的股票不按今天的名称整段排除。",
    ]
    if notes:
        preface.append("数据备注：" + "；".join(notes[:6]))
    return preface


def _sentiment_line(info) -> str:
    if info is None:
        return "没有该日的市场状态。"
    prev = "无" if not math.isfinite(info.prev_limit_return) else f"{info.prev_limit_return * 100:.2f}%"
    versus = "无" if not math.isfinite(info.index_vs_ma) else f"{info.index_vs_ma * 100:.2f}%"
    breadth = "无" if not math.isfinite(info.breadth) else f"{info.breadth * 100:.2f}%"
    return (
        f"涨停 {info.limit_up_count} 家，炸板率 {info.broken_rate * 100:.1f}%，"
        f"最高连板 {info.max_height}，昨日涨停今日平均涨跌 {prev}，"
        f"站上 MA20 的比例 {breadth}（{info.breadth_count} 只），等权指数相对均线 {versus}。"
    )


def _write_market_snapshot(
    report_dir: Path, day: date, info, model_ready: bool, use_ranker: bool
) -> None:
    if info is None:
        return
    payload = market_day_payload(day, info)
    payload["model_ready"] = model_ready
    payload["use_ranker"] = use_ranker
    payload["entry_note"] = "允许开新仓" if info.risk_on else "今日不开新仓"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "market_latest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def run_research(
    settings: Settings,
    today: date,
    cache_dir: Path,
    report_dir: Path,
    start: date | None = None,
) -> Path:
    symbols, notes, chosen, stats = load_universe(settings, today, cache_dir)
    calendar = master_calendar(symbols)
    end = calendar[-1]
    start = start or date(end.year - 2, end.month, min(end.day, 28))
    # Keep a warmup buffer inside the downloaded history when the requested
    # start is early enough.
    if start < calendar[0]:
        start = calendar[0]
    if chosen:
        notes.extend(coverage_notes(chosen, stats, start, end, len(symbols)))
    strategies = builtin_strategies()
    full: list[BacktestResult] = []
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
    logger.info("组合全样本")
    combined = run_backtest(
        symbols,
        [(item, item.default_params()) for item in strategies],
        settings,
        start,
        end,
        label="五策略组合",
    )
    logger.info("走步选参（各策略独立，组合复用同一组冻结参数）")
    folds, schedules, outcomes = build_walk_forward_schedules(
        symbols, strategies, settings, start, end, calendar
    )
    oos_pairs: list[tuple[BacktestResult, list[FoldOutcome]]] = []
    for strategy in strategies:
        logger.info("走步样本外 %s", strategy.name)
        oos = run_oos_backtest(
            symbols, [strategy], settings, folds, schedules, start, end
        )
        oos_pairs.append((oos, outcomes[strategy.id]))
    logger.info("组合走步样本外")
    combined_oos = run_oos_backtest(
        symbols,
        strategies,
        settings,
        folds,
        schedules,
        start,
        end,
        label="五策略组合",
    )
    combined_folds = [item for strategy in strategies for item in outcomes[strategy.id]]
    oos_pairs.append((combined_oos, combined_folds))
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
        f"样本区间 {start.isoformat()} 至 {end.isoformat()}。信号日价格带 {settings.price_min:.0f}–{settings.price_max:.0f} 元，成交额不低于 {settings.min_amount / 1e8:.2f} 亿元，近20日振幅不低于 {settings.min_swing_20d:.0%}。",
        "组合样本外不是五条策略参数的全排列。每一折先为每条策略单独选参，再把这五组参数放在同一个账户里交易。",
    ]
    comparison = _baseline_lines(report_dir)
    body = render_backtest_report(
        full_results=full,
        oos_results=oos_pairs,
        combined=combined,
        notes=notes,
        universe=[(item.code, item.name) for item in symbols],
        assumptions=assumptions,
        chart_names=chart_names,
        comparison=comparison,
    )
    path = report_dir / f"backtest_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}.md"
    path.write_text(body, encoding="utf-8")
    save_backtest_bundle(report_dir / "backtest_latest.json", full + [combined], oos_pairs, notes)
    return path


def _fetch_optional_intraday(cache_dir: Path, symbols: list[SymbolSeries], day: date, settings: Settings) -> None:
    try:
        from ashare.data.intraday import fetch_intraday

        fetch_intraday(cache_dir, symbols, day, hosts=list(settings.tdx_hosts))
    except Exception as exc:  # noqa: BLE001
        logger.warning("分时封板特征跳过：%s", exc)


def _snapshot_optional_pools(cache_dir: Path, report_dir: Path, day: date, settings: Settings) -> None:
    try:
        from ashare.data.limit_pool import snapshot_limit_pools

        snapshot_limit_pools(cache_dir, report_dir, day, settings)
    except Exception as exc:  # noqa: BLE001
        logger.warning("涨停池快照跳过：%s", exc)


def _baseline_lines(report_dir: Path) -> list[str]:
    path = report_dir / "baseline_50_sample.md"
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8").strip()
    return [text, ""]


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
        series = build_symbol(code, name, bars)
        apply_trading_status(series, load_status(cache_dir, code) or {})
        series_by_code[code] = series
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
