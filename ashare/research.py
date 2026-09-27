"""One-pass comparison: rule book, regime filter, then regime plus the ranker.

The holdout 2025-04-25 .. 2026-06-23 is evaluated after hyperparameters are
frozen. This module does not read that result and run again.
"""

from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path

from ashare.backtest.engine import BacktestResult, PendingBuy, _signal_book, run_backtest
from ashare.backtest.metrics import summarize
from ashare.backtest.walkforward import _trim_to_test
from ashare.config import Settings
from ashare.data.universe import coverage_notes
from ashare.paths import model_path
from ashare.pipeline import load_universe
from ashare.ranker import (
    FEATURE_NAMES,
    FULL_END,
    FULL_START,
    HOLDOUT_END,
    HOLDOUT_START,
    build_labeled_rows,
    fit_schedule,
    load_ranker,
    paper_verdict,
    pnl_tstat,
    precompute_scores,
    fit_model,
    rows_for_training,
    save_ranker,
    select_hyperparams,
)
from ashare.sentiment import attach_inferred_st, build_market_panel, entry_mask, market_day_payload
from ashare.strategies.library import core_strategies

logger = logging.getLogger(__name__)

ARM_LABELS = {
    "baseline": "规则打分",
    "regime": "规则打分 + 行情过滤",
    "regime_ml": "行情过滤 + 模型排序",
}


def run_regime_research(
    settings: Settings,
    today: date,
    cache_dir: Path,
    report_dir: Path,
) -> Path:
    """Train on pre-holdout trades, then backtest each arm once."""
    symbols, notes, chosen, stats = load_universe(settings, today, cache_dir)
    flagged = attach_inferred_st(symbols)
    panel = build_market_panel(symbols, settings)
    calendar = panel.calendar
    strategies = core_strategies()
    bundled = [(item, item.default_params()) for item in strategies]
    pool = max(settings.ml_pool, settings.top_n)
    logger.info("生成候选池，每日最多 %s 只", pool)
    pool_book = _signal_book(symbols, bundled, settings, None, cap=pool)
    rows = build_labeled_rows(symbols, pool_book, panel, settings)
    params, hp_records, used_fallback = select_hyperparams(rows, calendar)
    schedule, importances, frozen = fit_schedule(rows, calendar, params)
    scores = precompute_scores(rows, schedule) if schedule.segments else {}
    if frozen is not None:
        save_ranker(model_path(), frozen, params, importances, HOLDOUT_START)
        importance_path = report_dir / "ranker_importance.json"
        importance_path.parent.mkdir(parents=True, exist_ok=True)
        ranked = sorted(importances.items(), key=lambda item: item[1], reverse=True)
        importance_path.write_text(
            json.dumps(
                {
                    "trained_through": HOLDOUT_START.isoformat(),
                    "params": params,
                    "used_fallback": used_fallback,
                    "importances": [{"feature": name, "importance": value} for name, value in ranked],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    full_start = _snap_start(calendar, FULL_START)
    full_end = _snap_end(calendar, min(FULL_END, calendar[-1]))
    oos_start = _snap_start(calendar, HOLDOUT_START)
    oos_end = _snap_end(calendar, HOLDOUT_END)
    gate = entry_mask(panel)
    rule_book = {day: orders[: settings.top_n] for day, orders in pool_book.items()}

    arms = {
        "baseline": _both(
            symbols, bundled, settings, rule_book, None, None, full_start, full_end, oos_start, oos_end, calendar, "规则打分"
        ),
        "regime": _both(
            symbols, bundled, settings, rule_book, gate, None, full_start, full_end, oos_start, oos_end, calendar, "规则打分 + 行情过滤"
        ),
        "regime_ml": _both(
            symbols,
            bundled,
            settings,
            pool_book,
            gate,
            scores,
            full_start,
            full_end,
            oos_start,
            oos_end,
            calendar,
            "行情过滤 + 模型排序",
        ),
    }
    per_strategy = []
    for strategy in strategies:
        one = [(strategy, strategy.default_params())]
        logger.info("单策略书 %s", strategy.name)
        book = _signal_book(symbols, one, settings, None, cap=settings.top_n)
        plain = _both(
            symbols, one, settings, book, None, None, full_start, full_end, oos_start, oos_end, calendar, strategy.name
        )
        gated = _both(
            symbols, one, settings, book, gate, None, full_start, full_end, oos_start, oos_end, calendar, strategy.name + " + 行情过滤"
        )
        per_strategy.append({"strategy": strategy.name, "baseline": plain, "regime": gated})

    train_rows = len(rows_for_training(rows, HOLDOUT_START))
    labeled = sum(row.label is not None for row in rows)
    payload = _payload(
        arms,
        per_strategy,
        params,
        used_fallback,
        hp_records,
        importances,
        full_start,
        full_end,
        oos_start,
        oos_end,
        flagged,
        train_rows,
        labeled,
        len(rows),
        frozen is not None,
    )
    if chosen:
        notes = list(notes)
        notes.extend(coverage_notes(chosen, stats, full_start, full_end, len(symbols)))
    payload["notes"] = _notes(notes, flagged, train_rows, used_fallback, params, frozen is not None, settings)
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / "regime_ml_latest.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    stem = f"regime_ml_{full_start.strftime('%Y%m%d')}_{full_end.strftime('%Y%m%d')}"
    markdown = report_dir / f"{stem}.md"
    markdown.write_text(_markdown(payload), encoding="utf-8")
    from ashare.pipeline import publish_daily

    publish_daily(symbols, panel, settings, latest_day(calendar, today), report_dir, None, notes)
    logger.info("已写出 %s", markdown)
    return markdown


def run_train(settings: Settings, today: date, cache_dir: Path, report_dir: Path) -> Path:
    """Refit on every completed trade, keeping hyperparameters chosen before the holdout.

    This replaces ``data/models/ranker.pkl``. The holdout backtest uses the model
    written by ``research``, which was fit only on exits before 2025-04-25.
    """
    symbols, _notes, _chosen, _stats = load_universe(settings, today, cache_dir)
    attach_inferred_st(symbols)
    panel = build_market_panel(symbols, settings)
    strategies = core_strategies()
    bundled = [(item, item.default_params()) for item in strategies]
    pool = max(settings.ml_pool, settings.top_n)
    book = _signal_book(symbols, bundled, settings, None, cap=pool)
    rows = build_labeled_rows(symbols, book, panel, settings)
    existing = load_ranker()
    if existing and isinstance(existing.get("params"), dict):
        params = dict(existing["params"])
        used_fallback = False
        logger.info("沿用已保存的超参 %s，不在含样本外的数据上重选", params)
    else:
        params, _records, used_fallback = select_hyperparams(rows, panel.calendar)
    trainable = [row for row in rows if row.label is not None and row.exit_date is not None]
    if len(trainable) < 200:
        raise RuntimeError(f"完整交易只有 {len(trainable)} 笔，不够训练。")
    model = fit_model(trainable, params)
    importances = {
        name: float(value) for name, value in zip(FEATURE_NAMES, model.feature_importances_, strict=True)
    }
    destination = model_path()
    # Exclusive cutoff: every stored label already exited on or before the last bar.
    last = panel.calendar[-1] if panel.calendar else today
    trained_through = date.fromordinal(last.toordinal() + 1)
    save_ranker(destination, model, params, importances, trained_through)
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / "ranker_importance.json"
    ranked = sorted(importances.items(), key=lambda item: item[1], reverse=True)
    path.write_text(
        json.dumps(
            {
                "trained_through": trained_through.isoformat(),
                "params": params,
                "used_fallback": used_fallback,
                "rows": len(trainable),
                "importances": [{"feature": name, "importance": value} for name, value in ranked],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    _write_market_file(report_dir, panel, panel.calendar[-1], True)
    logger.info("已保存模型 %s，训练样本 %s", destination, len(trainable))
    return destination


def _write_market_file(report_dir: Path, panel, day: date, model_ready: bool) -> None:
    info = panel.days.get(day)
    if info is None:
        return
    payload = market_day_payload(day, info)
    payload["model_ready"] = model_ready
    payload["entry_note"] = "允许开新仓" if info.risk_on else "今日不开新仓"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "market_latest.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _both(
    symbols,
    bundled,
    settings: Settings,
    book,
    gate,
    scores,
    full_start: date,
    full_end: date,
    oos_start: date,
    oos_end: date,
    calendar: list[date],
    label: str,
) -> dict:
    logger.info("回测 %s", label)
    full = _simulate(symbols, bundled, settings, book, gate, scores, full_start, full_end, label)
    oos = _simulate_oos(symbols, bundled, settings, book, gate, scores, oos_start, oos_end, calendar, label)
    return {"full": _pack(full), "oos": _pack(oos)}


def _simulate(symbols, bundled, settings, book, gate, scores, start, end, label) -> BacktestResult:
    def rescore(order: PendingBuy) -> float | None:
        if scores is None:
            return None
        return scores.get((order.signal_date, order.code, order.strategy_id))

    return run_backtest(
        symbols,
        bundled,
        settings,
        start,
        end,
        label=label,
        entry_allowed=gate,
        rescore=None if scores is None else rescore,
        prepared_book=book,
    )


def _simulate_oos(symbols, bundled, settings, book, gate, scores, oos_start, oos_end, calendar, label) -> BacktestResult:
    earlier = [day for day in calendar if day < oos_start]
    begin = earlier[-1] if earlier else oos_start
    result = _simulate(symbols, bundled, settings, book, gate, scores, begin, oos_end, label)
    _trim_to_test(result, oos_start)
    return result


def _pack(result: BacktestResult) -> dict:
    metrics = summarize(result)
    pnls = [trade.pnl for trade in result.trades]
    tstat = pnl_tstat(pnls)
    return {
        "total_return": metrics.total_return,
        "max_drawdown": metrics.max_drawdown,
        "trade_count": metrics.trade_count,
        "win_rate": metrics.win_rate,
        "avg_win": metrics.avg_win,
        "avg_loss": metrics.avg_loss,
        "final_equity": metrics.final_equity,
        "tstat": tstat,
        "verdict": paper_verdict(metrics.total_return, metrics.trade_count, tstat),
        "start": result.start.isoformat(),
        "end": result.end.isoformat(),
    }


def _payload(
    arms,
    per_strategy,
    params,
    used_fallback,
    hp_records,
    importances,
    full_start,
    full_end,
    oos_start,
    oos_end,
    flagged,
    train_rows,
    labeled,
    candidates,
    model_ready,
) -> dict:
    ranked = sorted(importances.items(), key=lambda item: item[1], reverse=True)
    return {
        "full_start": full_start.isoformat(),
        "full_end": full_end.isoformat(),
        "oos_start": oos_start.isoformat(),
        "oos_end": oos_end.isoformat(),
        "hyperparams": params,
        "used_fallback": used_fallback,
        "hyperparam_records": hp_records,
        "importances": [{"feature": name, "importance": value} for name, value in ranked],
        "comparison": [
            {"arm": arm, "label": ARM_LABELS[arm], "window": window, **arms[arm][window]}
            for arm in ("baseline", "regime", "regime_ml")
            for window in ("full", "oos")
        ],
        "per_strategy": [
            {
                "strategy": item["strategy"],
                "baseline_full": item["baseline"]["full"],
                "baseline_oos": item["baseline"]["oos"],
                "regime_full": item["regime"]["full"],
                "regime_oos": item["regime"]["oos"],
            }
            for item in per_strategy
        ],
        "st_names_flagged": flagged,
        "train_rows": train_rows,
        "labeled_rows": labeled,
        "candidate_rows": candidates,
        "model_ready": model_ready,
    }


def latest_day(calendar: list[date], today: date) -> date:
    if not calendar:
        return today
    return calendar[-1]


def _notes(
    notes: list[str],
    flagged: int,
    train_rows: int,
    used_fallback: bool,
    params: dict,
    model_ready: bool,
    settings: Settings,
) -> list[str]:
    lines = [
        "比较的是同一套默认参数下的首板次日承接和均线回踩，不是五策略走步。三组实验都使用了按 5% 封板推断的 ST 标记，因此差额来自行情过滤和模型排序。",
        (
            f"行情过滤（写死，未按样本外调整）：至少 {settings.regime_min_names} 只股票有 MA20，"
            f"其中收盘在 MA20 之上的比例 ≥ {settings.regime_breadth_min:.0%}，"
            f"且等权指数不低于自身 {settings.regime_ma_window} 日均线。否则不新开仓，已有持仓仍卖出。"
        ),
        "模型是 scikit-learn 的 GradientBoostingRegressor。标签是单笔交易扣费后的收益率，买不进或直到行情结束仍未卖出的信号没有标签。训练行要求信号日和卖出日都早于截止日。",
        f"超参只在 2025-04-25 之前的扩展窗口里挑选。本次使用 {params}。"
        + ("验证折不够，用的是预先写好的中间格点。" if used_fallback else "验证折里平均标签更高的格点被留下，相同得分保留更早的格点。"),
        f"样本外模型的训练样本是 {train_rows} 笔在 2025-04-25 之前已经平仓的交易。样本外区间没有再训练。"
        if model_ready
        else "样本外之前的完整交易不够 200 笔，没有训练模型，模型排序这一组不会开仓。",
        f"按 5% 涨跌停封板推断、至少有一天像 ST 的股票有 {flagged} 只。这不是交易所的 ST 名单。当前名称里带 ST 的股票仍然整段排除。创业板、科创板和北交所没有 5% 档，不打这个标记。",
        "样本外账户从区间前一个交易日的收盘开始，本金仍按 2 万元计，这样区间第一天的开盘可以成交。总收益的分母是 2 万元，不是区间第一天的权益。",
        "t 统计量是已平仓交易盈亏的均值除以标准误（样本标准差，分母 n-1）。它描述这一段历史里这些交易的离散程度，不是未来还会重复的证明。",
    ]
    lines.extend(notes[:8])
    return lines


def _markdown(payload: dict) -> str:
    lines = [
        "# 行情过滤与模型排序",
        "",
        f"全样本 {payload['full_start']} 至 {payload['full_end']}。样本外 {payload['oos_start']} 至 {payload['oos_end']}，只在参数冻结之后评估一次。",
        "",
        "资金 20000 元，沿用 T+1、涨停买不进、跌停卖不出、最低 5 元佣金、印花税、过户费和滑点。候选是首板次日承接和均线回踩的默认参数。",
        "",
        "## 三组对照",
        "",
        "| 方案 | 区间 | 总收益 | 最大回撤 | 成交笔数 | 胜率 | 平均盈利 | 平均亏损 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    window_name = {"full": "全样本", "oos": "样本外"}
    for row in payload["comparison"]:
        lines.append(
            "| {label} | {window} | {ret} | {dd} | {trades} | {win} | {aw} | {al} |".format(
                label=row["label"],
                window=window_name[row["window"]],
                ret=_pct(row["total_return"]),
                dd=_pct(row["max_drawdown"]),
                trades=row["trade_count"],
                win=_pct(row["win_rate"]),
                aw=f"{row['avg_win']:.2f}",
                al=f"{row['avg_loss']:.2f}",
            )
        )
    lines.extend(["", "## 样本外能不能当模拟盘", ""])
    for row in payload["comparison"]:
        if row["window"] != "oos":
            continue
        lines.append(
            f"- **{row['label']}**：{row['trade_count']} 笔，收益 {_pct(row['total_return'])}，"
            f"t = {row['tstat']:.2f}。{row['verdict']}"
        )
    lines.extend(["", "## 行情过滤对单策略的影响", ""])
    lines.append("| 策略 | 过滤 | 区间 | 总收益 | 最大回撤 | 成交笔数 | 胜率 | 平均盈利 | 平均亏损 |")
    lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for item in payload["per_strategy"]:
        for label, key in (("不过滤", "baseline"), ("行情过滤", "regime")):
            for window, suffix in (("全样本", "full"), ("样本外", "oos")):
                row = item[f"{key}_{suffix}"]
                lines.append(
                    "| {strategy} | {label} | {window} | {ret} | {dd} | {trades} | {win} | {aw} | {al} |".format(
                        strategy=item["strategy"],
                        label=label,
                        window=window,
                        ret=_pct(row["total_return"]),
                        dd=_pct(row["max_drawdown"]),
                        trades=row["trade_count"],
                        win=_pct(row["win_rate"]),
                        aw=f"{row['avg_win']:.2f}",
                        al=f"{row['avg_loss']:.2f}",
                    )
                )
    lines.extend(["", "## 模型", ""])
    lines.append(f"- 参数：`{json.dumps(payload['hyperparams'], ensure_ascii=False)}`")
    lines.append(f"- 预先写好的中间格点：{'是' if payload['used_fallback'] else '否'}")
    lines.append(f"- 候选行 {payload['candidate_rows']}，有标签 {payload['labeled_rows']}，样本外模型使用其中 {payload['train_rows']} 行。")
    if payload["importances"]:
        lines.extend(["", "特征重要性（冻结的样本外模型）：", ""])
        lines.append("| 特征 | 重要性 |")
        lines.append("| --- | ---: |")
        for item in payload["importances"]:
            lines.append(f"| {item['feature']} | {item['importance']:.4f} |")
    lines.extend(["", "## 数据与规则", ""])
    lines.extend(f"- {note}" for note in payload.get("notes") or [])
    lines.append("")
    return "\n".join(lines)


def _pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def _snap_start(calendar: list[date], day: date) -> date:
    for item in calendar:
        if item >= day:
            return item
    return calendar[-1]


def _snap_end(calendar: list[date], day: date) -> date:
    found = calendar[0]
    for item in calendar:
        if item <= day:
            found = item
        else:
            break
    return found
