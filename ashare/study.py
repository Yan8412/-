"""One out-of-sample pass for sentiment features, terciles, and the factor screen.

The holdout 2025-04-25 .. 2026-06-23 is scored after the factor labels and the
optional sentiment gate have been frozen. This module does not read that
account and run again. Config defaults are not written.
"""

from __future__ import annotations

import gc
import json
import logging
import math
from datetime import date
from pathlib import Path

import numpy as np

from ashare.backtest.engine import _signal_book
from ashare.config import Settings
from ashare.factor_screen import confirmed_names, screen_factors
from ashare.ranker import (
    FULL_START,
    HOLDOUT_END,
    HOLDOUT_START,
    build_labeled_rows,
    feature_names,
    fit_schedule,
    pnl_tstat,
    precompute_scores,
    rows_for_training,
    select_hyperparams,
)
from ashare.research import _pack, _simulate, _simulate_oos, _snap_end, _snap_start
from ashare.sentiment import attach_inferred_st, build_market_panel, entry_mask
from ashare.strategies.library import strategies_from_ids
from ashare.terciles import (
    TERCILE_FEATURES,
    WalkForwardTerciles,
    gated_entry_mask,
    iter_folds_with_tail,
    select_gate,
    tag_trades,
    tercile_table,
)

logger = logging.getLogger(__name__)

REPORT_NAME = "sentiment_factor_study.md"


def run_sentiment_factor_study(
    settings: Settings,
    today: date,
    cache_dir: Path,
    report_dir: Path,
) -> Path:
    """MA pullback + regime, then the same book with the new ranker features."""
    from ashare.pipeline import load_universe

    symbols, notes, _chosen, _stats = load_universe(settings, today, cache_dir)
    attach_inferred_st(symbols)
    panel = build_market_panel(symbols, settings)
    calendar = panel.calendar
    if not calendar:
        raise RuntimeError("交易日历为空，无法做样本外对照。")
    full_start = _snap_start(calendar, FULL_START)
    oos_start = _snap_start(calendar, HOLDOUT_START)
    oos_end = _snap_end(calendar, HOLDOUT_END)
    pre_end = _day_before(calendar, oos_start)

    logger.info("因子筛选，训练标签在 %s 之前结束，确认标签在 %s 之前结束", "2024-09-24", HOLDOUT_START.isoformat())
    factor_rows = screen_factors(symbols, settings, batch_size=250)
    confirmed = confirmed_names(factor_rows)
    logger.info("确认因子 %s", list(confirmed) or "无")
    gc.collect()

    strategy = strategies_from_ids(["ma_pullback"])[0]
    bundled = [(strategy, strategy.default_params())]
    pool = max(settings.ml_pool, settings.top_n)
    logger.info("均线回踩候选池，每日最多 %s 只", pool)
    pool_book = _signal_book(symbols, bundled, settings, None, cap=pool)
    rule_book = {day: orders[: settings.top_n] for day, orders in pool_book.items()}
    gate = entry_mask(panel)

    pre_trades = []
    if pre_end is not None and pre_end >= full_start:
        logger.info("样本外之前的均线回踩 + 行情过滤，用于选定至多一条收紧规则")
        pre = _simulate(
            symbols,
            bundled,
            settings,
            rule_book,
            gate,
            None,
            full_start,
            pre_end,
            "均线回踩 + 行情过滤（样本外之前）",
        )
        pre_trades = [trade for trade in pre.trades if trade.buy_date < HOLDOUT_START]

    folds = iter_folds_with_tail(calendar, settings.train_days, settings.test_days)
    walker = WalkForwardTerciles(panel, calendar, folds)
    frozen_gate, train_diag = select_gate(tag_trades(pre_trades), walker)
    if frozen_gate is None:
        logger.info("样本外之前没有足够的分档差异，不收紧行情过滤")
    else:
        logger.info(
            "预承诺规则：%s 丢掉第 %s 档（0 低，2 高），训练差 %.4f",
            frozen_gate.feature,
            frozen_gate.drop_tercile,
            frozen_gate.train_spread,
        )

    logger.info("样本外：均线回踩 + 行情过滤")
    oos_result = _simulate_oos(
        symbols,
        bundled,
        settings,
        rule_book,
        gate,
        None,
        oos_start,
        oos_end,
        calendar,
        "均线回踩 + 行情过滤",
    )
    baseline = _pack(oos_result)
    oos_tagged = [
        trade
        for trade in tag_trades(oos_result.trades)
        if HOLDOUT_START <= trade.buy_date <= HOLDOUT_END
    ]
    terciles = {
        feature: tercile_table(oos_tagged, walker, feature, settings.initial_capital)
        for feature, _label in TERCILE_FEATURES
    }

    sentiment_arm = _ranker_arm(
        symbols,
        bundled,
        settings,
        pool_book,
        panel,
        gate,
        (),
        full_start,
        oos_start,
        oos_end,
        calendar,
        "均线回踩 + 行情过滤 + 情绪特征排序",
    )
    gc.collect()
    if confirmed:
        factor_arm = _ranker_arm(
            symbols,
            bundled,
            settings,
            pool_book,
            panel,
            gate,
            confirmed,
            full_start,
            oos_start,
            oos_end,
            calendar,
            "均线回踩 + 行情过滤 + 情绪特征 + 确认因子",
        )
    else:
        factor_arm = None
    gc.collect()

    tight = None
    if frozen_gate is not None:
        logger.info("样本外：预承诺的收紧规则，只评估这一次")
        tight_mask = gated_entry_mask(panel, calendar, walker, frozen_gate, gate)
        tight = _pack(
            _simulate_oos(
                symbols,
                bundled,
                settings,
                rule_book,
                tight_mask,
                None,
                oos_start,
                oos_end,
                calendar,
                "均线回踩 + 收紧后的行情过滤",
            )
        )

    payload = {
        "oos_start": oos_start.isoformat(),
        "oos_end": oos_end.isoformat(),
        "full_start": full_start.isoformat(),
        "initial_capital": settings.initial_capital,
        "baseline": baseline,
        "sentiment_ranker": sentiment_arm,
        "factor_ranker": factor_arm,
        "confirmed_factors": list(confirmed),
        "factors": [_jsonable(row) for row in factor_rows],
        "terciles": {feature: [_jsonable(row) for row in rows] for feature, rows in terciles.items()},
        "oos_trade_count": len(oos_tagged),
        "gate": None
        if frozen_gate is None
        else {
            "feature": frozen_gate.feature,
            "label": frozen_gate.label,
            "drop_tercile": frozen_gate.drop_tercile,
            "train_spread": frozen_gate.train_spread,
            "train_worst_mean": frozen_gate.train_worst_mean,
        },
        "gate_oos": tight,
        "train_gate_diag": [_jsonable(row) for row in train_diag],
        "archived_baseline": _archived_pullback(report_dir),
        "notes": _study_notes(notes, settings, frozen_gate, confirmed),
        "config_unchanged": True,
        "proposed_config": _proposal(tight, frozen_gate),
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / "sentiment_factor_study.json"
    json_path.write_text(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    markdown = report_dir / REPORT_NAME
    markdown.write_text(render_report(payload), encoding="utf-8")
    logger.info("已写出 %s", markdown)
    return markdown


def render_report(payload: dict) -> str:
    lines = [
        "# 涨停情绪、三分位和 OHLCV 因子",
        "",
        (
            f"样本外 {payload['oos_start']} 至 {payload['oos_end']}，资金 "
            f"{payload['initial_capital']:.0f} 元，扣费后。"
            "最终持有期没有参与因子筛选，也没有参与收紧规则的选择。"
            "默认配置没有改：`use_ranker` 仍是 false，策略默认值也没动。"
        ),
        "",
        "## 结论",
        "",
        _conclusion(payload),
        "",
        "## 任务 1：情绪特征与排序",
        "",
        (
            "新增的市场特征都只用到当天收盘和更早的 K 线：1进2、2进3、3板及以上晋级率，"
            "昨日连板高度至少为 2 的股票今日平均涨跌（连板溢价），以及 1 到最高连板之间缺失的高度个数（梯队断层）。"
            "晋级率在没有候选时是缺失，不是 0。想法来自 simonlin1212/vibe-astock 的情绪指标，计算用的是本项目自己的涨停标记。"
        ),
        "",
        "对照是同一套均线回踩默认参数加原来的行情过滤。排序模型的超参仍只在 2025-04-25 之前的验证折里挑选，样本外不再训练。",
        "",
        _account_table(
            [
                ("均线回踩 + 行情过滤", payload["baseline"]),
                ("均线回踩 + 行情过滤 + 情绪特征排序", payload["sentiment_ranker"].get("oos")),
                (
                    "均线回踩 + 行情过滤 + 情绪特征 + 确认因子",
                    None if payload.get("factor_ranker") is None else payload["factor_ranker"].get("oos"),
                ),
            ]
        ),
        "",
        _ranker_blurb(payload["sentiment_ranker"], "情绪特征模型"),
        "",
    ]
    if payload.get("factor_ranker") is not None:
        lines.append(_ranker_blurb(payload["factor_ranker"], "加入确认因子后的模型"))
        lines.append("")
    elif not payload.get("confirmed_factors"):
        lines.append("没有因子被标为确认，因此没有第二套排序模型。上一行就是情绪特征这一套。")
        lines.append("")

    lines.extend(
        [
            "## 任务 2：入场前夜情绪的三分位",
            "",
            (
                "样本是上面「均线回踩 + 行情过滤」样本外账户里已经平仓的交易。"
                "入场前夜是信号日收盘。某一天落在哪一档，只看它所在走步测试窗对应的训练窗："
                f"训练 {payload.get('baseline', {}).get('start', '')} 这段日历上，"
                "训练窗内、且当时允许开仓的交易日，用该指标的平均秩百分位切成三档。"
                "测试日不进入切点。分档不齐的交易单独计，不挪到相邻档。"
            ),
            "",
            (
                "表里的收益是这些交易扣费后盈亏之和除以 2 万元，表示它们对这个账户的贡献，"
                "不是「只做这一档」再跑出来的账户收益。t 是单笔盈亏的均值除以标准误。"
            ),
            "",
        ]
    )
    labels = dict(TERCILE_FEATURES)
    for feature, rows in (payload.get("terciles") or {}).items():
        lines.append(f"### {labels.get(feature, feature)}")
        lines.append("")
        lines.append(_tercile_table(rows))
        lines.append("")
    lines.extend(["### 收紧规则", ""])
    gate = payload.get("gate")
    if gate is None:
        lines.append(
            "样本外之前的走步测试交易里，三个指标的两端档要么笔数不够，要么较差的一端并没有亏钱。"
            "没有预承诺一条收紧规则，因此也没有拿样本外再跑一遍。样本外三分位表只作描述，不拿来选档。"
        )
    else:
        side = "低档" if gate["drop_tercile"] == 0 else "高档"
        lines.append(
            f"规则在样本外之前就定死：{gate['label']}（`{gate['feature']}`）丢掉{side}，"
            f"训练窗两端平均盈亏之差为 {gate['train_spread']:.2f} 元，较差一端平均盈亏 {gate['train_worst_mean']:.2f} 元。"
            "样本外只把这条规则评估了一次。切点仍按每个走步训练窗更新，丢掉哪一端不再改。"
        )
        lines.append("")
        lines.append(_account_table([("均线回踩 + 预承诺的收紧过滤", payload.get("gate_oos"))]))
    proposal = payload.get("proposed_config")
    lines.extend(["", proposal or "不建议修改默认配置。", ""])

    lines.extend(
        [
            "## 任务 3：OHLCV 因子",
            "",
            (
                "15 个因子按 qlib Alpha158 的定义重写（Apache-2.0，`qlib/contrib/data/loader.py`）。"
                "价格用因果前复权，成交量用原始股数。BETA10 是收盘价对时间序号 1..10 的回归斜率再除以收盘价，"
                "不是个股对市场的 CAPM beta。qlib 的注释写明每天涨 10 元则斜率为 10；"
                "Vibe-Trading 里若把 beta10 说成市场 beta，和它自己的斜率代码以及 qlib 都不一致，这里以 qlib 为准。"
            ),
            "",
            (
                "标签是 5 个交易日之后的前复权收益。训练段的标签在 2024-09-24 之前结束；"
                "确认段从 2024-09-24 起，标签在 2025-04-25 之前结束。横截面至少 100 只可交易股票才计算当日秩相关。"
                "t 用 Newey-West，Bartlett 滞后 4 天，对应持有期重叠。确认要求训练和确认同号且两边 |t| 都 ≥ 2。"
                "只在确认段显著、训练段不显著的，记为噪声，不能入选。"
            ),
            "",
            _factor_table(payload.get("factors") or []),
            "",
            "入选排序模型的确认因子："
            + ("、".join(payload.get("confirmed_factors") or []) or "无")
            + "。",
            "",
        ]
    )
    archived = payload.get("archived_baseline")
    if archived:
        lines.extend(
            [
                "## 和归档的均线回踩 + 行情过滤相比",
                "",
                (
                    "归档数字来自上一轮 `regime_ml` 报告，当时用的是已有的逐日 ST 缓存。"
                    "本轮如果重新下载了日线，成交和 t 可以和归档差一截，差的那一截不是这次特征改出来的。"
                    "这次有没有比基线更好，只看本轮这张表内部的对照。"
                ),
                "",
                _account_table(
                    [
                        ("归档：均线回踩 + 行情过滤", archived),
                        ("本轮：均线回踩 + 行情过滤", payload["baseline"]),
                    ]
                ),
                "",
            ]
        )
    lines.extend(["## 说明", ""])
    lines.extend(f"- {note}" for note in payload.get("notes") or [])
    lines.append("")
    return "\n".join(lines)


def _ranker_arm(
    symbols,
    bundled,
    settings,
    pool_book,
    panel,
    gate,
    extra,
    full_start,
    oos_start,
    oos_end,
    calendar,
    label,
) -> dict:
    logger.info("标注 %s", label)
    rows = build_labeled_rows(symbols, pool_book, panel, settings, extra_factors=extra)
    params, records, used_fallback = select_hyperparams(rows, calendar)
    names = feature_names(extra)
    schedule, importances, frozen = fit_schedule(rows, calendar, params, names=names)
    scores = precompute_scores(rows, schedule) if schedule.segments else {}
    if not scores or frozen is None:
        logger.info("%s 没有可用模型", label)
        packed = {
            "total_return": 0.0,
            "max_drawdown": 0.0,
            "trade_count": 0,
            "win_rate": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "final_equity": settings.initial_capital,
            "tstat": 0.0,
            "verdict": "样本外之前的完整交易不够，没有训练模型。",
            "start": oos_start.isoformat(),
            "end": oos_end.isoformat(),
        }
    else:
        packed = _pack(
            _simulate_oos(symbols, bundled, settings, pool_book, gate, scores, oos_start, oos_end, calendar, label)
        )
    ranked = sorted(importances.items(), key=lambda item: item[1], reverse=True)
    train_rows = len(rows_for_training(rows, HOLDOUT_START))
    del rows
    del scores
    gc.collect()
    return {
        "oos": packed,
        "params": params,
        "used_fallback": used_fallback,
        "hyperparam_folds": len(records),
        "train_rows": train_rows,
        "importances": [{"feature": name, "importance": value} for name, value in ranked[:12]],
        "model_ready": frozen is not None,
        "extra_factors": list(extra),
    }


def _conclusion(payload: dict) -> str:
    base = payload["baseline"]
    parts = [
        (
            f"均线回踩 + 行情过滤：收益 {_pct(base['total_return'])}，t = {base['tstat']:.2f}，"
            f"{base['trade_count']} 笔，最大回撤 {_pct(base['max_drawdown'])}。"
        )
    ]
    sentiment = payload["sentiment_ranker"]["oos"]
    parts.append(
        "加上情绪特征后的排序："
        f"收益 {_pct(sentiment['total_return'])}，t = {sentiment['tstat']:.2f}，"
        f"{sentiment['trade_count']} 笔，最大回撤 {_pct(sentiment['max_drawdown'])}。"
    )
    factor_arm = payload.get("factor_ranker")
    if factor_arm is not None:
        row = factor_arm["oos"]
        parts.append(
            "再加入确认因子："
            f"收益 {_pct(row['total_return'])}，t = {row['tstat']:.2f}，"
            f"{row['trade_count']} 笔，最大回撤 {_pct(row['max_drawdown'])}。"
        )
    else:
        parts.append("没有确认因子，排序模型没有再加股票层面的 OHLCV 因子。")
    gate = payload.get("gate_oos")
    if gate is not None:
        parts.append(
            "预承诺的收紧过滤："
            f"收益 {_pct(gate['total_return'])}，t = {gate['tstat']:.2f}，"
            f"{gate['trade_count']} 笔，最大回撤 {_pct(gate['max_drawdown'])}。"
        )
    parts.append(payload.get("proposed_config") or "没有任何一条达到 |t| ≥ 2，默认配置保持原样。")
    return " ".join(parts)


def _proposal(tight: dict | None, gate) -> str:
    if tight is None or gate is None:
        return "不建议修改默认配置。样本外没有一条预承诺规则达到 |t| ≥ 2。"
    if tight["total_return"] > 0 and abs(tight["tstat"]) >= 2 and tight["trade_count"] >= 30:
        side = "低于训练窗低分位" if gate.drop_tercile == 0 else "高于训练窗高分位"
        return (
            f"可以考虑把行情过滤再加一条：信号日的{gate.label}{side}时不开新仓。"
            f"这条规则的样本外收益 {_pct(tight['total_return'])}，t = {tight['tstat']:.2f}，"
            f"{tight['trade_count']} 笔。默认配置没有改，需要人工确认后再写入。"
        )
    return (
        "不建议修改默认配置。预承诺的收紧规则已经在样本外评估过一次，"
        f"收益 {_pct(tight['total_return'])}，t = {tight['tstat']:.2f}，没有达到 |t| ≥ 2。"
    )


def _account_table(rows: list[tuple[str, dict | None]]) -> str:
    lines = [
        "| 方案 | 总收益 | t | 成交笔数 | 最大回撤 | 胜率 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, row in rows:
        if not row:
            lines.append(f"| {label} | — | — | — | — | — |")
            continue
        lines.append(
            "| {label} | {ret} | {tstat} | {trades} | {dd} | {win} |".format(
                label=label,
                ret=_pct(row["total_return"]),
                tstat=f"{float(row['tstat']):.2f}",
                trades=row["trade_count"],
                dd=_pct(row["max_drawdown"]),
                win=_pct(row.get("win_rate") or 0.0),
            )
        )
    return "\n".join(lines)


def _tercile_table(rows: list[dict]) -> str:
    names = {0: "低", 1: "中", 2: "高", None: "未分档"}
    lines = [
        "| 档 | 笔数 | 盈亏贡献 | 平均单笔收益率 | t |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {name} | {count} | {contrib} | {mean} | {tstat} |".format(
                name=names.get(row["tercile"], str(row["tercile"])),
                count=row["trade_count"],
                contrib=_pct(row["contribution"]),
                mean=_pct(row["mean_return"]),
                tstat=f"{float(row['tstat']):.2f}",
            )
        )
    return "\n".join(lines)


def _factor_table(rows: list[dict]) -> str:
    lines = [
        "| 因子 | 训练 IC | 训练 t | 训练天数 | 确认 IC | 确认 t | 确认天数 | 分类 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    labels = {
        "confirmed": "确认",
        "train-only": "仅训练",
        "reversed": "反向",
        "noise": "噪声",
    }
    for row in rows:
        lines.append(
            "| {name} | {tr} | {tt} | {td} | {sr} | {st} | {sd} | {klass} |".format(
                name=row["name"],
                tr=_ic(row.get("train_mean_ic")),
                tt=_t(row.get("train_t")),
                td=row.get("train_days") or 0,
                sr=_ic(row.get("test_mean_ic")),
                st=_t(row.get("test_t")),
                sd=row.get("test_days") or 0,
                klass=labels.get(row.get("klass") or "", row.get("klass") or ""),
            )
        )
    return "\n".join(lines)


def _ranker_blurb(arm: dict, title: str) -> str:
    params = arm.get("params") or {}
    ready = "已训练" if arm.get("model_ready") else "未训练"
    top = arm.get("importances") or []
    bits = "、".join(f"{item['feature']} {item['importance']:.3f}" for item in top[:6]) or "无"
    fallback = "是" if arm.get("used_fallback") else "否"
    return (
        f"{title}：{ready}，样本外之前的训练行 {arm.get('train_rows', 0)}，"
        f"参数 `{json.dumps(params, ensure_ascii=False)}`，用预先写好的中间格点：{fallback}。"
        f"重要性前几名：{bits}。"
    )


def _study_notes(notes: list[str], settings: Settings, gate, confirmed: tuple[str, ...]) -> list[str]:
    lines = [
        "行情过滤仍是原来的三条：至少 "
        f"{settings.regime_min_names} 只股票有 MA20，站上 MA20 的比例 ≥ {settings.regime_breadth_min:.0%}，"
        f"等权指数不低于自身 {settings.regime_ma_window} 日均线。",
        "排序模型是 scikit-learn 的 GradientBoostingRegressor。标签是单笔扣费后收益率。买入失败或到样本结束仍未卖出的信号没有标签。",
        "超参网格没有改。挑选只用信号日和卖出日都早于 2025-04-25 的交易。",
        f"确认因子 {len(confirmed)} 个。"
        + ("名单：" + "、".join(confirmed) + "。" if confirmed else "没有因子同时通过训练和确认。"),
        "收紧规则如果存在，也只在样本外之前的交易上选定，样本外三分位表不参与选择。",
        "t 描述的是这一段历史里这些交易有多散，不是未来还会重复的证明。|t| ≥ 2 才考虑提议改配置，而且默认值不会自动改掉。",
    ]
    if gate is None:
        lines.append("这次没有提出收紧规则。")
    lines.extend(notes[:6])
    return lines


def _archived_pullback(report_dir: Path) -> dict | None:
    path = report_dir / "regime_ml_latest.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    for item in payload.get("per_strategy") or []:
        if item.get("strategy") != "均线回踩":
            continue
        row = item.get("regime_oos") or {}
        if row.get("trade_count") is None:
            return None
        return {
            "total_return": row.get("total_return"),
            "max_drawdown": row.get("max_drawdown"),
            "trade_count": row.get("trade_count"),
            "win_rate": row.get("win_rate"),
            "tstat": row.get("tstat"),
        }
    return None


def _day_before(calendar: list[date], day: date) -> date | None:
    earlier = [item for item in calendar if item < day]
    return earlier[-1] if earlier else None


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, np.integer):
        return int(value)
    return value


def _pct(value: float | None) -> str:
    if value is None or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return "—"
    return f"{float(value) * 100:.2f}%"


def _ic(value) -> str:
    if value is None or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return "—"
    return f"{float(value):.4f}"


def _t(value) -> str:
    if value is None or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return "—"
    return f"{float(value):.2f}"


# pnl_tstat is re-exported for callers that want the same t as the tables.
__all__ = ["REPORT_NAME", "render_report", "run_sentiment_factor_study", "pnl_tstat"]
