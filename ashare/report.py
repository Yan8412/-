"""Chinese Markdown / CSV reports and equity charts."""

from __future__ import annotations

from pathlib import Path

from ashare.backtest.engine import BacktestResult
from ashare.backtest.metrics import Metrics, summarize
from ashare.backtest.walkforward import FoldOutcome


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def cny(value: float) -> str:
    return f"{value:,.2f}"


def write_equity_chart(result: BacktestResult, path: Path, title: str) -> bool:
    """Save a PNG equity curve. Returns False when there is nothing to draw."""
    if len(result.equity) < 2:
        return False
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    for font_path in (
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
    ):
        if Path(font_path).exists():
            font_manager.fontManager.addfont(font_path)
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=font_path).get_name()
            break
    plt.rcParams["axes.unicode_minus"] = False
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    ax.plot(result.equity_dates, result.equity, color="#0b6e4f", linewidth=1.4)
    ax.axhline(result.initial_capital, color="#888888", linewidth=0.8, linestyle="--")
    ax.set_title(title)
    ax.set_ylabel("权益（元）")
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return True


def metrics_row(name: str, metrics: Metrics) -> str:
    return (
        f"| {name} | {pct(metrics.total_return)} | {pct(metrics.annualized_return)} | "
        f"{pct(metrics.win_rate)} | {cny(metrics.avg_win)} | {cny(metrics.avg_loss)} | "
        f"{pct(metrics.max_drawdown)} | {metrics.trade_count} | {cny(metrics.final_equity)} |"
    )


def metrics_header() -> str:
    return (
        "| 策略 | 总收益 | 年化收益 | 胜率 | 平均盈利 | 平均亏损 | 最大回撤 | 成交笔数 | 期末权益 |\n"
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    )


def render_backtest_report(
    *,
    full_results: list[BacktestResult],
    oos_results: list[tuple[BacktestResult, list[FoldOutcome]]],
    combined: BacktestResult,
    notes: list[str],
    universe: list[tuple[str, str]],
    assumptions: list[str],
    chart_names: dict[str, str],
    comparison: list[str] | None = None,
) -> str:
    lines = [
        "# A股短线回测报告",
        "",
        "本报告由本次运行的真实日线计算，不是手填示例。费用、滑点和无法成交的涨跌停都已计入。",
        "历史收益不代表未来收益。",
        "",
        "## 假设",
        "",
    ]
    lines.extend(f"- {item}" for item in assumptions)
    lines.extend(["", "## 股票池", ""])
    lines.append(f"进入回测的股票共 {len(universe)} 只。能不能买，按信号当天的价格、成交额、停牌、涨跌停和上市天数判断。")
    lines.append("")
    if len(universe) > 80:
        lines.append("名单过长，报告不逐行展开。覆盖情况写在文末。")
    else:
        lines.append("| 代码 | 名称 |")
        lines.append("| --- | --- |")
        for code, name in universe:
            lines.append(f"| {code} | {name} |")
    lines.extend(["", "## 全样本（默认参数）", "", metrics_header()])
    for result in full_results:
        lines.append(metrics_row(result.strategy_name, summarize(result)))
    lines.append(metrics_row(combined.strategy_name, summarize(combined)))
    lines.extend(
        [
            "",
            "全样本使用同一套默认参数跑完整段行情，只适合观察策略形态。它会高估实盘，因为参数是看着这段历史定的。",
            "",
            "## 走步样本外",
            "",
            "每个测试段开始前，只用它之前的训练窗口在小网格里选参数，然后参数冻结，只交易后面的测试段。",
            "训练窗口看不到该测试段的价格。指标用因果复权，后来的分红不会改写更早的信号。",
            "尾部凑不满一个测试窗口的交易日不会进入样本外曲线。",
            "",
            metrics_header(),
        ]
    )
    if not oos_results:
        lines.append("| （样本太短，无法切出训练/测试窗口） | | | | | | | | |")
    for result, _folds in oos_results:
        lines.append(metrics_row(f"{result.strategy_name}（{result.start} ~ {result.end}）", summarize(result)))
    for result, folds in oos_results:
        if not folds:
            lines.extend(["", f"{result.strategy_name} 没有形成有效的走步窗口。"])
            continue
        lines.extend(["", f"### {result.strategy_name} 的参数选择", ""])
        lines.append("| 策略 | 测试区间 | 训练目标值 | 最大持有天数 | 其他关键参数 | 训练期笔数 |")
        lines.append("| --- | --- | ---: | ---: | --- | ---: |")
        for outcome in folds:
            params = outcome.chosen_params
            extra = ", ".join(
                f"{key}={value}"
                for key, value in params.items()
                if key not in {"max_hold_days", "entry_low_pct", "entry_high_pct", "stop_loss_pct"}
            )
            lines.append(
                f"| {outcome.strategy_name or outcome.strategy_id} | "
                f"{outcome.fold.test_start} ~ {outcome.fold.test_end} | "
                f"{outcome.train_score:.3f} | {params.get('max_hold_days', '')} | {extra} | "
                f"{outcome.train_metrics.trade_count} |"
            )
        lines.append("")
    lines.extend(["", "## 权益曲线", ""])
    for key, filename in chart_names.items():
        lines.append(f"![{key}]({filename})")
        lines.append("")
    lines.extend(["", "## 最近成交（全样本组合）", ""])
    if not combined.trades:
        lines.append("组合在这段样本里没有完成任何买卖。")
    else:
        lines.append("| 代码 | 策略 | 买入日 | 卖出日 | 股数 | 买入价 | 卖出价 | 盈亏 | 原因 |")
        lines.append("| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |")
        for trade in combined.trades[-12:]:
            lines.append(
                f"| {trade.code} | {trade.strategy_name} | {trade.buy_date} | {trade.sell_date} | "
                f"{trade.shares} | {trade.buy_price:.2f} | {trade.sell_price:.2f} | "
                f"{trade.pnl:.2f} | {trade.reason} |"
            )
    lines.extend(["", "## 未成交统计（组合）", ""])
    if combined.rejects:
        for reason, count in sorted(combined.rejects.items(), key=lambda item: -item[1]):
            lines.append(f"- {reason}: {count}")
    else:
        lines.append("- 无")
    if comparison:
        lines.extend(["", "## 对照：上一版 50 只高成交额样本", ""])
        lines.extend(comparison)
    lines.extend(["", "## 数据与局限", ""])
    if notes:
        lines.extend(f"- {note}" for note in notes)
    else:
        lines.append("- 数据源没有记录额外异常。")
    lines.extend(
        [
            "- 没有逐笔行情，涨停开板又封死的日内过程用日线近似，真实成交会更差一些。",
            "",
        ]
    )
    return "\n".join(lines)


def write_recommendations_markdown(path: Path, rows: list[dict], preface: list[str]) -> None:
    lines = ["# 下一交易日候选", ""]
    lines.extend(preface)
    lines.append("")
    if not rows:
        lines.append("今天没有符合条件的候选。空仓是允许的结果。")
    else:
        lines.append(
            "| 代码 | 名称 | 策略 | 得分 | 模型分数 | 买入下限 | 买入上限 | 止盈 | 止损 | 最长持有 | 建议股数 | 预估金额 |"
        )
        lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for row in rows:
            model_score = row.get("model_score")
            model_cell = "—" if model_score is None else f"{float(model_score):.4f}"
            lines.append(
                f"| {row['code']} | {row['name']} | {row['strategy']} | {row['score']:.1f} | {model_cell} | "
                f"{row['entry_low']:.2f} | {row['entry_high']:.2f} | {row['take_profit']:.2f} | "
                f"{row['stop_loss']:.2f} | {row['max_hold_days']} | {row['shares']} | {row['budget']:.0f} |"
            )
        lines.extend(["", "## 理由", ""])
        for row in rows:
            lines.append(f"- **{row['code']} {row['name']}**（{row['strategy']}）：{row['reason']}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_recommendations_csv(path: Path, rows: list[dict]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "代码",
        "名称",
        "策略",
        "得分",
        "模型分数",
        "理由",
        "建议买入下限",
        "建议买入上限",
        "止盈价",
        "止损价",
        "最大持有天数",
        "建议股数",
        "预估金额",
        "信号日",
        "收盘价",
        "首封时间",
        "炸板次数",
        "回封",
        "竞价价",
        "竞价量",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "代码": row["code"],
                    "名称": row["name"],
                    "策略": row["strategy"],
                    "得分": f"{row['score']:.2f}",
                    "模型分数": "" if row.get("model_score") is None else f"{float(row['model_score']):.4f}",
                    "理由": row["reason"],
                    "建议买入下限": f"{row['entry_low']:.2f}",
                    "建议买入上限": f"{row['entry_high']:.2f}",
                    "止盈价": f"{row['take_profit']:.2f}",
                    "止损价": f"{row['stop_loss']:.2f}",
                    "最大持有天数": row["max_hold_days"],
                    "建议股数": row["shares"],
                    "预估金额": f"{row['budget']:.2f}",
                    "信号日": row["signal_date"],
                    "收盘价": f"{row['close']:.2f}",
                    "首封时间": row.get("first_seal") or "",
                    "炸板次数": row.get("broken_seals") if row.get("broken_seals") is not None else "",
                    "回封": row.get("resealed") or "",
                    "竞价价": row.get("auction_price") if row.get("auction_price") is not None else "",
                    "竞价量": row.get("auction_volume") if row.get("auction_volume") is not None else "",
                }
            )
