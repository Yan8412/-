"""Local Chinese dashboard. Launch with ``python -m ashare dashboard``."""

from __future__ import annotations

from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from ashare.dashboard_data import latest_shortlist, load_backtest_bundle, load_json_object, load_ledger, sell_hint
from ashare.paths import cache_dir, default_config_path, ledger_path, report_dir

REPORT_DIR = report_dir()
LEDGER_PATH = ledger_path()
CACHE_DIR = cache_dir()
BUNDLE_NAME = "backtest_latest.json"


def _font() -> str:
    if Path("C:/Windows/Fonts/msyh.ttc").exists():
        return "Microsoft YaHei"
    if Path("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc").exists():
        return "WenQuanYi Micro Hei"
    return "sans-serif"


def _apply_style() -> None:
    font = _font()
    st.markdown(
        f"""
        <style>
        .stApp {{
          font-family: "{font}", "Microsoft YaHei", "Noto Sans CJK SC", sans-serif;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def _cny(value: float) -> str:
    return f"{value:,.2f}"


def _metrics_frame(rows: list[dict]) -> pd.DataFrame:
    table = []
    for row in rows:
        metrics = row.get("metrics") or {}
        table.append(
            {
                "策略": row.get("strategy_name", ""),
                "区间": f"{row.get('start', '')} ~ {row.get('end', '')}",
                "总收益": _pct(float(metrics.get("total_return") or 0)),
                "年化收益": _pct(float(metrics.get("annualized_return") or 0)),
                "胜率": _pct(float(metrics.get("win_rate") or 0)),
                "平均盈利": _cny(float(metrics.get("avg_win") or 0)),
                "平均亏损": _cny(float(metrics.get("avg_loss") or 0)),
                "最大回撤": _pct(float(metrics.get("max_drawdown") or 0)),
                "成交笔数": int(metrics.get("trade_count") or 0),
                "期末权益": _cny(float(metrics.get("final_equity") or 0)),
            }
        )
    return pd.DataFrame(table)


def _equity_frame(rows: list[dict]) -> pd.DataFrame:
    records = []
    for row in rows:
        name = row.get("strategy_name", "")
        for point in row.get("equity") or []:
            records.append({"日期": point["date"], "权益": point["equity"], "策略": name})
    return pd.DataFrame(records)


def _equity_chart(frame: pd.DataFrame, title: str) -> alt.Chart:
    font = _font()
    base = alt.Chart(frame).mark_line().encode(
        x=alt.X("日期:T", title="日期"),
        y=alt.Y("权益:Q", title="权益（元）", scale=alt.Scale(zero=False)),
        color=alt.Color("策略:N", title="策略"),
        tooltip=["策略:N", "日期:T", alt.Tooltip("权益:Q", format=",.2f")],
    )
    return (
        base.properties(title=title, height=360)
        .configure_axis(labelFont=font, titleFont=font)
        .configure_legend(labelFont=font, titleFont=font)
        .configure_title(font=font)
        .interactive()
    )


def page_shortlist() -> None:
    st.header("今日候选")
    st.caption("收盘信号，委托目标是下一个交易日的开盘价。涨停开盘、或开盘价落在区间外，都不会成交。")
    _market_banner()
    path, rows = latest_shortlist(REPORT_DIR)
    if st.button("运行收盘更新", type="primary"):
        _run_daily_update()
    if path is None or not rows:
        st.info("还没有候选名单。点上面的按钮，或在仓库根目录运行 python -m ashare daily。")
        return
    st.subheader(path.stem.replace("daily_", "信号日 "))
    frame = pd.DataFrame(rows)
    preferred = [
        "代码",
        "名称",
        "策略",
        "建议买入下限",
        "建议买入上限",
        "止盈价",
        "止损价",
        "最大持有天数",
        "建议股数",
        "预估金额",
        "收盘价",
        "得分",
        "模型分数",
        "首封时间",
        "炸板次数",
        "回封",
        "竞价价",
        "竞价量",
    ]
    columns = [name for name in preferred if name in frame.columns]
    show = frame[columns].copy()
    for name in ("建议买入下限", "建议买入上限", "止盈价", "止损价", "收盘价", "预估金额", "得分", "模型分数"):
        if name in show.columns:
            show[name] = pd.to_numeric(show[name], errors="coerce")
    if "建议股数" in show.columns:
        show["建议股数"] = pd.to_numeric(show["建议股数"], errors="coerce").astype("Int64")
    if "最大持有天数" in show.columns:
        show["最大持有天数"] = pd.to_numeric(show["最大持有天数"], errors="coerce").astype("Int64")
    st.dataframe(show, hide_index=True, use_container_width=True)
    st.subheader("理由")
    reason_col = "理由" if "理由" in frame.columns else None
    if reason_col is None:
        st.info("这份名单没有写理由。")
        return
    for _, row in frame.iterrows():
        title = f"{row.get('代码', '')} {row.get('名称', '')} · {row.get('策略', '')}"
        st.markdown(f"**{title}**")
        st.write(row.get(reason_col) or "无")


def _run_daily_update() -> None:
    from datetime import date

    from ashare.config import load_settings
    from ashare.pipeline import run_daily

    settings = load_settings(default_config_path())
    with st.spinner("正在更新行情、结算模拟盘并重算候选。这只写模拟盘委托，不会向券商下单。"):
        try:
            written = run_daily(settings, date.today(), CACHE_DIR, REPORT_DIR, LEDGER_PATH)
        except Exception as exc:  # noqa: BLE001 - show the failure on the page
            st.error(f"更新失败：{exc}")
            return
    st.success(f"已更新 {written.name}")
    st.rerun()


def _render_sample(rows: list[dict], title: str, key: str, empty_message: str) -> None:
    st.dataframe(_metrics_frame(rows), hide_index=True, use_container_width=True)
    equity = _equity_frame(rows)
    if equity.empty:
        st.info(empty_message)
        return
    names = list(equity["策略"].unique())
    chosen = st.multiselect("显示哪些曲线", options=names, default=names, key=key)
    shown = equity[equity["策略"].isin(chosen)] if chosen else equity.iloc[0:0]
    if shown.empty:
        st.info("没有选中的曲线。")
        return
    st.altair_chart(_equity_chart(shown, title), use_container_width=True)


def _market_banner() -> None:
    snapshot = load_json_object(REPORT_DIR / "market_latest.json")
    if not snapshot:
        st.info("还没有行情状态。运行 python -m ashare daily 之后，这里会显示过滤结果和涨停情绪。")
        return
    note = snapshot.get("entry_note") or ""
    gate = snapshot.get("promo_gate")
    if gate == "closed":
        st.warning(f"{snapshot.get('date', '')} {note or '1进2 落在低档，模拟盘不写新委托'}")
    elif gate == "unavailable":
        st.warning(f"{snapshot.get('date', '')} {snapshot.get('promo_gate_warning') or '1进2闸门无法计算，这次不拦截'}")
    elif snapshot.get("risk_on"):
        st.success(f"{snapshot.get('date', '')} {note}")
    else:
        st.warning(f"{snapshot.get('date', '')} {note or '今日不开新仓'}")
    breadth = snapshot.get("breadth")
    versus = snapshot.get("index_vs_ma")
    prev = snapshot.get("prev_limit_return")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("站上 MA20", "—" if breadth is None else f"{float(breadth) * 100:.2f}%")
    c2.metric("涨停家数", int(snapshot.get("limit_up_count") or 0))
    c3.metric("炸板率", f"{float(snapshot.get('broken_rate') or 0) * 100:.1f}%")
    c4.metric("最高连板", int(snapshot.get("max_height") or 0))
    extra = []
    if versus is not None:
        extra.append(f"等权指数相对均线 {float(versus) * 100:.2f}%")
    if prev is not None:
        extra.append(f"昨日涨停今日平均 {float(prev) * 100:.2f}%")
    promo = snapshot.get("promo_1_2")
    cut = snapshot.get("promo_gate_cut")
    if gate == "open":
        promo_text = "—" if promo is None else f"{float(promo) * 100:.1f}%"
        cut_text = "—" if cut is None else f"{float(cut) * 100:.1f}%"
        extra.append(f"1进2 {promo_text}，低档切点 {cut_text}，闸门开启")
    elif gate == "closed":
        promo_text = "—" if promo is None else f"{float(promo) * 100:.1f}%"
        cut_text = "—" if cut is None else f"{float(cut) * 100:.1f}%"
        extra.append(f"1进2 {promo_text}，低档切点 {cut_text}，闸门关闭，已有持仓照常卖出")
    elif gate == "off":
        extra.append("1进2低档闸门未开启")
    if snapshot.get("use_ranker") is False:
        extra.append("排序模型已关闭")
    elif snapshot.get("model_ready"):
        extra.append("排序模型已加载")
    else:
        extra.append("排序模型未训练，候选按规则分")
    if extra:
        st.caption(" · ".join(extra))
    _pool_banner()


def _pool_banner() -> None:
    snapshot = load_json_object(REPORT_DIR / "pool_latest.json")
    if not snapshot:
        return
    source = {"ths": "同花顺", "akshare": "东财", "none": "无数据"}.get(snapshot.get("source") or "", snapshot.get("source") or "")
    broken = snapshot.get("broken_rate")
    broken_text = "—" if broken is None else f"{float(broken) * 100:.1f}%"
    st.caption(
        f"涨停池 {snapshot.get('date', '')}（{source}）："
        f"涨停 {int(snapshot.get('limit_up_count') or 0)}，"
        f"炸板率 {broken_text}，"
        f"最高连板 {int(snapshot.get('max_streak') or 0)}，"
        f"跌停 {int(snapshot.get('limit_down_count') or 0)}"
    )
    examples = snapshot.get("examples") or []
    if examples:
        bits = []
        for row in examples:
            seal = row.get("seal_time") or ""
            reason = row.get("reason") or ""
            extra = " ".join(part for part in (seal, reason) if part)
            bits.append(f"{row.get('code', '')} {row.get('name', '')}" + (f"（{extra}）" if extra else ""))
        st.caption("封板时间 / 原因：" + "；".join(bits))
    if snapshot.get("warning"):
        st.caption(snapshot["warning"])


def _regime_section() -> None:
    payload = load_json_object(REPORT_DIR / "regime_ml_latest.json")
    if not payload or not payload.get("comparison"):
        return
    st.subheader("行情过滤与模型排序")
    st.caption("这组和上面的五策略回测分开。样本外只评估了一次，参数没有按这段结果再调。")
    table = []
    window_name = {"full": "全样本", "oos": "样本外"}
    for row in payload["comparison"]:
        table.append(
            {
                "方案": row.get("label", ""),
                "区间": window_name.get(row.get("window", ""), row.get("window", "")),
                "总收益": _pct(float(row.get("total_return") or 0)),
                "最大回撤": _pct(float(row.get("max_drawdown") or 0)),
                "成交笔数": int(row.get("trade_count") or 0),
                "胜率": _pct(float(row.get("win_rate") or 0)),
                "平均盈利": _cny(float(row.get("avg_win") or 0)),
                "平均亏损": _cny(float(row.get("avg_loss") or 0)),
            }
        )
    st.dataframe(pd.DataFrame(table), hide_index=True, use_container_width=True)
    for row in payload["comparison"]:
        if row.get("window") == "oos" and row.get("verdict"):
            st.write(f"{row.get('label', '')}：{row['verdict']}")


def page_backtest() -> None:
    st.header("回测")
    st.caption("左边这一组和右边这一组不要混着看。全样本用的是事先定好的默认参数；样本外的参数只在更早的训练窗口里挑选。")
    bundle = load_backtest_bundle(REPORT_DIR / BUNDLE_NAME)
    if not bundle or not bundle.get("full_sample"):
        st.info("还没有回测数据。在仓库根目录运行 python -m ashare backtest 之后，刷新本页。")
        return
    full_rows = bundle["full_sample"]
    oos_rows = bundle["out_of_sample"]
    st.subheader("全样本（默认参数）")
    st.caption("同一套默认参数跑完整段历史。参数是对着这段行情定的，数字通常偏乐观。")
    _render_sample(full_rows, "全样本权益", "full_strategies", "全样本没有权益曲线。")
    st.subheader("走步样本外")
    st.caption("每一段测试开始前，只用更早的训练窗口挑参数，再交易后面没参与挑选的行情。")
    if not oos_rows:
        st.info("样本太短，没有走出样本外窗口。")
    else:
        _render_sample(oos_rows, "样本外权益", "oos_strategies", "样本外没有权益曲线。")
    _regime_section()
    notes = bundle.get("notes") or []
    if notes:
        with st.expander("数据说明"):
            for note in notes:
                st.write(note)


def page_paper() -> None:
    st.header("模拟盘")
    st.caption("账本只存在本机。这里没有券商连接。")
    ledger = load_ledger(LEDGER_PATH)
    if ledger is None:
        st.info("还没有模拟盘账本。在「今日候选」运行收盘更新后，候选会变成下一交易日开盘的待成交委托。成交之前，持仓、成交和盈亏曲线都是空的。")
        return
    cash = float(ledger.get("cash") or 0)
    initial = float(ledger.get("initial_cash") or cash)
    equity_rows = ledger.get("equity") or []
    latest_equity = float(equity_rows[-1]["equity"]) if equity_rows else cash
    c1, c2, c3 = st.columns(3)
    c1.metric("现金", _cny(cash))
    c2.metric("最近权益", _cny(latest_equity))
    c3.metric("相对本金", _cny(latest_equity - initial))
    st.subheader("持仓与卖出提示")
    positions = ledger.get("positions") or []
    if not positions:
        st.info("当前没有持仓。")
    else:
        holding = pd.DataFrame(positions)
        keep = [
            name
            for name in ("code", "name", "shares", "buy_date", "buy_price", "stop", "take_profit", "max_hold_days", "strategy_name", "sessions_held")
            if name in holding.columns
        ]
        renamed = holding[keep].rename(
            columns={
                "code": "代码",
                "name": "名称",
                "shares": "股数",
                "buy_date": "买入日",
                "buy_price": "买入价",
                "stop": "止损价",
                "take_profit": "止盈价",
                "max_hold_days": "最长持有",
                "strategy_name": "策略",
                "sessions_held": "已过可卖天数",
            }
        )
        st.dataframe(renamed, hide_index=True, use_container_width=True)
        for position in positions:
            label = f"{position.get('code', '')} {position.get('name', '')}"
            st.markdown(f"**{label}**")
            st.write(sell_hint(position))
    st.subheader("待成交委托")
    pending = [item for item in ledger.get("pending") or [] if item.get("status", "pending") == "pending"]
    if not pending:
        st.info("没有等待开盘的委托。")
    else:
        pending_frame = pd.DataFrame(pending)
        keep = [
            name
            for name in ("code", "name", "signal_date", "strategy_name", "entry_low", "entry_high", "max_hold_days", "shares_hint")
            if name in pending_frame.columns
        ]
        st.dataframe(
            pending_frame[keep].rename(
                columns={
                    "code": "代码",
                    "name": "名称",
                    "signal_date": "信号日",
                    "strategy_name": "策略",
                    "entry_low": "买入下限",
                    "entry_high": "买入上限",
                    "max_hold_days": "最长持有",
                    "shares_hint": "参考股数",
                }
            ),
            hide_index=True,
            use_container_width=True,
        )
    st.subheader("成交记录")
    fills = ledger.get("fills") or []
    if not fills:
        st.info("还没有成交。")
    else:
        st.dataframe(pd.DataFrame(fills), hide_index=True, use_container_width=True)
    st.subheader("盈亏曲线")
    if len(equity_rows) < 2:
        st.info("还没有足够的结算点，画不出盈亏曲线。下一个交易日收盘后运行 python -m ashare paper settle。")
        return
    curve = pd.DataFrame(
        {"日期": [item["date"] for item in equity_rows], "权益": [float(item["equity"]) for item in equity_rows]}
    )
    curve["策略"] = "模拟盘"
    st.altair_chart(_equity_chart(curve, "模拟盘权益"), use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="A股短线研究台", layout="wide")
    _apply_style()
    st.sidebar.title("A股短线研究台")
    st.sidebar.caption("本地页面。v1 不会下实盘单，历史回测也不是未来收益。")
    page = st.sidebar.radio("页面", ["今日候选", "回测", "模拟盘"], label_visibility="collapsed")
    if page == "今日候选":
        page_shortlist()
    elif page == "回测":
        page_backtest()
    else:
        page_paper()


main()
