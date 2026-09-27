"""Command line entry. Examples are in the Chinese README."""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

from ashare import __version__
from ashare.broker.paper import PaperBroker
from ashare.config import load_settings
from ashare.pipeline import run_daily, run_research, settle_paper


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ashare",
        description="A股短线候选、回测和模拟盘。v1 不会向券商下单。",
    )
    parser.add_argument("--version", action="version", version=f"ashare {__version__}")
    parser.add_argument("--config", type=Path, default=None, help="可选 JSON 配置，默认读取 ./config.json")
    sub = parser.add_subparsers(dest="command", required=True)

    daily = sub.add_parser("daily", help="收盘后更新数据并写出下一交易日候选")
    daily.add_argument("--top", type=int, default=None)
    daily.add_argument("--universe", type=int, default=None)
    daily.add_argument("--report-dir", type=Path, default=Path("reports"))
    daily.add_argument("--cache-dir", type=Path, default=Path("data/cache"))
    daily.add_argument("--paper", type=Path, default=Path("data/paper/ledger.json"))
    daily.add_argument("--no-paper", action="store_true", help="只出报告，不写入模拟盘委托")

    backtest = sub.add_parser("backtest", help="用已实现的规则跑历史回测和走步样本外")
    backtest.add_argument("--start", default=None, help="YYYY-MM-DD，默认大约两年")
    backtest.add_argument("--universe", type=int, default=None)
    backtest.add_argument("--report-dir", type=Path, default=Path("reports"))
    backtest.add_argument("--cache-dir", type=Path, default=Path("data/cache"))

    paper = sub.add_parser("paper", help="查看或结算模拟盘")
    paper_sub = paper.add_subparsers(dest="paper_command", required=True)
    status = paper_sub.add_parser("status", help="打印现金、持仓和未成交委托")
    status.add_argument("--ledger", type=Path, default=Path("data/paper/ledger.json"))
    settle = paper_sub.add_parser("settle", help="用最新日线撮合待成交委托并更新盈亏")
    settle.add_argument("--ledger", type=Path, default=Path("data/paper/ledger.json"))
    settle.add_argument("--cache-dir", type=Path, default=Path("data/cache"))

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    settings = load_settings(args.config) if args.config else load_settings(Path("config.json"))
    today = date.today()

    if args.command == "daily":
        if args.top is not None:
            settings.top_n = args.top
        if args.universe is not None:
            settings.universe_size = args.universe
        path = run_daily(
            settings,
            today,
            args.cache_dir,
            args.report_dir,
            None if args.no_paper else args.paper,
        )
        print(f"已写出 {path}")
        return 0

    if args.command == "backtest":
        if args.universe is not None:
            settings.universe_size = args.universe
        start = date.fromisoformat(args.start) if args.start else None
        path = run_research(settings, today, args.cache_dir, args.report_dir, start)
        print(f"已写出 {path}")
        return 0

    if args.command == "paper" and args.paper_command == "status":
        broker = PaperBroker(args.ledger, settings)
        snap = broker.snapshot()
        print(f"现金 {snap.cash:.2f} 元")
        print(f"最近权益 {snap.equity:.2f} 元")
        if not snap.positions:
            print("持仓：空")
        for pos in snap.positions:
            print(
                f"持仓 {pos['code']} {pos['name']} {pos['shares']} 股，"
                f"成本价 {pos['buy_price']}，买入日 {pos['buy_date']}，策略 {pos['strategy_name']}"
            )
        if not snap.pending:
            print("待成交：无")
        for order in snap.pending:
            print(f"待成交 {order['code']} {order['name']} 信号日 {order['signal_date']} {order['strategy_name']}")
        return 0

    if args.command == "paper" and args.paper_command == "settle":
        for line in settle_paper(settings, args.cache_dir, args.ledger, today):
            print(line)
        return 0

    parser.error("未知命令")
    return 2


if __name__ == "__main__":
    sys.exit(main())
