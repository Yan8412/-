"""Command line entry. Examples are in the Chinese README."""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from datetime import date
from pathlib import Path

from ashare import __version__
from ashare.broker.paper import PaperBroker
from ashare.config import load_settings
from ashare.paths import cache_dir, default_config_path, ledger_path, report_dir
from ashare.pipeline import run_daily, run_research, settle_paper


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ashare",
        description="A股短线候选、回测和模拟盘。v1 不会向券商下单。",
    )
    parser.add_argument("--version", action="version", version=f"ashare {__version__}")
    parser.add_argument("--config", type=Path, default=None, help="可选 JSON 配置，默认读取项目根目录的 config.json")
    sub = parser.add_subparsers(dest="command", required=True)

    daily = sub.add_parser("daily", help="收盘后更新数据并写出下一交易日候选")
    daily.add_argument("--top", type=int, default=None)
    daily.add_argument("--universe", type=int, default=None, help="只用成交额前 N 只；省略则扫描全市场")
    daily.add_argument("--report-dir", type=Path, default=None)
    daily.add_argument("--cache-dir", type=Path, default=None)
    daily.add_argument("--paper", type=Path, default=None)
    daily.add_argument("--no-paper", action="store_true", help="只出报告，不写入模拟盘委托")

    backtest = sub.add_parser("backtest", help="用已实现的规则跑历史回测和走步样本外")
    backtest.add_argument("--start", default=None, help="YYYY-MM-DD，默认大约两年")
    backtest.add_argument("--universe", type=int, default=None, help="只用成交额前 N 只；省略则扫描全市场")
    backtest.add_argument("--report-dir", type=Path, default=None)
    backtest.add_argument("--cache-dir", type=Path, default=None)

    research = sub.add_parser("research", help="行情过滤和模型排序的一次对照回测，不改样本外参数")
    research.add_argument("--report-dir", type=Path, default=None)
    research.add_argument("--cache-dir", type=Path, default=None)

    train = sub.add_parser("train", help="用已经完成的交易重训排序模型，超参仍只在样本外之前选定")
    train.add_argument("--report-dir", type=Path, default=None)
    train.add_argument("--cache-dir", type=Path, default=None)

    paper = sub.add_parser("paper", help="查看或结算模拟盘")
    paper_sub = paper.add_subparsers(dest="paper_command", required=True)
    status = paper_sub.add_parser("status", help="打印现金、持仓和未成交委托")
    status.add_argument("--ledger", type=Path, default=None)
    settle = paper_sub.add_parser("settle", help="用最新日线撮合待成交委托并更新盈亏")
    settle.add_argument("--ledger", type=Path, default=None)
    settle.add_argument("--cache-dir", type=Path, default=None)

    dashboard = sub.add_parser("dashboard", help="打开本地中文看盘页面")
    dashboard.add_argument("--port", type=int, default=8501)
    dashboard.add_argument("--report-dir", type=Path, default=None)
    dashboard.add_argument("--cache-dir", type=Path, default=None)
    dashboard.add_argument("--ledger", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    settings = load_settings(args.config)
    today = date.today()

    if args.command == "daily":
        if args.top is not None:
            settings.top_n = args.top
        if args.universe is not None:
            settings.universe_size = args.universe
            settings.full_market = False
        path = run_daily(
            settings,
            today,
            cache_dir(args.cache_dir),
            report_dir(args.report_dir),
            None if args.no_paper else ledger_path(args.paper),
        )
        print(f"已写出 {path}")
        return 0

    if args.command == "backtest":
        if args.universe is not None:
            settings.universe_size = args.universe
            settings.full_market = False
        start = date.fromisoformat(args.start) if args.start else None
        path = run_research(settings, today, cache_dir(args.cache_dir), report_dir(args.report_dir), start)
        print(f"已写出 {path}")
        return 0

    if args.command == "research":
        from ashare.research import run_regime_research

        path = run_regime_research(
            settings, today, cache_dir(args.cache_dir), report_dir(args.report_dir)
        )
        print(f"已写出 {path}")
        return 0

    if args.command == "train":
        from ashare.research import run_train

        path = run_train(settings, today, cache_dir(args.cache_dir), report_dir(args.report_dir))
        print(f"已保存 {path}")
        return 0

    if args.command == "paper" and args.paper_command == "status":
        broker = PaperBroker(ledger_path(args.ledger), settings)
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
        for line in settle_paper(settings, cache_dir(args.cache_dir), ledger_path(args.ledger), today):
            print(line)
        return 0

    if args.command == "dashboard":
        config = Path(args.config).expanduser().resolve() if args.config else default_config_path()
        return _launch_dashboard(
            args.port,
            report_dir(args.report_dir),
            cache_dir(args.cache_dir),
            ledger_path(args.ledger),
            config,
        )

    parser.error("未知命令")
    return 2


def dashboard_command(
    port: int,
    report_dir: Path,
    cache_dir: Path,
    ledger: Path,
    config: Path,
) -> tuple[list[str], dict[str, str]]:
    """Streamlit command that does not stop on the first-run email prompt."""
    app = Path(__file__).resolve().parent / "dashboard_app.py"
    env = os.environ.copy()
    env["ASHARE_REPORT_DIR"] = str(report_dir)
    env["ASHARE_CACHE_DIR"] = str(cache_dir)
    env["ASHARE_LEDGER"] = str(ledger)
    env["ASHARE_CONFIG"] = str(config)
    env["STREAMLIT_SERVER_HEADLESS"] = "true"
    env["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app),
        "--server.port",
        str(port),
        "--server.address",
        "127.0.0.1",
        "--server.headless",
        "true",
        "--browser.gatherUsageStats",
        "false",
    ]
    return command, env


def _launch_dashboard(port: int, report_dir: Path, cache_dir: Path, ledger: Path, config: Path) -> int:
    """Start the Streamlit page. Same command on Windows and Linux."""
    command, env = dashboard_command(port, report_dir, cache_dir, ledger, config)
    print(f"看盘页面： http://127.0.0.1:{port}")
    return subprocess.call(command, env=env)


if __name__ == "__main__":
    sys.exit(main())
