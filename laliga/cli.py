"""Command-line interface for fetching, training, backtesting, and predicting."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date, datetime, timedelta, timezone

import pandas as pd

from laliga import pipeline
from laliga.config import (
    DEFAULT_MIN_TRAIN_MATCHES,
    DEFAULT_SEASONS,
    DEFAULT_XI,
    TRACKS,
    ModelConfig,
    PromotionConfig,
    api_token,
    project_data_dir,
)
from laliga.data.client import SportMonksError, open_client
from laliga.data.fetch import fetch_historical, fetch_window
from laliga.data.odds import fetch_odds, parse_odds
from laliga.data.parse import fixtures_to_frame
from laliga.data.store import MatchStore
from laliga.model import promotion
from laliga.model.dixon_coles import FitError
from laliga.model.service import fit_models
from laliga.report import (
    format_backtest,
    format_predictions,
    format_team_table,
    write_predictions_csv,
    write_predictions_json,
)
from laliga.schedule import scheduled_between, select_next_matchday
from laliga.synthetic import generate_synthetic_fixtures, generate_synthetic_odds

logger = logging.getLogger(__name__)


class CliError(Exception):
    def __init__(self, message: str, exit_code: int = 2):
        super().__init__(message)
        self.exit_code = exit_code


def main(argv: list[str] | None = None) -> int:
    _load_env()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except CliError as exc:
        print(str(exc), file=sys.stderr)
        return exc.exit_code
    except (SportMonksError, FitError) as exc:
        print(str(exc), file=sys.stderr)
        return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="laliga",
        description=(
            "西甲（La Liga）比赛预测：全场胜平负、半场胜平负、最可能的三个比分。"
            "三套模型：纯足球（Dixon–Coles）、纯市场（赔率）、综合；按回测准确率自动选用。"
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    fetch = sub.add_parser("fetch", help="从 SportMonks 抓取西甲赛季并写入本地缓存")
    _add_data_dir(fetch)
    fetch.add_argument(
        "--seasons",
        type=int,
        default=DEFAULT_SEASONS,
        help=(
            f"最近几个赛季，含当前赛季（默认 {DEFAULT_SEASONS}）。"
            "接口列出的更少时，按实际数量抓取，不会因为数字更大而失败。"
        ),
    )
    fetch.add_argument("--refresh", action="store_true", help="忽略本地原始响应缓存，重新请求 API")
    fetch.add_argument("--replace", action="store_true", help="用本次抓取结果覆盖本地比赛表，而不是按 fixture_id 合并")
    fetch.set_defaults(func=cmd_fetch)

    train = sub.add_parser("train", help="用全部完场比赛拟合模型并保存")
    _add_model_args(train)
    train.set_defaults(func=cmd_train)

    backtest = sub.add_parser("backtest", help="按时间顺序走步回测，并与历史频率基准比较")
    _add_model_args(backtest)
    backtest.add_argument(
        "--min-train",
        type=int,
        default=DEFAULT_MIN_TRAIN_MATCHES,
        help=f"每个评测日之前至少要有多少场完场比赛（默认 {DEFAULT_MIN_TRAIN_MATCHES}）",
    )
    backtest.add_argument("--output", type=str, default="", help="把回测 JSON 写到这个路径")
    _add_promotion_args(backtest)
    backtest.add_argument("--dry-run", action="store_true", help="只显示自动升级判定，不写入启用模型和综合权重")
    backtest.set_defaults(func=cmd_backtest)

    odds = sub.add_parser("odds", help="从 SportMonks 抓取本地比赛的赛前赔率（需要 Odds add-on）")
    _add_data_dir(odds)
    odds.add_argument("--scheduled-only", action="store_true", help="只抓未开赛比赛的赔率")
    odds.add_argument("--from", dest="start", type=_parse_date, help="只抓这一天（含）之后开球的比赛")
    odds.add_argument("--to", dest="end", type=_parse_date, help="只抓这一天（含）之前开球的比赛")
    odds.add_argument("--refresh", action="store_true", help="完场比赛也忽略缓存重新请求")
    odds.set_defaults(func=cmd_odds)

    promote = sub.add_parser("promote", help="按已保存的回测结果重新判定每个市场启用哪套模型")
    _add_data_dir(promote)
    promote.add_argument("--input", type=str, default="", help="回测 JSON，默认 <数据目录>/predictions/backtest.json")
    _add_promotion_args(promote)
    promote.add_argument("--dry-run", action="store_true", help="只显示判定，不写入")
    promote.set_defaults(func=cmd_promote)

    daily = sub.add_parser("daily", help="每日刷新：更新近期赛程和赔率、回测并自动升级、预测下一轮")
    _add_model_args(daily)
    daily.add_argument(
        "--min-train",
        type=int,
        default=DEFAULT_MIN_TRAIN_MATCHES,
        help=f"回测时每个评测日之前至少要有多少场完场比赛（默认 {DEFAULT_MIN_TRAIN_MATCHES}）",
    )
    daily.add_argument("--days-back", type=int, default=7, help="向前补齐多少天的赛果（默认 7）")
    daily.add_argument("--days-ahead", type=int, default=14, help="向后抓取多少天的赛程和赔率（默认 14）")
    daily.add_argument("--skip-backtest", action="store_true", help="不重跑回测，沿用上次的启用模型")
    daily.add_argument("--offline", action="store_true", help="不访问 SportMonks，只用本地缓存")
    _add_promotion_args(daily)
    daily.set_defaults(func=cmd_daily)

    predict = sub.add_parser("predict", help="预测未开赛比赛")
    _add_model_args(predict)
    predict.add_argument("--from", dest="start", type=_parse_date, help="开始日期 YYYY-MM-DD（含）")
    predict.add_argument("--to", dest="end", type=_parse_date, help="结束日期 YYYY-MM-DD（含）")
    predict.add_argument("--next", action="store_true", help="预测下一轮（同一 round_id）")
    predict.add_argument("--refresh", action="store_true", help="预测前向 API 刷新这段赛程")
    predict.add_argument("--refit", action="store_true", help="即使已有模型文件也按赛前数据重新拟合")
    predict.add_argument("--csv", type=str, default="", help="把预测写入 CSV")
    predict.add_argument("--json", type=str, default="", help="把预测写入 JSON")
    predict.add_argument(
        "--track",
        choices=("active", *TRACKS, "all"),
        default="active",
        help="终端显示哪套模型：active=按回测启用的模型（默认），all=三套都显示",
    )
    predict.add_argument("--no-odds", action="store_true", help="不请求赔率，只用本地已有的赔率")
    predict.set_defaults(func=cmd_predict)

    demo = sub.add_parser("demo", help="用合成数据离线跑一遍回测和下一轮预测（不需要 API token）")
    demo.add_argument("--data-dir", default="", help="合成数据目录，默认是 <数据目录>/demo")
    demo.add_argument("--seed", type=int, default=7)
    demo.add_argument("--min-train", type=int, default=18)
    demo.add_argument("--max-iter", type=int, default=80)
    demo.add_argument("--xi", type=float, default=DEFAULT_XI)
    demo.set_defaults(func=cmd_demo)

    web = sub.add_parser("web", help="在本机打开中文操作台")
    _add_data_dir(web)
    web.add_argument("--host", default="127.0.0.1", help="监听地址，默认只对本机开放")
    web.add_argument("--port", type=int, default=8765)
    web.add_argument(
        "--demo",
        action="store_true",
        help="用合成示例数据预填界面，每页都会标明不是真实西甲。默认关闭。",
    )
    web.set_defaults(func=cmd_web)
    return parser


def cmd_web(args: argparse.Namespace) -> int:
    from laliga.webapp import serve

    if args.demo and not args.data_dir:
        root = project_data_dir() / "web-demo"
    else:
        root = project_data_dir(args.data_dir or None)
    print(f"操作台：http://{args.host}:{args.port}")
    if args.demo:
        print("合成示例模式已打开。页面上的比赛不是真实西甲。关掉 --demo 后重启才会使用真实缓存。")
    else:
        print("未开启示例模式。没有 token 或本地缓存时，页面只显示空状态和错误，不会填入合成比赛。")
    serve(root, host=args.host, port=args.port, demo=args.demo)
    return 0


def cmd_fetch(args: argparse.Namespace) -> int:
    store = _store(args)
    client = _client(store, refresh=args.refresh)
    frame = fetch_historical(client, store, seasons=args.seasons, replace=args.replace)
    _print_inventory(frame)
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    store = _store(args)
    history = _require_history(store)
    config = _config(args, min_train=4)
    as_of = pd.Timestamp(history.loc[history["status"] == "finished", "starting_at"].max()) + pd.Timedelta(seconds=1)
    model = fit_models(history, as_of, config)
    store.save_model_document(model.to_dict())
    document = model.to_dict()
    print(format_team_table(document["ft"], "全场"))
    print()
    source = "半场进球单独拟合" if model.ht_source == "first_half" else "半场由全场参数按进球比例缩放（半场样本不足）"
    print(format_team_table(document["ht"], f"半场（{source}）"))
    print()
    print(f"模型已保存到 {store.model_path}")
    return 0


def cmd_backtest(args: argparse.Namespace) -> int:
    from laliga.backtest import walk_forward

    store = _store(args)
    history = _require_history(store)
    config = _config(args, min_train=args.min_train)
    report = walk_forward(
        history,
        config,
        store.load_odds(),
        promotion=_promotion_config(args),
        incumbent=promotion.load_active(store),
    )
    print(format_backtest(report))
    if args.output:
        from pathlib import Path

        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n回测 JSON 已写入 {output}")
    if args.dry_run:
        print("\n--dry-run：没有写入启用模型和综合权重。")
        return 0
    _persist_promotion(store, report)
    return 0


def cmd_odds(args: argparse.Namespace) -> int:
    store = _store(args)
    history = store.load()
    if history.empty:
        raise CliError("本地没有比赛。请先 `python -m laliga fetch`。", exit_code=2)
    statuses = ["scheduled"] if args.scheduled_only else ["scheduled", "finished"]
    fixtures = history[history["status"].isin(statuses)]
    if args.start:
        fixtures = fixtures[fixtures["starting_at"] >= pd.Timestamp(args.start, tz="UTC")]
    if args.end:
        fixtures = fixtures[fixtures["starting_at"] < pd.Timestamp(args.end + timedelta(days=1), tz="UTC")]
    if fixtures.empty:
        raise CliError("没有符合条件的比赛。", exit_code=1)
    client = _client(store, refresh=args.refresh)
    print(f"将请求 {len(fixtures)} 场比赛的赔率（每场一次请求；完场比赛优先读缓存）。", file=sys.stderr)
    summary = fetch_odds(client, store, fixtures)
    print(summary.message())
    if summary.with_odds == 0:
        print("这些比赛都没有返回赔率。订阅可能不含西甲赔率，或这些历史比赛没有赔率覆盖。纯市场和综合模型会标注为缺赔率。")
    print(f"赔率表：{store.odds_path}")
    return 0


def cmd_promote(args: argparse.Namespace) -> int:
    from pathlib import Path

    from laliga.backtest import rows_from_document

    store = _store(args)
    source = Path(args.input) if args.input else store.predictions_dir / "backtest.json"
    if not source.exists():
        raise CliError(f"找不到回测文件 {source}。请先运行 `python -m laliga backtest`。", exit_code=2)
    rows = rows_from_document(json.loads(source.read_text(encoding="utf-8")))
    if not rows:
        raise CliError("回测文件里没有逐场记录（旧版本回测）。请重新运行 `python -m laliga backtest`。", exit_code=2)
    decision = promotion.decide(rows, promotion.load_active(store), _promotion_config(args))
    for body in decision["markets"].values():
        print(body["reason"])
    if args.dry_run:
        print("--dry-run：没有写入。")
        return 0
    promotion.apply(store, decision)
    print(f"启用模型已写入 {store.active_path}")
    return 0


def cmd_daily(args: argparse.Namespace) -> int:
    from laliga.backtest import walk_forward

    store = _store(args)
    today = datetime.now(timezone.utc).date()
    start = today - timedelta(days=max(args.days_back, 0))
    end = today + timedelta(days=max(args.days_ahead, 1))
    if not args.offline:
        if not api_token():
            raise CliError("每日刷新需要 SPORTMONKS_API_TOKEN。只用本地缓存时加 --offline。", exit_code=2)
        fetch_window(_client(store, refresh=False), store, start, end)
        history = store.load()
        window = history[
            (history["starting_at"] >= pd.Timestamp(start, tz="UTC"))
            & (history["starting_at"] < pd.Timestamp(end + timedelta(days=1), tz="UTC"))
            & history["status"].isin(["scheduled", "finished"])
        ]
        pipeline.refresh_odds(store, window)
    history = _require_history(store)
    if not args.skip_backtest:
        report = walk_forward(
            history,
            _config(args, min_train=args.min_train),
            store.load_odds(),
            promotion=_promotion_config(args),
            incumbent=promotion.load_active(store),
        )
        print(format_backtest(report))
        _persist_promotion(store, report)
        print()
    fixtures, note = select_next_matchday(history)
    if fixtures.empty:
        raise CliError("本地没有未开赛比赛，今天没有可刷新的预测。", exit_code=1)
    if note:
        print(note, file=sys.stderr)
    forecasts = pipeline.forecast_fixtures(history, fixtures, _config(args, min_train=4), store, save_model=True)
    pipeline.write_latest(store, forecasts, "每日刷新：下一轮", note)
    print(format_predictions(forecasts, "active"))
    print(f"\n预测已写入 {store.predictions_dir / 'latest.json'}，操作台「预测结果」页会显示。")
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    if args.next == bool(args.start or args.end):
        raise CliError("请指定 --next，或同时指定 --from 和 --to。")
    if (args.start is None) ^ (args.end is None):
        raise CliError("--from 和 --to 需要一起使用。")
    store = _store(args)
    if args.next:
        fixtures, note = _load_next(store, args.refresh)
    else:
        fixtures = _load_range(store, args.start, args.end, args.refresh)
        note = ""
    if fixtures.empty:
        raise CliError("没有可预测的未开赛比赛。", exit_code=1)
    if note:
        print(note, file=sys.stderr)
    if not args.no_odds:
        pipeline.refresh_odds(store, fixtures)
    history = store.load()
    config = _config(args, min_train=4)
    predictions = pipeline.forecast_fixtures(history, fixtures, config, store, refit=args.refit)
    print(format_predictions(predictions, args.track))
    if args.csv:
        from pathlib import Path

        write_predictions_csv(predictions, Path(args.csv))
        print(f"\nCSV 已写入 {args.csv}")
    if args.json:
        from pathlib import Path

        write_predictions_json(predictions, Path(args.json))
        print(f"JSON 已写入 {args.json}")
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    from laliga.backtest import walk_forward

    root = project_data_dir(args.data_dir) if args.data_dir else project_data_dir() / "demo"
    store = MatchStore(root)
    raw = generate_synthetic_fixtures(seed=args.seed)
    frame = fixtures_to_frame(raw)
    store.replace(frame)
    odds = parse_odds(generate_synthetic_odds(raw, seed=args.seed + 4))
    store.upsert_odds(odds, {int(value) for value in frame["fixture_id"]})
    config = ModelConfig(xi=args.xi, max_iter=args.max_iter, min_train_matches=args.min_train, min_ht_matches=12)
    print(f"合成数据已写入 {store.processed_path}（{len(frame)} 场，含合成赔率，不访问网络）。")
    report = walk_forward(frame, config, odds, promotion=DEMO_PROMOTION, incumbent=promotion.load_active(store))
    print()
    print(format_backtest(report))
    pipeline.save_backtest(store, report, promote=True)
    fixtures, note = select_next_matchday(frame)
    print()
    if note:
        print(note)
    if fixtures.empty:
        print("合成数据里没有未开赛轮次。")
        return 0
    forecasts = pipeline.forecast_fixtures(frame, fixtures, config, store, refit=True)
    print()
    print("下一轮合成比赛：")
    print(format_predictions(forecasts, "all"))
    return 0


def _load_next(store: MatchStore, refresh: bool):
    history = store.load()
    fixtures, note = select_next_matchday(history)
    if refresh or fixtures.empty:
        token = api_token()
        if token:
            today = datetime.now(timezone.utc).date()
            client = _client(store, refresh=refresh)
            fetch_window(client, store, today, today + timedelta(days=45))
            history = store.load()
            fixtures, note = select_next_matchday(history)
        elif fixtures.empty:
            raise CliError(
                "本地没有未开赛比赛。请设置 SPORTMONKS_API_TOKEN 后运行 "
                "`python -m laliga fetch`，或用 `python -m laliga demo` 先看离线示例。",
                exit_code=2,
            )
    return fixtures, note


def _load_range(store: MatchStore, start: date, end: date, refresh: bool):
    if end < start:
        raise CliError("结束日期早于开始日期。")
    start_ts = pd.Timestamp(datetime.combine(start, datetime.min.time()), tz="UTC")
    end_ts = pd.Timestamp(datetime.combine(end + timedelta(days=1), datetime.min.time()), tz="UTC")
    history = store.load()
    fixtures = scheduled_between(history, start_ts, end_ts)
    if refresh or fixtures.empty:
        token = api_token()
        if token:
            client = _client(store, refresh=refresh)
            fetch_window(client, store, start, end)
            history = store.load()
            fixtures = scheduled_between(history, start_ts, end_ts)
        elif fixtures.empty:
            raise CliError(
                "该日期范围内没有本地未开赛比赛。请设置 SPORTMONKS_API_TOKEN 后执行 fetch，"
                "或用 --from/--to 指向已经写入比赛表的日期。",
                exit_code=2,
            )
    return fixtures


def _print_inventory(frame) -> None:
    finished = int((frame["status"] == "finished").sum())
    scheduled = int((frame["status"] == "scheduled").sum())
    print(f"本地比赛表现有 {len(frame)} 场：完场 {finished}，未开赛 {scheduled}。")
    if "season_name" in frame.columns:
        for name, part in frame.groupby(frame["season_name"].replace("", "未知赛季")):
            print(f"  {name}: {len(part)} 场")


def _require_history(store: MatchStore):
    history = store.load()
    finished = history[history["status"] == "finished"] if not history.empty else history
    if finished.empty:
        raise CliError("本地没有完场比赛。请先 `python -m laliga fetch`，或运行 `python -m laliga demo`。", exit_code=2)
    return history


def _store(args: argparse.Namespace) -> MatchStore:
    return MatchStore(project_data_dir(getattr(args, "data_dir", "") or None))


def _client(store: MatchStore, *, refresh: bool):
    try:
        return open_client(store.cache_dir, refresh=refresh)
    except SportMonksError as exc:
        raise CliError(str(exc), exit_code=2) from exc


def _config(args: argparse.Namespace, min_train: int) -> ModelConfig:
    return ModelConfig(
        xi=args.xi,
        max_iter=args.max_iter,
        min_matches=args.min_matches,
        min_train_matches=min_train,
    )


DEMO_PROMOTION = PromotionConfig(min_matches=20, min_ensemble_rows=20)


def _persist_promotion(store: MatchStore, report) -> None:
    pipeline.save_backtest(store, report, promote=True)
    print(f"\n回测已写入 {store.predictions_dir}；综合权重和启用模型已写入 {store.ensemble_path.parent}。")


def _promotion_config(args: argparse.Namespace) -> PromotionConfig:
    defaults = PromotionConfig()
    min_matches = getattr(args, "promote_min_matches", None)
    min_z = getattr(args, "promote_min_z", None)
    return PromotionConfig(
        min_matches=defaults.min_matches if min_matches is None else min_matches,
        min_z=defaults.min_z if min_z is None else min_z,
        min_ensemble_rows=defaults.min_ensemble_rows,
    )


def _add_promotion_args(parser: argparse.ArgumentParser) -> None:
    defaults = PromotionConfig()
    parser.add_argument(
        "--promote-min-matches",
        type=int,
        default=None,
        help=f"三套模型都有预测的回测比赛至少多少场才允许升级（默认 {defaults.min_matches}）",
    )
    parser.add_argument(
        "--promote-min-z",
        type=float,
        default=None,
        help=f"挑战者逐场配对改进至少要有几个标准误才升级（默认 {defaults.min_z}）",
    )


def _add_data_dir(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-dir", default="", help="数据目录，默认 ./data 或环境变量 LALIGA_DATA_DIR")


def _add_model_args(parser: argparse.ArgumentParser) -> None:
    _add_data_dir(parser)
    parser.add_argument("--xi", type=float, default=DEFAULT_XI, help=f"时间衰减系数，越大越偏重近期（默认 {DEFAULT_XI}）")
    parser.add_argument("--max-iter", type=int, default=150, help="每次拟合的最大迭代次数")
    parser.add_argument("--min-matches", type=int, default=10, help="少于此场次的球队向先验收缩")


def _parse_date(value: str) -> date:
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise argparse.ArgumentTypeError("日期格式应为 YYYY-MM-DD") from exc


def _load_env() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(override=False)
