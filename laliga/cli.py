"""Command-line interface for fetching, training, backtesting, and predicting."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from laliga.config import DEFAULT_MIN_TRAIN_MATCHES, DEFAULT_SEASONS, DEFAULT_XI, ModelConfig, api_token, project_data_dir
from laliga.data.client import SportMonksError, open_client
from laliga.data.football_data import HistoryError, format_history_build, import_history, load_history
from laliga.data.fetch import fetch_historical, fetch_window
from laliga.data.markets import fetch_stored_markets, format_market_fetch
from laliga.data.parse import fixtures_to_frame
from laliga.data.store import MatchStore
from laliga.model.dixon_coles import FitError
from laliga.model.service import ScorelineModel, fit_models, model_is_fresh
from laliga.report import (
    format_backtest,
    format_comparison,
    format_predictions,
    format_team_table,
    write_predictions_csv,
    write_predictions_json,
)
from laliga.schedule import scheduled_between, select_next_matchday
from laliga.synthetic import generate_synthetic_fixtures

logger = logging.getLogger(__name__)


class CliError(Exception):
    def __init__(self, message: str, exit_code: int = 2):
        super().__init__(message)
        self.exit_code = exit_code


def main(argv: list[str] | None = None) -> int:
    _configure_stdio()
    _load_env()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except CliError as exc:
        print(str(exc), file=sys.stderr)
        return exc.exit_code
    except (SportMonksError, FitError, HistoryError) as exc:
        print(str(exc), file=sys.stderr)
        return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="laliga",
        description="西甲（La Liga）比赛预测：全场胜平负、半场胜平负、最可能的三个比分。",
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

    markets = sub.add_parser(
        "fetch-markets",
        help="为已经存好的比赛回填赛前 1X2 赔率和 xG（不进每日更新）",
    )
    _add_data_dir(markets)
    markets.add_argument("--refresh", action="store_true", help="忽略本地缓存，并重新请求已经填过的比赛")
    markets.set_defaults(func=cmd_fetch_markets)

    history = sub.add_parser(
        "import-history",
        help="从 football-data.co.uk 下载 2012/13 之后的西甲，写入 processed/history.csv（不进每日更新）",
    )
    _add_data_dir(history)
    history.add_argument("--since", type=int, default=2012, help="从这个赛季的起始年下载，例如 2012")
    history.add_argument("--until", type=int, default=None, help="下载到这个起始年，默认到今年")
    history.add_argument("--refresh", action="store_true", help="忽略已缓存的 SP1 CSV，重新下载")
    history.set_defaults(func=cmd_import_history)

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
    backtest.set_defaults(func=cmd_backtest)

    compare = sub.add_parser(
        "compare",
        help="在同一次走步回测里比较 Dixon–Coles、历史频率基准和 XGBoost；有赔率时再混合两者（不进入每日更新）",
    )
    _add_model_args(compare)
    compare.add_argument(
        "--min-train",
        type=int,
        default=DEFAULT_MIN_TRAIN_MATCHES,
        help=f"每个评测日之前至少要有多少场完场比赛（默认 {DEFAULT_MIN_TRAIN_MATCHES}）",
    )
    compare.add_argument(
        "--output",
        type=str,
        default="",
        help="对照 JSON 的路径。SportMonks 默认 predictions/model_comparison.json，长历史默认 model_comparison_history.json",
    )
    compare.add_argument(
        "--history",
        choices=("sportmonks", "football-data"),
        default="sportmonks",
        help="sportmonks 读 matches.csv；football-data 读 import-history 写出的 history.csv",
    )
    compare.add_argument("--since", type=int, default=None, help="只评测这个起始年及以后的赛季，例如 2012")
    compare.add_argument(
        "--refit-every",
        type=int,
        default=0,
        help="Dixon–Coles 每隔多少个 UTC 日重拟合。0 表示 SportMonks 每天、football-data 每 7 天",
    )
    boost = compare.add_mutually_exclusive_group()
    boost.add_argument("--xgboost", dest="with_xgboost", action="store_const", const=True, help="训练 XGBoost 对照")
    boost.add_argument("--no-xgboost", dest="with_xgboost", action="store_const", const=False, help="不训练 XGBoost")
    compare.set_defaults(with_xgboost=None, func=cmd_compare)

    predict = sub.add_parser("predict", help="预测未开赛比赛")
    _add_model_args(predict)
    predict.add_argument("--from", dest="start", type=_parse_date, help="开始日期 YYYY-MM-DD（含）")
    predict.add_argument("--to", dest="end", type=_parse_date, help="结束日期 YYYY-MM-DD（含）")
    predict.add_argument("--next", action="store_true", help="预测下一轮（同一 round_id）")
    predict.add_argument("--refresh", action="store_true", help="预测前向 API 刷新这段赛程")
    predict.add_argument("--refit", action="store_true", help="即使已有模型文件也按赛前数据重新拟合")
    predict.add_argument("--csv", type=str, default="", help="把预测写入 CSV")
    predict.add_argument("--json", type=str, default="", help="把预测写入 JSON")
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


def cmd_import_history(args: argparse.Namespace) -> int:
    store = _store(args)
    sport = store.load()
    built = import_history(
        store.root,
        None if sport.empty else sport,
        since=args.since,
        until=args.until,
        refresh=args.refresh,
    )
    print(format_history_build(built))
    return 0


def cmd_fetch_markets(args: argparse.Namespace) -> int:
    store = _store(args)
    if store.load().empty:
        raise CliError("本地没有比赛。请先 `python -m laliga fetch`，再回填赔率和 xG。", exit_code=2)
    client = _client(store, refresh=args.refresh)
    result = fetch_stored_markets(client, store, force=args.refresh)
    print(format_market_fetch(result))
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
    report = walk_forward(history, config)
    output = Path(args.output) if args.output else None
    if output is not None:
        _write_utf8(output, json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    print(format_backtest(report))
    if output is not None:
        print(f"\n回测 JSON 已写入 {output}")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    from laliga.backtest import compare_models

    store = _store(args)
    if args.history == "football-data":
        history = load_history(store.root, since=args.since)
    else:
        history = _filter_since(_require_history(store), args.since)
    refit_every = args.refit_every
    if refit_every == 0:
        refit_every = 7 if args.history == "football-data" else 1
    if refit_every < 1:
        raise CliError("--refit-every 至少是 1。0 表示按数据源使用默认间隔。")
    with_xgboost = args.history != "football-data" if args.with_xgboost is None else args.with_xgboost
    config = _config(args, min_train=args.min_train)
    report = compare_models(
        history,
        config,
        refit_every_days=refit_every,
        with_xgboost=with_xgboost,
        history_source=args.history,
    )
    if args.output:
        output = Path(args.output)
    elif args.history == "football-data":
        output = store.predictions_dir / "model_comparison_history.json"
    else:
        output = store.predictions_dir / "model_comparison.json"
    _write_utf8(output, json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    print(format_comparison(report))
    print(f"\n对照 JSON 已写入 {output}")
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
    history = store.load()
    config = _config(args, min_train=4)
    predictions = _predict(history, fixtures, config, store, refit=args.refit)
    if args.csv:
        write_predictions_csv(predictions, Path(args.csv))
    if args.json:
        write_predictions_json(predictions, Path(args.json))
    print(format_predictions(predictions))
    if args.csv:
        print(f"\nCSV 已写入 {args.csv}")
    if args.json:
        print(f"JSON 已写入 {args.json}")
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    from laliga.backtest import walk_forward

    root = project_data_dir(args.data_dir) if args.data_dir else project_data_dir() / "demo"
    store = MatchStore(root)
    frame = fixtures_to_frame(generate_synthetic_fixtures(seed=args.seed))
    store.replace(frame)
    config = ModelConfig(xi=args.xi, max_iter=args.max_iter, min_train_matches=args.min_train, min_ht_matches=12)
    print(f"合成数据已写入 {store.processed_path}（{len(frame)} 场，不访问网络）。")
    report = walk_forward(frame, config)
    print()
    print(format_backtest(report))
    fixtures, note = select_next_matchday(frame)
    print()
    if note:
        print(note)
    if fixtures.empty:
        print("合成数据里没有未开赛轮次。")
        return 0
    model = fit_models(frame, pd.Timestamp(fixtures["starting_at"].min()), config)
    predictions = [model.predict_row(row) for _, row in fixtures.iterrows()]
    print()
    print("下一轮合成比赛：")
    print(format_predictions(predictions))
    return 0


def _predict(history, fixtures, config: ModelConfig, store: MatchStore, *, refit: bool):
    saved = None
    document = store.load_model_document()
    if document and not refit:
        try:
            saved = ScorelineModel.from_dict(document)
        except (KeyError, TypeError, ValueError, FitError) as exc:
            logger.warning("已保存的模型无法读取，将重新拟合：%s", exc)
            saved = None
    eligible = history[~history["fixture_id"].isin(set(fixtures["fixture_id"].astype(int)))]
    if saved is not None and model_is_fresh(saved, eligible, fixtures):
        print("使用已保存的模型（训练截止日期不早于最近完场、且早于这些比赛）。", file=sys.stderr)
        return [saved.predict_row(row) for _, row in fixtures.sort_values("starting_at").iterrows()]

    print("按每场比赛开球日前的完场数据重新拟合。", file=sys.stderr)
    predictions = []
    previous: ScorelineModel | None = None
    ordered = fixtures.sort_values(["starting_at", "fixture_id"])
    for day, group in ordered.groupby(ordered["starting_at"].dt.floor("D"), sort=True):
        model = fit_models(
            eligible,
            pd.Timestamp(day),
            config,
            initial_ft=None if previous is None else previous.ft,
            initial_ht=None if previous is None else previous.ht,
        )
        previous = model
        for _, row in group.iterrows():
            predictions.append(model.predict_row(row))
    return predictions


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


def _filter_since(frame: pd.DataFrame, since: int | None) -> pd.DataFrame:
    if since is None or frame.empty:
        return frame
    if "season_start" in frame.columns and frame["season_start"].notna().any():
        kept = frame[pd.to_numeric(frame["season_start"], errors="coerce") >= int(since)].copy()
    else:
        kickoff = pd.to_datetime(frame["starting_at"], utc=True).dt.tz_convert("Europe/Madrid")
        start_year = kickoff.dt.year.where(kickoff.dt.month >= 7, kickoff.dt.year - 1)
        kept = frame.loc[start_year >= int(since)].copy()
    if kept.empty:
        raise CliError(f"没有 {since} 年及以后开赛的赛季。")
    return kept.reset_index(drop=True)


def _config(args: argparse.Namespace, min_train: int) -> ModelConfig:
    return ModelConfig(
        xi=args.xi,
        max_iter=args.max_iter,
        min_matches=args.min_matches,
        min_train_matches=min_train,
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


def _configure_stdio() -> None:
    """Point CLI stdout and stderr at UTF-8 before any log line or table is printed.

    A Windows console already accepts Unicode. A pipe or redirected file uses
    the ANSI code page (GBK on a Chinese locale) and raises UnicodeEncodeError
    for characters the comparison table prints: the minus sign in paired
    labels, the en dash in Dixon–Coles, the em dash for a missing metric, and
    the box-drawing rule. That used to abort ``compare`` before the JSON
    existed. UTF-8 keeps those characters. If the stream refuses an encoding
    change, replacing unencodable characters is still better than crashing.
    """

    for stream in (sys.stdout, sys.stderr):
        if stream is None:
            continue
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError, AttributeError):
            try:
                reconfigure(errors="replace")
            except (OSError, ValueError, AttributeError):
                pass


def _write_utf8(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _load_env() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(override=False)
