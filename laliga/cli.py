"""Command-line interface for fetching, training, backtesting, and predicting."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from laliga.archive import PredictionArchive, history_report, reconstruct_season
from dataclasses import replace

from laliga.config import DEFAULT_MIN_TRAIN_MATCHES, DEFAULT_SEASONS, DEFAULT_XI, ModelConfig, api_token, project_data_dir
from laliga.data.client import SportMonksError, open_client
from laliga.data.football_data import HistoryError, format_history_build, import_history, load_history
from laliga.data.fetch import fetch_historical, fetch_window
from laliga.data.markets import fetch_stored_markets, format_market_fetch
from laliga.data.parse import fixtures_to_frame
from laliga.data.store import MatchStore
from laliga.model.dixon_coles import FitError
from laliga.model.service import ScorelineModel, fit_models
from laliga.pipeline import (
    annotate_market_items,
    archive_alternative_tracks,
    describe_alternative_tracks,
    archive_scored,
    dashboard_items,
    score_fixtures,
    write_dashboard_latest,
    write_latest_next,
)
from laliga.report import (
    format_availability_report,
    format_backtest,
    format_comparison,
    format_history,
    format_predictions,
    format_team_table,
    latest_next_problem,
    write_predictions_csv,
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
        description="西甲（La Liga）比赛预测：联合半全场分布下的全场胜平负、半场胜平负、前两比分（仍保留前三）和半全场。",
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
    train.add_argument("--ignore-champion", action="store_true", help="不用注册表里的冠军参数")
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
    backtest.add_argument("--ignore-champion", action="store_true", help="不用注册表里的冠军参数")
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
    compare.add_argument("--ignore-champion", action="store_true", help="不用注册表里的冠军参数")
    compare.set_defaults(with_xgboost=None, func=cmd_compare)

    tracks = sub.add_parser(
        "compare-tracks",
        help="走步比较纯足球、市场反推进球、以及两者的混合（不进每日更新）",
    )
    _add_model_args(tracks)
    tracks.add_argument("--min-train", type=int, default=DEFAULT_MIN_TRAIN_MATCHES)
    tracks.add_argument("--output", type=str, default="", help="默认 predictions/tracks_comparison.json")
    tracks.add_argument(
        "--refit-every",
        type=int,
        default=0,
        help="每隔多少个 UTC 日重拟合足球模型。0 表示 SportMonks 每天、football-data 每 7 天",
    )
    tracks.add_argument("--history", choices=("sportmonks", "football-data"), default="sportmonks")
    tracks.add_argument("--since", type=int, default=None)
    tracks.add_argument("--ignore-champion", action="store_true")
    tracks.set_defaults(func=cmd_compare_tracks)

    upgrade = sub.add_parser("upgrade", help="按样本外准确率调参，只有挑战者更好才升为冠军")
    _add_model_args(upgrade)
    upgrade.add_argument("--trials", type=int, default=12, help="optuna 试验次数。没装 optuna 时改用少量固定参数")
    upgrade.add_argument("--holdout", type=int, default=80, help="留作样本外的最近完场场数，不参与调参")
    upgrade.add_argument("--min-train", type=int, default=DEFAULT_MIN_TRAIN_MATCHES)
    upgrade.add_argument("--min-promote", type=int, default=40, help="样本外少于此场数时不更换已有冠军")
    upgrade.add_argument("--rule", choices=("bootstrap", "strict"), default="bootstrap")
    upgrade.add_argument("--refit-every", type=int, default=14)
    upgrade.add_argument("--rollback", action="store_true", help="退回上一版冠军，不重新搜索")
    upgrade.add_argument("--history", choices=("sportmonks", "football-data"), default="sportmonks")
    upgrade.add_argument("--since", type=int, default=None, help="football-data 历史从哪个赛季起始年算起")
    upgrade.set_defaults(func=cmd_upgrade)

    probe = sub.add_parser("probe", help="用真实 token 查看最近西甲比赛实际返回了哪些赔率市场和 xG 字段")
    _add_data_dir(probe)
    probe.add_argument("--days", type=int, default=21, help="往回看多少天")
    probe.add_argument("--limit", type=int, default=6, help="最多检查多少场")
    probe.add_argument("--json", type=str, default="", help="把探测结果写入这个 JSON")
    probe.set_defaults(func=cmd_probe)

    predict = sub.add_parser("predict", help="预测未开赛比赛")
    _add_model_args(predict)
    predict.add_argument("--from", dest="start", type=_parse_date, help="开始日期 YYYY-MM-DD（含）")
    predict.add_argument("--to", dest="end", type=_parse_date, help="结束日期 YYYY-MM-DD（含）")
    predict.add_argument("--next", action="store_true", help="预测下一轮（同一 round_id）")
    predict.add_argument("--refresh", action="store_true", help="预测前向 API 刷新这段赛程")
    predict.add_argument("--refit", action="store_true", help="即使已有模型文件也按赛前数据重新拟合")
    predict.add_argument("--csv", type=str, default="", help="把预测写入 CSV")
    predict.add_argument("--json", type=str, default="", help="把预测写入 JSON（每日脚本的 latest_next.json 用这个，字段不变）")
    predict.add_argument(
        "--publish",
        action="store_true",
        help="同时写操作台读取的 predictions/latest.json 和 latest.csv（含比分矩阵）",
    )
    predict.add_argument("--no-archive", action="store_true", help="不写入 archive/predictions.jsonl（临时查看时用）")
    predict.add_argument("--ignore-champion", action="store_true", help="不用注册表里的冠军参数")
    predict.add_argument(
        "--availability",
        action="store_true",
        help="按伤停/首发调整期望进球。默认关闭；需要先 fetch-availability 和 compare-availability",
    )
    predict.set_defaults(func=cmd_predict)

    history_cmd = sub.add_parser("history", help="开球前存档的预测与赛果对照，以及基准命中率")
    _add_data_dir(history_cmd)
    history_cmd.add_argument("--json", type=str, default="", help="把完整历史写入 JSON")
    history_cmd.add_argument("--limit", type=int, default=20, help="终端最多显示最近多少场（默认 20）")
    history_cmd.set_defaults(func=cmd_history)

    backfill = sub.add_parser(
        "backfill-archive",
        help="用严格时间顺序的走步回测重建本赛季已赛比赛（标为回测重建，不算实时预测）",
    )
    _add_model_args(backfill)
    backfill.add_argument("--season-id", type=int, default=None, help="赛季 id，默认是比赛表里最新的赛季")
    backfill.add_argument(
        "--min-train",
        type=int,
        default=DEFAULT_MIN_TRAIN_MATCHES,
        help=f"每个比赛日之前至少要有多少场完场比赛（默认 {DEFAULT_MIN_TRAIN_MATCHES}）",
    )
    backfill.set_defaults(func=cmd_backfill_archive)

    fresh = sub.add_parser("check-freshness", help="每日更新前的数据新鲜度检查：正常退出码 0，STALE 退出码 2")
    _add_data_dir(fresh)
    fresh.add_argument("--now", type=str, default="", help="用这个 UTC 时间代替当前时间（也可用环境变量 LALIGA_FRESHNESS_NOW）")
    fresh.add_argument("--json", action="store_true", help="输出 JSON")
    fresh.set_defaults(func=cmd_check_freshness)

    doctor = sub.add_parser("doctor", help="部署后快速自检：数据、模型、预测文件格式、存档、新鲜度")
    _add_data_dir(doctor)
    doctor.set_defaults(func=cmd_doctor)

    avail = sub.add_parser("fetch-availability", help="为已存比赛回填首发阵容和伤停（不进每日更新，除非打开伤停开关）")
    _add_data_dir(avail)
    avail.add_argument("--refresh", action="store_true", help="完场比赛也重新请求")
    avail.add_argument("--days-ahead", type=int, default=14, help="未来多少天内的未开赛比赛每次重新请求（默认 14）")
    avail.set_defaults(func=cmd_fetch_availability)

    avail_compare = sub.add_parser(
        "compare-availability",
        help="走步评估伤停/首发调整，对比 Dixon–Coles 与历史频率基准，并保存系数",
    )
    _add_model_args(avail_compare)
    avail_compare.add_argument(
        "--min-train",
        type=int,
        default=DEFAULT_MIN_TRAIN_MATCHES,
        help=f"每个评测日之前至少要有多少场完场比赛（默认 {DEFAULT_MIN_TRAIN_MATCHES}）",
    )
    avail_compare.add_argument("--output", type=str, default="", help="默认 predictions/availability_comparison.json")
    avail_compare.set_defaults(func=cmd_compare_availability)

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
    print(format_team_table(document["ht"], f"上半场（{source}）"))
    print()
    if model.joint_model and model.sh is not None:
        print(format_team_table(document["sh"], "下半场"))
        print()
        print(
            f"联合分布：上半场 × 下半场，状态系数 η = {model.state_eta:.3f}。"
            "全场胜平负、半场胜平负、比分和半全场都从这张分布边际得到。"
        )
        if model.xg_weight:
            print(f"xG 权重 {model.xg_weight:.3f}（几何混合到上下半场期望进球）。")
    else:
        print("半场样本不足，未做联合分布。半场概率和全场概率各自加总为 1，但不保证半场比分不超过全场。")
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


def cmd_compare_tracks(args: argparse.Namespace) -> int:
    from laliga.model.tracks import compare_tracks, quotes_for_root
    from laliga.report import format_tracks

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
    config = _config(args, min_train=args.min_train)

    def progress(text: str) -> None:
        print(text, file=sys.stderr, flush=True)

    report = compare_tracks(
        history,
        config,
        quotes=quotes_for_root(store.root),
        refit_every_days=refit_every,
        notify=progress,
    )
    output = Path(args.output) if args.output else store.predictions_dir / "tracks_comparison.json"
    _write_utf8(output, json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    print(format_tracks(report.to_dict()))
    print(f"\n三条预测对照已写入 {output}")
    return 0


def cmd_upgrade(args: argparse.Namespace) -> int:
    from laliga.model.tracks import quotes_for_root
    from laliga.upgrade import format_upgrade_report, rollback, upgrade

    store = _store(args)
    if args.rollback:
        print(rollback(store.root))
        return 0
    if args.history == "football-data":
        history = load_history(store.root, since=args.since)
    else:
        history = _require_history(store)
    config = _config(args, min_train=args.min_train)
    # Search from the defaults, not from the current champion, so a challenger can move.
    config = replace(config, xg_weight=0.0, eta_scale=1.0, nb_dispersion=1.0, sh_rate_scale=1.0, temperature=1.0)
    result = upgrade(
        history,
        config,
        store.root,
        quotes=quotes_for_root(store.root),
        n_trials=args.trials,
        holdout_matches=args.holdout,
        min_promote=args.min_promote,
        rule=args.rule,
        refit_every_days=max(args.refit_every, 1),
    )
    print(format_upgrade_report(result.report))
    print(f"\n注册表：{store.root / 'models' / 'registry.json'}")
    print(f"准确率记录：{store.root / 'predictions' / 'accuracy_history.md'}")
    return 0


def cmd_probe(args: argparse.Namespace) -> int:
    from laliga.data.probe import format_probe, probe_recent

    store = _store(args)
    client = _client(store, refresh=False)
    report = probe_recent(client, days=args.days, limit=args.limit)
    print(format_probe(report))
    if args.json:
        _write_utf8(Path(args.json), json.dumps(report, ensure_ascii=False, indent=2))
        print(f"\n探测 JSON 已写入 {args.json}")
    if report.get("error") and not report.get("fixtures"):
        return 1
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
    adjuster = _availability_adjuster(store) if args.availability else None
    scored = score_fixtures(
        history,
        fixtures,
        config,
        store,
        refit=args.refit,
        adjuster=adjuster,
        notify=lambda text: print(text, file=sys.stderr),
    )
    predictions = [item.prediction for item in scored]
    if args.csv:
        write_predictions_csv(predictions, Path(args.csv))
    if args.json:
        write_latest_next(Path(args.json), predictions)
    if args.publish:
        label = "下一轮" if args.next else f"{args.start.isoformat()} 至 {args.end.isoformat()}"
        items = dashboard_items(scored)
        annotate_market_items(items, scored, history, store.root)
        write_dashboard_latest(store, items, label=label, note=note)
    archived = None
    if not args.no_archive:
        origin = "cli:predict --next" if args.next else "cli:predict --from/--to"
        archived = archive_scored(store, scored, history, origin=origin)
        others = archive_alternative_tracks(store, scored, history, origin=origin)
        if others:
            print(f"市场/混合存档：另写入 {others} 行（不计入纯足球命中率）。")
    print(format_predictions(predictions))
    print(describe_alternative_tracks(scored, history, store.root))
    if args.csv:
        print(f"\nCSV 已写入 {args.csv}")
    if args.json:
        print(f"JSON 已写入 {args.json}")
    if args.publish:
        print(f"操作台预测已写入 {store.predictions_dir / 'latest.json'}")
    if archived is not None:
        print(f"预测存档：写入 {archived.accepted} 场，拒绝 {len(archived.refused)} 场（已开球或不是未开赛）。")
    if adjuster is not None:
        used = sum(1 for item in scored if item.availability and item.availability.get("used"))
        print(f"伤停/首发调整：{used}/{len(scored)} 场生效。其余场次等于 Dixon–Coles。")
    return 0


def _availability_adjuster(store: MatchStore):
    from laliga.data.availability import AvailabilityStore
    from laliga.model.availability import AvailabilityFeatures, load_coefficients, make_adjuster

    coefficients = load_coefficients(store.root)
    if coefficients is None:
        raise CliError(
            "没有伤停系数文件 models/availability.json。先运行 "
            "`python -m laliga fetch-availability` 和 `python -m laliga compare-availability`。",
            exit_code=2,
        )
    lineups, sidelined, fetched = AvailabilityStore(store.root).load()
    return make_adjuster(AvailabilityFeatures(store.load(), lineups, sidelined, fetched), coefficients)


def cmd_history(args: argparse.Namespace) -> int:
    store = _store(args)
    report = history_report(store.root, store.load())
    if args.json:
        _write_utf8(Path(args.json), json.dumps(report, ensure_ascii=False, indent=2))
    print(format_history(report, limit=args.limit))
    if args.json:
        print(f"\n历史 JSON 已写入 {args.json}")
    return 0


def cmd_backfill_archive(args: argparse.Namespace) -> int:
    store = _store(args)
    history = _require_history(store)
    config = _config(args, min_train=args.min_train)
    built = reconstruct_season(history, config, season_id=args.season_id)
    if not built.records:
        raise CliError(
            f"赛季 {built.season_id} 没有可以重建的完场比赛（跳过 {built.skipped_days} 个训练样本不足的比赛日）。"
            "可以降低 --min-train。",
            exit_code=1,
        )
    PredictionArchive(store.root).replace_reconstruction(built.records)
    print(
        f"回测重建：赛季 {built.season_id}，{len(built.records)} 场，{built.days} 个比赛日"
        f"（跳过 {built.skipped_days} 个）。每个比赛日只用当天 00:00 UTC 之前的完场比赛。"
    )
    print(f"写入 {PredictionArchive(store.root).reconstruction_path}。这些是事后重建，不是当时的实时预测，单独统计。")
    return 0


def cmd_check_freshness(args: argparse.Namespace) -> int:
    from laliga.freshness import check_freshness, resolve_now

    store = _store(args)
    try:
        now = resolve_now(args.now or None)
    except ValueError as exc:
        raise CliError(str(exc), exit_code=1) from exc
    seasons_doc = None
    if store.seasons_path.exists():
        seasons_doc = json.loads(store.seasons_path.read_text(encoding="utf-8"))
    result = check_freshness(store.load(), now, seasons_doc=seasons_doc)
    if args.json:
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    else:
        print(result.format())
    return result.exit_code


def cmd_doctor(args: argparse.Namespace) -> int:
    from laliga import __version__
    from laliga.freshness import check_freshness, resolve_now

    store = _store(args)
    problems: list[str] = []
    print(f"laliga {__version__}，Python {sys.version.split()[0]}")
    print(f"数据目录：{store.root}")
    frame = store.load()
    finished = int((frame["status"] == "finished").sum()) if not frame.empty else 0
    scheduled = int((frame["status"] == "scheduled").sum()) if not frame.empty else 0
    print(f"比赛表：{len(frame)} 场（完场 {finished}，未开赛 {scheduled}）")
    if frame.empty:
        problems.append("比赛表是空的")
    document = store.load_model_document()
    if document:
        try:
            model = ScorelineModel.from_dict(document)
            print(f"模型：训练截止 {model.ft.trained_through}，样本 {model.ft.n_matches} 场")
        except (KeyError, TypeError, ValueError, FitError) as exc:
            problems.append(f"模型文件无法读取：{exc}")
    else:
        print("模型：还没有 models/dixon_coles.json")
    latest_next = store.predictions_dir / "latest_next.json"
    if latest_next.exists():
        issue = latest_next_problem(json.loads(latest_next.read_text(encoding="utf-8")))
        if issue:
            problems.append(f"latest_next.json 格式不对：{issue}")
        else:
            print("latest_next.json：字段完整")
    else:
        print("latest_next.json：还没有（每日脚本跑过一次后会出现）")
    archive = PredictionArchive(store.root)
    print(f"预测存档：实时 {len(archive.live_records())} 行，回测重建 {len(archive.reconstruction_records())} 行")
    seasons_doc = json.loads(store.seasons_path.read_text(encoding="utf-8")) if store.seasons_path.exists() else None
    fresh = check_freshness(frame, resolve_now(), seasons_doc=seasons_doc)
    print(fresh.format())
    for problem in problems:
        print(f"问题：{problem}", file=sys.stderr)
    if problems:
        return 1
    print("自检通过。" if not fresh.stale else "自检通过，但数据 STALE。")
    return 0


def cmd_fetch_availability(args: argparse.Namespace) -> int:
    from laliga.data.availability import AvailabilityStore, fetch_availability, format_availability_fetch

    store = _store(args)
    if store.load().empty:
        raise CliError("本地没有比赛。请先 `python -m laliga fetch`，再回填阵容和伤停。", exit_code=2)
    client = _client(store, refresh=False)
    result = fetch_availability(client, store, AvailabilityStore(store.root), force=args.refresh, days_ahead=args.days_ahead)
    print(format_availability_fetch(result))
    return 1 if result.denied else 0


def cmd_compare_availability(args: argparse.Namespace) -> int:
    from laliga.data.availability import AvailabilityStore
    from laliga.model.availability import AvailabilityFeatures, compare_availability, save_coefficients

    store = _store(args)
    history = _require_history(store)
    lineups, sidelined, fetched = AvailabilityStore(store.root).load()
    if fetched.empty:
        raise CliError("还没有阵容/伤停数据。先运行 `python -m laliga fetch-availability`。", exit_code=2)
    config = _config(args, min_train=args.min_train)
    report = compare_availability(history, AvailabilityFeatures(history, lineups, sidelined, fetched), config)
    output = Path(args.output) if args.output else store.predictions_dir / "availability_comparison.json"
    _write_utf8(output, json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    saved = save_coefficients(store.root, report)
    print(format_availability_report(report))
    print(f"\n评估 JSON 已写入 {output}")
    print(f"系数已写入 {saved}（只有 predict --availability 会读它）")
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
    config = ModelConfig(
        xi=args.xi,
        max_iter=args.max_iter,
        min_matches=args.min_matches,
        min_train_matches=min_train,
    )
    if getattr(args, "ignore_champion", False):
        return config
    from laliga.upgrade import apply_champion

    config = apply_champion(config, project_data_dir(getattr(args, "data_dir", "") or None))
    if getattr(args, "xi", DEFAULT_XI) != DEFAULT_XI:
        config = replace(config, xi=args.xi)
    return config


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
