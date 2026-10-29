"""Local dashboard. Bind to localhost and call the same pipeline as the CLI.

Synthetic matches are loaded only when the process is started with demo=True.
That mode prints a banner on every page. The default mode shows an empty state
until a real SportMonks fetch writes the local cache.
"""

from __future__ import annotations

import json
import logging
import threading
import uuid
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from flask import Flask, abort, redirect, render_template, request, send_file, url_for
from werkzeug.serving import make_server

from laliga import pipeline
from laliga.backtest import walk_forward
from laliga.config import (
    DEFAULT_SEASONS,
    PROMOTION_MARKET_LABELS,
    PROMOTION_MARKETS,
    TRACK_LABELS,
    TRACKS,
    ModelConfig,
    PromotionConfig,
    api_token,
)
from laliga.data.client import SportMonksError, open_client
from laliga.data.fetch import fetch_historical, fetch_window
from laliga.data.odds import fetch_odds
from laliga.data.store import MatchStore
from laliga.model import promotion
from laliga.model.dixon_coles import FitError
from laliga.model.market import SOURCE_LABELS
from laliga.model.service import fit_models
from laliga.report import backtest_metric_rows, split_kickoff
from laliga.schedule import scheduled_between, select_next_matchday

WEB_MAX_ITER = 80
DEMO_FETCH_MESSAGE = (
    "示例模式不会请求 SportMonks，避免把真实响应写进示例目录。"
    "请停掉这个进程，执行 python -m laliga web（不要加 --demo），再点「更新数据」。"
)
DEMO_PROMOTION = PromotionConfig(min_matches=20, min_ensemble_rows=20)
VIEW_NAMES = ("active", *TRACKS)
VIEW_LABELS = {"active": "启用模型", **TRACK_LABELS}


class Job:
    def __init__(self, running_title: str, done_title: str, failed_title: str) -> None:
        self.id = uuid.uuid4().hex
        self.running_title = running_title
        self.done_title = done_title
        self.failed_title = failed_title
        self.lines: list[str] = []
        self.done = False
        self.ok = False
        self.error = ""
        self.result_url = ""
        self.result_label = ""
        self._lock = threading.Lock()

    def add(self, line: str) -> None:
        with self._lock:
            self.lines.append(line)

    def finish(self, *, ok: bool, result_url: str = "", result_label: str = "", error: str = "") -> None:
        with self._lock:
            self.done = True
            self.ok = ok
            self.result_url = result_url
            self.result_label = result_label
            self.error = error

    def view(self) -> dict:
        with self._lock:
            if not self.done:
                title = self.running_title
            elif self.ok:
                title = self.done_title
            else:
                title = self.failed_title
            return {
                "id": self.id,
                "title": title,
                "lines": list(self.lines),
                "done": self.done,
                "ok": self.ok,
                "error": self.error,
                "result_url": self.result_url,
                "result_label": self.result_label,
            }


def create_app(data_dir: Path, *, demo: bool = False) -> Flask:
    root = Path(__file__).resolve().parent
    app = Flask(
        __name__,
        template_folder=str(root / "templates"),
        static_folder=str(root / "static"),
    )
    store = MatchStore(data_dir)
    if demo:
        _preload_demo(store)
    app.extensions["store"] = store
    app.extensions["demo"] = demo
    app.extensions["jobs"] = {}
    app.extensions["job_lock"] = threading.Lock()
    _register(app)
    return app


def serve(data_dir: Path, *, host: str = "127.0.0.1", port: int = 8765, demo: bool = False) -> None:
    app = create_app(data_dir, demo=demo)
    server = make_server(host, port, app, threaded=True)
    server.serve_forever()


def format_when(value: str) -> str:
    """Clock time in Asia/Singapore, with UTC, and no microseconds."""

    text = (value or "").strip()
    if not text:
        return ""
    try:
        utc, singapore = split_kickoff(text)
    except (TypeError, ValueError, OverflowError):
        return text
    return f"{singapore} 新加坡（{utc} UTC）"


def _preload_demo(store: MatchStore) -> None:
    frame = store.load()
    if not frame.empty:
        return
    from laliga.data.odds import parse_odds
    from laliga.data.parse import fixtures_to_frame
    from laliga.synthetic import generate_synthetic_fixtures, generate_synthetic_odds

    raw = generate_synthetic_fixtures()
    frame = fixtures_to_frame(raw)
    store.replace(frame)
    store.upsert_odds(parse_odds(generate_synthetic_odds(raw)), {int(value) for value in frame["fixture_id"]})


def _register(app: Flask) -> None:
    def store() -> MatchStore:
        return app.extensions["store"]

    def demo() -> bool:
        return bool(app.extensions["demo"])

    @app.context_processor
    def inject() -> dict:
        return {
            "demo": demo(),
            "token_set": bool(api_token()),
            "default_seasons": DEFAULT_SEASONS,
            "track_labels": TRACK_LABELS,
            "market_labels": PROMOTION_MARKET_LABELS,
            "source_labels": SOURCE_LABELS,
        }

    @app.get("/")
    def home():
        return render_template("home.html", status=_status(store()))

    @app.get("/predictions")
    def predictions():
        view = request.args.get("track", "active")
        if view not in VIEW_NAMES:
            view = "active"
        payload = _read_json(store().predictions_dir / "latest.json")
        items = []
        created_label = ""
        if payload:
            created_label = format_when(str(payload.get("created_at") or ""))
            for item in payload.get("items") or []:
                utc, singapore = split_kickoff(item["starting_at"])
                items.append({**item, "utc": utc, "singapore": singapore, "shown": _shown(item, view)})
        return render_template(
            "predictions.html",
            payload=payload,
            items=items,
            created_label=created_label,
            view=view,
            views=[(name, VIEW_LABELS[name]) for name in VIEW_NAMES],
            active=_active_summary(payload),
        )

    @app.get("/fixtures/<int:fixture_id>")
    def fixture_detail(fixture_id: int):
        payload = _read_json(store().predictions_dir / "latest.json") or {}
        item = next((row for row in payload.get("items") or [] if int(row["fixture_id"]) == fixture_id), None)
        match = _match_row(store(), fixture_id)
        if item is None and match is None:
            return render_template("not_found.html", fixture_id=fixture_id), 404
        utc = singapore = ""
        grids = None
        if item is not None:
            utc, singapore = split_kickoff(item["starting_at"])
            grids = {
                "ft": _grid(item.get("ft_matrix") or [], item.get("top_scores") or []),
                "ht": _grid(item.get("ht_matrix") or [], []),
            }
        actual = None
        if match is not None and match["status"] == "finished" and pd.notna(match["home_goals_ft"]):
            actual = f"{int(match['home_goals_ft'])}-{int(match['away_goals_ft'])}"
        return render_template(
            "fixture.html",
            tracks=_track_rows(item) if item is not None else [],
            item=item,
            match=match,
            utc=utc,
            singapore=singapore,
            grids=grids,
            actual=actual,
            fixture_id=fixture_id,
        )

    @app.get("/backtest")
    def backtest_page():
        payload = _read_json(store().predictions_dir / "backtest.json")
        return render_template(
            "backtest.html",
            report=payload,
            rows=backtest_metric_rows(payload),
            promotion=(payload or {}).get("promotion"),
            ensemble=(payload or {}).get("ensemble"),
        )

    @app.get("/data")
    def data_page():
        return render_template("data.html", status=_status(store()))

    @app.get("/predictions.csv")
    def download_predictions_csv():
        return _send_or_missing(store().predictions_dir / "latest.csv", "predictions.csv", "预测 CSV")

    @app.get("/predictions.json")
    def download_predictions_json():
        return _send_or_missing(store().predictions_dir / "latest.json", "predictions.json", "预测 JSON")

    @app.get("/backtest.csv")
    def download_backtest_csv():
        return _send_or_missing(store().predictions_dir / "backtest.csv", "backtest.csv", "回测 CSV")

    @app.get("/backtest.json")
    def download_backtest_json():
        return _send_or_missing(store().predictions_dir / "backtest.json", "backtest.json", "回测 JSON")

    @app.post("/actions/fetch")
    def start_fetch():
        seasons = request.form.get("seasons", str(DEFAULT_SEASONS))
        refresh = request.form.get("refresh") == "1"
        try:
            count = int(seasons)
        except ValueError:
            count = 0

        def work() -> str:
            if demo():
                raise FitError(DEMO_FETCH_MESSAGE)
            if count < 1 or count > 20:
                raise FitError("赛季数要在 1 到 20 之间。")
            client = open_client(store().cache_dir, refresh=refresh)
            frame = fetch_historical(client, store(), seasons=count, replace=False)
            finished = int((frame["status"] == "finished").sum())
            scheduled = int((frame["status"] == "scheduled").sum())
            logging.getLogger("laliga.webapp").info("已写入 %s 场（完场 %s，未开赛 %s）。", len(frame), finished, scheduled)
            return "/data"

        return _launch(app, "正在更新数据", work, "查看数据状态", done_title="数据更新完成", failed_title="数据更新失败")

    @app.post("/actions/train")
    def start_train():
        def work() -> str:
            history = _require_finished(store())
            config = _web_config(min_train=4)
            as_of = pd.Timestamp(history.loc[history["status"] == "finished", "starting_at"].max()) + pd.Timedelta(seconds=1)
            model = fit_models(history, as_of, config)
            store().save_model_document(model.to_dict())
            logging.getLogger("laliga.webapp").info(
                "训练完成，样本 %s 场，截止 %s。",
                model.ft.n_matches,
                format_when(str(model.ft.trained_through)),
            )
            return "/data"

        return _launch(app, "正在训练", work, "查看数据状态", done_title="训练完成", failed_title="训练失败")

    @app.post("/actions/odds")
    def start_odds():
        scheduled_only = request.form.get("scheduled_only") == "1"

        def work() -> str:
            if demo():
                raise FitError(DEMO_FETCH_MESSAGE)
            history = store().load()
            statuses = ["scheduled"] if scheduled_only else ["scheduled", "finished"]
            fixtures = history[history["status"].isin(statuses)] if not history.empty else history
            if fixtures.empty:
                raise FitError("本地没有比赛可以抓赔率。请先点「更新数据」。")
            client = open_client(store().cache_dir, refresh=False)
            logging.getLogger("laliga.webapp").info("将请求 %s 场比赛的赔率（每场一次请求，完场比赛优先读缓存）。", len(fixtures))
            summary = fetch_odds(client, store(), fixtures)
            logging.getLogger("laliga.webapp").info(summary.message())
            if summary.with_odds == 0:
                raise FitError("这些比赛都没有返回赔率。订阅可能不含西甲赔率（Odds add-on），或历史比赛没有赔率覆盖。")
            return "/data"

        return _launch(app, "正在更新赔率", work, "查看数据状态", done_title="赔率更新完成", failed_title="赔率更新失败")

    @app.post("/actions/backtest")
    def start_backtest():
        try:
            min_train = int(request.form.get("min_train", "320"))
        except ValueError:
            min_train = 0

        def work() -> str:
            if min_train < 4:
                raise FitError("最少训练场次至少为 4。")
            history = _require_finished(store())
            report = walk_forward(
                history,
                _web_config(min_train=min_train),
                store().load_odds(),
                promotion=DEMO_PROMOTION if demo() else PromotionConfig(),
                incumbent=promotion.load_active(store()),
            )
            pipeline.save_backtest(store(), report, promote=True)
            logging.getLogger("laliga.webapp").info(
                "回测完成，评测 %s 场，其中三套模型都有预测的 %s 场。", report.model.n, report.n_with_odds
            )
            return "/backtest"

        return _launch(app, "正在回测", work, "查看回测", done_title="回测完成", failed_title="回测失败")

    @app.post("/actions/predict-next")
    def start_predict_next():
        def work() -> str:
            fixtures, note = _next_fixtures(store(), demo_mode=demo())
            pipeline.refresh_odds(store(), fixtures, demo=demo())
            _write_predictions(store(), fixtures, "下一轮", note)
            return "/predictions"

        return _launch(app, "正在预测下一轮", work, "查看预测", done_title="下一轮预测完成", failed_title="下一轮预测失败")

    @app.post("/actions/predict-range")
    def start_predict_range():
        start_text = request.form.get("start", "")
        end_text = request.form.get("end", "")

        def work() -> str:
            start = _parse_date(start_text)
            end = _parse_date(end_text)
            if end < start:
                raise FitError("结束日期早于开始日期。")
            fixtures = _range_fixtures(store(), start, end, demo_mode=demo())
            pipeline.refresh_odds(store(), fixtures, demo=demo())
            _write_predictions(store(), fixtures, f"{start.isoformat()} 至 {end.isoformat()}", "")
            return "/predictions"

        return _launch(app, "正在按日期预测", work, "查看预测", done_title="按日期预测完成", failed_title="按日期预测失败")

    @app.get("/jobs/<job_id>")
    def job_page(job_id: str):
        job = app.extensions["jobs"].get(job_id)
        if job is None:
            return render_template("not_found.html", fixture_id=None), 404
        return render_template("job.html", job=job.view())

    @app.get("/jobs/<job_id>/status")
    def job_status(job_id: str):
        job = app.extensions["jobs"].get(job_id)
        if job is None:
            abort(404)
        return job.view()


def _launch(app: Flask, running_title: str, work, result_label: str, *, done_title: str, failed_title: str):
    lock: threading.Lock = app.extensions["job_lock"]
    with lock:
        running = next((job for job in app.extensions["jobs"].values() if not job.done), None)
        if running is not None:
            return redirect(url_for("job_page", job_id=running.id), code=303)
        job = Job(running_title, done_title, failed_title)
        app.extensions["jobs"][job.id] = job

    def runner() -> None:
        handler = _JobHandler(job)
        logger = logging.getLogger("laliga")
        logger.addHandler(handler)
        previous = logger.level
        logger.setLevel(logging.INFO)
        try:
            url = work()
            job.finish(ok=True, result_url=url, result_label=result_label)
        except (SportMonksError, FitError, ValueError) as exc:
            job.add(str(exc))
            job.finish(ok=False, error=str(exc))
        except Exception as exc:  # surface unexpected failures instead of a blank page
            logging.getLogger("laliga.webapp").exception("操作台任务失败")
            job.finish(ok=False, error=f"未预期的错误：{exc}")
        finally:
            logger.removeHandler(handler)
            logger.setLevel(previous)

    job.add(running_title)
    threading.Thread(target=runner, daemon=True).start()
    return redirect(url_for("job_page", job_id=job.id), code=303)


class _JobHandler(logging.Handler):
    def __init__(self, job: Job) -> None:
        super().__init__()
        self.job = job
        self.setFormatter(logging.Formatter("%(message)s"))

    def emit(self, record: logging.LogRecord) -> None:
        self.job.add(self.format(record))


def _web_config(*, min_train: int) -> ModelConfig:
    return ModelConfig(max_iter=WEB_MAX_ITER, min_train_matches=min_train, min_ht_matches=20)


def _require_finished(store: MatchStore):
    history = store.load()
    finished = history[history["status"] == "finished"] if not history.empty else history
    if finished.empty:
        raise FitError("本地没有完场比赛。请先点「更新数据」。免费计划不含西甲，需要能读到联赛 564 的订阅。")
    return history


def _next_fixtures(store: MatchStore, *, demo_mode: bool):
    history = store.load()
    fixtures, note = select_next_matchday(history)
    if fixtures.empty and api_token() and not demo_mode:
        today = datetime.now(timezone.utc).date()
        client = open_client(store.cache_dir, refresh=False)
        fetch_window(client, store, today, today + timedelta(days=45))
        history = store.load()
        fixtures, note = select_next_matchday(history)
    if fixtures.empty:
        raise FitError("本地没有未开赛比赛。请先点「更新数据」，或确认订阅里包含未来赛程。")
    return fixtures, note


def _range_fixtures(store: MatchStore, start: date, end: date, *, demo_mode: bool):
    start_ts = pd.Timestamp(datetime.combine(start, datetime.min.time()), tz="UTC")
    end_ts = pd.Timestamp(datetime.combine(end + timedelta(days=1), datetime.min.time()), tz="UTC")
    history = store.load()
    fixtures = scheduled_between(history, start_ts, end_ts)
    if fixtures.empty and api_token() and not demo_mode:
        client = open_client(store.cache_dir, refresh=False)
        fetch_window(client, store, start, end)
        history = store.load()
        fixtures = scheduled_between(history, start_ts, end_ts)
    if fixtures.empty:
        raise FitError("该日期范围内没有未开赛比赛。")
    return fixtures


def _write_predictions(store: MatchStore, fixtures, label: str, note: str) -> None:
    history = store.load()
    forecasts = pipeline.forecast_fixtures(history, fixtures, _web_config(min_train=4), store, save_model=True)
    pipeline.write_latest(store, forecasts, label, note)


def _shown(item: dict, view: str) -> dict:
    """Probabilities to display for one prediction under the chosen view."""

    tracks = item.get("tracks") or {}
    if view == "active" and item.get("active_view"):
        body = item["active_view"]
        marks = {market: bool((item.get("active") or {}).get(market, {}).get("fallback")) for market in PROMOTION_MARKETS}
        return {"available": True, "ft": body["ft"], "ht": body["ht"], "top_scores": body["top_scores"], "fallback": marks}
    if view in tracks:
        body = tracks[view]
        return {**body, "fallback": {}}
    return {"available": True, "ft": item["ft"], "ht": item["ht"], "top_scores": item["top_scores"], "fallback": {}}


def _active_summary(payload: dict | None) -> list[dict]:
    if not payload:
        return []
    active = payload.get("active") or {}
    return [
        {"market": PROMOTION_MARKET_LABELS[market], "track": TRACK_LABELS.get(active.get(market, "football"), "纯足球")}
        for market in PROMOTION_MARKETS
    ]


def _track_rows(item: dict) -> list[dict]:
    rows = []
    tracks = item.get("tracks") or {}
    active = item.get("active") or {}
    for track in TRACKS:
        body = tracks.get(track)
        if body is None:
            continue
        sources = body.get("sources") or {}
        rows.append(
            {
                "track": track,
                "label": TRACK_LABELS[track],
                "available": body.get("available"),
                "ft": body.get("ft"),
                "ht": body.get("ht"),
                "top_scores": body.get("top_scores") or [],
                "notes": body.get("notes") or [],
                "sources": {market: SOURCE_LABELS.get(value, _source_label(value)) for market, value in sources.items()},
                "active_for": [PROMOTION_MARKET_LABELS[market] for market in PROMOTION_MARKETS if (active.get(market) or {}).get("track") == track],
                "weights": body.get("weights"),
                "bookmakers": body.get("bookmakers"),
                "overround_ft": body.get("overround_ft"),
            }
        )
    return rows


def _source_label(value: str) -> str:
    if value.startswith("dixon_coles"):
        return "Dixon–Coles" + ("（按全场比例缩放）" if value.endswith("scaled_full_time") else "")
    if value == "log_pool":
        return "对数意见池"
    return value


def _status(store: MatchStore) -> dict:
    frame = store.load()
    seasons_doc = _read_json(store.seasons_path)
    model = store.load_model_document()
    grouped = []
    if not frame.empty:
        for (season_id, season_name), part in frame.groupby(["season_id", "season_name"], dropna=False):
            grouped.append(
                {
                    "season_id": "" if pd.isna(season_id) else int(season_id),
                    "season_name": season_name or "未命名赛季",
                    "matches": int(len(part)),
                    "finished": int((part["status"] == "finished").sum()),
                    "scheduled": int((part["status"] == "scheduled").sum()),
                }
            )
    cache_files = list(store.cache_dir.glob("*.json")) if store.cache_dir.exists() else []
    trained = None
    if model and model.get("ft"):
        raw = model["ft"].get("trained_through") or ""
        utc, singapore = ("", "")
        if raw:
            utc, singapore = split_kickoff(raw)
        trained = {
            "utc": utc,
            "singapore": singapore,
            "n_matches": model["ft"].get("n_matches"),
            "home_adv": model["ft"].get("home_adv"),
            "rho": model["ft"].get("rho"),
            "ht_source": model.get("ht_source"),
        }
    return {
        "matches": int(len(frame)),
        "finished": int((frame["status"] == "finished").sum()) if not frame.empty else 0,
        "scheduled": int((frame["status"] == "scheduled").sum()) if not frame.empty else 0,
        "seasons_api": (seasons_doc or {}).get("seasons") or [],
        "league_name": (seasons_doc or {}).get("league_name") or "",
        "local_seasons": grouped,
        "cache_files": len(cache_files),
        "model": trained,
        "has_predictions": (store.predictions_dir / "latest.json").exists(),
        "has_backtest": (store.predictions_dir / "backtest.json").exists(),
        "odds": _odds_status(store, frame),
        "active": _active_status(store),
        "ensemble": store.load_json(store.ensemble_path),
    }


def _odds_status(store: MatchStore, frame) -> dict:
    odds = store.load_odds()
    if odds.empty or frame.empty:
        return {"rows": int(len(odds)), "fixtures": 0, "finished": 0, "scheduled": 0, "markets": []}
    fixture_ids = set(int(value) for value in odds["fixture_id"].unique())
    status = frame.set_index("fixture_id")["status"]
    finished = sum(1 for fixture_id in fixture_ids if status.get(fixture_id) == "finished")
    scheduled = sum(1 for fixture_id in fixture_ids if status.get(fixture_id) == "scheduled")
    names = {1: "全场胜平负（1）", 31: "半场胜平负（31）", 29: "半全场（29）", 57: "正确比分（57）"}
    markets = [
        {"name": names.get(int(market), str(market)), "fixtures": int(part["fixture_id"].nunique())}
        for market, part in odds.groupby("market_id")
    ]
    return {"rows": int(len(odds)), "fixtures": len(fixture_ids), "finished": finished, "scheduled": scheduled, "markets": markets}


def _active_status(store: MatchStore) -> dict:
    document = store.load_json(store.active_path) or {}
    active = promotion.load_active(store)
    decision = document.get("decision") or {}
    return {
        "saved": bool(document),
        "rows": [
            {
                "market": PROMOTION_MARKET_LABELS[market],
                "track": TRACK_LABELS[active[market]],
                "reason": ((decision.get("markets") or {}).get(market) or {}).get("reason", ""),
            }
            for market in PROMOTION_MARKETS
        ],
        "decided_at": format_when(str(decision.get("decided_at") or "")),
        "history": [
            {
                "decided_at": format_when(str(entry.get("decided_at") or "")),
                "n_common": entry.get("n_common", 0),
                "tracks": [TRACK_LABELS.get((entry.get("tracks") or {}).get(market), "—") for market in PROMOTION_MARKETS],
                "changed": "、".join(PROMOTION_MARKET_LABELS[market] for market in entry.get("changed") or []) or "无",
            }
            for entry in reversed(document.get("history") or [])
        ][:10],
    }


def _match_row(store: MatchStore, fixture_id: int):
    frame = store.load()
    if frame.empty:
        return None
    found = frame[frame["fixture_id"] == fixture_id]
    if found.empty:
        return None
    return found.iloc[0]


def _grid(matrix: list, top_scores: list) -> dict | None:
    if not matrix:
        return None
    highlights = {f"{score['home_goals']}-{score['away_goals']}" for score in top_scores}
    rows = []
    for home_goals, line in enumerate(matrix):
        cells = []
        for away_goals, probability in enumerate(line):
            cells.append(
                {
                    "away": away_goals,
                    "probability": float(probability),
                    "top": f"{home_goals}-{away_goals}" in highlights,
                }
            )
        rows.append({"home": home_goals, "cells": cells})
    return {"away_headers": list(range(len(matrix[0]))), "rows": rows}


def _read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _send_or_missing(path: Path, download_name: str, label: str):
    if not path.exists():
        return render_template("missing_file.html", label=label), 404
    return send_file(path, as_attachment=True, download_name=download_name)


def _parse_date(value: str) -> date:
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise FitError("日期格式应为 YYYY-MM-DD。") from exc
