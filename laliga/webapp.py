"""Local dashboard. Bind to localhost and call the same pipeline as the CLI.

Synthetic matches are loaded only when the process is started with demo=True.
That mode prints a banner on every page. The default mode shows an empty state
until a real SportMonks fetch writes the local cache.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import threading
import uuid
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from flask import Flask, abort, redirect, render_template, request, send_file, url_for
from werkzeug.serving import make_server

from laliga.backtest import walk_forward
from laliga.config import DEFAULT_SEASONS, ModelConfig, api_token
from laliga.data.client import SportMonksError, open_client
from laliga.data.fetch import fetch_historical, fetch_window
from laliga.data.store import MatchStore
from laliga.model.dixon_coles import FitError
from laliga.model.service import ScorelineModel, fit_models, model_is_fresh
from laliga.schedule import scheduled_between, select_next_matchday

WEB_MAX_ITER = 80
DEMO_FETCH_MESSAGE = (
    "示例模式不会请求 SportMonks，避免把真实响应写进示例目录。"
    "请停掉这个进程，执行 python -m laliga web（不要加 --demo），再点「更新数据」。"
)


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


def split_kickoff(value: str) -> tuple[str, str]:
    """Return (UTC, Asia/Singapore) clock times as YYYY-MM-DD HH:MM."""

    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")
    else:
        stamp = stamp.tz_convert("UTC")
    singapore = stamp.tz_convert("Asia/Singapore")
    return stamp.strftime("%Y-%m-%d %H:%M"), singapore.strftime("%Y-%m-%d %H:%M")


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
    from laliga.data.parse import fixtures_to_frame
    from laliga.synthetic import generate_synthetic_fixtures

    store.replace(fixtures_to_frame(generate_synthetic_fixtures()))


def _register(app: Flask) -> None:
    def store() -> MatchStore:
        return app.extensions["store"]

    def demo() -> bool:
        return bool(app.extensions["demo"])

    @app.context_processor
    def inject() -> dict:
        return {"demo": demo(), "token_set": bool(api_token()), "default_seasons": DEFAULT_SEASONS}

    @app.get("/")
    def home():
        return render_template("home.html", status=_status(store()))

    @app.get("/predictions")
    def predictions():
        payload = _read_json(store().predictions_dir / "latest.json")
        items = []
        created_label = ""
        if payload:
            created_label = format_when(str(payload.get("created_at") or ""))
            for item in payload.get("items") or []:
                utc, singapore = split_kickoff(item["starting_at"])
                items.append({**item, "utc": utc, "singapore": singapore})
        return render_template("predictions.html", payload=payload, items=items, created_label=created_label)

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
        return render_template("backtest.html", report=payload, rows=_metric_rows(payload) if payload else [])

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
            report = walk_forward(history, _web_config(min_train=min_train))
            _write_backtest(store(), report)
            logging.getLogger("laliga.webapp").info("回测完成，评测 %s 场。", report.model.n)
            return "/backtest"

        return _launch(app, "正在回测", work, "查看回测", done_title="回测完成", failed_title="回测失败")

    @app.post("/actions/predict-next")
    def start_predict_next():
        def work() -> str:
            fixtures, note = _next_fixtures(store(), demo_mode=demo())
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
    config = _web_config(min_train=4)
    items = _score_fixtures(history, fixtures, config, store)
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "label": label,
        "note": note,
        "items": items,
    }
    store.predictions_dir.mkdir(parents=True, exist_ok=True)
    (store.predictions_dir / "latest.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (store.predictions_dir / "latest.csv").write_text(_predictions_csv(items), encoding="utf-8")
    logging.getLogger("laliga.webapp").info("已预测 %s 场（%s）。", len(items), label)


def _score_fixtures(history, fixtures, config: ModelConfig, store: MatchStore) -> list[dict]:
    ordered = fixtures.sort_values(["starting_at", "fixture_id"])
    eligible = history[~history["fixture_id"].isin({int(value) for value in ordered["fixture_id"]})]
    saved = _load_saved_model(store)
    items: list[dict] = []
    if saved is not None and model_is_fresh(saved, eligible, ordered):
        for _, row in ordered.iterrows():
            items.append(_pack(saved, row))
        return items
    previous: ScorelineModel | None = None
    model: ScorelineModel | None = None
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
            items.append(_pack(model, row))
    if model is not None:
        store.save_model_document(model.to_dict())
    return items


def _load_saved_model(store: MatchStore) -> ScorelineModel | None:
    document = store.load_model_document()
    if not document:
        return None
    try:
        return ScorelineModel.from_dict(document)
    except (KeyError, TypeError, ValueError, FitError):
        return None


def _pack(model: ScorelineModel, row) -> dict:
    prediction = model.predict_row(row)
    home_id = int(row["home_team_id"])
    away_id = int(row["away_team_id"])
    document = prediction.to_dict()
    document["ft_matrix"] = model.ft.score_matrix(home_id, away_id).round(6).tolist()
    document["ht_matrix"] = model.ht.score_matrix(home_id, away_id).round(6).tolist()
    return document


def _predictions_csv(items: list[dict]) -> str:
    buffer = io.StringIO()
    fieldnames = [
        "fixture_id",
        "kickoff_utc",
        "kickoff_singapore",
        "home_team",
        "away_team",
        "ft_home",
        "ft_draw",
        "ft_away",
        "ht_home",
        "ht_draw",
        "ht_away",
        "score_1",
        "score_1_prob",
        "score_2",
        "score_2_prob",
        "score_3",
        "score_3_prob",
    ]
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    for item in items:
        utc, singapore = split_kickoff(item["starting_at"])
        row = {
            "fixture_id": item["fixture_id"],
            "kickoff_utc": utc,
            "kickoff_singapore": singapore,
            "home_team": item["home_team"],
            "away_team": item["away_team"],
            "ft_home": item["ft"]["home"],
            "ft_draw": item["ft"]["draw"],
            "ft_away": item["ft"]["away"],
            "ht_home": item["ht"]["home"],
            "ht_draw": item["ht"]["draw"],
            "ht_away": item["ht"]["away"],
        }
        for index, score in enumerate(item["top_scores"], start=1):
            row[f"score_{index}"] = f"{score['home_goals']}-{score['away_goals']}"
            row[f"score_{index}_prob"] = score["probability"]
        writer.writerow(row)
    return buffer.getvalue()


def _write_backtest(store: MatchStore, report) -> None:
    payload = report.to_dict()
    store.predictions_dir.mkdir(parents=True, exist_ok=True)
    (store.predictions_dir / "backtest.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (store.predictions_dir / "backtest.csv").write_text(_backtest_csv(payload), encoding="utf-8")


def _backtest_csv(payload: dict) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["metric", "model", "baseline"])
    for label, model_value, baseline_value, _direction in _metric_rows(payload):
        writer.writerow([label, model_value, baseline_value])
    return buffer.getvalue()


def _metric_rows(payload: dict | None) -> list[tuple[str, str, str, str]]:
    if not payload:
        return []
    model = payload["model"]
    baseline = payload["baseline"]

    def num(value: float | None, digits: int, percent: bool) -> str:
        if value is None:
            return "—"
        if percent:
            return f"{value * 100:.1f}%"
        return f"{value:.{digits}f}"

    specs = [
        ("全场对数损失", "ft_log_loss", False, "越低越好"),
        ("全场 Brier 分数", "ft_brier", False, "越低越好"),
        ("全场最可能结果命中率", "ft_accuracy", True, "越高越好"),
        ("半场对数损失", "ht_log_loss", False, "越低越好"),
        ("半场 Brier 分数", "ht_brier", False, "越低越好"),
        ("半场最可能结果命中率", "ht_accuracy", True, "越高越好"),
        ("实际比分落在前三的比例", "top3_hit_rate", True, "越高越好"),
    ]
    rows = []
    for label, key, percent, direction in specs:
        rows.append((label, num(model.get(key), 4, percent), num(baseline.get(key), 4, percent), direction))
    return rows


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
