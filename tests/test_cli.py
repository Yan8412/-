import io
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from laliga.backtest import BacktestReport, ComparisonReport
from laliga.cli import main
from laliga.data.parse import fixtures_to_frame
from laliga.data.store import MatchStore
from laliga.model.metrics import MetricBlock
from laliga.synthetic import generate_synthetic_fixtures

REPO = Path(__file__).resolve().parents[1]


def _env(tmp_path: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["LALIGA_DATA_DIR"] = str(tmp_path)
    env["PYTHONPATH"] = str(REPO)
    env.pop("SPORTMONKS_API_TOKEN", None)
    return env


def _run(args: list[str], tmp_path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=REPO,
        env=_env(tmp_path),
        text=True,
        encoding="utf-8",
        capture_output=True,
        check=False,
    )


def test_fetch_without_token_fails(tmp_path):
    result = _run([sys.executable, "-m", "laliga", "fetch", "--data-dir", str(tmp_path)], tmp_path)
    assert result.returncode == 2
    assert "SPORTMONKS_API_TOKEN" in result.stderr


def test_demo_and_predict_next_run_offline(tmp_path):
    demo = _run(
        [sys.executable, "-m", "laliga", "demo", "--data-dir", str(tmp_path), "--max-iter", "40", "--min-train", "18", "--seed", "7"],
        tmp_path,
    )
    assert demo.returncode == 0, demo.stderr
    assert "全场对数损失" in demo.stdout
    assert "半场" in demo.stdout
    assert "比分1" in demo.stdout or "1-" in demo.stdout or "-" in demo.stdout

    predict = _run(
        [
            sys.executable,
            "-m",
            "laliga",
            "predict",
            "--data-dir",
            str(tmp_path),
            "--next",
            "--max-iter",
            "40",
            "--csv",
            str(tmp_path / "next.csv"),
            "--json",
            str(tmp_path / "next.json"),
        ],
        tmp_path,
    )
    assert predict.returncode == 0, predict.stderr
    assert "全场胜" in predict.stdout
    assert "不是投注建议" in predict.stdout
    table = pd.read_csv(tmp_path / "next.csv")
    assert len(table) == 3
    assert (table[["ft_home", "ft_draw", "ft_away"]].sum(axis=1) - 1).abs().max() < 1e-5
    text = (tmp_path / "next.json").read_text(encoding="utf-8")
    assert "top_scores" in text


def test_store_roundtrip_preserves_goals(tmp_path):
    frame = fixtures_to_frame(generate_synthetic_fixtures(seed=2, today=pd.Timestamp("2026-09-25").date()))
    store = MatchStore(tmp_path)
    store.replace(frame)
    loaded = store.load()
    assert len(loaded) == len(frame)
    assert int((loaded["status"] == "finished").sum()) == int((frame["status"] == "finished").sum())
    finished = loaded[loaded["status"] == "finished"].iloc[0]
    assert int(finished["home_goals_ft"]) >= int(finished["home_goals_ht"])


def test_repository_ignores_env_file():
    gitignore = (REPO / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert ".env" in {line.strip() for line in gitignore}
    example = (REPO / ".env.example").read_text(encoding="utf-8")
    assert "SPORTMONKS_API_TOKEN" in example
    for line in example.splitlines():
        if line.startswith("SPORTMONKS_API_TOKEN"):
            assert line.split("=", 1)[1].strip() == ""
    source = "\n".join(path.read_text(encoding="utf-8") for path in (REPO / "laliga").rglob("*.py"))
    assert "api_token = \"" not in source
    assert "SPORTMONKS_API_TOKEN = \"" not in source


def _metric() -> MetricBlock:
    return MetricBlock(
        n=1,
        n_ht=1,
        ft_log_loss=0.9,
        ft_brier=0.5,
        ft_accuracy=1.0,
        ht_log_loss=1.0,
        ht_brier=0.6,
        ht_accuracy=0.0,
        top3_hit_rate=1.0,
    )


def _comparison() -> ComparisonReport:
    block = _metric()
    return ComparisonReport(
        n_folds=1,
        first_test_day="2026-01-02",
        last_test_day="2026-01-02",
        min_train_matches=4,
        xg_status="absent",
        odds_status="absent",
        odds_used=False,
        odds_coverage=None,
        xg_features_used=False,
        feature_names=[],
        odds_feature_names=[],
        notes=["含赔率的 XGBoost 没有运行"],
        models={"dixon_coles": block, "baseline": block, "xgboost": block},
        calibration={},
        paired=[
            {
                "label": "XGBoost（无赔率）− Dixon–Coles",
                "mean_logloss_difference": 0.0148,
                "ci_low": -0.1,
                "ci_high": 0.1,
                "n": 1,
            }
        ],
        xgboost={"rounds": 1},
    )


def _seed_finished(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        [
            {
                "fixture_id": 1,
                "league_id": 564,
                "season_id": 1,
                "season_name": "2025/2026",
                "round_id": 1,
                "round_name": "1",
                "starting_at": "2026-01-01 20:00:00+00:00",
                "state_id": 5,
                "state_name": "FT",
                "home_team_id": 10,
                "home_team_name": "Home",
                "away_team_id": 20,
                "away_team_name": "Away",
                "home_goals_ht": 0,
                "away_goals_ht": 0,
                "home_goals_ft": 1,
                "away_goals_ft": 0,
                "status": "finished",
            }
        ]
    )
    MatchStore(tmp_path).replace(frame)


def _gbk_pipe() -> tuple[io.BytesIO, io.TextIOWrapper]:
    buffer = io.BytesIO()
    stream = io.TextIOWrapper(buffer, encoding="gbk", errors="strict")
    return buffer, stream


def _redirect_stdio(monkeypatch: pytest.MonkeyPatch, stdout, stderr) -> None:
    monkeypatch.setattr(sys, "stdout", stdout)
    monkeypatch.setattr(sys, "stderr", stderr)


@pytest.fixture
def restore_logging():
    root = logging.getLogger()
    handlers = root.handlers[:]
    level = root.level
    yield
    root.handlers[:] = handlers
    root.setLevel(level)


@pytest.mark.usefixtures("restore_logging")
def test_compare_on_gbk_stdout_prints_utf8_and_writes_json(tmp_path, monkeypatch):
    _seed_finished(tmp_path)
    output = tmp_path / "model_comparison.json"
    stdout_buffer, stdout = _gbk_pipe()
    stderr = _gbk_pipe()[1]
    _redirect_stdio(monkeypatch, stdout, stderr)
    monkeypatch.setattr("laliga.backtest.compare_models", lambda *_args, **_kwargs: _comparison())

    code = main(["compare", "--data-dir", str(tmp_path), "--min-train", "4", "--output", str(output)])

    assert code == 0
    assert stdout.encoding == "utf-8"
    assert stderr.encoding == "utf-8"
    stdout.flush()
    printed = stdout_buffer.getvalue().decode("utf-8")
    assert "XGBoost（无赔率）− Dixon–Coles" in printed
    assert "\u2212" in printed
    assert "\u2500" in printed
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["computed_from_stored_matches"] is True
    assert payload["paired"][0]["label"] == "XGBoost（无赔率）− Dixon–Coles"


@pytest.mark.usefixtures("restore_logging")
def test_compare_writes_json_when_stdout_still_rejects_the_minus_sign(tmp_path, monkeypatch):
    _seed_finished(tmp_path)
    output = tmp_path / "model_comparison.json"

    class RejectingGbk:
        encoding = "gbk"
        errors = "strict"

        def reconfigure(self, **_kwargs):
            raise OSError("cannot reconfigure")

        def write(self, text: str):
            text.encode("gbk")
            return len(text)

        def flush(self):
            return None

    _redirect_stdio(monkeypatch, RejectingGbk(), RejectingGbk())
    monkeypatch.setattr("laliga.backtest.compare_models", lambda *_args, **_kwargs: _comparison())

    with pytest.raises(UnicodeEncodeError):
        main(["compare", "--data-dir", str(tmp_path), "--min-train", "4", "--output", str(output)])

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["computed_from_stored_matches"] is True
    assert payload["paired"][0]["label"] == "XGBoost（无赔率）− Dixon–Coles"


@pytest.mark.usefixtures("restore_logging")
def test_backtest_writes_json_before_printing(tmp_path, monkeypatch):
    _seed_finished(tmp_path)
    output = tmp_path / "backtest.json"
    block = _metric()
    report = BacktestReport(
        n_folds=1,
        first_test_day="2026-01-02",
        last_test_day="2026-01-02",
        min_train_matches=4,
        model=block,
        baseline=block,
        model_by_season={},
        baseline_by_season={},
    )
    order: list[str] = []
    real_write_text = Path.write_text

    def tracking_write(self: Path, data: str, *args, **kwargs):
        if self == output:
            order.append("write")
        return real_write_text(self, data, *args, **kwargs)

    def tracking_print(*_args, **_kwargs):
        order.append("print")

    monkeypatch.setattr(Path, "write_text", tracking_write)
    monkeypatch.setattr("builtins.print", tracking_print)
    monkeypatch.setattr("laliga.backtest.walk_forward", lambda *_args, **_kwargs: report)
    _redirect_stdio(monkeypatch, io.StringIO(), io.StringIO())

    code = main(["backtest", "--data-dir", str(tmp_path), "--min-train", "4", "--output", str(output)])

    assert code == 0
    assert order.index("write") < order.index("print")
    assert json.loads(output.read_text(encoding="utf-8"))["n_folds"] == 1


@pytest.mark.usefixtures("restore_logging")
def test_help_on_a_gbk_pipe_is_utf8(monkeypatch):
    stdout_buffer, stdout = _gbk_pipe()
    _redirect_stdio(monkeypatch, stdout, _gbk_pipe()[1])

    with pytest.raises(SystemExit) as caught:
        main(["web", "--help"])

    assert caught.value.code == 0
    stdout.flush()
    text = stdout_buffer.getvalue().decode("utf-8")
    assert "监听地址，默认只对本机开放" in text

    compare_buffer, compare_stdout = _gbk_pipe()
    _redirect_stdio(monkeypatch, compare_stdout, _gbk_pipe()[1])
    with pytest.raises(SystemExit) as compare_help:
        main(["-h"])
    assert compare_help.value.code == 0
    compare_stdout.flush()
    compare_text = compare_buffer.getvalue().decode("utf-8")
    assert "Dixon–Coles" in compare_text


NOTIFIER_KEYS = {
    "fixture_id",
    "starting_at",
    "season_id",
    "season_name",
    "round_name",
    "home_team_id",
    "home_team",
    "away_team_id",
    "away_team",
    "ft",
    "ht",
    "top_scores",
    "expected_goals_ft",
    "expected_goals_ht",
}


def _predict_next(tmp_path: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    return _run(
        [sys.executable, "-m", "laliga", "predict", "--data-dir", str(tmp_path), "--next", "--max-iter", "40", *extra],
        tmp_path,
    )


def test_daily_predict_keeps_latest_next_format_publishes_and_archives(tmp_path):
    from laliga.archive import PredictionArchive
    from laliga.report import latest_next_problem

    MatchStore(tmp_path).replace(fixtures_to_frame(generate_synthetic_fixtures(seed=7)))
    latest_next = tmp_path / "predictions" / "latest_next.json"
    result = _predict_next(tmp_path, "--json", str(latest_next), "--publish")
    assert result.returncode == 0, result.stderr
    assert "预测存档：写入 3 场" in result.stdout

    payload = json.loads(latest_next.read_text(encoding="utf-8"))
    assert isinstance(payload, list) and len(payload) == 3
    assert latest_next_problem(payload) == ""
    for item in payload:
        assert set(item) == NOTIFIER_KEYS
        assert item["starting_at"].endswith("+00:00")
        assert set(item["ft"]) == set(item["ht"]) == {"home", "draw", "away"}
        assert [set(score) for score in item["top_scores"]] == [{"home_goals", "away_goals", "probability"}] * 3
        assert set(item["expected_goals_ft"]) == set(item["expected_goals_ht"]) == {"home", "away"}

    dashboard = json.loads((tmp_path / "predictions" / "latest.json").read_text(encoding="utf-8"))
    assert dashboard["label"] == "下一轮"
    assert [item["fixture_id"] for item in dashboard["items"]] == [item["fixture_id"] for item in payload]
    assert dashboard["items"][0]["ft_matrix"]
    assert (tmp_path / "predictions" / "latest.csv").exists()

    archive = PredictionArchive(tmp_path)
    first = archive.live_path.read_bytes()
    again = _predict_next(tmp_path, "--json", str(latest_next))
    assert again.returncode == 0, again.stderr
    assert archive.live_path.read_bytes().startswith(first)
    records = archive.live_records()
    assert len(records) == 6
    assert {record["origin"] for record in records} == {"cli:predict --next"}
    assert all(record["model_version"].startswith("laliga-") for record in records)

    skipped = _predict_next(tmp_path, "--no-archive")
    assert skipped.returncode == 0, skipped.stderr
    assert len(archive.live_records()) == 6

    history = _run([sys.executable, "-m", "laliga", "history", "--data-dir", str(tmp_path), "--json", str(tmp_path / "h.json")], tmp_path)
    assert history.returncode == 0, history.stderr
    assert "还没有开球后的存档预测" in history.stdout
    assert json.loads((tmp_path / "h.json").read_text(encoding="utf-8"))["live"]["rows"] == []

    doctor = _run([sys.executable, "-m", "laliga", "doctor", "--data-dir", str(tmp_path)], tmp_path)
    assert doctor.returncode == 0, doctor.stderr
    assert "latest_next.json：字段完整" in doctor.stdout
    assert "实时 6 行" in doctor.stdout


def test_doctor_flags_a_broken_latest_next(tmp_path):
    MatchStore(tmp_path).replace(fixtures_to_frame(generate_synthetic_fixtures(seed=7)))
    (tmp_path / "predictions").mkdir()
    (tmp_path / "predictions" / "latest_next.json").write_text(json.dumps([{"home_team": "A"}]), encoding="utf-8")
    doctor = _run([sys.executable, "-m", "laliga", "doctor", "--data-dir", str(tmp_path)], tmp_path)
    assert doctor.returncode == 1
    assert "latest_next.json 格式不对" in doctor.stderr


def test_backfill_archive_writes_only_the_reconstruction_file(tmp_path):
    from laliga.archive import PredictionArchive

    MatchStore(tmp_path).replace(fixtures_to_frame(generate_synthetic_fixtures(seed=7)))
    result = _run(
        [sys.executable, "-m", "laliga", "backfill-archive", "--data-dir", str(tmp_path), "--min-train", "18", "--max-iter", "40"],
        tmp_path,
    )
    assert result.returncode == 0, result.stderr
    assert "不是当时的实时预测" in result.stdout
    archive = PredictionArchive(tmp_path)
    assert not archive.live_path.exists()
    records = archive.reconstruction_records()
    assert records and {record["source"] for record in records} == {"backtest_reconstruction"}
    history = _run([sys.executable, "-m", "laliga", "history", "--data-dir", str(tmp_path)], tmp_path)
    assert "回测重建" in history.stdout


def test_predict_with_availability_needs_the_evaluated_coefficients(tmp_path):
    MatchStore(tmp_path).replace(fixtures_to_frame(generate_synthetic_fixtures(seed=7)))
    result = _predict_next(tmp_path, "--availability")
    assert result.returncode == 2
    assert "compare-availability" in result.stderr
