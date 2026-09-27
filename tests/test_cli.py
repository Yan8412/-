import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

from laliga.data.parse import fixtures_to_frame
from laliga.data.store import MatchStore
from laliga.synthetic import generate_synthetic_fixtures

REPO = Path(__file__).resolve().parents[1]


def _env(tmp_path: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["LALIGA_DATA_DIR"] = str(tmp_path)
    env["PYTHONPATH"] = str(REPO)
    env.pop("SPORTMONKS_API_TOKEN", None)
    return env


def test_fetch_without_token_fails(tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "laliga", "fetch", "--data-dir", str(tmp_path)],
        cwd=REPO,
        env=_env(tmp_path),
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 2
    assert "SPORTMONKS_API_TOKEN" in result.stderr


def test_demo_and_predict_next_run_offline(tmp_path):
    demo = subprocess.run(
        [sys.executable, "-m", "laliga", "demo", "--data-dir", str(tmp_path), "--max-iter", "40", "--min-train", "18", "--seed", "7"],
        cwd=REPO,
        env=_env(tmp_path),
        text=True,
        capture_output=True,
        check=False,
    )
    assert demo.returncode == 0, demo.stderr
    assert "全场对数损失" in demo.stdout
    assert "半场" in demo.stdout
    assert "比分1" in demo.stdout or "1-" in demo.stdout or "-" in demo.stdout

    predict = subprocess.run(
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
        cwd=REPO,
        env=_env(tmp_path),
        text=True,
        capture_output=True,
        check=False,
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
