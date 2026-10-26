"""Blend weights ignore the current day, and the champion ignores the holdout."""

from __future__ import annotations

import pandas as pd
import pytest

from laliga.archive import PredictionArchive, select_shown
from laliga.config import ModelConfig
from laliga.data.parse import fixtures_to_frame
from laliga.model.dixon_coles import FitError
from laliga.model.joint import HalfRates
from laliga.model.tracks import _attach_blends
from laliga.synthetic import generate_synthetic_fixtures
from laliga.upgrade import _decide, load_registry, rollback, save_registry, upgrade


def _priced_row(day: str, fixture_id: int, result: int) -> dict:
    return {
        "fixture_id": fixture_id,
        "match_day": day,
        "y_ft": result,
        "football_ft": (0.46, 0.28, 0.26),
        "market_ft": (0.55, 0.25, 0.20),
        "football_rates": HalfRates(0.55, 0.45, 0.65, 0.55, -0.05, -0.05, 0.0),
        "market_rates": HalfRates(0.80, 0.40, 0.90, 0.45, -0.05, -0.05, 0.0),
        "market_ht": (0.4, 0.35, 0.25),
        "football_ht": (0.38, 0.36, 0.26),
    }


def test_rate_weight_ignores_the_day_being_scored():
    rows = []
    for day_index, day in enumerate(("2026-09-01", "2026-09-08", "2026-09-15")):
        for offset in range(3):
            rows.append(_priced_row(day, 100 + day_index * 10 + offset, (day_index + offset) % 3))
    original = [dict(row) for row in rows]
    _attach_blends(rows, min_history=3)
    last = [row for row in rows if row["match_day"] == "2026-09-15"]
    assert last and all("blend_rate_weight" in row for row in last)
    weight = last[0]["blend_rate_weight"]
    assert all(row["match_day"] != "2026-09-15" or "blend_rate_ft" in row for row in rows)
    early = [row for row in rows if row["match_day"] == "2026-09-01"]
    assert all("blend_rate_weight" not in row for row in early)

    flipped = [dict(row) for row in original]
    for row in flipped:
        if row["match_day"] == "2026-09-15":
            row["y_ft"] = (row["y_ft"] + 1) % 3
    _attach_blends(flipped, min_history=3)
    assert flipped[-1]["blend_rate_weight"] == pytest.approx(weight)

    shifted = [dict(row) for row in original]
    for row in shifted:
        if row["match_day"] == "2026-09-08":
            row["y_ft"] = 2
    _attach_blends(shifted, min_history=3)
    # The earlier day is allowed to move the later weight. This assertion only
    # documents that the function reads history; the same-day check above is the leak guard.
    assert "blend_rate_weight" in shifted[-1]


def test_promotion_rules():
    worse = {"football": {"n": 50, "ft_log_loss": 1.00, "ft_rps": 0.22}}
    better = {"football": {"n": 50, "ft_log_loss": 0.90, "ft_rps": 0.21}}
    champion = {"football": {"n": 50, "ft_log_loss": 0.99, "ft_rps": 0.21}}
    promoted, _reason = _decide(better, None, rule="bootstrap", min_promote=40)
    assert promoted
    promoted, reason = _decide(worse, champion, rule="bootstrap", min_promote=40)
    assert promoted is False
    assert "冠军不变" in reason
    promoted, _reason = _decide(better, champion, rule="bootstrap", min_promote=40)
    assert promoted
    promoted, reason = _decide(better, champion, rule="strict", min_promote=40)
    assert promoted
    assert "升为冠军" in reason
    small = {"football": {"n": 10, "ft_log_loss": 0.50, "ft_rps": 0.10}}
    promoted, reason = _decide(small, champion, rule="strict", min_promote=40)
    assert promoted is False
    assert "40" in reason


def test_market_archive_line_does_not_replace_the_football_line(tmp_path):
    archive = PredictionArchive(tmp_path)
    football = {
        "fixture_id": 7,
        "starting_at": "2026-10-18T19:00:00+00:00",
        "season_id": 1,
        "season_name": "2026/2027",
        "round_name": "9",
        "home_team_id": 1,
        "home_team": "Home",
        "away_team_id": 2,
        "away_team": "Away",
        "ft": {"home": 0.51, "draw": 0.27, "away": 0.22},
        "ht": {"home": 0.40, "draw": 0.40, "away": 0.20},
        "top_scores": [{"home_goals": 1, "away_goals": 0, "probability": 0.12}],
        "expected_goals_ft": {"home": 1.4, "away": 0.9},
        "expected_goals_ht": {"home": 0.6, "away": 0.4},
    }
    market = dict(football)
    market["ft"] = {"home": 0.20, "draw": 0.30, "away": 0.50}
    archive.append([football], model_version="v", origin="test", generated_at="2026-10-17T08:00:00Z", extras={7: {"track": "football"}})
    archive.append(
        [market],
        model_version="v+market",
        origin="test:market",
        generated_at="2026-10-17T09:00:00Z",
        extras={7: {"track": "market"}},
    )
    shown = select_shown(archive.live_records())
    assert shown[7]["ft"]["home"] == pytest.approx(0.51)
    assert shown[7]["n_generations"] == 1


def test_upgrade_holdout_is_not_used_to_fit_and_rollback_restores_the_previous(tmp_path, monkeypatch):
    import laliga.upgrade as upgrade_mod

    frame = fixtures_to_frame(generate_synthetic_fixtures(seed=5, today=pd.Timestamp("2026-09-25").date()))
    finished = frame[frame["status"] == "finished"].sort_values(["starting_at", "fixture_id"])
    holdout_matches = 12
    cutoff = pd.Timestamp(finished.iloc[-holdout_matches]["starting_at"])
    if cutoff.tzinfo is None:
        cutoff = cutoff.tz_localize("UTC")
    cutoff = cutoff.floor("D")
    real_fit = upgrade_mod.fit_models
    seen = []

    def wrapped(matches, as_of, config, **kwargs):
        times = pd.to_datetime(matches["starting_at"], utc=True)
        assert times.max() < cutoff
        seen.append(pd.Timestamp(as_of))
        return real_fit(matches, as_of, config, **kwargs)

    monkeypatch.setattr(upgrade_mod, "fit_models", wrapped)
    config = ModelConfig(max_iter=12, min_matches=3, min_train_matches=8, min_ht_matches=8, xi=0.002)
    result = upgrade(
        frame,
        config,
        tmp_path,
        n_trials=1,
        holdout_matches=holdout_matches,
        min_promote=4,
        rule="strict",
        refit_every_days=28,
    )
    assert seen
    assert result.report["tuning_matches"] + result.report["holdout_matches"] == len(finished)
    assert result.promoted
    registry = load_registry(tmp_path)
    assert registry["champion"]["id"] == result.report["champion_id"]
    assert "optuna" in " ".join(result.report["libraries"]) or "三组固定参数" in " ".join(result.report["libraries"])
    text = (tmp_path / "predictions" / "accuracy_history.md").read_text(encoding="utf-8")
    assert "准确率记录" in text
    assert "样本外" in text
    with pytest.raises(FitError):
        rollback(tmp_path)
    registry["previous"] = {"id": "v0", "params": {"xi": 0.003}}
    save_registry(tmp_path, registry)
    message = rollback(tmp_path)
    assert "v0" in message
    assert load_registry(tmp_path)["champion"]["id"] == "v0"
    assert load_registry(tmp_path)["previous"]["id"] == result.report["champion_id"]
