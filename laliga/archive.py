"""Append-only archive of pre-kickoff predictions, and the history built from it.

Every live generation appends one JSON line per fixture to
``archive/predictions.jsonl``. Lines are never edited or removed. A line
generated at or after kickoff is refused, so nothing enters the archive for
a match that has started. For each fixture the history uses the latest line
generated before kickoff: the last numbers a user could have seen.

Results are not written into the archive. They are joined at read time from
``processed/matches.csv``, so a corrected score updates the hit marks
without touching what was predicted.

``archive/backtest_reconstruction.jsonl`` is a separate, regenerable file.
It holds strict walk-forward predictions for matches that were already
played before the live archive existed. Those rows are labelled
``backtest_reconstruction`` and never count toward the live hit rates.
"""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from laliga import __version__
from laliga.config import ModelConfig
from laliga.model.baseline import FrequencyBaseline, fit_baseline
from laliga.model.dixon_coles import MODEL_VERSION, FitError
from laliga.model.metrics import iter_calendar_days, outcome_index, per_match_log_loss, per_match_rps
from laliga.model.service import Prediction, ScorelineModel, fit_models

logger = logging.getLogger(__name__)

ARCHIVE_SCHEMA = 1
LIVE = "live"
RECONSTRUCTION = "backtest_reconstruction"
OUTCOMES = ("home", "draw", "away")


def utc_now() -> pd.Timestamp:
    return pd.Timestamp(datetime.now(timezone.utc))


def model_version(*, availability: bool = False) -> str:
    text = f"laliga-{__version__}/dixon_coles-v{MODEL_VERSION}"
    if availability:
        text += "+availability-v1"
    return text


def _utc(value: Any) -> pd.Timestamp | None:
    if value is None or value == "":
        return None
    stamp = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(stamp):
        return None
    return pd.Timestamp(stamp)


def _iso(value: Any) -> str:
    stamp = _utc(value)
    if stamp is None:
        raise ValueError(f"无法解析时间：{value!r}")
    return stamp.isoformat()


def archive_record(
    prediction: Prediction | dict,
    *,
    generated_at: Any,
    model_version: str,
    source: str = LIVE,
    origin: str = "",
    information_cutoff: Any = None,
    extra: dict | None = None,
) -> dict:
    """One archive line. ``information_cutoff`` is the latest data the prediction could use."""

    body = prediction.to_dict() if isinstance(prediction, Prediction) else dict(prediction)
    generated = _iso(generated_at)
    record = {
        "schema": ARCHIVE_SCHEMA,
        "source": source,
        "fixture_id": int(body["fixture_id"]),
        "kickoff_utc": _iso(body["starting_at"]),
        "season_id": body.get("season_id"),
        "season_name": body.get("season_name") or "",
        "round_name": body.get("round_name") or "",
        "home_team_id": int(body["home_team_id"]),
        "home_team": body["home_team"],
        "away_team_id": int(body["away_team_id"]),
        "away_team": body["away_team"],
        "ft": {key: float(body["ft"][key]) for key in OUTCOMES},
        "ht": {key: float(body["ht"][key]) for key in OUTCOMES},
        "top_scores": [
            {
                "home_goals": int(score["home_goals"]),
                "away_goals": int(score["away_goals"]),
                "probability": float(score["probability"]),
            }
            for score in body["top_scores"][:3]
        ],
        "expected_goals_ft": {
            "home": float(body["expected_goals_ft"]["home"]),
            "away": float(body["expected_goals_ft"]["away"]),
        },
        "expected_goals_ht": {
            "home": float(body["expected_goals_ht"]["home"]),
            "away": float(body["expected_goals_ht"]["away"]),
        },
        "model_version": model_version,
        "generated_at": generated,
        "information_cutoff": _iso(information_cutoff) if information_cutoff is not None else generated,
        "origin": origin,
    }
    if extra:
        record["extra"] = extra
    return record


@dataclass
class AppendResult:
    accepted: int = 0
    refused: list[tuple[int, str]] = field(default_factory=list)


class PredictionArchive:
    def __init__(self, root: Path) -> None:
        self.dir = Path(root) / "archive"
        self.live_path = self.dir / "predictions.jsonl"
        self.reconstruction_path = self.dir / "backtest_reconstruction.jsonl"

    def append(
        self,
        predictions: Iterable[Prediction | dict],
        *,
        model_version: str,
        origin: str,
        generated_at: Any = None,
        matches: pd.DataFrame | None = None,
        extras: dict[int, dict] | None = None,
    ) -> AppendResult:
        """Append live predictions. Anything at or after kickoff is refused, never written."""

        generated = _utc(generated_at) if generated_at is not None else utc_now()
        known = _known_fixtures(matches)
        result = AppendResult()
        lines: list[str] = []
        for prediction in predictions:
            record = archive_record(
                prediction,
                generated_at=generated,
                model_version=model_version,
                source=LIVE,
                origin=origin,
                extra=(extras or {}).get(int(_field(prediction, "fixture_id"))),
            )
            reason = _refusal(record, generated, known)
            if reason:
                result.refused.append((record["fixture_id"], reason))
                continue
            lines.append(json.dumps(record, ensure_ascii=False, sort_keys=True))
            result.accepted += 1
        if lines:
            _append_lines(self.live_path, lines)
        for fixture_id, reason in result.refused:
            logger.warning("未写入预测存档：比赛 %s，%s", fixture_id, reason)
        return result

    def live_records(self) -> list[dict]:
        return _read_lines(self.live_path)

    def reconstruction_records(self) -> list[dict]:
        return _read_lines(self.reconstruction_path)

    def replace_reconstruction(self, records: list[dict]) -> None:
        for record in records:
            if record.get("source") != RECONSTRUCTION:
                raise ValueError("回测重建文件只接受 backtest_reconstruction 记录。")
        self.dir.mkdir(parents=True, exist_ok=True)
        temporary = self.reconstruction_path.with_suffix(".jsonl.tmp")
        text = "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records)
        temporary.write_text(text, encoding="utf-8", newline="\n")
        temporary.replace(self.reconstruction_path)


def _field(prediction: Prediction | dict, name: str):
    return getattr(prediction, name) if isinstance(prediction, Prediction) else prediction[name]


def _known_fixtures(matches: pd.DataFrame | None) -> dict[int, tuple[pd.Timestamp | None, str]]:
    if matches is None or matches.empty:
        return {}
    known: dict[int, tuple[pd.Timestamp | None, str]] = {}
    for fixture_id, kickoff, status in zip(matches["fixture_id"], matches["starting_at"], matches["status"], strict=True):
        if pd.isna(fixture_id):
            continue
        known[int(fixture_id)] = (_utc(kickoff), str(status))
    return known


def _refusal(record: dict, generated: pd.Timestamp, known: dict[int, tuple[pd.Timestamp | None, str]]) -> str:
    kickoff = _utc(record["kickoff_utc"])
    if kickoff is None or generated >= kickoff:
        return "生成时间不早于开球时间"
    stored = known.get(record["fixture_id"])
    if stored is not None:
        stored_kickoff, status = stored
        if status != "scheduled":
            return f"比赛表里的状态是 {status}，不是未开赛"
        if stored_kickoff is not None and generated >= stored_kickoff:
            return "生成时间不早于比赛表里的开球时间"
    return ""


def _append_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(line + "\n" for line in lines)
    with open(path, "a", encoding="utf-8", newline="\n") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _read_lines(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    with open(path, encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                record = json.loads(text)
            except json.JSONDecodeError:
                logger.warning("预测存档 %s 第 %s 行不是完整的 JSON，已跳过。", path.name, number)
                continue
            if isinstance(record, dict) and "fixture_id" in record:
                records.append(record)
    return records


def select_shown(records: list[dict], matches: pd.DataFrame | None = None) -> dict[int, dict]:
    """Latest record per fixture generated strictly before kickoff.

    Kickoff is the earlier of the archived kickoff and the current one in the
    match table, so a match brought forward cannot pick up a line that was
    written after it actually started.
    """

    known = _known_fixtures(matches)
    shown: dict[int, dict] = {}
    counts: dict[int, int] = {}
    for record in records:
        fixture_id = int(record["fixture_id"])
        generated = _effective_time(record)
        kickoff = _utc(record.get("kickoff_utc"))
        if generated is None or kickoff is None:
            continue
        actual = known.get(fixture_id, (None, ""))[0]
        if actual is not None:
            kickoff = min(kickoff, actual)
        if generated >= kickoff:
            continue
        if not _is_football_track(record):
            continue
        counts[fixture_id] = counts.get(fixture_id, 0) + 1
        current = shown.get(fixture_id)
        if current is None or generated >= _effective_time(current):
            shown[fixture_id] = record
    return {fixture_id: {**record, "n_generations": counts[fixture_id]} for fixture_id, record in shown.items()}


def _is_football_track(record: dict) -> bool:
    """Live hit rates stay on the football track. Other tracks are settled separately."""

    extra = record.get("extra") if isinstance(record.get("extra"), dict) else {}
    track = extra.get("track") or record.get("track") or "football"
    return track == "football"


def _effective_time(record: dict) -> pd.Timestamp | None:
    """Live lines count from when they were written. A reconstruction counts from its data cutoff."""

    if record.get("source") == RECONSTRUCTION:
        return _utc(record.get("information_cutoff"))
    return _utc(record.get("generated_at"))


class _BaselineCache:
    """Frequency baseline on finished matches strictly before a cutoff, cached by sample size."""

    def __init__(self, matches: pd.DataFrame | None) -> None:
        if matches is None or matches.empty:
            self.finished = pd.DataFrame()
            self.kickoffs = np.array([], dtype="datetime64[ns]")
        else:
            finished = matches[matches["status"] == "finished"].dropna(subset=["home_goals_ft", "away_goals_ft"])
            self.finished = finished.sort_values(["starting_at", "fixture_id"]).reset_index(drop=True)
            self.kickoffs = self.finished["starting_at"].dt.tz_convert("UTC").dt.tz_localize(None).to_numpy()
        self._cache: dict[int, FrequencyBaseline | None] = {}

    def before(self, cutoff: pd.Timestamp) -> FrequencyBaseline | None:
        stamp = np.datetime64(pd.Timestamp(cutoff).tz_convert("UTC").tz_localize(None))
        count = int(np.searchsorted(self.kickoffs, stamp, side="left"))
        if count not in self._cache:
            if count == 0:
                self._cache[count] = None
            else:
                try:
                    self._cache[count] = fit_baseline(self.finished.iloc[:count])
                except FitError:
                    self._cache[count] = None
        return self._cache[count]


def _pick(probs: dict[str, float] | tuple[float, float, float]) -> str:
    values = [probs[key] for key in OUTCOMES] if isinstance(probs, dict) else list(probs)
    return OUTCOMES[int(np.argmax(values))]


def _scores(probs: dict[str, float] | tuple[float, float, float], actual: int) -> tuple[float, float]:
    values = np.array([[probs[key] for key in OUTCOMES] if isinstance(probs, dict) else list(probs)], dtype=float)
    y = np.array([actual], dtype=int)
    return float(per_match_log_loss(y, values)[0]), float(per_match_rps(y, values)[0])


def _int_or_none(value) -> int | None:
    if value is None or pd.isna(value):
        return None
    return int(value)


def build_history(records: list[dict], matches: pd.DataFrame | None, *, now: pd.Timestamp | None = None) -> list[dict]:
    """Shown predictions joined to results, oldest kickoff first, with running hit rates."""

    now = utc_now() if now is None else _utc(now)
    shown = select_shown(records, matches)
    lookup: dict[int, pd.Series] = {}
    if matches is not None and not matches.empty:
        for _, row in matches.drop_duplicates("fixture_id", keep="last").iterrows():
            lookup[int(row["fixture_id"])] = row
    baselines = _BaselineCache(matches)
    rows: list[dict] = []
    for fixture_id, record in shown.items():
        match = lookup.get(fixture_id)
        kickoff = _utc(record["kickoff_utc"])
        if match is not None and pd.notna(match["starting_at"]):
            kickoff = min(kickoff, _utc(match["starting_at"]))
        if kickoff > now:
            continue
        row = {
            **{key: record[key] for key in record if key not in ("schema",)},
            "kickoff_utc": kickoff.isoformat(),
            "predicted_kickoff_utc": record["kickoff_utc"],
            "status": "awaiting",
            "actual_ft": None,
            "actual_ht": None,
        }
        home_ft = away_ft = home_ht = away_ht = None
        if match is not None:
            if match["status"] == "finished":
                home_ft = _int_or_none(match["home_goals_ft"])
                away_ft = _int_or_none(match["away_goals_ft"])
                home_ht = _int_or_none(match["home_goals_ht"])
                away_ht = _int_or_none(match["away_goals_ht"])
            elif match["status"] == "skipped":
                row["status"] = "not_played"
                row["state_name"] = str(match.get("state_name") or "")
        if home_ft is not None and away_ft is not None:
            row.update(_marks(record, home_ft, away_ft, home_ht, away_ht, baselines))
        rows.append(row)
    rows.sort(key=lambda item: (item["kickoff_utc"], item["fixture_id"]))
    _running(rows)
    return rows


def _marks(record: dict, home_ft: int, away_ft: int, home_ht, away_ht, baselines: _BaselineCache) -> dict:
    y_ft = outcome_index(home_ft, away_ft)
    actual_score = (home_ft, away_ft)
    top3 = [(int(item["home_goals"]), int(item["away_goals"])) for item in record["top_scores"]]
    ft_log_loss, ft_rps = _scores(record["ft"], y_ft)
    marks: dict[str, Any] = {
        "status": "finished",
        "actual_ft": f"{home_ft}-{away_ft}",
        "home_goals_ft": home_ft,
        "away_goals_ft": away_ft,
        "ft_actual": OUTCOMES[y_ft],
        "ft_pick": _pick(record["ft"]),
        "top3_hit": actual_score in top3,
        "ft_log_loss": ft_log_loss,
        "ft_rps": ft_rps,
        "always_home_ft_hit": y_ft == 0,
    }
    marks["ft_hit"] = marks["ft_pick"] == marks["ft_actual"]
    y_ht = None
    if home_ht is not None and away_ht is not None:
        y_ht = outcome_index(home_ht, away_ht)
        marks.update(
            {
                "actual_ht": f"{home_ht}-{away_ht}",
                "home_goals_ht": home_ht,
                "away_goals_ht": away_ht,
                "ht_actual": OUTCOMES[y_ht],
                "ht_pick": _pick(record["ht"]),
                "always_home_ht_hit": y_ht == 0,
            }
        )
        marks["ht_hit"] = marks["ht_pick"] == marks["ht_actual"]
    else:
        marks.update({"ht_hit": None, "always_home_ht_hit": None})
    cutoff = _utc(record.get("information_cutoff")) or _utc(record["generated_at"])
    baseline = baselines.before(cutoff)
    if baseline is None:
        marks["baseline"] = None
        return marks
    base_log_loss, base_rps = _scores(baseline.ft_probs, y_ft)
    base_top3 = [(home, away) for home, away, _ in baseline.top_scores]
    base = {
        "n_matches": baseline.n_matches,
        "ft": dict(zip(OUTCOMES, baseline.ft_probs, strict=True)),
        "ht": dict(zip(OUTCOMES, baseline.ht_probs, strict=True)),
        "top_scores": [f"{home}-{away}" for home, away in base_top3],
        "ft_pick": _pick(baseline.ft_probs),
        "top3_hit": actual_score in base_top3,
        "ft_log_loss": base_log_loss,
        "ft_rps": base_rps,
    }
    base["ft_hit"] = base["ft_pick"] == OUTCOMES[y_ft]
    if y_ht is not None:
        base["ht_pick"] = _pick(baseline.ht_probs)
        base["ht_hit"] = base["ht_pick"] == OUTCOMES[y_ht]
    else:
        base["ht_pick"] = None
        base["ht_hit"] = None
    marks["baseline"] = base
    return marks


def _running(rows: list[dict]) -> None:
    n = ft = top3 = n_ht = ht = 0
    for row in rows:
        if row["status"] != "finished":
            row["running"] = None
            continue
        n += 1
        ft += int(bool(row["ft_hit"]))
        top3 += int(bool(row["top3_hit"]))
        if row.get("ht_hit") is not None:
            n_ht += 1
            ht += int(bool(row["ht_hit"]))
        row["running"] = {
            "n": n,
            "ft_hit_rate": ft / n,
            "top3_hit_rate": top3 / n,
            "ht_hit_rate": ht / n_ht if n_ht else None,
        }


def _rate(values: list) -> float | None:
    kept = [bool(value) for value in values if value is not None]
    if not kept:
        return None
    return sum(kept) / len(kept)


def _mean(values: list) -> float | None:
    kept = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not kept:
        return None
    return float(np.mean(kept))


def summarize_history(rows: list[dict]) -> dict:
    """Hit rates for the model and the honest baselines on the same finished matches."""

    finished = [row for row in rows if row["status"] == "finished"]
    with_baseline = [row for row in finished if row.get("baseline")]
    return {
        "n_finished": len(finished),
        "n_ht": sum(1 for row in finished if row.get("ht_hit") is not None),
        "n_awaiting": sum(1 for row in rows if row["status"] == "awaiting"),
        "n_not_played": sum(1 for row in rows if row["status"] == "not_played"),
        "n_with_baseline": len(with_baseline),
        "model": {
            "ft_hit_rate": _rate([row["ft_hit"] for row in finished]),
            "ht_hit_rate": _rate([row.get("ht_hit") for row in finished]),
            "top3_hit_rate": _rate([row["top3_hit"] for row in finished]),
            "ft_log_loss": _mean([row["ft_log_loss"] for row in finished]),
            "ft_rps": _mean([row["ft_rps"] for row in finished]),
        },
        "always_home": {
            "ft_hit_rate": _rate([row["always_home_ft_hit"] for row in finished]),
            "ht_hit_rate": _rate([row.get("always_home_ht_hit") for row in finished]),
        },
        "frequency": {
            "ft_hit_rate": _rate([row["baseline"]["ft_hit"] for row in with_baseline]),
            "ht_hit_rate": _rate([row["baseline"]["ht_hit"] for row in with_baseline]),
            "top3_hit_rate": _rate([row["baseline"]["top3_hit"] for row in with_baseline]),
            "ft_log_loss": _mean([row["baseline"]["ft_log_loss"] for row in with_baseline]),
            "ft_rps": _mean([row["baseline"]["ft_rps"] for row in with_baseline]),
        },
        "uniform_ft_log_loss": math.log(3.0),
    }


def history_report(root: Path, matches: pd.DataFrame | None, *, now: Any = None) -> dict:
    archive = PredictionArchive(root)
    live_rows = build_history(archive.live_records(), matches, now=now)
    rebuilt_rows = build_history(archive.reconstruction_records(), matches, now=now)
    return {
        "generated_at": (utc_now() if now is None else _utc(now)).isoformat(),
        "live": {"rows": live_rows, "summary": summarize_history(live_rows)},
        "backtest_reconstruction": {"rows": rebuilt_rows, "summary": summarize_history(rebuilt_rows)},
    }


def current_season_id(matches: pd.DataFrame) -> int | None:
    seasons = matches.dropna(subset=["season_id", "starting_at"])
    if seasons.empty:
        return None
    return int(seasons.sort_values("starting_at").iloc[-1]["season_id"])


@dataclass
class Reconstruction:
    records: list[dict]
    season_id: int | None
    days: int
    skipped_days: int


def reconstruct_season(
    matches: pd.DataFrame,
    config: ModelConfig,
    *,
    season_id: int | None = None,
    generated_at: Any = None,
) -> Reconstruction:
    """Walk-forward predictions for finished matches of one season.

    Each UTC day is predicted by a fit on finished matches that kicked off
    before that day's 00:00 UTC. That is the same cut as ``walk_forward``.
    Days with fewer than ``config.min_train_matches`` earlier results are
    skipped rather than predicted from a thin sample.
    """

    if season_id is None:
        season_id = current_season_id(matches)
    finished = matches[matches["status"] == "finished"].dropna(subset=["home_goals_ft", "away_goals_ft"])
    generated = _utc(generated_at) if generated_at is not None else utc_now()
    records: list[dict] = []
    previous: ScorelineModel | None = None
    days = skipped = 0
    version = model_version()
    for day, train, test in iter_calendar_days(finished):
        target = test[test["season_id"] == season_id] if season_id is not None else test
        if target.empty:
            continue
        if len(train) < config.min_train_matches:
            skipped += 1
            continue
        try:
            model = fit_models(
                train,
                pd.Timestamp(day),
                config,
                initial_ft=None if previous is None else previous.ft,
                initial_ht=None if previous is None else previous.ht,
                initial_sh=None if previous is None else previous.sh,
            )
        except FitError:
            skipped += 1
            continue
        previous = model
        days += 1
        for _, row in target.iterrows():
            records.append(
                archive_record(
                    model.predict_row(row),
                    generated_at=generated,
                    model_version=version,
                    source=RECONSTRUCTION,
                    origin="backfill-archive",
                    information_cutoff=pd.Timestamp(day),
                )
            )
    return Reconstruction(records=records, season_id=season_id, days=days, skipped_days=skipped)
