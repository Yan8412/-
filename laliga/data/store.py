"""Local CSV store for the normalised match table and the fitted model."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from laliga.data.parse import MATCH_COLUMNS, empty_matches


class MatchStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.processed_path = self.root / "processed" / "matches.csv"
        self.model_path = self.root / "models" / "dixon_coles.json"
        self.cache_dir = self.root / "cache"
        self.seasons_path = self.root / "processed" / "seasons.json"
        self.predictions_dir = self.root / "predictions"

    def load(self) -> pd.DataFrame:
        if not self.processed_path.exists():
            return empty_matches()
        frame = pd.read_csv(self.processed_path)
        return _coerce(frame)

    def replace(self, frame: pd.DataFrame) -> None:
        self._write(_coerce(frame))

    def upsert(self, frame: pd.DataFrame) -> pd.DataFrame:
        incoming = _coerce(frame)
        if incoming.empty:
            return self.load()
        current = self.load()
        if current.empty:
            merged = incoming
        else:
            merged = pd.concat([current, incoming], ignore_index=True)
            merged = merged.drop_duplicates("fixture_id", keep="last")
        merged = merged.sort_values(["starting_at", "fixture_id"]).reset_index(drop=True)
        self._write(merged)
        return merged

    def save_model_document(self, document: dict) -> None:
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        self.model_path.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")

    def load_model_document(self) -> dict | None:
        if not self.model_path.exists():
            return None
        return json.loads(self.model_path.read_text(encoding="utf-8"))

    def save_seasons(self, document: dict) -> None:
        self.seasons_path.parent.mkdir(parents=True, exist_ok=True)
        self.seasons_path.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")

    def _write(self, frame: pd.DataFrame) -> None:
        self.processed_path.parent.mkdir(parents=True, exist_ok=True)
        out = frame.copy()
        if not out.empty and pd.api.types.is_datetime64_any_dtype(out["starting_at"]):
            out["starting_at"] = out["starting_at"].dt.strftime("%Y-%m-%d %H:%M:%S%z")
        out.to_csv(self.processed_path, index=False)


def _coerce(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return empty_matches()
    for column in MATCH_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.NA
    frame = frame[MATCH_COLUMNS].copy()
    frame["starting_at"] = pd.to_datetime(frame["starting_at"], utc=True, errors="coerce")
    for column in (
        "fixture_id",
        "league_id",
        "season_id",
        "round_id",
        "state_id",
        "home_team_id",
        "away_team_id",
        "home_goals_ht",
        "away_goals_ht",
        "home_goals_ft",
        "away_goals_ft",
    ):
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("Int64")
    for column in (
        "season_name",
        "round_name",
        "state_name",
        "home_team_name",
        "away_team_name",
        "status",
    ):
        frame[column] = frame[column].fillna("").astype(str)
    frame = frame.dropna(subset=["fixture_id", "starting_at", "home_team_id", "away_team_id"])
    return frame.sort_values(["starting_at", "fixture_id"]).reset_index(drop=True)
