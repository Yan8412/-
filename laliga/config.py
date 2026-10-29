"""Paths, league constants, and model defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# Spanish Primera División. SportMonks Football API v3 league id.
# https://docs.sportmonks.com/v3/sportmonks-ai-docs/cursor-rules
LA_LIGA_LEAGUE_ID = 564

API_BASE_URL = "https://api.sportmonks.com/v3/football"
TOKEN_ENV = "SPORTMONKS_API_TOKEN"
DATA_DIR_ENV = "LALIGA_DATA_DIR"

# Regulation-time score descriptions. See SportMonks "scores" include:
# 1ST_HALF is the score at the break; 2ND_HALF is the cumulative 90-minute score.
SCORE_HT = "1ST_HALF"
SCORE_FT = "2ND_HALF"
SCORE_CURRENT = "CURRENT"

# Fixture state developer_name values used when the state include is present.
# state_id 5 is FT in current v3 payloads; ids are a fallback when the include
# is missing. See the states filter examples in the fixtures endpoint docs.
FINISHED_STATE_NAMES = frozenset({"FT", "AWARDED"})
SCHEDULED_STATE_NAMES = frozenset({"NS", "TBA", "PENDING", "DELAYED"})
# Regulation score is still 2ND_HALF if a cup tie went to extra time.
REGULATION_STATE_NAMES = FINISHED_STATE_NAMES | frozenset({"AET", "FT_PEN"})
SKIP_STATE_NAMES = frozenset(
    {
        "POSTPONED",
        "CANCELLED",
        "SUSPENDED",
        "DELETED",
        "ABANDONED",
        "WO",
        "INTERRUPTED",
        "AWAITING_UPDATES",
        "INPLAY_1ST_HALF",
        "HT",
        "BREAK",
        "INPLAY_ET",
        "INPLAY_PENALTIES",
        "EXTRA_TIME_BREAK",
        "INPLAY_2ND_HALF",
        "PEN_BREAK",
    }
)

FINISHED_STATE_IDS = frozenset({5, 17})  # FT, AWARDED
SCHEDULED_STATE_IDS = frozenset({1, 13, 16, 26})  # NS, TBA, DELAYED, PENDING

# Date-range endpoint rejects windows longer than 100 days (inclusive).
MAX_BETWEEN_DAYS = 100

# xi = 0.0025 => half-life ln(2)/xi ≈ 277 days, so several seasons still contribute.
DEFAULT_XI = 0.0025
# Many La Liga subscriptions, including the one used to verify this project,
# only list seasons from 2024/2025 onward (three seasons). Asking for more
# just keeps however many the league endpoint actually returns.
DEFAULT_SEASONS = 3
DEFAULT_MIN_TRAIN_MATCHES = 320
DEFAULT_HT_GOAL_RATIO = 0.45

# SportMonks v3 market ids (GET /v3/odds/markets). Standard pre-match feed.
MARKET_FULLTIME_RESULT = 1
MARKET_HALF_TIME_FULL_TIME = 29
MARKET_HALF_TIME_RESULT = 31
MARKET_CORRECT_SCORE = 57
ODDS_MARKETS = (
    MARKET_FULLTIME_RESULT,
    MARKET_HALF_TIME_RESULT,
    MARKET_HALF_TIME_FULL_TIME,
    MARKET_CORRECT_SCORE,
)

TRACKS = ("football", "market", "combined")
TRACK_LABELS = {"football": "纯足球", "market": "纯市场", "combined": "综合"}
PROMOTION_MARKETS = ("ft", "ht", "scores")
PROMOTION_MARKET_LABELS = {"ft": "全场胜平负", "ht": "半场胜平负", "scores": "前三比分"}


@dataclass(frozen=True)
class PromotionConfig:
    """When a challenger track replaces the active one for a market."""

    # Matches where all three tracks had a forecast. Below this, nothing changes.
    min_matches: int = 60
    # Paired mean improvement divided by its standard error.
    min_z: float = 1.0
    # Combined-weight fitting needs this many earlier matches with odds; equal weights before.
    min_ensemble_rows: int = 40


@dataclass(frozen=True)
class ModelConfig:
    """Hyperparameters shared by training, backtest, and prediction."""

    xi: float = DEFAULT_XI
    max_goals: int = 10
    # Gaussian prior precision on attack/defence, added to the total log-likelihood.
    # Influence per match shrinks as the sample grows.
    l2: float = 4.0
    rho_prior: float = -0.05
    rho_precision: float = 80.0
    max_iter: int = 150
    min_matches: int = 10
    prior_strength: float = 12.0
    min_train_matches: int = DEFAULT_MIN_TRAIN_MATCHES
    # Direct half-time fit is used only when at least this many HT scores exist.
    min_ht_matches: int = 30


def project_data_dir(explicit: str | Path | None = None) -> Path:
    """Resolve the data directory: CLI flag, then env, then ./data."""

    if explicit:
        return Path(explicit).expanduser().resolve()
    env = os.environ.get(DATA_DIR_ENV)
    if env:
        return Path(env).expanduser().resolve()
    return Path.cwd() / "data"


def api_token() -> str | None:
    token = os.environ.get(TOKEN_ENV, "").strip()
    return token or None


def api_base_url() -> str:
    """Football API base. Override with SPORTMONKS_API_BASE only in tests."""

    return os.environ.get("SPORTMONKS_API_BASE", API_BASE_URL).rstrip("/")
