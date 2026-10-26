"""Walk-forward retune, calibrate, and promote a challenger without leakage.

The most recent finished matches are a holdout. Optuna (or a small grid, if
optuna is not installed) only sees matches before that window. The champion is replaced when the challenger improves the holdout target:
blend full-time log loss when both sides scored a blend on every holdout
match, otherwise the football full-time log loss. The other metrics stay
inside a tolerance. Rollback is unchanged. The holdout model is fit on
matches strictly before the window, so the promotion decision cannot see
those results during tuning.

Temperature is fit on the tuning window's out-of-sample 1X2 probabilities
and then frozen. The geometric blend weight is fit the same way and frozen.
Neither is refit on the holdout.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from laliga.config import ModelConfig
from laliga.model.dixon_coles import FitError
from laliga.model.joint import temper_matrix
from laliga.model.metrics import multiclass_log_loss, multiclass_rps, outcome_index
from laliga.model.service import ScorelineModel, fit_models
from laliga.model.tracks import TRACK_MAX_FT, TRACK_MAX_HALF, TRACK_TEMPERATURE, geometric_rates, market_rates_for_row
from laliga.model.joint import convolve

REGISTRY_SCHEMA = 1


@dataclass
class UpgradeResult:
    promoted: bool
    reason: str
    report: dict

    def to_dict(self) -> dict:
        return self.report


def registry_path(root: Path) -> Path:
    return Path(root) / "models" / "registry.json"


def load_registry(root: Path) -> dict | None:
    path = registry_path(root)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def save_registry(root: Path, payload: dict) -> None:
    path = registry_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def rollback(root: Path) -> str:
    registry = load_registry(root)
    if not registry or not registry.get("previous"):
        raise FitError("注册表里没有可以退回的上一版冠军。")
    current = registry.get("champion")
    registry["champion"] = registry["previous"]
    registry["previous"] = current
    registry.setdefault("history", []).append({"event": "rollback", "at": _now(), "champion": registry["champion"]["id"]})
    save_registry(root, registry)
    return f"已退回 {registry['champion']['id']}。当前生产参数以注册表冠军为准。"


def apply_champion(config: ModelConfig, root: Path) -> ModelConfig:
    """Overlay champion hyperparameters when a registry exists. CLI max_iter is kept."""

    registry = load_registry(root)
    if not registry or not isinstance(registry.get("champion"), dict):
        return config
    params = registry["champion"].get("params") or {}
    return config_from_params(config, params)


def config_from_params(config: ModelConfig, params: dict) -> ModelConfig:
    values = {
        "xi": float(params.get("xi", config.xi)),
        "l2": float(params.get("l2", config.l2)),
        "rho_prior": float(params.get("rho_prior", config.rho_prior)),
        "rho_precision": float(params.get("rho_precision", config.rho_precision)),
        "xg_weight": float(params.get("xg_weight", config.xg_weight)),
        "eta_scale": float(params.get("eta_scale", config.eta_scale)),
        "nb_dispersion": float(params.get("nb_dispersion", config.nb_dispersion)),
        "sh_rate_scale": float(params.get("sh_rate_scale", config.sh_rate_scale)),
        "temperature": float(params.get("temperature", config.temperature)),
        "max_iter": config.max_iter,
        "min_matches": config.min_matches,
        "min_train_matches": config.min_train_matches,
        "max_goals": config.max_goals,
        "prior_strength": config.prior_strength,
        "min_ht_matches": config.min_ht_matches,
    }
    return ModelConfig(**values)


def upgrade(
    matches: pd.DataFrame,
    config: ModelConfig,
    root: Path,
    *,
    quotes: dict | None = None,
    n_trials: int = 12,
    holdout_matches: int = 80,
    min_promote: int = 40,
    rule: str = "bootstrap",
    refit_every_days: int = 14,
) -> UpgradeResult:
    finished = matches[matches["status"] == "finished"].dropna(subset=["home_goals_ft", "away_goals_ft"]).sort_values(["starting_at", "fixture_id"])
    if len(finished) < holdout_matches + config.min_train_matches + 4:
        raise FitError(
            f"完场比赛 {len(finished)} 场，不够留出 {holdout_matches} 场样本外并保留 {config.min_train_matches} 场训练。"
            "可以降低 --holdout 或 --min-train。"
        )
    cutoff = pd.Timestamp(finished.iloc[-holdout_matches]["starting_at"])
    if cutoff.tzinfo is None:
        cutoff = cutoff.tz_localize("UTC")
    cutoff = cutoff.floor("D")
    tuning = finished[finished["starting_at"] < cutoff].copy()
    holdout = finished[finished["starting_at"] >= cutoff].copy()
    if tuning.empty or holdout.empty:
        raise FitError("样本外切分后有一边是空的。")
    has_xg = "home_xg" in finished.columns and finished["home_xg"].notna().any()
    search = _search(tuning, config, n_trials=n_trials, has_xg=has_xg, refit_every_days=refit_every_days)
    best_params = dict(search["params"])
    temperature, _tune_rows = _temperature_from_tuning(tuning, config_from_params(config, best_params), refit_every_days)
    best_params["temperature"] = temperature
    blend_weight = _blend_weight(tuning, config_from_params(config, best_params), quotes or {}, refit_every_days)
    best_params["blend_market_weight"] = blend_weight
    challenger_config = config_from_params(config, best_params)
    challenger_model = fit_models(tuning, cutoff, challenger_config)
    challenger_metrics = _score(challenger_model, holdout, quotes or {}, blend_weight)
    registry = load_registry(root) or {"schema": REGISTRY_SCHEMA, "history": []}
    champion = registry.get("champion")
    champion_metrics = None
    if champion and champion.get("params"):
        champion_model = fit_models(tuning, cutoff, config_from_params(config, champion["params"]))
        champion_metrics = _score(champion_model, holdout, quotes or {}, float(champion["params"].get("blend_market_weight") or 0.0))
    promoted, reason = _decide(challenger_metrics, champion_metrics, rule=rule, min_promote=min_promote)
    version = _next_version(registry)
    entry = {
        "id": version,
        "params": best_params,
        "promoted_at": _now(),
        "holdout_from": cutoff.isoformat(),
        "holdout": challenger_metrics,
        "champion_holdout": champion_metrics,
        "search": {"engine": search["engine"], "n_trials": search["n_trials"], "best_value": search["best_value"]},
        "promoted": promoted,
        "reason": reason,
    }
    if promoted:
        registry["previous"] = champion
        registry["champion"] = entry
    registry.setdefault("history", []).append({"event": "upgrade", "at": _now(), **entry})
    registry["schema"] = REGISTRY_SCHEMA
    save_registry(root, registry)
    report = {
        "promoted": promoted,
        "reason": reason,
        "challenger": entry,
        "champion_id": (registry.get("champion") or {}).get("id"),
        "holdout_matches": int(len(holdout)),
        "tuning_matches": int(len(tuning)),
        "libraries": _library_notes(),
    }
    text = format_upgrade_report(report)
    report_path = Path(root) / "predictions" / "accuracy_history.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    previous_text = report_path.read_text(encoding="utf-8") if report_path.exists() else ""
    report_path.write_text(text + ("\n\n---\n\n" + previous_text if previous_text else ""), encoding="utf-8")
    (Path(root) / "predictions" / "accuracy_history.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return UpgradeResult(promoted, reason, report)


def format_upgrade_report(report: dict) -> str:
    challenger = report["challenger"]
    holdout = challenger["holdout"]
    lines = [
        f"# 准确率记录 {challenger['promoted_at']}",
        "",
        f"挑战者 {challenger['id']}：{'已升为冠军' if report['promoted'] else '未升冠军'}。{report['reason']}",
        f"调参只用 {report['tuning_matches']} 场（样本外窗口开始之前）。样本外 {report['holdout_matches']} 场，模型在窗口开始时冻结。",
        "",
        "## 样本外",
        "",
        f"- 纯足球 全场对数损失 {holdout['football']['ft_log_loss']:.4f}，RPS {holdout['football']['ft_rps']:.4f}，"
        f"前两比分命中 { _pct(holdout['football'].get('top2_hit_rate')) }，n = {holdout['football']['n']}",
    ]
    if holdout["football"].get("ht_log_loss") is not None:
        lines.append(
            f"- 纯足球 半场对数损失 {holdout['football']['ht_log_loss']:.4f}，半场 RPS {holdout['football'].get('ht_rps', float('nan')):.4f}"
        )
    if holdout["football"].get("htft_log_loss") is not None:
        lines.append(f"- 纯足球 半全场对数损失 {holdout['football']['htft_log_loss']:.4f}（n = {holdout['football'].get('n_htft')}）")
    market = holdout.get("market") or {}
    if market.get("n"):
        lines.append(
            f"- 市场 全场对数损失 {market['ft_log_loss']:.4f}，RPS {market['ft_rps']:.4f}，前两比分命中 {_pct(market.get('top2_hit_rate'))}，n = {market['n']}"
        )
    blend = holdout.get("blend") or {}
    if blend.get("n"):
        lines.append(
            f"- 混合 全场对数损失 {blend['ft_log_loss']:.4f}，RPS {blend['ft_rps']:.4f}，"
            f"市场权重 {challenger['params'].get('blend_market_weight')}，n = {blend['n']}"
        )
    champion_holdout = challenger.get("champion_holdout") if isinstance(challenger.get("champion_holdout"), dict) else {}
    old = champion_holdout.get("football") if isinstance(champion_holdout, dict) else None
    if old:
        lines.append(f"- 原冠军在同一窗口的全场对数损失 {old['ft_log_loss']:.4f}，RPS {old['ft_rps']:.4f}")
    old_blend = champion_holdout.get("blend") if isinstance(champion_holdout, dict) else None
    if isinstance(old_blend, dict) and old_blend.get("n") and old_blend.get("ft_log_loss") is not None:
        lines.append(f"- 原冠军在同一窗口的混合全场对数损失 {old_blend['ft_log_loss']:.4f}，n = {old_blend['n']}")
    lines.extend(["", "## 参数", "", "```"])
    for key, value in challenger["params"].items():
        lines.append(f"{key}: {value}")
    lines.extend(["```", "", "## 库", ""])
    lines.extend(f"- {note}" for note in report.get("libraries") or [])
    lines.append("")
    lines.append("这是概率估计的记录，不是投注建议。")
    return "\n".join(lines)


def _search(tuning: pd.DataFrame, config: ModelConfig, *, n_trials: int, has_xg: bool, refit_every_days: int) -> dict:
    def evaluate(params: dict) -> float:
        trial_config = config_from_params(config, {**params, "temperature": 1.0})
        rows = _oos_rows(tuning, trial_config, refit_every_days)
        if len(rows) < 8:
            return 5.0
        y = np.array([row["y"] for row in rows], dtype=int)
        probs = np.array([row["p"] for row in rows], dtype=float)
        return multiclass_log_loss(y, probs)

    try:
        import optuna
    except ImportError:
        grid = _fallback_grid(config, has_xg)
        best_value = np.inf
        best = grid[0]
        for params in grid[: max(n_trials, 1)]:
            value = evaluate(params)
            if value < best_value:
                best_value = value
                best = params
        return {"engine": "grid", "n_trials": min(len(grid), max(n_trials, 1)), "best_value": float(best_value), "params": best}

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=7)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    def objective(trial: "optuna.Trial") -> float:
        params = {
            "xi": trial.suggest_float("xi", 0.0008, 0.008, log=True),
            "l2": trial.suggest_float("l2", 0.5, 12.0, log=True),
            "rho_prior": trial.suggest_float("rho_prior", -0.15, 0.02),
            "rho_precision": trial.suggest_float("rho_precision", 20.0, 200.0, log=True),
            "eta_scale": trial.suggest_float("eta_scale", 0.0, 1.5),
            "nb_dispersion": trial.suggest_categorical("nb_dispersion", [1.0, 1.15]),
            "sh_rate_scale": trial.suggest_float("sh_rate_scale", 0.9, 1.1),
            "xg_weight": trial.suggest_float("xg_weight", 0.0, 0.5) if has_xg else 0.0,
        }
        return evaluate(params)

    study.optimize(objective, n_trials=max(n_trials, 1), show_progress_bar=False)
    params = dict(study.best_params)
    params.setdefault("xg_weight", 0.0)
    return {"engine": "optuna", "n_trials": max(n_trials, 1), "best_value": float(study.best_value), "params": params}


def _fallback_grid(config: ModelConfig, has_xg: bool) -> list[dict]:
    return [
        {"xi": config.xi, "l2": config.l2, "rho_prior": config.rho_prior, "rho_precision": config.rho_precision, "eta_scale": 1.0, "nb_dispersion": 1.0, "sh_rate_scale": 1.0, "xg_weight": 0.0},
        {"xi": config.xi * 1.5, "l2": config.l2, "rho_prior": -0.08, "rho_precision": 60.0, "eta_scale": 0.0, "nb_dispersion": 1.0, "sh_rate_scale": 1.0, "xg_weight": 0.2 if has_xg else 0.0},
        {"xi": max(config.xi * 0.6, 0.0008), "l2": 2.0, "rho_prior": -0.03, "rho_precision": 120.0, "eta_scale": 1.2, "nb_dispersion": 1.15, "sh_rate_scale": 0.95, "xg_weight": 0.0},
    ]


def _oos_rows(matches: pd.DataFrame, config: ModelConfig, refit_every_days: int) -> list[dict]:
    from laliga.model.metrics import iter_evaluation_folds

    rows = []
    previous: ScorelineModel | None = None
    last_fit = None
    model = None
    for day, train, test in iter_evaluation_folds(matches, config.min_train_matches):
        as_of = pd.Timestamp(day)
        if last_fit is None or (as_of - last_fit).days >= max(refit_every_days, 1) or model is None:
            try:
                model = fit_models(
                    train,
                    as_of,
                    config,
                    initial_ft=None if previous is None else previous.ft,
                    initial_ht=None if previous is None else previous.ht,
                    initial_sh=None if previous is None else previous.sh,
                )
            except FitError:
                continue
            previous = model
            last_fit = as_of
        for _, match in test.iterrows():
            prediction = model.predict_row(match)
            rows.append(
                {
                    "y": outcome_index(int(match["home_goals_ft"]), int(match["away_goals_ft"])),
                    "p": prediction.ft.as_tuple(),
                    "fixture_id": int(match["fixture_id"]),
                    "home_team_id": int(match["home_team_id"]),
                    "away_team_id": int(match["away_team_id"]),
                    "match": match,
                    "model_marker": True,
                }
            )
    return rows


def _temperature_from_tuning(tuning: pd.DataFrame, config: ModelConfig, refit_every_days: int) -> tuple[float, list[dict]]:
    rows = _oos_rows(tuning, config, refit_every_days)
    if len(rows) < 8:
        return 1.0, rows
    y = np.array([row["y"] for row in rows], dtype=int)
    probs = np.array([row["p"] for row in rows], dtype=float)
    best_t = 1.0
    best_loss = multiclass_log_loss(y, probs)
    for temperature in (0.85, 0.95, 1.0, 1.05, 1.15, 1.3):
        scaled = np.vstack([_temper_probs(row, temperature) for row in probs])
        loss = multiclass_log_loss(y, scaled)
        if loss < best_loss - 1e-6:
            best_loss = loss
            best_t = temperature
    return best_t, rows


def _temper_probs(probs: np.ndarray, temperature: float) -> np.ndarray:
    matrix = np.asarray(probs, dtype=float).reshape(1, -1)
    return temper_matrix(matrix, temperature).reshape(-1)


def _blend_weight(tuning: pd.DataFrame, config: ModelConfig, quotes: dict, refit_every_days: int) -> float:
    """Freeze the last walk-forward market weight. The holdout is not in ``tuning``."""

    from laliga.model.tracks import compare_tracks

    has_price = bool(quotes)
    for column in ("raw_implied_home", "odds_home", "implied_home"):
        if column in tuning.columns and tuning[column].notna().any():
            has_price = True
    if not has_price:
        return 0.0
    try:
        report = compare_tracks(
            tuning,
            config,
            quotes=quotes,
            min_blend_history=8,
            refit_every_days=refit_every_days,
        )
    except FitError:
        return 0.0
    summary = (report.to_dict().get("blend_rate_weight") or {})
    if not summary:
        return 0.0
    return float(summary.get("end") or 0.0)


def _score(model: ScorelineModel, holdout: pd.DataFrame, quotes: dict, blend_weight: float) -> dict:
    football_probs = []
    market_probs = []
    blend_probs = []
    y = []
    y_ht = []
    ht_probs = []
    top2_hit = []
    htft_y = []
    htft_p = []
    from laliga.model.joint import HTFT_CODES, htft_code

    for _, match in holdout.iterrows():
        prediction = model.predict_row(match)
        y.append(outcome_index(int(match["home_goals_ft"]), int(match["away_goals_ft"])))
        football_probs.append(prediction.ft.as_tuple())
        top2_hit.append((int(match["home_goals_ft"]), int(match["away_goals_ft"])) in {(item.home_goals, item.away_goals) for item in prediction.top2_scores})
        if prediction.htft and pd.notna(match.get("home_goals_ht")) and pd.notna(match.get("away_goals_ht")):
            code = htft_code(int(match["home_goals_ht"]), int(match["away_goals_ht"]), int(match["home_goals_ft"]), int(match["away_goals_ft"]))
            htft_y.append(HTFT_CODES.index(code))
            htft_p.append([prediction.htft.get(item, 1e-15) for item in HTFT_CODES])
            y_ht.append(outcome_index(int(match["home_goals_ht"]), int(match["away_goals_ht"])))
            ht_probs.append(prediction.ht.as_tuple())
        implied = market_rates_for_row(match, quotes)
        rates = model.half_rates(int(match["home_team_id"]), int(match["away_team_id"]))
        if implied is not None:
            forecast = implied.forecast(
                eta=0.0, max_half=TRACK_MAX_HALF, max_ft=TRACK_MAX_FT, temperature=TRACK_TEMPERATURE
            )
            market_probs.append(forecast.ft_probs)
            if rates is not None and blend_weight > 0:
                blended = geometric_rates(rates, implied.half_rates(), blend_weight)
                blend_probs.append(
                    convolve(
                        blended, max_half=TRACK_MAX_HALF, max_ft=TRACK_MAX_FT, temperature=TRACK_TEMPERATURE
                    ).ft_probs
                )
    y_arr = np.array(y, dtype=int)
    football = _pack(y_arr, np.array(football_probs, dtype=float), top2_hit)
    if ht_probs:
        y_ht_arr = np.array(y_ht, dtype=int)
        ht = np.array(ht_probs, dtype=float)
        football["n_ht"] = len(y_ht)
        football["ht_log_loss"] = multiclass_log_loss(y_ht_arr, ht)
        football["ht_rps"] = multiclass_rps(y_ht_arr, ht)
        football["ht_brier"] = _brier(y_ht_arr, ht)
    if htft_p:
        football["n_htft"] = len(htft_y)
        football["htft_log_loss"] = multiclass_log_loss(np.array(htft_y, dtype=int), np.array(htft_p, dtype=float))
    payload = {"football": football, "market": {"n": 0}, "blend": {"n": 0}}
    if market_probs:
        # Market rows can be a subset. Score football and market on the same tail only when every row had a price.
        if len(market_probs) == len(y):
            payload["market"] = _pack(y_arr, np.array(market_probs, dtype=float), None)
        else:
            payload["market"] = {"n": len(market_probs)}
    if len(blend_probs) == len(y) and blend_probs:
        payload["blend"] = _pack(y_arr, np.array(blend_probs, dtype=float), None)
    return payload


def _pack(y: np.ndarray, probs: np.ndarray, top2_hit) -> dict:
    return {
        "n": int(len(y)),
        "ft_log_loss": multiclass_log_loss(y, probs),
        "ft_rps": multiclass_rps(y, probs),
        "ft_brier": _brier(y, probs),
        "top2_hit_rate": None if top2_hit is None else float(np.mean(top2_hit)),
    }


def _brier(y: np.ndarray, probs: np.ndarray) -> float:
    from laliga.model.metrics import multiclass_brier

    return multiclass_brier(y, probs)


def _decide(challenger: dict, champion: dict | None, *, rule: str, min_promote: int) -> tuple[bool, str]:
    football = challenger["football"]
    if football["n"] < min_promote and champion is not None:
        return False, f"样本外只有 {football['n']} 场，少于 {min_promote}，不更换冠军。"
    if champion is None or not champion:
        return True, "注册表还没有冠军。这一版用样本外之前的数据调参，并记为冠军。样本外数字只作记录，没有拿来选参数。"
    old = champion["football"]
    loss_delta = football["ft_log_loss"] - old["ft_log_loss"]
    rps_delta = football["ft_rps"] - old["ft_rps"]
    blend_target = _blend_target(challenger, champion)
    if blend_target is not None:
        return _decide_blend(blend_target, loss_delta, rps_delta, challenger, champion, rule=rule)
    if rule == "strict":
        if loss_delta < -1e-4 and rps_delta <= 0.005:
            return True, "样本外全场对数损失更低，RPS 没有明显变差，升为冠军。"
        return False, _rejection("strict", loss_delta, rps_delta, gain=1e-4, rps_limit=0.005)
    # Bootstrap the paired gap when both models were scored on the same matches.
    # The stored metrics are aggregates, so the strict gap is the decision when
    # per-match vectors were not kept. Require a clear log-loss gain.
    if loss_delta < -0.005 and rps_delta <= 0.002:
        return True, "样本外全场对数损失至少低 0.005，且 RPS 没有上升超过 0.002，升为冠军。"
    return False, _rejection("bootstrap", loss_delta, rps_delta, gain=0.005, rps_limit=0.002)


def _blend_target(challenger: dict, champion: dict) -> float | None:
    """Blend full-time log loss gap when both models scored a blend on the whole holdout."""

    football_n = int(challenger["football"]["n"])
    old_n = int((champion.get("football") or {}).get("n") or -1)
    if football_n <= 0 or football_n != old_n:
        return None
    new = _track_loss(challenger.get("blend"), "ft_log_loss", expected_n=football_n)
    old = _track_loss(champion.get("blend"), "ft_log_loss", expected_n=old_n)
    if new is None or old is None:
        return None
    return new - old


def _decide_blend(
    blend_delta: float,
    football_delta: float,
    rps_delta: float,
    challenger: dict,
    champion: dict,
    *,
    rule: str,
) -> tuple[bool, str]:
    """Promote on blend log loss. Football, RPS, and HT/FT may not regress past the tolerance."""

    strict = rule == "strict"
    gain = 1e-4 if strict else 0.005
    football_limit = 0.005 if strict else 0.01
    rps_limit = 0.005 if strict else 0.002
    htft_delta = _optional_delta(challenger, champion, "htft_log_loss")
    htft_ok = htft_delta is None or htft_delta <= football_limit
    if blend_delta < -gain and football_delta <= football_limit and rps_delta <= rps_limit and htft_ok:
        return True, "样本外混合全场对数损失更低，其余指标没有超出容差，升为冠军。"
    failed = []
    improved = -blend_delta
    if improved <= gain:
        if blend_delta > 0:
            failed.append(f"混合全场对数损失变差 {blend_delta:.4f}，需要至少降低 {gain:g}")
        else:
            failed.append(f"混合全场对数损失只降低 {max(improved, 0.0):.4f}，小于 {gain:g}")
    if football_delta > football_limit:
        failed.append(f"纯足球对数损失变差 {football_delta:.4f}，超过 {football_limit:g}")
    if rps_delta > rps_limit:
        failed.append(f"RPS 变差 {rps_delta:.4f}，超过 {rps_limit:g}")
    if htft_delta is not None and htft_delta > football_limit:
        failed.append(f"半全场对数损失变差 {htft_delta:.4f}，超过 {football_limit:g}")
    return False, "样本外未升冠军：" + "；".join(failed) + "。冠军不变。"


def _rejection(rule: str, loss_delta: float, rps_delta: float, *, gain: float, rps_limit: float) -> str:
    failed = []
    if loss_delta >= -gain:
        if loss_delta > 0:
            failed.append(f"全场对数损失变差 {loss_delta:.4f}，需要至少降低 {gain:g}")
        else:
            failed.append(f"全场对数损失只降低 {-loss_delta:.4f}，小于 {gain:g}")
    if rps_delta > rps_limit:
        failed.append(f"RPS 变差 {rps_delta:.4f}，超过 {rps_limit:g}")
    if not failed:
        failed.append(f"没有同时满足对数损失和 RPS（规则 {rule}）")
    return "样本外未升冠军：" + "；".join(failed) + "。冠军不变。"


def _track_loss(block, key: str, *, expected_n: int) -> float | None:
    if not isinstance(block, dict) or int(block.get("n") or 0) != expected_n:
        return None
    value = block.get(key)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _optional_delta(challenger: dict, champion: dict, key: str) -> float | None:
    new = _finite((challenger.get("football") or {}).get(key))
    old = _finite((champion.get("football") or {}).get(key))
    if new is None or old is None:
        return None
    return new - old


def _finite(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _library_notes() -> list[str]:
    notes = []
    try:
        import optuna

        notes.append(f"optuna {optuna.__version__}：调 xi、先验、ρ、xG 权重、状态系数和负二项离散度。")
    except ImportError:
        notes.append("没有安装 optuna，这次用了三组固定参数，不是贝叶斯搜索。")
    notes.append("shin 包没有加入依赖：去水位沿用仓库里的纯 Python Shin，Windows 上的 Python 3.13 不用编译。")
    notes.append("penaltyblog 没有加入依赖：它的 Dixon–Coles 和本仓库是同一似然，比分分布已经有梯度测试；Pi rating 不产出联合比分。")
    notes.append("soccerdata 没有加入依赖：football-data.co.uk 的 SP1 用标准库直接下载，队名对照已经在仓库里。")
    try:
        import lightgbm

        notes.append(f"lightgbm {lightgbm.__version__} 已安装，但默认升级不把它放进冠军：上下半场速率仍由 Dixon–Coles 给出。")
    except ImportError:
        notes.append("lightgbm 未安装。可选的泊松目标半场模型没有进这次升级，避免多一套重依赖。")
    notes.append("负二项只放在预测分布上（goalmodel 的想法）：离散度 1 就是泊松，不改攻防参数的拟合。")
    return notes


def _next_version(registry: dict) -> str:
    history = registry.get("history") or []
    return f"v{len(history) + 1}"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _pct(value) -> str:
    if value is None or not isinstance(value, (int, float)) or not math.isfinite(value):
        return "—"
    return f"{value * 100:.1f}%"
