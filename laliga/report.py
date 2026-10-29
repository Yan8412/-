"""Text, CSV, and JSON renderings of predictions and backtest reports."""

from __future__ import annotations

import csv
import io
import json
import unicodedata
from pathlib import Path

import pandas as pd

from laliga.backtest import BacktestReport
from laliga.config import PROMOTION_MARKET_LABELS, PROMOTION_MARKETS, TRACK_LABELS, TRACKS
from laliga.model.metrics import MetricBlock
from laliga.model.tracks import FixtureForecast

VIEW_LABELS = {"active": "启用模型（按回测准确率选择）", **{track: TRACK_LABELS[track] for track in TRACKS}}
DISCLAIMER = "以上为模型估计的概率，不是投注建议。"

_METRIC_SPECS = [
    ("全场对数损失", "ft_log_loss", False),
    ("全场 Brier 分数", "ft_brier", False),
    ("全场最可能结果命中率", "ft_accuracy", True),
    ("半场对数损失", "ht_log_loss", False),
    ("半场 Brier 分数", "ht_brier", False),
    ("半场最可能结果命中率", "ht_accuracy", True),
    ("实际比分落在前三的比例", "top3_hit_rate", True),
]


def format_predictions(forecasts: list[FixtureForecast], view: str = "active") -> str:
    if not forecasts:
        return "没有比赛可显示。"
    views = list(TRACKS) if view == "all" else [view]
    blocks = []
    if view in ("active", "all"):
        blocks.append(active_summary(forecasts))
    for name in views:
        blocks.append(f"【{VIEW_LABELS[name]}】\n" + _prediction_table(forecasts, name))
    return "\n\n".join(blocks) + "\n" + DISCLAIMER


def active_summary(forecasts: list[FixtureForecast]) -> str:
    first = forecasts[0].active
    parts = [f"{PROMOTION_MARKET_LABELS[market]}={TRACK_LABELS[first[market]['promoted']]}" for market in PROMOTION_MARKETS]
    fallbacks = sum(1 for forecast in forecasts for choice in forecast.active.values() if choice["fallback"])
    line = "当前启用：" + "，".join(parts) + "。"
    if fallbacks:
        line += "带 * 的格子：启用的模型这场缺赔率，改用纯足球。"
    return line


def _prediction_table(forecasts: list[FixtureForecast], view: str) -> str:
    headers = ["开球时间(UTC)", "主队", "客队", "全场胜", "全场平", "全场负", "半场胜", "半场平", "半场负", "比分1", "比分2", "比分3"]
    body = [headers]
    for forecast in forecasts:
        item = forecast.prediction
        shown = forecast.view(view)
        kickoff = pd.Timestamp(item.starting_at).strftime("%Y-%m-%d %H:%M")
        row = [kickoff, item.home_team, item.away_team]
        if not shown.available:
            row.extend(["—"] * 9)
        else:
            marks = {market: "*" if view == "active" and forecast.active[market]["fallback"] else "" for market in PROMOTION_MARKETS}
            row.extend(_pct(value) + marks["ft"] for value in shown.ft)
            row.extend(_pct(value) + marks["ht"] for value in shown.ht)
            row.extend(f"{home}-{away} {_pct(probability)}{marks['scores']}" for home, away, probability in shown.top_scores[:3])
        body.append(row)
    return _table(body)


def track_columns(item: dict) -> dict:
    """Flat per-track columns for one prediction document (``FixtureForecast.to_dict``)."""

    row: dict = {}
    for track in TRACKS:
        body = (item.get("tracks") or {}).get(track) or {}
        available = bool(body.get("available"))
        row[f"{track}_available"] = available
        for market in ("ft", "ht"):
            values = body.get(market) or {}
            for side in ("home", "draw", "away"):
                row[f"{track}_{market}_{side}"] = round(values[side], 6) if available and side in values else None
        scores = body.get("top_scores") or []
        for index in range(3):
            score = scores[index] if available and index < len(scores) else None
            row[f"{track}_score_{index + 1}"] = None if score is None else f"{score['home_goals']}-{score['away_goals']}"
            row[f"{track}_score_{index + 1}_prob"] = None if score is None else round(score["probability"], 6)
    for market in PROMOTION_MARKETS:
        choice = (item.get("active") or {}).get(market) or {}
        row[f"active_{market}_track"] = choice.get("track")
    return row


def predictions_frame(forecasts: list[FixtureForecast]) -> pd.DataFrame:
    rows = []
    for forecast in forecasts:
        item = forecast.prediction
        row = {
            "fixture_id": item.fixture_id,
            "starting_at": item.starting_at,
            "home_team": item.home_team,
            "away_team": item.away_team,
            "home_team_id": item.home_team_id,
            "away_team_id": item.away_team_id,
            "ft_home": round(item.ft.home, 6),
            "ft_draw": round(item.ft.draw, 6),
            "ft_away": round(item.ft.away, 6),
            "ht_home": round(item.ht.home, 6),
            "ht_draw": round(item.ht.draw, 6),
            "ht_away": round(item.ht.away, 6),
            "exp_home_goals": round(item.expected_goals_ft[0], 4),
            "exp_away_goals": round(item.expected_goals_ft[1], 4),
            "exp_home_goals_ht": round(item.expected_goals_ht[0], 4),
            "exp_away_goals_ht": round(item.expected_goals_ht[1], 4),
        }
        for index, score in enumerate(item.top_scores, start=1):
            row[f"score_{index}"] = f"{score.home_goals}-{score.away_goals}"
            row[f"score_{index}_prob"] = round(score.probability, 6)
        row.update(track_columns(forecast.to_dict()))
        rows.append(row)
    return pd.DataFrame(rows)


def write_predictions_csv(forecasts: list[FixtureForecast], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    predictions_frame(forecasts).to_csv(path, index=False)


def write_predictions_json(forecasts: list[FixtureForecast], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [forecast.to_dict() for forecast in forecasts]
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def predictions_csv_text(items: list[dict]) -> str:
    """Dashboard CSV: kickoff in UTC and Singapore time, football columns, then every track."""

    rows = []
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
        row.update(track_columns(item))
        rows.append(row)
    buffer = io.StringIO()
    if rows:
        writer = csv.DictWriter(buffer, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return buffer.getvalue()


def split_kickoff(value: str) -> tuple[str, str]:
    """Return (UTC, Asia/Singapore) clock times as YYYY-MM-DD HH:MM."""

    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")
    else:
        stamp = stamp.tz_convert("UTC")
    singapore = stamp.tz_convert("Asia/Singapore")
    return stamp.strftime("%Y-%m-%d %H:%M"), singapore.strftime("%Y-%m-%d %H:%M")


def backtest_metric_rows(payload: dict | None) -> list[dict]:
    """Rows for the dashboard and CSV: all evaluated matches, then the common subset per track."""

    if not payload:
        return []
    model = payload["model"]
    baseline = payload["baseline"]
    tracks = payload.get("tracks") or {}
    rows = []
    for label, key, percent in _METRIC_SPECS:
        row = {
            "label": label,
            "direction": "越高越好" if percent else "越低越好",
            "model": _number(model.get(key), percent),
            "baseline": _number(baseline.get(key), percent),
        }
        for track in (*TRACKS, "baseline"):
            row[f"common_{track}"] = _number((tracks.get(track) or {}).get(key), percent) if tracks else "—"
        rows.append(row)
    return rows


def backtest_csv_text(payload: dict) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(
        ["metric", "model", "baseline", "common_football", "common_market", "common_combined", "common_baseline", "direction"]
    )
    for row in backtest_metric_rows(payload):
        writer.writerow(
            [
                row["label"],
                row["model"],
                row["baseline"],
                row["common_football"],
                row["common_market"],
                row["common_combined"],
                row["common_baseline"],
                row["direction"],
            ]
        )
    return buffer.getvalue()


def format_backtest(report: BacktestReport) -> str:
    lines = [
        f"走步回测：{report.n_folds} 个评测日，"
        f"{report.first_test_day} 至 {report.last_test_day}，"
        f"每日前至少 {report.min_train_matches} 场完场比赛用于训练。",
        f"评测比赛 {report.model.n} 场（其中有半场比分的 {report.model.n_ht} 场）。",
        "某一天的比赛只使用该日 00:00 UTC 之前的完场数据。",
        "",
        _metric_table("全部评测比赛（纯足球）", report.model, report.baseline),
    ]
    for season in report.model_by_season:
        lines.append("")
        lines.append(
            _metric_table(
                f"赛季 {season}",
                report.model_by_season[season],
                report.baseline_by_season.get(season),
            )
        )
    lines.append("")
    lines.extend(format_tracks(report))
    lines.append("")
    lines.append("对数损失和 Brier 分数越低越好；命中率越高越好。")
    return "\n".join(lines)


def format_tracks(report: BacktestReport) -> list[str]:
    if not report.tracks:
        return [
            "三套模型对比：评测比赛里没有一场带本地赔率，只评测了纯足球模型。",
            "先运行 `python -m laliga odds` 抓取赔率（需要 SportMonks Odds add-on）。没有赔率时不会自动升级到纯市场或综合模型。",
        ]
    blocks = [_track_table(report.tracks, f"三套模型对比（{report.n_with_odds} 场三套都有预测的比赛）")]
    weights = report.ensemble
    blocks.append(
        f"综合模型权重（对数意见池，按对数损失拟合）：全场 足球 {weights.ft.football:.3f} / 市场 {weights.ft.market:.3f}，"
        f"半场 足球 {weights.ht.football:.3f} / 市场 {weights.ht.market:.3f}"
        + ("" if weights.ft.fitted else "（样本不足，等权）")
    )
    if report.n_with_odds < report.model.n:
        blocks.append(f"有 {report.model.n - report.n_with_odds} 场评测比赛没有赔率，不计入三套模型对比。")
    if report.promotion:
        blocks.append("自动升级判定：")
        blocks.extend("  " + body["reason"] for body in report.promotion["markets"].values())
    return blocks


def _track_table(tracks: dict[str, MetricBlock], title: str) -> str:
    names = [name for name in (*TRACKS, "baseline") if name in tracks]
    rows = [["指标", *[TRACK_LABELS.get(name, "历史频率基准") for name in names], "方向"]]
    for label, key, percent in _METRIC_SPECS:
        digits = 1 if percent else 4
        values = [_format_metric(getattr(tracks[name], key), percent, digits) for name in names]
        rows.append([label, *values, "越高越好" if percent else "越低越好"])
    return title + "\n" + _table(rows)


def format_team_table(model_document_ft: dict, title: str) -> str:
    teams = []
    for team_id, body in (model_document_ft.get("teams") or {}).items():
        teams.append((body.get("name") or team_id, body.get("attack"), body.get("defence"), body.get("matches")))
    teams.sort(key=lambda item: (-(item[1] or 0), item[0]))
    rows = [["球队", "进攻", "防守", "样本场次"]]
    for name, attack, defence, matches in teams:
        rows.append([str(name), f"{attack:.3f}", f"{defence:.3f}", str(matches)])
    header = (
        f"{title}  μ={model_document_ft['mu']:.3f}  "
        f"主场={model_document_ft['home_adv']:.3f}  ρ={model_document_ft['rho']:.3f}  "
        f"样本={model_document_ft['n_matches']}"
    )
    return header + "\n" + _table(rows)


def _metric_table(title: str, model: MetricBlock, baseline: MetricBlock | None) -> str:
    rows = [["指标", "模型", "历史频率基准", "方向"]]
    for label, key, percent in _METRIC_SPECS:
        digits = 1 if percent else 4
        rows.append(
            [
                label,
                _format_metric(getattr(model, key), percent, digits),
                _format_metric(None if baseline is None else getattr(baseline, key), percent, digits),
                "越高越好" if percent else "越低越好",
            ]
        )
    return title + "\n" + _table(rows)


def _number(value: float | None, percent: bool) -> str:
    if value is None or (isinstance(value, float) and value != value):
        return "—"
    if percent:
        return f"{value * 100:.1f}%"
    return f"{value:.4f}"


def _format_metric(value: float | None, higher_better: bool, digits: int) -> str:
    if value is None or (isinstance(value, float) and value != value):
        return "—"
    if higher_better and digits == 1:
        return f"{value * 100:.1f}%"
    return f"{value:.{digits}f}"


def _pct(probability: float) -> str:
    return f"{probability * 100:.1f}%"


def _table(rows: list[list[str]]) -> str:
    widths = [0] * len(rows[0])
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], _display_width(cell))
    lines = []
    for row_index, row in enumerate(rows):
        pieces = [_pad(cell, widths[index]) for index, cell in enumerate(row)]
        lines.append("  ".join(pieces).rstrip())
        if row_index == 0:
            lines.append("  ".join("─" * width for width in widths))
    return "\n".join(lines)


def _display_width(text: str) -> int:
    width = 0
    for char in text:
        width += 2 if unicodedata.east_asian_width(char) in {"W", "F"} else 1
    return width


def _pad(text: str, width: int) -> str:
    gap = width - _display_width(text)
    if gap <= 0:
        return text
    return text + (" " * gap)
