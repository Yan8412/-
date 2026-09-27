"""Text, CSV, and JSON renderings of predictions and backtest reports."""

from __future__ import annotations

import json
import unicodedata
from pathlib import Path

import pandas as pd

from laliga.backtest import BacktestReport, ComparisonReport, iter_model_keys, model_label
from laliga.model.metrics import MetricBlock
from laliga.model.service import Prediction


def format_predictions(predictions: list[Prediction]) -> str:
    if not predictions:
        return "没有比赛可显示。"
    headers = [
        "开球时间(UTC)",
        "主队",
        "客队",
        "全场胜",
        "全场平",
        "全场负",
        "半场胜",
        "半场平",
        "半场负",
        "比分1",
        "比分2",
        "比分3",
    ]
    body = [headers]
    for item in predictions:
        kickoff = pd.Timestamp(item.starting_at).strftime("%Y-%m-%d %H:%M")
        body.append(
            [
                kickoff,
                item.home_team,
                item.away_team,
                _pct(item.ft.home),
                _pct(item.ft.draw),
                _pct(item.ft.away),
                _pct(item.ht.home),
                _pct(item.ht.draw),
                _pct(item.ht.away),
                _score(item.top_scores[0]),
                _score(item.top_scores[1]),
                _score(item.top_scores[2]),
            ]
        )
    return _table(body) + "\n以上为模型估计的概率，不是投注建议。"


def predictions_frame(predictions: list[Prediction]) -> pd.DataFrame:
    rows = []
    for item in predictions:
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
        rows.append(row)
    return pd.DataFrame(rows)


def write_predictions_csv(predictions: list[Prediction], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    predictions_frame(predictions).to_csv(path, index=False)


def write_predictions_json(predictions: list[Prediction], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [item.to_dict() for item in predictions]
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def format_backtest(report: BacktestReport) -> str:
    lines = [
        f"走步回测：{report.n_folds} 个评测日，"
        f"{report.first_test_day} 至 {report.last_test_day}，"
        f"每日前至少 {report.min_train_matches} 场完场比赛用于训练。",
        f"评测比赛 {report.model.n} 场（其中有半场比分的 {report.model.n_ht} 场）。",
        "某一天的比赛只使用该日 00:00 UTC 之前的完场数据。",
        "",
        _metric_table("全部评测比赛", report.model, report.baseline),
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
    lines.append("对数损失和 Brier 分数越低越好；命中率越高越好。")
    return "\n".join(lines)


def format_comparison(report: ComparisonReport) -> str:
    scored = report.models["dixon_coles"]
    title = "模型对照：Dixon–Coles、历史频率基准"
    if report.with_xgboost:
        title = "XGBoost 对照：与 Dixon–Coles、历史频率基准同一走步、同一批比赛。"
    lines = [
        title,
        f"评测比赛 {scored.n} 场（有半场比分的 {scored.n_ht} 场），"
        f"{report.first_test_day} 至 {report.last_test_day}，{report.n_folds} 个评测日，"
        f"Dixon–Coles 拟合 {report.n_refits} 次（每 {report.refit_every_days} 个 UTC 日）。",
        f"每个评测日前至少 {report.min_train_matches} 场完场比赛。训练只用该日 00:00 UTC 之前的数据。",
        "生产预测仍是 Dixon–Coles。这个对照不在每日更新里。",
        "",
        _comparison_table(report),
    ]
    if report.odds_coverage is not None:
        lines.append(f"评测比赛中带完整赛前赔率的比例：{report.odds_coverage * 100:.1f}%。")
    lines.append("")
    lines.append("校准：每一类的平均预测概率，以及实际出现的频率。")
    lines.append(_calibration_table(report))
    home_bins = _home_bin_table(report)
    if home_bins is not None:
        lines.append("")
        lines.append("主胜概率 0.6–0.8：这一档的平均预测概率和实际主胜频率。")
        lines.append(home_bins)
    season_table = _season_table(report)
    if season_table is not None:
        lines.append("")
        lines.append("分赛季：全场对数损失和 RPS。")
        lines.append(season_table)
    lines.append("")
    lines.append("配对比较：每场全场对数损失之差、以及 RPS 之差的均值。负值表示减号前面的模型更好。")
    lines.append("区间是同一次 1000 次 bootstrap 的 2.5% 和 97.5% 分位。")
    lines.append(_paired_table(report))
    lines.append("")
    for note in report.notes:
        lines.append(note)
    return "\n".join(lines)


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


def _comparison_table(report: ComparisonReport) -> str:
    header = ["模型", "全场对数损失", "全场 Brier", "全场 RPS", "全场命中率", "半场对数损失", "半场 Brier", "半场命中率", "n"]
    rows = [header]
    for key in iter_model_keys(report.models):
        block = report.models.get(key)
        if block is None:
            continue
        rows.append(
            [
                model_label(key, report.history_source, report.blend),
                _format_metric(block.ft_log_loss, False, 4),
                _format_metric(block.ft_brier, False, 4),
                _format_metric(block.ft_rps, False, 4),
                _format_metric(block.ft_accuracy, True, 1),
                _format_metric(block.ht_log_loss, False, 4),
                _format_metric(block.ht_brier, False, 4),
                _format_metric(block.ht_accuracy, True, 1),
                str(block.n),
            ]
        )
    return _table(rows)


def _calibration_table(report: ComparisonReport) -> str:
    rows = [["模型", "结果", "平均预测概率", "实际频率", "n"]]
    labels = {"home": "主胜", "draw": "平", "away": "客胜"}
    for key in iter_model_keys(report.models):
        summary = report.calibration.get(key)
        if not summary:
            continue
        for item in summary.get("per_class") or []:
            rows.append(
                [
                    model_label(key, report.history_source, report.blend),
                    labels.get(item["outcome"], item["outcome"]),
                    f"{item['mean_predicted']:.4f}",
                    f"{item['observed_frequency']:.4f}",
                    str(item["n"]),
                ]
            )
    return _table(rows)


def _home_bin_table(report: ComparisonReport) -> str | None:
    rows = [["模型", "平均预测主胜", "实际主胜频率", "n"]]
    for key in iter_model_keys(report.models):
        summary = report.calibration.get(key) or {}
        for item in summary.get("home_probability_bins") or []:
            if abs(float(item.get("bin_low", -1)) - 0.6) > 1e-9:
                continue
            if abs(float(item.get("bin_high", -1)) - 0.8) > 1e-9:
                continue
            rows.append(
                [
                    model_label(key, report.history_source, report.blend),
                    f"{item['mean_predicted_home']:.4f}",
                    f"{item['observed_home_frequency']:.4f}",
                    str(item["n"]),
                ]
            )
    if len(rows) == 1:
        return None
    return _table(rows)


def _season_table(report: ComparisonReport) -> str | None:
    blocks = report.by_season or {}
    seasons = blocks.get("dixon_coles") or {}
    if len(seasons) < 2:
        return None
    rows = [["赛季", "模型", "全场对数损失", "全场 RPS", "n"]]
    for key in iter_model_keys(report.models):
        for season, block in (blocks.get(key) or {}).items():
            rows.append(
                [
                    season,
                    model_label(key, report.history_source, report.blend),
                    _format_metric(block.ft_log_loss, False, 4),
                    _format_metric(block.ft_rps, False, 4),
                    str(block.n),
                ]
            )
    return _table(rows)


def _paired_table(report: ComparisonReport) -> str:
    rows = [["比较", "对数损失之差", "95% 区间低", "95% 区间高", "RPS 之差", "RPS 区间低", "RPS 区间高", "n"]]
    for item in report.paired:
        rows.append(
            [
                item["label"],
                f"{item['mean_logloss_difference']:.4f}",
                f"{item['ci_low']:.4f}",
                f"{item['ci_high']:.4f}",
                _format_optional(item.get("mean_rps_difference")),
                _format_optional(item.get("rps_ci_low")),
                _format_optional(item.get("rps_ci_high")),
                str(item["n"]),
            ]
        )
    return _table(rows)


def _format_optional(value) -> str:
    if value is None:
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    if number != number:
        return "—"
    return f"{number:.4f}"


def _metric_table(title: str, model: MetricBlock, baseline: MetricBlock | None) -> str:
    rows = [["指标", "模型", "历史频率基准", "方向"]]
    rows.extend(
        [
            _metric_row("全场对数损失", model.ft_log_loss, None if baseline is None else baseline.ft_log_loss, False, 4),
            _metric_row("全场 Brier 分数", model.ft_brier, None if baseline is None else baseline.ft_brier, False, 4),
            _metric_row("全场最可能结果命中率", model.ft_accuracy, None if baseline is None else baseline.ft_accuracy, True, 1),
            _metric_row("半场对数损失", model.ht_log_loss, None if baseline is None else baseline.ht_log_loss, False, 4),
            _metric_row("半场 Brier 分数", model.ht_brier, None if baseline is None else baseline.ht_brier, False, 4),
            _metric_row("半场最可能结果命中率", model.ht_accuracy, None if baseline is None else baseline.ht_accuracy, True, 1),
            _metric_row("实际比分落在前三的比例", model.top3_hit_rate, None if baseline is None else baseline.top3_hit_rate, True, 1),
        ]
    )
    return title + "\n" + _table(rows)


def _metric_row(name: str, model: float | None, baseline: float | None, higher_better: bool, digits: int) -> list[str]:
    return [
        name,
        _format_metric(model, higher_better, digits),
        _format_metric(baseline, higher_better, digits),
        "越高越好" if higher_better else "越低越好",
    ]


def _format_metric(value: float | None, higher_better: bool, digits: int) -> str:
    if value is None or (isinstance(value, float) and value != value):
        return "—"
    if higher_better and digits == 1:
        return f"{value * 100:.1f}%"
    return f"{value:.{digits}f}"


def _pct(probability: float) -> str:
    return f"{probability * 100:.1f}%"


def _score(item) -> str:
    return f"{item.home_goals}-{item.away_goals} {_pct(item.probability)}"


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
