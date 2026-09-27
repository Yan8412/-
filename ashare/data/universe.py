"""Full A-share catalog: current listings plus delisted names when Baostock has them."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from ashare.rules.limits import classify_board, is_risk_name, normalize_code

logger = logging.getLogger(__name__)

CATALOG_NAME = "universe/catalog.json"
# Delistings older than this are outside a two-year test plus warmup.
LOOKBACK_DAYS = 1200


@dataclass
class Instrument:
    code: str
    name: str
    ipo_date: str | None = None
    out_date: str | None = None
    listed: bool = True
    board: str = ""
    bars: int = 0
    last_bar: str | None = None
    error: str | None = None

    def ipo(self) -> date | None:
        return date.fromisoformat(self.ipo_date) if self.ipo_date else None

    def out(self) -> date | None:
        return date.fromisoformat(self.out_date) if self.out_date else None


def merge_names(sina: list[tuple[str, str]], basics: list[dict]) -> list[Instrument]:
    """Sina supplies the current name. Baostock supplies IPO, delist date, and old codes."""
    by_code: dict[str, Instrument] = {}
    for raw_code, raw_name in sina:
        try:
            code = normalize_code(raw_code)
        except ValueError:
            continue
        board = classify_board(code)
        if board in {"b_share", "other"}:
            continue
        by_code[code] = Instrument(code=code, name=raw_name or code, listed=True, board=board)
    for row in basics:
        code = str(row.get("code") or "")
        if not code:
            continue
        board = classify_board(code) if _is_six_digits(code) else "other"
        if board in {"b_share", "other"}:
            continue
        listed = bool(row.get("listed", True))
        existing = by_code.get(code)
        if existing is None:
            by_code[code] = Instrument(
                code=code,
                name=str(row.get("name") or code),
                ipo_date=_blank_date(row.get("ipo_date")),
                out_date=_blank_date(row.get("out_date")),
                listed=listed,
                board=board,
            )
            continue
        existing.ipo_date = existing.ipo_date or _blank_date(row.get("ipo_date"))
        existing.out_date = existing.out_date or _blank_date(row.get("out_date"))
        existing.board = existing.board or board
        # A code still quoted by Sina is listed, even if Baostock's status is stale.
        if not listed and existing.listed:
            continue
        existing.listed = listed
    return [by_code[code] for code in sorted(by_code)]


def load_catalog(cache_dir: Path, today: date, source) -> tuple[list[Instrument], list[str]]:
    """Current SH/SZ/ChiNext/STAR names, optional BSE, plus Baostock delistings."""
    notes: list[str] = []
    path = cache_dir / CATALOG_NAME
    cached = _read_catalog(path, today)
    if cached is not None:
        notes.append(f"股票池目录使用 {path.name} 缓存，共 {len(cached)} 只。")
        return cached, notes

    sina = source.list_node("hs_a")
    notes.append(f"新浪 hs_a 当前名单 {len(sina)} 只（沪深主板、创业板、科创板）。")
    bse: list[tuple[str, str]] = []
    bse_note = "北交所：新浪 hs_a 不含北交所。"
    for node in ("hs_bjs", "bj", "bj_a"):
        rows = source.list_node(node, page_cap=40)
        if rows:
            bse = rows
            bse_note = f"北交所：新浪节点 {node} 返回 {len(rows)} 只，已并入。"
            break
        bse_note = f"北交所：新浪节点 {node} 没有返回代码，本次不纳入北交所。"
    notes.append(bse_note)

    basics, basic_notes = _baostock_basics(cache_dir)
    notes.extend(basic_notes)
    merged = merge_names(sina + bse, basics)
    _write_catalog(path, merged)
    notes.append(f"合并后的目录共 {len(merged)} 只，已写入缓存。")
    return merged, notes


def select_for_download(catalog: list[Instrument], today: date) -> tuple[list[Instrument], dict[str, int]]:
    """Drop names that delisted before the lookback. Everything else is fetched."""
    floor = today - timedelta(days=LOOKBACK_DAYS)
    chosen: list[Instrument] = []
    stats = {
        "catalog": len(catalog),
        "listed": 0,
        "delisted": 0,
        "delisted_before_lookback": 0,
        "current_st": 0,
        "boards": {},
    }
    boards: dict[str, int] = {}
    for item in catalog:
        if item.listed:
            stats["listed"] += 1
        else:
            stats["delisted"] += 1
        if is_risk_name(item.name):
            stats["current_st"] += 1
        out = item.out()
        if (not item.listed) and out is not None and out < floor:
            stats["delisted_before_lookback"] += 1
            continue
        ipo = item.ipo()
        if ipo is not None and ipo > today:
            continue
        chosen.append(item)
        boards[item.board] = boards.get(item.board, 0) + 1
    stats["boards"] = boards
    stats["download"] = len(chosen)
    return chosen, stats


def coverage_notes(
    chosen: list[Instrument],
    stats: dict,
    start: date,
    end: date,
    loaded: int,
) -> list[str]:
    """Chinese sentences a reader can check against the counts. No invented totals."""
    in_window = []
    for item in chosen:
        out = item.out()
        if item.listed:
            continue
        if out is None or out >= start:
            in_window.append(item)
    with_bars = [item for item in in_window if item.bars > 0 and item.last_bar and item.last_bar >= start.isoformat()]
    ended_early = [
        item for item in in_window if item.bars > 0 and item.last_bar and item.last_bar < start.isoformat()
    ]
    missing = [item for item in in_window if item.bars <= 0]
    boards = stats.get("boards") or {}
    board_text = "、".join(f"{name} {count}" for name, count in sorted(boards.items()))
    notes = [
        f"目录 {stats.get('catalog', 0)} 只，其中当前仍上市 {stats.get('listed', 0)} 只，"
        f"Baostock 标记已退市 {stats.get('delisted', 0)} 只。",
        f"退市早于回看起点（约 {LOOKBACK_DAYS} 个自然日）而未下载的有 {stats.get('delisted_before_lookback', 0)} 只。",
        f"实际发起日线下载 {stats.get('download', 0)} 只，进入回测 {loaded} 只。"
        f"K线短于策略所需而跳过 {stats.get('too_short', 0)} 只。"
        f"当前风险名称整段跳过 {stats.get('st_skipped', 0)} 只。板块：{board_text or '无'}。",
        f"样本 {start.isoformat()} 至 {end.isoformat()} 内需要覆盖的退市股票 {len(in_window)} 只："
        f"日线覆盖到样本起点之后的 {len(with_bars)} 只，"
        f"有日线但最后一根早于样本起点的 {len(ended_early)} 只，"
        f"免费日线没有返回的 {len(missing)} 只。",
        f"当前名称含 ST、*ST 或退市字样的有 {stats.get('current_st', 0)} 只。"
        "腾讯和新浪日线没有逐日 ST 字段。Baostock 有逐日 isST，但单只查询实测约 0.8 秒，"
        "全市场大约要一小时，这次没有拉取。因此这些股票按当前名称整段排除；"
        "已经摘帽、样本期内曾经 ST 的日期没有被去掉。这不是逐日 ST 过滤。",
        "停牌日在腾讯日线里通常直接缺 bar，回测按当天没有行情处理，不使用未来的复牌价。",
        "次新用上市日期（Baostock 有的话）或首根 K 线晚于样本起点来判断，默认上市不满 120 根日线不买。"
        "成交额下限和价格带按信号当日的成交额和收盘价判断。",
        "近 20 个交易日最高价到最低价的振幅低于 10% 的股票不进短线信号。"
        "这是为了去掉工商银行这类低波动大盘股，不是按行业名称拉黑。",
    ]
    failures = [item for item in chosen if item.error]
    if failures:
        sample = "；".join(f"{item.code} {item.error}" for item in failures[:8])
        notes.append(f"日线下载失败 {len(failures)} 只。例子：{sample}")
    return notes


def _baostock_basics(cache_dir: Path) -> tuple[list[dict], list[str]]:
    path = cache_dir / "universe" / "baostock_basic.json"
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            rows = payload.get("rows") if isinstance(payload, dict) else None
            if isinstance(rows, list) and rows:
                return rows, [f"Baostock 基础信息使用缓存，{len(rows)} 条股票记录。"]
        except (OSError, json.JSONDecodeError):
            pass
    try:
        import baostock as bs
    except ImportError:
        return [], ["未安装 baostock，退市名单和上市日期缺失。股票池只有新浪当前上市的股票。"]
    login = bs.login()
    if getattr(login, "error_code", "1") != "0":
        return [], [f"Baostock 登录失败：{getattr(login, 'error_msg', login)}。退市名单缺失。"]
    try:
        cursor = bs.query_stock_basic()
        parsed: list[dict] = []
        delisted = 0
        listed = 0
        while cursor.error_code == "0" and cursor.next():
            raw = cursor.get_row_data()
            fields = list(cursor.fields or [])
            record = _parse_basic_row(raw, fields)
            if record is None:
                continue
            parsed.append(record)
            if record["listed"]:
                listed += 1
            else:
                delisted += 1
    finally:
        bs.logout()
    if not parsed:
        return [], ["Baostock query_stock_basic 没有返回股票记录。退市名单缺失。"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"rows": parsed}, ensure_ascii=False),
        encoding="utf-8",
    )
    return parsed, [
        f"Baostock 股票基础信息：纳入 {len(parsed)} 只（上市 {listed}，退市 {delisted}）。"
        "指数和无法识别的代码已丢掉。"
    ]


def _parse_basic_row(raw: list[str], fields: list[str]) -> dict | None:
    data = {name: raw[index] if index < len(raw) else "" for index, name in enumerate(fields)}
    if not data and len(raw) >= 6:
        data = {
            "code": raw[0],
            "code_name": raw[1],
            "ipoDate": raw[2],
            "outDate": raw[3],
            "type": raw[4],
            "status": raw[5],
        }
    kind = str(data.get("type") or "")
    if kind != "1":
        return None
    text = str(data.get("code") or "").strip().lower()
    if "." not in text:
        return None
    prefix, digits = text.split(".", 1)
    if prefix not in {"sh", "sz", "bj"} or not _is_six_digits(digits):
        return None
    board = classify_board(digits)
    if board in {"b_share", "other"}:
        return None
    status = str(data.get("status") or "")
    return {
        "code": digits,
        "name": str(data.get("code_name") or digits),
        "ipo_date": _blank_date(data.get("ipoDate")),
        "out_date": _blank_date(data.get("outDate")),
        "listed": status == "1",
        "board": board,
    }


def _read_catalog(path: Path, today: date) -> list[Instrument] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    fetched = str(payload.get("fetched_at") or "")
    try:
        stamp = datetime.fromisoformat(fetched)
    except ValueError:
        return None
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    age = datetime.now(timezone.utc) - stamp
    if age > timedelta(days=7):
        return None
    rows = payload.get("names")
    if not isinstance(rows, list) or not rows:
        return None
    loaded = [Instrument(**{key: row.get(key) for key in Instrument.__dataclass_fields__}) for row in rows]
    # A catalog fetched today is fine. ``today`` is accepted so callers can
    # force a refresh later by deleting the file; the age check is enough.
    del today
    return loaded


def _write_catalog(path: Path, names: list[Instrument]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "names": [asdict(item) for item in names],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _blank_date(value: object) -> str | None:
    text = str(value or "").strip()
    if not text or text.startswith("0"):
        return None
    try:
        date.fromisoformat(text[:10])
    except ValueError:
        return None
    return text[:10]


def _is_six_digits(value: str) -> bool:
    return len(value) == 6 and value.isdigit()
