"""Small tabular cache. Parquet when pyarrow is installed, JSON otherwise."""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def write_records(path: Path, records: list[dict]) -> None:
    """Write rows to ``path`` (parquet) or a sibling JSON file if pyarrow is missing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        logger.warning("未安装 pyarrow，%s 改写为 JSON。", path.name)
        _write_json(path.with_suffix(".json"), records)
        return
    table = pa.Table.from_pylist(records)
    pq.write_table(table, path)


def read_records(path: Path) -> list[dict]:
    """Read parquet, then the JSON fallback. Missing or unreadable files are empty."""
    if path.exists():
        try:
            import pyarrow.parquet as pq
        except ImportError:
            logger.warning("未安装 pyarrow，无法读取 %s。", path.name)
        else:
            try:
                return pq.read_table(path).to_pylist()
            except Exception as exc:  # noqa: BLE001 - corrupt cache should not stop the run
                logger.warning("读取 %s 失败：%s", path.name, exc)
    fallback = path.with_suffix(".json")
    if not fallback.exists():
        return []
    try:
        payload = json.loads(fallback.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("读取 %s 失败：%s", fallback.name, exc)
        return []
    return payload if isinstance(payload, list) else []


def _write_json(path: Path, records: list[dict]) -> None:
    path.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
