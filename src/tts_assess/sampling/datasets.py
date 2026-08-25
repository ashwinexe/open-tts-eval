from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TextItem:
    id: str
    text: str
    language: str | None = None


def load_texts(path: Path, *, limit: int | None = None) -> list[TextItem]:
    """Load utterances from a .txt (one per line), .jsonl, or .csv dataset.

    JSONL/CSV rows use a ``text`` field (required) and optional ``id`` and
    ``language`` fields. Plain-text lines are auto-numbered.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".txt":
        items = _load_txt(path)
    elif suffix == ".jsonl":
        items = _load_jsonl(path)
    elif suffix == ".json":
        items = _load_json(path)
    elif suffix == ".csv":
        items = _load_csv(path)
    else:
        raise ValueError(f"unsupported dataset format: {suffix}. Use .txt, .json, .jsonl, or .csv")
    if limit is not None:
        items = items[:limit]
    if not items:
        raise ValueError(f"no texts loaded from {path}")
    return items


def _load_txt(path: Path) -> list[TextItem]:
    items = []
    index = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        items.append(TextItem(id=f"utt_{index:04d}", text=text))
        index += 1
    return items


def _load_jsonl(path: Path) -> list[TextItem]:
    items = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
        items.append(_item_from_record(record, len(items), f"{path}:{line_no}"))
    return items


def _load_json(path: Path) -> list[TextItem]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a JSON array of objects or strings")
    items = []
    for index, record in enumerate(data):
        if isinstance(record, str):
            record = {"text": record}
        if not isinstance(record, dict):
            raise ValueError(f"{path}[{index}]: expected an object or string")
        items.append(_item_from_record(record, len(items), f"{path}[{index}]"))
    return items


def _load_csv(path: Path) -> list[TextItem]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        items = []
        for line_no, record in enumerate(reader, start=2):
            items.append(_item_from_record(record, len(items), f"{path}:{line_no}"))
        return items


def _item_from_record(record: dict, index: int, where: str) -> TextItem:
    text = (record.get("text") or "").strip()
    if not text:
        raise ValueError(f"{where}: row is missing a non-empty 'text' field")
    raw_id = record.get("id")
    item_id = str(raw_id).strip() if raw_id not in (None, "") else f"utt_{index:04d}"
    language = record.get("language") or None
    return TextItem(id=item_id, text=text, language=language)
