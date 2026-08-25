from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator


class ManifestItem(BaseModel):
    id: str
    text: str
    audio_path: Path
    speaker_id: str | None = None
    reference_audio_path: Path | None = None
    language: str = "en"
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("id", "text")
    @classmethod
    def _not_empty(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("field must be non-empty")
        return value.strip()


def load_manifest(path: Path) -> list[ManifestItem]:
    if path.suffix.lower() == ".jsonl":
        return _load_jsonl(path)
    if path.suffix.lower() == ".csv":
        return _load_csv(path)
    raise ValueError(f"Unsupported manifest format: {path.suffix}. Use .jsonl or .csv")


def resolve_manifest_paths(items: list[ManifestItem], manifest_path: Path) -> list[ManifestItem]:
    base_dir = manifest_path.parent.resolve()
    resolved = []
    for item in items:
        update = {"audio_path": _resolve_path(item.audio_path, base_dir)}
        if item.reference_audio_path is not None:
            update["reference_audio_path"] = _resolve_path(item.reference_audio_path, base_dir)
        resolved.append(item.model_copy(update=update))
    return resolved


def validate_manifest_paths(items: list[ManifestItem]) -> list[str]:
    errors: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item.id in seen:
            errors.append(f"{item.id}: duplicate id")
        seen.add(item.id)
        if not item.audio_path.exists():
            errors.append(f"{item.id}: audio_path does not exist: {item.audio_path}")
        if item.reference_audio_path is not None and not item.reference_audio_path.exists():
            errors.append(
                f"{item.id}: reference_audio_path does not exist: {item.reference_audio_path}"
            )
    return errors


def _load_jsonl(path: Path) -> list[ManifestItem]:
    items = []
    for line_no, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            items.append(ManifestItem.model_validate(json.loads(line)))
        except Exception as exc:
            raise ValueError(f"{path}:{line_no}: invalid manifest row: {exc}") from exc
    return items


def _load_csv(path: Path) -> list[ManifestItem]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = []
        for line_no, row in enumerate(reader, start=2):
            try:
                rows.append(ManifestItem.model_validate(_clean_csv_row(row)))
            except Exception as exc:
                raise ValueError(f"{path}:{line_no}: invalid manifest row: {exc}") from exc
        return rows


def _clean_csv_row(row: dict[str, str | None]) -> dict[str, Any]:
    cleaned = {key: value for key, value in row.items() if value not in (None, "")}
    metadata = cleaned.get("metadata")
    if isinstance(metadata, str):
        cleaned["metadata"] = json.loads(metadata)
    return cleaned


def _resolve_path(path: Path, base_dir: Path) -> Path:
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()
