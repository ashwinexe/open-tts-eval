from __future__ import annotations

import hashlib
import json
import random
import re
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tts_assess.io.manifest import ManifestItem
from tts_assess.sampling.datasets import TextItem, load_texts
from tts_assess.sampling.providers.base import (
    SynthesisRequest,
    SynthesisResult,
    TTSProvider,
    file_extension,
)

ProgressCallback = Callable[[dict[str, Any]], None]
_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass
class SamplingConfig:
    language: str | None = "en-US"
    audio_encoding: str = "WAV"
    sample_rate_hz: int = 24000
    speaking_rate: float | None = None
    temperature: float | None = None
    concurrency: int = 4
    overwrite: bool = False


def resolve_voices(
    provider: TTSProvider,
    voices: Sequence[str] | None,
    num_voices: int | None,
    *,
    shuffle: bool = False,
    seed: int = 0,
) -> list[str]:
    if voices:
        return list(voices)
    if num_voices:
        catalog = provider.list_voices()
        if not catalog:
            raise ValueError(f"provider {provider.name!r} returned no prebuilt voices")
        voice_ids = [voice.voice_id for voice in catalog]
        if shuffle:
            # Seeded so the random-but-diverse selection is reproducible (and cached).
            return random.Random(seed).sample(voice_ids, min(num_voices, len(voice_ids)))
        return voice_ids[:num_voices]
    raise ValueError("specify one or more --voice values or --num-voices")


def run_sampling(
    dataset_path: Path,
    output_dir: Path,
    provider: TTSProvider,
    models: Sequence[str | tuple[str, str]] | None,
    *,
    voices: Sequence[str] | None = None,
    num_voices: int | None = None,
    shuffle_voices: bool = False,
    seed: int = 0,
    config: SamplingConfig | None = None,
    limit: int | None = None,
    progress_cb: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Synthesize dataset x voices x models into eval-ready run directories.

    Produces one ``<provider>-<label>/`` directory per model, each containing an
    ``audio/`` folder and a ``manifest.jsonl`` compatible with ``tts-assess run``.
    Voices are recorded as ``speaker_id`` so the evaluator groups by voice.

    A model may be a plain id string, or an ``(api_id, label)`` pair when the
    request should use one id but the run should be recorded under another name
    (e.g. call ``inworld-tts-2`` but present it as ``inworld-tts-2-preview``).
    """
    config = config or SamplingConfig()
    texts = load_texts(dataset_path, limit=limit)
    voice_ids = resolve_voices(provider, voices, num_voices, shuffle=shuffle_voices, seed=seed)
    specs = _model_specs(models, provider.default_model)
    extension = file_extension(provider.output_encoding(config.audio_encoding))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = []
    for api_id, label in specs:
        run = _sample_model(
            provider, api_id, label, texts, voice_ids, output_dir, extension, config, progress_cb
        )
        runs.append(run)

    summary = {
        "provider": provider.name,
        "models": [label for _api, label in specs],
        "voices": voice_ids,
        "text_count": len(texts),
        "audio_encoding": config.audio_encoding,
        "sample_rate_hz": config.sample_rate_hz,
        "runs": runs,
    }
    _write_json(output_dir / "sampling_summary.json", summary)
    return summary


def _model_specs(
    models: Sequence[str | tuple[str, str]] | None, default_model: str
) -> list[tuple[str, str]]:
    if not models:
        return [(default_model, default_model)]
    specs = []
    for model in models:
        if isinstance(model, str):
            specs.append((model, model))
        else:
            api_id, label = model
            specs.append((api_id, label))
    return specs


def _sample_model(
    provider: TTSProvider,
    model_id: str,
    label: str,
    texts: list[TextItem],
    voice_ids: list[str],
    output_dir: Path,
    extension: str,
    config: SamplingConfig,
    progress_cb: ProgressCallback | None,
) -> dict[str, Any]:
    run_dir = output_dir / _safe(f"{provider.name}-{label}")
    audio_dir = run_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "manifest.jsonl"
    cached_rows = _load_cached_rows(manifest_path)
    tasks = [(text, voice) for text in texts for voice in voice_ids]

    def worker(task: tuple[TextItem, str]) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        text, voice = task
        sample_id = f"{voice}__{text.id}"
        filename = f"{_safe(sample_id)}.{extension}"
        audio_path = audio_dir / filename
        request = SynthesisRequest(
            text=text.text,
            voice_id=voice,
            model_id=model_id,
            language=text.language or config.language,
            audio_encoding=config.audio_encoding,
            sample_rate_hz=config.sample_rate_hz,
            speaking_rate=config.speaking_rate,
            temperature=config.temperature,
        )
        fingerprint = _sampling_fingerprint(provider, label, text, request)
        cached = cached_rows.get(sample_id)
        if (
            not config.overwrite
            and _cached_audio_matches(cached, audio_path, filename, fingerprint)
        ):
            return cached, None
        try:
            result = provider.synthesize(request)
            if not result.audio:
                raise ValueError("provider returned empty audio")
        except Exception as exc:  # noqa: BLE001 - record and continue over the batch
            return None, {"id": sample_id, "voice": voice, "model": label, "error": str(exc)}
        audio_path.write_bytes(result.audio)
        return (
            _manifest_row(
                text, voice, label, provider, filename, request, result, fingerprint
            ),
            None,
        )

    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, config.concurrency)) as executor:
        for row, error in executor.map(worker, tasks):
            if row is not None:
                rows.append(row)
            if error is not None:
                errors.append(error)
            if progress_cb is not None:
                progress_cb({"model": label, "ok": row is not None, "error": error})

    _write_manifest(manifest_path, rows)
    meta = {
        "provider": provider.name,
        "model": label,
        "voices": voice_ids,
        "sample_count": len(rows),
        "error_count": len(errors),
        "errors": errors,
        "audio_encoding": config.audio_encoding,
        "sample_rate_hz": config.sample_rate_hz,
    }
    _write_json(run_dir / "sampling_meta.json", meta)
    return {
        "model": label,
        "run_dir": str(run_dir),
        "manifest": str(run_dir / "manifest.jsonl"),
        "samples": len(rows),
        "errors": len(errors),
    }


def _manifest_row(
    text: TextItem,
    voice: str,
    model: str,
    provider: TTSProvider,
    audio_filename: str,
    request: SynthesisRequest,
    result: SynthesisResult,
    fingerprint: str,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "provider": provider.name,
        "model": model,
        "voice": voice,
        "api_model": request.model_id,
        "audio_encoding": result.audio_encoding,
        "sample_rate_hz": result.sample_rate_hz,
        "requested_audio_encoding": request.audio_encoding,
        "requested_sample_rate_hz": request.sample_rate_hz,
        "speaking_rate": request.speaking_rate,
        "temperature": request.temperature,
        "source_text_id": text.id,
        "sampling_fingerprint": fingerprint,
    }
    if result.usage:
        metadata["provider_usage"] = result.usage
    return {
        "id": f"{voice}__{text.id}",
        "text": text.text,
        "audio_path": f"audio/{audio_filename}",
        "speaker_id": voice,
        "language": request.language or "en",
        "metadata": metadata,
    }


def _sampling_fingerprint(
    provider: TTSProvider,
    label: str,
    text: TextItem,
    request: SynthesisRequest,
) -> str:
    payload = {
        "provider": provider.name,
        "label": label,
        "text_id": text.id,
        "text": request.text,
        "voice_id": request.voice_id,
        "model_id": request.model_id,
        "language": request.language,
        "audio_encoding": request.audio_encoding,
        "sample_rate_hz": request.sample_rate_hz,
        "speaking_rate": request.speaking_rate,
        "temperature": request.temperature,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _load_cached_rows(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    except (OSError, ValueError, TypeError):
        return {}
    return {
        row["id"]: row
        for row in rows
        if isinstance(row, dict) and isinstance(row.get("id"), str)
    }


def _cached_audio_matches(
    row: dict[str, Any] | None,
    audio_path: Path,
    filename: str,
    fingerprint: str,
) -> bool:
    if row is None or row.get("audio_path") != f"audio/{filename}":
        return False
    metadata = row.get("metadata")
    if not isinstance(metadata, dict) or metadata.get("sampling_fingerprint") != fingerprint:
        return False
    try:
        return audio_path.is_file() and audio_path.stat().st_size > 0
    except OSError:
        return False


def _write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            # Validate against the evaluator's schema so the manifest is
            # guaranteed loadable by `tts-assess run`.
            ManifestItem.model_validate(row)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _safe(value: str) -> str:
    cleaned = _UNSAFE.sub("_", value).strip("_")
    return cleaned or "item"
