from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from rich.progress import track

from tts_assess.asr import transcribe
from tts_assess.audio import analyze_features, read_audio, resample_mono
from tts_assess.config import AssessmentConfig
from tts_assess.io.manifest import (
    ManifestItem,
    load_manifest,
    resolve_manifest_paths,
    validate_manifest_paths,
)
from tts_assess.metrics import audio_metric_dict, compute_text_metrics
from tts_assess.metrics.optional import SAMPLE_RATE as OPTIONAL_METRICS_SAMPLE_RATE
from tts_assess.metrics.optional import compute_optional_metrics
from tts_assess.normalization import build_normalizer
from tts_assess.reporting.aggregate import summarize
from tts_assess.reporting.compare import build_run_report
from tts_assess.reporting.comparison_html import render_comparison_html
from tts_assess.reporting.thresholds import classify_row
from tts_assess.reporting.writers import write_csv, write_jsonl, write_summary

_MEASUREMENT_CACHE_VERSION = 2


def run_assessment(
    manifest_path: Path,
    output_dir: Path,
    config: AssessmentConfig,
    *,
    fail_on_manifest_errors: bool = True,
    use_cache: bool = True,
    cache_dir: Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    items = resolve_manifest_paths(load_manifest(manifest_path), manifest_path)
    errors = validate_manifest_paths(items)
    if errors and fail_on_manifest_errors:
        raise ValueError("Manifest validation failed:\n" + "\n".join(errors))

    output_dir.mkdir(parents=True, exist_ok=True)
    normalizer = build_normalizer(config.normalization)
    cache = _MeasurementCache(cache_dir or output_dir / ".measure_cache") if use_cache else None

    rows = []
    for item in track(items, description="Assessing audio"):
        rows.append(_assess_item(item, config, normalizer, cache))

    # Store audio paths relative to the output dir when the audio lives under it,
    # so a self-contained run dir (audio + report together) is portable and its
    # report's <audio> players resolve without machine-specific absolute paths.
    for row in rows:
        row["audio_path"] = _relative_audio_path(row.get("audio_path"), output_dir)

    summary = summarize(
        rows,
        confidence_level=config.reporting.confidence_level,
        bootstrap_resamples=config.reporting.bootstrap_resamples,
        group_by_voice=config.reporting.group_by_voice,
    )
    summary["manifest_errors"] = errors
    summary["config"] = config.model_dump()
    if cache is not None:
        summary["cache"] = {"hits": cache.hits, "misses": cache.misses, "dir": str(cache.directory)}
    write_jsonl(output_dir / "results.jsonl", rows)
    write_summary(output_dir / "summary.json", summary)
    if config.reporting.csv:
        write_csv(output_dir / "results.csv", rows)
    if config.reporting.html:
        report_data = build_run_report(
            rows, summary, config, output_dir=output_dir,
            title=config.reporting.title, label=output_dir.name,
        )
        write_summary(output_dir / "report_data.json", report_data)
        (output_dir / "report.html").write_text(
            render_comparison_html(report_data), encoding="utf-8"
        )
    return rows, summary


def _relative_audio_path(audio_path: str | None, output_dir: Path) -> str | None:
    """Return the path relative to output_dir when the audio is inside it, else unchanged."""
    if not audio_path:
        return audio_path
    try:
        return Path(audio_path).resolve().relative_to(Path(output_dir).resolve()).as_posix()
    except (ValueError, OSError):
        return audio_path


class _MeasurementCache:
    """Content-addressed cache of the expensive per-sample measurement.

    The key hashes the decoded audio plus the text and the ASR / normalization /
    optional-metric config, so a re-run reuses ASR transcripts and metrics when
    nothing that affects them changed. Thresholds and reporting are always applied
    fresh, so tuning bands or improving the report needs no recomputation.
    """

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0

    def load(self, key: str) -> dict[str, Any] | None:
        path = self.directory / f"{key}.json"
        if path.exists():
            try:
                measured = json.loads(path.read_text(encoding="utf-8"))
                self.hits += 1
                return measured
            except (OSError, ValueError):
                return None
        return None

    def store(self, key: str, measured: dict[str, Any]) -> None:
        self.misses += 1
        path = self.directory / f"{key}.json"
        path.write_text(json.dumps(measured, default=str, ensure_ascii=False), encoding="utf-8")


def _cache_key(
    audio,
    sample_rate: int,
    channels: int,
    item: ManifestItem,
    config: AssessmentConfig,
) -> str:
    digest = hashlib.sha1()
    digest.update(audio.tobytes())
    reference_digest = None
    if config.optional_metrics.speaker_similarity and item.reference_audio_path is not None:
        try:
            reference_digest = hashlib.sha1(item.reference_audio_path.read_bytes()).hexdigest()
        except OSError as exc:
            reference_digest = f"unreadable:{type(exc).__name__}"
    fingerprint = json.dumps(
        {
            "cache_version": _MEASUREMENT_CACHE_VERSION,
            "text": item.text,
            "language": item.language,
            "sample_rate": sample_rate,
            "channels": channels,
            "reference_audio": reference_digest,
            "asr": config.asr.model_dump(),
            "normalization": config.normalization.model_dump(),
            "optional_metrics": config.optional_metrics.model_dump(),
        },
        sort_keys=True,
    )
    digest.update(fingerprint.encode("utf-8"))
    return digest.hexdigest()


def _assess_item(
    item: ManifestItem,
    config: AssessmentConfig,
    normalizer,
    cache: _MeasurementCache | None,
) -> dict[str, Any]:
    reference_path = str(item.reference_audio_path) if item.reference_audio_path else None
    row: dict[str, Any] = {
        "id": item.id,
        "text": item.text,
        "audio_path": str(item.audio_path),
        "speaker_id": item.speaker_id,
        "reference_audio_path": reference_path,
        "language": item.language,
        "metadata": item.metadata,
    }
    try:
        audio, sample_rate, channels = read_audio(item.audio_path)
    except Exception as exc:
        row.update({"status": "fail", "error_audio": str(exc), "failure_labels": ["fail:audio"]})
        return row

    key = (
        _cache_key(audio, sample_rate, channels, item, config)
        if cache is not None
        else None
    )
    measured = cache.load(key) if key is not None else None
    if measured is None:
        measured = _measure_item(item, config, normalizer, audio, sample_rate, channels)
        if "error_audio" in measured or "error_asr" in measured:
            row.update(measured)
            return row
        if key is not None:
            cache.store(key, measured)

    row.update(measured)
    status, metric_statuses, labels = classify_row(row, config.thresholds)
    row["status"] = status
    row["metric_statuses"] = metric_statuses
    row["failure_labels"] = labels
    return row


def _measure_item(
    item: ManifestItem,
    config: AssessmentConfig,
    normalizer,
    audio,
    sample_rate: int,
    channels: int,
) -> dict[str, Any]:
    """Compute the cacheable measurement (audio + ASR + text + optional metrics)."""
    measured: dict[str, Any] = {}
    try:
        audio_features = analyze_features(audio, sample_rate, channels)
        measured.update(audio_metric_dict(audio_features, item.text))
    except Exception as exc:
        return {"error_audio": str(exc), "status": "fail", "failure_labels": ["fail:audio"]}

    try:
        transcript = transcribe(item.audio_path, config.asr, expected_text=item.text)
        measured.update(
            {
                "transcript": transcript.text,
                "asr_backend": transcript.backend,
                "asr_model": transcript.model,
                "asr_segments": [segment.__dict__ for segment in transcript.segments],
            }
        )
    except Exception as exc:
        return {"error_asr": str(exc), "status": "fail", "failure_labels": ["fail:asr"]}

    normalized_text = normalizer(item.text)
    normalized_transcript = normalizer(measured["transcript"])
    measured["normalized_text"] = normalized_text
    measured["normalized_transcript"] = normalized_transcript
    measured.update(compute_text_metrics(normalized_text, normalized_transcript).to_dict())

    # Reuse the already-decoded audio: resample once to the optional-metric rate
    # and decode the reference (if any) once, instead of re-reading per metric.
    audio_16k = resample_mono(audio, sample_rate, OPTIONAL_METRICS_SAMPLE_RATE)
    reference_16k = None
    if item.reference_audio_path is not None:
        try:
            ref_audio, ref_sample_rate, _ = read_audio(item.reference_audio_path)
            reference_16k = resample_mono(ref_audio, ref_sample_rate, OPTIONAL_METRICS_SAMPLE_RATE)
        except Exception:
            reference_16k = None
    measured.update(compute_optional_metrics(audio_16k, reference_16k, config.optional_metrics))
    return measured
