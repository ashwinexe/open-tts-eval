from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from tts_assess.config import ASRConfig


@dataclass(frozen=True)
class ASRSegment:
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class ASRTranscript:
    text: str
    segments: list[ASRSegment]
    backend: str
    model: str


def transcribe(path: Path, config: ASRConfig, *, expected_text: str | None = None) -> ASRTranscript:
    if config.backend == "mock":
        text = expected_text or ""
        return ASRTranscript(
            text=text,
            segments=[ASRSegment(start=0.0, end=0.0, text=text)] if text else [],
            backend="mock",
            model="mock",
        )
    return _transcribe_faster_whisper(path, config)


@lru_cache(maxsize=4)
def _load_whisper_model(model: str, device: str, compute_type: str):
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError(
            "faster-whisper is not installed. Install with `pip install .[asr]` or use "
            "`asr.backend: mock` for tests."
        ) from exc
    return WhisperModel(model, device=device, compute_type=compute_type)


def _transcribe_faster_whisper(path: Path, config: ASRConfig) -> ASRTranscript:
    # Cached so the model loads once per (model, device, compute_type) instead of
    # being re-instantiated for every manifest row.
    model = _load_whisper_model(config.model, config.device, config.compute_type)
    segments_iter, _info = model.transcribe(
        str(path),
        language=config.language,
        beam_size=5,
        vad_filter=True,
    )
    segments = [
        ASRSegment(start=float(segment.start), end=float(segment.end), text=segment.text.strip())
        for segment in segments_iter
    ]
    return ASRTranscript(
        text=" ".join(segment.text for segment in segments).strip(),
        segments=segments,
        backend="faster-whisper",
        model=config.model,
    )
