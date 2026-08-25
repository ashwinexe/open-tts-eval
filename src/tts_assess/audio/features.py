from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal


@dataclass(frozen=True)
class AudioFeatures:
    sample_rate: int
    channels: int
    duration_sec: float
    rms: float
    peak: float
    clipping_ratio: float
    leading_silence_sec: float
    trailing_silence_sec: float
    silence_ratio: float
    tail_click_detected: bool
    tail_click_score: float


def read_audio(path: Path) -> tuple[np.ndarray, int, int]:
    """Decode a file once into (mono float32, sample_rate, channels)."""
    raw, sample_rate = sf.read(path, always_2d=True, dtype="float32")
    channels = raw.shape[1]
    mono = raw.mean(axis=1)
    return mono, sample_rate, channels


def resample_mono(audio: np.ndarray, sample_rate: int, target_sample_rate: int) -> np.ndarray:
    if not target_sample_rate or sample_rate == target_sample_rate:
        return audio
    gcd = np.gcd(sample_rate, target_sample_rate)
    return signal.resample_poly(
        audio, target_sample_rate // gcd, sample_rate // gcd
    ).astype(np.float32)


def load_audio(path: Path, target_sample_rate: int | None = None) -> tuple[np.ndarray, int]:
    mono, sample_rate, _ = read_audio(path)
    if target_sample_rate and sample_rate != target_sample_rate:
        mono = resample_mono(mono, sample_rate, target_sample_rate)
        sample_rate = target_sample_rate
    return mono, sample_rate


def analyze_audio(path: Path) -> AudioFeatures:
    audio, sample_rate, channels = read_audio(path)
    return analyze_features(audio, sample_rate, channels)


def analyze_features(audio: np.ndarray, sample_rate: int, channels: int) -> AudioFeatures:
    duration = float(len(audio) / sample_rate) if sample_rate else 0.0
    abs_audio = np.abs(audio)
    rms = float(np.sqrt(np.mean(np.square(audio)))) if len(audio) else 0.0
    peak = float(abs_audio.max()) if len(audio) else 0.0
    clipping_ratio = float(np.mean(abs_audio >= 0.999)) if len(audio) else 0.0
    silence_mask = _silence_mask(audio)
    leading = _edge_silence_seconds(silence_mask, sample_rate, from_start=True)
    trailing = _edge_silence_seconds(silence_mask, sample_rate, from_start=False)
    silence_ratio = float(np.mean(silence_mask)) if len(silence_mask) else 1.0
    click_score = _tail_click_score(audio, sample_rate)
    return AudioFeatures(
        sample_rate=sample_rate,
        channels=channels,
        duration_sec=duration,
        rms=rms,
        peak=peak,
        clipping_ratio=clipping_ratio,
        leading_silence_sec=leading,
        trailing_silence_sec=trailing,
        silence_ratio=silence_ratio,
        tail_click_detected=click_score >= 4.0,
        tail_click_score=click_score,
    )


def chars_per_second(text: str, duration_sec: float) -> float | None:
    if duration_sec <= 0:
        return None
    return len(text) / duration_sec


def _silence_mask(audio: np.ndarray, frame_size: int = 1024) -> np.ndarray:
    if len(audio) == 0:
        return np.array([], dtype=bool)
    pad = (-len(audio)) % frame_size
    padded = np.pad(audio, (0, pad)) if pad else audio
    frames = padded.reshape(-1, frame_size)
    frame_rms = np.sqrt(np.mean(np.square(frames), axis=1))
    noise_floor = float(np.percentile(frame_rms, 20)) * 1.5
    speech_floor = float(np.percentile(frame_rms, 95)) * 0.1
    threshold = max(min(noise_floor, speech_floor), 1e-4)
    frame_silence = frame_rms <= threshold
    return np.repeat(frame_silence, frame_size)[: len(audio)]


def _edge_silence_seconds(mask: np.ndarray, sample_rate: int, *, from_start: bool) -> float:
    if len(mask) == 0:
        return 0.0
    iterable = mask if from_start else mask[::-1]
    count = 0
    for silent in iterable:
        if not silent:
            break
        count += 1
    return count / sample_rate


def _tail_click_score(audio: np.ndarray, sample_rate: int) -> float:
    if len(audio) < int(sample_rate * 0.25):
        return 0.0
    tail = audio[-max(1, int(sample_rate * 0.03)) :]
    context = audio[-int(sample_rate * 0.23) : -int(sample_rate * 0.03)]
    if len(context) == 0:
        return 0.0
    tail_peak = float(np.max(np.abs(np.diff(tail)))) if len(tail) > 1 else 0.0
    context_rms = float(np.sqrt(np.mean(np.square(context)))) + 1e-8
    return tail_peak / context_rms
