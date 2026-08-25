from __future__ import annotations

from tts_assess.audio.features import AudioFeatures, chars_per_second


def audio_metric_dict(features: AudioFeatures, text: str) -> dict[str, float | int | bool | None]:
    return {
        "sample_rate": features.sample_rate,
        "channels": features.channels,
        "duration_sec": features.duration_sec,
        "rms": features.rms,
        "peak": features.peak,
        "clipping_ratio": features.clipping_ratio,
        "leading_silence_sec": features.leading_silence_sec,
        "trailing_silence_sec": features.trailing_silence_sec,
        "silence_ratio": features.silence_ratio,
        "tail_click_detected": features.tail_click_detected,
        "tail_click_score": features.tail_click_score,
        "chars_per_second": chars_per_second(text, features.duration_sec),
    }
