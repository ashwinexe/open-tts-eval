import numpy as np

from tts_assess.config import OptionalMetricsConfig
from tts_assess.metrics.optional import SAMPLE_RATE, compute_optional_metrics


def _tone(seconds: float = 1.0) -> np.ndarray:
    t = np.linspace(0, seconds, int(SAMPLE_RATE * seconds), endpoint=False)
    return (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def test_lightweight_metrics_run_on_decoded_array():
    options = OptionalMetricsConfig(vowel_prolongation=True, voice_lens=True)
    metrics = compute_optional_metrics(_tone(), None, options)
    assert "vowel_prolongation_score" in metrics
    assert "vowel_prolongation_detected" in metrics
    assert metrics["expressiveness_proxy"] >= 0.0
    assert metrics["arousal_proxy"] >= 0.0


def test_speaker_similarity_reports_missing_reference():
    options = OptionalMetricsConfig(speaker_similarity=True)
    metrics = compute_optional_metrics(_tone(), None, options)
    assert metrics["speaker_similarity_error"] == "reference_audio_path_missing"


def test_no_options_returns_empty():
    assert compute_optional_metrics(_tone(), None, OptionalMetricsConfig()) == {}
