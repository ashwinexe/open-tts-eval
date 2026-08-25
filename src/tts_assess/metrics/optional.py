from __future__ import annotations

import importlib.util
from functools import lru_cache

import numpy as np

from tts_assess.config import OptionalMetricsConfig

# Optional metrics operate on audio already decoded to mono at this rate.
SAMPLE_RATE = 16000


def compute_optional_metrics(
    audio: np.ndarray,
    reference_audio: np.ndarray | None,
    options: OptionalMetricsConfig,
) -> dict[str, float | bool | str | None]:
    metrics: dict[str, float | bool | str | None] = {}
    if options.vowel_prolongation:
        metrics.update(_vowel_prolongation(audio))
    if options.voice_lens:
        metrics.update(_voice_lens_proxy(audio))
    if options.nisqa_v2:
        metrics.update(_nisqa_v2(audio))
    if options.speaker_similarity:
        metrics.update(_speaker_similarity(audio, reference_audio))
    if options.open_aligner:
        metrics.update(_open_aligner_status())
    return metrics


def _vowel_prolongation(audio: np.ndarray) -> dict[str, float | bool]:
    frame = int(0.05 * SAMPLE_RATE)
    hop = int(0.025 * SAMPLE_RATE)
    if len(audio) < frame:
        return {"vowel_prolongation_score": 0.0, "vowel_prolongation_detected": False}
    energies = []
    centroids = []
    for start in range(0, len(audio) - frame + 1, hop):
        window = audio[start : start + frame]
        spectrum = np.abs(np.fft.rfft(window * np.hanning(len(window))))
        freqs = np.fft.rfftfreq(len(window), 1 / SAMPLE_RATE)
        energy = float(np.sqrt(np.mean(np.square(window))))
        centroid = float(np.sum(freqs * spectrum) / (np.sum(spectrum) + 1e-8))
        energies.append(energy)
        centroids.append(centroid)
    energies_arr = np.asarray(energies)
    centroids_arr = np.asarray(centroids)
    voiced = (energies_arr > max(np.percentile(energies_arr, 60), 1e-4)) & (centroids_arr < 2200)
    longest = _longest_true_run(voiced) * hop / SAMPLE_RATE
    return {
        "vowel_prolongation_score": float(longest),
        "vowel_prolongation_detected": bool(longest >= 0.8),
    }


def _voice_lens_proxy(audio: np.ndarray) -> dict[str, float]:
    if len(audio) == 0:
        return {"expressiveness_proxy": 0.0, "arousal_proxy": 0.0}
    frame = int(0.04 * SAMPLE_RATE)
    hop = int(0.02 * SAMPLE_RATE)
    rms = []
    zcr = []
    for start in range(0, max(1, len(audio) - frame + 1), hop):
        window = audio[start : start + frame]
        if len(window) == 0:
            continue
        rms.append(float(np.sqrt(np.mean(np.square(window)))))
        zcr.append(float(np.mean(np.abs(np.diff(np.signbit(window).astype(np.int8))))))
    rms_arr = np.asarray(rms) if rms else np.asarray([0.0])
    zcr_arr = np.asarray(zcr) if zcr else np.asarray([0.0])
    return {
        "expressiveness_proxy": float(np.std(rms_arr) / (np.mean(rms_arr) + 1e-8)),
        "arousal_proxy": float(np.mean(rms_arr) * 0.7 + np.mean(zcr_arr) * 0.3),
    }


@lru_cache(maxsize=1)
def _nisqa_metric():
    from torchmetrics.audio.nisqa import NonIntrusiveSpeechQualityAssessment

    return NonIntrusiveSpeechQualityAssessment(SAMPLE_RATE)


def _nisqa_v2(audio: np.ndarray) -> dict[str, float | str]:
    try:
        import torch
        from torchmetrics.audio.nisqa import NonIntrusiveSpeechQualityAssessment  # noqa: F401
    except Exception as exc:
        return {"nisqa_error": f"not_available: {exc}"}
    try:
        metric = _nisqa_metric()
        scores = metric(torch.tensor(audio).unsqueeze(0))
        names = [
            "nisqa_mos",
            "nisqa_noisiness",
            "nisqa_discontinuity",
            "nisqa_coloration",
            "nisqa_loudness",
        ]
        return {
            name: float(value)
            for name, value in zip(names, scores.squeeze().tolist(), strict=False)
        }
    except Exception as exc:
        return {"nisqa_error": str(exc)}


@lru_cache(maxsize=1)
def _speaker_classifier():
    from speechbrain.inference.speaker import EncoderClassifier

    return EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")


def _speaker_similarity(
    audio: np.ndarray, reference_audio: np.ndarray | None
) -> dict[str, float | str]:
    if reference_audio is None:
        return {"speaker_similarity_error": "reference_audio_path_missing"}
    try:
        import torch
        from speechbrain.inference.speaker import EncoderClassifier  # noqa: F401
    except Exception as exc:
        return {"speaker_similarity_error": f"not_available: {exc}"}
    try:
        classifier = _speaker_classifier()
        emb = classifier.encode_batch(torch.tensor(audio).unsqueeze(0))
        ref_emb = classifier.encode_batch(torch.tensor(reference_audio).unsqueeze(0))
        similarity = torch.nn.functional.cosine_similarity(emb.squeeze(), ref_emb.squeeze(), dim=0)
        return {"speaker_similarity": float(similarity)}
    except Exception as exc:
        return {"speaker_similarity_error": str(exc)}


def _open_aligner_status() -> dict[str, str]:
    if importlib.util.find_spec("stable_whisper") is not None:
        return {"open_aligner_backend": "stable-ts", "open_aligner_status": "available"}
    return {"open_aligner_backend": "stable-ts", "open_aligner_status": "not_installed"}


def _longest_true_run(values: np.ndarray) -> int:
    longest = 0
    current = 0
    for value in values:
        if value:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest
