import wave
from pathlib import Path

from tts_assess.audio import analyze_audio


def test_audio_features_for_synthetic_wav(tmp_path: Path):
    path = tmp_path / "tone.wav"
    _write_tone(path)

    features = analyze_audio(path)

    assert features.sample_rate == 16000
    assert features.channels == 1
    assert 0.49 <= features.duration_sec <= 0.51
    assert features.peak > 0
    assert features.clipping_ratio == 0


def _write_tone(path: Path):
    sample_rate = 16000
    samples = [int(0.1 * 32767) for _ in range(sample_rate // 2)]
    with wave.open(str(path), "w") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        f.writeframes(b"".join(sample.to_bytes(2, "little", signed=True) for sample in samples))
