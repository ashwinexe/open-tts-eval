import wave
from pathlib import Path

from tts_assess.config import AssessmentConfig
from tts_assess.pipeline import run_assessment


def test_pipeline_with_mock_asr_writes_artifacts(tmp_path: Path):
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    audio_path = audio_dir / "sample.wav"
    _write_tone(audio_path)
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text('{"id":"sample","text":"Hello world.","audio_path":"audio/sample.wav"}\n')
    config = AssessmentConfig.default()
    config.asr.backend = "mock"
    config.asr.model = "mock"
    config.optional_metrics.vowel_prolongation = True
    config.optional_metrics.voice_lens = True

    rows, summary = run_assessment(manifest, tmp_path / "out", config)

    assert rows[0]["status"] in {"pass", "warn", "fail"}
    assert rows[0]["wer"] == 0
    assert summary["sample_count"] == 1
    assert (tmp_path / "out" / "results.jsonl").exists()
    assert (tmp_path / "out" / "summary.json").exists()
    assert (tmp_path / "out" / "results.csv").exists()
    assert (tmp_path / "out" / "report.html").exists()
    assert (tmp_path / "out" / "report_data.json").exists()


def _write_tone(path: Path):
    sample_rate = 16000
    samples = [int(0.1 * 32767) for _ in range(sample_rate)]
    with wave.open(str(path), "w") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        f.writeframes(b"".join(sample.to_bytes(2, "little", signed=True) for sample in samples))
