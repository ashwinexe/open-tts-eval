import io
import json
import wave

import pytest

from tts_assess.config import AssessmentConfig
from tts_assess.io.manifest import load_manifest, resolve_manifest_paths, validate_manifest_paths
from tts_assess.pipeline import run_assessment
from tts_assess.sampling.providers.base import SynthesisResult, TTSProvider, Voice
from tts_assess.sampling.sampler import SamplingConfig, resolve_voices, run_sampling


def _tiny_wav(seconds: float = 0.2, sample_rate: int = 24000) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "w") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        frame = int(0.1 * 32767).to_bytes(2, "little", signed=True)
        handle.writeframes(frame * int(sample_rate * seconds))
    return buffer.getvalue()


class FakeProvider(TTSProvider):
    name = "fake"
    default_model = "fake-1"

    def __init__(self):
        self.requests = []

    def synthesize(self, request):
        self.requests.append(request)
        return SynthesisResult(
            audio=_tiny_wav(sample_rate=request.sample_rate_hz),
            audio_encoding=request.audio_encoding,
            sample_rate_hz=request.sample_rate_hz,
        )

    def list_voices(self):
        return [Voice("v1"), Voice("v2"), Voice("v3")]


def test_resolve_voices():
    provider = FakeProvider()
    assert resolve_voices(provider, ["a", "b"], None) == ["a", "b"]
    assert resolve_voices(provider, None, 2) == ["v1", "v2"]
    with pytest.raises(ValueError, match="--voice"):
        resolve_voices(provider, None, None)


def test_sampling_writes_audio_and_valid_manifest(tmp_path):
    dataset = tmp_path / "texts.txt"
    dataset.write_text("Hello world.\nThe quick brown fox.\n")
    provider = FakeProvider()

    summary = run_sampling(
        dataset,
        tmp_path / "out",
        provider,
        ["m1"],
        voices=["nova", "atlas"],
        config=SamplingConfig(concurrency=2),
    )

    # One run dir per model; 2 texts x 2 voices = 4 clips.
    run = summary["runs"][0]
    assert run["samples"] == 4 and run["errors"] == 0
    run_dir = tmp_path / "out" / "fake-m1"
    assert len(list((run_dir / "audio").glob("*.wav"))) == 4

    # Manifest loads and validates under the evaluator's own loader.
    manifest = run_dir / "manifest.jsonl"
    items = resolve_manifest_paths(load_manifest(manifest), manifest)
    assert validate_manifest_paths(items) == []
    assert {item.speaker_id for item in items} == {"nova", "atlas"}


def test_sampling_cache_requires_matching_request_fingerprint_and_nonempty_audio(tmp_path):
    dataset = tmp_path / "texts.txt"
    dataset.write_text("Hello world.\n")
    provider = FakeProvider()
    output = tmp_path / "out"
    config = SamplingConfig(temperature=0.2)

    run_sampling(dataset, output, provider, ["m1"], voices=["nova"], config=config)
    assert len(provider.requests) == 1

    provider.requests.clear()
    run_sampling(dataset, output, provider, ["m1"], voices=["nova"], config=config)
    assert provider.requests == []

    provider.requests.clear()
    changed = SamplingConfig(temperature=0.7)
    run_sampling(dataset, output, provider, ["m1"], voices=["nova"], config=changed)
    assert len(provider.requests) == 1
    manifest = output / "fake-m1" / "manifest.jsonl"
    row = json.loads(manifest.read_text().strip())
    assert row["metadata"]["temperature"] == 0.7
    assert row["metadata"]["sampling_fingerprint"]

    provider.requests.clear()
    dataset.write_text("Changed text, same generated id.\n")
    run_sampling(dataset, output, provider, ["m1"], voices=["nova"], config=changed)
    assert len(provider.requests) == 1

    provider.requests.clear()
    audio_path = next((output / "fake-m1" / "audio").glob("*.wav"))
    audio_path.write_bytes(b"")
    run_sampling(dataset, output, provider, ["m1"], voices=["nova"], config=changed)
    assert len(provider.requests) == 1


def test_sampled_manifest_runs_through_evaluator(tmp_path):
    dataset = tmp_path / "texts.txt"
    dataset.write_text("Hello world.\nSecond utterance here.\n")
    run_sampling(
        dataset,
        tmp_path / "out",
        FakeProvider(),
        ["m1"],
        voices=["nova", "atlas"],
    )
    manifest = tmp_path / "out" / "fake-m1" / "manifest.jsonl"

    config = AssessmentConfig.default()
    config.asr.backend = "mock"
    rows, report = run_assessment(manifest, tmp_path / "eval", config)

    assert report["sample_count"] == 4
    assert set(report["by_voice"]) == {"nova", "atlas"}
    assert (tmp_path / "eval" / "report.html").exists()


def test_synthesis_errors_are_recorded_not_raised(tmp_path):
    class FlakyProvider(FakeProvider):
        def synthesize(self, request):
            if request.text.startswith("bad"):
                raise RuntimeError("boom")
            return super().synthesize(request)

    dataset = tmp_path / "texts.txt"
    dataset.write_text("good one\nbad two\n")
    summary = run_sampling(dataset, tmp_path / "out", FlakyProvider(), ["m1"], voices=["v"])

    run = summary["runs"][0]
    assert run["samples"] == 1
    assert run["errors"] == 1
