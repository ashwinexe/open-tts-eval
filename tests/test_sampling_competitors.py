import io
import json
import wave

import pytest

from tts_assess.sampling.providers import build_provider
from tts_assess.sampling.providers.base import ProviderError, SynthesisRequest
from tts_assess.sampling.providers.elevenlabs import ElevenLabsProvider
from tts_assess.sampling.providers.hume import HumeProvider


class FakeTransport:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, method, url, headers, body, timeout):
        self.calls.append({"method": method, "url": url, "headers": headers, "body": body})
        return self.responses.pop(0)


def _tiny_wav() -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(24000)
        w.writeframes(b"\x00\x01" * 2400)
    return buf.getvalue()


REQ = SynthesisRequest(
    text="Hi.", voice_id="V1", model_id="m1", language="en-US", sample_rate_hz=24000
)


def test_competitors_registered():
    for name in ("elevenlabs", "hume"):
        assert build_provider(name, "k").name == name


def test_elevenlabs_synthesize_and_voices():
    audio = _tiny_wav()
    voices_json = json.dumps(
        {"voices": [{"voice_id": "abc", "name": "Rachel", "labels": {"language": "en"}}]}
    ).encode()
    t = FakeTransport([(200, audio), (200, voices_json)])
    p = ElevenLabsProvider("k", transport=t)
    result = p.synthesize(REQ)
    assert result.audio == audio and result.audio_encoding == "WAV"
    assert "/text-to-speech/V1?output_format=wav_44100" in t.calls[0]["url"]
    assert json.loads(t.calls[0]["body"])["model_id"] == "m1"
    assert t.calls[0]["headers"]["xi-api-key"] == "k"
    voices = p.list_voices()
    assert voices[0].voice_id == "abc" and voices[0].name == "Rachel"


def test_hume_payload_and_wav():
    audio = _tiny_wav()
    t = FakeTransport([(200, audio)])
    p = HumeProvider("k", transport=t)
    req = SynthesisRequest(text="Hi.", voice_id="Ava", model_id="", sample_rate_hz=24000)
    result = p.synthesize(req)
    assert result.audio == audio
    body = json.loads(t.calls[0]["body"])
    assert body["utterances"][0]["voice"] == {"name": "Ava", "provider": "HUME_AI"}
    assert body["format"] == {"type": "wav"}
    assert t.calls[0]["headers"]["X-Hume-Api-Key"] == "k"


def test_http_error_raises_provider_error():
    p = ElevenLabsProvider("k", transport=FakeTransport([(401, b'{"detail":"bad key"}')]))
    with pytest.raises(ProviderError, match="HTTP 401"):
        p.synthesize(REQ)
