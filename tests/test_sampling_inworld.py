import base64
import json

import pytest

from tts_assess.sampling.providers.base import ProviderError, SynthesisRequest
from tts_assess.sampling.providers.inworld import InworldProvider


class FakeTransport:
    """Records requests and replays a scripted list of (status, body) responses."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, method, url, headers, body, timeout):
        self.calls.append({"method": method, "url": url, "headers": headers, "body": body})
        return self.responses.pop(0)


def _audio_response(audio: bytes):
    payload = {
        "audioContent": base64.b64encode(audio).decode(),
        "usage": {"processedCharactersCount": 3},
    }
    return (200, json.dumps(payload).encode())


def test_synthesize_builds_payload_and_decodes_audio():
    transport = FakeTransport([_audio_response(b"RIFFwav")])
    provider = InworldProvider("secret-key", transport=transport)
    request = SynthesisRequest(
        text="Hi there",
        voice_id="Sarah",
        model_id="inworld-tts-1.5-max",
        language="en-US",
        audio_encoding="WAV",
        sample_rate_hz=24000,
        temperature=0.7,
    )

    result = provider.synthesize(request)

    assert result.audio == b"RIFFwav"
    assert result.usage["processedCharactersCount"] == 3
    call = transport.calls[0]
    assert call["method"] == "POST" and call["url"].endswith("/tts/v1/voice")
    assert call["headers"]["Authorization"] == "Basic secret-key"
    body = json.loads(call["body"])
    assert body["voiceId"] == "Sarah"
    assert body["modelId"] == "inworld-tts-1.5-max"
    assert body["language"] == "en-US"
    assert body["temperature"] == 0.7
    assert body["audioConfig"] == {"audioEncoding": "WAV", "sampleRateHertz": 24000}


def test_retries_on_503_then_succeeds(monkeypatch):
    monkeypatch.setattr("tts_assess.sampling.providers.inworld.time.sleep", lambda *_: None)
    transport = FakeTransport([(503, b"busy"), _audio_response(b"ok")])
    provider = InworldProvider("k", transport=transport, max_retries=3)
    result = provider.synthesize(SynthesisRequest(text="x", voice_id="v", model_id="m"))
    assert result.audio == b"ok"
    assert len(transport.calls) == 2


def test_client_error_raises_provider_error():
    transport = FakeTransport([(400, b'{"error":"unknown voice"}')])
    provider = InworldProvider("k", transport=transport)
    with pytest.raises(ProviderError, match="HTTP 400"):
        provider.synthesize(SynthesisRequest(text="x", voice_id="bad", model_id="m"))


def test_missing_audio_content_raises():
    transport = FakeTransport([(200, json.dumps({"usage": {}}).encode())])
    provider = InworldProvider("k", transport=transport)
    with pytest.raises(ProviderError, match="missing audioContent"):
        provider.synthesize(SynthesisRequest(text="x", voice_id="v", model_id="m"))


def test_list_voices_parses_varied_shapes():
    payload = {
        "voices": [
            {"voiceId": "Sarah", "displayName": "Sarah", "languages": ["en-US"]},
            {"name": "Ashley", "language": "en-GB"},
            {"unrelated": "skip-me"},
        ]
    }
    transport = FakeTransport([(200, json.dumps(payload).encode())])
    voices = InworldProvider("k", transport=transport).list_voices()
    assert [v.voice_id for v in voices] == ["Sarah", "Ashley"]
    assert voices[1].languages == ("en-GB",)


def test_empty_api_key_rejected():
    with pytest.raises(ValueError, match="API key is required"):
        InworldProvider("")
