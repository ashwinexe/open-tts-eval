from __future__ import annotations

import json

from tts_assess.sampling.providers import _http
from tts_assess.sampling.providers.base import (
    ProviderError,
    SynthesisRequest,
    SynthesisResult,
    TTSProvider,
    Voice,
)


class HumeProvider(TTSProvider):
    """Hume Octave TTS. Docs: https://dev.hume.ai/docs/text-to-speech-tts/overview"""

    name = "hume"
    default_model = "octave-2"  # latest; "octave-1" for the older model
    forced_encoding = "WAV"
    base_url = "https://api.hume.ai"
    voice_provider = "HUME_AI"

    @staticmethod
    def _version(model_id: str) -> str | None:
        """Map a friendly model label to Hume's ``version`` field (None omits it)."""
        model = (model_id or "").lower()
        if model in ("2", "octave-2", "octave2"):
            return "2"
        if model in ("", "1", "octave-1", "octave1", "octave"):
            return None
        return model_id

    def __init__(
        self, api_key: str, *, timeout: float = 120.0, max_retries: int = 3, transport=None
    ):
        if not api_key:
            raise ValueError("Hume API key is required")
        self._key = api_key
        self._timeout = timeout
        self._max_retries = max_retries
        self._transport = transport or _http.urllib_transport

    def _headers(self) -> dict[str, str]:
        return {"X-Hume-Api-Key": self._key, "Content-Type": "application/json"}

    def synthesize(self, request: SynthesisRequest) -> SynthesisResult:
        voice = {"name": request.voice_id, "provider": self.voice_provider}
        payload = {
            "utterances": [{"text": request.text, "voice": voice}],
            "format": {"type": "wav"},
        }
        version = self._version(request.model_id)
        if version:
            payload["version"] = version
        status, data = _http.call(
            self._transport,
            "POST",
            f"{self.base_url}/v0/tts/file",
            self._headers(),
            body=json.dumps(payload).encode("utf-8"),
            timeout=self._timeout,
            max_retries=self._max_retries,
        )
        if status != 200:
            raise ProviderError(f"Hume synthesize failed: HTTP {status}: {_http.snippet(data)}")
        return SynthesisResult(
            audio=data, audio_encoding="WAV", sample_rate_hz=request.sample_rate_hz, usage={}
        )

    def list_voices(self) -> list[Voice]:
        url = f"{self.base_url}/v0/tts/voices?provider={self.voice_provider}&page_size=100"
        status, data = _http.call(
            self._transport, "GET", url, self._headers(), timeout=self._timeout
        )
        if status != 200:
            raise ProviderError(f"Hume voices HTTP {status}: {_http.snippet(data)}")
        body = json.loads(data)
        entries = body.get("voices_page") or body.get("voices") or []
        voices = []
        for entry in entries:
            voice_id = entry.get("name") or entry.get("id")
            if voice_id:
                voices.append(Voice(voice_id=voice_id, name=entry.get("name"), metadata=entry))
        return voices
