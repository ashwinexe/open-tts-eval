from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

# Map an Inworld/GCP-style audio encoding to a file extension. Container formats
# (WAV/FLAC/OGG) are what the evaluation suite can decode; raw PCM variants are
# written with a .wav name only when the provider wraps them in a container.
_EXTENSIONS = {
    "WAV": "wav",
    "LINEAR16": "wav",
    "PCM": "wav",
    "MP3": "mp3",
    "FLAC": "flac",
    "OGG_OPUS": "ogg",
    "ALAW": "wav",
    "MULAW": "wav",
}


class ProviderError(RuntimeError):
    """Raised when a TTS provider request fails."""


@dataclass(frozen=True)
class SynthesisRequest:
    text: str
    voice_id: str
    model_id: str
    language: str | None = None
    audio_encoding: str = "WAV"
    sample_rate_hz: int = 24000
    speaking_rate: float | None = None
    temperature: float | None = None


@dataclass(frozen=True)
class SynthesisResult:
    audio: bytes
    audio_encoding: str
    sample_rate_hz: int
    usage: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Voice:
    voice_id: str
    name: str | None = None
    languages: tuple[str, ...] = ()
    metadata: dict = field(default_factory=dict)


class TTSProvider(ABC):
    """A pluggable text-to-speech backend.

    Add a new provider by subclassing this, setting ``name``/``default_model``,
    and registering it in ``providers/__init__.py``.
    """

    name: str = "base"
    default_model: str = ""
    # When set, the provider always emits this audio encoding regardless of the
    # requested one (competitor APIs each have a preferred decodable format).
    forced_encoding: str | None = None

    @abstractmethod
    def synthesize(self, request: SynthesisRequest) -> SynthesisResult:
        """Synthesize one utterance and return decoded audio bytes."""

    @abstractmethod
    def list_voices(self) -> list[Voice]:
        """Return the provider's catalog of prebuilt voices."""

    def output_encoding(self, requested: str) -> str:
        """The encoding this provider will actually produce for a request."""
        return self.forced_encoding or requested


def decode_audio_payload(encoded: str) -> bytes:
    """Decode an audio string payload, trying hex first then base64."""
    import base64
    import binascii

    text = encoded.strip()
    if not text:
        return b""
    try:
        return bytes.fromhex(text)
    except ValueError:
        pass
    try:
        return base64.b64decode(text, validate=False)
    except (ValueError, binascii.Error) as exc:
        raise ProviderError("could not decode audio payload from provider response") from exc


def file_extension(encoding: str) -> str:
    return _EXTENSIONS.get(encoding.upper(), "bin")
