from __future__ import annotations

import time
import urllib.error
import urllib.request
from collections.abc import Callable

from tts_assess.sampling.providers.base import ProviderError

# transport(method, url, headers, body, timeout) -> (status_code, response_bytes).
# Single-shot; retries are layered on top by `call`. Injectable for testing.
Transport = Callable[[str, str, dict[str, str], bytes | None, float], tuple[int, bytes]]

RETRY_STATUS = frozenset({429, 500, 502, 503, 504})

# Some providers sit behind Cloudflare, which blocks the default "Python-urllib"
# User-Agent with a 1010 error. Present a normal UA unless the caller set one.
_DEFAULT_USER_AGENT = "Mozilla/5.0 (compatible; tts-assess/0.1; +https://inworld.ai)"


def urllib_transport(
    method: str, url: str, headers: dict[str, str], body: bytes | None, timeout: float
) -> tuple[int, bytes]:
    if not any(key.lower() == "user-agent" for key in headers):
        headers = {**headers, "User-Agent": _DEFAULT_USER_AGENT}
    request = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()
    except urllib.error.URLError as exc:
        raise ProviderError(f"network error calling {url}: {exc.reason}") from exc


def call(
    transport: Transport,
    method: str,
    url: str,
    headers: dict[str, str],
    *,
    body: bytes | None = None,
    timeout: float = 120.0,
    max_retries: int = 3,
) -> tuple[int, bytes]:
    """Perform an HTTP request with retry/backoff on transient statuses."""
    for attempt in range(max(1, max_retries)):
        status, data = transport(method, url, headers, body, timeout)
        if status in RETRY_STATUS and attempt < max_retries - 1:
            time.sleep(min(2**attempt, 8))
            continue
        return status, data
    return status, data


def snippet(data: bytes, limit: int = 300) -> str:
    text = data.decode("utf-8", errors="replace").strip()
    return text if len(text) <= limit else text[:limit] + "…"
