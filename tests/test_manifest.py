from pathlib import Path

from tts_assess.io.manifest import load_manifest, resolve_manifest_paths, validate_manifest_paths


def test_jsonl_manifest_loads_and_resolves_relative_paths(tmp_path: Path):
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"not a real wav")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text('{"id":"a","text":"Hello","audio_path":"audio.wav"}\n')

    items = resolve_manifest_paths(load_manifest(manifest), manifest)

    assert items[0].id == "a"
    assert items[0].audio_path == audio.resolve()
    assert validate_manifest_paths(items) == []
