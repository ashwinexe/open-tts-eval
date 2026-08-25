from pathlib import Path

from tts_assess.io.manifest import (
    load_manifest,
    resolve_manifest_paths,
    validate_manifest_paths,
)


def test_csv_manifest_parses_metadata_and_resolves_paths(tmp_path: Path):
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"not a real wav")
    manifest = tmp_path / "m.csv"
    manifest.write_text(
        "id,text,audio_path,metadata\n"
        'c1,"Hello, world.",a.wav,"{""source"":""demo""}"\n'
    )

    items = resolve_manifest_paths(load_manifest(manifest), manifest)

    assert items[0].id == "c1"
    assert items[0].text == "Hello, world."
    assert items[0].metadata == {"source": "demo"}
    assert items[0].audio_path == audio.resolve()
    assert validate_manifest_paths(items) == []


def test_validate_reports_duplicate_ids_and_missing_paths(tmp_path: Path):
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"x")
    manifest = tmp_path / "m.jsonl"
    manifest.write_text(
        '{"id":"dup","text":"a","audio_path":"a.wav"}\n'
        '{"id":"dup","text":"b","audio_path":"missing.wav"}\n'
    )

    items = resolve_manifest_paths(load_manifest(manifest), manifest)
    errors = validate_manifest_paths(items)

    assert any("duplicate id" in error for error in errors)
    assert any("does not exist" in error for error in errors)
