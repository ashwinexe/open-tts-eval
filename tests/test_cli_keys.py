import pytest
import typer

from tts_assess.cli import _resolve_api_key


def test_key_file_takes_precedence_and_is_stripped(tmp_path, monkeypatch):
    monkeypatch.setenv("MY_KEY", "from-env")
    key_file = tmp_path / "key"
    key_file.write_text("  from-file\n")
    assert _resolve_api_key(key_file, "MY_KEY") == "from-file"


def test_falls_back_to_env(tmp_path, monkeypatch):
    monkeypatch.setenv("MY_KEY", "from-env")
    assert _resolve_api_key(None, "MY_KEY") == "from-env"


def test_empty_file_rejected(tmp_path, monkeypatch):
    monkeypatch.delenv("MY_KEY", raising=False)
    key_file = tmp_path / "key"
    key_file.write_text("   \n")
    with pytest.raises(typer.BadParameter, match="empty"):
        _resolve_api_key(key_file, "MY_KEY")


def test_missing_file_rejected(tmp_path):
    with pytest.raises(typer.BadParameter, match="cannot read"):
        _resolve_api_key(tmp_path / "nope", "MY_KEY")


def test_no_source_rejected(monkeypatch):
    monkeypatch.delenv("MY_KEY", raising=False)
    with pytest.raises(typer.BadParameter, match="no API key"):
        _resolve_api_key(None, "MY_KEY")
