import pytest
import yaml
from pydantic import ValidationError

from tts_assess.config import load_config


def test_load_config_none_returns_defaults():
    config = load_config(None)
    assert config.asr.backend == "faster-whisper"
    assert "wer" in config.thresholds


def test_load_config_deep_merges_over_defaults(tmp_path):
    path = tmp_path / "c.yml"
    path.write_text(
        yaml.safe_dump(
            {
                "asr": {"backend": "mock"},
                "thresholds": {"wer": {"warn": 0.2}},
            }
        )
    )
    config = load_config(path)
    # Explicit override applied.
    assert config.asr.backend == "mock"
    # Sibling default preserved rather than reset.
    assert config.asr.model == "small"
    # Threshold merged field-by-field: warn overridden, default fail kept.
    assert config.thresholds["wer"].warn == 0.2
    assert config.thresholds["wer"].fail == 0.10
    # Untouched default threshold still present.
    assert "cer" in config.thresholds


def test_unknown_config_fields_are_rejected(tmp_path):
    path = tmp_path / "bad.yml"
    path.write_text(yaml.safe_dump({"reporting": {"embed_audo": False}}))
    with pytest.raises(ValidationError, match="embed_audo"):
        load_config(path)
