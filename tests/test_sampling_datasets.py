import pytest

from tts_assess.sampling.datasets import load_texts


def test_load_txt_autonumbers_and_skips_blank_lines(tmp_path):
    path = tmp_path / "d.txt"
    path.write_text("Hello world.\n\n  Second line.  \n")
    items = load_texts(path)
    assert [i.id for i in items] == ["utt_0000", "utt_0001"]
    assert items[1].text == "Second line."


def test_load_jsonl_uses_id_and_language(tmp_path):
    path = tmp_path / "d.jsonl"
    path.write_text(
        '{"id":"a","text":"First","language":"en-US"}\n'
        '{"text":"Second"}\n'
    )
    items = load_texts(path)
    assert items[0].id == "a"
    assert items[0].language == "en-US"
    assert items[1].id == "utt_0001"


def test_load_json_array_of_objects_and_strings(tmp_path):
    obj_path = tmp_path / "d.json"
    obj_path.write_text(
        '[{"text":"First","language":"en-US","category":"x"},{"id":"b","text":"Second"}]'
    )
    items = load_texts(obj_path)
    assert items[0].text == "First" and items[0].language == "en-US"
    assert items[1].id == "b"

    str_path = tmp_path / "s.json"
    str_path.write_text('["Alpha","Beta"]')
    items = load_texts(str_path)
    assert [i.text for i in items] == ["Alpha", "Beta"]
    assert items[0].id == "utt_0000"


def test_load_json_rejects_non_array(tmp_path):
    path = tmp_path / "d.json"
    path.write_text('{"text":"nope"}')
    with pytest.raises(ValueError, match="expected a JSON array"):
        load_texts(path)


def test_load_csv_requires_text(tmp_path):
    path = tmp_path / "d.csv"
    path.write_text("id,text\nc1,Hello\n")
    items = load_texts(path)
    assert items[0].id == "c1" and items[0].text == "Hello"


def test_limit_and_unsupported_and_empty(tmp_path):
    path = tmp_path / "d.txt"
    path.write_text("a\nb\nc\n")
    assert len(load_texts(path, limit=2)) == 2

    with pytest.raises(ValueError, match="unsupported dataset format"):
        load_texts(tmp_path / "d.wav")

    empty = tmp_path / "empty.txt"
    empty.write_text("\n\n")
    with pytest.raises(ValueError, match="no texts loaded"):
        load_texts(empty)


def test_missing_text_field_raises(tmp_path):
    path = tmp_path / "d.jsonl"
    path.write_text('{"id":"a"}\n')
    with pytest.raises(ValueError, match="missing a non-empty 'text'"):
        load_texts(path)
