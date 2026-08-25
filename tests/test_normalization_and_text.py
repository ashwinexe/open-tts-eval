from tts_assess.metrics.text import compute_text_metrics
from tts_assess.normalization import normalize_english


def test_english_normalizer_expands_small_numbers_and_punctuation():
    assert normalize_english("I can't do 42 things!") == "i cannot do forty two things"
    assert normalize_english("I’ve seen it.") == "i have seen it"


def test_normalizer_folds_time_markers_and_glued_numbers():
    # "9 AM", "9am", and "9 a.m." must all normalize identically so ASR variants
    # do not inflate WER.
    assert normalize_english("9 AM") == "nine am"
    assert normalize_english("9 AM") == normalize_english("9am") == normalize_english("9 a.m.")


def test_normalizer_folds_hyphenated_compounds_and_abbreviations():
    assert normalize_english("text-to-speech") == "text to speech"
    assert normalize_english("the U.S. economy") == "the us economy"


def test_text_metrics_detect_repetition_and_tail_extra_words():
    metrics = compute_text_metrics(
        "hello world",
        "hello world hello world extra words",
    )
    assert metrics.wer > 0
    assert metrics.repeated_span is True
    assert metrics.tail_hallucination is True


def test_text_metrics_allow_reference_repetition_but_flag_added_repetition():
    expected = "not now not now"
    assert compute_text_metrics(expected, expected).repeated_span is False
    assert compute_text_metrics(expected, "not now not now not now").repeated_span is True
    assert compute_text_metrics("very very good", "very very good").repeated_span is False
