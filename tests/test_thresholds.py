from tts_assess.config import BoundThreshold
from tts_assess.reporting.thresholds import evaluate_thresholds

THRESHOLDS = {
    "wer": BoundThreshold(warn=0.05, fail=0.10),
    "speaker_similarity": BoundThreshold(warn_below=0.65, fail_below=0.55),
    "tail_click_detected": BoundThreshold(fail_if_true=True),
}


def test_upper_bound_fail_warn_pass():
    assert evaluate_thresholds({"wer": 0.20}, THRESHOLDS)[0] == "fail"
    assert evaluate_thresholds({"wer": 0.07}, THRESHOLDS)[0] == "warn"
    assert evaluate_thresholds({"wer": 0.01}, THRESHOLDS)[0] == "pass"


def test_below_bound_warn_and_fail():
    _, statuses, _ = evaluate_thresholds({"speaker_similarity": 0.60}, THRESHOLDS)
    assert statuses["speaker_similarity"] == "warn"
    _, statuses, _ = evaluate_thresholds({"speaker_similarity": 0.50}, THRESHOLDS)
    assert statuses["speaker_similarity"] == "fail"


def test_boolean_fail_if_true():
    overall, statuses, labels = evaluate_thresholds({"tail_click_detected": True}, THRESHOLDS)
    assert overall == "fail"
    assert statuses["tail_click_detected"] == "fail"
    assert "fail:tail_click_detected" in labels
    _, statuses, _ = evaluate_thresholds({"tail_click_detected": False}, THRESHOLDS)
    assert statuses["tail_click_detected"] == "pass"


def test_missing_and_none_metrics_are_skipped():
    overall, statuses, labels = evaluate_thresholds({"wer": None}, THRESHOLDS)
    assert overall == "pass"
    assert statuses == {}
    assert labels == []
