from tts_assess.reporting.aggregate import summarize

ROWS = [
    {
        "status": "pass",
        "speaker_id": "spk_a",
        "wer": 0.01,
        "metric_statuses": {"wer": "pass"},
        "failure_labels": [],
    },
    {
        "status": "warn",
        "speaker_id": "spk_a",
        "wer": 0.06,
        "metric_statuses": {"wer": "warn"},
        "failure_labels": ["warn:wer"],
    },
    {
        "status": "fail",
        "speaker_id": "spk_b",
        "wer": 0.30,
        "metric_statuses": {"wer": "fail", "clipping_ratio": "fail"},
        "failure_labels": ["fail:wer", "fail:clipping_ratio"],
    },
]


def test_summarize_reports_confidence_interval_fields():
    summary = summarize(ROWS, bootstrap_resamples=200)
    wer = summary["metrics"]["wer"]
    assert wer["count"] == 3
    assert wer["ci_low"] is not None and wer["ci_high"] is not None
    assert wer["ci_low"] <= wer["mean"] <= wer["ci_high"]
    assert wer["ci_level"] == 0.95


def test_summarize_counts_threshold_violations_per_metric():
    violations = summarize(ROWS, bootstrap_resamples=200)["violations"]
    assert violations["wer"] == {"warn": 1, "fail": 1, "total": 2}
    assert violations["clipping_ratio"] == {"warn": 0, "fail": 1, "total": 1}
    # Ordered by total descending.
    assert list(violations)[0] == "wer"


def test_summarize_groups_by_voice():
    summary = summarize(ROWS, bootstrap_resamples=200, group_by_voice=True)
    by_voice = summary["by_voice"]
    assert set(by_voice) == {"spk_a", "spk_b"}
    assert by_voice["spk_a"]["sample_count"] == 2
    assert by_voice["spk_a"]["passed"] == 1
    assert by_voice["spk_b"]["failed"] == 1
    assert by_voice["spk_b"]["violations"]["wer"]["fail"] == 1


def test_group_by_voice_can_be_disabled():
    assert "by_voice" not in summarize(ROWS, bootstrap_resamples=200, group_by_voice=False)


def test_unspecified_voice_bucket():
    rows = [{"status": "pass", "wer": 0.0, "metric_statuses": {}, "failure_labels": []}]
    by_voice = summarize(rows, bootstrap_resamples=100)["by_voice"]
    assert "unspecified" in by_voice
