from tts_assess.reporting.aggregate import percentile, summarize


def test_percentile_interpolates_between_samples():
    assert percentile([0.0, 10.0], 50) == 5.0
    assert percentile([1.0, 2.0, 3.0, 4.0], 100) == 4.0
    assert percentile([5.0], 95) == 5.0


def test_summarize_counts_metrics_and_failure_labels():
    rows = [
        {"status": "pass", "wer": 0.0, "failure_labels": []},
        {"status": "fail", "wer": 0.5, "failure_labels": ["fail:wer"]},
        {"status": "warn", "wer": 0.06, "failure_labels": ["warn:wer"]},
    ]
    summary = summarize(rows)
    assert summary["sample_count"] == 3
    assert (summary["passed"], summary["warned"], summary["failed"]) == (1, 1, 1)
    assert abs(summary["pass_rate"] - 1 / 3) < 1e-9
    assert summary["metrics"]["wer"]["count"] == 3
    assert summary["metrics"]["wer"]["max"] == 0.5
    assert summary["failure_counts"]["fail:wer"] == 1


def test_summarize_empty_rows_is_safe():
    summary = summarize([])
    assert summary["sample_count"] == 0
    assert summary["pass_rate"] == 0.0
    assert summary["metrics"] == {}
