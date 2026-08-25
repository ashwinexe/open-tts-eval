from tts_assess.reporting.stats import bootstrap_ci, intervals_separated, summarize_values


def test_bootstrap_ci_collapses_for_constant_and_tiny_inputs():
    assert bootstrap_ci([5.0, 5.0, 5.0]) == (5.0, 5.0)
    assert bootstrap_ci([3.0]) == (3.0, 3.0)
    assert bootstrap_ci([]) == (0.0, 0.0)


def test_bootstrap_ci_brackets_the_mean_and_is_deterministic():
    values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    low, high = bootstrap_ci(values, resamples=1000)
    mean = sum(values) / len(values)
    assert low <= mean <= high
    assert min(values) <= low < high <= max(values)
    # Fixed seed -> identical interval on repeat.
    assert bootstrap_ci(values, resamples=1000) == (low, high)


def test_summarize_values_reports_full_shape():
    interval = summarize_values([1.0, 2.0, 3.0, 4.0])
    assert interval["n"] == 4
    assert interval["mean"] == 2.5
    assert interval["median"] == 2.5
    assert interval["min"] == 1.0
    assert interval["max"] == 4.0
    assert interval["ci_low"] <= interval["mean"] <= interval["ci_high"]
    assert interval["ci_level"] == 0.95


def test_summarize_values_empty_is_all_none():
    interval = summarize_values([])
    assert interval["n"] == 0
    assert interval["mean"] is None
    assert interval["ci_low"] is None


def test_intervals_separated():
    a = {"ci_low": 0.0, "ci_high": 0.1}
    b = {"ci_low": 0.2, "ci_high": 0.3}
    overlapping = {"ci_low": 0.05, "ci_high": 0.25}
    assert intervals_separated(a, b) is True
    assert intervals_separated(a, overlapping) is False
    assert intervals_separated(a, {"ci_low": None, "ci_high": None}) is False
