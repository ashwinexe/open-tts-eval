import json
from pathlib import Path

from tts_assess.config import AssessmentConfig
from tts_assess.reporting.aggregate import summarize
from tts_assess.reporting.compare import (
    build_comparison,
    build_run_report,
    default_label,
    load_run_rows,
    run_comparison,
)
from tts_assess.reporting.comparison_html import render_comparison_html
from tts_assess.reporting.thresholds import classify_row


def _rows(prefix: str, wer_scale: float):
    rows = []
    for i in range(8):
        rows.append(
            {
                "id": f"{prefix}_{i}",
                "speaker_id": "spk_a" if i % 2 == 0 else "spk_b",
                "text": "hello world",
                "transcript": f"hello world {i}",
                "audio_path": f"/tmp/audio/{prefix}_{i}.wav",
                "wer": round(wer_scale * (0.02 + 0.01 * i), 4),
                "cer": round(wer_scale * 0.01 * (i + 1), 4),
                "empty_transcript": False,
                "tail_hallucination": False,
                "repeated_span": False,
                "tail_click_detected": False,
            }
        )
    return rows


def _config():
    config = AssessmentConfig.default()
    config.reporting.bootstrap_resamples = 300
    return config


def _metric_row(report, key):
    for group in report["metric_table"]["groups"]:
        for row in group["rows"]:
            if row["key"] == key:
                return row
    raise KeyError(key)


def _metric_group(report, key):
    for group in report["metric_table"]["groups"]:
        if any(r["key"] == key for r in group["rows"]):
            return group
    raise KeyError(key)


def _health_row(report, key):
    for row in report["health_table"]["rows"]:
        if row["key"] == key:
            return row
    raise KeyError(key)


def test_build_comparison_structure():
    runs = [("alpha", _rows("alpha", 1.0)), ("beta", _rows("beta", 3.0))]
    report = build_comparison(runs, _config())
    assert report["title"] == "Comparative Report"
    assert report["labels"] == ["alpha", "beta"]
    assert {"metric_table", "health_table"} <= set(report)
    wer = _metric_row(report, "wer")
    assert wer["better"] == "lower" and len(wer["cells"]) == 2


def test_metric_grouping_and_hidden():
    runs = []
    for prefix, scale in (("alpha", 1.0), ("beta", 3.0)):
        rows = _rows(prefix, scale)
        for i, r in enumerate(rows):
            r["length_ratio"] = 1.0 + 0.01 * i   # must be hidden
            r["silence_ratio"] = 0.2 + 0.02 * i   # Silence group, present
        runs.append((prefix, rows))
    report = build_comparison(runs, _config())
    titles = [g["title"] for g in report["metric_table"]["groups"]]
    assert titles == ["Accuracy", "Silence"]  # only groups with present metrics
    assert _metric_group(report, "wer")["title"] == "Accuracy"
    assert _metric_group(report, "silence_ratio")["title"] == "Silence"
    # length_ratio is hidden everywhere.
    with __import__("pytest").raises(KeyError):
        _metric_row(report, "length_ratio")


def test_others_group_is_not_colorized():
    runs = []
    for prefix in ("alpha", "beta"):
        rows = _rows(prefix, 1.0)
        for i, r in enumerate(rows):
            r["silence_ratio"] = 0.2 + 0.05 * i * (1 if prefix == "alpha" else 2)
        runs.append((prefix, rows))
    report = build_comparison(runs, _config())
    others = _metric_group(report, "silence_ratio")
    assert others["colorize"] is False
    sil = _metric_row(report, "silence_ratio")
    assert all(not c.get("best") and not c.get("worst") for c in sil["cells"])
    assert sil["better"] is None


def test_vowel_health_uses_fault_rate():
    runs = []
    for prefix in ("alpha", "beta"):
        rows = _rows(prefix, 1.0)
        # 2 of 8 clips exceed 0.8 (a fault); the rest are short.
        for i, r in enumerate(rows):
            r["vowel_prolongation_score"] = 1.2 if i < 2 else 0.3
        runs.append((prefix, rows))
    report = build_comparison(runs, _config())
    # Vowel prolongation is not in Table 1 anymore.
    import pytest

    with pytest.raises(KeyError):
        _metric_row(report, "vowel_prolongation_score")
    # Table 2 uses the configured threshold classification: pass = 6/8 = 0.75.
    h = _health_row(report, "vowel_prolongation_score")
    assert abs(h["cells"][0]["good_rate"] - 0.75) < 1e-9
    assert h["pass_rule"] == "< 0.8"


def test_vowel_health_uses_custom_threshold_and_inclusive_failure_boundary():
    config = _config()
    config.thresholds["vowel_prolongation_score"].fail = 1.2
    rows_a = _rows("alpha", 1.0)
    rows_b = _rows("beta", 1.0)
    for rows in (rows_a, rows_b):
        for i, row in enumerate(rows):
            row["vowel_prolongation_score"] = 1.2 if i == 0 else 1.0

    report = build_comparison([("alpha", rows_a), ("beta", rows_b)], config)
    health = _health_row(report, "vowel_prolongation_score")
    assert health["pass_rule"] == "< 1.2"
    assert health["cells"][0]["good_rate"] == 7 / 8


def test_metric_best_worst_and_significance():
    runs = [("alpha", _rows("alpha", 1.0)), ("beta", _rows("beta", 3.0))]
    report = build_comparison(runs, _config())
    wer = _metric_row(report, "wer")
    # alpha (lower WER) wins, beta loses.
    assert wer["cells"][0]["best"] is True
    assert wer["cells"][1]["worst"] is True
    # beta's interval is far from alpha's -> significant.
    assert wer["cells"][1]["sig"] is True
    assert wer["cells"][0]["mean"] < wer["cells"][1]["mean"]


def test_health_pass_rates_bands_and_hidden():
    runs = []
    for prefix, sil in (("alpha", 0.1), ("beta", 0.6)):  # silence warn=0.45, fail=0.65
        rows = _rows(prefix, 1.0)
        for r in rows:
            r["silence_ratio"] = sil
        runs.append((prefix, rows))
    report = build_comparison(runs, _config())
    sil = _health_row(report, "silence_ratio")
    # alpha: all 0.1 (<0.45) -> 100% pass -> good; beta: 0.6 (warn band) -> 0% pass -> fail.
    assert sil["cells"][0]["good_rate"] == 1.0 and sil["cells"][0]["status"] == "good"
    assert sil["cells"][1]["good_rate"] == 0.0 and sil["cells"][1]["status"] == "fail"
    # WER / CER / insertions are hidden from Model Health.
    import pytest

    for hidden in ("wer", "cer", "insertion_rate"):
        with pytest.raises(KeyError):
            _health_row(report, hidden)


def test_render_html_is_minimal_and_static():
    runs = [("alpha", _rows("alpha", 1.0)), ("beta", _rows("beta", 3.0))]
    config = _config()
    config.reporting.question = "Which model is healthier?"
    html = render_comparison_html(build_comparison(runs, config))
    assert "<h1>Comparative Report</h1>" in html
    assert "Which model is healthier?" in html
    assert "Metric Comparison" in html and "Model Health" in html
    # Removed blocks must not appear.
    for gone in ("Run Overview", "By-Voice", "sources", "artifact", "toc", "Notes"):
        assert gone not in html


def test_run_comparison_writes_artifacts(tmp_path: Path):
    run_a, run_b = tmp_path / "alpha", tmp_path / "beta"
    run_a.mkdir()
    run_b.mkdir()
    (run_a / "results.jsonl").write_text("\n".join(json.dumps(r) for r in _rows("alpha", 1.0)))
    (run_b / "results.jsonl").write_text("\n".join(json.dumps(r) for r in _rows("beta", 3.0)))

    report = run_comparison(
        [run_a, run_b], tmp_path / "out", _config(), title="Provider A vs B"
    )

    assert (tmp_path / "out" / "comparison.html").exists()
    assert (tmp_path / "out" / "comparison.json").exists()
    assert report["labels"] == ["alpha", "beta"]
    assert report["title"] == "Provider A vs B"
    assert "<h1>Provider A vs B</h1>" in (tmp_path / "out" / "comparison.html").read_text()


def test_labels_override_and_mismatch_raises(tmp_path: Path):
    run_a, run_b = tmp_path / "r1", tmp_path / "r2"
    run_a.mkdir()
    run_b.mkdir()
    (run_a / "results.jsonl").write_text(
        "\n".join(json.dumps(row) for row in _rows("r1", 1.0))
    )
    (run_b / "results.jsonl").write_text(
        "\n".join(json.dumps(row) for row in _rows("r2", 2.0))
    )

    report = run_comparison(
        [run_a, run_b], tmp_path / "out", _config(), labels=["Custom A", "Custom B"]
    )
    assert report["labels"] == ["Custom A", "Custom B"]

    try:
        run_comparison([run_a, run_b], tmp_path / "out2", _config(), labels=["a"])
    except ValueError as exc:
        assert "one --label per run" in str(exc)
    else:
        raise AssertionError("expected ValueError on label/run count mismatch")

    with __import__("pytest").raises(ValueError, match="at least two"):
        run_comparison([run_a], tmp_path / "out3", _config())


def test_comparison_rejects_different_content_and_speaker_profiles():
    alpha = _rows("alpha", 1.0)
    missing = _rows("beta", 1.0)[:-1]
    with __import__("pytest").raises(ValueError, match="cohort differs"):
        build_comparison([("alpha", alpha), ("missing", missing)], _config())

    different_speakers = _rows("beta", 1.0)
    for row in different_speakers:
        row["speaker_id"] = "only_one_voice"
    with __import__("pytest").raises(ValueError, match="per-speaker sampling profile"):
        build_comparison([("alpha", alpha), ("different", different_speakers)], _config())


def test_run_report_violations_dedup_columns_and_audio(tmp_path: Path):
    (tmp_path / "audio").mkdir()
    (tmp_path / "audio" / "s0.wav").write_bytes(b"x")  # only sample 0 has audio on disk
    cfg = _config()
    rows = []
    for i in range(5):
        wer = 0.2 if i < 3 else 0.0  # s0,s1,s2 fail WER
        row = {
            "id": f"s{i}",
            "speaker_id": "v",
            "text": "hello world",
            "transcript": "hello",
            "normalized_text": "hello world",
            "normalized_transcript": "hello",
            "wer": wer,
            "cer": wer / 2,
            "tail_click_detected": (i == 0),  # s0 also violates -> must dedup
            "audio_path": f"audio/{'s0' if i == 0 else 'missing'}.wav",
        }
        status, ms, labels = classify_row(row, cfg.thresholds)
        row["status"], row["metric_statuses"], row["failure_labels"] = status, ms, labels
        rows.append(row)
    summary = summarize(rows, bootstrap_resamples=100, group_by_voice=False)

    report = build_run_report(rows, summary, cfg, output_dir=tmp_path, title="T", label="run-x")
    assert report["title"] == "T" and report["labels"] == ["run-x"]
    assert {"metric_table", "health_table", "violations"} <= set(report)

    ex = report["violations"]["examples"]
    # Hidden id/status: example rows expose only these fields.
    assert set(ex["rows"][0]) == {"metric", "voice", "value", "expected", "heard", "audio"}
    # Dedup: s0 violates WER + tail-click but appears once -> 3 rows, all WER.
    assert len(ex["rows"]) == 3
    assert all(r["metric"] == "WER" for r in ex["rows"])
    # Audio shown only when the file exists on disk.
    assert ex["has_audio"] is True
    assert ex["rows"][0]["audio"] == "audio/s0.wav"
    assert all(r["audio"] is None for r in ex["rows"][1:])

    cfg.reporting.embed_audio = False
    cfg.reporting.max_worst_samples = 2
    limited = build_run_report(rows, summary, cfg, output_dir=tmp_path, title="T", label="run-x")
    limited_examples = limited["violations"]["examples"]
    assert len(limited_examples["rows"]) == 2
    assert limited_examples["has_audio"] is False
    assert all(row["audio"] is None for row in limited_examples["rows"])


def test_run_report_renders_violations_chapter(tmp_path: Path):
    cfg = _config()
    rows = []
    for i in range(4):
        row = {"id": f"s{i}", "speaker_id": "v", "text": "hi", "transcript": "hi",
               "wer": 0.3, "cer": 0.1, "audio_path": "audio/x.wav"}
        status, ms, labels = classify_row(row, cfg.thresholds)
        row["status"], row["metric_statuses"], row["failure_labels"] = status, ms, labels
        rows.append(row)
    summary = summarize(rows, bootstrap_resamples=100, group_by_voice=False)
    html = render_comparison_html(build_run_report(rows, summary, cfg, output_dir=tmp_path,
                                                   title="Run", label="r"))
    assert "Threshold Violations" in html
    assert "expected" in html and "heard" in html


def test_load_run_rows_and_default_label(tmp_path: Path):
    run = tmp_path / "myrun"
    run.mkdir()
    (run / "results.jsonl").write_text('{"id":"a"}\n\n{"id":"b"}\n')
    rows = load_run_rows(run)
    assert [r["id"] for r in rows] == ["a", "b"]
    assert default_label(run) == "myrun"
    assert default_label(run / "results.jsonl") == "results"

