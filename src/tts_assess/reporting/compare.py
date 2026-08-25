from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tts_assess.config import AssessmentConfig
from tts_assess.reporting.aggregate import summarize
from tts_assess.reporting.comparison_html import render_comparison_html
from tts_assess.reporting.metrics_meta import KEY_METRICS, direction, display_name
from tts_assess.reporting.stats import intervals_separated
from tts_assess.reporting.thresholds import classify_row
from tts_assess.reporting.writers import write_summary

STATIC_TITLE = "Comparative Report"

# Metric Comparison table layout: (group title, colorize best/worst?, [(key, label)]).
# Only metrics present in the data are shown; anything not listed here is hidden.
COMPARISON_GROUPS: list[tuple[str, bool, list[tuple[str, str]]]] = [
    ("Accuracy", True, [
        ("wer", "WER"),
        ("insertion_rate", "WER.Insertions"),
        ("deletion_rate", "WER.Deletions"),
        ("substitution_rate", "WER.Substitutions"),
        ("cer", "CER"),
    ]),
    ("NISQAv2", True, [
        ("nisqa_mos", "MOS"),
        ("nisqa_coloration", "coloration"),
        ("nisqa_discontinuity", "discontinuity"),
        ("nisqa_loudness", "loudness"),
        ("nisqa_noisiness", "noisiness"),
    ]),
    ("Subjective", True, [
        ("chars_per_second", "Chars/sec"),
        ("arousal_proxy", "Arousal"),
        ("expressiveness_proxy", "Expressiveness"),
    ]),
    ("Silence", False, [
        ("silence_ratio", "Silence"),
        ("leading_silence_sec", "Lead silence (s)"),
        ("trailing_silence_sec", "Tail silence (s)"),
    ]),
]

# Metrics excluded from the Model Health (Table 2) grid.
HEALTH_HIDE: frozenset[str] = frozenset({"wer", "cer", "insertion_rate"})


def load_run_rows(path: Path) -> list[dict[str, Any]]:
    """Load one run's per-sample rows from its results.jsonl."""
    path = Path(path)
    jsonl = path if path.suffix == ".jsonl" else path / "results.jsonl"
    if not jsonl.exists():
        raise FileNotFoundError(f"no results.jsonl found for run: {path}")
    rows = []
    for line in jsonl.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def default_label(path: Path) -> str:
    path = Path(path)
    return path.stem if path.suffix == ".jsonl" else path.name


def run_comparison(
    run_paths: list[Path],
    output_dir: Path,
    config: AssessmentConfig,
    *,
    labels: list[str] | None = None,
    title: str | None = None,
) -> dict[str, Any]:
    if labels and len(labels) != len(run_paths):
        raise ValueError(
            f"got {len(labels)} labels for {len(run_paths)} runs; provide one --label per run"
        )
    runs = []
    for index, path in enumerate(run_paths):
        rows = load_run_rows(path)
        label = labels[index] if labels else default_label(path)
        runs.append((label, rows))

    report_data = build_comparison(runs, config, title=title)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_summary(output_dir / "comparison.json", report_data)
    (output_dir / "comparison.html").write_text(
        render_comparison_html(report_data), encoding="utf-8"
    )
    return report_data


def build_run_report(
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    config: AssessmentConfig,
    *,
    output_dir: Path,
    title: str,
    label: str,
) -> dict[str, Any]:
    """Single-run report: the comparison layout (one column) + a violations chapter.

    Rows are expected already classified by the pipeline under this config.
    """
    return {
        "title": title,
        "subtitle": f"{len(rows)} samples",
        "question": config.reporting.question,
        "labels": [label],
        "metric_table": _metric_table([summary], config.reporting.confidence_level),
        "health_table": _health_table([rows], config),
        "violations": _run_violations(rows, config, Path(output_dir)),
    }


def _run_violations(
    rows: list[dict[str, Any]], config: AssessmentConfig, output_dir: Path
) -> dict[str, Any]:
    per_metric_limit = config.reporting.max_violation_examples
    total_limit = config.reporting.max_worst_samples
    counts: dict[str, dict[str, int]] = {}
    for row in rows:
        for metric, status in (row.get("metric_statuses") or {}).items():
            if status in ("warn", "fail"):
                bucket = counts.setdefault(metric, {"warn": 0, "fail": 0, "total": 0})
                bucket[status] += 1
                bucket["total"] += 1
    counts = dict(sorted(counts.items(), key=lambda kv: kv[1]["total"], reverse=True))
    count_rows = [{"metric": display_name(m), **b} for m, b in counts.items()]

    used: set[str] = set()  # sample ids already shown, so each appears once
    examples: list[dict[str, Any]] = []
    has_audio = False
    for metric in counts:
        if len(examples) >= total_limit:
            break
        offenders = [
            row
            for row in rows
            if (row.get("metric_statuses") or {}).get(metric) in ("warn", "fail")
            and row.get("id") not in used
        ]
        numeric = [
            row
            for row in offenders
            if isinstance(row.get(metric), int | float) and not isinstance(row.get(metric), bool)
        ]
        if numeric:
            numeric.sort(key=lambda row: row.get(metric), reverse=direction(metric) != "higher")
            chosen = numeric[:per_metric_limit]
        else:
            chosen = offenders[:per_metric_limit]
        chosen = chosen[: total_limit - len(examples)]
        for row in chosen:
            used.add(row.get("id"))
            audio = (
                _existing_audio(row.get("audio_path"), output_dir)
                if config.reporting.embed_audio
                else None
            )
            has_audio = has_audio or bool(audio)
            examples.append(
                {
                    "metric": display_name(metric),
                    "voice": row.get("speaker_id") or "\u2014",
                    "value": _fmt_value(row.get(metric)),
                    "expected": _snippet(row.get("normalized_text") or row.get("text") or ""),
                    "heard": _snippet(
                        row.get("normalized_transcript") or row.get("transcript") or ""
                    ),
                    "audio": audio,
                }
            )
    total = sum(b["total"] for b in counts.values())
    state = (
        "No configured threshold was violated."
        if not count_rows
        else f"{total} violations across {len(counts)} metric(s)."
    )
    return {
        "state": state,
        "counts": count_rows,
        "examples": {"has_audio": has_audio, "rows": examples},
    }


def _existing_audio(audio_path: str | None, output_dir: Path) -> str | None:
    if not audio_path:
        return None
    path = Path(audio_path)
    full = path if path.is_absolute() else output_dir / path
    return audio_path if full.exists() else None


def _snippet(text: str, limit: int = 90) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1] + "\u2026"


def _fmt_value(value: Any) -> str:
    if isinstance(value, bool):
        return "flagged" if value else "ok"
    if isinstance(value, int | float):
        return _num(value)
    return "" if value is None else str(value)


def build_comparison(
    runs: list[tuple[str, list[dict[str, Any]]]],
    config: AssessmentConfig,
    *,
    title: str | None = None,
    subtitle: str | None = None,
) -> dict[str, Any]:
    _validate_comparable_runs(runs)
    confidence = config.reporting.confidence_level
    resamples = config.reporting.bootstrap_resamples
    labels = [label for label, _ in runs]

    all_rows: list[list[dict[str, Any]]] = []
    summaries: list[dict[str, Any]] = []
    for _label, rows in runs:
        # Re-apply the shared thresholds so both tables are comparable even if the
        # runs were originally assessed with different bands.
        for row in rows:
            status, metric_statuses, failure_labels = classify_row(row, config.thresholds)
            row["status"] = status
            row["metric_statuses"] = metric_statuses
            row["failure_labels"] = failure_labels
        all_rows.append(rows)
        summaries.append(
            summarize(
                rows,
                confidence_level=confidence,
                bootstrap_resamples=resamples,
                group_by_voice=False,
            )
        )

    return {
        "title": title or STATIC_TITLE,
        "subtitle": subtitle,
        "question": config.reporting.question,
        "generated": datetime.now(timezone.utc).isoformat(),
        "labels": labels,
        "metric_table": _metric_table(summaries, confidence),
        "health_table": _health_table(all_rows, config),
    }


def _validate_comparable_runs(runs: list[tuple[str, list[dict[str, Any]]]]) -> None:
    """Require equivalent content and voice-sampling shapes across comparison runs."""
    if len(runs) < 2:
        raise ValueError("comparison requires at least two runs")

    reference_label, reference_rows = runs[0]
    reference_content, reference_speakers = _comparison_profile(reference_label, reference_rows)
    for label, rows in runs[1:]:
        content, speakers = _comparison_profile(label, rows)
        if content != reference_content:
            missing = sum((reference_content - content).values())
            extra = sum((content - reference_content).values())
            raise ValueError(
                f"run {label!r} is not comparable to {reference_label!r}: "
                f"text/language cohort differs ({missing} missing, {extra} extra samples)"
            )
        if speakers != reference_speakers:
            raise ValueError(
                f"run {label!r} is not comparable to {reference_label!r}: "
                "per-speaker sampling profile differs"
            )


def _comparison_profile(
    label: str, rows: list[dict[str, Any]]
) -> tuple[Counter[tuple[str, str]], Counter[tuple[tuple[tuple[str, str], int], ...]]]:
    if not rows:
        raise ValueError(f"run {label!r} has no samples")

    content: Counter[tuple[str, str]] = Counter()
    by_speaker: dict[str, Counter[tuple[str, str]]] = {}
    seen_ids: set[str] = set()
    speaker_presence: list[bool] = []
    for index, row in enumerate(rows):
        sample_id = row.get("id")
        if not isinstance(sample_id, str) or not sample_id.strip():
            raise ValueError(f"run {label!r} sample {index} has no non-empty id")
        if sample_id in seen_ids:
            raise ValueError(f"run {label!r} has duplicate sample id {sample_id!r}")
        seen_ids.add(sample_id)

        text = row.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"run {label!r} sample {sample_id!r} has no non-empty text")
        key = (text, str(row.get("language") or ""))
        content[key] += 1

        speaker = row.get("speaker_id")
        has_speaker = isinstance(speaker, str) and bool(speaker.strip())
        speaker_presence.append(has_speaker)
        if has_speaker:
            by_speaker.setdefault(speaker, Counter())[key] += 1

    if any(speaker_presence) and not all(speaker_presence):
        raise ValueError(f"run {label!r} mixes samples with and without speaker_id")

    speaker_profiles: Counter[tuple[tuple[tuple[str, str], int], ...]] = Counter()
    if all(speaker_presence):
        for samples in by_speaker.values():
            speaker_profiles[tuple(sorted(samples.items()))] += 1
    return content, speaker_profiles


def _metric_table(summaries: list[dict[str, Any]], confidence: float) -> dict[str, Any]:
    present = _union_metrics(summaries)
    groups = []
    for title, colorize, members in COMPARISON_GROUPS:
        rows = []
        for key, label in members:
            if key not in present:
                continue
            stats = [summary.get("metrics", {}).get(key) for summary in summaries]
            rows.append(
                {
                    "metric": label,
                    "key": key,
                    "better": direction(key) if colorize else None,
                    "cells": _metric_cells(key, stats, colorize=colorize),
                }
            )
        if rows:
            groups.append({"title": title, "colorize": colorize, "rows": rows})
    return {"ci_label": f"{round(confidence * 100)}% CI", "groups": groups}


def _metric_cells(
    metric: str, stats: list[dict[str, Any] | None], *, colorize: bool
) -> list[dict[str, Any]]:
    best, worst = _best_worst_stats(metric, stats) if colorize else (None, None)
    cells = []
    for index, stat in enumerate(stats):
        if not stat or stat.get("mean") is None:
            cells.append({"na": True})
            continue
        sig = (
            best is not None
            and index != best
            and stats[best] is not None
            and intervals_separated(stats[best], stat)
        )
        cells.append(
            {
                "na": False,
                "mean": stat.get("mean"),
                "lo": stat.get("ci_low"),
                "hi": stat.get("ci_high"),
                "best": index == best,
                "worst": index == worst,
                "sig": bool(sig),
            }
        )
    return cells


def _health_table(
    all_rows: list[list[dict[str, Any]]], config: AssessmentConfig
) -> dict[str, Any]:
    good_band = config.reporting.health_good_rate
    warn_band = config.reporting.health_warn_rate
    # Only metrics with a configured threshold define a pass/fail, hence a "good %".
    metrics = [
        m
        for m in _ordered(set(config.thresholds))
        if m not in HEALTH_HIDE and _evaluated_anywhere(m, all_rows)
    ]
    rows = []
    for metric in metrics:
        cells = []
        for run_rows in all_rows:
            rate = _pass_rate(metric, run_rows)
            if rate is None:
                cells.append({"na": True})
            else:
                cells.append(
                    {"na": False, "good_rate": rate, "status": _band(rate, good_band, warn_band)}
                )
        pass_rule = _pass_rule(config.thresholds[metric])
        rows.append(
            {
                "metric": display_name(metric),
                "key": metric,
                "pass_rule": pass_rule,
                "cells": cells,
            }
        )
    return {"good_rate": good_band, "warn_rate": warn_band, "rows": rows}


def _pass_rule(threshold: Any) -> str:
    """Human-readable per-sample condition for a sample to count as passing."""
    if threshold.fail_if_true is not None:
        return "not flagged"
    # A sample passes when it is strictly better than the warn band (or the fail
    # band if no warn is set) in the metric's direction.
    if threshold.warn is not None or threshold.fail is not None:
        bound = threshold.warn if threshold.warn is not None else threshold.fail
        return f"< {_num(bound)}"
    if threshold.warn_below is not None or threshold.fail_below is not None:
        bound = threshold.warn_below if threshold.warn_below is not None else threshold.fail_below
        return f"> {_num(bound)}"
    return "\u2014"


def _num(value: float) -> str:
    return f"{value:.3f}".rstrip("0").rstrip(".") or "0"


def _pass_rate(metric: str, rows: list[dict[str, Any]]) -> float | None:
    evaluated = good = 0
    for row in rows:
        status = (row.get("metric_statuses") or {}).get(metric)
        if status in ("pass", "warn", "fail"):
            evaluated += 1
            if status == "pass":
                good += 1
    return good / evaluated if evaluated else None


def _evaluated_anywhere(metric: str, all_rows: list[list[dict[str, Any]]]) -> bool:
    return any(_pass_rate(metric, rows) is not None for rows in all_rows)


def _band(rate: float, good: float, warn: float) -> str:
    if rate >= good:
        return "good"
    if rate >= warn:
        return "warn"
    return "fail"


def _best_worst_stats(
    metric: str, stats: list[dict[str, Any] | None]
) -> tuple[int | None, int | None]:
    facing = direction(metric)
    means = {
        index: stat["mean"]
        for index, stat in enumerate(stats)
        if stat and stat.get("mean") is not None
    }
    if not facing or len(means) < 2:
        return None, None
    if facing == "lower":
        best = min(means, key=means.get)
        worst = max(means, key=means.get)
    else:
        best = max(means, key=means.get)
        worst = min(means, key=means.get)
    if means[best] == means[worst]:
        return None, None
    return best, worst


def _union_metrics(summaries: list[dict[str, Any]]) -> list[str]:
    names: set[str] = set()
    for summary in summaries:
        names.update(summary.get("metrics", {}))
    return _ordered(names)


def _ordered(names: set[str]) -> list[str]:
    lead = [name for name in KEY_METRICS if name in names]
    rest = sorted(name for name in names if name not in KEY_METRICS)
    return lead + rest
