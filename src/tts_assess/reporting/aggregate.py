from __future__ import annotations

from typing import Any

from tts_assess.reporting.stats import summarize_values

UNSPECIFIED_VOICE = "unspecified"


def summarize(
    rows: list[dict[str, Any]],
    *,
    confidence_level: float = 0.95,
    bootstrap_resamples: int = 2000,
    group_by_voice: bool = True,
) -> dict[str, Any]:
    summary = _group_summary(rows, confidence_level, bootstrap_resamples)

    failures: dict[str, int] = {}
    for row in rows:
        for label in row.get("failure_labels", []):
            failures[label] = failures.get(label, 0) + 1
    summary["failure_counts"] = dict(
        sorted(failures.items(), key=lambda item: item[1], reverse=True)
    )

    if group_by_voice:
        groups: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            voice = row.get("speaker_id") or UNSPECIFIED_VOICE
            groups.setdefault(voice, []).append(row)
        summary["by_voice"] = {
            voice: _group_summary(voice_rows, confidence_level, bootstrap_resamples)
            for voice, voice_rows in sorted(groups.items())
        }
    return summary


def _group_summary(
    rows: list[dict[str, Any]],
    confidence_level: float,
    bootstrap_resamples: int,
) -> dict[str, Any]:
    total = len(rows)
    passed = sum(1 for row in rows if row.get("status") == "pass")
    warned = sum(1 for row in rows if row.get("status") == "warn")
    failed = sum(1 for row in rows if row.get("status") == "fail")

    metric_summary: dict[str, Any] = {}
    for metric in _numeric_metric_names(rows):
        values = [
            float(row[metric])
            for row in rows
            if isinstance(row.get(metric), int | float) and not isinstance(row.get(metric), bool)
        ]
        interval = summarize_values(
            values, confidence=confidence_level, resamples=bootstrap_resamples
        )
        metric_summary[metric] = {
            "count": interval["n"],
            "mean": interval["mean"],
            "std": interval["std"],
            "median": interval["median"],
            "p95": interval["p95"],
            "min": interval["min"],
            "max": interval["max"],
            "ci_low": interval["ci_low"],
            "ci_high": interval["ci_high"],
            "ci_level": interval["ci_level"],
        }

    return {
        "sample_count": total,
        "passed": passed,
        "warned": warned,
        "failed": failed,
        "pass_rate": passed / total if total else 0.0,
        "metrics": metric_summary,
        "violations": _violation_counts(rows),
    }


def _violation_counts(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    """Per-metric warn/fail counts derived from each row's threshold statuses."""
    violations: dict[str, dict[str, int]] = {}
    for row in rows:
        for metric, status in (row.get("metric_statuses") or {}).items():
            if status in ("warn", "fail"):
                bucket = violations.setdefault(metric, {"warn": 0, "fail": 0, "total": 0})
                bucket[status] += 1
                bucket["total"] += 1
    return dict(sorted(violations.items(), key=lambda item: item[1]["total"], reverse=True))


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("percentile requires non-empty values")
    index = (len(ordered) - 1) * p / 100
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _numeric_metric_names(rows: list[dict[str, Any]]) -> list[str]:
    excluded = {"sample_rate", "channels"}
    names: set[str] = set()
    for row in rows:
        for key, value in row.items():
            if key in excluded:
                continue
            if isinstance(value, int | float) and not isinstance(value, bool):
                names.add(key)
    return sorted(names)
