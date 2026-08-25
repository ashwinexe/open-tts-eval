from __future__ import annotations

from tts_assess.config import BoundThreshold, Status

# Boolean row signals that count as hard failures regardless of numeric bands.
HEURISTIC_FAIL_KEYS = (
    "empty_transcript",
    "tail_hallucination",
    "repeated_span",
    "tail_click_detected",
)


def classify_row(
    row: dict[str, object],
    thresholds: dict[str, BoundThreshold],
) -> tuple[Status, dict[str, Status], list[str]]:
    """Overall status, per-metric statuses, and failure labels for one row.

    Shared by the assessment pipeline and the cross-run comparison so both apply
    thresholds identically.
    """
    labels: list[str] = [f"fail:{key}" for key in HEURISTIC_FAIL_KEYS if row.get(key)]
    status, metric_statuses, threshold_labels = evaluate_thresholds(row, thresholds)
    labels.extend(threshold_labels)
    if any(label.startswith("fail:") for label in labels):
        status = "fail"
    return status, metric_statuses, sorted(set(labels))


def evaluate_thresholds(
    metrics: dict[str, object],
    thresholds: dict[str, BoundThreshold],
) -> tuple[Status, dict[str, Status], list[str]]:
    statuses: dict[str, Status] = {}
    labels: list[str] = []
    for name, threshold in thresholds.items():
        if name not in metrics or metrics[name] is None:
            continue
        status = _evaluate_metric(metrics[name], threshold)
        statuses[name] = status
        if status in {"warn", "fail"}:
            labels.append(f"{status}:{name}")
    overall: Status = "pass"
    if any(status == "fail" for status in statuses.values()):
        overall = "fail"
    elif any(status == "warn" for status in statuses.values()):
        overall = "warn"
    return overall, statuses, labels


def _evaluate_metric(value: object, threshold: BoundThreshold) -> Status:
    if isinstance(value, bool):
        if threshold.fail_if_true and value:
            return "fail"
        return "pass"
    if not isinstance(value, int | float):
        return "skip"
    numeric = float(value)
    if threshold.fail is not None and numeric >= threshold.fail:
        return "fail"
    if threshold.fail_below is not None and numeric <= threshold.fail_below:
        return "fail"
    if threshold.warn is not None and numeric >= threshold.warn:
        return "warn"
    if threshold.warn_below is not None and numeric <= threshold.warn_below:
        return "warn"
    return "pass"
