from __future__ import annotations

# Direction of "better" for each metric. Metrics absent from both sets are
# treated as neutral (no winner is highlighted and no metric is scored as an
# improvement or regression).
LOWER_IS_BETTER = frozenset(
    {
        "wer",
        "cer",
        "insertion_rate",
        "deletion_rate",
        "substitution_rate",
        "clipping_ratio",
        "silence_ratio",
        "leading_silence_sec",
        "trailing_silence_sec",
        "tail_click_score",
        "vowel_prolongation_score",
    }
)
HIGHER_IS_BETTER = frozenset(
    {
        "speaker_similarity",
        "nisqa_mos",
        "nisqa_noisiness",
        "nisqa_discontinuity",
        "nisqa_coloration",
        "nisqa_loudness",
    }
)

# Compact set used for dense per-voice / cross-run tables. The single-report
# averages table still shows every numeric metric.
KEY_METRICS = (
    "wer",
    "cer",
    "insertion_rate",
    "silence_ratio",
    "clipping_ratio",
    "speaker_similarity",
    "nisqa_mos",
    "chars_per_second",
)

_DISPLAY = {
    "wer": "WER",
    "cer": "CER",
    "insertion_rate": "Insertions",
    "deletion_rate": "Deletions",
    "substitution_rate": "Substitutions",
    "clipping_ratio": "Clipping",
    "silence_ratio": "Silence",
    "leading_silence_sec": "Lead silence (s)",
    "trailing_silence_sec": "Tail silence (s)",
    "tail_click_score": "Tail-click score",
    "speaker_similarity": "Speaker similarity",
    "nisqa_mos": "NISQA MOS",
    "chars_per_second": "Chars/sec",
    "duration_sec": "Duration (s)",
    "expressiveness_proxy": "Expressiveness",
    "arousal_proxy": "Arousal",
    "vowel_prolongation_score": "Vowel prolongation",
}


# Metric groupings for the report's Metric Averages section, in display order.
METRIC_CATEGORIES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "Intelligibility",
        ("wer", "cer", "insertion_rate", "deletion_rate", "substitution_rate", "length_ratio"),
    ),
    ("Pacing & duration", ("duration_sec", "chars_per_second")),
    (
        "Audio health",
        (
            "clipping_ratio",
            "silence_ratio",
            "leading_silence_sec",
            "trailing_silence_sec",
            "tail_click_score",
            "rms",
            "peak",
        ),
    ),
    (
        "Predicted quality (MOS)",
        (
            "nisqa_mos",
            "nisqa_noisiness",
            "nisqa_discontinuity",
            "nisqa_coloration",
            "nisqa_loudness",
        ),
    ),
    ("Speaker", ("speaker_similarity",)),
    ("Expressiveness", ("expressiveness_proxy", "arousal_proxy", "vowel_prolongation_score")),
)

_CATEGORY_OF = {metric: name for name, metrics in METRIC_CATEGORIES for metric in metrics}


def category_of(metric: str) -> str:
    return _CATEGORY_OF.get(metric, "Other")


def group_by_category(metric_names: list[str]) -> list[tuple[str, list[str]]]:
    """Group metric names into display categories, preserving category order.

    Within a category, metrics follow the canonical order; unknown metrics fall
    into a trailing "Other" group sorted alphabetically.
    """
    present = set(metric_names)
    groups: list[tuple[str, list[str]]] = []
    for name, metrics in METRIC_CATEGORIES:
        members = [metric for metric in metrics if metric in present]
        if members:
            groups.append((name, members))
    others = sorted(name for name in present if name not in _CATEGORY_OF)
    if others:
        groups.append(("Other", others))
    return groups


def direction(metric: str) -> str | None:
    if metric in LOWER_IS_BETTER:
        return "lower"
    if metric in HIGHER_IS_BETTER:
        return "higher"
    return None


def is_better(metric: str, candidate: float, incumbent: float) -> bool:
    """Whether ``candidate`` is a better value than ``incumbent`` for ``metric``."""
    dir_ = direction(metric)
    if dir_ == "lower":
        return candidate < incumbent
    if dir_ == "higher":
        return candidate > incumbent
    return False


def display_name(metric: str) -> str:
    return _DISPLAY.get(metric, metric.replace("_", " "))
