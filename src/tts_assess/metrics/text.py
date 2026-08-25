from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass

import jiwer


@dataclass(frozen=True)
class TextMetrics:
    wer: float
    cer: float
    insertion_rate: float
    deletion_rate: float
    substitution_rate: float
    empty_transcript: bool
    tail_hallucination: bool
    repeated_span: bool
    length_ratio: float | None

    def to_dict(self) -> dict[str, float | bool | None]:
        return asdict(self)


def compute_text_metrics(expected: str, transcript: str) -> TextMetrics:
    expected = expected.strip()
    transcript = transcript.strip()
    word_measures = jiwer.process_words(expected, transcript)
    char_measures = jiwer.process_characters(expected, transcript)
    ref_words = max(1, len(expected.split()))
    return TextMetrics(
        wer=float(word_measures.wer),
        cer=float(char_measures.cer),
        insertion_rate=float(word_measures.insertions / ref_words),
        deletion_rate=float(word_measures.deletions / ref_words),
        substitution_rate=float(word_measures.substitutions / ref_words),
        empty_transcript=not bool(transcript),
        tail_hallucination=_tail_hallucination(expected, transcript),
        repeated_span=_has_repeated_span(expected, transcript),
        length_ratio=(len(transcript) / len(expected)) if expected else None,
    )


def _tail_hallucination(expected: str, transcript: str) -> bool:
    expected_words = expected.split()
    transcript_words = transcript.split()
    if len(transcript_words) <= len(expected_words) + 3:
        return False
    prefix = transcript_words[: len(expected_words)]
    prefix_error = jiwer.wer(" ".join(expected_words), " ".join(prefix)) if expected_words else 1.0
    extra_ratio = (len(transcript_words) - len(expected_words)) / max(1, len(expected_words))
    return prefix_error <= 0.25 and extra_ratio >= 0.25


def _has_repeated_span(expected: str, transcript: str) -> bool:
    '''Whether ASR added an adjacent repetition that is absent from the reference.'''
    expected_repeats = _adjacent_repeat_counts(expected)
    transcript_repeats = _adjacent_repeat_counts(transcript)
    return any(count > expected_repeats[span] for span, count in transcript_repeats.items())


def _adjacent_repeat_counts(text: str) -> Counter[tuple[str, ...]]:
    words = text.split()
    repeats: Counter[tuple[str, ...]] = Counter()
    for size in range(1, min(6, len(words) // 2 + 1)):
        for start in range(0, len(words) - size * 2 + 1):
            span = tuple(words[start : start + size])
            if span == tuple(words[start + size : start + size * 2]):
                repeats[span] += 1
    return repeats
