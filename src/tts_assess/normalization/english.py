from __future__ import annotations

import re

_CONTRACTIONS = {
    "can't": "cannot",
    "won't": "will not",
    "n't": " not",
    "'re": " are",
    "'ve": " have",
    "'ll": " will",
    "'d": " would",
    "'m": " am",
}

_NUMBERS_0_TO_19 = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
]
_TENS = {
    20: "twenty",
    30: "thirty",
    40: "forty",
    50: "fifty",
    60: "sixty",
    70: "seventy",
    80: "eighty",
    90: "ninety",
}


def normalize_english(text: str, *, expand_numbers: bool = True) -> str:
    text = (
        text.replace("’", "'")
        .replace("‘", "'")
        .replace("`", "'")
        .replace("“", '"')
        .replace("”", '"')
        .lower()
        .strip()
    )
    for source, replacement in _CONTRACTIONS.items():
        text = text.replace(source, replacement)
    # Fold intra-word abbreviation dots so "a.m." -> "am" and "u.s." -> "us"
    # instead of splitting into "a m" when punctuation is later stripped. This
    # removes a common source of spurious WER against ASR output.
    text = re.sub(r"(?<=[a-z])\.(?=[a-z])", "", text)
    # Split glued digit/letter runs so "9am" and "9 AM" normalize the same way.
    text = re.sub(r"(?<=\d)(?=[a-z])", " ", text)
    text = re.sub(r"(?<=[a-z])(?=\d)", " ", text)
    # Normalize common time markers to a single token ("a m" / "am" -> "am").
    text = re.sub(r"\b([ap])\s+m\b", r"\1m", text)
    if expand_numbers:
        text = re.sub(r"\b\d+\b", lambda match: _number_to_words(int(match.group(0))), text)
    text = re.sub(r"[$€£]", " ", text)
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _number_to_words(value: int) -> str:
    if value < 20:
        return _NUMBERS_0_TO_19[value]
    if value < 100:
        tens = value // 10 * 10
        remainder = value % 10
        return _TENS[tens] if remainder == 0 else f"{_TENS[tens]} {_NUMBERS_0_TO_19[remainder]}"
    if value < 1000:
        hundreds = value // 100
        remainder = value % 100
        prefix = f"{_NUMBERS_0_TO_19[hundreds]} hundred"
        return prefix if remainder == 0 else f"{prefix} {_number_to_words(remainder)}"
    return " ".join(_NUMBERS_0_TO_19[int(digit)] for digit in str(value))
