from __future__ import annotations

import re

NEG_PATTERNS = [
    r"\bnon[-_\s]?defective\b",
    r"\bnot\s+defective\b",
    r"\bclean\b",
    r"\bsafe\b",
    r"\bnegative\b",
    r"\b0\b",
]
POS_PATTERNS = [
    r"\bdefective\b",
    r"\bbuggy\b",
    r"\bbug\b",
    r"\bvulnerable\b",
    r"\bpositive\b",
    r"\b1\b",
]


def normalize_label(text: str, default: int = 0) -> int:
    """Map generated text to 0=non-defective or 1=defective.

    Negative patterns are checked first because "non-defective" contains the substring
    "defective". This avoids the common substring-normalization bug.
    """
    s = (text or "").strip().lower()
    s = s.replace("_", "-")
    first_line = s.splitlines()[0] if s else ""

    for pattern in NEG_PATTERNS:
        if re.search(pattern, first_line):
            return 0
    for pattern in POS_PATTERNS:
        if re.search(pattern, first_line):
            return 1
    for pattern in NEG_PATTERNS:
        if re.search(pattern, s):
            return 0
    for pattern in POS_PATTERNS:
        if re.search(pattern, s):
            return 1
    return default


def is_conforming_output(text: str) -> bool:
    cleaned = (text or "").strip().lower().strip("` .,:;\"'")
    return cleaned in {"defective", "non-defective"}
