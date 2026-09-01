"""Sentence splitting and text-cleaning helpers for the ASR pipeline.

Pure text logic: no Qt, no torch, no I/O — unit-testable in isolation.
"""

import logging
from typing import Any

log = logging.getLogger("LiveTranslate.Text")

_pysbd_cache: dict[str, Any] = {}  # lang -> pysbd.Segmenter


def _get_segmenter(lang: str) -> Any:
    import pysbd  # type: ignore[import-untyped]  # pysbd is unmaintained and ships no stubs

    if lang not in _pysbd_cache:
        pysbd_lang = lang if lang in pysbd.languages.LANGUAGE_CODES else "en"
        _pysbd_cache[lang] = pysbd.Segmenter(language=pysbd_lang, clean=False)
    return _pysbd_cache[lang]


def split_sentences(text: str, lang: str = "en") -> list[str]:
    """Split text into sentences using pysbd, with comma fallback for long text."""
    seg = _get_segmenter(lang)
    parts: list[str] = [p for p in seg.segment(text) if p.strip()]
    if len(parts) > 1:
        return parts

    # Comma fallback for long unsplit text — split at last balanced comma
    # CJK 「、」at 25 chars; all commas at 60 chars (long sentence, reduce latency)
    min_len = 25 if any(c == "、" for c in text) else 60
    if len(text) > min_len:
        for i in range(len(text) - 8, 5, -1):
            if text[i] in ",，;；、":  # noqa: RUF001
                before = text[: i + 1].strip()
                after = text[i + 1 :].strip()
                if before and after and len(before) > 15 and len(after) > 3:
                    return [before, after]

    return parts


def is_short_utterance(text: str) -> bool:
    """Check if text has ≤8 alphanumeric chars (likely noise/filler/fragment)."""
    alnum = sum(1 for c in text if c.isalnum())
    return alnum <= 8


def strip_committed_overlap(text: str, committed_tail: str) -> str:
    """Remove text that overlaps with previously committed content."""
    if not committed_tail:
        return text
    tail = committed_tail.lower().rstrip()
    text_lower = text.lower()
    # Check if text starts with a suffix of the committed tail
    max_check = min(len(tail), len(text_lower))
    for overlap_len in range(max_check, 2, -1):
        if text_lower[:overlap_len] == tail[-overlap_len:]:
            stripped = text[overlap_len:].strip()
            if stripped:
                log.debug(f"Stripped echo overlap ({overlap_len} chars): '{text[:overlap_len]}...'")
                return stripped
            return ""
    return text
