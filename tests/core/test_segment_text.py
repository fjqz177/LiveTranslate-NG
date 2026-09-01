import pytest

from livetranslate.core.segment_text import (
    is_short_utterance,
    split_sentences,
    strip_committed_overlap,
)


class TestSplitSentences:
    def test_pysbd_splits_on_period(self):
        parts = split_sentences("Hello there. This is another sentence.", "en")
        assert len(parts) == 2
        assert parts[0].startswith("Hello there")

    def test_short_text_returns_single_part(self):
        assert split_sentences("short text", "en") == ["short text"]

    def test_unknown_lang_falls_back_to_english_segmenter(self):
        parts = split_sentences("One. Two.", "xx")
        assert len(parts) == 2

    def test_cjk_comma_fallback_at_25_chars(self):
        text = "A" * 16 + "\u3001" + "B" * 20  # 37 chars, contains 、
        parts = split_sentences(text, "ja")
        assert len(parts) == 2
        assert parts[0].endswith("\u3001")
        assert parts[1] == "B" * 20

    def test_western_comma_fallback_at_60_chars(self):
        text = "A" * 40 + "," + "B" * 30  # 71 chars
        parts = split_sentences(text, "en")
        assert len(parts) == 2
        assert parts[0].endswith(",")

    def test_no_fallback_below_threshold(self):
        # 40 chars but no comma of any kind -> single part
        assert split_sentences("A" * 40, "en") == ["A" * 40]

    def test_comma_fallback_rejects_unbalanced_sides(self):
        # before is only 10 chars (must be > 15) -> no split
        text = "A" * 10 + "," + "B" * 55
        assert split_sentences(text, "en") == [text]


class TestIsShortUtterance:
    @pytest.mark.parametrize(
        "text",
        ["", "ok", "嗯嗯", "12345678", "a" * 8],
    )
    def test_short(self, text):
        assert is_short_utterance(text) is True

    @pytest.mark.parametrize(
        "text",
        ["thank you very much", "a" * 9, "123456789"],
    )
    def test_not_short(self, text):
        assert is_short_utterance(text) is False


class TestStripCommittedOverlap:
    def test_no_tail_returns_text(self):
        assert strip_committed_overlap("hello world", "") == "hello world"

    def test_no_overlap_returns_text(self):
        assert (
            strip_committed_overlap("brand new sentence", "unrelated tail") == "brand new sentence"
        )

    def test_strips_matching_tail_suffix(self):
        result = strip_committed_overlap("is committed and more", "this is committed")
        assert result == "and more"

    def test_case_insensitive(self):
        result = strip_committed_overlap("committed rest", "COMMITTED")
        assert result == "rest"

    def test_full_overlap_returns_empty(self):
        assert strip_committed_overlap("committed", "committed") == ""

    def test_overlap_shorter_than_3_chars_ignored(self):
        # 2-char suffix match is not stripped (loop lower bound is 2)
        assert strip_committed_overlap("ab hello", "xxab") == "ab hello"
