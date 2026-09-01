import pytest

from livetranslate.core.translator import DEFAULT_PROMPT, Translator


class TestCheckRepetition:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("", False),
            ("short text", False),
            ("abcdefgh" * 4, False),  # 32 chars, below the 40-char threshold
            ("abcdefgh" * 5, True),  # 40 chars, 8-char pattern repeats
            (
                "the quick brown fox jumps over the lazy dog",
                False,  # 43 chars, no repeated pattern
            ),
        ],
    )
    def test_repetition_detection(self, text, expected):
        assert Translator._check_repetition(text) is expected


class TestExtractJsonTranslation:
    @staticmethod
    def _make() -> Translator:
        return Translator(
            api_base="http://127.0.0.1:9/v1",
            api_key="test-key",
            model="test-model",
        )

    def test_valid_json(self):
        assert self._make()._extract_json_translation('{"t": "hello"}') == "hello"

    def test_malformed_json_returns_raw(self):
        assert self._make()._extract_json_translation("not json") == "not json"

    def test_json_without_t_key_returns_raw(self):
        assert self._make()._extract_json_translation('{"other": 1}') == '{"other": 1}'

    def test_non_dict_json_returns_raw(self):
        assert self._make()._extract_json_translation("[1, 2]") == "[1, 2]"


class TestPromptFallback:
    def _make(self, system_prompt=None):
        return Translator(
            api_base="http://127.0.0.1:9/v1",
            api_key="test-key",
            model="test-model",
            system_prompt=system_prompt,
        )

    def test_bad_template_falls_back_to_default(self):
        t = self._make(system_prompt="bad {nonexistent_key}")
        prompt = t._build_system_prompt("en")
        assert "English" in prompt
        assert "nonexistent" not in prompt

    def test_valid_template_formats_placeholders(self):
        t = self._make(system_prompt="TR {source_lang}->{target_lang}")
        assert t._build_system_prompt("ja") == "TR Japanese->Chinese"

    def test_default_prompt_formats_placeholders(self):
        t = self._make()
        prompt = t._build_system_prompt("en")
        assert "English" in prompt
        assert "Chinese" in prompt
        assert "{source_lang}" not in prompt
        assert "You are a real-time subtitle translator" in DEFAULT_PROMPT
