import pytest

from livetranslate.modeling.registry import (
    DEFAULT_FUNASR_MODEL,
    WHISPER_SIZES,
    asr_model_id,
    funasr_display_name,
    funasr_model_id,
    funasr_model_options,
    funasr_profile,
    funasr_supports_padding,
    migrate_funasr_settings,
    normalize_asr_engine_selection,
    normalize_funasr_model_key,
)


class TestNormalizeFunasrModelKey:
    def test_known_key_passthrough(self):
        assert normalize_funasr_model_key("sensevoice-small") == "sensevoice-small"

    def test_legacy_alias(self):
        assert normalize_funasr_model_key("sensevoice") == "sensevoice-small"
        assert normalize_funasr_model_key("funasr-nano") == "funasr-nano-2512"

    @pytest.mark.parametrize("bad", [None, "", "unknown-model", 42])
    def test_unknown_falls_back_to_default(self, bad):
        assert normalize_funasr_model_key(bad) == DEFAULT_FUNASR_MODEL


class TestNormalizeAsrEngineSelection:
    def test_legacy_engine_merges_into_funasr(self):
        assert normalize_asr_engine_selection("sensevoice", None) == (
            "funasr",
            "sensevoice-small",
        )

    def test_funasr_engine_with_model(self):
        assert normalize_asr_engine_selection("funasr", "funasr-nano-2512") == (
            "funasr",
            "funasr-nano-2512",
        )

    def test_other_engines_passthrough(self):
        assert normalize_asr_engine_selection("whisper", None) == (
            "whisper",
            DEFAULT_FUNASR_MODEL,
        )

    def test_none_defaults_to_funasr(self):
        assert normalize_asr_engine_selection(None, None) == (
            "funasr",
            DEFAULT_FUNASR_MODEL,
        )


class TestMigrateFunasrSettings:
    def test_legacy_engine_rewritten(self):
        s = {"asr_engine": "sensevoice"}
        result = migrate_funasr_settings(s)
        assert result is s  # in-place migration
        assert s["asr_engine"] == "funasr"
        assert s["funasr_model"] == "sensevoice-small"

    def test_non_funasr_engine_gets_default_model_key(self):
        s = {"asr_engine": "whisper"}
        migrate_funasr_settings(s)
        assert s["asr_engine"] == "whisper"
        assert s["funasr_model"] == DEFAULT_FUNASR_MODEL

    def test_none_settings_returns_none(self):
        assert migrate_funasr_settings(None) is None

    def test_existing_model_key_preserved(self):
        s = {"asr_engine": "funasr", "funasr_model": "funasr-mlt-nano-2512"}
        migrate_funasr_settings(s)
        assert s["funasr_model"] == "funasr-mlt-nano-2512"


class TestProfiles:
    def test_profile_lookup(self):
        p = funasr_profile("sensevoice-small")
        assert p["family"] == "sensevoice"
        assert p["supports_padding"] is True

    def test_display_name(self):
        assert funasr_display_name("funasr-nano-2512") == "Fun-ASR-Nano"

    def test_supports_padding_differs_per_model(self):
        assert funasr_supports_padding("sensevoice-small") is True
        assert funasr_supports_padding("funasr-nano-2512") is False

    def test_model_options_cover_all_profiles(self):
        options = funasr_model_options()
        assert len(options) == 3
        assert next(k for k, _ in options) == "sensevoice-small"


class TestRepoIds:
    def test_funasr_ids_per_hub(self):
        assert funasr_model_id("sensevoice-small", "ms") == "iic/SenseVoiceSmall"
        assert funasr_model_id("sensevoice-small", "hf") == "FunAudioLLM/SenseVoiceSmall"

    def test_asr_model_id_hf_namespace_difference(self):
        assert asr_model_id("sensevoice", "ms") == "iic/SenseVoiceSmall"
        assert asr_model_id("sensevoice", "hf") == "FunAudioLLM/SenseVoiceSmall"

    def test_asr_model_id_anime_whisper(self):
        assert asr_model_id("anime-whisper", "hf") == "litagin/anime-whisper"

    def test_whisper_sizes_list(self):
        assert WHISPER_SIZES == ["tiny", "base", "small", "medium", "large-v3"]


class TestSensevoiceOnnxModeling:
    def test_display_name_and_size_registered(self):
        from livetranslate.modeling.registry import ASR_DISPLAY_NAMES, MODEL_SIZE_BYTES

        # M-MATRIX: sensevoice-onnx is the recommended base-only CPU engine;
        # its display name/size feed the GUI and the cache gates.
        assert ASR_DISPLAY_NAMES["sensevoice-onnx"] == "SenseVoice (ONNX)"
        assert MODEL_SIZE_BYTES["sensevoice-onnx"] == 900_000_000
