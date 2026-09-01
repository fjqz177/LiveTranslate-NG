"""Tests for the ASR engine registry and per-platform recommendation.

M-MATRIX (2026-08-31): EngineSpec.engine_type is the single source of truth for
the worker engine type (= asr/worker._ENGINE_FACTORIES key = persisted
settings['asr_engine'] = worker_config['engine_type']). funasr-nano is no
longer a standalone ENGINE entry (Nanos live as MODELS under the legacy
"funasr" engine).
"""

from livetranslate.asr.registry import (
    ENGINE_REGISTRY,
    GUI_ENGINE_ORDER,
    engine_id_for_type,
    engine_type_for_engine,
    engines_for_platform,
    recommend_engine,
)
from livetranslate.platform.system import AcceleratorInfo


class TestRegistry:
    def test_all_planned_engines_present(self):
        assert set(ENGINE_REGISTRY) == {
            "faster-whisper",
            "sensevoice-funasr",
            "sensevoice-onnx",
            "anime-whisper",
            "remote",
        }

    def test_extras_only_declared_where_needed(self):
        assert ENGINE_REGISTRY["sensevoice-onnx"].extras == (
            "engine-sensevoice-onnx",
        )  # numpy fbank frontend lives in base; the ONNX model gates status
        assert ENGINE_REGISTRY["faster-whisper"].extras == ("engine-whisper",)

    def test_every_engine_supports_at_least_one_platform(self):
        for spec in ENGINE_REGISTRY.values():
            assert spec.platforms

    def test_engine_type_is_source_of_truth(self):
        """M-MATRIX: engine_type must be exactly the worker factory key
        (or remote-whisper for the in-process remote client)."""
        expected = {
            "faster-whisper": "whisper",
            "sensevoice-funasr": "funasr",
            "sensevoice-onnx": "sensevoice-onnx",
            "anime-whisper": "anime-whisper",
            "remote": "remote-whisper",
        }
        assert {eid: spec.engine_type for eid, spec in ENGINE_REGISTRY.items()} == expected

    def test_engine_type_matches_worker_factories(self):
        """Every engine_type is loadable by the ASR worker (_ENGINE_FACTORIES)
        or is the in-process remote client (remote-whisper)."""
        from livetranslate.asr.worker import _ENGINE_FACTORIES

        loadable = set(_ENGINE_FACTORIES) | {"remote-whisper"}
        for engine_id, spec in ENGINE_REGISTRY.items():
            assert spec.engine_type in loadable, f"{engine_id} -> {spec.engine_type!r}"

    def test_worker_ready_matches_worker_factories(self):
        """ASR-1: worker_ready must mirror asr/worker._ENGINE_FACTORIES
        (plus the in-process remote engine). A declared-but-unwired engine
        must be flagged, never selectable as if the extras were merely missing."""
        from livetranslate.asr.worker import _ENGINE_FACTORIES

        loadable = set(_ENGINE_FACTORIES) | {"remote-whisper"}
        for engine_id, spec in ENGINE_REGISTRY.items():
            if spec.engine_type in loadable:
                assert spec.worker_ready, f"{engine_id} has a factory but is not worker_ready"
            else:
                assert not spec.worker_ready, f"{engine_id} claims worker_ready but has none"

    def test_no_funasr_nano_standalone_engine(self):
        """M-MATRIX: funasr-nano is a phantom standalone ENGINE entry (no
        engine-nano extra, no worker factory). Nanos stay as MODELS under the
        legacy "funasr" engine (see modeling/registry.FUNASR_MODEL_PROFILES)."""
        assert "funasr-nano" not in ENGINE_REGISTRY


class TestGuiOrderAndHelpers:
    def test_gui_engine_order(self):
        assert GUI_ENGINE_ORDER == (
            "faster-whisper",
            "sensevoice-onnx",
            "sensevoice-funasr",
            "anime-whisper",
            "remote",
        )

    def test_engines_for_platform(self):
        assert engines_for_platform("win32") == list(ENGINE_REGISTRY)
        assert engines_for_platform("linux") == []

    def test_engine_type_for_engine(self):
        assert engine_type_for_engine("faster-whisper") == "whisper"
        assert engine_type_for_engine("sensevoice-onnx") == "sensevoice-onnx"
        assert engine_type_for_engine("remote") == "remote-whisper"

    def test_engine_id_for_type_round_trips_legacy_values(self):
        """M-MATRIX: persisted settings['asr_engine'] values resolve back
        exactly (one-to-one with worker factories / remote-whisper)."""
        assert engine_id_for_type("whisper") == "faster-whisper"
        assert engine_id_for_type("funasr") == "sensevoice-funasr"
        assert engine_id_for_type("sensevoice-onnx") == "sensevoice-onnx"
        assert engine_id_for_type("anime-whisper") == "anime-whisper"
        assert engine_id_for_type("remote-whisper") == "remote"

    def test_engine_id_for_type_unknown_and_unselectable_return_none(self):
        """Choice: unknown type or type not selectable on the platform returns
        None (not KeyError) so legacy/mismatched persisted values degrade
        gracefully; callers decide the fallback."""
        assert engine_id_for_type("no-such-type") is None
        assert engine_id_for_type("whisper", platform="linux") is None


class TestRecommend:
    def test_windows_cuda(self):
        assert recommend_engine(AcceleratorInfo("cuda", "RTX 4090")) == "faster-whisper"

    def test_windows_cpu(self):
        assert recommend_engine(AcceleratorInfo("cpu")) == "faster-whisper"
