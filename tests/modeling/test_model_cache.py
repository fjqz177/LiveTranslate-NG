"""Regression tests for model_cache detection logic.

Reproduces the three historical cache-layout bugs:
- ModelScope 1.38+ snapshot layout change (2026-07-11)
- HF empty-dir / interrupted-download false positives (2026-04-18)
- Qwen3-0.6B weights missing from nano models (2026-07-11)
"""

from pathlib import Path

import pytest

import livetranslate.modeling.cache as cache_module
from livetranslate.modeling.cache import (
    _hf_repo_complete,
    _ms_model_path,
    _ms_repo_complete,
    get_local_model_path,
    get_whisper_local_path,
    is_asr_cached,
    list_local_faster_whisper_models,
    qwen_weights_present,
    resolve_custom_whisper_model,
)


def _write(path, content: bytes = b"x"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


# ── ModelScope cache layouts ──


class TestModelScopeLayouts:
    @pytest.mark.parametrize(
        "layout",
        [
            "iic/SenseVoiceSmall",  # <=1.37 explicit cache_dir
            "models/iic/SenseVoiceSmall",  # 1.34~1.37 env-default
            "hub/models/iic/SenseVoiceSmall",  # hub tree
            "hub/iic/SenseVoiceSmall",  # older hub tree
        ],
    )
    def test_legacy_layouts_detected(self, tmp_path, layout):
        target = tmp_path / "modelscope" / layout
        target.mkdir(parents=True)
        assert _ms_model_path("iic", "SenseVoiceSmall", tmp_path).exists()

    def test_ms_138_snapshot_layout_detected(self, tmp_path):
        snap = tmp_path / "modelscope" / "models" / "iic--SenseVoiceSmall" / "snapshots" / "abc123"
        snap.mkdir(parents=True)
        found = _ms_model_path("iic", "SenseVoiceSmall", tmp_path)
        assert found == snap

    def test_ms_fallback_path_when_nothing_cached(self, tmp_path):
        found = _ms_model_path("iic", "SenseVoiceSmall", tmp_path)
        assert not found.exists()
        assert found.name == "SenseVoiceSmall"

    def test_is_asr_cached_funasr_via_ms_layout(self, tmp_path):
        # CORE-1: an aborted download's empty dir must NOT count as cached.
        (tmp_path / "modelscope" / "iic" / "SenseVoiceSmall").mkdir(parents=True)
        assert not is_asr_cached("funasr", "sensevoice-small", "ms", models_dir=tmp_path)

    def test_is_asr_cached_funasr_complete_snapshot(self, tmp_path):
        # A complete snapshot (>= 50MB via sparse file, zero disk usage)
        # counts as cached — both legacy and snapshots layouts.
        for layout in (
            tmp_path / "modelscope" / "iic" / "SenseVoiceSmall",
            tmp_path / "modelscope" / "models" / "iic--SenseVoiceSmall" / "snapshots" / "s1",
        ):
            layout.mkdir(parents=True)
            with (layout / "model.pt").open("wb") as f:
                f.seek(50_000_000 - 1)
                f.write(b"\x00")
            assert is_asr_cached("funasr", "sensevoice-small", "ms", models_dir=tmp_path), layout

    def test_is_asr_cached_not_cached_when_empty(self, tmp_path):
        assert not is_asr_cached("funasr", "sensevoice-small", "ms", models_dir=tmp_path)


# ── HuggingFace interrupted downloads ──


class TestHfRepoComplete:
    def test_complete_snapshot_over_min_bytes(self, tmp_path):
        _write(
            tmp_path / "huggingface" / "hub" / "models--a--b" / "snapshots" / "s1" / "model.bin",
            b"x" * 100,
        )
        assert _hf_repo_complete("a", "b", min_bytes=10, models_dir=tmp_path)

    def test_below_min_bytes_not_complete(self, tmp_path):
        _write(
            tmp_path / "huggingface" / "hub" / "models--a--b" / "snapshots" / "s1" / "model.bin",
            b"x" * 5,
        )
        assert not _hf_repo_complete("a", "b", min_bytes=10, models_dir=tmp_path)

    def test_incomplete_blob_not_complete(self, tmp_path):
        # .incomplete blobs from an aborted download must not count as cached
        _write(
            tmp_path
            / "huggingface"
            / "hub"
            / "models--a--b"
            / "snapshots"
            / "s1"
            / "model.bin.incomplete",
            b"x" * 100,
        )
        assert not _hf_repo_complete("a", "b", min_bytes=10, models_dir=tmp_path)

    def test_broken_symlink_not_complete(self, tmp_path):
        snap = tmp_path / "huggingface" / "hub" / "models--a--b" / "snapshots" / "s1"
        snap.mkdir(parents=True)
        try:
            (snap / "blob").symlink_to(tmp_path / "does-not-exist")
        except (OSError, NotImplementedError):
            pytest.skip("symlinks unavailable on this system")
        assert not _hf_repo_complete("a", "b", min_bytes=10, models_dir=tmp_path)

    def test_empty_snapshot_dir_not_complete(self, tmp_path):
        (tmp_path / "huggingface" / "hub" / "models--a--b" / "snapshots" / "s1").mkdir(parents=True)
        assert not _hf_repo_complete("a", "b", models_dir=tmp_path)


# ── Qwen weights for nano models ──


class TestQwenWeights:
    def test_no_qwen_subdir_counts_as_present(self, tmp_path):
        assert qwen_weights_present(tmp_path)

    def test_empty_qwen_subdir_missing(self, tmp_path):
        (tmp_path / "Qwen3-0.6B").mkdir()
        assert not qwen_weights_present(tmp_path)

    def test_qwen_safetensors_present(self, tmp_path):
        _write(tmp_path / "Qwen3-0.6B" / "model.safetensors")
        assert qwen_weights_present(tmp_path)

    def test_nano_not_cached_without_qwen_weights(self, tmp_path):
        (tmp_path / "modelscope" / "FunAudioLLM" / "Fun-ASR-Nano-2512").mkdir(parents=True)
        (tmp_path / "modelscope" / "FunAudioLLM" / "Fun-ASR-Nano-2512" / "Qwen3-0.6B").mkdir()
        assert not is_asr_cached("funasr", "funasr-nano-2512", "ms", models_dir=tmp_path)

    def test_nano_cached_with_qwen_weights(self, tmp_path):
        model_dir = tmp_path / "modelscope" / "FunAudioLLM" / "Fun-ASR-Nano-2512"
        model_dir.mkdir(parents=True)
        # Base model must itself be complete (CORE-1) before the Qwen gate.
        with (model_dir / "model.pt").open("wb") as f:
            f.seek(50_000_000 - 1)
            f.write(b"\x00")
        _write(model_dir / "Qwen3-0.6B" / "model.safetensors")
        assert is_asr_cached("funasr", "funasr-nano-2512", "ms", models_dir=tmp_path)


# ── Anime-Whisper ──


class TestAnimeWhisper:
    def test_cached_with_weights_and_config(self, tmp_path):
        snap = (
            tmp_path / "huggingface" / "hub" / "models--litagin--anime-whisper" / "snapshots" / "s1"
        )
        _write(snap / "model.safetensors")
        _write(snap / "config.json")
        assert is_asr_cached("anime-whisper", hub="hf", models_dir=tmp_path)

    def test_not_cached_without_weights(self, tmp_path):
        snap = (
            tmp_path / "huggingface" / "hub" / "models--litagin--anime-whisper" / "snapshots" / "s1"
        )
        _write(snap / "config.json")
        assert not is_asr_cached("anime-whisper", hub="hf", models_dir=tmp_path)


# ── Whisper ModelScope cache (hub-aware download/caching) ──


@pytest.fixture
def tiny_min_bytes(monkeypatch):
    """Shrink the whisper-tiny size table so tests only write tiny files."""
    monkeypatch.setattr(
        cache_module,
        "MODEL_SIZE_BYTES",
        {"whisper-tiny": 20},  # min_bytes = 10
    )


class TestWhisperMsCache:
    def test_cached_via_modelscope_flat(self, tmp_path, tiny_min_bytes):
        _write(
            tmp_path / "modelscope" / "Systran" / "faster-whisper-tiny" / "model.bin",
            b"x" * 20,
        )
        assert is_asr_cached("whisper", "tiny", "ms", models_dir=tmp_path)

    def test_cached_via_ms_138_snapshot_layout(self, tmp_path, tiny_min_bytes):
        snap = (
            tmp_path
            / "modelscope"
            / "models"
            / "Systran--faster-whisper-tiny"
            / "snapshots"
            / "abc123"
        )
        _write(snap / "model.bin", b"x" * 20)
        assert is_asr_cached("whisper", "tiny", "ms", models_dir=tmp_path)

    def test_not_cached_when_ms_partial(self, tmp_path, tiny_min_bytes):
        _write(
            tmp_path / "modelscope" / "Systran" / "faster-whisper-tiny" / "model.bin",
            b"x" * 5,
        )
        assert not is_asr_cached("whisper", "tiny", "ms", models_dir=tmp_path)

    def test_cached_via_hf_still_works(self, tmp_path, tiny_min_bytes):
        _write(
            tmp_path
            / "huggingface"
            / "hub"
            / "models--Systran--faster-whisper-tiny"
            / "snapshots"
            / "s1"
            / "model.bin",
            b"x" * 20,
        )
        assert is_asr_cached("whisper", "tiny", "hf", models_dir=tmp_path)

    def test_ms_cache_accepted_regardless_of_hub(self, tmp_path, tiny_min_bytes):
        # Either hub's cache counts, so switching download source never
        # triggers a duplicate download (same policy as funasr).
        _write(
            tmp_path / "modelscope" / "Systran" / "faster-whisper-tiny" / "model.bin",
            b"x" * 20,
        )
        assert is_asr_cached("whisper", "tiny", "hf", models_dir=tmp_path)
        assert is_asr_cached("whisper", "tiny", "ms", models_dir=tmp_path)

    def test_custom_model_size_still_resolves(self, tmp_path):
        # Custom paths resolve relative to models_dir.parent (the app root).
        app = tmp_path / "app"
        _write(app / "my-whisper" / "model.bin")
        _write(app / "my-whisper" / "config.json")
        assert is_asr_cached("whisper", "my-whisper", "ms", models_dir=app / "models")

    def test_ms_repo_complete_direct(self, tmp_path):
        _write(
            tmp_path / "modelscope" / "Systran" / "faster-whisper-tiny" / "model.bin",
            b"x" * 15,
        )
        assert _ms_repo_complete(
            "Systran", "faster-whisper-tiny", min_bytes=10, models_dir=tmp_path
        )
        assert not _ms_repo_complete(
            "Systran", "faster-whisper-tiny", min_bytes=50, models_dir=tmp_path
        )


class TestGetWhisperLocalPath:
    def test_ms_preferred(self, tmp_path, tiny_min_bytes):
        ms = tmp_path / "modelscope" / "Systran" / "faster-whisper-tiny"
        _write(ms / "model.bin", b"x" * 20)
        result = get_whisper_local_path("tiny", hub="ms", models_dir=tmp_path)
        assert Path(result) == ms

    def test_hf_fallback(self, tmp_path, tiny_min_bytes):
        snap = (
            tmp_path
            / "huggingface"
            / "hub"
            / "models--Systran--faster-whisper-tiny"
            / "snapshots"
            / "s1"
        )
        _write(snap / "model.bin", b"x" * 20)
        result = get_whisper_local_path("tiny", hub="ms", models_dir=tmp_path)
        assert Path(result) == snap

    def test_none_when_missing(self, tmp_path, tiny_min_bytes):
        assert get_whisper_local_path("tiny", hub="ms", models_dir=tmp_path) is None

    def test_none_for_custom_path(self, tmp_path, tiny_min_bytes):
        assert get_whisper_local_path("my-whisper", hub="ms", models_dir=tmp_path) is None

    def test_incomplete_ms_dir_ignored(self, tmp_path, tiny_min_bytes):
        _write(
            tmp_path / "modelscope" / "Systran" / "faster-whisper-tiny" / "model.bin",
            b"x" * 5,
        )
        assert get_whisper_local_path("tiny", hub="ms", models_dir=tmp_path) is None


# ── get_local_model_path ──


class TestGetLocalModelPath:
    def test_ms_preferred(self, tmp_path):
        ms = tmp_path / "modelscope" / "iic" / "SenseVoiceSmall"
        ms.mkdir(parents=True)
        hf_snap = (
            tmp_path
            / "huggingface"
            / "hub"
            / "models--FunAudioLLM--SenseVoiceSmall"
            / "snapshots"
            / "s1"
        )
        hf_snap.mkdir(parents=True)
        result = get_local_model_path(
            "funasr", hub="ms", funasr_model="sensevoice-small", models_dir=tmp_path
        )
        assert Path(result) == ms

    def test_hf_fallback(self, tmp_path):
        hf_snap = (
            tmp_path
            / "huggingface"
            / "hub"
            / "models--FunAudioLLM--SenseVoiceSmall"
            / "snapshots"
            / "s1"
        )
        hf_snap.mkdir(parents=True)
        result = get_local_model_path(
            "funasr", hub="ms", funasr_model="sensevoice-small", models_dir=tmp_path
        )
        assert Path(result) == hf_snap

    def test_none_when_missing(self, tmp_path):
        assert (
            get_local_model_path(
                "funasr", hub="ms", funasr_model="sensevoice-small", models_dir=tmp_path
            )
            is None
        )


# ── Local faster-whisper discovery ──


class TestLocalWhisperDiscovery:
    def test_finds_custom_model_dir(self, tmp_path):
        _write(tmp_path / "my-whisper" / "model.bin")
        _write(tmp_path / "my-whisper" / "config.json")
        entries = list_local_faster_whisper_models(tmp_path)
        assert [e["name"] for e in entries] == ["my-whisper"]

    def test_ignores_builtin_cache(self, tmp_path):
        _write(
            tmp_path
            / "huggingface"
            / "hub"
            / "models--Systran--faster-whisper-tiny"
            / "snapshots"
            / "s1"
            / "model.bin"
        )
        _write(
            tmp_path
            / "huggingface"
            / "hub"
            / "models--Systran--faster-whisper-tiny"
            / "snapshots"
            / "s1"
            / "config.json"
        )
        assert list_local_faster_whisper_models(tmp_path) == []

    def test_ignores_modelscope_builtin_cache(self, tmp_path):
        # ModelScope downloads of builtin sizes must not show up as user
        # "local" whisper models in the size combo.
        ms = tmp_path / "modelscope" / "Systran" / "faster-whisper-tiny"
        _write(ms / "model.bin")
        _write(ms / "config.json")
        assert list_local_faster_whisper_models(tmp_path) == []

    def test_requires_both_bin_and_config(self, tmp_path):
        _write(tmp_path / "half" / "model.bin")
        assert list_local_faster_whisper_models(tmp_path) == []

    def test_resolve_custom_relative_path(self, tmp_path):
        # Mirror production layout: models/ sits next to the custom dir,
        # relative paths resolve against models_dir.parent (the app root).
        app = tmp_path / "app"
        models_dir = app / "models"
        _write(app / "rel-whisper" / "model.bin")
        _write(app / "rel-whisper" / "config.json")
        resolved = resolve_custom_whisper_model("rel-whisper", models_dir=models_dir)
        assert resolved == str((app / "rel-whisper").resolve())

    def test_resolve_builtin_size_returns_none(self, tmp_path):
        assert resolve_custom_whisper_model("tiny", models_dir=tmp_path) is None
