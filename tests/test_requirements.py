from pathlib import Path

import pytest


def _pyproject() -> str:
    return Path("pyproject.toml").read_text(encoding="utf-8").lower()


def _deps_block() -> str:
    """Return the [project] dependencies section only (not the extras)."""
    text = _pyproject()
    start = text.index("dependencies = [")
    end = text.index("]", start)
    # Only dependency strings count — comments inside the block (which may
    # legitimately mention "torch-free") must not pollute substring checks.
    return "\n".join(
        line.strip() for line in text[start:end].splitlines() if line.strip().startswith('"')
    )


def _extras_block() -> str:
    """The extras table only — cut off at the next TOML section so the
    dependency-groups below cannot pollute the assertions."""
    text = _pyproject()
    start = text.index("[project.optional-dependencies]")
    end = text.find("\n[", start + 1)
    if end == -1:
        end = len(text)
    return text[start:end]


def test_base_dependencies_are_torch_free():
    """The base install must not pull any GPU framework; engines own their
    backends via extras so the GUI/CLI can run on plain CPU wheels."""
    deps = _deps_block()

    assert "torch" not in deps
    assert "torchaudio" not in deps
    assert "funasr" not in deps
    assert "silero-vad" not in deps
    assert "faster-whisper" not in deps
    assert "onnxruntime" in deps


def test_base_dependencies_are_hub_sdk_free():
    """Model downloads use the lightweight httpx client (hub_downloader.py);
    the SDKs must not re-enter base — they pulled ~130 MB of transitive deps
    (transformers/jieba/...) into the frozen bundle (SelfServe P0-A2)."""
    deps = _deps_block()
    assert "modelscope" not in deps
    assert "huggingface-hub" not in deps
    assert "huggingface_hub" not in deps


def test_engine_extras_carry_their_backends():
    extras = _extras_block()

    assert "engine-whisper" in extras
    assert "engine-funasr" in extras
    assert "engine-torch-vad" not in extras  # VAD moved to onnxruntime (Phase 2)
    assert "funasr==1.4.2" in extras
    assert "faster-whisper>=1.0,<2" in extras
    assert '"silero-vad' not in extras  # onnx model loads at runtime, no torch dep


def test_torch_stays_a_matched_pair_inside_extras():
    """torch/torchaudio pins must appear together wherever torch is declared."""
    extras = _extras_block()
    assert "torch==2.11.0" in extras
    assert "torchaudio==2.11.0" in extras
    # engine-cuda provides the CUDA torch and engine-funasr carries
    # torch+torchaudio together. torch may appear in both groups; only the
    # pairing invariant holds. sensevoice-onnx is a numpy fbank port (P0-A3),
    # so torchaudio appears exactly once.
    assert extras.count('"torchaudio==2.11.0"') == 1
    assert extras.count('"torch==2.11.0"') >= 1


def test_sensevoice_onnx_extra_is_empty():
    """The numpy fbank port removed torchaudio from the sensevoice-onnx path
    (SelfServe P0-A3): the extra must stay empty so base stays torch-free and
    the frozen bundle can run SenseVoice offline."""
    extras = _extras_block()
    assert "engine-sensevoice-onnx = []" in extras


def test_windows_audio_capture_is_platform_marked():
    """PyAudioWPatch ships Windows-only wheels; the marker keeps macOS/Linux
    resolution viable."""
    deps = _deps_block()
    assert "pyaudiowpatch==0.2.12.8" in deps
    assert "sys_platform == 'win32'" in deps


def test_critical_dependency_pins():
    """Major-version drift on the HTTP/UI stack stays blocked."""
    deps = _deps_block()

    assert "openai>=3.0.0" in deps
    assert "numpy>=2.2.6" in deps
    assert "pyqt6>=6.5.0,<7" in deps
    assert "pysbd>=0.3.4,<1" in deps


def test_funasr_uses_published_dependency_metadata():
    """Install a current FunASR normally instead of copying its dependencies."""
    extras = _extras_block()

    assert "hydra-core>=1.3.2" in extras
    assert "soundfile>=0.12.1" in extras
    assert "numba>=0.59.0" in extras
    assert "editdistance-s" not in extras


@pytest.mark.parametrize("readme", ["README.md", "README_en.md"])
def test_readmes_do_not_describe_the_removed_editdistance_workaround(readme: str):
    if not Path(readme).exists():
        pytest.skip(f"{readme} not present yet (docs deferred)")
    text = Path(readme).read_text(encoding="utf-8").lower()
    assert "--no-deps" not in text
    assert "editdistance-s" not in text


def test_project_is_installable_with_a_console_script():
    text = _pyproject()
    assert "[project.scripts]" in text
    assert 'livetranslate = "livetranslate.__main__:main"' in text
    assert "[build-system]" in text
    assert "hatchling" in text
    assert "package = true" in text


def test_workspace_contains_the_server_package():
    text = _pyproject()
    assert "[tool.uv.workspace]" in text
    assert "src/livetranslate_server" in text
    server = Path("src/livetranslate_server/pyproject.toml").read_text(encoding="utf-8").lower()
    assert "fastapi" in server
    assert "uvicorn" in server
    assert "faster-whisper" in server
