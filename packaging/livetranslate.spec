# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller onedir spec (plan §4.9, ADR-004).

onefile is forbidden: the ASR worker runs via multiprocessing, which needs
real files on disk. Engine extras (funasr / faster-whisper / torch) are NOT
bundled — they ship as engine packs per ADR-006, and the app degrades
gracefully when they are absent. The GUI never imports engine backends.

Build:  uv run pyinstaller packaging/livetranslate.spec --noconfirm
Output: dist/LiveTranslate/
"""

from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_data_files

# Spec-relative paths resolve against packaging/; anchor to the repo root.
ROOT = Path(SPECPATH).resolve().parent

datas = [
    (str(ROOT / "config.yaml"), "."),
    (str(ROOT / "assets" / "icons"), "assets/icons"),
    (str(ROOT / "i18n" / "zh.yaml"), "i18n"),
    (str(ROOT / "i18n" / "en.yaml"), "i18n"),
    (str(ROOT / "i18n" / "CHANGELOG_zh.md"), "i18n"),
    (str(ROOT / "i18n" / "CHANGELOG_en.md"), "i18n"),
]
binaries = []
hiddenimports = []

# onnxruntime ships native DLLs inside its package. collect_all() would also
# hidden-import the whole submodule tree, and onnxruntime.transformers (the
# quantization tooling) imports the real transformers package — which drags
# av/scipy/librosa/sklearn/llvmlite into the frozen bundle (+300 MB). Collect
# the native binaries only; the runtime import path (onnxruntime.capi) is
# found by normal static analysis.
from PyInstaller.utils.hooks import collect_dynamic_libs

binaries += collect_dynamic_libs("onnxruntime")

# pysbd bundles per-language abbreviation data.
datas += collect_data_files("pysbd")

# asr/engines/funasr_nano.py imports the vendored model.py via a sys.path
# insert (funasr's remote-code protocol), so static analysis never sees it.
# Bundle the directory at the path __file__-relative resolution expects:
# _MEIPASS/livetranslate/asr/vendor/funasr_nano.
datas.append(
    (
        str(ROOT / "src" / "livetranslate" / "asr" / "vendor" / "funasr_nano"),
        "livetranslate/asr/vendor/funasr_nano",
    )
)

# Model downloads go through modeling/hub_downloader.py (httpx direct REST);
# the SDKs stay out of the frozen bundle (SelfServe P0-A2).
hiddenimports += []

# Silero VAD ONNX (torch-free runtime): silero-vad is a dev-only package
# (it pulls torch), so bundle its model export instead of the package.
# The scorer resolves _MEIPASS/models/vad/silero_vad.onnx in frozen mode.
import importlib.util as _ilu

_sv_spec = _ilu.find_spec("silero_vad")
if _sv_spec and _sv_spec.submodule_search_locations:
    _sv_onnx = Path(_sv_spec.submodule_search_locations[0]) / "data" / "silero_vad.onnx"
    if _sv_onnx.exists():
        datas.append((str(_sv_onnx), "models/vad"))

a = Analysis(
    [str(ROOT / "src" / "livetranslate" / "__main__.py")],
    pathex=[str(ROOT / "src")],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=[
        "torch",
        "torchaudio",
        "funasr",
        "faster_whisper",
        "tkinter",
        # Engine-side transitive chains that static analysis pulls in through
        # lazy imports (asr/engines/anime_whisper's `from transformers import
        # pipeline`): the frozen bundle must stay engine-free — workers load
        # these from the engine venv (ADR-007). Excluding the whole chain
        # (not just the tops) keeps av/scipy/llvmlite/sklearn/PIL out too.
        # sentencepiece is NOT excluded: sensevoice-onnx needs it in base.
        "transformers",
        "tokenizers",
        "safetensors",
        "jieba",
        "datasets",
        "av",
        "scipy",
        "sklearn",
        "librosa",
        "numba",
        "llvmlite",
        "PIL",
        "soundfile",
        "soxr",
        "lazy_loader",
        "imageio",
        "einops",
        "hf_xet",
        "modelscope",
        "huggingface_hub",
    ],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="LiveTranslate",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    icon=str(ROOT / "assets" / "icons" / "app.ico"),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="LiveTranslate",
)

