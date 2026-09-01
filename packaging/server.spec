# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the remote ASR server (plan Phase 5 尾项).

Onefile is acceptable: the server is a single uvicorn process (no
multiprocessing children). Requires faster-whisper + fastapi + uvicorn.

Build:  uv run pyinstaller packaging/server.spec --noconfirm
Output: dist/livetranslate-server
"""

from pathlib import Path

ROOT = Path(SPECPATH).resolve().parent

a = Analysis(
    [str(ROOT / "src" / "livetranslate_server" / "__main__.py")],
    pathex=[str(ROOT / "src")],
    binaries=[],
    datas=[],
    hiddenimports=["uvicorn.logging", "uvicorn.loops", "uvicorn.protocols"],
    hookspath=[],
    runtime_hooks=[],
    excludes=["PyQt6", "onnxruntime", "modelscope", "huggingface_hub", "pysbd"],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="livetranslate-server",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
)

