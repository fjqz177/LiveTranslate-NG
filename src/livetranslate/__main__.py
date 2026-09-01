"""LiveTranslate entry point (the only import-order owner).

apply_cache_env() must run before torch (TORCH_HOME resolution), and on
Windows torch must be imported before PyQt6 to avoid the c10.dll conflict
(pytorch#166628). The torch-before-Qt constraint is Windows-only; keep it
platform-gated so non-Windows starts clean.

The console script ([project.scripts]) points here too, so smoke-mode
environment setup runs for both 'uv run livetranslate' and 'python -m
livetranslate'.
"""

import os
import sys
import tempfile

# Smoke mode (CI L4 packaging checks): isolate all data in a temp dir and
# run headless. Must run BEFORE paths.py computes CONFIG_DIR (import-time).
if "--smoke" in sys.argv:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ["LIVETRANSLATE_PORTABLE_DIR"] = tempfile.mkdtemp(prefix="livetranslate-smoke-")

# Frozen worker bootstrap (PyInstaller + multiprocessing spawn): the ASR
# worker child re-executes this entry point with "--multiprocessing-fork"
# args. freeze_support() diverts it into multiprocessing.spawn.spawn_main()
# BEFORE any application code runs. Without it the child re-runs the whole
# app — the single-instance gate wakes the primary and the child exits 0,
# which the client surfaces as a bare ASRWorkerExited instead of the real
# engine error (spawn bootstrap also dead-locks without an import guard).
if sys.platform == "win32":
    import multiprocessing

    multiprocessing.freeze_support()

from livetranslate.modeling.manager import apply_cache_env

# Set cache env BEFORE importing torch so TORCH_HOME is respected.
apply_cache_env()

if sys.platform == "win32":
    # torch must precede PyQt6 on Windows (DLL order); engine deps ship with
    # the base install, so torch is always importable.
    try:
        import torch
    except ImportError:
        torch = None

from livetranslate.app import main as _app_main


def main() -> None:
    _app_main()


if __name__ == "__main__":
    main()
