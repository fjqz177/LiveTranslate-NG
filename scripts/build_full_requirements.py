"""Generate the full runtime ``requirements-full-*.txt`` that pyappify installs.

uv is the authoritative resolver (``uv.lock`` + ``pyproject.toml`` engine-*
extras). pyappify installs with **pip**; this bridges the two by exporting uv's
resolution as fully-pinned pip input. The default (CPU) profile bundles
base + ``engine-funasr`` + ``engine-whisper``; the gpu profile additionally
pulls the cu126 torch wheels via the pytorch index.

A CI "drift gate" re-runs this and diffs the committed ``requirements-full-*.txt``
(see ``docs/development/重写方案.md`` §4.7) so the two sources never drift.

Run:  uv run python scripts/build_full_requirements.py
"""

from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# --no-dev:  exclude the dev toolchain. --no-emit-project: excluded the project
# package itself (the app code runs from source via main.py, not pip-installed).
# uv export no longer accepts `--python-platform` (removed in uv 0.12); the
# export follows the lock's platform resolution. The build/CI runs on Windows,
# so the current-platform export is exactly the target wheel set.
_EXPORT = [
    "uv",
    "export",
    "--no-dev",
    "--no-emit-project",
]


def _export(extra: list[str]) -> str:
    return subprocess.run(
        [*_EXPORT, *extra],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def main() -> None:
    cpu = _export(["--extra", "engine-funasr", "--extra", "engine-whisper"])
    gpu = _export(
        [
            "--extra",
            "engine-funasr",
            "--extra",
            "engine-whisper",
            "--extra",
            "engine-cuda",
        ]
    )
    (ROOT / "requirements-full-cpu.txt").write_text(cpu, encoding="utf-8")
    (ROOT / "requirements-full-gpu.txt").write_text(gpu, encoding="utf-8")
    print("wrote requirements-full-cpu.txt / requirements-full-gpu.txt")


if __name__ == "__main__":
    main()
