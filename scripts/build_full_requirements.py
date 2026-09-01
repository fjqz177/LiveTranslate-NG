"""Generate the full runtime ``requirements-full-*.txt`` that pyappify installs.

uv is the authoritative resolver (``uv.lock`` + ``pyproject.toml`` engine-*
extras). pyappify installs with **pip**; this bridges the two by exporting uv's
resolution as fully-pinned pip input.

Profiles:
  - **cpu** (default): base + ``engine-funasr`` + ``engine-whisper``, torch
    resolved from PyPI (``torch==2.11.0`` — the Windows CPU wheel, no CUDA
    localizer). Keeps a CPU install ~1.5GB and lets the user download it via
    the CN PyPI mirror (pyappify ``pip_args`` index-url).
  - **gpu**: additionally ``engine-cuda``, torch from the cu126 index
    (``torch==2.11.0+cu126``). The cu126 index is overridable via
    ``LT_REQUIREMENTS_PYTORCH_INDEX`` (CI uses PyTorch official; local dev uses
    the NJU mirror).

A CI "drift gate" re-runs this and diffs the committed ``requirements-full-*.txt``
(see ``docs/development/重写方案.md`` §4.7) so the two sources never drift.

Run:  uv run python scripts/build_full_requirements.py
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

_DEV_PYPI = "https://pypi.tuna.tsinghua.edu.cn/simple"  # CN-mirror dev default
_NJU_CU126 = "https://mirror.nju.edu.cn/pytorch/whl/cu126"  # CN-mirror cu126


def _pypi_index() -> str:
    """Default PyPI for resolving the base/engine wheels. Local dev uses the
    Tsinghua mirror (fast in mainland CN); CI overrides via env to the official
    index (GitHub runners sit outside China and treat the mirrors poorly)."""
    return os.environ.get("LT_REQUIREMENTS_PYPI_INDEX", _DEV_PYPI)


def _torch_index() -> str:
    """cu126 PyTorch index: local dev uses the NJU mirror (reachable in CN);
    CI overrides via env to the official index."""
    return os.environ.get("LT_REQUIREMENTS_PYTORCH_INDEX", _NJU_CU126)


def _export(extra: list[str], *, torch_index: str) -> str:
    # `--index "pytorch=<url>"` overrides the ``[tool.uv.sources] torch`` index
    # for this export, so pointing it at PyPI yields the plain CPU wheel while
    # pointing it at cu126 yields the CUDA build. Export never downloads — it
    # only emits pinned requirements, so the chosen index just has to be
    # reachable for metadata.
    return subprocess.run(
        [
            "uv",
            "export",
            "--no-dev",
            "--no-emit-project",
            "--default-index",
            _pypi_index(),
            "--index",
            f"pytorch={torch_index}",
            # The cu126 / cpu PyTorch index is a mirror that also carries old
            # non-torch packages; uv pins each package to the first index that
            # has it, which would drag e.g. idna==3.4 from the pytorch mirror
            # and break openai. unsafe-best-match lets every non-torch package
            # resolve from all indexes (same strategy ci.yml uses for sync).
            "--index-strategy", "unsafe-best-match",
            *extra,
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def main() -> None:
    # CPU: torch from PyPI = ``torch==2.11.0`` (CPU wheel).
    cpu = _export(
        ["--extra", "engine-funasr", "--extra", "engine-whisper"],
        torch_index=_pypi_index(),
    )
    # GPU: torch from the cu126 index = ``torch==2.11.0+cu126`` (+ cuda libs).
    gpu = _export(
        [
            "--extra",
            "engine-funasr",
            "--extra",
            "engine-whisper",
            "--extra",
            "engine-cuda",
        ],
        torch_index=_torch_index(),
    )
    (ROOT / "requirements-full-cpu.txt").write_text(cpu, encoding="utf-8")
    (ROOT / "requirements-full-gpu.txt").write_text(gpu, encoding="utf-8")
    print("wrote requirements-full-cpu.txt / requirements-full-gpu.txt")


if __name__ == "__main__":
    main()
