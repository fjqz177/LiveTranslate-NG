"""Generate the full runtime ``requirements-full-*.txt`` that pyappify installs.

uv is the authoritative resolver (``uv.lock`` + ``pyproject.toml`` engine-*
extras). pyappify installs with **pip**; this bridges the two by exporting uv's
resolution as fully-pinned pip input.

Profiles:
  - **cpu** (default): base + ``engine-funasr`` + ``engine-whisper``. The
    ``torch == 2.11.0`` wheel comes from **PyPI** (CPU build, no CUDA bundle), so
    a CPU install stays ~1.5GB and the user fetches it from a CN mirror
    (pyappify ``pip_args`` index-url).
  - **gpu**: additionally ``engine-cuda``; ``torch == 2.11.0+cu126`` comes from
    the **cu126** PyTorch index, pinned in ``pyproject.toml`` as the
    ``explicit`` "pytorch" index. That scopes cu126 ONLY to torch/torchaudio;
    every non-torch package resolves from the ``pypi`` default index, so each
    emitted ``--hash=`` is complete. (A package missing its hash — e.g. jinja2
    pulled off the cu126 index — makes pip's implicit ``--require-hashes``
    reject the whole file: that was the gpu `pip install -r` exit-1.)

The torch source is chosen per-variant, NOT via ``--index-strategy``:
  - cpu: override ``--index "pytorch=<pypi>"`` so torch picks up the CPU wheel.
  - gpu: emit NO index override — rely on the explicit "pytorch" cu126 index.
  Both deliberately avoid ``--index-strategy unsafe-best-match``: it promotes the
  cu126 index to a global candidate, which BOTH drags old non-torch packages
  (idna==3.4) into resolution AND pulls jinja2/markupsafe off the cu126 index
  where uv cannot record a wheel hash — the exact cause of the gpu build failure.

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


def _pypi_index() -> str:
    """Default PyPI for resolving the base/engine wheels. Local dev uses the
    Tsinghua mirror (fast in mainland CN); CI overrides via env to the official
    index (GitHub runners sit outside China and treat the mirrors poorly)."""
    return os.environ.get("LT_REQUIREMENTS_PYPI_INDEX", _DEV_PYPI)


def _export(extras: list[str], *, pypi_index: str, cpu: bool) -> str:
    args = [
        "uv",
        "export",
        "--no-dev",
        "--no-emit-project",
        "--default-index",
        pypi_index,
    ]
    if cpu:
        # Override the explicit "pytorch" index to PyPI so torch == 2.11.0
        # resolves the CPU wheel (the cu126 index only ships +cu126 builds).
        args += ["--index", f"pytorch={pypi_index}"]
    for extra in extras:
        args += ["--extra", extra]
    return subprocess.run(
        args,
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def main() -> None:
    # CPU: torch from PyPI = ``torch==2.11.0`` (CPU wheel).
    cpu = _export(
        ["engine-funasr", "engine-whisper"],
        pypi_index=_pypi_index(),
        cpu=True,
    )
    # GPU: torch from the explicit cu126 "pytorch" index (pyproject.toml) =
    # ``torch==2.11.0+cu126``; non-torch packages from the pypi default index.
    gpu = _export(
        ["engine-funasr", "engine-whisper", "engine-cuda"],
        pypi_index=_pypi_index(),
        cpu=False,
    )
    (ROOT / "requirements-full-cpu.txt").write_text(cpu, encoding="utf-8")
    (ROOT / "requirements-full-gpu.txt").write_text(gpu, encoding="utf-8")
    print("wrote requirements-full-cpu.txt / requirements-full-gpu.txt")


if __name__ == "__main__":
    main()
