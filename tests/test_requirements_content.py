"""Content guards for the shipped install requirements (requirements-full-*.txt).

These two files are what pyappify actually ``pip install``s on the end-user
machine, so they are the real install surface — yet test_requirements.py only
asserts substrings against pyproject and never reads them. Historically this is
where the bug lived: a package with a **missing ``--hash=``** made the whole file
uninstallable under pip's implicit ``--require-hashes`` (the CPU/GPU install
exit-1 root cause), and the CPU/GPU torch variant split is easy to silently
invert.

This file locks the content contract directly, so a future regeneration that
drops a hash, reverses the variant, or lets a non-Linux CUDA-13 bundle into the
Windows CPU profile fails loudly instead of shipping to users.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CPU = PROJECT_ROOT / "requirements-full-cpu.txt"
GPU = PROJECT_ROOT / "requirements-full-gpu.txt"

# A top-level pin line: no leading whitespace, non-comment, `name==version` or
# `name>=version`, optionally ` ; sys_platform == 'linux'`.
_PIN_RE = re.compile(r"^[^\s#].*?(==|>=)\d")

# CUDA-13 bundle packages that must NEVER occur ungated (the plain `torch==2.11.0`
# wheel carries these as Linux-only transitive deps; on the Windows target they
# must be inert, i.e. gated `; sys_platform == 'linux'`).
_CUDA13_MARKERS = ("nvidia-", "cuda-", "triton")


def _pin_blocks(text: str):
    """Yield (spec, has_hash) for each top-level pinned package block."""
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if _PIN_RE.match(line):
            spec = line
            has_hash = False
            j = i + 1
            while j < len(lines):
                nxt = lines[j]
                # Next top-level package line terminates this block.
                if nxt and not nxt[0].isspace() and not nxt.strip().startswith("#"):
                    break
                if "--hash=" in nxt:
                    has_hash = True
                j += 1
            yield spec, has_hash
            i = j
        else:
            i += 1


def _assert_requirements_exist(path: Path):
    if not path.exists():
        pytest.fail(f"{path.name} missing — run scripts/build_full_requirements.py and commit")


@pytest.mark.parametrize("req", [CPU, GPU], ids=["cpu", "gpu"])
def test_every_pin_has_a_hash(req):
    """No package may ship without a --hash= (a missing hash makes pip's
    implicit --require-hashes reject the whole file)."""
    _assert_requirements_exist(req)
    no_hash = [
        spec for spec, has_hash in _pin_blocks(req.read_text(encoding="utf-8")) if not has_hash
    ]
    assert not no_hash, f"{req.name}: packages missing --hash=: {no_hash}"


def test_cpu_profile_torch_is_cpu_wheel():
    """CPU profile pins the plain CPU wheel, never the +cu126 CUDA build."""
    _assert_requirements_exist(CPU)
    text = CPU.read_text(encoding="utf-8")
    assert re.search(r"^torch==2\.11\.0(?:\s|\\|$)", text, re.MULTILINE), (
        "cpu must pin torch==2.11.0"
    )
    assert not re.search(r"^torch==2\.11\.0\+cu", text, re.MULTILINE), (
        "cpu must NOT use the +cu126 wheel"
    )


def test_gpu_profile_torch_is_cuda_wheel():
    _assert_requirements_exist(GPU)
    text = GPU.read_text(encoding="utf-8")
    assert re.search(r"^torch==2\.11\.0\+cu126", text, re.MULTILINE), (
        "gpu must pin torch==2.11.0+cu126"
    )


@pytest.mark.parametrize("req", [CPU, GPU], ids=["cpu", "gpu"])
def test_cuda13_bundle_is_linux_gated(req):
    """nvidia-*/cuda-toolkit/triton may only appear behind `; sys_platform ==
    'linux'`. They are transitive deps of the plain torch wheel — harmless ON
    Linux but must never leak into the Windows target's install."""
    _assert_requirements_exist(req)
    text = req.read_text(encoding="utf-8")
    ungated = [
        spec
        for spec, _ in _pin_blocks(text)
        if any(m in spec for m in _CUDA13_MARKERS) and "sys_platform == 'linux'" not in spec
    ]
    assert not ungated, f"{req.name}: ungated CUDA-13 bundle packages: {ungated}"
