"""Runtime variant requirements guard (M-VENV, architecture hardening Milestone 1).

The shipped engine-venv requirements (`runtime/requirements/<variant>.txt`)
are `uv export --no-dev` output for the engine extras
(engine-funasr + engine-whisper): the engine venv gets those extras and
their transitive closure and NOTHING else. The dev toolchain
([dependency-groups] dev: mypy/ruff/pytest/pre-commit/pyinstaller/pillow/
silero-vad/onnx) must never leak into these files — the embedded uv runs
`uv pip install -r` against them on end-user machines.

Boundary (see docs/development/dependency-distribution.md): the base closure
from [project.dependencies] is structurally still carried by uv export, but
that is harmless redundancy (the worker subprocess inherits the frozen base
env); it is neither removed nor re-installed by the engine venv.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

from livetranslate.core import uv_runner

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REQ_DIR = PROJECT_ROOT / "runtime" / "requirements"

# Top-level package names that must NEVER appear as a pin in a shipped
# engine-venv requirements file. All of them come from the [dependency-groups]
# dev toolchain — they have no business in an end-user engine venv.
DEV_DENY = {
    "mypy",
    "ruff",
    "pytest",
    "pytest-qt",
    "pre-commit",
    "pyinstaller",
    "pyinstaller-hooks-contrib",
    "pillow",
    "silero-vad",
    "onnx",
    "black",
    "flake8",
    "coverage",
}

# torch's runtime deps (jinja2, markupsafe) resolve from the CPU pytorch
# index (download.pytorch.org/whl/cpu), which does not publish PEP 658 hash
# metadata for the non-torch wheels it serves — real `uv export` emits those
# exact pins hash-less. Nothing else may be: the files are regeneration output,
# never hand-written, so this carve-out is the documented seam rather than a
# bypass.
_HASHLESS_ALLOWED = {"jinja2", "markupsafe"}

_PIN_RE = re.compile(r"^(\w[\w\-]*)==\s*")


def _load_build_runtime_variants() -> object:
    """Import scripts/build_runtime_variants.py (no scripts/__init__.py)."""
    path = PROJECT_ROOT / "scripts" / "build_runtime_variants.py"
    spec = importlib.util.spec_from_file_location("build_runtime_variants", path)
    assert spec is not None and spec.loader is not None, "spec_from_file_location failed"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def variant_files():
    return sorted((p.stem, p) for p in REQ_DIR.glob("*.txt") if p.is_file())


def _pins(lines: list[str]) -> list[tuple[int, str, str]]:
    """(line_index, package_name, raw_line) for every `name==` pin line."""
    pins = []
    for i, line in enumerate(lines):
        if line.startswith("--index-url") or not line.strip():
            continue
        m = _PIN_RE.match(line)
        if m:
            pins.append((i, m.group(1), line))
    return pins


def test_variant_name_sets_are_consistent():
    """On-disk files == uv_runner.VARIANTS == build_runtime_variants.VARIANTS.

    The engine manager (core/uv_runner.py) can only offer variants whose
    requirements file exists; the generator (scripts/build_runtime_variants.py)
    lists the variants it exports. All three sources of truth must agree.
    """
    build = _load_build_runtime_variants()
    build_names = {name for name, _index in build.VARIANTS}
    disk_names = {p.stem for p in REQ_DIR.glob("*.txt")}
    assert disk_names == set(uv_runner.VARIANTS) == build_names


def test_no_dev_toolchain_in_variant_requirements(variant_files):
    offenders: dict[str, list[str]] = {}
    for name, path in variant_files:
        lines = path.read_text(encoding="utf-8").splitlines()
        pinned_names = {pname for _i, pname, _line in _pins(lines)}
        leaked = sorted(pinned_names & DEV_DENY)
        if leaked:
            offenders[name] = leaked
    assert not offenders, f"dev toolchain leaked into engine-venv requirements: {offenders}"


def test_pins_are_hash_pinned(variant_files):
    """Reproducibility: every `==` pin carries a `--hash=sha256:` continuation.

    The only hash-less pins allowed are the CPU-index torch transitive deps
    (see _HASHLESS_ALLOWED); any other pin without a hash is a drift that a
    fresh `uv export --no-dev` must repair.
    """
    missing: dict[str, list[str]] = {}
    for name, path in variant_files:
        lines = path.read_text(encoding="utf-8").splitlines()
        nohash = []
        for i, pname, _line in _pins(lines):
            nxt = lines[i + 1].strip() if i + 1 < len(lines) else ""
            if not nxt.startswith("--hash=sha256:"):
                nohash.append(pname)
        leftover = [p for p in nohash if p not in _HASHLESS_ALLOWED]
        if leftover:
            missing[name] = leftover
    assert not missing, f"pins missing a --hash=sha256 continuation: {missing}"


def test_variant_files_are_pin_lists(variant_files):
    """Each file is a non-empty pin list — no `-e` entry may slip in."""
    for name, path in variant_files:
        lines = path.read_text(encoding="utf-8").splitlines()
        assert lines, f"{name}: file is empty"
        assert not [ln for ln in lines if re.match(r"^\s*-e\s", ln)], (
            f"{name}: contains an editable (-e) entry"
        )
