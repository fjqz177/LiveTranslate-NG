"""Embedded uv runner (SelfServe P1-B2): install engine variants.

Frozen builds ship `tools/uv.exe` next to `app\\` and `data\\`; dev runs use
the uv on PATH. Variant requirements (runtime/requirements/<variant>.txt,
bundled into the frozen app) are pinned + hashed exports, so installing is
`uv pip install -r` into the staging venv — no resolution drift on user
machines, exactly the "networked but never re-resolved" contract.

Mirror selection rewrites the PyPI index URL; the pytorch wheel index
follows the variant (cpu / cu126). All state changes go through
engine_runtime's staging protocol (begin/complete/abort).
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

from livetranslate.core import engine_runtime as er
from livetranslate.core.paths import PROJECT_ROOT, data_root

log = logging.getLogger("LiveTranslate.UvRunner")

ProgressCb = Callable[[str], None]

PYPI_MIRRORS = {
    "official": "https://pypi.org/simple",
    "tsinghua": "https://pypi.tuna.tsinghua.edu.cn/simple",
    "aliyun": "https://mirrors.aliyun.com/pypi/simple",
    "nju": "https://mirrors.nju.edu.cn/pypi/simple",
    "ustc": "https://mirrors.ustc.edu.cn/pypi/simple",
}
# The pytorch wheel index stays OFFICIAL for every variant (P2-C3 measured:
# 188 ms from this network). Chinese pytorch-wheels mirrors (aliyun/SJTU) are
# flat wheel directories, not PEP 503 indexes — uv cannot resolve from them,
# verified 2026-08-17 with a dry-run install. Mirror selection therefore
# applies to PyPI only.
PYTORCH_INDEX = "https://download.pytorch.org/whl/{cu}"
# torch wheel index candidates (same shape as the official index: PEP 503 with a
# {cu} branch). nju verified 2026-08-31: cu126/ and cpu/ exist and carry
# torch-2.11.0+cu126/cpu-cp312-win_amd64.whl. The old note that "CN pytorch
# mirrors are flat dirs" applies only to non-PEP503 ones (aliyun/SJTU).
TORCH_INDEX_MIRRORS: dict[str, str] = {
    "official": PYTORCH_INDEX,
    "nju": "https://mirrors.nju.edu.cn/pytorch/whl/{cu}",
}
# Keep in lockstep with scripts/build_runtime_variants.VARIANTS (M-VENV enforces
# the agreement via tests/test_runtime_requirements.py::test_variant_name_sets_are_consistent).
# A variant only exists here if torch publishes that wheel (this project ships
# only cpu + the default CUDA slot); no higher-CUDA wheel is available.
VARIANTS = ("cpu", "cu126")
DEFAULT_MIRROR = "official"


class UvRunnerError(RuntimeError):
    """uv invocation failed (missing binary, bad variant, install error)."""


def uv_binary() -> Path | None:
    """Frozen: <install root>\\tools\\uv(.exe); dev: uv on PATH."""
    if getattr(sys, "frozen", False):
        exe_dir = Path(sys.executable).resolve().parent  # <install>\app
        for name in ("uv.exe", "uv"):
            candidate = exe_dir.parent / "tools" / name
            if candidate.exists():
                return candidate
        return None
    found = shutil.which("uv")
    return Path(found) if found else None


def bundled_python() -> Path | None:
    """The CPython shipped with the frozen package (tools\\python\\python.exe).

    The runtime never discovers a Python on the user machine: the packaging
    chain copies the uv-managed CPython 3.12 into <install root>\\tools\\python,
    and uv venv always gets this concrete path. That removes the whole class
    of user-machine failures (managed-install discovery, version-alias
    junction trust / os error 448, no-local-python downloads). Dev runs have
    no bundle and return None (uv on PATH handles it).
    """
    if not getattr(sys, "frozen", False):
        return None
    exe_dir = Path(sys.executable).resolve().parent  # <install>\app
    candidate = exe_dir.parent / "tools" / "python" / "python.exe"
    return candidate if candidate.exists() else None


def venv_python(staging: Path) -> Path:
    if os.name == "nt":
        return staging / ".venv" / "Scripts" / "python.exe"
    return staging / ".venv" / "bin" / "python"


def requirements_file(variant: str) -> Path:
    if variant not in VARIANTS:
        raise UvRunnerError(f"unknown variant: {variant!r}")
    path = PROJECT_ROOT / "runtime" / "requirements" / f"{variant}.txt"
    if not path.exists():
        raise UvRunnerError(f"variant requirements missing: {path}")
    return path


def _resolve_mirror(mirror: str) -> str:
    if mirror == "auto":
        mirror = DEFAULT_MIRROR
    if mirror not in PYPI_MIRRORS:
        raise UvRunnerError(f"unknown pypi mirror: {mirror!r}")
    return PYPI_MIRRORS[mirror]


def _torch_index(variant: str, torch_mirror: str = "official") -> str:
    template = TORCH_INDEX_MIRRORS.get(torch_mirror)
    if template is None:
        raise UvRunnerError(f"unknown torch mirror: {torch_mirror!r}")
    return template.format(cu=variant)


def _managed_python(major_minor: str) -> Path | None:
    """Concrete python.exe from uv's managed installs, if one exists.

    uv resolves "--python 3.12" through its version-alias directory, which is
    a symlink (mount point). Windows can mark that mount point untrusted and
    return os error 448 to low-trust processes (unsigned app launched from
    Explorer), measured on a user machine as "untrusted mount point". The
    real install dirs carry the patch version in their name, so globbing
    `cpython-3.12.*` selects them directly and never traverses the alias.
    """
    base = os.environ.get("UV_PYTHON_INSTALL_DIR")
    if base:
        root = Path(base)
    elif os.name == "nt":
        root = Path.home() / "AppData" / "Roaming" / "uv" / "python"
    else:
        root = Path.home() / ".local" / "share" / "uv" / "python"
    if not root.is_dir():
        return None
    for entry in sorted(root.glob(f"cpython-{major_minor}.*"), reverse=True):
        exe = entry / "python.exe" if os.name == "nt" else entry / "bin" / "python"
        if exe.is_file():
            return exe
    return None


def _fix_pyvenv_home(venv: Path, python_exe: Path) -> None:
    """Rewrite pyvenv.cfg's home to the concrete interpreter dir.

    uv canonicalizes a managed interpreter back to its version-alias
    directory (measured: passing the real path still yields home=<alias>).
    That alias is a symlink which Windows can mark as an untrusted mount
    point; the venv redirector then fails to launch the base python for
    low-trust process chains with os error 448. Point home at the real dir.
    """
    cfg = venv / "pyvenv.cfg"
    if not cfg.exists():
        return
    home = str(python_exe.resolve().parent)
    lines = cfg.read_text(encoding="utf-8").splitlines()
    with cfg.open("w", encoding="utf-8") as fh:
        for line in lines:
            if line.startswith("home"):
                fh.write(f"home = {home}\n")
            else:
                fh.write(line + "\n")


def _uv_cache_env() -> dict[str, str] | None:
    """Pin uv's wheel cache into the install tree for frozen builds.

    Installed and portable packages must leave nothing in the user profile:
    UV_CACHE_DIR points at <data root>\\uv-cache so uninstalling removes it
    with the rest of the data. Dev runs keep uv's own default cache location
    (or an explicit user UV_CACHE_DIR) untouched.
    """
    if not getattr(sys, "frozen", False):
        return None
    env = os.environ.copy()
    env["UV_CACHE_DIR"] = str(data_root() / "uv-cache")
    return env


def _run_uv(
    argv: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run uv with UTF-8 decoding: uv always emits UTF-8, while text=True
    decodes with the locale (GBK on Chinese Windows) — a decode crash in the
    reader thread then leaves stderr None and masks the real error (the
    "'NoneType' object has no attribute 'strip'" failure)."""
    # win32: CREATE_NO_WINDOW so the frozen GUI never flashes a console for uv.
    create = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    return subprocess.run(
        argv,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(cwd) if cwd else None,
        env=env,
        creationflags=create,
    )


def _emit(cb: ProgressCb | None, message: str) -> None:
    if cb:
        cb(message)
    log.info("%s", message)


def install_variant(
    variant: str,
    *,
    app_version: str,
    pypi_mirror: str = "auto",
    torch_mirror: str = "official",
    python_version: str = "3.12",
    progress_cb: ProgressCb | None = None,
) -> Path:
    """Install one engine variant into the engine area (staging protocol).

    Returns the installed variant dir. Raises UvRunnerError on failure; the
    engine area is rolled back (abort_install) and resumable.
    """
    uv = uv_binary()
    if uv is None:
        raise UvRunnerError(
            "uv not found: frozen builds need tools/uv(.exe) next to app\\; "
            "dev runs need uv on PATH (https://docs.astral.sh/uv/)"
        )
    req = requirements_file(variant)
    pypi_url = _resolve_mirror(pypi_mirror)
    torch_url = _torch_index(variant, torch_mirror)
    uv_env = _uv_cache_env()  # frozen: keep uv's cache inside the install tree

    staging = er.begin_install(variant, app_version)
    _emit(progress_cb, f"creating venv: {variant}")

    # Interpreter priority: the bundled CPython (frozen packages ship
    # tools\python — never discover on the user machine), then the dev
    # machine's uv-managed install, then the bare version string (uv
    # resolves it itself). Run uv from the staging dir so no relative entry
    # in the requirements can resolve against a random CWD (Explorer
    # launches children with CWD=C:\Windows\system32).
    interpreter = bundled_python() or _managed_python(python_version)
    if getattr(sys, "frozen", False) and interpreter is None:
        er.abort_install(variant)  # roll back the staging area (P6: close the
        # raise-without-abort asymmetry that stranded state=installing)
        raise UvRunnerError(
            "bundled python missing: frozen builds ship tools\\python\\python.exe "
            "(scripts/package_windows.ps1 copies the uv-managed CPython 3.12)"
        )
    log.info("venv interpreter: %s", interpreter or python_version)
    venv_cmd = [
        str(uv),
        "venv",
        str(staging / ".venv"),
        "--python",
        str(interpreter) if interpreter else python_version,
    ]
    proc = _run_uv(venv_cmd, cwd=staging, env=uv_env)
    if proc.returncode != 0:
        er.abort_install(variant)
        raise UvRunnerError(f"uv venv failed: {(proc.stderr or '').strip()[-400:]}")

    if interpreter:
        _fix_pyvenv_home(staging / ".venv", interpreter)

    python = venv_python(staging)
    if not python.exists():
        er.abort_install(variant)
        raise UvRunnerError(f"venv python missing: {python}")

    install_cmd = [
        str(uv),
        "pip",
        "install",
        "-r",
        str(req),
        "--python",
        str(python),
        "--index-url",
        pypi_url,
        "--extra-index-url",
        torch_url,
        # The pinned requirements carry per-package --index-url annotations
        # (uv export output). Without this, uv's default first-index-wins
        # strategy dead-ends on packages that exist on the pytorch index at
        # a different version (measured: certifi==2026.7.22, cu126).
        "--index-strategy",
        "unsafe-best-match",
    ]
    _emit(progress_cb, f"installing {variant} (pypi: {pypi_url}, torch: {torch_url})")

    proc = _run_uv(install_cmd, cwd=staging, env=uv_env)
    if proc.returncode != 0:
        tail = (proc.stderr or "").strip()[-400:]
        er.abort_install(variant)
        raise UvRunnerError(f"uv pip install failed: {tail}")

    er.complete_install(variant)
    _emit(progress_cb, f"variant ready: {variant}")
    return er.variant_dir(variant)
